//! WGPU render pass helpers.

use super::frustum::Frustum;
use super::*;

/// Conservative pixel bounding rectangle `(x, y, w, h)` of a scene-space AABB
/// within the `scaled_width × scaled_height` render sub-rect. Used to scissor
/// the fullscreen volume raycast down to the pixels a model can actually
/// cover, so small-on-screen models stop paying per-pixel ray setup across
/// the whole frame. `None` means the projection cannot bound the box (a
/// corner reaches the eye plane) and the caller must scissor to the full
/// sub-rect; a zero-area rect means the box projects entirely off-screen.
fn aabb_scissor_rect(
    view_proj: &glam::Mat4,
    bounds_min: glam::Vec3,
    bounds_max: glam::Vec3,
    scaled_width: f32,
    scaled_height: f32,
) -> Option<(u32, u32, u32, u32)> {
    let mut ndc_min = glam::Vec2::splat(f32::MAX);
    let mut ndc_max = glam::Vec2::splat(f32::MIN);
    for corner in 0..8u32 {
        let p = glam::vec3(
            if corner & 1 == 0 {
                bounds_min.x
            } else {
                bounds_max.x
            },
            if corner & 2 == 0 {
                bounds_min.y
            } else {
                bounds_max.y
            },
            if corner & 4 == 0 {
                bounds_min.z
            } else {
                bounds_max.z
            },
        );
        let clip = *view_proj * p.extend(1.0);
        if clip.w <= 1.0e-6 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        ndc_min = ndc_min.min(glam::vec2(ndc.x, ndc.y));
        ndc_max = ndc_max.max(glam::vec2(ndc.x, ndc.y));
    }
    // NDC → pixels (y flips), one pixel of padding against raster rounding.
    let x0 = ((ndc_min.x.max(-1.0) + 1.0) * 0.5 * scaled_width - 1.0)
        .floor()
        .max(0.0) as u32;
    let x1 = ((ndc_max.x.min(1.0) + 1.0) * 0.5 * scaled_width + 1.0)
        .ceil()
        .clamp(0.0, scaled_width.ceil()) as u32;
    let y0 = ((1.0 - ndc_max.y.min(1.0)) * 0.5 * scaled_height - 1.0)
        .floor()
        .max(0.0) as u32;
    let y1 = ((1.0 - ndc_min.y.max(-1.0)) * 0.5 * scaled_height + 1.0)
        .ceil()
        .clamp(0.0, scaled_height.ceil()) as u32;
    Some((x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)))
}

impl<'a> Graphics<'a> {
    pub(super) fn render_scene_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        editor: &EditorState,
        triangulations: &[OpenTriangulation],
        block_models: &[OpenBlockModel],
    ) {
        let bg_color = editor.renderer_background_color;
        let clear_color = [
            linear_to_srgb(bg_color[0]) as f64,
            linear_to_srgb(bg_color[1]) as f64,
            linear_to_srgb(bg_color[2]) as f64,
        ];
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.msaa_view,
                resolve_target: Some(view),
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: clear_color[0],
                        g: clear_color[1],
                        b: clear_color[2],
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // Block model chunks carry their own AABB, so cheaply skip GPU draw
        // calls for chunks that are entirely outside the current view
        // instead of always uploading/drawing every renderable block.
        let frustum = Frustum::from_view_proj(glam::Mat4::from_cols_array_2d(
            &self.camera_uniform.view_proj,
        ));

        if !self.triangulation_gpu.is_empty() || !self.block_model_gpu.is_empty() {
            render_pass.set_pipeline(&self.surface_render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            for triangulation in triangulations {
                let entity = triangulation.entity_id();
                if !triangulation.visible || editor.hidden_handles.contains(&entity) {
                    continue;
                }
                let Some(cached) = self.triangulation_gpu.get(triangulation.id) else {
                    continue;
                };
                if cached.color[3] < 0.999 {
                    continue;
                }
                render_pass.set_bind_group(1, &cached.surface_style_bind_group, &[]);
                for chunk in &cached.surface_chunks {
                    render_pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..chunk.index_count, 0, 0..1);
                }
            }
            for block_model in block_models {
                let entity = block_model.entity_id();
                if !block_model.visible || editor.hidden_handles.contains(&entity) {
                    continue;
                }
                let Some(cached) = self.block_model_gpu.get(block_model.id) else {
                    continue;
                };
                if cached.volume.is_some() {
                    continue;
                }
                if cached.surface_chunks.is_empty() {
                    continue;
                }
                render_pass.set_pipeline(&self.block_model_render_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_bind_group(1, &cached.surface_style_bind_group, &[]);
                for chunk in &cached.surface_chunks {
                    if !frustum.intersects_aabb(chunk.gpu.bounds_min, chunk.gpu.bounds_max) {
                        continue;
                    }
                    // Non-indexed: the shader expands 36 vertices per instance
                    // into a cube. One instance per block.
                    render_pass.set_vertex_buffer(0, chunk.gpu.instance_buffer.slice(..));
                    render_pass.draw(0..36, 0..chunk.gpu.instance_count);
                }
                render_pass.set_pipeline(&self.surface_render_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            }

            let mut transparent: Vec<_> = triangulations
                .iter()
                .filter(|triangulation| {
                    let entity = triangulation.entity_id();
                    triangulation.visible
                        && !editor.hidden_handles.contains(&entity)
                        && self
                            .triangulation_gpu
                            .get(triangulation.id)
                            .is_some_and(|cached| cached.color[3] < 0.999)
                })
                .collect();
            let forward = self.camera.forward();
            transparent.sort_by(|a, b| {
                let ac = a.mesh.bounds().center();
                let bc = b.mesh.bounds().center();
                let ad = (DVec3::new(ac.x, ac.y, ac.z) - self.camera.position).dot(forward);
                let bd = (DVec3::new(bc.x, bc.y, bc.z) - self.camera.position).dot(forward);
                bd.total_cmp(&ad)
            });
            render_pass.set_pipeline(&self.transparent_surface_render_pipeline);
            for triangulation in transparent {
                let Some(cached) = self.triangulation_gpu.get(triangulation.id) else {
                    continue;
                };
                render_pass.set_bind_group(1, &cached.surface_style_bind_group, &[]);
                for chunk in &cached.surface_chunks {
                    render_pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..chunk.index_count, 0, 0..1);
                }
            }
        }

        drop(render_pass);
        self.render_volume_block_models(encoder, view, editor, block_models, &frustum);
        self.render_fallback_transparent_block_models(
            encoder,
            view,
            editor,
            block_models,
            &frustum,
        );

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Overlay Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.msaa_view,
                resolve_target: Some(view),
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        if !self.lyon_buffer.vertices.is_empty() && !self.lyon_buffer.indices.is_empty() {
            render_pass.set_pipeline(if editor.xray_enabled {
                &self.xray_render_pipeline
            } else {
                &self.render_pipeline
            });
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.lyon_vertex_gpu.slice(..));
            render_pass.set_index_buffer(self.lyon_index_gpu.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.lyon_buffer.indices.len() as u32, 0, 0..1);
        }

        if !self.stroke_vertex_buf.is_empty() && !self.stroke_index_buf.is_empty() {
            render_pass.set_pipeline(if editor.xray_enabled {
                &self.overlay_render_pipeline
            } else {
                &self.stroke_render_pipeline
            });
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.stroke_vertex_gpu.slice(..));
            render_pass
                .set_index_buffer(self.stroke_index_gpu.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.stroke_index_buf.len() as u32, 0, 0..1);
        }

        if !self.dynamic_vertex_buf.is_empty() && !self.dynamic_index_buf.is_empty() {
            render_pass.set_pipeline(if editor.xray_enabled {
                &self.overlay_render_pipeline
            } else {
                &self.stroke_render_pipeline
            });
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.dynamic_vertex_gpu.slice(..));
            render_pass
                .set_index_buffer(self.dynamic_index_gpu.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.dynamic_index_buf.len() as u32, 0, 0..1);
        }

        render_pass.set_pipeline(&self.edge_render_pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        for triangulation in triangulations {
            let entity = triangulation.entity_id();
            if !triangulation.visible || editor.hidden_handles.contains(&entity) {
                continue;
            }
            let Some(cached) = self.triangulation_gpu.get(triangulation.id) else {
                continue;
            };
            if cached.edge_width <= 0.0 || cached.edge_chunks.is_empty() {
                continue;
            }
            render_pass.set_bind_group(1, &cached.edge_style_bind_group, &[]);
            for chunk in &cached.edge_chunks {
                render_pass.set_vertex_buffer(0, chunk.instance_buffer.slice(..));
                render_pass.draw(0..6, 0..chunk.instance_count);
            }
        }
        for block_model in block_models {
            let entity = block_model.entity_id();
            if !block_model.visible || editor.hidden_handles.contains(&entity) {
                continue;
            }
            let Some(cached) = self.block_model_gpu.get(block_model.id) else {
                continue;
            };
            if cached.edge_chunks.is_empty() {
                continue;
            }
            render_pass.set_bind_group(1, &cached.edge_style_bind_group, &[]);
            for chunk in &cached.edge_chunks {
                render_pass.set_vertex_buffer(0, chunk.instance_buffer.slice(..));
                render_pass.draw(0..6, 0..chunk.instance_count);
            }
        }

        if !self.overlay_vertex_buf.is_empty() && !self.overlay_index_buf.is_empty() {
            render_pass.set_pipeline(&self.overlay_render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.overlay_vertex_gpu.slice(..));
            render_pass
                .set_index_buffer(self.overlay_index_gpu.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.overlay_index_buf.len() as u32, 0, 0..1);
        }

        if let Err(e) = self.text_system.text_renderer.render(
            &self.text_system.text_atlas,
            &self.text_system.viewport,
            &mut render_pass,
        ) {
            log::error!("Text render failed: {e:?}");
        }
    }

    fn render_volume_block_models(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        editor: &EditorState,
        block_models: &[OpenBlockModel],
        frustum: &Frustum,
    ) {
        // Advance cell-pool streaming for the camera's current position before
        // drawing: near bricks become resident, far ones fall back to their
        // aggregates. No-op for volumes that fit the pool.
        let camera_scene = (self.camera.position - self.scene_origin).as_vec3();
        self.block_model_gpu
            .stream_volumes(&self.queue, camera_scene);

        let mut visible_volume_blocks = block_models
            .iter()
            .filter(|block_model| {
                let entity = block_model.entity_id();
                block_model.visible
                    && !editor.hidden_handles.contains(&entity)
                    && self
                        .block_model_gpu
                        .get(block_model.id)
                        .is_some_and(|cached| {
                            cached.volume.as_ref().is_some_and(|volume| {
                                frustum.intersects_aabb(volume.bounds_min, volume.bounds_max)
                            })
                        })
            })
            .collect::<Vec<_>>();
        if visible_volume_blocks.is_empty() {
            return;
        }
        // Each model raycasts independently and blends One/OneMinusSrcAlpha
        // into the shared target, so correctness requires back-to-front
        // ordering — the same convention as the transparent-triangulation
        // sort above. (Truly interpenetrating volumes would need a merged
        // march; distinct models composited far-to-near is the honest
        // approximation.)
        let forward = self.camera.forward();
        let camera_position = self.camera.position;
        let depth_along_view = |block_model: &OpenBlockModel| {
            block_model
                .world_bounds()
                .map(|(min, max)| ((min + max) * 0.5 - camera_position).dot(forward))
                .unwrap_or(0.0)
        };
        visible_volume_blocks.sort_by(|a, b| depth_along_view(b).total_cmp(&depth_along_view(a)));

        // Draw the raycast into an off-screen target at `render_scale` of full
        // resolution (reduced while the camera moves), then upscale it over the
        // scene. The target is full-size; we fill only the top-left sub-rect so
        // the scale can change per frame with no reallocation. Keeping the two
        // in lockstep: the raycast shader derives the same sub-rect from
        // `camera.viewport.xy * render_scale`, and the upscale samples exactly
        // that region.
        let render_scale = self.camera_uniform.render_scale();
        let scaled_width = (self.config.width as f32 * render_scale).max(1.0);
        let scaled_height = (self.config.height as f32 * render_scale).max(1.0);
        self.queue.write_buffer(
            &self.block_model_volume_target.params_buffer,
            0,
            bytemuck::cast_slice(&[scaled_width, scaled_height, 0.0f32, 0.0f32]),
        );

        {
            let mut volume_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block Model Volume Raycast Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.block_model_volume_target.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            volume_pass.set_viewport(0.0, 0.0, scaled_width, scaled_height, 0.0, 1.0);
            volume_pass.set_pipeline(&self.block_model_volume_pipeline);
            volume_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            volume_pass.set_bind_group(2, &self.block_model_peel_targets.peel_bind_groups[0], &[]);
            let view_proj = glam::Mat4::from_cols_array_2d(&self.camera_uniform.view_proj);
            let full_rect = (
                0u32,
                0u32,
                scaled_width.ceil() as u32,
                scaled_height.ceil() as u32,
            );
            for block_model in visible_volume_blocks {
                let Some(cached) = self.block_model_gpu.get(block_model.id) else {
                    continue;
                };
                let Some(volume) = cached.volume.as_ref() else {
                    continue;
                };
                // Scissor the fullscreen triangle to the model's projected
                // bounds; set per draw because scissor state persists across
                // draws within the pass.
                let (x, y, w, h) = aabb_scissor_rect(
                    &view_proj,
                    volume.bounds_min,
                    volume.bounds_max,
                    scaled_width,
                    scaled_height,
                )
                .unwrap_or(full_rect);
                if w == 0 || h == 0 {
                    continue;
                }
                volume_pass.set_scissor_rect(x, y, w, h);
                volume_pass.set_bind_group(1, &volume.bind_group, &[]);
                volume_pass.draw(0..3, 0..1);
            }
        }

        let mut upscale_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Block Model Volume Upscale Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.msaa_view,
                resolve_target: Some(view),
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        upscale_pass.set_pipeline(&self.block_model_volume_upscale_pipeline);
        upscale_pass.set_bind_group(0, &self.block_model_volume_target.bind_group, &[]);
        upscale_pass.draw(0..3, 0..1);
    }

    fn render_fallback_transparent_block_models(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        editor: &EditorState,
        block_models: &[OpenBlockModel],
        frustum: &Frustum,
    ) {
        let visible_transparent_blocks = block_models
            .iter()
            .filter(|block_model| {
                let entity = block_model.entity_id();
                block_model.visible
                    && !editor.hidden_handles.contains(&entity)
                    && self
                        .block_model_gpu
                        .get(block_model.id)
                        .is_some_and(|cached| {
                            cached.volume.is_none() && !cached.transparent_surface_chunks.is_empty()
                        })
            })
            .collect::<Vec<_>>();
        if visible_transparent_blocks.is_empty() {
            return;
        }

        {
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block Model Transparency Clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.block_model_peel_targets.accum_views[0],
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }

        {
            let mut peel_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block Model Order-Independent Transparency Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.block_model_peel_targets.accum_views[0],
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            peel_pass.set_pipeline(&self.block_model_peel_pipeline);
            peel_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            peel_pass.set_bind_group(2, &self.block_model_peel_targets.peel_bind_groups[0], &[]);
            for block_model in &visible_transparent_blocks {
                let Some(cached) = self.block_model_gpu.get(block_model.id) else {
                    continue;
                };
                peel_pass.set_bind_group(1, &cached.surface_style_bind_group, &[]);
                for chunk in &cached.transparent_surface_chunks {
                    if !frustum.intersects_aabb(chunk.gpu.bounds_min, chunk.gpu.bounds_max) {
                        continue;
                    }
                    peel_pass.set_vertex_buffer(0, chunk.gpu.instance_buffer.slice(..));
                    peel_pass.draw(0..36, 0..chunk.gpu.instance_count);
                }
            }
        }

        let mut composite_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Block Model Peel Composite Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.msaa_view,
                resolve_target: Some(view),
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        composite_pass.set_pipeline(&self.block_model_peel_composite_pipeline);
        composite_pass.set_bind_group(
            0,
            &self.block_model_peel_targets.composite_bind_groups[0],
            &[],
        );
        composite_pass.draw(0..3, 0..1);
    }
}
