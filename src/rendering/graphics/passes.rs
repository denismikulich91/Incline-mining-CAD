//! WGPU render pass helpers.

use super::frustum::Frustum;
use super::*;

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
                if cached.translucent {
                    continue;
                }
                render_pass.set_pipeline(&self.block_model_render_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_bind_group(1, &cached.surface_style_bind_group, &[]);
                for chunk in &cached.surface_chunks {
                    if !frustum.intersects_aabb(chunk.gpu.bounds_min, chunk.gpu.bounds_max) {
                        continue;
                    }
                    render_pass.set_vertex_buffer(0, chunk.gpu.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        chunk.gpu.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..chunk.gpu.index_count, 0, 0..1);
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
            let mut transparent_blocks: Vec<_> = block_models
                .iter()
                .filter(|block_model| {
                    let entity = block_model.entity_id();
                    block_model.visible
                        && !editor.hidden_handles.contains(&entity)
                        && self
                            .block_model_gpu
                            .get(block_model.id)
                            .is_some_and(|cached| cached.translucent)
                })
                .collect();
            let forward = self.camera.forward();
            transparent_blocks.sort_by(|a, b| {
                let ac = a
                    .world_bounds()
                    .map(|(min, max)| (min + max) * 0.5)
                    .unwrap_or(DVec3::ZERO);
                let bc = b
                    .world_bounds()
                    .map(|(min, max)| (min + max) * 0.5)
                    .unwrap_or(DVec3::ZERO);
                let ad = (ac - self.camera.position).dot(forward);
                let bd = (bc - self.camera.position).dot(forward);
                bd.total_cmp(&ad)
            });
            for block_model in transparent_blocks {
                let Some(cached) = self.block_model_gpu.get(block_model.id) else {
                    continue;
                };
                render_pass.set_pipeline(&self.transparent_block_model_render_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_bind_group(1, &cached.surface_style_bind_group, &[]);
                // Overlapping translucent blocks only blend correctly when
                // drawn back-to-front; sorting whole models isn't enough
                // when a single model's own chunks overlap each other, so
                // order each model's chunks by depth too.
                let mut chunks: Vec<_> = cached.surface_chunks.iter().collect();
                chunks.sort_by(|a, b| {
                    let ac = self.scene_origin
                        + ((a.gpu.bounds_min + a.gpu.bounds_max) * 0.5).as_dvec3();
                    let bc = self.scene_origin
                        + ((b.gpu.bounds_min + b.gpu.bounds_max) * 0.5).as_dvec3();
                    let ad = (ac - self.camera.position).dot(forward);
                    let bd = (bc - self.camera.position).dot(forward);
                    bd.total_cmp(&ad)
                });
                for chunk in chunks {
                    if !frustum.intersects_aabb(chunk.gpu.bounds_min, chunk.gpu.bounds_max) {
                        continue;
                    }
                    render_pass.set_vertex_buffer(0, chunk.gpu.vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        chunk.gpu.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..chunk.gpu.index_count, 0, 0..1);
                }
                render_pass.set_pipeline(&self.transparent_surface_render_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            }
        }

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
            if cached.edge_chunks.is_empty() {
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
}
