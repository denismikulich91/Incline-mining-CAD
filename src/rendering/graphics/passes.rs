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

/// Choose a point LOD with a 20% dead band around each boundary. The dead
/// band prevents tiny camera movements or projection rounding from toggling a
/// chunk between two visibly different densities every frame.
fn point_lod_with_hysteresis(counts: [u32; 3], desired: usize, current: usize) -> usize {
    let mut level = current.min(counts.len() - 1);
    while level + 1 < counts.len() && desired.saturating_mul(5) < counts[level + 1] as usize * 4 {
        level += 1;
    }
    while level > 0 && desired.saturating_mul(4) > counts[level] as usize * 5 {
        level -= 1;
    }
    level
}

fn clamped_document_batch_range(
    range: (u32, u32),
    available_indices: usize,
) -> Option<std::ops::Range<u32>> {
    let start = (range.0 as usize).min(available_indices);
    let end = (range.1 as usize).min(available_indices);
    let end = start + (end.saturating_sub(start) / 3) * 3;
    (start < end).then_some(start as u32..end as u32)
}

fn document_primitive_order(primitive: DocumentPrimitive) -> u8 {
    match primitive {
        DocumentPrimitive::Fill => 0,
        DocumentPrimitive::Stroke => 1,
    }
}

impl<'a> Graphics<'a> {
    fn draw_document_batches<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        stage: DocumentRenderStage,
        xray_enabled: bool,
        primitive_filter: Option<DocumentPrimitive>,
    ) {
        let mut batches = self
            .document_draw_batches
            .iter()
            .filter(|batch| {
                batch.stage(xray_enabled) == stage
                    && primitive_filter.is_none_or(|primitive| batch.primitive == primitive)
            })
            .collect::<Vec<_>>();
        if stage == DocumentRenderStage::Translucent {
            let forward = self.camera.forward();
            let position = self.camera.position;
            batches.sort_by(|a, b| {
                let a_depth = (a.center - position).dot(forward);
                let b_depth = (b.center - position).dot(forward);
                b_depth.total_cmp(&a_depth).then_with(|| {
                    document_primitive_order(a.primitive)
                        .cmp(&document_primitive_order(b.primitive))
                })
            });
        } else {
            // Preserve the traditional fill-before-outline ordering. Stable
            // sorting retains object order within each primitive stream.
            batches.sort_by_key(|batch| document_primitive_order(batch.primitive));
        }

        let mut bound_primitive = None;
        for batch in batches {
            if bound_primitive != Some(batch.primitive) {
                let pipeline = match (stage, batch.primitive, xray_enabled) {
                    (_, DocumentPrimitive::Fill, true) => &self.xray_render_pipeline,
                    (DocumentRenderStage::Opaque, DocumentPrimitive::Fill, false)
                    | (DocumentRenderStage::Overlay, DocumentPrimitive::Fill, false) => {
                        &self.render_pipeline
                    }
                    (DocumentRenderStage::Translucent, DocumentPrimitive::Fill, false) => {
                        &self.transparent_document_fill_pipeline
                    }
                    (_, DocumentPrimitive::Stroke, true) => &self.overlay_render_pipeline,
                    (DocumentRenderStage::Opaque, DocumentPrimitive::Stroke, false) => {
                        &self.opaque_stroke_render_pipeline
                    }
                    (DocumentRenderStage::Translucent, DocumentPrimitive::Stroke, false)
                    | (DocumentRenderStage::Overlay, DocumentPrimitive::Stroke, false) => {
                        &self.stroke_render_pipeline
                    }
                };
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                match batch.primitive {
                    DocumentPrimitive::Fill => {
                        render_pass.set_vertex_buffer(0, self.lyon_vertex_gpu.slice(..));
                        render_pass.set_index_buffer(
                            self.lyon_index_gpu.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                    }
                    DocumentPrimitive::Stroke => {
                        render_pass.set_vertex_buffer(0, self.stroke_vertex_gpu.slice(..));
                        render_pass.set_index_buffer(
                            self.stroke_index_gpu.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                    }
                }
                bound_primitive = Some(batch.primitive);
            }

            let available = match batch.primitive {
                DocumentPrimitive::Fill => self.lyon_buffer.indices.len(),
                DocumentPrimitive::Stroke => self.stroke_index_buf.len(),
            };
            if let Some(range) = clamped_document_batch_range(batch.index_range, available) {
                render_pass.draw_indexed(range, 0, 0..1);
            }
        }
    }

    fn draw_static_document_strokes<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        xray_enabled: bool,
    ) {
        if !self
            .static_strokes
            .chunks()
            .iter()
            .any(|chunk| chunk.drawable())
        {
            return;
        }
        render_pass.set_pipeline(if xray_enabled {
            &self.overlay_render_pipeline
        } else {
            &self.opaque_stroke_render_pipeline
        });
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        for chunk in self.static_strokes.chunks() {
            if !chunk.drawable() {
                continue;
            }
            let (Some(vertex_gpu), Some(index_gpu)) = (&chunk.vertex_gpu, &chunk.index_gpu) else {
                continue;
            };
            render_pass.set_vertex_buffer(0, vertex_gpu.slice(..));
            render_pass.set_index_buffer(index_gpu.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..chunk.index_count, 0, 0..1);
        }
    }

    /// Scene-origin-relative AABB of a triangulation mesh, in the same space as
    /// the uploaded surface vertices (`world - scene_origin`, no vertical
    /// exaggeration — that lives in `view_proj`). Matches `surface_vertex` so a
    /// frustum built from `camera_uniform.view_proj` culls it correctly.
    fn mesh_scene_aabb(
        &self,
        mesh: &crate::model::formats::tri00t::Triangulation,
    ) -> (glam::Vec3, glam::Vec3) {
        let bounds = mesh.bounds();
        let min = glam::DVec3::new(bounds.min.x, bounds.min.y, bounds.min.z) - self.scene_origin;
        let max = glam::DVec3::new(bounds.max.x, bounds.max.y, bounds.max.z) - self.scene_origin;
        (min.as_vec3(), max.as_vec3())
    }

    /// Whether the camera is looking exactly down in orthographic mode — the
    /// only view in which flat plan-view raster images are drawn.
    fn plan_view_active(&self) -> bool {
        !self.projection.is_perspective() && self.camera.forward().z <= -(1.0 - 1.0e-6)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_scene_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        editor: &EditorState,
        triangulations: &[OpenTriangulation],
        block_models: &[OpenBlockModel],
        point_clouds: &[crate::model::point_cloud::OpenPointCloud],
        rasters: &[OpenRasterTexture],
        include_editor_overlays: bool,
    ) {
        let bg_color = editor.renderer_background_color;
        let clear_color = [
            bg_color[0].clamp(0.0, 1.0) as f64,
            bg_color[1].clamp(0.0, 1.0) as f64,
            bg_color[2].clamp(0.0, 1.0) as f64,
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
                    load: wgpu::LoadOp::Clear(0.0),
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
        let view_proj = glam::Mat4::from_cols_array_2d(&self.camera_uniform.view_proj);
        let frustum = Frustum::from_view_proj(view_proj);

        // Developer chunk-debug view: colour each surface chunk distinctly and
        // report how many chunks survive frustum culling.
        let debug_chunks = editor.debug_chunk_coloring;
        let mut rendered_chunks: u32 = 0;
        let mut total_chunks: u32 = 0;

        // Undraped rasters show as flat plan-view images: drawn before all
        // scene geometry, pinned to the far plane with depth writes off, and
        // only when the view is exactly top-down orthographic — from any
        // other angle a heightless image would be misleading.
        if self.plan_view_active() {
            let draped: std::collections::HashSet<_> = triangulations
                .iter()
                .filter_map(|triangulation| triangulation.raster_texture)
                .collect();
            let mut pipeline_bound = false;
            for raster in rasters {
                if draped.contains(&raster.id) {
                    continue;
                }
                let Some((bind_group, vertex_buffer)) = self.raster_gpu.plane(raster.id) else {
                    continue;
                };
                if !pipeline_bound {
                    render_pass.set_pipeline(&self.raster_plane_render_pipeline);
                    render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                    pipeline_bound = true;
                }
                render_pass.set_bind_group(1, bind_group, &[]);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.draw(0..4, 0..1);
            }
        }

        // Point splats write depth, so they draw with the opaque geometry —
        // before the transparent surfaces that must blend over them.
        if !self.point_cloud_gpu.is_empty() {
            let mut colored_pipeline_active = None;
            for point_cloud in point_clouds {
                if !point_cloud.visible {
                    continue;
                }
                let Some(cached) = self.point_cloud_gpu.get(point_cloud.id) else {
                    continue;
                };
                if colored_pipeline_active != Some(cached.colored) {
                    let pipeline = if cached.colored {
                        &self.point_cloud_colored_render_pipeline
                    } else {
                        &self.point_cloud_uncolored_render_pipeline
                    };
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                    colored_pipeline_active = Some(cached.colored);
                }
                render_pass.set_bind_group(1, &cached.style_bind_group, &[]);
                for chunk in cached.chunks.iter().filter_map(Option::as_ref) {
                    let bounds_min = chunk.bounds_min + cached.origin_scene;
                    let bounds_max = chunk.bounds_max + cached.origin_scene;
                    // Projected bounds are conservative and include raster
                    // padding. Unlike a separate plane test, an uncertain box
                    // at the eye plane returns `None` and is always retained.
                    let projected = aabb_scissor_rect(
                        &view_proj,
                        bounds_min,
                        bounds_max,
                        self.size.width.max(1) as f32,
                        self.size.height.max(1) as f32,
                    );
                    if projected.is_some_and(|(_, _, width, height)| width == 0 || height == 0) {
                        continue;
                    }
                    let desired_points = projected
                        .map(|(_, _, width, height)| {
                            ((u64::from(width) * u64::from(height)) / 4).max(1) as usize
                        })
                        .unwrap_or(usize::MAX);
                    let level = point_lod_with_hysteresis(
                        chunk.level_counts,
                        desired_points,
                        chunk.selected_level.get(),
                    );
                    chunk.selected_level.set(level);
                    let instance_count = chunk.level_counts[level];
                    render_pass.set_vertex_buffer(0, chunk.instance_buffer.slice(..));
                    render_pass.draw(0..4, 0..instance_count);
                }
            }
        }

        // Opaque document fills and strokes must establish colour and depth
        // before any translucent surface or block-model composite. X-ray
        // intentionally remains an editor overlay and is deferred.
        if !editor.xray_enabled {
            self.draw_document_batches(
                &mut render_pass,
                DocumentRenderStage::Opaque,
                false,
                Some(DocumentPrimitive::Fill),
            );
            self.draw_static_document_strokes(&mut render_pass, false);
            self.draw_document_batches(
                &mut render_pass,
                DocumentRenderStage::Opaque,
                false,
                Some(DocumentPrimitive::Stroke),
            );
        }

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
                total_chunks += cached.surface_chunks.len() as u32;
                // Cheap whole-mesh reject before touching individual chunks.
                let (aabb_min, aabb_max) = self.mesh_scene_aabb(&triangulation.mesh);
                if !frustum.intersects_aabb(aabb_min, aabb_max) {
                    continue;
                }
                if !debug_chunks {
                    render_pass.set_bind_group(1, &cached.surface_style_bind_group, &[]);
                }
                render_pass.set_bind_group(
                    3,
                    self.raster_gpu.bind_group(triangulation.raster_texture),
                    &[],
                );
                for chunk in &cached.surface_chunks {
                    // Per-chunk frustum cull: chunks are Morton-spatial, so their
                    // AABBs are tight enough for this to reject real geometry.
                    if !frustum.intersects_aabb(chunk.bounds_min, chunk.bounds_max) {
                        continue;
                    }
                    if debug_chunks {
                        render_pass.set_bind_group(1, &chunk.debug_style_bind_group, &[]);
                    }
                    render_pass.set_bind_group(2, &chunk.chunk_bind_group, &[]);
                    rendered_chunks += 1;
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
                total_chunks += cached.surface_chunks.len() as u32;
                let (aabb_min, aabb_max) = self.mesh_scene_aabb(&triangulation.mesh);
                if !frustum.intersects_aabb(aabb_min, aabb_max) {
                    continue;
                }
                if !debug_chunks {
                    render_pass.set_bind_group(1, &cached.surface_style_bind_group, &[]);
                }
                render_pass.set_bind_group(
                    3,
                    self.raster_gpu.bind_group(triangulation.raster_texture),
                    &[],
                );
                for chunk in &cached.surface_chunks {
                    if !frustum.intersects_aabb(chunk.bounds_min, chunk.bounds_max) {
                        continue;
                    }
                    if debug_chunks {
                        render_pass.set_bind_group(1, &chunk.debug_style_bind_group, &[]);
                    }
                    render_pass.set_bind_group(2, &chunk.chunk_bind_group, &[]);
                    rendered_chunks += 1;
                    render_pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..chunk.index_count, 0, 0..1);
                }
            }
        }

        drop(render_pass);
        // Secondary viewports (slice previews) must not overwrite the main
        // viewport's statistics or advance the shared volume residency
        // streamer: the main pass may already be encoded against the current
        // residency tables, and two cameras would evict each other's bricks
        // every frame. Previews read whatever is resident and fall back to
        // brick aggregates elsewhere.
        if include_editor_overlays {
            self.chunk_render_stats = (rendered_chunks, total_chunks);
        }
        let needs_volume_target = block_models.iter().any(|block_model| {
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
        });
        let needs_transparency_target = needs_volume_target
            || block_models.iter().any(|block_model| {
                let entity = block_model.entity_id();
                block_model.visible
                    && !editor.hidden_handles.contains(&entity)
                    && self
                        .block_model_gpu
                        .get(block_model.id)
                        .is_some_and(|cached| {
                            cached.volume.is_none()
                                && cached.transparent_surface_chunks.iter().any(|chunk| {
                                    frustum
                                        .intersects_aabb(chunk.gpu.bounds_min, chunk.gpu.bounds_max)
                                })
                        })
            });
        self.update_block_model_optional_targets(needs_transparency_target, needs_volume_target);
        self.render_volume_block_models(
            encoder,
            view,
            editor,
            block_models,
            &frustum,
            include_editor_overlays,
        );
        self.render_fallback_transparent_block_models(
            encoder,
            view,
            editor,
            block_models,
            &frustum,
        );

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Document Transparency and Overlay Render Pass"),
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

        if editor.xray_enabled {
            // X-ray deliberately bypasses scene depth and stays above all
            // composited transparency, matching the previous editor behavior.
            self.draw_document_batches(
                &mut render_pass,
                DocumentRenderStage::Overlay,
                true,
                Some(DocumentPrimitive::Fill),
            );
            self.draw_static_document_strokes(&mut render_pass, true);
            self.draw_document_batches(
                &mut render_pass,
                DocumentRenderStage::Overlay,
                true,
                Some(DocumentPrimitive::Stroke),
            );
        } else {
            // Alpha document primitives test the complete opaque depth buffer
            // but never update it, so farther translucent fills still blend.
            self.draw_document_batches(
                &mut render_pass,
                DocumentRenderStage::Translucent,
                false,
                None,
            );
            self.draw_document_batches(&mut render_pass, DocumentRenderStage::Overlay, false, None);
        }

        if include_editor_overlays
            && !self.dynamic_vertex_buf.is_empty()
            && !self.dynamic_index_buf.is_empty()
        {
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

        if include_editor_overlays
            && !self.overlay_vertex_buf.is_empty()
            && !self.overlay_index_buf.is_empty()
        {
            render_pass.set_pipeline(&self.overlay_render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.overlay_vertex_gpu.slice(..));
            render_pass
                .set_index_buffer(self.overlay_index_gpu.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.overlay_index_buf.len() as u32, 0, 0..1);
        }

        if include_editor_overlays
            && let Err(e) = self.text_system.text_renderer.render(
                &self.text_system.text_atlas,
                &self.text_system.viewport,
                &mut render_pass,
            )
        {
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
        stream_residency: bool,
    ) {
        // Advance cell-pool streaming for the camera's current position before
        // drawing: near bricks become resident, far ones fall back to their
        // aggregates. No-op for volumes that fit the pool. Only the primary
        // viewport streams; preview cameras render from the main viewport's
        // residency so they cannot mutate or thrash it.
        if stream_residency {
            let camera_scene = (self.camera.position - self.scene_origin).as_vec3();
            self.block_model_gpu
                .stream_volumes(&self.queue, camera_scene);
        }

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
        let volume_target = self
            .block_model_volume_target
            .as_ref()
            .expect("visible volume block model must have a volume target");
        let transparency_targets = self
            .block_model_transparency_targets
            .as_ref()
            .expect("visible volume block model must have a depth bind group");
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
            &volume_target.params_buffer,
            0,
            bytemuck::cast_slice(&[scaled_width, scaled_height, 0.0f32, 0.0f32]),
        );

        {
            let mut volume_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block Model Volume Raycast Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &volume_target.view,
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
            volume_pass.set_bind_group(
                2,
                &transparency_targets.transparency_fallback_bind_groups[0],
                &[],
            );
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
        upscale_pass.set_bind_group(0, &volume_target.bind_group, &[]);
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
                            cached.volume.is_none()
                                && cached.transparent_surface_chunks.iter().any(|chunk| {
                                    frustum
                                        .intersects_aabb(chunk.gpu.bounds_min, chunk.gpu.bounds_max)
                                })
                        })
            })
            .collect::<Vec<_>>();
        if visible_transparent_blocks.is_empty() {
            return;
        }
        let transparency_targets = self
            .block_model_transparency_targets
            .as_ref()
            .expect("visible transparent block model must have transparency targets");

        {
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block Model Transparency Clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &transparency_targets.accum_views[0],
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
            let mut transparency_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block Model Order-Independent Transparency Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &transparency_targets.accum_views[0],
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
            transparency_pass.set_pipeline(&self.block_model_transparency_fallback_pipeline);
            transparency_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            transparency_pass.set_bind_group(
                2,
                &transparency_targets.transparency_fallback_bind_groups[0],
                &[],
            );
            for block_model in &visible_transparent_blocks {
                let Some(cached) = self.block_model_gpu.get(block_model.id) else {
                    continue;
                };
                transparency_pass.set_bind_group(1, &cached.surface_style_bind_group, &[]);
                for chunk in &cached.transparent_surface_chunks {
                    if !frustum.intersects_aabb(chunk.gpu.bounds_min, chunk.gpu.bounds_max) {
                        continue;
                    }
                    transparency_pass.set_vertex_buffer(0, chunk.gpu.instance_buffer.slice(..));
                    transparency_pass.draw(0..36, 0..chunk.gpu.instance_count);
                }
            }
        }

        let mut composite_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Block Model Transparency Composite Pass"),
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
        composite_pass.set_pipeline(&self.block_model_transparency_composite_pipeline);
        composite_pass.set_bind_group(0, &transparency_targets.composite_bind_groups[0], &[]);
        composite_pass.draw(0..3, 0..1);
    }
}
