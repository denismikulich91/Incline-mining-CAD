use super::*;
use crate::rendering::scene::{
    build::{
        DocumentSceneBuildInput, DynamicSceneBuildInput, rebuild_document_scene,
        rebuild_dynamic_scene,
    },
    overlays::{OverlaySceneBuildInput, rebuild_editor_overlay},
};

impl<'a> Graphics<'a> {
    pub(crate) fn render(
        &mut self,
        editor: &mut EditorState,
        document: &mut Document,
        triangulations: &[OpenTriangulation],
        block_models: &[crate::model::block_model::OpenBlockModel],
        point_clouds: &[crate::model::point_cloud::OpenPointCloud],
        project: &UiProjectView,
    ) -> Result<UiFrameOutput, RenderSurfaceError> {
        self.vertical_exaggeration = editor.vertical_exaggeration.clamp(0.1, 20.0);
        self.fit_depth_to_scene(
            document,
            triangulations,
            block_models,
            point_clouds,
            &editor.hidden_handles,
        );
        self.include_pending_stroke_in_depth(editor);
        self.include_batter_berm_preview_in_depth(editor);
        self.include_road_preview_in_depth(editor);
        self.upload_camera_uniform(editor.block_model_interaction_resolution_divisor);
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(output)
            | wgpu::CurrentSurfaceTexture::Suboptimal(output) => output,
            wgpu::CurrentSurfaceTexture::Timeout => return Err(RenderSurfaceError::Timeout),
            wgpu::CurrentSurfaceTexture::Occluded => return Err(RenderSurfaceError::Occluded),
            wgpu::CurrentSurfaceTexture::Outdated => return Err(RenderSurfaceError::Outdated),
            wgpu::CurrentSurfaceTexture::Lost => return Err(RenderSurfaceError::Lost),
            wgpu::CurrentSurfaceTexture::Validation => return Err(RenderSurfaceError::Validation),
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let scale_factor = self.window.scale_factor() as f32;
        self.triangulation_gpu.sync(
            &self.device,
            &self.queue,
            self.scene_origin,
            scale_factor,
            triangulations,
            editor,
            &self.surface_style_bind_group_layout,
            &self.surface_chunk_bind_group_layout,
            &self.edge_style_bind_group_layout,
        );
        self.block_model_gpu.sync(
            &self.device,
            &self.queue,
            self.scene_origin,
            scale_factor,
            block_models,
            editor,
            &self.surface_style_bind_group_layout,
            &self.block_model_volume_bind_group_layout,
            &self.edge_style_bind_group_layout,
        );
        self.point_cloud_gpu.sync(
            &self.device,
            &self.queue,
            self.scene_origin,
            scale_factor,
            point_clouds,
            &self.edge_style_bind_group_layout,
        );
        let needs_geometry_rebuild = self.geometry_dirty
            || self.cached_document_revision != document.revision()
            || (self.cached_scale_factor - scale_factor).abs() > f32::EPSILON;

        if needs_geometry_rebuild {
            // Reconcile the static stroke chunks first so the stream rebuild
            // below knows which objects they own and can skip them.
            self.static_strokes.sync(
                &self.device,
                &self.queue,
                document,
                editor,
                self.scene_origin,
                scale_factor,
            );
            rebuild_document_scene(DocumentSceneBuildInput {
                editor,
                document,
                static_ids: self.static_strokes.claimed(),
                text_system: &mut self.text_system,
                lyon_buffer: &mut self.lyon_buffer,
                stroke_vertex_buf: &mut self.stroke_vertex_buf,
                stroke_index_buf: &mut self.stroke_index_buf,
                cached_textareas: &mut self.cached_textareas,
                textarea_depths: &mut self.textarea_depths,
                pick_records: &mut self.pick_records,
                text_pick_records: &mut self.text_pick_records,
                scene_origin: self.scene_origin,
                scale_factor,
            });

            self.upload_scene_stream_buffers();

            self.cached_scale_factor = scale_factor;
            self.cached_document_revision = document.revision();
            self.geometry_dirty = false;
        }

        // Per-frame pass for the live drawing tools; the static scene above no
        // longer rebuilds while they run. Rebuilt while a tool is active and
        // once more after it deactivates (to clear the buffers).
        let dynamic_active = editor.active_tool == crate::ui::state::ActiveTool::MakeRoad
            || editor.batter_berm_dialog_open;
        if dynamic_active || !self.dynamic_vertex_buf.is_empty() {
            rebuild_dynamic_scene(DynamicSceneBuildInput {
                editor,
                document,
                dynamic_vertex_buf: &mut self.dynamic_vertex_buf,
                dynamic_index_buf: &mut self.dynamic_index_buf,
                scene_origin: self.scene_origin,
                scale_factor,
            });
            Self::clamp_stream_geometry(
                &self.device,
                &mut self.dynamic_vertex_buf,
                &mut self.dynamic_index_buf,
                "Dynamic Scene Buffer",
            );
            if !self.dynamic_vertex_buf.is_empty() {
                Self::ensure_stream_capacity(
                    &self.device,
                    &mut self.dynamic_vertex_gpu,
                    &mut self.dynamic_vertex_capacity,
                    self.dynamic_vertex_buf.len(),
                    std::mem::size_of::<StrokeVertex>(),
                    wgpu::BufferUsages::VERTEX,
                    "Dynamic Scene Vertex Buffer",
                );
                self.queue.write_buffer(
                    &self.dynamic_vertex_gpu,
                    0,
                    bytemuck::cast_slice(&self.dynamic_vertex_buf),
                );
            }
            if !self.dynamic_index_buf.is_empty() {
                Self::ensure_stream_capacity(
                    &self.device,
                    &mut self.dynamic_index_gpu,
                    &mut self.dynamic_index_capacity,
                    self.dynamic_index_buf.len(),
                    std::mem::size_of::<u32>(),
                    wgpu::BufferUsages::INDEX,
                    "Dynamic Scene Index Buffer",
                );
                self.queue.write_buffer(
                    &self.dynamic_index_gpu,
                    0,
                    bytemuck::cast_slice(&self.dynamic_index_buf),
                );
            }
        }

        let measurement_state = (
            matches!(
                editor.active_tool,
                crate::ui::state::ActiveTool::MeasureDistance
                    | crate::ui::state::ActiveTool::MeasureBermAngle
            ),
            editor.measurement_start,
            editor.measurement_end,
            editor.berm_angle_points.clone(),
        );
        if measurement_state != self.cached_measurement_state {
            self.cached_measurement_state = measurement_state;
            self.overlay_dirty = true;
        }
        if editor.poly_finish_dialog != self.cached_poly_finish_dialog {
            self.cached_poly_finish_dialog = editor.poly_finish_dialog;
            self.overlay_dirty = true;
        }

        if self.overlay_dirty {
            let overlay_vp = self.view_proj();
            let overlay_screen = self.screen_size();
            rebuild_editor_overlay(OverlaySceneBuildInput {
                editor,
                document,
                overlay_vertex_buf: &mut self.overlay_vertex_buf,
                overlay_index_buf: &mut self.overlay_index_buf,
                view_proj: overlay_vp,
                screen_size: overlay_screen,
                scene_origin: self.scene_origin,
                scale_factor,
            });

            Self::clamp_stream_geometry(
                &self.device,
                &mut self.overlay_vertex_buf,
                &mut self.overlay_index_buf,
                "Editor Overlay Buffer",
            );
            if !self.overlay_vertex_buf.is_empty() {
                Self::ensure_stream_capacity(
                    &self.device,
                    &mut self.overlay_vertex_gpu,
                    &mut self.overlay_vertex_capacity,
                    self.overlay_vertex_buf.len(),
                    std::mem::size_of::<StrokeVertex>(),
                    wgpu::BufferUsages::VERTEX,
                    "Editor Overlay Vertex Buffer",
                );
                self.queue.write_buffer(
                    &self.overlay_vertex_gpu,
                    0,
                    bytemuck::cast_slice(&self.overlay_vertex_buf),
                );
            }
            if !self.overlay_index_buf.is_empty() {
                Self::ensure_stream_capacity(
                    &self.device,
                    &mut self.overlay_index_gpu,
                    &mut self.overlay_index_capacity,
                    self.overlay_index_buf.len(),
                    std::mem::size_of::<u32>(),
                    wgpu::BufferUsages::INDEX,
                    "Editor Overlay Index Buffer",
                );
                self.queue.write_buffer(
                    &self.overlay_index_gpu,
                    0,
                    bytemuck::cast_slice(&self.overlay_index_buf),
                );
            }
            self.overlay_dirty = false;
        }

        // Glyph vertices are stored pre-view_proj (the text shader applies the
        // camera from the params uniform), so prepare only has to run when the
        // text areas changed — camera-only frames reuse the prepared buffers.
        if needs_geometry_rebuild || self.text_prepare_pending {
            // Trim resets the atlas's in-use protection set; prepare rebuilds
            // it, so the two must run in the same frame or a later prepare
            // could evict glyphs the live vertex buffer still references.
            self.text_system.text_atlas.trim();

            let text_areas: Vec<TextArea> = self
                .cached_textareas
                .iter()
                .filter_map(|c: &CachedTextArea| c.text_area(&self.text_system.text_cache))
                .collect();

            if let Err(e) = self.text_system.text_renderer.prepare(
                &self.device,
                &self.queue,
                &mut self.text_system.font_system,
                &mut self.text_system.text_atlas,
                &self.text_system.viewport,
                text_areas,
                &mut self.text_system.swash_cache,
            ) {
                log::error!("Text prepare failed: {e:?}");
            }
            self.text_prepare_pending = false;
            if needs_geometry_rebuild
                && self
                    .frame_index
                    .is_multiple_of(TEXT_CACHE_TRIM_INTERVAL_FRAMES)
            {
                self.text_system.text_cache.trim();
            }
        }

        self.render_scene_pass(
            &mut encoder,
            &view,
            editor,
            triangulations,
            block_models,
            point_clouds,
        );

        self.update_tool_projections(editor, document);

        let orbit_marker_screen = self.orbit_marker_screen_pos();
        let camera_forward = self.camera.forward();
        let camera_up = self.camera.up();
        let ui_output = self.gui.render(
            &self.window,
            &self.device,
            &self.queue,
            &mut encoder,
            &view,
            editor,
            document,
            project,
            block_models,
            [self.size.width, self.size.height],
            orbit_marker_screen,
            [
                camera_forward.x as f32,
                camera_forward.y as f32,
                camera_forward.z as f32,
            ],
            [camera_up.x as f32, camera_up.y as f32, camera_up.z as f32],
        );
        if ui_output.geometry_dirty {
            self.invalidate_geometry();
        }
        // Dragging a UI widget that affects the render (e.g. a colour-gradient
        // stop) starts the interaction cooldown so the volume raycaster drops
        // to its low-quality path, like camera drags / zoom / resize. The
        // volume was already drawn this frame, so this takes effect next frame
        // — fine, since dragging spans many frames.
        if ui_output.ui_pointer_active {
            self.mark_interaction();
        }
        if ui_output.repaint {
            self.window.request_redraw();
        }
        // Keep redrawing through the interaction cooldown so the volume
        // raycaster's reduced-quality frames are always followed by a
        // full-quality one once the camera settles / resizing stops.
        if self.interaction_active() {
            self.window.request_redraw();
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        self.frame_index = self.frame_index.wrapping_add(1);

        Ok(ui_output)
    }
}
