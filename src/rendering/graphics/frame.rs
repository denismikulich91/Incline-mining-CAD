use super::*;
use crate::rendering::scene::{
    build::{
        DocumentSceneBuildInput, DynamicSceneBuildInput, rebuild_document_scene,
        rebuild_dynamic_scene,
    },
    overlays::{OverlaySceneBuildInput, rebuild_editor_overlay},
};

pub(crate) struct RenderInput<'frame> {
    pub(crate) editor: &'frame mut EditorState,
    pub(crate) document: &'frame mut Document,
    pub(crate) triangulations: &'frame [OpenTriangulation],
    pub(crate) block_models: &'frame [crate::model::block_model::OpenBlockModel],
    pub(crate) point_clouds: &'frame [crate::model::point_cloud::OpenPointCloud],
    pub(crate) rasters: &'frame [OpenRasterTexture],
    pub(crate) project: &'frame UiProjectView,
}

impl<'a> Graphics<'a> {
    pub(crate) fn render(
        &mut self,
        input: RenderInput<'_>,
    ) -> Result<UiFrameOutput, RenderSurfaceError> {
        let RenderInput {
            editor,
            document,
            triangulations,
            block_models,
            point_clouds,
            rasters,
            project,
        } = input;
        self.vertical_exaggeration = editor.vertical_exaggeration.clamp(0.1, 20.0);
        let slice_visible_half_length =
            slice_visible_half_length(self.projection.zoom, self.screen_size());
        if let Some(slice) = self.slice_view.as_mut() {
            // Slice mode owns the clip planes: the symmetric depth extent *is*
            // the slab, so the scene-fitting passes below must not run — they
            // would blow the clip range back out to the scene bounds.
            slice.width = editor.slice_width_input.clamp(0.1, 1.0e6);
            slice.move_speed = editor.slice_speed_input.clamp(0.0, 1.0e6);
            slice.rotate_speed = editor.slice_rotate_input.clamp(1.0, 720.0).to_radians();
            self.projection
                .set_symmetric_depth_extent(slice.width * 0.5);
            editor.slice_center = [slice.center.x, slice.center.y, slice.center.z];
            editor.slice_direction = [slice.direction.x, slice.direction.y];
            editor.slice_half_length = slice_visible_half_length;
        } else {
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
        }
        editor.debug_clip_plane_distances = Some(self.projection.clip_planes());
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
        // Non-sRGB view of the same texture for the egui pass; egui applies
        // gamma itself, so it wants raw byte writes (see `Gui::new`).
        let gui_view = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.config.format.remove_srgb_suffix()),
            ..Default::default()
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let scale_factor = self.window.scale_factor() as f32;
        self.raster_gpu.sync(
            &self.device,
            &self.queue,
            self.scene_origin,
            rasters,
            &self.raster_surface_bind_group_layout,
        );
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
            glam::Mat4::from_cols_array_2d(&self.camera_uniform.view_proj),
            (self.camera.position - self.scene_origin).as_vec3(),
            point_clouds,
            &self.edge_style_bind_group_layout,
        );
        let render_style_key = editor.render_style_key();
        if self.cached_render_style_key != Some(render_style_key) {
            self.cached_render_style_key = Some(render_style_key);
            self.geometry_dirty = true;
            self.overlay_dirty = true;
        }
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
                document_draw_batches: &mut self.document_draw_batches,
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
                    .wrapping_sub(self.last_text_cache_trim_frame)
                    >= TEXT_CACHE_TRIM_INTERVAL_FRAMES
            {
                self.text_system.text_cache.trim();
                self.last_text_cache_trim_frame = self.frame_index;
            }
        }

        self.render_scene_pass(
            &mut encoder,
            &view,
            editor,
            triangulations,
            block_models,
            point_clouds,
            rasters,
            true,
        );

        // One-shot viewport export: re-render the scene (without the egui
        // chrome) into an offscreen texture and queue a readback on this
        // frame's encoder; the PNG is written after submit below.
        let pending_screenshot = self.pending_screenshot.take().map(|path| {
            self.encode_screenshot_capture(
                &mut encoder,
                editor,
                triangulations,
                block_models,
                point_clouds,
                rasters,
                path,
            )
        });

        // Render the in-viewport plan preview through the same shaded scene
        // pass as the main and detached viewports. The egui panel samples this
        // offscreen texture below.
        self.render_embedded_slice_preview(
            document,
            triangulations,
            block_models,
            point_clouds,
            rasters,
            editor,
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
            &gui_view,
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
        // Keep redrawing through the interaction cooldown so the volume
        // raycaster's reduced-quality frames are always followed by a
        // full-quality one once the camera settles / resizing stops.
        if self.interaction_active() {
            self.window.request_redraw();
        }
        // Continue draining the bounded point-cloud upload queue without
        // treating background upload progress as camera interaction.
        if self.point_cloud_gpu.has_pending_uploads() || self.block_model_gpu.has_pending_builds() {
            self.window.request_redraw();
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        if let Some(capture) = pending_screenshot {
            self.finish_screenshot_capture(capture);
        }
        self.frame_index = self.frame_index.wrapping_add(1);

        Ok(ui_output)
    }
}
