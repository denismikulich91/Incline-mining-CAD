use std::time::{Duration, Instant};

use winit::{
    event::*,
    keyboard::{KeyCode, PhysicalKey},
    window::CursorIcon,
};

use crate::{
    app::App, rendering::graphics::RenderSurfaceError, ui::state::ActiveTool, userspace_error,
    userspace_log,
};

const RIGHT_CLICK_DRAG_THRESHOLD_PX: f32 = 3.0;

impl<'a> App<'a> {
    pub(crate) fn handle_window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if self
            .graphics
            .as_ref()
            .and_then(|graphics| graphics.slice_preview_window_id())
            == Some(window_id)
        {
            self.handle_slice_preview_event(event);
            return;
        }
        // A dropped secondary window can still have a final queued event.
        // Never feed such an event into the root window's egui/camera state.
        if self.window.as_ref().map(|window| window.id()) != Some(window_id) {
            return;
        }
        if let WindowEvent::ModifiersChanged(modifiers) = &event {
            self.modifiers = modifiers.state();
        }

        if let WindowEvent::Focused(focused) = &event {
            self.window_focused = *focused;
            if !focused {
                self.finish_left_button_interactions();
                if let Some(graphics) = self.graphics.as_mut() {
                    graphics.release_mouse_capture();
                    graphics.release_slice_keys();
                }
            }
            self.redraw_requested = true;
        }

        if let WindowEvent::MouseWheel { .. } = &event {
            self.last_scroll_instant = Some(Instant::now());
        }

        if self.handle_cursor_mode_cycle(&event) {
            return;
        }

        let gui_response = self
            .graphics
            .as_mut()
            .map(|graphics| graphics.gui_input(&event));
        if gui_response.is_some_and(|response| response.repaint) {
            self.redraw_requested = true;
        }

        if let Some(graphics) = self.graphics.as_mut() {
            graphics.set_fly_mode_enabled(self.editor.fly_mode_enabled);
        }

        let gui_consumed = gui_response.is_some_and(|response| response.consumed);

        // Slice mode claims plain W/S (slab forward/back) and Q/E (rotate the
        // line). Presses are dropped when the GUI consumed the event (e.g.
        // typing in the slice panel), but releases always pass through so
        // keys can't get stuck held.
        if self.editor.slice_mode_enabled
            && !self.modifiers.control_key()
            && !(cfg!(target_os = "macos") && self.modifiers.super_key())
            && let WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        physical_key:
                            PhysicalKey::Code(
                                key @ (KeyCode::KeyW
                                | KeyCode::KeyS
                                | KeyCode::KeyQ
                                | KeyCode::KeyE),
                            ),
                        ..
                    },
                ..
            } = &event
            && (!gui_consumed || *state == ElementState::Released)
            && let Some(graphics) = self.graphics.as_mut()
        {
            graphics.slice_process_key(*key, *state == ElementState::Pressed);
            self.redraw_requested = true;
        }

        if !gui_consumed {
            let suppress_fly_canvas_click = (self.editor.fly_mode_enabled
                || self.editor.slice_mode_enabled)
                && matches!(
                    event,
                    WindowEvent::MouseInput {
                        button: MouseButton::Left,
                        ..
                    }
                );
            if !suppress_fly_canvas_click {
                self.handle_mouse_press(&event);
                self.handle_mouse_release(&event);
            }
            self.handle_right_press_for_orbit(&event);
            self.handle_right_release(&event);
            self.handle_control_shortcuts(&event);
        } else if is_left_mouse_release(&event) {
            self.finish_left_button_interactions();
        }

        let graphics_consumed = self.graphics.as_mut().is_some_and(|graphics| {
            if gui_consumed && !graphics.should_receive_fly_event(&event) {
                return false;
            }

            let consumed = graphics.input(&event);
            if consumed {
                self.redraw_requested = true;
            }
            consumed
        });
        let input_consumed = gui_consumed || graphics_consumed;

        if !input_consumed {
            match event {
                WindowEvent::CloseRequested => {
                    if let Err(error) = self.request_exit() {
                        userspace_error!("Couldn't exit: {}", error);
                    }
                }
                WindowEvent::KeyboardInput { .. } => self.handle_key_action(&event),
                WindowEvent::Resized(physical_size)
                    if physical_size.width > 0 && physical_size.height > 0 =>
                {
                    self.pending_resize = Some(physical_size);
                    self.redraw_requested = true;
                }
                WindowEvent::RedrawRequested => {
                    self.poll_triangulation_loads();
                    self.poll_block_model_loads();
                    self.poll_point_cloud_loads();
                    self.poll_raster_loads();
                    self.poll_saves();
                    self.poll_jobs();
                    let now = Instant::now();
                    let frame_interval = if self.pending_resize.is_some() {
                        super::rate_interval(self.editor.resize_frame_rate_cap)
                    } else {
                        super::rate_interval(self.editor.frame_rate_cap)
                    };
                    if self
                        .last_render_time
                        .is_some_and(|last_render| now.duration_since(last_render) < frame_interval)
                    {
                        // Winit/compositors can issue redraw events directly. Defer
                        // early events so both normal and resize rendering obey the
                        // configured frame limits.
                        self.redraw_requested = true;
                        return;
                    }
                    // This event satisfies any queued redraw request, including
                    // redraws generated directly by the compositor during resize.
                    self.redraw_requested = false;
                    let project = self.project_view();
                    let completing_topology_load = self.topology_uploads_pending();
                    let applied_resize = self.pending_resize.take();
                    if let Some(graphics) = self.graphics.as_mut() {
                        if let Some(size) = applied_resize {
                            graphics.resize(size);
                        }
                        let last_render_time = *self.last_render_time.get_or_insert(now);
                        let dt = now - last_render_time;
                        self.last_render_time = Some(now);
                        if !dt.is_zero() {
                            let instantaneous_fps = 1.0 / dt.as_secs_f32();
                            self.editor.measured_fps =
                                Some(self.editor.measured_fps.map_or(instantaneous_fps, |fps| {
                                    fps * 0.9 + instantaneous_fps * 0.1
                                }));
                        }
                        graphics.update(dt, self.editor.block_model_interaction_resolution_divisor);
                        self.editor.can_undo = self.history.can_undo();
                        self.editor.can_redo = self.history.can_redo();
                        match graphics.render(crate::rendering::graphics::frame::RenderInput {
                            editor: &mut self.editor,
                            document: &mut self.scene_document,
                            triangulations: &self.triangulations,
                            block_models: &self.block_models,
                            point_clouds: &self.point_clouds,
                            rasters: &self.raster_textures,
                            project: &project,
                        }) {
                            Ok(ui_output) => {
                                self.render_validation_recovery_attempts = 0;
                                self.next_ui_repaint_deadline = ui_output
                                    .repaint_after
                                    .and_then(|delay| Instant::now().checked_add(delay));
                                if ui_output.repaint_after.is_some_and(|delay| delay.is_zero()) {
                                    self.redraw_requested = true;
                                }
                                if let Some(graphics) = self.graphics.as_mut() {
                                    graphics.set_fly_mode_enabled(self.editor.fly_mode_enabled);
                                    self.editor.debug_chunk_stats =
                                        Some(graphics.chunk_render_stats);
                                }
                                if completing_topology_load
                                    && !self.graphics.as_ref().is_some_and(|graphics| {
                                        graphics.point_cloud_uploads_pending()
                                    })
                                {
                                    self.finish_topology_load();
                                }
                                if self.background_tasks_pending()
                                    && let Some(window) = &self.window
                                {
                                    window.set_cursor(CursorIcon::Progress);
                                }
                                self.handle_ui_commands(ui_output.commands);
                                self.sync_slice_preview_window(event_loop);
                                if let Some(graphics) = self.graphics.as_mut() {
                                    graphics.request_slice_preview_redraw_if_scene_changed(
                                        &self.editor,
                                        &self.scene_document,
                                        &self.triangulations,
                                        &self.block_models,
                                        &self.point_clouds,
                                        &self.raster_textures,
                                    );
                                }
                                // Keep move panel preview in sync — apply whenever the panel delta
                                // differs from the last applied preview (catches typed values that
                                // don't always trigger changed() on every frame).
                                if self.editor.active_tool == ActiveTool::Move
                                    && self.gizmo_drag.is_none()
                                    && !self.editor.selected_handles.is_empty()
                                    && self.editor.move_panel_delta
                                        != self.editor.move_panel_last_preview
                                {
                                    let d = self.editor.move_panel_delta;
                                    self.ensure_move_session_original();
                                    self.preview_move_delta(glam::DVec3::new(d[0], d[1], d[2]));
                                    self.editor.move_panel_last_preview = d;
                                    self.invalidate_geometry();
                                }
                                // Clean up per-tool state when the tool is switched via the
                                // toolbar.
                                if self.editor.active_tool != ActiveTool::Move
                                    && self.has_pending_move_delta()
                                {
                                    self.editor.move_panel_last_preview = [f64::NAN; 3];
                                    self.cancel_move_delta();
                                }
                                if self.editor.active_tool != ActiveTool::ExplodePolygon
                                    && self.editor.tool_highlight_id.is_some()
                                    && self.editor.offset_target_ids.is_empty()
                                    && self.editor.batter_berm_target_id.is_none()
                                    && !self.editor.tri_cut_poly_open
                                {
                                    self.editor.tool_highlight_id = None;
                                    self.invalidate_geometry();
                                }
                                // Update batter berm preview every frame while dialog is open.
                                if self.editor.batter_berm_dialog_open {
                                    self.update_batter_berm_preview();
                                    self.invalidate_overlay();
                                }
                                // Clear in-progress stroke when switching away from drawing tools.
                                if !matches!(
                                    self.editor.active_tool,
                                    ActiveTool::MakeLine
                                        | ActiveTool::MakePoly
                                        | ActiveTool::MakeRoad
                                ) && !self.editor.pending_stroke.is_empty()
                                {
                                    self.editor.pending_stroke.clear();
                                    self.invalidate_overlay();
                                }
                                // Clear the pending slice line when switching
                                // away from the slice tool.
                                if self.editor.active_tool != ActiveTool::VerticalSlice
                                    && self.editor.slice_pending_start.is_some()
                                {
                                    self.editor.slice_pending_start = None;
                                    self.invalidate_overlay();
                                }
                                // Cancel fuse state when switching away from the fuse tool.
                                if self.editor.active_tool != ActiveTool::FuseIntoPolygon
                                    && (self.editor.fuse_awaiting_endpoint.is_some()
                                        || !self.editor.fuse_segments.is_empty())
                                {
                                    self.cancel_fuse();
                                }
                                // Auto-initiate fuse from an existing selection when the fuse
                                // tool is first activated with exactly one selected line/polygon.
                                if self.editor.active_tool == ActiveTool::FuseIntoPolygon
                                    && self.editor.fuse_awaiting_endpoint.is_none()
                                    && self.editor.fuse_segments.is_empty()
                                {
                                    self.fuse_init_from_selection();
                                }
                                if self.editor.active_tool == ActiveTool::SplitAtPoints
                                    && self.editor.split_poly_id.is_none()
                                {
                                    self.split_init_from_selection();
                                }
                            }
                            Err(RenderSurfaceError::Lost | RenderSurfaceError::Outdated) => {
                                graphics.reconfigure();
                                self.redraw_requested = true;
                            }
                            Err(RenderSurfaceError::Timeout | RenderSurfaceError::Occluded) => {
                                self.redraw_requested = true;
                            }
                            Err(RenderSurfaceError::Validation) => {
                                // Bounded surface/device recovery first; only
                                // repeated failures without one good frame in
                                // between are treated as fatal.
                                const MAX_VALIDATION_RECOVERY_ATTEMPTS: u32 = 3;
                                if self.render_validation_recovery_attempts
                                    < MAX_VALIDATION_RECOVERY_ATTEMPTS
                                {
                                    self.render_validation_recovery_attempts += 1;
                                    log::warn!(
                                        "Renderer validation error; attempting surface recovery ({}/{})",
                                        self.render_validation_recovery_attempts,
                                        MAX_VALIDATION_RECOVERY_ATTEMPTS
                                    );
                                    graphics.reconfigure();
                                    self.redraw_requested = true;
                                } else {
                                    self.begin_fatal_shutdown("renderer validation error");
                                }
                            }
                        }
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    self.editor.cursor_screen_px = Some((position.x as f32, position.y as f32));
                    if self.editor.selection_box_start_px.is_some() {
                        self.editor.selection_box_current_px = self.editor.cursor_screen_px;
                    }
                    if self
                        .graphics
                        .as_mut()
                        .is_some_and(|g| g.set_mouse_location(position.into()))
                    {
                        self.redraw_requested = true;
                    }
                    self.maybe_start_right_orbit_drag();
                    let z = self.editor.z_level;
                    let raw = self.graphics.as_ref().and_then(|g| g.cursor_world(z));
                    let is_drawing_tool = matches!(
                        self.editor.active_tool,
                        ActiveTool::MakePoint
                            | ActiveTool::MakeLine
                            | ActiveTool::MakePoly
                            | ActiveTool::MakeText
                            | ActiveTool::MeasureDistance
                            | ActiveTool::MeasureBermAngle
                            | ActiveTool::MakeRoad
                            | ActiveTool::VerticalSlice
                    );
                    let is_scrolling = self
                        .last_scroll_instant
                        .is_some_and(|t| t.elapsed() < Duration::from_millis(250));
                    let snap_mode_enabled = matches!(
                        self.editor.cursor_mode,
                        crate::ui::state::CursorMode::SnapToPoint
                            | crate::ui::state::CursorMode::SnapToLine
                            | crate::ui::state::CursorMode::SnapToSurface
                    );
                    let camera_active =
                        self.graphics.as_ref().is_some_and(|g| g.is_camera_active());
                    let snap_eligible =
                        is_drawing_tool && snap_mode_enabled && !camera_active && !is_scrolling;
                    let now = Instant::now();
                    let snap_poll_due = self.last_snap_poll_instant.is_none_or(|last_poll| {
                        now.duration_since(last_poll)
                            >= super::rate_interval(self.editor.snap_poll_rate)
                    });
                    let snapped = if snap_eligible && snap_poll_due {
                        self.last_snap_poll_instant = Some(now);
                        self.refresh_snap_index();
                        // Line snapping targets resolved road centerlines;
                        // hand it the cached ghost-free network so the snap
                        // query never re-resolves the whole network per poll.
                        if self.editor.cursor_mode == crate::ui::state::CursorMode::SnapToLine {
                            self.refresh_scene_road_network();
                        }
                        let document = &self.scene_document;
                        self.graphics.as_ref().and_then(|g| {
                            g.snap_cursor(
                                document,
                                &self.snap_index,
                                self.scene_road_network.as_ref(),
                                &self.triangulations,
                                &self.editor.hidden_handles,
                                &self.editor.frozen_handles,
                                &self.editor.cursor_mode,
                                self.editor.xray_enabled,
                            )
                        })
                    } else if snap_eligible && self.editor.cursor_snapped {
                        self.editor.cursor_world
                    } else {
                        None
                    };
                    let was_snapped = self.editor.cursor_snapped;
                    // During a camera drag the mouse steers the view, not the
                    // cursor: keep the last world cursor (and its snapped flag)
                    // so in-progress tool previews don't chase the unprojection
                    // of a rotating camera.
                    let effective = if camera_active {
                        self.editor.cursor_world
                    } else {
                        snapped.or(raw)
                    };
                    self.editor.cursor_snapped = if camera_active {
                        was_snapped
                    } else {
                        snapped.is_some()
                    };
                    let cursor_world_changed = self.editor.cursor_world != effective;
                    if cursor_world_changed {
                        self.editor.cursor_world = effective;
                        self.redraw_requested = true;
                    }
                    self.drag_to_cursor();
                    if self.gizmo_drag.is_some() {
                        self.move_gizmo_to_cursor();
                        self.invalidate_overlay();
                    }
                    if self.editor.active_tool == ActiveTool::Move
                        && let Some(cursor_px) = self.editor.cursor_screen_px
                    {
                        let new_hover = hit_gizmo_axis(&self.editor, cursor_px).map(|(idx, _)| idx);
                        if self.editor.move_gizmo_hovered_axis != new_hover {
                            self.editor.move_gizmo_hovered_axis = new_hover;
                            self.invalidate_overlay();
                        }
                    }
                    // Chamfer gizmo: hover detection and radius drag
                    if self.editor.active_tool == ActiveTool::Chamfer {
                        // If no corner has been picked yet, show a hover sphere on the nearest
                        // valid vertex.
                        if self.editor.chamfer_corner_index.is_none()
                            && let Some(cursor_px) = self.editor.cursor_screen_px
                        {
                            let vp = self
                                .graphics
                                .as_ref()
                                .map(|g| g.view_proj())
                                .unwrap_or_default();
                            let screen = self
                                .graphics
                                .as_ref()
                                .map(|g| g.screen_size_pub())
                                .unwrap_or_default();
                            let nearest = crate::rendering::pick::pick_nearest_vertex(
                                &self.scene_document,
                                &self.editor.hidden_handles,
                                &self.editor.frozen_handles,
                                &vp,
                                screen,
                                cursor_px,
                                crate::app::PICK_THRESHOLD_PX * 2.5,
                            );
                            let hover_px = nearest.and_then(|(oid, _vi, world)| {
                                let is_closed_polygon =
                                    self.scene_document.get_object(oid).is_some_and(|o| {
                                        matches!(
                                            o,
                                            crate::model::Object::Polyline {
                                                verts,
                                                closed: true,
                                                ..
                                            } if verts.len() >= 3
                                        )
                                    });
                                if !is_closed_polygon {
                                    return None;
                                }
                                crate::rendering::pick::world_to_screen(&vp, world, screen)
                                    .map(|s| (s.x as f32, s.y as f32))
                            });
                            if hover_px != self.editor.chamfer_hover_corner_px {
                                self.editor.chamfer_hover_corner_px = hover_px;
                                self.invalidate_overlay();
                            }
                        }
                    }
                    let hover_pick_due = snap_poll_due
                        && (self.editor.active_tool == ActiveTool::DeleteElement
                            || self.editor.active_tool == ActiveTool::ExplodePolygon
                            || self.editor.relimit_waiting_for_pick
                            || self.editor.relimit_confirming_end);
                    if hover_pick_due {
                        self.last_snap_poll_instant = Some(now);
                    }
                    if self.editor.active_tool == ActiveTool::DeleteElement && hover_pick_due {
                        self.update_delete_hover_vertex();
                    } else if self.editor.active_tool != ActiveTool::DeleteElement
                        && self.editor.delete_hover_vertex_px.is_some()
                    {
                        self.editor.delete_hover_vertex_px = None;
                        self.invalidate_overlay();
                    }
                    if let Some(cursor_px) = self.editor.cursor_screen_px {
                        // Drag: update radius from cursor offset along edge screen direction
                        if let Some(start_px) = self.editor.chamfer_gizmo_drag_start_px {
                            if let Some(dir) = self.editor.chamfer_gizmo_edge_screen_dir {
                                let dx = cursor_px.0 - start_px.0;
                                let dy = cursor_px.1 - start_px.1;
                                let dot = dx * dir.0 + dy * dir.1;
                                let new_radius = (self.editor.chamfer_gizmo_drag_start_radius
                                    + dot as f64 / self.editor.chamfer_gizmo_px_per_world)
                                    .max(0.001)
                                    .min(self.editor.chamfer_max_radius);
                                self.editor.chamfer_radius = new_radius;
                                self.invalidate_overlay();
                            }
                        } else if let (Some(corner), Some(handle)) = (
                            self.editor.chamfer_gizmo_corner_px,
                            self.editor.chamfer_gizmo_handle_px,
                        ) {
                            // Hover detection: anywhere along the visible arrow shaft.
                            // When radius≈0 the handle coincides with the corner, so use
                            // the stub direction to compute the visual tip (matches UI drawing).
                            const STUB_LEN: f32 = 40.0;
                            let threshold = 10.0f32;
                            let near = {
                                let raw_ax = handle.0 - corner.0;
                                let raw_ay = handle.1 - corner.1;
                                let raw_len = (raw_ax * raw_ax + raw_ay * raw_ay).sqrt();
                                // Compute the visual tip the same way the UI does.
                                let (tip_x, tip_y) = if raw_len > STUB_LEN {
                                    (handle.0, handle.1)
                                } else {
                                    // Use stub dir (edge direction) when handle is near corner.
                                    let (dx, dy) = if raw_len > 4.0 {
                                        (raw_ax / raw_len, raw_ay / raw_len)
                                    } else if let Some(ed) = self.editor.chamfer_gizmo_bisector_px {
                                        let el = (ed.0 * ed.0 + ed.1 * ed.1).sqrt().max(1e-6);
                                        (ed.0 / el, ed.1 / el)
                                    } else {
                                        (1.0, 0.0)
                                    };
                                    (corner.0 + dx * STUB_LEN, corner.1 + dy * STUB_LEN)
                                };
                                // Point-to-segment distance from cursor to corner→tip.
                                let ax = tip_x - corner.0;
                                let ay = tip_y - corner.1;
                                let bx = cursor_px.0 - corner.0;
                                let by = cursor_px.1 - corner.1;
                                let seg_len_sq = ax * ax + ay * ay;
                                let dist_sq = if seg_len_sq < 1e-6 {
                                    bx * bx + by * by
                                } else {
                                    let t = ((bx * ax + by * ay) / seg_len_sq).clamp(0.0, 1.0);
                                    let px = bx - t * ax;
                                    let py = by - t * ay;
                                    px * px + py * py
                                };
                                dist_sq < threshold * threshold
                            };
                            if self.editor.chamfer_gizmo_hovered != near {
                                self.editor.chamfer_gizmo_hovered = near;
                                self.invalidate_overlay();
                            }
                        }
                    }
                    // Bezier tool: hover detection and CP drag
                    if self.editor.active_tool == ActiveTool::Bezier
                        && let Some(cursor_px) = self.editor.cursor_screen_px
                    {
                        // Update CP drag position using cursor_world
                        if let Some(cp_idx) = self.editor.bezier_dragging_cp
                            && let Some(world) = self
                                .graphics
                                .as_ref()
                                .and_then(|g| g.cursor_world(self.editor.z_level))
                        {
                            if cp_idx == 0 {
                                self.editor.bezier_cp1 = [world.x, world.y, world.z];
                            } else {
                                self.editor.bezier_cp2 = [world.x, world.y, world.z];
                            }
                        }
                        // Update hover state
                        let hover = bezier_cp_hit(&self.editor, cursor_px);
                        if hover != self.editor.bezier_hover_cp {
                            self.editor.bezier_hover_cp = hover;
                            self.invalidate_overlay();
                        }
                    }
                    if hover_pick_due {
                        self.update_relimit_hover_end();
                    }
                    self.update_offset_preview();
                    let road_preview_due =
                        self.last_road_preview_update_instant
                            .is_none_or(|last_update| {
                                now.duration_since(last_update)
                                    >= super::rate_interval(self.editor.snap_poll_rate)
                            });
                    if self.editor.active_tool == ActiveTool::MakeRoad
                        && cursor_world_changed
                        && road_preview_due
                    {
                        self.last_road_preview_update_instant = Some(now);
                        self.update_road_preview();
                    }
                    if self.editor.active_tool == ActiveTool::ExplodePolygon && hover_pick_due {
                        self.update_explode_hover();
                    }
                    if !self.editor.pending_stroke.is_empty()
                        || self.editor.cursor_snapped
                        || was_snapped
                        || self.editor.relimit_confirming_end
                        || self.editor.relimit_waiting_for_pick
                        || self.editor.offset_awaiting_side_pick
                        || self.gizmo_drag.is_some()
                        || self.editor.active_tool == ActiveTool::Move
                        || self.editor.active_tool == ActiveTool::MeasureDistance
                        || self.editor.active_tool == ActiveTool::MeasureBermAngle
                        || self.editor.active_tool == ActiveTool::DeleteElement
                        || self.editor.active_tool == ActiveTool::Chamfer
                        || self.editor.active_tool == ActiveTool::MakeRoad
                        || self.editor.active_tool == ActiveTool::Bezier
                        || self.editor.slice_pending_start.is_some()
                    {
                        self.invalidate_overlay();
                    }
                }
                _ => {}
            }
        }
    }

    fn handle_slice_preview_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
            }
            WindowEvent::Focused(false) => {
                if let Some(graphics) = self.graphics.as_mut() {
                    graphics.release_slice_keys();
                }
                self.redraw_requested = true;
            }
            WindowEvent::CloseRequested => {
                if let Some(graphics) = self.graphics.as_mut() {
                    graphics.close_slice_preview();
                }
                self.editor.slice_preview_detached = false;
                self.redraw_requested = true;
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => {
                if let Some(graphics) = self.graphics.as_mut() {
                    graphics.close_slice_preview();
                }
                self.editor.slice_preview_detached = false;
                self.redraw_requested = true;
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        physical_key:
                            PhysicalKey::Code(
                                key @ (KeyCode::KeyW
                                | KeyCode::KeyS
                                | KeyCode::KeyQ
                                | KeyCode::KeyE),
                            ),
                        ..
                    },
                ..
            } if !self.modifiers.control_key()
                && !(cfg!(target_os = "macos") && self.modifiers.super_key()) =>
            {
                if let Some(graphics) = self.graphics.as_mut() {
                    graphics.slice_process_key(key, state == ElementState::Pressed);
                    graphics.request_slice_preview_redraw();
                }
                self.redraw_requested = true;
            }
            WindowEvent::Resized(size) => {
                if let Some(graphics) = self.graphics.as_mut() {
                    graphics.resize_slice_preview(size);
                }
            }
            WindowEvent::RedrawRequested => {
                let result = self.graphics.as_mut().map(|graphics| {
                    graphics.render_slice_preview(
                        &self.scene_document,
                        &self.triangulations,
                        &self.block_models,
                        &self.point_clouds,
                        &self.raster_textures,
                        &self.editor,
                    )
                });
                match result {
                    Some(Err(RenderSurfaceError::Lost | RenderSurfaceError::Outdated)) => {
                        if let Some(graphics) = self.graphics.as_mut() {
                            graphics.reconfigure_slice_preview();
                            graphics.request_slice_preview_redraw();
                        }
                    }
                    Some(Err(RenderSurfaceError::Timeout | RenderSurfaceError::Occluded)) => {
                        if let Some(graphics) = self.graphics.as_ref() {
                            graphics.request_slice_preview_redraw();
                        }
                    }
                    Some(Err(RenderSurfaceError::Validation)) => {
                        if let Some(graphics) = self.graphics.as_mut() {
                            graphics.close_slice_preview();
                        }
                        self.editor.slice_preview_detached = false;
                    }
                    Some(Ok(())) | None => {}
                }
            }
            _ => {}
        }
    }
}

impl<'a> App<'a> {
    fn handle_cursor_mode_cycle(&mut self, event: &WindowEvent) -> bool {
        let WindowEvent::MouseInput {
            state: ElementState::Pressed,
            button,
            ..
        } = event
        else {
            return false;
        };

        self.editor.cursor_mode = match button {
            MouseButton::Forward => self.editor.cursor_mode.next(),
            MouseButton::Back => self.editor.cursor_mode.previous(),
            _ => return false,
        };
        self.editor.cursor_snapped = false;
        self.redraw_requested = true;
        true
    }

    fn handle_mouse_press(&mut self, event: &WindowEvent) {
        if let WindowEvent::MouseInput {
            state: ElementState::Pressed,
            button: MouseButton::Left,
            ..
        } = event
        {
            if self.editor.active_tool != ActiveTool::None {
                userspace_log!(
                    "Left click with tool {:?} at {:?}",
                    self.editor.active_tool,
                    self.editor.cursor_screen_px
                );
            }
            self.editor.canvas_context_menu_open = false;
            match self.editor.active_tool {
                ActiveTool::MakePoint => self.place_point_at_cursor(),
                ActiveTool::MakeText if !self.editor.text_editing_enabled => self.text_tool_click(),
                ActiveTool::MakeLine | ActiveTool::MakePoly if !self.editor.poly_finish_dialog => {
                    self.place_stroke_point()
                }
                ActiveTool::MakeRoad => self.place_road_point(),
                ActiveTool::MeasureDistance => self.measure_distance_click(),
                ActiveTool::MeasureBermAngle => self.measure_berm_angle_click(),
                ActiveTool::VerticalSlice => self.slice_line_click(),
                ActiveTool::DeleteElement => {
                    self.editor.selection_box_start_px = self.editor.cursor_screen_px;
                    self.editor.selection_box_current_px = self.editor.cursor_screen_px;
                }
                ActiveTool::None => self.begin_select_or_drag(),
                ActiveTool::Move => {
                    if let Some(cursor_px) = self.editor.cursor_screen_px {
                        if let Some((axis_idx, axis)) = hit_gizmo_axis(&self.editor, cursor_px) {
                            self.begin_gizmo_drag(axis_idx, axis, cursor_px);
                        } else if !self.pick_move_vertex_target() {
                            self.begin_select_or_drag();
                        }
                    } else {
                        self.begin_select_or_drag();
                    }
                }
                ActiveTool::Chamfer => {
                    if let Some(cursor_px) = self.editor.cursor_screen_px {
                        if self.editor.chamfer_gizmo_hovered {
                            self.editor.chamfer_gizmo_drag_start_px = Some(cursor_px);
                            self.editor.chamfer_gizmo_drag_start_radius =
                                self.editor.chamfer_radius;
                        } else {
                            self.pick_chamfer_corner();
                        }
                    }
                }
                ActiveTool::Bezier => {
                    if let Some(cursor_px) = self.editor.cursor_screen_px {
                        // Check if clicking on a CP handle first
                        let hit_cp = bezier_cp_hit(&self.editor, cursor_px);
                        if let Some(cp_idx) = hit_cp {
                            self.editor.bezier_dragging_cp = Some(cp_idx);
                        } else {
                            self.bezier_click();
                        }
                    }
                }
                ActiveTool::ExplodePolygon => self.explode_at_cursor(),
                ActiveTool::FuseIntoPolygon => self.fuse_click(),
                ActiveTool::SplitAtPoints => self.split_at_points_click(),
                ActiveTool::RelimitLine => self.relimit_line_click(),
                ActiveTool::OffsetElement => {
                    if self.editor.offset_awaiting_side_pick {
                        self.commit_offset();
                    } else if !self.editor.offset_dialog_open {
                        self.pick_offset_target();
                    }
                }
                ActiveTool::BatterBermOffset if !self.editor.batter_berm_dialog_open => {
                    self.pick_batter_berm_target();
                }
                _ => {}
            }
        }
    }

    fn handle_mouse_release(&mut self, event: &WindowEvent) {
        if is_left_mouse_release(event) {
            self.finish_left_button_interactions();
        }
    }

    fn finish_left_button_interactions(&mut self) {
        self.finish_box_selection();
        self.finish_drag();
        if self.gizmo_drag.is_some() {
            self.finish_gizmo_drag();
        }
        if self.editor.chamfer_gizmo_drag_start_px.is_some() {
            self.editor.chamfer_gizmo_drag_start_px = None;
            self.invalidate_overlay();
        }
        if self.editor.bezier_dragging_cp.is_some() {
            self.editor.bezier_dragging_cp = None;
            self.invalidate_overlay();
        }
    }

    fn handle_right_press_for_orbit(&mut self, event: &WindowEvent) {
        if let WindowEvent::MouseInput {
            button: MouseButton::Right,
            state: ElementState::Pressed,
            ..
        } = event
        {
            self.right_press_px = self.editor.cursor_screen_px;
            self.right_orbit_active = false;
        }
    }

    fn handle_right_release(&mut self, event: &WindowEvent) {
        if self.editor.fly_mode_enabled || self.editor.slice_mode_enabled {
            // Slice mode: a right click has nothing to cancel and no context
            // menu to open.
            self.right_press_px = None;
            self.right_orbit_active = false;
            return;
        }
        if let WindowEvent::MouseInput {
            state: ElementState::Released,
            button: MouseButton::Right,
            ..
        } = event
        {
            let orbit_was_active = self.right_orbit_active;
            self.right_orbit_active = false;
            let is_quick_press = match (self.right_press_px.take(), self.editor.cursor_screen_px) {
                (Some(press), Some(cur)) => {
                    let dx = cur.0 - press.0;
                    let dy = cur.1 - press.1;
                    (dx * dx + dy * dy).sqrt() < RIGHT_CLICK_DRAG_THRESHOLD_PX
                }
                _ => false,
            } && !orbit_was_active;
            if is_quick_press && self.editor.active_tool == ActiveTool::MakeRoad {
                if self.editor.pending_stroke.is_empty() {
                    self.cancel_road();
                    userspace_log!("Right click exited road tool");
                } else if self.editor.pending_stroke.len() == 1 {
                    self.discard_road_stroke();
                    userspace_log!("Right click cancelled road stroke");
                } else {
                    self.commit_road();
                    userspace_log!("Right click finished road");
                }
                self.redraw_requested = true;
            } else if is_quick_press
                && matches!(
                    self.editor.active_tool,
                    ActiveTool::MakeLine | ActiveTool::MakePoly
                )
            {
                self.try_finish_tool();
                self.redraw_requested = true;
                userspace_log!(
                    "Right click finished drawing tool {:?}",
                    self.editor.active_tool
                );
            } else if is_quick_press && self.editor.active_tool != ActiveTool::None {
                self.cancel_active_tool();
                self.redraw_requested = true;
                userspace_log!("Right click cancelled active tool");
            } else if is_quick_press
                && self.editor.active_tool == ActiveTool::None
                && !self.editor.tri_create_open
            {
                let frozen = &self.editor.frozen_handles;
                let picked = self.graphics.as_ref().and_then(|g| {
                    g.pick_at_cursor(
                        crate::app::PICK_THRESHOLD_PX,
                        &self.triangulations,
                        &self.editor.hidden_handles,
                        frozen,
                        self.editor.xray_enabled,
                    )
                });
                if let Some((handle, world)) = picked {
                    if let crate::model::SceneEntityId::Object(id) = handle {
                        self.activate_project_for_object(id);
                        let Some(layer) = self
                            .workspace
                            .active_document()
                            .and_then(|document| document.get_object(id))
                            .map(crate::model::Object::layer)
                        else {
                            return;
                        };
                        self.editor.active_layer = Some(layer);
                    }
                    if !self.editor.selected_handles.contains(&handle) {
                        self.editor.on_canvas_pick(
                            handle,
                            world,
                            crate::ui::state::SelectionMode::Replace,
                        );
                        self.invalidate_geometry();
                    }
                    self.active_triangulation = match handle {
                        crate::model::SceneEntityId::Triangulation(id) => Some(id),
                        crate::model::SceneEntityId::Object(_)
                        | crate::model::SceneEntityId::BlockModel(_) => None,
                    };
                    userspace_log!("Opened canvas context menu for {:?}", handle);
                    self.editor.canvas_context_menu_open = true;
                    self.editor.canvas_context_menu_px = self.editor.cursor_screen_px;
                    self.redraw_requested = true;
                }
            }
        }
    }

    fn handle_control_shortcuts(&mut self, event: &WindowEvent) {
        if !self.editor.text_editing_enabled
            && (self.modifiers.control_key()
                || (cfg!(target_os = "macos") && self.modifiers.super_key()))
            && let WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(key),
                        ..
                    },
                ..
            } = event
        {
            match key {
                KeyCode::KeyZ if self.modifiers.shift_key() => {
                    userspace_log!("Ctrl+Shift+Z");
                    self.apply_history_step(false)
                }
                KeyCode::KeyZ => {
                    userspace_log!("Ctrl+Z");
                    self.apply_history_step(true)
                }
                KeyCode::KeyY => {
                    userspace_log!("Ctrl+Y");
                    self.apply_history_step(false)
                }
                KeyCode::KeyS => {
                    userspace_log!("Ctrl+S");
                    if let Err(error) = self.save_all_dirty_projects() {
                        userspace_error!("Couldn't save: {}", error);
                    }
                }
                KeyCode::KeyA => {
                    userspace_log!("Ctrl+A");
                    self.select_all_active_objects();
                }
                KeyCode::KeyD => {
                    userspace_log!("Ctrl+D");
                    self.duplicate_selection();
                }
                _ => {}
            }
        }
    }

    fn maybe_start_right_orbit_drag(&mut self) {
        if self.editor.fly_mode_enabled || self.right_orbit_active {
            return;
        }
        let (Some(press), Some(cur)) = (self.right_press_px, self.editor.cursor_screen_px) else {
            return;
        };
        let dx = cur.0 - press.0;
        let dy = cur.1 - press.1;
        if (dx * dx + dy * dy).sqrt() < RIGHT_CLICK_DRAG_THRESHOLD_PX {
            return;
        }

        if self.editor.slice_mode_enabled {
            // No orbiting in slice mode; the camera is fully derived from the
            // slice state (Q/E rotates the slice line instead).
            return;
        }

        self.refresh_snap_index();
        let Some(graphics) = self.graphics.as_mut() else {
            return;
        };
        graphics.begin_orbit_at_surface(
            &self.triangulations,
            &self.editor.hidden_handles,
            &self.editor.frozen_handles,
            &self.scene_document,
            &self.snap_index,
        );
        graphics.begin_right_orbit_drag();
        self.right_orbit_active = true;
        self.redraw_requested = true;
    }

    fn handle_key_action(&mut self, event: &WindowEvent) {
        if let WindowEvent::KeyboardInput {
            event:
                KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(key),
                    ..
                },
            ..
        } = event
        {
            self.handle_key_code(*key);
        }
    }

    fn handle_key_code(&mut self, key: KeyCode) {
        match &key {
            KeyCode::Escape => {
                if self.editor.canvas_context_menu_open {
                    self.editor.canvas_context_menu_open = false;
                    self.redraw_requested = true;
                    userspace_log!("Escape closed canvas context menu");
                } else if self.editor.text_editing_enabled {
                    self.cancel_text_edit();
                    userspace_log!("Escape cancelled text editing");
                } else if self.editor.slice_mode_enabled {
                    self.set_slice_mode_enabled(false);
                    userspace_log!("Escape exited slice view");
                } else if self.editor.active_tool == ActiveTool::VerticalSlice {
                    self.editor.slice_pending_start = None;
                    self.editor.active_tool = ActiveTool::None;
                    self.invalidate_overlay();
                    userspace_log!("Escape cancelled slice line placement");
                } else if self.editor.offset_awaiting_side_pick || self.editor.offset_dialog_open {
                    self.cancel_offset();
                    userspace_log!("Escape cancelled offset tool");
                } else if self.editor.relimit_confirming_end
                    || self.editor.relimit_waiting_for_pick
                    || self.editor.relimit_awaiting_source_pick
                    || self.editor.relimit_dialog_open
                {
                    self.cancel_relimit();
                    userspace_log!("Escape cancelled relimit tool");
                } else if self.editor.active_tool == ActiveTool::FuseIntoPolygon {
                    self.cancel_fuse();
                    self.editor.active_tool = ActiveTool::None;
                    userspace_log!("Escape cancelled fuse tool");
                } else if self.editor.active_tool == ActiveTool::SplitAtPoints {
                    self.cancel_split_at_points();
                    userspace_log!("Escape cancelled split at points tool");
                } else if self.editor.active_tool == ActiveTool::Move {
                    self.gizmo_drag = None;
                    self.editor.gizmo_drag_axis_index = None;
                    self.cancel_move_delta();
                    self.editor.active_tool = ActiveTool::None;
                    userspace_log!("Escape cancelled move tool");
                } else if self.editor.active_tool == ActiveTool::Chamfer {
                    self.cancel_chamfer();
                    userspace_log!("Escape cancelled chamfer tool");
                } else if self.editor.active_tool == ActiveTool::Bezier {
                    self.cancel_bezier();
                    userspace_log!("Escape cancelled bezier tool");
                } else if self.editor.active_tool == ActiveTool::ExplodePolygon {
                    self.editor.tool_highlight_id = None;
                    self.editor.active_tool = ActiveTool::None;
                    self.invalidate_geometry();
                    userspace_log!("Escape cancelled explode tool");
                } else if self.editor.active_tool == ActiveTool::MakeRoad {
                    self.cancel_road();
                    userspace_log!("Escape cancelled road tool");
                } else if self.editor.active_tool == ActiveTool::BatterBermOffset
                    || self.editor.batter_berm_dialog_open
                    || self.editor.batter_berm_target_id.is_some()
                {
                    self.cancel_batter_berm();
                    userspace_log!("Escape cancelled batter berm tool");
                } else if !self.editor.pending_stroke.is_empty()
                    || self.editor.active_tool != ActiveTool::None
                {
                    self.discard_stroke();
                    userspace_log!("Escape discarded in-progress stroke");
                }
            }
            KeyCode::Backquote => {
                let picked = self.graphics.as_ref().and_then(|graphics| {
                    graphics.pick_at_cursor(
                        crate::app::PICK_THRESHOLD_PX,
                        &self.triangulations,
                        &self.editor.hidden_handles,
                        &self.editor.frozen_handles,
                        self.editor.xray_enabled,
                    )
                });
                if let Some((_handle, world)) = picked
                    && world.z.is_finite()
                {
                    let z = (world.z * 10_000.0).round() / 10_000.0;
                    self.editor.z_level = z;
                    self.editor.z_input = z;
                    self.redraw_requested = true;
                    userspace_log!("Backquote set Z level to cursor hit Z {:.4}", z);
                }
            }
            KeyCode::Enter | KeyCode::NumpadEnter if !self.editor.text_editing_enabled => {
                userspace_log!("Enter");
                if self.editor.active_tool == ActiveTool::MakeRoad {
                    self.commit_road();
                } else if self.editor.active_tool == ActiveTool::Move {
                    let d = self.editor.move_panel_delta;
                    self.apply_move_delta(glam::DVec3::new(d[0], d[1], d[2]));
                    self.editor.active_tool = ActiveTool::None;
                } else if self.editor.active_tool == ActiveTool::OffsetElement
                    && self.editor.offset_awaiting_side_pick
                {
                    self.commit_offset();
                } else if self.editor.active_tool == ActiveTool::Bezier {
                    self.apply_bezier();
                } else {
                    self.try_finish_tool();
                }
            }
            KeyCode::Delete | KeyCode::Backspace if !self.editor.text_editing_enabled => {
                let object_count = self
                    .editor
                    .selected_handles
                    .iter()
                    .filter(|h| matches!(h, crate::model::SceneEntityId::Object(_)))
                    .count();
                if object_count > 0 {
                    userspace_log!("Delete requested for {object_count} selected object(s)");
                    self.editor.delete_confirm_open = true;
                    self.redraw_requested = true;
                }
            }
            _ => {}
        }
    }

    /// Cancel the current active tool, mirroring the Escape key behaviour.
    pub(crate) fn cancel_active_tool(&mut self) {
        if self.editor.text_editing_enabled {
            return; // let the text editor handle it
        }
        if self.editor.offset_awaiting_side_pick || self.editor.offset_dialog_open {
            self.cancel_offset();
        } else if self.editor.relimit_confirming_end
            || self.editor.relimit_waiting_for_pick
            || self.editor.relimit_awaiting_source_pick
            || self.editor.relimit_dialog_open
        {
            self.cancel_relimit();
        } else if self.editor.active_tool == ActiveTool::FuseIntoPolygon {
            self.cancel_fuse();
            self.editor.active_tool = ActiveTool::None;
        } else if self.editor.active_tool == ActiveTool::SplitAtPoints {
            self.cancel_split_at_points();
        } else if self.editor.active_tool == ActiveTool::Move {
            self.gizmo_drag = None;
            self.editor.gizmo_drag_axis_index = None;
            self.cancel_move_delta();
            self.editor.active_tool = ActiveTool::None;
        } else if self.editor.active_tool == ActiveTool::ExplodePolygon {
            self.editor.tool_highlight_id = None;
            self.editor.active_tool = ActiveTool::None;
            self.invalidate_geometry();
        } else if self.editor.active_tool == ActiveTool::Chamfer {
            self.cancel_chamfer();
        } else if self.editor.active_tool == ActiveTool::Bezier {
            self.cancel_bezier();
        } else if self.editor.active_tool == ActiveTool::MakeRoad {
            self.cancel_road();
        } else if self.editor.active_tool == ActiveTool::BatterBermOffset
            || self.editor.batter_berm_dialog_open
            || self.editor.batter_berm_target_id.is_some()
        {
            self.cancel_batter_berm();
        } else if self.editor.active_tool == ActiveTool::MeasureBermAngle {
            self.editor.berm_angle_points.clear();
            self.editor.active_tool = ActiveTool::None;
            self.invalidate_overlay();
        } else if self.editor.active_tool == ActiveTool::VerticalSlice {
            self.editor.slice_pending_start = None;
            self.editor.active_tool = ActiveTool::None;
            self.invalidate_overlay();
        } else if !self.editor.pending_stroke.is_empty() {
            self.discard_stroke();
        } else {
            self.editor.active_tool = ActiveTool::None;
        }
    }

    pub(crate) fn set_active_tool_from_toolbar(&mut self, tool: ActiveTool) {
        if self.editor.text_editing_enabled {
            return;
        }
        if (self.editor.fly_mode_enabled || self.editor.slice_mode_enabled)
            && tool != ActiveTool::None
        {
            return;
        }

        let previous_tool = self.editor.active_tool;
        let next_tool = if previous_tool == tool {
            ActiveTool::None
        } else {
            tool
        };

        match previous_tool {
            ActiveTool::OffsetElement if next_tool != ActiveTool::OffsetElement => {
                self.cancel_offset();
            }
            ActiveTool::RelimitLine if next_tool != ActiveTool::RelimitLine => {
                self.cancel_relimit();
            }
            ActiveTool::BatterBermOffset if next_tool != ActiveTool::BatterBermOffset => {
                self.cancel_batter_berm();
            }
            ActiveTool::MakeRoad if next_tool != ActiveTool::MakeRoad => {
                self.cancel_road();
            }
            ActiveTool::Bezier if next_tool != ActiveTool::Bezier => {
                self.cancel_bezier();
            }
            ActiveTool::Chamfer if next_tool != ActiveTool::Chamfer => {
                self.cancel_chamfer();
            }
            ActiveTool::SplitAtPoints if next_tool != ActiveTool::SplitAtPoints => {
                self.cancel_split_at_points();
            }
            _ => {}
        }

        self.editor.active_tool = next_tool;
        if next_tool != ActiveTool::None {
            self.editor.new_layer_dialog_open = false;
        }

        if matches!(
            tool,
            ActiveTool::MeasureDistance | ActiveTool::MeasureBermAngle
        ) {
            self.editor.measurement_start = None;
            self.editor.measurement_end = None;
            self.editor.berm_angle_points.clear();
        }
    }

    pub(crate) fn set_fly_mode_enabled(&mut self, enabled: bool) {
        if self.editor.fly_mode_enabled == enabled {
            return;
        }

        if enabled {
            if self.editor.text_editing_enabled {
                self.cancel_text_edit();
            } else {
                self.cancel_active_tool();
            }
            // Fly and slice modes are mutually exclusive: both claim W/S and
            // right-drag.
            self.set_slice_mode_enabled(false);
            self.finish_left_button_interactions();
            self.editor.canvas_context_menu_open = false;
            self.editor.cursor_snapped = false;
            self.right_press_px = None;
            self.right_orbit_active = false;
        }

        self.editor.fly_mode_enabled = enabled;
        self.redraw_requested = true;
    }
}

fn bezier_cp_hit(editor: &crate::ui::state::EditorState, cursor_px: (f32, f32)) -> Option<u8> {
    if editor.bezier_selected_verts[0].is_none() || editor.bezier_selected_verts[1].is_none() {
        return None;
    }
    const HANDLE_THRESHOLD_PX: f32 = 12.0;
    let dist = |(ax, ay): (f32, f32), (bx, by): (f32, f32)| -> f32 {
        ((ax - bx) * (ax - bx) + (ay - by) * (ay - by)).sqrt()
    };
    let mut best: Option<(u8, f32)> = None;
    if let Some(cp1_px) = editor.bezier_cp1_screen_px {
        let d = dist(cursor_px, cp1_px);
        if d < HANDLE_THRESHOLD_PX {
            best = Some((0, d));
        }
    }
    if let Some(cp2_px) = editor.bezier_cp2_screen_px {
        let d = dist(cursor_px, cp2_px);
        if d < HANDLE_THRESHOLD_PX && best.is_none_or(|(_, bd)| d < bd) {
            best = Some((1, d));
        }
    }
    best.map(|(idx, _)| idx)
}

fn is_left_mouse_release(event: &WindowEvent) -> bool {
    matches!(
        event,
        WindowEvent::MouseInput {
            state: ElementState::Released,
            button: MouseButton::Left,
            ..
        }
    )
}

fn hit_gizmo_axis(
    editor: &crate::ui::state::EditorState,
    cursor_px: (f32, f32),
) -> Option<(u8, glam::DVec3)> {
    const THRESHOLD_PX: f32 = 12.0;
    let center = editor.move_gizmo_center_px?;
    let axes = [
        (editor.move_gizmo_x_tip_px, glam::DVec3::X, 0u8),
        (editor.move_gizmo_y_tip_px, glam::DVec3::Y, 1u8),
        (editor.move_gizmo_z_tip_px, glam::DVec3::Z, 2u8),
    ];
    let dist_to_segment = |p: (f32, f32), a: (f32, f32), b: (f32, f32)| -> f32 {
        let px = p.0 - a.0;
        let py = p.1 - a.1;
        let bx = b.0 - a.0;
        let by = b.1 - a.1;
        let len2 = bx * bx + by * by;
        if len2 < 1e-8 {
            return (px * px + py * py).sqrt();
        }
        let t = ((px * bx + py * by) / len2).clamp(0.0, 1.0);
        let rx = px - t * bx;
        let ry = py - t * by;
        (rx * rx + ry * ry).sqrt()
    };

    axes.into_iter().find_map(|(tip, axis, idx)| {
        let tip = tip?;
        (dist_to_segment(cursor_px, center, tip) < THRESHOLD_PX).then_some((idx, axis))
    })
}
