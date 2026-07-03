use winit::keyboard::PhysicalKey;

use super::*;

impl<'a> Graphics<'a> {
    pub(crate) fn process_mouse_motion(&mut self, dx: f64, dy: f64) -> bool {
        if self.fly_mode_enabled && self.mouse_pressed == Some(MouseButton::Right) {
            self.fly_camera_controller.process_mouse_motion(dx, dy);
            return true;
        }

        if !self.fly_mode_enabled && self.mouse_pressed == Some(MouseButton::Middle) {
            return self
                .camera_controller
                .process_mouse(self.mouse_pressed, dx, dy);
        }

        false
    }

    pub(crate) fn set_mouse_location(&mut self, mouse_loc: (f32, f32)) -> bool {
        let previous_mouse_loc = self.camera_controller.mouse_loc;
        self.camera_controller.mouse_loc = mouse_loc;

        if self.mouse_pressed == Some(MouseButton::Right) && !self.fly_mode_enabled {
            let dx = mouse_loc.0 - previous_mouse_loc.0;
            let dy = mouse_loc.1 - previous_mouse_loc.1;
            return self
                .camera_controller
                .process_mouse(self.mouse_pressed, dx.into(), dy.into());
        }

        false
    }

    pub(super) fn screen_size(&self) -> Size {
        (self.size.width as f32, self.size.height as f32)
    }

    pub(crate) fn zoom(&self) -> f64 {
        self.projection.zoom
    }

    pub(crate) fn screen_size_pub(&self) -> Size {
        self.screen_size()
    }

    pub(crate) fn view_proj(&self) -> glam::DMat4 {
        self.projection.calc_matrix() * self.camera.calc_matrix() * self.exaggeration_matrix()
    }

    pub(super) fn exaggeration_matrix(&self) -> glam::DMat4 {
        DMat4::from_translation(self.scene_origin)
            * DMat4::from_scale(DVec3::new(1.0, 1.0, self.vertical_exaggeration))
            * DMat4::from_translation(-self.scene_origin)
    }

    pub(super) fn exaggerate_point(&self, point: DVec3) -> DVec3 {
        self.scene_origin
            + DVec3::new(
                point.x - self.scene_origin.x,
                point.y - self.scene_origin.y,
                (point.z - self.scene_origin.z) * self.vertical_exaggeration,
            )
    }

    pub(super) fn unexaggerate_point(&self, point: DVec3) -> DVec3 {
        self.scene_origin
            + DVec3::new(
                point.x - self.scene_origin.x,
                point.y - self.scene_origin.y,
                (point.z - self.scene_origin.z) / self.vertical_exaggeration,
            )
    }

    pub(super) fn cursor_model_ray(&self) -> (DVec3, DVec3) {
        // Unproject through the exact matrix used for rendering. The previous
        // implementation moved the origin 1e9 units behind the cursor, which
        // lost enough floating-point precision to visibly offset BVH hits.
        let screen = self.screen_size();
        let cursor = self.camera_controller.mouse_loc;
        let ndc = crate::rendering::camera::point(cursor.0, cursor.1, screen);
        let inverse = self.view_proj().inverse();
        let near_h = inverse * DVec4::new(ndc.x, ndc.y, 0.0, 1.0);
        let far_h = inverse * DVec4::new(ndc.x, ndc.y, 1.0, 1.0);
        let near = near_h.truncate() / near_h.w;
        let far = far_h.truncate() / far_h.w;
        (near, (far - near).normalize())
    }

    /// World coordinate under the current cursor on the plane `z = plane_z`,
    /// using the last cursor position tracked by the camera controller.
    pub(crate) fn cursor_world(&self, plane_z: f64) -> Option<DVec3> {
        let screen = self.screen_size();
        let aspect = screen.0 as f64 / screen.1.max(1.0) as f64;
        let displayed_plane_z =
            self.scene_origin.z + (plane_z - self.scene_origin.z) * self.vertical_exaggeration;
        screen_to_world_on_plane(
            &self.camera,
            self.projection.zoom,
            aspect,
            screen,
            self.camera_controller.mouse_loc,
            displayed_plane_z,
        )
        .map(|point| self.unexaggerate_point(point))
    }

    /// Nearest rendered entity geometry under the cursor, as `(handle, world)`
    /// with the geometry's true world position (including Z). Frozen handles are
    /// visible but excluded from picking. `None` if no geometry is within
    /// `threshold_px`.
    pub(crate) fn pick_at_cursor(
        &self,
        threshold_px: f32,
        triangulations: &[OpenTriangulation],
        hidden: &std::collections::HashSet<SceneEntityId>,
        frozen: &std::collections::HashSet<SceneEntityId>,
        xray_enabled: bool,
    ) -> Option<(SceneEntityId, DVec3)> {
        let view_proj = self.view_proj();
        let screen = self.screen_size();
        let geometry_hit = pick_nearest(
            &self.pick_records,
            &self.stroke_vertex_buf,
            &self.stroke_index_buf,
            &self.lyon_buffer.vertices,
            &self.lyon_buffer.indices,
            self.scene_origin,
            &view_proj,
            screen,
            self.camera_controller.mouse_loc,
            threshold_px,
            frozen,
        );
        let text_hit = pick_text(
            &self.text_pick_records,
            &view_proj,
            screen,
            self.camera_controller.mouse_loc,
            frozen,
        );

        let (ray_origin, direction) = self.cursor_model_ray();
        let document_hit = match (geometry_hit, text_hit) {
            (Some(geometry), Some(text)) => {
                let geometry_depth = (geometry.world - ray_origin).dot(direction);
                let text_depth = (text.world - ray_origin).dot(direction);
                Some(if text_depth < geometry_depth {
                    text
                } else {
                    geometry
                })
            }
            (geometry, text) => geometry.or(text),
        };
        let surface_hit = SceneQuery::nearest_surface(
            triangulations,
            hidden,
            Some(frozen),
            ray_origin,
            direction,
        );

        match (document_hit, surface_hit) {
            (Some(document), Some(surface)) => {
                // In x-ray mode visible lines render through surfaces, so a document
                // hit always takes precedence over the surface beneath it.
                if xray_enabled {
                    Some((document.entity, document.world))
                } else {
                    // Compare camera-ray depths: prefer whichever hit is closer.
                    // `pick_nearest` returns the nearest 2D vertex which may not be
                    // exactly under the cursor, but for geometry placed above a surface
                    // the vertex depth is still smaller than the surface hit below it.
                    let doc_depth = (document.world - ray_origin).dot(direction);
                    let surf_depth = (surface.1 - ray_origin).dot(direction);
                    if doc_depth <= surf_depth {
                        Some((document.entity, document.world))
                    } else {
                        Some(surface)
                    }
                }
            }
            (Some(document), None) => Some((document.entity, document.world)),
            (None, surface) => surface,
        }
    }

    /// Return design entities whose rendered geometry is fully enclosed by a
    /// physical-pixel selection rectangle.
    pub(crate) fn entities_in_screen_rect(
        &self,
        start_px: (f32, f32),
        end_px: (f32, f32),
    ) -> Vec<SceneEntityId> {
        let min_x = f64::from(start_px.0.min(end_px.0));
        let max_x = f64::from(start_px.0.max(end_px.0));
        let min_y = f64::from(start_px.1.min(end_px.1));
        let max_y = f64::from(start_px.1.max(end_px.1));
        let view_proj = self.view_proj();
        let screen = self.screen_size();

        let geometry_hits = self.pick_records.iter().filter_map(|record| {
            let stroke = record.stroke_range.0 as usize
                ..(record.stroke_range.1 as usize).min(self.stroke_vertex_buf.len());
            let fill = record.fill_range.0 as usize
                ..(record.fill_range.1 as usize).min(self.lyon_buffer.vertices.len());
            let points = self.stroke_vertex_buf[stroke]
                .iter()
                .map(|vertex| vertex.pos)
                .chain(
                    self.lyon_buffer.vertices[fill]
                        .iter()
                        .map(|vertex| vertex.pos),
                );
            let mut any = false;
            let enclosed = points
                .filter_map(|position| {
                    let world = DVec3::from_array(position.map(f64::from)) + self.scene_origin;
                    crate::rendering::pick::world_to_screen(&view_proj, world, screen)
                })
                .all(|point| {
                    any = true;
                    point.x >= min_x && point.x <= max_x && point.y >= min_y && point.y <= max_y
                });
            (any && enclosed).then_some(record.entity)
        });

        let text_hits = self.text_pick_records.iter().filter_map(|record| {
            let screen_corners: Vec<_> = record
                .corners
                .iter()
                .filter_map(|&corner| {
                    crate::rendering::pick::world_to_screen(&view_proj, corner, screen)
                })
                .collect();
            if screen_corners.len() < 4 {
                return None;
            }
            let enclosed = screen_corners.iter().all(|point| {
                point.x >= min_x && point.x <= max_x && point.y >= min_y && point.y <= max_y
            });
            enclosed.then_some(record.entity)
        });

        geometry_hits.chain(text_hits).collect()
    }

    /// Cross-select: entities where ANY vertex falls inside the rectangle.
    /// Used for left-to-right drag (Vulcan-style touch/cross selection).
    pub(crate) fn entities_touching_screen_rect(
        &self,
        start_px: (f32, f32),
        end_px: (f32, f32),
    ) -> Vec<SceneEntityId> {
        let min_x = f64::from(start_px.0.min(end_px.0));
        let max_x = f64::from(start_px.0.max(end_px.0));
        let min_y = f64::from(start_px.1.min(end_px.1));
        let max_y = f64::from(start_px.1.max(end_px.1));
        let view_proj = self.view_proj();
        let screen = self.screen_size();

        let geometry_hits = self.pick_records.iter().filter_map(|record| {
            let stroke = record.stroke_range.0 as usize
                ..(record.stroke_range.1 as usize).min(self.stroke_vertex_buf.len());
            let fill = record.fill_range.0 as usize
                ..(record.fill_range.1 as usize).min(self.lyon_buffer.vertices.len());
            let points = self.stroke_vertex_buf[stroke]
                .iter()
                .map(|vertex| vertex.pos)
                .chain(
                    self.lyon_buffer.vertices[fill]
                        .iter()
                        .map(|vertex| vertex.pos),
                );
            let any_vertex_inside = points
                .filter_map(|position| {
                    let world = DVec3::from_array(position.map(f64::from)) + self.scene_origin;
                    crate::rendering::pick::world_to_screen(&view_proj, world, screen)
                })
                .any(|point| {
                    point.x >= min_x && point.x <= max_x && point.y >= min_y && point.y <= max_y
                });
            if any_vertex_inside {
                return Some(record.entity);
            }
            // Cross-select: also include if any segment crosses the box boundary.
            let any_segment_crosses = record.segments.iter().any(|&[a3, b3]| {
                let (Some(a), Some(b)) = (
                    crate::rendering::pick::world_to_screen(&view_proj, a3, screen),
                    crate::rendering::pick::world_to_screen(&view_proj, b3, screen),
                ) else {
                    return false;
                };
                segment_intersects_rect(a, b, min_x, max_x, min_y, max_y)
            });
            any_segment_crosses.then_some(record.entity)
        });

        let text_hits = self.text_pick_records.iter().filter_map(|record| {
            let screen_corners: Vec<_> = record
                .corners
                .iter()
                .filter_map(|&corner| {
                    crate::rendering::pick::world_to_screen(&view_proj, corner, screen)
                })
                .collect();
            let any_inside = screen_corners.iter().any(|point| {
                point.x >= min_x && point.x <= max_x && point.y >= min_y && point.y <= max_y
            });
            any_inside.then_some(record.entity)
        });

        geometry_hits.chain(text_hits).collect()
    }

    /// Begin an orbit with the anchor at the surface or geometry point under the cursor.
    /// Falls back to the current-target depth when nothing is hit.
    /// Called from the app level where triangulations are available.
    pub(crate) fn begin_orbit_at_surface(
        &mut self,
        triangulations: &[OpenTriangulation],
        hidden: &std::collections::HashSet<SceneEntityId>,
        frozen: &std::collections::HashSet<SceneEntityId>,
        document: &Document,
        snap_index: &crate::model::spatial::ObjectSnapIndex,
    ) {
        // Prefer snapping to a nearby document vertex so the orbit pivot lands
        // on actual geometry (lines, polygons, points) when one is close.
        let view_proj = self.view_proj();
        let screen = self.screen_size();
        let snap_pt = SceneQuery::snap(
            document,
            snap_index,
            triangulations,
            hidden,
            frozen,
            &CursorMode::SnapToPoint,
            &view_proj,
            screen,
            self.camera_controller.mouse_loc,
            SNAP_THRESHOLD_PX,
            false,
        );

        let pt = if let Some(p) = snap_pt {
            p
        } else {
            let (ray_origin, direction) = self.cursor_model_ray();
            SceneQuery::nearest_surface(triangulations, hidden, Some(frozen), ray_origin, direction)
                .map(|(_, world)| world)
                .unwrap_or_else(|| {
                    // No triangulation surface hit — try picking any document object
                    // near the cursor so the pivot lands on visible geometry rather than
                    // at the (possibly stale) camera-target depth.
                    self.pick_at_cursor(SNAP_THRESHOLD_PX, triangulations, hidden, frozen, false)
                        .map(|(_, world)| world)
                        .unwrap_or_else(|| {
                            self.unexaggerate_point(self.cursor_world_at_target_depth())
                        })
                })
        };

        self.camera.sync_angles_from_forward();
        self.camera_controller
            .begin_orbit(self.exaggerate_point(pt));
        self.orbit_marker = Some(pt);
    }

    /// Find the nearest snap target for the current cursor position.
    /// Returns `None` in `CursorMode::Select` or when nothing is within the snap threshold.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn snap_cursor(
        &self,
        document: &Document,
        snap_index: &crate::model::spatial::ObjectSnapIndex,
        triangulations: &[OpenTriangulation],
        hidden: &std::collections::HashSet<SceneEntityId>,
        frozen: &std::collections::HashSet<SceneEntityId>,
        mode: &CursorMode,
        xray_enabled: bool,
    ) -> Option<DVec3> {
        if *mode == CursorMode::Select {
            return None;
        }
        let view_proj = self.view_proj();
        let screen = self.screen_size();
        SceneQuery::snap(
            document,
            snap_index,
            triangulations,
            hidden,
            frozen,
            mode,
            &view_proj,
            screen,
            self.camera_controller.mouse_loc,
            SNAP_THRESHOLD_PX,
            xray_enabled,
        )
    }

    /// Reset the camera to a top-down plan view that fits all visible content.
    /// Falls back to a default unit zoom centred on the origin when empty.
    pub(crate) fn fit_to_extents(
        &mut self,
        document: &Document,
        triangulations: &[OpenTriangulation],
        block_models: &[OpenBlockModel],
        hidden: &std::collections::HashSet<crate::model::SceneEntityId>,
    ) {
        let screen = self.screen_size();
        let aspect = (screen.0 as f64 / screen.1.max(1.0) as f64).max(1e-9);

        let (center, zoom) = match scene_bounds(document, triangulations, block_models, hidden) {
            Some((min, max)) => {
                let center = (min + max) * 0.5;
                let size = max - min;
                // Degenerate (single point or all collinear on one axis): unit zoom.
                if size.length() < 1e-6 {
                    (center, 1.0_f64)
                } else {
                    // Plan view: right == X, up == Y.
                    let zoom_h = size.y / 2.0;
                    let zoom_w = size.x / (2.0 * aspect);
                    // 10 % padding so content never touches the viewport edge.
                    (center, zoom_h.max(zoom_w) * 1.1)
                }
            }
            None => (DVec3::ZERO, 1.0_f64),
        };

        let zoom = zoom.max(1e-4);
        self.projection.zoom = zoom;
        let camera_distance = if self.fly_mode_enabled {
            zoom / (crate::rendering::camera::PERSPECTIVE_FOV_Y * 0.5).tan()
        } else {
            zoom
        };
        self.camera.reset_to_plan_view(center, camera_distance);
        self.scene_origin = center;
        self.triangulation_gpu.clear();
        self.block_model_gpu.clear();
        self.geometry_dirty = true;
        // Update znear/zfar immediately so snap/pick work before the first render.
        self.fit_depth_to_scene(document, triangulations, block_models, hidden);
    }

    /// Frame all visible content while keeping the current camera orientation
    /// (orbit/tilt unchanged) — only position, target and zoom are adjusted.
    /// No-op when there is nothing visible to frame.
    pub(crate) fn zoom_to_extents(
        &mut self,
        document: &Document,
        triangulations: &[OpenTriangulation],
        block_models: &[OpenBlockModel],
        hidden: &std::collections::HashSet<crate::model::SceneEntityId>,
    ) {
        let Some((min, max)) = scene_bounds(document, triangulations, block_models, hidden) else {
            return;
        };
        let center = (min + max) * 0.5;
        let forward = self.camera.forward();
        let right = forward.cross(self.camera.up()).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();

        // Project the eight bounding-box corners onto the view's right/up axes
        // to find the half-extents the ortho frustum must cover at this angle.
        let mut half_w = 0.0_f64;
        let mut half_h = 0.0_f64;
        for i in 0..8 {
            let corner = DVec3::new(
                if (i & 1) == 0 { min.x } else { max.x },
                if (i & 2) == 0 { min.y } else { max.y },
                if (i & 4) == 0 { min.z } else { max.z },
            );
            let mut d = corner - center;
            d.z *= self.vertical_exaggeration;
            half_w = half_w.max(d.dot(right).abs());
            half_h = half_h.max(d.dot(up).abs());
        }

        let screen = self.screen_size();
        let aspect = (screen.0 as f64 / screen.1.max(1.0) as f64).max(1e-9);
        let zoom = if half_w <= 1e-9 && half_h <= 1e-9 {
            1.0
        } else {
            // 10 % padding, matching fit_to_extents.
            (half_h.max(half_w / aspect) * 1.1).max(1e-4)
        };

        self.projection.zoom = zoom;
        self.camera.frame_keep_orientation(center, zoom);
        self.scene_origin = center;
        self.triangulation_gpu.clear();
        self.block_model_gpu.clear();
        self.geometry_dirty = true;
        // Update znear/zfar immediately so snap/pick work before the first render.
        self.fit_depth_to_scene(document, triangulations, block_models, hidden);
    }

    pub(crate) fn set_standard_view(&mut self, view: crate::ui::state::StandardView) {
        let (forward, up) = match view {
            crate::ui::state::StandardView::Up => (DVec3::NEG_Z, DVec3::Y),
            crate::ui::state::StandardView::Down => (DVec3::Z, DVec3::Y),
            crate::ui::state::StandardView::North => (DVec3::NEG_Y, DVec3::Z),
            crate::ui::state::StandardView::South => (DVec3::Y, DVec3::Z),
            crate::ui::state::StandardView::West => (DVec3::X, DVec3::Z),
            crate::ui::state::StandardView::East => (DVec3::NEG_X, DVec3::Z),
        };
        self.camera
            .set_target_orientation(forward, up, self.projection.zoom);
        self.camera_controller.end_orbit();
        self.orbit_marker = None;
    }

    /// Keep the orthographic depth range tight around the current scene. A
    /// billion-unit fixed range loses enough depth precision that back-side
    /// mesh edges can compare equal to the front surface and bleed through it.
    pub(super) fn fit_depth_to_scene(
        &mut self,
        document: &Document,
        triangulations: &[OpenTriangulation],
        block_models: &[OpenBlockModel],
        hidden: &std::collections::HashSet<crate::model::SceneEntityId>,
    ) {
        if self.geometry_dirty || self.cached_bounds_document_revision != document.revision() {
            self.cached_scene_bounds = scene_bounds(document, triangulations, block_models, hidden);
            self.cached_bounds_document_revision = document.revision();
        }
        let Some((min, max)) = self.cached_scene_bounds else {
            self.projection
                .set_symmetric_depth_extent((self.projection.zoom * 4.0).max(10.0));
            return;
        };
        let forward = self.camera.forward();
        let mut half_depth = 0.0_f64;
        for i in 0..8 {
            let corner = DVec3::new(
                if (i & 1) == 0 { min.x } else { max.x },
                if (i & 2) == 0 { min.y } else { max.y },
                if (i & 4) == 0 { min.z } else { max.z },
            );
            let corner = self.exaggerate_point(corner);
            half_depth = half_depth.max((corner - self.camera.position).dot(forward).abs());
        }
        let padding = (self.projection.zoom * 0.25).max(1.0);
        self.projection
            .set_symmetric_depth_extent(half_depth + padding);
    }

    /// Ensure an in-progress line or polygon remains inside the camera's clip
    /// volume even when its drawing plane lies outside the committed scene.
    pub(super) fn include_pending_stroke_in_depth(&mut self, editor: &EditorState) {
        if editor.pending_stroke.is_empty() {
            return;
        }

        let forward = self.camera.forward();
        let depth_from_camera = |point: DVec3| {
            let point = self.exaggerate_point(point);
            (point - self.camera.position).dot(forward).abs()
        };
        let mut half_depth = editor
            .pending_stroke
            .iter()
            .copied()
            .filter(|point| point.x.is_finite() && point.y.is_finite() && point.z.is_finite())
            .map(depth_from_camera)
            .fold(0.0_f64, f64::max);

        if !editor.poly_finish_dialog
            && let Some(cursor) = editor.cursor_world
            && cursor.x.is_finite()
            && cursor.y.is_finite()
            && cursor.z.is_finite()
        {
            half_depth = half_depth.max(depth_from_camera(cursor));
        }

        let padding = (self.projection.zoom * 0.25).max(1.0);
        self.projection
            .expand_symmetric_depth_extent(half_depth + padding);
    }

    /// Keep generated batter/berm preview rings inside the clip volume. They
    /// are not committed document geometry yet, so scene bounds omit them.
    pub(super) fn include_batter_berm_preview_in_depth(&mut self, editor: &EditorState) {
        if !editor.batter_berm_dialog_open || editor.batter_berm_rings_world.is_empty() {
            return;
        }

        let forward = self.camera.forward();
        let half_depth = editor
            .batter_berm_rings_world
            .iter()
            .flatten()
            .copied()
            .filter(|point| point.x.is_finite() && point.y.is_finite() && point.z.is_finite())
            .map(|point| {
                let point = self.exaggerate_point(point);
                (point - self.camera.position).dot(forward).abs()
            })
            .fold(0.0_f64, f64::max);

        let padding = (self.projection.zoom * 0.25).max(1.0);
        self.projection
            .expand_symmetric_depth_extent(half_depth + padding);
    }

    pub(super) fn include_road_preview_in_depth(&mut self, editor: &EditorState) {
        use crate::ui::state::ActiveTool;
        if editor.active_tool != ActiveTool::MakeRoad {
            return;
        }
        let forward = self.camera.forward();
        let depth_from_camera = |point: glam::DVec3| {
            let point = self.exaggerate_point(point);
            (point - self.camera.position).dot(forward).abs()
        };
        let half_depth = editor
            .road_preview_left_world
            .iter()
            .chain(editor.road_preview_right_world.iter())
            .copied()
            .filter(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
            .map(depth_from_camera)
            .fold(0.0_f64, f64::max);
        if half_depth > 0.0 {
            let padding = (self.projection.zoom * 0.25).max(1.0);
            self.projection
                .expand_symmetric_depth_extent(half_depth + padding);
        }
    }

    pub(super) fn upload_camera_uniform(&mut self) {
        let screen_size = self.screen_size();
        self.camera_uniform.update_view_proj(
            &self.camera,
            &self.projection,
            self.scene_origin,
            self.vertical_exaggeration,
        );
        self.text_system.viewport.update(
            &self.queue,
            Resolution {
                width: screen_size.0 as u32,
                height: screen_size.1 as u32,
            },
            glyphon::CameraUniform {
                view_proj: self.camera_uniform.view_proj,
            },
        );
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    pub(crate) fn update(&mut self, dt: Duration) {
        if self.fly_mode_enabled {
            self.fly_camera_controller
                .update_camera(&mut self.camera, dt);
        } else {
            let screen_size = self.screen_size();
            self.camera_controller.update_camera(
                &mut self.camera,
                &mut self.projection,
                dt,
                screen_size,
            );
        }
        self.upload_camera_uniform();
    }

    pub(crate) fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::MouseWheel { delta, .. } => {
                if self.fly_mode_enabled {
                    self.fly_camera_controller.process_scroll(delta);
                } else {
                    self.camera_controller.process_scroll(delta);
                }
                true
            }
            WindowEvent::MouseInput { button, state, .. } => {
                let pressing = *state == ElementState::Pressed;
                if pressing {
                    if *button == MouseButton::Right && !self.fly_mode_enabled {
                        // The app promotes a right press to orbit only after it
                        // moves past the context-click threshold.
                    } else if self.mouse_pressed.is_none() || *button == MouseButton::Right {
                        self.mouse_pressed = Some(*button);
                    }
                    if self.fly_mode_enabled && *button == MouseButton::Right {
                        self.fly_camera_controller.begin_capture();
                    }
                } else if self.mouse_pressed == Some(*button) {
                    self.mouse_pressed = None;
                }
                if !pressing && *button == MouseButton::Right {
                    self.camera_controller.end_orbit();
                    self.fly_camera_controller.clear_input();
                    self.orbit_marker = None;
                }
                self.sync_cursor_grab();
                true
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        physical_key: PhysicalKey::Code(key),
                        ..
                    },
                ..
            } => {
                let fly_active =
                    self.fly_mode_enabled && self.mouse_pressed == Some(MouseButton::Right);
                fly_active
                    && self
                        .fly_camera_controller
                        .process_key(*key, *state == ElementState::Pressed)
            }
            _ => false,
        }
    }

    pub(super) fn cursor_world_at_target_depth(&self) -> DVec3 {
        let forward = self.camera.forward();
        let right = forward.cross(self.camera.up()).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        let screen = self.screen_size();
        let mouse_ndc = crate::rendering::camera::point(
            self.camera_controller.mouse_loc.0,
            self.camera_controller.mouse_loc.1,
            screen,
        );
        let aspect = screen.0 as f64 / screen.1.max(1.0) as f64;
        let focal_dist = (self.camera.target() - self.camera.position)
            .dot(forward)
            .abs();
        self.camera.position
            + forward * focal_dist
            + right * mouse_ndc.x * aspect * self.projection.zoom
            + up * mouse_ndc.y * self.projection.zoom
    }

    pub(crate) fn orbit_marker_screen_pos(&self) -> Option<(f32, f32)> {
        let marker = self.orbit_marker?;
        let view_proj = self.view_proj();
        let screen = self.screen_size();
        crate::rendering::pick::world_to_screen(&view_proj, marker, screen)
            .map(|v| (v.x as f32, v.y as f32))
    }
}
