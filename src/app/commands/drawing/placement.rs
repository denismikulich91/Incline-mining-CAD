use crate::{
    app::App,
    model::{Command, LayerId, Object, PolyVertex},
    ui::state::ActiveTool,
    userspace_log,
};

impl<'a> App<'a> {
    pub(crate) fn measure_distance_click(&mut self) {
        if matches!(
            self.editor.cursor_mode,
            crate::ui::state::CursorMode::SnapToPoint
                | crate::ui::state::CursorMode::SnapToLine
                | crate::ui::state::CursorMode::SnapToSurface
        ) && !self.editor.cursor_snapped
        {
            return;
        }
        let Some(point) = self.editor.cursor_world else {
            return;
        };
        if let Some(_) = self.editor.measurement_start
            && self.editor.measurement_end.is_none()
        {
            self.editor.measurement_end = Some(point);
        } else {
            self.editor.measurement_start = Some(point);
            self.editor.measurement_end = None;
        }
        self.invalidate_overlay();
    }

    pub(crate) fn place_point_at_cursor(&mut self) {
        if !self.editing_ready() {
            return;
        }
        // Block if snap mode is active but cursor isn't snapped
        if matches!(
            self.editor.cursor_mode,
            crate::ui::state::CursorMode::SnapToPoint
                | crate::ui::state::CursorMode::SnapToLine
                | crate::ui::state::CursorMode::SnapToSurface
        ) && !self.editor.cursor_snapped
        {
            return;
        }
        let Some(world) = self.editor.cursor_world else {
            return;
        };
        let Some(layer) = self.active_layer() else {
            return;
        };
        let color = crate::model::ObjectColor::Fixed(self.editor.tool_line_color);
        if let Some(project) = self.workspace.active_project_mut() {
            let doc = &mut project.pidb.document;
            let id = doc.allocate_object_id();
            self.history.execute(
                doc,
                Command::AddObject(Object::Point {
                    id,
                    layer,
                    pos: world,
                    color,
                }),
            );
            project.dirty = true;
        }
        userspace_log!(
            "Placed point @ {:.3}, {:.3}, {:.3}",
            world.x,
            world.y,
            world.z
        );
        self.invalidate_geometry();
    }

    pub(crate) fn place_stroke_point(&mut self) {
        if !self.editing_ready() {
            return;
        }
        // Block if snap mode is active but cursor isn't snapped
        if matches!(
            self.editor.cursor_mode,
            crate::ui::state::CursorMode::SnapToPoint
                | crate::ui::state::CursorMode::SnapToLine
                | crate::ui::state::CursorMode::SnapToSurface
        ) && !self.editor.cursor_snapped
        {
            return;
        }
        let Some(world) = self.editor.cursor_world else {
            return;
        };
        let Some(layer) = self.active_layer() else {
            return;
        };
        self.editor.pending_stroke.push(world);
        match self.editor.active_tool {
            ActiveTool::MakeLine if self.editor.pending_stroke.len() >= 2 => {
                let verts: Vec<PolyVertex> = self
                    .editor
                    .pending_stroke
                    .iter()
                    .map(|&p| PolyVertex::straight(p))
                    .collect();
                let chain_start = *self.editor.pending_stroke.last().unwrap();
                self.commit_polyline(verts, false, layer);
                // Chain: end of the last segment becomes start of next
                self.editor.pending_stroke.clear();
                self.editor.pending_stroke.push(chain_start);
            }
            ActiveTool::MakePoly => {
                // If the user clicked the first vertex, auto-close without showing a dialog.
                let is_closing = self.editor.pending_stroke.len() >= 2
                    && self
                        .editor
                        .pending_stroke
                        .first()
                        .zip(self.editor.pending_stroke.last())
                        .is_some_and(|(first, last)| (*first - *last).length_squared() <= 1.0e-16);
                if is_closing {
                    self.finish_poly_closed();
                    return;
                }
            }
            ActiveTool::MakeLine if !self.editor.pending_stroke.is_empty() => {}
            _ => {}
        }
        self.invalidate_geometry();
    }

    /// Finish an in-progress MakePoly stroke as a closed polygon.
    pub(crate) fn finish_poly_closed(&mut self) {
        if self.editor.active_tool != ActiveTool::MakePoly {
            return;
        }
        // Clicking the first vertex to close the preview records that point a
        // second time. Closed polylines already add the last-to-first edge, so
        // discard the duplicate instead of creating a zero-length segment.
        if self.editor.pending_stroke.len() >= 2
            && self
                .editor
                .pending_stroke
                .first()
                .zip(self.editor.pending_stroke.last())
                .is_some_and(|(first, last)| (*first - *last).length_squared() <= 1.0e-16)
        {
            self.editor.pending_stroke.pop();
        }
        if self.editor.pending_stroke.len() < 3 {
            return;
        }
        let Some(layer) = self.active_layer() else {
            return;
        };
        let verts: Vec<PolyVertex> = self
            .editor
            .pending_stroke
            .iter()
            .map(|&p| PolyVertex::straight(p))
            .collect();
        self.commit_polyline(verts, true, layer);
        self.editor.pending_stroke.clear();
        self.editor.poly_finish_dialog = false;
        self.editor.poly_finish_dialog_px = None;
        self.invalidate_geometry();
        userspace_log!("Finished closed polygon");
    }

    /// Discard the current in-progress stroke without committing anything.
    pub(crate) fn discard_stroke(&mut self) {
        let discarded_vertices = self.editor.pending_stroke.len();
        self.editor.pending_stroke.clear();
        self.editor.measurement_start = None;
        self.editor.measurement_end = None;
        self.editor.poly_finish_dialog = false;
        self.editor.poly_finish_dialog_px = None;
        self.editor.active_tool = ActiveTool::None;
        self.invalidate_geometry();
        userspace_log!("Discarded in-progress stroke ({discarded_vertices} vertices)");
    }

    /// Cancel the current stroke (commits open polyline if ≥2 verts, otherwise discards).
    pub(crate) fn cancel_stroke(&mut self) {
        if self.editor.active_tool == ActiveTool::MakePoly
            && self.editor.pending_stroke.len() >= 2
            && let Some(layer) = self.active_layer()
        {
            let verts: Vec<PolyVertex> = self
                .editor
                .pending_stroke
                .iter()
                .map(|&p| PolyVertex::straight(p))
                .collect();
            self.commit_polyline(verts, false, layer);
            userspace_log!(
                "Cancelled polygon (committed {} vertex/vertices as open polyline)",
                self.editor.pending_stroke.len()
            );
        }
        self.editor.pending_stroke.clear();
        self.editor.poly_finish_dialog = false;
        self.editor.poly_finish_dialog_px = None;
        self.invalidate_geometry();
    }

    /// Attempt to finish the active drawing tool via Enter.
    /// If no stroke is in progress, exits the active drawing tool.
    /// For MakePoly with enough verts, opens the finish dialog at the cursor.
    /// For MakeLine, clears the chain anchor so the next click starts a new string.
    pub(crate) fn try_finish_tool(&mut self) {
        match self.editor.active_tool {
            ActiveTool::MakePoly if self.editor.pending_stroke.len() >= 2 => {
                self.editor.poly_finish_dialog = true;
                self.editor.poly_finish_dialog_px = self.editor.cursor_screen_px;
                self.redraw_requested = true;
            }
            ActiveTool::MakePoly if !self.editor.pending_stroke.is_empty() => {
                self.editor.pending_stroke.clear();
                self.editor.poly_finish_dialog = false;
                self.editor.poly_finish_dialog_px = None;
                self.invalidate_geometry();
                userspace_log!("Finished polygon stroke; MakePoly remains active");
            }
            ActiveTool::MakePoly => {
                self.editor.active_tool = ActiveTool::None;
                self.invalidate_geometry();
            }
            ActiveTool::MakeLine => {
                if self.editor.pending_stroke.len() >= 2 {
                    let verts: Vec<PolyVertex> = self
                        .editor
                        .pending_stroke
                        .iter()
                        .map(|&p| PolyVertex::straight(p))
                        .collect();
                    if let Some(layer) = self.active_layer() {
                        self.commit_polyline(verts, false, layer);
                        userspace_log!(
                            "Finished line via Enter ({} vertices)",
                            self.editor.pending_stroke.len()
                        );
                    }
                }
                if self.editor.pending_stroke.is_empty() {
                    self.editor.active_tool = ActiveTool::None;
                    self.invalidate_geometry();
                } else {
                    self.editor.pending_stroke.clear();
                    self.invalidate_geometry();
                    userspace_log!("Finished line string; MakeLine remains active");
                }
            }
            _ => {}
        }
    }

    fn commit_polyline(&mut self, verts: Vec<PolyVertex>, closed: bool, layer: LayerId) {
        if verts.len() < 2 {
            return;
        }
        let color = crate::model::ObjectColor::Fixed(self.editor.tool_line_color);
        let fill_color = Some(crate::model::ObjectColor::Fixed(
            self.editor.tool_fill_color,
        ));
        let line_weight = self.editor.tool_line_weight;
        let fill = self.editor.tool_hatch.to_fill_style();
        if let Some(project) = self.workspace.active_project_mut() {
            let doc = &mut project.pidb.document;
            let id = doc.allocate_object_id();
            self.history.execute(
                doc,
                Command::AddObject(Object::Polyline {
                    id,
                    layer,
                    verts,
                    closed,
                    color,
                    fill,
                    fill_color,
                    line_weight,
                }),
            );
            project.dirty = true;
        }
    }
}
