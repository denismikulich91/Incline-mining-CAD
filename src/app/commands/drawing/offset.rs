use crate::{
    app::{App, PICK_THRESHOLD_PX},
    model::{Command, Object, ObjectId, PolyVertex, SceneEntityId},
    ui::state::ActiveTool,
    userspace_log,
};

impl<'a> App<'a> {
    /// Open the offset dialog for the given object, or pick from selection.
    pub(crate) fn open_offset_dialog(&mut self) {
        let object_id = self.editor.selected_handles.iter().find_map(|h| match h {
            SceneEntityId::Object(id) => {
                if matches!(self.active_layer_object(*id), Some(Object::Polyline { .. })) {
                    Some(*id)
                } else {
                    None
                }
            }
            _ => None,
        });
        if let Some(id) = object_id {
            self.editor.offset_target_id = Some(id);
            self.editor.offset_dialog_open = true;
            self.editor.tool_highlight_id = Some(id);
            userspace_log!("Opened offset dialog for object {:?}", id);
            self.invalidate_geometry();
        }
    }

    /// Pick an element to offset when the tool is active but no target yet.
    pub(crate) fn pick_offset_target(&mut self) {
        let frozen = &self.editor.frozen_handles;
        let picked = self.graphics.as_ref().and_then(|g| {
            g.pick_at_cursor(
                PICK_THRESHOLD_PX,
                &self.triangulations,
                &self.editor.hidden_handles,
                frozen,
                self.editor.xray_enabled,
            )
        });
        if let Some((SceneEntityId::Object(id), _)) = picked
            && matches!(self.active_layer_object(id), Some(Object::Polyline { .. }))
        {
            self.editor.offset_target_id = Some(id);
            self.editor.offset_dialog_open = true;
            self.editor.tool_highlight_id = Some(id);
            self.invalidate_geometry();
        }
    }

    /// Called when the dialog Apply button is pressed. Computes horiz_dist and
    /// z_delta from dialog inputs, then enters the side-pick phase.
    pub(crate) fn begin_offset_pick(
        &mut self,
        object_id: ObjectId,
        horiz_dist: f64,
        z_delta: f64,
        project_to_rl: Option<(f64, f64)>,
    ) {
        self.editor.offset_target_id = Some(object_id);
        self.editor.offset_horiz_dist = horiz_dist;
        self.editor.offset_z_delta = z_delta;
        self.editor.offset_project_to_rl = project_to_rl;
        self.editor.offset_dialog_open = false;
        self.editor.offset_awaiting_side_pick = true;
        let closed = matches!(
            self.active_layer_object(object_id),
            Some(Object::Polyline { closed: true, .. })
        );
        self.editor.offset_preview_closed = closed;
    }

    /// Compute the offset result geometry for the current side-pick settings,
    /// choosing between a uniform offset and a per-vertex angled projection to a
    /// target absolute RL depending on `offset_project_to_rl`.
    fn compute_offset_result(
        &self,
        src_verts: &[glam::DVec3],
        closed: bool,
        cursor_world_xy: glam::DVec2,
    ) -> Vec<glam::DVec3> {
        if let Some((tan_angle, target_rl)) = self.editor.offset_project_to_rl {
            // Per-vertex horizontal distance implied by each vertex's own elevation,
            // used only to size the cursor-side probe below.
            let probe_dist = if tan_angle.abs() < 1e-9 {
                0.0
            } else {
                src_verts
                    .iter()
                    .map(|v| ((target_rl - v.z) / tan_angle).abs())
                    .fold(0.0_f64, f64::max)
            };
            let side = crate::model::geometry::offset_side_from_cursor(
                src_verts,
                closed,
                cursor_world_xy,
                probe_dist,
            );
            crate::model::geometry::geometric_offset_project_to_rl(
                src_verts, closed, side, tan_angle, target_rl,
            )
        } else {
            let horiz_dist = self.editor.offset_horiz_dist;
            let abs_dist = horiz_dist.abs();
            let z_delta = self.editor.offset_z_delta;
            let side = crate::model::geometry::offset_side_from_cursor(
                src_verts,
                closed,
                cursor_world_xy,
                abs_dist,
            );
            crate::model::geometry::geometric_offset(src_verts, closed, side * abs_dist, z_delta)
        }
    }

    /// Recompute the offset preview based on current cursor position.
    pub(crate) fn update_offset_preview(&mut self) {
        if !self.editor.offset_awaiting_side_pick {
            return;
        }
        let Some(object_id) = self.editor.offset_target_id else {
            return;
        };
        if self.editor.cursor_screen_px.is_none() {
            return;
        }
        let Some(graphics) = self.graphics.as_ref() else {
            return;
        };

        // Get source verts.
        let (src_verts, closed) = match self.active_document().get_object(object_id) {
            Some(Object::Polyline { verts, closed, .. }) => {
                (verts.iter().map(|v| v.pos).collect::<Vec<_>>(), *closed)
            }
            _ => return,
        };

        // Unproject cursor to a world XY position (use Z=0 plane for side determination).
        let cursor_world_xy = graphics
            .cursor_world(0.0)
            .map(|w| glam::DVec2::new(w.x, w.y))
            .unwrap_or(glam::DVec2::ZERO);

        let preview = self.compute_offset_result(&src_verts, closed, cursor_world_xy);

        if self.editor.offset_preview_world != preview
            || self.editor.offset_preview_closed != closed
        {
            self.editor.offset_source_world = src_verts;
            self.editor.offset_preview_world = preview;
            self.editor.offset_preview_closed = closed;
            userspace_log!("Updated offset preview for object {:?}", object_id);
        }
    }

    /// Commit the offset using the current preview side.
    pub(crate) fn commit_offset(&mut self) {
        let Some(object_id) = self.editor.offset_target_id else {
            return;
        };
        if self.editor.cursor_screen_px.is_none() {
            return;
        }
        let Some(graphics) = self.graphics.as_ref() else {
            return;
        };

        let (src_verts, closed, layer, color, fill, fill_color, line_weight) =
            match self.active_document().get_object(object_id) {
                Some(Object::Polyline {
                    verts,
                    closed,
                    layer,
                    color,
                    fill,
                    fill_color,
                    line_weight,
                    ..
                }) => (
                    verts.iter().map(|v| v.pos).collect::<Vec<_>>(),
                    *closed,
                    *layer,
                    *color,
                    *fill,
                    *fill_color,
                    *line_weight,
                ),
                _ => return,
            };

        let cursor_world_xy = graphics
            .cursor_world(0.0)
            .map(|w| glam::DVec2::new(w.x, w.y))
            .unwrap_or(glam::DVec2::ZERO);

        let new_positions = self.compute_offset_result(&src_verts, closed, cursor_world_xy);

        let new_verts: Vec<PolyVertex> = new_positions
            .into_iter()
            .map(PolyVertex::straight)
            .collect();

        if let Some(project) = self.workspace.active_project_mut() {
            let doc = &mut project.pidb.document;
            let id = doc.allocate_object_id();
            self.history.execute(
                doc,
                Command::AddObject(Object::Polyline {
                    id,
                    layer,
                    verts: new_verts,
                    closed,
                    color,
                    fill,
                    fill_color,
                    line_weight,
                }),
            );
            project.dirty = true;
        }

        self.cancel_offset();
        userspace_log!("Created offset of object {:?}", object_id);
        self.invalidate_geometry();
    }

    pub(crate) fn cancel_offset(&mut self) {
        self.editor.offset_dialog_open = false;
        self.editor.offset_target_id = None;
        self.editor.offset_awaiting_side_pick = false;
        self.editor.offset_project_to_rl = None;
        self.editor.offset_preview_world.clear();
        self.editor.offset_source_world.clear();
        self.editor.offset_preview_screen_px.clear();
        self.editor.offset_source_screen_px.clear();
        self.editor.tool_highlight_id = None;
        self.editor.active_tool = ActiveTool::None;
        self.invalidate_geometry();
        userspace_log!("Cancelled offset tool");
    }
}
