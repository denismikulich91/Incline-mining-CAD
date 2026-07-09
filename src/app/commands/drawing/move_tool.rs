use glam::DVec3;

use crate::{
    app::{App, GizmoDragState},
    model::{Command, Object, ObjectId, SceneEntityId},
    userspace_log,
};

impl<'a> App<'a> {
    pub(crate) fn apply_move_delta(&mut self, delta: DVec3) {
        self.ensure_move_session_original();
        self.preview_move_delta(delta);
        let Some(originals) = self.move_session_original.take() else {
            return;
        };
        let commands: Vec<Command> = originals
            .into_iter()
            .filter_map(|before| {
                let after = self.active_document().get_object(before.id()).cloned()?;
                objects_differ(&before, &after).then_some(Command::Replace { before, after })
            })
            .collect();

        let moved = commands.len();
        if moved > 0 {
            self.history.push_applied(Command::Batch(commands));
            userspace_log!("Applied move delta ({delta}) to {moved} object(s)");
        }
        self.editor.move_vertex_target = None;
        self.editor.move_panel_delta = [0.0; 3];
        self.editor.move_panel_last_preview = [f64::NAN; 3];
        self.invalidate_geometry();
        self.invalidate_overlay();
    }

    pub(crate) fn cancel_move_delta(&mut self) {
        self.gizmo_drag = None;
        self.editor.gizmo_drag_axis_index = None;
        self.restore_move_session_original();
        self.editor.move_vertex_target = None;
        self.editor.move_panel_delta = [0.0; 3];
        self.editor.move_panel_last_preview = [f64::NAN; 3];
        userspace_log!("Cancelled move operation");
        self.invalidate_geometry();
        self.invalidate_overlay();
    }

    /// Commit any pending move to history without applying an additional delta.
    ///
    /// Unlike `cancel_move_delta` (which reverts objects), this keeps the
    /// current document state — whatever `preview_move_delta` last produced —
    /// and pushes a Replace command for each changed object.  Used by save
    /// paths so that an in-progress move is preserved rather than silently lost.
    pub(crate) fn commit_pending_move(&mut self) {
        self.gizmo_drag = None;
        self.editor.gizmo_drag_axis_index = None;

        let Some(originals) = self.move_session_original.take() else {
            self.editor.move_vertex_target = None;
            self.editor.move_panel_delta = [0.0; 3];
            self.editor.move_panel_last_preview = [f64::NAN; 3];
            return;
        };
        let commands: Vec<Command> = originals
            .into_iter()
            .filter_map(|before| {
                let after = self.active_document().get_object(before.id()).cloned()?;
                objects_differ(&before, &after).then_some(Command::Replace { before, after })
            })
            .collect();

        let moved = commands.len();
        if moved > 0 {
            self.history.push_applied(Command::Batch(commands));
        }

        self.editor.move_vertex_target = None;
        self.editor.move_panel_delta = [0.0; 3];
        self.editor.move_panel_last_preview = [f64::NAN; 3];
        self.invalidate_geometry();
        self.invalidate_overlay();
    }

    pub(crate) fn has_pending_move_delta(&self) -> bool {
        self.move_session_original.is_some()
            || self.editor.move_vertex_target.is_some()
            || self.gizmo_drag.is_some()
    }

    pub(crate) fn begin_gizmo_drag(&mut self, axis_idx: u8, axis: DVec3, cursor_px: (f32, f32)) {
        if !self.editing_ready() {
            return;
        }
        let selected_ids = self.selected_object_ids();
        if selected_ids.is_empty() {
            return;
        }
        self.ensure_move_session_original();
        let (axis_screen_dir, px_per_world_unit) =
            self.gizmo_axis_screen_basis(axis_idx, cursor_px);
        let start_delta = DVec3::new(
            self.editor.move_panel_delta[0],
            self.editor.move_panel_delta[1],
            self.editor.move_panel_delta[2],
        );

        self.editor.gizmo_drag_axis_index = Some(axis_idx);
        self.gizmo_drag = Some(GizmoDragState {
            axis,
            start_cursor_screen_px: cursor_px,
            axis_screen_dir,
            px_per_world_unit,
            start_delta,
        });
        userspace_log!(
            "Started gizmo drag on {} axis",
            ["X", "Y", "Z"][axis_idx as usize]
        );
    }

    pub(crate) fn move_gizmo_to_cursor(&mut self) {
        let Some(gizmo) = self.gizmo_drag.as_ref() else {
            return;
        };
        let Some(cursor_px) = self.editor.cursor_screen_px else {
            return;
        };
        let dcx = cursor_px.0 - gizmo.start_cursor_screen_px.0;
        let dcy = cursor_px.1 - gizmo.start_cursor_screen_px.1;
        let pixel_delta_along_axis = dcx * gizmo.axis_screen_dir.0 + dcy * gizmo.axis_screen_dir.1;
        let world_delta = gizmo.start_delta
            + gizmo.axis * (f64::from(pixel_delta_along_axis) / gizmo.px_per_world_unit);
        self.editor.move_panel_delta = [world_delta.x, world_delta.y, world_delta.z];
        self.preview_move_delta(world_delta);
        self.invalidate_geometry();
    }

    pub(crate) fn finish_gizmo_drag(&mut self) {
        let Some(_gizmo) = self.gizmo_drag.take() else {
            return;
        };
        self.editor.gizmo_drag_axis_index = None;
        userspace_log!("Finished gizmo drag");
        self.invalidate_geometry();
        self.invalidate_overlay();
    }

    pub(crate) fn ensure_move_session_original(&mut self) {
        let selected_ids = self.move_target_object_ids();
        if selected_ids.is_empty() {
            self.move_session_original = None;
            return;
        }
        let should_refresh = self
            .move_session_original
            .as_ref()
            .map(|originals| {
                originals.len() != selected_ids.len()
                    || originals
                        .iter()
                        .any(|object| !selected_ids.contains(&object.id()))
            })
            .unwrap_or(true);
        if should_refresh {
            self.move_session_original = Some(
                selected_ids
                    .iter()
                    .filter_map(|&object_id| self.active_document().get_object(object_id).cloned())
                    .collect(),
            );
        }
    }

    pub(crate) fn preview_move_delta(&mut self, delta: DVec3) {
        let Some(originals) = self.move_session_original.clone() else {
            return;
        };
        let vertex_target = self.editor.move_vertex_target;
        if let Some(project) = self.workspace.active_project_mut() {
            for object in originals {
                let mut moved = object.clone();
                translate_move_target(&mut moved, vertex_target, delta);
                project.pidb.document.replace_object(moved);
            }
        }
    }

    fn restore_move_session_original(&mut self) {
        let Some(originals) = self.move_session_original.take() else {
            return;
        };
        if let Some(project) = self.workspace.active_project_mut() {
            for object in originals {
                project.pidb.document.replace_object(object);
            }
            project.invalidate_dirty_layers();
        }
    }

    fn selected_object_ids(&self) -> Vec<ObjectId> {
        self.editor
            .selected_handles
            .iter()
            .filter_map(|handle| match handle {
                SceneEntityId::Object(object_id) => Some(*object_id),
                SceneEntityId::Triangulation(_) | SceneEntityId::BlockModel(_) => None,
            })
            .collect()
    }

    fn move_target_object_ids(&self) -> Vec<ObjectId> {
        if let Some((object_id, _)) = self.editor.move_vertex_target {
            vec![object_id]
        } else {
            self.selected_object_ids()
        }
    }

    fn gizmo_axis_screen_basis(&self, axis_idx: u8, cursor_px: (f32, f32)) -> ((f32, f32), f64) {
        let center = self.editor.move_gizmo_center_px.unwrap_or(cursor_px);
        let (tip, px_per_world_unit) = match axis_idx {
            0 => (
                self.editor.move_gizmo_x_tip_px.unwrap_or(cursor_px),
                self.editor.move_gizmo_x_px_per_world,
            ),
            1 => (
                self.editor.move_gizmo_y_tip_px.unwrap_or(cursor_px),
                self.editor.move_gizmo_y_px_per_world,
            ),
            _ => (
                self.editor.move_gizmo_z_tip_px.unwrap_or(cursor_px),
                self.editor.move_gizmo_z_px_per_world,
            ),
        };
        let dx = tip.0 - center.0;
        let dy = tip.1 - center.1;
        let len = (dx * dx + dy * dy).sqrt().max(0.001);
        ((dx / len, dy / len), px_per_world_unit.max(0.001))
    }
}

fn translate_move_target(
    object: &mut Object,
    vertex_target: Option<(ObjectId, usize)>,
    delta: DVec3,
) {
    let Some((target_id, vertex_index)) = vertex_target else {
        object.translate(delta);
        return;
    };
    if object.id() != target_id {
        return;
    }
    match object {
        Object::Polyline { verts, .. }
        | Object::Road {
            centerline: verts, ..
        } => {
            if let Some(vertex) = verts.get_mut(vertex_index) {
                vertex.pos += delta;
            }
        }
        Object::Point { pos, .. } if vertex_index == 0 => {
            *pos += delta;
        }
        _ => {}
    }
}

fn objects_differ(before: &Object, after: &Object) -> bool {
    before != after
}
