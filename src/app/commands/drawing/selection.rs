use crate::{
    app::{App, PICK_THRESHOLD_PX},
    model::{Command, Object, ObjectId, SceneEntityId},
    rendering::pick,
    ui::state::ActiveTool,
    userspace_log,
};

struct DeleteVertexHit {
    object_id: ObjectId,
    vertex_index: usize,
    screen_px: (f32, f32),
}

impl<'a> App<'a> {
    pub(crate) fn select_all_active_objects(&mut self) {
        if self.has_pending_move_delta() {
            self.cancel_move_delta();
        }
        self.editor.selected_handles.clear();
        let handles = self
            .workspace
            .active_project()
            .map_or_else(Vec::new, |project| {
                project
                    .pidb
                    .document
                    .objects()
                    .iter()
                    .filter(|object| project.loaded_layers.contains(&object.layer()))
                    .map(|object| SceneEntityId::Object(object.id()))
                    .filter(|handle| {
                        !self.editor.hidden_handles.contains(handle)
                            && !self.editor.frozen_handles.contains(handle)
                    })
                    .collect::<Vec<_>>()
            });
        let count = handles.len();
        self.editor.selected_handles.extend(handles);
        userspace_log!("Selected {count} object(s)");
        self.invalidate_overlay();
    }

    pub(crate) fn delete_at_cursor(&mut self) {
        if !self.editing_ready() {
            return;
        }
        if self.delete_polygon_vertex_at_cursor() {
            return;
        }
        let frozen = &self.editor.frozen_handles;
        let picked = self.graphics.as_ref().and_then(|graphics| {
            graphics.pick_at_cursor(
                PICK_THRESHOLD_PX,
                &self.triangulations,
                &self.editor.hidden_handles,
                frozen,
                self.editor.xray_enabled,
            )
        });
        let Some((SceneEntityId::Object(object_id), world)) = picked else {
            return;
        };
        self.editor.on_canvas_pick(
            SceneEntityId::Object(object_id),
            world,
            crate::ui::state::SelectionMode::Replace,
        );
        self.invalidate_geometry();
        self.editor.delete_confirm_open = true;
    }

    pub(crate) fn update_delete_hover_vertex(&mut self) {
        let hover_px = self.delete_polygon_vertex_hit().map(|hit| hit.screen_px);
        if self.editor.delete_hover_vertex_px != hover_px {
            self.editor.delete_hover_vertex_px = hover_px;
            self.invalidate_overlay();
        }
    }

    fn delete_polygon_vertex_at_cursor(&mut self) -> bool {
        let Some(hit) = self.delete_polygon_vertex_hit() else {
            return false;
        };
        let Some(before) = self.active_document().get_object(hit.object_id).cloned() else {
            return false;
        };
        let mut after = before.clone();
        let Object::Polyline { verts, closed, .. } = &mut after else {
            return false;
        };
        if !*closed || verts.len() <= 3 || hit.vertex_index >= verts.len() {
            return false;
        }
        verts.remove(hit.vertex_index);
        let previous = if hit.vertex_index == 0 {
            verts.len() - 1
        } else {
            hit.vertex_index - 1
        };
        verts[previous].bulge = 0.0;

        let Some(project) = self.workspace.active_project_mut() else {
            return false;
        };
        self.history.execute(
            &mut project.pidb.document,
            Command::Replace { before, after },
        );
        self.editor.delete_hover_vertex_px = None;
        self.editor.selected_handles.clear();
        self.editor
            .selected_handles
            .insert(SceneEntityId::Object(hit.object_id));
        userspace_log!(
            "Deleted vertex {} from polygon {:?}",
            hit.vertex_index,
            hit.object_id
        );
        self.invalidate_geometry();
        self.invalidate_overlay();
        true
    }

    fn delete_polygon_vertex_hit(&self) -> Option<DeleteVertexHit> {
        let graphics = self.graphics.as_ref()?;
        let cursor_px = self.editor.cursor_screen_px?;
        let (SceneEntityId::Object(object_id), _) = graphics.pick_at_cursor(
            PICK_THRESHOLD_PX,
            &self.triangulations,
            &self.editor.hidden_handles,
            &self.editor.frozen_handles,
            self.editor.xray_enabled,
        )?
        else {
            return None;
        };
        let Object::Polyline {
            verts,
            closed: true,
            ..
        } = self.scene_document.get_object(object_id)?
        else {
            return None;
        };
        if verts.len() <= 3 {
            return None;
        }

        let cursor = glam::DVec2::new(f64::from(cursor_px.0), f64::from(cursor_px.1));
        let view_proj = graphics.view_proj();
        let screen = graphics.screen_size_pub();
        let mut best: Option<(usize, (f32, f32), f64)> = None;
        let threshold = f64::from(PICK_THRESHOLD_PX * 2.0);
        for (index, vertex) in verts.iter().enumerate() {
            if let Some(screen_pos) = pick::world_to_screen(&view_proj, vertex.pos, screen) {
                let distance = screen_pos.distance(cursor);
                if distance <= threshold
                    && best.is_none_or(|(_, _, best_distance)| distance < best_distance)
                {
                    best = Some((index, (screen_pos.x as f32, screen_pos.y as f32), distance));
                }
            }
        }

        best.map(|(vertex_index, screen_px, _)| DeleteVertexHit {
            object_id,
            vertex_index,
            screen_px,
        })
    }

    pub(crate) fn delete_selection(&mut self) {
        if self.editor.selected_handles.is_empty() {
            return;
        }
        if !self.editing_ready() {
            return;
        }
        let handles: Vec<SceneEntityId> = self.editor.selected_handles.iter().copied().collect();
        let batch: Vec<Command> = handles
            .iter()
            .filter_map(|&handle| {
                if let SceneEntityId::Object(id) = handle {
                    self.active_document()
                        .get_object(id)
                        .cloned()
                        .map(Command::delete_object)
                } else {
                    None
                }
            })
            .collect();
        let deleted = batch.len();
        if deleted > 0
            && let Some(project) = self.workspace.active_project_mut()
        {
            self.history
                .execute(&mut project.pidb.document, Command::Batch(batch));
        }
        if deleted > 0 {
            self.editor.selected_handles.clear();
            // Deleting a tool target must also discard its transient markers.
            self.editor.fuse_segments.clear();
            self.editor.fuse_awaiting_endpoint = None;
            self.editor.fuse_endpoint_markers.clear();
            self.editor.fuse_chain_tail = None;
            self.editor.fuse_close_marker = None;
            self.editor.active_tool = ActiveTool::None;
            userspace_log!("Deleted {deleted} selected object(s)");
            self.invalidate_geometry();
            self.invalidate_overlay();
        }
    }

    pub(crate) fn duplicate_selection(&mut self) {
        if self.editor.selected_handles.is_empty() {
            return;
        }
        if !self.editing_ready() {
            return;
        }
        let originals: Vec<Object> = self
            .editor
            .selected_handles
            .iter()
            .filter_map(|&handle| {
                if let SceneEntityId::Object(id) = handle {
                    self.active_document().get_object(id).cloned()
                } else {
                    None
                }
            })
            .collect();
        if originals.is_empty() {
            return;
        }
        let Some(project) = self.workspace.active_project_mut() else {
            return;
        };
        let mut copies: Vec<Object> = Vec::with_capacity(originals.len());
        for obj in &originals {
            let new_id = project.pidb.document.allocate_object_id();
            copies.push(obj.with_id_and_layer(new_id, obj.layer()));
        }
        let new_ids: Vec<SceneEntityId> = copies
            .iter()
            .map(|o| SceneEntityId::Object(o.id()))
            .collect();
        let count = copies.len();
        let batch = Command::Batch(copies.into_iter().map(Command::AddObject).collect());
        self.history.execute(&mut project.pidb.document, batch);
        self.editor.selected_handles.clear();
        for id in new_ids {
            self.editor.selected_handles.insert(id);
        }
        userspace_log!("Duplicated {count} object(s)");
        self.invalidate_geometry();
        self.invalidate_overlay();
    }

    pub(crate) fn pick_move_vertex_target(&mut self) -> bool {
        if !self.editing_ready() {
            return false;
        }
        let Some(graphics) = self.graphics.as_ref() else {
            return false;
        };
        let Some(cursor_px) = self.editor.cursor_screen_px else {
            return false;
        };
        let Some((object_id, vertex_index, _world)) = pick::pick_nearest_vertex(
            self.active_document(),
            &self.editor.hidden_handles,
            &self.editor.frozen_handles,
            &graphics.view_proj(),
            graphics.screen_size_pub(),
            cursor_px,
            PICK_THRESHOLD_PX * 2.0,
        ) else {
            self.editor.move_vertex_target = None;
            return false;
        };

        self.cancel_move_delta();
        self.editor.move_vertex_target = Some((object_id, vertex_index));
        self.editor.selected_handles.clear();
        self.editor
            .selected_handles
            .insert(SceneEntityId::Object(object_id));
        self.editor.move_panel_delta = [0.0; 3];
        self.invalidate_geometry();
        self.invalidate_overlay();
        true
    }
}
