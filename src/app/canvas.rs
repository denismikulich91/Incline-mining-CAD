use crate::{
    app::{App, PICK_THRESHOLD_PX},
    model::{Command, Object, SceneEntityId},
    ui::state::SelectionMode,
};

impl<'a> App<'a> {
    pub(crate) fn begin_select_or_drag(&mut self) {
        self.pending_topology_click = None;

        // In triangulation creation mode, every canvas press starts a potential
        // box selection. On release, a short press becomes a normal click-pick.
        // This allows box drags to begin over a polygon instead of requiring
        // empty space.
        if self.editor.tri_create_open {
            self.editor.selection_box_start_px = self.editor.cursor_screen_px;
            self.editor.selection_box_current_px = self.editor.cursor_screen_px;
            return;
        }

        // Polygon-pick mode for cut-by-polygon: intercept the click and look for a closed polyline.
        if self.editor.tri_cut_poly_awaiting_pick {
            let frozen = &self.editor.frozen_handles;
            if let Some((crate::model::SceneEntityId::Object(oid), _)) =
                self.graphics.as_ref().and_then(|g| {
                    g.pick_at_cursor(
                        crate::app::PICK_THRESHOLD_PX,
                        &self.triangulations,
                        &self.editor.hidden_handles,
                        frozen,
                        self.editor.xray_enabled,
                    )
                })
            {
                let is_closed_poly = self.scene_document.get_object(oid).is_some_and(|o| {
                    matches!(o, crate::model::Object::Polyline { closed: true, .. })
                });
                if is_closed_poly {
                    let name = self
                        .scene_document
                        .get_object(oid)
                        .and_then(|o| {
                            let layer_id = o.layer();
                            self.scene_document
                                .layer(layer_id)
                                .map(|l| format!("Polygon on '{}'", l.name))
                        })
                        .unwrap_or_else(|| "Polygon".to_owned());
                    self.editor.tri_cut_poly_object_id = Some(oid);
                    self.editor.tri_cut_poly_object_name = name;
                    self.editor.tri_cut_poly_awaiting_pick = false;
                    self.editor.tool_highlight_id = Some(oid);
                    self.invalidate_geometry();
                }
            }
            return;
        }

        if !self.workspace.has_active_project() && self.triangulations.is_empty() {
            self.editing_ready();
            return;
        }
        if self.editor.active_tool == crate::ui::state::ActiveTool::Move {
            self.cancel_move_delta();
            self.editor.move_vertex_target = None;
        }
        let frozen = &self.editor.frozen_handles;
        let picked = self.graphics.as_ref().and_then(|graphics| {
            graphics
                .pick_at_cursor(
                    PICK_THRESHOLD_PX,
                    &self.triangulations,
                    &self.editor.hidden_handles,
                    frozen,
                    self.editor.xray_enabled,
                )
                .map(|(handle, world)| (Some(handle), world))
                .or_else(|| {
                    graphics
                        .cursor_world(self.editor.z_level)
                        .map(|world| (None, world))
                })
        });
        match picked {
            Some((Some(handle), world)) => {
                if matches!(handle, SceneEntityId::Triangulation(_)) {
                    self.pending_topology_click = Some((handle, world));
                    self.editor.selection_box_start_px = self.editor.cursor_screen_px;
                    self.editor.selection_box_current_px = self.editor.cursor_screen_px;
                    return;
                }

                let owner_changed = if let SceneEntityId::Object(object_id) = handle
                    && let Some(index) = self.workspace.project_index_for_object(object_id)
                {
                    let changed = self.workspace.active_index != Some(index);
                    if changed {
                        self.history.clear();
                        self.editor.selected_handles.clear();
                    }
                    self.workspace.set_active_index(index);
                    self.editor.active_layer = self
                        .workspace
                        .active_document()
                        .and_then(|document| document.get_object(object_id))
                        .map(Object::layer);
                    changed
                } else {
                    false
                };
                let selection_mode = if owner_changed {
                    SelectionMode::Replace
                } else if self.modifiers.shift_key() {
                    SelectionMode::Toggle
                } else if self.modifiers.control_key() {
                    SelectionMode::Add
                } else if matches!(
                    handle,
                    SceneEntityId::Triangulation(_) | SceneEntityId::BlockModel(_)
                ) && self.editor.selected_handles.contains(&handle)
                {
                    SelectionMode::Toggle
                } else {
                    SelectionMode::Replace
                };
                self.editor.on_canvas_pick(handle, world, selection_mode);
                self.active_triangulation = match handle {
                    SceneEntityId::Triangulation(id)
                        if self.editor.selected_handles.contains(&handle) =>
                    {
                        Some(id)
                    }
                    SceneEntityId::Triangulation(_) => None,
                    SceneEntityId::BlockModel(_) => None,
                    SceneEntityId::Object(_) => None,
                };
            }
            Some((None, world)) => {
                self.active_triangulation = None;
                if matches!(
                    self.editor.active_tool,
                    crate::ui::state::ActiveTool::None | crate::ui::state::ActiveTool::Move
                ) {
                    self.editor.selection_box_start_px = self.editor.cursor_screen_px;
                    self.editor.selection_box_current_px = self.editor.cursor_screen_px;
                } else if self.workspace.has_active_project() {
                    self.editor.on_canvas_click(world);
                } else {
                    self.editor.selected_handles.clear();
                }
            }
            None => {
                // A background click may not intersect the current Z plane
                // (for example, when clicking above the horizon). It should
                // still begin a selection gesture so a short click can clear
                // the current selection.
                self.active_triangulation = None;
                if matches!(
                    self.editor.active_tool,
                    crate::ui::state::ActiveTool::None | crate::ui::state::ActiveTool::Move
                ) {
                    self.editor.selection_box_start_px = self.editor.cursor_screen_px;
                    self.editor.selection_box_current_px = self.editor.cursor_screen_px;
                } else {
                    return;
                }
            }
        }
        self.invalidate_geometry();
    }

    pub(crate) fn finish_box_selection(&mut self) {
        let (Some(start), Some(end)) = (
            self.editor.selection_box_start_px.take(),
            self.editor.selection_box_current_px.take(),
        ) else {
            self.pending_topology_click = None;
            return;
        };
        let dragged = (end.0 - start.0).abs().max((end.1 - start.1).abs()) >= 3.0;
        let pending_topology_click = self.pending_topology_click.take();

        // Delete element tool: box drag selects and confirms deletion; single click deletes at
        // cursor.
        if self.editor.active_tool == crate::ui::state::ActiveTool::DeleteElement {
            if dragged {
                let cross_select = end.0 > start.0;
                let enclosed = self
                    .graphics
                    .as_ref()
                    .map(|g| {
                        if cross_select {
                            g.entities_touching_screen_rect(start, end)
                        } else {
                            g.entities_in_screen_rect(start, end)
                        }
                    })
                    .unwrap_or_default();
                let active_project = self.workspace.active_index;
                let active_object_ids = self.active_project_object_ids();
                self.editor.selected_handles.clear();
                for handle in enclosed {
                    if let crate::model::SceneEntityId::Object(id) = handle
                        && active_project.is_some()
                        && active_object_ids.contains(&id)
                    {
                        self.editor.selected_handles.insert(handle);
                    }
                }
                // Always invalidate geometry so deselected objects lose their highlight color.
                self.invalidate_geometry();
                if !self.editor.selected_handles.is_empty() {
                    self.editor.delete_confirm_open = true;
                }
            } else {
                self.delete_at_cursor();
            }
            return;
        }

        // In triangulation creation mode, drag-select adds closed polygons to the tri pick list.
        if self.editor.tri_create_open {
            if dragged {
                // Same left/right direction convention as regular selection.
                let cross_select = end.0 > start.0;
                let enclosed = self
                    .graphics
                    .as_ref()
                    .map(|g| {
                        if cross_select {
                            g.entities_touching_screen_rect(start, end)
                        } else {
                            g.entities_in_screen_rect(start, end)
                        }
                    })
                    .unwrap_or_default();
                let mut added = 0usize;
                for handle in enclosed {
                    if let SceneEntityId::Object(oid) = handle
                        && !self.editor.tri_selected_object_ids.contains(&oid)
                        && self
                            .scene_document
                            .get_object(oid)
                            .is_some_and(is_closed_polygon)
                    {
                        self.editor.tri_selected_object_ids.push(oid);
                        self.editor
                            .selected_handles
                            .insert(SceneEntityId::Object(oid));
                        added += 1;
                    }
                }
                if added > 0 {
                    self.invalidate_geometry();
                }
            } else {
                self.tri_pick_at_cursor();
            }
            return;
        }

        if !dragged {
            if let Some((handle, world)) = pending_topology_click {
                let selection_mode = if self.modifiers.shift_key() {
                    SelectionMode::Toggle
                } else if self.modifiers.control_key() {
                    SelectionMode::Add
                } else if self.editor.selected_handles.contains(&handle) {
                    SelectionMode::Toggle
                } else {
                    SelectionMode::Replace
                };
                self.editor.on_canvas_pick(handle, world, selection_mode);
                self.active_triangulation = match handle {
                    SceneEntityId::Triangulation(id)
                        if self.editor.selected_handles.contains(&handle) =>
                    {
                        Some(id)
                    }
                    SceneEntityId::Triangulation(_) => None,
                    SceneEntityId::BlockModel(_) => None,
                    SceneEntityId::Object(_) => None,
                };
                self.invalidate_geometry();
                return;
            }

            let preserve = self.modifiers.shift_key() || self.modifiers.control_key();
            if !preserve {
                self.editor.selected_handles.clear();
                self.active_triangulation = None;
                if self.editor.active_tool == crate::ui::state::ActiveTool::Move {
                    self.editor.move_vertex_target = None;
                }
            }
            self.invalidate_geometry();
            return;
        }

        // Vulcan-style selection: left-to-right (end.x > start.x) = cross select (any vertex
        // inside box); right-to-left (end.x < start.x) = window select (all vertices inside).
        let cross_select = end.0 > start.0;
        let mut enclosed = self
            .graphics
            .as_ref()
            .map(|graphics| {
                if cross_select {
                    graphics.entities_touching_screen_rect(start, end)
                } else {
                    graphics.entities_in_screen_rect(start, end)
                }
            })
            .unwrap_or_default();
        let active_project = self.workspace.active_index;
        let active_object_ids = self.active_project_object_ids();
        enclosed.retain(|handle| match handle {
            SceneEntityId::Object(object_id) => {
                active_project.is_some() && active_object_ids.contains(object_id)
            }
            SceneEntityId::Triangulation(_) => true,
            SceneEntityId::BlockModel(_) => true,
        });
        if self.modifiers.shift_key() {
            for handle in enclosed {
                if !self.editor.selected_handles.remove(&handle) {
                    self.editor.selected_handles.insert(handle);
                }
            }
        } else {
            if !self.modifiers.control_key() {
                self.editor.selected_handles.clear();
            }
            self.editor.selected_handles.extend(enclosed);
        }
        if self.editor.active_tool == crate::ui::state::ActiveTool::Move {
            self.editor.move_vertex_target = None;
        }
        self.invalidate_geometry();
    }

    /// Click-pick in triangulation creation mode: toggle the picked object in the tri selection.
    fn tri_pick_at_cursor(&mut self) {
        let Some(picked) = self.graphics.as_ref().and_then(|g| {
            g.pick_at_cursor(
                PICK_THRESHOLD_PX,
                &self.triangulations,
                &self.editor.hidden_handles,
                &self.editor.frozen_handles,
                self.editor.xray_enabled,
            )
        }) else {
            return;
        };
        let (handle, _world) = picked;
        let SceneEntityId::Object(oid) = handle else {
            return;
        };
        if self.editor.tri_selected_object_ids.contains(&oid) {
            self.editor.tri_selected_object_ids.retain(|&o| o != oid);
            self.editor
                .selected_handles
                .remove(&SceneEntityId::Object(oid));
        } else if self
            .scene_document
            .get_object(oid)
            .is_some_and(is_closed_polygon)
        {
            self.editor.tri_selected_object_ids.push(oid);
            self.editor
                .selected_handles
                .insert(SceneEntityId::Object(oid));
        }
        self.invalidate_geometry();
    }

    pub(crate) fn drag_to_cursor(&mut self) {
        let Some(drag) = self.drag.as_ref() else {
            return;
        };
        let (object_id, plane_z, last) = (drag.object_id, drag.plane_z, drag.last_world);
        let Some(world) = self.graphics.as_ref().and_then(|g| g.cursor_world(plane_z)) else {
            return;
        };
        let delta = world - last;
        if delta.length_squared() <= 0.0 {
            return;
        }
        if let Some(doc) = self.workspace.active_document_mut() {
            doc.translate_object(object_id, delta);
        }
        if let Some(drag) = self.drag.as_mut() {
            drag.last_world = world;
            drag.moved = true;
        }
        self.workspace.mark_dirty();
        self.invalidate_geometry();
    }

    pub(crate) fn finish_drag(&mut self) {
        let Some(drag) = self.drag.take() else {
            return;
        };
        if drag.moved
            && let Some(after) = self.active_document().get_object(drag.object_id).cloned()
        {
            self.history.push_applied(Command::Replace {
                before: drag.before,
                after,
            });
            self.workspace.mark_dirty();
        }
    }

    fn active_project_object_ids(&self) -> std::collections::HashSet<crate::model::ObjectId> {
        self.workspace
            .active_project()
            .map(|project| {
                project
                    .pidb
                    .document
                    .objects()
                    .iter()
                    .map(Object::id)
                    .collect()
            })
            .unwrap_or_default()
    }
}

fn is_closed_polygon(obj: &Object) -> bool {
    matches!(obj, Object::Polyline { verts, closed: true, .. } if verts.len() >= 3)
}
