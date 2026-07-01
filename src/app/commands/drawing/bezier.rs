use glam::DVec3;

use crate::{
    app::{App, PICK_THRESHOLD_PX},
    model::{Command, Object, PolyVertex, SceneEntityId},
    rendering::pick,
    ui::state::ActiveTool,
    userspace_log,
};

impl<'a> App<'a> {
    /// Dispatch a canvas click for the Bezier tool.
    pub(crate) fn bezier_click(&mut self) {
        if self.editor.bezier_poly_id.is_none() {
            self.pick_bezier_polygon();
        } else if self.editor.bezier_selected_verts[0].is_none()
            || self.editor.bezier_selected_verts[1].is_none()
        {
            self.pick_bezier_vertex();
        }
        // Both verts selected: user interacts through the panel/gizmo only
    }

    fn pick_bezier_polygon(&mut self) {
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

        let Some((SceneEntityId::Object(oid), _)) = picked else {
            return;
        };

        if !self.active_layer_object(oid).is_some_and(
            |o| matches!(o, Object::Polyline { closed: true, verts, .. } if verts.len() >= 3),
        ) {
            return;
        }

        self.editor.bezier_poly_id = Some(oid);
        self.editor.bezier_selected_verts = [None; 2];
        self.editor.bezier_dialog_open = true;
        self.invalidate_overlay();
    }

    fn pick_bezier_vertex(&mut self) {
        let Some(oid) = self.editor.bezier_poly_id else {
            return;
        };
        let Some(graphics) = self.graphics.as_ref() else {
            return;
        };
        let Some(cursor_px) = self.editor.cursor_screen_px else {
            return;
        };

        // Find the nearest vertex of the selected polygon only
        let vp = graphics.view_proj();
        let screen = graphics.screen_size_pub();
        let verts = match self.scene_document.get_object(oid) {
            Some(Object::Polyline {
                verts,
                closed: true,
                ..
            }) => verts.clone(),
            _ => return,
        };
        let n = verts.len();

        let cursor_d = glam::DVec2::new(f64::from(cursor_px.0), f64::from(cursor_px.1));
        let mut best_dist = (PICK_THRESHOLD_PX * 2.5) as f64;
        let mut best_idx: Option<usize> = None;

        for (i, vert) in verts.iter().enumerate() {
            if let Some(sp) = pick::world_to_screen(&vp, vert.pos, screen) {
                let d = sp.distance(cursor_d);
                if d < best_dist {
                    best_dist = d;
                    best_idx = Some(i);
                }
            }
        }

        let Some(vi) = best_idx else {
            return;
        };

        if self.editor.bezier_selected_verts[0].is_none() {
            // Picking the first vertex
            self.editor.bezier_selected_verts[0] = Some(vi);
            self.invalidate_overlay();
        } else {
            let first = self.editor.bezier_selected_verts[0].unwrap();
            if vi == first {
                return;
            }

            // Check adjacency: must be consecutive in the polygon
            let adjacent = vi == (first + 1) % n || first == (vi + 1) % n;
            if !adjacent {
                return;
            }

            // Normalise so that bezier_selected_verts[0] is always the "start" of the edge
            // i.e., verts[0] leads to verts[1] = (verts[0]+1)%n
            let (start, end) = if (first + 1) % n == vi {
                (first, vi)
            } else {
                (vi, first)
            };

            self.editor.bezier_selected_verts[0] = Some(start);
            self.editor.bezier_selected_verts[1] = Some(end);

            // Initialise control points at 1/3 and 2/3 along the selected edge
            let v0 = verts[start].pos;
            let v1 = verts[end].pos;
            let cp1 = v0 + (v1 - v0) * (1.0 / 3.0);
            let cp2 = v0 + (v1 - v0) * (2.0 / 3.0);
            self.editor.bezier_cp1 = [cp1.x, cp1.y, cp1.z];
            self.editor.bezier_cp2 = [cp2.x, cp2.y, cp2.z];
            self.invalidate_overlay();
        }
    }

    pub(crate) fn apply_bezier(&mut self) {
        let (Some(oid), [Some(vi), Some(vj)]) = (
            self.editor.bezier_poly_id,
            self.editor.bezier_selected_verts,
        ) else {
            return;
        };
        let Some(project) = self.workspace.active_project_mut() else {
            return;
        };
        let doc = &mut project.pidb.document;
        let Some(obj) = doc.get_object(oid) else {
            return;
        };
        let Object::Polyline {
            verts,
            closed: true,
            ..
        } = obj
        else {
            return;
        };

        let before = obj.clone();
        let verts = verts.clone();
        let n = verts.len();

        let v_start = verts[vi].pos;
        let v_end = verts[vj].pos;
        let cp1 = DVec3::from(self.editor.bezier_cp1);
        let cp2 = DVec3::from(self.editor.bezier_cp2);
        let segments = self.editor.bezier_segments.max(2) as usize;

        // Build new vertex list: insert bezier intermediate points between vi and vj
        let mut new_verts: Vec<PolyVertex> = Vec::with_capacity(n + segments);
        for (k, vert) in verts.iter().enumerate().take(n) {
            new_verts.push(*vert);
            if k == vi {
                for seg in 1..segments {
                    let t = seg as f64 / segments as f64;
                    let p = bezier_eval(v_start, cp1, cp2, v_end, t);
                    new_verts.push(PolyVertex::straight(p));
                }
            }
        }

        let mut after = before.clone();
        if let Object::Polyline {
            verts: after_verts, ..
        } = &mut after
        {
            *after_verts = new_verts;
        }

        if before != after {
            self.history
                .execute(doc, Command::Replace { before, after });
            project.dirty = true;
            userspace_log!(
                "Bezier applied: {} intermediate points on edge {vi}→{vj}",
                segments - 1
            );
        }

        self.clear_bezier_state();
        self.editor.active_tool = ActiveTool::None;
        self.invalidate_geometry();
        self.invalidate_overlay();
    }

    pub(crate) fn cancel_bezier(&mut self) {
        self.clear_bezier_state();
        self.editor.active_tool = ActiveTool::None;
        self.invalidate_overlay();
    }

    pub(crate) fn clear_bezier_state(&mut self) {
        self.editor.bezier_poly_id = None;
        self.editor.bezier_selected_verts = [None; 2];
        self.editor.bezier_cp1 = [0.0; 3];
        self.editor.bezier_cp2 = [0.0; 3];
        self.editor.bezier_poly_verts_screen_px.clear();
        self.editor.bezier_cp1_screen_px = None;
        self.editor.bezier_cp2_screen_px = None;
        self.editor.bezier_preview_screen_px.clear();
        self.editor.bezier_dragging_cp = None;
        self.editor.bezier_hover_cp = None;
        self.editor.bezier_dialog_open = false;
    }
}

/// Evaluate a cubic bezier at parameter `t` in [0, 1].
pub(crate) fn bezier_eval(p0: DVec3, p1: DVec3, p2: DVec3, p3: DVec3, t: f64) -> DVec3 {
    let u = 1.0 - t;
    u * u * u * p0 + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 + t * t * t * p3
}
