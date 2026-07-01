use glam::DVec3;

use crate::{
    app::{App, PICK_THRESHOLD_PX},
    model::{Command, Object, ObjectId, SceneEntityId},
    rendering::pick,
    ui::state::{ActiveTool, RelimitCandidate, RelimitMode, TrimEnd},
    userspace_log, userspace_warn,
};

impl<'a> App<'a> {
    pub(crate) fn open_relimit_dialog(&mut self) {
        // Relimiting modifies an existing object, so it isn't restricted to the
        // active layer (that restriction only matters for where *new* geometry
        // gets created).
        let object_id = self.editor.selected_handles.iter().find_map(|h| match h {
            SceneEntityId::Object(id) => {
                if matches!(
                    self.active_document().get_object(*id),
                    Some(Object::Polyline { closed: false, .. })
                ) {
                    Some(*id)
                } else {
                    None
                }
            }
            _ => None,
        });
        if let Some(id) = object_id {
            self.editor.relimit_source_id = Some(id);
            self.editor.relimit_awaiting_source_pick = false;
            self.editor.relimit_dialog_open = true;
        } else {
            self.editor.relimit_awaiting_source_pick = true;
        }
    }

    /// Phase 0 pick: user clicked the source line while no dialog was open.
    fn pick_relimit_source(&mut self) {
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
            && matches!(
                self.active_document().get_object(id),
                Some(Object::Polyline { closed: false, .. })
            )
        {
            self.editor.relimit_source_id = Some(id);
            self.editor.relimit_awaiting_source_pick = false;
            self.editor.relimit_dialog_open = true;
        }
    }

    pub(crate) fn relimit_line_click(&mut self) {
        if !self.editing_ready() {
            return;
        }

        // Phase 0: no source selected yet.
        if self.editor.relimit_awaiting_source_pick {
            self.pick_relimit_source();
            return;
        }

        // Phase 2: user clicks to confirm which end to move.
        if self.editor.relimit_confirming_end {
            self.commit_relimit_intersect();
            return;
        }

        // Phase 1: pick the second line (only when dialog is closed).
        if !self.editor.relimit_waiting_for_pick {
            userspace_warn!(
                "Relimit: click ignored, tool is not currently waiting for a target pick"
            );
            return;
        }
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
        let Some((SceneEntityId::Object(second_id), _)) = picked else {
            userspace_warn!("Relimit: click did not hit any object (nothing under cursor)");
            return;
        };
        let Some(source_id) = self.editor.relimit_source_id else {
            userspace_warn!("Relimit: no source line is set, aborting pick");
            return;
        };
        if second_id == source_id {
            userspace_warn!("Relimit: clicked the source line itself, pick a different line");
            return;
        }

        // Both must be open polylines with at least 1 segment. Relimiting modifies
        // existing objects, so neither is restricted to the active layer.
        let (src_verts, tgt_verts) = {
            let src = match self.active_document().get_object(source_id) {
                Some(Object::Polyline {
                    verts,
                    closed: false,
                    ..
                }) if verts.len() >= 2 => verts.clone(),
                _ => {
                    userspace_warn!(
                        "Relimit: source object {source_id:?} is no longer a valid open polyline"
                    );
                    return;
                }
            };
            let tgt = match self.active_document().get_object(second_id) {
                Some(Object::Polyline { verts, closed, .. }) if verts.len() >= 2 => {
                    (verts.clone(), *closed)
                }
                Some(Object::Polyline { verts, .. }) => {
                    userspace_warn!(
                        "Relimit: clicked polyline {second_id:?} has only {} vertex/vertices, need at least 2",
                        verts.len()
                    );
                    return;
                }
                Some(other) => {
                    userspace_warn!(
                        "Relimit: clicked object {second_id:?} is not a polyline (it's a {})",
                        other.kind_name()
                    );
                    return;
                }
                None => {
                    userspace_warn!("Relimit: clicked object {second_id:?} no longer exists");
                    return;
                }
            };
            (src, tgt)
        };
        let (tgt_verts, tgt_closed) = tgt_verts;

        let Some(graphics) = self.graphics.as_ref() else {
            userspace_warn!("Relimit: no graphics context available");
            return;
        };
        let vp = graphics.view_proj();
        let screen = graphics.screen_size_pub();

        // Collect every intersection of the *infinite* source line with the
        // target's segments, recorded as the parameter t along A→B (t=0 at A,
        // t=1 at B; t<0 is beyond A, t>1 is beyond B).
        //
        // This is computed in world-space XY (ground plane), not screen space:
        // a screen-space test depends on the current camera angle and clips out
        // any endpoint that projects off-screen or behind the camera, which made
        // picking unreliable in anything but a straight-down plan view.
        let src_last = src_verts.len() - 1;
        let a_world = src_verts[0].pos;
        let b_world = src_verts[src_last].pos;
        let a_xy = a_world.truncate();
        let b_xy = b_world.truncate();
        let mut crossings: Vec<(f64, DVec3)> = Vec::new();
        let n = tgt_verts.len();
        // Edge count: n-1 for an open polyline, plus the closing edge when closed.
        let edge_count = if tgt_closed { n } else { n - 1 };
        for i in 0..edge_count {
            let c = tgt_verts[i].pos;
            let d = tgt_verts[(i + 1) % n].pos;
            if let Some(t) = line_line_intersect_t(a_xy, b_xy, c.truncate(), d.truncate()) {
                crossings.push((t, a_world + t * (b_world - a_world)));
            }
        }
        if crossings.is_empty() {
            userspace_warn!(
                "Relimit: source line does not cross the selected target line in plan view"
            );
            return;
        }

        // Build the candidate operations. Endpoint A may move to any crossing on
        // its side of B (t < 1); endpoint B to any on its side of A (t > 0). A
        // crossing beyond the current span (t<0 for A, t>1 for B) extends the
        // line (yellow); one inside the span (0<t<1) trims it (red).
        //
        // The screen-space handle position is only used to pick which candidate
        // is nearest the cursor and to draw the hover marker — it's fine for it
        // to be approximate (or fall back to the anchor) when a point happens to
        // project off-screen, rather than dropping an otherwise valid candidate.
        let mid_screen = |anchor_world: DVec3, target: DVec3| -> (f32, f32) {
            let sa = pick::world_to_screen(&vp, anchor_world, screen);
            let st = pick::world_to_screen(&vp, target, screen);
            match (sa, st) {
                (Some(a), Some(t)) => {
                    let m = (a + t) * 0.5;
                    (m.x as f32, m.y as f32)
                }
                (Some(a), None) => (a.x as f32, a.y as f32),
                (None, Some(t)) => (t.x as f32, t.y as f32),
                (None, None) => (0.0, 0.0),
            }
        };

        let mut candidates: Vec<RelimitCandidate> = Vec::new();

        // A-extend: nearest crossing with t < 0 (closest to A from outside).
        // The trim/extend split is offset by TOUCH_EPSILON so a crossing that
        // lands right at (or a hair past) A's current endpoint — e.g. the other
        // line was already relimited to touch here — is treated as a (near
        // zero-length) trim rather than falling into neither bucket.
        if let Some(&(_, world)) = crossings
            .iter()
            .filter(|(t, _)| *t < -TOUCH_EPSILON)
            .max_by(|(t1, _), (t2, _)| t1.total_cmp(t2))
        {
            candidates.push(RelimitCandidate {
                end: TrimEnd::Start,
                target: world,
                is_extension: true,
                handle_px: mid_screen(a_world, world),
            });
        }
        // A-trim: nearest interior crossing (smallest t in (0,1), closest to A).
        if let Some(&(_, world)) = crossings
            .iter()
            .filter(|(t, _)| *t > -TOUCH_EPSILON && *t < 1.0)
            .min_by(|(t1, _), (t2, _)| t1.total_cmp(t2))
        {
            candidates.push(RelimitCandidate {
                end: TrimEnd::Start,
                target: world,
                is_extension: false,
                handle_px: mid_screen(a_world, world),
            });
        }
        // B-trim: nearest interior crossing (largest t in (0,1), closest to B).
        if let Some(&(_, world)) = crossings
            .iter()
            .filter(|(t, _)| *t > 0.0 && *t < 1.0 + TOUCH_EPSILON)
            .max_by(|(t1, _), (t2, _)| t1.total_cmp(t2))
        {
            candidates.push(RelimitCandidate {
                end: TrimEnd::End,
                target: world,
                is_extension: false,
                handle_px: mid_screen(b_world, world),
            });
        }
        // B-extend: nearest crossing with t > 1 (closest to B from outside).
        if let Some(&(_, world)) = crossings
            .iter()
            .filter(|(t, _)| *t > 1.0 + TOUCH_EPSILON)
            .min_by(|(t1, _), (t2, _)| t1.total_cmp(t2))
        {
            candidates.push(RelimitCandidate {
                end: TrimEnd::End,
                target: world,
                is_extension: true,
                handle_px: mid_screen(b_world, world),
            });
        }

        if candidates.is_empty() {
            userspace_warn!(
                "Relimit: found {} crossing(s) but none usable (source endpoint sits exactly on the target)",
                crossings.len()
            );
            return;
        }

        userspace_log!(
            "Relimit: picked target {second_id:?}, {} candidate end(s) available",
            candidates.len()
        );
        self.editor.relimit_second_id = Some(second_id);
        self.editor.relimit_candidates = candidates;
        self.editor.relimit_waiting_for_pick = false;
        self.editor.relimit_hover_target_id = None;
        self.editor.relimit_hover_target_screen_px.clear();
        self.editor.relimit_confirming_end = true;
        self.select_relimit_candidate_nearest_cursor();
        self.invalidate_overlay();
    }

    /// Pick whichever candidate's screen handle is nearest the cursor and make
    /// it the active operation (updating the preview overlay).
    fn select_relimit_candidate_nearest_cursor(&mut self) {
        if self.editor.relimit_candidates.is_empty() {
            return;
        }
        let best_idx = if let Some((cx, cy)) = self.editor.cursor_screen_px {
            let c = glam::DVec2::new(f64::from(cx), f64::from(cy));
            self.editor
                .relimit_candidates
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = glam::DVec2::new(f64::from(a.handle_px.0), f64::from(a.handle_px.1))
                        .distance(c);
                    let db = glam::DVec2::new(f64::from(b.handle_px.0), f64::from(b.handle_px.1))
                        .distance(c);
                    da.total_cmp(&db)
                })
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            0
        };
        self.apply_relimit_candidate(best_idx);
    }

    /// Activate candidate `idx`: set the moving end, target, and preview segment.
    fn apply_relimit_candidate(&mut self, idx: usize) {
        let Some(cand) = self.editor.relimit_candidates.get(idx).copied() else {
            return;
        };
        let Some(graphics) = self.graphics.as_ref() else {
            return;
        };
        let vp = graphics.view_proj();
        let screen = graphics.screen_size_pub();
        let Some(source_id) = self.editor.relimit_source_id else {
            return;
        };
        let (a_pos, b_pos) = match self.active_document().get_object(source_id) {
            Some(Object::Polyline { verts, .. }) if verts.len() >= 2 => {
                (verts[0].pos, verts[verts.len() - 1].pos)
            }
            _ => return,
        };
        let moving_pos = match cand.end {
            TrimEnd::Start => a_pos,
            TrimEnd::End => b_pos,
        };
        self.editor.relimit_hover_end = cand.end;
        self.editor.relimit_intersection_3d = Some(cand.target);
        self.editor.relimit_preview_is_extension = cand.is_extension;
        self.editor.relimit_preview_from_px =
            pick::world_to_screen(&vp, moving_pos, screen).map(|sp| (sp.x as f32, sp.y as f32));
        self.editor.relimit_preview_to_px =
            pick::world_to_screen(&vp, cand.target, screen).map(|sp| (sp.x as f32, sp.y as f32));
    }

    pub(crate) fn update_relimit_hover_end(&mut self) {
        let Some(graphics) = self.graphics.as_ref() else {
            return;
        };
        let vp = graphics.view_proj();
        let screen = graphics.screen_size_pub();

        // Phase 1: soft-pick the candidate target line and project it to screen for a yellow
        // highlight.
        if self.editor.relimit_waiting_for_pick {
            let picked = self.graphics.as_ref().and_then(|g| {
                g.pick_at_cursor(
                    PICK_THRESHOLD_PX,
                    &self.triangulations,
                    &self.editor.hidden_handles,
                    &self.editor.frozen_handles,
                    self.editor.xray_enabled,
                )
            });
            let candidate = picked.and_then(|(e, _)| match e {
                SceneEntityId::Object(id) if Some(id) != self.editor.relimit_source_id => Some(id),
                _ => None,
            });
            if candidate != self.editor.relimit_hover_target_id {
                self.editor.relimit_hover_target_id = candidate;
                self.editor.relimit_hover_target_screen_px.clear();
                if let Some(id) = candidate {
                    let (pts, closed) = match self.active_document().get_object(id) {
                        Some(Object::Polyline { verts, closed, .. }) => (
                            verts
                                .iter()
                                .filter_map(|v| pick::world_to_screen(&vp, v.pos, screen))
                                .map(|sp| (sp.x as f32, sp.y as f32))
                                .collect::<Vec<_>>(),
                            *closed,
                        ),
                        _ => (Vec::new(), false),
                    };
                    self.editor.relimit_hover_target_screen_px = pts;
                    self.editor.relimit_hover_target_closed = closed;
                }
            }
            return;
        }

        // Phase 2: select whichever candidate operation's handle is nearest the cursor.
        if !self.editor.relimit_confirming_end {
            return;
        }
        self.select_relimit_candidate_nearest_cursor();
    }

    fn commit_relimit_intersect(&mut self) {
        let Some(source_id) = self.editor.relimit_source_id else {
            return;
        };
        let Some(intersection) = self.editor.relimit_intersection_3d else {
            return;
        };
        let trim_end = self.editor.relimit_hover_end;

        let before = match self.active_document().get_object(source_id) {
            Some(obj) => obj.clone(),
            None => return,
        };
        let mut after = before.clone();
        if let Object::Polyline { verts, .. } = &mut after {
            match trim_end {
                TrimEnd::Start => {
                    if let Some(v) = verts.first_mut() {
                        v.pos = intersection;
                    }
                }
                TrimEnd::End => {
                    if let Some(v) = verts.last_mut() {
                        v.pos = intersection;
                    }
                }
            }
        }

        if let Some(project) = self.workspace.active_project_mut() {
            self.history.execute(
                &mut project.pidb.document,
                Command::Replace { before, after },
            );
            project.dirty = true;
        }
        self.editor.relimit_confirming_end = false;
        self.editor.relimit_waiting_for_pick = false;
        self.editor.relimit_awaiting_source_pick = false;
        self.editor.relimit_dialog_open = false;
        self.editor.relimit_source_id = None;
        self.editor.relimit_second_id = None;
        self.editor.relimit_intersection_3d = None;
        self.editor.relimit_candidates.clear();
        self.editor.relimit_hover_target_id = None;
        self.editor.relimit_hover_target_screen_px.clear();
        self.editor.relimit_preview_from_px = None;
        self.editor.relimit_preview_to_px = None;
        self.editor.active_tool = ActiveTool::None;
        self.invalidate_geometry();
    }

    pub(crate) fn relimit_resize(&mut self, source_id: ObjectId, mode: RelimitMode, value: f64) {
        let before = match self.active_document().get_object(source_id) {
            Some(obj @ Object::Polyline { .. }) => obj.clone(),
            _ => return,
        };
        let mut after = before.clone();
        let resize_end = self.editor.relimit_resize_end;
        if let Object::Polyline { verts, .. } = &mut after {
            if verts.len() < 2 {
                return;
            }
            let start = verts[0].pos;
            let end = verts[verts.len() - 1].pos;
            let dir = end - start;
            let current_len = dir.length();
            if current_len < 1e-9 {
                return;
            }
            let unit = dir / current_len;
            let new_len = match mode {
                RelimitMode::AbsoluteLength => value,
                RelimitMode::RelativeLength => current_len + value,
                RelimitMode::Intersect => return, // handled separately
            };
            if !new_len.is_finite() || new_len <= 0.0 {
                return;
            }
            let last = verts.len() - 1;
            match resize_end {
                crate::ui::state::TrimEnd::End => verts[last].pos = start + unit * new_len,
                crate::ui::state::TrimEnd::Start => verts[0].pos = end - unit * new_len,
            }
        }

        if let Some(project) = self.workspace.active_project_mut() {
            self.history.execute(
                &mut project.pidb.document,
                Command::Replace { before, after },
            );
            project.dirty = true;
        }
        self.editor.relimit_dialog_open = false;
        self.editor.relimit_awaiting_source_pick = false;
        self.editor.relimit_source_id = None;
        self.editor.relimit_second_id = None;
        self.editor.relimit_intersection_3d = None;
        self.editor.relimit_candidates.clear();
        self.editor.relimit_hover_target_id = None;
        self.editor.relimit_hover_target_screen_px.clear();
        self.editor.relimit_preview_from_px = None;
        self.editor.relimit_preview_to_px = None;
        self.editor.active_tool = ActiveTool::None;
        self.invalidate_geometry();
    }
}

// -------------------------------------------------------------------------
// World-space (XY plane) line intersection helper (outside the impl)
// -------------------------------------------------------------------------

/// Relative slack allowed on the target segment parameter `u` beyond `[0, 1]` so
/// that two lines which already *touch* at a point (e.g. one endpoint sitting
/// exactly on the other line, perhaps from a prior relimit) still count as
/// intersecting there, rather than being rejected by floating-point drift that
/// pushes the computed `u` a hair outside the exact boundary.
const TOUCH_EPSILON: f64 = 1e-6;

/// Intersect the *infinite* line through `a`,`b` with the *segment* `c`–`d`.
/// Returns the parameter `t` along `a`→`b` (any value), but only when the hit
/// lies on (or within `TOUCH_EPSILON` of) the actual `c`–`d` segment — so a
/// polygon edge is never extended to an arbitrary off-edge point, while a
/// touching endpoint still resolves.
fn line_line_intersect_t(
    a: glam::DVec2,
    b: glam::DVec2,
    c: glam::DVec2,
    d: glam::DVec2,
) -> Option<f64> {
    let r = b - a;
    let s = d - c;
    let denom = r.x * s.y - r.y * s.x;
    if denom.abs() < 1e-9 {
        return None;
    }
    let t = ((c.x - a.x) * s.y - (c.y - a.y) * s.x) / denom;
    let u = ((c.x - a.x) * r.y - (c.y - a.y) * r.x) / denom;
    if !(-TOUCH_EPSILON..=1.0 + TOUCH_EPSILON).contains(&u) {
        return None;
    }
    Some(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec2;

    #[test]
    fn finds_a_proper_crossing() {
        // Infinite line through (0,0)->(10,0) crosses segment (5,-5)->(5,5) at t=0.5.
        let t = line_line_intersect_t(
            DVec2::new(0.0, 0.0),
            DVec2::new(10.0, 0.0),
            DVec2::new(5.0, -5.0),
            DVec2::new(5.0, 5.0),
        );
        assert!(t.is_some());
        assert!((t.unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn finds_a_hit_exactly_at_the_target_segment_endpoint() {
        // The target segment starts exactly where it touches the source's line.
        let t = line_line_intersect_t(
            DVec2::new(0.0, 0.0),
            DVec2::new(10.0, 0.0),
            DVec2::new(5.0, 0.0),
            DVec2::new(5.0, 5.0),
        );
        assert!(t.is_some());
        assert!((t.unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn tolerates_floating_point_drift_just_past_the_target_endpoint() {
        // Simulates a touch point computed by a previous relimit landing a hair
        // outside the exact [0, 1] segment range due to float rounding.
        let t = line_line_intersect_t(
            DVec2::new(0.0, 0.0),
            DVec2::new(10.0, 0.0),
            DVec2::new(5.0, -1e-9),
            DVec2::new(5.0, 5.0),
        );
        assert!(t.is_some(), "a touch point just past u=0 should still hit");
    }

    #[test]
    fn rejects_a_miss_well_outside_the_segment() {
        let t = line_line_intersect_t(
            DVec2::new(0.0, 0.0),
            DVec2::new(10.0, 0.0),
            DVec2::new(5.0, 1.0),
            DVec2::new(5.0, 5.0),
        );
        assert!(t.is_none());
    }
}
