use crate::{
    app::{App, PICK_THRESHOLD_PX},
    model::{Command, Object, PolyVertex, SceneEntityId},
    ui::state::{ActiveTool, BatterBermMode, BatterBermPreviewKey},
    userspace_log,
};

impl<'a> App<'a> {
    pub(crate) fn open_batter_berm_dialog(&mut self) {
        let object_id = self.editor.selected_handles.iter().find_map(|h| match h {
            SceneEntityId::Object(id)
                if self
                    .workspace
                    .active_document()
                    .and_then(|document| document.get_object(*id))
                    .is_some_and(|object| matches!(object, Object::Polyline { .. })) =>
            {
                Some(*id)
            }
            _ => None,
        });
        if let Some(id) = object_id {
            if !self.activate_project_for_object(id) {
                return;
            }
            self.editor.batter_berm_target_id = Some(id);
            self.editor.batter_berm_dialog_open = true;
            self.editor.batter_berm_preview_key = None;
            self.editor.tool_highlight_id = Some(id);
            let closed = matches!(
                self.active_document().get_object(id),
                Some(Object::Polyline { closed: true, .. })
            );
            self.editor.batter_berm_preview_closed = closed;
            userspace_log!("Opened batter berm dialog for object {:?}", id);
            self.invalidate_geometry();
        }
    }

    pub(crate) fn pick_batter_berm_target(&mut self) {
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
            && self.activate_project_for_object(id)
            && matches!(
                self.active_document().get_object(id),
                Some(Object::Polyline { .. })
            )
        {
            self.editor.batter_berm_target_id = Some(id);
            self.editor.batter_berm_dialog_open = true;
            self.editor.batter_berm_preview_key = None;
            self.editor.tool_highlight_id = Some(id);
            let closed = matches!(
                self.active_document().get_object(id),
                Some(Object::Polyline { closed: true, .. })
            );
            self.editor.batter_berm_preview_closed = closed;
            self.invalidate_geometry();
        }
    }

    /// Recompute preview rings from current dialog inputs. Called every render frame while open.
    pub(crate) fn update_batter_berm_preview(&mut self) {
        if !self.editor.batter_berm_dialog_open {
            return;
        }
        let Some(object_id) = self.editor.batter_berm_target_id else {
            return;
        };
        if !self.activate_project_for_object(object_id) {
            return;
        }

        let document_revision = self.active_document().revision();
        let berm_width = self.editor.batter_berm_width;
        let angle_deg = self.editor.batter_berm_angle;
        let bench_height = self.editor.batter_berm_bench_height;
        let benches = self.editor.batter_berm_benches;
        let mode = self.editor.batter_berm_mode;
        let preview_key = BatterBermPreviewKey {
            target_id: object_id,
            document_revision,
            width: berm_width,
            angle: angle_deg,
            bench_height,
            benches,
            mode,
        };
        if self.editor.batter_berm_preview_key == Some(preview_key) {
            return;
        }

        let (src_verts, closed) = match self.active_document().get_object(object_id) {
            Some(Object::Polyline { verts, closed, .. }) => (
                crate::model::geometry::tessellate_polyline_bulges(verts, *closed),
                *closed,
            ),
            _ => return,
        };

        if berm_width <= 0.0
            || angle_deg <= 0.0
            || angle_deg >= 90.0
            || bench_height <= 0.0
            || benches == 0
        {
            self.editor.batter_berm_max_benches = 1;
            self.editor.batter_berm_rings_world.clear();
            self.editor.batter_berm_guides_world.clear();
            self.editor.batter_berm_preview_key = Some(preview_key);
            return;
        }

        let (side, delta_height) = mode_to_side_and_dz(&src_verts, closed, mode, bench_height);
        let batter_horiz = batter_horizontal_dist(angle_deg, bench_height);
        let max_benches = max_batter_berm_benches(
            &src_verts,
            closed,
            side,
            batter_horiz,
            berm_width,
            delta_height,
            100,
        );
        self.editor.batter_berm_max_benches = max_benches;
        self.editor.batter_berm_benches = benches.min(max_benches.max(1));

        let rings = if max_benches == 0 {
            Vec::new()
        } else {
            compute_batter_berm_rings(
                &src_verts,
                closed,
                side,
                batter_horiz,
                berm_width,
                delta_height,
                self.editor.batter_berm_benches,
            )
        };
        let guides = compute_guide_segments(&src_verts, &rings, closed);

        self.editor.batter_berm_source_world = src_verts;
        self.editor.batter_berm_rings_world = rings;
        self.editor.batter_berm_guides_world = guides;
        self.editor.batter_berm_preview_closed = closed;
        self.editor.batter_berm_preview_key = Some(BatterBermPreviewKey {
            benches: self.editor.batter_berm_benches,
            ..preview_key
        });
    }

    /// Commit all rings from the current preview as new polylines.
    pub(crate) fn commit_batter_berm(&mut self) {
        let Some(object_id) = self.editor.batter_berm_target_id else {
            return;
        };
        if !self.activate_project_for_object(object_id) {
            return;
        }

        if self.editor.batter_berm_rings_world.is_empty() {
            return;
        }

        let (layer, color, fill, line_weight, closed) =
            match self.active_document().get_object(object_id) {
                Some(Object::Polyline {
                    layer,
                    color,
                    fill,
                    line_weight,
                    closed,
                    ..
                }) => (*layer, *color, *fill, *line_weight, *closed),
                _ => return,
            };

        let rings = self.editor.batter_berm_rings_world.clone();

        if let Some(project) = self.workspace.active_project_mut() {
            let doc = &mut project.pidb.document;
            let commands = rings
                .into_iter()
                .map(|ring_verts| {
                    let new_verts: Vec<PolyVertex> =
                        ring_verts.into_iter().map(PolyVertex::straight).collect();
                    let id = doc.allocate_object_id();
                    Command::AddObject(Object::Polyline {
                        id,
                        layer,
                        verts: new_verts,
                        closed,
                        color,
                        fill,
                        line_weight,
                    })
                })
                .collect::<Vec<_>>();
            if !commands.is_empty() {
                self.history.execute(doc, Command::Batch(commands));
            }
        }

        self.cancel_batter_berm();
        userspace_log!("Created batter berm from object {:?}", object_id);
        self.invalidate_geometry();
    }

    pub(crate) fn cancel_batter_berm(&mut self) {
        self.editor.batter_berm_dialog_open = false;
        self.editor.batter_berm_target_id = None;
        self.editor.batter_berm_rings_world.clear();
        self.editor.batter_berm_source_world.clear();
        self.editor.batter_berm_guides_world.clear();
        self.editor.batter_berm_rings_screen_px.clear();
        self.editor.batter_berm_source_screen_px.clear();
        self.editor.batter_berm_guides_screen_px.clear();
        self.editor.tool_highlight_id = None;
        self.editor.batter_berm_preview_key = None;
        self.editor.active_tool = ActiveTool::None;
        self.invalidate_geometry();
        userspace_log!("Cancelled batter berm tool");
    }
}

/// Both pit and stockpile go inward. Pit goes down, stockpile goes up.
fn mode_to_side_and_dz(
    verts: &[glam::DVec3],
    closed: bool,
    mode: BatterBermMode,
    bench_height: f64,
) -> (f64, f64) {
    let side = inward_side(verts, closed);
    let delta_z = match mode {
        BatterBermMode::Pit => -bench_height,
        BatterBermMode::Stockpile => bench_height,
    };
    (side, delta_z)
}

/// Returns the sign that offsets geometry inward for the given polygon.
fn inward_side(verts: &[glam::DVec3], closed: bool) -> f64 {
    if closed && verts.len() >= 3 {
        // Positive offset = left of directed edges = inward for CCW polygons.
        let area = crate::model::geometry::signed_area_xy(verts);
        if area > 0.0 { 1.0 } else { -1.0 }
    } else {
        -1.0
    }
}

/// Horizontal run of the batter face for a given angle and bench height.
fn batter_horizontal_dist(angle_deg: f64, bench_height: f64) -> f64 {
    let tan = angle_deg.to_radians().tan();
    if tan.abs() < 1e-9 {
        0.0
    } else {
        bench_height / tan
    }
}

/// Produce iteration rings: toe_0, berm_0, toe_1, berm_1, …, toe_(N-1).
/// The last iteration has no trailing berm.
fn compute_batter_berm_rings(
    src_verts: &[glam::DVec3],
    closed: bool,
    side: f64,
    batter_horiz: f64,
    berm_width: f64,
    delta_height: f64,
    benches: u32,
) -> Vec<Vec<glam::DVec3>> {
    use crate::model::geometry::geometric_offset;
    let n = benches as usize;
    let mut rings = Vec::with_capacity(2 * n - 1);
    let mut current = src_verts.to_vec();
    for i in 0..n {
        let toe = clean_offset(
            geometric_offset(&current, closed, side * batter_horiz, delta_height),
            closed,
        );
        if !valid_inward_offset(&current, &toe, closed) {
            break;
        }
        let is_last = i + 1 == n;
        if is_last {
            rings.push(toe);
        } else {
            let berm = clean_offset(
                geometric_offset(&toe, closed, side * berm_width, 0.0),
                closed,
            );
            rings.push(toe);
            if !valid_inward_offset(rings.last().expect("toe was just pushed"), &berm, closed) {
                break;
            }
            rings.push(berm.clone());
            current = berm;
        }
    }
    rings
}

fn max_batter_berm_benches(
    src_verts: &[glam::DVec3],
    closed: bool,
    side: f64,
    batter_horiz: f64,
    berm_width: f64,
    delta_height: f64,
    hard_limit: u32,
) -> u32 {
    use crate::model::geometry::geometric_offset;

    let mut current = src_verts.to_vec();
    let mut max_benches = 0;
    for bench in 1..=hard_limit {
        let toe = clean_offset(
            geometric_offset(&current, closed, side * batter_horiz, delta_height),
            closed,
        );
        if !valid_inward_offset(&current, &toe, closed) {
            break;
        }
        max_benches = bench;

        let berm = clean_offset(
            geometric_offset(&toe, closed, side * berm_width, 0.0),
            closed,
        );
        if !valid_inward_offset(&toe, &berm, closed) {
            break;
        }
        current = berm;
    }
    max_benches
}

fn valid_inward_offset(previous: &[glam::DVec3], next: &[glam::DVec3], closed: bool) -> bool {
    use crate::model::geometry::{point_in_polygon_xy, signed_area_xy};

    if !closed {
        return next.len() >= 2;
    }
    if previous.len() < 3 || next.len() < 3 {
        return false;
    }

    let previous_area = signed_area_xy(previous);
    let next_area = signed_area_xy(next);
    let area_tolerance = previous_area.abs().max(1.0) * 1e-9;
    if previous_area.signum() != next_area.signum()
        || next_area.abs() >= previous_area.abs() - area_tolerance
    {
        return false;
    }

    // Checking vertices plus edge midpoints catches a cleaned ring that has
    // folded across or escaped the preceding boundary.
    for i in 0..next.len() {
        let a = next[i];
        let b = next[(i + 1) % next.len()];
        if !point_in_polygon_xy(a.truncate(), previous)
            || !point_in_polygon_xy(a.lerp(b, 0.5).truncate(), previous)
        {
            return false;
        }
    }
    true
}

fn clean_offset(verts: Vec<glam::DVec3>, closed: bool) -> Vec<glam::DVec3> {
    if closed {
        crate::model::geometry::remove_self_intersections(verts)
    } else {
        verts
    }
}

/// Connect each generated vertex to the nearest point on the preceding ring.
/// This stays stable when loop cleanup changes vertex counts between offsets.
fn compute_guide_segments(
    source: &[glam::DVec3],
    rings: &[Vec<glam::DVec3>],
    closed: bool,
) -> Vec<(glam::DVec3, glam::DVec3)> {
    let mut guides = Vec::new();
    let mut previous = source;
    for ring in rings {
        guides.extend(ring.iter().filter_map(|&point| {
            nearest_point_on_polyline_xy(point, previous, closed).map(|nearest| (nearest, point))
        }));
        previous = ring;
    }
    guides
}

fn nearest_point_on_polyline_xy(
    point: glam::DVec3,
    polyline: &[glam::DVec3],
    closed: bool,
) -> Option<glam::DVec3> {
    if polyline.len() < 2 {
        return None;
    }

    let edge_count = if closed {
        polyline.len()
    } else {
        polyline.len() - 1
    };
    let point_xy = point.truncate();
    let mut best = None;
    let mut best_distance_sq = f64::INFINITY;

    for i in 0..edge_count {
        let a = polyline[i];
        let b = polyline[(i + 1) % polyline.len()];
        let ab = b.truncate() - a.truncate();
        let length_sq = ab.length_squared();
        let t = if length_sq <= 1e-20 {
            0.0
        } else {
            ((point_xy - a.truncate()).dot(ab) / length_sq).clamp(0.0, 1.0)
        };
        let candidate = a.lerp(b, t);
        let distance_sq = (point_xy - candidate.truncate()).length_squared();
        if distance_sq < best_distance_sq {
            best_distance_sq = distance_sq;
            best = Some(candidate);
        }
    }

    best
}
