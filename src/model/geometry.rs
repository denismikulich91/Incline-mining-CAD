//! Double-precision geometry shared by import, editing and scene preparation.

use glam::{DMat4, DQuat, DVec2, DVec3};

pub(crate) fn point_to_dvec3(point: &dxf::Point) -> DVec3 {
    DVec3::new(point.x, point.y, point.z)
}

pub(crate) fn vector_to_dvec3(vector: &dxf::Vector) -> DVec3 {
    DVec3::new(vector.x, vector.y, vector.z)
}

pub(crate) fn normalize_or_z(vector: DVec3) -> DVec3 {
    if vector.length_squared() <= f64::EPSILON {
        DVec3::Z
    } else {
        vector.normalize()
    }
}

#[derive(Clone)]
pub(crate) struct Transform {
    stack: Vec<DMat4>,
}

impl Transform {
    pub(crate) fn identity() -> Self {
        Self {
            stack: vec![DMat4::IDENTITY],
        }
    }

    pub(crate) fn apply(&self, point: DVec3) -> DVec3 {
        self.matrix().transform_point3(point)
    }

    pub(crate) fn matrix(&self) -> DMat4 {
        self.stack.last().copied().unwrap_or(DMat4::IDENTITY)
    }

    pub(crate) fn push(&mut self, base: DVec3, translation: DVec3, rotation: f64, scale: DVec3) {
        self.push_with_affine(base, translation, rotation, scale, DMat4::IDENTITY);
    }

    pub(crate) fn push_with_affine(
        &mut self,
        base: DVec3,
        translation: DVec3,
        rotation: f64,
        scale: DVec3,
        affine: DMat4,
    ) {
        let local = affine
            * DMat4::from_translation(translation)
            * DMat4::from_quat(DQuat::from_rotation_z(rotation))
            * DMat4::from_scale(scale)
            * DMat4::from_translation(-base);
        self.stack.push(self.matrix() * local);
    }

    pub(crate) fn pop(&mut self) {
        if self.stack.len() > 1 {
            self.stack.pop();
        }
    }
}

// -------------------------------------------------------------------------
// Geometric polygon/polyline offset
// -------------------------------------------------------------------------

/// Intersect adjacent offset edge lines; `None` when (near-)parallel.
fn intersect_offset_edges(a: DVec2, r: DVec2, b: DVec2, s: DVec2) -> Option<DVec2> {
    crate::model::kernel::line_line(a, r, b, s)
}

/// When the mitre extension at a corner is larger than this multiple of the offset
/// distance, the corner is bevelled instead to prevent self-intersecting output.
const MITER_LIMIT: f64 = 4.0;

/// Offset each edge of a polyline/polygon perpendicular to itself in the XY plane,
/// then intersect adjacent offset edges to find the new vertices.
///
/// `signed_horiz_dist > 0` offsets to the **left** of each directed edge (CCW = inward).
/// `z_delta` is added to every vertex Z (use `0.0` for a horizontal berm offset).
///
/// For open polylines the first/last vertices are the respective endpoints of the
/// first/last offset edges (no intersection needed).
pub(crate) fn geometric_offset(
    verts: &[DVec3],
    closed: bool,
    signed_horiz_dist: f64,
    z_delta: f64,
) -> Vec<DVec3> {
    let n = verts.len();
    if n < 2 {
        return verts.to_vec();
    }

    // Compute per-edge normals (left perpendicular, XY only).
    let edge_count = if closed { n } else { n - 1 };
    let mut normals: Vec<DVec2> = Vec::with_capacity(edge_count);
    let mut dirs: Vec<DVec2> = Vec::with_capacity(edge_count);
    for i in 0..edge_count {
        let a = verts[i].truncate();
        let b = verts[(i + 1) % n].truncate();
        let d = b - a;
        let len = d.length();
        let dir = if len > 1e-10 { d / len } else { DVec2::X };
        dirs.push(dir);
        normals.push(DVec2::new(-dir.y, dir.x)); // left perpendicular
    }

    // Build offset edge start points.
    let offset_starts: Vec<DVec2> = (0..edge_count)
        .map(|i| verts[i].truncate() + signed_horiz_dist * normals[i])
        .collect();

    // Compute new vertex positions.
    let mut result = Vec::with_capacity(n);

    if closed {
        for i in 0..n {
            let prev = (i + edge_count - 1) % edge_count;
            let a = offset_starts[prev];
            let b = offset_starts[i];
            // Apply mitre limit: if the corner extension exceeds MITER_LIMIT × offset
            // the polygon would self-intersect (tight inward corner); bevel instead.
            let max_ext = MITER_LIMIT * signed_horiz_dist.abs();
            let intersection = intersect_offset_edges(a, dirs[prev], b, dirs[i])
                .filter(|&pt| {
                    signed_horiz_dist.abs() < 1e-10
                        || (pt - verts[i].truncate()).length() <= max_ext
                })
                .unwrap_or_else(|| {
                    // Parallel or overconstrained corner: bevel by averaging edge endpoints.
                    let end_of_prev = offset_starts[prev]
                        + dirs[prev] * (verts[i].truncate() - verts[prev % n].truncate()).length();
                    (end_of_prev + b) * 0.5
                });
            result.push(DVec3::new(
                intersection.x,
                intersection.y,
                verts[i].z + z_delta,
            ));
        }
    } else {
        // First vertex: start of first offset edge.
        result.push(DVec3::new(
            offset_starts[0].x,
            offset_starts[0].y,
            verts[0].z + z_delta,
        ));
        // Interior vertices: intersection of adjacent offset edges.
        for i in 1..n - 1 {
            let a = offset_starts[i - 1];
            let b = offset_starts[i];
            let max_ext = MITER_LIMIT * signed_horiz_dist.abs();
            let intersection = intersect_offset_edges(a, dirs[i - 1], b, dirs[i])
                .filter(|&pt| {
                    signed_horiz_dist.abs() < 1e-10
                        || (pt - verts[i].truncate()).length() <= max_ext
                })
                .unwrap_or_else(|| {
                    let end_of_prev = offset_starts[i - 1]
                        + dirs[i - 1] * (verts[i].truncate() - verts[i - 1].truncate()).length();
                    (end_of_prev + b) * 0.5
                });
            result.push(DVec3::new(
                intersection.x,
                intersection.y,
                verts[i].z + z_delta,
            ));
        }
        // Last vertex: end of last offset edge.
        let last_edge = edge_count - 1;
        let end = offset_starts[last_edge]
            + dirs[last_edge] * {
                let a = verts[n - 2].truncate();
                let b = verts[n - 1].truncate();
                (b - a).length()
            };
        result.push(DVec3::new(end.x, end.y, verts[n - 1].z + z_delta));
    }

    result
}

/// Project each vertex of a polyline outward (perpendicular, XY) by the horizontal
/// distance implied by *its own* elevation and a fixed batter angle, so every output
/// vertex lands flat at `target_rl` — as if a batter wall of that angle were cut from
/// the string down (or up) to the target level at each point along its length.
///
/// Unlike `geometric_offset` (uniform per-edge XY offset + uniform Z shift), the
/// horizontal offset here varies per vertex with `verts[i].z`, so a string that
/// changes elevation along its length (e.g. a ramp crest) ends up flat rather than
/// retaining its original elevation profile.
///
/// `side` selects which side of the polyline to project toward (see
/// `offset_side_from_cursor`).
pub(crate) fn geometric_offset_project_to_rl(
    verts: &[DVec3],
    closed: bool,
    side: f64,
    tan_angle: f64,
    target_rl: f64,
) -> Vec<DVec3> {
    let n = verts.len();
    if n < 2 {
        return verts.to_vec();
    }

    let edge_count = if closed { n } else { n - 1 };
    let mut normals: Vec<DVec2> = Vec::with_capacity(edge_count);
    for i in 0..edge_count {
        let a = verts[i].truncate();
        let b = verts[(i + 1) % n].truncate();
        let d = b - a;
        let len = d.length();
        let dir = if len > 1e-10 { d / len } else { DVec2::X };
        normals.push(DVec2::new(-dir.y, dir.x));
    }

    let vertex_normal = |i: usize| -> DVec2 {
        let (prev, next) = if closed {
            (
                normals[(i + edge_count - 1) % edge_count],
                normals[i % edge_count],
            )
        } else if i == 0 {
            return normals[0];
        } else if i == n - 1 {
            return normals[edge_count - 1];
        } else {
            (normals[i - 1], normals[i])
        };
        let sum = prev + next;
        if sum.length() > 1e-10 {
            sum.normalize()
        } else {
            next
        }
    };

    verts
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dist = if tan_angle.abs() < 1e-9 {
                0.0
            } else {
                (target_rl - v.z) / tan_angle
            };
            let xy = v.truncate() + side * dist * vertex_normal(i);
            DVec3::new(xy.x, xy.y, target_rl)
        })
        .collect()
}

pub(crate) const ROAD_INTERSECTION_FLAT_CLEARANCE_M: f64 = 25.0;

pub(crate) fn points_coincident(a: DVec3, b: DVec3) -> bool {
    crate::model::kernel::points_coincident_3d(a, b)
}

/// Signed area of a polygon (XY plane only). Positive = CCW, negative = CW.
pub(crate) fn signed_area_xy(verts: &[DVec3]) -> f64 {
    let n = verts.len();
    let mut area = 0.0_f64;
    for i in 0..n {
        let a = verts[i];
        let b = verts[(i + 1) % n];
        area += a.x * b.y - b.x * a.y;
    }
    area * 0.5
}

/// Return which sign of `horiz_dist` places the offset on the cursor's side.
/// Returns `1.0` or `-1.0`. For closed polygons uses a point-in-polygon test
/// so the result correctly tracks inside vs outside regardless of polygon size.
pub(crate) fn offset_side_from_cursor(
    verts: &[DVec3],
    closed: bool,
    cursor_xy: DVec2,
    abs_dist: f64,
) -> f64 {
    if verts.len() < 2 || abs_dist < 1e-10 {
        return 1.0;
    }

    if closed && verts.len() >= 3 {
        // geometric_offset positive d = left of each edge = inward for CCW, outward for CW.
        let area = signed_area_xy(verts);
        let positive_is_outward = area < 0.0; // CW polygon: left normal points outward
        let cursor_inside = point_in_polygon_xy(cursor_xy, verts);
        // cursor inside → want inward; cursor outside → want outward.
        let want_outward = !cursor_inside;
        if want_outward == positive_is_outward {
            1.0
        } else {
            -1.0
        }
    } else {
        // Open polyline: compare which offset centroid is closer to the cursor.
        let pos = compute_offset_centroid(verts, closed, abs_dist);
        let neg = compute_offset_centroid(verts, closed, -abs_dist);
        if (pos - cursor_xy).length_squared() <= (neg - cursor_xy).length_squared() {
            1.0
        } else {
            -1.0
        }
    }
}

/// Point-in-polygon test on the XY plane. Points on the boundary (within
/// `kernel::XY_TOL`) count as inside; use [`crate::model::kernel::point_in_polygon`]
/// directly when the boundary case needs explicit handling.
pub(crate) fn point_in_polygon_xy(point: DVec2, verts: &[DVec3]) -> bool {
    !matches!(
        crate::model::kernel::point_in_polygon(point, verts.iter().map(|v| v.truncate())),
        crate::model::kernel::PolyContainment::Outside
    )
}

/// Check if segment AB strictly intersects segment CD (not at shared endpoints).
/// Returns the intersection point and its parameter along AB if found.
fn seg_seg_intersection_2d(a: DVec2, b: DVec2, c: DVec2, d: DVec2) -> Option<(DVec2, f64)> {
    match crate::model::kernel::segment_segment(a, b, c, d) {
        crate::model::kernel::SegSeg::Crossing { point, t, .. } => Some((point, t)),
        _ => None,
    }
}

/// Remove self-intersections from a closed polygon offset result (XY plane).
///
/// When an inward offset is too large for a sharp corner the adjacent edges
/// fold back and cross, creating a small loop. Keeps the loop with the
/// greatest XY area (the polygon body); see `split_self_intersection_loops`
/// for callers that want the discarded lobes too.
pub(crate) fn remove_self_intersections(verts: Vec<DVec3>) -> Vec<DVec3> {
    if verts.len() < 4 {
        return verts;
    }
    split_self_intersection_loops(verts)
        .into_iter()
        .next()
        .unwrap_or_default()
}

/// Split a closed ring at its self-crossings into simple loops, largest
/// XY area first, in a single sweep.
///
/// All pairwise edge crossings are collected once (`kernel::segment_segment`
/// `Crossing`s only — endpoint touches don't split), inserted along their
/// edges, then loops are peeled with a stack: when the ring returns to a
/// crossing it has already passed, the vertices in between form one loop.
/// Interleaved (non-nested) crossing pairs — which offset fold-backs don't
/// produce — degrade gracefully: the orphaned occurrence stays a pass-through
/// vertex. Each crossing keeps the Z interpolated on its own edge.
pub(crate) fn split_self_intersection_loops(verts: Vec<DVec3>) -> Vec<Vec<DVec3>> {
    let n = verts.len();
    if n < 4 {
        return vec![verts];
    }

    // Crossing occurrences per edge: (t along edge, pair id, position).
    let mut edge_hits: Vec<Vec<(f64, usize, DVec3)>> = vec![Vec::new(); n];
    let mut pair_count = 0usize;
    for i in 0..n {
        let a = verts[i];
        let b = verts[(i + 1) % n];
        // Adjacent edges share an endpoint and cannot properly cross.
        let j_end = if i == 0 { n - 1 } else { n };
        for j in (i + 2)..j_end {
            let c = verts[j];
            let d = verts[(j + 1) % n];
            if let Some((point, t)) =
                seg_seg_intersection_2d(a.truncate(), b.truncate(), c.truncate(), d.truncate())
            {
                let u = {
                    let cd = (d - c).truncate();
                    let len_sq = cd.length_squared();
                    if len_sq > 0.0 {
                        (point - c.truncate()).dot(cd) / len_sq
                    } else {
                        0.0
                    }
                };
                let z_i = a.z + t * (b.z - a.z);
                let z_j = c.z + u * (d.z - c.z);
                edge_hits[i].push((t, pair_count, DVec3::new(point.x, point.y, z_i)));
                edge_hits[j].push((u, pair_count, DVec3::new(point.x, point.y, z_j)));
                pair_count += 1;
            }
        }
    }
    if pair_count == 0 {
        return vec![verts];
    }

    // The ring with crossings inserted in edge order.
    struct AugVertex {
        pos: DVec3,
        pair: Option<usize>,
    }
    let mut ring: Vec<AugVertex> = Vec::with_capacity(n + 2 * pair_count);
    for (i, hits) in edge_hits.iter_mut().enumerate() {
        ring.push(AugVertex {
            pos: verts[i],
            pair: None,
        });
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        for &(_, pair, pos) in hits.iter() {
            ring.push(AugVertex {
                pos,
                pair: Some(pair),
            });
        }
    }

    // Peel loops: `open[pair]` is the stack index of the crossing's first
    // occurrence; the second occurrence closes the loop between them.
    let mut open: Vec<Option<usize>> = vec![None; pair_count];
    let mut stack: Vec<AugVertex> = Vec::with_capacity(ring.len());
    let mut loops: Vec<Vec<DVec3>> = Vec::new();
    for vertex in ring {
        let Some(pair) = vertex.pair else {
            stack.push(vertex);
            continue;
        };
        match open[pair] {
            None => {
                open[pair] = Some(stack.len());
                stack.push(vertex);
            }
            Some(start) => {
                let peeled: Vec<DVec3> = stack.drain(start..).map(|v| v.pos).collect();
                // Crossings whose partner left with the peeled loop become
                // plain vertices when the partner shows up later.
                for slot in open.iter_mut() {
                    if slot.is_some_and(|index| index >= start) {
                        *slot = None;
                    }
                }
                if peeled.len() >= 3 {
                    loops.push(peeled);
                }
                // The ring passes through the crossing once more.
                stack.push(AugVertex {
                    pos: vertex.pos,
                    pair: None,
                });
            }
        }
    }
    let remainder: Vec<DVec3> = stack.into_iter().map(|v| v.pos).collect();
    if remainder.len() >= 3 {
        loops.push(remainder);
    }

    loops.sort_by(|a, b| signed_area_xy(b).abs().total_cmp(&signed_area_xy(a).abs()));
    loops
}

fn compute_offset_centroid(verts: &[DVec3], closed: bool, signed_dist: f64) -> DVec2 {
    let offset = geometric_offset(verts, closed, signed_dist, 0.0);
    if offset.is_empty() {
        return DVec2::ZERO;
    }
    let sum: DVec2 = offset.iter().map(|v| v.truncate()).sum();
    sum / offset.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bow_tie_splits_into_both_triangles_and_keeps_one() {
        const E: f64 = 476_000.0;
        const N: f64 = 7_654_000.0;
        let ring = vec![
            DVec3::new(E, N, 100.0),
            DVec3::new(E + 10.0, N, 100.0),
            DVec3::new(E, N + 10.0, 100.0),
            DVec3::new(E + 10.0, N + 10.0, 100.0),
        ];
        let loops = split_self_intersection_loops(ring.clone());
        assert_eq!(loops.len(), 2, "bow-tie must peel into two triangles");
        for a_loop in &loops {
            assert!((signed_area_xy(a_loop).abs() - 25.0).abs() < 1e-6);
            // Each triangle contains the crossing point (E+5, N+5).
            assert!(
                a_loop
                    .iter()
                    .any(|v| (v.truncate() - DVec2::new(E + 5.0, N + 5.0)).length() < 1e-6)
            );
        }
        let kept = remove_self_intersections(ring);
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn double_pinch_peels_all_three_lobes_largest_first() {
        // A comb-shaped ring whose base edge is crossed twice: two 5x4 body
        // lobes above y=0 and one 5x2 pocket below. The old implementation
        // silently discarded all but one lobe.
        let ring = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(15.0, 0.0, 0.0),
            DVec3::new(15.0, 4.0, 0.0),
            DVec3::new(10.0, 4.0, 0.0),
            DVec3::new(10.0, -2.0, 0.0),
            DVec3::new(5.0, -2.0, 0.0),
            DVec3::new(5.0, 4.0, 0.0),
            DVec3::new(0.0, 4.0, 0.0),
        ];
        let loops = split_self_intersection_loops(ring.clone());
        let areas: Vec<f64> = loops
            .iter()
            .map(|a_loop| signed_area_xy(a_loop).abs())
            .collect();
        assert_eq!(areas.len(), 3, "expected two body lobes and the pocket");
        assert!((areas[0] - 20.0).abs() < 1e-9);
        assert!((areas[1] - 20.0).abs() < 1e-9);
        assert!((areas[2] - 10.0).abs() < 1e-9);

        let kept = remove_self_intersections(ring);
        assert!((signed_area_xy(&kept).abs() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn simple_ring_passes_through_unsplit() {
        let ring = vec![
            DVec3::new(0.0, 0.0, 5.0),
            DVec3::new(10.0, 0.0, 5.0),
            DVec3::new(10.0, 10.0, 5.0),
            DVec3::new(0.0, 10.0, 5.0),
        ];
        let loops = split_self_intersection_loops(ring.clone());
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0], ring);
    }

    #[test]
    fn project_to_rl_lands_flat_and_recedes_as_ramp_descends() {
        // A "ramp crest" string descending in Z along +X, like a crest that follows
        // a ramp as it's mined down.
        let verts = vec![
            DVec3::new(0.0, 0.0, 282.0),
            DVec3::new(10.0, 0.0, 279.0),
            DVec3::new(20.0, 0.0, 276.0),
        ];
        let tan_angle = 60f64.to_radians().tan();
        let target_rl = 282.0;
        let result = geometric_offset_project_to_rl(&verts, false, 1.0, tan_angle, target_rl);

        // Every vertex must land exactly at the target RL (flat string).
        for v in &result {
            assert!((v.z - target_rl).abs() < 1e-9);
        }

        // The vertex that started exactly at target RL should not move in XY.
        assert!((result[0].truncate() - verts[0].truncate()).length() < 1e-9);

        // As the source string descends further below the target RL, the crest
        // should recede further away (monotonically increasing offset distance).
        let d1 = (result[1].truncate() - verts[1].truncate()).length();
        let d2 = (result[2].truncate() - verts[2].truncate()).length();
        assert!(
            d2 > d1,
            "expected offset distance to grow as elevation drops further below target RL"
        );

        // Sanity-check the magnitude against the batter geometry: horiz = height / tan(angle).
        let expected_d2 = (target_rl - verts[2].z) / tan_angle;
        assert!((d2 - expected_d2).abs() < 1e-6);
    }

    #[test]
    fn project_to_rl_handles_a_corner_via_bisector_normal() {
        // Open string with a 90-degree turn, so the interior vertex's normal must
        // come from the (prev + next) edge-normal bisector, not a single edge normal.
        let verts = vec![
            DVec3::new(0.0, 0.0, 282.0),
            DVec3::new(10.0, 0.0, 279.0),
            DVec3::new(10.0, 10.0, 276.0),
        ];
        let tan_angle = 60f64.to_radians().tan();
        let target_rl = 282.0;
        let result = geometric_offset_project_to_rl(&verts, false, 1.0, tan_angle, target_rl);

        assert_eq!(result.len(), verts.len());
        for v in &result {
            assert!((v.z - target_rl).abs() < 1e-9);
        }

        // First edge runs along +X, so its left normal is +Y; second edge runs
        // along +Y, so its left normal is -X. The interior vertex's bisector
        // normal should be their (normalized) sum, i.e. pointing into (-X, +Y).
        let dist1 = (target_rl - verts[1].z) / tan_angle;
        let bisector = (DVec2::new(0.0, 1.0) + DVec2::new(-1.0, 0.0)).normalize();
        let expected_xy = verts[1].truncate() + dist1 * bisector;
        assert!((result[1].truncate() - expected_xy).length() < 1e-9);
    }

    #[test]
    fn project_to_rl_on_closed_polygon_keeps_vertex_count_and_flat_z() {
        let verts = vec![
            DVec3::new(0.0, 0.0, 280.0),
            DVec3::new(10.0, 0.0, 278.0),
            DVec3::new(10.0, 10.0, 276.0),
            DVec3::new(0.0, 10.0, 278.0),
        ];
        let tan_angle = 45f64.to_radians().tan();
        let target_rl = 280.0;
        let result = geometric_offset_project_to_rl(&verts, true, 1.0, tan_angle, target_rl);

        assert_eq!(result.len(), verts.len());
        for v in &result {
            assert!((v.z - target_rl).abs() < 1e-9);
        }
        // The vertex already at target RL should be unmoved in XY.
        assert!((result[0].truncate() - verts[0].truncate()).length() < 1e-9);
    }

    #[test]
    fn project_to_rl_flips_side_with_the_sign_argument() {
        let verts = vec![DVec3::new(0.0, 0.0, 280.0), DVec3::new(10.0, 0.0, 270.0)];
        let tan_angle = 45f64.to_radians().tan();
        let target_rl = 280.0;
        let pos_side = geometric_offset_project_to_rl(&verts, false, 1.0, tan_angle, target_rl);
        let neg_side = geometric_offset_project_to_rl(&verts, false, -1.0, tan_angle, target_rl);

        let d_pos = pos_side[1].y - verts[1].y;
        let d_neg = neg_side[1].y - verts[1].y;
        assert!(d_pos > 0.0);
        assert!(d_neg < 0.0);
        assert!((d_pos + d_neg).abs() < 1e-9);
    }
}
