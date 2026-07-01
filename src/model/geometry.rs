//! Double-precision geometry shared by import, editing and scene preparation.

use glam::{DMat4, DQuat, DVec2, DVec3};

use super::{Document, Object};

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

/// Compute the 2D intersection parameter `t` on line A+t*r where it meets B+u*s.
/// Returns `None` when lines are parallel.
fn intersect_offset_edges(a: DVec2, r: DVec2, b: DVec2, s: DVec2) -> Option<DVec2> {
    let denom = r.x * s.y - r.y * s.x;
    if denom.abs() < 1e-10 {
        return None;
    }
    let t = ((b.x - a.x) * s.y - (b.y - a.y) * s.x) / denom;
    Some(a + t * r)
}

/// When the mitre extension at a corner is larger than this multiple of the offset
/// distance, the corner is bevelled instead to prevent self-intersecting output.
const MITER_LIMIT: f64 = 4.0;
pub(crate) const MIN_ROAD_TURN_ANGLE_DEGREES: f64 = 30.0;

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

pub(crate) const ROAD_INTERSECTION_FLAT_CLEARANCE_M: f64 = 10.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RoadCenterlineRuleError {
    SegmentTooShort {
        required: f64,
        actual: f64,
    },
    DegenerateSegment,
    TurnTooSharp {
        minimum_degrees: f64,
        actual_degrees: f64,
    },
    SegmentTooSteep {
        maximum_degrees: f64,
        actual_degrees: f64,
    },
}

impl std::fmt::Display for RoadCenterlineRuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoadCenterlineRuleError::SegmentTooShort { required, actual } => write!(
                f,
                "Road segment is too short for {required:.1} m flat intersection approach(es): {actual:.2} m available"
            ),
            RoadCenterlineRuleError::DegenerateSegment => {
                f.write_str("Road segment is too short to place an intersection approach")
            }
            RoadCenterlineRuleError::TurnTooSharp {
                minimum_degrees,
                actual_degrees,
            } => write!(
                f,
                "Road turn is too sharp: {actual_degrees:.1}° is below the {minimum_degrees:.0}° minimum"
            ),
            RoadCenterlineRuleError::SegmentTooSteep {
                maximum_degrees,
                actual_degrees,
            } => write!(
                f,
                "Road segment angle is too steep: {actual_degrees:.1}° exceeds the {maximum_degrees:.1}° maximum"
            ),
        }
    }
}

/// Insert mandatory flat road approaches next to road gradients and intersections.
///
/// Any segment that changes Z reserves flat clearance at both ends before the
/// ramp begins/ends. Junctions can require more clearance so road edges miter
/// cleanly around the connected road width.
pub(crate) fn road_centerline_with_intersection_flats(
    centerline: &[DVec3],
    document: &Document,
    road_width: f64,
) -> Result<Vec<DVec3>, RoadCenterlineRuleError> {
    if centerline.len() < 2 {
        return Ok(centerline.to_vec());
    }
    validate_road_turn_angles(centerline)?;
    validate_road_endpoint_attachment_turns(centerline, document)?;

    let mut flat_after: Vec<f64> = vec![0.0; centerline.len()];
    let mut flat_before: Vec<f64> = vec![0.0; centerline.len()];

    for index in 0..centerline.len() - 1 {
        let a = centerline[index];
        let b = centerline[index + 1];
        if !segment_is_inclined(a, b) {
            continue;
        }

        flat_after[index] =
            flat_after[index].max(road_flat_clearance_at_point(a, b, document, road_width));
        flat_before[index + 1] =
            flat_before[index + 1].max(road_flat_clearance_at_point(b, a, document, road_width));
    }

    for index in 0..centerline.len() - 1 {
        let required = flat_after[index] + flat_before[index + 1];
        if required > 0.0 {
            let actual = horizontal_distance(centerline[index], centerline[index + 1]);
            if actual + 1e-9 < required {
                return Err(RoadCenterlineRuleError::SegmentTooShort { required, actual });
            }
        }
    }

    let extra_vertices = flat_after
        .iter()
        .filter(|&&clearance| clearance > 0.0)
        .count()
        + flat_before
            .iter()
            .filter(|&&clearance| clearance > 0.0)
            .count();
    let mut result = Vec::with_capacity(centerline.len() + extra_vertices);
    result.push(centerline[0]);

    for index in 0..centerline.len() - 1 {
        let a = centerline[index];
        let b = centerline[index + 1];
        if flat_after[index] > 0.0 {
            result.push(point_horiz_distance_from(a, b, flat_after[index], a.z)?);
        }
        if flat_before[index + 1] > 0.0 {
            result.push(point_horiz_distance_from(
                b,
                a,
                flat_before[index + 1],
                b.z,
            )?);
        }
        result.push(b);
    }

    validate_road_turn_angles(&result)?;
    Ok(result)
}

fn road_flat_clearance_at_junction(
    junction: DVec3,
    branch_toward: DVec3,
    document: &Document,
    road_width: f64,
) -> f64 {
    let branch_delta = branch_toward.truncate() - junction.truncate();
    let branch_len = branch_delta.length();
    if branch_len < 1e-9 {
        return ROAD_INTERSECTION_FLAT_CLEARANCE_M;
    }

    let branch_dir = branch_delta / branch_len;
    let own_half_width = (road_width * 0.5).max(0.0);
    connected_road_branches_at_junction(document, junction)
        .into_iter()
        .filter_map(|(other_dir, other_width)| {
            let angle = branch_dir.dot(other_dir).clamp(-1.0, 1.0).acos();
            if angle < 1e-6 {
                return None;
            }

            let half_width = own_half_width.max(other_width * 0.5);
            let tangent = (angle * 0.5).tan();
            if tangent.abs() < 1e-9 {
                return None;
            }

            Some(ROAD_INTERSECTION_FLAT_CLEARANCE_M + half_width / tangent)
        })
        .fold(ROAD_INTERSECTION_FLAT_CLEARANCE_M, f64::max)
}

fn road_flat_clearance_at_point(
    point: DVec3,
    branch_toward: DVec3,
    document: &Document,
    road_width: f64,
) -> f64 {
    if road_point_is_junction(point, document) {
        road_flat_clearance_at_junction(point, branch_toward, document, road_width)
    } else {
        ROAD_INTERSECTION_FLAT_CLEARANCE_M
    }
}

fn connected_road_branches_at_junction(document: &Document, junction: DVec3) -> Vec<(DVec2, f64)> {
    let mut branches = Vec::new();
    for object in document.objects() {
        let Object::Road {
            centerline, width, ..
        } = object
        else {
            continue;
        };
        if centerline.len() < 2 {
            continue;
        }

        for (index, vertex) in centerline.iter().enumerate() {
            if !points_coincident(vertex.pos, junction) {
                continue;
            }
            if index > 0 {
                push_branch_dir(&mut branches, centerline[index - 1].pos, junction, *width);
            }
            if index + 1 < centerline.len() {
                push_branch_dir(&mut branches, centerline[index + 1].pos, junction, *width);
            }
        }

        for segment in centerline.windows(2) {
            let a = segment[0].pos;
            let b = segment[1].pos;
            if point_on_segment_3d(junction, a, b) {
                push_branch_dir(&mut branches, a, junction, *width);
                push_branch_dir(&mut branches, b, junction, *width);
            }
        }
    }
    branches
}

fn push_branch_dir(branches: &mut Vec<(DVec2, f64)>, toward: DVec3, junction: DVec3, width: f64) {
    let delta = toward.truncate() - junction.truncate();
    let len = delta.length();
    if len < 1e-9 {
        return;
    }
    let dir = delta / len;
    if branches
        .iter()
        .any(|(existing, _)| existing.dot(dir) > 1.0 - 1e-8)
    {
        return;
    }
    branches.push((dir, width));
}

pub(crate) fn validate_road_turn_angles(
    centerline: &[DVec3],
) -> Result<(), RoadCenterlineRuleError> {
    if centerline.len() < 3 {
        return Ok(());
    }

    for window in centerline.windows(3) {
        let prev = window[0].truncate();
        let joint = window[1].truncate();
        let next = window[2].truncate();
        let a = prev - joint;
        let b = next - joint;
        let a_len = a.length();
        let b_len = b.length();
        if a_len < 1e-9 || b_len < 1e-9 {
            return Err(RoadCenterlineRuleError::DegenerateSegment);
        }

        let cos = (a.dot(b) / (a_len * b_len)).clamp(-1.0, 1.0);
        let angle_degrees = cos.acos().to_degrees();
        if angle_degrees + 1e-6 < MIN_ROAD_TURN_ANGLE_DEGREES {
            return Err(RoadCenterlineRuleError::TurnTooSharp {
                minimum_degrees: MIN_ROAD_TURN_ANGLE_DEGREES,
                actual_degrees: angle_degrees,
            });
        }
    }

    Ok(())
}

fn validate_road_endpoint_attachment_turns(
    centerline: &[DVec3],
    document: &Document,
) -> Result<(), RoadCenterlineRuleError> {
    let Some((&start, &end)) = centerline.first().zip(centerline.last()) else {
        return Ok(());
    };
    if centerline.len() < 2 {
        return Ok(());
    }

    validate_road_endpoint_attachment_turn(document, start, centerline[1])?;
    if !points_coincident(start, end) {
        validate_road_endpoint_attachment_turn(document, end, centerline[centerline.len() - 2])?;
    }

    Ok(())
}

fn validate_road_endpoint_attachment_turn(
    document: &Document,
    junction: DVec3,
    new_branch: DVec3,
) -> Result<(), RoadCenterlineRuleError> {
    for object in document.objects() {
        let Object::Road { centerline, .. } = object else {
            continue;
        };
        if centerline.len() < 2 {
            continue;
        }

        if points_coincident(centerline[0].pos, junction) {
            validate_road_turn_angles(&[centerline[1].pos, junction, new_branch])?;
        }
        let last_index = centerline.len() - 1;
        if points_coincident(centerline[last_index].pos, junction) {
            validate_road_turn_angles(&[centerline[last_index - 1].pos, junction, new_branch])?;
        }

        for segment in centerline.windows(2) {
            let a = segment[0].pos;
            let b = segment[1].pos;
            if point_on_segment_3d(junction, a, b) {
                validate_road_turn_angles(&[a, junction, new_branch])?;
                validate_road_turn_angles(&[b, junction, new_branch])?;
            }
        }
    }

    Ok(())
}

pub(crate) fn validate_road_segment_angles(
    centerline: &[DVec3],
    max_degrees: f64,
) -> Result<(), RoadCenterlineRuleError> {
    let max_degrees = max_degrees.clamp(0.0, 89.9);
    for segment in centerline.windows(2) {
        let delta = segment[1] - segment[0];
        let horizontal = delta.truncate().length();
        let vertical = delta.z.abs();
        if horizontal < 1e-9 {
            if vertical < 1e-9 {
                continue;
            }
            return Err(RoadCenterlineRuleError::SegmentTooSteep {
                maximum_degrees: max_degrees,
                actual_degrees: 90.0,
            });
        }

        let angle_degrees = vertical.atan2(horizontal).to_degrees();
        if angle_degrees > max_degrees + 1e-6 {
            return Err(RoadCenterlineRuleError::SegmentTooSteep {
                maximum_degrees: max_degrees,
                actual_degrees: angle_degrees,
            });
        }
    }

    Ok(())
}

fn road_point_is_junction(point: DVec3, document: &Document) -> bool {
    document.objects().iter().any(|object| {
        let Object::Road { centerline, .. } = object else {
            return false;
        };
        if centerline.len() < 2 {
            return false;
        }

        if centerline
            .iter()
            .any(|vertex| points_coincident(vertex.pos, point))
        {
            return true;
        }

        centerline
            .windows(2)
            .any(|segment| point_on_segment_3d(point, segment[0].pos, segment[1].pos))
    })
}

fn point_on_segment_3d(point: DVec3, a: DVec3, b: DVec3) -> bool {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-10 {
        return points_coincident(point, a);
    }
    let t = (point - a).dot(ab) / len_sq;
    if !(1e-6..=1.0 - 1e-6).contains(&t) {
        return false;
    }
    points_coincident(point, a + ab * t)
}

fn point_horiz_distance_from(
    from: DVec3,
    toward: DVec3,
    distance: f64,
    z: f64,
) -> Result<DVec3, RoadCenterlineRuleError> {
    let delta = toward.truncate() - from.truncate();
    let len = delta.length();
    if len < 1e-9 {
        return Err(RoadCenterlineRuleError::DegenerateSegment);
    }
    if len + 1e-9 < distance {
        return Err(RoadCenterlineRuleError::SegmentTooShort {
            required: distance,
            actual: len,
        });
    }
    let xy = from.truncate() + delta / len * distance;
    Ok(DVec3::new(xy.x, xy.y, z))
}

fn segment_is_inclined(a: DVec3, b: DVec3) -> bool {
    (b.z - a.z).abs() > 1e-6
}

fn horizontal_distance(a: DVec3, b: DVec3) -> f64 {
    (b.truncate() - a.truncate()).length()
}

pub(crate) fn points_coincident(a: DVec3, b: DVec3) -> bool {
    (a - b).length_squared() < 1e-8
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

/// Ray-casting point-in-polygon test on the XY plane.
pub(crate) fn point_in_polygon_xy(point: DVec2, verts: &[DVec3]) -> bool {
    let n = verts.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (verts[i].x, verts[i].y);
        let (xj, yj) = (verts[j].x, verts[j].y);
        if ((yi > point.y) != (yj > point.y))
            && (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Check if segment AB strictly intersects segment CD (not at shared endpoints).
/// Returns the intersection point and its parameter along AB if found.
fn seg_seg_intersection_2d(a: DVec2, b: DVec2, c: DVec2, d: DVec2) -> Option<(DVec2, f64)> {
    let r = b - a;
    let s = d - c;
    let denom = r.x * s.y - r.y * s.x;
    if denom.abs() < 1e-10 {
        return None; // Parallel or collinear
    }
    let t = ((c.x - a.x) * s.y - (c.y - a.y) * s.x) / denom;
    let u = ((c.x - a.x) * r.y - (c.y - a.y) * r.x) / denom;
    const EPS: f64 = 1e-8;
    if t > EPS && t < 1.0 - EPS && u > EPS && u < 1.0 - EPS {
        Some((a + t * r, t))
    } else {
        None
    }
}

/// Remove self-intersections from a closed polygon offset result (XY plane).
///
/// When an inward offset is too large for a sharp corner the adjacent edges
/// fold back and cross, creating a small loop. This function splits the polygon
/// at each crossing, retains the loop with the greater XY area, and repeats
/// until no crossings remain.
pub(crate) fn remove_self_intersections(verts: Vec<DVec3>) -> Vec<DVec3> {
    if verts.len() < 4 {
        return verts;
    }
    let mut pts = verts;

    let max_passes = pts.len() * pts.len();
    for _ in 0..max_passes {
        let n = pts.len();
        if n < 3 {
            break;
        }
        let mut found = false;
        'search: for i in 0..n {
            let a = pts[i].truncate();
            let b = pts[(i + 1) % n].truncate();
            // Only check edges that are non-adjacent to edge i→(i+1).
            let j_end = if i == 0 { n - 1 } else { n };
            for j in (i + 2)..j_end {
                let c = pts[j].truncate();
                let d = pts[(j + 1) % n].truncate();
                if let Some((intersection_xy, t)) = seg_seg_intersection_2d(a, b, c, d) {
                    let intersection_z = pts[i].z + t * (pts[(i + 1) % n].z - pts[i].z);
                    let intersection =
                        DVec3::new(intersection_xy.x, intersection_xy.y, intersection_z);

                    let mut first = Vec::with_capacity(j - i + 1);
                    first.push(intersection);
                    first.extend_from_slice(&pts[i + 1..=j]);

                    let mut second = Vec::with_capacity(n - (j - i) + 1);
                    second.push(intersection);
                    second.extend_from_slice(&pts[j + 1..]);
                    second.extend_from_slice(&pts[..=i]);

                    let first_area = signed_area_xy(&first).abs();
                    let second_area = signed_area_xy(&second).abs();
                    pts = if first_area > second_area {
                        first
                    } else {
                        second
                    };
                    found = true;
                    break 'search;
                }
            }
        }
        if !found {
            break;
        }
    }

    pts
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
