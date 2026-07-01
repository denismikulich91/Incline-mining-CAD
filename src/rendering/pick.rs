//! Geometry picking against the world-space vertex buffers the renderer
//! already produces.
//!
//! Each top-level entity records the range of stroke/fill vertices it emitted
//! (`PickRecord`). To pick, we project those world-space vertices to the screen
//! and find the geometry nearest the cursor, returning its true world position
//! (including Z) and owning entity handle.

use std::collections::HashSet;

use glam::{DMat4, DVec2, DVec3};

use crate::{
    Size,
    model::{Document, Object, ObjectId, SceneEntityId},
    rendering::{StrokeVertex, Vertex},
};

#[derive(Clone, Debug)]
pub(crate) struct PickRecord {
    pub(crate) entity: SceneEntityId,
    /// Half-open range into the stroke vertex buffer.
    pub(crate) stroke_range: (u32, u32),
    /// Half-open range into the stroke index buffer, used for picking.
    pub(crate) stroke_index_range: (u32, u32),
    /// Half-open range into the fill vertex buffer.
    pub(crate) fill_range: (u32, u32),
    /// Half-open range into the fill index buffer.
    pub(crate) fill_index_range: (u32, u32),
    /// Original polyline segment endpoints (3-D world space, before tessellation).
    /// Used for cross-select edge-crossing tests. Empty for points.
    pub(crate) segments: Vec<[DVec3; 2]>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PickHit {
    pub(crate) entity: SceneEntityId,
    pub(crate) world: DVec3,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TextPickRecord {
    pub(crate) entity: SceneEntityId,
    pub(crate) corners: [DVec3; 4],
}

/// Project a world point to physical screen pixels. Returns `None` when the
/// point is on/behind the camera plane (`w <= 0`).
pub(crate) fn world_to_screen(view_proj: &DMat4, world: DVec3, screen: Size) -> Option<DVec2> {
    let clip = *view_proj * world.extend(1.0);
    if clip.w.abs() <= f64::EPSILON {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if !(0.0..=1.0).contains(&ndc.z) {
        return None;
    }
    Some(DVec2::new(
        (ndc.x * 0.5 + 0.5) * screen.0 as f64,
        (0.5 - ndc.y * 0.5) * screen.1 as f64,
    ))
}

/// Parameter `t in [0, 1]` of the closest point on segment `a-b` to `p`.
pub(crate) fn closest_t_on_segment(p: DVec2, a: DVec2, b: DVec2) -> f64 {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 <= f64::EPSILON {
        return 0.0;
    }
    ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
}

/// Find the entity geometry nearest the cursor within `threshold_px`.
///
/// Stroke vertices arrive in quads (`start, start, end, end`) per segment, so
/// we step in 4s and measure point-to-segment distance in screen space, which
/// handles long straight lines correctly. Fill vertices are tested as points.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pick_nearest(
    records: &[PickRecord],
    stroke_verts: &[StrokeVertex],
    stroke_indices: &[u32],
    fill_verts: &[Vertex],
    fill_indices: &[u32],
    scene_origin: DVec3,
    view_proj: &DMat4,
    screen: Size,
    cursor_px: (f32, f32),
    threshold_px: f32,
    frozen: &HashSet<SceneEntityId>,
) -> Option<PickHit> {
    let cursor = DVec2::new(cursor_px.0 as f64, cursor_px.1 as f64);
    let mut best_dist = threshold_px as f64;
    let mut best_hit: Option<PickHit> = None;
    let mut best_fill_depth = f64::INFINITY;

    for rec in records {
        // Frozen entities are visible but not selectable.
        if frozen.contains(&rec.entity) {
            continue;
        }
        let (s0, s1) = (
            rec.stroke_index_range.0 as usize,
            (rec.stroke_index_range.1 as usize).min(stroke_indices.len()),
        );
        for triangle in stroke_indices[s0..s1].chunks_exact(3) {
            for (a, b) in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ] {
                let (Some(a), Some(b)) =
                    (stroke_verts.get(a as usize), stroke_verts.get(b as usize))
                else {
                    continue;
                };
                let wa = DVec3::from_array(a.pos.map(f64::from)) + scene_origin;
                let wb = DVec3::from_array(b.pos.map(f64::from)) + scene_origin;
                if let (Some(sa), Some(sb)) = (
                    world_to_screen(view_proj, wa, screen),
                    world_to_screen(view_proj, wb, screen),
                ) {
                    let t = closest_t_on_segment(cursor, sa, sb);
                    let dist = (sa + (sb - sa) * t).distance(cursor);
                    if dist < best_dist {
                        best_dist = dist;
                        best_hit = Some(PickHit {
                            entity: rec.entity,
                            world: wa + (wb - wa) * t,
                        });
                    }
                }
            }
        }

        let (f0, f1) = (rec.fill_range.0 as usize, rec.fill_range.1 as usize);
        for vert in &fill_verts[f0..f1.min(fill_verts.len())] {
            let world = DVec3::from_array(vert.pos.map(f64::from)) + scene_origin;
            if let Some(sp) = world_to_screen(view_proj, world, screen) {
                let dist = sp.distance(cursor);
                if dist < best_dist {
                    best_dist = dist;
                    best_hit = Some(PickHit {
                        entity: rec.entity,
                        world,
                    });
                }
            }
        }

        let (i0, i1) = (
            rec.fill_index_range.0 as usize,
            (rec.fill_index_range.1 as usize).min(fill_indices.len()),
        );
        for triangle in fill_indices[i0..i1].chunks_exact(3) {
            let [Some(a), Some(b), Some(c)] = [
                fill_verts.get(triangle[0] as usize),
                fill_verts.get(triangle[1] as usize),
                fill_verts.get(triangle[2] as usize),
            ] else {
                continue;
            };
            let wa = DVec3::from_array(a.pos.map(f64::from)) + scene_origin;
            let wb = DVec3::from_array(b.pos.map(f64::from)) + scene_origin;
            let wc = DVec3::from_array(c.pos.map(f64::from)) + scene_origin;
            let (Some(sa), Some(sb), Some(sc)) = (
                world_to_screen(view_proj, wa, screen),
                world_to_screen(view_proj, wb, screen),
                world_to_screen(view_proj, wc, screen),
            ) else {
                continue;
            };
            if let Some(weights) = triangle_weights(cursor, sa, sb, sc) {
                let world = wa * weights.x + wb * weights.y + wc * weights.z;
                let clip = *view_proj * world.extend(1.0);
                let depth = if clip.w.abs() > f64::EPSILON {
                    clip.z / clip.w
                } else {
                    f64::INFINITY
                };
                if depth < best_fill_depth {
                    best_fill_depth = depth;
                    best_dist = 0.0;
                    best_hit = Some(PickHit {
                        entity: rec.entity,
                        world,
                    });
                }
            }
        }
    }

    best_hit
}

pub(crate) fn pick_text(
    records: &[TextPickRecord],
    view_proj: &DMat4,
    screen: Size,
    cursor_px: (f32, f32),
    frozen: &HashSet<SceneEntityId>,
) -> Option<PickHit> {
    let cursor = DVec2::new(f64::from(cursor_px.0), f64::from(cursor_px.1));
    let mut best: Option<(f64, PickHit)> = None;
    for record in records {
        if frozen.contains(&record.entity) {
            continue;
        }
        let [Some(a), Some(b), Some(c), Some(d)] = record
            .corners
            .map(|corner| world_to_screen(view_proj, corner, screen))
        else {
            continue;
        };
        let world = if let Some(weights) = triangle_weights(cursor, a, b, c) {
            record.corners[0] * weights.x
                + record.corners[1] * weights.y
                + record.corners[2] * weights.z
        } else if let Some(weights) = triangle_weights(cursor, a, c, d) {
            record.corners[0] * weights.x
                + record.corners[2] * weights.y
                + record.corners[3] * weights.z
        } else {
            continue;
        };
        let center_distance = ((a + b + c + d) * 0.25).distance_squared(cursor);
        if best
            .as_ref()
            .is_none_or(|(distance, _)| center_distance < *distance)
        {
            best = Some((
                center_distance,
                PickHit {
                    entity: record.entity,
                    world,
                },
            ));
        }
    }
    best.map(|(_, hit)| hit)
}

/// Find the polyline vertex nearest the cursor. Returns `(object_id, vertex_index, world_pos)`.
/// Only considers unfrozen, visible objects whose layer is visible in `doc`.
pub(crate) fn pick_nearest_vertex(
    doc: &Document,
    hidden: &HashSet<SceneEntityId>,
    frozen: &HashSet<SceneEntityId>,
    view_proj: &DMat4,
    screen: Size,
    cursor_px: (f32, f32),
    threshold_px: f32,
) -> Option<(ObjectId, usize, DVec3)> {
    let cursor = DVec2::new(f64::from(cursor_px.0), f64::from(cursor_px.1));
    let mut best_dist = threshold_px as f64;
    let mut best: Option<(ObjectId, usize, DVec3)> = None;

    for object in doc.objects() {
        let entity = SceneEntityId::Object(object.id());
        if hidden.contains(&entity) || frozen.contains(&entity) {
            continue;
        }
        if !doc.layer(object.layer()).map(|l| l.visible).unwrap_or(true) {
            continue;
        }
        match object {
            Object::Polyline { verts, .. }
            | Object::Road {
                centerline: verts, ..
            } => {
                for (i, vert) in verts.iter().enumerate() {
                    if let Some(sp) = world_to_screen(view_proj, vert.pos, screen) {
                        let d = sp.distance(cursor);
                        if d < best_dist {
                            best_dist = d;
                            best = Some((object.id(), i, vert.pos));
                        }
                    }
                }
            }
            Object::Point { pos, .. } => {
                if let Some(sp) = world_to_screen(view_proj, *pos, screen) {
                    let d = sp.distance(cursor);
                    if d < best_dist {
                        best_dist = d;
                        best = Some((object.id(), 0, *pos));
                    }
                }
            }
            _ => {}
        }
    }

    best
}

pub(crate) fn triangle_weights(point: DVec2, a: DVec2, b: DVec2, c: DVec2) -> Option<DVec3> {
    let denominator = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    if denominator.abs() <= f64::EPSILON {
        return None;
    }
    let u = ((b.y - c.y) * (point.x - c.x) + (c.x - b.x) * (point.y - c.y)) / denominator;
    let v = ((c.y - a.y) * (point.x - c.x) + (a.x - c.x) * (point.y - c.y)) / denominator;
    let w = 1.0 - u - v;
    (u >= 0.0 && v >= 0.0 && w >= 0.0).then_some(DVec3::new(u, v, w))
}
