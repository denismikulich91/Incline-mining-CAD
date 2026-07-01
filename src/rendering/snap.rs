use std::collections::HashSet;

use glam::{DMat4, DVec2, DVec3};

use crate::{
    Size,
    model::{
        Document, Object, SceneEntityId, spatial::ObjectSnapIndex, triangulation::OpenTriangulation,
    },
    rendering::pick::{closest_t_on_segment, triangle_weights, world_to_screen},
    ui::state::CursorMode,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct SnapHit {
    pub(crate) world: DVec3,
}

pub(crate) const SNAP_THRESHOLD_PX: f32 = 15.0;

/// Find the nearest snap target to `cursor_px` given `mode`.
///
/// Returns `Some(world_pos)` if any target is within `threshold_px`,
/// otherwise `None` (caller should fall back to the raw cursor ray).
#[allow(clippy::too_many_arguments)]
pub(crate) fn snap_cursor(
    document: &Document,
    snap_index: &ObjectSnapIndex,
    triangulations: &[OpenTriangulation],
    hidden: &HashSet<SceneEntityId>,
    frozen: &HashSet<SceneEntityId>,
    mode: &CursorMode,
    view_proj: &DMat4,
    screen: Size,
    cursor_px: (f32, f32),
    threshold_px: f32,
) -> Option<SnapHit> {
    let cursor = DVec2::new(cursor_px.0 as f64, cursor_px.1 as f64);
    let threshold = threshold_px as f64;
    let mut best_dist_sq = threshold * threshold;
    let mut best: Option<SnapHit> = None;
    let mut best_surface_depth = f64::INFINITY;

    // BVH narrows candidates to objects whose projected AABB overlaps the cursor region.
    let candidates = snap_index.candidates(view_proj, screen, cursor, threshold);

    for obj_idx in candidates {
        let object = &document.objects()[obj_idx];
        let entity = SceneEntityId::Object(object.id());
        if frozen.contains(&entity) || hidden.contains(&entity) {
            continue;
        }
        if !document
            .layer(object.layer())
            .map(|l| l.visible)
            .unwrap_or(true)
        {
            continue;
        }

        match mode {
            CursorMode::SnapToSurface => {}
            CursorMode::SnapToPoint => match object {
                Object::Point { pos, .. } => {
                    if let Some(sp) = world_to_screen(view_proj, *pos, screen) {
                        let d = sp.distance_squared(cursor);
                        if d < best_dist_sq {
                            best_dist_sq = d;
                            best = Some(SnapHit { world: *pos });
                        }
                    }
                }
                Object::Polyline { verts, .. }
                | Object::Road {
                    centerline: verts, ..
                } => {
                    for v in verts.iter() {
                        if let Some(sp) = world_to_screen(view_proj, v.pos, screen) {
                            let d = sp.distance_squared(cursor);
                            if d < best_dist_sq {
                                best_dist_sq = d;
                                best = Some(SnapHit { world: v.pos });
                            }
                        }
                    }
                }
                _ => {}
            },

            CursorMode::SnapToLine => {
                let (verts, closed) = match object {
                    Object::Polyline { verts, closed, .. } => (verts.as_slice(), *closed),
                    Object::Road { centerline, .. } => (centerline.as_slice(), false),
                    _ => continue,
                };
                let n = verts.len();
                if n < 2 {
                    continue;
                }
                let seg_count = if closed { n } else { n - 1 };
                for i in 0..seg_count {
                    let a = verts[i].pos;
                    let b = verts[(i + 1) % n].pos;
                    let bulge = verts[i].bulge;
                    if bulge.abs() <= f64::EPSILON {
                        let (Some(sa), Some(sb)) = (
                            world_to_screen(view_proj, a, screen),
                            world_to_screen(view_proj, b, screen),
                        ) else {
                            continue;
                        };
                        let t = closest_t_on_segment(cursor, sa, sb);
                        let d = (sa + (sb - sa) * t).distance_squared(cursor);
                        if d < best_dist_sq {
                            best_dist_sq = d;
                            best = Some(SnapHit {
                                world: a + (b - a) * t,
                            });
                        }
                    } else {
                        let points = segment_points(a, b, bulge);
                        for pair in points.windows(2) {
                            let (Some(sa), Some(sb)) = (
                                world_to_screen(view_proj, pair[0], screen),
                                world_to_screen(view_proj, pair[1], screen),
                            ) else {
                                continue;
                            };
                            let t = closest_t_on_segment(cursor, sa, sb);
                            let d = (sa + (sb - sa) * t).distance_squared(cursor);
                            if d < best_dist_sq {
                                best_dist_sq = d;
                                best = Some(SnapHit {
                                    world: pair[0] + (pair[1] - pair[0]) * t,
                                });
                            }
                        }
                    }
                }
            }

            CursorMode::Select => {}
        }
    }

    for tri in triangulations {
        let entity = tri.entity_id();
        if !tri.visible || hidden.contains(&entity) || frozen.contains(&entity) {
            continue;
        }
        let tri_candidates =
            tri.spatial
                .screen_candidates(&tri.mesh, view_proj, screen, cursor_px, threshold_px);
        match mode {
            CursorMode::SnapToSurface => {
                for triangle in &tri_candidates {
                    let [a, b, c] = *triangle;
                    let (Some(sa), Some(sb), Some(sc)) = (
                        world_to_screen(view_proj, a, screen),
                        world_to_screen(view_proj, b, screen),
                        world_to_screen(view_proj, c, screen),
                    ) else {
                        continue;
                    };
                    if let Some(weights) = triangle_weights(cursor, sa, sb, sc) {
                        let world = a * weights.x + b * weights.y + c * weights.z;
                        let clip = *view_proj * world.extend(1.0);
                        let depth = if clip.w.abs() > f64::EPSILON {
                            clip.z / clip.w
                        } else {
                            f64::INFINITY
                        };
                        if depth < best_surface_depth {
                            best_surface_depth = depth;
                            best = Some(SnapHit { world });
                        }
                    }
                }
            }
            CursorMode::SnapToPoint => {
                for triangle in &tri_candidates {
                    for &pos in triangle {
                        if let Some(sp) = world_to_screen(view_proj, pos, screen) {
                            let d = sp.distance_squared(cursor);
                            if d < best_dist_sq {
                                best_dist_sq = d;
                                best = Some(SnapHit { world: pos });
                            }
                        }
                    }
                }
            }
            CursorMode::SnapToLine => {
                for triangle in &tri_candidates {
                    for (a, b) in [
                        (triangle[0], triangle[1]),
                        (triangle[1], triangle[2]),
                        (triangle[2], triangle[0]),
                    ] {
                        let (Some(sa), Some(sb)) = (
                            world_to_screen(view_proj, a, screen),
                            world_to_screen(view_proj, b, screen),
                        ) else {
                            continue;
                        };
                        let ab = sb - sa;
                        let t = closest_t_on_segment(cursor, sa, sb);
                        let d = (sa + ab * t).distance_squared(cursor);
                        if d < best_dist_sq {
                            best_dist_sq = d;
                            best = Some(SnapHit {
                                world: a + (b - a) * t,
                            });
                        }
                    }
                }
            }
            CursorMode::Select => {}
        }
    }

    best
}

fn segment_points(start: DVec3, end: DVec3, bulge: f64) -> Vec<DVec3> {
    let chord = end - start;
    let length = chord.length();
    if length <= f64::EPSILON {
        return vec![start];
    }
    let theta = 4.0 * bulge.atan();
    let midpoint = (start + end) * 0.5;
    let perpendicular = DVec3::Z.cross(chord / length).normalize_or_zero();
    let center = midpoint + perpendicular * length * (1.0 - bulge * bulge) / (4.0 * bulge);
    let radius = start - center;
    let count = ((theta.abs() * 16.0).ceil() as usize).clamp(8, 128);
    (0..=count)
        .map(|index| {
            if index == count {
                end
            } else {
                center
                    + glam::DQuat::from_rotation_z(theta * index as f64 / count as f64)
                        .mul_vec3(radius)
            }
        })
        .collect()
}
