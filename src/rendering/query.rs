//! Unified scene interaction queries, independent of GPU buffer ownership.

use std::collections::HashSet;

use glam::{DMat4, DVec3};

use crate::{
    Size,
    model::{Document, SceneEntityId, spatial::ObjectSnapIndex, triangulation::OpenTriangulation},
    rendering::snap,
    ui::state::CursorMode,
};

pub(crate) struct SceneQuery;

impl SceneQuery {
    pub(crate) fn nearest_surface(
        triangulations: &[OpenTriangulation],
        hidden: &HashSet<SceneEntityId>,
        frozen: Option<&HashSet<SceneEntityId>>,
        ray_origin: DVec3,
        ray_direction: DVec3,
    ) -> Option<(SceneEntityId, DVec3)> {
        triangulations
            .iter()
            .filter(|triangulation| {
                let entity = triangulation.entity_id();
                triangulation.visible
                    && !hidden.contains(&entity)
                    && frozen.is_none_or(|set| !set.contains(&entity))
            })
            .filter_map(|triangulation| {
                triangulation
                    .spatial
                    .ray_hit(&triangulation.mesh, ray_origin, ray_direction)
                    .map(|point| (triangulation.entity_id(), point))
            })
            .min_by(|(_, a), (_, b)| {
                (*a - ray_origin)
                    .dot(ray_direction)
                    .total_cmp(&(*b - ray_origin).dot(ray_direction))
            })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn snap(
        document: &Document,
        snap_index: &ObjectSnapIndex,
        road_network: Option<&crate::model::road_network::ResolvedNetwork>,
        triangulations: &[OpenTriangulation],
        hidden: &HashSet<SceneEntityId>,
        frozen: &HashSet<SceneEntityId>,
        mode: &CursorMode,
        view_projection: &DMat4,
        screen: Size,
        cursor: (f32, f32),
        threshold: f32,
        xray_enabled: bool,
    ) -> Option<DVec3> {
        let candidate = snap::snap_cursor(
            document,
            snap_index,
            road_network,
            triangulations,
            hidden,
            frozen,
            mode,
            view_projection,
            screen,
            cursor,
            threshold,
        )?;
        if xray_enabled {
            return Some(candidate.world);
        }
        // Surface snapping is already a nearest-hit ray cast along the cursor
        // ray, so the candidate is the front-most surface by construction —
        // re-ray-casting every triangulation here would only repeat the work.
        if matches!(mode, CursorMode::SnapToSurface) {
            return Some(candidate.world);
        }
        // Test visibility along the candidate's own screen-space ray. Using the
        // cursor ray here skews the accepted snap region on sloped surfaces:
        // the candidate may be several pixels from the cursor, so the cursor ray
        // can hit the same surface in front of an otherwise visible vertex.
        let (ray_origin, ray_direction) =
            ray_through_world_point(view_projection, candidate.world)?;
        let occluder =
            Self::nearest_surface(triangulations, hidden, None, ray_origin, ray_direction);
        let candidate_depth = (candidate.world - ray_origin).dot(ray_direction);
        // A triangulation must occlude its own back-side vertices too. The small
        // relative tolerance only absorbs ray/triangle floating-point noise at
        // the visible surface; it does not open a path through the mesh.
        if occluder.is_some_and(|(_, point)| {
            let surface_depth = (point - ray_origin).dot(ray_direction);
            let tolerance = 1.0e-5_f64.max(surface_depth.abs() * 1.0e-9);
            candidate_depth > surface_depth + tolerance
        }) {
            None
        } else {
            Some(candidate.world)
        }
    }
}

fn ray_through_world_point(view_projection: &DMat4, point: DVec3) -> Option<(DVec3, DVec3)> {
    let clip = *view_projection * point.extend(1.0);
    if clip.w.abs() <= f64::EPSILON {
        return None;
    }

    let ndc = clip.truncate() / clip.w;
    let inverse = view_projection.inverse();
    let near_h = inverse * DVec3::new(ndc.x, ndc.y, 0.0).extend(1.0);
    let far_h = inverse * DVec3::new(ndc.x, ndc.y, 1.0).extend(1.0);
    if near_h.w.abs() <= f64::EPSILON || far_h.w.abs() <= f64::EPSILON {
        return None;
    }

    let near = near_h.truncate() / near_h.w;
    let far = far_h.truncate() / far_h.w;
    let direction = (far - near).try_normalize()?;
    Some((near, direction))
}
