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
        // Test visibility along the candidate's own screen-space ray. Using the
        // cursor ray here skews the accepted snap region on sloped surfaces:
        // the candidate may be several pixels from the cursor, so the cursor ray
        // can hit the same surface in front of an otherwise visible vertex.
        let (ray_origin, ray_direction) =
            ray_through_world_point(view_projection, candidate.world)?;
        let candidate_depth = (candidate.world - ray_origin).dot(ray_direction);
        // Surface snapping already found the nearest triangulation along this
        // ray. Other snap modes still need the triangulation visibility test.
        let surface_depth = (!matches!(mode, CursorMode::SnapToSurface))
            .then(|| Self::nearest_surface(triangulations, hidden, None, ray_origin, ray_direction))
            .flatten()
            .map(|(_, point)| (point - ray_origin).dot(ray_direction));
        let document_fill_depth =
            nearest_opaque_document_fill(document, snap_index, hidden, ray_origin, ray_direction)
                .map(|point| (point - ray_origin).dot(ray_direction));
        let occluder_depth = surface_depth
            .into_iter()
            .chain(document_fill_depth)
            .min_by(f64::total_cmp);
        // A triangulation must occlude its own back-side vertices too. The small
        // relative tolerance only absorbs ray/triangle floating-point noise at
        // the visible surface; it does not open a path through the mesh.
        if occluder_depth.is_some_and(|depth| {
            let tolerance = 1.0e-5_f64.max(depth.abs() * 1.0e-9);
            candidate_depth > depth + tolerance
        }) {
            None
        } else {
            Some(candidate.world)
        }
    }
}

fn nearest_opaque_document_fill(
    document: &Document,
    snap_index: &ObjectSnapIndex,
    hidden: &HashSet<SceneEntityId>,
    ray_origin: DVec3,
    ray_direction: DVec3,
) -> Option<DVec3> {
    snap_index.nearest_filled_polygon_hit(ray_origin, ray_direction, |object_index| {
        let Some(object) = document.objects().get(object_index) else {
            return false;
        };
        let entity = SceneEntityId::Object(object.id());
        !hidden.contains(&entity)
            && document
                .layer(object.layer())
                .is_none_or(|layer| layer.visible)
            && document.object_fill_rgba(object)[3] >= 1.0 - f32::EPSILON
    })
}

pub(crate) fn ray_through_world_point(
    view_projection: &DMat4,
    point: DVec3,
) -> Option<(DVec3, DVec3)> {
    let clip = *view_projection * point.extend(1.0);
    if clip.w.abs() <= f64::EPSILON {
        return None;
    }

    let ndc = clip.truncate() / clip.w;
    let inverse = view_projection.inverse();
    let near_h = inverse * DVec3::new(ndc.x, ndc.y, 1.0).extend(1.0);
    let far_h = inverse * DVec3::new(ndc.x, ndc.y, 0.0).extend(1.0);
    if near_h.w.abs() <= f64::EPSILON || far_h.w.abs() <= f64::EPSILON {
        return None;
    }

    let near = near_h.truncate() / near_h.w;
    let far = far_h.truncate() / far_h.w;
    let direction = (far - near).try_normalize()?;
    Some((near, direction))
}
