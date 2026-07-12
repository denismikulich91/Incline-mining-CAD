//! Scene bounds helpers.

use glam::DVec3;

use crate::model::{
    Document, Object, SceneEntityId,
    block_model::OpenBlockModel,
    geometry::{polyline_bulge_bounds, text_bounds_corners},
    point_cloud::OpenPointCloud,
    road_network::{RoadKey, resolve},
    triangulation::OpenTriangulation,
};

pub(crate) fn scene_bounds(
    document: &Document,
    triangulations: &[OpenTriangulation],
    block_models: &[OpenBlockModel],
    point_clouds: &[OpenPointCloud],
    hidden: &std::collections::HashSet<SceneEntityId>,
) -> Option<(DVec3, DVec3)> {
    let mut min = DVec3::splat(f64::MAX);
    let mut max = DVec3::splat(f64::MIN);
    let mut any = false;

    let hidden_layers: std::collections::HashSet<_> = document
        .layers()
        .iter()
        .filter(|layer| !layer.visible)
        .map(|layer| layer.id)
        .collect();
    let road_network = resolve(document, None);
    for object in document.objects() {
        if hidden_layers.contains(&object.layer())
            || hidden.contains(&SceneEntityId::Object(object.id()))
        {
            continue;
        }

        match object {
            Object::Point { pos, .. } => {
                min = min.min(*pos);
                max = max.max(*pos);
                any = true;
            }
            Object::Text {
                pos,
                content,
                height,
                rotation,
                ..
            } => {
                for corner in text_bounds_corners(*pos, content, *height, *rotation) {
                    min = min.min(corner);
                    max = max.max(corner);
                    any = true;
                }
            }
            Object::Polyline { verts, closed, .. } => {
                if let Some((object_min, object_max)) = polyline_bulge_bounds(verts, *closed) {
                    min = min.min(object_min);
                    max = max.max(object_max);
                    any = true;
                }
            }
            Object::Road { id, centerline, .. } => {
                let mut found_resolved = false;
                for edge in road_network.edges_for(RoadKey::Object(*id)) {
                    for point in edge.center.iter().chain(&edge.left).chain(&edge.right) {
                        min = min.min(*point);
                        max = max.max(*point);
                        any = true;
                        found_resolved = true;
                    }
                }
                if !found_resolved
                    && let Some((object_min, object_max)) = polyline_bulge_bounds(centerline, false)
                {
                    min = min.min(object_min);
                    max = max.max(object_max);
                    any = true;
                }
            }
        }
    }

    for triangulation in triangulations.iter().filter(|triangulation| {
        triangulation.visible && !hidden.contains(&triangulation.entity_id())
    }) {
        let bounds = triangulation.mesh.bounds();
        min = min.min(DVec3::new(bounds.min.x, bounds.min.y, bounds.min.z));
        max = max.max(DVec3::new(bounds.max.x, bounds.max.y, bounds.max.z));
        any = true;
    }

    for block_model in block_models
        .iter()
        .filter(|block_model| block_model.visible && !hidden.contains(&block_model.entity_id()))
    {
        if let Some((block_min, block_max)) = block_model.world_bounds() {
            min = min.min(block_min);
            max = max.max(block_max);
            any = true;
        }
    }

    for point_cloud in point_clouds
        .iter()
        .filter(|point_cloud| point_cloud.visible)
    {
        let (cloud_min, cloud_max) = point_cloud.bounds;
        min = min.min(cloud_min);
        max = max.max(cloud_max);
        any = true;
    }

    any.then_some((min, max))
}
