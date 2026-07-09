//! Scene bounds helpers.

use glam::DVec3;

use crate::model::{
    Document, Object, SceneEntityId, block_model::OpenBlockModel, point_cloud::OpenPointCloud,
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
    for object in document.objects() {
        if hidden_layers.contains(&object.layer())
            || hidden.contains(&SceneEntityId::Object(object.id()))
        {
            continue;
        }

        match object {
            Object::Point { pos, .. } | Object::Text { pos, .. } => {
                min = min.min(*pos);
                max = max.max(*pos);
                any = true;
            }
            Object::Polyline { verts, .. }
            | Object::Road {
                centerline: verts, ..
            } => {
                for vertex in verts {
                    min = min.min(vertex.pos);
                    max = max.max(vertex.pos);
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
