use std::path::PathBuf;

use crate::model::formats::tri00t;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TriangulationId(pub(crate) u64);

/// The computed outputs of loading a triangulation file, produced on a background
/// thread and sent back to the main thread via channel.
pub(crate) struct LoadedTriangulation {
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) mesh: tri00t::Triangulation,
    pub(crate) spatial: crate::model::spatial::TriangleBvh,
    pub(crate) edges: Vec<[u32; 2]>,
    pub(crate) scene_was_empty: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct OpenTriangulation {
    pub(crate) id: TriangulationId,
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    /// Whether `path` points to a mesh that has been written to disk.
    pub(crate) is_saved: bool,
    pub(crate) mesh: tri00t::Triangulation,
    pub(crate) spatial: crate::model::spatial::TriangleBvh,
    pub(crate) edges: Vec<[u32; 2]>,
    pub(crate) visible: bool,
    pub(crate) color: [f32; 4],
    pub(crate) line_color: [f32; 4],
    pub(crate) line_weight: Option<f32>,
}

impl OpenTriangulation {
    pub(crate) fn entity_id(&self) -> crate::model::SceneEntityId {
        crate::model::SceneEntityId::Triangulation(self.id)
    }
}

pub(crate) fn unique_edges(mesh: &tri00t::Triangulation) -> Vec<[u32; 2]> {
    use rayon::prelude::*;

    let mut packed: Vec<u64> = mesh
        .face_vertex_indices_iter()
        .flat_map(|face| {
            [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])].map(|(a, b)| {
                let (a, b) = if a <= b {
                    (a as u32, b as u32)
                } else {
                    (b as u32, a as u32)
                };
                (u64::from(a) << 32) | u64::from(b)
            })
        })
        .collect();

    packed.par_sort_unstable();
    packed.dedup();

    packed
        .into_iter()
        .map(|e| [(e >> 32) as u32, e as u32])
        .collect()
}
