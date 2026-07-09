use std::path::PathBuf;
use std::sync::Arc;

use glam::DVec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PointCloudId(pub(crate) u64);

/// The decoded contents of a point cloud file, produced on a background
/// thread and sent back to the main thread via channel.
pub(crate) struct LoadedPointCloud {
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) points: Arc<Vec<DVec3>>,
    /// Per-point packed RGBA8 colours (r in the low byte), parallel to
    /// `points`. `None` when the source file carries no colour data.
    pub(crate) colors: Option<Arc<Vec<u32>>>,
    pub(crate) bounds: (DVec3, DVec3),
    pub(crate) scene_was_empty: bool,
}

#[derive(Clone)]
pub(crate) struct OpenPointCloud {
    pub(crate) id: PointCloudId,
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) points: Arc<Vec<DVec3>>,
    pub(crate) colors: Option<Arc<Vec<u32>>>,
    pub(crate) bounds: (DVec3, DVec3),
    pub(crate) visible: bool,
    /// Uniform colour used when the file carries no per-point colours.
    pub(crate) color: [f32; 4],
    /// On-screen splat size in logical pixels.
    pub(crate) point_size: f32,
}
