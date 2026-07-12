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
    /// Spatially ordered, cloud-local render data prepared by the loader.
    pub(crate) prepared: Arc<PreparedPointCloud>,
    pub(crate) bounds: (DVec3, DVec3),
}

#[derive(Clone)]
pub(crate) struct OpenPointCloud {
    pub(crate) id: PointCloudId,
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) points: Arc<Vec<DVec3>>,
    pub(crate) prepared: Arc<PreparedPointCloud>,
    pub(crate) bounds: (DVec3, DVec3),
    pub(crate) visible: bool,
    /// Uniform colour used when the file carries no per-point colours.
    pub(crate) color: [f32; 4],
    /// On-screen splat size in logical pixels.
    pub(crate) point_size: f32,
}

/// Position-only instance used by clouds without per-point colours.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PointPosition {
    pub(crate) pos: [f32; 3],
}

/// Position and packed RGBA8 used by coloured clouds.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PointInstance {
    pub(crate) pos: [f32; 3],
    pub(crate) color: u32,
}

pub(crate) enum PreparedPointData {
    Uncolored(Vec<PointPosition>),
    Colored(Vec<PointInstance>),
}

impl PreparedPointData {
    pub(crate) fn bytes(&self) -> &[u8] {
        match self {
            Self::Uncolored(points) => bytemuck::cast_slice(points),
            Self::Colored(points) => bytemuck::cast_slice(points),
        }
    }

    pub(crate) fn position(&self, index: usize) -> Option<[f32; 3]> {
        match self {
            Self::Uncolored(points) => points.get(index).map(|point| point.pos),
            Self::Colored(points) => points.get(index).map(|point| point.pos),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct PointPickGroup {
    pub(crate) start: u32,
    pub(crate) end: u32,
    pub(crate) bounds_min: glam::Vec3,
    pub(crate) bounds_max: glam::Vec3,
}

pub(crate) struct PreparedPointChunk {
    /// Points are ordered so the 1/4 and 1/2 representative sets are
    /// prefixes. All LODs therefore share one CPU and GPU buffer.
    pub(crate) data: PreparedPointData,
    /// Full, 1/2 and 1/4 resolution instance counts.
    pub(crate) level_counts: [u32; 3],
    pub(crate) bounds_min: glam::Vec3,
    pub(crate) bounds_max: glam::Vec3,
    /// Small Morton-coherent ranges used for CPU visibility queries without
    /// scanning an entire 256k-point render chunk on every snap poll.
    pub(crate) pick_groups: Vec<PointPickGroup>,
}

pub(crate) struct PreparedPointCloud {
    pub(crate) origin: DVec3,
    pub(crate) chunks: Vec<PreparedPointChunk>,
    pub(crate) colored: bool,
}

const POINTS_PER_SPATIAL_CHUNK: usize = 256 * 1024;
const POINTS_PER_PICK_GROUP: usize = 512;

/// Convert source doubles into a spatially coherent, render-ready hierarchy.
/// This is called by the point-cloud loader, never by the render thread.
pub(crate) fn prepare_for_render(
    points: &[DVec3],
    colors: Option<&[u32]>,
    bounds: (DVec3, DVec3),
) -> PreparedPointCloud {
    let origin = (bounds.0 + bounds.1) * 0.5;
    let extent = (bounds.1 - bounds.0).max(DVec3::splat(f64::EPSILON));

    if let Some(colors) = colors {
        let mut sorted = points
            .iter()
            .zip(colors.iter().copied().chain(std::iter::repeat(0xffff_ffff)))
            .filter(|(point, _)| point.is_finite())
            .map(|(point, color)| PointInstance {
                pos: (*point - origin).as_vec3().to_array(),
                color,
            })
            .collect::<Vec<_>>();
        sorted.sort_unstable_by_key(|point| morton_key(point.pos, origin, bounds.0, extent));
        PreparedPointCloud {
            origin,
            chunks: sorted
                .chunks(POINTS_PER_SPATIAL_CHUNK)
                .map(build_colored_chunk)
                .collect(),
            colored: true,
        }
    } else {
        let mut sorted = points
            .iter()
            .filter(|point| point.is_finite())
            .map(|point| PointPosition {
                pos: (*point - origin).as_vec3().to_array(),
            })
            .collect::<Vec<_>>();
        sorted.sort_unstable_by_key(|point| morton_key(point.pos, origin, bounds.0, extent));
        PreparedPointCloud {
            origin,
            chunks: sorted
                .chunks(POINTS_PER_SPATIAL_CHUNK)
                .map(build_uncolored_chunk)
                .collect(),
            colored: false,
        }
    }
}

fn build_colored_chunk(points: &[PointInstance]) -> PreparedPointChunk {
    let (bounds_min, bounds_max) = local_bounds(points.iter().map(|point| point.pos));
    let (ordered, level_counts) = lod_prefix_order(points);
    let pick_groups = build_pick_groups(&ordered, level_counts, |point| point.pos);
    PreparedPointChunk {
        data: PreparedPointData::Colored(ordered),
        level_counts,
        bounds_min,
        bounds_max,
        pick_groups,
    }
}

fn build_uncolored_chunk(points: &[PointPosition]) -> PreparedPointChunk {
    let (bounds_min, bounds_max) = local_bounds(points.iter().map(|point| point.pos));
    let (ordered, level_counts) = lod_prefix_order(points);
    let pick_groups = build_pick_groups(&ordered, level_counts, |point| point.pos);
    PreparedPointChunk {
        data: PreparedPointData::Uncolored(ordered),
        level_counts,
        bounds_min,
        bounds_max,
        pick_groups,
    }
}

fn build_pick_groups<T>(
    points: &[T],
    level_counts: [u32; 3],
    position: impl Fn(&T) -> [f32; 3] + Copy,
) -> Vec<PointPickGroup> {
    let coarse = level_counts[2] as usize;
    let medium = level_counts[1] as usize;
    let full = level_counts[0] as usize;
    let mut groups = Vec::new();
    for (tier_start, tier_end) in [(0, coarse), (coarse, medium), (medium, full)] {
        for start in (tier_start..tier_end).step_by(POINTS_PER_PICK_GROUP) {
            let end = (start + POINTS_PER_PICK_GROUP).min(tier_end);
            let (bounds_min, bounds_max) = local_bounds(points[start..end].iter().map(position));
            groups.push(PointPickGroup {
                start: start as u32,
                end: end as u32,
                bounds_min,
                bounds_max,
            });
        }
    }
    groups
}

fn lod_prefix_order<T: Copy>(points: &[T]) -> (Vec<T>, [u32; 3]) {
    let mut ordered = Vec::with_capacity(points.len());
    ordered.extend(points.iter().step_by(4).copied());
    let coarse_count = ordered.len();
    ordered.extend(
        points
            .iter()
            .enumerate()
            .filter(|(index, _)| index % 2 == 0 && index % 4 != 0)
            .map(|(_, point)| *point),
    );
    let medium_count = ordered.len();
    ordered.extend(
        points
            .iter()
            .enumerate()
            .filter(|(index, _)| index % 2 != 0)
            .map(|(_, point)| *point),
    );
    (
        ordered,
        [
            points.len() as u32,
            medium_count as u32,
            coarse_count as u32,
        ],
    )
}

fn local_bounds(points: impl Iterator<Item = [f32; 3]>) -> (glam::Vec3, glam::Vec3) {
    points.map(glam::Vec3::from_array).fold(
        (
            glam::Vec3::splat(f32::INFINITY),
            glam::Vec3::splat(f32::NEG_INFINITY),
        ),
        |(min, max), point| (min.min(point), max.max(point)),
    )
}

fn morton_key(local: [f32; 3], origin: DVec3, min: DVec3, extent: DVec3) -> u64 {
    let world = DVec3::from_array(local.map(f64::from)) + origin;
    let normalized = ((world - min) / extent).clamp(DVec3::ZERO, DVec3::ONE);
    let scale = ((1u32 << 21) - 1) as f64;
    let x = (normalized.x * scale) as u32;
    let y = (normalized.y * scale) as u32;
    let z = (normalized.z * scale) as u32;
    interleave_21(x) | (interleave_21(y) << 1) | (interleave_21(z) << 2)
}

fn interleave_21(value: u32) -> u64 {
    let mut value = u64::from(value & 0x1f_ffff);
    value = (value | value << 32) & 0x001f_0000_0000_ffff;
    value = (value | value << 16) & 0x001f_0000_ff00_00ff;
    value = (value | value << 8) & 0x100f_00f0_0f00_f00f;
    value = (value | value << 4) & 0x10c3_0c30_c30c_30c3;
    (value | value << 2) & 0x1249_2492_4924_9249
}
