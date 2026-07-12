//! Persistent, incrementally uploaded GPU representation of point clouds.

use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    sync::Arc,
};

use glam::{DMat4, DVec2, DVec3, Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::model::point_cloud::{OpenPointCloud, PointCloudId, PreparedPointCloud};
use crate::rendering::graphics::frustum::Frustum;

pub(crate) use crate::model::point_cloud::{PointInstance, PointPosition};

/// Mirrors `PointCloudStyle` in `point_cloud.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PointCloudStyleUniform {
    color: [f32; 4],
    /// x: splat size in physical pixels.
    options: [f32; 4],
    /// Cloud-local origin relative to the current floating scene origin.
    origin: [f32; 4],
}

pub(crate) struct CachedPointChunk {
    pub(crate) instance_buffer: wgpu::Buffer,
    pub(crate) level_counts: [u32; 3],
    pub(crate) selected_level: Cell<usize>,
    /// Bounds in cloud-local coordinates.
    pub(crate) bounds_min: glam::Vec3,
    pub(crate) bounds_max: glam::Vec3,
    resident_bytes: usize,
    last_used_frame: u64,
}

pub(crate) struct CachedPointCloudGpu {
    /// Slots retain prepared-chunk indexing even when the GPU allocation is
    /// evicted, keeping render and CPU-pick ranges aligned.
    pub(crate) chunks: Vec<Option<CachedPointChunk>>,
    pub(crate) style_bind_group: wgpu::BindGroup,
    pub(crate) colored: bool,
    pub(crate) origin_scene: glam::Vec3,
    style_buffer: wgpu::Buffer,
    color: [f32; 4],
    point_size: f32,
    scene_origin: DVec3,
    prepared: Arc<PreparedPointCloud>,
    visible: bool,
}

#[derive(Default)]
pub(crate) struct PointCloudGpuCache {
    clouds: HashMap<PointCloudId, CachedPointCloudGpu>,
    frame_index: u64,
    pending_uploads: bool,
    rejected_chunks: HashSet<ChunkKey>,
}

/// Limit upload work in any one render call. Prepared chunks are at most about
/// 4 MiB, so this normally advances several chunks.
const UPLOAD_BUDGET_BYTES: usize = 16 * 1024 * 1024;
/// Point instances share a hard global residency budget. wgpu does not expose
/// physical VRAM, so the effective budget is additionally bounded by twice
/// the adapter's supported single-buffer size.
const MAX_RESIDENCY_BUDGET_BYTES: usize = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ChunkKey {
    cloud: PointCloudId,
    chunk: usize,
}

#[derive(Clone, Copy, Debug)]
struct ResidencyCandidate {
    key: ChunkKey,
    bytes: usize,
    distance_squared: f32,
}

impl PointCloudGpuCache {
    pub(crate) fn is_empty(&self) -> bool {
        !self
            .clouds
            .values()
            .any(|cloud| cloud.chunks.iter().any(Option::is_some))
    }

    pub(crate) fn has_pending_uploads(&self) -> bool {
        self.pending_uploads
    }

    pub(crate) fn get(&self, id: PointCloudId) -> Option<&CachedPointCloudGpu> {
        self.clouds.get(&id)
    }

    /// Nearest depth-writing point splat covering `screen_point` at the LOD
    /// used by the most recent render pass.
    pub(crate) fn nearest_depth_at_screen(
        &self,
        view_proj: &DMat4,
        screen: (f32, f32),
        screen_point: DVec2,
    ) -> Option<f64> {
        let mut nearest = f64::NEG_INFINITY;
        for cached in self.clouds.values().filter(|cached| cached.visible) {
            let half_size = f64::from(cached.point_size) * 0.5;
            for (chunk_index, chunk) in cached
                .chunks
                .iter()
                .enumerate()
                .filter_map(|(index, chunk)| chunk.as_ref().map(|chunk| (index, chunk)))
            {
                let Some(prepared) = cached.prepared.chunks.get(chunk_index) else {
                    continue;
                };
                let count = chunk.level_counts[chunk.selected_level.get()] as usize;
                for group in prepared
                    .pick_groups
                    .iter()
                    .filter(|group| (group.start as usize) < count)
                {
                    let bounds_min = cached.prepared.origin + group.bounds_min.as_dvec3();
                    let bounds_max = cached.prepared.origin + group.bounds_max.as_dvec3();
                    if !projected_bounds_overlap(
                        view_proj,
                        screen,
                        screen_point,
                        half_size,
                        bounds_min,
                        bounds_max,
                    ) {
                        continue;
                    }
                    let end = (group.end as usize).min(count);
                    for index in group.start as usize..end {
                        let Some(local) = prepared.data.position(index) else {
                            continue;
                        };
                        let world =
                            cached.prepared.origin + DVec3::from_array(local.map(f64::from));
                        let Some(projected) =
                            crate::rendering::pick::world_to_screen(view_proj, world, screen)
                        else {
                            continue;
                        };
                        if (projected.x - screen_point.x).abs() > half_size
                            || (projected.y - screen_point.y).abs() > half_size
                        {
                            continue;
                        }
                        let clip = *view_proj * world.extend(1.0);
                        if clip.w.abs() > f64::EPSILON {
                            nearest = nearest.max(clip.z / clip.w);
                        }
                    }
                }
            }
        }
        nearest.is_finite().then_some(nearest)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sync(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene_origin: DVec3,
        scale_factor: f32,
        view_proj: Mat4,
        camera_scene: Vec3,
        point_clouds: &[OpenPointCloud],
        style_layout: &wgpu::BindGroupLayout,
    ) {
        self.frame_index = self.frame_index.wrapping_add(1).max(1);
        let loaded: HashSet<_> = point_clouds.iter().map(|cloud| cloud.id).collect();
        self.clouds.retain(|id, _| loaded.contains(id));
        self.rejected_chunks
            .retain(|key| loaded.contains(&key.cloud));

        for cloud in point_clouds {
            let point_size = (cloud.point_size * scale_factor).max(1.0);
            let replace = self
                .clouds
                .get(&cloud.id)
                .is_some_and(|cached| !Arc::ptr_eq(&cached.prepared, &cloud.prepared));
            if replace {
                self.clouds.remove(&cloud.id);
            }

            if let Some(cached) = self.clouds.get_mut(&cloud.id) {
                cached.visible = cloud.visible;
                if cached.color != cloud.color
                    || cached.point_size != point_size
                    || cached.scene_origin != scene_origin
                {
                    let style = style_uniform(cloud, point_size, scene_origin);
                    queue.write_buffer(&cached.style_buffer, 0, bytemuck::bytes_of(&style));
                    cached.color = cloud.color;
                    cached.point_size = point_size;
                    cached.scene_origin = scene_origin;
                    cached.origin_scene = (cloud.prepared.origin - scene_origin).as_vec3();
                }
                continue;
            }

            let style = style_uniform(cloud, point_size, scene_origin);
            let style_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Point Cloud Style Uniform"),
                contents: bytemuck::bytes_of(&style),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let style_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: style_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: style_buffer.as_entire_binding(),
                }],
                label: Some("Point Cloud Style Bind Group"),
            });
            self.clouds.insert(
                cloud.id,
                CachedPointCloudGpu {
                    style_buffer,
                    style_bind_group,
                    colored: cloud.prepared.colored,
                    origin_scene: (cloud.prepared.origin - scene_origin).as_vec3(),
                    color: cloud.color,
                    point_size,
                    scene_origin,
                    prepared: Arc::clone(&cloud.prepared),
                    chunks: std::iter::repeat_with(|| None)
                        .take(cloud.prepared.chunks.len())
                        .collect(),
                    visible: cloud.visible,
                },
            );
        }

        self.update_residency(device, view_proj, camera_scene);
    }

    fn update_residency(&mut self, device: &wgpu::Device, view_proj: Mat4, camera_scene: Vec3) {
        let max_buffer_bytes =
            usize::try_from(device.limits().max_buffer_size).unwrap_or(usize::MAX);
        let residency_budget = point_cloud_residency_budget(device.limits().max_buffer_size);
        let frustum = Frustum::from_view_proj(view_proj);
        let mut candidates = Vec::new();
        for (&cloud_id, cloud) in &self.clouds {
            if !cloud.visible {
                continue;
            }
            for (chunk_index, prepared) in cloud.prepared.chunks.iter().enumerate() {
                let min = prepared.bounds_min + cloud.origin_scene;
                let max = prepared.bounds_max + cloud.origin_scene;
                if !frustum.intersects_aabb(min, max) {
                    continue;
                }
                let key = ChunkKey {
                    cloud: cloud_id,
                    chunk: chunk_index,
                };
                let bytes = prepared.data.bytes().len();
                if bytes > max_buffer_bytes || bytes > residency_budget {
                    if self.rejected_chunks.insert(key) {
                        crate::userspace_error!(
                            "Point-cloud chunk needs {} MiB, exceeding the GPU allocation budget; the chunk will not be displayed",
                            bytes / (1024 * 1024),
                        );
                    }
                    continue;
                }
                candidates.push(ResidencyCandidate {
                    key,
                    bytes,
                    distance_squared: distance_squared_to_aabb(camera_scene, min, max),
                });
            }
        }

        let desired = select_residency_candidates(candidates, residency_budget);
        let desired_keys = desired
            .iter()
            .map(|candidate| candidate.key)
            .collect::<HashSet<_>>();

        for candidate in &desired {
            if let Some(chunk) = self
                .clouds
                .get_mut(&candidate.key.cloud)
                .and_then(|cloud| cloud.chunks.get_mut(candidate.key.chunk))
                .and_then(Option::as_mut)
            {
                chunk.last_used_frame = self.frame_index;
            }
        }

        let mut resident_bytes = self.resident_bytes();
        let missing_bytes = desired
            .iter()
            .filter(|candidate| !self.is_resident(candidate.key))
            .fold(0usize, |total, candidate| {
                total.saturating_add(candidate.bytes)
            });
        let mut bytes_to_free = resident_bytes
            .saturating_add(missing_bytes)
            .saturating_sub(residency_budget);
        if bytes_to_free > 0 {
            let mut evictable = Vec::new();
            for (&cloud_id, cloud) in &self.clouds {
                for (chunk_index, chunk) in cloud.chunks.iter().enumerate() {
                    let Some(chunk) = chunk else {
                        continue;
                    };
                    let key = ChunkKey {
                        cloud: cloud_id,
                        chunk: chunk_index,
                    };
                    if !desired_keys.contains(&key) {
                        evictable.push((key, chunk.last_used_frame, chunk.resident_bytes));
                    }
                }
            }
            evictable
                .sort_unstable_by_key(|(key, last_used, _)| (*last_used, key.cloud.0, key.chunk));
            for (key, _, bytes) in evictable {
                if bytes_to_free == 0 {
                    break;
                }
                if let Some(slot) = self
                    .clouds
                    .get_mut(&key.cloud)
                    .and_then(|cloud| cloud.chunks.get_mut(key.chunk))
                {
                    *slot = None;
                    resident_bytes = resident_bytes.saturating_sub(bytes);
                    bytes_to_free = bytes_to_free.saturating_sub(bytes);
                }
            }
        }

        let mut upload_budget = UPLOAD_BUDGET_BYTES;
        for candidate in &desired {
            if self.is_resident(candidate.key) {
                continue;
            }
            if candidate.bytes > upload_budget
                || resident_bytes.saturating_add(candidate.bytes) > residency_budget
            {
                continue;
            }
            let (instance_buffer, level_counts, bounds_min, bounds_max) = {
                let prepared =
                    &self.clouds[&candidate.key.cloud].prepared.chunks[candidate.key.chunk];
                (
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Cached Point Cloud Instances"),
                        contents: prepared.data.bytes(),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                    prepared.level_counts,
                    prepared.bounds_min,
                    prepared.bounds_max,
                )
            };
            let cloud = self
                .clouds
                .get_mut(&candidate.key.cloud)
                .expect("residency candidate cloud disappeared");
            cloud.chunks[candidate.key.chunk] = Some(CachedPointChunk {
                instance_buffer,
                level_counts,
                selected_level: Cell::new(0),
                bounds_min,
                bounds_max,
                resident_bytes: candidate.bytes,
                last_used_frame: self.frame_index,
            });
            resident_bytes += candidate.bytes;
            upload_budget -= candidate.bytes;
        }

        self.pending_uploads = desired
            .iter()
            .any(|candidate| !self.is_resident(candidate.key));
    }

    fn resident_bytes(&self) -> usize {
        self.clouds
            .values()
            .flat_map(|cloud| cloud.chunks.iter().filter_map(Option::as_ref))
            .fold(0usize, |total, chunk| {
                total.saturating_add(chunk.resident_bytes)
            })
    }

    fn is_resident(&self, key: ChunkKey) -> bool {
        self.clouds
            .get(&key.cloud)
            .and_then(|cloud| cloud.chunks.get(key.chunk))
            .is_some_and(Option::is_some)
    }
}

fn point_cloud_residency_budget(max_buffer_size: u64) -> usize {
    usize::try_from(max_buffer_size)
        .unwrap_or(usize::MAX)
        .saturating_mul(2)
        .min(MAX_RESIDENCY_BUDGET_BYTES)
}

fn distance_squared_to_aabb(point: Vec3, min: Vec3, max: Vec3) -> f32 {
    let nearest = point.clamp(min, max);
    point.distance_squared(nearest)
}

fn select_residency_candidates(
    mut candidates: Vec<ResidencyCandidate>,
    budget: usize,
) -> Vec<ResidencyCandidate> {
    candidates.sort_by(|a, b| {
        a.distance_squared
            .total_cmp(&b.distance_squared)
            .then_with(|| a.key.cloud.0.cmp(&b.key.cloud.0))
            .then_with(|| a.key.chunk.cmp(&b.key.chunk))
    });
    let mut used = 0usize;
    candidates
        .into_iter()
        .filter(|candidate| {
            let Some(combined) = used.checked_add(candidate.bytes) else {
                return false;
            };
            if combined > budget {
                return false;
            }
            used = combined;
            true
        })
        .collect()
}

fn projected_bounds_overlap(
    view_proj: &DMat4,
    screen: (f32, f32),
    point: DVec2,
    padding: f64,
    min: DVec3,
    max: DVec3,
) -> bool {
    let mut projected_min = DVec2::splat(f64::INFINITY);
    let mut projected_max = DVec2::splat(f64::NEG_INFINITY);
    let mut projected_any = false;
    for x in [min.x, max.x] {
        for y in [min.y, max.y] {
            for z in [min.z, max.z] {
                let clip = *view_proj * DVec3::new(x, y, z).extend(1.0);
                if clip.w <= f64::EPSILON {
                    return true;
                }
                let ndc = clip.truncate() / clip.w;
                let screen_point = DVec2::new(
                    (ndc.x * 0.5 + 0.5) * f64::from(screen.0),
                    (0.5 - ndc.y * 0.5) * f64::from(screen.1),
                );
                projected_min = projected_min.min(screen_point);
                projected_max = projected_max.max(screen_point);
                projected_any = true;
            }
        }
    }
    projected_any
        && point.x >= projected_min.x - padding
        && point.x <= projected_max.x + padding
        && point.y >= projected_min.y - padding
        && point.y <= projected_max.y + padding
}

fn style_uniform(
    cloud: &OpenPointCloud,
    point_size: f32,
    scene_origin: DVec3,
) -> PointCloudStyleUniform {
    let origin = (cloud.prepared.origin - scene_origin).as_vec3();
    PointCloudStyleUniform {
        color: cloud.color,
        options: [point_size, 0.0, 0.0, 0.0],
        origin: [origin.x, origin.y, origin.z, 0.0],
    }
}
