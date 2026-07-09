//! Persistent GPU representation of loaded point clouds.

use std::collections::{HashMap, HashSet};

use glam::DVec3;
use wgpu::util::DeviceExt;

use crate::model::point_cloud::{OpenPointCloud, PointCloudId};

/// Per-instance point splat: scene-origin-relative position plus packed RGBA8.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PointInstance {
    pub(crate) pos: [f32; 3],
    pub(crate) color: u32,
}

/// Mirrors `PointCloudStyle` in `point_cloud.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PointCloudStyleUniform {
    color: [f32; 4],
    /// x: splat size in physical pixels, y: 1.0 when per-point colours exist.
    options: [f32; 4],
}

pub(crate) struct CachedPointChunk {
    pub(crate) instance_buffer: wgpu::Buffer,
    pub(crate) instance_count: u32,
    /// Scene-origin-relative AABB for per-chunk frustum culling.
    pub(crate) bounds_min: glam::Vec3,
    pub(crate) bounds_max: glam::Vec3,
}

pub(crate) struct CachedPointCloudGpu {
    pub(crate) chunks: Vec<CachedPointChunk>,
    pub(crate) style_buffer: wgpu::Buffer,
    pub(crate) style_bind_group: wgpu::BindGroup,
    color: [f32; 4],
    point_size: f32,
    scene_origin: DVec3,
}

#[derive(Default)]
pub(crate) struct PointCloudGpuCache {
    clouds: HashMap<PointCloudId, CachedPointCloudGpu>,
}

/// Points per GPU chunk: the culling granularity, sized so a chunk's
/// instance buffer stays a few tens of MB. LiDAR files are written in scan
/// order, so consecutive runs stay spatially compact enough to cull.
const POINTS_PER_CHUNK: usize = 2 * 1024 * 1024;

impl PointCloudGpuCache {
    pub(crate) fn clear(&mut self) {
        self.clouds.clear();
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.clouds.is_empty()
    }

    pub(crate) fn get(&self, id: PointCloudId) -> Option<&CachedPointCloudGpu> {
        self.clouds.get(&id)
    }

    pub(crate) fn sync(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene_origin: DVec3,
        scale_factor: f32,
        point_clouds: &[OpenPointCloud],
        style_layout: &wgpu::BindGroupLayout,
    ) {
        let loaded: HashSet<_> = point_clouds.iter().map(|cloud| cloud.id).collect();
        self.clouds.retain(|id, _| loaded.contains(id));
        for cloud in point_clouds {
            let point_size = (cloud.point_size * scale_factor).max(1.0);
            if let Some(cached) = self.clouds.get_mut(&cloud.id) {
                if cached.scene_origin != scene_origin {
                    // Instance positions bake in the scene origin; rebuild.
                    // (Camera code clears the whole cache on origin changes,
                    // so this path is a backstop.)
                    cached.chunks = build_point_chunks(device, scene_origin, cloud);
                    cached.scene_origin = scene_origin;
                }
                if cached.color != cloud.color || cached.point_size != point_size {
                    let style = style_uniform(cloud, point_size);
                    queue.write_buffer(&cached.style_buffer, 0, bytemuck::bytes_of(&style));
                    cached.color = cloud.color;
                    cached.point_size = point_size;
                }
            } else {
                let chunks = build_point_chunks(device, scene_origin, cloud);
                if chunks.is_empty() {
                    continue;
                }
                let style = style_uniform(cloud, point_size);
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
                        chunks,
                        style_buffer,
                        style_bind_group,
                        color: cloud.color,
                        point_size,
                        scene_origin,
                    },
                );
            }
        }
    }
}

fn style_uniform(cloud: &OpenPointCloud, point_size: f32) -> PointCloudStyleUniform {
    PointCloudStyleUniform {
        color: cloud.color,
        options: [
            point_size,
            if cloud.colors.is_some() { 1.0 } else { 0.0 },
            0.0,
            0.0,
        ],
    }
}

fn build_point_chunks(
    device: &wgpu::Device,
    scene_origin: DVec3,
    cloud: &OpenPointCloud,
) -> Vec<CachedPointChunk> {
    let limit = device.limits().max_buffer_size as usize;
    let points_per_chunk = POINTS_PER_CHUNK
        .min(limit / std::mem::size_of::<PointInstance>())
        .max(1);
    let mut chunks = Vec::new();
    let mut instances = Vec::with_capacity(cloud.points.len().min(points_per_chunk));
    for (chunk_index, run) in cloud.points.chunks(points_per_chunk).enumerate() {
        instances.clear();
        let mut bounds_min = glam::Vec3::splat(f32::INFINITY);
        let mut bounds_max = glam::Vec3::splat(f32::NEG_INFINITY);
        for (offset, point) in run.iter().enumerate() {
            if !point.is_finite() {
                continue;
            }
            let scene_rel = (*point - scene_origin).as_vec3();
            bounds_min = bounds_min.min(scene_rel);
            bounds_max = bounds_max.max(scene_rel);
            let color = cloud
                .colors
                .as_ref()
                .and_then(|colors| colors.get(chunk_index * points_per_chunk + offset))
                .copied()
                .unwrap_or(0xffff_ffff);
            instances.push(PointInstance {
                pos: scene_rel.to_array(),
                color,
            });
        }
        if instances.is_empty() {
            continue;
        }
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cached Point Cloud Instances"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });
        chunks.push(CachedPointChunk {
            instance_buffer,
            instance_count: instances.len() as u32,
            bounds_min,
            bounds_max,
        });
    }
    chunks
}
