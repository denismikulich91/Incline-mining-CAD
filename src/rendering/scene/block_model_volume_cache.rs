//! CPU scene representation and dynamic GPU streaming cache for block model volumes.

use super::block_model_ramp::{
    VISIBLE_ALPHA_EPSILON, is_hidden_block_appearance, ramp_rgba, volume_sigma_for_alpha,
};
use super::gpu_cache::{block_model_color_values, grade_for_block};
use crate::model::block_model::{ColorTransferFunction, OpenBlockModel};
use glam::{DVec3, Vec3};
use std::collections::VecDeque;
use wgpu::util::DeviceExt;

/// Hard CPU/backing-store ceiling for the optional volume path. Models above
/// it fall back to the already bounded cube renderer instead of doing several
/// GiB of synchronous cell construction during a frame.
pub(crate) const MAX_VOLUME_CELL_BYTES: usize = 1024 * 1024 * 1024;
const MAX_VOLUME_METADATA_BYTES: usize = 512 * 1024 * 1024;
pub(crate) const BRICK_SIZE: usize = 8;
pub(crate) const CELLS_PER_BRICK: usize = BRICK_SIZE * BRICK_SIZE * BRICK_SIZE;
pub(crate) const EMPTY_BRICK: u32 = u32::MAX;
pub(crate) const UNIFORM_BRICK_FLAG: u32 = 0x8000_0000;
pub(crate) const NOT_RESIDENT_SLOT: u32 = 0x7fff_ffff;
const EMPTY_CELL_PAYLOAD: u32 = u32::MAX;
const FALLBACK_CELL_FLAG: u32 = 1 << 31;
const PLANE_DEDUP_EPSILON: f32 = 1.0e-4;

const VOLUME_POOL_BUDGET_FRACTION: f64 = 0.6;
const VOLUME_MAX_UPLOADS_PER_UPDATE: usize = 1024;
const VOLUME_OPACITY_CUTOFF: f32 = 0.95;
const VOLUME_MAX_STEPS: u32 = 4096;
const VOLUME_LOD_FOOTPRINT_FACTOR: f32 = 3.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ColorStopUniform {
    color: [f32; 4],
    pos: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BlockVolumeUniform {
    fallback_color: [f32; 4],
    options: [f32; 4],
    lod: [f32; 4],
    dims: [u32; 4],
    brick_dims: [u32; 4],
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
    scene_to_local_0: [f32; 4],
    scene_to_local_1: [f32; 4],
    scene_to_local_2: [f32; 4],
    stops: [ColorStopUniform; crate::model::block_model::MAX_COLOR_STOPS],
}

pub(crate) struct CachedBlockVolumeGpu {
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) uniform_buffer: wgpu::Buffer,
    pub(crate) _x_planes_buffer: wgpu::Buffer,
    pub(crate) _y_planes_buffer: wgpu::Buffer,
    pub(crate) _z_planes_buffer: wgpu::Buffer,
    pub(crate) cell_pool_buffer: wgpu::Buffer,
    pub(crate) _brick_table_buffer: wgpu::Buffer,
    pub(crate) brick_aggregate_buffer: wgpu::Buffer,
    pub(crate) brick_info_buffer: wgpu::Buffer,
    pub(crate) streamer: BrickStreamer,
    pub(crate) scene_to_local: [[f32; 4]; 3],
    pub(crate) asset: BlockVolumeAsset,
    pub(crate) bounds_min: glam::Vec3,
    pub(crate) bounds_max: glam::Vec3,
}

impl CachedBlockVolumeGpu {
    /// Nearest visible volume cell along a model-local ray. This mirrors the
    /// volume asset's sparse occupancy rather than treating its outer AABB as
    /// solid, so document picks remain possible through genuine holes.
    pub(crate) fn nearest_opaque_cell_hit(
        &self,
        ray_origin: Vec3,
        ray_direction: Vec3,
        color_transfer: &ColorTransferFunction,
        fallback_alpha: f32,
    ) -> Option<f32> {
        let asset = &self.asset;
        let brick_dims = asset.brick_dims.map(|value| value as usize);
        let dims = asset.dims.map(|value| value as usize);
        let mut nearest = f32::INFINITY;
        for (ordinal, &brick_index) in asset.occupied_brick_indices.iter().enumerate() {
            if asset.brick_aggregates[ordinal][3] <= 0.0 {
                continue;
            }
            let brick_index = brick_index as usize;
            let bi = brick_index % brick_dims[0];
            let bj = (brick_index / brick_dims[0]) % brick_dims[1];
            let bk = brick_index / (brick_dims[0] * brick_dims[1]);
            let starts = [bi * BRICK_SIZE, bj * BRICK_SIZE, bk * BRICK_SIZE];
            let ends = [
                (starts[0] + BRICK_SIZE).min(dims[0]),
                (starts[1] + BRICK_SIZE).min(dims[1]),
                (starts[2] + BRICK_SIZE).min(dims[2]),
            ];
            let brick_min = Vec3::new(
                asset.x_planes[starts[0]],
                asset.y_planes[starts[1]],
                asset.z_planes[starts[2]],
            );
            let brick_max = Vec3::new(
                asset.x_planes[ends[0]],
                asset.y_planes[ends[1]],
                asset.z_planes[ends[2]],
            );
            if ray_box_distance_f32(ray_origin, ray_direction, brick_min, brick_max)
                .is_none_or(|distance| distance >= nearest)
            {
                continue;
            }
            let cells = asset.cells.brick_cells(ordinal as u32);
            for k in starts[2]..ends[2] {
                for j in starts[1]..ends[1] {
                    for i in starts[0]..ends[0] {
                        let local_index =
                            brick_local_index(i - starts[0], j - starts[1], k - starts[2]);
                        let payload = cells[local_index];
                        let alpha = if payload == EMPTY_CELL_PAYLOAD {
                            0.0
                        } else if payload & FALLBACK_CELL_FLAG != 0 {
                            fallback_alpha
                        } else {
                            ramp_rgba(color_transfer, (payload & 0xffff) as f32 / 65535.0)[3]
                        };
                        if alpha < 0.98 {
                            continue;
                        }
                        let cell_min =
                            Vec3::new(asset.x_planes[i], asset.y_planes[j], asset.z_planes[k]);
                        let cell_max = Vec3::new(
                            asset.x_planes[i + 1],
                            asset.y_planes[j + 1],
                            asset.z_planes[k + 1],
                        );
                        if let Some(distance) =
                            ray_box_distance_f32(ray_origin, ray_direction, cell_min, cell_max)
                            && distance < nearest
                        {
                            nearest = distance;
                        }
                    }
                }
            }
        }
        nearest.is_finite().then_some(nearest)
    }
}

fn ray_box_distance_f32(origin: Vec3, direction: Vec3, min: Vec3, max: Vec3) -> Option<f32> {
    let mut near = 0.0_f32;
    let mut far = f32::INFINITY;
    for axis in 0..3 {
        if direction[axis].abs() <= f32::EPSILON {
            if origin[axis] < min[axis] || origin[axis] > max[axis] {
                return None;
            }
            continue;
        }
        let inverse = direction[axis].recip();
        let first = (min[axis] - origin[axis]) * inverse;
        let second = (max[axis] - origin[axis]) * inverse;
        near = near.max(first.min(second));
        far = far.min(first.max(second));
        if near > far {
            return None;
        }
    }
    Some(near)
}

pub(crate) enum CellBacking {
    Ram(Vec<u32>),
    Mapped(MappedCells),
}

pub(crate) struct MappedCells {
    mmap: memmap2::Mmap,
    // Keep the anonymous temporary file alive for exactly as long as its
    // mapping. `tempfile()` unlinks immediately on Unix and uses
    // delete-on-close on Windows, so the OS also cleans it after a crash.
    // Declaring the mapping first makes it drop before the file handle.
    _file: std::fs::File,
}

impl CellBacking {
    pub(crate) fn brick_cells(&self, ordinal: u32) -> &[u32] {
        let start = ordinal as usize * CELLS_PER_BRICK;
        let end = start + CELLS_PER_BRICK;
        match self {
            CellBacking::Ram(v) => &v[start..end],
            CellBacking::Mapped(m) => bytemuck::cast_slice(&m.mmap[start * 4..end * 4]),
        }
    }
}

enum CellBackingBuilder {
    Ram(Vec<u32>),
    Mapped(MappedCellBackingBuilder),
}

struct MappedCellBackingBuilder {
    // Drop the mapping before its delete-on-close file handle on every early
    // return from sizing, initialization, flush, or read-only conversion.
    mmap: memmap2::MmapMut,
    file: std::fs::File,
}

const MAX_VOLUME_RAM_CELL_BYTES: usize = 256 * 1024 * 1024;

impl CellBackingBuilder {
    fn new(cell_count: usize) -> Result<Self, String> {
        let bytes = cell_count
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| "cell byte size overflows usize".to_owned())?;
        if bytes <= MAX_VOLUME_RAM_CELL_BYTES {
            return Ok(CellBackingBuilder::Ram(vec![
                EMPTY_CELL_PAYLOAD;
                cell_count
            ]));
        }
        Self::new_mapped(cell_count)
    }

    fn new_mapped(cell_count: usize) -> Result<Self, String> {
        Self::new_mapped_in(cell_count, &std::env::temp_dir())
    }

    fn new_mapped_in(cell_count: usize, directory: &std::path::Path) -> Result<Self, String> {
        let bytes = cell_count
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| "cell byte size overflows usize".to_owned())?;
        // Unlike a named PID/counter path, `tempfile_in` creates a private,
        // unpredictable file and asks the OS to make it anonymous. Its File
        // handle owns cleanup immediately, including every failure below.
        let file = create_anonymous_cell_file(directory)?;
        file.set_len(bytes as u64)
            .map_err(|e| format!("sizing temp cell file to {bytes} bytes: {e}"))?;
        let mut mmap = unsafe { memmap2::MmapMut::map_mut(&file) }
            .map_err(|e| format!("mapping temp cell file: {e}"))?;
        let words: &mut [u32] = bytemuck::cast_slice_mut(&mut mmap[..]);
        words.fill(EMPTY_CELL_PAYLOAD);
        Ok(CellBackingBuilder::Mapped(MappedCellBackingBuilder {
            mmap,
            file,
        }))
    }

    fn as_mut_slice(&mut self) -> &mut [u32] {
        match self {
            CellBackingBuilder::Ram(v) => v.as_mut_slice(),
            CellBackingBuilder::Mapped(mapped) => bytemuck::cast_slice_mut(&mut mapped.mmap[..]),
        }
    }

    fn finish(self) -> Result<CellBacking, String> {
        match self {
            CellBackingBuilder::Ram(v) => Ok(CellBacking::Ram(v)),
            CellBackingBuilder::Mapped(mapped) => {
                mapped
                    .mmap
                    .flush()
                    .map_err(|e| format!("flushing temp cell file: {e}"))?;
                let mmap = mapped
                    .mmap
                    .make_read_only()
                    .map_err(|e| format!("sealing temp cell file: {e}"))?;
                Ok(CellBacking::Mapped(MappedCells {
                    mmap,
                    _file: mapped.file,
                }))
            }
        }
    }
}

fn create_anonymous_cell_file(directory: &std::path::Path) -> Result<std::fs::File, String> {
    let file = tempfile::tempfile_in(directory)
        .map_err(|e| format!("creating anonymous temp cell file: {e}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        // Linux's O_TMPFILE path can inherit OpenOptions' default mode. The
        // file is already anonymous here, so tightening it has no exposure
        // window, and the owned handle still cleans up if chmod fails.
        file.set_permissions(std::fs::Permissions::from_mode(0o600))
            .map_err(|e| format!("restricting anonymous temp cell file permissions: {e}"))?;
    }
    Ok(file)
}

const NO_OCCUPANT: u32 = u32::MAX;

pub(crate) struct BrickStreamer {
    pool_slots: u32,
    ordinal_slot: Vec<u32>,
    slot_occupant: Vec<u32>,
    free_slots: Vec<u32>,
    uniform: Vec<bool>,
    centers: Vec<[f32; 3]>,
    pending: VecDeque<u32>,
    fits: bool,
    dirty: bool,
    last_camera: Option<Vec3>,
    replan_distance: f32,
    desired_scratch: Vec<bool>,
}

#[derive(Default)]
struct StreamPlan {
    uploads: Vec<(u32, u32)>,
    info_updates: Vec<(u32, u32)>,
}

impl BrickStreamer {
    fn new(
        pool_slots: u32,
        uniform: Vec<bool>,
        centers: Vec<[f32; 3]>,
        replan_distance: f32,
    ) -> Self {
        let occupied = uniform.len();
        let streamable = uniform.iter().filter(|&&u| !u).count();
        BrickStreamer {
            pool_slots,
            ordinal_slot: vec![NOT_RESIDENT_SLOT; occupied],
            slot_occupant: vec![NO_OCCUPANT; pool_slots as usize],
            free_slots: (0..pool_slots).rev().collect(),
            uniform,
            centers,
            pending: VecDeque::new(),
            fits: streamable <= pool_slots as usize,
            dirty: true,
            last_camera: None,
            replan_distance,
            desired_scratch: vec![false; occupied],
        }
    }

    fn info(&self, ordinal: u32) -> u32 {
        let flag = if self.uniform[ordinal as usize] {
            UNIFORM_BRICK_FLAG
        } else {
            0
        };
        flag | self.ordinal_slot[ordinal as usize]
    }

    fn all_info(&self) -> Vec<u32> {
        (0..self.ordinal_slot.len() as u32)
            .map(|o| self.info(o))
            .collect()
    }

    fn set_uniform(&mut self, uniform: Vec<bool>) {
        debug_assert_eq!(uniform.len(), self.uniform.len());
        for (ordinal, &now_uniform) in uniform.iter().enumerate() {
            let was_resident = self.ordinal_slot[ordinal] != NOT_RESIDENT_SLOT;
            if now_uniform && was_resident {
                let slot = self.ordinal_slot[ordinal];
                self.slot_occupant[slot as usize] = NO_OCCUPANT;
                self.free_slots.push(slot);
                self.ordinal_slot[ordinal] = NOT_RESIDENT_SLOT;
            }
        }
        let streamable = uniform.iter().filter(|&&u| !u).count();
        self.fits = streamable <= self.pool_slots as usize;
        self.uniform = uniform;
        self.dirty = true;
    }

    fn needs_replan(&self, camera_local: Vec3) -> bool {
        if self.dirty {
            return true;
        }
        if self.fits {
            return false;
        }
        self.last_camera
            .is_none_or(|c| c.distance(camera_local) > self.replan_distance)
    }

    fn replan(&mut self, camera_local: Vec3) -> Vec<(u32, u32)> {
        self.dirty = false;
        self.last_camera = Some(camera_local);
        self.pending.clear();
        let occupied = self.ordinal_slot.len();

        for d in self.desired_scratch.iter_mut() {
            *d = false;
        }
        let mut order: Vec<u32> = (0..occupied as u32)
            .filter(|&o| !self.uniform[o as usize])
            .collect();
        if !self.fits {
            let cam = camera_local;
            let dist2 = |o: u32| {
                let c = self.centers[o as usize];
                (Vec3::from(c) - cam).length_squared()
            };
            let k = self.pool_slots as usize;
            order.select_nth_unstable_by(k - 1, |&a, &b| dist2(a).total_cmp(&dist2(b)));
            order.truncate(k);
        }
        for &o in &order {
            self.desired_scratch[o as usize] = true;
        }

        let mut evictions = Vec::new();
        for slot in 0..self.slot_occupant.len() {
            let occupant = self.slot_occupant[slot];
            if occupant != NO_OCCUPANT && !self.desired_scratch[occupant as usize] {
                self.slot_occupant[slot] = NO_OCCUPANT;
                self.free_slots.push(slot as u32);
                self.ordinal_slot[occupant as usize] = NOT_RESIDENT_SLOT;
                evictions.push((occupant, self.info(occupant)));
            }
        }

        if !self.fits {
            let cam = camera_local;
            order.sort_by(|&a, &b| {
                let da = (Vec3::from(self.centers[a as usize]) - cam).length_squared();
                let db = (Vec3::from(self.centers[b as usize]) - cam).length_squared();
                da.total_cmp(&db)
            });
        }
        for &o in &order {
            if self.ordinal_slot[o as usize] == NOT_RESIDENT_SLOT {
                self.pending.push_back(o);
            }
        }
        evictions
    }

    fn drain(&mut self, max_uploads: usize) -> StreamPlan {
        let mut plan = StreamPlan::default();
        while plan.uploads.len() < max_uploads {
            let Some(ordinal) = self.pending.pop_front() else {
                break;
            };
            if self.ordinal_slot[ordinal as usize] != NOT_RESIDENT_SLOT
                || self.uniform[ordinal as usize]
            {
                continue;
            }
            let Some(slot) = self.free_slots.pop() else {
                self.pending.push_front(ordinal);
                break;
            };
            self.slot_occupant[slot as usize] = ordinal;
            self.ordinal_slot[ordinal as usize] = slot;
            plan.uploads.push((slot, ordinal));
            plan.info_updates.push((ordinal, self.info(ordinal)));
        }
        plan
    }

    fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }
}

pub(crate) struct BlockVolumeAsset {
    x_planes: Vec<f32>,
    y_planes: Vec<f32>,
    z_planes: Vec<f32>,
    pub(crate) cells: CellBacking,
    pub(crate) brick_table: Vec<u32>,
    /// Brick-table index for each occupied ordinal. Cached because style
    /// changes are frequent and rediscovering/sorting this mapping walked the
    /// entire sparse address space on every colour-picker tick.
    occupied_brick_indices: Vec<u32>,
    pub(crate) brick_aggregates: Vec<[f32; 4]>,
    pub(crate) brick_uniform: Vec<bool>,
    brick_centers: Vec<[f32; 3]>,
    occupied_count: usize,
    dims: [u32; 3],
    brick_dims: [u32; 3],
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    reference_len: f32,
}

pub(crate) fn upload_block_volume_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene_origin: DVec3,
    block_model: &OpenBlockModel,
    layout: &wgpu::BindGroupLayout,
    asset: BlockVolumeAsset,
) -> Option<CachedBlockVolumeGpu> {
    let storage_limit = device.limits().max_storage_buffer_binding_size;
    let max_buffer = device.limits().max_buffer_size;

    let pool_budget_bytes =
        ((storage_limit.min(max_buffer) as f64) * VOLUME_POOL_BUDGET_FRACTION) as u64;
    let slot_bytes = (CELLS_PER_BRICK * std::mem::size_of::<u32>()) as u64;
    let budget_slots = (pool_budget_bytes / slot_bytes).max(1);
    let mixed_count = asset.brick_uniform.iter().filter(|&&u| !u).count() as u64;
    let pool_slots = mixed_count.clamp(1, budget_slots) as u32;

    let plane_bytes = |n: usize| (n * std::mem::size_of::<f32>()) as u64;
    let brick_table_bytes = (asset.brick_table.len() * std::mem::size_of::<u32>()) as u64;
    let brick_aggregate_bytes =
        (asset.brick_aggregates.len() * std::mem::size_of::<[f32; 4]>()) as u64;
    let brick_info_bytes = (asset.occupied_count * std::mem::size_of::<u32>()) as u64;
    let pool_bytes = pool_slots as u64 * slot_bytes;
    if brick_table_bytes > storage_limit
        || brick_aggregate_bytes > storage_limit
        || brick_info_bytes > storage_limit
        || pool_bytes > storage_limit
        || plane_bytes(asset.x_planes.len()) > storage_limit
        || plane_bytes(asset.y_planes.len()) > storage_limit
        || plane_bytes(asset.z_planes.len()) > storage_limit
    {
        log::warn!(
            "Block model '{}' volume metadata exceeds the storage-buffer limit ({} MiB brick table); falling back to cube transparency",
            block_model.name,
            brick_table_bytes / (1024 * 1024),
        );
        return None;
    }
    if mixed_count > budget_slots {
        log::info!(
            "Block model '{}' streams: {} mixed bricks, pool holds {} ({} MiB); far regions render from aggregates until they stream in",
            block_model.name,
            mixed_count,
            pool_slots,
            pool_bytes / (1024 * 1024),
        );
    }

    let mut streamer = BrickStreamer::new(
        pool_slots,
        asset.brick_uniform.clone(),
        asset.brick_centers.clone(),
        streamer_replan_distance(&asset),
    );
    let seed_camera =
        0.5 * (Vec3::from_array(asset.bounds_min) + Vec3::from_array(asset.bounds_max));
    streamer.replan(seed_camera);
    let seed = streamer.drain(pool_slots as usize);

    let uniform = block_volume_uniform(block_model, scene_origin, &asset);
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Block Model Volume Uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let x_planes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Block Model Volume X Planes"),
        contents: bytemuck::cast_slice(&asset.x_planes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let y_planes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Block Model Volume Y Planes"),
        contents: bytemuck::cast_slice(&asset.y_planes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let z_planes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Block Model Volume Z Planes"),
        contents: bytemuck::cast_slice(&asset.z_planes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let cell_pool_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Block Model Volume Cell Pool"),
        size: pool_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let brick_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Block Model Volume Brick Table"),
        contents: bytemuck::cast_slice(&asset.brick_table),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let brick_aggregate_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Block Model Volume Brick Aggregates"),
        contents: bytemuck::cast_slice(&asset.brick_aggregates),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let brick_info_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Block Model Volume Brick Info"),
        contents: bytemuck::cast_slice(&streamer.all_info()),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    write_cell_uploads(
        queue,
        &cell_pool_buffer,
        &asset.cells,
        &seed.uploads,
        slot_bytes,
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_planes_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: y_planes_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: z_planes_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: cell_pool_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: brick_table_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: brick_aggregate_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: brick_info_buffer.as_entire_binding(),
            },
        ],
        label: Some("Block Model Volume Bind Group"),
    });

    let (bounds_min, bounds_max) = block_model
        .world_bounds()
        .map(|(min, max)| {
            (
                (min - scene_origin).as_vec3(),
                (max - scene_origin).as_vec3(),
            )
        })
        .unwrap_or((
            glam::Vec3::from_array(asset.bounds_min),
            glam::Vec3::from_array(asset.bounds_max),
        ));

    Some(CachedBlockVolumeGpu {
        bind_group,
        uniform_buffer,
        _x_planes_buffer: x_planes_buffer,
        _y_planes_buffer: y_planes_buffer,
        _z_planes_buffer: z_planes_buffer,
        cell_pool_buffer,
        _brick_table_buffer: brick_table_buffer,
        brick_aggregate_buffer,
        brick_info_buffer,
        streamer,
        scene_to_local: scene_to_local_rows(block_model, scene_origin),
        asset,
        bounds_min,
        bounds_max,
    })
}

fn streamer_replan_distance(asset: &BlockVolumeAsset) -> f32 {
    let extent = Vec3::from_array(asset.bounds_max) - Vec3::from_array(asset.bounds_min);
    (extent.length() * 0.05).max(1.0e-3)
}

pub(crate) fn stream_volume_bricks(
    queue: &wgpu::Queue,
    volume: &mut CachedBlockVolumeGpu,
    camera_local: Vec3,
) {
    if volume.streamer.needs_replan(camera_local) {
        let evictions = volume.streamer.replan(camera_local);
        write_info_updates(queue, &volume.brick_info_buffer, evictions);
    }
    if !volume.streamer.has_pending() {
        return;
    }
    let plan = volume.streamer.drain(VOLUME_MAX_UPLOADS_PER_UPDATE);
    let slot_bytes = (CELLS_PER_BRICK * std::mem::size_of::<u32>()) as u64;
    write_cell_uploads(
        queue,
        &volume.cell_pool_buffer,
        &volume.asset.cells,
        &plan.uploads,
        slot_bytes,
    );
    write_info_updates(queue, &volume.brick_info_buffer, plan.info_updates);
}

/// Upload consecutive pool slots together. The streamer commonly assigns a
/// long run of neighbouring free slots; issuing one write per 2 KiB brick made
/// camera movement spend much more CPU time in queue submission than copying.
/// Cap each temporary batch so seeding a very large pool stays memory-bounded.
fn write_cell_uploads(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    cells: &CellBacking,
    uploads: &[(u32, u32)],
    slot_bytes: u64,
) {
    const MAX_BATCH_BRICKS: usize = 1024;
    if uploads.is_empty() {
        return;
    }
    let mut ordered = uploads.to_vec();
    ordered.sort_unstable_by_key(|&(slot, _)| slot);
    let mut start = 0;
    while start < ordered.len() {
        let first_slot = ordered[start].0;
        let mut end = start + 1;
        while end < ordered.len()
            && end - start < MAX_BATCH_BRICKS
            && ordered[end].0 == ordered[end - 1].0 + 1
        {
            end += 1;
        }
        let mut packed = Vec::with_capacity((end - start) * CELLS_PER_BRICK);
        for &(_, ordinal) in &ordered[start..end] {
            packed.extend_from_slice(cells.brick_cells(ordinal));
        }
        queue.write_buffer(
            buffer,
            u64::from(first_slot) * slot_bytes,
            bytemuck::cast_slice(&packed),
        );
        start = end;
    }
}

/// Coalesce neighbouring ordinal updates into a small number of queue writes.
fn write_info_updates(queue: &wgpu::Queue, buffer: &wgpu::Buffer, mut updates: Vec<(u32, u32)>) {
    const MAX_BATCH_INFOS: usize = 16 * 1024;
    updates.sort_unstable_by_key(|&(ordinal, _)| ordinal);
    let mut start = 0;
    while start < updates.len() {
        let first_ordinal = updates[start].0;
        let mut end = start + 1;
        while end < updates.len()
            && end - start < MAX_BATCH_INFOS
            && updates[end].0 == updates[end - 1].0 + 1
        {
            end += 1;
        }
        let infos: Vec<u32> = updates[start..end].iter().map(|&(_, info)| info).collect();
        queue.write_buffer(
            buffer,
            u64::from(first_ordinal) * std::mem::size_of::<u32>() as u64,
            bytemuck::cast_slice(&infos),
        );
        start = end;
    }
}

pub(crate) fn update_block_volume_style(
    queue: &wgpu::Queue,
    volume: &mut CachedBlockVolumeGpu,
    scene_origin: DVec3,
    block_model: &OpenBlockModel,
) {
    let style = compute_brick_style_data(
        &volume.asset,
        &block_model.color_transfer,
        block_model.color,
    );
    queue.write_buffer(
        &volume.brick_aggregate_buffer,
        0,
        bytemuck::cast_slice(&style.aggregates),
    );
    volume.asset.brick_aggregates = style.aggregates;
    volume.asset.brick_uniform = style.uniform_flags.clone();
    volume.streamer.set_uniform(style.uniform_flags);
    queue.write_buffer(
        &volume.brick_info_buffer,
        0,
        bytemuck::cast_slice(&volume.streamer.all_info()),
    );
    let uniform = block_volume_uniform(block_model, scene_origin, &volume.asset);
    queue.write_buffer(&volume.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
}

fn block_volume_uniform(
    block_model: &OpenBlockModel,
    scene_origin: DVec3,
    asset: &BlockVolumeAsset,
) -> BlockVolumeUniform {
    let mut fallback_color = block_model.color;
    fallback_color[3] = fallback_color[3].clamp(0.0, 1.0);

    let mut stops = [ColorStopUniform {
        color: [0.0; 4],
        pos: [0.0; 4],
    }; crate::model::block_model::MAX_COLOR_STOPS];
    let stop_count = block_model
        .color_transfer
        .stops
        .len()
        .clamp(2, crate::model::block_model::MAX_COLOR_STOPS);
    for (slot, stop) in stops.iter_mut().zip(
        block_model
            .color_transfer
            .stops
            .iter()
            .take(crate::model::block_model::MAX_COLOR_STOPS),
    ) {
        slot.color = stop.color;
        slot.pos = [stop.t, 0.0, 0.0, 0.0];
    }

    let [row_x, row_y, row_z] = scene_to_local_rows(block_model, scene_origin);

    BlockVolumeUniform {
        fallback_color,
        options: [
            stop_count as f32,
            asset.reference_len.max(1.0e-6),
            VOLUME_OPACITY_CUTOFF,
            VOLUME_MAX_STEPS as f32,
        ],
        lod: [VOLUME_LOD_FOOTPRINT_FACTOR, 0.0, 0.0, 0.0],
        dims: [asset.dims[0], asset.dims[1], asset.dims[2], 0],
        brick_dims: [
            asset.brick_dims[0],
            asset.brick_dims[1],
            asset.brick_dims[2],
            BRICK_SIZE as u32,
        ],
        bounds_min: [
            asset.bounds_min[0],
            asset.bounds_min[1],
            asset.bounds_min[2],
            0.0,
        ],
        bounds_max: [
            asset.bounds_max[0],
            asset.bounds_max[1],
            asset.bounds_max[2],
            0.0,
        ],
        scene_to_local_0: row_x,
        scene_to_local_1: row_y,
        scene_to_local_2: row_z,
        stops,
    }
}

fn scene_to_local_rows(block_model: &OpenBlockModel, scene_origin: DVec3) -> [[f32; 4]; 3] {
    let rotation = block_model.model.rotation();
    let model_origin_scene = block_model.model.origin() - scene_origin;
    let row = |axis: DVec3| {
        [
            axis.x as f32,
            axis.y as f32,
            axis.z as f32,
            -axis.dot(model_origin_scene) as f32,
        ]
    };
    [
        row(rotation.x_axis),
        row(rotation.y_axis),
        row(rotation.z_axis),
    ]
}

pub(crate) fn apply_scene_to_local(rows: &[[f32; 4]; 3], scene: Vec3) -> Vec3 {
    let axis = |r: &[f32; 4]| r[0] * scene.x + r[1] * scene.y + r[2] * scene.z + r[3];
    Vec3::new(axis(&rows[0]), axis(&rows[1]), axis(&rows[2]))
}

fn validate_volume_metadata_budget(brick_count: usize) -> Result<(), String> {
    // Dense address/occupancy tables plus the worst-case per-occupied-brick
    // aggregate, centre, uniform/residency and streaming metadata. Keep this
    // estimate deliberately conservative; cell payloads have their own cap.
    const BYTES_PER_BRICK_UPPER_BOUND: usize = 64;
    let bytes = brick_count
        .checked_mul(BYTES_PER_BRICK_UPPER_BOUND)
        .ok_or_else(|| "brick metadata byte size overflows usize".to_owned())?;
    if bytes > MAX_VOLUME_METADATA_BYTES || brick_count > u32::MAX as usize {
        return Err(format!(
            "brick grid metadata would require at least {} MiB",
            bytes / (1024 * 1024)
        ));
    }
    Ok(())
}

pub(crate) fn build_block_volume_asset(
    block_model: &OpenBlockModel,
) -> Result<Option<BlockVolumeAsset>, String> {
    let Some((x_planes, y_planes, z_planes)) = block_volume_planes(block_model) else {
        return Ok(None);
    };
    if x_planes.len() < 2 || y_planes.len() < 2 || z_planes.len() < 2 {
        return Ok(None);
    }

    let dims_usize = [x_planes.len() - 1, y_planes.len() - 1, z_planes.len() - 1];
    let brick_dims_usize = brick_grid_dims(dims_usize);
    let brick_count = brick_dims_usize
        .into_iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
        .ok_or_else(|| "brick grid dimensions overflow usize".to_owned())?;
    validate_volume_metadata_budget(brick_count)?;
    let color_values = block_model_color_values(block_model);

    struct PlacedBlock {
        payload: u32,
        range: [[usize; 2]; 3],
    }
    let mut placed: Vec<PlacedBlock> = Vec::new();
    let mut brick_occupied = Vec::new();
    brick_occupied
        .try_reserve_exact(brick_count)
        .map_err(|error| format!("could not allocate brick occupancy table: {error}"))?;
    brick_occupied.resize(brick_count, false);
    for &block_index in block_model.renderable_block_indices.iter() {
        let Some(block) = block_model.blocks.get(block_index).copied() else {
            continue;
        };
        let grade = grade_for_block(
            &color_values,
            block_index,
            block_model.hide_empty_color_values,
        );
        if is_hidden_block_appearance(grade, color_values.is_some(), &block_model.color_transfer) {
            continue;
        }
        let Some(ix0) = plane_index(&x_planes, block.lower.x as f32) else {
            continue;
        };
        let Some(ix1) = plane_index(&x_planes, block.upper.x as f32) else {
            continue;
        };
        let Some(iy0) = plane_index(&y_planes, block.lower.y as f32) else {
            continue;
        };
        let Some(iy1) = plane_index(&y_planes, block.upper.y as f32) else {
            continue;
        };
        let Some(iz0) = plane_index(&z_planes, block.lower.z as f32) else {
            continue;
        };
        let Some(iz1) = plane_index(&z_planes, block.upper.z as f32) else {
            continue;
        };
        if ix1 <= ix0 || iy1 <= iy0 || iz1 <= iz0 {
            continue;
        }
        let payload = pack_cell_payload(grade, color_values.is_some());
        for bk in (iz0 / BRICK_SIZE)..=((iz1 - 1) / BRICK_SIZE) {
            for bj in (iy0 / BRICK_SIZE)..=((iy1 - 1) / BRICK_SIZE) {
                for bi in (ix0 / BRICK_SIZE)..=((ix1 - 1) / BRICK_SIZE) {
                    brick_occupied[brick_index(brick_dims_usize, bi, bj, bk)] = true;
                }
            }
        }
        placed.push(PlacedBlock {
            payload,
            range: [[ix0, ix1], [iy0, iy1], [iz0, iz1]],
        });
    }

    let occupied_count = brick_occupied.iter().filter(|&&occ| occ).count();
    if occupied_count == 0 {
        return Ok(None);
    }

    let sparse_cell_count = occupied_count
        .checked_mul(CELLS_PER_BRICK)
        .ok_or_else(|| "sparse cell count overflows usize".to_owned())?;
    let sparse_cell_bytes = sparse_cell_count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "sparse cell byte size overflows usize".to_owned())?;
    if sparse_cell_bytes > MAX_VOLUME_CELL_BYTES {
        return Err(format!(
            "sparse brick volume would require {} MiB of cell payloads",
            sparse_cell_bytes / (1024 * 1024)
        ));
    }

    let mut brick_table = Vec::new();
    brick_table
        .try_reserve_exact(brick_count)
        .map_err(|error| format!("could not allocate brick address table: {error}"))?;
    brick_table.resize(brick_count, EMPTY_BRICK);
    let mut brick_centers = vec![[0.0f32; 3]; occupied_count];
    let mut occupied_brick_indices = Vec::with_capacity(occupied_count);
    let mut next_ordinal = 0u32;
    for (bindex, &occ) in brick_occupied.iter().enumerate() {
        if !occ {
            continue;
        }
        brick_table[bindex] = next_ordinal;
        occupied_brick_indices.push(bindex as u32);
        let bi = bindex % brick_dims_usize[0];
        let bj = (bindex / brick_dims_usize[0]) % brick_dims_usize[1];
        let bk = bindex / (brick_dims_usize[0] * brick_dims_usize[1]);
        let axis_center = |planes: &[f32], b: usize, dim: usize| {
            let lo = b * BRICK_SIZE;
            let hi = ((b + 1) * BRICK_SIZE).min(dim);
            (planes[lo] + planes[hi]) * 0.5
        };
        brick_centers[next_ordinal as usize] = [
            axis_center(&x_planes, bi, dims_usize[0]),
            axis_center(&y_planes, bj, dims_usize[1]),
            axis_center(&z_planes, bk, dims_usize[2]),
        ];
        next_ordinal += 1;
    }

    let mut builder = CellBackingBuilder::new(sparse_cell_count)?;
    {
        let cells = builder.as_mut_slice();
        for placed_block in &placed {
            let [[ix0, ix1], [iy0, iy1], [iz0, iz1]] = placed_block.range;
            for k in iz0..iz1 {
                for j in iy0..iy1 {
                    for i in ix0..ix1 {
                        let bindex = brick_index(
                            brick_dims_usize,
                            i / BRICK_SIZE,
                            j / BRICK_SIZE,
                            k / BRICK_SIZE,
                        );
                        let ordinal = brick_table[bindex] as usize;
                        let local =
                            brick_local_index(i % BRICK_SIZE, j % BRICK_SIZE, k % BRICK_SIZE);
                        cells[ordinal * CELLS_PER_BRICK + local] = placed_block.payload;
                    }
                }
            }
        }
    }
    let cells = builder.finish()?;

    let reference_len = reference_cell_length(&x_planes, &y_planes, &z_planes);

    let dims = [
        dims_usize[0]
            .try_into()
            .map_err(|_| "x cell count exceeds u32".to_owned())?,
        dims_usize[1]
            .try_into()
            .map_err(|_| "y cell count exceeds u32".to_owned())?,
        dims_usize[2]
            .try_into()
            .map_err(|_| "z cell count exceeds u32".to_owned())?,
    ];
    let brick_dims = [
        brick_dims_usize[0] as u32,
        brick_dims_usize[1] as u32,
        brick_dims_usize[2] as u32,
    ];
    let mut asset = BlockVolumeAsset {
        bounds_min: [x_planes[0], y_planes[0], z_planes[0]],
        bounds_max: [
            *x_planes.last().unwrap(),
            *y_planes.last().unwrap(),
            *z_planes.last().unwrap(),
        ],
        x_planes,
        y_planes,
        z_planes,
        cells,
        brick_table,
        occupied_brick_indices,
        brick_aggregates: Vec::new(),
        brick_uniform: Vec::new(),
        brick_centers,
        occupied_count,
        dims,
        brick_dims,
        reference_len,
    };
    let style = compute_brick_style_data(&asset, &block_model.color_transfer, block_model.color);
    asset.brick_aggregates = style.aggregates;
    asset.brick_uniform = style.uniform_flags;
    Ok(Some(asset))
}

struct VolumeRampLut {
    entries: Vec<([f32; 4], f32)>,
    fallback: ([f32; 4], f32),
}

impl VolumeRampLut {
    fn build(
        color_transfer: &ColorTransferFunction,
        fallback_color: [f32; 4],
        reference_len: f32,
    ) -> Self {
        let entry = |color: [f32; 4]| {
            (
                color,
                volume_sigma_for_alpha(color[3].clamp(0.0, 1.0), reference_len),
            )
        };
        Self {
            entries: (0..=u16::MAX)
                .map(|grade| entry(ramp_rgba(color_transfer, f32::from(grade) / 65535.0)))
                .collect(),
            fallback: entry(fallback_color),
        }
    }

    fn resolve(&self, payload: u32) -> Option<&([f32; 4], f32)> {
        if payload == EMPTY_CELL_PAYLOAD {
            return None;
        }
        if payload & FALLBACK_CELL_FLAG != 0 {
            return Some(&self.fallback);
        }
        Some(&self.entries[(payload & 0xffff) as usize])
    }
}

struct BrickStyleData {
    aggregates: Vec<[f32; 4]>,
    uniform_flags: Vec<bool>,
}

fn compute_brick_style_data(
    asset: &BlockVolumeAsset,
    color_transfer: &ColorTransferFunction,
    fallback_color: [f32; 4],
) -> BrickStyleData {
    use rayon::prelude::*;

    let mut fallback_color = fallback_color;
    fallback_color[3] = fallback_color[3].clamp(0.0, 1.0);
    let lut = VolumeRampLut::build(color_transfer, fallback_color, asset.reference_len);
    let dims = [
        asset.dims[0] as usize,
        asset.dims[1] as usize,
        asset.dims[2] as usize,
    ];
    let brick_dims = [
        asset.brick_dims[0] as usize,
        asset.brick_dims[1] as usize,
        asset.brick_dims[2] as usize,
    ];
    let cell_lengths =
        |planes: &[f32]| -> Vec<f32> { planes.windows(2).map(|pair| pair[1] - pair[0]).collect() };
    let x_lengths = cell_lengths(&asset.x_planes);
    let y_lengths = cell_lengths(&asset.y_planes);
    let z_lengths = cell_lengths(&asset.z_planes);

    let per_brick: Vec<([f32; 4], bool)> = asset
        .occupied_brick_indices
        .par_iter()
        .enumerate()
        .map(|(ordinal, &bindex)| {
            let bindex = bindex as usize;
            let ordinal = ordinal as u32;
            let bi = bindex % brick_dims[0];
            let bj = (bindex / brick_dims[0]) % brick_dims[1];
            let bk = bindex / (brick_dims[0] * brick_dims[1]);
            let brick_cells = asset.cells.brick_cells(ordinal);
            let mut volume_sum = 0.0f64;
            let mut sigma_volume_sum = 0.0f64;
            let mut rgb_sum = [0.0f64; 3];
            let mut first_appearance: Option<Option<[f32; 4]>> = None;
            let mut uniform = true;
            for lk in 0..BRICK_SIZE.min(dims[2] - bk * BRICK_SIZE) {
                let k = bk * BRICK_SIZE + lk;
                for lj in 0..BRICK_SIZE.min(dims[1] - bj * BRICK_SIZE) {
                    let j = bj * BRICK_SIZE + lj;
                    for li in 0..BRICK_SIZE.min(dims[0] - bi * BRICK_SIZE) {
                        let i = bi * BRICK_SIZE + li;
                        let cell_volume = (x_lengths[i] * y_lengths[j] * z_lengths[k]) as f64;
                        volume_sum += cell_volume;
                        let payload = brick_cells[brick_local_index(li, lj, lk)];
                        let visible = lut
                            .resolve(payload)
                            .filter(|(rgba, _)| rgba[3] >= VISIBLE_ALPHA_EPSILON);
                        let appearance = visible.map(|(rgba, _)| *rgba);
                        match first_appearance {
                            None => first_appearance = Some(appearance),
                            Some(first) => uniform &= first == appearance,
                        }
                        let Some((rgba, sigma)) = visible else {
                            continue;
                        };
                        let weight = f64::from(*sigma) * cell_volume;
                        sigma_volume_sum += weight;
                        rgb_sum[0] += f64::from(rgba[0]) * weight;
                        rgb_sum[1] += f64::from(rgba[1]) * weight;
                        rgb_sum[2] += f64::from(rgba[2]) * weight;
                    }
                }
            }
            let aggregate = if sigma_volume_sum > 0.0 && volume_sum > 0.0 {
                [
                    (rgb_sum[0] / sigma_volume_sum) as f32,
                    (rgb_sum[1] / sigma_volume_sum) as f32,
                    (rgb_sum[2] / sigma_volume_sum) as f32,
                    (sigma_volume_sum / volume_sum) as f32,
                ]
            } else {
                [0.0; 4]
            };
            (aggregate, uniform)
        })
        .collect();

    let (aggregates, uniform_flags) = per_brick.into_iter().unzip();
    BrickStyleData {
        aggregates,
        uniform_flags,
    }
}

fn block_volume_planes(block_model: &OpenBlockModel) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let dims = block_model.model.metadata.dims;
    let regular_count = dims
        .into_iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(dim))?;
    if !block_model.model.metadata.is_irregular
        && regular_count == block_model.model.metadata.n_blocks
        && dims.into_iter().all(|dim| dim > 0)
    {
        let lower = block_model.model.metadata.lower;
        let upper = block_model.model.metadata.upper;
        return Some((
            regular_planes(lower.x as f32, upper.x as f32, dims[0]),
            regular_planes(lower.y as f32, upper.y as f32, dims[1]),
            regular_planes(lower.z as f32, upper.z as f32, dims[2]),
        ));
    }

    let mut x_planes = Vec::new();
    let mut y_planes = Vec::new();
    let mut z_planes = Vec::new();
    for &index in block_model.renderable_block_indices.iter() {
        let Some(block) = block_model.blocks.get(index) else {
            continue;
        };
        x_planes.push(block.lower.x as f32);
        x_planes.push(block.upper.x as f32);
        y_planes.push(block.lower.y as f32);
        y_planes.push(block.upper.y as f32);
        z_planes.push(block.lower.z as f32);
        z_planes.push(block.upper.z as f32);
    }
    dedup_planes(&mut x_planes);
    dedup_planes(&mut y_planes);
    dedup_planes(&mut z_planes);
    Some((x_planes, y_planes, z_planes))
}

fn regular_planes(lower: f32, upper: f32, cells: usize) -> Vec<f32> {
    let step = (upper - lower) / cells as f32;
    (0..=cells)
        .map(|index| lower + step * index as f32)
        .collect()
}

fn dedup_planes(planes: &mut Vec<f32>) {
    planes.sort_by(f32::total_cmp);
    planes.dedup_by(|a, b| (*a - *b).abs() <= PLANE_DEDUP_EPSILON);
}

fn plane_index(planes: &[f32], value: f32) -> Option<usize> {
    let insertion = planes.partition_point(|plane| *plane < value - PLANE_DEDUP_EPSILON);
    if insertion < planes.len() && (planes[insertion] - value).abs() <= PLANE_DEDUP_EPSILON {
        Some(insertion)
    } else {
        None
    }
}

fn brick_grid_dims(dims: [usize; 3]) -> [usize; 3] {
    [
        dims[0].div_ceil(BRICK_SIZE),
        dims[1].div_ceil(BRICK_SIZE),
        dims[2].div_ceil(BRICK_SIZE),
    ]
}

fn brick_index(brick_dims: [usize; 3], bi: usize, bj: usize, bk: usize) -> usize {
    (bk * brick_dims[1] + bj) * brick_dims[0] + bi
}

fn brick_local_index(li: usize, lj: usize, lk: usize) -> usize {
    (lk * BRICK_SIZE + lj) * BRICK_SIZE + li
}

fn pack_cell_payload(grade: f32, has_grade: bool) -> u32 {
    if has_grade && grade >= 0.0 {
        let quantized = (grade.clamp(0.0, 1.0) * 65535.0).round() as u32;
        quantized & 0xffff
    } else {
        FALLBACK_CELL_FLAG
    }
}

fn reference_cell_length(x_planes: &[f32], y_planes: &[f32], z_planes: &[f32]) -> f32 {
    let avg_delta = |planes: &[f32]| -> f32 {
        if planes.len() < 2 {
            return 1.0;
        }
        let span = (planes[planes.len() - 1] - planes[0]).abs();
        (span / (planes.len() - 1) as f32).max(1.0e-6)
    };
    (avg_delta(x_planes) * avg_delta(y_planes) * avg_delta(z_planes)).cbrt()
}
