//! Persistent GPU representation of immutable and infrequently-changing scene assets.

use std::collections::{HashMap, HashSet};

use glam::{DVec3, Vec3};
use wgpu::util::DeviceExt;

use crate::{
    model::{
        block_model::{
            BlockModelId, ColorTransferFunction, MAX_COLOR_STOPS, OpenBlockModel,
            numeric_variable_default,
        },
        formats::tri00t,
        triangulation::{OpenTriangulation, TriangulationId},
    },
    rendering::{BlockInstance, SurfaceVertex},
    ui::state::EditorState,
};

const YELLOW_HIGHLIGHT_COLOR: [f32; 4] = [1.0, 0.85, 0.0, 1.0];
const MAX_SURFACE_CHUNK_BYTES: usize = 256 * 1024 * 1024;
const FALLBACK_BLOCK_GRADE: f32 = -1.0;
const HIDDEN_BLOCK_GRADE: f32 = -2.0;
const EMPTY_CELL_PAYLOAD: u32 = u32::MAX;
const FALLBACK_CELL_FLAG: u32 = 1 << 31;
const PLANE_DEDUP_EPSILON: f32 = 1.0e-4;
/// Ceiling on the packed sparse cell payloads. Only matters when a temp-file
/// backing store cannot be created (then the payloads live in RAM and this
/// bounds that allocation); with the mmap backing store the payloads live on
/// disk and never occupy committed RAM, so a model far larger than this still
/// renders — its cells stream into a bounded GPU pool by camera proximity, and
/// regions not resident render from their brick aggregates. The value is a
/// sanity limit against pathological grids / filling the disk, not the render
/// budget (that is `volume_pool_budget_bytes`).
const MAX_VOLUME_CELL_BYTES: usize = 8 * 1024 * 1024 * 1024;
/// Edge length in cells of a macro-brick. Cells are packed brick-by-brick so
/// empty regions cost no payload storage and can be skipped in one DDA step.
/// Must match `BRICK_SIZE` in `block_model_volume.wgsl`.
const BRICK_SIZE: usize = 8;
/// Cell payloads per brick (`BRICK_SIZE^3`); one pool slot holds this many.
const CELLS_PER_BRICK: usize = BRICK_SIZE * BRICK_SIZE * BRICK_SIZE;
/// `brick_table` sentinel for a brick with no occupied cells. Must match
/// `EMPTY_BRICK` in the shader.
const EMPTY_BRICK: u32 = u32::MAX;
/// `brick_info` bit 31: the brick is uniform (one appearance) so its aggregate
/// reproduces per-cell output exactly and it needs no pool slot. Must match
/// `UNIFORM_BRICK_FLAG` in `block_model_volume.wgsl`.
const UNIFORM_BRICK_FLAG: u32 = 0x8000_0000;
/// `brick_info` low bits when a mixed brick's cells are not resident in the
/// pool; the march renders it from its aggregate until it streams in. Must
/// match `NOT_RESIDENT_SLOT` in `block_model_volume.wgsl`.
const NOT_RESIDENT_SLOT: u32 = 0x7fff_ffff;
/// Fraction of the device storage-buffer limit the streaming cell pool may
/// use. Leaves headroom for the other volume buffers (planes, brick table,
/// aggregates, info) which are always fully resident.
const VOLUME_POOL_BUDGET_FRACTION: f64 = 0.6;
/// Upper bound on cell-pool slots refilled per streaming update, so a camera
/// jump spreads its uploads over several frames instead of one hitch; the
/// not-yet-streamed bricks render from their aggregates meanwhile.
const VOLUME_MAX_UPLOADS_PER_UPDATE: usize = 1024;
const VOLUME_OPACITY_CUTOFF: f32 = 0.95;
const VOLUME_MAX_STEPS: u32 = 4096;
/// Pixel-footprint multiple of the reference cell length beyond which the
/// volume raycaster integrates whole bricks from their LOD aggregate instead
/// of stepping individual (sub-pixel) cells. Larger = better far-field image,
/// smaller = faster. Never visually tuned yet.
const VOLUME_LOD_FOOTPRINT_FACTOR: f32 = 3.0;
/// Grades below this are discarded by `block_model.wgsl` (`grade < -1.5`).
/// Geometry building must treat such blocks as absent — a discarded block
/// leaves a hole, so it can't be allowed to cull its neighbours' faces.
const HIDDEN_GRADE_DISCARD_THRESHOLD: f32 = -1.5;
/// Mirrors `VISIBLE_ALPHA_EPSILON` in `block_model.wgsl`: graded fragments
/// whose ramp alpha falls below this are discarded.
const VISIBLE_ALPHA_EPSILON: f32 = 0.004;

fn is_hidden_block_grade(grade: f32) -> bool {
    grade < HIDDEN_GRADE_DISCARD_THRESHOLD
}

/// Whether the fragment shader will discard every fragment of a block with
/// this grade — either the hidden-grade sentinel, or (when a colour variable
/// is active, `has_grade`) a colour-ramp alpha below the visibility epsilon.
/// Must match `fs_main` in `block_model.wgsl` exactly; geometry building
/// treats such blocks as absent.
fn is_hidden_block_appearance(
    grade: f32,
    has_grade: bool,
    color_transfer: &ColorTransferFunction,
) -> bool {
    is_hidden_block_grade(grade)
        || (has_grade && grade >= 0.0 && ramp_alpha(color_transfer, grade) < VISIBLE_ALPHA_EPSILON)
}

fn block_alpha(
    grade: f32,
    has_grade: bool,
    color_transfer: &ColorTransferFunction,
    fallback_alpha: f32,
) -> f32 {
    if has_grade && grade >= 0.0 {
        ramp_alpha(color_transfer, grade)
    } else {
        fallback_alpha
    }
}

/// CPU replica of `ramp_color` in the block-model WGSL shaders. The first
/// stop is a lower cutoff: values below it are transparent. Each stop then
/// starts at its own position and remains active until the next.
///
/// Called per block/cell on hot paths (fingerprints, LUT build), so it works
/// directly on the stops slice — no per-call copy. Fewer than two stops is
/// not a valid ramp; the GPU path zero-pads to two, which resolves every `t`
/// to the zeroed second stop, i.e. fully transparent — replicated here.
fn ramp_rgba(color_transfer: &ColorTransferFunction, t: f32) -> [f32; 4] {
    let stops = &color_transfer.stops[..color_transfer.stops.len().min(MAX_COLOR_STOPS)];
    let [first, .., last] = stops else {
        return [0.0; 4];
    };
    if t < first.t {
        return [0.0; 4];
    }
    if t >= last.t {
        return last.color;
    }
    for pair in stops.windows(2) {
        if t < pair[1].t {
            return pair[0].color;
        }
    }
    last.color
}

fn ramp_alpha(color_transfer: &ColorTransferFunction, t: f32) -> f32 {
    ramp_rgba(color_transfer, t)[3]
}

/// CPU replica of `sigma_for_alpha` in `block_model_volume.wgsl`.
fn volume_sigma_for_alpha(alpha: f32, reference_len: f32) -> f32 {
    let ref_len = reference_len.max(1.0e-6);
    if alpha >= 0.98 {
        return -(0.001f32).ln() / ref_len;
    }
    -((1.0 - alpha).max(0.001)).ln() / ref_len
}

fn make_translucent(color: &mut [f32; 4]) {
    color[3] *= 0.3;
}

#[derive(Clone, Copy)]
enum BlockSurfaceSelection {
    All,
    OpaqueOnly,
    TransparentOnly,
}

pub(crate) struct CachedTriangulationGpu {
    pub(crate) surface_chunks: Vec<CachedSurfaceChunk>,
    pub(crate) surface_style_buffer: wgpu::Buffer,
    pub(crate) surface_style_bind_group: wgpu::BindGroup,
    /// Cached computed colour; used for transparency sorting and dirty-checking.
    pub(crate) color: [f32; 4],
    pub(crate) edge_chunks: Vec<CachedEdgeChunk>,
    pub(crate) edge_style_buffer: wgpu::Buffer,
    pub(crate) edge_style_bind_group: wgpu::BindGroup,
    line_color: [f32; 4],
    pub(crate) edge_width: f32,
}

pub(crate) struct CachedSurfaceChunk {
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) index_buffer: wgpu::Buffer,
    pub(crate) index_count: u32,
}

pub(crate) struct CachedEdgeChunk {
    pub(crate) instance_buffer: wgpu::Buffer,
    pub(crate) instance_count: u32,
}

/// Per-instance edge geometry — position only. Color and width live in EdgeStyleUniform.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct EdgeInstance {
    pub(crate) start: [f32; 3],
    pub(crate) end: [f32; 3],
}

/// Per-triangulation surface colour uniform. Updated cheaply when colour changes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SurfaceStyleUniform {
    color: [f32; 4],
}

/// Mirrors `ColorStop` for upload; `pos.x` holds the stop's `t`, the rest is
/// padding to keep the array's stride 16-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ColorStopUniform {
    color: [f32; 4],
    pos: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlockModelStyleUniform {
    fallback_color: [f32; 4],
    options: [f32; 4],
    /// Local->scene rotation as three column vectors (`.xyz` used), and the
    /// scene-relative translation. The block-model shader expands each
    /// instance's local bounds with `rotation * local + translation`. Field
    /// order and vec4 padding must match `BlockModelStyle` in the WGSL.
    rotation_0: [f32; 4],
    rotation_1: [f32; 4],
    rotation_2: [f32; 4],
    translation: [f32; 4],
    stops: [ColorStopUniform; MAX_COLOR_STOPS],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BlockVolumeUniform {
    fallback_color: [f32; 4],
    // x: active stop count, y: reference cell length, z: opacity cutoff,
    // w: maximum DDA steps.
    options: [f32; 4],
    // x: pixel-footprint multiple of the reference cell length at which the
    // march switches to whole-brick LOD integration; yzw unused.
    lod: [f32; 4],
    dims: [u32; 4],
    // xyz: brick counts per axis; w: brick edge length in cells.
    brick_dims: [u32; 4],
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
    // scene-relative position -> model-local position affine rows.
    scene_to_local_0: [f32; 4],
    scene_to_local_1: [f32; 4],
    scene_to_local_2: [f32; 4],
    stops: [ColorStopUniform; MAX_COLOR_STOPS],
}

/// Per-triangulation edge style uniform. Updated cheaply when colour/width changes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EdgeStyleUniform {
    color: [f32; 4],
    width: f32,
    _pad: [f32; 3],
}

#[derive(Default)]
pub(crate) struct TriangulationGpuCache {
    meshes: HashMap<TriangulationId, CachedTriangulationGpu>,
}

#[derive(Default)]
pub(crate) struct BlockModelGpuCache {
    models: HashMap<BlockModelId, CachedBlockModelGpu>,
}

pub(crate) struct CachedBlockModelGpu {
    /// Depth-writing chunks: all chunks for an opaque model, or only the
    /// alpha-opaque grade subset for a mixed-alpha ramp.
    pub(crate) surface_chunks: Vec<CachedBlockModelSurfaceChunk>,
    /// Order-independent transparent chunks. Empty unless the whole model is
    /// translucent or the active ramp has genuinely partial-alpha grades.
    pub(crate) transparent_surface_chunks: Vec<CachedBlockModelSurfaceChunk>,
    /// Solid-volume raycast representation used for transparent/mixed-alpha
    /// block models. When present it replaces both the opaque and transparent
    /// cube draws for this model, preserving front-to-back volumetric opacity.
    pub(crate) volume: Option<CachedBlockVolumeGpu>,
    pub(crate) surface_style_buffer: wgpu::Buffer,
    pub(crate) surface_style_bind_group: wgpu::BindGroup,
    pub(crate) translucent: bool,
    force_translucent: bool,
    pub(crate) edge_chunks: Vec<CachedEdgeChunk>,
    pub(crate) edge_style_buffer: wgpu::Buffer,
    pub(crate) edge_style_bind_group: wgpu::BindGroup,
    line_color: [f32; 4],
    edge_width: f32,
    variable: Option<String>,
    color_transfer: ColorTransferFunction,
    hide_empty_color_values: bool,
    /// See [`hidden_blocks_fingerprint`]: geometry was built for this set of
    /// shader-hidden blocks; when it changes, recolor-in-place is not enough.
    hidden_fingerprint: u64,
    scene_origin: DVec3,
}

pub(crate) struct CachedBlockVolumeGpu {
    pub(crate) bind_group: wgpu::BindGroup,
    /// Kept (with `COPY_DST`) so a colour-ramp change that leaves occupancy
    /// unchanged can re-upload the stops in place — see
    /// [`update_block_volume_style`].
    uniform_buffer: wgpu::Buffer,
    pub(crate) _x_planes_buffer: wgpu::Buffer,
    pub(crate) _y_planes_buffer: wgpu::Buffer,
    pub(crate) _z_planes_buffer: wgpu::Buffer,
    /// Bounded cell-payload pool (`COPY_DST`): `pool_slots * CELLS_PER_BRICK`
    /// payloads. Bricks stream in/out of its slots by camera proximity.
    cell_pool_buffer: wgpu::Buffer,
    pub(crate) _brick_table_buffer: wgpu::Buffer,
    /// Kept (with `COPY_DST`) for in-place LOD-aggregate re-upload on a
    /// ramp-only change.
    brick_aggregate_buffer: wgpu::Buffer,
    /// Per-ordinal `brick_info` (`COPY_DST`): `UNIFORM_BRICK_FLAG | slot`.
    /// Rewritten fully on restyle (uniformity changed) and per-entry as bricks
    /// stream in and out.
    brick_info_buffer: wgpu::Buffer,
    /// Residency manager for the cell pool — decides which mixed bricks are
    /// resident and emits the pool/`brick_info` updates each frame.
    streamer: BrickStreamer,
    /// Affine rows mapping a scene-relative position to model-local space
    /// (the same transform the shader's uniform carries), so the streamer can
    /// score bricks by the camera's model-local distance. Rebuilt with the
    /// volume on a `scene_origin` change.
    scene_to_local: [[f32; 4]; 3],
    /// CPU copy of the built asset, retained so a ramp-only restyle can
    /// recompute just the LOD aggregates + uniformity without re-walking
    /// blocks, and so the streamer can source cells for the pool from the
    /// backing store. The cell backing is mmap'd for large models, so this is
    /// not a large committed-RAM copy.
    asset: BlockVolumeAsset,
    pub(crate) bounds_min: glam::Vec3,
    pub(crate) bounds_max: glam::Vec3,
}

/// A block model surface chunk plus enough CPU-side state to re-colour it
/// (on an active-variable/legend switch) without re-walking blocks, redoing
/// face culling, or reallocating GPU buffers — only geometry changes
/// (translucency toggling, or the block model itself changing) need a full
/// rebuild.
pub(crate) struct CachedBlockModelSurfaceChunk {
    pub(crate) gpu: CachedBlockModelChunkGpu,
    /// Mirrors the instance buffer's current contents so `grade` can be
    /// patched in place and re-uploaded with a single `write_buffer`.
    cpu_instances: Vec<BlockInstance>,
    /// Source block index for each instance, parallel to `cpu_instances`, so
    /// recolour can recompute each block's grade without re-deriving geometry.
    block_indices: Vec<usize>,
}

/// GPU handles for one instanced block-model chunk. Unlike a triangulation's
/// [`CachedSurfaceChunk`] there is no index buffer: each block is a single
/// 32-byte instance the shader expands into a cube via a non-indexed draw.
pub(crate) struct CachedBlockModelChunkGpu {
    pub(crate) instance_buffer: wgpu::Buffer,
    pub(crate) instance_count: u32,
    /// Scene-relative bounds of this chunk's blocks, for frustum culling and
    /// transparency depth sorting.
    pub(crate) bounds_min: glam::Vec3,
    pub(crate) bounds_max: glam::Vec3,
}

impl BlockModelGpuCache {
    pub(crate) fn clear(&mut self) {
        self.models.clear();
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Advance cell-pool streaming for every cached volume: bricks near the
    /// camera become resident (their cells uploaded into the bounded pool),
    /// far bricks fall back to their aggregates. A no-op for volumes whose
    /// mixed bricks all fit the pool. `camera_scene` is the camera position
    /// relative to `scene_origin` (each volume maps it into its own local
    /// space). Call once per frame before the volume pass.
    pub(crate) fn stream_volumes(&mut self, queue: &wgpu::Queue, camera_scene: Vec3) {
        for cached in self.models.values_mut() {
            if let Some(volume) = cached.volume.as_mut() {
                let camera_local = apply_scene_to_local(&volume.scene_to_local, camera_scene);
                stream_volume_bricks(queue, volume, camera_local);
            }
        }
    }

    pub(crate) fn get(&self, id: BlockModelId) -> Option<&CachedBlockModelGpu> {
        self.models.get(&id)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sync(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene_origin: DVec3,
        scale_factor: f32,
        block_models: &[OpenBlockModel],
        editor: &EditorState,
        surface_style_layout: &wgpu::BindGroupLayout,
        volume_layout: &wgpu::BindGroupLayout,
        edge_style_layout: &wgpu::BindGroupLayout,
    ) {
        let loaded: HashSet<_> = block_models.iter().map(|model| model.id).collect();
        self.models.retain(|id, _| loaded.contains(id));
        for block_model in block_models {
            let entity = block_model.entity_id();
            let selected = editor.selected_handles.contains(&entity);
            let force_translucent = editor.translucent_handles.contains(&entity);
            let has_partial_alpha_stops = block_model_has_partial_alpha_stops(block_model);
            let translucent = force_translucent || has_partial_alpha_stops;
            let mut line_color = if selected {
                crate::ui::SELECTION_COLOR_F32
            } else {
                [0.03, 0.05, 0.06, 1.0]
            };
            if translucent {
                make_translucent(&mut line_color);
            }
            let show_edges = selected || editor.topology_wireframes_enabled;
            let edge_width = if show_edges {
                scale_factor.max(1.0)
            } else {
                0.0
            };

            if let Some(cached) = self.models.get_mut(&block_model.id) {
                let variable_dirty = cached.variable != block_model.active_numeric_variable;
                let geometry_dirty = cached.translucent != translucent
                    || cached.force_translucent != force_translucent;
                let style_dirty = cached.color_transfer != block_model.color_transfer;
                let empty_visibility_dirty =
                    cached.hide_empty_color_values != block_model.hide_empty_color_values;
                let scene_origin_dirty = cached.scene_origin != scene_origin;
                let edge_geom_dirty = (cached.edge_width == 0.0) != (edge_width == 0.0);
                let edge_style_dirty =
                    cached.line_color != line_color || cached.edge_width != edge_width;
                if !variable_dirty
                    && !geometry_dirty
                    && !style_dirty
                    && !empty_visibility_dirty
                    && !scene_origin_dirty
                    && !edge_geom_dirty
                    && !edge_style_dirty
                {
                    continue;
                }
                // The shader-hidden set (which also determines volume cell
                // occupancy) is needed by both the volume fast-path decision
                // and the chunk branches below, and walking it is O(blocks) —
                // compute it once, but only when something that can change it
                // is dirty (skip it for edge/scene-origin-only updates).
                let hidden_fingerprint =
                    if variable_dirty || geometry_dirty || style_dirty || empty_visibility_dirty {
                        hidden_blocks_fingerprint(block_model)
                    } else {
                        cached.hidden_fingerprint
                    };
                // (Re)build the volume first. When it renders the model the
                // surface chunks below are dead weight, so `volume_present`
                // lets each path skip building them (see
                // `build_surface_chunks_unless_volume`). A colour-ramp-only
                // change that leaves occupancy unchanged takes the cheap
                // in-place restyle (recompute aggregates + stops uniform)
                // instead of a full rebuild + re-upload of the large buffers —
                // this is what keeps dragging a gradient stop smooth.
                let style_only = style_dirty
                    && !variable_dirty
                    && !geometry_dirty
                    && !empty_visibility_dirty
                    && !scene_origin_dirty;
                if style_only && hidden_fingerprint == cached.hidden_fingerprint {
                    if let Some(volume) = cached.volume.as_mut() {
                        update_block_volume_style(queue, volume, scene_origin, block_model);
                    }
                } else if variable_dirty
                    || geometry_dirty
                    || style_dirty
                    || empty_visibility_dirty
                    || scene_origin_dirty
                {
                    cached.volume = build_block_volume_gpu(
                        device,
                        queue,
                        scene_origin,
                        block_model,
                        translucent,
                        volume_layout,
                    );
                    cached.scene_origin = scene_origin;
                }
                let volume_present = cached.volume.is_some();
                if geometry_dirty {
                    // Translucency changes which pipeline/chunking the model
                    // draws with, so it needs a full rebuild.
                    let (surface_chunks, transparent_surface_chunks) =
                        build_surface_chunks_unless_volume(
                            device,
                            scene_origin,
                            block_model,
                            force_translucent,
                            has_partial_alpha_stops,
                            volume_present,
                        );
                    cached.surface_chunks = surface_chunks;
                    cached.transparent_surface_chunks = transparent_surface_chunks;
                    let style = block_model_style(block_model, force_translucent);
                    queue.write_buffer(&cached.surface_style_buffer, 0, bytemuck::bytes_of(&style));
                    cached.variable = block_model.active_numeric_variable.clone();
                    cached.color_transfer = block_model.color_transfer.clone();
                    cached.hide_empty_color_values = block_model.hide_empty_color_values;
                    cached.hidden_fingerprint = hidden_fingerprint;
                    cached.translucent = translucent;
                    cached.force_translucent = force_translucent;
                } else if variable_dirty || empty_visibility_dirty {
                    if hidden_fingerprint == cached.hidden_fingerprint && !translucent {
                        // The attribute/legend switch hides the same set of
                        // blocks, so which blocks and faces render is
                        // unchanged — only re-colour, without re-walking
                        // blocks or reallocating GPU buffers. (Only reached
                        // when `!translucent`, i.e. `volume_present == false`.)
                        recolor_block_model_surface_chunks(
                            queue,
                            block_model,
                            &mut cached.surface_chunks,
                        );
                        recolor_block_model_surface_chunks(
                            queue,
                            block_model,
                            &mut cached.transparent_surface_chunks,
                        );
                    } else {
                        // The set of shader-hidden blocks or the alpha class
                        // changed: skipped blocks, exposed-face culling, and
                        // opaque/transparent routing are stale.
                        let (surface_chunks, transparent_surface_chunks) =
                            build_surface_chunks_unless_volume(
                                device,
                                scene_origin,
                                block_model,
                                force_translucent,
                                has_partial_alpha_stops,
                                volume_present,
                            );
                        cached.surface_chunks = surface_chunks;
                        cached.transparent_surface_chunks = transparent_surface_chunks;
                        if cached.edge_width > 0.0 {
                            cached.edge_chunks =
                                build_block_model_edge_chunks(device, scene_origin, block_model);
                        }
                        cached.hidden_fingerprint = hidden_fingerprint;
                    }
                    if variable_dirty || style_dirty {
                        let style = block_model_style(block_model, force_translucent);
                        queue.write_buffer(
                            &cached.surface_style_buffer,
                            0,
                            bytemuck::bytes_of(&style),
                        );
                    }
                    cached.variable = block_model.active_numeric_variable.clone();
                    cached.color_transfer = block_model.color_transfer.clone();
                    cached.hide_empty_color_values = block_model.hide_empty_color_values;
                } else if style_dirty {
                    // The colour-transfer function changed. This can change
                    // which blocks are shader-hidden and which blocks are
                    // routed to the depth-writing vs transparent pass, so
                    // rebuild geometry whenever the model has partial-alpha
                    // semantics. Otherwise a style-uniform write is enough.
                    // When a volume renders the model this rebuild collapses
                    // to clearing the (unused) chunks — that is the win that
                    // kills the per-colour-change hitch on large models.
                    if hidden_fingerprint != cached.hidden_fingerprint || translucent {
                        let (surface_chunks, transparent_surface_chunks) =
                            build_surface_chunks_unless_volume(
                                device,
                                scene_origin,
                                block_model,
                                force_translucent,
                                has_partial_alpha_stops,
                                volume_present,
                            );
                        cached.surface_chunks = surface_chunks;
                        cached.transparent_surface_chunks = transparent_surface_chunks;
                        if cached.edge_width > 0.0 {
                            cached.edge_chunks =
                                build_block_model_edge_chunks(device, scene_origin, block_model);
                        }
                        cached.hidden_fingerprint = hidden_fingerprint;
                    }
                    let style = block_model_style(block_model, force_translucent);
                    queue.write_buffer(&cached.surface_style_buffer, 0, bytemuck::bytes_of(&style));
                    cached.color_transfer = block_model.color_transfer.clone();
                }
                if edge_geom_dirty {
                    cached.edge_chunks = if edge_width > 0.0 {
                        build_block_model_edge_chunks(device, scene_origin, block_model)
                    } else {
                        Vec::new()
                    };
                }
                if edge_style_dirty || edge_geom_dirty {
                    let style = EdgeStyleUniform {
                        color: line_color,
                        width: edge_width,
                        _pad: [0.0; 3],
                    };
                    queue.write_buffer(&cached.edge_style_buffer, 0, bytemuck::bytes_of(&style));
                    cached.line_color = line_color;
                    cached.edge_width = edge_width;
                }
            } else {
                // Build the volume first so its surface chunks (never drawn
                // when a volume renders the model) are skipped, and so the
                // "nothing to render" guard below doesn't drop a model that
                // renders purely through the raycaster.
                let volume = build_block_volume_gpu(
                    device,
                    queue,
                    scene_origin,
                    block_model,
                    translucent,
                    volume_layout,
                );
                let (surface_chunks, transparent_surface_chunks) =
                    build_surface_chunks_unless_volume(
                        device,
                        scene_origin,
                        block_model,
                        force_translucent,
                        has_partial_alpha_stops,
                        volume.is_some(),
                    );
                if volume.is_none()
                    && surface_chunks.is_empty()
                    && transparent_surface_chunks.is_empty()
                {
                    continue;
                }
                let surface_style = block_model_style(block_model, force_translucent);
                let surface_style_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Block Model Surface Style Uniform"),
                        contents: bytemuck::bytes_of(&surface_style),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });
                let surface_style_bind_group =
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: surface_style_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: surface_style_buffer.as_entire_binding(),
                        }],
                        label: Some("Block Model Surface Style Bind Group"),
                    });
                let edge_chunks = if edge_width > 0.0 {
                    build_block_model_edge_chunks(device, scene_origin, block_model)
                } else {
                    Vec::new()
                };
                let edge_style = EdgeStyleUniform {
                    color: line_color,
                    width: edge_width,
                    _pad: [0.0; 3],
                };
                let edge_style_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Block Model Edge Style Uniform"),
                        contents: bytemuck::bytes_of(&edge_style),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });
                let edge_style_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: edge_style_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: edge_style_buffer.as_entire_binding(),
                    }],
                    label: Some("Block Model Edge Style Bind Group"),
                });
                self.models.insert(
                    block_model.id,
                    CachedBlockModelGpu {
                        surface_chunks,
                        transparent_surface_chunks,
                        volume,
                        surface_style_buffer,
                        surface_style_bind_group,
                        translucent,
                        force_translucent,
                        edge_chunks,
                        edge_style_buffer,
                        edge_style_bind_group,
                        line_color,
                        edge_width,
                        variable: block_model.active_numeric_variable.clone(),
                        color_transfer: block_model.color_transfer.clone(),
                        hide_empty_color_values: block_model.hide_empty_color_values,
                        hidden_fingerprint: hidden_blocks_fingerprint(block_model),
                        scene_origin,
                    },
                );
            }
        }
    }
}

impl TriangulationGpuCache {
    pub(crate) fn clear(&mut self) {
        self.meshes.clear();
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }

    pub(crate) fn get(&self, id: TriangulationId) -> Option<&CachedTriangulationGpu> {
        self.meshes.get(&id)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sync(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene_origin: DVec3,
        scale_factor: f32,
        triangulations: &[OpenTriangulation],
        editor: &EditorState,
        surface_style_layout: &wgpu::BindGroupLayout,
        edge_style_layout: &wgpu::BindGroupLayout,
    ) {
        let loaded: HashSet<_> = triangulations.iter().map(|tri| tri.id).collect();
        self.meshes.retain(|id, _| loaded.contains(id));
        for triangulation in triangulations {
            let entity = triangulation.entity_id();
            let selected = editor.selected_handles.contains(&entity);
            // Selection is represented by the highlighted wireframe below; it must
            // never replace the topology's configured face colour.
            let mut color = if !selected && editor.tri_hover_handles.contains(&entity) {
                YELLOW_HIGHLIGHT_COLOR
            } else {
                triangulation.color
            };
            if editor.translucent_handles.contains(&entity) {
                make_translucent(&mut color);
            }
            let mut line_color = if selected {
                crate::ui::SELECTION_COLOR_F32
            } else {
                triangulation.line_color
            };
            if editor.translucent_handles.contains(&entity) {
                make_translucent(&mut line_color);
            }
            // Selection always shows edges, even when global wireframes are off.
            let show_edges = selected || editor.topology_wireframes_enabled;
            let edge_width = if show_edges {
                (triangulation.line_weight.unwrap_or(1.0) * scale_factor).max(1.0)
            } else {
                0.0
            };

            if let Some(cached) = self.meshes.get_mut(&triangulation.id) {
                let surface_dirty = cached.color != color;
                // Rebuild edge geometry only when edges flip between present and absent.
                let edge_geom_dirty = (cached.edge_width == 0.0) != (edge_width == 0.0);
                let edge_style_dirty =
                    cached.line_color != line_color || cached.edge_width != edge_width;

                if !surface_dirty && !edge_geom_dirty && !edge_style_dirty {
                    continue;
                }

                if surface_dirty {
                    let style = SurfaceStyleUniform { color };
                    queue.write_buffer(&cached.surface_style_buffer, 0, bytemuck::bytes_of(&style));
                    cached.color = color;
                }

                if edge_geom_dirty && edge_width > 0.0 && cached.edge_chunks.is_empty() {
                    cached.edge_chunks = build_edge_chunks(device, scene_origin, triangulation);
                }

                if edge_style_dirty || edge_geom_dirty {
                    let style = EdgeStyleUniform {
                        color: line_color,
                        width: edge_width,
                        _pad: [0.0; 3],
                    };
                    queue.write_buffer(&cached.edge_style_buffer, 0, bytemuck::bytes_of(&style));
                    cached.line_color = line_color;
                    cached.edge_width = edge_width;
                }
            } else {
                // Not in cache — release any stale handle then do a full build.
                self.meshes.remove(&triangulation.id);

                let surface_chunks = build_surface_chunks(device, scene_origin, triangulation);
                if surface_chunks.is_empty() {
                    continue;
                }

                let surface_style = SurfaceStyleUniform { color };
                let surface_style_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Surface Style Uniform"),
                        contents: bytemuck::bytes_of(&surface_style),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });
                let surface_style_bind_group =
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout: surface_style_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: surface_style_buffer.as_entire_binding(),
                        }],
                        label: Some("Surface Style Bind Group"),
                    });

                let edge_chunks = if edge_width > 0.0 {
                    build_edge_chunks(device, scene_origin, triangulation)
                } else {
                    Vec::new()
                };
                let edge_style = EdgeStyleUniform {
                    color: line_color,
                    width: edge_width,
                    _pad: [0.0; 3],
                };
                let edge_style_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Edge Style Uniform"),
                        contents: bytemuck::bytes_of(&edge_style),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });
                let edge_style_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: edge_style_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: edge_style_buffer.as_entire_binding(),
                    }],
                    label: Some("Edge Style Bind Group"),
                });

                self.meshes.insert(
                    triangulation.id,
                    CachedTriangulationGpu {
                        surface_chunks,
                        surface_style_buffer,
                        surface_style_bind_group,
                        color,
                        edge_chunks,
                        edge_style_buffer,
                        edge_style_bind_group,
                        line_color,
                        edge_width,
                    },
                );
            }
        }
    }
}

/// Upload indexed surface geometry. Most meshes use their source vertex buffer
/// directly; exceptionally large meshes are remapped into independently
/// drawable chunks which each respect the device's maximum buffer size.
/// Colour is NOT baked in — it lives in a per-draw SurfaceStyle uniform.
fn build_surface_chunks(
    device: &wgpu::Device,
    scene_origin: DVec3,
    triangulation: &OpenTriangulation,
) -> Vec<CachedSurfaceChunk> {
    let source = triangulation.mesh.vertices();
    let face_count = triangulation.mesh.face_count();
    if source.is_empty() || face_count == 0 {
        return Vec::new();
    }
    let limit = (device.limits().max_buffer_size as usize).min(MAX_SURFACE_CHUNK_BYTES);
    let max_vertices = (limit / std::mem::size_of::<SurfaceVertex>()).min(u32::MAX as usize);
    let max_indices = (limit / std::mem::size_of::<u32>()).min(u32::MAX as usize);
    let vertex_bytes = source
        .len()
        .checked_mul(std::mem::size_of::<SurfaceVertex>());
    let index_count = face_count.checked_mul(3);
    let index_bytes = index_count.and_then(|count| count.checked_mul(std::mem::size_of::<u32>()));

    if vertex_bytes.is_some_and(|bytes| bytes <= limit)
        && index_bytes.is_some_and(|bytes| bytes <= limit)
        && source.len() <= u32::MAX as usize
    {
        let vertices = source
            .iter()
            .map(|point| surface_vertex(*point, scene_origin))
            .collect::<Vec<_>>();
        // Pre-allocate with the known index count to avoid repeated reallocation.
        let mut indices = Vec::with_capacity(face_count * 3);
        indices.extend(
            triangulation
                .mesh
                .face_vertex_indices_iter()
                .flat_map(|face| face.map(|index| index as u32)),
        );
        return upload_surface_chunk(device, &vertices, &indices)
            .into_iter()
            .collect();
    }

    log::info!(
        "Chunking triangulation '{}' for GPU upload ({} vertices, {} faces, max buffer {} bytes)",
        triangulation.name,
        source.len(),
        face_count,
        limit
    );
    let mut chunks = Vec::new();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    // Dense sentinel array: remap[global_index] == u32::MAX means "not yet in this chunk".
    // O(1) array access replaces HashMap hash-probing for each vertex lookup.
    let mut remap = vec![u32::MAX; source.len()];
    let mut dirty: Vec<usize> = Vec::new();
    for face in triangulation.mesh.face_vertex_indices_iter() {
        let missing = face
            .iter()
            .filter(|&&i| remap.get(i).is_some_and(|&v| v == u32::MAX))
            .count();
        if !indices.is_empty()
            && (indices.len() + 3 > max_indices || vertices.len() + missing > max_vertices)
        {
            if let Some(chunk) = upload_surface_chunk(device, &vertices, &indices) {
                chunks.push(chunk);
            }
            vertices.clear();
            indices.clear();
            // Reset only the entries we touched — avoids O(source.len()) clear.
            for &i in &dirty {
                remap[i] = u32::MAX;
            }
            dirty.clear();
        }
        for global_index in face {
            let local_index = if let Some(&v) = remap.get(global_index)
                && v != u32::MAX
            {
                v
            } else {
                let Some(point) = source.get(global_index) else {
                    continue;
                };
                let index = vertices.len() as u32;
                vertices.push(surface_vertex(*point, scene_origin));
                if global_index < remap.len() {
                    remap[global_index] = index;
                    dirty.push(global_index);
                }
                index
            };
            indices.push(local_index);
        }
    }
    if let Some(chunk) = upload_surface_chunk(device, &vertices, &indices) {
        chunks.push(chunk);
    }
    chunks
}

fn surface_vertex(point: tri00t::Vertex, scene_origin: DVec3) -> SurfaceVertex {
    let local = DVec3::new(point.x, point.y, point.z) - scene_origin;
    SurfaceVertex {
        pos: local.as_vec3().to_array(),
    }
}

fn upload_surface_chunk(
    device: &wgpu::Device,
    vertices: &[SurfaceVertex],
    indices: &[u32],
) -> Option<CachedSurfaceChunk> {
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }
    let limit = device.limits().max_buffer_size;
    let vertex_bytes = std::mem::size_of_val(vertices) as u64;
    let index_bytes = std::mem::size_of_val(indices) as u64;
    if vertex_bytes > limit || index_bytes > limit {
        log::error!(
            "Triangulation GPU chunk rejected before allocation: vertices={vertex_bytes} bytes, indices={index_bytes} bytes, limit={limit} bytes"
        );
        return None;
    }
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cached Triangulation Surface Vertices"),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cached Triangulation Surface Indices"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    Some(CachedSurfaceChunk {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
    })
}

fn build_edge_chunks(
    device: &wgpu::Device,
    scene_origin: DVec3,
    triangulation: &OpenTriangulation,
) -> Vec<CachedEdgeChunk> {
    const TARGET_CHUNK_BYTES: usize = 64 * 1024 * 1024;
    let buffer_limit = (device.limits().max_buffer_size as usize).min(TARGET_CHUNK_BYTES);
    let edges_per_chunk = (buffer_limit / std::mem::size_of::<EdgeInstance>()).max(1);
    let mut chunks = Vec::new();
    let vertices = triangulation.mesh.vertices();
    for source_edges in triangulation.edges.chunks(edges_per_chunk) {
        let instances = source_edges
            .iter()
            .map(|[a, b]| {
                let a = vertices[*a as usize];
                let b = vertices[*b as usize];
                let start = DVec3::new(a.x, a.y, a.z) - scene_origin;
                let end = DVec3::new(b.x, b.y, b.z) - scene_origin;
                EdgeInstance {
                    start: start.as_vec3().to_array(),
                    end: end.as_vec3().to_array(),
                }
            })
            .collect::<Vec<_>>();
        if let Some(chunk) = upload_edge_chunk(device, &instances) {
            chunks.push(chunk);
        }
    }
    chunks
}

fn upload_edge_chunk(device: &wgpu::Device, instances: &[EdgeInstance]) -> Option<CachedEdgeChunk> {
    if instances.is_empty() {
        return None;
    }
    let limit = device.limits().max_buffer_size;
    let instance_bytes = std::mem::size_of_val(instances) as u64;
    if instance_bytes > limit {
        log::error!(
            "Triangulation edge chunk rejected before GPU allocation: instances={instance_bytes} bytes, limit={limit} bytes"
        );
        return None;
    }
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cached Triangulation Edge Instances"),
        contents: bytemuck::cast_slice(instances),
        usage: wgpu::BufferUsages::VERTEX,
    });
    Some(CachedEdgeChunk {
        instance_buffer,
        instance_count: instances.len() as u32,
    })
}

/// Decoded values/default/render-range for the block model's currently
/// active colour variable, if any. Shared by initial geometry build and by
/// the cheap re-colour path so both compute `grade` identically.
type BlockModelColorValues = (std::sync::Arc<Vec<f64>>, Option<f64>, Option<(f64, f64)>);

fn block_model_color_values(block_model: &OpenBlockModel) -> Option<BlockModelColorValues> {
    let name = block_model.active_numeric_variable.as_deref()?;
    let var = block_model.model.variable(name)?;
    let values = block_model.active_numeric_values()?;
    let default = numeric_variable_default(var);
    // Cached alongside the decoded values, so a rebuild doesn't re-scan the
    // whole column just to recover the render range.
    let range = block_model.active_value_range();
    Some((values, default, range))
}

/// Fingerprint of the set of blocks the fragment shader will hide outright.
/// Chunk geometry (skipped blocks, exposed-face culling) depends on this set,
/// so the cheap recolor-in-place path is only valid while it is unchanged.
///
/// XOR of per-index mixes rather than a sequential hasher: it is
/// order-independent, so the O(blocks) scan — which runs on every ramp-drag
/// tick to validate the restyle fast path — parallelizes cleanly. Block
/// indices are unique, so equal hidden sets always fingerprint equal;
/// distinct sets collide with ~2^-64 probability.
fn hidden_blocks_fingerprint(block_model: &OpenBlockModel) -> u64 {
    use rayon::prelude::*;

    // Sebastiano Vigna's SplitMix64 finalizer: a bijective 64-bit mix, so
    // single-index "sets" can't collide and XOR combinations scatter well.
    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    let color_values = block_model_color_values(block_model);
    // Captured by field: `OpenBlockModel` holds a `RefCell` cache and is not
    // `Sync`, but the pieces this scan needs all are.
    let color_transfer = &block_model.color_transfer;
    let hide_empty = block_model.hide_empty_color_values;
    block_model
        .renderable_block_indices
        .par_iter()
        .filter(|&&index| block_is_hidden(&color_values, color_transfer, index, hide_empty))
        .map(|&index| splitmix64(index as u64))
        .reduce(|| 0, |a, b| a ^ b)
}

fn grade_for_block(
    color_values: &Option<BlockModelColorValues>,
    block_index: usize,
    hide_empty: bool,
) -> f32 {
    let Some((values, default, range)) = color_values.as_ref() else {
        return FALLBACK_BLOCK_GRADE;
    };
    let Some(value) = values.get(block_index).copied() else {
        return empty_grade(hide_empty);
    };
    if is_empty_grade_value(value, *default) {
        return empty_grade(hide_empty);
    }
    range
        .map(|range| normalized_grade(value, range))
        .unwrap_or(FALLBACK_BLOCK_GRADE)
}

/// Whether the fragment shader will discard every fragment of the block at
/// `block_index` — combining the hidden-grade sentinel with a colour-ramp
/// alpha below the visibility epsilon. Geometry building treats such blocks as
/// absent, so this must stay in lockstep with `fs_main` in `block_model.wgsl`.
fn block_is_hidden(
    color_values: &Option<BlockModelColorValues>,
    color_transfer: &ColorTransferFunction,
    block_index: usize,
    hide_empty: bool,
) -> bool {
    let grade = grade_for_block(color_values, block_index, hide_empty);
    is_hidden_block_appearance(grade, color_values.is_some(), color_transfer)
}

/// Recomputes and re-uploads `grade` for every already-built chunk of a
/// block model, without re-walking blocks, redoing face culling, or
/// reallocating GPU buffers. Valid whenever chunk geometry (which blocks are
/// rendered, and which faces they have) hasn't changed — i.e. everything
/// except a translucency toggle or the block model's own data changing.
fn recolor_block_model_surface_chunks(
    queue: &wgpu::Queue,
    block_model: &OpenBlockModel,
    chunks: &mut [CachedBlockModelSurfaceChunk],
) {
    let color_values = block_model_color_values(block_model);
    for chunk in chunks {
        for (instance, &block_index) in chunk.cpu_instances.iter_mut().zip(&chunk.block_indices) {
            instance.grade = grade_for_block(
                &color_values,
                block_index,
                block_model.hide_empty_color_values,
            );
        }
        queue.write_buffer(
            &chunk.gpu.instance_buffer,
            0,
            bytemuck::cast_slice(&chunk.cpu_instances),
        );
    }
}

/// Build the opaque/transparent surface-chunk sets for a block model, unless
/// the volume raycaster will render it. When a volume asset is present the
/// surface chunks are never drawn — the opaque pass skips any model with a
/// volume, and the fallback transparent pass only runs when the volume is
/// absent (see `graphics/passes.rs`) — so building them (which walks every
/// block and re-culls faces) is pure waste. Return empty sets in that case.
fn build_surface_chunks_unless_volume(
    device: &wgpu::Device,
    scene_origin: DVec3,
    block_model: &OpenBlockModel,
    force_translucent: bool,
    has_partial_alpha_stops: bool,
    volume_present: bool,
) -> (
    Vec<CachedBlockModelSurfaceChunk>,
    Vec<CachedBlockModelSurfaceChunk>,
) {
    if volume_present {
        (Vec::new(), Vec::new())
    } else {
        build_block_model_surface_chunk_sets(
            device,
            scene_origin,
            block_model,
            force_translucent,
            has_partial_alpha_stops,
        )
    }
}

fn build_block_model_surface_chunk_sets(
    device: &wgpu::Device,
    scene_origin: DVec3,
    block_model: &OpenBlockModel,
    force_translucent: bool,
    has_partial_alpha_stops: bool,
) -> (
    Vec<CachedBlockModelSurfaceChunk>,
    Vec<CachedBlockModelSurfaceChunk>,
) {
    if force_translucent {
        return (
            Vec::new(),
            build_block_model_surface_chunks(
                device,
                scene_origin,
                block_model,
                false,
                BlockSurfaceSelection::All,
            ),
        );
    }
    if has_partial_alpha_stops {
        return (
            build_block_model_surface_chunks(
                device,
                scene_origin,
                block_model,
                true,
                BlockSurfaceSelection::OpaqueOnly,
            ),
            build_block_model_surface_chunks(
                device,
                scene_origin,
                block_model,
                false,
                BlockSurfaceSelection::TransparentOnly,
            ),
        );
    }
    (
        build_block_model_surface_chunks(
            device,
            scene_origin,
            block_model,
            true,
            BlockSurfaceSelection::All,
        ),
        Vec::new(),
    )
}

fn build_block_model_surface_chunks(
    device: &wgpu::Device,
    scene_origin: DVec3,
    block_model: &OpenBlockModel,
    cull_shared_faces: bool,
    selection: BlockSurfaceSelection,
) -> Vec<CachedBlockModelSurfaceChunk> {
    const BLOCKS_PER_CHUNK: usize = 8192;

    let color_values = block_model_color_values(block_model);
    let renderable_blocks = cull_shared_faces.then(|| {
        renderable_block_keys(
            &block_model.blocks,
            &block_model.renderable_block_indices,
            &color_values,
            &block_model.color_transfer,
            block_model.hide_empty_color_values,
            selection,
            block_model.color[3],
        )
    });

    // The shader places blocks with `rotation * local + translation`. Offset
    // local bounds by `local_ref` — the local point that maps to the scene
    // origin — so the f32 instance coordinates stay small (block-extent scale)
    // and keep the precision the old CPU `local_to_world - scene_origin` had.
    // With this reference the translation is exactly zero (see
    // `block_model_style`), so only the rotation is uploaded.
    let rotation = block_model.model.rotation();
    let local_ref = rotation.inverse() * (scene_origin - block_model.model.origin());

    let mut chunks = Vec::new();
    for source_indices in block_model
        .renderable_block_indices
        .chunks(BLOCKS_PER_CHUNK)
    {
        let mut instances = Vec::with_capacity(source_indices.len());
        let mut block_indices = Vec::with_capacity(source_indices.len());
        let mut bounds_min = glam::Vec3::splat(f32::INFINITY);
        let mut bounds_max = glam::Vec3::splat(f32::NEG_INFINITY);
        for &block_index in source_indices {
            let Some(block) = block_model.blocks.get(block_index) else {
                continue;
            };
            let grade = grade_for_block(
                &color_values,
                block_index,
                block_model.hide_empty_color_values,
            );
            if is_hidden_block_appearance(
                grade,
                color_values.is_some(),
                &block_model.color_transfer,
            ) {
                // The shader would discard every fragment; emitting the
                // block would only stale the culling when values change.
                continue;
            }
            if !block_matches_surface_selection(
                grade,
                color_values.is_some(),
                &block_model.color_transfer,
                selection,
                block_model.color[3],
            ) {
                continue;
            }
            // Whole-interior-block cull: when culling shared faces (opaque),
            // a block whose six neighbours are all present-and-drawn shows no
            // face, so it needs no instance. Hidden blocks are already absent
            // from the neighbour set, so this preserves the Phase-1 hole fix.
            if let Some(renderable_blocks) = renderable_blocks.as_ref()
                && visible_block_faces(*block, renderable_blocks)
                    .iter()
                    .all(|visible| !visible)
            {
                continue;
            }
            instances.push(BlockInstance {
                lower: (block.lower - local_ref).as_vec3().to_array(),
                grade,
                upper: (block.upper - local_ref).as_vec3().to_array(),
                _pad: 0.0,
            });
            block_indices.push(block_index);
            // Chunk bounds are the scene-relative world AABB, so keep walking
            // the block's rotated corners here (frustum culling / depth sort).
            for corner in block_corners(block_model, *block) {
                let scene_rel = (corner - scene_origin).as_vec3();
                bounds_min = bounds_min.min(scene_rel);
                bounds_max = bounds_max.max(scene_rel);
            }
        }
        if let Some(gpu) =
            upload_block_model_surface_chunk(device, &instances, bounds_min, bounds_max)
        {
            chunks.push(CachedBlockModelSurfaceChunk {
                gpu,
                cpu_instances: instances,
                block_indices,
            });
        }
    }
    chunks
}

fn block_model_style(block_model: &OpenBlockModel, translucent: bool) -> BlockModelStyleUniform {
    let has_grade = block_model_color_values(block_model).is_some();
    let mut fallback_color = block_model.color;
    if translucent {
        make_translucent(&mut fallback_color);
    } else {
        fallback_color[3] = 1.0;
    }
    let mut stops = [ColorStopUniform {
        color: [0.0; 4],
        pos: [0.0; 4],
    }; MAX_COLOR_STOPS];
    let stop_count = block_model.color_transfer.stops.len().min(MAX_COLOR_STOPS);
    for (slot, stop) in stops.iter_mut().zip(
        block_model
            .color_transfer
            .stops
            .iter()
            .take(MAX_COLOR_STOPS),
    ) {
        slot.color = stop.color;
        slot.pos = [stop.t, 0.0, 0.0, 0.0];
    }
    // Local->scene rotation columns. Translation is zero because
    // `build_block_model_surface_chunks` pre-offsets instance bounds by the
    // local point that maps to the scene origin.
    let rotation = block_model.model.rotation();
    let col = |axis: DVec3| {
        let v = axis.as_vec3();
        [v.x, v.y, v.z, 0.0]
    };
    BlockModelStyleUniform {
        fallback_color,
        options: [
            if has_grade { 1.0 } else { 0.0 },
            stop_count as f32,
            0.0,
            0.0,
        ],
        rotation_0: col(rotation.x_axis),
        rotation_1: col(rotation.y_axis),
        rotation_2: col(rotation.z_axis),
        translation: [0.0; 4],
        stops,
    }
}

/// Off-GPU backing store for occupied bricks' packed cell payloads,
/// ordinal-indexed (`CELLS_PER_BRICK` contiguous `u32`s per brick, empty cells
/// = [`EMPTY_CELL_PAYLOAD`]). Kept off the GPU so the resident subset can be
/// streamed into a bounded pool without holding every payload in VRAM; a large
/// model spills to an mmap'd temp file so the payloads need not sit in
/// committed RAM either — the OS page cache handles CPU-side residency and the
/// working set stays near the streamed subset.
enum CellBacking {
    Ram(Vec<u32>),
    Mapped(MappedCells),
}

/// An mmap'd temp file holding the packed cells. The file is removed on drop;
/// on Unix the mapping outlives the unlink, on Windows the delete lands once
/// the mapping is dropped first (fields drop in declaration order).
struct MappedCells {
    mmap: memmap2::Mmap,
    path: std::path::PathBuf,
}

impl Drop for MappedCells {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

impl CellBacking {
    /// The `CELLS_PER_BRICK` payloads of the brick with this ordinal.
    fn brick_cells(&self, ordinal: u32) -> &[u32] {
        let start = ordinal as usize * CELLS_PER_BRICK;
        let end = start + CELLS_PER_BRICK;
        match self {
            CellBacking::Ram(v) => &v[start..end],
            // The mmap base is page-aligned and `start * 4` is a multiple of 4,
            // so the subslice is `u32`-aligned as `cast_slice` requires.
            CellBacking::Mapped(m) => bytemuck::cast_slice(&m.mmap[start * 4..end * 4]),
        }
    }
}

/// Writer for [`CellBacking`]. RAM-backed under [`MAX_VOLUME_RAM_CELL_BYTES`],
/// otherwise mmap-backed on a temp file. Both start filled with
/// [`EMPTY_CELL_PAYLOAD`] so bricks' interior holes read as empty.
enum CellBackingBuilder {
    Ram(Vec<u32>),
    Mapped {
        file: std::fs::File,
        mmap: memmap2::MmapMut,
        path: std::path::PathBuf,
    },
}

/// Above this many bytes of packed cells, the backing store spills to a temp
/// file instead of RAM. Below it, RAM keeps small/medium models fast.
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

    /// The mmap-backed builder path, factored out so tests can exercise it
    /// without allocating past [`MAX_VOLUME_RAM_CELL_BYTES`].
    fn new_mapped(cell_count: usize) -> Result<Self, String> {
        let bytes = cell_count * std::mem::size_of::<u32>();
        // Spill to a uniquely-named temp file. Any failure (no temp dir, disk
        // full) propagates so the caller can fall back to the cube path.
        let path = std::env::temp_dir().join(format!(
            "proinspector-volume-{}-{}.cells",
            std::process::id(),
            NEXT_BACKING_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| format!("temp cell file {}: {e}", path.display()))?;
        file.set_len(bytes as u64)
            .map_err(|e| format!("sizing temp cell file to {bytes} bytes: {e}"))?;
        // SAFETY: we exclusively own this freshly-created temp file; nothing
        // else maps or truncates it for the lifetime of the mapping.
        let mut mmap = unsafe { memmap2::MmapMut::map_mut(&file) }
            .map_err(|e| format!("mapping temp cell file: {e}"))?;
        let words: &mut [u32] = bytemuck::cast_slice_mut(&mut mmap[..]);
        words.fill(EMPTY_CELL_PAYLOAD);
        Ok(CellBackingBuilder::Mapped { file, mmap, path })
    }

    fn as_mut_slice(&mut self) -> &mut [u32] {
        match self {
            CellBackingBuilder::Ram(v) => v.as_mut_slice(),
            CellBackingBuilder::Mapped { mmap, .. } => bytemuck::cast_slice_mut(&mut mmap[..]),
        }
    }

    fn finish(self) -> Result<CellBacking, String> {
        match self {
            CellBackingBuilder::Ram(v) => Ok(CellBacking::Ram(v)),
            CellBackingBuilder::Mapped { file, mmap, path } => {
                mmap.flush()
                    .map_err(|e| format!("flushing temp cell file: {e}"))?;
                let mmap = mmap
                    .make_read_only()
                    .map_err(|e| format!("sealing temp cell file: {e}"))?;
                drop(file);
                Ok(CellBacking::Mapped(MappedCells { mmap, path }))
            }
        }
    }
}

/// Distinguishes concurrently-open temp backing files within this process.
static NEXT_BACKING_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Sentinel for an unoccupied pool slot in [`BrickStreamer::slot_occupant`].
const NO_OCCUPANT: u32 = u32::MAX;

/// Manages which occupied bricks' cells are resident in the bounded GPU cell
/// pool. Uniform bricks never need residency (their aggregate is exact); the
/// remaining *mixed* bricks compete for `pool_slots` slots by camera
/// proximity. When they all fit, every mixed brick is assigned a slot once and
/// the streamer goes idle; otherwise the nearest ones stay resident and the
/// rest render from their aggregates until they stream in.
///
/// Pure CPU state — no GPU handles — so it is unit-testable in isolation. It
/// emits *plans* (`load`/evict slot assignments and `brick_info` updates) that
/// the GPU layer applies.
struct BrickStreamer {
    pool_slots: u32,
    /// Per-ordinal assigned pool slot, or [`NOT_RESIDENT_SLOT`]. Uniform
    /// bricks are permanently `NOT_RESIDENT_SLOT`.
    ordinal_slot: Vec<u32>,
    /// Per-slot occupying ordinal, or [`NO_OCCUPANT`].
    slot_occupant: Vec<u32>,
    /// Free slot indices available for assignment.
    free_slots: Vec<u32>,
    /// Per-ordinal uniform flag (ramp-dependent; updated on restyle). Uniform
    /// bricks are excluded from streaming.
    uniform: Vec<bool>,
    /// Per-ordinal brick centre in model-local space, for proximity scoring.
    centers: Vec<[f32; 3]>,
    /// Ordinals desired resident but not yet uploaded, nearest-first.
    pending: std::collections::VecDeque<u32>,
    /// `true` when every mixed brick fits the pool: residency is then
    /// camera-independent and computed once.
    fits: bool,
    /// Forces a replan regardless of camera movement (first frame / restyle).
    dirty: bool,
    /// Camera-local position at the last replan, to gate replans on movement.
    last_camera: Option<Vec3>,
    /// Movement (model-local units) beyond which a replan is warranted.
    replan_distance: f32,
    /// Scratch reused by `replan` to mark the desired set without allocating.
    desired_scratch: Vec<bool>,
}

/// One streaming step's GPU work: pool-slot uploads (`slot`, `ordinal`) and
/// `brick_info` entry rewrites (`ordinal`, packed value).
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
            pending: std::collections::VecDeque::new(),
            fits: streamable <= pool_slots as usize,
            dirty: true,
            last_camera: None,
            replan_distance,
            desired_scratch: vec![false; occupied],
        }
    }

    /// The `brick_info` value for an ordinal from current residency.
    fn info(&self, ordinal: u32) -> u32 {
        let flag = if self.uniform[ordinal as usize] {
            UNIFORM_BRICK_FLAG
        } else {
            0
        };
        flag | self.ordinal_slot[ordinal as usize]
    }

    /// The full `brick_info` array (for the initial upload / a restyle).
    fn all_info(&self) -> Vec<u32> {
        (0..self.ordinal_slot.len() as u32)
            .map(|o| self.info(o))
            .collect()
    }

    /// React to a ramp change that may have flipped some bricks' uniformity.
    /// Newly-uniform bricks are evicted (their slots freed); the next
    /// `plan` re-selects residency. Returns nothing — `plan` emits the GPU
    /// updates — but forces a replan.
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

    /// Recompute the desired resident set for the current camera, evicting
    /// bricks that fell out of it and queueing the ones that entered it.
    /// Returns the `brick_info` rewrites from evictions; the queued loads are
    /// applied incrementally by [`Self::drain`].
    fn replan(&mut self, camera_local: Vec3) -> Vec<(u32, u32)> {
        self.dirty = false;
        self.last_camera = Some(camera_local);
        self.pending.clear();
        let occupied = self.ordinal_slot.len();

        // Desired set: all mixed bricks when they fit, else the nearest
        // `pool_slots` of them.
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

        // Evict resident bricks no longer desired.
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

        // Queue desired-but-not-resident bricks, nearest first (already the
        // order from the partial sort in the streaming case; for the fits case
        // order is arbitrary, which is fine).
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

    /// Assign up to `max_uploads` queued bricks to free slots, returning the
    /// pool uploads and `brick_info` rewrites to apply. Draining is bounded so
    /// a big camera jump spreads over several frames.
    fn drain(&mut self, max_uploads: usize) -> StreamPlan {
        let mut plan = StreamPlan::default();
        while plan.uploads.len() < max_uploads {
            let Some(ordinal) = self.pending.pop_front() else {
                break;
            };
            // Skip if it became resident or uniform since being queued.
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

    /// Whether `drain` still has queued work.
    fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }
}

struct BlockVolumeAsset {
    x_planes: Vec<f32>,
    y_planes: Vec<f32>,
    z_planes: Vec<f32>,
    /// Off-GPU backing of occupied bricks' cells, ordinal-indexed.
    cells: CellBacking,
    /// brick index -> ordinal (`0..occupied_count`), or [`EMPTY_BRICK`].
    brick_table: Vec<u32>,
    /// One LOD aggregate per *occupied* brick, indexed by ordinal: rgb =
    /// extinction-weighted mean ramp colour, w = volume-weighted mean
    /// extinction (sigma). The march integrates whole bricks from these when
    /// cells project sub-pixel, when the brick is uniform (exact), or when its
    /// cells are not resident in the pool.
    brick_aggregates: Vec<[f32; 4]>,
    /// One uniformity flag per *occupied* brick (by ordinal): every real cell
    /// resolves to one appearance, so the aggregate reproduces per-cell output
    /// exactly and the brick needs no pool residency. See
    /// [`compute_brick_style_data`].
    brick_uniform: Vec<bool>,
    /// One model-local brick centre per *occupied* brick (by ordinal), for
    /// streaming proximity scoring. Ramp-independent.
    brick_centers: Vec<[f32; 3]>,
    occupied_count: usize,
    /// Number of cells along each axis.
    dims: [u32; 3],
    /// Number of bricks along each axis (`brick_grid_dims(dims)`).
    brick_dims: [u32; 3],
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    reference_len: f32,
}

fn build_block_volume_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene_origin: DVec3,
    block_model: &OpenBlockModel,
    translucent: bool,
    layout: &wgpu::BindGroupLayout,
) -> Option<CachedBlockVolumeGpu> {
    if !translucent {
        return None;
    }

    let asset = match build_block_volume_asset(block_model) {
        Ok(Some(asset)) => asset,
        Ok(None) => return None,
        Err(message) => {
            log::warn!(
                "Block model '{}' volume raycast cache rejected: {message}; falling back to cube transparency",
                block_model.name
            );
            return None;
        }
    };

    let storage_limit = device.limits().max_storage_buffer_binding_size;
    let max_buffer = device.limits().max_buffer_size;

    // The cell backing (possibly many GB) never goes to the GPU whole; only a
    // bounded pool does. Size it to hold every mixed brick if that fits the
    // budget, else as many as the budget allows — the rest render from their
    // aggregates and stream in by proximity.
    let pool_budget_bytes =
        ((storage_limit.min(max_buffer) as f64) * VOLUME_POOL_BUDGET_FRACTION) as u64;
    let slot_bytes = (CELLS_PER_BRICK * std::mem::size_of::<u32>()) as u64;
    let budget_slots = (pool_budget_bytes / slot_bytes).max(1);
    let mixed_count = asset.brick_uniform.iter().filter(|&&u| !u).count() as u64;
    // Uniform-only models need no pool at all; keep one slot so the buffer is
    // never zero-sized.
    let pool_slots = mixed_count.clamp(1, budget_slots) as u32;

    // The always-resident metadata buffers must each fit a storage binding.
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

    // Seed residency. In the fits case this assigns every mixed brick a
    // permanent slot (camera-independent); in the streaming case it fills the
    // pool with a first proximity set around the model centre, corrected each
    // frame by `stream_volume_bricks`.
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
    // Upload the seed residency's cells into their pool slots.
    for &(slot, ordinal) in &seed.uploads {
        queue.write_buffer(
            &cell_pool_buffer,
            slot as u64 * slot_bytes,
            bytemuck::cast_slice(asset.cells.brick_cells(ordinal)),
        );
    }
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

/// Distance (model-local units) the camera must move to warrant recomputing
/// streaming residency: a fraction of the model's diagonal, so it scales with
/// model size.
fn streamer_replan_distance(asset: &BlockVolumeAsset) -> f32 {
    let extent = Vec3::from_array(asset.bounds_max) - Vec3::from_array(asset.bounds_min);
    (extent.length() * 0.05).max(1.0e-3)
}

/// Per-frame streaming step: for a volume that needs it, replan residency
/// against the camera (in model-local space) if it moved enough, then upload a
/// bounded batch of newly-resident bricks' cells into the pool and patch their
/// `brick_info` slots. A no-op for volumes whose mixed bricks all fit the pool
/// once nothing is pending.
fn stream_volume_bricks(
    queue: &wgpu::Queue,
    volume: &mut CachedBlockVolumeGpu,
    camera_local: Vec3,
) {
    if volume.streamer.needs_replan(camera_local) {
        let evictions = volume.streamer.replan(camera_local);
        for (ordinal, info) in evictions {
            queue.write_buffer(
                &volume.brick_info_buffer,
                ordinal as u64 * std::mem::size_of::<u32>() as u64,
                bytemuck::bytes_of(&info),
            );
        }
    }
    if !volume.streamer.has_pending() {
        return;
    }
    let plan = volume.streamer.drain(VOLUME_MAX_UPLOADS_PER_UPDATE);
    let slot_bytes = (CELLS_PER_BRICK * std::mem::size_of::<u32>()) as u64;
    for (slot, ordinal) in plan.uploads {
        queue.write_buffer(
            &volume.cell_pool_buffer,
            slot as u64 * slot_bytes,
            bytemuck::cast_slice(volume.asset.cells.brick_cells(ordinal)),
        );
    }
    for (ordinal, info) in plan.info_updates {
        queue.write_buffer(
            &volume.brick_info_buffer,
            ordinal as u64 * std::mem::size_of::<u32>() as u64,
            bytemuck::bytes_of(&info),
        );
    }
}

/// In-place update of a cached volume for a colour-ramp change that leaves
/// cell occupancy unchanged (verified by the caller via
/// [`hidden_blocks_fingerprint`]). Only the ramp-dependent data — the LOD
/// aggregates and the stops/fallback uniform — is recomputed and re-uploaded;
/// the planes, cells and brick table (the large buffers) are untouched. This
/// is what keeps dragging a colour-gradient stop cheap on large models.
fn update_block_volume_style(
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
    // Feed the new uniformity to the streamer (evicting bricks that became
    // uniform and freeing their slots), then re-upload the whole `brick_info`
    // — uniform flags may have flipped for many bricks. Bricks that became
    // mixed are non-resident until the next frame's streaming pass loads them
    // (they render from their aggregates meanwhile).
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
    }; MAX_COLOR_STOPS];
    let stop_count = block_model
        .color_transfer
        .stops
        .len()
        .clamp(2, MAX_COLOR_STOPS);
    for (slot, stop) in stops.iter_mut().zip(
        block_model
            .color_transfer
            .stops
            .iter()
            .take(MAX_COLOR_STOPS),
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

/// Affine rows mapping a scene-relative position to model-local space, shared
/// by the shader uniform and the CPU streaming proximity. `local = [dot(row.xyz,
/// scene) + row.w]` per axis.
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

/// Apply [`scene_to_local_rows`] to a scene-relative point.
fn apply_scene_to_local(rows: &[[f32; 4]; 3], scene: Vec3) -> Vec3 {
    let axis = |r: &[f32; 4]| r[0] * scene.x + r[1] * scene.y + r[2] * scene.z + r[3];
    Vec3::new(axis(&rows[0]), axis(&rows[1]), axis(&rows[2]))
}

fn build_block_volume_asset(
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
    let color_values = block_model_color_values(block_model);

    // Pass 1: resolve each renderable block to a compressed cell range and
    // payload, and mark every brick it touches as occupied. We keep the
    // resolved ranges so pass 2 need not recompute plane indices.
    struct PlacedBlock {
        payload: u32,
        range: [[usize; 2]; 3],
    }
    let mut placed: Vec<PlacedBlock> = Vec::new();
    let mut brick_occupied = vec![false; brick_count];
    for &block_index in &block_model.renderable_block_indices {
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
    // The backing store (mmap'd for large models) no longer has to fit in RAM
    // or VRAM — only the streamed subset does. This ceiling guards against a
    // pathological grid or filling the disk, not the render budget.
    if sparse_cell_bytes > MAX_VOLUME_CELL_BYTES {
        return Err(format!(
            "sparse brick volume would require {} MiB of cell payloads",
            sparse_cell_bytes / (1024 * 1024)
        ));
    }

    // Assign occupied bricks sequential ordinals (0..occupied_count); empty
    // bricks stay `EMPTY_BRICK` and cost no cell storage. The ordinal indexes
    // the aggregates, uniformity/residency (`brick_info`) and the cell backing
    // (brick n at `cells[n * CELLS_PER_BRICK ..]`).
    let mut brick_table = vec![EMPTY_BRICK; brick_count];
    let mut brick_centers = vec![[0.0f32; 3]; occupied_count];
    let mut next_ordinal = 0u32;
    for (bindex, &occ) in brick_occupied.iter().enumerate() {
        if !occ {
            continue;
        }
        brick_table[bindex] = next_ordinal;
        let bi = bindex % brick_dims_usize[0];
        let bj = (bindex / brick_dims_usize[0]) % brick_dims_usize[1];
        let bk = bindex / (brick_dims_usize[0] * brick_dims_usize[1]);
        // Brick centre in model-local space (plane midpoint of the brick's cell
        // span), clamped to the grid on edge bricks. Feeds streaming proximity.
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

    // Pass 2: write each block's payload into its bricks' cell slots in the
    // backing store (an mmap'd temp file for large models).
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
        // Ramp-dependent; filled below and recomputed in place on a ramp-only
        // restyle (see `update_block_volume_style`).
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

/// Resolved appearance per quantized u16 grade (plus the fallback payload):
/// straight RGBA and its precomputed extinction. Payloads store grades
/// already quantized to 16 bits, so one table lookup replaces a per-cell
/// `ramp_rgba` scan — this is what keeps the per-cell brick passes cheap
/// while a colour stop is being dragged.
struct VolumeRampLut {
    /// `(rgba, sigma)` for grade `g / 65535.0`, `g` the table index.
    entries: Vec<([f32; 4], f32)>,
    /// `(rgba, sigma)` for `FALLBACK_CELL_FLAG` payloads.
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

    /// The appearance a cell payload resolves to, or `None` for empty cells.
    /// Must match `color_for_payload` in `block_model_volume.wgsl` exactly.
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

/// Ramp-dependent per-occupied-brick data, indexed by
/// `brick_table[b] / BRICK_SIZE^3` and recomputed on every ramp change.
///
/// `aggregates`: rgb = extinction-weighted mean ramp colour, w =
/// volume-weighted mean extinction (sigma), used by the shader when it
/// integrates a whole brick (coarse LOD, uniform, or not resident).
/// `uniform_flags`: `true` when every real cell in the brick resolves to one
/// appearance (a single RGBA, or invisible — empty and sub-epsilon alpha
/// alike), so the aggregate reproduces per-cell output exactly and the brick
/// needs no pool residency. The resolution here must match `color_for_payload`
/// + `VISIBLE_ALPHA_EPSILON` in the shader.
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

    // Occupied bricks paired with their ordinal (= their brick_table value and
    // their index in the aggregate/uniform outputs), in ordinal order.
    let mut occupied: Vec<(usize, u32)> = asset
        .brick_table
        .iter()
        .enumerate()
        .filter(|&(_, &ordinal)| ordinal != EMPTY_BRICK)
        .map(|(bindex, &ordinal)| (bindex, ordinal))
        .collect();
    occupied.sort_unstable_by_key(|&(_, ordinal)| ordinal);

    let per_brick: Vec<([f32; 4], bool)> = occupied
        .par_iter()
        .map(|&(bindex, ordinal)| {
            let bi = bindex % brick_dims[0];
            let bj = (bindex / brick_dims[0]) % brick_dims[1];
            let bk = bindex / (brick_dims[0] * brick_dims[1]);
            let brick_cells = asset.cells.brick_cells(ordinal);
            let mut volume_sum = 0.0f64;
            let mut sigma_volume_sum = 0.0f64;
            let mut rgb_sum = [0.0f64; 3];
            // Appearance of the brick's first real cell; `uniform` stays true
            // while every other cell resolves identically (invisible cells —
            // empty or sub-epsilon — all count as the `None` appearance).
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
    for &index in &block_model.renderable_block_indices {
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

/// Number of macro-bricks along each axis needed to cover `dims` cells.
fn brick_grid_dims(dims: [usize; 3]) -> [usize; 3] {
    [
        dims[0].div_ceil(BRICK_SIZE),
        dims[1].div_ceil(BRICK_SIZE),
        dims[2].div_ceil(BRICK_SIZE),
    ]
}

/// Linear index of a brick within the dense `brick_table`.
fn brick_index(brick_dims: [usize; 3], bi: usize, bj: usize, bk: usize) -> usize {
    (bk * brick_dims[1] + bj) * brick_dims[0] + bi
}

/// Offset of a cell within its brick's `BRICK_SIZE^3` packed payload block.
fn brick_local_index(li: usize, lj: usize, lk: usize) -> usize {
    (lk * BRICK_SIZE + lj) * BRICK_SIZE + li
}

/// Sparse cell lookup into the ordinal-packed backing: resolve `(i, j, k)`
/// through the brick table (which stores ordinals) to a payload, returning
/// [`EMPTY_CELL_PAYLOAD`] for cells in an empty brick. Validates the CPU
/// packing; the shader instead indirects the ordinal through `brick_info` to a
/// pool slot.
#[cfg_attr(not(test), allow(dead_code))]
fn sparse_cell_payload(
    brick_dims: [usize; 3],
    brick_table: &[u32],
    cells: &[u32],
    i: usize,
    j: usize,
    k: usize,
) -> u32 {
    let bindex = brick_index(brick_dims, i / BRICK_SIZE, j / BRICK_SIZE, k / BRICK_SIZE);
    let ordinal = brick_table[bindex];
    if ordinal == EMPTY_BRICK {
        return EMPTY_CELL_PAYLOAD;
    }
    let local = brick_local_index(i % BRICK_SIZE, j % BRICK_SIZE, k % BRICK_SIZE);
    cells[ordinal as usize * CELLS_PER_BRICK + local]
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

fn block_matches_surface_selection(
    grade: f32,
    has_grade: bool,
    color_transfer: &ColorTransferFunction,
    selection: BlockSurfaceSelection,
    fallback_alpha: f32,
) -> bool {
    match selection {
        BlockSurfaceSelection::All => true,
        BlockSurfaceSelection::OpaqueOnly => {
            block_alpha(grade, has_grade, color_transfer, fallback_alpha) >= 0.98
        }
        BlockSurfaceSelection::TransparentOnly => {
            block_alpha(grade, has_grade, color_transfer, fallback_alpha) < 0.98
        }
    }
}

/// Whether any of the block model's colour-transfer stops has partial
/// (neither fully transparent nor fully opaque) alpha. Such a model must be
/// drawn through the translucent pipeline for grade colouring to blend
/// correctly, even if the user hasn't toggled translucency manually.
fn block_model_has_partial_alpha_stops(block_model: &OpenBlockModel) -> bool {
    if block_model.active_numeric_variable.is_none() {
        return false;
    }
    block_model
        .color_transfer
        .stops
        .iter()
        .any(|stop| stop.color[3] > 0.02 && stop.color[3] < 0.98)
}

fn upload_block_model_surface_chunk(
    device: &wgpu::Device,
    instances: &[BlockInstance],
    bounds_min: glam::Vec3,
    bounds_max: glam::Vec3,
) -> Option<CachedBlockModelChunkGpu> {
    if instances.is_empty() {
        return None;
    }
    let limit = device.limits().max_buffer_size;
    let instance_bytes = std::mem::size_of_val(instances) as u64;
    if instance_bytes > limit {
        log::error!(
            "Block model surface chunk rejected before GPU allocation: instances={instance_bytes} bytes, limit={limit} bytes"
        );
        return None;
    }
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cached Block Model Instances"),
        contents: bytemuck::cast_slice(instances),
        // COPY_DST is required here: `recolor_block_model_surface_chunks`
        // patches `grade` in place with `queue.write_buffer` on a legend/
        // attribute switch, instead of reallocating this buffer.
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });
    Some(CachedBlockModelChunkGpu {
        instance_buffer,
        instance_count: instances.len() as u32,
        bounds_min,
        bounds_max,
    })
}

/// Keys of the blocks that will actually be drawn: renderable geometry minus
/// blocks the fragment shader hides outright. Only drawn blocks may cull a
/// neighbour's shared face — a hidden block leaves a see-through hole, so the
/// face behind it must exist. This mirrors the shader's discard exactly,
/// including blocks made invisible by a fully-transparent colour-ramp stop, so
/// changing the ramp forces a geometry rebuild (correctness over speed).
fn renderable_block_keys(
    blocks: &[crate::model::block_model::BlockBounds],
    renderable_block_indices: &[usize],
    color_values: &Option<BlockModelColorValues>,
    color_transfer: &ColorTransferFunction,
    hide_empty: bool,
    selection: BlockSurfaceSelection,
    fallback_alpha: f32,
) -> HashSet<[u64; 6]> {
    renderable_block_indices
        .iter()
        .filter(|&&index| {
            let grade = grade_for_block(color_values, index, hide_empty);
            !is_hidden_block_appearance(grade, color_values.is_some(), color_transfer)
                && block_matches_surface_selection(
                    grade,
                    color_values.is_some(),
                    color_transfer,
                    selection,
                    fallback_alpha,
                )
        })
        .filter_map(|&index| blocks.get(index).copied())
        .map(block_key)
        .collect()
}

fn visible_block_faces(
    block: crate::model::block_model::BlockBounds,
    renderable_blocks: &HashSet<[u64; 6]>,
) -> [bool; 6] {
    let size = block.upper - block.lower;
    let neighbours = [
        DVec3::new(-size.x, 0.0, 0.0),
        DVec3::new(size.x, 0.0, 0.0),
        DVec3::new(0.0, -size.y, 0.0),
        DVec3::new(0.0, size.y, 0.0),
        DVec3::new(0.0, 0.0, -size.z),
        DVec3::new(0.0, 0.0, size.z),
    ];
    neighbours.map(|delta| {
        let neighbour = block_key(crate::model::block_model::BlockBounds {
            lower: block.lower + delta,
            upper: block.upper + delta,
        });
        !renderable_blocks.contains(&neighbour)
    })
}

fn block_key(block: crate::model::block_model::BlockBounds) -> [u64; 6] {
    [
        quantize(block.lower.x),
        quantize(block.lower.y),
        quantize(block.lower.z),
        quantize(block.upper.x),
        quantize(block.upper.y),
        quantize(block.upper.z),
    ]
}

fn quantize(value: f64) -> u64 {
    (value * 1_000_000.0).round().to_bits()
}

fn is_empty_grade_value(value: f64, default: Option<f64>) -> bool {
    !value.is_finite()
        || default.is_some_and(|default| (value - default).abs() < 1e-8)
        || value <= -90.0
}

fn empty_grade(hide_empty: bool) -> f32 {
    if hide_empty {
        HIDDEN_BLOCK_GRADE
    } else {
        FALLBACK_BLOCK_GRADE
    }
}

fn normalized_grade(value: f64, range: (f64, f64)) -> f32 {
    ((value - range.0) / (range.1 - range.0)).clamp(0.0, 1.0) as f32
}

fn build_block_model_edge_chunks(
    device: &wgpu::Device,
    scene_origin: DVec3,
    block_model: &OpenBlockModel,
) -> Vec<CachedEdgeChunk> {
    const EDGES_PER_CHUNK: usize = 64 * 1024;
    let color_values = block_model_color_values(block_model);
    let mut chunks = Vec::new();
    let mut instances = Vec::new();
    for &block_index in &block_model.renderable_block_indices {
        let Some(block) = block_model.blocks.get(block_index) else {
            continue;
        };
        // Shader-hidden blocks draw no faces; skip their outlines too.
        if block_is_hidden(
            &color_values,
            &block_model.color_transfer,
            block_index,
            block_model.hide_empty_color_values,
        ) {
            continue;
        }
        let corners = block_corners(block_model, *block);
        for [a, b] in [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ] {
            instances.push(EdgeInstance {
                start: (corners[a] - scene_origin).as_vec3().to_array(),
                end: (corners[b] - scene_origin).as_vec3().to_array(),
            });
            if instances.len() >= EDGES_PER_CHUNK {
                if let Some(chunk) = upload_edge_chunk(device, &instances) {
                    chunks.push(chunk);
                }
                instances.clear();
            }
        }
    }
    if let Some(chunk) = upload_edge_chunk(device, &instances) {
        chunks.push(chunk);
    }
    chunks
}

fn block_corners(
    block_model: &OpenBlockModel,
    block: crate::model::block_model::BlockBounds,
) -> [DVec3; 8] {
    let lo = block.lower;
    let hi = block.upper;
    [
        block_model
            .model
            .local_to_world(DVec3::new(lo.x, lo.y, lo.z)),
        block_model
            .model
            .local_to_world(DVec3::new(hi.x, lo.y, lo.z)),
        block_model
            .model
            .local_to_world(DVec3::new(hi.x, hi.y, lo.z)),
        block_model
            .model
            .local_to_world(DVec3::new(lo.x, hi.y, lo.z)),
        block_model
            .model
            .local_to_world(DVec3::new(lo.x, lo.y, hi.z)),
        block_model
            .model
            .local_to_world(DVec3::new(hi.x, lo.y, hi.z)),
        block_model
            .model
            .local_to_world(DVec3::new(hi.x, hi.y, hi.z)),
        block_model
            .model
            .local_to_world(DVec3::new(lo.x, hi.y, hi.z)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grade_for_block_is_neutral_with_no_active_variable() {
        assert_eq!(grade_for_block(&None, 0, true), FALLBACK_BLOCK_GRADE);
    }

    #[test]
    fn grade_for_block_hides_an_out_of_range_index_when_enabled() {
        let color_values = Some((std::sync::Arc::new(vec![1.0, 2.0]), None, Some((0.0, 2.0))));
        assert_eq!(grade_for_block(&color_values, 5, true), HIDDEN_BLOCK_GRADE);
    }

    #[test]
    fn grade_for_block_uses_fallback_for_empty_values_when_hide_disabled() {
        let color_values = Some((
            std::sync::Arc::new(vec![-99.0]),
            Some(-99.0),
            Some((0.0, 2.0)),
        ));
        assert_eq!(
            grade_for_block(&color_values, 0, false),
            FALLBACK_BLOCK_GRADE
        );
    }

    #[test]
    fn grade_for_block_hides_empty_values_when_enabled() {
        let color_values = Some((
            std::sync::Arc::new(vec![-99.0]),
            Some(-99.0),
            Some((0.0, 2.0)),
        ));
        assert_eq!(grade_for_block(&color_values, 0, true), HIDDEN_BLOCK_GRADE);
    }

    #[test]
    fn grade_for_block_normalizes_within_the_render_range() {
        let color_values = Some((
            std::sync::Arc::new(vec![0.0, 5.0, 10.0]),
            None,
            Some((0.0, 10.0)),
        ));
        assert_eq!(grade_for_block(&color_values, 1, true), 0.5);
    }

    #[test]
    fn hidden_neighbour_does_not_cull_the_shared_face() {
        let a = crate::model::block_model::BlockBounds {
            lower: DVec3::ZERO,
            upper: DVec3::ONE,
        };
        let b = crate::model::block_model::BlockBounds {
            lower: DVec3::new(1.0, 0.0, 0.0),
            upper: DVec3::new(2.0, 1.0, 1.0),
        };
        let blocks = [a, b];
        let indices = [0usize, 1];
        // Block 1's value is the empty sentinel; with hide-empty on the
        // shader discards it, so it must not suppress block 0's +X face.
        let color_values = Some((
            std::sync::Arc::new(vec![1.0, -99.0]),
            Some(-99.0),
            Some((0.0, 2.0)),
        ));

        let ramp = opaque_ramp();
        let keys = renderable_block_keys(
            &blocks,
            &indices,
            &color_values,
            &ramp,
            true,
            BlockSurfaceSelection::All,
            1.0,
        );
        assert_eq!(visible_block_faces(a, &keys), [true; 6]);

        // With hide-empty off the block draws (fallback colour), so the
        // shared face culls as before.
        let keys = renderable_block_keys(
            &blocks,
            &indices,
            &color_values,
            &ramp,
            false,
            BlockSurfaceSelection::All,
            1.0,
        );
        let faces = visible_block_faces(a, &keys);
        assert!(!faces[1], "+X face should be culled by a drawn neighbour");
    }

    /// A fully-opaque two-stop ramp, so `ramp_alpha` never hides a block.
    fn opaque_ramp() -> ColorTransferFunction {
        use crate::model::block_model::ColorStop;
        ColorTransferFunction {
            stops: vec![
                ColorStop {
                    id: 1,
                    t: 0.0,
                    color: [1.0, 0.0, 0.0, 1.0],
                },
                ColorStop {
                    id: 2,
                    t: 1.0,
                    color: [0.0, 0.0, 1.0, 1.0],
                },
            ],
        }
    }

    #[test]
    fn ramp_alpha_matches_the_shader_hard_cutoffs() {
        use crate::model::block_model::ColorStop;
        // Transparent low end, opaque middle, transparent high end.
        let ramp = ColorTransferFunction {
            stops: vec![
                ColorStop {
                    id: 1,
                    t: 0.0,
                    color: [0.0, 0.0, 0.0, 0.0],
                },
                ColorStop {
                    id: 2,
                    t: 0.5,
                    color: [1.0, 1.0, 1.0, 1.0],
                },
                ColorStop {
                    id: 3,
                    t: 1.0,
                    color: [0.0, 0.0, 0.0, 0.0],
                },
            ],
        };
        // Endpoints clamp to the first/last stop alpha (both transparent).
        assert_eq!(ramp_alpha(&ramp, 0.0), 0.0);
        assert_eq!(ramp_alpha(&ramp, 1.0), 0.0);
        // Stops are hard cutoffs: the first stop remains active until the
        // middle stop, then the middle stop remains active until the last.
        assert_eq!(ramp_alpha(&ramp, 0.25), 0.0);
        assert_eq!(ramp_alpha(&ramp, 0.5), 1.0);
        assert_eq!(ramp_alpha(&ramp, 0.75), 1.0);
    }

    #[test]
    fn first_ramp_stop_is_a_lower_visibility_cutoff() {
        use crate::model::block_model::ColorStop;
        let ramp = ColorTransferFunction {
            stops: vec![
                ColorStop {
                    id: 1,
                    t: 0.25,
                    color: [1.0, 0.0, 0.0, 0.8],
                },
                ColorStop {
                    id: 2,
                    t: 0.50,
                    color: [0.0, 1.0, 0.0, 0.4],
                },
                ColorStop {
                    id: 3,
                    t: 0.75,
                    color: [0.0, 0.0, 1.0, 1.0],
                },
            ],
        };

        assert_eq!(ramp_alpha(&ramp, 0.24), 0.0);
        assert_eq!(ramp_alpha(&ramp, 0.25), 0.8);
        assert_eq!(ramp_alpha(&ramp, 0.49), 0.8);
        assert_eq!(ramp_alpha(&ramp, 0.50), 0.4);
        assert_eq!(ramp_alpha(&ramp, 0.74), 0.4);
        assert_eq!(ramp_alpha(&ramp, 0.75), 1.0);
    }

    #[test]
    fn transparent_ramp_stop_hides_the_block_and_exposes_the_shared_face() {
        let a = crate::model::block_model::BlockBounds {
            lower: DVec3::ZERO,
            upper: DVec3::ONE,
        };
        let b = crate::model::block_model::BlockBounds {
            lower: DVec3::new(1.0, 0.0, 0.0),
            upper: DVec3::new(2.0, 1.0, 1.0),
        };
        let blocks = [a, b];
        let indices = [0usize, 1];
        // Both blocks have in-range values, so grade-only culling would draw
        // block 1 and hide block 0's +X face. Block 1's grade is 1.0, which a
        // fully-transparent top stop discards in the shader — so it must be
        // treated as absent and expose the shared face.
        let color_values = Some((std::sync::Arc::new(vec![0.0, 2.0]), None, Some((0.0, 2.0))));
        use crate::model::block_model::ColorStop;
        let ramp = ColorTransferFunction {
            stops: vec![
                ColorStop {
                    id: 1,
                    t: 0.0,
                    color: [1.0, 0.0, 0.0, 1.0],
                },
                ColorStop {
                    id: 2,
                    t: 1.0,
                    color: [0.0, 0.0, 1.0, 0.0],
                },
            ],
        };

        let keys = renderable_block_keys(
            &blocks,
            &indices,
            &color_values,
            &ramp,
            false,
            BlockSurfaceSelection::All,
            1.0,
        );
        assert_eq!(
            visible_block_faces(a, &keys),
            [true; 6],
            "a ramp-transparent neighbour must not cull the shared face"
        );

        // The same geometry with an opaque ramp keeps block 1, so the shared
        // face culls — confirming the transparency is what changed it.
        let keys = renderable_block_keys(
            &blocks,
            &indices,
            &color_values,
            &opaque_ramp(),
            false,
            BlockSurfaceSelection::All,
            1.0,
        );
        assert!(
            !visible_block_faces(a, &keys)[1],
            "+X face should be culled by an opaque neighbour"
        );
    }

    #[test]
    fn mixed_alpha_ramp_keeps_opaque_blocks_in_the_opaque_set_only() {
        use crate::model::block_model::ColorStop;
        let blocks = [
            crate::model::block_model::BlockBounds {
                lower: DVec3::ZERO,
                upper: DVec3::ONE,
            },
            crate::model::block_model::BlockBounds {
                lower: DVec3::new(1.0, 0.0, 0.0),
                upper: DVec3::new(2.0, 1.0, 1.0),
            },
        ];
        let indices = [0usize, 1];
        let color_values = Some((std::sync::Arc::new(vec![0.0, 2.0]), None, Some((0.0, 2.0))));
        let ramp = ColorTransferFunction {
            stops: vec![
                ColorStop {
                    id: 1,
                    t: 0.0,
                    color: [1.0, 0.0, 0.0, 1.0],
                },
                ColorStop {
                    id: 2,
                    t: 1.0,
                    color: [0.0, 1.0, 0.0, 0.35],
                },
            ],
        };

        let opaque_keys = renderable_block_keys(
            &blocks,
            &indices,
            &color_values,
            &ramp,
            false,
            BlockSurfaceSelection::OpaqueOnly,
            1.0,
        );
        assert!(opaque_keys.contains(&block_key(blocks[0])));
        assert!(!opaque_keys.contains(&block_key(blocks[1])));

        let transparent_keys = renderable_block_keys(
            &blocks,
            &indices,
            &color_values,
            &ramp,
            false,
            BlockSurfaceSelection::TransparentOnly,
            1.0,
        );
        assert!(!transparent_keys.contains(&block_key(blocks[0])));
        assert!(transparent_keys.contains(&block_key(blocks[1])));
    }

    #[test]
    fn compressed_planes_sort_and_deduplicate_with_tolerance() {
        let mut planes = vec![2.0, 0.0, 1.0, 1.0 + PLANE_DEDUP_EPSILON * 0.5];
        dedup_planes(&mut planes);
        assert_eq!(planes, vec![0.0, 1.0, 2.0]);
        assert_eq!(
            plane_index(&planes, 1.0 + PLANE_DEDUP_EPSILON * 0.25),
            Some(1)
        );
        assert_eq!(plane_index(&planes, 1.5), None);
    }

    #[test]
    fn regular_planes_preserve_exact_grid_extents() {
        assert_eq!(
            regular_planes(-1.0, 1.0, 4),
            vec![-1.0, -0.5, 0.0, 0.5, 1.0]
        );
    }

    #[test]
    fn cell_payload_packs_grade_or_fallback_flag() {
        assert_eq!(pack_cell_payload(0.5, true), 32768);
        assert_eq!(
            pack_cell_payload(FALLBACK_BLOCK_GRADE, false),
            FALLBACK_CELL_FLAG
        );
        assert_ne!(pack_cell_payload(0.0, true), EMPTY_CELL_PAYLOAD);
    }

    /// Packs a dense grid into the sparse brick layout exactly as
    /// `build_block_volume_asset` does, for equivalence testing.
    fn pack_dense_to_bricks(dims: [usize; 3], dense: &[u32]) -> (Vec<u32>, Vec<u32>, [usize; 3]) {
        let brick_dims = brick_grid_dims(dims);
        let brick_count = brick_dims[0] * brick_dims[1] * brick_dims[2];
        let cells_per_brick = BRICK_SIZE * BRICK_SIZE * BRICK_SIZE;
        let dense_index = |i: usize, j: usize, k: usize| (k * dims[1] + j) * dims[0] + i;

        let mut occupied = vec![false; brick_count];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    if dense[dense_index(i, j, k)] != EMPTY_CELL_PAYLOAD {
                        let b =
                            brick_index(brick_dims, i / BRICK_SIZE, j / BRICK_SIZE, k / BRICK_SIZE);
                        occupied[b] = true;
                    }
                }
            }
        }
        // Assign sequential ordinals (matching `build_block_volume_asset`);
        // cells are ordinal-packed at `ordinal * cells_per_brick`.
        let mut brick_table = vec![EMPTY_BRICK; brick_count];
        let mut next_ordinal = 0u32;
        for (b, &occ) in occupied.iter().enumerate() {
            if occ {
                brick_table[b] = next_ordinal;
                next_ordinal += 1;
            }
        }
        let mut cells = vec![EMPTY_CELL_PAYLOAD; next_ordinal as usize * cells_per_brick];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let payload = dense[dense_index(i, j, k)];
                    if payload != EMPTY_CELL_PAYLOAD {
                        let ordinal = brick_table[brick_index(
                            brick_dims,
                            i / BRICK_SIZE,
                            j / BRICK_SIZE,
                            k / BRICK_SIZE,
                        )] as usize;
                        let local =
                            brick_local_index(i % BRICK_SIZE, j % BRICK_SIZE, k % BRICK_SIZE);
                        cells[ordinal * cells_per_brick + local] = payload;
                    }
                }
            }
        }
        (brick_table, cells, brick_dims)
    }

    /// Build a RAM-backed `BlockVolumeAsset` from a dense-packed brick table +
    /// cells for the style/aggregate tests. Ramp-dependent fields are left
    /// empty (filled by `compute_brick_style_data`).
    fn test_volume_asset(
        brick_table: Vec<u32>,
        cells: Vec<u32>,
        brick_dims: [usize; 3],
        dims: [u32; 3],
        bounds_max: [f32; 3],
    ) -> BlockVolumeAsset {
        let unit_planes = |n: u32| (0..=n).map(|c| c as f32).collect::<Vec<_>>();
        let occupied_count = brick_table.iter().filter(|&&o| o != EMPTY_BRICK).count();
        BlockVolumeAsset {
            x_planes: unit_planes(dims[0]),
            y_planes: unit_planes(dims[1]),
            z_planes: unit_planes(dims[2]),
            cells: CellBacking::Ram(cells),
            brick_table,
            brick_aggregates: Vec::new(),
            brick_uniform: Vec::new(),
            brick_centers: Vec::new(),
            occupied_count,
            dims,
            brick_dims: [
                brick_dims[0] as u32,
                brick_dims[1] as u32,
                brick_dims[2] as u32,
            ],
            bounds_min: [0.0; 3],
            bounds_max,
            reference_len: 1.0,
        }
    }

    #[test]
    fn sparse_brick_lookup_matches_dense_for_every_cell() {
        // Spans several bricks per axis with edge bricks that aren't full
        // (20/11/9 are not multiples of BRICK_SIZE = 8).
        let dims = [20usize, 11, 9];
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut dense = vec![EMPTY_CELL_PAYLOAD; dims[0] * dims[1] * dims[2]];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    // Leave the middle brick column (i in [8, 16)) entirely
                    // empty so whole bricks are unoccupied; fill the rest
                    // deterministically at ~40%.
                    if !(8..16).contains(&i) && next() % 5 < 2 {
                        dense[(k * dims[1] + j) * dims[0] + i] = next() & 0xffff;
                    }
                }
            }
        }

        let (brick_table, cells, brick_dims) = pack_dense_to_bricks(dims, &dense);
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let expected = dense[(k * dims[1] + j) * dims[0] + i];
                    let got = sparse_cell_payload(brick_dims, &brick_table, &cells, i, j, k);
                    assert_eq!(got, expected, "sparse lookup mismatch at ({i}, {j}, {k})");
                }
            }
        }

        // The test is only meaningful if the layout actually exercised both
        // empty and occupied bricks.
        assert!(
            brick_table.contains(&EMPTY_BRICK),
            "expected at least one empty brick"
        );
        assert!(
            brick_table.iter().any(|&b| b != EMPTY_BRICK),
            "expected at least one occupied brick"
        );
    }

    #[test]
    fn volume_ramp_lut_matches_ramp_rgba_and_sigma() {
        use crate::model::block_model::ColorStop;
        let ramp = ColorTransferFunction {
            stops: vec![
                ColorStop {
                    id: 1,
                    t: 0.1,
                    color: [1.0, 0.2, 0.1, 0.5],
                },
                ColorStop {
                    id: 2,
                    t: 0.5,
                    color: [0.1, 0.2, 1.0, 1.0],
                },
                ColorStop {
                    id: 3,
                    t: 0.9,
                    color: [0.1, 0.9, 0.2, 0.0],
                },
            ],
        };
        let fallback = [0.6, 0.6, 0.6, 0.5];
        let reference_len = 2.5;
        let lut = VolumeRampLut::build(&ramp, fallback, reference_len);
        // Every quantized grade must resolve exactly as the direct ramp scan
        // the shader replicates — including sigma, which feeds the LOD
        // aggregates. Full sweep: it's the whole point of the table.
        for grade in 0..=u16::MAX {
            let t = f32::from(grade) / 65535.0;
            let expected = ramp_rgba(&ramp, t);
            let (rgba, sigma) = lut.entries[grade as usize];
            assert_eq!(rgba, expected, "LUT colour diverged at grade {grade}");
            assert_eq!(
                sigma,
                volume_sigma_for_alpha(expected[3].clamp(0.0, 1.0), reference_len),
                "LUT sigma diverged at grade {grade}"
            );
        }
        assert_eq!(lut.resolve(FALLBACK_CELL_FLAG), Some(&lut.fallback));
        assert_eq!(lut.fallback.0, fallback);
        assert_eq!(lut.resolve(EMPTY_CELL_PAYLOAD), None);
        // Grade payloads with the fallback bit clear resolve through entries.
        assert_eq!(lut.resolve(1234), Some(&lut.entries[1234]));
    }

    #[test]
    fn brick_style_flags_mark_exactly_the_uniform_appearance_bricks() {
        use crate::model::block_model::ColorStop;
        // Three bricks along x. Brick 0: different quantized grades that all
        // land in one stop bin — one appearance, uniform. Brick 1: grades
        // from two bins — not uniform. Brick 2: EMPTY payloads mixed with
        // below-first-stop grades — both invisible, one appearance, uniform.
        let dims = [24usize, 8, 8];
        let mut dense = vec![EMPTY_CELL_PAYLOAD; dims[0] * dims[1] * dims[2]];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let payload = if i < 8 {
                        40_000 + ((i + j + k) as u32 % 7) * 100
                    } else if i < 16 {
                        if (i + j + k) % 2 == 0 { 40_000 } else { 20_000 }
                    } else if (i + j + k) % 2 == 0 {
                        EMPTY_CELL_PAYLOAD
                    } else {
                        3_000 // grade ~0.046, below the first stop at 0.1
                    };
                    dense[(k * dims[1] + j) * dims[0] + i] = payload;
                }
            }
        }
        let (brick_table, cells, brick_dims) = pack_dense_to_bricks(dims, &dense);
        let asset = test_volume_asset(brick_table, cells, brick_dims, [24, 8, 8], [24.0, 8.0, 8.0]);
        let ramp = ColorTransferFunction {
            stops: vec![
                ColorStop {
                    id: 1,
                    t: 0.1,
                    color: [1.0, 0.2, 0.1, 0.5],
                },
                ColorStop {
                    id: 2,
                    t: 0.5,
                    color: [0.1, 0.2, 1.0, 1.0],
                },
            ],
        };
        let style = compute_brick_style_data(&asset, &ramp, [0.6, 0.6, 0.6, 0.5]);

        assert_eq!(style.uniform_flags.len(), 3);
        assert_eq!(
            style.uniform_flags,
            vec![true, false, true],
            "expected uniform, mixed, invisible-uniform"
        );

        // Brick 0 is fully filled with one appearance (the second stop:
        // blue, alpha 1.0), so its aggregate must be exactly that colour and
        // extinction — the coarse LOD path then reproduces the fine path.
        let sigma = volume_sigma_for_alpha(1.0, 1.0);
        let aggregate = style.aggregates[0];
        assert!(
            (aggregate[0] - 0.1).abs() < 1.0e-5
                && (aggregate[1] - 0.2).abs() < 1.0e-5
                && (aggregate[2] - 1.0).abs() < 1.0e-5,
            "homogeneous brick aggregate colour diverged: {aggregate:?}"
        );
        assert!(
            (aggregate[3] - sigma).abs() < 1.0e-4,
            "homogeneous brick aggregate sigma diverged: {} vs {sigma}",
            aggregate[3]
        );
        // Brick 2 holds only invisible cells: no aggregate contribution.
        assert_eq!(style.aggregates[2], [0.0; 4]);
    }

    #[test]
    fn brick_style_flags_match_brute_force_on_clipped_grids() {
        use crate::model::block_model::ColorStop;
        // Edge bricks are clipped (19/10/9 are not multiples of BRICK_SIZE),
        // exercising the partial-range iteration that the flag computation
        // shares with the aggregates. Flags must agree with a direct
        // per-brick scan of the dense grid.
        let dims = [19usize, 10, 9];
        let mut state = 0xdead_beefu32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut dense = vec![EMPTY_CELL_PAYLOAD; dims[0] * dims[1] * dims[2]];
        for slot in dense.iter_mut() {
            *slot = match next() % 6 {
                // Bias towards few distinct appearances so some bricks come
                // out uniform and others mixed.
                0 => EMPTY_CELL_PAYLOAD,
                1 => FALLBACK_CELL_FLAG,
                2 | 3 => 40_000,
                4 => 20_000,
                _ => next() & 0xffff,
            };
        }
        let (brick_table, cells, brick_dims) = pack_dense_to_bricks(dims, &dense);
        let asset = test_volume_asset(
            brick_table.clone(),
            cells,
            brick_dims,
            [dims[0] as u32, dims[1] as u32, dims[2] as u32],
            [dims[0] as f32, dims[1] as f32, dims[2] as f32],
        );
        let ramp = ColorTransferFunction {
            stops: vec![
                ColorStop {
                    id: 1,
                    t: 0.1,
                    color: [1.0, 0.2, 0.1, 0.5],
                },
                ColorStop {
                    id: 2,
                    t: 0.5,
                    color: [0.1, 0.2, 1.0, 1.0],
                },
            ],
        };
        let fallback = [0.6, 0.6, 0.6, 0.5];
        let style = compute_brick_style_data(&asset, &ramp, fallback);

        // Brute-force appearance per cell, straight from the dense grid.
        let appearance = |payload: u32| -> Option<[f32; 4]> {
            if payload == EMPTY_CELL_PAYLOAD {
                return None;
            }
            let rgba = if payload & FALLBACK_CELL_FLAG != 0 {
                fallback
            } else {
                ramp_rgba(&ramp, (payload & 0xffff) as f32 / 65535.0)
            };
            (rgba[3] >= VISIBLE_ALPHA_EPSILON).then_some(rgba)
        };
        let mut checked = 0;
        for (bindex, &ordinal) in brick_table.iter().enumerate() {
            if ordinal == EMPTY_BRICK {
                continue;
            }
            let bi = bindex % brick_dims[0];
            let bj = (bindex / brick_dims[0]) % brick_dims[1];
            let bk = bindex / (brick_dims[0] * brick_dims[1]);
            let mut classes = std::collections::HashSet::new();
            for k in bk * BRICK_SIZE..((bk + 1) * BRICK_SIZE).min(dims[2]) {
                for j in bj * BRICK_SIZE..((bj + 1) * BRICK_SIZE).min(dims[1]) {
                    for i in bi * BRICK_SIZE..((bi + 1) * BRICK_SIZE).min(dims[0]) {
                        let payload = dense[(k * dims[1] + j) * dims[0] + i];
                        classes.insert(appearance(payload).map(|c| c.map(f32::to_bits)));
                    }
                }
            }
            let expected = classes.len() <= 1;
            let got = style.uniform_flags[ordinal as usize];
            assert_eq!(
                got,
                expected,
                "flag mismatch for brick ({bi}, {bj}, {bk}): {} classes",
                classes.len()
            );
            checked += 1;
        }
        assert!(
            checked > 4,
            "expected several occupied bricks, got {checked}"
        );
    }

    /// Fully drive a streamer to a steady state for the given camera: replan,
    /// then drain in bounded batches until nothing is pending. Returns the
    /// resident ordinal set. Asserts the invariants that must always hold.
    fn settle_streamer(
        streamer: &mut BrickStreamer,
        camera: Vec3,
    ) -> std::collections::HashSet<u32> {
        if streamer.needs_replan(camera) {
            streamer.replan(camera);
        }
        while streamer.has_pending() {
            let plan = streamer.drain(3); // small batch to exercise multi-frame draining
            // A load must never exceed capacity or double-book a slot.
            for &(slot, ordinal) in &plan.uploads {
                assert!(slot < streamer.pool_slots, "slot out of range");
                assert_eq!(streamer.slot_occupant[slot as usize], ordinal);
                assert_eq!(streamer.ordinal_slot[ordinal as usize], slot);
            }
        }
        // Invariant: the slot map and ordinal map agree, and no uniform brick
        // is resident.
        let mut resident = std::collections::HashSet::new();
        for (ordinal, &slot) in streamer.ordinal_slot.iter().enumerate() {
            if slot != NOT_RESIDENT_SLOT {
                assert!(!streamer.uniform[ordinal], "uniform brick resident");
                assert_eq!(streamer.slot_occupant[slot as usize], ordinal as u32);
                resident.insert(ordinal as u32);
            }
        }
        resident
    }

    #[test]
    fn mapped_cell_backing_round_trips_through_a_temp_file() {
        // Force the mmap path (bypassing the RAM threshold) and verify the
        // whole write→seal→read cycle: initial fill is EMPTY, per-ordinal
        // writes land, and the sealed mapping reads them back exactly. Also
        // confirms the temp file is cleaned up on drop.
        let brick_count = 5;
        let cell_count = brick_count * CELLS_PER_BRICK;
        let mut builder =
            CellBackingBuilder::new_mapped(cell_count).expect("temp-file backing should build");
        {
            let cells = builder.as_mut_slice();
            assert_eq!(cells.len(), cell_count);
            assert!(
                cells.iter().all(|&c| c == EMPTY_CELL_PAYLOAD),
                "starts empty"
            );
            // Write a distinct pattern per brick, leaving some holes.
            for (idx, cell) in cells.iter_mut().enumerate() {
                let ordinal = idx / CELLS_PER_BRICK;
                let local = idx % CELLS_PER_BRICK;
                if local.is_multiple_of(3) {
                    *cell = (ordinal as u32) << 8 | local as u32;
                }
            }
        }
        let path = match &builder {
            CellBackingBuilder::Mapped { path, .. } => path.clone(),
            CellBackingBuilder::Ram(_) => panic!("expected the mmap path"),
        };
        let backing = builder.finish().expect("sealing should succeed");
        for ordinal in 0..brick_count as u32 {
            for (local, &cell) in backing.brick_cells(ordinal).iter().enumerate() {
                let expected = if local.is_multiple_of(3) {
                    ordinal << 8 | local as u32
                } else {
                    EMPTY_CELL_PAYLOAD
                };
                assert_eq!(cell, expected, "brick {ordinal} cell {local}");
            }
        }
        assert!(path.exists(), "temp file present while mapping is alive");
        drop(backing);
        assert!(!path.exists(), "temp file removed on drop");
    }

    #[test]
    fn streamer_fits_case_makes_every_mixed_brick_resident_once() {
        // 5 mixed bricks, pool of 8 slots: all fit. After settling, every
        // mixed brick is resident and uniform bricks never are. A second
        // settle at a different camera does nothing (camera-independent).
        let uniform = vec![false, true, false, false, true, false, false];
        let centers = (0..7).map(|i| [i as f32, 0.0, 0.0]).collect::<Vec<_>>();
        let mut s = BrickStreamer::new(8, uniform.clone(), centers, 1.0);
        let resident = settle_streamer(&mut s, Vec3::ZERO);
        let expected: std::collections::HashSet<u32> =
            (0..7).filter(|&o| !uniform[o as usize]).collect();
        assert_eq!(resident, expected, "all mixed bricks should be resident");
        assert!(
            !s.needs_replan(Vec3::new(100.0, 100.0, 100.0)),
            "fits case is camera-independent"
        );
    }

    #[test]
    fn streamer_keeps_the_nearest_bricks_when_over_budget() {
        // 10 mixed bricks along x at 0..9, pool holds only 3. The camera near
        // x=0 must keep bricks {0,1,2}; moved near x=9 must keep {7,8,9}.
        let uniform = vec![false; 10];
        let centers = (0..10).map(|i| [i as f32, 0.0, 0.0]).collect::<Vec<_>>();
        let mut s = BrickStreamer::new(3, uniform, centers, 0.5);

        let near0 = settle_streamer(&mut s, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(near0, [0, 1, 2].into_iter().collect(), "nearest to x=0");
        assert_eq!(near0.len(), 3, "pool exactly filled");

        // A large move must trigger a replan and swap the resident set.
        assert!(s.needs_replan(Vec3::new(9.0, 0.0, 0.0)));
        let near9 = settle_streamer(&mut s, Vec3::new(9.0, 0.0, 0.0));
        assert_eq!(near9, [7, 8, 9].into_iter().collect(), "nearest to x=9");
    }

    #[test]
    fn streamer_ignores_camera_jitter_below_the_replan_distance() {
        let uniform = vec![false; 10];
        let centers = (0..10).map(|i| [i as f32, 0.0, 0.0]).collect::<Vec<_>>();
        let mut s = BrickStreamer::new(3, uniform, centers, 2.0);
        settle_streamer(&mut s, Vec3::new(0.0, 0.0, 0.0));
        // Move less than replan_distance: no replan, resident set unchanged.
        assert!(!s.needs_replan(Vec3::new(1.5, 0.0, 0.0)));
    }

    #[test]
    fn streamer_evicts_bricks_that_become_uniform() {
        // 4 mixed bricks, pool of 4 (fits). After settling all are resident.
        // A restyle makes brick 1 uniform: it must be evicted, its slot freed,
        // and its `brick_info` flips to the uniform sentinel.
        let uniform = vec![false; 4];
        let centers = (0..4).map(|i| [i as f32, 0.0, 0.0]).collect::<Vec<_>>();
        let mut s = BrickStreamer::new(4, uniform, centers, 0.5);
        settle_streamer(&mut s, Vec3::ZERO);
        assert_eq!(
            s.ordinal_slot
                .iter()
                .filter(|&&x| x != NOT_RESIDENT_SLOT)
                .count(),
            4
        );

        s.set_uniform(vec![false, true, false, false]);
        assert_eq!(
            s.ordinal_slot[1], NOT_RESIDENT_SLOT,
            "uniform brick evicted"
        );
        assert_eq!(s.info(1), UNIFORM_BRICK_FLAG | NOT_RESIDENT_SLOT);
        // The freed slot is reusable; settling keeps the other three resident
        // and never re-admits the uniform brick.
        let resident = settle_streamer(&mut s, Vec3::ZERO);
        assert_eq!(resident, [0, 2, 3].into_iter().collect());
    }

    #[test]
    fn streamer_admits_bricks_that_become_mixed() {
        // Start with brick 2 uniform (never resident). A restyle makes it
        // mixed; it must then become resident on the next settle.
        let uniform = vec![false, false, true, false];
        let centers = (0..4).map(|i| [i as f32, 0.0, 0.0]).collect::<Vec<_>>();
        let mut s = BrickStreamer::new(4, uniform, centers, 0.5);
        let resident = settle_streamer(&mut s, Vec3::ZERO);
        assert!(!resident.contains(&2), "uniform brick starts non-resident");

        s.set_uniform(vec![false; 4]);
        let resident = settle_streamer(&mut s, Vec3::ZERO);
        assert!(resident.contains(&2), "newly-mixed brick becomes resident");
        assert_eq!(resident.len(), 4);
    }

    #[test]
    fn streamer_info_encodes_flag_and_slot() {
        let uniform = vec![false, true];
        let centers = vec![[0.0; 3], [1.0; 3]];
        let mut s = BrickStreamer::new(2, uniform, centers, 0.5);
        settle_streamer(&mut s, Vec3::ZERO);
        // Mixed brick 0 resident in some slot: high bit clear, low bits = slot.
        let info0 = s.info(0);
        assert_eq!(info0 & UNIFORM_BRICK_FLAG, 0);
        assert!(info0 < s.pool_slots, "mixed resident encodes its slot");
        // Uniform brick 1: flag set, not resident.
        assert_eq!(s.info(1), UNIFORM_BRICK_FLAG | NOT_RESIDENT_SLOT);
    }

    #[test]
    fn visible_block_faces_keeps_all_faces_for_isolated_block() {
        let block = crate::model::block_model::BlockBounds {
            lower: DVec3::ZERO,
            upper: DVec3::ONE,
        };
        let renderable = HashSet::from([block_key(block)]);

        assert_eq!(visible_block_faces(block, &renderable), [true; 6]);
    }

    #[test]
    fn visible_block_faces_culls_shared_regular_grid_face() {
        let a = crate::model::block_model::BlockBounds {
            lower: DVec3::ZERO,
            upper: DVec3::ONE,
        };
        let b = crate::model::block_model::BlockBounds {
            lower: DVec3::X,
            upper: DVec3::new(2.0, 1.0, 1.0),
        };
        let renderable = HashSet::from([block_key(a), block_key(b)]);

        assert_eq!(
            visible_block_faces(a, &renderable),
            [true, false, true, true, true, true]
        );
        assert_eq!(
            visible_block_faces(b, &renderable),
            [false, true, true, true, true, true]
        );
    }
}
