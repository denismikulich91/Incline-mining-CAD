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
pub(crate) const FALLBACK_BLOCK_GRADE: f32 = -1.0;
const HIDDEN_BLOCK_GRADE: f32 = -2.0;
/// Grades below this are discarded by `block_model.wgsl` (`grade < -1.5`).
/// Geometry building must treat such blocks as absent — a discarded block
/// leaves a hole, so it can't be allowed to cull its neighbours' faces.
use super::block_model_ramp::{block_alpha, is_hidden_block_appearance, make_translucent};

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
    /// Scene-origin-relative AABB, for per-chunk frustum culling. Vertex
    /// positions are stored relative to the chunk's own AABB centre instead
    /// (small magnitudes keep f32 interpolation precise far from the scene
    /// origin); the shader re-adds the scene-relative centre from the chunk
    /// uniform in `chunk_bind_group`.
    pub(crate) bounds_min: glam::Vec3,
    pub(crate) bounds_max: glam::Vec3,
    /// Uniform bind group (group 2) holding this chunk's scene-origin-relative
    /// rebase offset; the bind group keeps the buffer alive.
    pub(crate) chunk_bind_group: wgpu::BindGroup,
    /// A `SurfaceStyle` bind group holding a distinct per-chunk debug colour,
    /// bound instead of the mesh colour when the chunk-debug view is on.
    pub(crate) debug_style_bind_group: wgpu::BindGroup,
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

pub(crate) use super::block_model_volume_cache::{
    CachedBlockVolumeGpu, apply_scene_to_local, build_block_volume_gpu, stream_volume_bricks,
    update_block_volume_style,
};

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
                if scene_origin_dirty && !geometry_dirty {
                    // Surface and edge chunks bake in scene-origin-relative
                    // positions, so an origin change invalidates them along
                    // with the volume. (Camera code currently clears the whole
                    // cache on origin changes, so this path is a backstop.)
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
                }
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
        surface_chunk_layout: &wgpu::BindGroupLayout,
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
                let surface_chunks = build_surface_chunks(
                    device,
                    scene_origin,
                    triangulation,
                    surface_style_layout,
                    surface_chunk_layout,
                );
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

/// Target triangles per spatial chunk. Chunks are the granularity of frustum
/// culling (and per-chunk debug colouring), so this trades culling precision
/// (smaller = tighter) against per-chunk draw-call/AABB-test overhead
/// (smaller = more). ~100k keeps a multi-million-face mesh in tens of chunks.
const TARGET_FACES_PER_CHUNK: usize = 100_000;

/// Upload indexed surface geometry as spatially-coherent chunks, walking faces
/// in the precomputed XY-Morton order (`triangulation.surface_face_order`,
/// built off-thread) so each chunk covers a compact region with a tight AABB
/// the renderer can frustum-cull — rather than the old size-only split whose
/// chunks each spanned the whole mesh. No sorting happens here, so a huge
/// mesh's first upload doesn't hitch the render thread. Colour is NOT baked in;
/// it lives in a per-draw SurfaceStyle uniform (plus a per-chunk debug-colour
/// uniform for the chunk-debug view).
fn build_surface_chunks(
    device: &wgpu::Device,
    scene_origin: DVec3,
    triangulation: &OpenTriangulation,
    surface_style_layout: &wgpu::BindGroupLayout,
    surface_chunk_layout: &wgpu::BindGroupLayout,
) -> Vec<CachedSurfaceChunk> {
    let mesh = &triangulation.mesh;
    let source = mesh.vertices();
    let face_count = mesh.face_count();
    if source.is_empty() || face_count == 0 || source.len() > u32::MAX as usize {
        if source.len() > u32::MAX as usize {
            log::error!(
                "Triangulation '{}' has {} vertices (> u32::MAX); cannot chunk for GPU",
                triangulation.name,
                source.len()
            );
        }
        return Vec::new();
    }
    let order = &triangulation.surface_face_order;

    // Chunk size: the target, capped so a chunk's vertex/index buffers fit the
    // device limit and local indices stay within u32.
    let limit = (device.limits().max_buffer_size as usize).min(MAX_SURFACE_CHUNK_BYTES);
    let max_indices = limit / std::mem::size_of::<u32>();
    let max_vertices = limit / std::mem::size_of::<SurfaceVertex>();
    let faces_cap = (max_indices / 3)
        .min(max_vertices / 3)
        .min(u32::MAX as usize / 3)
        .max(1);
    let faces_per_chunk = TARGET_FACES_PER_CHUNK.min(faces_cap).max(1);

    let mut chunks = Vec::new();
    // Dense sentinel remap reused across chunks: remap[global] == u32::MAX means
    // "not yet in this chunk". Reset only touched entries between chunks.
    let mut remap = vec![u32::MAX; source.len()];
    let mut dirty: Vec<usize> = Vec::new();
    let mut vertices: Vec<SurfaceVertex> = Vec::new();
    // Flat shading takes the normal from the triangle's provoking (first)
    // vertex, so each face needs a first vertex no other face uses. Tracks,
    // per local vertex, whether a face has already claimed it.
    let mut provoking_claimed: Vec<bool> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for (chunk_index, run) in order.chunks(faces_per_chunk).enumerate() {
        vertices.clear();
        provoking_claimed.clear();
        indices.clear();

        // First pass: world-space AABB in f64. Its centre becomes the chunk's
        // local origin so vertex magnitudes stay small no matter how far the
        // chunk sits from the scene origin.
        let mut world_min = DVec3::splat(f64::INFINITY);
        let mut world_max = DVec3::splat(f64::NEG_INFINITY);
        for &face_index in run {
            let Some(face) = mesh.face_vertex_indices(face_index as usize) else {
                continue;
            };
            for global_index in face {
                if let Some(point) = source.get(global_index) {
                    let pos = DVec3::new(point.x, point.y, point.z);
                    world_min = world_min.min(pos);
                    world_max = world_max.max(pos);
                }
            }
        }
        if !world_min.x.is_finite() {
            continue;
        }
        let chunk_origin = (world_min + world_max) * 0.5;

        for &face_index in run {
            let Some(face) = mesh.face_vertex_indices(face_index as usize) else {
                continue;
            };
            let [Some(pa), Some(pb), Some(pc)] = face.map(|index| {
                source
                    .get(index)
                    .map(|point| DVec3::new(point.x, point.y, point.z))
            }) else {
                continue;
            };
            // Flat face normal in f64; oriented z >= 0 to match the two-sided
            // surface lighting (previously done per-fragment in the shader).
            let mut face_normal = (pb - pa).cross(pc - pa).normalize_or_zero();
            if face_normal.z < 0.0 {
                face_normal = -face_normal;
            }
            let normal = face_normal.as_vec3().to_array();

            let mut local = [0u32; 3];
            for (slot, global_index) in face.into_iter().enumerate() {
                local[slot] = if remap[global_index] != u32::MAX {
                    remap[global_index]
                } else {
                    let point = source[global_index];
                    let index = vertices.len() as u32;
                    vertices.push(surface_vertex(point, chunk_origin, normal));
                    provoking_claimed.push(false);
                    remap[global_index] = index;
                    dirty.push(global_index);
                    index
                };
            }
            // Rotate the face (winding-preserving) so an unclaimed vertex
            // provokes it; duplicate one vertex only when all three are taken.
            match (0..3).find(|&i| !provoking_claimed[local[i] as usize]) {
                Some(i) => local.rotate_left(i),
                None => {
                    let duplicate = SurfaceVertex {
                        pos: vertices[local[0] as usize].pos,
                        normal,
                    };
                    local[0] = vertices.len() as u32;
                    vertices.push(duplicate);
                    provoking_claimed.push(false);
                }
            }
            let provoking = local[0] as usize;
            vertices[provoking].normal = normal;
            provoking_claimed[provoking] = true;
            indices.extend_from_slice(&local);
        }
        if let Some(chunk) = upload_surface_chunk(
            device,
            &vertices,
            &indices,
            (world_min - scene_origin).as_vec3(),
            (world_max - scene_origin).as_vec3(),
            (chunk_origin - scene_origin).as_vec3(),
            chunk_index,
            surface_style_layout,
            surface_chunk_layout,
        ) {
            chunks.push(chunk);
        }
        for &global_index in &dirty {
            remap[global_index] = u32::MAX;
        }
        dirty.clear();
    }
    if chunks.len() > 1 {
        log::info!(
            "Triangulation '{}' uploaded in {} spatial chunks ({} faces)",
            triangulation.name,
            chunks.len(),
            face_count
        );
    }
    chunks
}

/// A distinct, well-spread debug colour per chunk index (golden-ratio hue).
fn chunk_debug_color(chunk_index: usize) -> [f32; 4] {
    let hue = (chunk_index as f32 * 0.618_034).fract();
    let [r, g, b] = hsv_to_rgb(hue, 0.65, 0.95);
    [r, g, b, 1.0]
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match (i as i32).rem_euclid(6) {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

fn surface_vertex(point: tri00t::Vertex, chunk_origin: DVec3, normal: [f32; 3]) -> SurfaceVertex {
    let local = DVec3::new(point.x, point.y, point.z) - chunk_origin;
    SurfaceVertex {
        pos: local.as_vec3().to_array(),
        normal,
    }
}

#[allow(clippy::too_many_arguments)]
fn upload_surface_chunk(
    device: &wgpu::Device,
    vertices: &[SurfaceVertex],
    indices: &[u32],
    bounds_min: glam::Vec3,
    bounds_max: glam::Vec3,
    chunk_offset: glam::Vec3,
    chunk_index: usize,
    surface_style_layout: &wgpu::BindGroupLayout,
    surface_chunk_layout: &wgpu::BindGroupLayout,
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
    // Per-chunk debug colour uniform; the bind group keeps the buffer alive.
    let debug_style = SurfaceStyleUniform {
        color: chunk_debug_color(chunk_index),
    };
    let debug_style_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Chunk Debug Style Uniform"),
        contents: bytemuck::bytes_of(&debug_style),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let debug_style_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: surface_style_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: debug_style_buffer.as_entire_binding(),
        }],
        label: Some("Chunk Debug Style Bind Group"),
    });
    // Scene-origin-relative rebase offset re-added in the vertex shader.
    let chunk_uniform: [f32; 4] = [chunk_offset.x, chunk_offset.y, chunk_offset.z, 0.0];
    let chunk_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Surface Chunk Offset Uniform"),
        contents: bytemuck::bytes_of(&chunk_uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let chunk_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: surface_chunk_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: chunk_buffer.as_entire_binding(),
        }],
        label: Some("Surface Chunk Offset Bind Group"),
    });
    Some(CachedSurfaceChunk {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        bounds_min,
        bounds_max,
        chunk_bind_group,
        debug_style_bind_group,
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

pub(crate) type BlockModelColorValues = (std::sync::Arc<Vec<f64>>, Option<f64>, Option<(f64, f64)>);

pub(crate) fn block_model_color_values(
    block_model: &OpenBlockModel,
) -> Option<BlockModelColorValues> {
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

pub(crate) fn grade_for_block(
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
        || crate::model::block_model::is_no_data_sentinel(value)
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
