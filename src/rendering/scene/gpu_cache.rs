//! Persistent GPU representation of immutable and infrequently-changing scene assets.

use std::collections::{HashMap, HashSet};

use glam::DVec3;
use wgpu::util::DeviceExt;

use crate::{
    model::{
        block_model::{
            BlockModelId, ColorTransferFunction, MAX_COLOR_STOPS, OpenBlockModel,
            numeric_variable_default, render_value_range,
        },
        formats::tri00t,
        triangulation::{OpenTriangulation, TriangulationId},
    },
    rendering::{BlockModelVertex, SurfaceVertex},
    ui::state::EditorState,
};

const YELLOW_HIGHLIGHT_COLOR: [f32; 4] = [1.0, 0.85, 0.0, 1.0];
const MAX_SURFACE_CHUNK_BYTES: usize = 256 * 1024 * 1024;
const FALLBACK_BLOCK_GRADE: f32 = -1.0;
const HIDDEN_BLOCK_GRADE: f32 = -2.0;

fn make_translucent(color: &mut [f32; 4]) {
    color[3] *= 0.3;
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
    edge_width: f32,
}

pub(crate) struct CachedSurfaceChunk {
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) index_buffer: wgpu::Buffer,
    pub(crate) index_count: u32,
    /// Scene-relative bounds of this chunk's geometry, used for frustum
    /// culling and transparency depth sorting.
    pub(crate) bounds_min: glam::Vec3,
    pub(crate) bounds_max: glam::Vec3,
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
    pub(crate) surface_chunks: Vec<CachedBlockModelSurfaceChunk>,
    pub(crate) surface_style_buffer: wgpu::Buffer,
    pub(crate) surface_style_bind_group: wgpu::BindGroup,
    pub(crate) translucent: bool,
    pub(crate) edge_chunks: Vec<CachedEdgeChunk>,
    pub(crate) edge_style_buffer: wgpu::Buffer,
    pub(crate) edge_style_bind_group: wgpu::BindGroup,
    line_color: [f32; 4],
    edge_width: f32,
    variable: Option<String>,
    color_transfer: ColorTransferFunction,
    hide_empty_color_values: bool,
}

/// A block model surface chunk plus enough CPU-side state to re-colour it
/// (on an active-variable/legend switch) without re-walking blocks, redoing
/// face culling, or reallocating GPU buffers — only geometry changes
/// (translucency toggling, or the block model itself changing) need a full
/// rebuild.
pub(crate) struct CachedBlockModelSurfaceChunk {
    pub(crate) gpu: CachedSurfaceChunk,
    /// Mirrors the vertex buffer's current contents so `grade` can be
    /// patched in place and re-uploaded with a single `write_buffer`.
    cpu_vertices: Vec<BlockModelVertex>,
    /// CPU vertex ranges generated for each block, used to patch grade in
    /// place even when exposed-face culling gives each block a different
    /// vertex count.
    block_vertex_ranges: Vec<BlockVertexRange>,
}

struct BlockVertexRange {
    block_index: usize,
    start: usize,
    len: usize,
}

impl BlockModelGpuCache {
    pub(crate) fn clear(&mut self) {
        self.models.clear();
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.models.is_empty()
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
        edge_style_layout: &wgpu::BindGroupLayout,
    ) {
        let loaded: HashSet<_> = block_models.iter().map(|model| model.id).collect();
        self.models.retain(|id, _| loaded.contains(id));
        for block_model in block_models {
            let entity = block_model.entity_id();
            let selected = editor.selected_handles.contains(&entity);
            let translucent = editor.translucent_handles.contains(&entity)
                || block_model_has_partial_alpha_stops(block_model);
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
                let geometry_dirty = cached.translucent != translucent;
                let style_dirty = cached.color_transfer != block_model.color_transfer;
                let empty_visibility_dirty =
                    cached.hide_empty_color_values != block_model.hide_empty_color_values;
                let edge_geom_dirty = (cached.edge_width == 0.0) != (edge_width == 0.0);
                let edge_style_dirty =
                    cached.line_color != line_color || cached.edge_width != edge_width;
                if !variable_dirty
                    && !geometry_dirty
                    && !style_dirty
                    && !empty_visibility_dirty
                    && !edge_geom_dirty
                    && !edge_style_dirty
                {
                    continue;
                }
                if geometry_dirty {
                    // Translucency changes which pipeline/chunking the model
                    // draws with, so it needs a full rebuild.
                    cached.surface_chunks = build_block_model_surface_chunks(
                        device,
                        scene_origin,
                        block_model,
                        !translucent,
                    );
                    let style = block_model_style(block_model, translucent);
                    queue.write_buffer(&cached.surface_style_buffer, 0, bytemuck::bytes_of(&style));
                    cached.variable = block_model.active_numeric_variable.clone();
                    cached.color_transfer = block_model.color_transfer.clone();
                    cached.hide_empty_color_values = block_model.hide_empty_color_values;
                    cached.translucent = translucent;
                } else if variable_dirty || empty_visibility_dirty {
                    // A legend/attribute switch alone doesn't change which
                    // blocks or faces are rendered — only re-colour, without
                    // re-walking blocks or reallocating GPU buffers.
                    recolor_block_model_surface_chunks(
                        queue,
                        block_model,
                        &mut cached.surface_chunks,
                    );
                    if variable_dirty || style_dirty {
                        let style = block_model_style(block_model, translucent);
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
                    // Only the colour-transfer function (stop positions or
                    // colours) changed — dragging a handle doesn't touch
                    // which blocks/faces render or their `grade`, so a
                    // single style-uniform write is enough, independent of
                    // model size.
                    let style = block_model_style(block_model, translucent);
                    queue.write_buffer(&cached.surface_style_buffer, 0, bytemuck::bytes_of(&style));
                    cached.color_transfer = block_model.color_transfer.clone();
                }
                if edge_geom_dirty {
                    cached.edge_chunks =
                        build_block_model_edge_chunks(device, scene_origin, block_model);
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
                let surface_chunks = build_block_model_surface_chunks(
                    device,
                    scene_origin,
                    block_model,
                    !translucent,
                );
                if surface_chunks.is_empty() {
                    continue;
                }
                let surface_style = block_model_style(block_model, translucent);
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
                        surface_style_buffer,
                        surface_style_bind_group,
                        translucent,
                        edge_chunks,
                        edge_style_buffer,
                        edge_style_bind_group,
                        line_color,
                        edge_width,
                        variable: block_model.active_numeric_variable.clone(),
                        color_transfer: block_model.color_transfer.clone(),
                        hide_empty_color_values: block_model.hide_empty_color_values,
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

                if edge_geom_dirty {
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
    let (bounds_min, bounds_max) = vertex_bounds(vertices.iter().map(|vertex| vertex.pos));
    Some(CachedSurfaceChunk {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        bounds_min,
        bounds_max,
    })
}

/// Axis-aligned bounds of a set of vertex positions. Returns a degenerate
/// (zero-sized, all-zero) box for an empty iterator; callers only use this
/// on non-empty vertex slices.
fn vertex_bounds(positions: impl Iterator<Item = [f32; 3]>) -> (glam::Vec3, glam::Vec3) {
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for pos in positions {
        let pos = glam::Vec3::from(pos);
        min = min.min(pos);
        max = max.max(pos);
    }
    if min.x.is_finite() {
        (min, max)
    } else {
        (glam::Vec3::ZERO, glam::Vec3::ZERO)
    }
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
    let range = render_value_range(&values, &block_model.renderable_block_indices, default);
    Some((values, default, range))
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
        for range in &chunk.block_vertex_ranges {
            let grade = grade_for_block(
                &color_values,
                range.block_index,
                block_model.hide_empty_color_values,
            );
            for vertex in &mut chunk.cpu_vertices[range.start..range.start + range.len] {
                vertex.grade = grade;
            }
        }
        queue.write_buffer(
            &chunk.gpu.vertex_buffer,
            0,
            bytemuck::cast_slice(&chunk.cpu_vertices),
        );
    }
}

fn build_block_model_surface_chunks(
    device: &wgpu::Device,
    scene_origin: DVec3,
    block_model: &OpenBlockModel,
    cull_shared_faces: bool,
) -> Vec<CachedBlockModelSurfaceChunk> {
    const BLOCKS_PER_CHUNK: usize = 8192;

    let color_values = block_model_color_values(block_model);
    let renderable_blocks = cull_shared_faces.then(|| block_model_renderable_keys(block_model));

    const QUADS: [[u32; 4]; 6] = [
        [0, 3, 7, 4], // -X
        [1, 5, 6, 2], // +X
        [0, 4, 5, 1], // -Y
        [3, 2, 6, 7], // +Y
        [0, 1, 2, 3], // -Z
        [4, 7, 6, 5], // +Z
    ];

    let mut chunks = Vec::new();
    for source_indices in block_model
        .renderable_block_indices
        .chunks(BLOCKS_PER_CHUNK)
    {
        let mut vertices = Vec::with_capacity(source_indices.len() * 8);
        let mut indices = Vec::with_capacity(source_indices.len() * 36);
        let mut block_vertex_ranges = Vec::with_capacity(source_indices.len());
        for &block_index in source_indices {
            let Some(block) = block_model.blocks.get(block_index) else {
                continue;
            };
            let grade = grade_for_block(
                &color_values,
                block_index,
                block_model.hide_empty_color_values,
            );
            let visible_faces = renderable_blocks
                .as_ref()
                .map(|renderable_blocks| visible_block_faces(*block, renderable_blocks))
                .unwrap_or([true; 6]);
            if !visible_faces.iter().any(|visible| *visible) {
                continue;
            }
            let corners = block_corners(block_model, *block);
            let start = vertices.len();
            for (face_index, _) in visible_faces
                .into_iter()
                .enumerate()
                .filter(|(_, visible)| *visible)
            {
                let quad = QUADS[face_index];
                let Ok(base) = u32::try_from(vertices.len()) else {
                    break;
                };
                for corner_index in quad {
                    vertices.push(BlockModelVertex {
                        pos: (corners[corner_index as usize] - scene_origin)
                            .as_vec3()
                            .to_array(),
                        grade,
                    });
                }
                indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
            }
            let len = vertices.len() - start;
            if len > 0 {
                block_vertex_ranges.push(BlockVertexRange {
                    block_index,
                    start,
                    len,
                });
            }
        }
        if let Some(gpu) = upload_block_model_surface_chunk(device, &vertices, &indices) {
            chunks.push(CachedBlockModelSurfaceChunk {
                gpu,
                cpu_vertices: vertices,
                block_vertex_ranges,
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
    BlockModelStyleUniform {
        fallback_color,
        options: [
            if has_grade { 1.0 } else { 0.0 },
            stop_count as f32,
            0.0,
            0.0,
        ],
        stops,
    }
}

/// Whether any of the block model's colour-transfer stops has partial
/// (neither fully transparent nor fully opaque) alpha. Such a model must be
/// drawn through the translucent pipeline for grade colouring to blend
/// correctly, even if the user hasn't toggled translucency manually.
fn block_model_has_partial_alpha_stops(block_model: &OpenBlockModel) -> bool {
    block_model
        .color_transfer
        .stops
        .iter()
        .any(|stop| stop.color[3] > 0.02 && stop.color[3] < 0.98)
}

fn upload_block_model_surface_chunk(
    device: &wgpu::Device,
    vertices: &[BlockModelVertex],
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
            "Block model surface chunk rejected before GPU allocation: vertices={vertex_bytes} bytes, indices={index_bytes} bytes, limit={limit} bytes"
        );
        return None;
    }
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cached Block Model Surface Vertices"),
        contents: bytemuck::cast_slice(vertices),
        // COPY_DST is required here: `recolor_block_model_surface_chunks`
        // patches `grade` in place with `queue.write_buffer` on a legend/
        // attribute switch, instead of reallocating this buffer.
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Cached Block Model Surface Indices"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let (bounds_min, bounds_max) = vertex_bounds(vertices.iter().map(|vertex| vertex.pos));
    Some(CachedSurfaceChunk {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        bounds_min,
        bounds_max,
    })
}

fn block_model_renderable_keys(block_model: &OpenBlockModel) -> HashSet<[u64; 6]> {
    block_model
        .renderable_block_indices
        .iter()
        .filter_map(|&index| block_model.blocks.get(index).copied())
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
    let mut chunks = Vec::new();
    let mut instances = Vec::new();
    for &block_index in &block_model.renderable_block_indices {
        let Some(block) = block_model.blocks.get(block_index) else {
            continue;
        };
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
    fn vertex_bounds_of_empty_iterator_is_zero() {
        let (min, max) = vertex_bounds(std::iter::empty());
        assert_eq!(min, glam::Vec3::ZERO);
        assert_eq!(max, glam::Vec3::ZERO);
    }

    #[test]
    fn vertex_bounds_covers_all_positions() {
        let positions = [[1.0, -2.0, 0.5], [-3.0, 4.0, 2.0], [0.0, 0.0, -1.0]];
        let (min, max) = vertex_bounds(positions.into_iter());
        assert_eq!(min, glam::Vec3::new(-3.0, -2.0, -1.0));
        assert_eq!(max, glam::Vec3::new(1.0, 4.0, 2.0));
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
