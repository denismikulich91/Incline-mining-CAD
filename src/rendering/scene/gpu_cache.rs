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
    /// The block index each consecutive group of 8 vertices in
    /// `cpu_vertices` was generated from, in emission order.
    block_order: Vec<usize>,
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
                editor.selection_color
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
                let edge_geom_dirty = (cached.edge_width == 0.0) != (edge_width == 0.0);
                let edge_style_dirty =
                    cached.line_color != line_color || cached.edge_width != edge_width;
                if !variable_dirty
                    && !geometry_dirty
                    && !style_dirty
                    && !edge_geom_dirty
                    && !edge_style_dirty
                {
                    continue;
                }
                if geometry_dirty {
                    // Translucency changes which pipeline/chunking the model
                    // draws with, so it needs a full rebuild.
                    cached.surface_chunks =
                        build_block_model_surface_chunks(device, scene_origin, block_model);
                    let style = block_model_style(block_model, translucent);
                    queue.write_buffer(&cached.surface_style_buffer, 0, bytemuck::bytes_of(&style));
                    cached.variable = block_model.active_numeric_variable.clone();
                    cached.color_transfer = block_model.color_transfer.clone();
                    cached.translucent = translucent;
                } else if variable_dirty {
                    // A legend/attribute switch alone doesn't change which
                    // blocks or faces are rendered — only re-colour, without
                    // re-walking blocks or reallocating GPU buffers.
                    recolor_block_model_surface_chunks(
                        queue,
                        block_model,
                        &mut cached.surface_chunks,
                    );
                    let style = block_model_style(block_model, translucent);
                    queue.write_buffer(&cached.surface_style_buffer, 0, bytemuck::bytes_of(&style));
                    cached.variable = block_model.active_numeric_variable.clone();
                    cached.color_transfer = block_model.color_transfer.clone();
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
                let surface_chunks =
                    build_block_model_surface_chunks(device, scene_origin, block_model);
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
                editor.selection_color
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
type BlockModelColorValues = (Vec<f64>, Option<f64>, (f64, f64));

fn block_model_color_values(block_model: &OpenBlockModel) -> Option<BlockModelColorValues> {
    block_model
        .active_numeric_variable
        .as_deref()
        .and_then(|name| block_model.model.variable(name).map(|var| (name, var)))
        .and_then(|(name, var)| {
            let values = block_model.model.numeric_values(name).ok()?;
            let default = numeric_variable_default(var);
            let range =
                render_value_range(&values, &block_model.renderable_block_indices, default)?;
            Some((values, default, range))
        })
}

fn grade_for_block(color_values: &Option<BlockModelColorValues>, block_index: usize) -> f32 {
    color_values
        .as_ref()
        .and_then(|(values, default, range)| {
            values
                .get(block_index)
                .copied()
                .map(|value| normalized_grade(value, *default, *range))
        })
        .unwrap_or(-1.0)
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
        for (slot, &block_index) in chunk.block_order.iter().enumerate() {
            let grade = grade_for_block(&color_values, block_index);
            for vertex in &mut chunk.cpu_vertices[slot * 8..slot * 8 + 8] {
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
) -> Vec<CachedBlockModelSurfaceChunk> {
    const BLOCKS_PER_CHUNK: usize = 8192;

    let color_values = block_model_color_values(block_model);

    // Faces shared by two touching, both-rendered blocks are never visible
    // (interior geometry), so skip emitting them entirely. This only applies
    // to regular (non-sub-blocked) grids, where every block shares one cell
    // size and sits on an integer lattice we can index in O(1); irregular
    // schemas fall back to emitting all 6 faces, as before.
    const QUADS: [[u32; 4]; 6] = [
        [0, 3, 7, 4], // -X
        [1, 5, 6, 2], // +X
        [0, 4, 5, 1], // -Y
        [3, 2, 6, 7], // +Y
        [0, 1, 2, 3], // -Z
        [4, 7, 6, 5], // +Z
    ];
    let neighbor_grid = regular_block_grid_index(block_model);

    let mut chunks = Vec::new();
    for source_indices in block_model
        .renderable_block_indices
        .chunks(BLOCKS_PER_CHUNK)
    {
        let mut vertices = Vec::with_capacity(source_indices.len() * 8);
        let mut indices = Vec::with_capacity(source_indices.len() * 36);
        let mut block_order = Vec::with_capacity(source_indices.len());
        for &block_index in source_indices {
            let Some(block) = block_model.blocks.get(block_index) else {
                continue;
            };
            let visible_faces = neighbor_grid
                .as_ref()
                .and_then(|(grid, cell, origin)| {
                    let coord = grid_coord(block.lower, *origin, *cell)?;
                    Some([
                        !grid.contains(&(coord.0 - 1, coord.1, coord.2)),
                        !grid.contains(&(coord.0 + 1, coord.1, coord.2)),
                        !grid.contains(&(coord.0, coord.1 - 1, coord.2)),
                        !grid.contains(&(coord.0, coord.1 + 1, coord.2)),
                        !grid.contains(&(coord.0, coord.1, coord.2 - 1)),
                        !grid.contains(&(coord.0, coord.1, coord.2 + 1)),
                    ])
                })
                .unwrap_or([true; 6]);
            if visible_faces == [false; 6] {
                // Fully interior block: every neighbouring cell is also
                // rendered, so none of its faces can ever be seen.
                continue;
            }
            let Ok(base) = u32::try_from(vertices.len()) else {
                break;
            };
            let grade = grade_for_block(&color_values, block_index);
            block_order.push(block_index);
            for corner in block_corners(block_model, *block) {
                vertices.push(BlockModelVertex {
                    pos: (corner - scene_origin).as_vec3().to_array(),
                    grade,
                });
            }
            for (quad, visible) in QUADS.iter().zip(visible_faces) {
                if !visible {
                    continue;
                }
                indices.extend([
                    base + quad[0],
                    base + quad[1],
                    base + quad[2],
                    base + quad[0],
                    base + quad[2],
                    base + quad[3],
                ]);
            }
        }
        if let Some(gpu) = upload_block_model_surface_chunk(device, &vertices, &indices) {
            chunks.push(CachedBlockModelSurfaceChunk {
                gpu,
                cpu_vertices: vertices,
                block_order,
            });
        }
    }
    chunks
}

/// Grid-cell occupancy index, cell size, and lattice origin for a regular
/// block model's rendered blocks.
type RegularBlockGridIndex = (HashSet<(i32, i32, i32)>, DVec3, DVec3);

/// Builds a lattice-coordinate index of every rendered block in a regular
/// (non-sub-blocked) grid, for O(1) touching-neighbour lookups. Returns
/// `None` for sub-blocked (`is_irregular`) models, where block sizes vary
/// and a shared face can't be assumed just because two blocks touch.
fn regular_block_grid_index(block_model: &OpenBlockModel) -> Option<RegularBlockGridIndex> {
    let metadata = &block_model.model.metadata;
    if metadata.is_irregular {
        return None;
    }
    let [dim_x, dim_y, dim_z] = metadata.dims;
    if dim_x == 0 || dim_y == 0 || dim_z == 0 {
        return None;
    }
    let cell = DVec3::new(
        (metadata.upper.x - metadata.lower.x) / dim_x as f64,
        (metadata.upper.y - metadata.lower.y) / dim_y as f64,
        (metadata.upper.z - metadata.lower.z) / dim_z as f64,
    );
    if !(cell.x > 0.0 && cell.y > 0.0 && cell.z > 0.0) {
        return None;
    }
    let origin = metadata.lower;
    let mut grid = HashSet::with_capacity(block_model.renderable_block_indices.len());
    for &index in &block_model.renderable_block_indices {
        let Some(block) = block_model.blocks.get(index) else {
            continue;
        };
        // A block whose lower corner doesn't land cleanly on the lattice
        // (e.g. explicit bound variables that don't describe a uniform
        // grid) means we can't trust face-adjacency-by-index for this
        // model; bail out rather than risk culling a face that's really
        // visible.
        let coord = grid_coord(block.lower, origin, cell)?;
        grid.insert(coord);
    }
    Some((grid, cell, origin))
}

fn grid_coord(lower: DVec3, origin: DVec3, cell: DVec3) -> Option<(i32, i32, i32)> {
    let rel = (lower - origin) / cell;
    let rounded = rel.round();
    if (rel - rounded).abs().max_element() > 1e-3 {
        return None;
    }
    Some((rounded.x as i32, rounded.y as i32, rounded.z as i32))
}

fn block_model_style(block_model: &OpenBlockModel, translucent: bool) -> BlockModelStyleUniform {
    let has_grade = block_model
        .active_numeric_variable
        .as_deref()
        .and_then(|name| block_model.model.variable(name).map(|var| (name, var)))
        .and_then(|(name, var)| {
            let values = block_model.model.numeric_values(name).ok()?;
            let default = numeric_variable_default(var);
            render_value_range(&values, &block_model.renderable_block_indices, default)
        })
        .is_some();
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

fn normalized_grade(value: f64, default: Option<f64>, range: (f64, f64)) -> f32 {
    if !value.is_finite()
        || default.is_some_and(|default| (value - default).abs() < 1e-8)
        || value <= -90.0
    {
        return -1.0;
    }
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
        assert_eq!(grade_for_block(&None, 0), -1.0);
    }

    #[test]
    fn grade_for_block_is_neutral_for_an_out_of_range_index() {
        let color_values = Some((vec![1.0, 2.0], None, (0.0, 2.0)));
        assert_eq!(grade_for_block(&color_values, 5), -1.0);
    }

    #[test]
    fn grade_for_block_normalizes_within_the_render_range() {
        let color_values = Some((vec![0.0, 5.0, 10.0], None, (0.0, 10.0)));
        assert_eq!(grade_for_block(&color_values, 1), 0.5);
    }

    #[test]
    fn grid_coord_matches_points_exactly_on_the_lattice() {
        let origin = DVec3::new(100.0, -50.0, 0.0);
        let cell = DVec3::new(2.0, 2.0, 5.0);
        let lower = origin + DVec3::new(2.0 * 3.0, 2.0 * -4.0, 5.0 * 7.0);
        assert_eq!(grid_coord(lower, origin, cell), Some((3, -4, 7)));
    }

    #[test]
    fn grid_coord_rejects_points_off_the_lattice() {
        let origin = DVec3::ZERO;
        let cell = DVec3::new(2.0, 2.0, 2.0);
        // Off by a quarter cell on X: not a real grid boundary.
        let lower = DVec3::new(2.5, 4.0, 6.0);
        assert_eq!(grid_coord(lower, origin, cell), None);
    }

    #[test]
    fn grid_coord_tolerates_float_rounding_noise() {
        let origin = DVec3::ZERO;
        let cell = DVec3::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
        // Accumulated float error from repeated division, still "on" cell 5.
        let lower = DVec3::splat(5.0 * (1.0 / 3.0) + 1e-9);
        assert_eq!(grid_coord(lower, origin, cell), Some((5, 5, 5)));
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
}
