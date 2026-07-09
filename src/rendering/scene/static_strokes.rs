//! Persistent GPU cache for stroke geometry of stable polylines.
//!
//! The per-rebuild scene builder re-tessellates every object on any document
//! change, which is fine for hand-drawn documents but ruinous once an
//! operation (contouring, string imports) drops tens of thousands of dense
//! polylines into the document. This cache claims those polylines, groups them
//! into per-layer chunks with their own GPU buffers, and re-tessellates only
//! the chunks whose members actually changed. Everything else — points, text,
//! roads, filled polygons, and any polyline the editor is currently styling
//! (selection, highlight, translucency) — stays on the per-rebuild path.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use glam::DVec3;

use crate::{
    model::{Document, FillStyle, LayerId, Object, ObjectId, SceneEntityId},
    rendering::{
        StrokeVertex, Vertex,
        geometry::{DrawContext, polyline_segments, tessellate_polyline_stroke},
        pick::PickRecord,
    },
    ui::state::EditorState,
};

/// Soft per-chunk stroke-vertex budget (~27 MiB of vertex data). Small enough
/// that re-tessellating one chunk (an edited or newly selected member) stays
/// within a frame; large enough that contour-scale documents need only dozens
/// of draw calls.
const CHUNK_VERTEX_BUDGET: usize = 512 * 1024;

pub(crate) struct StaticStrokeChunk {
    layer: LayerId,
    /// Mirrors the layer's visibility each sync so draw and pick can skip the
    /// chunk without rebuilding anything when a layer is toggled.
    pub(crate) layer_visible: bool,
    members: Vec<ObjectId>,
    /// Estimated stroke vertices including members assigned since the last
    /// rebuild; used only to decide when to start a new chunk.
    estimated_vertices: usize,
    dirty: bool,
    /// CPU copies are retained for picking (cursor pick and box select read
    /// vertex positions back).
    pub(crate) vertices: Vec<StrokeVertex>,
    pub(crate) indices: Vec<u32>,
    /// Per-member pick records with ranges into this chunk's buffers (fill
    /// ranges are always empty — filled polylines are ineligible).
    pub(crate) records: Vec<PickRecord>,
    pub(crate) vertex_gpu: Option<wgpu::Buffer>,
    pub(crate) index_gpu: Option<wgpu::Buffer>,
    vertex_capacity: usize,
    index_capacity: usize,
    pub(crate) index_count: u32,
}

impl StaticStrokeChunk {
    fn new(layer: LayerId) -> Self {
        Self {
            layer,
            layer_visible: true,
            members: Vec::new(),
            estimated_vertices: 0,
            dirty: false,
            vertices: Vec::new(),
            indices: Vec::new(),
            records: Vec::new(),
            vertex_gpu: None,
            index_gpu: None,
            vertex_capacity: 0,
            index_capacity: 0,
            index_count: 0,
        }
    }

    pub(crate) fn drawable(&self) -> bool {
        self.layer_visible && self.index_count > 0
    }
}

#[derive(Default)]
pub(crate) struct StaticStrokeCache {
    chunks: Vec<StaticStrokeChunk>,
    /// Chunk index and last-built fingerprint per claimed object.
    object_chunk: HashMap<ObjectId, (usize, u64)>,
    claimed: HashSet<ObjectId>,
    cached_scene_origin: DVec3,
    cached_scale_factor: f32,
}

/// Everything the cache bakes into vertices besides the object's own data.
/// A mismatch forces the member's chunk to rebuild.
fn fingerprint(object_revision: u64, rgba: [f32; 4], layer: LayerId) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    object_revision.hash(&mut hasher);
    rgba.map(f32::to_bits).hash(&mut hasher);
    layer.hash(&mut hasher);
    hasher.finish()
}

/// Whether the cache may own this object's stroke geometry. Anything the
/// editor is currently restyling stays on the per-rebuild path, which already
/// implements recoloring, fills, and draw-on-top ordering for it.
fn eligible(object: &Object, editor: &EditorState) -> bool {
    let Object::Polyline {
        verts,
        closed,
        fill,
        ..
    } = object
    else {
        return false;
    };
    if *closed && verts.len() >= 3 && *fill != FillStyle::Clear {
        return false;
    }
    let handle = SceneEntityId::Object(object.id());
    !(editor.hidden_handles.contains(&handle)
        || editor.frozen_handles.contains(&handle)
        || editor.selected_handles.contains(&handle)
        || editor.translucent_handles.contains(&handle)
        || editor.tri_hover_handles.contains(&handle)
        || editor.tool_highlight_id == Some(object.id()))
}

/// Rough stroke-vertex count for chunk assignment: 4 per segment plus a round
/// join allowance. Only steers the budget split, so precision is unimportant.
fn estimate_vertices(verts_len: usize, closed: bool) -> usize {
    let segments = verts_len.saturating_sub(1) + usize::from(closed);
    segments * 4 + verts_len * 2
}

impl StaticStrokeCache {
    /// Object ids whose stroke geometry this cache owns; the scene builder
    /// must skip them.
    pub(crate) fn claimed(&self) -> &HashSet<ObjectId> {
        &self.claimed
    }

    pub(crate) fn chunks(&self) -> &[StaticStrokeChunk] {
        &self.chunks
    }

    /// Reconcile the cache with the current document and editor state,
    /// re-tessellating and re-uploading only chunks with changed members.
    /// Runs on every geometry rebuild, so per-object work here must stay
    /// cheap (a fingerprint compare) for unchanged objects.
    pub(crate) fn sync(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        document: &Document,
        editor: &EditorState,
        scene_origin: DVec3,
        scale_factor: f32,
    ) {
        // Origin and scale factor are baked into every vertex.
        if scene_origin != self.cached_scene_origin
            || (scale_factor - self.cached_scale_factor).abs() > f32::EPSILON
        {
            self.chunks.clear();
            self.object_chunk.clear();
            self.cached_scene_origin = scene_origin;
            self.cached_scale_factor = scale_factor;
        }

        self.claimed.clear();
        for object in document.objects() {
            if !eligible(object, editor) {
                continue;
            }
            let id = object.id();
            let rgba = document.object_rgba(object);
            let fp = fingerprint(document.object_revision(id), rgba, object.layer());
            match self.object_chunk.get(&id).copied() {
                Some((chunk_index, cached_fp)) => {
                    let moved_layer = self.chunks[chunk_index].layer != object.layer();
                    if moved_layer {
                        self.remove_member(id, chunk_index);
                        self.assign(object, id);
                    } else if cached_fp != fp {
                        self.chunks[chunk_index].dirty = true;
                    }
                }
                None => self.assign(object, id),
            }
            self.claimed.insert(id);
        }

        // Members that vanished or became ineligible release their chunk.
        let stale: Vec<(ObjectId, usize)> = self
            .object_chunk
            .iter()
            .filter(|(id, _)| !self.claimed.contains(id))
            .map(|(&id, &(chunk_index, _))| (id, chunk_index))
            .collect();
        for (id, chunk_index) in stale {
            self.remove_member(id, chunk_index);
            self.object_chunk.remove(&id);
        }

        for chunk in &mut self.chunks {
            chunk.layer_visible = document
                .layer(chunk.layer)
                .map(|layer| layer.visible)
                .unwrap_or(true);
        }

        for chunk_index in 0..self.chunks.len() {
            if self.chunks[chunk_index].dirty {
                self.rebuild_chunk(
                    chunk_index,
                    device,
                    queue,
                    document,
                    scene_origin,
                    scale_factor,
                );
            }
        }
    }

    fn assign(&mut self, object: &Object, id: ObjectId) {
        let Object::Polyline { verts, closed, .. } = object else {
            return;
        };
        let estimate = estimate_vertices(verts.len(), *closed);
        let layer = object.layer();
        let chunk_index = self
            .chunks
            .iter()
            .position(|chunk| {
                chunk.layer == layer && chunk.estimated_vertices < CHUNK_VERTEX_BUDGET
            })
            .unwrap_or_else(|| {
                self.chunks.push(StaticStrokeChunk::new(layer));
                self.chunks.len() - 1
            });
        let chunk = &mut self.chunks[chunk_index];
        chunk.members.push(id);
        chunk.estimated_vertices += estimate;
        chunk.dirty = true;
        // The real fingerprint is stored when the chunk rebuilds.
        self.object_chunk.insert(id, (chunk_index, 0));
    }

    fn remove_member(&mut self, id: ObjectId, chunk_index: usize) {
        let chunk = &mut self.chunks[chunk_index];
        chunk.members.retain(|member| *member != id);
        chunk.dirty = true;
    }

    fn rebuild_chunk(
        &mut self,
        chunk_index: usize,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        document: &Document,
        scene_origin: DVec3,
        scale_factor: f32,
    ) {
        let chunk = &mut self.chunks[chunk_index];
        chunk.vertices.clear();
        chunk.indices.clear();
        chunk.records.clear();

        // Eligible polylines never emit fill geometry; these stay empty.
        let mut unused_fill_vertices: Vec<Vertex> = Vec::new();
        let mut unused_fill_indices: Vec<u32> = Vec::new();

        for member_index in 0..chunk.members.len() {
            let id = chunk.members[member_index];
            let Some(object) = document.get_object(id) else {
                continue;
            };
            let Object::Polyline {
                verts,
                closed,
                line_weight,
                ..
            } = object
            else {
                continue;
            };
            let rgba = document.object_rgba(object);
            let stroke_start = chunk.vertices.len() as u32;
            let stroke_index_start = chunk.indices.len() as u32;
            {
                let mut draw_ctx = DrawContext {
                    stroke_vertex_buf: &mut chunk.vertices,
                    stroke_index_buf: &mut chunk.indices,
                    fill_vertex_buf: &mut unused_fill_vertices,
                    fill_index_buf: &mut unused_fill_indices,
                    scene_origin,
                    scale_factor,
                };
                tessellate_polyline_stroke(&mut draw_ctx, verts, *closed, *line_weight, rgba);
            }
            let stroke_end = chunk.vertices.len() as u32;
            if stroke_end > stroke_start {
                chunk.records.push(PickRecord {
                    entity: SceneEntityId::Object(id),
                    stroke_range: (stroke_start, stroke_end),
                    stroke_index_range: (stroke_index_start, chunk.indices.len() as u32),
                    fill_range: (0, 0),
                    fill_index_range: (0, 0),
                    segments: polyline_segments(verts, *closed),
                });
            }
            let fp = fingerprint(document.object_revision(id), rgba, object.layer());
            self.object_chunk.insert(id, (chunk_index, fp));
        }

        chunk.estimated_vertices = chunk.vertices.len();
        chunk.index_count = chunk.indices.len() as u32;
        chunk.dirty = false;

        upload(
            device,
            queue,
            &mut chunk.vertex_gpu,
            &mut chunk.vertex_capacity,
            bytemuck::cast_slice(&chunk.vertices),
            wgpu::BufferUsages::VERTEX,
            "Static Stroke Chunk Vertex Buffer",
        );
        upload(
            device,
            queue,
            &mut chunk.index_gpu,
            &mut chunk.index_capacity,
            bytemuck::cast_slice(&chunk.indices),
            wgpu::BufferUsages::INDEX,
            "Static Stroke Chunk Index Buffer",
        );
    }
}

/// (Re)create `buffer` if `data` outgrew it, then write `data`.
fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &mut Option<wgpu::Buffer>,
    capacity_bytes: &mut usize,
    data: &[u8],
    usage: wgpu::BufferUsages,
    label: &'static str,
) {
    if data.is_empty() {
        return;
    }
    if buffer.is_none() || data.len() > *capacity_bytes {
        *capacity_bytes = data.len().next_power_of_two();
        *buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: *capacity_bytes as wgpu::BufferAddress,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }
    if let Some(buffer) = buffer {
        queue.write_buffer(buffer, 0, data);
    }
}
