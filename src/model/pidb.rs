use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fs,
    io::{Cursor, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use dxf::{
    Color, Drawing, LwPolylineVertex, Point,
    entities::{
        Entity, EntityType, LwPolyline, ModelPoint, Polyline as DxfPolyline, Text as DxfText,
        Vertex as DxfVertex,
    },
    enums::AcadVersion,
};
use serde::{Deserialize, Serialize};

use crate::{
    model::{Document, FillStyle, LayerId, Object, ObjectColor, PolyVertex},
    rendering::color::{COLOR_TABLE, linear_to_srgb_byte},
    userspace_log,
};

pub(crate) const PIDB_FORMAT_VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PidbMetadata {
    pub(crate) name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct PidbFile {
    pub(crate) format_version: u32,
    pub(crate) document: Document,
    pub(crate) metadata: PidbMetadata,
}

#[derive(Clone, Debug)]
pub(crate) struct OpenProject {
    /// Stable identity for this open instance and the runtime namespace used
    /// by all of its layer/object ids. Namespace zero is reserved for disk.
    pub(crate) runtime_id: u32,
    pub(crate) path: Option<PathBuf>,
    pub(crate) pidb: PidbFile,
    /// Layers currently present in the shared scene. The PIDB remains the
    /// source of truth even while a layer is unloaded.
    pub(crate) loaded_layers: HashSet<LayerId>,
    /// Namespace-invariant content fingerprint of the complete PIDB at its
    /// last successful load/save. `None` means the project has never been
    /// saved. Replaces whole-document JSON snapshots: an interactive drag
    /// used to re-clone and re-serialize the full project per pointer event.
    saved_content_hash: Option<u64>,
    /// Per-object hash cache backing [`Document::content_hash`].
    content_hash_cache: RefCell<HashMap<crate::model::ObjectId, (u64, u64)>>,
    dirty_cache: RefCell<Option<PidbDirtyCache>>,
}

/// Cached whole-PIDB dirty result. Document mutations bump `revision`; the
/// other serialized PIDB fields are included directly in the cache key.
#[derive(Clone, Debug)]
struct PidbDirtyCache {
    revision: u64,
    format_version: u32,
    metadata: PidbMetadata,
    dirty: bool,
}

impl OpenProject {
    pub(crate) fn has_unsaved_changes(&self) -> bool {
        let Some(saved_content_hash) = self.saved_content_hash else {
            return true;
        };
        let revision = self.pidb.document.revision();
        if let Some(cache) = self.dirty_cache.borrow().as_ref()
            && cache.revision == revision
            && cache.format_version == self.pidb.format_version
            && cache.metadata == self.pidb.metadata
        {
            return cache.dirty;
        }

        let dirty = self.content_hash() != saved_content_hash;
        *self.dirty_cache.borrow_mut() = Some(PidbDirtyCache {
            revision,
            format_version: self.pidb.format_version,
            metadata: self.pidb.metadata.clone(),
            dirty,
        });
        dirty
    }

    /// Fingerprint of everything the PIDB file serializes, computed from the
    /// runtime document with cached per-object hashes — only objects touched
    /// since the previous call re-hash.
    fn content_hash(&self) -> u64 {
        use std::hash::{DefaultHasher, Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.pidb.format_version.hash(&mut hasher);
        self.pidb.metadata.name.hash(&mut hasher);
        self.pidb
            .document
            .content_hash(&mut self.content_hash_cache.borrow_mut())
            .hash(&mut hasher);
        hasher.finish()
    }

    pub(crate) fn current_content_hash(&self) -> u64 {
        self.content_hash()
    }

    /// Record the exact snapshot written by an asynchronous saver. If edits
    /// happened while it was writing, `has_unsaved_changes` compares the live
    /// content with this hash and correctly remains dirty.
    pub(crate) fn mark_snapshot_saved(&mut self, snapshot_hash: u64) {
        self.saved_content_hash = Some(snapshot_hash);
        self.dirty_cache = RefCell::new(None);
    }

    pub(crate) fn mark_saved(&mut self) {
        self.saved_content_hash = Some(self.content_hash());
        self.dirty_cache = RefCell::new(None);
    }
}

fn portable_pidb(pidb: &PidbFile) -> PidbFile {
    let mut portable = pidb.clone();
    portable.document.apply_runtime_namespace(0);
    portable
}

#[derive(Clone, Debug)]
pub(crate) struct Workspace {
    /// Every PIDB currently open and parsed in memory.
    pub(crate) projects: Vec<OpenProject>,
    /// The project receiving editing commands. Other projects remain open and
    /// may contribute loaded layers to the shared scene.
    pub(crate) active_index: Option<usize>,
    next_runtime_namespace: u32,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            projects: Vec::new(),
            active_index: None,
            next_runtime_namespace: 1,
        }
    }
}

impl Workspace {
    pub(crate) fn active_project(&self) -> Option<&OpenProject> {
        self.active_index.and_then(|index| self.projects.get(index))
    }

    pub(crate) fn active_project_mut(&mut self) -> Option<&mut OpenProject> {
        self.active_index
            .and_then(|index| self.projects.get_mut(index))
    }

    pub(crate) fn active_document(&self) -> Option<&Document> {
        self.active_project().map(|p| &p.pidb.document)
    }

    pub(crate) fn active_document_mut(&mut self) -> Option<&mut Document> {
        self.active_project_mut().map(|p| &mut p.pidb.document)
    }

    pub(crate) fn has_active_project(&self) -> bool {
        self.active_index
            .is_some_and(|index| index < self.projects.len())
    }

    pub(crate) fn project_index_for_path(&self, path: &Path) -> Option<usize> {
        self.projects
            .iter()
            .position(|project| project.path.as_deref() == Some(path))
    }

    pub(crate) fn project_index_for_runtime_id(&self, runtime_id: u32) -> Option<usize> {
        self.projects
            .iter()
            .position(|project| project.runtime_id == runtime_id)
    }

    pub(crate) fn project_index_for_object(
        &self,
        object_id: crate::model::ObjectId,
    ) -> Option<usize> {
        self.projects
            .iter()
            .position(|project| project.pidb.document.get_object(object_id).is_some())
    }

    pub(crate) fn project_index_for_layer(&self, layer_id: LayerId) -> Option<usize> {
        self.projects
            .iter()
            .position(|project| project.pidb.document.layer(layer_id).is_some())
    }

    /// Add a project without changing the editing target. Existing paths are
    /// deduplicated and return the already-open index.
    pub(crate) fn add_inactive(&mut self, mut project: OpenProject) -> usize {
        if let Some(path) = project.path.as_deref()
            && let Some(index) = self.project_index_for_path(path)
        {
            return index;
        }
        self.prepare_project(&mut project);
        self.projects.push(project);
        self.projects.len() - 1
    }

    /// Add a project and make it the editing target. Existing paths are
    /// activated rather than opened twice.
    pub(crate) fn add_and_activate(&mut self, project: OpenProject) -> usize {
        let index = self.add_inactive(project);
        self.active_index = Some(index);
        index
    }

    fn prepare_project(&mut self, project: &mut OpenProject) {
        let namespace = self.next_runtime_namespace;
        self.next_runtime_namespace = self.next_runtime_namespace.saturating_add(1);
        project.runtime_id = namespace;
        project.pidb.document.apply_runtime_namespace(namespace);
        project.loaded_layers.clear();
    }

    pub(crate) fn set_active_index(&mut self, index: usize) {
        if index < self.projects.len() {
            self.active_index = Some(index);
        }
    }

    /// Fingerprint of everything `scene_document()` reads: the active
    /// document's revision (bumped by every document mutation) and loaded-layer
    /// set. Equal keys guarantee an identical composite, letting callers skip
    /// the rebuild.
    pub(crate) fn composite_key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for project in &self.projects {
            project.runtime_id.hash(&mut hasher);
            project.pidb.document.revision().hash(&mut hasher);
            // Order-insensitive fold over the loaded-layer set.
            let mut layers_fold: u64 = 0;
            for layer in &project.loaded_layers {
                layers_fold ^= layer.0.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            }
            layers_fold.hash(&mut hasher);
            project.loaded_layers.len().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Build the composite document rendered and queried by the viewport.
    ///
    /// One pass per project's objects (bucketed by loaded layer) instead of
    /// rescanning every object once per loaded layer, and one index rebuild
    /// at the end instead of per-insert map maintenance. Draw order matches
    /// the incremental path: per project, loaded layers in layer order, each
    /// layer's objects in document order.
    pub(crate) fn scene_document(&self) -> Document {
        let mut scene = Document::new();
        for project in &self.projects {
            let document = &project.pidb.document;
            let mut per_layer: HashMap<LayerId, Vec<usize>> = HashMap::new();
            for (index, object) in document.objects().iter().enumerate() {
                if project.loaded_layers.contains(&object.layer()) {
                    per_layer.entry(object.layer()).or_default().push(index);
                }
            }
            for layer in document.layers() {
                if !project.loaded_layers.contains(&layer.id) {
                    continue;
                }
                let indices = per_layer.remove(&layer.id).unwrap_or_default();
                scene.append_layer_snapshot_unindexed(
                    layer,
                    indices.iter().map(|&index| {
                        let object = &document.objects()[index];
                        (object, document.object_revision(object.id()))
                    }),
                );
            }
        }
        scene.rebuild_object_index();
        scene
    }
}

pub(crate) fn new_empty(path: Option<PathBuf>) -> PidbFile {
    let document = Document::new();

    PidbFile {
        format_version: PIDB_FORMAT_VERSION,
        document,
        metadata: PidbMetadata {
            name: project_name(path.as_deref(), "Untitled.pidb"),
        },
    }
}

pub(crate) fn load(path: impl AsRef<Path>) -> Result<PidbFile> {
    let path = path.as_ref();
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut pidb: PidbFile =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    validate(&mut pidb)?;
    if pidb.metadata.name.trim().is_empty() {
        pidb.metadata.name = project_name(Some(path), "Untitled.pidb");
    }
    Ok(pidb)
}

pub(crate) fn save(path: impl AsRef<Path>, pidb: &PidbFile) -> Result<()> {
    let path = path.as_ref();
    let portable = portable_pidb(pidb);
    let json = serde_json::to_string_pretty(&portable)?;
    crate::model::atomic_file::write_atomic(path, |file| {
        file.write_all(json.as_bytes())
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    })
}

pub(crate) fn validate(pidb: &mut PidbFile) -> Result<()> {
    if pidb.format_version != PIDB_FORMAT_VERSION {
        bail!(
            "Unsupported PIDB format version {} (expected {})",
            pidb.format_version,
            PIDB_FORMAT_VERSION
        );
    }
    let repaired = pidb.document.repair_degenerate_closed_polylines();
    if repaired > 0 {
        userspace_log!("Reopened {repaired} degenerate two-vertex polygon(s) while loading");
    }
    // Enforce model invariants before runtime namespacing: duplicate or
    // out-of-range ids would otherwise silently alias distinct records once
    // ids are masked into the 32-bit local namespace.
    pidb.document.validate().context("invalid PIDB document")?;
    // Serialized counters are advisory only; derive them from the actual ids.
    pidb.document.recompute_id_counters();
    pidb.document.rebuild_object_index();
    Ok(())
}

/// Parse a DXF file into a standalone document without touching any project.
/// Lets multi-file imports parse everything up front and commit atomically.
pub(crate) fn parse_dxf_document(path: impl AsRef<Path>) -> Result<Document> {
    let path = path.as_ref();
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut cursor = Cursor::new(bytes.as_slice());
    let drawing =
        Drawing::load(&mut cursor).with_context(|| format!("parse {}", path.display()))?;
    let document = crate::model::formats::dxf::from_dxf(&drawing);
    document
        .validate()
        .with_context(|| format!("validate imported DXF {}", path.display()))?;
    Ok(document)
}

pub(crate) fn merge_document(target: &mut Document, imported: &Document) -> usize {
    let mut layer_map = HashMap::new();
    for layer in imported.layers() {
        let target_layer = target.layer_id_by_name(&layer.name).unwrap_or_else(|| {
            target.add_layer(
                layer.name.clone(),
                layer.color_index,
                layer.color,
                layer.visible,
                layer.elevation,
            )
        });
        layer_map.insert(layer.id, target_layer);
    }

    let mut added = 0;
    for object in imported.objects() {
        let layer = layer_map
            .get(&object.layer())
            .copied()
            .unwrap_or_else(|| target.ensure_default_layer());
        let id = target.allocate_object_id();
        target.insert_object(object.with_id_and_layer(id, layer));
        added += 1;
    }
    added
}

pub(crate) fn export_to_dxf(pidb: &PidbFile, path: impl AsRef<Path>) -> Result<()> {
    export_layers_to_dxf(pidb, None, path)
}

pub(crate) fn export_layer_to_dxf(
    pidb: &PidbFile,
    layer: LayerId,
    path: impl AsRef<Path>,
) -> Result<()> {
    export_layers_to_dxf(pidb, Some(layer), path)
}

fn export_layers_to_dxf(
    pidb: &PidbFile,
    only_layer: Option<LayerId>,
    path: impl AsRef<Path>,
) -> Result<()> {
    let mut drawing = Drawing::new();
    drawing.header.version = AcadVersion::R2004;
    for layer in pidb.document.layers() {
        if only_layer.is_some_and(|id| id != layer.id) {
            continue;
        }
        drawing.add_layer(dxf::tables::Layer {
            name: layer.name.clone(),
            color: Color::from_index(layer.color_index.unwrap_or(7).clamp(1, 255)),
            is_layer_on: layer.visible,
            ..Default::default()
        });
    }

    // Resolved lazily on the first road: junction pads, seam blending and arc
    // tessellation come from the same network resolve the editor draws from.
    let mut road_network: Option<crate::model::road_network::ResolvedNetwork> = None;
    for object in pidb.document.objects() {
        if only_layer.is_some_and(|id| id != object.layer()) {
            continue;
        }
        add_object_to_drawing(&pidb.document, object, &mut road_network, &mut drawing);
    }
    // Atomic replacement keeps the previous valid export intact when the
    // write fails part-way (disk full, permissions, encoding errors).
    crate::model::atomic_file::write_atomic(path.as_ref(), |file| {
        let mut writer = std::io::BufWriter::new(file);
        drawing.save(&mut writer)?;
        std::io::Write::flush(&mut writer)?;
        Ok(())
    })
    .with_context(|| format!("write {}", path.as_ref().display()))
}

fn add_object_to_drawing(
    document: &Document,
    object: &Object,
    road_network: &mut Option<crate::model::road_network::ResolvedNetwork>,
    drawing: &mut Drawing,
) {
    let Some(layer) = document.layer(object.layer()) else {
        return;
    };
    let layer_name = layer.name.clone();
    let object_color = object.color();

    match object {
        Object::Point { pos, .. } => {
            let mut entity = Entity::new(EntityType::ModelPoint(ModelPoint {
                location: point_from_vec3(*pos),
                ..Default::default()
            }));
            entity.common.layer = layer_name;
            apply_object_color(&mut entity.common, object_color);
            drawing.add_entity(entity);
        }
        Object::Polyline { verts, closed, .. } => {
            if verts.len() < 2 {
                return;
            }
            let z0 = verts[0].pos.z;
            let is_flat = verts.iter().all(|v| (v.pos.z - z0).abs() < 1e-9);
            if is_flat {
                // All vertices share the same elevation — use the compact 2D form.
                let mut poly = LwPolyline {
                    vertices: verts.iter().map(lw_vertex_from_poly_vertex).collect(),
                    ..Default::default()
                };
                poly.set_is_closed(*closed);
                let mut entity = Entity::new(EntityType::LwPolyline(poly));
                entity.common.layer = layer_name;
                apply_object_color(&mut entity.common, object_color);
                entity.common.elevation = z0;
                drawing.add_entity(entity);
            } else {
                // DXF 3D polyline vertices do not portably support bulge.
                // Tessellate bulged segments (including interpolated Z) into
                // straight 3D vertices before export.
                let mut poly = DxfPolyline::default();
                poly.set_is_3d_polyline(true);
                poly.set_is_closed(*closed);
                let positions: Vec<_> =
                    if verts.iter().any(|vertex| vertex.bulge.abs() > f64::EPSILON) {
                        crate::model::geometry::tessellate_polyline_bulges(verts, *closed)
                    } else {
                        verts.iter().map(|vertex| vertex.pos).collect()
                    };
                for pos in positions {
                    let mut vertex = DxfVertex {
                        location: Point::new(pos.x, pos.y, pos.z),
                        bulge: 0.0,
                        ..Default::default()
                    };
                    vertex.set_is_3d_polyline_vertex(true);
                    poly.add_vertex(drawing, vertex);
                }
                let mut entity = Entity::new(EntityType::Polyline(poly));
                entity.common.layer = layer_name;
                apply_object_color(&mut entity.common, object_color);
                drawing.add_entity(entity);
            }
        }
        Object::Text {
            pos,
            content,
            height,
            rotation,
            ..
        } => {
            let mut entity = Entity::new(EntityType::Text(DxfText {
                location: point_from_vec3(*pos),
                text_height: *height,
                value: content.clone(),
                rotation: rotation.to_degrees(),
                ..Default::default()
            }));
            entity.common.layer = layer_name;
            apply_object_color(&mut entity.common, object_color);
            drawing.add_entity(entity);
        }
        Object::Road { id, .. } => {
            use crate::model::road_network::{self, RoadKey};
            let network = road_network.get_or_insert_with(|| road_network::resolve(document, None));
            for edge in network.edges_for(RoadKey::Object(*id)) {
                emit_road_points(&edge.center, &layer_name, object_color, drawing);
                emit_road_points(&edge.left, &layer_name, object_color, drawing);
                emit_road_points(&edge.right, &layer_name, object_color, drawing);
                if edge.start_cap
                    && let (Some(&l), Some(&r)) = (edge.left.first(), edge.right.first())
                {
                    emit_road_points(&[l, r], &layer_name, object_color, drawing);
                }
                if edge.end_cap
                    && let (Some(&l), Some(&r)) = (edge.left.last(), edge.right.last())
                {
                    emit_road_points(&[l, r], &layer_name, object_color, drawing);
                }
            }
        }
    }
}

fn emit_road_points(
    points: &[glam::DVec3],
    layer_name: &str,
    color: ObjectColor,
    drawing: &mut Drawing,
) {
    let verts: Vec<PolyVertex> = points.iter().copied().map(PolyVertex::straight).collect();
    emit_road_verts(verts, layer_name, color, drawing);
}

fn emit_road_verts(
    verts: Vec<PolyVertex>,
    layer_name: &str,
    color: ObjectColor,
    drawing: &mut Drawing,
) {
    if verts.len() < 2 {
        return;
    }
    let z0 = verts[0].pos.z;
    let is_flat = verts.iter().all(|v| (v.pos.z - z0).abs() < 1e-9);
    if is_flat {
        let mut poly = LwPolyline {
            vertices: verts.iter().map(lw_vertex_from_poly_vertex).collect(),
            ..Default::default()
        };
        poly.set_is_closed(false);
        let mut entity = Entity::new(EntityType::LwPolyline(poly));
        entity.common.layer = layer_name.to_owned();
        apply_object_color(&mut entity.common, color);
        entity.common.elevation = z0;
        drawing.add_entity(entity);
    } else {
        let mut poly = DxfPolyline::default();
        poly.set_is_3d_polyline(true);
        poly.set_is_closed(false);
        for v in &verts {
            let mut vertex = DxfVertex {
                location: Point::new(v.pos.x, v.pos.y, v.pos.z),
                bulge: v.bulge,
                ..Default::default()
            };
            vertex.set_is_3d_polyline_vertex(true);
            poly.add_vertex(drawing, vertex);
        }
        let mut entity = Entity::new(EntityType::Polyline(poly));
        entity.common.layer = layer_name.to_owned();
        apply_object_color(&mut entity.common, color);
        drawing.add_entity(entity);
    }
}

fn lw_vertex_from_poly_vertex(vertex: &PolyVertex) -> LwPolylineVertex {
    LwPolylineVertex {
        x: vertex.pos.x,
        y: vertex.pos.y,
        bulge: vertex.bulge,
        ..Default::default()
    }
}

fn point_from_vec3(pos: glam::DVec3) -> Point {
    Point::new(pos.x, pos.y, pos.z)
}

fn apply_object_color(common: &mut dxf::entities::EntityCommon, color: ObjectColor) {
    match color {
        ObjectColor::ByLayer => {
            common.color = Color::by_layer();
            common.color_24_bit = 0;
            common.transparency = 0;
        }
        ObjectColor::Fixed(rgba) => {
            common.color = Color::from_index(nearest_aci(rgba));
            let red = u32::from(linear_to_srgb_byte(rgba[0]));
            let green = u32::from(linear_to_srgb_byte(rgba[1]));
            let blue = u32::from(linear_to_srgb_byte(rgba[2]));
            common.color_24_bit = ((red << 16) | (green << 8) | blue) as i32;
            let alpha = (rgba[3].clamp(0.0, 1.0) * 255.0).round() as i32;
            common.transparency = 0x0200_0000 | alpha;
        }
    }
}

fn nearest_aci(rgba: [f32; 4]) -> u8 {
    let target = [
        linear_to_srgb_byte(rgba[0]) as i32,
        linear_to_srgb_byte(rgba[1]) as i32,
        linear_to_srgb_byte(rgba[2]) as i32,
    ];
    (1..=255)
        .min_by_key(|index| {
            let rgb = COLOR_TABLE[*index as usize];
            let dr = target[0] - i32::from(rgb[0]);
            let dg = target[1] - i32::from(rgb[1]);
            let db = target[2] - i32::from(rgb[2]);
            dr * dr + dg * dg + db * db
        })
        .unwrap_or(7)
}

fn project_name(path: Option<&Path>, fallback: &str) -> String {
    path.and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .unwrap_or(fallback)
        .to_string()
}

pub(crate) fn pidb_from_dxf_path(path: impl AsRef<Path>) -> Result<PidbFile> {
    let path = path.as_ref();
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut cursor = Cursor::new(bytes);
    let drawing = Drawing::load(&mut cursor)?;
    let document = crate::model::formats::dxf::from_dxf(&drawing);
    Ok(PidbFile {
        format_version: PIDB_FORMAT_VERSION,
        document,
        metadata: PidbMetadata {
            name: project_name(Some(path), "Imported.pidb"),
        },
    })
}

pub(crate) fn pidb_from_dgd_isis(path: impl AsRef<Path>) -> Result<PidbFile> {
    let path = path.as_ref();
    let design = crate::model::formats::isis::read_dgd_design(path)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
    let index_entries = match crate::model::formats::isis::same_stem_isix_path(path) {
        Some(index_path) => {
            let entries = crate::model::formats::isis::read_dgd_index(&index_path)
                .map_err(|e| anyhow::anyhow!("reading {}: {e}", index_path.display()))?;
            userspace_log!(
                "Loaded DGD index sidecar {} ({} current layer entries)",
                index_path.display(),
                entries.len()
            );
            entries
        }
        None => {
            userspace_log!(
                "No DGD index sidecar found next to {}; empty DGD layers may be omitted",
                path.display()
            );
            Vec::new()
        }
    };
    let mut document = document_from_dgd_points(
        &design.points,
        &design.texts,
        &index_entries,
        &design.layer_names,
        design.palette.as_ref(),
    );
    document.repair_degenerate_closed_polylines();
    Ok(PidbFile {
        format_version: PIDB_FORMAT_VERSION,
        document,
        metadata: PidbMetadata {
            name: project_name(Some(path), "Imported.pidb"),
        },
    })
}

pub(crate) fn pidb_from_duf(path: impl AsRef<Path>) -> Result<PidbFile> {
    let path = path.as_ref();
    let duf = crate::model::formats::duf::read_duf(path)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
    let mut document = document_from_duf_design(&duf);
    document.repair_degenerate_closed_polylines();
    if duf.skipped.unsupported_mesh_entities > 0 || !duf.polyfaces.is_empty() {
        userspace_log!(
            "Ignored {} DUF mesh entit{} while importing {} as PIDB",
            duf.polyfaces
                .len()
                .max(duf.skipped.unsupported_mesh_entities),
            if duf
                .polyfaces
                .len()
                .max(duf.skipped.unsupported_mesh_entities)
                == 1
            {
                "y"
            } else {
                "ies"
            },
            path.display()
        );
    }
    Ok(PidbFile {
        format_version: PIDB_FORMAT_VERSION,
        document,
        metadata: PidbMetadata {
            name: project_name(Some(path), "Imported.pidb"),
        },
    })
}

fn document_from_duf_design(duf: &crate::model::formats::duf::DufData) -> Document {
    let mut doc = Document::new();
    let mut layer_ids: HashMap<String, LayerId> = HashMap::new();

    let mut layer_id_for = |doc: &mut Document, name: &str| {
        *layer_ids.entry(name.to_owned()).or_insert_with(|| {
            doc.add_layer(name.to_owned(), None, [1.0, 1.0, 1.0, 1.0], true, 0.0)
        })
    };

    for point in &duf.points {
        let layer = layer_id_for(&mut doc, &point.layer_name);
        doc.add_object(|id| Object::Point {
            id,
            layer,
            pos: point.position,
            color: ObjectColor::ByLayer,
        });
    }

    for polyline in &duf.polylines {
        if polyline.vertices.len() < 2 {
            continue;
        }
        let layer = layer_id_for(&mut doc, &polyline.layer_name);
        let verts = polyline
            .vertices
            .iter()
            .copied()
            .map(PolyVertex::straight)
            .collect();
        doc.add_object(|id| Object::Polyline {
            id,
            layer,
            verts,
            closed: false,
            color: ObjectColor::ByLayer,
            fill: FillStyle::Clear,
            line_weight: 1.0,
        });
    }

    doc
}

fn document_from_dgd_points(
    points: &[crate::model::formats::isis::DesignPoint],
    texts: &[crate::model::formats::isis::DesignText],
    index_entries: &[crate::model::formats::isis::DesignIndexEntry],
    _embedded_layer_names: &[String],
    palette: Option<&crate::model::formats::isis::DgdColorTable>,
) -> Document {
    use crate::model::formats::isis::{DGD_COORD_RECORD_LEN, DesignGeometryKind};
    use glam::DVec3;

    let mut doc = Document::new();
    let mut layer_ids: HashMap<String, LayerId> = HashMap::new();
    let layer_resolver = DgdLayerResolver::new(index_entries);

    let mut layer_id_for = |doc: &mut Document, name: &str| {
        *layer_ids.entry(name.to_owned()).or_insert_with(|| {
            doc.add_layer(name.to_owned(), None, [1.0, 1.0, 1.0, 1.0], true, 0.0)
        })
    };

    for entry in index_entries {
        if let Some(layer_name) = dgd_index_layer_name(&entry.name) {
            layer_id_for(&mut doc, layer_name);
        }
    }

    let mut current_layer = None;
    let mut current_layer_name: Option<String> = None;
    let mut current_geometry = DesignGeometryKind::Unknown;
    let mut current_closed = false;
    let mut current_color_index: Option<u8> = None;
    let mut previous_offset = None;
    let mut current_verts: Vec<PolyVertex> = Vec::new();

    fn finish_segment(
        doc: &mut Document,
        layer_id: Option<LayerId>,
        geometry_kind: DesignGeometryKind,
        closed: bool,
        color_index: Option<u8>,
        palette: Option<&crate::model::formats::isis::DgdColorTable>,
        verts: &mut Vec<PolyVertex>,
    ) {
        let Some(layer_id) = layer_id else {
            verts.clear();
            return;
        };
        let color = dgd_object_color(color_index, palette);
        if geometry_kind == DesignGeometryKind::Point {
            for vertex in verts.drain(..) {
                add_dgd_point(doc, layer_id, vertex.pos, color);
            }
            return;
        }
        match verts.len() {
            0 => {}
            1 => add_dgd_point(doc, layer_id, verts[0].pos, color),
            _ => add_dgd_polyline(doc, layer_id, std::mem::take(verts), closed, color),
        }
        verts.clear();
    }

    for point in points {
        let vertex = PolyVertex::straight(DVec3::new(point.x, point.y, point.z));
        let has_record_gap =
            previous_offset.is_some_and(|offset| point.offset != offset + DGD_COORD_RECORD_LEN);
        if point.seg_type == 0 || current_layer.is_none() || has_record_gap {
            finish_segment(
                &mut doc,
                current_layer,
                current_geometry,
                current_closed,
                current_color_index,
                palette,
                &mut current_verts,
            );
            current_closed = point.closed;
            current_color_index = point.color_index;
            let layer_name = point
                .layer_name
                .as_deref()
                .or_else(|| layer_resolver.layer_name_at(point.offset))
                .or_else(|| dgd_candidate_layer_name(&point.name))
                .or_else(|| dgd_candidate_layer_name(&point.secondary_name))
                .map(str::to_owned);
            if let Some(layer_name) = layer_name {
                current_layer = Some(layer_id_for(&mut doc, &layer_name));
                current_geometry = dgd_segment_geometry_kind(point.geometry_kind, &layer_name);
                current_layer_name = Some(layer_name);
            } else if current_layer.is_some() && !has_record_gap {
                let layer_name = current_layer_name.as_deref().unwrap_or("DGD Import");
                current_geometry = dgd_segment_geometry_kind(point.geometry_kind, layer_name);
            } else {
                let layer_name = "DGD Import";
                current_layer = Some(layer_id_for(&mut doc, layer_name));
                current_geometry = dgd_segment_geometry_kind(point.geometry_kind, layer_name);
                current_layer_name = Some(layer_name.to_owned());
            }
        }
        current_verts.push(vertex);
        previous_offset = Some(point.offset);
    }

    finish_segment(
        &mut doc,
        current_layer,
        current_geometry,
        current_closed,
        current_color_index,
        palette,
        &mut current_verts,
    );

    for text in texts {
        let layer_name = text
            .layer_name
            .as_deref()
            .or_else(|| layer_resolver.layer_name_at(text.offset))
            .unwrap_or("DGD Import");
        let layer_id = layer_id_for(&mut doc, layer_name);
        doc.add_object(|id| Object::Text {
            id,
            layer: layer_id,
            pos: DVec3::new(text.x, text.y, text.z),
            content: text.content.clone(),
            height: text.height,
            rotation: text.rotation_degrees.to_radians(),
            color: dgd_object_color(text.color_index, palette),
        });
    }

    doc
}

/// Resolve a Vulcan object colour index to an [`ObjectColor`]. The database's
/// embedded `dig$colour256` palette takes precedence; indices it does not
/// define fall back to the built-in default palette
/// ([`crate::rendering::color::vulcan_color_to_linear_rgba`]), and a blank or
/// unmapped index resolves to `ByLayer`.
fn dgd_object_color(
    color_index: Option<u8>,
    palette: Option<&crate::model::formats::isis::DgdColorTable>,
) -> ObjectColor {
    let Some(index) = color_index else {
        return ObjectColor::ByLayer;
    };
    palette
        .and_then(|palette| palette.rgb(index))
        .map(crate::rendering::color::rgb_bytes_to_linear_rgba)
        .or_else(|| crate::rendering::color::vulcan_color_to_linear_rgba(index))
        .map_or(ObjectColor::ByLayer, ObjectColor::Fixed)
}

fn add_dgd_point(doc: &mut Document, layer_id: LayerId, pos: glam::DVec3, color: ObjectColor) {
    doc.add_object(|id| Object::Point {
        id,
        layer: layer_id,
        pos,
        color,
    });
}

fn add_dgd_polyline(
    doc: &mut Document,
    layer_id: LayerId,
    verts: Vec<PolyVertex>,
    closed: bool,
    color: ObjectColor,
) {
    doc.add_object(|id| Object::Polyline {
        id,
        layer: layer_id,
        verts,
        closed,
        color,
        fill: FillStyle::Clear,
        line_weight: 1.0,
    });
}

struct DgdLayerResolver<'a> {
    entries: Vec<&'a crate::model::formats::isis::DesignIndexEntry>,
}

impl<'a> DgdLayerResolver<'a> {
    fn new(entries: &'a [crate::model::formats::isis::DesignIndexEntry]) -> Self {
        let mut entries: Vec<_> = entries.iter().collect();
        entries.sort_by_key(|entry| entry.offset);
        Self { entries }
    }

    fn layer_name_at(&self, offset: usize) -> Option<&'a str> {
        let index = self.entries.partition_point(|entry| entry.offset <= offset);
        [
            index
                .checked_sub(1)
                .and_then(|index| self.entries.get(index)),
            self.entries.get(index),
        ]
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            dgd_index_layer_name(&entry.name).map(|name| {
                let distance = entry.offset.abs_diff(offset);
                (distance, name)
            })
        })
        .min_by_key(|(distance, _)| *distance)
        .map(|(_, name)| name)
    }
}

fn dgd_segment_geometry_kind(
    geometry_kind: crate::model::formats::isis::DesignGeometryKind,
    layer_name: &str,
) -> crate::model::formats::isis::DesignGeometryKind {
    if geometry_kind == crate::model::formats::isis::DesignGeometryKind::Unknown
        && is_dgd_point_collection_layer_name(layer_name)
    {
        crate::model::formats::isis::DesignGeometryKind::Point
    } else {
        geometry_kind
    }
}

fn dgd_candidate_layer_name(raw_name: &str) -> Option<&str> {
    let name = raw_name.trim();
    if crate::model::formats::isis::is_dgd_meaningful_layer_name(name) {
        Some(name)
    } else {
        None
    }
}

fn dgd_index_layer_name(raw_name: &str) -> Option<&str> {
    let name = raw_name.trim();
    if crate::model::formats::isis::is_dgd_index_layer_name(name) {
        Some(name)
    } else {
        None
    }
}

fn is_dgd_point_collection_layer_name(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    upper == "POINTS"
        || upper == "REFERENCE_POINTS"
        || upper.ends_with("_PTS")
        || upper.ends_with("POINTS")
}

pub(crate) fn open_project(path: Option<PathBuf>, mut pidb: PidbFile) -> Result<OpenProject> {
    validate(&mut pidb)?;
    let mut project = OpenProject {
        runtime_id: 0,
        path,
        pidb,
        loaded_layers: HashSet::new(),
        saved_content_hash: None,
        content_hash_cache: RefCell::new(HashMap::new()),
        dirty_cache: RefCell::new(None),
    };
    // A pathless project has never been saved and stays dirty; the hash is
    // namespace-invariant, so capturing it before runtime namespacing is fine.
    if project.path.is_some() {
        project.mark_saved();
    }
    Ok(project)
}
