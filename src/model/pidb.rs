use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    io::{Cursor, Write},
    path::{Path, PathBuf},
    time::SystemTime,
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
    model::{Document, FillStyle, Layer, LayerId, Object, ObjectColor, ObjectId, PolyVertex},
    rendering::color::{COLOR_TABLE, linear_to_srgb_byte},
    userspace_log,
};

pub(crate) const PIDB_FORMAT_VERSION: u32 = 2;

#[derive(Clone, Debug, Serialize, Deserialize)]
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
    /// Stable identity for this open instance. Unlike its vector index, this
    /// does not change when another project is closed or projects are sorted.
    pub(crate) runtime_id: u32,
    pub(crate) path: Option<PathBuf>,
    pub(crate) pidb: PidbFile,
    /// Layers currently present in the shared scene. The PIDB remains the
    /// source of truth even while a layer is unloaded.
    pub(crate) loaded_layers: HashSet<LayerId>,
    dirty_layers_cache: RefCell<Option<DirtyLayersCache>>,
    /// Parsed-and-serialized snapshot of the on-disk file, keyed by its
    /// modification time. The disk contents only change when we save, so this
    /// spares `dirty_layers()` from re-reading and re-serializing the file on
    /// every in-memory edit (it runs per render frame).
    disk_snapshot_cache: RefCell<Option<DiskSnapshot>>,
    /// Serialized view of the in-memory document, updated incrementally via
    /// per-object revisions so a revision bump (e.g. every frame of a vertex
    /// drag) only re-serializes the objects that edit actually touched.
    memory_snapshot_cache: RefCell<MemorySnapshot>,
}

/// Cached `dirty_layers()` result, valid for one (document revision, on-disk
/// mtime) pair.
#[derive(Clone, Debug)]
struct DirtyLayersCache {
    revision: u64,
    mtime: Option<SystemTime>,
    layers: HashSet<LayerId>,
}

/// Normalized (namespace-0, parsed-then-serialized) view of an on-disk PIDB,
/// grouped by local layer id, used to diff against in-memory state.
#[derive(Clone, Debug)]
struct DiskSnapshot {
    mtime: Option<std::time::SystemTime>,
    layers: HashMap<LayerId, String>,
    objects: HashMap<LayerId, BTreeMap<u64, String>>,
}

/// Namespace-0 serialization of the in-memory document, mirroring
/// `DiskSnapshot`'s shape. Kept in sync incrementally: only objects whose
/// per-object revision moved since the last sync are re-serialized, so
/// interactive edits on large documents don't pay a full-document clone and
/// JSON pass per frame.
#[derive(Clone, Debug, Default)]
struct MemorySnapshot {
    /// Per runtime object id: the revision its cached JSON reflects and the
    /// local-layer bucket it was filed under (to relocate on layer moves).
    object_state: HashMap<ObjectId, (u64, LayerId)>,
    layers: HashMap<LayerId, String>,
    objects: HashMap<LayerId, BTreeMap<u64, String>>,
}

impl MemorySnapshot {
    fn sync(&mut self, document: &Document) {
        const LOCAL_MASK: u64 = u32::MAX as u64;

        // Layers are few — reserialize them all at their local ids.
        self.layers = document
            .layers()
            .iter()
            .map(|layer| {
                let local_id = LayerId(layer.id.0 & LOCAL_MASK);
                let mut portable = layer.clone();
                portable.id = local_id;
                (
                    local_id,
                    serde_json::to_string(&portable).unwrap_or_default(),
                )
            })
            .collect();

        let mut seen = HashSet::with_capacity(document.objects().len());
        for object in document.objects() {
            let id = object.id();
            seen.insert(id);
            let revision = document.object_revision(id);
            let local_layer = LayerId(object.layer().0 & LOCAL_MASK);
            if let Some(&(cached_revision, cached_layer)) = self.object_state.get(&id) {
                if cached_revision == revision && cached_layer == local_layer {
                    continue;
                }
                if cached_layer != local_layer
                    && let Some(bucket) = self.objects.get_mut(&cached_layer)
                {
                    bucket.remove(&(id.0 & LOCAL_MASK));
                }
            }
            let local_id = ObjectId(id.0 & LOCAL_MASK);
            let portable = object.with_id_and_layer(local_id, local_layer);
            self.objects.entry(local_layer).or_default().insert(
                local_id.0,
                serde_json::to_string(&portable).unwrap_or_default(),
            );
            self.object_state.insert(id, (revision, local_layer));
        }

        // Drop entries for objects deleted since the last sync.
        if self.object_state.len() != seen.len() {
            let stale: Vec<ObjectId> = self
                .object_state
                .keys()
                .filter(|id| !seen.contains(*id))
                .copied()
                .collect();
            for id in stale {
                if let Some((_, layer)) = self.object_state.remove(&id)
                    && let Some(bucket) = self.objects.get_mut(&layer)
                {
                    bucket.remove(&(id.0 & LOCAL_MASK));
                    if bucket.is_empty() {
                        self.objects.remove(&layer);
                    }
                }
            }
        }
    }
}

impl OpenProject {
    pub(crate) fn dirty_layers(&self) -> HashSet<LayerId> {
        let revision = self.pidb.document.revision();
        // The disk mtime is part of the cache key: an external rewrite of the
        // file must invalidate the result even though the in-memory revision
        // is untouched.
        let mtime = self
            .path
            .as_ref()
            .and_then(|path| fs::metadata(path).and_then(|m| m.modified()).ok());
        if let Some(cache) = self.dirty_layers_cache.borrow().as_ref()
            && cache.revision == revision
            && cache.mtime == mtime
        {
            return cache.layers.clone();
        }
        let layers = match self.path.as_ref() {
            Some(path) => self.dirty_layers_against_disk(path, mtime),
            None => self
                .pidb
                .document
                .layers()
                .iter()
                .map(|layer| layer.id)
                .collect(),
        };
        *self.dirty_layers_cache.borrow_mut() = Some(DirtyLayersCache {
            revision,
            mtime,
            layers: layers.clone(),
        });
        layers
    }

    /// Diff the in-memory document against the on-disk file, reusing a cached
    /// disk snapshot (refreshed only when the file's mtime changes) so an
    /// in-memory edit doesn't re-read or re-parse the file.
    fn dirty_layers_against_disk(
        &self,
        path: &Path,
        mtime: Option<SystemTime>,
    ) -> HashSet<LayerId> {
        let snapshot_matches = self
            .disk_snapshot_cache
            .borrow()
            .as_ref()
            .is_some_and(|snapshot| snapshot.mtime == mtime);
        if !snapshot_matches {
            match DiskSnapshot::load(path, mtime) {
                Some(snapshot) => *self.disk_snapshot_cache.borrow_mut() = Some(snapshot),
                // File missing/unreadable: every layer is unsaved.
                None => {
                    return self
                        .pidb
                        .document
                        .layers()
                        .iter()
                        .map(|layer| layer.id)
                        .collect();
                }
            }
        }
        let cache = self.disk_snapshot_cache.borrow();
        let disk = cache.as_ref().expect("snapshot populated above");

        // Bring the incrementally maintained namespace-0 memory snapshot up to
        // date; only objects touched since the last sync are re-serialized.
        let mut memory_cache = self.memory_snapshot_cache.borrow_mut();
        memory_cache.sync(&self.pidb.document);
        let memory = &*memory_cache;
        let empty_objects = BTreeMap::new();

        self.pidb
            .document
            .layers()
            .iter()
            .filter_map(|runtime_layer| {
                let local_id = LayerId(runtime_layer.id.0 & u64::from(u32::MAX));
                let memory_layer = memory.layers.get(&local_id);
                let disk_layer = disk.layers.get(&local_id);
                let memory_objects = memory.objects.get(&local_id).unwrap_or(&empty_objects);
                let disk_objects = disk.objects.get(&local_id).unwrap_or(&empty_objects);
                layer_differs(
                    memory_layer.map(String::as_str),
                    disk_layer.map(String::as_str),
                    memory_objects,
                    disk_objects,
                )
                .then_some(runtime_layer.id)
            })
            .collect()
    }

    pub(crate) fn invalidate_dirty_layers(&self) {
        *self.dirty_layers_cache.borrow_mut() = None;
    }

    pub(crate) fn invalidate_disk_snapshot(&self) {
        self.invalidate_dirty_layers();
        *self.disk_snapshot_cache.borrow_mut() = None;
    }

    pub(crate) fn has_unsaved_changes(&self) -> bool {
        !self.dirty_layers().is_empty()
    }
}

impl DiskSnapshot {
    fn load(path: &Path, mtime: Option<std::time::SystemTime>) -> Option<Self> {
        let disk = load(path).ok()?;
        let objects = objects_by_layer(disk.document.objects());
        let layers = disk
            .document
            .layers()
            .iter()
            .map(|layer| {
                let local_id = LayerId(layer.id.0 & u64::from(u32::MAX));
                (local_id, serde_json::to_string(layer).unwrap_or_default())
            })
            .collect();
        Some(DiskSnapshot {
            mtime,
            layers,
            objects,
        })
    }
}

/// Re-serialize a JSON string after parsing it into `T`, so its float fields
/// take the same bit patterns the disk side got when it was parsed from the
/// file. serde_json's float parser can land a value 1 ULP off its serialized
/// form, and object/layer floats are `f32`, so a value serialized straight from
/// memory can differ textually from the round-tripped disk copy even when
/// nothing was edited.
fn normalized_json<T: Serialize + serde::de::DeserializeOwned>(json: &str) -> String {
    serde_json::from_str::<T>(json)
        .ok()
        .and_then(|value| serde_json::to_string(&value).ok())
        .unwrap_or_else(|| json.to_owned())
}

/// True if a layer's in-memory state diverges from disk. Both sides are
/// serialized JSON. A direct string compare is tried first (unchanged layers —
/// the common case every frame — cost one serialization pass); only on a
/// mismatch are the memory strings normalised through the same typed parse the
/// disk side went through, so a 1-ULP float-parser artefact isn't mistaken for
/// an edit.
fn layer_differs(
    memory_layer: Option<&str>,
    disk_layer: Option<&str>,
    memory_objects: &BTreeMap<u64, String>,
    disk_objects: &BTreeMap<u64, String>,
) -> bool {
    if memory_layer == disk_layer && memory_objects == disk_objects {
        return false;
    }
    let layer_matches =
        memory_layer.map(normalized_json::<Layer>) == disk_layer.map(normalized_json::<Layer>);
    if !layer_matches {
        return true;
    }
    if memory_objects.len() != disk_objects.len() {
        return true;
    }
    memory_objects.iter().any(|(id, memory_json)| {
        disk_objects.get(id).is_none_or(|disk_json| {
            normalized_json::<Object>(memory_json) != normalized_json::<Object>(disk_json)
        })
    })
}

#[derive(Clone, Debug)]
pub(crate) struct Workspace {
    pub(crate) projects: Vec<OpenProject>,
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
        self.active_index.and_then(|i| self.projects.get(i))
    }

    pub(crate) fn active_project_mut(&mut self) -> Option<&mut OpenProject> {
        self.active_index.and_then(|i| self.projects.get_mut(i))
    }

    pub(crate) fn active_document(&self) -> Option<&Document> {
        self.active_project().map(|p| &p.pidb.document)
    }

    pub(crate) fn active_document_mut(&mut self) -> Option<&mut Document> {
        self.active_project_mut().map(|p| &mut p.pidb.document)
    }

    pub(crate) fn has_active_project(&self) -> bool {
        self.active_index.is_some_and(|i| i < self.projects.len())
    }

    /// Add a project and make it active. If a project with the same path is
    /// already open, just switch to it instead of adding a duplicate.
    pub(crate) fn add_and_activate(&mut self, mut project: OpenProject) -> usize {
        if let Some(path) = &project.path
            && let Some(i) = self
                .projects
                .iter()
                .position(|p| p.path.as_deref() == Some(path.as_path()))
        {
            self.active_index = Some(i);
            return i;
        }
        self.prepare_project(&mut project);
        self.projects.push(project);
        let idx = self.projects.len() - 1;
        self.active_index = Some(idx);
        idx
    }

    /// Add a project without changing the active project.
    pub(crate) fn add_inactive(&mut self, mut project: OpenProject) -> usize {
        if let Some(path) = &project.path
            && let Some(index) = self
                .projects
                .iter()
                .position(|candidate| candidate.path.as_deref() == Some(path.as_path()))
        {
            return index;
        }
        self.prepare_project(&mut project);
        self.projects.push(project);
        self.projects.len() - 1
    }

    fn prepare_project(&mut self, project: &mut OpenProject) {
        let namespace = self.next_runtime_namespace;
        self.next_runtime_namespace = self.next_runtime_namespace.saturating_add(1);
        project.runtime_id = namespace;
        project.pidb.document.apply_runtime_namespace(namespace);
        project.loaded_layers.clear();
    }

    pub(crate) fn project_index_for_runtime_id(&self, runtime_id: u32) -> Option<usize> {
        self.projects
            .iter()
            .position(|project| project.runtime_id == runtime_id)
    }

    pub(crate) fn project_index_for_loaded_layer(&self, layer_id: LayerId) -> Option<usize> {
        self.projects
            .iter()
            .position(|project| project.loaded_layers.contains(&layer_id))
    }

    pub(crate) fn project_index_for_object(
        &self,
        object_id: crate::model::ObjectId,
    ) -> Option<usize> {
        self.projects
            .iter()
            .position(|project| project.pidb.document.get_object(object_id).is_some())
    }

    /// Fingerprint of everything `scene_document()` reads: per project, its
    /// namespace, document revision (bumped by every document mutation) and
    /// loaded-layer set. Equal keys guarantee an identical composite, letting
    /// callers skip the rebuild.
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
    pub(crate) fn scene_document(&self) -> Document {
        let mut scene = Document::new();
        for project in &self.projects {
            for layer in project.pidb.document.layers() {
                if !project.loaded_layers.contains(&layer.id) {
                    continue;
                }
                scene.append_layer_snapshot(
                    layer,
                    project
                        .pidb
                        .document
                        .objects()
                        .iter()
                        .filter(|object| object.layer() == layer.id),
                );
            }
        }
        scene
    }

    pub(crate) fn set_active_index(&mut self, index: usize) {
        if index < self.projects.len() {
            self.active_index = Some(index);
        }
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

fn objects_by_layer(
    objects: &[crate::model::Object],
) -> HashMap<LayerId, std::collections::BTreeMap<u64, String>> {
    let mut map: HashMap<LayerId, std::collections::BTreeMap<u64, String>> = HashMap::new();
    for object in objects {
        map.entry(object.layer()).or_default().insert(
            object.id().0,
            serde_json::to_string(object).unwrap_or_default(),
        );
    }
    map
}

/// Save one layer into an existing PIDB without overwriting unsaved changes on
/// other in-memory layers.
pub(crate) fn save_layer(path: impl AsRef<Path>, pidb: &PidbFile, layer_id: LayerId) -> Result<()> {
    let path = path.as_ref();
    // Older/newly-created projects may already have a selected save path even
    // though no file has been written yet. Establish the database with the
    // complete in-memory document before attempting an incremental layer save.
    if !path.exists() {
        return save(path, pidb);
    }
    let mut portable = pidb.clone();
    portable.document.apply_runtime_namespace_for_io(0);
    let local_layer_id = LayerId(layer_id.0 & u64::from(u32::MAX));
    let layer = portable
        .document
        .layer(local_layer_id)
        .cloned()
        .context("The selected layer no longer exists")?;
    let objects: Vec<Object> = portable
        .document
        .objects()
        .iter()
        .filter(|object| object.layer() == local_layer_id)
        .cloned()
        .collect();

    // "Incremental" only in what it preserves: the whole file is re-parsed
    // and re-serialized, so this save is O(file size).
    let mut disk = load(path)?;
    let old_object_ids: Vec<_> = disk
        .document
        .objects()
        .iter()
        .filter(|object| object.layer() == local_layer_id)
        .map(Object::id)
        .collect();
    for object_id in old_object_ids {
        disk.document.remove_object(object_id);
    }
    // Replace the layer in place so saving does not move it to the end of the
    // on-disk layer list.
    disk.document.upsert_layer(&layer);
    for object in &objects {
        disk.document.insert_object(object.clone());
    }
    save(path, &disk)
}

pub(crate) fn save(path: impl AsRef<Path>, pidb: &PidbFile) -> Result<()> {
    let path = path.as_ref();
    // Runtime ids carry a per-open-project namespace so several PIDBs can
    // share one scene. Keep the on-disk format project-local and stable.
    let mut portable = pidb.clone();
    portable.document.apply_runtime_namespace(0);
    let json = serde_json::to_string_pretty(&portable)?;
    // Write-then-rename so a crash or full disk mid-write cannot destroy the
    // previous copy of the project.
    let tmp_path = path.with_extension("pidb.tmp");
    {
        let mut file = fs::File::create(&tmp_path)
            .with_context(|| format!("create {}", tmp_path.display()))?;
        file.write_all(json.as_bytes())
            .with_context(|| format!("write {}", tmp_path.display()))?;
        file.sync_all()
            .with_context(|| format!("sync {}", tmp_path.display()))?;
    }
    fs::rename(&tmp_path, path).with_context(|| format!("write {}", path.display()))
}

pub(crate) fn validate(pidb: &mut PidbFile) -> Result<()> {
    if pidb.format_version != PIDB_FORMAT_VERSION {
        bail!(
            "Unsupported PIDB format version {} (expected {})",
            pidb.format_version,
            PIDB_FORMAT_VERSION
        );
    }
    pidb.document.rebuild_object_index();
    Ok(())
}

pub(crate) fn import_dxf_into(pidb: &mut PidbFile, path: impl AsRef<Path>) -> Result<usize> {
    let path = path.as_ref();
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let count = import_dxf_bytes_into(pidb, &bytes)?;
    userspace_log!("Imported {count} object(s) from {}", path.display());
    Ok(count)
}

pub(crate) fn import_dxf_bytes_into(pidb: &mut PidbFile, bytes: &[u8]) -> Result<usize> {
    let mut cursor = Cursor::new(bytes);
    let drawing = Drawing::load(&mut cursor)?;
    let imported = crate::model::formats::dxf::from_dxf(&drawing);
    Ok(merge_document(&mut pidb.document, &imported))
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
    drawing.header.version = AcadVersion::R2000;
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
    drawing
        .save_file(path.as_ref())
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
    let dxf_color = color_to_dxf(object.color());

    match object {
        Object::Point { pos, .. } => {
            let mut entity = Entity::new(EntityType::ModelPoint(ModelPoint {
                location: point_from_vec3(*pos),
                ..Default::default()
            }));
            entity.common.layer = layer_name;
            entity.common.color = dxf_color;
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
                entity.common.color = dxf_color;
                entity.common.elevation = z0;
                drawing.add_entity(entity);
            } else {
                // 3D polyline — preserve per-vertex Z.
                let mut poly = DxfPolyline::default();
                poly.set_is_3d_polyline(true);
                poly.set_is_closed(*closed);
                for v in verts {
                    let mut vertex = DxfVertex {
                        location: Point::new(v.pos.x, v.pos.y, v.pos.z),
                        bulge: v.bulge,
                        ..Default::default()
                    };
                    vertex.set_is_3d_polyline_vertex(true);
                    poly.add_vertex(drawing, vertex);
                }
                let mut entity = Entity::new(EntityType::Polyline(poly));
                entity.common.layer = layer_name;
                entity.common.color = dxf_color;
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
            entity.common.color = dxf_color;
            drawing.add_entity(entity);
        }
        Object::Road { id, .. } => {
            use crate::model::road_network::{self, RoadKey};
            let network = road_network.get_or_insert_with(|| road_network::resolve(document, None));
            for edge in network.edges_for(RoadKey::Object(*id)) {
                emit_road_points(&edge.center, &layer_name, dxf_color.clone(), drawing);
                emit_road_points(&edge.left, &layer_name, dxf_color.clone(), drawing);
                emit_road_points(&edge.right, &layer_name, dxf_color.clone(), drawing);
                if edge.start_cap
                    && let (Some(&l), Some(&r)) = (edge.left.first(), edge.right.first())
                {
                    emit_road_points(&[l, r], &layer_name, dxf_color.clone(), drawing);
                }
                if edge.end_cap
                    && let (Some(&l), Some(&r)) = (edge.left.last(), edge.right.last())
                {
                    emit_road_points(&[l, r], &layer_name, dxf_color.clone(), drawing);
                }
            }
        }
    }
}

fn emit_road_points(points: &[glam::DVec3], layer_name: &str, color: Color, drawing: &mut Drawing) {
    let verts: Vec<PolyVertex> = points.iter().copied().map(PolyVertex::straight).collect();
    emit_road_verts(verts, layer_name, color, drawing);
}

fn emit_road_verts(verts: Vec<PolyVertex>, layer_name: &str, color: Color, drawing: &mut Drawing) {
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
        entity.common.color = color;
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
        entity.common.color = color;
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

fn color_to_dxf(color: ObjectColor) -> Color {
    match color {
        ObjectColor::ByLayer => Color::by_layer(),
        ObjectColor::Fixed(rgba) => Color::from_index(nearest_aci(rgba)),
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
    let document = document_from_dgd_points(
        &design.points,
        &design.texts,
        &index_entries,
        &design.layer_names,
        design.palette.as_ref(),
    );
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
    let document = document_from_duf_design(&duf);
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
    Ok(OpenProject {
        runtime_id: 0,
        path,
        pidb,
        loaded_layers: HashSet::new(),
        dirty_layers_cache: RefCell::new(None),
        disk_snapshot_cache: RefCell::new(None),
        memory_snapshot_cache: RefCell::new(MemorySnapshot::default()),
    })
}
