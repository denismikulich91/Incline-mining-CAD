use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fs,
    io::Cursor,
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
    model::{
        Document, FillStyle, LayerId, Object, ObjectColor, PolyVertex, geometry::geometric_offset,
    },
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
    pub(crate) dirty: bool,
    /// Layers currently present in the shared scene. The PIDB remains the
    /// source of truth even while a layer is unloaded.
    pub(crate) loaded_layers: HashSet<LayerId>,
    dirty_layers_cache: RefCell<Option<(u64, HashSet<LayerId>)>>,
}

impl OpenProject {
    pub(crate) fn dirty_layers(&self) -> HashSet<LayerId> {
        let revision = self.pidb.document.revision();
        if let Some((cached_revision, layers)) = self.dirty_layers_cache.borrow().as_ref()
            && *cached_revision == revision
        {
            return layers.clone();
        }
        let layers = self
            .path
            .as_ref()
            .map(|path| dirty_layers_from_disk(&self.pidb, path))
            .unwrap_or_else(|| {
                self.pidb
                    .document
                    .layers()
                    .iter()
                    .map(|layer| layer.id)
                    .collect()
            });
        *self.dirty_layers_cache.borrow_mut() = Some((revision, layers.clone()));
        layers
    }

    pub(crate) fn invalidate_dirty_layers(&self) {
        *self.dirty_layers_cache.borrow_mut() = None;
    }
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

    pub(crate) fn mark_dirty(&mut self) {
        if let Some(p) = self.active_project_mut() {
            p.dirty = true;
        }
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

/// True if `pidb`'s visible contents differ from the file saved at `path`.
///
/// Compares layers and objects by id (order-insensitive) after normalising the
/// runtime namespace, and ignores bookkeeping that doesn't change the document
/// the user sees (id counters, revision). Used to recompute a project's dirty
/// flag after discarding a layer's edits, so the `*` indicator clears when the
/// in-memory state matches disk again.
pub(crate) fn differs_from_disk(pidb: &PidbFile, path: impl AsRef<Path>) -> bool {
    let Ok(disk) = load(path) else {
        return true;
    };
    fn signature(
        pidb: &PidbFile,
    ) -> (
        std::collections::BTreeMap<u64, String>,
        std::collections::BTreeMap<u64, String>,
        String,
    ) {
        let mut pidb = pidb.clone();
        pidb.document.apply_runtime_namespace(0);
        let layers = pidb
            .document
            .layers()
            .iter()
            .map(|l| (l.id.0, serde_json::to_string(l).unwrap_or_default()))
            .collect();
        let objects = pidb
            .document
            .objects()
            .iter()
            .map(|o| (o.id().0, serde_json::to_string(o).unwrap_or_default()))
            .collect();
        (layers, objects, pidb.metadata.name.clone())
    }
    signature(pidb) != signature(&disk)
}

/// Return all runtime layer ids whose layer metadata or objects differ from disk.
pub(crate) fn dirty_layers_from_disk(pidb: &PidbFile, path: impl AsRef<Path>) -> HashSet<LayerId> {
    let Ok(disk) = load(path) else {
        return pidb
            .document
            .layers()
            .iter()
            .map(|layer| layer.id)
            .collect();
    };
    let mut portable = pidb.clone();
    portable.document.apply_runtime_namespace(0);
    // serde_json's float parser has edge cases where certain f64 values parse to a
    // different bit pattern than their serialized form (off by 1 ULP). Normalising the
    // in-memory objects through the same JSON round-trip that save() uses ensures the
    // comparison uses the same float representation as the disk side.
    if let Ok(json) = serde_json::to_string(&portable)
        && let Ok(normalised) = serde_json::from_str::<PidbFile>(&json)
    {
        portable = normalised;
    }

    // Pre-group objects by layer once (O(L+O)) instead of re-scanning all objects per layer
    // (O(L×O)).
    let memory_by_layer = objects_by_layer(portable.document.objects());
    let disk_by_layer = objects_by_layer(disk.document.objects());

    pidb.document
        .layers()
        .iter()
        .filter_map(|runtime_layer| {
            let local_id = LayerId(runtime_layer.id.0 & u64::from(u32::MAX));
            let memory_layer = portable.document.layer(local_id);
            let disk_layer = disk.document.layer(local_id);
            let empty = std::collections::BTreeMap::new();
            let memory_objects = memory_by_layer.get(&local_id).unwrap_or(&empty);
            let disk_objects = disk_by_layer.get(&local_id).unwrap_or(&empty);
            (memory_layer != disk_layer || memory_objects != disk_objects)
                .then_some(runtime_layer.id)
        })
        .collect()
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
    disk.document.delete_layer(local_layer_id);
    disk.document.append_layer_snapshot(&layer, objects.iter());
    save(path, &disk)
}

pub(crate) fn save(path: impl AsRef<Path>, pidb: &PidbFile) -> Result<()> {
    let path = path.as_ref();
    // Runtime ids carry a per-open-project namespace so several PIDBs can
    // share one scene. Keep the on-disk format project-local and stable.
    let mut portable = pidb.clone();
    portable.document.apply_runtime_namespace(0);
    let json = serde_json::to_string_pretty(&portable)?;
    fs::write(path, json).with_context(|| format!("write {}", path.display()))
}

pub(crate) fn validate(pidb: &mut PidbFile) -> Result<()> {
    if pidb.format_version != PIDB_FORMAT_VERSION {
        bail!(
            "Unsupported PIDB format version {} (expected {})",
            pidb.format_version,
            PIDB_FORMAT_VERSION
        );
    }
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

    for object in pidb.document.objects() {
        if only_layer.is_some_and(|id| id != object.layer()) {
            continue;
        }
        add_object_to_drawing(&pidb.document, object, &mut drawing);
    }
    drawing
        .save_file(path.as_ref())
        .with_context(|| format!("write {}", path.as_ref().display()))
}

fn add_object_to_drawing(document: &Document, object: &Object, drawing: &mut Drawing) {
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
        Object::Road {
            centerline,
            width,
            camber_degrees,
            shape,
            ..
        } => {
            if centerline.len() < 2 {
                return;
            }
            let half_width = width / 2.0;
            let (left_z, right_z) = shape.z_offsets(*width, *camber_degrees);
            let cl_pts: Vec<glam::DVec3> = centerline.iter().map(|v| v.pos).collect();
            // Centerline
            emit_road_verts(centerline.clone(), &layer_name, dxf_color.clone(), drawing);
            // Left edge
            let left_pts = geometric_offset(&cl_pts, false, half_width, 0.0);
            let left_verts: Vec<PolyVertex> = left_pts
                .iter()
                .map(|&p| PolyVertex::straight(glam::DVec3::new(p.x, p.y, p.z + left_z)))
                .collect();
            emit_road_verts(left_verts, &layer_name, dxf_color.clone(), drawing);
            // Right edge
            let right_pts = geometric_offset(&cl_pts, false, -half_width, 0.0);
            let right_verts: Vec<PolyVertex> = right_pts
                .iter()
                .map(|&p| PolyVertex::straight(glam::DVec3::new(p.x, p.y, p.z + right_z)))
                .collect();
            emit_road_verts(right_verts, &layer_name, dxf_color, drawing);
        }
    }
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
    let points = crate::model::formats::isis::read_dgd_points(path)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
    let document = document_from_dgd_points(&points);
    Ok(PidbFile {
        format_version: PIDB_FORMAT_VERSION,
        document,
        metadata: PidbMetadata {
            name: project_name(Some(path), "Imported.pidb"),
        },
    })
}

fn document_from_dgd_points(points: &[crate::model::formats::isis::DesignPoint]) -> Document {
    use glam::DVec3;

    let mut doc = Document::new();
    let mut layer_ids: HashMap<String, LayerId> = HashMap::new();
    // Current open polyline verts per layer.
    let mut current_polys: HashMap<LayerId, Vec<PolyVertex>> = HashMap::new();
    let mut completed: Vec<(LayerId, Vec<PolyVertex>)> = Vec::new();

    for point in points {
        let layer_id = *layer_ids.entry(point.name.clone()).or_insert_with(|| {
            doc.add_layer(point.name.clone(), None, [1.0, 1.0, 1.0, 1.0], true, 0.0)
        });
        let vertex = PolyVertex::straight(DVec3::new(point.x, point.y, point.z));
        if point.seg_type == 0 {
            if let Some(verts) = current_polys.remove(&layer_id)
                && !verts.is_empty()
            {
                completed.push((layer_id, verts));
            }
            current_polys.insert(layer_id, vec![vertex]);
        } else {
            current_polys.entry(layer_id).or_default().push(vertex);
        }
    }
    for (layer_id, verts) in current_polys {
        if !verts.is_empty() {
            completed.push((layer_id, verts));
        }
    }
    for (layer_id, verts) in completed {
        doc.add_object(|id| Object::Polyline {
            id,
            layer: layer_id,
            verts,
            closed: false,
            color: ObjectColor::ByLayer,
            fill: FillStyle::Clear,
            fill_color: None,
            line_weight: 1.0,
        });
    }
    doc
}

pub(crate) fn open_project(
    path: Option<PathBuf>,
    mut pidb: PidbFile,
    dirty: bool,
) -> Result<OpenProject> {
    validate(&mut pidb)?;
    Ok(OpenProject {
        runtime_id: 0,
        path,
        pidb,
        dirty,
        loaded_layers: HashSet::new(),
        dirty_layers_cache: RefCell::new(None),
    })
}
