//! Editable mine-design document model and source of truth for the editor.

pub(crate) mod block_model;
pub(crate) mod formats;
pub(crate) mod geometry;
pub(crate) mod kernel;
pub(crate) mod pidb;
pub(crate) mod road_network;
pub(crate) mod spatial;
pub(crate) mod triangulation;

use glam::DVec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct LayerId(pub(crate) u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct ObjectId(pub(crate) u64);

/// Stable identity used by rendering, selection and spatial queries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum SceneEntityId {
    Object(ObjectId),
    Triangulation(triangulation::TriangulationId),
    BlockModel(block_model::BlockModelId),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Layer {
    pub(crate) id: LayerId,
    pub(crate) name: String,
    pub(crate) color_index: Option<u8>,
    /// Resolved RGBA used for rendering objects with `ObjectColor::ByLayer`.
    pub(crate) color: [f32; 4],
    pub(crate) visible: bool,
    pub(crate) elevation: f32,
}

/// Cross-section profile for a road object.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum RoadShape {
    /// Symmetric crown — both edges drop from the centreline (^).
    #[default]
    Crown,
    /// Left edge high, right edge low — full cross-fall to the right (/).
    CrossFallRight,
    /// Right edge high, left edge low — full cross-fall to the left (\).
    CrossFallLeft,
}

impl RoadShape {
    /// Returns `(left_z_offset, right_z_offset)` for the given half-width and camber.
    pub(crate) fn z_offsets(self, width: f64, camber_degrees: f64) -> (f64, f64) {
        let drop = (width / 2.0) * camber_degrees.to_radians().tan();
        match self {
            RoadShape::Crown => (-drop, -drop),
            RoadShape::CrossFallRight => (drop, -drop),
            RoadShape::CrossFallLeft => (-drop, drop),
        }
    }
}

/// Fill style for closed polylines.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum FillStyle {
    #[default]
    Clear,
    Crosses,
    Slashes,
    Solid,
}

/// How an object's colour is determined.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum ObjectColor {
    /// Follow the owning layer's colour.
    ByLayer,
    /// An explicit resolved RGBA colour.
    Fixed([f32; 4]),
}

/// A polyline vertex. `bulge` is the DXF arc encoding (`tan(includedAngle/4)`)
/// for the segment starting at this vertex; `0.0` is a straight segment.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct PolyVertex {
    pub(crate) pos: DVec3,
    pub(crate) bulge: f64,
}

impl PolyVertex {
    pub(crate) fn straight(pos: DVec3) -> Self {
        Self { pos, bulge: 0.0 }
    }
}

fn default_poly_line_weight() -> f32 {
    1.0
}

/// A drawable design element. `Polyline` with `closed == true` represents a
/// polygon; vertices carry bulges so arcs/circles are preserved.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum Object {
    Point {
        id: ObjectId,
        layer: LayerId,
        pos: DVec3,
        color: ObjectColor,
    },
    Polyline {
        id: ObjectId,
        layer: LayerId,
        verts: Vec<PolyVertex>,
        closed: bool,
        color: ObjectColor,
        #[serde(default)]
        fill: FillStyle,
        #[serde(default)]
        fill_color: Option<ObjectColor>,
        #[serde(default = "default_poly_line_weight")]
        line_weight: f32,
    },
    Text {
        id: ObjectId,
        layer: LayerId,
        pos: DVec3,
        content: String,
        height: f64,
        rotation: f64,
        color: ObjectColor,
    },
    Road {
        id: ObjectId,
        layer: LayerId,
        color: ObjectColor,
        centerline: Vec<PolyVertex>,
        width: f64,
        camber_degrees: f64,
        shape: RoadShape,
    },
}

impl Object {
    pub(crate) fn id(&self) -> ObjectId {
        match self {
            Object::Point { id, .. }
            | Object::Polyline { id, .. }
            | Object::Text { id, .. }
            | Object::Road { id, .. } => *id,
        }
    }

    pub(crate) fn layer(&self) -> LayerId {
        match self {
            Object::Point { layer, .. }
            | Object::Polyline { layer, .. }
            | Object::Text { layer, .. }
            | Object::Road { layer, .. } => *layer,
        }
    }

    /// Short human-readable variant name, for diagnostics/logging.
    pub(crate) fn kind_name(&self) -> &'static str {
        match self {
            Object::Point { .. } => "Point",
            Object::Polyline { closed: true, .. } => "Polygon",
            Object::Polyline { closed: false, .. } => "Polyline",
            Object::Text { .. } => "Text",
            Object::Road { .. } => "Road",
        }
    }

    pub(crate) fn color(&self) -> ObjectColor {
        match self {
            Object::Point { color, .. }
            | Object::Polyline { color, .. }
            | Object::Text { color, .. }
            | Object::Road { color, .. } => *color,
        }
    }

    pub(crate) fn translate(&mut self, delta: DVec3) {
        match self {
            Object::Point { pos, .. } | Object::Text { pos, .. } => *pos += delta,
            Object::Polyline { verts, .. }
            | Object::Road {
                centerline: verts, ..
            } => {
                for vertex in verts {
                    vertex.pos += delta;
                }
            }
        }
    }

    pub(crate) fn set_z_position(&mut self, z: f64) {
        match self {
            Object::Point { pos, .. } | Object::Text { pos, .. } => pos.z = z,
            Object::Polyline { verts, .. }
            | Object::Road {
                centerline: verts, ..
            } => {
                for vertex in verts {
                    vertex.pos.z = z;
                }
            }
        }
    }

    pub(crate) fn with_id_and_layer(&self, id: ObjectId, layer: LayerId) -> Self {
        match self {
            Object::Point { pos, color, .. } => Object::Point {
                id,
                layer,
                pos: *pos,
                color: *color,
            },
            Object::Polyline {
                verts,
                closed,
                color,
                fill,
                fill_color,
                line_weight,
                ..
            } => Object::Polyline {
                id,
                layer,
                verts: verts.clone(),
                closed: *closed,
                color: *color,
                fill: *fill,
                fill_color: *fill_color,
                line_weight: *line_weight,
            },
            Object::Text {
                pos,
                content,
                height,
                rotation,
                color,
                ..
            } => Object::Text {
                id,
                layer,
                pos: *pos,
                content: content.clone(),
                height: *height,
                rotation: *rotation,
                color: *color,
            },
            Object::Road {
                centerline,
                width,
                camber_degrees,
                shape,
                color,
                ..
            } => Object::Road {
                id,
                layer,
                color: *color,
                centerline: centerline.clone(),
                width: *width,
                camber_degrees: *camber_degrees,
                shape: *shape,
            },
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct Document {
    layers: Vec<Layer>,
    objects: Vec<Object>,
    next_layer_id: u64,
    next_object_id: u64,
    #[serde(skip)]
    revision: u64,
}

impl Document {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn layers(&self) -> &[Layer] {
        &self.layers
    }

    pub(crate) fn objects(&self) -> &[Object] {
        &self.objects
    }

    pub(crate) fn add_layer(
        &mut self,
        name: String,
        color_index: Option<u8>,
        color: [f32; 4],
        visible: bool,
        elevation: f32,
    ) -> LayerId {
        let id = LayerId(self.next_layer_id);
        self.next_layer_id += 1;
        self.layers.push(Layer {
            id,
            name,
            color_index,
            color,
            visible,
            elevation,
        });
        self.touch();
        id
    }

    pub(crate) fn allocate_layer_id(&mut self) -> LayerId {
        let id = LayerId(self.next_layer_id);
        self.next_layer_id += 1;
        id
    }

    /// Append an object, supplying its freshly allocated id to the constructor.
    pub(crate) fn add_object(&mut self, make: impl FnOnce(ObjectId) -> Object) -> ObjectId {
        let id = ObjectId(self.next_object_id);
        self.next_object_id += 1;
        self.objects.push(make(id));
        self.touch();
        id
    }

    /// Reserve a fresh object id without inserting anything.
    pub(crate) fn allocate_object_id(&mut self) -> ObjectId {
        let id = ObjectId(self.next_object_id);
        self.next_object_id += 1;
        id
    }

    /// Insert an object that already carries its id (used by commands/redo).
    pub(crate) fn insert_object(&mut self, object: Object) {
        self.next_object_id = self.next_object_id.max(object.id().0 + 1);
        self.objects.push(object);
        self.touch();
    }

    /// Remove the object with `id`, returning it if present.
    pub(crate) fn remove_object(&mut self, id: ObjectId) -> Option<Object> {
        let index = self.objects.iter().position(|object| object.id() == id)?;
        let object = self.objects.remove(index);
        self.touch();
        Some(object)
    }

    pub(crate) fn layer(&self, id: LayerId) -> Option<&Layer> {
        self.layers.iter().find(|layer| layer.id == id)
    }

    /// Remove a layer by id. Returns false if the layer did not exist.
    pub(crate) fn delete_layer(&mut self, id: LayerId) -> bool {
        let before = self.layers.len();
        self.layers.retain(|l| l.id != id);
        let removed = self.layers.len() < before;
        if removed {
            self.touch();
        }
        removed
    }

    /// Resolved RGBA for an object, following its layer when `ByLayer`.
    pub(crate) fn object_rgba(&self, object: &Object) -> [f32; 4] {
        match object.color() {
            ObjectColor::Fixed(rgba) => rgba,
            ObjectColor::ByLayer => self
                .layer(object.layer())
                .map(|layer| layer.color)
                .unwrap_or([1.0, 1.0, 1.0, 1.0]),
        }
    }

    /// Resolved fill RGBA for an object. For polylines with an explicit fill
    /// color, that color is used; otherwise falls back to `object_rgba`.
    pub(crate) fn object_fill_rgba(&self, object: &Object) -> [f32; 4] {
        self.object_rgba(object)
    }

    pub(crate) fn rename_layer(&mut self, id: LayerId, new_name: String) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) {
            layer.name = new_name;
            self.touch();
        }
    }

    pub(crate) fn layer_id_by_name(&self, name: &str) -> Option<LayerId> {
        self.layers
            .iter()
            .find(|layer| layer.name == name)
            .map(|layer| layer.id)
    }

    pub(crate) fn first_layer_id(&self) -> Option<LayerId> {
        self.layers.first().map(|layer| layer.id)
    }

    pub(crate) fn ensure_default_layer(&mut self) -> LayerId {
        self.first_layer_id().unwrap_or_else(|| {
            self.add_layer("0".to_string(), Some(7), [1.0, 1.0, 1.0, 1.0], true, 0.0)
        })
    }

    /// Assign every layer and object a runtime namespace. PIDB ids are local
    /// to a file; namespacing makes them safe to combine in one scene.
    pub(crate) fn apply_runtime_namespace(&mut self, namespace: u32) {
        self.apply_runtime_namespace_inner(namespace);
        self.touch();
    }

    /// Normalize ids for serialization without marking the document as edited.
    pub(crate) fn apply_runtime_namespace_for_io(&mut self, namespace: u32) {
        self.apply_runtime_namespace_inner(namespace);
    }

    fn apply_runtime_namespace_inner(&mut self, namespace: u32) {
        const LOCAL_MASK: u64 = u32::MAX as u64;
        let prefix = u64::from(namespace) << 32;
        let runtime_id = |id: u64| prefix | (id & LOCAL_MASK);

        for layer in &mut self.layers {
            layer.id = LayerId(runtime_id(layer.id.0));
        }
        self.objects = self
            .objects
            .iter()
            .map(|object| {
                object.with_id_and_layer(
                    ObjectId(runtime_id(object.id().0)),
                    LayerId(runtime_id(object.layer().0)),
                )
            })
            .collect();
        self.next_layer_id = runtime_id(self.next_layer_id);
        self.next_object_id = runtime_id(self.next_object_id);
    }

    /// Append a layer and its objects while retaining their runtime ids.
    pub(crate) fn append_layer_snapshot<'a>(
        &mut self,
        layer: &Layer,
        objects: impl Iterator<Item = &'a Object>,
    ) {
        if self.layer(layer.id).is_none() {
            self.next_layer_id = self.next_layer_id.max(layer.id.0.saturating_add(1));
            self.layers.push(layer.clone());
        }
        for object in objects {
            self.insert_object(object.clone());
        }
        self.touch();
    }

    pub(crate) fn get_object(&self, id: ObjectId) -> Option<&Object> {
        self.objects.iter().find(|object| object.id() == id)
    }

    /// Translate the object with `id` by `delta`. Returns `true` if found.
    pub(crate) fn translate_object(&mut self, id: ObjectId, delta: DVec3) -> bool {
        match self.objects.iter_mut().find(|object| object.id() == id) {
            Some(object) => {
                object.translate(delta);
                self.touch();
                true
            }
            None => false,
        }
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    fn touch(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }
}

/// A reversible edit to the document.
#[derive(Clone, Debug)]
pub(crate) enum Command {
    AddObject(Object),
    DeleteObject(Object),
    /// Replace an object's state (e.g. after a move). `before`/`after` share an id.
    Replace {
        before: Object,
        after: Object,
    },
    /// Apply/revert a sequence of commands atomically (single Ctrl-Z).
    Batch(Vec<Command>),
    /// Rename a layer (undo restores the old name).
    RenameLayer {
        id: LayerId,
        before: String,
        after: String,
    },
    /// Add a complete layer and all objects generated with it.
    AddLayerSnapshot {
        layer: Layer,
        objects: Vec<Object>,
    },
    /// Remove a complete layer and all objects on it.
    DeleteLayerSnapshot {
        layer: Layer,
        objects: Vec<Object>,
    },
}

impl Command {
    fn apply(&self, doc: &mut Document) {
        match self {
            Command::AddObject(object) => doc.insert_object(object.clone()),
            Command::DeleteObject(object) => {
                doc.remove_object(object.id());
            }
            Command::Replace { after, .. } => {
                doc.remove_object(after.id());
                doc.insert_object(after.clone());
            }
            Command::Batch(cmds) => {
                for cmd in cmds {
                    cmd.apply(doc);
                }
            }
            Command::RenameLayer { id, after, .. } => doc.rename_layer(*id, after.clone()),
            Command::AddLayerSnapshot { layer, objects } => {
                doc.append_layer_snapshot(layer, objects.iter())
            }
            Command::DeleteLayerSnapshot { layer, objects } => {
                for object in objects {
                    doc.remove_object(object.id());
                }
                doc.delete_layer(layer.id);
            }
        }
    }

    fn revert(&self, doc: &mut Document) {
        match self {
            Command::AddObject(object) => {
                doc.remove_object(object.id());
            }
            Command::DeleteObject(object) => doc.insert_object(object.clone()),
            Command::Replace { before, .. } => {
                doc.remove_object(before.id());
                doc.insert_object(before.clone());
            }
            Command::Batch(cmds) => {
                for cmd in cmds.iter().rev() {
                    cmd.revert(doc);
                }
            }
            Command::RenameLayer { id, before, .. } => doc.rename_layer(*id, before.clone()),
            Command::AddLayerSnapshot { layer, objects } => {
                for object in objects {
                    doc.remove_object(object.id());
                }
                doc.delete_layer(layer.id);
            }
            Command::DeleteLayerSnapshot { layer, objects } => {
                doc.append_layer_snapshot(layer, objects.iter())
            }
        }
    }
}

/// Undo/redo stacks. Every document edit goes through `execute`.
#[derive(Default)]
pub(crate) struct History {
    undo: Vec<Command>,
    redo: Vec<Command>,
}

impl History {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }

    pub(crate) fn execute(&mut self, doc: &mut Document, command: Command) {
        command.apply(doc);
        self.undo.push(command);
        self.redo.clear();
    }

    /// Record a command whose effect is already applied to the document
    /// (e.g. an interactive drag-move committed on mouse release).
    pub(crate) fn push_applied(&mut self, command: Command) {
        self.undo.push(command);
        self.redo.clear();
    }

    /// Revert the most recent command. Returns `true` if something was undone.
    pub(crate) fn undo(&mut self, doc: &mut Document) -> bool {
        match self.undo.pop() {
            Some(command) => {
                command.revert(doc);
                self.redo.push(command);
                true
            }
            None => false,
        }
    }

    pub(crate) fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub(crate) fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Re-apply the most recently undone command. Returns `true` on success.
    pub(crate) fn redo(&mut self, doc: &mut Document) -> bool {
        match self.redo.pop() {
            Some(command) => {
                command.apply(doc);
                self.undo.push(command);
                true
            }
            None => false,
        }
    }
}
