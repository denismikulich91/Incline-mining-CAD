use glam::{DVec2, DVec3};

use crate::{
    app::App,
    model::{
        Command, Document, FillStyle, Object, ObjectColor, ObjectId, PolyVertex, SceneEntityId,
        formats::tri00t,
        road_network::{
            EdgeGeom, GhostRoad, RoadKey, prepare, resolve, resolve_prepared, validate_ghost,
            validate_ghost_prepared,
        },
    },
    ui::state::{ActiveTool, CursorMode, TriSurfaceType},
    userspace_log, userspace_warn,
};

const ROAD_MESH_NODE_TOL: f64 = 1e-6;
const ROAD_SIDE_POLYLINE_COLOR: ObjectColor = ObjectColor::Fixed([1.0, 0.85, 0.0, 1.0]);

impl<'a> App<'a> {
    /// Rebuild the preview side lines for the road being drawn.
    ///
    /// The preview runs the exact same resolver as committed roads (with the
    /// pending stroke as a ghost), so what is shown while drawing is what
    /// commits. Also refreshes `road_preview_violation`, which the viewport
    /// label and preview colouring read.
    pub(crate) fn update_road_preview(&mut self) {
        if self.editor.active_tool != ActiveTool::MakeRoad {
            self.clear_road_preview_geometry();
            return;
        }
        if self.editor.cursor_world.is_none() && self.editor.pending_stroke.len() < 2 {
            self.clear_road_preview_geometry();
            return;
        }

        // Grandfathered violations come from the ghost-free topology, which
        // is cached per document revision — without it every cursor move
        // would rebuild the network topology a second time just to diff.
        let preexisting = self.road_preexisting_compromised();

        let Some(document) = self
            .workspace
            .active_project()
            .map(|project| &project.pidb.document)
        else {
            self.clear_road_preview_geometry();
            return;
        };

        let Some(ghost) = crate::rendering::scene::road::make_ghost_candidate(&self.editor) else {
            self.clear_road_preview_geometry();
            return;
        };
        // One ghost-inclusive topology build shared by validation and
        // resolution; `resolve` from scratch here would build it again.
        let prepared = prepare(document, Some(&ghost));
        let violation = validate_ghost_prepared(
            &prepared,
            &ghost,
            self.editor.road_max_angle_degrees,
            &preexisting,
        )
        .err();
        if violation.is_some() {
            self.clear_road_preview_geometry();
            self.editor.road_preview_violation = violation;
            return;
        }

        let network = resolve_prepared(prepared);

        let mut center_world: Vec<DVec3> = Vec::new();
        let mut left_world: Vec<DVec3> = Vec::new();
        let mut right_world: Vec<DVec3> = Vec::new();
        let mut affected_edges = Vec::new();
        for edge in network.edges {
            match edge.road {
                RoadKey::Ghost => {
                    push_polyline_with_break(&mut center_world, &edge.center);
                    push_polyline_with_break(&mut left_world, &edge.left);
                    push_polyline_with_break(&mut right_world, &edge.right);
                }
                RoadKey::Object(id) if network.ghost_affected.contains(&id) => {
                    affected_edges.push(edge);
                }
                RoadKey::Object(_) => {}
            }
        }
        self.editor.road_preview_center_world = center_world;
        self.editor.road_preview_left_world = left_world;
        self.editor.road_preview_right_world = right_world;
        self.editor.road_preview_affected_edges = affected_edges;
        self.editor.road_preview_violation = None;

        // The affected roads are suppressed in the static scene and drawn
        // ghost-inclusive by the dynamic pass; the static scene only needs a
        // rebuild when that set changes, not on every cursor move.
        if self.editor.road_preview_affected_roads != network.ghost_affected {
            self.editor.road_preview_affected_roads = network.ghost_affected;
            self.invalidate_geometry();
        }
    }

    pub(crate) fn clear_road_preview_geometry(&mut self) {
        self.editor.road_preview_center_world.clear();
        self.editor.road_preview_left_world.clear();
        self.editor.road_preview_right_world.clear();
        self.editor.road_preview_affected_edges.clear();
        self.editor.road_preview_violation = None;
        if !self.editor.road_preview_affected_roads.is_empty() {
            self.editor.road_preview_affected_roads.clear();
            // Un-suppress the previously affected roads in the static scene.
            self.invalidate_geometry();
        }
    }

    pub(crate) fn place_road_point(&mut self) {
        if !self.editing_ready() {
            return;
        }
        let loop_closure = self.pending_road_loop_closure_point();
        if matches!(
            self.editor.cursor_mode,
            CursorMode::SnapToPoint | CursorMode::SnapToLine | CursorMode::SnapToSurface
        ) && !self.editor.cursor_snapped
            && loop_closure.is_none()
        {
            return;
        }
        let Some(world) = loop_closure.or(self.editor.cursor_world) else {
            return;
        };
        if self.active_layer().is_none() {
            return;
        }
        if !self.editor.pending_stroke.is_empty() {
            let mut candidate = self.editor.pending_stroke.clone();
            candidate.push(world);
            if let Err(error) = self.validate_road_candidate(candidate) {
                userspace_warn!("Road point rejected: {error}");
                return;
            }
        }
        self.editor.pending_stroke.push(world);
        self.invalidate_geometry();
    }

    /// Run every placement rule (grade, turn angles including across
    /// junctions, flat-zone clearances) against the candidate centerline.
    fn validate_road_candidate(
        &self,
        centerline: Vec<DVec3>,
    ) -> Result<(), crate::model::road_network::RoadRuleViolation> {
        let ghost = GhostRoad {
            centerline,
            width: self.editor.road_width,
            camber_degrees: self.editor.road_camber_degrees,
            shape: self.editor.road_shape,
        };
        validate_ghost(
            self.active_document(),
            &ghost,
            self.editor.road_max_angle_degrees,
        )
    }

    pub(crate) fn commit_road(&mut self) {
        if self.editor.pending_stroke.len() < 2 {
            return;
        }
        let Some(layer) = self.active_layer() else {
            return;
        };
        // Rule check only: flat approaches, junction clearances and side
        // lines are derived by the network resolver, never stored.
        if let Err(error) = self.validate_road_candidate(self.editor.pending_stroke.clone()) {
            userspace_warn!("Could not place road: {error}");
            return;
        }

        let color = ObjectColor::Fixed(self.editor.tool_line_color);
        if let Some(project) = self.workspace.active_project_mut() {
            let doc = &mut project.pidb.document;
            let id = doc.allocate_object_id();
            let new_road = Object::Road {
                id,
                layer,
                color,
                centerline: self
                    .editor
                    .pending_stroke
                    .iter()
                    .copied()
                    .map(PolyVertex::straight)
                    .collect(),
                width: self.editor.road_width,
                camber_degrees: self.editor.road_camber_degrees,
                shape: self.editor.road_shape,
            };
            self.history.execute(doc, Command::AddObject(new_road));
        }

        self.discard_road_stroke();
    }

    /// Clear the in-progress road without leaving the road tool.
    pub(crate) fn discard_road_stroke(&mut self) {
        self.editor.pending_stroke.clear();
        self.clear_road_preview_geometry();
        self.editor.road_preview_left_screen_px.clear();
        self.editor.road_preview_right_screen_px.clear();
        self.editor.road_preview_center_screen_px.clear();
        self.invalidate_geometry();
    }

    pub(crate) fn cancel_road(&mut self) {
        self.discard_road_stroke();
        self.editor.road_dialog_open = false;
        self.editor.active_tool = ActiveTool::None;
    }

    pub(crate) fn create_triangulation_from_roads(
        &mut self,
        name: String,
        object_ids: Vec<ObjectId>,
    ) -> anyhow::Result<()> {
        let RoadTriangulationMesh {
            road_count,
            vertices,
            faces,
        } = build_road_triangulation_mesh(self.active_document(), object_ids)?;

        crate::userspace_log!(
            "Converted {} road(s) to triangulation '{}'",
            road_count,
            name
        );
        self.finish_generated_triangulation(name, vertices, faces, TriSurfaceType::Surface)
    }

    pub(crate) fn convert_selected_roads_to_polylines(&mut self) -> anyhow::Result<()> {
        let selected_ids: Vec<ObjectId> = self
            .editor
            .selected_handles
            .iter()
            .filter_map(|handle| match handle {
                SceneEntityId::Object(id) => Some(*id),
                SceneEntityId::Triangulation(_) | SceneEntityId::BlockModel(_) => None,
            })
            .collect();

        if selected_ids.is_empty() {
            userspace_warn!("Select one or more roads before converting to polylines");
            return Ok(());
        }

        let replacement = build_road_polyline_replacement(self.active_document(), selected_ids)?;
        let road_count = replacement.sources.len();
        let polyline_count = replacement.polylines.len();

        if let Some(project) = self.workspace.active_project_mut() {
            let doc = &mut project.pidb.document;
            let mut commands = Vec::with_capacity(road_count + polyline_count);
            let mut new_ids = Vec::with_capacity(polyline_count);

            for source in replacement.sources {
                commands.push(Command::delete_object(source));
            }
            for spec in replacement.polylines {
                let id = doc.allocate_object_id();
                new_ids.push(id);
                commands.push(Command::AddObject(Object::Polyline {
                    id,
                    layer: spec.layer,
                    verts: spec.verts,
                    closed: false,
                    color: spec.color,
                    fill: FillStyle::Clear,
                    line_weight: 1.0,
                }));
            }

            self.history.execute(doc, Command::Batch(commands));
            self.editor.selected_handles = new_ids.into_iter().map(SceneEntityId::Object).collect();
            userspace_log!(
                "Converted {road_count} road(s) to {polyline_count} resolved polyline(s)"
            );
        }

        self.invalidate_geometry();
        Ok(())
    }

    fn pending_road_loop_closure_point(&self) -> Option<DVec3> {
        if self.editor.pending_stroke.len() < 3 {
            return None;
        }
        let cursor = self.editor.cursor_screen_px?;
        let first_screen = self
            .editor
            .road_preview_center_screen_px
            .first()
            .copied()
            .flatten()?;
        let dx = cursor.0 - first_screen.0;
        let dy = cursor.1 - first_screen.1;
        let threshold = crate::rendering::snap::SNAP_THRESHOLD_PX;
        (dx * dx + dy * dy <= threshold * threshold).then_some(self.editor.pending_stroke[0])
    }
}

/// Append a polyline to a preview buffer, separated from prior content by a
/// NaN break vertex (skipped by both projection and drawing).
fn push_polyline_with_break(buffer: &mut Vec<DVec3>, points: &[DVec3]) {
    if points.is_empty() {
        return;
    }
    if !buffer.is_empty() {
        buffer.push(DVec3::splat(f64::NAN));
    }
    buffer.extend_from_slice(points);
}

struct RoadTriangulationMesh {
    road_count: usize,
    vertices: Vec<tri00t::Vertex>,
    faces: Vec<[u32; 3]>,
}

struct RoadPolylineReplacement {
    sources: Vec<Object>,
    polylines: Vec<RoadPolylineSpec>,
}

struct RoadPolylineSpec {
    layer: crate::model::LayerId,
    verts: Vec<PolyVertex>,
    color: ObjectColor,
}

#[derive(Clone, Copy)]
struct RoadMeshEndpoint {
    center: DVec3,
    left: DVec3,
    right: DVec3,
}

fn build_road_polyline_replacement(
    document: &Document,
    object_ids: Vec<ObjectId>,
) -> anyhow::Result<RoadPolylineReplacement> {
    if object_ids.is_empty() {
        anyhow::bail!("No roads selected for polyline conversion");
    }

    let selected: std::collections::HashSet<_> = object_ids.into_iter().collect();
    let mut road_style_by_id = std::collections::HashMap::new();
    let mut sources = Vec::new();

    for object_id in &selected {
        if let Some(object @ Object::Road { layer, color, .. }) = document.get_object(*object_id) {
            road_style_by_id.insert(*object_id, (*layer, *color));
            sources.push(object.clone());
        }
    }

    if sources.is_empty() {
        anyhow::bail!("No road objects selected for polyline conversion");
    }

    let network = resolve(document, None);
    let mut polylines = Vec::new();
    for edge in network.edges {
        let RoadKey::Object(object_id) = edge.road else {
            continue;
        };
        let Some(&(layer, center_color)) = road_style_by_id.get(&object_id) else {
            continue;
        };

        push_road_polyline_spec(&mut polylines, layer, center_color, &edge.center);
        push_road_polyline_spec(&mut polylines, layer, ROAD_SIDE_POLYLINE_COLOR, &edge.left);
        push_road_polyline_spec(&mut polylines, layer, ROAD_SIDE_POLYLINE_COLOR, &edge.right);
        if edge.start_cap {
            push_road_cap_polyline_spec(
                &mut polylines,
                layer,
                edge.left.first().copied(),
                edge.right.first().copied(),
            );
        }
        if edge.end_cap {
            push_road_cap_polyline_spec(
                &mut polylines,
                layer,
                edge.left.last().copied(),
                edge.right.last().copied(),
            );
        }
    }

    if polylines.is_empty() {
        anyhow::bail!("Selected roads produced no polyline geometry");
    }

    Ok(RoadPolylineReplacement { sources, polylines })
}

fn push_road_polyline_spec(
    polylines: &mut Vec<RoadPolylineSpec>,
    layer: crate::model::LayerId,
    color: ObjectColor,
    points: &[DVec3],
) {
    if points.len() < 2 {
        return;
    }
    polylines.push(RoadPolylineSpec {
        layer,
        verts: points.iter().copied().map(PolyVertex::straight).collect(),
        color,
    });
}

fn push_road_cap_polyline_spec(
    polylines: &mut Vec<RoadPolylineSpec>,
    layer: crate::model::LayerId,
    left: Option<DVec3>,
    right: Option<DVec3>,
) {
    let (Some(left), Some(right)) = (left, right) else {
        return;
    };
    push_road_polyline_spec(polylines, layer, ROAD_SIDE_POLYLINE_COLOR, &[left, right]);
}

fn build_road_triangulation_mesh(
    document: &Document,
    object_ids: Vec<ObjectId>,
) -> anyhow::Result<RoadTriangulationMesh> {
    if object_ids.is_empty() {
        anyhow::bail!("No roads selected for triangulation");
    }

    let selected: std::collections::HashSet<_> = object_ids.into_iter().collect();
    let network = resolve(document, None);
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut endpoints = Vec::new();
    let mut road_count = 0usize;

    for object_id in &selected {
        if document
            .get_object(*object_id)
            .is_some_and(|object| matches!(object, Object::Road { .. }))
        {
            road_count += 1;
        }
    }

    for edge in network.edges {
        let RoadKey::Object(object_id) = edge.road else {
            continue;
        };
        if !selected.contains(&object_id) {
            continue;
        }
        append_road_edge_strips(
            &edge.left,
            &edge.center,
            &edge.right,
            &mut vertices,
            &mut faces,
        )?;
        push_edge_endpoint_records(&edge, &mut endpoints);
    }
    append_junction_fans(&endpoints, &mut vertices, &mut faces)?;

    if road_count == 0 {
        anyhow::bail!("No road objects selected for triangulation");
    }
    if faces.is_empty() {
        anyhow::bail!("Selected roads produced no triangulation faces");
    }

    Ok(RoadTriangulationMesh {
        road_count,
        vertices,
        faces,
    })
}

fn push_edge_endpoint_records(edge: &EdgeGeom, endpoints: &mut Vec<RoadMeshEndpoint>) {
    if let (Some(&center), Some(&left), Some(&right)) =
        (edge.center.first(), edge.left.first(), edge.right.first())
    {
        endpoints.push(RoadMeshEndpoint {
            center,
            left,
            right,
        });
    }
    if let (Some(&center), Some(&left), Some(&right)) =
        (edge.center.last(), edge.left.last(), edge.right.last())
    {
        endpoints.push(RoadMeshEndpoint {
            center,
            left,
            right,
        });
    }
}

fn append_road_edge_strips(
    left: &[DVec3],
    center: &[DVec3],
    right: &[DVec3],
    vertices: &mut Vec<tri00t::Vertex>,
    faces: &mut Vec<[u32; 3]>,
) -> anyhow::Result<()> {
    append_half_road_panel(left, center, vertices, faces)?;
    append_half_road_panel(center, right, vertices, faces)
}

fn append_half_road_panel(
    first_edge: &[DVec3],
    second_edge: &[DVec3],
    vertices: &mut Vec<tri00t::Vertex>,
    faces: &mut Vec<[u32; 3]>,
) -> anyhow::Result<()> {
    if first_edge.len() < 2 || second_edge.len() < 2 {
        return Ok(());
    }
    let first_stations = normalized_polyline_stations(first_edge);
    let second_stations = normalized_polyline_stations(second_edge);
    let mut stations = Vec::with_capacity(first_edge.len() + second_edge.len());
    stations.extend(first_stations.iter().copied());
    stations.extend(second_stations.iter().copied());
    stations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    stations.dedup_by(|a, b| (*a - *b).abs() <= ROAD_MESH_NODE_TOL);
    if stations.len() < 2 {
        return Ok(());
    }
    if vertices.len().saturating_add(stations.len() * 2) > u32::MAX as usize {
        anyhow::bail!("Road triangulation is too large");
    }

    let base = vertices.len() as u32;
    for station in &stations {
        vertices.push(vertex_from_dvec3(sample_polyline_normalized(
            first_edge,
            &first_stations,
            *station,
        )));
        vertices.push(vertex_from_dvec3(sample_polyline_normalized(
            second_edge,
            &second_stations,
            *station,
        )));
    }

    for index in 0..(stations.len() - 1) {
        let a0 = base + index as u32 * 2;
        let b0 = a0 + 1;
        let a1 = a0 + 2;
        let b1 = a0 + 3;
        push_face_if_non_degenerate(faces, vertices, [a0, b0, b1]);
        push_face_if_non_degenerate(faces, vertices, [a0, b1, a1]);
    }
    Ok(())
}

fn normalized_polyline_stations(points: &[DVec3]) -> Vec<f64> {
    let mut distances = Vec::with_capacity(points.len());
    let mut total = 0.0;
    distances.push(0.0);
    for pair in points.windows(2) {
        total += (pair[1].truncate() - pair[0].truncate()).length();
        distances.push(total);
    }
    if total <= ROAD_MESH_NODE_TOL {
        return vec![0.0; points.len()];
    }
    distances
        .into_iter()
        .map(|distance| distance / total)
        .collect()
}

fn sample_polyline_normalized(points: &[DVec3], stations: &[f64], station: f64) -> DVec3 {
    if station <= 0.0 {
        return points[0];
    }
    if station >= 1.0 {
        return *points.last().expect("len >= 2");
    }
    for index in 0..(stations.len() - 1) {
        let a = stations[index];
        let b = stations[index + 1];
        if station <= b + ROAD_MESH_NODE_TOL {
            let denom = b - a;
            if denom.abs() <= ROAD_MESH_NODE_TOL {
                return points[index];
            }
            let t = ((station - a) / denom).clamp(0.0, 1.0);
            return points[index].lerp(points[index + 1], t);
        }
    }
    *points.last().expect("len >= 2")
}

fn push_face_if_non_degenerate(
    faces: &mut Vec<[u32; 3]>,
    vertices: &[tri00t::Vertex],
    face: [u32; 3],
) {
    let a = vertex_to_dvec3(vertices[face[0] as usize]);
    let b = vertex_to_dvec3(vertices[face[1] as usize]);
    let c = vertex_to_dvec3(vertices[face[2] as usize]);
    if (b - a).cross(c - a).length_squared() > ROAD_MESH_NODE_TOL * ROAD_MESH_NODE_TOL {
        faces.push(face);
    }
}

fn append_junction_fans(
    endpoints: &[RoadMeshEndpoint],
    vertices: &mut Vec<tri00t::Vertex>,
    faces: &mut Vec<[u32; 3]>,
) -> anyhow::Result<()> {
    let mut groups: Vec<Vec<RoadMeshEndpoint>> = Vec::new();
    for endpoint in endpoints {
        if let Some(group) = groups
            .iter_mut()
            .find(|group| points_same_node(group[0].center, endpoint.center))
        {
            group.push(*endpoint);
        } else {
            groups.push(vec![*endpoint]);
        }
    }

    for group in groups {
        if group.len() < 3 {
            continue;
        }
        let center = average_points(group.iter().map(|endpoint| endpoint.center));
        let mut boundary = Vec::new();
        for endpoint in &group {
            push_unique_point(&mut boundary, endpoint.left);
            push_unique_point(&mut boundary, endpoint.right);
        }
        if boundary.len() < 3 {
            continue;
        }
        boundary.sort_by(|a, b| {
            angle_around(center, *a)
                .partial_cmp(&angle_around(center, *b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        append_fan(center, &boundary, vertices, faces)?;
    }
    Ok(())
}

fn append_fan(
    center: DVec3,
    boundary: &[DVec3],
    vertices: &mut Vec<tri00t::Vertex>,
    faces: &mut Vec<[u32; 3]>,
) -> anyhow::Result<()> {
    if vertices.len().saturating_add(boundary.len() + 1) > u32::MAX as usize {
        anyhow::bail!("Road triangulation is too large");
    }
    let base = vertices.len() as u32;
    vertices.push(vertex_from_dvec3(center));
    for point in boundary {
        vertices.push(vertex_from_dvec3(*point));
    }
    for index in 0..boundary.len() {
        let a = base + 1 + index as u32;
        let b = base + 1 + ((index + 1) % boundary.len()) as u32;
        faces.push([base, a, b]);
    }
    Ok(())
}

fn points_same_node(a: DVec3, b: DVec3) -> bool {
    a.distance_squared(b) <= ROAD_MESH_NODE_TOL * ROAD_MESH_NODE_TOL
}

fn push_unique_point(points: &mut Vec<DVec3>, point: DVec3) {
    if !points
        .iter()
        .any(|existing| existing.distance_squared(point) <= ROAD_MESH_NODE_TOL * ROAD_MESH_NODE_TOL)
    {
        points.push(point);
    }
}

fn average_points(points: impl Iterator<Item = DVec3>) -> DVec3 {
    let mut sum = DVec3::ZERO;
    let mut count = 0.0;
    for point in points {
        sum += point;
        count += 1.0;
    }
    if count > 0.0 { sum / count } else { sum }
}

fn angle_around(center: DVec3, point: DVec3) -> f64 {
    let delta: DVec2 = (point - center).truncate();
    delta.y.atan2(delta.x)
}

fn vertex_from_dvec3(point: DVec3) -> tri00t::Vertex {
    tri00t::Vertex {
        x: point.x,
        y: point.y,
        z: point.z,
    }
}

fn vertex_to_dvec3(vertex: tri00t::Vertex) -> DVec3 {
    DVec3::new(vertex.x, vertex.y, vertex.z)
}
