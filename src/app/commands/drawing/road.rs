use glam::DVec3;

use crate::{
    app::App,
    model::{
        Command, Object, ObjectColor, PolyVertex,
        road_network::{GhostRoad, RoadKey, resolve, validate_ghost},
    },
    ui::state::{ActiveTool, CursorMode},
    userspace_warn,
};

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
        let violation = validate_ghost(document, &ghost, self.editor.road_max_angle_degrees).err();
        if violation.is_some() {
            self.clear_road_preview_geometry();
            self.editor.road_preview_violation = violation;
            return;
        }

        let network = resolve(document, Some(&ghost));

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

    fn clear_road_preview_geometry(&mut self) {
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
            project.dirty = true;
        }

        self.editor.pending_stroke.clear();
        self.clear_road_preview_geometry();
        self.editor.road_preview_left_screen_px.clear();
        self.editor.road_preview_right_screen_px.clear();
        self.editor.road_preview_center_screen_px.clear();
        self.invalidate_geometry();
    }

    pub(crate) fn cancel_road(&mut self) {
        self.editor.pending_stroke.clear();
        self.clear_road_preview_geometry();
        self.editor.road_preview_left_screen_px.clear();
        self.editor.road_preview_right_screen_px.clear();
        self.editor.road_preview_center_screen_px.clear();
        self.editor.road_dialog_open = false;
        self.editor.active_tool = ActiveTool::None;
        self.invalidate_geometry();
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
