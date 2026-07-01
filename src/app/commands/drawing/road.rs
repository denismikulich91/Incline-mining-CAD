use glam::DVec3;

use crate::{
    app::App,
    model::{
        Command, Document, Object, ObjectColor, ObjectId, PolyVertex,
        geometry::{
            ROAD_INTERSECTION_FLAT_CLEARANCE_M, RoadCenterlineRuleError, geometric_offset,
            road_centerline_with_intersection_flats, validate_road_segment_angles,
        },
    },
    ui::state::{ActiveTool, CursorMode},
    userspace_warn,
};

impl<'a> App<'a> {
    pub(crate) fn update_road_preview(&mut self) {
        if self.editor.active_tool != ActiveTool::MakeRoad {
            self.editor.road_preview_left_world.clear();
            self.editor.road_preview_right_world.clear();
            return;
        }

        let Some(cursor) = self.editor.cursor_world else {
            if self.editor.pending_stroke.len() < 2 {
                self.editor.road_preview_left_world.clear();
                self.editor.road_preview_right_world.clear();
            }
            return;
        };

        let mut raw_centerline: Vec<DVec3> = self.editor.pending_stroke.clone();
        raw_centerline.push(cursor);

        if validate_road_segment_angles(&raw_centerline, self.editor.road_max_angle_degrees)
            .is_err()
        {
            self.editor.road_preview_left_world.clear();
            self.editor.road_preview_right_world.clear();
            return;
        }

        let mut centerline = match self.workspace.active_project().and_then(|project| {
            road_centerline_with_intersection_flats(
                &raw_centerline,
                &project.pidb.document,
                self.editor.road_width,
            )
            .ok()
        }) {
            Some(centerline) => centerline,
            None => {
                self.editor.road_preview_left_world.clear();
                self.editor.road_preview_right_world.clear();
                return;
            }
        };

        if centerline.len() < 2 {
            self.editor.road_preview_left_world.clear();
            self.editor.road_preview_right_world.clear();
            return;
        }

        // Prepend a phantom point if our first placed point snaps to the end of an
        // existing road, so geometric_offset can compute the correct miter there.
        let start_pos = centerline[0];
        let prepend_phantom = self.workspace.active_project().and_then(|proj| {
            let candidates: Vec<DVec3> = proj
                .pidb
                .document
                .objects()
                .iter()
                .filter_map(|obj| {
                    if let Object::Road {
                        centerline: other_cl,
                        ..
                    } = obj
                    {
                        let n = other_cl.len();
                        if n >= 2 {
                            if (other_cl[0].pos - start_pos).length_squared() < 1e-10 {
                                return Some(other_cl[1].pos);
                            }
                            if (other_cl[n - 1].pos - start_pos).length_squared() < 1e-10 {
                                return Some(other_cl[n - 2].pos);
                            }
                        }
                    }
                    None
                })
                .collect();
            (candidates.len() == 1).then(|| candidates[0])
        });

        // Junction detected: rebuild 3D geometry this frame so the connecting road's
        // edge dynamically miters toward the cursor while we're still drawing.
        if prepend_phantom.is_some() {
            self.invalidate_geometry();
        }

        let attachment_centerline = centerline.clone();
        let skip = if let Some(p) = prepend_phantom {
            centerline.insert(0, p);
            1
        } else {
            0
        };

        let (left_z, right_z) = self
            .editor
            .road_shape
            .z_offsets(self.editor.road_width, self.editor.road_camber_degrees);
        let half_w = self.editor.road_width / 2.0;
        let left_xy = geometric_offset(&centerline, false, half_w, 0.0);
        let right_xy = geometric_offset(&centerline, false, -half_w, 0.0);

        let mut left_world: Vec<DVec3> = left_xy
            .into_iter()
            .skip(skip)
            .map(|p| DVec3::new(p.x, p.y, p.z + left_z))
            .collect();
        let mut right_world: Vec<DVec3> = right_xy
            .into_iter()
            .skip(skip)
            .map(|p| DVec3::new(p.x, p.y, p.z + right_z))
            .collect();

        if let Some(project) = self.workspace.active_project() {
            let attachments =
                attached_polyline_clips(&project.pidb.document, &attachment_centerline);
            left_world = snap_edge_endpoints_to_attached_objects(left_world, &attachments);
            right_world = snap_edge_endpoints_to_attached_objects(right_world, &attachments);
            left_world = clip_edge_to_attached_objects(left_world, &attachments);
            right_world = clip_edge_to_attached_objects(right_world, &attachments);
            left_world = snap_edge_endpoints_to_attached_objects(left_world, &attachments);
            right_world = snap_edge_endpoints_to_attached_objects(right_world, &attachments);
            left_world = taper_edge_to_attached_objects(left_world, &attachments);
            right_world = taper_edge_to_attached_objects(right_world, &attachments);
        }

        self.editor.road_preview_left_world = left_world;
        self.editor.road_preview_right_world = right_world;
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
            if let Err(error) =
                validate_road_segment_angles(&candidate, self.editor.road_max_angle_degrees)
            {
                userspace_warn!("Road point rejected: {error}");
                return;
            }
            if let Err(error) = road_centerline_with_intersection_flats(
                &candidate,
                self.active_document(),
                self.editor.road_width,
            ) {
                userspace_warn!("Road point rejected: {error}");
                return;
            }
        }
        self.editor.pending_stroke.push(world);
        self.invalidate_geometry();
    }

    pub(crate) fn commit_road(&mut self) {
        if self.editor.pending_stroke.len() < 2 {
            return;
        }
        let Some(layer) = self.active_layer() else {
            return;
        };
        if let Err(error) = validate_road_segment_angles(
            &self.editor.pending_stroke,
            self.editor.road_max_angle_degrees,
        ) {
            userspace_warn!("Could not place road: {error}");
            return;
        }
        let color = ObjectColor::Fixed(self.editor.tool_line_color);
        let centerline_points = match road_centerline_with_intersection_flats(
            &self.editor.pending_stroke,
            self.active_document(),
            self.editor.road_width,
        ) {
            Ok(centerline) => centerline,
            Err(error) => {
                userspace_warn!("Could not place road: {error}");
                return;
            }
        };

        if let Some(project) = self.workspace.active_project_mut() {
            let doc = &mut project.pidb.document;
            let id = doc.allocate_object_id();
            let mut new_road = Object::Road {
                id,
                layer,
                color,
                centerline: centerline_points
                    .iter()
                    .copied()
                    .map(PolyVertex::straight)
                    .collect(),
                width: self.editor.road_width,
                camber_degrees: self.editor.road_camber_degrees,
                shape: self.editor.road_shape,
            };
            let mut replacements = match normalize_new_and_existing_roads_for_junctions(
                doc,
                id,
                &mut new_road,
                &centerline_points,
            ) {
                Ok(replacements) => replacements,
                Err(error) => {
                    userspace_warn!("Could not normalize road junctions: {error}");
                    return;
                }
            };
            let mut commands = vec![Command::AddObject(new_road)];
            commands.append(&mut replacements);
            self.history.execute(doc, Command::Batch(commands));
            project.dirty = true;
        }

        self.editor.pending_stroke.clear();
        self.editor.road_preview_left_world.clear();
        self.editor.road_preview_right_world.clear();
        self.editor.road_dialog_open = false;
        self.editor.active_tool = ActiveTool::None;
        self.invalidate_geometry();
    }

    pub(crate) fn cancel_road(&mut self) {
        self.editor.pending_stroke.clear();
        self.editor.road_preview_left_world.clear();
        self.editor.road_preview_right_world.clear();
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

fn normalize_new_and_existing_roads_for_junctions(
    document: &Document,
    new_road_id: ObjectId,
    new_road: &mut Object,
    new_centerline: &[DVec3],
) -> Result<Vec<Command>, RoadCenterlineRuleError> {
    let Some((&start, &end)) = new_centerline.first().zip(new_centerline.last()) else {
        return Ok(Vec::new());
    };

    let mut replacements = Vec::new();
    let mut working: Vec<Object> = document.objects().to_vec();
    for junction in unique_junction_points([start, end]) {
        let mut context = working.clone();
        context.push(new_road.clone());
        for object in &mut working {
            if object.id() == new_road_id {
                continue;
            }
            let before = object.clone();
            if normalize_road_object_at_junction(object, junction, &context)? && *object != before {
                replacements.push((before, object.clone()));
            }
        }
        let mut context = working.clone();
        context.push(new_road.clone());
        normalize_road_object_at_junction(new_road, junction, &context)?;
    }

    Ok(replacements
        .into_iter()
        .map(|(before, after)| Command::Replace { before, after })
        .collect())
}

fn unique_junction_points(points: [DVec3; 2]) -> Vec<DVec3> {
    let mut unique = Vec::new();
    for point in points {
        if unique
            .iter()
            .all(|&existing| !points_coincident(existing, point))
        {
            unique.push(point);
        }
    }
    unique
}

fn normalize_road_object_at_junction(
    object: &mut Object,
    junction: DVec3,
    all_roads: &[Object],
) -> Result<bool, RoadCenterlineRuleError> {
    let Object::Road {
        centerline, width, ..
    } = object
    else {
        return Ok(false);
    };
    if centerline.len() < 2 {
        return Ok(false);
    }
    let road_width = *width;

    let Some(junction_index) = ensure_junction_vertex(centerline, junction)? else {
        return Ok(false);
    };

    let mut changed = false;
    if junction_index > 0 {
        let branch = centerline[junction_index - 1].pos;
        if branch_is_inclined(junction, branch) {
            let clearance =
                road_flat_clearance_for_objects(junction, branch, road_width, all_roads);
            changed |= normalize_branch_flat_approach(centerline, junction_index, -1, clearance)?;
        }
    }
    let current_index = centerline
        .iter()
        .position(|vertex| points_coincident(vertex.pos, junction))
        .unwrap_or(junction_index);
    if current_index + 1 < centerline.len() {
        let branch = centerline[current_index + 1].pos;
        if branch_is_inclined(junction, branch) {
            let clearance =
                road_flat_clearance_for_objects(junction, branch, road_width, all_roads);
            changed |= normalize_branch_flat_approach(centerline, current_index, 1, clearance)?;
        }
    }

    Ok(changed)
}

fn ensure_junction_vertex(
    centerline: &mut Vec<PolyVertex>,
    junction: DVec3,
) -> Result<Option<usize>, RoadCenterlineRuleError> {
    if let Some(index) = centerline
        .iter()
        .position(|vertex| points_coincident(vertex.pos, junction))
    {
        return Ok(Some(index));
    }

    for index in 0..centerline.len() - 1 {
        let a = centerline[index].pos;
        let b = centerline[index + 1].pos;
        if let Some(t) = point_on_segment_t(junction, a, b) {
            let split = DVec3::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, junction.z);
            centerline.insert(index + 1, PolyVertex::straight(split));
            return Ok(Some(index + 1));
        }
    }

    Ok(None)
}

fn normalize_branch_flat_approach(
    centerline: &mut Vec<PolyVertex>,
    junction_index: usize,
    direction: isize,
    clearance: f64,
) -> Result<bool, RoadCenterlineRuleError> {
    let junction = centerline[junction_index].pos;
    let junction_z = junction.z;
    let mut changed = false;
    let mut current_index = junction_index;
    let mut travelled = 0.0;

    loop {
        let Some(next_index) = offset_index(current_index, direction, centerline.len()) else {
            return Ok(changed);
        };

        let current = centerline[current_index].pos;
        let next = centerline[next_index].pos;
        let segment_len = horizontal_distance(current, next);
        if segment_len < 1e-9 {
            return Err(RoadCenterlineRuleError::DegenerateSegment);
        }

        let remaining = clearance - travelled;
        if segment_len + 1e-9 < remaining {
            if (centerline[next_index].pos.z - junction_z).abs() > 1e-6 {
                centerline[next_index].pos.z = junction_z;
                changed = true;
            }
            travelled += segment_len;
            current_index = next_index;
            continue;
        }

        let t = (remaining / segment_len).clamp(0.0, 1.0);
        if t >= 1.0 - 1e-9 {
            if (centerline[next_index].pos.z - junction_z).abs() > 1e-6 {
                centerline[next_index].pos.z = junction_z;
                changed = true;
            }
            return Ok(changed);
        }

        let clearance_point = DVec3::new(
            current.x + (next.x - current.x) * t,
            current.y + (next.y - current.y) * t,
            junction_z,
        );

        if points_coincident(current, clearance_point) {
            return Ok(changed);
        }

        let insert_index = if direction > 0 {
            next_index
        } else {
            current_index
        };
        centerline.insert(insert_index, PolyVertex::straight(clearance_point));
        return Ok(true);
    }
}

fn offset_index(index: usize, direction: isize, len: usize) -> Option<usize> {
    if direction > 0 {
        (index + 1 < len).then_some(index + 1)
    } else {
        index.checked_sub(1)
    }
}

fn road_flat_clearance_for_objects(
    junction: DVec3,
    branch_toward: DVec3,
    road_width: f64,
    objects: &[Object],
) -> f64 {
    let branch_delta = branch_toward.truncate() - junction.truncate();
    let branch_len = branch_delta.length();
    if branch_len < 1e-9 {
        return ROAD_INTERSECTION_FLAT_CLEARANCE_M;
    }

    let branch_dir = branch_delta / branch_len;
    let own_half_width = (road_width * 0.5).max(0.0);
    connected_road_branches_for_objects(objects, junction)
        .into_iter()
        .filter_map(|(other_dir, other_width)| {
            let angle = branch_dir.dot(other_dir).clamp(-1.0, 1.0).acos();
            if angle < 1e-6 {
                return None;
            }

            let half_width = own_half_width.max(other_width * 0.5);
            let tangent = (angle * 0.5).tan();
            if tangent.abs() < 1e-9 {
                return None;
            }

            Some(ROAD_INTERSECTION_FLAT_CLEARANCE_M + half_width / tangent)
        })
        .fold(ROAD_INTERSECTION_FLAT_CLEARANCE_M, f64::max)
}

fn connected_road_branches_for_objects(
    objects: &[Object],
    junction: DVec3,
) -> Vec<(glam::DVec2, f64)> {
    let mut branches = Vec::new();
    for object in objects {
        let Object::Road {
            centerline, width, ..
        } = object
        else {
            continue;
        };
        if centerline.len() < 2 {
            continue;
        }

        for (index, vertex) in centerline.iter().enumerate() {
            if !points_coincident(vertex.pos, junction) {
                continue;
            }
            if index > 0 {
                push_branch_dir(&mut branches, centerline[index - 1].pos, junction, *width);
            }
            if index + 1 < centerline.len() {
                push_branch_dir(&mut branches, centerline[index + 1].pos, junction, *width);
            }
        }

        for segment in centerline.windows(2) {
            let a = segment[0].pos;
            let b = segment[1].pos;
            if point_on_segment_t(junction, a, b).is_some() {
                push_branch_dir(&mut branches, a, junction, *width);
                push_branch_dir(&mut branches, b, junction, *width);
            }
        }
    }
    branches
}

fn push_branch_dir(
    branches: &mut Vec<(glam::DVec2, f64)>,
    toward: DVec3,
    junction: DVec3,
    width: f64,
) {
    let delta = toward.truncate() - junction.truncate();
    let len = delta.length();
    if len < 1e-9 {
        return;
    }
    let dir = delta / len;
    if branches
        .iter()
        .any(|(existing, _)| existing.dot(dir) > 1.0 - 1e-8)
    {
        return;
    }
    branches.push((dir, width));
}

fn point_on_segment_t(point: DVec3, a: DVec3, b: DVec3) -> Option<f64> {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-10 {
        return None;
    }
    let t = (point - a).dot(ab) / len_sq;
    if !(1e-6..=1.0 - 1e-6).contains(&t) {
        return None;
    }
    let projected = a + ab * t;
    points_coincident(projected, point).then_some(t)
}

fn horizontal_distance(a: DVec3, b: DVec3) -> f64 {
    (b.truncate() - a.truncate()).length()
}

fn branch_is_inclined(junction: DVec3, branch: DVec3) -> bool {
    (branch.z - junction.z).abs() > 1e-6
}

fn points_coincident(a: DVec3, b: DVec3) -> bool {
    (a - b).length_squared() < 1e-8
}

#[derive(Clone)]
struct AttachedClip {
    inward_dir: glam::DVec2,
    segments: Vec<[DVec3; 2]>,
    clip_boundary: bool,
}

fn attached_polyline_clips(
    document: &crate::model::Document,
    centerline: &[DVec3],
) -> Vec<AttachedClip> {
    if centerline.len() < 2 {
        return Vec::new();
    }

    let mut clips = Vec::new();
    let start = centerline[0];
    let end = *centerline.last().unwrap();
    let start_dir = (centerline[1].truncate() - start.truncate()).normalize_or_zero();
    let end_dir =
        (centerline[centerline.len() - 2].truncate() - end.truncate()).normalize_or_zero();

    clips.extend(attached_polyline_clips_at(document, start, start_dir));
    clips.extend(attached_road_edge_clips_at(document, start, start_dir));
    if !points_coincident(start, end) {
        clips.extend(attached_polyline_clips_at(document, end, end_dir));
        clips.extend(attached_road_edge_clips_at(document, end, end_dir));
    }
    clips
}

fn attached_polyline_clips_at(
    document: &crate::model::Document,
    junction: DVec3,
    inward_dir: glam::DVec2,
) -> Vec<AttachedClip> {
    if inward_dir.length_squared() < 1e-12 {
        return Vec::new();
    }

    document
        .objects()
        .iter()
        .filter_map(|object| {
            let Object::Polyline { verts, closed, .. } = object else {
                return None;
            };
            if verts.len() < 2 {
                return None;
            }

            let mut segments: Vec<[DVec3; 2]> =
                verts.windows(2).map(|w| [w[0].pos, w[1].pos]).collect();
            if *closed && verts.len() >= 3 {
                segments.push([verts.last().unwrap().pos, verts[0].pos]);
            }

            let mut attached_segments = Vec::new();
            for segment in segments {
                if point_on_segment_t(junction, segment[0], segment[1]).is_some()
                    || points_coincident(junction, segment[0])
                    || points_coincident(junction, segment[1])
                {
                    attached_segments.push(segment);
                }
            }

            (!attached_segments.is_empty()).then_some(AttachedClip {
                inward_dir,
                segments: attached_segments,
                clip_boundary: true,
            })
        })
        .collect()
}

fn attached_road_edge_clips_at(
    document: &crate::model::Document,
    junction: DVec3,
    inward_dir: glam::DVec2,
) -> Vec<AttachedClip> {
    if inward_dir.length_squared() < 1e-12 {
        return Vec::new();
    }

    let mut segments = Vec::new();
    for object in document.objects() {
        let Object::Road {
            centerline,
            width,
            camber_degrees,
            shape,
            ..
        } = object
        else {
            continue;
        };
        if centerline.len() < 2 || !road_centerline_contains_junction(centerline, junction) {
            continue;
        }

        let mut cl: Vec<DVec3> = centerline.iter().map(|vertex| vertex.pos).collect();
        if points_coincident(cl[0], junction) {
            cl.insert(0, junction + inward_dir.extend(0.0));
        } else if points_coincident(
            *cl.last().expect("road has at least two vertices"),
            junction,
        ) {
            cl.push(junction + inward_dir.extend(0.0));
        }

        let half_w = *width / 2.0;
        let (left_z, right_z) = shape.z_offsets(*width, *camber_degrees);
        let left = edge_points_with_z_offset(geometric_offset(&cl, false, half_w, 0.0), left_z);
        let right = edge_points_with_z_offset(geometric_offset(&cl, false, -half_w, 0.0), right_z);
        segments.extend(left.windows(2).map(|pair| [pair[0], pair[1]]));
        segments.extend(right.windows(2).map(|pair| [pair[0], pair[1]]));
    }

    (!segments.is_empty())
        .then_some(AttachedClip {
            inward_dir,
            segments,
            clip_boundary: false,
        })
        .into_iter()
        .collect()
}

fn road_centerline_contains_junction(centerline: &[PolyVertex], junction: DVec3) -> bool {
    centerline
        .iter()
        .any(|vertex| points_coincident(vertex.pos, junction))
        || centerline
            .windows(2)
            .any(|pair| point_on_segment_t(junction, pair[0].pos, pair[1].pos).is_some())
}

fn edge_points_with_z_offset(points: Vec<DVec3>, z_offset: f64) -> Vec<DVec3> {
    points
        .into_iter()
        .map(|point| DVec3::new(point.x, point.y, point.z + z_offset))
        .collect()
}

fn taper_edge_to_attached_objects(edge: Vec<DVec3>, attachments: &[AttachedClip]) -> Vec<DVec3> {
    if attachments.is_empty() || edge.len() < 2 {
        return edge;
    }

    let mut tapered = Vec::with_capacity(edge.len() + attachments.len());
    for pair in edge.windows(2) {
        let a = pair[0];
        let b = pair[1];
        if !a.is_finite() || !b.is_finite() {
            push_preview_break(&mut tapered);
            continue;
        }

        let mut ts = vec![0.0, 1.0];
        for attachment in attachments {
            if let (Some(along_a), Some(along_b)) = (
                attachment_edge_along(a.truncate(), attachment),
                attachment_edge_along(b.truncate(), attachment),
            ) {
                add_along_split(&mut ts, along_a, along_b, 0.0);
                add_along_split(
                    &mut ts,
                    along_a,
                    along_b,
                    ROAD_INTERSECTION_FLAT_CLEARANCE_M,
                );
            }
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ts.dedup_by(|a, b| (*a - *b).abs() < 1e-8);

        for t in ts {
            let point = DVec3::new(
                a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t,
            );
            push_preview_point(
                &mut tapered,
                taper_edge_point_to_attached_objects(point, attachments),
            );
        }
    }

    tapered
}

fn add_along_split(ts: &mut Vec<f64>, along_a: f64, along_b: f64, target: f64) {
    let denom = along_b - along_a;
    if denom.abs() < 1e-9 {
        return;
    }
    let t = (target - along_a) / denom;
    if t > 1e-8 && t < 1.0 - 1e-8 {
        ts.push(t);
    }
}

fn taper_edge_point_to_attached_objects(mut point: DVec3, attachments: &[AttachedClip]) -> DVec3 {
    for attachment in attachments {
        let Some((along, attach_z)) = attachment_edge_contact(point.truncate(), attachment) else {
            continue;
        };
        if !(-1e-6..=ROAD_INTERSECTION_FLAT_CLEARANCE_M).contains(&along) {
            continue;
        }
        let scale = (along / ROAD_INTERSECTION_FLAT_CLEARANCE_M).clamp(0.0, 1.0);
        point.z = attach_z + (point.z - attach_z) * scale;
    }
    point
}

fn snap_edge_endpoints_to_attached_objects(
    mut edge: Vec<DVec3>,
    attachments: &[AttachedClip],
) -> Vec<DVec3> {
    if attachments.is_empty() {
        return edge;
    }
    let attachments: Vec<_> = attachments
        .iter()
        .filter(|attachment| attachment.clip_boundary)
        .collect();
    if attachments.is_empty() {
        return edge;
    }

    if let Some(index) = edge.iter().position(|point| point.is_finite()) {
        snap_edge_endpoint_to_attachment(&mut edge[index], &attachments);
    }
    if let Some(index) = edge.iter().rposition(|point| point.is_finite()) {
        snap_edge_endpoint_to_attachment(&mut edge[index], &attachments);
    }

    edge
}

fn snap_edge_endpoint_to_attachment(point: &mut DVec3, attachments: &[&AttachedClip]) {
    let Some((attachment, snapped)) = attachments
        .iter()
        .filter_map(|attachment| {
            attachment_edge_intersection(*point, attachment).map(|snapped| (attachment, snapped))
        })
        .min_by(|(_, a), (_, b)| {
            (a.truncate() - point.truncate())
                .length_squared()
                .partial_cmp(&(b.truncate() - point.truncate()).length_squared())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    else {
        return;
    };
    let attach_z = attachment_z_at(snapped.truncate(), attachment).unwrap_or(snapped.z);
    *point = DVec3::new(snapped.x, snapped.y, attach_z);
}

fn attachment_edge_intersection(point: DVec3, attachment: &AttachedClip) -> Option<DVec3> {
    attachment
        .segments
        .iter()
        .filter_map(|segment| {
            line_segment_intersection_xy(
                point.truncate(),
                attachment.inward_dir,
                segment[0].truncate(),
                segment[1].truncate(),
            )
        })
        .map(|xy| {
            DVec3::new(
                xy.x,
                xy.y,
                attachment_z_at(xy, attachment).unwrap_or(point.z),
            )
        })
        .min_by(|a, b| {
            (a.truncate() - point.truncate())
                .length_squared()
                .partial_cmp(&(b.truncate() - point.truncate()).length_squared())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn attachment_edge_along(point: glam::DVec2, attachment: &AttachedClip) -> Option<f64> {
    attachment_edge_contact(point, attachment).map(|(along, _)| along)
}

fn attachment_edge_contact(point: glam::DVec2, attachment: &AttachedClip) -> Option<(f64, f64)> {
    attachment
        .segments
        .iter()
        .filter_map(|segment| {
            line_segment_intersection_xy(
                point,
                attachment.inward_dir,
                segment[0].truncate(),
                segment[1].truncate(),
            )
            .map(|contact| {
                (
                    (point - contact).dot(attachment.inward_dir),
                    segment_z_at_xy(contact, *segment),
                )
            })
        })
        .min_by(|a, b| {
            a.0.abs()
                .partial_cmp(&b.0.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn attachment_z_at(point: glam::DVec2, attachment: &AttachedClip) -> Option<f64> {
    attachment_edge_contact(point, attachment).map(|(_, z)| z)
}

fn segment_z_at_xy(point: glam::DVec2, segment: [DVec3; 2]) -> f64 {
    let a = segment[0].truncate();
    let b = segment[1].truncate();
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-12 {
        return (segment[0].z + segment[1].z) * 0.5;
    }
    let t = ((point - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    segment[0].z + (segment[1].z - segment[0].z) * t
}

fn line_segment_intersection_xy(
    line_point: glam::DVec2,
    line_dir: glam::DVec2,
    seg_a: glam::DVec2,
    seg_b: glam::DVec2,
) -> Option<glam::DVec2> {
    let seg = seg_b - seg_a;
    let denom = line_dir.perp_dot(seg);
    if denom.abs() < 1e-10 {
        return None;
    }
    let delta = seg_a - line_point;
    let line_t = delta.perp_dot(seg) / denom;
    let seg_t = delta.perp_dot(line_dir) / denom;
    if (-1e-8..=1.0 + 1e-8).contains(&seg_t) {
        Some(line_point + line_dir * line_t)
    } else {
        None
    }
}

fn clip_edge_to_attached_objects(edge: Vec<DVec3>, attachments: &[AttachedClip]) -> Vec<DVec3> {
    if edge.len() < 2 {
        return edge;
    }

    if attachments.is_empty() {
        return edge;
    }
    let attachments: Vec<_> = attachments
        .iter()
        .filter(|attachment| attachment.clip_boundary)
        .collect();
    if attachments.is_empty() {
        return edge;
    }

    let mut clipped = Vec::with_capacity(edge.len());
    for pair in edge.windows(2) {
        let pa = pair[0];
        let pb = pair[1];
        if !pa.is_finite() || !pb.is_finite() {
            continue;
        }

        let pa2 = pa.truncate();
        let pb2 = pb.truncate();
        let mut ts = vec![0.0, 1.0];
        for attachment in &attachments {
            for segment in &attachment.segments {
                if let Some(t) =
                    road_edge_seg_t(pa2, pb2, segment[0].truncate(), segment[1].truncate())
                {
                    ts.push(t);
                }
            }
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ts.dedup_by(|a, b| (*a - *b).abs() < 1e-8);

        for w in ts.windows(2) {
            let (t0, t1) = (w[0], w[1]);
            if t1 - t0 < 1e-8 {
                continue;
            }
            let mid2d = pa2 + (pb2 - pa2) * ((t0 + t1) * 0.5);
            if attachments
                .iter()
                .filter_map(|attachment| attachment_edge_along(mid2d, attachment))
                .any(|along| along < -1e-8)
            {
                push_preview_break(&mut clipped);
                continue;
            }

            let interp = |t: f64| {
                DVec3::new(
                    pa.x + (pb.x - pa.x) * t,
                    pa.y + (pb.y - pa.y) * t,
                    pa.z + (pb.z - pa.z) * t,
                )
            };
            push_preview_point(&mut clipped, interp(t0));
            push_preview_point(&mut clipped, interp(t1));
        }
    }

    clipped
}

fn push_preview_point(points: &mut Vec<DVec3>, point: DVec3) {
    if points
        .last()
        .is_some_and(|last| last.is_finite() && (*last - point).length_squared() < 1e-12)
    {
        return;
    }
    points.push(point);
}

fn push_preview_break(points: &mut Vec<DVec3>) {
    if points.last().is_some_and(|last| !last.is_finite()) {
        return;
    }
    points.push(DVec3::splat(f64::NAN));
}

fn road_edge_seg_t(p: glam::DVec2, q: glam::DVec2, a: glam::DVec2, b: glam::DVec2) -> Option<f64> {
    let d = q - p;
    let e = b - a;
    let denom = d.x * e.y - d.y * e.x;
    if denom.abs() < 1e-10 {
        return None;
    }
    let t = ((a.x - p.x) * e.y - (a.y - p.y) * e.x) / denom;
    let u = ((a.x - p.x) * d.y - (a.y - p.y) * d.x) / denom;
    if t > -1e-8 && t < 1.0 + 1e-8 && u > -1e-8 && u < 1.0 + 1e-8 {
        Some(t.clamp(0.0, 1.0))
    } else {
        None
    }
}
