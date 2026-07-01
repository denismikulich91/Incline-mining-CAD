//! Road scene assembly and attachment clipping helpers.

use glam::DVec3;

use crate::{
    model::{Document, Object, geometry::geometric_offset},
    ui::state::{ActiveTool, EditorState},
};

pub(crate) fn build_render_road_data(document: &Document, editor: &EditorState) -> Vec<RoadData> {
    let road_junction_phantoms = road_junction_phantoms(document, editor);
    let preview_junctions = preview_junctions(document, editor);
    let mut all_road_data: Vec<RoadData> = document
        .objects()
        .iter()
        .filter_map(|obj| {
            if let Object::Road {
                id,
                centerline,
                width,
                camber_degrees,
                shape,
                ..
            } = obj
            {
                if centerline.len() < 2 {
                    return None;
                }
                let half_w = width / 2.0;
                let (left_z, right_z) = shape.z_offsets(*width, *camber_degrees);
                let (prepend_ph, append_ph) = road_junction_phantoms
                    .get(id)
                    .copied()
                    .unwrap_or((None, None));
                let mut raw_cl: Vec<DVec3> = centerline.iter().map(|v| v.pos).collect();
                for &junction in &preview_junctions {
                    normalize_render_centerline_at_junction(&mut raw_cl, junction);
                }
                let attachments = render_attached_polyline_clips(document, &raw_cl);
                let mut cl = raw_cl.clone();
                let draw_start = if let Some(p) = prepend_ph {
                    cl.insert(0, p);
                    1
                } else {
                    0
                };
                let draw_end_trim = if let Some(p) = append_ph {
                    cl.push(p);
                    1
                } else {
                    0
                };
                let mut left_pts = geometric_offset(&cl, false, half_w, 0.0);
                let mut right_pts = geometric_offset(&cl, false, -half_w, 0.0);
                render_snap_edge_endpoints_to_attachments(&mut left_pts, &attachments);
                render_snap_edge_endpoints_to_attachments(&mut right_pts, &attachments);
                let draw_end = left_pts.len().saturating_sub(draw_end_trim);
                Some(RoadData {
                    id: Some(*id),
                    left_pts,
                    right_pts,
                    draw_start,
                    draw_end,
                    left_z,
                    right_z,
                    half_w,
                    cl_raw: raw_cl,
                    attachments,
                })
            } else {
                None
            }
        })
        .collect();

    if editor.active_tool == ActiveTool::MakeRoad {
        let mut preview_centerline = editor.pending_stroke.clone();
        if let Some(cursor) = editor.cursor_world {
            preview_centerline.push(cursor);
        }
        if preview_centerline.len() >= 2
            && crate::model::geometry::validate_road_segment_angles(
                &preview_centerline,
                editor.road_max_angle_degrees,
            )
            .is_ok()
            && let Ok(preview_centerline) =
                crate::model::geometry::road_centerline_with_intersection_flats(
                    &preview_centerline,
                    document,
                    editor.road_width,
                )
        {
            let half_w = editor.road_width / 2.0;
            let (left_z, right_z) = editor
                .road_shape
                .z_offsets(editor.road_width, editor.road_camber_degrees);
            let left_pts = geometric_offset(&preview_centerline, false, half_w, 0.0);
            let right_pts = geometric_offset(&preview_centerline, false, -half_w, 0.0);
            all_road_data.push(RoadData {
                id: None,
                left_pts,
                right_pts,
                draw_start: 0,
                draw_end: preview_centerline.len(),
                left_z,
                right_z,
                half_w,
                cl_raw: preview_centerline,
                attachments: Vec::new(),
            });
        }
    }

    let road_attachment_clips = render_road_edge_attachment_clips(&all_road_data);
    for (road, clips) in all_road_data.iter_mut().zip(road_attachment_clips) {
        road.attachments.extend(clips);
    }
    all_road_data
}

fn road_junction_phantoms(
    document: &Document,
    editor: &EditorState,
) -> std::collections::HashMap<crate::model::ObjectId, (Option<DVec3>, Option<DVec3>)> {
    struct RoadEp {
        id: crate::model::ObjectId,
        start: DVec3,
        end: DVec3,
        second: DVec3,
        penult: DVec3,
        cl: Vec<DVec3>,
    }
    let eps: Vec<RoadEp> = document
        .objects()
        .iter()
        .filter_map(|obj| {
            if let Object::Road { id, centerline, .. } = obj {
                let n = centerline.len();
                if n >= 2 {
                    return Some(RoadEp {
                        id: *id,
                        start: centerline[0].pos,
                        end: centerline[n - 1].pos,
                        second: centerline[1].pos,
                        penult: centerline[n - 2].pos,
                        cl: centerline.iter().map(|v| v.pos).collect(),
                    });
                }
            }
            None
        })
        .collect();
    let mut map = std::collections::HashMap::new();
    for ep in &eps {
        let mut start_candidates = Vec::new();
        let mut end_candidates = Vec::new();
        for other in &eps {
            if other.id == ep.id {
                continue;
            }
            if (other.start - ep.start).length_squared() < 1e-10 {
                start_candidates.push(other.second);
            }
            if (other.end - ep.start).length_squared() < 1e-10 {
                start_candidates.push(other.penult);
            }
            if (other.start - ep.end).length_squared() < 1e-10 {
                end_candidates.push(other.second);
            }
            if (other.end - ep.end).length_squared() < 1e-10 {
                end_candidates.push(other.penult);
            }
        }
        if start_candidates.len() == 1 {
            map.entry(ep.id).or_insert((None, None)).0 = start_candidates.first().copied();
        }
        if end_candidates.len() == 1 {
            map.entry(ep.id).or_insert((None, None)).1 = end_candidates.first().copied();
        }
    }

    for ep in &eps {
        for other in &eps {
            if other.id == ep.id {
                continue;
            }
            for pair in other.cl.windows(2) {
                let a = pair[0];
                let b = pair[1];
                let ab = b - a;
                let len_sq = ab.length_squared();
                if len_sq < 1e-10 {
                    continue;
                }
                let seg_dir = ab / len_sq.sqrt();
                let t_s = (ep.start - a).dot(ab) / len_sq;
                if t_s > 1e-6
                    && t_s < 1.0 - 1e-6
                    && (ep.start - (a + ab * t_s)).length_squared() < 1e-8
                {
                    let entry = map.entry(ep.id).or_insert((None, None));
                    if entry.0.is_none() {
                        entry.0 = Some(ep.start - seg_dir);
                    }
                }
                let t_e = (ep.end - a).dot(ab) / len_sq;
                if t_e > 1e-6
                    && t_e < 1.0 - 1e-6
                    && (ep.end - (a + ab * t_e)).length_squared() < 1e-8
                {
                    let entry = map.entry(ep.id).or_insert((None, None));
                    if entry.1.is_none() {
                        entry.1 = Some(ep.end + seg_dir);
                    }
                }
            }
        }
    }

    if editor.active_tool == ActiveTool::MakeRoad {
        let preview_second = if editor.pending_stroke.len() >= 2 {
            Some(editor.pending_stroke[1])
        } else {
            editor.cursor_world
        };
        if let (Some(first), Some(second)) =
            (editor.pending_stroke.first().copied(), preview_second)
        {
            for ep in &eps {
                if (ep.end - first).length_squared() < 1e-10 {
                    map.entry(ep.id).or_insert((None, None)).1 = Some(second);
                }
                if (ep.start - first).length_squared() < 1e-10 {
                    map.entry(ep.id).or_insert((None, None)).0 = Some(second);
                }
            }
        }
    }

    map
}

fn preview_junctions(document: &Document, editor: &EditorState) -> Vec<DVec3> {
    if editor.active_tool != ActiveTool::MakeRoad {
        return Vec::new();
    }
    let mut preview_centerline = editor.pending_stroke.clone();
    if let Some(cursor) = editor.cursor_world {
        preview_centerline.push(cursor);
    }
    crate::model::geometry::validate_road_segment_angles(
        &preview_centerline,
        editor.road_max_angle_degrees,
    )
    .ok()
    .and_then(|_| {
        crate::model::geometry::road_centerline_with_intersection_flats(
            &preview_centerline,
            document,
            editor.road_width,
        )
        .ok()
    })
    .and_then(|centerline| {
        centerline
            .first()
            .copied()
            .zip(centerline.last().copied())
            .map(|(start, end)| unique_points([start, end]))
    })
    .unwrap_or_default()
}

/// Returns the t-parameter on segment PQ where it crosses segment AB, or `None` if they don't
/// intersect.
pub(crate) fn road_edge_seg_t(
    p: glam::DVec2,
    q: glam::DVec2,
    a: glam::DVec2,
    b: glam::DVec2,
) -> Option<f64> {
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

#[derive(Clone)]
pub(crate) struct RenderAttachedClip {
    inward_dir: glam::DVec2,
    pub(crate) segments: Vec<[DVec3; 2]>,
    pub(crate) clip_boundary: bool,
}

pub(crate) struct RoadData {
    pub(crate) id: Option<crate::model::ObjectId>,
    pub(crate) left_pts: Vec<DVec3>,
    pub(crate) right_pts: Vec<DVec3>,
    pub(crate) draw_start: usize,
    pub(crate) draw_end: usize,
    pub(crate) left_z: f64,
    pub(crate) right_z: f64,
    pub(crate) half_w: f64,
    pub(crate) cl_raw: Vec<DVec3>,
    pub(crate) attachments: Vec<RenderAttachedClip>,
}

struct RenderRoadAttachmentSource {
    id: Option<crate::model::ObjectId>,
    cl_raw: Vec<DVec3>,
    left_segments: Vec<[DVec3; 2]>,
    right_segments: Vec<[DVec3; 2]>,
}

pub(crate) fn render_attached_polyline_clips(
    document: &Document,
    centerline: &[DVec3],
) -> Vec<RenderAttachedClip> {
    if centerline.len() < 2 {
        return Vec::new();
    }
    let start = centerline[0];
    let end = *centerline.last().unwrap();
    let start_dir = (centerline[1].truncate() - start.truncate()).normalize_or_zero();
    let end_dir =
        (centerline[centerline.len() - 2].truncate() - end.truncate()).normalize_or_zero();

    let mut clips = render_attached_polyline_clips_at(document, start, start_dir);
    if !render_points_coincident(start, end) {
        clips.extend(render_attached_polyline_clips_at(document, end, end_dir));
    }
    clips
}

fn render_attached_polyline_clips_at(
    document: &Document,
    junction: DVec3,
    inward_dir: glam::DVec2,
) -> Vec<RenderAttachedClip> {
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
                if render_point_on_segment_t(junction, segment[0], segment[1]).is_some()
                    || render_points_coincident(junction, segment[0])
                    || render_points_coincident(junction, segment[1])
                {
                    attached_segments.push(segment);
                }
            }

            (!attached_segments.is_empty()).then_some(RenderAttachedClip {
                inward_dir,
                segments: attached_segments,
                clip_boundary: true,
            })
        })
        .collect()
}

pub(crate) fn render_tapered_edge_point(
    base: DVec3,
    z_off: f64,
    attachments: &[RenderAttachedClip],
) -> DVec3 {
    let mut point = DVec3::new(base.x, base.y, base.z + z_off);
    for attachment in attachments {
        let Some((along, attach_z)) = render_attachment_edge_contact(base.truncate(), attachment)
        else {
            continue;
        };
        if !(-1e-6..=crate::model::geometry::ROAD_INTERSECTION_FLAT_CLEARANCE_M).contains(&along) {
            continue;
        }
        let scale =
            (along / crate::model::geometry::ROAD_INTERSECTION_FLAT_CLEARANCE_M).clamp(0.0, 1.0);
        point.z = attach_z + (point.z - attach_z) * scale;
    }
    point
}

pub(crate) fn add_render_along_split(ts: &mut Vec<f64>, along_a: f64, along_b: f64, target: f64) {
    let denom = along_b - along_a;
    if denom.abs() < 1e-9 {
        return;
    }
    let t = (target - along_a) / denom;
    if t > 1e-8 && t < 1.0 - 1e-8 {
        ts.push(t);
    }
}

pub(crate) fn render_snap_edge_endpoints_to_attachments(
    edge: &mut [DVec3],
    attachments: &[RenderAttachedClip],
) {
    let attachments: Vec<_> = attachments
        .iter()
        .filter(|attachment| attachment.clip_boundary)
        .collect();
    if attachments.is_empty() {
        return;
    }
    if let Some(index) = edge.iter().position(|point| point.is_finite()) {
        render_snap_edge_endpoint_to_attachment(&mut edge[index], &attachments);
    }
    if let Some(index) = edge.iter().rposition(|point| point.is_finite()) {
        render_snap_edge_endpoint_to_attachment(&mut edge[index], &attachments);
    }
}

fn render_snap_edge_endpoint_to_attachment(point: &mut DVec3, attachments: &[&RenderAttachedClip]) {
    let Some((attachment, snapped)) = attachments
        .iter()
        .filter_map(|attachment| {
            render_attachment_edge_intersection(*point, attachment)
                .map(|snapped| (attachment, snapped))
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
    let attach_z = render_attachment_z_at(snapped.truncate(), attachment).unwrap_or(snapped.z);
    *point = DVec3::new(snapped.x, snapped.y, attach_z);
}

fn render_attachment_edge_intersection(
    point: DVec3,
    attachment: &RenderAttachedClip,
) -> Option<DVec3> {
    attachment
        .segments
        .iter()
        .filter_map(|segment| {
            render_line_segment_intersection_xy(
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
                render_attachment_z_at(xy, attachment).unwrap_or(point.z),
            )
        })
        .min_by(|a, b| {
            (a.truncate() - point.truncate())
                .length_squared()
                .partial_cmp(&(b.truncate() - point.truncate()).length_squared())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

pub(crate) fn render_attachment_edge_along(
    point: glam::DVec2,
    attachment: &RenderAttachedClip,
) -> Option<f64> {
    render_attachment_edge_contact(point, attachment).map(|(along, _)| along)
}

fn render_attachment_edge_contact(
    point: glam::DVec2,
    attachment: &RenderAttachedClip,
) -> Option<(f64, f64)> {
    attachment
        .segments
        .iter()
        .filter_map(|segment| {
            render_line_segment_intersection_xy(
                point,
                attachment.inward_dir,
                segment[0].truncate(),
                segment[1].truncate(),
            )
            .map(|contact| {
                (
                    (point - contact).dot(attachment.inward_dir),
                    render_segment_z_at_xy(contact, *segment),
                )
            })
        })
        .min_by(|a, b| {
            a.0.abs()
                .partial_cmp(&b.0.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn render_attachment_z_at(point: glam::DVec2, attachment: &RenderAttachedClip) -> Option<f64> {
    render_attachment_edge_contact(point, attachment).map(|(_, z)| z)
}

fn render_segment_z_at_xy(point: glam::DVec2, segment: [DVec3; 2]) -> f64 {
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

pub(crate) fn render_road_edge_attachment_clips(
    roads: &[RoadData],
) -> Vec<Vec<RenderAttachedClip>> {
    let sources: Vec<RenderRoadAttachmentSource> = roads
        .iter()
        .map(|road| RenderRoadAttachmentSource {
            id: road.id,
            cl_raw: road.cl_raw.clone(),
            left_segments: render_final_edge_segments(&road.left_pts, road.left_z),
            right_segments: render_final_edge_segments(&road.right_pts, road.right_z),
        })
        .collect();

    roads
        .iter()
        .map(|road| {
            if road.cl_raw.len() < 2 {
                return Vec::new();
            }
            let mut clips = Vec::new();
            let start = road.cl_raw[0];
            let start_dir = (road.cl_raw[1].truncate() - start.truncate()).normalize_or_zero();
            clips.extend(render_road_edge_attachment_clips_at(
                road.id, start, start_dir, &sources,
            ));

            let end = *road.cl_raw.last().expect("road has at least two vertices");
            if !render_points_coincident(start, end) {
                let end_dir = (road.cl_raw[road.cl_raw.len() - 2].truncate() - end.truncate())
                    .normalize_or_zero();
                clips.extend(render_road_edge_attachment_clips_at(
                    road.id, end, end_dir, &sources,
                ));
            }
            clips
        })
        .collect()
}

fn render_road_edge_attachment_clips_at(
    road_id: Option<crate::model::ObjectId>,
    junction: DVec3,
    inward_dir: glam::DVec2,
    sources: &[RenderRoadAttachmentSource],
) -> Vec<RenderAttachedClip> {
    if inward_dir.length_squared() < 1e-12 {
        return Vec::new();
    }

    let mut segments = Vec::new();
    for source in sources {
        if source.id == road_id || !render_centerline_contains_junction(&source.cl_raw, junction) {
            continue;
        }
        segments.extend(source.left_segments.iter().copied());
        segments.extend(source.right_segments.iter().copied());
    }

    (!segments.is_empty())
        .then_some(RenderAttachedClip {
            inward_dir,
            segments,
            clip_boundary: false,
        })
        .into_iter()
        .collect()
}

fn render_final_edge_segments(edge: &[DVec3], z_offset: f64) -> Vec<[DVec3; 2]> {
    edge.windows(2)
        .map(|pair| {
            [
                DVec3::new(pair[0].x, pair[0].y, pair[0].z + z_offset),
                DVec3::new(pair[1].x, pair[1].y, pair[1].z + z_offset),
            ]
        })
        .collect()
}

fn render_centerline_contains_junction(centerline: &[DVec3], junction: DVec3) -> bool {
    centerline
        .iter()
        .any(|&point| render_points_coincident(point, junction))
        || centerline
            .windows(2)
            .any(|pair| render_point_on_segment_t(junction, pair[0], pair[1]).is_some())
}

fn render_line_segment_intersection_xy(
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

pub(crate) fn normalize_render_centerline_at_junction(
    centerline: &mut Vec<DVec3>,
    junction: DVec3,
) {
    if centerline.len() < 2 {
        return;
    }
    let Some(junction_index) = ensure_render_junction_vertex(centerline, junction) else {
        return;
    };
    if junction_index > 0 {
        let branch = centerline[junction_index - 1];
        if render_branch_is_inclined(junction, branch) {
            normalize_render_branch_flat(centerline, junction_index, -1);
        }
    }
    let current_index = centerline
        .iter()
        .position(|&point| render_points_coincident(point, junction))
        .unwrap_or(junction_index);
    if current_index + 1 < centerline.len() {
        let branch = centerline[current_index + 1];
        if render_branch_is_inclined(junction, branch) {
            normalize_render_branch_flat(centerline, current_index, 1);
        }
    }
}

fn ensure_render_junction_vertex(centerline: &mut Vec<DVec3>, junction: DVec3) -> Option<usize> {
    if let Some(index) = centerline
        .iter()
        .position(|&point| render_points_coincident(point, junction))
    {
        return Some(index);
    }
    for index in 0..centerline.len() - 1 {
        let a = centerline[index];
        let b = centerline[index + 1];
        if let Some(t) = render_point_on_segment_t(junction, a, b) {
            let split = DVec3::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, junction.z);
            centerline.insert(index + 1, split);
            return Some(index + 1);
        }
    }
    None
}

fn normalize_render_branch_flat(
    centerline: &mut Vec<DVec3>,
    junction_index: usize,
    direction: isize,
) {
    let junction = centerline[junction_index];
    let junction_z = junction.z;
    let mut current_index = junction_index;
    let mut travelled = 0.0;
    loop {
        let Some(next_index) = render_offset_index(current_index, direction, centerline.len())
        else {
            return;
        };
        let current = centerline[current_index];
        let next = centerline[next_index];
        let segment_len = (next.truncate() - current.truncate()).length();
        if segment_len < 1e-9 {
            return;
        }
        let remaining = crate::model::geometry::ROAD_INTERSECTION_FLAT_CLEARANCE_M - travelled;
        if segment_len + 1e-9 < remaining {
            centerline[next_index].z = junction_z;
            travelled += segment_len;
            current_index = next_index;
            continue;
        }
        let t = (remaining / segment_len).clamp(0.0, 1.0);
        if t >= 1.0 - 1e-9 {
            centerline[next_index].z = junction_z;
            return;
        }
        let clearance = DVec3::new(
            current.x + (next.x - current.x) * t,
            current.y + (next.y - current.y) * t,
            junction_z,
        );
        let insert_index = if direction > 0 {
            next_index
        } else {
            current_index
        };
        centerline.insert(insert_index, clearance);
        return;
    }
}

fn render_offset_index(index: usize, direction: isize, len: usize) -> Option<usize> {
    if direction > 0 {
        (index + 1 < len).then_some(index + 1)
    } else {
        index.checked_sub(1)
    }
}

fn render_point_on_segment_t(point: DVec3, a: DVec3, b: DVec3) -> Option<f64> {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-10 {
        return None;
    }
    let t = (point - a).dot(ab) / len_sq;
    if !(1e-6..=1.0 - 1e-6).contains(&t) {
        return None;
    }
    render_points_coincident(a + ab * t, point).then_some(t)
}

fn render_points_coincident(a: DVec3, b: DVec3) -> bool {
    (a - b).length_squared() < 1e-8
}

fn render_branch_is_inclined(junction: DVec3, branch: DVec3) -> bool {
    (branch.z - junction.z).abs() > 1e-6
}

pub(crate) fn unique_points(points: [DVec3; 2]) -> Vec<DVec3> {
    let mut unique = Vec::new();
    for point in points {
        if unique
            .iter()
            .all(|&existing| !render_points_coincident(existing, point))
        {
            unique.push(point);
        }
    }
    unique
}

/// Returns `true` if the 2D point `p` lies strictly inside a road body
/// (within `half_w` of any centerline segment, with a small inset so boundary
/// points are not treated as inside).
pub(crate) fn road_point_in_body(p: glam::DVec2, cl: &[DVec3], half_w: f64) -> bool {
    let threshold = half_w * 0.98;
    for pair in cl.windows(2) {
        let a = pair[0].truncate();
        let b = pair[1].truncate();
        let ab = b - a;
        let len = ab.length();
        if len < 1e-10 {
            continue;
        }
        let dir = ab / len;
        let t = (p - a).dot(dir).clamp(0.0, len);
        if (p - (a + dir * t)).length() < threshold {
            return true;
        }
    }
    false
}
