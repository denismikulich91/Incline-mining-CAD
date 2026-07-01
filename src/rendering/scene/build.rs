//! Document/editor scene assembly entry points.

use glam::{DVec3, Mat4, Vec3};
use lyon::tessellation::VertexBuffers;

use crate::{
    model::{Document, Object, SceneEntityId},
    rendering::{
        StrokeVertex, Vertex,
        geometry::{
            DrawContext, draw_bulge_segment, draw_line, draw_round_join, draw_screen_cross,
        },
        graphics::{
            DOC_LINE_WIDTH, DOC_TEXT_FONT_SIZE, INVALID_PREVIEW_COLOR, PREVIEW_COLOR,
            TEXT_EDIT_INDICATOR_COLOR, YELLOW_HIGHLIGHT_COLOR, make_translucent,
            text_bounds_corners,
        },
        pick::{PickRecord, TextPickRecord},
        scene::{
            document::{
                draw_origin_marker, fill_polygon_hatch, fill_polygon_solid, polygon_plane_frame,
            },
            road::*,
        },
        text::{CachedTextArea, Text, TextBox, TextSystem},
    },
    ui::state::{ActiveTool, EditorState},
};

pub(crate) struct DocumentSceneBuildInput<'a> {
    pub(crate) editor: &'a EditorState,
    pub(crate) document: &'a Document,
    pub(crate) text_system: &'a mut TextSystem,
    pub(crate) lyon_buffer: &'a mut VertexBuffers<Vertex, u32>,
    pub(crate) stroke_vertex_buf: &'a mut Vec<StrokeVertex>,
    pub(crate) stroke_index_buf: &'a mut Vec<u32>,
    pub(crate) cached_textareas: &'a mut Vec<CachedTextArea>,
    pub(crate) textarea_depths: &'a mut Vec<f32>,
    pub(crate) pick_records: &'a mut Vec<PickRecord>,
    pub(crate) text_pick_records: &'a mut Vec<TextPickRecord>,
    pub(crate) scene_origin: DVec3,
    pub(crate) scale_factor: f32,
    pub(crate) selection_color: [f32; 4],
}

pub(crate) fn rebuild_document_scene(input: DocumentSceneBuildInput<'_>) {
    let DocumentSceneBuildInput {
        editor,
        document,
        text_system,
        lyon_buffer,
        stroke_vertex_buf,
        stroke_index_buf,
        cached_textareas,
        textarea_depths,
        pick_records,
        text_pick_records,
        scene_origin,
        scale_factor,
        selection_color,
    } = input;

    lyon_buffer.indices.clear();
    lyon_buffer.vertices.clear();
    stroke_index_buf.clear();
    stroke_vertex_buf.clear();
    cached_textareas.clear();
    textarea_depths.clear();
    pick_records.clear();
    text_pick_records.clear();

    {
        let mut draw_ctx = DrawContext {
            stroke_vertex_buf,
            stroke_index_buf,
            fill_vertex_buf: &mut lyon_buffer.vertices,
            fill_index_buf: &mut lyon_buffer.indices,
            scene_origin,
            scale_factor,
        };

        draw_origin_marker(&mut draw_ctx);

        let all_road_data = build_render_road_data(document, editor);

        // The document is the sole source of geometry. Draw each object in world
        // space, honoring layer visibility / hide / freeze, and record a pick range
        // unless frozen so objects select and highlight.
        for object in document.objects() {
            let handle = SceneEntityId::Object(object.id());
            let layer_visible = document
                .layer(object.layer())
                .map(|layer| layer.visible)
                .unwrap_or(true);
            if !layer_visible || editor.hidden_handles.contains(&handle) {
                continue;
            }
            let rgba = document.object_rgba(object);
            let stroke_start = draw_ctx.stroke_vertex_buf.len() as u32;
            let stroke_index_start = draw_ctx.stroke_index_buf.len() as u32;
            let fill_start = draw_ctx.fill_vertex_buf.len() as u32;
            let fill_index_start = draw_ctx.fill_index_buf.len() as u32;
            match object {
                Object::Point { pos, .. } => {
                    draw_screen_cross(&mut draw_ctx, *pos, 6.0, DOC_LINE_WIDTH, rgba);
                }
                Object::Polyline {
                    verts,
                    closed,
                    fill,
                    line_weight,
                    ..
                } => {
                    let line_rgba = rgba;
                    let fill_rgba = document.object_fill_rgba(object);
                    for segment in verts.windows(2) {
                        draw_bulge_segment(
                            &mut draw_ctx,
                            segment[0].pos,
                            segment[1].pos,
                            segment[0].bulge,
                            *line_weight,
                            line_rgba,
                        );
                    }
                    if *closed
                        && verts.len() >= 2
                        && let (Some(first), Some(last)) = (verts.first(), verts.last())
                    {
                        draw_bulge_segment(
                            &mut draw_ctx,
                            last.pos,
                            first.pos,
                            last.bulge,
                            *line_weight,
                            line_rgba,
                        );
                    }
                    let join_vertices: &[crate::model::PolyVertex] = if *closed {
                        verts
                    } else if verts.len() > 2 {
                        &verts[1..verts.len() - 1]
                    } else {
                        &[]
                    };
                    for vertex in join_vertices {
                        draw_round_join(&mut draw_ctx, vertex.pos, *line_weight, line_rgba);
                    }
                    if *closed && verts.len() >= 3 {
                        match fill {
                            crate::model::FillStyle::Solid => {
                                fill_polygon_solid(
                                    draw_ctx.fill_vertex_buf,
                                    draw_ctx.fill_index_buf,
                                    verts,
                                    fill_rgba,
                                    draw_ctx.scene_origin,
                                );
                            }
                            crate::model::FillStyle::Slashes | crate::model::FillStyle::Crosses => {
                                let hatch_spacing = {
                                    let (centroid, axis_u, axis_v) = polygon_plane_frame(verts);
                                    let pts_2d: Vec<[f64; 2]> = verts
                                        .iter()
                                        .map(|v| {
                                            let d = v.pos - centroid;
                                            [d.dot(axis_u), d.dot(axis_v)]
                                        })
                                        .collect();
                                    let min_x =
                                        pts_2d.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
                                    let max_x = pts_2d
                                        .iter()
                                        .map(|p| p[0])
                                        .fold(f64::NEG_INFINITY, f64::max);
                                    let min_y =
                                        pts_2d.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
                                    let max_y = pts_2d
                                        .iter()
                                        .map(|p| p[1])
                                        .fold(f64::NEG_INFINITY, f64::max);
                                    ((max_x - min_x).max(max_y - min_y) / 15.0).max(f64::EPSILON)
                                };
                                fill_polygon_hatch(
                                    &mut draw_ctx,
                                    verts,
                                    fill_rgba,
                                    45.0,
                                    hatch_spacing,
                                    *line_weight,
                                );
                                if *fill == crate::model::FillStyle::Crosses {
                                    fill_polygon_hatch(
                                        &mut draw_ctx,
                                        verts,
                                        fill_rgba,
                                        135.0,
                                        hatch_spacing,
                                        *line_weight,
                                    );
                                }
                            }
                            crate::model::FillStyle::Clear => {}
                        }
                    }
                }
                Object::Road {
                    id: road_id,
                    centerline,
                    ..
                } => {
                    if let Some(rd) = all_road_data.iter().find(|r| r.id == Some(*road_id)) {
                        for seg in centerline.windows(2) {
                            draw_bulge_segment(
                                &mut draw_ctx,
                                seg[0].pos,
                                seg[1].pos,
                                seg[0].bulge,
                                DOC_LINE_WIDTH,
                                rgba,
                            );
                        }

                        let edge_color = YELLOW_HIGHLIGHT_COLOR;
                        for (edge_pts, z_off) in
                            [(&rd.left_pts, rd.left_z), (&rd.right_pts, rd.right_z)]
                        {
                            for i in rd.draw_start..rd.draw_end.saturating_sub(1) {
                                let pa = edge_pts[i];
                                let pb = edge_pts[i + 1];
                                let pa2 = pa.truncate();
                                let pb2 = pb.truncate();

                                let mut ts: Vec<f64> = vec![0.0, 1.0];
                                for other in &all_road_data {
                                    if other.id == rd.id {
                                        continue;
                                    }
                                    for other_edge in [&other.left_pts, &other.right_pts] {
                                        for pair in other_edge.windows(2) {
                                            if let Some(t) = road_edge_seg_t(
                                                pa2,
                                                pb2,
                                                pair[0].truncate(),
                                                pair[1].truncate(),
                                            ) {
                                                ts.push(t);
                                            }
                                        }
                                    }
                                }
                                for attachment in &rd.attachments {
                                    for segment in &attachment.segments {
                                        if let Some(t) = road_edge_seg_t(
                                            pa2,
                                            pb2,
                                            segment[0].truncate(),
                                            segment[1].truncate(),
                                        ) {
                                            ts.push(t);
                                        }
                                    }
                                    if let (Some(along_a), Some(along_b)) = (
                                        render_attachment_edge_along(pa2, attachment),
                                        render_attachment_edge_along(pb2, attachment),
                                    ) {
                                        add_render_along_split(&mut ts, along_a, along_b, 0.0);
                                        add_render_along_split(
                                        &mut ts,
                                        along_a,
                                        along_b,
                                        crate::model::geometry::ROAD_INTERSECTION_FLAT_CLEARANCE_M,
                                    );
                                    }
                                }
                                ts.sort_by(|a, b| {
                                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                });

                                for w in ts.windows(2) {
                                    let (t0, t1) = (w[0], w[1]);
                                    if t1 - t0 < 1e-8 {
                                        continue;
                                    }
                                    let mid2d = pa2 + (pb2 - pa2) * ((t0 + t1) * 0.5);
                                    let inside = all_road_data.iter().any(|other| {
                                        if other.id == rd.id {
                                            return false;
                                        }
                                        road_point_in_body(mid2d, &other.cl_raw, other.half_w)
                                    });
                                    let clipped_by_attachment = rd
                                        .attachments
                                        .iter()
                                        .filter(|attachment| attachment.clip_boundary)
                                        .filter_map(|attachment| {
                                            render_attachment_edge_along(mid2d, attachment)
                                        })
                                        .any(|along| along < -1e-8);
                                    if !inside && !clipped_by_attachment {
                                        let interp = |t: f64| {
                                            let base = DVec3::new(
                                                pa.x + (pb.x - pa.x) * t,
                                                pa.y + (pb.y - pa.y) * t,
                                                pa.z + (pb.z - pa.z) * t,
                                            );
                                            render_tapered_edge_point(base, z_off, &rd.attachments)
                                        };
                                        draw_line(
                                            &mut draw_ctx,
                                            interp(t0),
                                            interp(t1),
                                            DOC_LINE_WIDTH,
                                            edge_color,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                Object::Text {
                    pos,
                    content,
                    height,
                    rotation,
                    ..
                } => {
                    let is_editing = editor.editing_labels_id == Some(object.id());
                    let (render_content, render_height, render_rotation, render_rgba) =
                        if is_editing {
                            (
                                &editor.pending_text,
                                editor.pending_text_height,
                                editor.pending_text_rotation_degrees.to_radians(),
                                editor.pending_text_color,
                            )
                        } else {
                            (content, *height, *rotation, rgba)
                        };
                    let corners =
                        text_bounds_corners(*pos, render_content, render_height, render_rotation);
                    text_pick_records.push(TextPickRecord {
                        entity: handle,
                        corners,
                    });
                    if is_editing || editor.selected_handles.contains(&handle) {
                        let indicator_color = if is_editing {
                            TEXT_EDIT_INDICATOR_COLOR
                        } else {
                            selection_color
                        };
                        for edge in corners
                            .iter()
                            .copied()
                            .zip(corners.iter().copied().cycle().skip(1))
                            .take(4)
                        {
                            draw_line(&mut draw_ctx, edge.0, edge.1, 2.0, indicator_color);
                        }
                    }
                    let mut textbox =
                        TextBox::new(vec![Text::new(render_content.clone(), render_rgba)], 1.0);
                    textbox.font_size = DOC_TEXT_FONT_SIZE;
                    let local = *pos - scene_origin;
                    let scale = (render_height / textbox.font_size as f64).max(f64::EPSILON) as f32;
                    let matrix = Mat4::from_translation(local.as_vec3())
                        * Mat4::from_rotation_z(render_rotation as f32)
                        * Mat4::from_scale(Vec3::new(scale, -scale, scale));
                    cached_textareas.push(textbox.text_areas(
                        text_system,
                        (0.0, 0.0),
                        (f32::MAX, f32::MAX),
                        1.0,
                        matrix,
                    ));
                    textarea_depths.push(local.z as f32);
                }
            }
            let stroke_end = draw_ctx.stroke_vertex_buf.len() as u32;
            let stroke_index_end = draw_ctx.stroke_index_buf.len() as u32;
            let fill_end = draw_ctx.fill_vertex_buf.len() as u32;
            let fill_index_end = draw_ctx.fill_index_buf.len() as u32;
            if (stroke_end > stroke_start || fill_end > fill_start)
                && !editor.frozen_handles.contains(&handle)
            {
                let segments = match object {
                    Object::Polyline { verts, closed, .. } => {
                        let mut segs: Vec<[DVec3; 2]> =
                            verts.windows(2).map(|w| [w[0].pos, w[1].pos]).collect();
                        if *closed && verts.len() >= 2 {
                            segs.push([verts.last().unwrap().pos, verts.first().unwrap().pos]);
                        }
                        segs
                    }
                    Object::Road { centerline, .. } => centerline
                        .windows(2)
                        .map(|w| [w[0].pos, w[1].pos])
                        .collect(),
                    _ => Vec::new(),
                };
                pick_records.push(PickRecord {
                    entity: handle,
                    stroke_range: (stroke_start, stroke_end),
                    stroke_index_range: (stroke_index_start, stroke_index_end),
                    fill_range: (fill_start, fill_end),
                    fill_index_range: (fill_index_start, fill_index_end),
                    segments,
                });
            }
        }

        if editor.batter_berm_dialog_open {
            const BATTER_BERM_PREVIEW_COLOR: [f32; 4] = [1.0, 0.86, 0.0, 1.0];
            const BATTER_BERM_GUIDE_COLOR: [f32; 4] = [1.0, 0.9, 0.16, 0.8];

            for &(from, to) in &editor.batter_berm_guides_world {
                draw_line(&mut draw_ctx, from, to, 1.5, BATTER_BERM_GUIDE_COLOR);
            }

            for ring in &editor.batter_berm_rings_world {
                for pair in ring.windows(2) {
                    draw_line(
                        &mut draw_ctx,
                        pair[0],
                        pair[1],
                        2.0,
                        BATTER_BERM_PREVIEW_COLOR,
                    );
                }
                if editor.batter_berm_preview_closed
                    && ring.len() >= 2
                    && let (Some(&first), Some(&last)) = (ring.first(), ring.last())
                {
                    draw_line(&mut draw_ctx, last, first, 2.0, BATTER_BERM_PREVIEW_COLOR);
                }
            }
        }

        if editor.active_tool == ActiveTool::MakeRoad {
            let mut raw_centerline = editor.pending_stroke.clone();
            if let Some(cursor) = editor.cursor_world {
                raw_centerline.push(cursor);
            }
            let centerline = crate::model::geometry::validate_road_segment_angles(
                &raw_centerline,
                editor.road_max_angle_degrees,
            )
            .ok()
            .and_then(|_| {
                crate::model::geometry::road_centerline_with_intersection_flats(
                    &raw_centerline,
                    document,
                    editor.road_width,
                )
                .ok()
            });
            let color = if centerline.is_some() {
                PREVIEW_COLOR
            } else {
                INVALID_PREVIEW_COLOR
            };
            let centerline = centerline.unwrap_or(raw_centerline);
            for pair in centerline.windows(2) {
                draw_line(&mut draw_ctx, pair[0], pair[1], 1.5, color);
            }
            for edge in [
                &editor.road_preview_left_world,
                &editor.road_preview_right_world,
            ] {
                for pair in edge.windows(2) {
                    draw_line(&mut draw_ctx, pair[0], pair[1], 2.0, YELLOW_HIGHLIGHT_COLOR);
                }
            }
        }
    }

    if !editor.translucent_handles.is_empty() {
        for rec in pick_records.iter() {
            if !editor.translucent_handles.contains(&rec.entity) {
                continue;
            }
            let stroke = rec.stroke_range.0 as usize..rec.stroke_range.1 as usize;
            for vertex in &mut stroke_vertex_buf[stroke] {
                make_translucent(&mut vertex.color);
            }
            let fill = rec.fill_range.0 as usize..rec.fill_range.1 as usize;
            for vertex in &mut lyon_buffer.vertices[fill] {
                make_translucent(&mut vertex.color);
            }
        }
    }

    if !editor.tri_hover_handles.is_empty() {
        for rec in pick_records.iter() {
            if !editor.tri_hover_handles.contains(&rec.entity) {
                continue;
            }
            let stroke = rec.stroke_range.0 as usize..rec.stroke_range.1 as usize;
            for vertex in &mut stroke_vertex_buf[stroke] {
                vertex.color = YELLOW_HIGHLIGHT_COLOR;
            }
            let fill = rec.fill_range.0 as usize..rec.fill_range.1 as usize;
            for vertex in &mut lyon_buffer.vertices[fill] {
                vertex.color = YELLOW_HIGHLIGHT_COLOR;
            }
        }
    }

    if !editor.selected_handles.is_empty() {
        for rec in pick_records.iter() {
            if !editor.selected_handles.contains(&rec.entity) {
                continue;
            }
            let stroke = rec.stroke_range.0 as usize..rec.stroke_range.1 as usize;
            for vertex in &mut stroke_vertex_buf[stroke] {
                vertex.color = selection_color;
            }
            let fill = rec.fill_range.0 as usize..rec.fill_range.1 as usize;
            for vertex in &mut lyon_buffer.vertices[fill] {
                vertex.color = selection_color;
            }
        }
    }

    if let Some(highlight_id) = editor.tool_highlight_id {
        let handle = SceneEntityId::Object(highlight_id);
        if let Some(rec) = pick_records.iter().find(|r| r.entity == handle) {
            let stroke = rec.stroke_range.0 as usize..rec.stroke_range.1 as usize;
            for vertex in &mut stroke_vertex_buf[stroke] {
                vertex.color = selection_color;
            }
            let fill = rec.fill_range.0 as usize..rec.fill_range.1 as usize;
            for vertex in &mut lyon_buffer.vertices[fill] {
                vertex.color = selection_color;
            }
        }
    }
}
