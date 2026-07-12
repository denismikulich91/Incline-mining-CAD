//! Document/editor scene assembly entry points.

use glam::{DVec3, Mat4, Vec3};
use lyon::tessellation::VertexBuffers;

use std::collections::HashSet;

use crate::{
    model::{Document, Object, ObjectId, SceneEntityId},
    rendering::{
        StrokeVertex, Vertex,
        geometry::{DrawContext, draw_line, draw_screen_cross, tessellate_polyline_stroke},
        graphics::{
            DOC_LINE_WIDTH, DOC_TEXT_FONT_SIZE, INVALID_PREVIEW_COLOR, PREVIEW_COLOR,
            TEXT_EDIT_INDICATOR_COLOR, YELLOW_HIGHLIGHT_COLOR, make_translucent,
            text_bounds_corners,
        },
        pick::{PickRecord, TextPickRecord, world_bounds_from_local_positions},
        scene::{
            document::{
                draw_origin_marker, fill_polygon_hatch, fill_polygon_solid, polygon_hatch_spacing,
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
    /// Objects owned by the static stroke chunk cache; skipped here.
    pub(crate) static_ids: &'a HashSet<ObjectId>,
    pub(crate) text_system: &'a mut TextSystem,
    pub(crate) lyon_buffer: &'a mut VertexBuffers<Vertex, u32>,
    pub(crate) stroke_vertex_buf: &'a mut Vec<StrokeVertex>,
    pub(crate) stroke_index_buf: &'a mut Vec<u32>,
    pub(crate) cached_textareas: &'a mut Vec<CachedTextArea>,
    pub(crate) textarea_depths: &'a mut Vec<f32>,
    pub(crate) pick_records: &'a mut Vec<PickRecord>,
    pub(crate) text_pick_records: &'a mut Vec<TextPickRecord>,
    pub(crate) document_draw_batches: &'a mut Vec<DocumentDrawBatch>,
    pub(crate) scene_origin: DVec3,
    pub(crate) scale_factor: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DocumentPrimitive {
    Fill,
    Stroke,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DocumentRenderStage {
    Opaque,
    Translucent,
    Overlay,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DocumentDrawBatch {
    pub(crate) primitive: DocumentPrimitive,
    pub(crate) index_range: (u32, u32),
    pub(crate) center: DVec3,
    stage: DocumentRenderStage,
}

impl DocumentDrawBatch {
    pub(crate) fn stage(self, xray_enabled: bool) -> DocumentRenderStage {
        if xray_enabled {
            DocumentRenderStage::Overlay
        } else {
            self.stage
        }
    }
}

#[derive(Clone, Copy)]
struct DocumentObjectRanges {
    entity: Option<SceneEntityId>,
    stroke_index_range: (u32, u32),
    fill_index_range: (u32, u32),
    center: DVec3,
}

pub(crate) fn rebuild_document_scene(input: DocumentSceneBuildInput<'_>) {
    let DocumentSceneBuildInput {
        editor,
        document,
        static_ids,
        text_system,
        lyon_buffer,
        stroke_vertex_buf,
        stroke_index_buf,
        cached_textareas,
        textarea_depths,
        pick_records,
        text_pick_records,
        document_draw_batches,
        scene_origin,
        scale_factor,
    } = input;

    lyon_buffer.indices.clear();
    lyon_buffer.vertices.clear();
    stroke_index_buf.clear();
    stroke_vertex_buf.clear();
    cached_textareas.clear();
    textarea_depths.clear();
    pick_records.clear();
    text_pick_records.clear();
    document_draw_batches.clear();

    let mut object_ranges = Vec::new();

    {
        let mut draw_ctx = DrawContext {
            stroke_vertex_buf,
            stroke_index_buf,
            fill_vertex_buf: &mut lyon_buffer.vertices,
            fill_index_buf: &mut lyon_buffer.indices,
            scene_origin,
            scale_factor,
        };

        let origin_stroke_start = draw_ctx.stroke_index_buf.len() as u32;
        draw_origin_marker(&mut draw_ctx);
        object_ranges.push(DocumentObjectRanges {
            entity: None,
            stroke_index_range: (origin_stroke_start, draw_ctx.stroke_index_buf.len() as u32),
            fill_index_range: (0, 0),
            center: DVec3::ZERO,
        });

        // Static pass: roads resolve without the ghost. Roads the ghost
        // reshapes are suppressed here and drawn by `rebuild_dynamic_scene`
        // from the ghost-inclusive resolve instead.
        let road_network = resolve(document, None);

        // The document is the sole source of geometry. Draw each object in world
        // space, honoring layer visibility / hide / freeze, and record a pick range
        // unless frozen so objects select and highlight. Selected objects are
        // emitted last so overlapping selected strokes remain visible.
        let mut objects: Vec<&Object> = document.objects().iter().collect();
        objects.sort_by_key(|object| {
            let handle = SceneEntityId::Object(object.id());
            editor.selected_handles.contains(&handle)
                || editor.tool_highlight_id == Some(object.id())
        });
        for object in objects {
            if static_ids.contains(&object.id()) {
                continue;
            }
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
            let mut fill_opaque = false;
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
                    fill_opaque = *fill == crate::model::FillStyle::Solid
                        && fill_rgba[3] >= 1.0 - f32::EPSILON;
                    tessellate_polyline_stroke(
                        &mut draw_ctx,
                        verts,
                        *closed,
                        *line_weight,
                        line_rgba,
                    );
                    if *closed && verts.len() >= 2 {
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
                                let hatch_spacing = polygon_hatch_spacing(verts, 15.0);
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
                Object::Road { id: road_id, .. } => {
                    if editor.road_preview_affected_roads.contains(road_id) {
                        continue;
                    }
                    // All geometry is pre-resolved by the road network: the
                    // resolved center carries junction flattening and grade
                    // flats, and the side lines already terminate exactly at
                    // shared junction corners — nothing to clip here.
                    for edge in road_network.edges_for(RoadKey::Object(*road_id)) {
                        for pair in edge.center.windows(2) {
                            draw_line(&mut draw_ctx, pair[0], pair[1], DOC_LINE_WIDTH, rgba);
                        }
                        for side in [&edge.left, &edge.right] {
                            for pair in side.windows(2) {
                                draw_line(
                                    &mut draw_ctx,
                                    pair[0],
                                    pair[1],
                                    DOC_LINE_WIDTH,
                                    YELLOW_HIGHLIGHT_COLOR,
                                );
                            }
                        }
                        if edge.start_cap
                            && let (Some(&l), Some(&r)) = (edge.left.first(), edge.right.first())
                        {
                            draw_line(&mut draw_ctx, l, r, DOC_LINE_WIDTH, YELLOW_HIGHLIGHT_COLOR);
                        }
                        if edge.end_cap
                            && let (Some(&l), Some(&r)) = (edge.left.last(), edge.right.last())
                        {
                            draw_line(&mut draw_ctx, l, r, DOC_LINE_WIDTH, YELLOW_HIGHLIGHT_COLOR);
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
                            crate::ui::SELECTION_COLOR_F32
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
            if stroke_index_end > stroke_index_start || fill_index_end > fill_index_start {
                object_ranges.push(DocumentObjectRanges {
                    entity: Some(handle),
                    stroke_index_range: (stroke_index_start, stroke_index_end),
                    fill_index_range: (fill_index_start, fill_index_end),
                    center: object_center(object),
                });
            }
            if (stroke_end > stroke_start || fill_end > fill_start)
                && !editor.frozen_handles.contains(&handle)
                && let Some(world_bounds) = world_bounds_from_local_positions(
                    draw_ctx.stroke_vertex_buf[stroke_start as usize..stroke_end as usize]
                        .iter()
                        .map(|vertex| vertex.pos)
                        .chain(
                            draw_ctx.fill_vertex_buf[fill_start as usize..fill_end as usize]
                                .iter()
                                .map(|vertex| vertex.pos),
                        ),
                    scene_origin,
                )
            {
                pick_records.push(PickRecord {
                    entity: handle,
                    world_bounds,
                    stroke_range: (stroke_start, stroke_end),
                    stroke_index_range: (stroke_index_start, stroke_index_end),
                    fill_range: (fill_start, fill_end),
                    fill_index_range: (fill_index_start, fill_index_end),
                    fill_opaque,
                });
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
                vertex.color = crate::ui::SELECTION_COLOR_F32;
            }
            let fill = rec.fill_range.0 as usize..rec.fill_range.1 as usize;
            for vertex in &mut lyon_buffer.vertices[fill] {
                vertex.color = crate::ui::SELECTION_COLOR_F32;
            }
        }
    }

    if let Some(highlight_id) = editor.tool_highlight_id {
        let handle = SceneEntityId::Object(highlight_id);
        if let Some(rec) = pick_records.iter().find(|r| r.entity == handle) {
            let stroke = rec.stroke_range.0 as usize..rec.stroke_range.1 as usize;
            for vertex in &mut stroke_vertex_buf[stroke] {
                vertex.color = crate::ui::SELECTION_COLOR_F32;
            }
            let fill = rec.fill_range.0 as usize..rec.fill_range.1 as usize;
            for vertex in &mut lyon_buffer.vertices[fill] {
                vertex.color = crate::ui::SELECTION_COLOR_F32;
            }
        }
    }

    rebuild_document_draw_batches(
        document_draw_batches,
        &object_ranges,
        editor,
        stroke_vertex_buf,
        stroke_index_buf,
        &lyon_buffer.vertices,
        &lyon_buffer.indices,
    );
}

fn object_center(object: &Object) -> DVec3 {
    match object {
        Object::Point { pos, .. } | Object::Text { pos, .. } => *pos,
        Object::Polyline { verts, .. } => average_positions(verts.iter().map(|vertex| vertex.pos)),
        Object::Road { centerline, .. } => {
            average_positions(centerline.iter().map(|vertex| vertex.pos))
        }
    }
}

fn average_positions(points: impl Iterator<Item = DVec3>) -> DVec3 {
    let (sum, count) = points.fold((DVec3::ZERO, 0usize), |(sum, count), point| {
        (sum + point, count + 1)
    });
    if count == 0 {
        DVec3::ZERO
    } else {
        sum / count as f64
    }
}

fn rebuild_document_draw_batches(
    batches: &mut Vec<DocumentDrawBatch>,
    objects: &[DocumentObjectRanges],
    editor: &EditorState,
    stroke_vertices: &[StrokeVertex],
    stroke_indices: &[u32],
    fill_vertices: &[Vertex],
    fill_indices: &[u32],
) {
    batches.clear();
    for object in objects {
        let overlay = object.entity.is_some_and(|entity| {
            editor.selected_handles.contains(&entity)
                || editor.tri_hover_handles.contains(&entity)
                || editor
                    .tool_highlight_id
                    .is_some_and(|id| entity == SceneEntityId::Object(id))
                || editor
                    .editing_labels_id
                    .is_some_and(|id| entity == SceneEntityId::Object(id))
        });
        append_alpha_runs(
            batches,
            DocumentPrimitive::Stroke,
            object.stroke_index_range,
            object.center,
            overlay,
            stroke_indices,
            |index| {
                stroke_vertices
                    .get(index as usize)
                    .map(|vertex| vertex.color[3])
            },
        );
        append_alpha_runs(
            batches,
            DocumentPrimitive::Fill,
            object.fill_index_range,
            object.center,
            overlay,
            fill_indices,
            |index| {
                fill_vertices
                    .get(index as usize)
                    .map(|vertex| vertex.color[3])
            },
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn append_alpha_runs(
    batches: &mut Vec<DocumentDrawBatch>,
    primitive: DocumentPrimitive,
    index_range: (u32, u32),
    center: DVec3,
    overlay: bool,
    indices: &[u32],
    alpha_for_vertex: impl Fn(u32) -> Option<f32>,
) {
    let start = (index_range.0 as usize).min(indices.len());
    let end = (index_range.1 as usize).min(indices.len());
    let end = start + (end.saturating_sub(start) / 3) * 3;
    if start >= end {
        return;
    }

    let stage_for_triangle = |triangle_start: usize| {
        if overlay {
            DocumentRenderStage::Overlay
        } else if indices
            .get(triangle_start)
            .and_then(|&index| alpha_for_vertex(index))
            .unwrap_or(1.0)
            < 0.999
        {
            DocumentRenderStage::Translucent
        } else {
            DocumentRenderStage::Opaque
        }
    };

    let mut run_start = start;
    let mut run_stage = stage_for_triangle(start);
    for triangle_start in (start + 3..end).step_by(3) {
        let stage = stage_for_triangle(triangle_start);
        if stage == run_stage {
            continue;
        }
        push_document_batch(
            batches,
            DocumentDrawBatch {
                primitive,
                index_range: (run_start as u32, triangle_start as u32),
                center,
                stage: run_stage,
            },
        );
        run_start = triangle_start;
        run_stage = stage;
    }
    push_document_batch(
        batches,
        DocumentDrawBatch {
            primitive,
            index_range: (run_start as u32, end as u32),
            center,
            stage: run_stage,
        },
    );
}

fn push_document_batch(batches: &mut Vec<DocumentDrawBatch>, batch: DocumentDrawBatch) {
    // Fill and stroke indices live in separate streams, so batches of one
    // primitive can remain contiguous even when the other primitive was
    // appended between them. Coalescing opaque/overlay runs keeps large point
    // and road documents at a handful of draw calls; translucent runs retain
    // per-object centers for back-to-front sorting.
    if let Some(previous) = batches
        .iter_mut()
        .rev()
        .find(|previous| previous.primitive == batch.primitive)
        && previous.stage == batch.stage
        && previous.index_range.1 == batch.index_range.0
        && (batch.stage != DocumentRenderStage::Translucent || previous.center == batch.center)
    {
        previous.index_range.1 = batch.index_range.1;
        return;
    }
    batches.push(batch);
}

pub(crate) struct DynamicSceneBuildInput<'a> {
    pub(crate) editor: &'a EditorState,
    pub(crate) document: &'a Document,
    pub(crate) dynamic_vertex_buf: &'a mut Vec<StrokeVertex>,
    pub(crate) dynamic_index_buf: &'a mut Vec<u32>,
    pub(crate) scene_origin: DVec3,
    pub(crate) scale_factor: f32,
}

/// Per-frame geometry for the live drawing tools: the road ghost preview, the
/// batter/berm preview, and the committed roads the ghost reshapes (suppressed
/// from the static pass). Stroke-only and tiny, so rebuilding it every frame
/// while a tool is active is cheap — unlike the full document scene it
/// replaces in that role.
pub(crate) fn rebuild_dynamic_scene(input: DynamicSceneBuildInput<'_>) {
    let DynamicSceneBuildInput {
        editor,
        document,
        dynamic_vertex_buf,
        dynamic_index_buf,
        scene_origin,
        scale_factor,
    } = input;

    dynamic_vertex_buf.clear();
    dynamic_index_buf.clear();

    let mut unused_fill_vertices: Vec<Vertex> = Vec::new();
    let mut unused_fill_indices: Vec<u32> = Vec::new();
    let mut draw_ctx = DrawContext {
        stroke_vertex_buf: dynamic_vertex_buf,
        stroke_index_buf: dynamic_index_buf,
        fill_vertex_buf: &mut unused_fill_vertices,
        fill_index_buf: &mut unused_fill_indices,
        scene_origin,
        scale_factor,
    };

    // Ghost-reshaped committed roads, resolved with the ghost included (kept
    // fresh by `update_road_preview`), drawn exactly like the static pass
    // draws roads and honouring the same visibility filters.
    for edge in &editor.road_preview_affected_edges {
        let RoadKey::Object(road_id) = edge.road else {
            continue;
        };
        let Some(object) = document.get_object(road_id) else {
            continue;
        };
        let handle = SceneEntityId::Object(road_id);
        let layer_visible = document
            .layer(object.layer())
            .map(|layer| layer.visible)
            .unwrap_or(true);
        if !layer_visible || editor.hidden_handles.contains(&handle) {
            continue;
        }
        let rgba = document.object_rgba(object);
        for pair in edge.center.windows(2) {
            draw_line(&mut draw_ctx, pair[0], pair[1], DOC_LINE_WIDTH, rgba);
        }
        for side in [&edge.left, &edge.right] {
            for pair in side.windows(2) {
                draw_line(
                    &mut draw_ctx,
                    pair[0],
                    pair[1],
                    DOC_LINE_WIDTH,
                    YELLOW_HIGHLIGHT_COLOR,
                );
            }
        }
        if edge.start_cap
            && let (Some(&l), Some(&r)) = (edge.left.first(), edge.right.first())
        {
            draw_line(&mut draw_ctx, l, r, DOC_LINE_WIDTH, YELLOW_HIGHLIGHT_COLOR);
        }
        if edge.end_cap
            && let (Some(&l), Some(&r)) = (edge.left.last(), edge.right.last())
        {
            draw_line(&mut draw_ctx, l, r, DOC_LINE_WIDTH, YELLOW_HIGHLIGHT_COLOR);
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
        // The resolved ghost centerline (flat pockets included) when the
        // stroke passes the placement rules; the raw stroke in the invalid
        // colour otherwise. Both are maintained by `update_road_preview`.
        let valid =
            editor.road_preview_violation.is_none() && !editor.road_preview_center_world.is_empty();
        let (centerline, color) = if valid {
            (editor.road_preview_center_world.clone(), PREVIEW_COLOR)
        } else {
            let mut raw_centerline = editor.pending_stroke.clone();
            if let Some(cursor) = editor.cursor_world {
                raw_centerline.push(cursor);
            }
            (raw_centerline, INVALID_PREVIEW_COLOR)
        };
        for pair in centerline.windows(2) {
            if pair[0].is_finite() && pair[1].is_finite() {
                draw_line(&mut draw_ctx, pair[0], pair[1], 1.5, color);
            }
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
