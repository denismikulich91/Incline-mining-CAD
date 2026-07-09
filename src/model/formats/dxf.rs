//! Convert a `dxf::Drawing` into the editable [`Document`] model.
//!
//! The DXF is flattened and simplified for open-pit design: block/INSERT
//! instances are baked into world space, ByLayer/ByBlock colour is resolved,
//! dimensions are dropped, and curves become bulge-aware polylines (arcs/circles
//! preserved; ellipses/splines tessellated). The renderer then reads only the
//! `Document`.

use std::collections::HashMap;

use dxf::{
    Block, Drawing,
    entities::{Entity, EntityType},
};
use glam::{DMat4, DQuat, DVec3, DVec4};

use crate::model::{Document, LayerId, Object, ObjectColor, PolyVertex};
use crate::{
    model::geometry::{Transform, normalize_or_z, point_to_dvec3, vector_to_dvec3},
    rendering::color::aci_to_linear_rgba,
    userspace_warn,
};

const MAX_BLOCK_DEPTH: usize = 32;
/// Target world-space segment length when tessellating non-bulge curves.
const TESSELLATION_SEGMENT_LEN: f64 = 0.15;
const MIN_TESSELLATION_SEGMENTS: usize = 32;
const MAX_TESSELLATION_SEGMENTS: usize = 4096;

/// Build a `Document` from a loaded DXF drawing.
pub(crate) fn from_dxf(drawing: &Drawing) -> Document {
    let mut doc = Document::new();

    let mut layer_ids: HashMap<String, LayerId> = HashMap::new();
    for layer in drawing.layers() {
        let aci = layer.color.index();
        let rgba = aci
            .and_then(|index| aci_to_linear_rgba(index as i32))
            .unwrap_or([1.0, 1.0, 1.0, 1.0]);
        let id = doc.add_layer(layer.name.clone(), aci, rgba, layer.is_layer_on, 0.0);
        layer_ids.insert(layer.name.clone(), id);
    }
    // Guarantee at least one layer to place geometry on.
    doc.ensure_default_layer();

    let blocks: HashMap<&str, &Block> = drawing
        .blocks()
        .map(|block| (block.name.as_str(), block))
        .collect();

    let mut ctx = ImportCtx {
        doc: &mut doc,
        layer_ids,
        blocks: &blocks,
        stack: Vec::new(),
        unknown_layer_ids: HashMap::new(),
        next_unknown_layer_index: 0,
    };
    let transform = Transform::identity();
    let scope = BlockScope::world();
    import_entities(drawing.entities(), &mut ctx, &transform, scope);

    // The DXF spec mandates a default "0" layer and `Drawing::new()` always
    // emits one, so an exported DXF carrying a single real layer still
    // includes an empty "0". Drop it on import when it has no geometry,
    // unless it is the only layer (the document must keep at least one).
    if doc.layers().len() > 1
        && let Some(zero_id) = doc.layer_id_by_name("0")
        && !doc.objects().iter().any(|object| object.layer() == zero_id)
    {
        doc.delete_layer(zero_id);
    }

    doc
}

struct ImportCtx<'a> {
    doc: &'a mut Document,
    layer_ids: HashMap<String, LayerId>,
    blocks: &'a HashMap<&'a str, &'a Block>,
    stack: Vec<String>,
    unknown_layer_ids: HashMap<String, LayerId>,
    next_unknown_layer_index: usize,
}

/// Inherited colour/layer while rendering the contents of a block.
#[derive(Clone, Copy, Default)]
struct BlockScope<'a> {
    color: Option<u8>,
    layer: Option<&'a str>,
}

impl<'a> BlockScope<'a> {
    fn world() -> Self {
        Self::default()
    }
}

impl ImportCtx<'_> {
    fn layer_id(&mut self, name: &str) -> LayerId {
        if let Some(id) = self
            .layer_ids
            .get(name)
            .or_else(|| self.unknown_layer_ids.get(name))
        {
            return *id;
        }

        let fallback_name = loop {
            let candidate = format!("unknown_{}", self.next_unknown_layer_index);
            self.next_unknown_layer_index += 1;
            if self.doc.layer_id_by_name(&candidate).is_none() {
                break candidate;
            }
        };
        let id = self
            .doc
            .add_layer(fallback_name.clone(), None, [1.0, 1.0, 1.0, 1.0], true, 0.0);
        self.unknown_layer_ids.insert(name.to_owned(), id);
        userspace_warn!(
            "DXF entity referenced undefined layer '{name}', imported as '{fallback_name}'"
        );
        id
    }
}

fn resolve_color(entity: &Entity, block_color: Option<u8>) -> ObjectColor {
    let color = &entity.common.color;
    let aci = if color.is_by_layer() {
        return ObjectColor::ByLayer;
    } else if color.is_by_block() {
        block_color
    } else {
        color.index()
    };
    match aci.and_then(|index| aci_to_linear_rgba(index as i32)) {
        Some(rgba) => ObjectColor::Fixed(rgba),
        None => ObjectColor::ByLayer,
    }
}

fn import_entities<'e>(
    entities: impl IntoIterator<Item = &'e Entity>,
    ctx: &mut ImportCtx<'_>,
    transform: &Transform,
    scope: BlockScope<'_>,
) {
    for entity in entities {
        if !entity.common.is_visible {
            continue;
        }
        let effective_layer = if entity.common.layer == "0" {
            scope.layer.unwrap_or("0")
        } else {
            entity.common.layer.as_str()
        };
        let layer = ctx.layer_id(effective_layer);
        let color = resolve_color(entity, scope.color);

        match &entity.specific {
            EntityType::Line(line) => {
                let verts = vec![
                    PolyVertex::straight(transform.apply(point_to_dvec3(&line.p1))),
                    PolyVertex::straight(transform.apply(point_to_dvec3(&line.p2))),
                ];
                push_polyline(ctx, layer, verts, false, color);
            }
            EntityType::LwPolyline(poly) => {
                let elevation = entity.common.elevation;
                let verts: Vec<PolyVertex> = poly
                    .vertices
                    .iter()
                    .map(|v| PolyVertex {
                        pos: DVec3::new(v.x, v.y, elevation),
                        bulge: v.bulge,
                    })
                    .collect();
                if verts.len() >= 2 {
                    push_bulge_polyline(
                        ctx,
                        layer,
                        verts,
                        poly.is_closed(),
                        normalize_or_z(vector_to_dvec3(&poly.extrusion_direction)),
                        transform,
                        color,
                    );
                }
            }
            EntityType::Arc(arc) => {
                let verts = arc_to_verts(
                    point_to_dvec3(&arc.center),
                    arc.radius,
                    normalize_or_z(vector_to_dvec3(&arc.normal)),
                    arc.start_angle.to_radians(),
                    arc.end_angle.to_radians(),
                    transform,
                );
                push_polyline(ctx, layer, verts, false, color);
            }
            EntityType::Circle(circle) => {
                let (verts, closed) = circle_to_verts(
                    point_to_dvec3(&circle.center),
                    circle.radius,
                    normalize_or_z(vector_to_dvec3(&circle.normal)),
                    transform,
                );
                push_polyline(ctx, layer, verts, closed, color);
            }
            EntityType::Ellipse(ell) => {
                let closed = ellipse_is_full_loop(ell.start_parameter, ell.end_parameter);
                let mut verts = ellipse_to_verts(ell, transform);
                // For a full (closed) ellipse, ellipse_to_verts generates segments+1 points
                // where the first and last coincide — drop the duplicate closing vertex.
                if closed && verts.len() > 1 {
                    verts.pop();
                }
                if verts.len() >= 2 {
                    push_polyline(ctx, layer, verts, closed, color);
                }
            }
            EntityType::Spline(spline) => {
                // v1: approximate with fit points (on-curve) or control points.
                let source = if spline.fit_points.len() >= 2 {
                    &spline.fit_points
                } else {
                    &spline.control_points
                };
                if source.len() >= 2 {
                    let verts = source
                        .iter()
                        .map(|p| PolyVertex::straight(transform.apply(point_to_dvec3(p))))
                        .collect();
                    push_polyline(ctx, layer, verts, spline.is_closed(), color);
                }
            }
            EntityType::Solid(solid) => {
                // Outline as a closed quad (DXF solid corner order is 1,2,4,3).
                let verts = vec![
                    PolyVertex::straight(transform.apply(point_to_dvec3(&solid.first_corner))),
                    PolyVertex::straight(transform.apply(point_to_dvec3(&solid.second_corner))),
                    PolyVertex::straight(transform.apply(point_to_dvec3(&solid.fourth_corner))),
                    PolyVertex::straight(transform.apply(point_to_dvec3(&solid.third_corner))),
                ];
                push_polyline(ctx, layer, verts, true, color);
            }
            EntityType::Leader(leader) => {
                let verts: Vec<PolyVertex> = leader
                    .vertices
                    .iter()
                    .map(|p| PolyVertex::straight(transform.apply(point_to_dvec3(p))))
                    .collect();
                if verts.len() >= 2 {
                    push_polyline(ctx, layer, verts, false, color);
                }
            }
            EntityType::ModelPoint(point) => {
                let pos = transform.apply(point_to_dvec3(&point.location));
                ctx.doc.add_object(|id| Object::Point {
                    id,
                    layer,
                    pos,
                    color,
                });
            }
            EntityType::Text(text) => {
                let pos = transform.apply(point_to_dvec3(&text.location));
                push_text(
                    ctx,
                    layer,
                    pos,
                    plain_text(&text.value),
                    text.text_height,
                    text.rotation.to_radians(),
                    color,
                );
            }
            EntityType::MText(mtext) => {
                let pos = transform.apply(point_to_dvec3(&mtext.insertion_point));
                let mut raw = String::new();
                for chunk in &mtext.extended_text {
                    raw.push_str(chunk);
                }
                raw.push_str(&mtext.text);
                push_text(
                    ctx,
                    layer,
                    pos,
                    plain_mtext(&raw),
                    mtext.initial_text_height,
                    mtext.rotation_angle.to_radians(),
                    color,
                );
            }
            EntityType::Polyline(poly) => {
                // Skip polygon meshes and polyface meshes; only import 2D/3D polylines.
                if poly.is_polyface_mesh() || poly.is_3d_polygon_mesh() {
                    continue;
                }
                let is_3d = poly.is_3d_polyline();
                let elevation = poly.location.z;
                let verts: Vec<PolyVertex> = poly
                    .vertices()
                    .filter(|v| {
                        // Skip spline control points and curve-fit extras;
                        // keep plain vertices and 3D polyline vertices.
                        !v.is_spline_frame_control_point()
                            && !v.is_extra_created_by_curve_fit()
                            && !v.is_spline_vertex_created_by_spline_fitting()
                    })
                    .map(|v| {
                        let z = if is_3d { v.location.z } else { elevation };
                        PolyVertex {
                            pos: DVec3::new(v.location.x, v.location.y, z),
                            bulge: v.bulge,
                        }
                    })
                    .collect();
                if verts.len() >= 2 {
                    push_bulge_polyline(
                        ctx,
                        layer,
                        verts,
                        poly.is_closed(),
                        normalize_or_z(vector_to_dvec3(&poly.normal)),
                        transform,
                        color,
                    );
                }
            }
            EntityType::Insert(insert) => {
                import_insert(entity, insert, ctx, transform, scope);
            }
            // Dimensions and anything else are dropped.
            _ => {}
        }
    }
}

fn import_insert(
    entity: &Entity,
    insert: &dxf::entities::Insert,
    ctx: &mut ImportCtx<'_>,
    transform: &Transform,
    scope: BlockScope<'_>,
) {
    let Some(block) = ctx.blocks.get(insert.name.as_str()).copied() else {
        userspace_warn!("DXF INSERT references unknown block '{}'", insert.name);
        return;
    };
    if ctx.stack.len() >= MAX_BLOCK_DEPTH {
        userspace_warn!(
            "DXF block nesting exceeds maximum depth ({}), skipping '{}'",
            MAX_BLOCK_DEPTH,
            insert.name
        );
        return;
    }
    if ctx.stack.iter().any(|n| n == &block.name) {
        userspace_warn!("DXF circular block reference detected: '{}'", insert.name);
        return;
    }

    let child_color = if entity.common.color.is_by_layer() {
        scope.color
    } else {
        entity.common.color.index().or(scope.color)
    };
    // Resolve the child layer name to an owned String so it outlives the borrow.
    let child_layer = if entity.common.layer == "0" {
        scope.layer.unwrap_or("0").to_string()
    } else {
        entity.common.layer.clone()
    };

    let base_point = point_to_dvec3(&block.base_point);
    let location = point_to_dvec3(&insert.location);
    let rotation = insert.rotation.to_radians();
    let scale = DVec3::new(
        insert.x_scale_factor,
        insert.y_scale_factor,
        insert.z_scale_factor,
    );
    let ocs = ocs_to_wcs_affine(&insert.extrusion_direction);

    ctx.stack.push(block.name.clone());
    let mut local = transform.clone();
    local.push_with_affine(base_point, location, rotation, scale, ocs);
    let child_scope = BlockScope {
        color: child_color,
        layer: Some(child_layer.as_str()),
    };
    for row in 0..insert.row_count.max(1) {
        for col in 0..insert.column_count.max(1) {
            let offset = DVec3::new(
                col as f64 * insert.column_spacing,
                row as f64 * insert.row_spacing,
                0.0,
            );
            local.push(offset, DVec3::ZERO, 0.0, DVec3::ONE);
            import_entities(block.entities.iter(), ctx, &local, child_scope);
            local.pop();
        }
    }
    ctx.stack.pop();
}

fn push_polyline(
    ctx: &mut ImportCtx<'_>,
    layer: LayerId,
    verts: Vec<PolyVertex>,
    closed: bool,
    color: ObjectColor,
) {
    if verts.len() < 2 {
        return;
    }
    ctx.doc.add_object(|id| Object::Polyline {
        id,
        layer,
        verts,
        closed,
        color,
        fill: crate::model::FillStyle::Clear,
        line_weight: 1.0,
    });
}

fn push_bulge_polyline(
    ctx: &mut ImportCtx<'_>,
    layer: LayerId,
    verts: Vec<PolyVertex>,
    closed: bool,
    normal: DVec3,
    transform: &Transform,
    color: ObjectColor,
) {
    let verts = trim_duplicate_open_endpoint(verts, closed);
    if verts.len() < 2 {
        return;
    }

    let has_bulges = verts.iter().any(|vertex| vertex.bulge.abs() > f64::EPSILON);
    if !has_bulges {
        push_polyline(
            ctx,
            layer,
            verts
                .into_iter()
                .map(|vertex| PolyVertex::straight(transform.apply(vertex.pos)))
                .collect(),
            closed,
            color,
        );
        return;
    }

    if let Some(sign) = xy_bulge_transform_sign(transform, normal) {
        push_polyline(
            ctx,
            layer,
            verts
                .into_iter()
                .map(|vertex| PolyVertex {
                    pos: transform.apply(vertex.pos),
                    bulge: vertex.bulge * sign,
                })
                .collect(),
            closed,
            color,
        );
        return;
    }

    let mut baked = Vec::new();
    for segment in verts.windows(2) {
        append_baked_bulge_segment(
            &mut baked,
            segment[0].pos,
            segment[1].pos,
            segment[0].bulge,
            normal,
            transform,
        );
    }
    if closed && let (Some(first), Some(last)) = (verts.first(), verts.last()) {
        append_baked_bulge_segment(
            &mut baked, last.pos, first.pos, last.bulge, normal, transform,
        );
    }
    let baked_closed = closed;
    push_polyline(ctx, layer, baked, baked_closed, color);
}

#[allow(clippy::too_many_arguments)]
fn push_text(
    ctx: &mut ImportCtx<'_>,
    layer: LayerId,
    pos: DVec3,
    content: String,
    height: f64,
    rotation: f64,
    color: ObjectColor,
) {
    if content.trim().is_empty() {
        return;
    }
    ctx.doc.add_object(|id| Object::Text {
        id,
        layer,
        pos,
        content,
        height,
        rotation,
        color,
    });
}

/// Normalise an arc sweep into `(0, TAU]`.
fn normalize_sweep(start: f64, end: f64) -> f64 {
    let mut sweep = end - start;
    while sweep <= 0.0 {
        sweep += std::f64::consts::TAU;
    }
    sweep
}

/// An arc in the XY plane becomes one bulged segment; otherwise it is
/// tessellated to straight segments.
fn arc_to_verts(
    center: DVec3,
    radius: f64,
    normal: DVec3,
    start: f64,
    end: f64,
    transform: &Transform,
) -> Vec<PolyVertex> {
    let sweep = normalize_sweep(start, end);
    if let Some(sign) = xy_bulge_transform_sign(transform, normal) {
        let start_pos = center + radius * DVec3::new(start.cos(), start.sin(), 0.0);
        let end_pos = center + radius * DVec3::new(end.cos(), end.sin(), 0.0);
        vec![
            PolyVertex {
                pos: transform.apply(start_pos),
                bulge: (sweep / 4.0).tan() * sign,
            },
            PolyVertex::straight(transform.apply(end_pos)),
        ]
    } else {
        let segments = arc_segment_count(radius, sweep);
        let start_vec =
            DQuat::from_axis_angle(normal, start).mul_vec3(orthonormal_basis(normal).0 * radius);
        (0..=segments)
            .map(|i| {
                let angle = start + sweep * (i as f64 / segments as f64);
                let pos =
                    center + DQuat::from_axis_angle(normal, angle - start).mul_vec3(start_vec);
                PolyVertex::straight(transform.apply(pos))
            })
            .collect()
    }
}

fn circle_to_verts(
    center: DVec3,
    radius: f64,
    normal: DVec3,
    transform: &Transform,
) -> (Vec<PolyVertex>, bool) {
    if let Some(sign) = xy_bulge_transform_sign(transform, normal) {
        let a = center + DVec3::new(radius, 0.0, 0.0);
        let b = center - DVec3::new(radius, 0.0, 0.0);
        (
            vec![
                PolyVertex {
                    pos: transform.apply(a),
                    bulge: sign,
                },
                PolyVertex {
                    pos: transform.apply(b),
                    bulge: sign,
                },
            ],
            true,
        )
    } else {
        let (u, v) = orthonormal_basis(normal);
        let segments = arc_segment_count(radius, std::f64::consts::TAU);
        let verts = (0..segments)
            .map(|i| {
                let angle = std::f64::consts::TAU * (i as f64 / segments as f64);
                let pos = center + radius * (u * angle.cos() + v * angle.sin());
                PolyVertex::straight(transform.apply(pos))
            })
            .collect();
        (verts, true)
    }
}

fn ellipse_to_verts(ell: &dxf::entities::Ellipse, transform: &Transform) -> Vec<PolyVertex> {
    let center = point_to_dvec3(&ell.center);
    let major = vector_to_dvec3(&ell.major_axis);
    let a = major.length();
    if a <= f64::EPSILON {
        return Vec::new();
    }
    let major_dir = major / a;
    let normal = normalize_or_z(vector_to_dvec3(&ell.normal));
    let minor_dir = normal.cross(major_dir).normalize_or_zero();
    let b = a * ell.minor_axis_ratio;
    let start = ell.start_parameter;
    let end = ell.end_parameter;
    let sweep = if ellipse_is_full_loop(start, end) {
        std::f64::consts::TAU
    } else {
        end - start
    };
    let segments = arc_segment_count(a.max(b), sweep.abs());
    (0..=segments)
        .map(|i| {
            let t = start + sweep * (i as f64 / segments as f64);
            let pos = center + major_dir * a * t.cos() + minor_dir * b * t.sin();
            PolyVertex::straight(transform.apply(pos))
        })
        .collect()
}

fn ellipse_is_full_loop(start: f64, end: f64) -> bool {
    let span = (end - start).abs();
    !(f64::EPSILON..std::f64::consts::TAU - 1.0e-6).contains(&span)
}

fn arc_segment_count(radius: f64, sweep: f64) -> usize {
    let arc_len = radius.abs() * sweep.abs();
    ((arc_len / TESSELLATION_SEGMENT_LEN).ceil() as usize)
        .clamp(MIN_TESSELLATION_SEGMENTS, MAX_TESSELLATION_SEGMENTS)
}

fn trim_duplicate_open_endpoint(mut verts: Vec<PolyVertex>, closed: bool) -> Vec<PolyVertex> {
    if closed || verts.len() <= 2 {
        return verts;
    }
    let Some(first) = verts.first() else {
        return verts;
    };
    let Some(last) = verts.last() else {
        return verts;
    };
    if crate::model::kernel::points_coincident_3d(first.pos, last.pos) {
        let _ = verts.pop();
    }
    verts
}

fn xy_bulge_transform_sign(transform: &Transform, normal: DVec3) -> Option<f64> {
    if (normal - DVec3::Z).length_squared() > 1.0e-6 {
        return None;
    }

    let matrix = transform.matrix();
    let origin = matrix.transform_point3(DVec3::ZERO);
    let x_axis = matrix.transform_point3(DVec3::X) - origin;
    let y_axis = matrix.transform_point3(DVec3::Y) - origin;
    if x_axis.z.abs() > 1.0e-5 || y_axis.z.abs() > 1.0e-5 {
        return None;
    }

    let x = x_axis.truncate();
    let y = y_axis.truncate();
    let x_len = x.length();
    let y_len = y.length();
    if x_len <= f64::EPSILON || y_len <= f64::EPSILON {
        return None;
    }
    if (x_len - y_len).abs() > x_len.max(y_len) * 1.0e-4 {
        return None;
    }
    if x.dot(y).abs() > x_len * y_len * 1.0e-4 {
        return None;
    }

    let orientation = x.perp_dot(y);
    if orientation.abs() <= f64::EPSILON {
        None
    } else {
        Some(orientation.signum())
    }
}

fn append_baked_bulge_segment(
    out: &mut Vec<PolyVertex>,
    start: DVec3,
    end: DVec3,
    bulge: f64,
    normal: DVec3,
    transform: &Transform,
) {
    if out.is_empty() {
        out.push(PolyVertex::straight(transform.apply(start)));
    }

    let chord = end - start;
    let chord_len = chord.length();
    if bulge.abs() <= f64::EPSILON || chord_len <= f64::EPSILON {
        out.push(PolyVertex::straight(transform.apply(end)));
        return;
    }

    let theta = 4.0 * bulge.atan();
    let radius = chord_len * (1.0 + bulge * bulge) / (4.0 * bulge.abs());
    let midpoint = (start + end) * 0.5;
    let chord_dir = chord / chord_len;
    let perp = normal.cross(chord_dir).normalize_or_zero();
    if perp.length_squared() <= f64::EPSILON {
        out.push(PolyVertex::straight(transform.apply(end)));
        return;
    }

    let offset = chord_len * (1.0 - bulge * bulge) / (4.0 * bulge);
    let center = midpoint + perp * offset;
    let start_vec = start - center;
    let segments = arc_segment_count(radius, theta.abs());
    for i in 1..=segments {
        let point = if i == segments {
            end
        } else {
            let t = theta * (i as f64 / segments as f64);
            center + DQuat::from_axis_angle(normal, t).mul_vec3(start_vec)
        };
        out.push(PolyVertex::straight(transform.apply(point)));
    }
}

fn plain_text(raw: &str) -> String {
    replace_legacy_text_codes(raw).trim().to_string()
}

fn plain_mtext(raw: &str) -> String {
    let mut out = String::new();
    let mut chars = raw.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\\' => match chars.next() {
                Some('P') => out.push('\n'),
                Some('p') => {
                    let _ = read_until_semicolon(&mut chars);
                }
                Some('X') => out.push('\n'),
                Some('~') => out.push(' '),
                Some('\\') => out.push('\\'),
                Some('{') => out.push('{'),
                Some('}') => out.push('}'),
                Some('U') if chars.peek() == Some(&'+') => {
                    let _ = chars.next();
                    let mut code = String::new();
                    for _ in 0..4 {
                        if let Some(hex) = chars.next() {
                            code.push(hex);
                        }
                    }
                    if let Ok(value) = u32::from_str_radix(&code, 16)
                        && let Some(decoded) = char::from_u32(value)
                    {
                        out.push(decoded);
                    }
                }
                Some('S') | Some('s') => {
                    let stack = read_until_semicolon(&mut chars);
                    out.push_str(&plain_stack_text(&stack));
                }
                Some(
                    'A' | 'a' | 'C' | 'c' | 'F' | 'f' | 'H' | 'h' | 'Q' | 'q' | 'T' | 't' | 'W'
                    | 'w',
                ) => {
                    let _ = read_until_semicolon(&mut chars);
                }
                Some('L' | 'l' | 'O' | 'o' | 'K' | 'k') => {}
                Some(other) => out.push(other),
                None => {}
            },
            '{' | '}' => {}
            '\r' => {}
            other => out.push(other),
        }
    }

    replace_legacy_text_codes(&out)
        .lines()
        .map(str::trim)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn read_until_semicolon(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut value = String::new();
    for ch in chars.by_ref() {
        if ch == ';' {
            break;
        }
        value.push(ch);
    }
    value
}

fn plain_stack_text(stack: &str) -> String {
    let mut text = stack
        .replace(['#', '^'], "/")
        .replace("\\P", "\n")
        .replace("\\~", " ");
    if text.contains('/') {
        text = text
            .split('/')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("/");
    }
    text
}

fn replace_legacy_text_codes(raw: &str) -> String {
    let mut out = String::new();
    let mut chars = raw.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '%' && chars.peek() == Some(&'%') {
            let _ = chars.next();
            match chars.next() {
                Some('d') | Some('D') => out.push('°'),
                Some('p') | Some('P') => out.push('±'),
                Some('c') | Some('C') => out.push('Ø'),
                Some('%') => out.push('%'),
                Some('u' | 'U' | 'o' | 'O') => {}
                Some(other) => {
                    out.push('%');
                    out.push('%');
                    out.push(other);
                }
                None => {
                    out.push('%');
                    out.push('%');
                }
            }
        } else {
            out.push(ch);
        }
    }
    out
}

/// An orthonormal pair spanning the plane perpendicular to `normal`.
fn orthonormal_basis(normal: DVec3) -> (DVec3, DVec3) {
    let seed = if normal.z.abs() < 0.9 {
        DVec3::Z
    } else {
        DVec3::X
    };
    let u = seed.cross(normal).normalize_or(DVec3::X);
    let v = normal.cross(u).normalize_or(DVec3::Y);
    (u, v)
}

/// DXF Arbitrary Axis Algorithm: OCS→WCS basis from an extrusion normal.
fn ocs_to_wcs_affine(extrusion_direction: &dxf::Vector) -> DMat4 {
    let mut n = vector_to_dvec3(extrusion_direction);
    if n.length_squared() <= f64::EPSILON {
        return DMat4::IDENTITY;
    }
    n = n.normalize();
    if (n - DVec3::Z).length_squared() <= 1.0e-10 {
        return DMat4::IDENTITY;
    }
    let axis_x = if n.x.abs() < (1.0 / 64.0) && n.y.abs() < (1.0 / 64.0) {
        DVec3::Y.cross(n).normalize_or_zero()
    } else {
        DVec3::Z.cross(n).normalize_or_zero()
    };
    if axis_x.length_squared() <= f64::EPSILON {
        return DMat4::IDENTITY;
    }
    let axis_y = n.cross(axis_x).normalize_or_zero();
    if axis_y.length_squared() <= f64::EPSILON {
        return DMat4::IDENTITY;
    }
    DMat4::from_cols(
        axis_x.extend(0.0),
        axis_y.extend(0.0),
        n.extend(0.0),
        DVec4::new(0.0, 0.0, 0.0, 1.0),
    )
}
