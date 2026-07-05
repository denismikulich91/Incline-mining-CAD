//! CPU scene tessellation primitives. Domain geometry remains double precision;
//! vertices are rebased to a scene origin only at the GPU boundary.

use std::sync::LazyLock;

use glam::DVec3;

use crate::rendering::{StrokeVertex, Vertex};

pub(crate) struct DrawContext<'a> {
    pub(crate) stroke_vertex_buf: &'a mut Vec<StrokeVertex>,
    pub(crate) stroke_index_buf: &'a mut Vec<u32>,
    pub(crate) fill_vertex_buf: &'a mut Vec<Vertex>,
    pub(crate) fill_index_buf: &'a mut Vec<u32>,
    pub(crate) scene_origin: DVec3,
    pub(crate) scale_factor: f32,
}

const MIN_CURVE_SEGMENTS: usize = 16;
const MAX_CURVE_SEGMENTS: usize = 4096;
const TARGET_SEGMENT_LENGTH_WORLD: f64 = 0.15;

fn local(point: DVec3, origin: DVec3) -> [f32; 3] {
    let p = point - origin;
    [p.x as f32, p.y as f32, p.z as f32]
}

pub(crate) fn draw_line(
    ctx: &mut DrawContext,
    start: DVec3,
    end: DVec3,
    line_width: f32,
    color: [f32; 4],
) {
    let delta = end - start;
    if delta.length_squared() <= f64::EPSILON {
        return;
    }
    let line_width = line_width * ctx.scale_factor;
    let i = ctx.stroke_vertex_buf.len() as u32;
    let start = local(start, ctx.scene_origin);
    let end = local(end, ctx.scene_origin);
    let half = line_width.max(1.0) * 0.5;

    ctx.stroke_vertex_buf.extend_from_slice(&[
        StrokeVertex {
            pos: start,
            color,
            other_pos: end,
            offset_px: [-half, 0.0],
            screen_space: 0.0,
        },
        StrokeVertex {
            pos: start,
            color,
            other_pos: end,
            offset_px: [half, 0.0],
            screen_space: 0.0,
        },
        StrokeVertex {
            pos: end,
            color,
            other_pos: start,
            offset_px: [half, 0.0],
            screen_space: 0.0,
        },
        StrokeVertex {
            pos: end,
            color,
            other_pos: start,
            offset_px: [-half, 0.0],
            screen_space: 0.0,
        },
    ]);
    ctx.stroke_index_buf
        .extend_from_slice(&[i + 1, i, i + 3, i, i + 2, i + 3]);
}

/// Draw a filled circle (sphere indicator) in screen space at the given world position.
/// `radius_px` is in device pixels.
pub(crate) fn draw_screen_sphere(
    ctx: &mut DrawContext,
    center: DVec3,
    radius_px: f32,
    color: [f32; 4],
) {
    const SEGMENTS: u32 = 12;
    let base = ctx.stroke_vertex_buf.len() as u32;
    let position = local(center, ctx.scene_origin);
    ctx.stroke_vertex_buf.push(StrokeVertex {
        pos: position,
        color,
        other_pos: position,
        offset_px: [0.0, 0.0],
        screen_space: 1.0,
    });
    for i in 0..SEGMENTS {
        let angle = std::f32::consts::TAU * i as f32 / SEGMENTS as f32;
        ctx.stroke_vertex_buf.push(StrokeVertex {
            pos: position,
            color,
            other_pos: position,
            offset_px: [angle.cos() * radius_px, angle.sin() * radius_px],
            screen_space: 1.0,
        });
    }
    for i in 0..SEGMENTS {
        ctx.stroke_index_buf.extend_from_slice(&[
            base,
            base + i + 1,
            base + (i + 1) % SEGMENTS + 1,
        ]);
    }
}

pub(crate) fn draw_screen_cross(
    ctx: &mut DrawContext,
    center: DVec3,
    half_size_px: f32,
    line_width: f32,
    color: [f32; 4],
) {
    draw_screen_segment(ctx, center, DVec3::X, half_size_px, line_width, color);
    draw_screen_segment(ctx, center, DVec3::Y, half_size_px, line_width, color);
}

fn draw_screen_segment(
    ctx: &mut DrawContext,
    center: DVec3,
    direction: DVec3,
    half_size_px: f32,
    line_width: f32,
    color: [f32; 4],
) {
    let index = ctx.stroke_vertex_buf.len() as u32;
    let position = local(center, ctx.scene_origin);
    let other = local(center + direction, ctx.scene_origin);
    let half_width = (line_width * ctx.scale_factor).max(1.0) * 0.5;
    ctx.stroke_vertex_buf.extend_from_slice(&[
        StrokeVertex {
            pos: position,
            color,
            other_pos: other,
            offset_px: [-half_width, -half_size_px],
            screen_space: 0.0,
        },
        StrokeVertex {
            pos: position,
            color,
            other_pos: other,
            offset_px: [half_width, -half_size_px],
            screen_space: 0.0,
        },
        StrokeVertex {
            pos: position,
            color,
            other_pos: other,
            offset_px: [-half_width, half_size_px],
            screen_space: 0.0,
        },
        StrokeVertex {
            pos: position,
            color,
            other_pos: other,
            offset_px: [half_width, half_size_px],
            screen_space: 0.0,
        },
    ]);
    ctx.stroke_index_buf.extend_from_slice(&[
        index + 1,
        index,
        index + 3,
        index,
        index + 2,
        index + 3,
    ]);
}

/// Add a camera-independent round join in pixel space at a world position.
pub(crate) fn draw_round_join(
    ctx: &mut DrawContext,
    center: DVec3,
    line_width: f32,
    color: [f32; 4],
) {
    const MAX_SEGMENTS: u32 = 16;
    static DIRECTIONS: LazyLock<[[f32; 2]; (MAX_SEGMENTS + 1) as usize]> = LazyLock::new(|| {
        std::array::from_fn(|index| {
            let angle = std::f32::consts::TAU * index as f32 / MAX_SEGMENTS as f32;
            [angle.cos(), angle.sin()]
        })
    });
    let radius = (line_width * ctx.scale_factor).max(1.0) * 0.5;
    let segments = if radius <= 2.0 { 8 } else { MAX_SEGMENTS };
    let direction_step = (MAX_SEGMENTS / segments) as usize;
    let base = ctx.stroke_vertex_buf.len() as u32;
    let position = local(center, ctx.scene_origin);
    ctx.stroke_vertex_buf.push(StrokeVertex {
        pos: position,
        color,
        other_pos: position,
        offset_px: [0.0, 0.0],
        screen_space: 1.0,
    });
    for index in (0..=MAX_SEGMENTS as usize).step_by(direction_step) {
        let direction = DIRECTIONS[index];
        ctx.stroke_vertex_buf.push(StrokeVertex {
            pos: position,
            color,
            other_pos: position,
            offset_px: [direction[0] * radius, direction[1] * radius],
            screen_space: 1.0,
        });
    }
    for index in 0..segments {
        ctx.stroke_index_buf
            .extend_from_slice(&[base, base + index + 1, base + index + 2]);
    }
}

pub(crate) fn draw_bulge_segment(
    ctx: &mut DrawContext,
    start: DVec3,
    end: DVec3,
    bulge: f64,
    line_width: f32,
    color: [f32; 4],
) {
    let chord = end - start;
    let chord_len = chord.length();
    if bulge.abs() <= f64::EPSILON || chord_len <= f64::EPSILON {
        draw_line(ctx, start, end, line_width, color);
        return;
    }

    let theta = 4.0 * bulge.atan();
    let radius = chord_len * (1.0 + bulge * bulge) / (4.0 * bulge.abs());
    let midpoint = (start + end) * 0.5;
    let chord_dir = chord / chord_len;
    let perp = DVec3::Z.cross(chord_dir);
    if perp.length_squared() <= f64::EPSILON {
        draw_line(ctx, start, end, line_width, color);
        return;
    }
    let center = midpoint + perp.normalize() * chord_len * (1.0 - bulge * bulge) / (4.0 * bulge);
    let start_vec = start - center;
    let segments = ((radius * theta.abs() / TARGET_SEGMENT_LENGTH_WORLD).ceil() as usize)
        .clamp(MIN_CURVE_SEGMENTS, MAX_CURVE_SEGMENTS);
    let mut previous = start;
    for i in 1..=segments {
        let point = if i == segments {
            end
        } else {
            center
                + glam::DQuat::from_axis_angle(DVec3::Z, theta * i as f64 / segments as f64)
                    .mul_vec3(start_vec)
        };
        draw_line(ctx, previous, point, line_width, color);
        if i < segments {
            draw_round_join(ctx, point, line_width, color);
        }
        previous = point;
    }
}
