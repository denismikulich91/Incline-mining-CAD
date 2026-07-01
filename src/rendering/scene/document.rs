//! Document entity scene assembly.

use glam::DVec3;
use lyon::{
    math::point as lyon_point,
    path::Path as LyonPath,
    tessellation::{BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers},
};

use crate::rendering::{
    Vertex,
    geometry::{DrawContext, draw_line},
};

pub(crate) fn draw_origin_marker(draw_ctx: &mut DrawContext<'_>) {
    // Origin marker: 10 metres end-to-end on each axis.
    const ORIGIN_MARKER_HALF_SIZE: f64 = 5.0;
    draw_line(
        draw_ctx,
        DVec3::new(-ORIGIN_MARKER_HALF_SIZE, 0.0, 0.0),
        DVec3::new(ORIGIN_MARKER_HALF_SIZE, 0.0, 0.0),
        1.0,
        [1.0, 0.0, 0.0, 1.0],
    );
    draw_line(
        draw_ctx,
        DVec3::new(0.0, -ORIGIN_MARKER_HALF_SIZE, 0.0),
        DVec3::new(0.0, ORIGIN_MARKER_HALF_SIZE, 0.0),
        1.0,
        [1.0, 0.0, 0.0, 1.0],
    );
    draw_line(
        draw_ctx,
        DVec3::new(0.0, 0.0, -ORIGIN_MARKER_HALF_SIZE),
        DVec3::new(0.0, 0.0, ORIGIN_MARKER_HALF_SIZE),
        1.0,
        [1.0, 0.0, 0.0, 1.0],
    );
}

/// Compute an orthonormal local frame for a polygon's best-fit plane using
/// Newell's method. Returns (centroid, axis_u, axis_v). For horizontal
/// polygons (normal ~= Z) the axes align with X and Y, so horizontal fills
/// are identical to the old XY projection. For tilted polygons the fill
/// sits in the actual face plane rather than being projected onto XY.
pub(crate) fn polygon_plane_frame(verts: &[crate::model::PolyVertex]) -> (DVec3, DVec3, DVec3) {
    let n = verts.len();
    let centroid = verts.iter().fold(DVec3::ZERO, |a, v| a + v.pos) / n as f64;

    let mut normal = DVec3::ZERO;
    for i in 0..n {
        let c = verts[i].pos;
        let d = verts[(i + 1) % n].pos;
        normal.x += (c.y - d.y) * (c.z + d.z);
        normal.y += (c.z - d.z) * (c.x + d.x);
        normal.z += (c.x - d.x) * (c.y + d.y);
    }
    let normal = if normal.length_squared() > f64::EPSILON {
        normal.normalize()
    } else {
        DVec3::Z
    };

    let up_hint = if normal.z.abs() < 0.9 {
        DVec3::Z
    } else {
        DVec3::Y
    };
    let axis_u = up_hint.cross(normal).normalize_or(DVec3::X);
    let axis_v = normal.cross(axis_u).normalize_or(DVec3::Y);
    (centroid, axis_u, axis_v)
}

/// Draw hatch lines at `angle_deg` clipped to the interior of a closed polygon.
/// Works in the polygon's own plane, so the fill is correct regardless of
/// camera orientation or polygon tilt.
pub(crate) fn fill_polygon_hatch(
    draw_ctx: &mut DrawContext<'_>,
    verts: &[crate::model::PolyVertex],
    color: [f32; 4],
    angle_deg: f32,
    spacing: f64,
    line_weight: f32,
) {
    if verts.len() < 3 {
        return;
    }
    let (centroid, axis_u, axis_v) = polygon_plane_frame(verts);

    let pts_2d: Vec<[f64; 2]> = verts
        .iter()
        .map(|v| {
            let d = v.pos - centroid;
            [d.dot(axis_u), d.dot(axis_v)]
        })
        .collect();

    let angle = (angle_deg as f64).to_radians();
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let rotated: Vec<[f64; 2]> = pts_2d
        .iter()
        .map(|p| [p[0] * cos_a + p[1] * sin_a, -p[0] * sin_a + p[1] * cos_a])
        .collect();

    let min_v = rotated.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
    let max_v = rotated
        .iter()
        .map(|p| p[1])
        .fold(f64::NEG_INFINITY, f64::max);
    let n = rotated.len();

    let mut scan_v = min_v + spacing * 0.5;
    while scan_v < max_v {
        let mut xs: Vec<f64> = Vec::new();
        for i in 0..n {
            let j = (i + 1) % n;
            let v0 = rotated[i][1];
            let u0 = rotated[i][0];
            let v1 = rotated[j][1];
            let u1 = rotated[j][0];
            if (v0 <= scan_v && scan_v < v1) || (v1 <= scan_v && scan_v < v0) {
                let t = (scan_v - v0) / (v1 - v0);
                xs.push(u0 + t * (u1 - u0));
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut i = 0;
        while i + 1 < xs.len() {
            let (su0, su1) = (xs[i], xs[i + 1]);
            let pu0 = su0 * cos_a - scan_v * sin_a;
            let pv0 = su0 * sin_a + scan_v * cos_a;
            let pu1 = su1 * cos_a - scan_v * sin_a;
            let pv1 = su1 * sin_a + scan_v * cos_a;
            let a = centroid + pu0 * axis_u + pv0 * axis_v;
            let b = centroid + pu1 * axis_u + pv1 * axis_v;
            draw_line(draw_ctx, a, b, line_weight, color);
            i += 2;
        }
        scan_v += spacing;
    }
}

/// Tessellate a closed polygon using lyon and push the triangles into the fill
/// buffers. Works in the polygon's own plane so the solid fill is correct
/// regardless of camera orientation or polygon tilt.
pub(crate) fn fill_polygon_solid(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    verts: &[crate::model::PolyVertex],
    color: [f32; 4],
    scene_origin: DVec3,
) {
    if verts.len() < 3 {
        return;
    }
    let (centroid, axis_u, axis_v) = polygon_plane_frame(verts);

    let pts_2d: Vec<[f32; 2]> = verts
        .iter()
        .map(|v| {
            let d = v.pos - centroid;
            [d.dot(axis_u) as f32, d.dot(axis_v) as f32]
        })
        .collect();

    let mut builder = LyonPath::builder();
    builder.begin(lyon_point(pts_2d[0][0], pts_2d[0][1]));
    for pt in &pts_2d[1..] {
        builder.line_to(lyon_point(pt[0], pt[1]));
    }
    builder.close();
    let path = builder.build();

    let base = vertices.len();
    if base > u32::MAX as usize {
        return;
    }
    let mut tessellator = FillTessellator::new();
    let mut output: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    let result = tessellator.tessellate_path(
        &path,
        &FillOptions::default(),
        &mut BuffersBuilder::new(&mut output, |vertex: FillVertex| {
            [vertex.position().x, vertex.position().y]
        }),
    );
    if result.is_err() {
        return;
    }
    for [u, v] in &output.vertices {
        let pos = centroid + *u as f64 * axis_u + *v as f64 * axis_v - scene_origin;
        vertices.push(Vertex {
            pos: [pos.x as f32, pos.y as f32, pos.z as f32],
            color,
        });
    }
    let base = base as u32;
    for idx in &output.indices {
        indices.push(base + idx);
    }
}
