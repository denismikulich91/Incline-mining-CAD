use std::collections::{HashMap, VecDeque};

use super::cuts::lerp_at_z;
use super::*;

impl<'a> App<'a> {
    /// Generate contour polylines from a triangulation and store them as a new
    /// layer in the chosen pidb project. Major contours (multiples of
    /// `major_interval`) use `major_color`; all others use `minor_color`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_contour_triangulation(
        &mut self,
        tri_id: TriangulationId,
        major_interval: f64,
        minor_interval: f64,
        major_color: [f32; 4],
        minor_color: [f32; 4],
        project_index: usize,
        z_range: Option<(f64, f64)>,
    ) -> Result<()> {
        if !minor_interval.is_finite()
            || !major_interval.is_finite()
            || minor_interval <= 0.0
            || major_interval <= 0.0
        {
            anyhow::bail!("Intervals must be positive finite numbers");
        }
        if minor_interval < 1e-6 {
            anyhow::bail!(
                "Minor contour interval is too small (minimum 0.000001) — this would generate an unbounded number of contour levels"
            );
        }
        if major_interval < minor_interval {
            anyhow::bail!("Major interval must be >= minor interval");
        }

        let (mesh, tri_name) = {
            let tri = self
                .triangulations
                .iter()
                .find(|t| t.id == tri_id)
                .ok_or_else(|| anyhow::anyhow!("Triangulation not found"))?;
            (tri.mesh.clone(), tri.name.clone())
        };

        let bounds = mesh.bounds();
        let mut z_lo = bounds.min.z;
        let mut z_hi = bounds.max.z;
        if let Some((range_lo, range_hi)) = z_range {
            if !range_lo.is_finite() || !range_hi.is_finite() || range_lo >= range_hi {
                anyhow::bail!("Contour Z range must be finite with min < max");
            }
            z_lo = z_lo.max(range_lo);
            z_hi = z_hi.min(range_hi);
            if z_lo >= z_hi {
                anyhow::bail!("Contour Z range does not overlap the triangulation");
            }
        }

        if (z_hi - z_lo).abs() < 1e-10 {
            anyhow::bail!("Triangulation has no Z extent to contour");
        }

        const MAX_CONTOUR_LEVELS: f64 = 100_000.0;
        let level_estimate = (z_hi - z_lo) / minor_interval;
        if level_estimate > MAX_CONTOUR_LEVELS {
            anyhow::bail!(
                "These settings would generate about {level_estimate:.0} contour levels over a Z extent of {:.1} m. \
                 Increase the minor interval, or restrict the Z range to contour a slice of the triangulation.",
                z_hi - z_lo
            );
        }

        if self.workspace.projects.get(project_index).is_none() {
            anyhow::bail!("Target PIDB not found");
        }

        let verts_raw = mesh.vertices();
        let contour_faces: Vec<ContourFace> = mesh
            .face_vertex_indices_iter()
            .map(|face| {
                let vertices = [verts_raw[face[0]], verts_raw[face[1]], verts_raw[face[2]]];
                let z_min = vertices
                    .iter()
                    .map(|vertex| vertex.z)
                    .fold(f64::INFINITY, f64::min);
                let z_max = vertices
                    .iter()
                    .map(|vertex| vertex.z)
                    .fold(f64::NEG_INFINITY, f64::max);
                ContourFace {
                    vertices,
                    z_min,
                    z_max,
                }
            })
            .collect();
        let mut polylines: Vec<(Vec<glam::DVec3>, crate::model::ObjectColor)> = Vec::new();

        // Bucket faces by the levels they cross so each level only visits the
        // faces that actually intersect it, instead of rescanning every face.
        let levels = contour_levels(z_lo, z_hi, minor_interval, major_interval);
        let mut level_faces: Vec<Vec<u32>> = vec![Vec::new(); levels.len()];
        for (face_index, face) in contour_faces.iter().enumerate() {
            let first = levels.partition_point(|level| *level < face.z_min);
            let last = levels.partition_point(|level| *level < face.z_max);
            for bucket in &mut level_faces[first..last] {
                bucket.push(face_index as u32);
            }
        }

        for (z_level, face_indices) in levels.iter().copied().zip(&level_faces) {
            let is_major = is_major_contour(z_level, major_interval);
            let color = if is_major { major_color } else { minor_color };
            let line_color = crate::model::ObjectColor::Fixed(color);
            let mut level_segments = Vec::new();

            for &face_index in face_indices {
                let face = &contour_faces[face_index as usize];
                if let Some(seg) = triangle_contour_segment(face.vertices, z_level) {
                    level_segments.push([
                        glam::DVec3::new(seg[0].x, seg[0].y, seg[0].z),
                        glam::DVec3::new(seg[1].x, seg[1].y, seg[1].z),
                    ]);
                }
            }

            polylines.extend(
                chain_contour_segments(level_segments)
                    .into_iter()
                    .map(|verts| (verts, line_color)),
            );
        }

        if polylines.is_empty() {
            anyhow::bail!("No contour segments were generated for the selected intervals");
        }

        if self.workspace.active_index != Some(project_index) {
            self.history.clear();
            self.workspace.set_active_index(project_index);
        }

        let layer_name = format!("{tri_name}_contour");
        let project = &mut self.workspace.projects[project_index];
        let layer_id = project.pidb.document.allocate_layer_id();
        let layer = Layer {
            id: layer_id,
            name: layer_name.clone(),
            color_index: None,
            color: [1.0, 1.0, 1.0, 1.0],
            visible: true,
            elevation: 0.0,
        };
        let objects: Vec<Object> = polylines
            .into_iter()
            .map(|(verts, color)| Object::Polyline {
                id: project.pidb.document.allocate_object_id(),
                layer: layer_id,
                verts: verts
                    .into_iter()
                    .map(crate::model::PolyVertex::straight)
                    .collect(),
                closed: false,
                color,
                fill: crate::model::FillStyle::Clear,
                line_weight: 1.0,
            })
            .collect();
        let line_count = objects.len();
        self.history.execute(
            &mut project.pidb.document,
            crate::model::Command::AddLayerSnapshot { layer, objects },
        );
        project.loaded_layers.insert(layer_id);
        self.editor.active_layer = Some(layer_id);
        userspace_log!("Generated {line_count} contour polyline(s) for triangulation '{tri_name}'");
        self.invalidate_geometry();
        Ok(())
    }
}

struct ContourFace {
    vertices: [tri00t::Vertex; 3],
    z_min: f64,
    z_max: f64,
}

fn contour_levels(z_lo: f64, z_hi: f64, minor_interval: f64, major_interval: f64) -> Vec<f64> {
    let mut levels = Vec::new();
    append_levels(&mut levels, z_lo, z_hi, minor_interval);
    append_levels(&mut levels, z_lo, z_hi, major_interval);
    levels.sort_by(f64::total_cmp);
    levels.dedup_by(|a, b| (*a - *b).abs() <= 1e-8);
    levels
}

fn append_levels(levels: &mut Vec<f64>, z_lo: f64, z_hi: f64, interval: f64) {
    let first = (z_lo / interval).ceil();
    let last = (z_hi / interval).floor();
    if first > last {
        return;
    }
    let count = (last - first) as usize;
    for i in 0..=count {
        levels.push((first + i as f64) * interval);
    }
}

fn is_major_contour(z_level: f64, major_interval: f64) -> bool {
    let nearest = (z_level / major_interval).round() * major_interval;
    (z_level - nearest).abs() <= 1e-6
}

/// Quantized endpoint key. Both triangles adjacent to a mesh edge interpolate
/// the crossing from the same two vertices, so matching endpoints agree to
/// float noise; quantizing at the kernel tolerance makes them hash-equal.
fn contour_point_key(p: glam::DVec3) -> (i64, i64, i64) {
    let q = crate::model::kernel::XY_TOL;
    (
        (p.x / q).round() as i64,
        (p.y / q).round() as i64,
        (p.z / q).round() as i64,
    )
}

fn chain_contour_segments(segments: Vec<[glam::DVec3; 2]>) -> Vec<Vec<glam::DVec3>> {
    // Endpoint key -> segments touching that point, for O(1) chain extension.
    let mut by_endpoint: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (index, segment) in segments.iter().enumerate() {
        by_endpoint
            .entry(contour_point_key(segment[0]))
            .or_default()
            .push(index);
        by_endpoint
            .entry(contour_point_key(segment[1]))
            .or_default()
            .push(index);
    }

    let mut used = vec![false; segments.len()];
    let mut chains = Vec::new();

    let mut take_neighbor = |used: &mut Vec<bool>, point: glam::DVec3| -> Option<glam::DVec3> {
        let candidates = by_endpoint.get_mut(&contour_point_key(point))?;
        while let Some(index) = candidates.pop() {
            if used[index] {
                continue;
            }
            used[index] = true;
            let [a, b] = segments[index];
            let key = contour_point_key(point);
            return Some(if contour_point_key(a) == key { b } else { a });
        }
        None
    };

    for seed in 0..segments.len() {
        if used[seed] {
            continue;
        }
        used[seed] = true;
        let [a, b] = segments[seed];
        let mut chain: VecDeque<glam::DVec3> = VecDeque::from([a, b]);
        while let Some(next) = take_neighbor(&mut used, *chain.back().unwrap()) {
            chain.push_back(next);
        }
        while let Some(previous) = take_neighbor(&mut used, *chain.front().unwrap()) {
            chain.push_front(previous);
        }
        if chain.len() >= 2 {
            chains.push(chain.into_iter().collect());
        }
    }
    chains
}

pub(super) fn triangle_contour_segment(
    v: [tri00t::Vertex; 3],
    z_level: f64,
) -> Option<[tri00t::Vertex; 2]> {
    let mut pts: Vec<tri00t::Vertex> = Vec::with_capacity(2);
    for i in 0..3 {
        let a = v[i];
        let b = v[(i + 1) % 3];
        if (a.z <= z_level && z_level < b.z) || (b.z <= z_level && z_level < a.z) {
            pts.push(lerp_at_z(a, b, z_level));
        }
    }
    if pts.len() == 2 {
        Some([pts[0], pts[1]])
    } else {
        None
    }
}
