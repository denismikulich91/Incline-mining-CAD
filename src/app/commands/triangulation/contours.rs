use super::cuts::lerp_at_z;
use super::*;

impl<'a> App<'a> {
    /// Generate contour polylines from a triangulation and store them as a new
    /// layer in the chosen pidb project. Major contours (multiples of
    /// `major_interval`) use `major_color`; all others use `minor_color`.
    pub(crate) fn generate_contour_triangulation(
        &mut self,
        tri_id: TriangulationId,
        major_interval: f64,
        minor_interval: f64,
        major_color: [f32; 4],
        minor_color: [f32; 4],
        project_index: usize,
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
        let z_lo = bounds.min.z;
        let z_hi = bounds.max.z;

        if (z_hi - z_lo).abs() < 1e-10 {
            anyhow::bail!("Triangulation has no Z extent to contour");
        }

        if self.workspace.projects.get(project_index).is_none() {
            anyhow::bail!("Target PIDB not found");
        }

        // First contour level at or above z_lo
        let first_level = (z_lo / minor_interval).ceil() * minor_interval;
        let verts_raw = mesh.vertices();
        let mut segments: Vec<([glam::DVec3; 2], crate::model::ObjectColor)> = Vec::new();

        let mut z_level = first_level;
        while z_level <= z_hi + 1e-10 {
            let is_major = (z_level / major_interval).abs() % 1.0 < 1e-6
                || (1.0 - (z_level / major_interval).abs() % 1.0) < 1e-6;
            let color = if is_major { major_color } else { minor_color };
            let line_color = crate::model::ObjectColor::Fixed(color);

            for face in mesh.face_vertex_indices_iter() {
                let raw = [verts_raw[face[0]], verts_raw[face[1]], verts_raw[face[2]]];
                if let Some(seg) = triangle_contour_segment(raw, z_level) {
                    segments.push((
                        [
                            glam::DVec3::new(seg[0].x, seg[0].y, seg[0].z),
                            glam::DVec3::new(seg[1].x, seg[1].y, seg[1].z),
                        ],
                        line_color,
                    ));
                }
            }

            z_level += minor_interval;
        }

        if segments.is_empty() {
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
        let objects: Vec<Object> = segments
            .into_iter()
            .map(|(segment, color)| Object::Polyline {
                id: project.pidb.document.allocate_object_id(),
                layer: layer_id,
                verts: segment
                    .into_iter()
                    .map(crate::model::PolyVertex::straight)
                    .collect(),
                closed: false,
                color,
                fill: crate::model::FillStyle::Clear,
                fill_color: None,
                line_weight: 1.0,
            })
            .collect();
        let line_count = objects.len();
        self.history.execute(
            &mut project.pidb.document,
            crate::model::Command::AddLayerSnapshot { layer, objects },
        );
        project.loaded_layers.insert(layer_id);
        project.dirty = true;
        self.editor.active_layer = Some(layer_id);
        userspace_log!("Generated {line_count} contour segments for triangulation '{tri_name}'");
        self.invalidate_geometry();
        Ok(())
    }
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
