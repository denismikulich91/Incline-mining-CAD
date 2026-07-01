use super::*;

impl<'a> App<'a> {
    /// Cut the triangulation, clipping each triangle against the XY polygon boundary
    /// using Sutherland-Hodgman. Produces smooth edges instead of centroid-based jagged cuts.
    pub(crate) fn cut_triangulation_by_polygon(
        &mut self,
        tri_id: TriangulationId,
        polygon_id: crate::model::ObjectId,
        name: String,
    ) -> Result<()> {
        let (mesh, tri_name) = {
            let tri = self
                .triangulations
                .iter()
                .find(|t| t.id == tri_id)
                .ok_or_else(|| anyhow::anyhow!("Triangulation not found"))?;
            (tri.mesh.clone(), tri.name.clone())
        };

        let poly_verts: Vec<glam::DVec3> = match self.scene_document.get_object(polygon_id) {
            Some(Object::Polyline {
                verts,
                closed: true,
                ..
            }) => verts.iter().map(|v| v.pos).collect(),
            _ => anyhow::bail!("Selected object is not a closed polygon"),
        };

        let (new_verts, new_faces) = clip_mesh_by_polygon_xy(&mesh, &poly_verts);

        if new_faces.is_empty() {
            anyhow::bail!("No triangulation geometry falls inside the selected polygon");
        }
        userspace_log!("Cut triangulation '{}' by polygon", tri_name);
        self.finish_generated_triangulation(name, new_verts, new_faces, TriSurfaceType::Surface)
    }

    /// Trim a topology to the region a pit shell does not excavate, so the two meshes meet at
    /// a seam that follows their true 3D contact line (where the shell crosses the terrain) —
    /// not a fixed design polygon. Topology under parts of the shell that float above the
    /// ground is kept. The pit shell mesh may be multi-valued in XY (walls, benches) or a
    /// watertight closed solid; its flat crest cap never forces removal on its own.
    pub(crate) fn cut_topology_by_pit_shell(
        &mut self,
        topology_id: TriangulationId,
        pit_shell_id: TriangulationId,
        name: String,
    ) -> Result<()> {
        if topology_id == pit_shell_id {
            anyhow::bail!("Topology and pit shell must be different triangulations");
        }

        let (topology_mesh, topology_name) = {
            let tri = self
                .triangulations
                .iter()
                .find(|t| t.id == topology_id)
                .ok_or_else(|| anyhow::anyhow!("Topology not found"))?;
            (tri.mesh.clone(), tri.name.clone())
        };
        let pit_shell_mesh = {
            let tri = self
                .triangulations
                .iter()
                .find(|t| t.id == pit_shell_id)
                .ok_or_else(|| anyhow::anyhow!("Pit shell not found"))?;
            tri.mesh.clone()
        };

        let pit_shell = prepare_pit_shell_surface(&pit_shell_mesh)?;
        let (new_verts, new_faces) = clip_topology_to_pit_shell(&topology_mesh, &pit_shell);

        if new_faces.is_empty() {
            anyhow::bail!("No topology geometry falls outside the pit shell");
        }
        userspace_log!("Cut topology '{}' to pit shell", topology_name);
        self.finish_generated_triangulation(name, new_verts, new_faces, TriSurfaceType::Surface)
    }

    /// Cut the triangulation to the Z band [z_min, z_max], clipping triangles
    /// that straddle the boundary planes.
    pub(crate) fn cut_triangulation_by_z(
        &mut self,
        tri_id: TriangulationId,
        z_min: f64,
        z_max: f64,
        name: String,
    ) -> Result<()> {
        if z_min >= z_max {
            anyhow::bail!("Z min must be less than Z max");
        }

        let (mesh, tri_name) = {
            let tri = self
                .triangulations
                .iter()
                .find(|t| t.id == tri_id)
                .ok_or_else(|| anyhow::anyhow!("Triangulation not found"))?;
            (tri.mesh.clone(), tri.name.clone())
        };

        let verts_raw = mesh.vertices();
        let mut new_verts: Vec<tri00t::Vertex> = Vec::new();
        let mut new_faces: Vec<[u32; 3]> = Vec::new();

        for face in mesh.face_vertex_indices_iter() {
            let raw = [verts_raw[face[0]], verts_raw[face[1]], verts_raw[face[2]]];
            for clipped in clip_triangle_z(raw, z_min, z_max) {
                let base = new_verts.len() as u32;
                new_verts.extend_from_slice(&clipped);
                new_faces.push([base, base + 1, base + 2]);
            }
        }

        if new_faces.is_empty() {
            anyhow::bail!("No triangulation geometry lies within the specified Z range");
        }
        userspace_log!(
            "Cut triangulation '{}' by Z band [{:.3}, {:.3}]",
            tri_name,
            z_min,
            z_max
        );
        self.finish_generated_triangulation(name, new_verts, new_faces, TriSurfaceType::Surface)
    }

    /// Vertically clip one triangulation against another topology. Only the XY
    /// overlap with the reference topology is emitted.
    pub(crate) fn cut_triangulation_by_surface(
        &mut self,
        target_id: TriangulationId,
        reference_id: TriangulationId,
        side: TriSurfaceCutSide,
        name: String,
    ) -> Result<()> {
        if target_id == reference_id {
            anyhow::bail!("Cut object and reference topology must be different triangulations");
        }

        let target = self
            .triangulations
            .iter()
            .find(|triangulation| triangulation.id == target_id)
            .ok_or_else(|| anyhow::anyhow!("Cut triangulation not found"))?;
        let reference = self
            .triangulations
            .iter()
            .find(|triangulation| triangulation.id == reference_id)
            .ok_or_else(|| anyhow::anyhow!("Reference topology not found"))?;

        let (new_vertices, new_faces) = clip_mesh_by_surface(&target.mesh, &reference.mesh, side)?;
        if new_faces.is_empty() {
            let retained = match side {
                TriSurfaceCutSide::CutTop => "at or below",
                TriSurfaceCutSide::CutBottom => "at or above",
            };
            anyhow::bail!(
                "No cut-object geometry lies {retained} the reference topology within its XY coverage"
            );
        }

        userspace_log!(
            "Cut triangulation '{}' by surface '{}' ({:?})",
            target.name,
            reference.name,
            side
        );
        self.finish_generated_triangulation(name, new_vertices, new_faces, TriSurfaceType::Surface)
    }
}

/// Clip a mesh against an XY polygon, keeping the inside (the polygon's own footprint).
pub(super) fn clip_mesh_by_polygon_xy(
    mesh: &tri00t::Triangulation,
    polygon: &[glam::DVec3],
) -> (Vec<tri00t::Vertex>, Vec<[u32; 3]>) {
    let verts_raw = mesh.vertices();
    let mut new_verts: Vec<tri00t::Vertex> = Vec::new();
    let mut new_faces: Vec<[u32; 3]> = Vec::new();

    for face in mesh.face_vertex_indices_iter() {
        let raw = [verts_raw[face[0]], verts_raw[face[1]], verts_raw[face[2]]];
        for clipped in clip_triangle_by_polygon_concave(raw, polygon) {
            let base = new_verts.len() as u32;
            new_verts.extend_from_slice(&clipped);
            new_faces.push([base, base + 1, base + 2]);
        }
    }
    (new_verts, new_faces)
}

pub(super) fn clip_triangle_by_polygon_concave(
    v: [tri00t::Vertex; 3],
    polygon: &[glam::DVec3],
) -> Vec<[tri00t::Vertex; 3]> {
    use geo::{BooleanOps, Coord, LineString, Polygon as GeoPoly};

    if polygon.len() < 3 {
        return vec![];
    }

    // Build triangle as a geo Polygon (closed ring: repeat first vertex at end)
    let mut tri_coords: Vec<Coord<f64>> = v.iter().map(|p| Coord { x: p.x, y: p.y }).collect();
    tri_coords.push(tri_coords[0]);
    let tri_poly = GeoPoly::new(LineString::new(tri_coords), vec![]);

    // Build clip polygon (closed ring)
    let mut clip_coords: Vec<Coord<f64>> =
        polygon.iter().map(|p| Coord { x: p.x, y: p.y }).collect();
    clip_coords.push(clip_coords[0]);
    let clip_poly = GeoPoly::new(LineString::new(clip_coords), vec![]);

    let result = tri_poly.intersection(&clip_poly);

    let mut output = Vec::new();
    for poly in &result {
        let exterior: Vec<Coord<f64>> = poly.exterior().coords().copied().collect();
        // geo closes rings (last == first), skip the duplicate
        let n = exterior.len().saturating_sub(1);
        if n < 3 {
            continue;
        }
        // Build a flat [x0, y0, x1, y1, ...] array for earcutr
        let flat: Vec<f64> = exterior[..n].iter().flat_map(|c| [c.x, c.y]).collect();
        let Ok(indices) = earcutr::earcut(&flat, &[], 2) else {
            continue;
        };
        for tri_idx in indices.chunks(3) {
            if tri_idx.len() < 3 {
                continue;
            }
            let make_vert = |i: usize| {
                let x = flat[i * 2];
                let y = flat[i * 2 + 1];
                tri00t::Vertex::new(x, y, bary_z(x, y, v))
            };
            output.push([
                make_vert(tri_idx[0]),
                make_vert(tri_idx[1]),
                make_vert(tri_idx[2]),
            ]);
        }
    }
    output
}

/// Barycentric Z interpolation: given XY point (x, y) inside triangle `v`, return its Z.
pub(super) fn bary_z(x: f64, y: f64, v: [tri00t::Vertex; 3]) -> f64 {
    let denom = (v[1].y - v[2].y) * (v[0].x - v[2].x) + (v[2].x - v[1].x) * (v[0].y - v[2].y);
    if denom.abs() < 1e-12 {
        return (v[0].z + v[1].z + v[2].z) / 3.0;
    }
    let w0 = ((v[1].y - v[2].y) * (x - v[2].x) + (v[2].x - v[1].x) * (y - v[2].y)) / denom;
    let w1 = ((v[2].y - v[0].y) * (x - v[2].x) + (v[0].x - v[2].x) * (y - v[2].y)) / denom;
    let w2 = 1.0 - w0 - w1;
    w0 * v[0].z + w1 * v[1].z + w2 * v[2].z
}

/// Clip a triangle to the band z_min <= z <= z_max, returning 0–2 result triangles.
pub(super) fn clip_triangle_z(
    v: [tri00t::Vertex; 3],
    z_min: f64,
    z_max: f64,
) -> Vec<[tri00t::Vertex; 3]> {
    let mut result = clip_triangle_plane(v, z_min, true);
    let above_min = std::mem::take(&mut result);
    for tri in above_min {
        result.extend(clip_triangle_plane(tri, z_max, false));
    }
    result
}

/// Clip a triangle against a single Z plane, keeping the side selected by `keep_above`.
pub(super) fn clip_triangle_plane(
    v: [tri00t::Vertex; 3],
    z_plane: f64,
    keep_above: bool,
) -> Vec<[tri00t::Vertex; 3]> {
    let inside: [bool; 3] = v.map(|vi| {
        if keep_above {
            vi.z >= z_plane
        } else {
            vi.z <= z_plane
        }
    });
    let count = inside.iter().filter(|&&b| b).count();
    match count {
        0 => vec![],
        3 => vec![v],
        1 => {
            let in_i = inside.iter().position(|&b| b).unwrap();
            let a = v[in_i];
            let b = v[(in_i + 1) % 3];
            let c = v[(in_i + 2) % 3];
            vec![[a, lerp_at_z(a, b, z_plane), lerp_at_z(a, c, z_plane)]]
        }
        2 => {
            let out_i = inside.iter().position(|&b| !b).unwrap();
            let c = v[out_i];
            let a = v[(out_i + 1) % 3];
            let b = v[(out_i + 2) % 3];
            let p = lerp_at_z(c, a, z_plane);
            let q = lerp_at_z(c, b, z_plane);
            vec![[a, b, q], [a, q, p]]
        }
        _ => unreachable!(),
    }
}

pub(super) fn lerp_at_z(a: tri00t::Vertex, b: tri00t::Vertex, z: f64) -> tri00t::Vertex {
    if (b.z - a.z).abs() < 1e-12 {
        return a;
    }
    let t = (z - a.z) / (b.z - a.z);
    tri00t::Vertex::new(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), z)
}

#[derive(Clone, Copy)]
pub(super) struct SurfaceClipVertex {
    pub(super) point: glam::DVec3,
    pub(super) height_delta: f64,
}

pub(super) fn clip_mesh_by_surface(
    target: &tri00t::Triangulation,
    reference: &tri00t::Triangulation,
    side: TriSurfaceCutSide,
) -> Result<(Vec<tri00t::Vertex>, Vec<[u32; 3]>)> {
    let target_vertices = target.vertices();
    let reference_surface = validate_reference_surface(reference)?;
    if reference_surface.skipped_vertical_faces > 0 {
        userspace_warn!(
            "Ignored {} vertical or degenerate reference topology face(s) with no XY area",
            reference_surface.skipped_vertical_faces
        );
    }

    let mut output_vertices = Vec::new();
    let mut output_faces = Vec::new();

    for face in target.face_vertex_indices_iter() {
        let target_triangle = [
            target_vertices[face[0]],
            target_vertices[face[1]],
            target_vertices[face[2]],
        ];
        let target_bounds = triangle_xy_bounds(target_triangle);

        for reference_index in reference_surface.spatial.xy_bounds_candidate_indices(
            &reference_surface.mesh,
            target_bounds.0,
            target_bounds.1,
        ) {
            let reference_triangle = reference_surface.triangles[reference_index];
            let overlap = clip_target_triangle_to_reference_xy(target_triangle, reference_triangle);
            if overlap.len() < 3 {
                continue;
            }

            let polygon: Vec<SurfaceClipVertex> = overlap
                .into_iter()
                .map(|point| {
                    let reference_z = bary_z(point.x, point.y, reference_triangle);
                    SurfaceClipVertex {
                        point,
                        height_delta: point.z - reference_z,
                    }
                })
                .collect();
            let clipped = clip_surface_polygon(polygon, side);
            append_surface_clip_polygon(&clipped, &mut output_vertices, &mut output_faces);
        }
    }

    Ok((output_vertices, output_faces))
}

/// Trim a topology to the region the pit shell does not excavate, so a separately rendered
/// pit shell fills the removed area and the two meshes meet along their true 3D contact
/// line (where the shell surface crosses the terrain) rather than a fixed design polygon.
///
/// For each topology triangle, every overlapping pit-shell face contributes the XY
/// sub-region where the topology lies *at or above* that face — i.e. where the shell digs
/// below the ground and genuinely removes material. Their union (across faces this is the
/// region above the shell's *lower* envelope) is removed; the rest is kept. Crucially,
/// where the shell floats *above* the terrain — e.g. a flat design crest standing over
/// undulating ground near the rim — the topology underneath is **kept**, so the cut is
/// flush with the real contact line instead of the shell's widest XY extent. Because a
/// point is removed as soon as *any* shell face there is below it, a watertight solid's
/// flat crest cap never forces removal on its own. Triangles with no XY overlap against
/// the shell pass through unchanged. A final cleanup pass (`trim_triangles_inside_void`)
/// clips away any excavated triangle portions the per-face clipping missed over near-vertical
/// walls.
pub(super) fn clip_topology_to_pit_shell(
    topology: &tri00t::Triangulation,
    pit_shell: &PreparedReferenceSurface,
) -> (Vec<tri00t::Vertex>, Vec<[u32; 3]>) {
    use geo::{BooleanOps, unary_union};

    let topology_vertices = topology.vertices();
    let mut output_vertices = Vec::new();
    let mut output_faces = Vec::new();

    let pass_through = |target: &[tri00t::Vertex; 3],
                        vertices: &mut Vec<tri00t::Vertex>,
                        faces: &mut Vec<[u32; 3]>| {
        let base = vertices.len() as u32;
        vertices.extend_from_slice(target);
        faces.push([base, base + 1, base + 2]);
    };

    for face in topology.face_vertex_indices_iter() {
        let target = [
            topology_vertices[face[0]],
            topology_vertices[face[1]],
            topology_vertices[face[2]],
        ];
        let bounds = triangle_xy_bounds(target);
        let candidates: Vec<[tri00t::Vertex; 3]> = pit_shell
            .spatial
            .xy_bounds_candidate_indices(&pit_shell.mesh, bounds.0, bounds.1)
            .into_iter()
            .map(|index| pit_shell.triangles[index])
            .collect();

        if candidates.is_empty() {
            // Outside the pit shell's footprint entirely: keep the original geometry.
            pass_through(&target, &mut output_vertices, &mut output_faces);
            continue;
        }

        // For each candidate, the sub-region of `target` within the candidate's XY footprint
        // AND at or above the candidate's surface there — the part the shell excavates. Each
        // region is a convex overlap split by one half-plane, so it stays a simple polygon.
        let mut removed_polygons: Vec<geo::Polygon<f64>> = Vec::new();
        for candidate in &candidates {
            let overlap = clip_target_triangle_to_reference_xy(target, *candidate);
            if overlap.len() < 3 {
                continue;
            }
            let polygon: Vec<SurfaceClipVertex> = overlap
                .into_iter()
                .map(|point| {
                    let reference_z = bary_z(point.x, point.y, *candidate);
                    SurfaceClipVertex {
                        point,
                        height_delta: point.z - reference_z,
                    }
                })
                .collect();
            // Keep the part where the topology is at or above the shell (height_delta >= 0):
            // that is where the shell has cut below the ground and removes material.
            let removed = clip_surface_polygon(polygon, TriSurfaceCutSide::CutBottom);
            if overlay_removed_region_is_true_void(pit_shell, target, &removed)
                && let Some(geo_polygon) = surface_clip_polygon_to_geo(&removed)
            {
                removed_polygons.push(geo_polygon);
            }
        }

        if removed_polygons.is_empty() {
            // The shell floats above the topology everywhere it overlaps here (or there is no
            // real overlap): nothing is excavated, so keep the whole triangle.
            pass_through(&target, &mut output_vertices, &mut output_faces);
            continue;
        }

        let removed_union = unary_union(&removed_polygons);
        let target_polygon = triangle_to_geo_polygon(target);
        let kept = target_polygon.difference(&removed_union);

        for polygon in &kept {
            emit_earcut_polygon(polygon, target, &mut output_vertices, &mut output_faces);
        }
    }

    // Cleanup pass: trim surviving triangles against the pit shell's lower envelope. The
    // per-face XY clipping above can miss geometry when its only overlapping shell faces are
    // near-vertical walls, whose XY projection is a degenerate sliver that Sutherland–Hodgman
    // collapses to nothing. A point-in-triangle query against the shell is robust for those
    // thin faces, and clipping (rather than dropping by centroid) removes partial seam lips.
    trim_triangles_inside_void(pit_shell, &mut output_vertices, &mut output_faces);

    (output_vertices, output_faces)
}

/// Numerical tolerance (metres) for trimming surviving topology against the pit shell's
/// lower envelope. This is deliberately tiny: the cleanup clips partial lips at the contact
/// line, while genuinely floating terrain is preserved because it is below the shell.
const VOID_CLEANUP_TOLERANCE: f64 = 1e-6;

/// Clip triangles against the pit shell's lower envelope, keeping only topology that is not
/// above the shell (plus a tiny numerical tolerance). Rebuilds the vertex/face buffers in
/// place, trimming partial seam lips instead of keeping/dropping whole triangles by centroid.
fn trim_triangles_inside_void(
    pit_shell: &PreparedReferenceSurface,
    vertices: &mut Vec<tri00t::Vertex>,
    faces: &mut Vec<[u32; 3]>,
) {
    trim_triangles_by_shell_face_overlays(pit_shell, vertices, faces);
    trim_triangles_by_envelope_samples(pit_shell, vertices, faces);
}

/// Re-clip emitted topology triangles against every overlapping shell face. This catches
/// triangles whose vertices look safe but whose interior crosses a bench/wall edge in the
/// shell lower envelope.
fn trim_triangles_by_shell_face_overlays(
    pit_shell: &PreparedReferenceSurface,
    vertices: &mut Vec<tri00t::Vertex>,
    faces: &mut Vec<[u32; 3]>,
) {
    use geo::{BooleanOps, unary_union};

    let mut kept_vertices = Vec::with_capacity(vertices.len());
    let mut kept_faces = Vec::with_capacity(faces.len());
    for face in faces.iter() {
        let triangle = [
            vertices[face[0] as usize],
            vertices[face[1] as usize],
            vertices[face[2] as usize],
        ];
        let bounds = triangle_xy_bounds(triangle);
        let mut removed_polygons: Vec<geo::Polygon<f64>> = Vec::new();
        for index in
            pit_shell
                .spatial
                .xy_bounds_candidate_indices(&pit_shell.mesh, bounds.0, bounds.1)
        {
            let shell_triangle = pit_shell.triangles[index];
            let overlap = clip_target_triangle_to_reference_xy(triangle, shell_triangle);
            if overlap.len() < 3 {
                continue;
            }
            let polygon: Vec<SurfaceClipVertex> = overlap
                .into_iter()
                .map(|point| {
                    let reference_z = bary_z(point.x, point.y, shell_triangle);
                    SurfaceClipVertex {
                        point,
                        height_delta: point.z - reference_z - VOID_CLEANUP_TOLERANCE,
                    }
                })
                .collect();
            let removed = clip_surface_polygon(polygon, TriSurfaceCutSide::CutBottom);
            if overlay_removed_region_is_true_void(pit_shell, triangle, &removed)
                && let Some(geo_polygon) = surface_clip_polygon_to_geo(&removed)
            {
                removed_polygons.push(geo_polygon);
            }
        }

        if removed_polygons.is_empty() {
            let base = kept_vertices.len() as u32;
            kept_vertices.extend_from_slice(&triangle);
            kept_faces.push([base, base + 1, base + 2]);
            continue;
        }

        let removed_union = unary_union(&removed_polygons);
        let target_polygon = triangle_to_geo_polygon(triangle);
        let kept = target_polygon.difference(&removed_union);
        for polygon in &kept {
            emit_earcut_polygon(polygon, triangle, &mut kept_vertices, &mut kept_faces);
        }
    }
    *vertices = kept_vertices;
    *faces = kept_faces;
}

/// The face-overlay pass can propose removals from a single shell face even in regions
/// where the true lower envelope still floats above the terrain. Validate each proposed
/// removal against the lower envelope at a representative point before cutting it away.
fn overlay_removed_region_is_true_void(
    pit_shell: &PreparedReferenceSurface,
    target: [tri00t::Vertex; 3],
    removed: &[SurfaceClipVertex],
) -> bool {
    let Some(point) = polygon_representative_xy(removed) else {
        return false;
    };
    let topo_z = bary_z(point.x, point.y, target);
    shell_lower_envelope_z_strict(pit_shell, point.x, point.y)
        .is_some_and(|shell_z| topo_z > shell_z + VOID_CLEANUP_TOLERANCE)
}

fn polygon_representative_xy(polygon: &[SurfaceClipVertex]) -> Option<glam::DVec2> {
    if polygon.len() < 3 {
        return None;
    }
    let sum = polygon.iter().fold(glam::DVec2::ZERO, |sum, vertex| {
        sum + glam::DVec2::new(vertex.point.x, vertex.point.y)
    });
    Some(sum / polygon.len() as f64)
}

/// Final point-query cleanup for near-vertical shell faces whose XY projections can be too
/// thin for polygon clipping.
fn trim_triangles_by_envelope_samples(
    pit_shell: &PreparedReferenceSurface,
    vertices: &mut Vec<tri00t::Vertex>,
    faces: &mut Vec<[u32; 3]>,
) {
    let mut kept_vertices = Vec::with_capacity(vertices.len());
    let mut kept_faces = Vec::with_capacity(faces.len());
    for face in faces.iter() {
        let triangle = [
            vertices[face[0] as usize],
            vertices[face[1] as usize],
            vertices[face[2] as usize],
        ];
        let mut has_shell_coverage = false;
        let polygon: Vec<SurfaceClipVertex> = triangle
            .iter()
            .map(|vertex| {
                let height_delta = shell_lower_envelope_z_strict(pit_shell, vertex.x, vertex.y)
                    .map(|shell_z| {
                        has_shell_coverage = true;
                        vertex.z - shell_z - VOID_CLEANUP_TOLERANCE
                    })
                    .unwrap_or(0.0);
                SurfaceClipVertex {
                    point: glam::DVec3::new(vertex.x, vertex.y, vertex.z),
                    height_delta,
                }
            })
            .collect();

        if !has_shell_coverage {
            let cx = (triangle[0].x + triangle[1].x + triangle[2].x) / 3.0;
            let cy = (triangle[0].y + triangle[1].y + triangle[2].y) / 3.0;
            let cz = (triangle[0].z + triangle[1].z + triangle[2].z) / 3.0;
            if let Some(shell_z) = shell_lower_envelope_z_strict(pit_shell, cx, cy)
                && cz > shell_z + VOID_CLEANUP_TOLERANCE
            {
                continue;
            }
            let base = kept_vertices.len() as u32;
            kept_vertices.extend_from_slice(&triangle);
            kept_faces.push([base, base + 1, base + 2]);
            continue;
        }

        let clipped = clip_surface_polygon(polygon, TriSurfaceCutSide::CutTop);
        append_surface_clip_polygon(&clipped, &mut kept_vertices, &mut kept_faces);
    }
    *vertices = kept_vertices;
    *faces = kept_faces;
}

/// Lower envelope for cleanup clipping, excluding points that lie only on shell XY
/// boundaries. Those boundary vertices are often the already-correct cut seam emitted by
/// the boolean difference pass; trimming them again shrinks the retained outside topology.
fn shell_lower_envelope_z_strict(
    pit_shell: &PreparedReferenceSurface,
    x: f64,
    y: f64,
) -> Option<f64> {
    let point = glam::DVec2::new(x, y);
    let mut lowest = f64::INFINITY;
    for index in pit_shell
        .spatial
        .xy_bounds_candidate_indices(&pit_shell.mesh, point, point)
    {
        if let Some(z) = point_strictly_in_triangle_bary_z(x, y, pit_shell.triangles[index]) {
            lowest = lowest.min(z);
        }
    }
    lowest.is_finite().then_some(lowest)
}

/// Barycentric Z of `(x, y)` on triangle `v`, excluding XY edges.
fn point_strictly_in_triangle_bary_z(x: f64, y: f64, v: [tri00t::Vertex; 3]) -> Option<f64> {
    let denom = (v[1].y - v[2].y) * (v[0].x - v[2].x) + (v[2].x - v[1].x) * (v[0].y - v[2].y);
    if denom.abs() < 1e-12 {
        return None;
    }
    let w0 = ((v[1].y - v[2].y) * (x - v[2].x) + (v[2].x - v[1].x) * (y - v[2].y)) / denom;
    let w1 = ((v[2].y - v[0].y) * (x - v[2].x) + (v[0].x - v[2].x) * (y - v[2].y)) / denom;
    let w2 = 1.0 - w0 - w1;
    if w0 <= 1e-9 || w1 <= 1e-9 || w2 <= 1e-9 {
        return None;
    }
    Some(w0 * v[0].z + w1 * v[1].z + w2 * v[2].z)
}

/// Convert a clipped surface polygon to a geo polygon, normalized to CCW winding so
/// `geo`'s boolean ops treat it as a positive-area region. Returns `None` for polygons
/// with fewer than three points or no XY area.
fn surface_clip_polygon_to_geo(polygon: &[SurfaceClipVertex]) -> Option<geo::Polygon<f64>> {
    use geo::{Coord, LineString, Polygon as GeoPoly};
    if polygon.len() < 3 {
        return None;
    }
    let signed_area: f64 = (0..polygon.len())
        .map(|index| {
            let a = polygon[index].point;
            let b = polygon[(index + 1) % polygon.len()].point;
            a.x * b.y - b.x * a.y
        })
        .sum();
    if signed_area.abs() < 1e-12 {
        return None;
    }
    let mut coords: Vec<Coord<f64>> = polygon
        .iter()
        .map(|vertex| Coord {
            x: vertex.point.x,
            y: vertex.point.y,
        })
        .collect();
    if signed_area < 0.0 {
        coords.reverse();
    }
    coords.push(coords[0]);
    Some(GeoPoly::new(LineString::new(coords), vec![]))
}

/// Build a geo polygon from a triangle's XY projection, normalized to CCW winding so
/// `geo`'s boolean ops treat it as a positive-area region regardless of the triangle's
/// original 3D orientation.
fn triangle_to_geo_polygon(v: [tri00t::Vertex; 3]) -> geo::Polygon<f64> {
    use geo::{Coord, LineString, Polygon as GeoPoly};
    let ccw = if triangle_xy_area(v) >= 0.0 {
        [v[0], v[1], v[2]]
    } else {
        [v[0], v[2], v[1]]
    };
    let mut coords: Vec<Coord<f64>> = ccw.iter().map(|p| Coord { x: p.x, y: p.y }).collect();
    coords.push(coords[0]);
    GeoPoly::new(LineString::new(coords), vec![])
}

/// Triangulate a (possibly holed) XY polygon via earcut, sampling Z from `target`'s plane
/// for every output vertex (valid because every point in `polygon` was derived purely from
/// XY set operations plus linear crossings on `target`'s own plane).
fn emit_earcut_polygon(
    polygon: &geo::Polygon<f64>,
    target: [tri00t::Vertex; 3],
    vertices: &mut Vec<tri00t::Vertex>,
    faces: &mut Vec<[u32; 3]>,
) {
    // Earcut in coordinates local to the polygon's first vertex. At real-world (UTM-scale)
    // magnitudes the absolute coordinates swamp f32-like precision inside the triangulator;
    // shifting to a local origin keeps the working range small and robust.
    let origin = polygon.exterior().0.first().copied();
    let Some(origin) = origin else {
        return;
    };

    let mut flat: Vec<f64> = Vec::new();
    let mut hole_indices: Vec<usize> = Vec::new();
    let exterior_len = push_seam_ring(polygon.exterior(), origin, &mut flat);
    if exterior_len < 3 {
        // The exterior collapsed to a sliver thinner than the seam tolerance: drop it. Any
        // gap is sub-millimetre and lies on the contact line, hidden by the pit shell.
        return;
    }
    for interior in polygon.interiors() {
        let start = flat.len() / 2;
        let hole_len = push_seam_ring(interior, origin, &mut flat);
        if hole_len >= 3 {
            hole_indices.push(start);
        } else {
            flat.truncate(start * 2);
        }
    }

    let mut emit_tri = |ia: usize, ib: usize, ic: usize| {
        let make_vert = |i: usize| {
            let x = flat[i * 2] + origin.x;
            let y = flat[i * 2 + 1] + origin.y;
            tri00t::Vertex::new(x, y, bary_z(x, y, target))
        };
        let a = make_vert(ia);
        let b = make_vert(ib);
        let c = make_vert(ic);
        // Drop needle triangles: sliver artefacts of clipping a triangle against the jagged
        // contact line. A triangle thinner than SEAM_TOLERANCE in its narrow dimension renders
        // as a floating edge but carries no visible area, so skipping it is purely cosmetic.
        let ab = glam::DVec2::new(b.x - a.x, b.y - a.y);
        let ac = glam::DVec2::new(c.x - a.x, c.y - a.y);
        let double_area = (ab.x * ac.y - ab.y * ac.x).abs();
        let longest_edge = (b.x - a.x)
            .hypot(b.y - a.y)
            .max((c.x - b.x).hypot(c.y - b.y))
            .max((a.x - c.x).hypot(a.y - c.y));
        if longest_edge <= 0.0 || double_area / longest_edge < SEAM_TOLERANCE {
            return;
        }
        let base = vertices.len() as u32;
        vertices.push(a);
        vertices.push(b);
        vertices.push(c);
        faces.push([base, base + 1, base + 2]);
    };

    match earcutr::earcut(&flat, &hole_indices, 2) {
        Ok(indices) => {
            for tri_idx in indices.chunks(3) {
                if tri_idx.len() == 3 {
                    emit_tri(tri_idx[0], tri_idx[1], tri_idx[2]);
                }
            }
        }
        Err(_) => {
            // Rather than silently dropping the polygon (which would leave a hole/floating
            // edge in the seam), fall back to a triangle fan over the exterior ring. This is
            // only exact for convex rings, but every clipped topology sub-polygon starts from
            // a single triangle, so its exterior is convex or near-convex in practice.
            for i in 1..exterior_len.saturating_sub(1) {
                emit_tri(0, i, i + 1);
            }
        }
    }
}

/// Tolerance (metres) for treating seam geometry as degenerate: vertices within this
/// distance of the line through their neighbours are dropped, and triangles thinner than
/// this are skipped. Well below any real mine-surface detail, so this only removes the
/// numerical slivers produced by clipping topology triangles against the jagged 3D contact
/// line — the source of "floating edge" artefacts along the cut.
const SEAM_TOLERANCE: f64 = 1e-3;

/// Push a ring's vertices (translated to `origin`) into `flat` as `[x, y, ...]`, dropping
/// vertices that are within `SEAM_TOLERANCE` of the segment joining their neighbours. This
/// collapses the redundant, near-collinear points left along the contact line by unioning
/// many per-face clip polygons, which would otherwise make earcut emit needle triangles.
/// Returns the number of vertices kept.
fn push_seam_ring(
    ring: &geo::LineString<f64>,
    origin: geo::Coord<f64>,
    flat: &mut Vec<f64>,
) -> usize {
    let coords: Vec<geo::Coord<f64>> = ring.coords().copied().collect();
    let n = coords.len().saturating_sub(1); // geo rings repeat the first point at the end
    if n < 3 {
        return 0;
    }
    let keep: Vec<bool> = (0..n)
        .map(|i| {
            let prev = coords[(i + n - 1) % n];
            let cur = coords[i];
            let next = coords[(i + 1) % n];
            let dx = next.x - prev.x;
            let dy = next.y - prev.y;
            let len = dx.hypot(dy);
            if len < 1e-9 {
                return true;
            }
            let perp = ((cur.x - prev.x) * dy - (cur.y - prev.y) * dx).abs() / len;
            perp >= SEAM_TOLERANCE
        })
        .collect();
    let kept = keep.iter().filter(|&&k| k).count();
    if kept < 3 {
        return 0;
    }
    for (i, c) in coords[..n].iter().enumerate() {
        if keep[i] {
            flat.push(c.x - origin.x);
            flat.push(c.y - origin.y);
        }
    }
    kept
}

#[derive(Debug)]
pub(super) struct PreparedReferenceSurface {
    pub(super) mesh: tri00t::Triangulation,
    pub(super) triangles: Vec<[tri00t::Vertex; 3]>,
    pub(super) spatial: crate::model::spatial::TriangleBvh,
    pub(super) skipped_vertical_faces: usize,
}

pub(super) fn validate_reference_surface(
    reference: &tri00t::Triangulation,
) -> Result<PreparedReferenceSurface> {
    let prepared = prepare_reference_surface_relaxed(reference)?;

    let overlap_area_tolerance = reference_xy_overlap_area_tolerance(reference);
    for (index, triangle) in prepared.triangles.iter().copied().enumerate() {
        let bounds = triangle_xy_bounds(triangle);
        for other_index in
            prepared
                .spatial
                .xy_bounds_candidate_indices(&prepared.mesh, bounds.0, bounds.1)
        {
            if other_index <= index {
                continue;
            }
            let overlap = triangle_intersection_xy(triangle, prepared.triangles[other_index]);
            let overlap_area = polygon_area_xy(&overlap).abs();
            if overlap_area > overlap_area_tolerance {
                let z_delta = overlap_z_delta(triangle, prepared.triangles[other_index], &overlap);
                anyhow::bail!(
                    "Reference topology overlaps itself in XY and is not single-valued \
                    (triangles {index} and {other_index} overlap by {overlap_area:.6} \
                    square units with up to {z_delta:.6} Z difference)"
                );
            }
        }
    }
    Ok(prepared)
}

fn reference_xy_overlap_area_tolerance(reference: &tri00t::Triangulation) -> f64 {
    let bounds = reference.bounds();
    let dx = bounds.max.x - bounds.min.x;
    let dy = bounds.max.y - bounds.min.y;
    (dx.abs().max(dy.abs()).powi(2) * 1.0e-14).max(1.0e-10)
}

fn overlap_z_delta(a: [tri00t::Vertex; 3], b: [tri00t::Vertex; 3], overlap: &[glam::DVec2]) -> f64 {
    if overlap.is_empty() {
        return 0.0;
    }
    let mut samples = overlap.to_vec();
    let centroid = overlap
        .iter()
        .copied()
        .fold(glam::DVec2::ZERO, |sum, point| sum + point)
        / overlap.len() as f64;
    samples.push(centroid);

    samples
        .into_iter()
        .map(|point| (bary_z(point.x, point.y, a) - bary_z(point.x, point.y, b)).abs())
        .fold(0.0, f64::max)
}

/// Prepare a pit shell mesh as a candidate-matching surface for `clip_topology_to_pit_shell`.
/// Unlike `prepare_reference_surface_relaxed`, this keeps near-vertical wall faces — they
/// have little or no XY-projected area but are exactly the geometry that determines how far
/// in from the rim the topology needs to be removed as depth increases. Only genuinely
/// degenerate faces (zero area in 3D — duplicate or collinear points) are dropped.
pub(super) fn prepare_pit_shell_surface(
    pit_shell: &tri00t::Triangulation,
) -> Result<PreparedReferenceSurface> {
    let vertices = pit_shell.vertices();
    if pit_shell.face_count() == 0 {
        anyhow::bail!("Pit shell contains no faces");
    }

    let mut prepared_vertices = Vec::new();
    let mut prepared_faces = Vec::new();
    let mut triangles = Vec::new();

    for face in pit_shell.face_vertex_indices_iter() {
        let triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]];
        if triangle_area_3d(triangle) <= 1e-9 {
            continue;
        }
        let base = prepared_vertices.len() as u32;
        prepared_vertices.extend_from_slice(&triangle);
        prepared_faces.push([base, base + 1, base + 2]);
        triangles.push(triangle);
    }

    if triangles.is_empty() {
        anyhow::bail!("Pit shell contains no usable (non-degenerate) faces");
    }

    let mesh = tri00t::Triangulation::from_vertices_and_faces(prepared_vertices, prepared_faces);
    let spatial = crate::model::spatial::TriangleBvh::build(&mesh);
    Ok(PreparedReferenceSurface {
        mesh,
        triangles,
        spatial,
        skipped_vertical_faces: 0,
    })
}

fn triangle_area_3d(triangle: [tri00t::Vertex; 3]) -> f64 {
    let a = glam::DVec3::new(triangle[0].x, triangle[0].y, triangle[0].z);
    let b = glam::DVec3::new(triangle[1].x, triangle[1].y, triangle[1].z);
    let c = glam::DVec3::new(triangle[2].x, triangle[2].y, triangle[2].z);
    (b - a).cross(c - a).length() * 0.5
}

pub(super) fn prepare_reference_surface_relaxed(
    reference: &tri00t::Triangulation,
) -> Result<PreparedReferenceSurface> {
    let vertices = reference.vertices();
    if reference.face_count() == 0 {
        anyhow::bail!("Reference topology contains no faces");
    }

    let xy_area_tolerance = reference_xy_overlap_area_tolerance(reference);
    let mut prepared_vertices = Vec::new();
    let mut prepared_faces = Vec::new();
    let mut triangles = Vec::new();
    let mut skipped_vertical_faces = 0usize;

    for face in reference.face_vertex_indices_iter() {
        let triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]];
        if triangle_xy_area(triangle).abs() <= xy_area_tolerance {
            skipped_vertical_faces += 1;
            continue;
        }
        let base = prepared_vertices.len() as u32;
        prepared_vertices.extend_from_slice(&triangle);
        prepared_faces.push([base, base + 1, base + 2]);
        triangles.push(triangle);
    }

    if triangles.is_empty() {
        anyhow::bail!("Reference topology must contain at least one non-vertical 2.5D face");
    }

    let mesh = tri00t::Triangulation::from_vertices_and_faces(prepared_vertices, prepared_faces);
    let spatial = crate::model::spatial::TriangleBvh::build(&mesh);
    Ok(PreparedReferenceSurface {
        mesh,
        triangles,
        spatial,
        skipped_vertical_faces,
    })
}

pub(super) fn triangle_xy_area(triangle: [tri00t::Vertex; 3]) -> f64 {
    let [a, b, c] = triangle;
    ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) * 0.5
}

pub(super) fn triangle_xy_bounds(triangle: [tri00t::Vertex; 3]) -> (glam::DVec2, glam::DVec2) {
    let mut min = glam::DVec2::splat(f64::INFINITY);
    let mut max = glam::DVec2::splat(f64::NEG_INFINITY);
    for point in triangle {
        let xy = glam::DVec2::new(point.x, point.y);
        min = min.min(xy);
        max = max.max(xy);
    }
    (min, max)
}

pub(super) fn triangle_intersection_xy(
    subject: [tri00t::Vertex; 3],
    clip: [tri00t::Vertex; 3],
) -> Vec<glam::DVec2> {
    let mut polygon: Vec<glam::DVec2> = subject
        .iter()
        .map(|point| glam::DVec2::new(point.x, point.y))
        .collect();
    let clip_points = clip.map(|point| glam::DVec2::new(point.x, point.y));
    let clip_ccw = triangle_xy_area(clip) > 0.0;

    for edge_index in 0..3 {
        let edge_a = clip_points[edge_index];
        let edge_b = clip_points[(edge_index + 1) % 3];
        polygon = clip_polygon_by_xy_edge(&polygon, edge_a, edge_b, clip_ccw);
        if polygon.len() < 3 {
            break;
        }
    }
    polygon
}

pub(super) fn polygon_area_xy(polygon: &[glam::DVec2]) -> f64 {
    if polygon.len() < 3 {
        return 0.0;
    }
    (0..polygon.len())
        .map(|index| {
            let a = polygon[index];
            let b = polygon[(index + 1) % polygon.len()];
            a.x * b.y - b.x * a.y
        })
        .sum::<f64>()
        * 0.5
}

pub(super) fn clip_target_triangle_to_reference_xy(
    target: [tri00t::Vertex; 3],
    reference: [tri00t::Vertex; 3],
) -> Vec<glam::DVec3> {
    let mut polygon: Vec<glam::DVec3> = target
        .iter()
        .map(|point| glam::DVec3::new(point.x, point.y, point.z))
        .collect();
    let reference_points = reference.map(|point| glam::DVec2::new(point.x, point.y));
    let reference_ccw = triangle_xy_area(reference) > 0.0;
    for edge_index in 0..3 {
        polygon = clip_polygon_3d_by_xy_edge(
            &polygon,
            reference_points[edge_index],
            reference_points[(edge_index + 1) % 3],
            reference_ccw,
        );
        if polygon.len() < 3 {
            break;
        }
    }
    polygon
}

pub(super) fn clip_polygon_3d_by_xy_edge(
    polygon: &[glam::DVec3],
    edge_a: glam::DVec2,
    edge_b: glam::DVec2,
    clip_ccw: bool,
) -> Vec<glam::DVec3> {
    if polygon.is_empty() {
        return Vec::new();
    }
    let edge = edge_b - edge_a;
    let signed_distance = |point: glam::DVec3| {
        let cross = edge.x * (point.y - edge_a.y) - edge.y * (point.x - edge_a.x);
        if clip_ccw { cross } else { -cross }
    };

    let mut output = Vec::new();
    let mut previous = polygon[polygon.len() - 1];
    let mut previous_distance = signed_distance(previous);
    let mut previous_inside = previous_distance >= -1e-10;
    for &current in polygon {
        let current_distance = signed_distance(current);
        let current_inside = current_distance >= -1e-10;
        if current_inside != previous_inside {
            let denominator = previous_distance - current_distance;
            if denominator.abs() > 1e-20 {
                output.push(
                    previous.lerp(current, (previous_distance / denominator).clamp(0.0, 1.0)),
                );
            }
        }
        if current_inside {
            output.push(current);
        }
        previous = current;
        previous_distance = current_distance;
        previous_inside = current_inside;
    }
    deduplicate_polygon_3d(output)
}

pub(super) fn deduplicate_polygon_3d(mut polygon: Vec<glam::DVec3>) -> Vec<glam::DVec3> {
    polygon.dedup_by(|a, b| a.distance_squared(*b) <= 1e-20);
    if polygon.len() > 1 && polygon[0].distance_squared(polygon[polygon.len() - 1]) <= 1e-20 {
        polygon.pop();
    }
    polygon
}

pub(super) fn clip_polygon_by_xy_edge(
    polygon: &[glam::DVec2],
    edge_a: glam::DVec2,
    edge_b: glam::DVec2,
    clip_ccw: bool,
) -> Vec<glam::DVec2> {
    if polygon.is_empty() {
        return Vec::new();
    }
    let edge = edge_b - edge_a;
    let signed_distance = |point: glam::DVec2| {
        let cross = edge.x * (point.y - edge_a.y) - edge.y * (point.x - edge_a.x);
        if clip_ccw { cross } else { -cross }
    };

    let mut output = Vec::new();
    let mut previous = *polygon.last().expect("polygon is non-empty");
    let mut previous_distance = signed_distance(previous);
    let mut previous_inside = previous_distance >= -1e-10;
    for &current in polygon {
        let current_distance = signed_distance(current);
        let current_inside = current_distance >= -1e-10;
        if current_inside != previous_inside {
            let denominator = previous_distance - current_distance;
            if denominator.abs() > 1e-20 {
                let t = (previous_distance / denominator).clamp(0.0, 1.0);
                output.push(previous.lerp(current, t));
            }
        }
        if current_inside {
            output.push(current);
        }
        previous = current;
        previous_distance = current_distance;
        previous_inside = current_inside;
    }
    deduplicate_polygon_xy(output)
}

pub(super) fn deduplicate_polygon_xy(mut polygon: Vec<glam::DVec2>) -> Vec<glam::DVec2> {
    polygon.dedup_by(|a, b| a.distance_squared(*b) <= 1e-20);
    if polygon.len() > 1
        && polygon[0].distance_squared(*polygon.last().expect("non-empty")) <= 1e-20
    {
        polygon.pop();
    }
    polygon
}

pub(super) fn clip_surface_polygon(
    polygon: Vec<SurfaceClipVertex>,
    side: TriSurfaceCutSide,
) -> Vec<SurfaceClipVertex> {
    if polygon.is_empty() {
        return polygon;
    }
    let retained = |delta: f64| match side {
        TriSurfaceCutSide::CutTop => delta <= 1e-9,
        TriSurfaceCutSide::CutBottom => delta >= -1e-9,
    };

    let mut output = Vec::new();
    let mut previous = *polygon.last().expect("polygon is non-empty");
    let mut previous_inside = retained(previous.height_delta);
    for current in polygon {
        let current_inside = retained(current.height_delta);
        if current_inside != previous_inside {
            let denominator = previous.height_delta - current.height_delta;
            if denominator.abs() > 1e-20 {
                let t = (previous.height_delta / denominator).clamp(0.0, 1.0);
                output.push(SurfaceClipVertex {
                    point: previous.point.lerp(current.point, t),
                    height_delta: 0.0,
                });
            }
        }
        if current_inside {
            output.push(current);
        }
        previous = current;
        previous_inside = current_inside;
    }
    output
}

pub(super) fn append_surface_clip_polygon(
    polygon: &[SurfaceClipVertex],
    vertices: &mut Vec<tri00t::Vertex>,
    faces: &mut Vec<[u32; 3]>,
) {
    if polygon.len() < 3 {
        return;
    }
    let base = vertices.len() as u32;
    vertices.extend(
        polygon
            .iter()
            .map(|vertex| tri00t::Vertex::new(vertex.point.x, vertex.point.y, vertex.point.z)),
    );
    for i in 1..polygon.len() - 1 {
        let face = [base, base + i as u32, base + i as u32 + 1];
        let a = vertices[face[0] as usize];
        let b = vertices[face[1] as usize];
        let c = vertices[face[2] as usize];
        let ab = glam::DVec3::new(b.x - a.x, b.y - a.y, b.z - a.z);
        let ac = glam::DVec3::new(c.x - a.x, c.y - a.y, c.z - a.z);
        if ab.cross(ac).length_squared() > 1e-20 {
            faces.push(face);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kept_area(verts: &[tri00t::Vertex], faces: &[[u32; 3]]) -> f64 {
        faces
            .iter()
            .map(|face| {
                let tri = [
                    verts[face[0] as usize],
                    verts[face[1] as usize],
                    verts[face[2] as usize],
                ];
                triangle_xy_area(tri).abs()
            })
            .sum()
    }

    fn covering_z_with_bvh(
        mesh: &tri00t::Triangulation,
        bvh: &crate::model::spatial::TriangleBvh,
        x: f64,
        y: f64,
    ) -> Option<f64> {
        let point = glam::DVec2::new(x, y);
        let vertices = mesh.vertices();
        bvh.xy_bounds_candidate_indices(mesh, point, point)
            .into_iter()
            .find_map(|index| {
                let face = mesh.face_vertex_indices(index)?;
                let triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]];
                point_in_triangle_bary_z_inclusive_for_test(x, y, triangle)
            })
    }

    fn point_in_triangle_bary_z_inclusive_for_test(
        x: f64,
        y: f64,
        v: [tri00t::Vertex; 3],
    ) -> Option<f64> {
        let denom = (v[1].y - v[2].y) * (v[0].x - v[2].x) + (v[2].x - v[1].x) * (v[0].y - v[2].y);
        if denom.abs() < 1e-12 {
            return None;
        }
        let w0 = ((v[1].y - v[2].y) * (x - v[2].x) + (v[2].x - v[1].x) * (y - v[2].y)) / denom;
        let w1 = ((v[2].y - v[0].y) * (x - v[2].x) + (v[0].x - v[2].x) * (y - v[2].y)) / denom;
        let w2 = 1.0 - w0 - w1;
        if w0 < -1e-9 || w1 < -1e-9 || w2 < -1e-9 {
            return None;
        }
        Some(w0 * v[0].z + w1 * v[1].z + w2 * v[2].z)
    }

    #[test]
    fn clip_topology_to_pit_shell_follows_the_true_contact_line() {
        // Pit shell: flat quad at z=0 over x:[0,10], y:[0,10].
        let pit_shell_mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![
                tri00t::Vertex::new(0.0, 0.0, 0.0),
                tri00t::Vertex::new(10.0, 0.0, 0.0),
                tri00t::Vertex::new(10.0, 10.0, 0.0),
                tri00t::Vertex::new(0.0, 10.0, 0.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        );
        // Topology: tilted plane z = x - 5, crossing the shell's z=0 exactly at x=5. Only the
        // half of the footprint where the topology is above the shell (x:[5,10]) is excavated;
        // where the shell floats above the topology (x:[0,5]) the topology is kept.
        let topology_mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![
                tri00t::Vertex::new(-10.0, -10.0, -15.0),
                tri00t::Vertex::new(20.0, -10.0, 15.0),
                tri00t::Vertex::new(20.0, 20.0, 15.0),
                tri00t::Vertex::new(-10.0, 20.0, -15.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        );

        let pit_shell = prepare_pit_shell_surface(&pit_shell_mesh).unwrap();
        let (verts, faces) = clip_topology_to_pit_shell(&topology_mesh, &pit_shell);

        assert!(!faces.is_empty());
        // 900 topology - 50 removed (rectangle x:[5,10], y:[0,10]).
        let total_area = kept_area(&verts, &faces);
        assert!(
            (total_area - 850.0).abs() < 1e-6,
            "expected kept area 850, got {total_area}"
        );

        // No kept vertex should fall strictly inside the removed rectangle x:[5,10],y:[0,10].
        for v in &verts {
            let inside_removed =
                v.x > 5.0 + 1e-9 && v.x < 10.0 - 1e-9 && v.y > 1e-9 && v.y < 10.0 - 1e-9;
            assert!(
                !inside_removed,
                "vertex ({}, {}) should have been clipped away",
                v.x, v.y
            );
        }
    }

    #[test]
    fn clip_topology_to_pit_shell_closed_solid_cap_does_not_cancel_footprint() {
        // Regression: a watertight solid whose up-facing crest cap and down-facing floor
        // project onto the same XY footprint with opposite winding. Without CCW
        // normalization these cancel in the union, leaving nothing removed.
        let r0 = tri00t::Vertex::new(0.0, 0.0, 0.0);
        let r1 = tri00t::Vertex::new(10.0, 0.0, 0.0);
        let r2 = tri00t::Vertex::new(10.0, 10.0, 0.0);
        let r3 = tri00t::Vertex::new(0.0, 10.0, 0.0);
        let f0 = tri00t::Vertex::new(0.0, 0.0, -10.0);
        let f1 = tri00t::Vertex::new(10.0, 0.0, -10.0);
        let f2 = tri00t::Vertex::new(10.0, 10.0, -10.0);
        let f3 = tri00t::Vertex::new(0.0, 10.0, -10.0);
        let pit_shell_mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![r0, r1, r2, r3, f0, f1, f2, f3],
            vec![
                // crest cap, up-facing (CCW in XY)
                [0, 1, 2],
                [0, 2, 3],
                // floor, down-facing (CW in XY)
                [4, 6, 5],
                [4, 7, 6],
                // walls
                [0, 5, 1],
                [0, 4, 5],
                [1, 6, 2],
                [1, 5, 6],
                [2, 7, 3],
                [2, 6, 7],
                [3, 4, 0],
                [3, 7, 4],
            ],
        );
        let topology_mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![
                tri00t::Vertex::new(-10.0, -10.0, 5.0),
                tri00t::Vertex::new(20.0, -10.0, 5.0),
                tri00t::Vertex::new(20.0, 20.0, 5.0),
                tri00t::Vertex::new(-10.0, 20.0, 5.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        );

        let pit_shell = prepare_pit_shell_surface(&pit_shell_mesh).unwrap();
        let (verts, faces) = clip_topology_to_pit_shell(&topology_mesh, &pit_shell);
        let total_area = kept_area(&verts, &faces);
        assert!(
            (total_area - 800.0).abs() < 1e-6,
            "closed-solid footprint (100) must be removed, got kept {total_area}"
        );
    }

    #[test]
    fn emit_earcut_polygon_drops_needle_slivers_but_keeps_real_area() {
        use geo::{Coord, LineString, Polygon as GeoPoly};
        let target = [
            tri00t::Vertex::new(0.0, 0.0, 0.0),
            tri00t::Vertex::new(10.0, 0.0, 0.0),
            tri00t::Vertex::new(0.0, 10.0, 0.0),
        ];

        // A 10m-long sliver only 0.1mm wide (below SEAM_TOLERANCE): the "floating edge"
        // artefact. It must not emit any faces.
        let needle = GeoPoly::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 1.0 },
                Coord { x: 10.0, y: 1.0 },
                Coord { x: 10.0, y: 1.0001 },
                Coord { x: 0.0, y: 1.0001 },
                Coord { x: 0.0, y: 1.0 },
            ]),
            vec![],
        );
        let mut verts = Vec::new();
        let mut faces = Vec::new();
        emit_earcut_polygon(&needle, target, &mut verts, &mut faces);
        assert!(
            faces.is_empty(),
            "needle sliver should emit no faces, got {}",
            faces.len()
        );

        // A real 4x4 square with a redundant near-collinear midpoint on one edge triangulates
        // to its full area with no needle triangles.
        let square = GeoPoly::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 2.0, y: 0.00005 }, // near-collinear midpoint on the bottom edge
                Coord { x: 4.0, y: 0.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 0.0, y: 4.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        );
        let mut verts = Vec::new();
        let mut faces = Vec::new();
        emit_earcut_polygon(&square, target, &mut verts, &mut faces);
        let area = kept_area(&verts, &faces);
        assert!((area - 16.0).abs() < 1e-3, "expected area ~16, got {area}");
        for face in &faces {
            let tri = [
                verts[face[0] as usize],
                verts[face[1] as usize],
                verts[face[2] as usize],
            ];
            let ar = triangle_xy_area(tri).abs() * 2.0;
            let longest = {
                let e = |p: tri00t::Vertex, q: tri00t::Vertex| (p.x - q.x).hypot(p.y - q.y);
                e(tri[0], tri[1])
                    .max(e(tri[1], tri[2]))
                    .max(e(tri[2], tri[0]))
            };
            assert!(
                ar / longest >= 1e-3,
                "no needle triangles should be emitted"
            );
        }
    }

    #[test]
    fn trim_triangles_inside_void_deletes_floaters_but_keeps_the_floating_band() {
        // Pit shell: crest at z=0 over [0,10]^2 sloping down to a floor at z=-10 over [3,7]^2.
        let r0 = tri00t::Vertex::new(0.0, 0.0, 0.0);
        let r1 = tri00t::Vertex::new(10.0, 0.0, 0.0);
        let r2 = tri00t::Vertex::new(10.0, 10.0, 0.0);
        let r3 = tri00t::Vertex::new(0.0, 10.0, 0.0);
        let f0 = tri00t::Vertex::new(3.0, 3.0, -10.0);
        let f1 = tri00t::Vertex::new(7.0, 3.0, -10.0);
        let f2 = tri00t::Vertex::new(7.0, 7.0, -10.0);
        let f3 = tri00t::Vertex::new(3.0, 7.0, -10.0);
        let pit_shell_mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![r0, r1, r2, r3, f0, f1, f2, f3],
            vec![
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
                [4, 5, 6],
                [4, 6, 7],
            ],
        );
        let pit_shell = prepare_pit_shell_surface(&pit_shell_mesh).unwrap();

        // Three planted triangles, each a small square split into two:
        //   A) at the pit centre, z=-5 — above the floor (-10): inside the void, must be dropped.
        //   B) near the rim, z=+2 — above the crest but the shell (0) is below +2 there: this is
        //      excavated too, must be dropped.
        //   C) outside the footprint entirely: must be kept.
        let planted = [
            ([4.0, 4.0], -5.0, false), // A: void
            ([1.0, 1.0], 2.0, false),  // B: excavated near rim
            ([20.0, 20.0], 5.0, true), // C: outside, keep
        ];
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        for ([x, y], z, _keep) in planted {
            let quad = [
                tri00t::Vertex::new(x, y, z),
                tri00t::Vertex::new(x + 1.0, y, z),
                tri00t::Vertex::new(x + 1.0, y + 1.0, z),
                tri00t::Vertex::new(x, y + 1.0, z),
            ];
            let base = vertices.len() as u32;
            vertices.extend_from_slice(&quad);
            faces.push([base, base + 1, base + 2]);
            faces.push([base, base + 2, base + 3]);
        }

        trim_triangles_inside_void(&pit_shell, &mut vertices, &mut faces);

        // Only the outside quad (2 triangles) should survive.
        assert_eq!(
            faces.len(),
            2,
            "only the outside-footprint quad should remain"
        );
        for face in &faces {
            let cx = (vertices[face[0] as usize].x
                + vertices[face[1] as usize].x
                + vertices[face[2] as usize].x)
                / 3.0;
            assert!(cx > 15.0, "a void/excavated triangle survived cleanup");
        }
    }

    #[test]
    fn trim_triangles_inside_void_clips_partial_lips() {
        let pit_shell_mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![
                tri00t::Vertex::new(0.0, 0.0, 0.0),
                tri00t::Vertex::new(10.0, 0.0, 0.0),
                tri00t::Vertex::new(10.0, 10.0, 0.0),
                tri00t::Vertex::new(0.0, 10.0, 0.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        );
        let pit_shell = prepare_pit_shell_surface(&pit_shell_mesh).unwrap();

        let mut vertices = vec![
            tri00t::Vertex::new(3.0, 2.0, 1.0),
            tri00t::Vertex::new(9.0, 2.0, -2.0),
            tri00t::Vertex::new(3.0, 8.0, -2.0),
        ];
        let mut faces = vec![[0, 1, 2]];

        trim_triangles_inside_void(&pit_shell, &mut vertices, &mut faces);

        let kept = kept_area(&vertices, &faces);
        assert!(
            (kept - 16.0).abs() < 1e-4,
            "expected only the lip corner clipped away, got kept_area={kept}"
        );
        assert!(
            vertices
                .iter()
                .all(|vertex| vertex.z <= VOID_CLEANUP_TOLERANCE + 1e-9),
            "cleanup left a vertex above the shell lower envelope"
        );
        assert!(
            faces.len() > 1,
            "partial cleanup should trim the triangle instead of dropping or keeping it whole"
        );
    }

    #[test]
    fn trim_triangles_inside_void_clips_interior_lips() {
        let pit_shell_mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![
                tri00t::Vertex::new(7.0, 1.0, -10.0),
                tri00t::Vertex::new(9.0, 1.0, -10.0),
                tri00t::Vertex::new(7.0, 3.0, -10.0),
            ],
            vec![[0, 1, 2]],
        );
        let pit_shell = prepare_pit_shell_surface(&pit_shell_mesh).unwrap();

        let mut vertices = vec![
            tri00t::Vertex::new(0.0, 0.0, -5.0),
            tri00t::Vertex::new(10.0, 0.0, -5.0),
            tri00t::Vertex::new(0.0, 10.0, -5.0),
        ];
        let mut faces = vec![[0, 1, 2]];

        trim_triangles_inside_void(&pit_shell, &mut vertices, &mut faces);

        let kept = kept_area(&vertices, &faces);
        assert!(
            (kept - 48.0).abs() < 1e-6,
            "expected the interior 2 m^2 lip clipped away, got kept_area={kept}"
        );
        for face in &faces {
            let triangle = [
                vertices[face[0] as usize],
                vertices[face[1] as usize],
                vertices[face[2] as usize],
            ];
            assert!(
                point_strictly_in_triangle_bary_z(7.25, 1.25, triangle).is_none(),
                "interior shell patch should not remain covered by topology"
            );
        }
    }

    #[test]
    #[ignore]
    fn reported_lip_point_not_covered_on_tutorial_data() {
        let topology = crate::model::formats::read_mesh("tutorial/example_data/topology.obj")
            .expect("topology.obj loads");
        let pit_surface =
            crate::model::formats::read_mesh("tutorial/example_data/tutorial_pit_surface.00t")
                .expect("pit surface loads");
        let pit_shell = prepare_pit_shell_surface(&pit_surface).unwrap();
        let (verts, faces) = clip_topology_to_pit_shell(&topology, &pit_shell);
        let x = 194_396.016;
        let y = 7_954_711.042;
        let shell_z = shell_lower_envelope_z_strict(&pit_shell, x, y);
        eprintln!(
            "output verts={} faces={} shell_z_at_reported={shell_z:?}",
            verts.len(),
            faces.len()
        );
        for (face_index, face) in faces.iter().enumerate() {
            let triangle = [
                verts[face[0] as usize],
                verts[face[1] as usize],
                verts[face[2] as usize],
            ];
            if let Some(z) = point_strictly_in_triangle_bary_z(x, y, triangle) {
                eprintln!(
                    "contains face={face_index} topo_z={z:.6} delta={:?} tri={triangle:?}",
                    shell_z.map(|s| z - s)
                );
                panic!("reported point is still covered by output topology");
            }
        }
    }

    #[test]
    #[ignore]
    fn scan_for_reported_hole_near_picked_triangle() {
        let topology = crate::model::formats::read_mesh("tutorial/example_data/topology.obj")
            .expect("topology.obj loads");
        let pit_surface =
            crate::model::formats::read_mesh("tutorial/example_data/tutorial_pit_surface.00t")
                .expect("pit surface loads");
        let pit_shell = prepare_pit_shell_surface(&pit_surface).unwrap();
        let (verts, faces) = clip_topology_to_pit_shell(&topology, &pit_shell);
        let output = tri00t::Triangulation::from_vertices_and_faces(verts, faces);
        let input_bvh = crate::model::spatial::TriangleBvh::build(&topology);
        let output_bvh = crate::model::spatial::TriangleBvh::build(&output);

        let anchor_x = 194_716.885;
        let anchor_y = 7_954_885.380;
        let mut missing = Vec::new();
        for yi in -5..=5 {
            for xi in -5..=5 {
                let x = anchor_x + xi as f64;
                let y = anchor_y + yi as f64;
                let Some(input_z) = covering_z_with_bvh(&topology, &input_bvh, x, y) else {
                    continue;
                };
                let shell_z = shell_lower_envelope_z_strict(&pit_shell, x, y);
                let should_keep = shell_z.is_none_or(|z| input_z <= z + 1e-4);
                if !should_keep {
                    continue;
                }
                let output_z = covering_z_with_bvh(&output, &output_bvh, x, y);
                if output_z.is_none() {
                    let dist = (x - anchor_x).hypot(y - anchor_y);
                    missing.push((dist, x, y, input_z, shell_z));
                }
            }
        }
        missing.sort_by(|a, b| a.0.total_cmp(&b.0));
        eprintln!(
            "missing_should_keep_samples={} nearest={:?}",
            missing.len(),
            missing.iter().take(20).collect::<Vec<_>>()
        );
        assert!(
            missing.is_empty(),
            "found should-keep samples missing from output"
        );
    }

    #[test]
    fn clip_topology_to_pit_shell_is_robust_at_utm_coordinates() {
        // Real mine data lives in UTM-scale coordinates; the boolean ops must still work.
        let ox = 194_500.0;
        let oy = 7_954_800.0;
        let v = |x: f64, y: f64, z: f64| tri00t::Vertex::new(ox + x, oy + y, z);
        let pit_shell_mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![
                v(0.0, 0.0, 250.0),
                v(10.0, 0.0, 250.0),
                v(10.0, 10.0, 250.0),
                v(0.0, 10.0, 250.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        );
        let topology_mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![
                v(-10.0, -10.0, 260.0),
                v(20.0, -10.0, 260.0),
                v(20.0, 20.0, 260.0),
                v(-10.0, 20.0, 260.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        );

        let pit_shell = prepare_pit_shell_surface(&pit_shell_mesh).unwrap();
        let (verts, faces) = clip_topology_to_pit_shell(&topology_mesh, &pit_shell);
        let total_area = kept_area(&verts, &faces);
        assert!(
            (total_area - 800.0).abs() < 1e-3,
            "expected kept area 800 at UTM scale, got {total_area}"
        );
    }

    #[test]
    fn clip_topology_to_pit_shell_passes_through_when_no_overlap() {
        let pit_shell_mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![
                tri00t::Vertex::new(100.0, 100.0, 0.0),
                tri00t::Vertex::new(110.0, 100.0, 0.0),
                tri00t::Vertex::new(110.0, 110.0, 0.0),
                tri00t::Vertex::new(100.0, 110.0, 0.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        );
        let topology_mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![
                tri00t::Vertex::new(0.0, 0.0, 5.0),
                tri00t::Vertex::new(10.0, 0.0, 5.0),
                tri00t::Vertex::new(0.0, 10.0, 5.0),
            ],
            vec![[0, 1, 2]],
        );

        let pit_shell = prepare_pit_shell_surface(&pit_shell_mesh).unwrap();
        let (verts, faces) = clip_topology_to_pit_shell(&topology_mesh, &pit_shell);

        assert_eq!(faces.len(), 1);
        assert_eq!(verts.len(), 3);
        assert_eq!(verts[0], tri00t::Vertex::new(0.0, 0.0, 5.0));
        assert_eq!(verts[1], tri00t::Vertex::new(10.0, 0.0, 5.0));
        assert_eq!(verts[2], tri00t::Vertex::new(0.0, 10.0, 5.0));
    }

    #[test]
    fn clip_topology_to_pit_shell_follows_walls_down_to_the_floor() {
        // Pit shell: a frustum — crest rim at z=0 over x/y:[0,10], sloping walls down to a
        // flat floor at z=-10 over x/y:[3,7]. This is the case prepare_reference_surface_relaxed
        // would break: the near-vertical walls have almost no XY-projected area and used to be
        // dropped as "degenerate", leaving only the rim and floor as candidates.
        let r0 = tri00t::Vertex::new(0.0, 0.0, 0.0);
        let r1 = tri00t::Vertex::new(10.0, 0.0, 0.0);
        let r2 = tri00t::Vertex::new(10.0, 10.0, 0.0);
        let r3 = tri00t::Vertex::new(0.0, 10.0, 0.0);
        let f0 = tri00t::Vertex::new(3.0, 3.0, -10.0);
        let f1 = tri00t::Vertex::new(7.0, 3.0, -10.0);
        let f2 = tri00t::Vertex::new(7.0, 7.0, -10.0);
        let f3 = tri00t::Vertex::new(3.0, 7.0, -10.0);
        let pit_shell_mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![r0, r1, r2, r3, f0, f1, f2, f3],
            vec![
                [0, 1, 5],
                [0, 5, 4], // front wall
                [1, 2, 6],
                [1, 6, 5], // right wall
                [2, 3, 7],
                [2, 7, 6], // back wall
                [3, 0, 4],
                [3, 4, 7], // left wall
                [4, 5, 6],
                [4, 6, 7], // floor
            ],
        );

        // Topology: flat plane at z=-5, well above the floor (-10) but below the rim (0).
        let topology_mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![
                tri00t::Vertex::new(-10.0, -10.0, -5.0),
                tri00t::Vertex::new(20.0, -10.0, -5.0),
                tri00t::Vertex::new(20.0, 20.0, -5.0),
                tri00t::Vertex::new(-10.0, 20.0, -5.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        );

        let pit_shell = prepare_pit_shell_surface(&pit_shell_mesh).unwrap();
        let (verts, faces) = clip_topology_to_pit_shell(&topology_mesh, &pit_shell);
        assert!(!faces.is_empty());

        // The cut follows the true contact line down the walls: only the inner part of the
        // footprint, where the sloping wall/floor has dropped below the z=-5 topology, is
        // excavated. The outer wall band (still above z=-5) and everything outside the crest
        // is kept, so strictly more than 900 - 100 = 800 survives and strictly less than 900.
        let kept = kept_area(&verts, &faces);
        assert!(
            kept > 800.0 + 1.0,
            "expected partial retention within the pit footprint (outer wall band above the \
             topology should survive), got kept_area={kept}"
        );
        assert!(
            kept < 900.0 - 1.0,
            "expected the excavated inner region removed, got kept_area={kept}"
        );
    }

    #[test]
    fn validate_reference_surface_reports_projected_overlap_details() {
        let mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![
                tri00t::Vertex::new(0.0, 0.0, 0.0),
                tri00t::Vertex::new(2.0, 0.0, 0.0),
                tri00t::Vertex::new(0.0, 2.0, 0.0),
                tri00t::Vertex::new(0.5, 0.5, 1.0),
                tri00t::Vertex::new(2.5, 0.5, 1.0),
                tri00t::Vertex::new(0.5, 2.5, 1.0),
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        );

        let error = validate_reference_surface(&mesh).unwrap_err().to_string();

        assert!(error.contains("triangles 0 and 1"));
        assert!(error.contains("Z difference"));
    }

    #[test]
    fn prepare_reference_surface_skips_zero_xy_faces() {
        let mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![
                tri00t::Vertex::new(0.0, 0.0, 0.0),
                tri00t::Vertex::new(2.0, 0.0, 0.0),
                tri00t::Vertex::new(0.0, 2.0, 0.0),
                tri00t::Vertex::new(3.0, 0.0, 0.0),
                tri00t::Vertex::new(3.0, 1.0, 1.0),
                tri00t::Vertex::new(3.0, 2.0, 2.0),
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        );

        let prepared = prepare_reference_surface_relaxed(&mesh).unwrap();

        assert_eq!(prepared.triangles.len(), 1);
        assert_eq!(prepared.skipped_vertical_faces, 1);
    }
}
