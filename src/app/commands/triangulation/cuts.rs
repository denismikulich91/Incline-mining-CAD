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
        let envelope = build_pit_shell_lower_envelope(&pit_shell)?;
        let (new_verts, new_faces) = clip_topology_to_pit_shell(&topology_mesh, &envelope);

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
        let flat: Vec<[f64; 2]> = exterior[..n].iter().map(|c| [c.x, c.y]).collect();
        let mut indices: Vec<usize> = Vec::new();
        earcut::Earcut::new().earcut(flat.iter().copied(), &[], &mut indices);
        for tri_idx in indices.chunks_exact(3) {
            let make_vert = |i: usize| {
                let [x, y] = flat[i];
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
/// `envelope` must be the shell's lower envelope (see `build_pit_shell_lower_envelope`):
/// a single-valued 2.5D surface giving, at every XY point of the shell's footprint, the
/// lowest shell surface there. A topology point is excavated exactly when it lies at or
/// above that envelope. Because the envelope cells tile the plane, each topology triangle
/// is rebuilt cell by cell: the overlap with an open cell is kept whole, and the overlap
/// with a covered cell keeps only the part below the cell's plane. Every fragment is a
/// convex polygon (a triangle–triangle overlap split by one half-plane), so emission is a
/// simple fan — no polygon booleans or ear-cutting, whose floating-point failure modes on
/// production data motivated this design. Where the shell floats *above* the terrain —
/// e.g. a flat design crest standing over undulating ground near the rim — the topology is
/// below the envelope and is **kept**, so the cut is flush with the real contact line
/// instead of the shell's widest XY extent. Because the lower envelope of a watertight
/// solid is its floor, a flat crest cap never forces removal on its own. Triangles that
/// touch no excavated region pass through unchanged and unfragmented.
pub(super) fn clip_topology_to_pit_shell(
    topology: &tri00t::Triangulation,
    envelope: &PitShellLowerEnvelope,
) -> (Vec<tri00t::Vertex>, Vec<[u32; 3]>) {
    use rayon::prelude::*;

    let topology_vertices = topology.vertices();
    let topology_faces: Vec<[usize; 3]> = topology.face_vertex_indices_iter().collect();

    let pass_through = |target: &[tri00t::Vertex; 3],
                        vertices: &mut Vec<tri00t::Vertex>,
                        faces: &mut Vec<[u32; 3]>| {
        let base = vertices.len() as u32;
        vertices.extend_from_slice(target);
        faces.push([base, base + 1, base + 2]);
    };

    let task_count = rayon::current_num_threads().saturating_mul(4).max(1);
    let chunk_size = topology_faces.len().div_ceil(task_count).max(1);
    let partials: Vec<(Vec<tri00t::Vertex>, Vec<[u32; 3]>)> = topology_faces
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut chunk_vertices = Vec::new();
            let mut chunk_faces = Vec::new();
            let mut candidate_stack = Vec::new();
            let mut fragments: Vec<Vec<SurfaceClipVertex>> = Vec::new();

            for face in chunk.iter().copied() {
                let target = [
                    topology_vertices[face[0]],
                    topology_vertices[face[1]],
                    topology_vertices[face[2]],
                ];
                let bounds = triangle_xy_bounds(target);

                // Fragment the triangle across the overlapping cells. `height_delta` is linear
                // over each fragment, so a positive delta at some overlap vertex is exactly the
                // condition for that cell to excavate part of the triangle.
                fragments.clear();
                let mut saw_candidate = false;
                let mut any_excavated = false;
                envelope
                    .spatial
                    .for_each_xy_bounds_candidate_index_with_stack(
                        bounds.0,
                        bounds.1,
                        &mut candidate_stack,
                        |index| {
                            saw_candidate = true;
                            let cell = envelope.triangles[index];
                            let overlap = clip_target_triangle_to_reference_xy(target, cell);
                            if overlap.len() < 3 {
                                return;
                            }
                            if !envelope.covered[index] {
                                fragments.push(
                                    overlap
                                        .into_iter()
                                        .map(|point| SurfaceClipVertex {
                                            point,
                                            height_delta: 0.0,
                                        })
                                        .collect(),
                                );
                                return;
                            }
                            let polygon: Vec<SurfaceClipVertex> = overlap
                                .into_iter()
                                .map(|point| {
                                    let reference_z = bary_z(point.x, point.y, cell);
                                    SurfaceClipVertex {
                                        point,
                                        height_delta: point.z - reference_z,
                                    }
                                })
                                .collect();
                            if polygon.iter().any(|vertex| vertex.height_delta > 1e-9) {
                                any_excavated = true;
                            }
                            // Keep the part below the envelope (height_delta <= 0): the shell
                            // floats above the terrain there and removes nothing.
                            fragments
                                .push(clip_surface_polygon(polygon, TriSurfaceCutSide::CutTop));
                        },
                    );

                if !saw_candidate || !any_excavated {
                    // Beyond even the envelope's padding, or nothing under this triangle is
                    // excavated (open cells only, or the shell floats above the terrain
                    // everywhere here): keep it whole and unfragmented.
                    pass_through(&target, &mut chunk_vertices, &mut chunk_faces);
                    continue;
                }

                for fragment in &fragments {
                    append_surface_clip_polygon(fragment, &mut chunk_vertices, &mut chunk_faces);
                }
            }

            (chunk_vertices, chunk_faces)
        })
        .collect();

    let mut output_vertices =
        Vec::with_capacity(partials.iter().map(|(vertices, _)| vertices.len()).sum());
    let mut output_faces = Vec::with_capacity(partials.iter().map(|(_, faces)| faces.len()).sum());
    for (vertices, faces) in partials {
        let base = output_vertices.len() as u32;
        output_vertices.extend(vertices);
        output_faces.extend(
            faces
                .into_iter()
                .map(|face| [base + face[0], base + face[1], base + face[2]]),
        );
    }

    (output_vertices, output_faces)
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

pub(super) fn reference_xy_overlap_area_tolerance(reference: &tri00t::Triangulation) -> f64 {
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

/// Prepare a pit shell mesh as input to `build_pit_shell_lower_envelope`. Unlike
/// `prepare_reference_surface_relaxed`, this keeps vertical and near-vertical wall faces —
/// they have little or no XY-projected area, but their edges are exactly the boundaries the
/// envelope arrangement must respect (bench crests, wall toes), and their surfaces define
/// the envelope over wall bands. Only genuinely degenerate faces (zero area in 3D —
/// duplicate or collinear points) are dropped.
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

pub(super) fn triangle_area_3d(triangle: [tri00t::Vertex; 3]) -> f64 {
    let a = glam::DVec3::new(triangle[0].x, triangle[0].y, triangle[0].z);
    let b = glam::DVec3::new(triangle[1].x, triangle[1].y, triangle[1].z);
    let c = glam::DVec3::new(triangle[2].x, triangle[2].y, triangle[2].z);
    (b - a).cross(c - a).length() * 0.5
}

/// The pit shell's lower envelope: a triangulated planar subdivision that tiles all of XY.
/// Cells with `covered[i] == true` carry the lowest shell surface over their footprint in
/// their vertex Z; open cells (outside the shell footprint, or padding around it) never
/// remove topology and carry no meaningful Z.
#[derive(Debug)]
pub(super) struct PitShellLowerEnvelope {
    #[cfg_attr(not(test), allow(dead_code))]
    mesh: tri00t::Triangulation,
    triangles: Vec<[tri00t::Vertex; 3]>,
    covered: Vec<bool>,
    spatial: crate::model::spatial::TriangleBvh,
}

/// How far beyond the shell bounds the envelope's open padding cells extend (metres).
/// Larger than any real survey extent, so every topology triangle lands inside the padded
/// triangulation and is handled by the same per-cell path.
pub(super) const ENVELOPE_PADDING: f64 = 1.0e7;

/// Build the pit shell's lower envelope: the single-valued 2.5D surface giving, at every
/// XY point of the shell's footprint, the lowest shell surface there. This reduces the
/// multi-valued shell (walls, benches, watertight solids) to the one surface that decides
/// excavation — a topology point is excavated exactly when it lies above the lower envelope.
///
/// The construction is exact, not sampled. Every shell triangle edge is projected to XY and
/// inserted as a CDT constraint (splitting where edges cross), so no output cell interior
/// crosses the projected boundary of any shell face. Within one cell the set of covering
/// shell faces is therefore constant, and — because a valid shell does not self-intersect,
/// so faces overlapping in XY never cross in 3D — their vertical order is constant too.
/// One interior sample per cell then identifies the lowest covering face exactly, and that
/// face's plane supplies the cell's corner elevations. Exactly vertical faces contribute
/// their edges as constraints (bench crests and wall toes land on cell boundaries, where
/// the envelope legitimately jumps) but never supply elevations. Cells with no covering
/// face — outside the shell's (possibly concave) footprint, or in the far padding — are
/// kept as open cells so the envelope tiles the whole plane.
pub(super) fn build_pit_shell_lower_envelope(
    pit_shell: &PreparedReferenceSurface,
) -> Result<PitShellLowerEnvelope> {
    use rayon::prelude::*;
    use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation as _};

    // Work in coordinates local to the shell's minimum corner: at real-world (UTM-scale)
    // magnitudes the absolute coordinates would erode the precision of the CDT's
    // constraint-splitting intersection points.
    let bounds = pit_shell.mesh.bounds();
    let origin = glam::DVec2::new(bounds.min.x, bounds.min.y);
    let extent = glam::DVec2::new(bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y);

    // Dedup projected points exactly (by bit pattern, with -0.0 normalised so it cannot
    // alias 0.0 under spade's positional dedup) and collect each undirected edge once, so
    // the CDT can be bulk-loaded instead of point-located per triangle corner — the
    // incremental build dominated the whole cut on large shells. Only edges the bulk
    // loader rejects as conflicting (crossing another constraint in projection, i.e.
    // walls over benches) go through the splitting insert.
    let normalized = |value: f64| if value == 0.0 { 0.0 } else { value };
    let mut points: Vec<Point2<f64>> = Vec::with_capacity(pit_shell.triangles.len() + 4);
    for (corner_x, corner_y) in [
        (-ENVELOPE_PADDING, -ENVELOPE_PADDING),
        (extent.x + ENVELOPE_PADDING, -ENVELOPE_PADDING),
        (extent.x + ENVELOPE_PADDING, extent.y + ENVELOPE_PADDING),
        (-ENVELOPE_PADDING, extent.y + ENVELOPE_PADDING),
    ] {
        points.push(Point2::new(corner_x, corner_y));
    }
    let mut point_indices: HashMap<(u64, u64), usize> = HashMap::new();
    let mut edges: HashSet<(usize, usize)> = HashSet::new();
    for triangle in &pit_shell.triangles {
        let mut indices = [0usize; 3];
        for (slot, point) in triangle.iter().enumerate() {
            if !point.x.is_finite() || !point.y.is_finite() {
                anyhow::bail!("Pit shell contains non-finite coordinates");
            }
            let local_x = normalized(point.x - origin.x);
            let local_y = normalized(point.y - origin.y);
            indices[slot] = *point_indices
                .entry((local_x.to_bits(), local_y.to_bits()))
                .or_insert_with(|| {
                    points.push(Point2::new(local_x, local_y));
                    points.len() - 1
                });
        }
        for i in 0..3 {
            let a = indices[i];
            let b = indices[(i + 1) % 3];
            if a != b {
                edges.insert(if a < b { (a, b) } else { (b, a) });
            }
        }
    }
    let mut edge_list: Vec<[usize; 2]> = edges.into_iter().map(|(a, b)| [a, b]).collect();
    edge_list.sort_unstable();

    let point_count = points.len();
    let mut conflicting_edges: Vec<[usize; 2]> = Vec::new();
    let mut cdt: ConstrainedDelaunayTriangulation<Point2<f64>> =
        ConstrainedDelaunayTriangulation::try_bulk_load_cdt(points, edge_list, |edge| {
            conflicting_edges.push(edge)
        })
        .map_err(|error| anyhow::anyhow!("Pit shell CDT bulk load failed: {error:?}"))?;
    // Our exact dedup means spade removed no duplicates, so bulk-load vertex indices
    // (which the conflict callback reports) still match ours; anything else would make
    // the splitting inserts below connect the wrong points.
    if cdt.num_vertices() != point_count {
        anyhow::bail!("Pit shell CDT dropped vertices unexpectedly during bulk load");
    }
    for [a, b] in conflicting_edges {
        cdt.add_constraint_and_split(
            spade::handles::FixedVertexHandle::from_index(a),
            spade::handles::FixedVertexHandle::from_index(b),
            |point| point,
        );
    }

    let cells: Vec<[glam::DVec2; 3]> = cdt
        .inner_faces()
        .map(|face| {
            face.vertices().map(|vertex| {
                let position = vertex.position();
                glam::DVec2::new(position.x + origin.x, position.y + origin.y)
            })
        })
        .collect();
    let classified: Vec<([tri00t::Vertex; 3], bool)> = cells
        .par_iter()
        .fold(
            || (Vec::new(), Vec::new()),
            |(mut candidate_stack, mut output), corners| {
                let centroid = (corners[0] + corners[1] + corners[2]) / 3.0;
                let lowest =
                    lowest_covering_shell_triangle(pit_shell, centroid, &mut candidate_stack);
                let cell = corners.map(|corner| {
                    let z = lowest.map_or(0.0, |lowest| bary_z(corner.x, corner.y, lowest));
                    tri00t::Vertex::new(corner.x, corner.y, z)
                });
                if triangle_xy_area(cell).abs() > 1e-12 {
                    output.push((cell, lowest.is_some()));
                }
                (candidate_stack, output)
            },
        )
        .map(|(_, output)| output)
        .reduce(Vec::new, |mut left, right| {
            left.extend(right);
            left
        });

    let mut vertices = Vec::with_capacity(classified.len() * 3);
    let mut faces = Vec::with_capacity(classified.len());
    let mut triangles = Vec::with_capacity(classified.len());
    let mut covered = Vec::with_capacity(classified.len());
    for (cell, is_covered) in classified {
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&cell);
        faces.push([base, base + 1, base + 2]);
        triangles.push(cell);
        covered.push(is_covered);
    }

    if !covered.iter().any(|&is_covered| is_covered) {
        anyhow::bail!("Pit shell has no XY footprint to build a lower envelope from");
    }

    let mesh = tri00t::Triangulation::from_vertices_and_faces(vertices, faces);
    let spatial = crate::model::spatial::TriangleBvh::build(&mesh);
    Ok(PitShellLowerEnvelope {
        mesh,
        triangles,
        covered,
        spatial,
    })
}

/// The lowest shell face covering `point` in XY. Inclusive of face edges; exactly vertical
/// faces (degenerate XY projection) never match.
fn lowest_covering_shell_triangle(
    pit_shell: &PreparedReferenceSurface,
    point: glam::DVec2,
    candidate_stack: &mut Vec<usize>,
) -> Option<[tri00t::Vertex; 3]> {
    let mut lowest: Option<(f64, [tri00t::Vertex; 3])> = None;
    pit_shell
        .spatial
        .for_each_xy_bounds_candidate_index_with_stack(point, point, candidate_stack, |index| {
            let triangle = pit_shell.triangles[index];
            if let Some(z) = point_in_triangle_bary_z(point.x, point.y, triangle)
                && lowest.is_none_or(|(lowest_z, _)| z < lowest_z)
            {
                lowest = Some((z, triangle));
            }
        });
    lowest.map(|(_, triangle)| triangle)
}

/// Barycentric Z of `(x, y)` on triangle `v`, inclusive of XY edges. `None` outside the
/// triangle or when its XY projection is degenerate.
pub(super) fn point_in_triangle_bary_z(x: f64, y: f64, v: [tri00t::Vertex; 3]) -> Option<f64> {
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
    if !triangles_overlap_xy_sat(target, reference) {
        return Vec::new();
    }
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

#[inline(always)]
fn triangles_overlap_xy_sat(a: [tri00t::Vertex; 3], b: [tri00t::Vertex; 3]) -> bool {
    for triangle in [a, b] {
        for edge_index in 0..3 {
            if separates_triangles_xy(a, b, triangle[edge_index], triangle[(edge_index + 1) % 3]) {
                return false;
            }
        }
    }

    true
}

#[inline(always)]
fn separates_triangles_xy(
    a: [tri00t::Vertex; 3],
    b: [tri00t::Vertex; 3],
    edge_a: tri00t::Vertex,
    edge_b: tri00t::Vertex,
) -> bool {
    let edge_x = edge_b.x - edge_a.x;
    let edge_y = edge_b.y - edge_a.y;
    let edge_len_sq = edge_x * edge_x + edge_y * edge_y;
    if edge_len_sq <= f64::EPSILON {
        return false;
    }

    let (a_min, a_max) = project_triangle_onto_edge_normal_xy(a, edge_a, edge_x, edge_y);
    let (b_min, b_max) = project_triangle_onto_edge_normal_xy(b, edge_a, edge_x, edge_y);
    let gap = if a_max < b_min {
        b_min - a_max
    } else if b_max < a_min {
        a_min - b_max
    } else {
        return false;
    };

    const XY_TOL_SQ: f64 = crate::model::kernel::XY_TOL * crate::model::kernel::XY_TOL;
    gap * gap > XY_TOL_SQ * edge_len_sq
}

#[inline(always)]
fn project_triangle_onto_edge_normal_xy(
    triangle: [tri00t::Vertex; 3],
    edge_a: tri00t::Vertex,
    edge_x: f64,
    edge_y: f64,
) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for point in triangle {
        let projected = edge_x * (point.y - edge_a.y) - edge_y * (point.x - edge_a.x);
        min = min.min(projected);
        max = max.max(projected);
    }
    (min, max)
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
    // Signed distance in metres (scale-independent of edge length), so the
    // on-boundary tolerance means the same thing for every clip edge.
    let signed_distance = |point: glam::DVec3| {
        let d = crate::model::kernel::signed_distance_to_line(
            glam::DVec2::new(point.x, point.y),
            edge_a,
            edge_b,
        );
        if clip_ccw { d } else { -d }
    };
    const INSIDE_TOL: f64 = crate::model::kernel::XY_TOL;

    let mut output = Vec::new();
    let mut previous = polygon[polygon.len() - 1];
    let mut previous_distance = signed_distance(previous);
    let mut previous_inside = previous_distance >= -INSIDE_TOL;
    for &current in polygon {
        let current_distance = signed_distance(current);
        let current_inside = current_distance >= -INSIDE_TOL;
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
    // Signed distance in metres (scale-independent of edge length), so the
    // on-boundary tolerance means the same thing for every clip edge.
    let signed_distance = |point: glam::DVec2| {
        let d = crate::model::kernel::signed_distance_to_line(point, edge_a, edge_b);
        if clip_ccw { d } else { -d }
    };
    const INSIDE_TOL: f64 = crate::model::kernel::XY_TOL;

    let mut output = Vec::new();
    let mut previous = *polygon.last().expect("polygon is non-empty");
    let mut previous_distance = signed_distance(previous);
    let mut previous_inside = previous_distance >= -INSIDE_TOL;
    for &current in polygon {
        let current_distance = signed_distance(current);
        let current_inside = current_distance >= -INSIDE_TOL;
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
                point_in_triangle_bary_z(x, y, triangle)
            })
    }

    fn envelope_of(pit_shell_mesh: &tri00t::Triangulation) -> PitShellLowerEnvelope {
        build_pit_shell_lower_envelope(&prepare_pit_shell_surface(pit_shell_mesh).unwrap()).unwrap()
    }

    fn envelope_z(envelope: &PitShellLowerEnvelope, x: f64, y: f64) -> Option<f64> {
        let point = glam::DVec2::new(x, y);
        envelope
            .spatial
            .xy_bounds_candidate_indices(&envelope.mesh, point, point)
            .into_iter()
            .filter(|&index| envelope.covered[index])
            .find_map(|index| point_in_triangle_bary_z(x, y, envelope.triangles[index]))
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

        let envelope = envelope_of(&pit_shell_mesh);
        let (verts, faces) = clip_topology_to_pit_shell(&topology_mesh, &envelope);

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
        // A watertight solid whose up-facing crest cap and down-facing floor project onto
        // the same XY footprint. The lower envelope must resolve to the floor, and the
        // topology above it must be removed over the full footprint.
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

        let envelope = envelope_of(&pit_shell_mesh);
        let (verts, faces) = clip_topology_to_pit_shell(&topology_mesh, &envelope);
        let total_area = kept_area(&verts, &faces);
        assert!(
            (total_area - 800.0).abs() < 1e-6,
            "closed-solid footprint (100) must be removed, got kept {total_area}"
        );
    }

    #[test]
    fn lower_envelope_of_closed_solid_is_the_floor() {
        // Same watertight box as above: crest cap at z=0, floor at z=-10 over [0,10]^2,
        // exactly vertical walls. The envelope must be the floor everywhere inside the
        // footprint and absent outside it.
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
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
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

        let envelope = envelope_of(&pit_shell_mesh);
        for (x, y) in [(5.0, 5.0), (0.5, 0.5), (9.5, 9.5), (5.0, 0.5)] {
            let z = envelope_z(&envelope, x, y);
            assert!(
                z.is_some_and(|z| (z - -10.0).abs() < 1e-9),
                "envelope at ({x}, {y}) should be the floor (-10), got {z:?}"
            );
        }
        assert!(
            envelope_z(&envelope, -1.0, 5.0).is_none(),
            "envelope should not extend outside the shell footprint"
        );
    }

    #[test]
    fn lower_envelope_of_frustum_follows_walls_down_to_the_floor() {
        // Frustum: crest rim at z=0 over [0,10]^2 sloping down to a floor at z=-10 over
        // [3,7]^2 (walls only, no crest cap). The envelope is the wall surface over the
        // wall band and the floor inside it.
        let pit_shell_mesh = frustum_pit_shell();
        let envelope = envelope_of(&pit_shell_mesh);

        // Floor.
        let z = envelope_z(&envelope, 5.0, 5.0);
        assert!(
            z.is_some_and(|z| (z - -10.0).abs() < 1e-9),
            "envelope at the pit centre should be the floor (-10), got {z:?}"
        );
        // Front wall band: the wall plane there is z = -10 * (y / 3).
        let z = envelope_z(&envelope, 5.0, 1.5);
        assert!(
            z.is_some_and(|z| (z - -5.0).abs() < 1e-9),
            "envelope on the front wall band should be -5, got {z:?}"
        );
        // Outside the rim.
        assert!(envelope_z(&envelope, 12.0, 5.0).is_none());
    }

    /// Frustum pit shell: crest rim at z=0 over [0,10]^2 sloping down to a floor at z=-10
    /// over [3,7]^2 (walls + floor, no crest cap).
    fn frustum_pit_shell() -> tri00t::Triangulation {
        let r0 = tri00t::Vertex::new(0.0, 0.0, 0.0);
        let r1 = tri00t::Vertex::new(10.0, 0.0, 0.0);
        let r2 = tri00t::Vertex::new(10.0, 10.0, 0.0);
        let r3 = tri00t::Vertex::new(0.0, 10.0, 0.0);
        let f0 = tri00t::Vertex::new(3.0, 3.0, -10.0);
        let f1 = tri00t::Vertex::new(7.0, 3.0, -10.0);
        let f2 = tri00t::Vertex::new(7.0, 7.0, -10.0);
        let f3 = tri00t::Vertex::new(3.0, 7.0, -10.0);
        tri00t::Triangulation::from_vertices_and_faces(
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
        )
    }

    #[test]
    fn clip_topology_to_pit_shell_removes_floaters_but_keeps_outside_geometry() {
        let envelope = envelope_of(&frustum_pit_shell());

        // Three planted quads (two triangles each):
        //   A) at the pit centre, z=-5 — above the floor (-10): inside the void, must be dropped.
        //   B) near the rim, z=+2 — above the wall band there: excavated too, must be dropped.
        //   C) outside the footprint entirely: must be kept.
        let planted = [
            ([4.0, 4.0], -5.0),  // A: void
            ([1.0, 1.0], 2.0),   // B: excavated near rim
            ([20.0, 20.0], 5.0), // C: outside, keep
        ];
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        for ([x, y], z) in planted {
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
        let topology_mesh = tri00t::Triangulation::from_vertices_and_faces(vertices, faces);

        let (verts, faces) = clip_topology_to_pit_shell(&topology_mesh, &envelope);

        // Only the outside quad (2 triangles) should survive.
        assert_eq!(
            faces.len(),
            2,
            "only the outside-footprint quad should remain"
        );
        for face in &faces {
            let cx =
                (verts[face[0] as usize].x + verts[face[1] as usize].x + verts[face[2] as usize].x)
                    / 3.0;
            assert!(cx > 15.0, "a void/excavated triangle survived the cut");
        }
    }

    #[test]
    fn clip_topology_to_pit_shell_clips_partial_lips() {
        let pit_shell_mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![
                tri00t::Vertex::new(0.0, 0.0, 0.0),
                tri00t::Vertex::new(10.0, 0.0, 0.0),
                tri00t::Vertex::new(10.0, 10.0, 0.0),
                tri00t::Vertex::new(0.0, 10.0, 0.0),
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        );
        let envelope = envelope_of(&pit_shell_mesh);

        // One topology triangle whose corner near (3, 2) pokes above the flat shell at z=0:
        // only that 2 m^2 lip is excavated.
        let topology_mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![
                tri00t::Vertex::new(3.0, 2.0, 1.0),
                tri00t::Vertex::new(9.0, 2.0, -2.0),
                tri00t::Vertex::new(3.0, 8.0, -2.0),
            ],
            vec![[0, 1, 2]],
        );

        let (verts, faces) = clip_topology_to_pit_shell(&topology_mesh, &envelope);

        let kept = kept_area(&verts, &faces);
        assert!(
            (kept - 16.0).abs() < 1e-4,
            "expected only the lip corner clipped away, got kept_area={kept}"
        );
        assert!(
            verts.iter().all(|vertex| vertex.z <= 1e-9),
            "cut left a vertex above the shell lower envelope"
        );
        assert!(
            faces.len() > 1,
            "the cut should trim the triangle instead of dropping or keeping it whole"
        );
    }

    #[test]
    fn clip_topology_to_pit_shell_clips_interior_lips() {
        // A small shell patch strictly inside one big topology triangle: the removed region
        // is a hole in the triangle's interior.
        let pit_shell_mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![
                tri00t::Vertex::new(7.0, 1.0, -10.0),
                tri00t::Vertex::new(9.0, 1.0, -10.0),
                tri00t::Vertex::new(7.0, 3.0, -10.0),
            ],
            vec![[0, 1, 2]],
        );
        let envelope = envelope_of(&pit_shell_mesh);

        let topology_mesh = tri00t::Triangulation::from_vertices_and_faces(
            vec![
                tri00t::Vertex::new(0.0, 0.0, -5.0),
                tri00t::Vertex::new(10.0, 0.0, -5.0),
                tri00t::Vertex::new(0.0, 10.0, -5.0),
            ],
            vec![[0, 1, 2]],
        );

        let (verts, faces) = clip_topology_to_pit_shell(&topology_mesh, &envelope);

        let kept = kept_area(&verts, &faces);
        assert!(
            (kept - 48.0).abs() < 1e-6,
            "expected the interior 2 m^2 lip clipped away, got kept_area={kept}"
        );
        for face in &faces {
            let triangle = [
                verts[face[0] as usize],
                verts[face[1] as usize],
                verts[face[2] as usize],
            ];
            assert!(
                point_in_triangle_bary_z(7.25, 1.25, triangle).is_none(),
                "interior shell patch should not remain covered by topology"
            );
        }
    }

    /// Timing run of the full pit-shell cut on large local data. Run with
    /// `cargo test --release -- --ignored` when the files are present.
    #[test]
    #[ignore]
    fn diag_pit_shell_cut_large_local_data() {
        let home = std::env::var("HOME").unwrap();
        let topology = crate::model::formats::read_mesh(format!("{home}/Downloads/NW_Surface.obj"))
            .expect("topo loads");
        let pit_shell = crate::model::formats::read_mesh(format!(
            "{home}/Downloads/20250520_ob35_604rl_osa.00t"
        ))
        .expect("pit shell loads");
        eprintln!(
            "topology {} faces, shell {} faces",
            topology.face_count(),
            pit_shell.face_count()
        );

        let start = std::time::Instant::now();
        let prepared = prepare_pit_shell_surface(&pit_shell).expect("prepare");
        eprintln!("prepare_pit_shell_surface: {:.2?}", start.elapsed());
        let start = std::time::Instant::now();
        let envelope = build_pit_shell_lower_envelope(&prepared).expect("envelope");
        eprintln!(
            "build_pit_shell_lower_envelope: {:.2?} ({} cells)",
            start.elapsed(),
            envelope.triangles.len()
        );
        let start = std::time::Instant::now();
        let (verts, faces) = clip_topology_to_pit_shell(&topology, &envelope);
        eprintln!(
            "clip_topology_to_pit_shell: {:.2?} ({} verts, {} faces)",
            start.elapsed(),
            verts.len(),
            faces.len()
        );
        assert!(!faces.is_empty());
        assert!(
            faces.len() < topology.face_count() + 4_000_000,
            "unexpected output explosion"
        );
    }

    #[test]
    #[ignore]
    fn reported_lip_point_not_covered_on_tutorial_data() {
        let topology = crate::model::formats::read_mesh("tutorial/example_data/topology.obj")
            .expect("topology.obj loads");
        let pit_surface =
            crate::model::formats::read_mesh("tutorial/example_data/tutorial_pit_surface.00t")
                .expect("pit surface loads");
        let envelope = envelope_of(&pit_surface);
        let (verts, faces) = clip_topology_to_pit_shell(&topology, &envelope);
        let x = 194_396.016;
        let y = 7_954_711.042;
        let shell_z = envelope_z(&envelope, x, y);
        eprintln!(
            "output verts={} faces={} envelope_z_at_reported={shell_z:?}",
            verts.len(),
            faces.len()
        );
        for (face_index, face) in faces.iter().enumerate() {
            let triangle = [
                verts[face[0] as usize],
                verts[face[1] as usize],
                verts[face[2] as usize],
            ];
            if let Some(z) = point_in_triangle_bary_z(x, y, triangle) {
                eprintln!(
                    "contains face={face_index} topo_z={z:.6} delta={:?} tri={triangle:?}",
                    shell_z.map(|s| z - s)
                );
                panic!("reported point is still covered by output topology");
            }
        }
    }

    /// Full-footprint verification of the pit-shell cut against real tutorial data, in both
    /// directions: no output geometry encroaches into the excavated void, and no topology
    /// that should survive goes missing. Run with `cargo test -- --ignored` when the
    /// tutorial example data is present.
    #[test]
    #[ignore]
    fn pit_shell_cut_preserves_coverage_and_removes_encroachment_on_tutorial_data() {
        let topology = crate::model::formats::read_mesh("tutorial/example_data/topology.obj")
            .expect("topology.obj loads");
        let pit_surface =
            crate::model::formats::read_mesh("tutorial/example_data/tutorial_pit_surface.00t")
                .expect("pit surface loads");
        let envelope = envelope_of(&pit_surface);
        let (verts, faces) = clip_topology_to_pit_shell(&topology, &envelope);
        let output = tri00t::Triangulation::from_vertices_and_faces(verts, faces);
        let input_bvh = crate::model::spatial::TriangleBvh::build(&topology);
        let output_bvh = crate::model::spatial::TriangleBvh::build(&output);

        // Samples closer than this (vertically) to the contact line are skipped: right on
        // the seam, coverage legitimately switches between topology and pit shell.
        const CONTACT_MARGIN: f64 = 0.01;

        let bounds = topology.bounds();
        let steps = 400;
        let step_x = (bounds.max.x - bounds.min.x) / steps as f64;
        let step_y = (bounds.max.y - bounds.min.y) / steps as f64;
        let mut worst_encroachment: Option<(f64, f64, f64)> = None;
        let mut missing: Vec<(f64, f64, f64, Option<f64>)> = Vec::new();
        let mut checked = 0usize;
        for yi in 0..=steps {
            for xi in 0..=steps {
                let x = bounds.min.x + xi as f64 * step_x;
                let y = bounds.min.y + yi as f64 * step_y;
                let Some(input_z) = covering_z_with_bvh(&topology, &input_bvh, x, y) else {
                    continue;
                };
                checked += 1;
                let shell_z = envelope_z(&envelope, x, y);
                let output_z = covering_z_with_bvh(&output, &output_bvh, x, y);
                match shell_z {
                    // Excavated: the topology is clearly above the envelope, so the output
                    // must not cover this point at all.
                    Some(shell_z) if input_z > shell_z + CONTACT_MARGIN => {
                        if output_z.is_some()
                            && worst_encroachment
                                .is_none_or(|(depth, ..)| input_z - shell_z > depth)
                        {
                            worst_encroachment = Some((input_z - shell_z, x, y));
                        }
                    }
                    // Kept: clearly below the envelope, or outside the shell footprint —
                    // the output must still cover this point.
                    Some(shell_z) if input_z < shell_z - CONTACT_MARGIN => {
                        if output_z.is_none() {
                            missing.push((x, y, input_z, Some(shell_z)));
                        }
                    }
                    None => {
                        if output_z.is_none() {
                            missing.push((x, y, input_z, None));
                        }
                    }
                    // Within the contact margin: either outcome is correct.
                    Some(_) => {}
                }
            }
        }
        eprintln!(
            "checked={checked} worst_encroachment={worst_encroachment:?} missing={} first_missing={:?}",
            missing.len(),
            missing.iter().take(20).collect::<Vec<_>>()
        );
        assert!(
            worst_encroachment.is_none(),
            "output topology encroaches into the excavated void: {worst_encroachment:?}"
        );
        assert!(
            missing.is_empty(),
            "{} should-keep samples missing from output",
            missing.len()
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

        let envelope = envelope_of(&pit_shell_mesh);
        let (verts, faces) = clip_topology_to_pit_shell(&topology_mesh, &envelope);
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

        let envelope = envelope_of(&pit_shell_mesh);
        let (verts, faces) = clip_topology_to_pit_shell(&topology_mesh, &envelope);

        assert_eq!(faces.len(), 1);
        assert_eq!(verts.len(), 3);
        assert_eq!(verts[0], tri00t::Vertex::new(0.0, 0.0, 5.0));
        assert_eq!(verts[1], tri00t::Vertex::new(10.0, 0.0, 5.0));
        assert_eq!(verts[2], tri00t::Vertex::new(0.0, 10.0, 5.0));
    }

    #[test]
    fn clip_topology_to_pit_shell_follows_walls_down_to_the_floor() {
        // Frustum pit shell: the walls determine how far in from the rim the topology must
        // be removed as depth increases, so the lower envelope must include them.
        let pit_shell_mesh = frustum_pit_shell();

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

        let envelope = envelope_of(&pit_shell_mesh);
        let (verts, faces) = clip_topology_to_pit_shell(&topology_mesh, &envelope);
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
