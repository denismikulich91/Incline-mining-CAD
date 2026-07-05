use super::cuts::{
    SurfaceClipVertex, append_surface_clip_polygon, bary_z, clip_surface_polygon,
    clip_target_triangle_to_reference_xy, point_in_triangle_bary_z,
    reference_xy_overlap_area_tolerance, triangle_area_3d, triangle_xy_area, triangle_xy_bounds,
};
use super::*;

impl<'a> App<'a> {
    /// Create a new topology by replacing the area covered by a pit or stockpile
    /// solid with that solid's mesh.
    pub(crate) fn include_solid_in_topology(
        &mut self,
        topology_id: TriangulationId,
        shape_id: TriangulationId,
        name: String,
    ) -> Result<()> {
        if topology_id == shape_id {
            anyhow::bail!("Topology and pit/stockpile solid must be different triangulations");
        }

        let topology = self
            .triangulations
            .iter()
            .find(|triangulation| triangulation.id == topology_id)
            .ok_or_else(|| anyhow::anyhow!("Topology triangulation not found"))?;
        let shape = self
            .triangulations
            .iter()
            .find(|triangulation| triangulation.id == shape_id)
            .ok_or_else(|| anyhow::anyhow!("Pit/stockpile solid triangulation not found"))?;

        let included = include_shape_mesh_in_topology_with_spatial(
            &topology.mesh,
            &topology.spatial,
            &shape.mesh,
        )?;
        if included.faces.is_empty() {
            anyhow::bail!("Combined triangulation produced no faces");
        }

        userspace_log!(
            "Included solid '{}' in topology '{}' (retained {} topology faces, skipped {} closure-cap faces)",
            shape.name,
            topology.name,
            included.retained_topology_faces,
            included.skipped_cap_faces
        );
        let edges = triangle_edge_list(&included.faces);
        self.finish_generated_triangulation_with_edges(
            name,
            included.vertices,
            included.faces,
            TriSurfaceType::Surface,
            edges,
        )
    }
}

pub(super) struct IncludedShapeMesh {
    pub(super) vertices: Vec<tri00t::Vertex>,
    pub(super) faces: Vec<[u32; 3]>,
    pub(super) retained_topology_faces: usize,
    pub(super) skipped_cap_faces: usize,
}

fn triangle_edge_list(faces: &[[u32; 3]]) -> Vec<[u32; 2]> {
    let mut edges = Vec::with_capacity(faces.len().saturating_mul(3));
    for [a, b, c] in faces.iter().copied() {
        edges.push(sorted_u32_edge(a, b));
        edges.push(sorted_u32_edge(b, c));
        edges.push(sorted_u32_edge(c, a));
    }
    edges
}

fn sorted_u32_edge(a: u32, b: u32) -> [u32; 2] {
    if a <= b { [a, b] } else { [b, a] }
}

#[cfg(test)]
pub(super) fn include_shape_mesh_in_topology(
    topology: &tri00t::Triangulation,
    shape: &tri00t::Triangulation,
) -> Result<IncludedShapeMesh> {
    let topology_spatial = crate::model::spatial::TriangleBvh::build(topology);
    include_shape_mesh_in_topology_with_spatial(topology, &topology_spatial, shape)
}

pub(super) fn include_shape_mesh_in_topology_with_spatial(
    topology: &tri00t::Triangulation,
    topology_spatial: &crate::model::spatial::TriangleBvh,
    shape: &tri00t::Triangulation,
) -> Result<IncludedShapeMesh> {
    if topology.face_count() == 0 {
        anyhow::bail!("Topology contains no faces");
    }
    if shape.face_count() == 0 {
        anyhow::bail!("Pit/stockpile solid contains no faces");
    }

    let topology_vertices = topology.vertices();
    let shape_vertices = shape.vertices();
    let shape_faces: Vec<[usize; 3]> = shape.face_vertex_indices_iter().collect();
    let cap = closure_cap_info(shape_vertices, &shape_faces, topology, topology_spatial)?;
    let xy_area_tolerance = reference_xy_overlap_area_tolerance(topology);

    let skipped_cap_faces = cap.face_mask.iter().filter(|masked| **masked).count();
    let clipped_shape = clip_shape_to_topology(
        shape_vertices,
        &shape_faces,
        &cap.face_mask,
        topology,
        topology_spatial,
        xy_area_tolerance,
        cap.shape_cut_side,
    )?;
    let cut_rings = rings_from_boundary_segments(&clipped_shape.topology_segments)?;
    if std::env::var("PI_INCLUDE_DIAG").is_ok() {
        let mut sizes: Vec<usize> = cut_rings.iter().map(|ring| ring.len()).collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        sizes.truncate(12);
        let areas: Vec<f64> = cut_rings
            .iter()
            .map(|ring| polygon_area_vertices_xy(ring).abs())
            .collect();
        let max_area = areas.iter().copied().fold(0.0f64, f64::max);
        eprintln!(
            "include diag: {} segments -> {} rings, largest sizes {:?}, max ring area {max_area:.0}",
            clipped_shape.topology_segments.len(),
            cut_rings.len(),
            sizes
        );
    }
    if cut_rings.is_empty() {
        anyhow::bail!("Pit/stockpile solid did not produce a topology intersection boundary");
    }

    // Partition the plane by the cut rings (a CDT constrained on every ring
    // edge, with far padding cells), classify each cell inside/outside the
    // footprint, then rebuild the topology cell by cell: faces that never
    // touch a covered cell pass through whole (vertex sharing preserved via
    // the remap), faces fully inside are dropped, and only faces straddling
    // the boundary are fragmented — each fragment a convex triangle–cell
    // overlap. This replaces the earlier per-face geo boolean difference,
    // whose cost was faces-in-ring-bbox × ring vertices and which stalled for
    // hours once a real multi-kilometre contact ring closed.
    let coverage = build_ring_coverage(&cut_rings)?;
    let affected = fragment_topology_faces_by_coverage(topology, &coverage);
    let mut affected_cursor = 0usize;

    let estimated_output_faces = topology
        .face_count()
        .saturating_add(clipped_shape.faces.len());
    let mut output_vertices = Vec::with_capacity(
        topology
            .vertex_count()
            .saturating_add(affected.len().saturating_mul(3))
            .saturating_add(clipped_shape.vertices.len()),
    );
    let mut output_faces = Vec::with_capacity(estimated_output_faces);
    let mut topology_vertex_remap = vec![u32::MAX; topology_vertices.len()];
    let mut retained_topology_faces = 0usize;

    for (face_index, face) in topology.face_vertex_indices_iter().enumerate() {
        let is_affected =
            affected.get(affected_cursor).map(|(index, _)| *index) == Some(face_index);
        if !is_affected {
            output_faces.push([
                remapped_topology_vertex(
                    face[0],
                    topology_vertices,
                    &mut topology_vertex_remap,
                    &mut output_vertices,
                ),
                remapped_topology_vertex(
                    face[1],
                    topology_vertices,
                    &mut topology_vertex_remap,
                    &mut output_vertices,
                ),
                remapped_topology_vertex(
                    face[2],
                    topology_vertices,
                    &mut topology_vertex_remap,
                    &mut output_vertices,
                ),
            ]);
            retained_topology_faces += 1;
            continue;
        }
        let fragments = &affected[affected_cursor].1;
        affected_cursor += 1;
        retained_topology_faces += fragments.len();
        for fragment in fragments {
            let base = output_vertices.len() as u32;
            output_vertices.extend_from_slice(fragment);
            output_faces.push([base, base + 1, base + 2]);
        }
    }

    append_mesh(
        clipped_shape.vertices,
        clipped_shape.faces,
        &mut output_vertices,
        &mut output_faces,
    );

    Ok(IncludedShapeMesh {
        vertices: output_vertices,
        faces: output_faces,
        retained_topology_faces,
        skipped_cap_faces,
    })
}

#[inline(always)]
fn remapped_topology_vertex(
    vertex_index: usize,
    topology_vertices: &[tri00t::Vertex],
    topology_vertex_remap: &mut [u32],
    output_vertices: &mut Vec<tri00t::Vertex>,
) -> u32 {
    if topology_vertex_remap[vertex_index] != u32::MAX {
        return topology_vertex_remap[vertex_index];
    }
    let mapped = output_vertices.len() as u32;
    output_vertices.push(topology_vertices[vertex_index]);
    topology_vertex_remap[vertex_index] = mapped;
    mapped
}

/// The plane partitioned by the cut rings: a constrained triangulation whose
/// cells never cross a ring edge, so each cell is entirely inside or outside
/// the replacement footprint.
pub(super) struct RingCoverage {
    triangles: Vec<[tri00t::Vertex; 3]>,
    covered: Vec<bool>,
    spatial: crate::model::spatial::TriangleBvh,
}

pub(super) fn build_ring_coverage(rings: &[Vec<tri00t::Vertex>]) -> Result<RingCoverage> {
    use rayon::prelude::*;
    use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation as _};

    let mut min = glam::DVec2::splat(f64::INFINITY);
    let mut max = glam::DVec2::splat(f64::NEG_INFINITY);
    for ring in rings {
        for vertex in ring {
            if !vertex.x.is_finite() || !vertex.y.is_finite() {
                anyhow::bail!("Cut ring contains non-finite coordinates");
            }
            min = min.min(glam::DVec2::new(vertex.x, vertex.y));
            max = max.max(glam::DVec2::new(vertex.x, vertex.y));
        }
    }
    if !min.x.is_finite() {
        anyhow::bail!("Cut rings are empty");
    }
    let origin = min;
    let extent = max - min;

    // Same bulk-load scheme as the pit-shell envelope: exact-bit point dedup
    // (-0.0 normalised) keeps our indices aligned with spade's, and only
    // edges the bulk loader rejects as crossing go through the splitting
    // insert. Rings from doubled survey sheets sit nearly on top of each
    // other in XY, so crossings are expected and handled.
    let normalized = |value: f64| if value == 0.0 { 0.0 } else { value };
    let padding = super::cuts::ENVELOPE_PADDING;
    let mut points: Vec<Point2<f64>> = Vec::new();
    for (corner_x, corner_y) in [
        (-padding, -padding),
        (extent.x + padding, -padding),
        (extent.x + padding, extent.y + padding),
        (-padding, extent.y + padding),
    ] {
        points.push(Point2::new(corner_x, corner_y));
    }
    let mut point_indices: HashMap<(u64, u64), usize> = HashMap::new();
    let mut edges: HashSet<(usize, usize)> = HashSet::new();
    for ring in rings {
        let mut indices = Vec::with_capacity(ring.len());
        for vertex in ring {
            let local_x = normalized(vertex.x - origin.x);
            let local_y = normalized(vertex.y - origin.y);
            indices.push(
                *point_indices
                    .entry((local_x.to_bits(), local_y.to_bits()))
                    .or_insert_with(|| {
                        points.push(Point2::new(local_x, local_y));
                        points.len() - 1
                    }),
            );
        }
        for i in 0..indices.len() {
            let a = indices[i];
            let b = indices[(i + 1) % indices.len()];
            if a != b {
                edges.insert(sorted_usize_pair(a, b));
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
        .map_err(|error| anyhow::anyhow!("Cut ring CDT bulk load failed: {error:?}"))?;
    if cdt.num_vertices() != point_count {
        anyhow::bail!("Cut ring CDT dropped vertices unexpectedly during bulk load");
    }
    for [a, b] in conflicting_edges {
        cdt.add_constraint_and_split(
            spade::handles::FixedVertexHandle::from_index(a),
            spade::handles::FixedVertexHandle::from_index(b),
            |point| point,
        );
    }

    let pips: Vec<RingPip> = rings.iter().map(|ring| RingPip::build(ring)).collect();
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
        .filter_map(|corners| {
            let cell = corners.map(|corner| tri00t::Vertex::new(corner.x, corner.y, 0.0));
            if triangle_xy_area(cell).abs() <= 1e-12 {
                return None;
            }
            let centroid = (corners[0] + corners[1] + corners[2]) / 3.0;
            // Inside ANY ring counts as covered: doubled sheets yield two
            // nearly coincident rings over the same footprint, and even-odd
            // across all rings together would cancel them out.
            let covered = pips.iter().any(|pip| pip.contains(centroid));
            Some((cell, covered))
        })
        .collect();

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
    let mesh = tri00t::Triangulation::from_vertices_and_faces(vertices, faces);
    let spatial = crate::model::spatial::TriangleBvh::build(&mesh);
    Ok(RingCoverage {
        triangles,
        covered,
        spatial,
    })
}

/// Even-odd point-in-polygon over one ring, with edges bucketed into
/// horizontal bands so a query touches only the few edges crossing its Y.
struct RingPip {
    min: glam::DVec2,
    max: glam::DVec2,
    band_height: f64,
    bands: Vec<Vec<(glam::DVec2, glam::DVec2)>>,
}

impl RingPip {
    fn build(ring: &[tri00t::Vertex]) -> Self {
        let mut min = glam::DVec2::splat(f64::INFINITY);
        let mut max = glam::DVec2::splat(f64::NEG_INFINITY);
        for vertex in ring {
            min = min.min(glam::DVec2::new(vertex.x, vertex.y));
            max = max.max(glam::DVec2::new(vertex.x, vertex.y));
        }
        let band_count = ring.len().clamp(1, 1024);
        let span = (max.y - min.y).max(1e-12);
        let band_height = span / band_count as f64;
        let mut bands = vec![Vec::new(); band_count];
        for index in 0..ring.len() {
            let a = ring[index];
            let b = ring[(index + 1) % ring.len()];
            let a = glam::DVec2::new(a.x, a.y);
            let b = glam::DVec2::new(b.x, b.y);
            let (y0, y1) = if a.y <= b.y { (a.y, b.y) } else { (b.y, a.y) };
            let first = (((y0 - min.y) / band_height) as usize).min(band_count - 1);
            let last = (((y1 - min.y) / band_height) as usize).min(band_count - 1);
            for band in &mut bands[first..=last] {
                band.push((a, b));
            }
        }
        Self {
            min,
            max,
            band_height,
            bands,
        }
    }

    fn contains(&self, point: glam::DVec2) -> bool {
        if point.x < self.min.x
            || point.x > self.max.x
            || point.y < self.min.y
            || point.y > self.max.y
        {
            return false;
        }
        let band = (((point.y - self.min.y) / self.band_height) as usize).min(self.bands.len() - 1);
        let mut inside = false;
        for (a, b) in &self.bands[band] {
            if (a.y > point.y) != (b.y > point.y) {
                let t = (point.y - a.y) / (b.y - a.y);
                if a.x + t * (b.x - a.x) > point.x {
                    inside = !inside;
                }
            }
        }
        inside
    }
}

/// For every topology face that overlaps a covered (inside-footprint) cell,
/// the fragments of it lying over uncovered cells — i.e. the part of the face
/// that survives the cut. Faces absent from the result never touch the
/// footprint and pass through whole; present with no fragments means fully
/// excavated. Returned sorted by face index.
fn fragment_topology_faces_by_coverage(
    topology: &tri00t::Triangulation,
    coverage: &RingCoverage,
) -> Vec<(usize, Vec<[tri00t::Vertex; 3]>)> {
    use rayon::prelude::*;

    let vertices = topology.vertices();
    let faces: Vec<[usize; 3]> = topology.face_vertex_indices_iter().collect();
    let task_count = rayon::current_num_threads().saturating_mul(4).max(1);
    let chunk_size = faces.len().div_ceil(task_count).max(1);
    let partials: Vec<Vec<(usize, Vec<[tri00t::Vertex; 3]>)>> = faces
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let mut affected = Vec::new();
            let mut candidate_stack = Vec::new();
            let mut candidates: Vec<usize> = Vec::new();
            let face_index_base = chunk_index * chunk_size;

            for (offset, face) in chunk.iter().enumerate() {
                let target = [vertices[face[0]], vertices[face[1]], vertices[face[2]]];
                let bounds = triangle_xy_bounds(target);
                candidates.clear();
                coverage
                    .spatial
                    .for_each_xy_bounds_candidate_index_with_stack(
                        bounds.0,
                        bounds.1,
                        &mut candidate_stack,
                        |index| candidates.push(index),
                    );

                // Most faces sit far from the footprint: test the covered
                // candidates first and skip all overlap work if none touch.
                let touched = candidates.iter().any(|&index| {
                    coverage.covered[index]
                        && clip_target_triangle_to_reference_xy(target, coverage.triangles[index])
                            .len()
                            >= 3
                });
                if !touched {
                    continue;
                }
                let mut fragments = Vec::new();
                for &index in &candidates {
                    if coverage.covered[index] {
                        continue;
                    }
                    let overlap =
                        clip_target_triangle_to_reference_xy(target, coverage.triangles[index]);
                    append_xy_polygon_fan(&overlap, &mut fragments);
                }
                affected.push((face_index_base + offset, fragments));
            }
            affected
        })
        .collect();

    let mut affected = Vec::with_capacity(partials.iter().map(Vec::len).sum());
    for partial in partials {
        affected.extend(partial);
    }
    affected
}

fn append_xy_polygon_fan(polygon: &[glam::DVec3], output: &mut Vec<[tri00t::Vertex; 3]>) {
    if polygon.len() < 3 {
        return;
    }
    let a = polygon[0];
    for i in 1..polygon.len() - 1 {
        let b = polygon[i];
        let c = polygon[i + 1];
        if (b - a).cross(c - a).length_squared() > 1e-20 {
            output.push([
                tri00t::Vertex::new(a.x, a.y, a.z),
                tri00t::Vertex::new(b.x, b.y, b.z),
                tri00t::Vertex::new(c.x, c.y, c.z),
            ]);
        }
    }
}

pub(super) struct ClosureCapInfo {
    face_mask: Vec<bool>,
    shape_cut_side: TriSurfaceCutSide,
}

/// Classify the solid as a pit (bulk below the topology; keep its lower
/// surface, CutTop) or a stockpile (bulk above; keep its upper surface,
/// CutBottom), and mask the flat closure cap on the removed side so it never
/// contributes geometry.
///
/// Classification is by the solid's position relative to the topology, not by
/// cap shape: real pit designs often have their largest flat area at the
/// *bottom* (floor and benches) and no flat crest cap at all — a cap-area
/// heuristic reads that as a stockpile and merges the wrong half. Flat
/// extreme caps are excluded from the score (they are closure artifacts, and
/// a wide flat crest cap would otherwise outweigh a small deep floor); every
/// other face votes with its 3D area times its centroid height above or
/// below the topology. Only when no non-cap face lies over the topology does
/// the old cap-area comparison decide.
pub(super) fn closure_cap_info(
    vertices: &[tri00t::Vertex],
    faces: &[[usize; 3]],
    topology: &tri00t::Triangulation,
    topology_spatial: &crate::model::spatial::TriangleBvh,
) -> Result<ClosureCapInfo> {
    use rayon::prelude::*;

    let mut mask = vec![false; faces.len()];
    if faces.is_empty() || vertices.is_empty() {
        anyhow::bail!("Pit/stockpile solid contains no cap faces");
    }

    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    for vertex in vertices {
        z_min = z_min.min(vertex.z);
        z_max = z_max.max(vertex.z);
    }
    let z_tolerance = ((z_max - z_min).abs() * 1e-8).max(1e-6);

    let mut min_area = 0.0;
    let mut max_area = 0.0;
    let mut candidates = Vec::new();
    let mut is_cap_candidate = vec![false; faces.len()];
    for (face_index, face) in faces.iter().copied().enumerate() {
        let triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]];
        if !triangle_is_flat_z(triangle, z_tolerance) {
            continue;
        }

        let area = triangle_xy_area(triangle).abs();
        if area <= 1e-10 {
            continue;
        }

        let z = (triangle[0].z + triangle[1].z + triangle[2].z) / 3.0;
        if (z - z_min).abs() <= z_tolerance {
            min_area += area;
            candidates.push((face_index, false));
            is_cap_candidate[face_index] = true;
        } else if (z - z_max).abs() <= z_tolerance {
            max_area += area;
            candidates.push((face_index, true));
            is_cap_candidate[face_index] = true;
        }
    }

    let topology_vertices = topology.vertices();
    let height_score: f64 = faces
        .par_iter()
        .enumerate()
        .fold(
            || (Vec::new(), 0.0f64),
            |(mut candidate_stack, mut score), (face_index, face)| {
                if is_cap_candidate[face_index] {
                    return (candidate_stack, score);
                }
                let triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]];
                let centroid_x = (triangle[0].x + triangle[1].x + triangle[2].x) / 3.0;
                let centroid_y = (triangle[0].y + triangle[1].y + triangle[2].y) / 3.0;
                let centroid_z = (triangle[0].z + triangle[1].z + triangle[2].z) / 3.0;
                // Uppermost topology sheet covering the centroid (overlapping
                // resurvey sheets can stack; the ground is the top one).
                let mut topology_z = f64::NEG_INFINITY;
                let centroid_xy = glam::DVec2::new(centroid_x, centroid_y);
                topology_spatial.for_each_xy_bounds_candidate_index_with_stack(
                    centroid_xy,
                    centroid_xy,
                    &mut candidate_stack,
                    |topology_index| {
                        if let Some(reference_face) = topology.face_vertex_indices(topology_index) {
                            let reference = [
                                topology_vertices[reference_face[0]],
                                topology_vertices[reference_face[1]],
                                topology_vertices[reference_face[2]],
                            ];
                            if let Some(z) =
                                point_in_triangle_bary_z(centroid_x, centroid_y, reference)
                            {
                                topology_z = topology_z.max(z);
                            }
                        }
                    },
                );
                if topology_z.is_finite() {
                    score += triangle_area_3d(triangle) * (centroid_z - topology_z);
                }
                (candidate_stack, score)
            },
        )
        .map(|(_, score)| score)
        .sum();

    let remove_max_cap = if height_score < -1e-9 {
        true
    } else if height_score > 1e-9 {
        false
    } else if let Some(rim_above_surface) = open_rim_above_surface(vertices, faces) {
        // Open surface with no topology overlap to vote on: a pit surface's
        // open rim (the crest) sits above the surface it bounds, a
        // stockpile's open rim (the toe) below it.
        rim_above_surface
    } else {
        if min_area <= 1e-10 && max_area <= 1e-10 {
            anyhow::bail!(
                "Pit/stockpile solid has no flat closure cap and does not overlap the topology"
            );
        }
        max_area >= min_area
    };
    let shape_cut_side = if remove_max_cap {
        TriSurfaceCutSide::CutTop
    } else {
        TriSurfaceCutSide::CutBottom
    };
    for (face_index, is_max_cap) in candidates {
        if is_max_cap == remove_max_cap {
            mask[face_index] = true;
        }
    }
    Ok(ClosureCapInfo {
        face_mask: mask,
        shape_cut_side,
    })
}

/// Whether the mesh's open rim (edges used by exactly one face) sits above
/// its area-weighted mean surface elevation. `None` when the mesh is closed,
/// the rim is too short to be a real crest/toe (shorter than the square root
/// of the surface area — stray cracks in an otherwise closed solid), or the
/// elevation difference is within noise.
fn open_rim_above_surface(vertices: &[tri00t::Vertex], faces: &[[usize; 3]]) -> Option<bool> {
    let mut edge_counts: HashMap<(usize, usize), usize> = HashMap::new();
    for face in faces {
        for i in 0..3 {
            *edge_counts
                .entry(sorted_usize_pair(face[i], face[(i + 1) % 3]))
                .or_insert(0) += 1;
        }
    }
    let mut rim_length = 0.0;
    let mut rim_z_weighted = 0.0;
    for ((a, b), count) in &edge_counts {
        if *count != 1 {
            continue;
        }
        let va = vertices[*a];
        let vb = vertices[*b];
        let dx = va.x - vb.x;
        let dy = va.y - vb.y;
        let dz = va.z - vb.z;
        let length = (dx * dx + dy * dy + dz * dz).sqrt();
        rim_length += length;
        rim_z_weighted += length * (va.z + vb.z) * 0.5;
    }
    if rim_length <= 1e-9 {
        return None;
    }

    let mut total_area = 0.0;
    let mut surface_z_weighted = 0.0;
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    for face in faces {
        let triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]];
        let area = triangle_area_3d(triangle);
        total_area += area;
        surface_z_weighted += area * (triangle[0].z + triangle[1].z + triangle[2].z) / 3.0;
        for vertex in triangle {
            z_min = z_min.min(vertex.z);
            z_max = z_max.max(vertex.z);
        }
    }
    if total_area <= 1e-9 || rim_length * rim_length <= total_area {
        return None;
    }
    let rim_z = rim_z_weighted / rim_length;
    let surface_z = surface_z_weighted / total_area;
    if (rim_z - surface_z).abs() <= (z_max - z_min).abs() * 0.01 {
        return None;
    }
    Some(rim_z > surface_z)
}

pub(super) fn triangle_is_flat_z(triangle: [tri00t::Vertex; 3], tolerance: f64) -> bool {
    let z_min = triangle
        .iter()
        .map(|vertex| vertex.z)
        .fold(f64::INFINITY, f64::min);
    let z_max = triangle
        .iter()
        .map(|vertex| vertex.z)
        .fold(f64::NEG_INFINITY, f64::max);
    z_max - z_min <= tolerance
}

pub(super) fn rings_from_boundary_segments(
    segments: &[[tri00t::Vertex; 2]],
) -> Result<Vec<Vec<tri00t::Vertex>>> {
    let tolerance_sq = 1e-8;
    let mut grid = BoundaryNodeGrid::new(tolerance_sq);
    // Segments shared by adjacent clip polygons appear an even number of
    // times and cancel; only odd-count edges lie on the cut boundary.
    let mut edge_counts: HashMap<(usize, usize), usize> = HashMap::new();
    for segment in segments {
        let a = grid.index_for(segment[0]);
        let b = grid.index_for(segment[1]);
        if a != b {
            *edge_counts.entry(sorted_usize_pair(a, b)).or_insert(0) += 1;
        }
    }
    let nodes = grid.nodes;
    let mut edges: Vec<(usize, usize)> = edge_counts
        .into_iter()
        .filter(|(_, count)| count % 2 == 1)
        .map(|(edge, _)| edge)
        .collect();
    edges.sort_unstable();
    stitch_contour_gaps(&nodes, &mut edges, segments);

    let mut adjacency = vec![Vec::<usize>::new(); nodes.len()];
    for (a, b) in &edges {
        if !adjacency[*a].contains(b) {
            adjacency[*a].push(*b);
        }
        if !adjacency[*b].contains(a) {
            adjacency[*b].push(*a);
        }
    }

    let mut visited_edges = HashSet::new();
    let mut rings = Vec::new();
    for (start, next) in edges {
        let edge_key = sorted_usize_pair(start, next);
        if visited_edges.contains(&edge_key) {
            continue;
        }

        let mut ring_nodes = vec![start, next];
        visited_edges.insert(edge_key);
        let mut previous = start;
        let mut current = next;
        let mut closed = false;
        loop {
            if current == start {
                closed = true;
                break;
            }

            let Some(candidate) = adjacency[current].iter().copied().find(|neighbor| {
                *neighbor != previous
                    && !visited_edges.contains(&sorted_usize_pair(current, *neighbor))
            }) else {
                break;
            };
            visited_edges.insert(sorted_usize_pair(current, candidate));
            previous = current;
            current = candidate;
            ring_nodes.push(current);
        }

        if ring_nodes.last() == Some(&start) {
            ring_nodes.pop();
        }
        let mut ring = ring_nodes
            .into_iter()
            .map(|node_index| nodes[node_index])
            .collect::<Vec<_>>();
        deduplicate_ring_xy(&mut ring, tolerance_sq);
        if closed && ring.len() >= 3 && polygon_area_vertices_xy(&ring).abs() > 1e-8 {
            rings.push(ring);
        }
    }

    if rings.is_empty() {
        anyhow::bail!(
            "Could not form a closed topology cut boundary from the clipped pit/stockpile"
        );
    }
    Ok(rings)
}

/// Bridge small gaps in the contact contour by pairing odd-degree (dead-end)
/// nodes with their nearest odd-degree partner.
///
/// Real topologies break the contour in two ways: numeric near-misses just
/// over the node-snap tolerance, and overlapping survey sheets that end
/// partway around the shape — the contact line jumps from one sheet to the
/// other, leaving a gap the size of the sheet separation (metres). Without
/// stitching, one gap discards the entire multi-kilometre contour and the cut
/// silently keeps every topology face. Pairs are joined closest-first and
/// only within a tolerance scaled from the median segment length, so distinct
/// far-apart contours are never fused.
fn stitch_contour_gaps(
    nodes: &[tri00t::Vertex],
    edges: &mut Vec<(usize, usize)>,
    segments: &[[tri00t::Vertex; 2]],
) {
    let mut degree = vec![0usize; nodes.len()];
    for (a, b) in edges.iter() {
        degree[*a] += 1;
        degree[*b] += 1;
    }
    let open: Vec<usize> = (0..nodes.len())
        .filter(|&index| degree[index] % 2 == 1)
        .collect();
    if open.is_empty() {
        return;
    }

    let mut lengths: Vec<f64> = segments
        .iter()
        .map(|[a, b]| {
            let dx = a.x - b.x;
            let dy = a.y - b.y;
            let dz = a.z - b.z;
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .collect();
    lengths.sort_unstable_by(f64::total_cmp);
    let median_length = lengths.get(lengths.len() / 2).copied().unwrap_or(0.0);
    let gap_tolerance = (median_length * 32.0).max(1.0e-3);
    let gap_tolerance_sq = gap_tolerance * gap_tolerance;

    let existing: HashSet<(usize, usize)> = edges.iter().copied().collect();
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..open.len() {
        for j in (i + 1)..open.len() {
            let a = nodes[open[i]];
            let b = nodes[open[j]];
            let dx = a.x - b.x;
            let dy = a.y - b.y;
            let dz = a.z - b.z;
            let distance_sq = dx * dx + dy * dy + dz * dz;
            if distance_sq <= gap_tolerance_sq
                && !existing.contains(&sorted_usize_pair(open[i], open[j]))
            {
                pairs.push((distance_sq, open[i], open[j]));
            }
        }
    }
    pairs.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));

    let mut used = HashSet::new();
    for (_, a, b) in pairs {
        if used.contains(&a) || used.contains(&b) {
            continue;
        }
        used.insert(a);
        used.insert(b);
        edges.push(sorted_usize_pair(a, b));
    }
    edges.sort_unstable();
}

/// Snaps boundary vertices to node indices through a uniform hash grid so
/// lookups stay near-constant regardless of how many segments the cut
/// produced (a linear scan here goes quadratic on dense topologies).
///
/// Matching is 3D: topologies with overlapping sheets (resurveys, tile
/// overlap) yield one contact contour per sheet at nearly the same XY but
/// different z; merging those by XY tangles the contours into one graph and
/// no ring closes. Kept apart, each contour closes or is discarded on its own.
pub(super) struct BoundaryNodeGrid {
    nodes: Vec<tri00t::Vertex>,
    cells: HashMap<(i64, i64), Vec<usize>>,
    cell_size: f64,
    tolerance_sq: f64,
}

impl BoundaryNodeGrid {
    pub(super) fn new(tolerance_sq: f64) -> Self {
        Self {
            nodes: Vec::new(),
            cells: HashMap::new(),
            cell_size: tolerance_sq.sqrt(),
            tolerance_sq,
        }
    }

    pub(super) fn index_for(&mut self, vertex: tri00t::Vertex) -> usize {
        let cell_x = (vertex.x / self.cell_size).floor() as i64;
        let cell_y = (vertex.y / self.cell_size).floor() as i64;
        // The cell size equals the snap tolerance, so any node within
        // tolerance of `vertex` lives in the 3x3 cell neighbourhood.
        for dx in -1i64..=1 {
            for dy in -1i64..=1 {
                let Some(bucket) = self.cells.get(&(cell_x + dx, cell_y + dy)) else {
                    continue;
                };
                for &index in bucket {
                    if vertices_close_xyz(self.nodes[index], vertex, self.tolerance_sq) {
                        return index;
                    }
                }
            }
        }
        let index = self.nodes.len();
        self.nodes.push(vertex);
        self.cells.entry((cell_x, cell_y)).or_default().push(index);
        index
    }
}

pub(super) fn sorted_usize_pair(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

pub(super) fn vertices_close_xy(a: tri00t::Vertex, b: tri00t::Vertex, tolerance_sq: f64) -> bool {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy <= tolerance_sq
}

pub(super) fn vertices_close_xyz(a: tri00t::Vertex, b: tri00t::Vertex, tolerance_sq: f64) -> bool {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dx * dx + dy * dy + dz * dz <= tolerance_sq
}

pub(super) fn deduplicate_ring_xy(ring: &mut Vec<tri00t::Vertex>, tolerance_sq: f64) {
    ring.dedup_by(|a, b| vertices_close_xy(*a, *b, tolerance_sq));
    if ring.len() > 1 && vertices_close_xy(ring[0], *ring.last().expect("non-empty"), tolerance_sq)
    {
        ring.pop();
    }
}

pub(super) fn polygon_area_vertices_xy(ring: &[tri00t::Vertex]) -> f64 {
    if ring.len() < 3 {
        return 0.0;
    }
    (0..ring.len())
        .map(|index| {
            let a = ring[index];
            let b = ring[(index + 1) % ring.len()];
            a.x * b.y - b.x * a.y
        })
        .sum::<f64>()
        * 0.5
}

pub(super) struct ClippedShapeMesh {
    vertices: Vec<tri00t::Vertex>,
    faces: Vec<[u32; 3]>,
    topology_segments: Vec<[tri00t::Vertex; 2]>,
}

pub(super) struct ShapeClipContext<'a> {
    topology: &'a tri00t::Triangulation,
    topology_spatial: &'a crate::model::spatial::TriangleBvh,
    xy_area_tolerance: f64,
    side: TriSurfaceCutSide,
}

#[derive(Default)]
pub(super) struct ShapeClipOutput {
    vertices: Vec<tri00t::Vertex>,
    faces: Vec<[u32; 3]>,
    topology_segments: Vec<[tri00t::Vertex; 2]>,
}

pub(super) fn clip_shape_to_topology(
    shape_vertices: &[tri00t::Vertex],
    shape_faces: &[[usize; 3]],
    cap_mask: &[bool],
    topology: &tri00t::Triangulation,
    topology_spatial: &crate::model::spatial::TriangleBvh,
    xy_area_tolerance: f64,
    side: TriSurfaceCutSide,
) -> Result<ClippedShapeMesh> {
    use rayon::prelude::*;

    let context = ShapeClipContext {
        topology,
        topology_spatial,
        xy_area_tolerance,
        side,
    };
    let task_count = rayon::current_num_threads().saturating_mul(4).max(1);
    let chunk_size = shape_faces.len().div_ceil(task_count).max(1);
    let partials = shape_faces
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, faces)| {
            let mut output = ShapeClipOutput::default();
            let mut candidate_stack = Vec::new();
            let face_index_base = chunk_index * chunk_size;

            for (offset, face) in faces.iter().copied().enumerate() {
                let face_index = face_index_base + offset;
                if cap_mask.get(face_index).copied().unwrap_or(false) {
                    continue;
                }

                let triangle = [
                    shape_vertices[face[0]],
                    shape_vertices[face[1]],
                    shape_vertices[face[2]],
                ];
                if triangle_xy_area(triangle).abs() <= 1.0e-10 {
                    clip_vertical_shape_triangle_to_topology(
                        triangle,
                        &context,
                        &mut output,
                        &mut candidate_stack,
                    );
                } else {
                    clip_surface_shape_triangle_to_topology(
                        triangle,
                        &context,
                        &mut output,
                        &mut candidate_stack,
                    );
                }
            }

            output
        })
        .collect::<Vec<_>>();

    let mut output = ShapeClipOutput {
        vertices: Vec::with_capacity(partials.iter().map(|partial| partial.vertices.len()).sum()),
        faces: Vec::with_capacity(partials.iter().map(|partial| partial.faces.len()).sum()),
        topology_segments: Vec::with_capacity(
            partials
                .iter()
                .map(|partial| partial.topology_segments.len())
                .sum(),
        ),
    };
    for partial in partials {
        append_shape_clip_output(&mut output, partial);
    }

    if output.faces.is_empty() {
        anyhow::bail!("Pit/stockpile solid has no geometry on the retained side of the topology");
    }
    if output.topology_segments.is_empty() {
        anyhow::bail!("Pit/stockpile solid did not intersect the topology surface");
    }
    Ok(ClippedShapeMesh {
        vertices: output.vertices,
        faces: output.faces,
        topology_segments: output.topology_segments,
    })
}

fn append_shape_clip_output(output: &mut ShapeClipOutput, partial: ShapeClipOutput) {
    let base = output.vertices.len() as u32;
    output.vertices.extend(partial.vertices);
    output.faces.extend(
        partial
            .faces
            .into_iter()
            .map(|face| [base + face[0], base + face[1], base + face[2]]),
    );
    output.topology_segments.extend(partial.topology_segments);
}

fn reference_triangle_for_topology_index(
    context: &ShapeClipContext<'_>,
    topology_index: usize,
) -> Option<[tri00t::Vertex; 3]> {
    let face = context.topology.face_vertex_indices(topology_index)?;
    let vertices = context.topology.vertices();
    let triangle = [vertices[face[0]], vertices[face[1]], vertices[face[2]]];
    (triangle_xy_area(triangle).abs() > context.xy_area_tolerance).then_some(triangle)
}

pub(super) fn clip_surface_shape_triangle_to_topology(
    triangle: [tri00t::Vertex; 3],
    context: &ShapeClipContext<'_>,
    output: &mut ShapeClipOutput,
    candidate_stack: &mut Vec<usize>,
) {
    let bounds = triangle_xy_bounds(triangle);
    context
        .topology_spatial
        .for_each_xy_bounds_candidate_index_with_stack(
            bounds.0,
            bounds.1,
            candidate_stack,
            |topology_index| {
                let Some(reference) =
                    reference_triangle_for_topology_index(context, topology_index)
                else {
                    return;
                };
                let overlap = clip_target_triangle_to_reference_xy(triangle, reference);
                if overlap.len() < 3 {
                    return;
                }

                let polygon = overlap
                    .into_iter()
                    .map(|point| {
                        let reference_z = bary_z(point.x, point.y, reference);
                        SurfaceClipVertex {
                            point,
                            height_delta: point.z - reference_z,
                        }
                    })
                    .collect::<Vec<_>>();
                let clipped = clip_surface_polygon(polygon, context.side);
                collect_topology_segments(&clipped, &mut output.topology_segments);
                append_surface_clip_polygon(&clipped, &mut output.vertices, &mut output.faces);
            },
        );
}

pub(super) fn clip_vertical_shape_triangle_to_topology(
    triangle: [tri00t::Vertex; 3],
    context: &ShapeClipContext<'_>,
    output: &mut ShapeClipOutput,
    candidate_stack: &mut Vec<usize>,
) {
    let Some((origin, axis, axis_len_sq)) = vertical_triangle_axis(triangle) else {
        return;
    };
    let segment_min = origin.min(origin + axis);
    let segment_max = origin.max(origin + axis);

    context
        .topology_spatial
        .for_each_xy_bounds_candidate_index_with_stack(
            segment_min,
            segment_max,
            candidate_stack,
            |topology_index| {
                let Some(reference) =
                    reference_triangle_for_topology_index(context, topology_index)
                else {
                    return;
                };
                let Some((t_min, t_max)) = segment_triangle_interval_xy(origin, axis, reference)
                else {
                    return;
                };
                if t_max - t_min <= 1.0e-10 {
                    return;
                }

                let mut polygon = triangle
                    .into_iter()
                    .map(|vertex| VerticalClipVertex {
                        t: vertical_t(origin, axis, axis_len_sq, vertex),
                        z: vertex.z,
                    })
                    .collect::<Vec<_>>();
                polygon = clip_vertical_polygon_t_min(&polygon, t_min);
                polygon = clip_vertical_polygon_t_max(&polygon, t_max);
                if polygon.len() < 3 {
                    return;
                }

                let surface_polygon = polygon
                    .into_iter()
                    .map(|vertex| {
                        let xy = origin + axis * vertex.t;
                        let reference_z = bary_z(xy.x, xy.y, reference);
                        SurfaceClipVertex {
                            point: glam::DVec3::new(xy.x, xy.y, vertex.z),
                            height_delta: vertex.z - reference_z,
                        }
                    })
                    .collect::<Vec<_>>();
                let clipped = clip_surface_polygon(surface_polygon, context.side);
                collect_topology_segments(&clipped, &mut output.topology_segments);
                append_surface_clip_polygon(&clipped, &mut output.vertices, &mut output.faces);
            },
        );
}

#[derive(Clone, Copy)]
pub(super) struct VerticalClipVertex {
    t: f64,
    z: f64,
}

pub(super) fn vertical_triangle_axis(
    triangle: [tri00t::Vertex; 3],
) -> Option<(glam::DVec2, glam::DVec2, f64)> {
    let points = triangle.map(|vertex| glam::DVec2::new(vertex.x, vertex.y));
    let pairs = [(0, 1), (0, 2), (1, 2)];
    let (a, b, axis_len_sq) = pairs
        .into_iter()
        .map(|(a, b)| (a, b, points[a].distance_squared(points[b])))
        .max_by(|left, right| left.2.total_cmp(&right.2))?;
    if axis_len_sq <= 1.0e-16 {
        return None;
    }
    Some((points[a], points[b] - points[a], axis_len_sq))
}

pub(super) fn vertical_t(
    origin: glam::DVec2,
    axis: glam::DVec2,
    axis_len_sq: f64,
    vertex: tri00t::Vertex,
) -> f64 {
    (glam::DVec2::new(vertex.x, vertex.y) - origin).dot(axis) / axis_len_sq
}

pub(super) fn segment_triangle_interval_xy(
    origin: glam::DVec2,
    axis: glam::DVec2,
    triangle: [tri00t::Vertex; 3],
) -> Option<(f64, f64)> {
    let points = triangle.map(|vertex| glam::DVec2::new(vertex.x, vertex.y));
    let clip_ccw = triangle_xy_area(triangle) > 0.0;
    let mut t_min: f64 = 0.0;
    let mut t_max: f64 = 1.0;

    for edge_index in 0..3 {
        let edge_a = points[edge_index];
        let edge_b = points[(edge_index + 1) % 3];
        let edge = edge_b - edge_a;
        let signed_at = |t: f64| {
            let point = origin + axis * t;
            let cross = edge.x * (point.y - edge_a.y) - edge.y * (point.x - edge_a.x);
            if clip_ccw { cross } else { -cross }
        };
        let d0 = signed_at(0.0);
        let d1 = signed_at(1.0);
        let inside0 = d0 >= -1.0e-10;
        let inside1 = d1 >= -1.0e-10;
        match (inside0, inside1) {
            (true, true) => {}
            (false, false) => return None,
            _ => {
                let denom = d0 - d1;
                if denom.abs() <= 1.0e-20 {
                    return None;
                }
                let t = (d0 / denom).clamp(0.0, 1.0);
                if inside0 {
                    t_max = t_max.min(t);
                } else {
                    t_min = t_min.max(t);
                }
                if t_min > t_max {
                    return None;
                }
            }
        }
    }

    Some((t_min, t_max))
}

pub(super) fn clip_vertical_polygon_t_min(
    polygon: &[VerticalClipVertex],
    t_min: f64,
) -> Vec<VerticalClipVertex> {
    clip_vertical_polygon_by_t(polygon, t_min, true)
}

pub(super) fn clip_vertical_polygon_t_max(
    polygon: &[VerticalClipVertex],
    t_max: f64,
) -> Vec<VerticalClipVertex> {
    clip_vertical_polygon_by_t(polygon, t_max, false)
}

pub(super) fn clip_vertical_polygon_by_t(
    polygon: &[VerticalClipVertex],
    t_plane: f64,
    keep_greater: bool,
) -> Vec<VerticalClipVertex> {
    if polygon.is_empty() {
        return Vec::new();
    }
    let retained = |vertex: VerticalClipVertex| {
        if keep_greater {
            vertex.t >= t_plane - 1.0e-10
        } else {
            vertex.t <= t_plane + 1.0e-10
        }
    };

    let mut output = Vec::new();
    let mut previous = *polygon.last().expect("polygon is non-empty");
    let mut previous_inside = retained(previous);
    for &current in polygon {
        let current_inside = retained(current);
        if current_inside != previous_inside {
            let denom = current.t - previous.t;
            if denom.abs() > 1.0e-20 {
                let u = ((t_plane - previous.t) / denom).clamp(0.0, 1.0);
                output.push(VerticalClipVertex {
                    t: t_plane,
                    z: previous.z + (current.z - previous.z) * u,
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

pub(super) fn collect_topology_segments(
    polygon: &[SurfaceClipVertex],
    segments: &mut Vec<[tri00t::Vertex; 2]>,
) {
    if polygon.len() < 2 {
        return;
    }
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        if a.height_delta.abs() > 1e-8 || b.height_delta.abs() > 1e-8 {
            continue;
        }
        if a.point.distance_squared(b.point) <= 1e-16 {
            continue;
        }
        segments.push([
            tri00t::Vertex::new(a.point.x, a.point.y, a.point.z),
            tri00t::Vertex::new(b.point.x, b.point.y, b.point.z),
        ]);
    }
}

pub(super) fn append_mesh(
    source_vertices: Vec<tri00t::Vertex>,
    source_faces: Vec<[u32; 3]>,
    output_vertices: &mut Vec<tri00t::Vertex>,
    output_faces: &mut Vec<[u32; 3]>,
) {
    let base = output_vertices.len() as u32;
    output_vertices.extend(source_vertices);
    output_faces.extend(
        source_faces
            .into_iter()
            .map(|face| [base + face[0], base + face[1], base + face[2]]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_topology(cells: usize, extent: f64) -> tri00t::Triangulation {
        let step = extent / cells as f64;
        let side = cells + 1;
        let mut vertices = Vec::with_capacity(side * side);
        for row in 0..side {
            for col in 0..side {
                vertices.push(tri00t::Vertex::new(
                    col as f64 * step,
                    row as f64 * step,
                    0.0,
                ));
            }
        }
        let mut faces = Vec::with_capacity(cells * cells * 2);
        for row in 0..cells {
            for col in 0..cells {
                let a = (row * side + col) as u32;
                let b = a + 1;
                let c = a + side as u32;
                let d = c + 1;
                faces.push([a, b, d]);
                faces.push([a, d, c]);
            }
        }
        tri00t::Triangulation::from_vertices_and_faces(vertices, faces)
    }

    /// Closed box solid: flat cap at `z_top`, flat floor at `z_floor`,
    /// vertical walls — the shape of a rough pit design.
    fn box_pit(min: [f64; 2], max: [f64; 2], z_floor: f64, z_top: f64) -> tri00t::Triangulation {
        let corner = |x: f64, y: f64, z: f64| tri00t::Vertex::new(x, y, z);
        let vertices = vec![
            corner(min[0], min[1], z_floor),
            corner(max[0], min[1], z_floor),
            corner(max[0], max[1], z_floor),
            corner(min[0], max[1], z_floor),
            corner(min[0], min[1], z_top),
            corner(max[0], min[1], z_top),
            corner(max[0], max[1], z_top),
            corner(min[0], max[1], z_top),
        ];
        let faces = vec![
            // floor + cap
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            // walls
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        tri00t::Triangulation::from_vertices_and_faces(vertices, faces)
    }

    fn kept_xy_area(vertices: &[tri00t::Vertex], faces: &[[u32; 3]]) -> f64 {
        faces
            .iter()
            .map(|face| {
                let triangle = [
                    vertices[face[0] as usize],
                    vertices[face[1] as usize],
                    vertices[face[2] as usize],
                ];
                triangle_xy_area(triangle).abs()
            })
            .sum()
    }

    #[test]
    fn include_box_pit_replaces_topology_inside_footprint() {
        // Flat topology at z=0 over [0,20]^2; box pit x/y:[5.3,10.3] with the
        // floor at z=-5 (below the topology) and the closure cap at z=2.
        let topology = grid_topology(20, 20.0);
        let pit = box_pit([5.3, 5.3], [10.3, 10.3], -5.0, 2.0);

        let included = include_shape_mesh_in_topology(&topology, &pit).expect("include succeeds");

        assert_eq!(included.skipped_cap_faces, 2, "top cap should be skipped");
        assert!(included.retained_topology_faces > 0);

        // The pit floor replaces the footprint exactly, so the XY-projected
        // area is unchanged: (400 - 25) retained topology + 25 floor. The
        // vertical walls contribute no XY area.
        let area = kept_xy_area(&included.vertices, &included.faces);
        assert!(
            (area - 400.0).abs() < 1e-6,
            "expected XY area 400, got {area}"
        );

        // The floor must survive at its own depth, and no topology-level
        // geometry may remain strictly inside the footprint.
        assert!(
            included.vertices.iter().any(|vertex| vertex.z == -5.0),
            "pit floor missing from output"
        );
        for vertex in &included.vertices {
            let strictly_inside = vertex.x > 5.3 + 1e-6
                && vertex.x < 10.3 - 1e-6
                && vertex.y > 5.3 + 1e-6
                && vertex.y < 10.3 - 1e-6;
            assert!(
                !(strictly_inside && vertex.z == 0.0),
                "topology vertex ({}, {}) left inside the pit footprint",
                vertex.x,
                vertex.y
            );
        }
    }

    /// The reported misclassification: a pit whose only flat extreme cap is
    /// its *floor* (the crest closure follows terrain, so it is not flat).
    /// The old cap-area heuristic read the flat floor as a stockpile's bottom
    /// closure and merged the upper surface; position relative to the
    /// topology must classify it as a pit.
    #[test]
    fn include_pit_with_flat_floor_and_uneven_crest_classifies_as_pit() {
        let topology = grid_topology(20, 20.0);
        // Box pit with the closure cap tilted (z from 1 to 2, never flat) and
        // a flat floor at z=-5 — the floor is the largest flat area.
        let corner = |x: f64, y: f64, z: f64| tri00t::Vertex::new(x, y, z);
        let vertices = vec![
            corner(5.3, 5.3, -5.0),
            corner(10.3, 5.3, -5.0),
            corner(10.3, 10.3, -5.0),
            corner(5.3, 10.3, -5.0),
            corner(5.3, 5.3, 1.0),
            corner(10.3, 5.3, 1.5),
            corner(10.3, 10.3, 2.0),
            corner(5.3, 10.3, 1.5),
        ];
        let faces = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        let pit = tri00t::Triangulation::from_vertices_and_faces(vertices, faces);

        let included = include_shape_mesh_in_topology(&topology, &pit).expect("include succeeds");

        assert!(
            included.vertices.iter().any(|vertex| vertex.z == -5.0),
            "pit floor missing from output — solid was treated as a stockpile"
        );
        for vertex in &included.vertices {
            let strictly_inside = vertex.x > 5.3 + 1e-6
                && vertex.x < 10.3 - 1e-6
                && vertex.y > 5.3 + 1e-6
                && vertex.y < 10.3 - 1e-6;
            assert!(
                !(strictly_inside && vertex.z == 0.0),
                "topology vertex ({}, {}) left inside the pit footprint",
                vertex.x,
                vertex.y
            );
        }
    }

    #[test]
    #[ignore]
    fn diag_pit_00t_structure() {
        let home = std::env::var("HOME").unwrap();
        let shape =
            crate::model::formats::read_mesh(format!("{home}/Downloads/pit.00t")).expect("loads");
        let vertices = shape.vertices();
        let faces: Vec<[usize; 3]> = shape.face_vertex_indices_iter().collect();
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;
        for v in vertices {
            z_min = z_min.min(v.z);
            z_max = z_max.max(v.z);
        }
        let z_tolerance = ((z_max - z_min).abs() * 1e-8).max(1e-6);
        eprintln!(
            "{} verts, {} faces, z [{z_min}, {z_max}], tol {z_tolerance}",
            vertices.len(),
            faces.len()
        );

        let mut flat = 0usize;
        let mut flat_at_min = (0usize, 0.0f64);
        let mut flat_at_max = (0usize, 0.0f64);
        let mut vertical = 0usize;
        let mut other = 0usize;
        let mut flat_z_samples: Vec<f64> = Vec::new();
        for face in &faces {
            let t = [vertices[face[0]], vertices[face[1]], vertices[face[2]]];
            let xy_area = triangle_xy_area(t).abs();
            if triangle_is_flat_z(t, z_tolerance) {
                flat += 1;
                let z = (t[0].z + t[1].z + t[2].z) / 3.0;
                flat_z_samples.push(z);
                if (z - z_min).abs() <= z_tolerance {
                    flat_at_min.0 += 1;
                    flat_at_min.1 += xy_area;
                } else if (z - z_max).abs() <= z_tolerance {
                    flat_at_max.0 += 1;
                    flat_at_max.1 += xy_area;
                }
            } else if xy_area <= 1e-10 {
                vertical += 1;
            } else {
                other += 1;
            }
        }
        flat_z_samples.sort_by(f64::total_cmp);
        flat_z_samples.dedup_by(|a, b| (*a - *b).abs() < 0.5);
        eprintln!(
            "flat {flat} (at z_min: {flat_at_min:?}, at z_max: {flat_at_max:?}), vertical {vertical}, sloped {other}"
        );
        eprintln!("distinct flat z levels: {flat_z_samples:?}");

        // Boundary (open) edges: count edges used by only one face.
        let mut edge_counts: HashMap<(usize, usize), usize> = HashMap::new();
        for face in &faces {
            for i in 0..3 {
                let a = face[i];
                let b = face[(i + 1) % 3];
                *edge_counts.entry(sorted_usize_pair(a, b)).or_insert(0usize) += 1;
            }
        }
        let open_edges = edge_counts.values().filter(|c| **c == 1).count();
        let over_shared = edge_counts.values().filter(|c| **c > 2).count();
        eprintln!(
            "edges: {} total, {open_edges} open (boundary), {over_shared} shared by >2 faces",
            edge_counts.len()
        );

        // Z histogram of face centroids weighted by 3D area.
        let mut hist = [0.0f64; 24];
        for face in &faces {
            let t = [vertices[face[0]], vertices[face[1]], vertices[face[2]]];
            let z = (t[0].z + t[1].z + t[2].z) / 3.0;
            let bin = (((z - z_min) / (z_max - z_min)) * 23.0).clamp(0.0, 23.0) as usize;
            hist[bin] += triangle_area_3d(t);
        }
        for (bin, area) in hist.iter().enumerate() {
            if *area > 0.0 {
                eprintln!(
                    "  z {:.0}..{:.0}: area {:.0}",
                    z_min + (z_max - z_min) * bin as f64 / 24.0,
                    z_min + (z_max - z_min) * (bin as f64 + 1.0) / 24.0,
                    area
                );
            }
        }
    }

    /// End-to-end Include Solid on the real reported data: the pit solid must
    /// classify as a pit (excavate below the topo), not a stockpile.
    #[test]
    #[ignore]
    fn diag_real_pit_include() {
        let home = std::env::var("HOME").unwrap();
        let topo_path = format!("{home}/Downloads/NW_Surface.obj");
        let topology = crate::model::formats::read_mesh(&topo_path).expect("topo loads");
        let start = std::time::Instant::now();
        let spatial = crate::model::spatial::TriangleBvh::build(&topology);
        eprintln!("BVH build: {:.2?}", start.elapsed());
        for name in ["pit.00t", "20250520_ob35_604rl_osa.00t"] {
            let path = format!("{home}/Downloads/{name}");
            let shape = crate::model::formats::read_mesh(&path).expect("shape loads");
            let sb = shape.bounds();
            eprintln!(
                "{name}: xy [{:.1},{:.1}]x[{:.1},{:.1}] z [{:.1},{:.1}]",
                sb.min.x, sb.max.x, sb.min.y, sb.max.y, sb.min.z, sb.max.z
            );
            let vertices = shape.vertices();
            let faces: Vec<[usize; 3]> = shape.face_vertex_indices_iter().collect();
            let start = std::time::Instant::now();
            let cap = match closure_cap_info(vertices, &faces, &topology, &spatial) {
                Ok(cap) => cap,
                Err(error) => {
                    eprintln!("  closure_cap_info failed: {error}");
                    continue;
                }
            };
            eprintln!(
                "  classified in {:.2?}: cut side {:?} ({} cap faces masked)",
                start.elapsed(),
                cap.shape_cut_side,
                cap.face_mask.iter().filter(|m| **m).count()
            );
            let start = std::time::Instant::now();
            match include_shape_mesh_in_topology_with_spatial(&topology, &spatial, &shape) {
                Ok(included) => {
                    let shape_z_min = sb.min.z;
                    let deep = included
                        .vertices
                        .iter()
                        .filter(|v| v.z < shape_z_min + 1.0)
                        .count();
                    eprintln!(
                        "  include in {:.2?}: {} faces ({} retained topo, {} caps skipped), {} verts near pit floor",
                        start.elapsed(),
                        included.faces.len(),
                        included.retained_topology_faces,
                        included.skipped_cap_faces,
                        deep
                    );
                }
                Err(error) => eprintln!("  include failed: {error}"),
            }
        }
    }

    /// Reproduces the hang reported on a 9.4M-face topology: without the
    /// ring-bounds early-out every topology face ran a geo boolean difference.
    /// Run with `cargo test --release -- --ignored` when NW_Surface.obj is
    /// present locally.
    #[test]
    #[ignore]
    fn include_box_pit_in_large_local_topo_completes() {
        let path = std::env::var("PROINSPECTOR_LARGE_TOPO").unwrap_or_else(|_| {
            format!(
                "{}/Downloads/NW_Surface.obj",
                std::env::var("HOME").unwrap()
            )
        });
        let topology = crate::model::formats::read_mesh(&path).expect("large topo loads");
        eprintln!(
            "topology: {} vertices, {} faces",
            topology.vertex_count(),
            topology.face_count()
        );

        // Place a 200x200 m box pit at the centre of the topo's bounds.
        let vertices = topology.vertices();
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        let (mut min_z, mut max_z) = (f64::INFINITY, f64::NEG_INFINITY);
        for vertex in vertices {
            min_x = min_x.min(vertex.x);
            max_x = max_x.max(vertex.x);
            min_y = min_y.min(vertex.y);
            max_y = max_y.max(vertex.y);
            min_z = min_z.min(vertex.z);
            max_z = max_z.max(vertex.z);
        }
        // Offset off the topo's 1 m grid so the walls don't coincide exactly
        // with shared triangle edges (contact segments emitted by both
        // neighbours cancel as duplicates on exact alignment).
        let cx = (min_x + max_x) / 2.0 + 0.37;
        let cy = (min_y + max_y) / 2.0 + 0.23;
        let pit = box_pit(
            [cx - 100.0, cy - 100.0],
            [cx + 100.0, cy + 100.0],
            min_z - 50.0,
            max_z + 50.0,
        );

        let start = std::time::Instant::now();
        let included = include_shape_mesh_in_topology(&topology, &pit).expect("include succeeds");
        eprintln!(
            "included in {:.2?}: {} faces ({} retained topology)",
            start.elapsed(),
            included.faces.len(),
            included.retained_topology_faces
        );
        // The pit floor must survive, and the footprint must actually have
        // been excavated (an unclosed cut ring degrades into a silent no-op
        // where every topology face is retained).
        let floor_z = min_z - 50.0;
        assert!(
            included.vertices.iter().any(|vertex| vertex.z == floor_z),
            "pit floor missing from output"
        );
        assert!(
            included.retained_topology_faces < topology.face_count(),
            "no topology faces were removed — cut ring did not close"
        );
    }

    /// Topologies with overlapping surface sheets (resurveys, tile overlap)
    /// produce one wall-contact contour per sheet at the same XY but
    /// different z. XY-only node snapping welded the contours into one
    /// tangled graph in which no ring closed; 3D snapping keeps them apart so
    /// the complete contour closes and the partial sheet's open chain is
    /// discarded.
    #[test]
    fn include_box_pit_closes_ring_despite_partial_overlapping_sheet() {
        let topology = grid_topology(20, 20.0);
        let mut vertices = topology.vertices().to_vec();
        let mut faces: Vec<[u32; 3]> = topology
            .face_vertex_indices_iter()
            .map(|face| [face[0] as u32, face[1] as u32, face[2] as u32])
            .collect();
        // Second sheet 0.5 above the main surface, covering only y:[0,8] —
        // it crosses the pit's south wall but ends inside the footprint, so
        // its contact contour cannot close.
        let base = vertices.len() as u32;
        vertices.push(tri00t::Vertex::new(0.0, 0.0, 0.5));
        vertices.push(tri00t::Vertex::new(20.0, 0.0, 0.5));
        vertices.push(tri00t::Vertex::new(20.0, 8.0, 0.5));
        vertices.push(tri00t::Vertex::new(0.0, 8.0, 0.5));
        faces.push([base, base + 1, base + 2]);
        faces.push([base, base + 2, base + 3]);
        let topology = tri00t::Triangulation::from_vertices_and_faces(vertices, faces);
        let pit = box_pit([5.3, 5.3], [10.3, 10.3], -5.0, 2.0);

        let included = include_shape_mesh_in_topology(&topology, &pit).expect("include succeeds");

        assert!(
            included.vertices.iter().any(|vertex| vertex.z == -5.0),
            "pit floor missing from output"
        );
        for vertex in &included.vertices {
            let strictly_inside = vertex.x > 5.3 + 1e-6
                && vertex.x < 10.3 - 1e-6
                && vertex.y > 5.3 + 1e-6
                && vertex.y < 10.3 - 1e-6;
            assert!(
                !(strictly_inside && vertex.z == 0.0),
                "main-sheet vertex ({}, {}) left inside the pit footprint",
                vertex.x,
                vertex.y
            );
        }
    }
}
