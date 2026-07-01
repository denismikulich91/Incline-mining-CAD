use super::cuts::{
    SurfaceClipVertex, append_surface_clip_polygon, bary_z, clip_surface_polygon,
    clip_target_triangle_to_reference_xy, prepare_reference_surface_relaxed, triangle_xy_area,
    triangle_xy_bounds,
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

        let included = include_shape_mesh_in_topology(&topology.mesh, &shape.mesh)?;
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
        self.finish_generated_triangulation(
            name,
            included.vertices,
            included.faces,
            TriSurfaceType::Surface,
        )
    }
}

pub(super) struct IncludedShapeMesh {
    pub(super) vertices: Vec<tri00t::Vertex>,
    pub(super) faces: Vec<[u32; 3]>,
    pub(super) retained_topology_faces: usize,
    pub(super) skipped_cap_faces: usize,
}

pub(super) fn include_shape_mesh_in_topology(
    topology: &tri00t::Triangulation,
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
    let cap = closure_cap_info(shape_vertices, &shape_faces)?;
    let topology_surface = prepare_reference_surface_relaxed(topology)?;
    if topology_surface.skipped_vertical_faces > 0 {
        userspace_warn!(
            "Ignored {} vertical or degenerate topology face(s) with no XY area",
            topology_surface.skipped_vertical_faces
        );
    }

    let skipped_cap_faces = cap.face_mask.iter().filter(|masked| **masked).count();
    let clipped_shape = clip_shape_to_topology(
        shape_vertices,
        &shape_faces,
        &cap.face_mask,
        &topology_surface.mesh,
        &topology_surface.triangles,
        &topology_surface.spatial,
        cap.shape_cut_side,
    )?;
    let cut_rings = rings_from_boundary_segments(&clipped_shape.topology_segments)?;
    if cut_rings.is_empty() {
        anyhow::bail!("Pit/stockpile solid did not produce a topology intersection boundary");
    }

    let mut output_vertices = Vec::new();
    let mut output_faces = Vec::new();
    let mut retained_topology_faces = 0usize;

    for face in topology.face_vertex_indices_iter() {
        let triangle = [
            topology_vertices[face[0]],
            topology_vertices[face[1]],
            topology_vertices[face[2]],
        ];
        let clipped = clip_triangle_outside_footprints(triangle, &cut_rings);
        retained_topology_faces += clipped.len();
        for clipped_triangle in clipped {
            let base = output_vertices.len() as u32;
            output_vertices.extend_from_slice(&clipped_triangle);
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

pub(super) struct ClosureCapInfo {
    face_mask: Vec<bool>,
    shape_cut_side: TriSurfaceCutSide,
}

pub(super) fn closure_cap_info(
    vertices: &[tri00t::Vertex],
    faces: &[[usize; 3]],
) -> Result<ClosureCapInfo> {
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
        } else if (z - z_max).abs() <= z_tolerance {
            max_area += area;
            candidates.push((face_index, true));
        }
    }

    if min_area <= 1e-10 && max_area <= 1e-10 {
        anyhow::bail!("Pit/stockpile solid has no flat closure cap to use as a footprint");
    }
    let remove_max_cap = max_area >= min_area;
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

pub(super) fn clip_triangle_outside_footprints(
    triangle: [tri00t::Vertex; 3],
    footprints: &[Vec<tri00t::Vertex>],
) -> Vec<[tri00t::Vertex; 3]> {
    use geo::{BooleanOps, Coord, LineString, MultiPolygon, Polygon as GeoPoly};

    let mut tri_coords: Vec<Coord<f64>> = triangle
        .iter()
        .map(|point| Coord {
            x: point.x,
            y: point.y,
        })
        .collect();
    tri_coords.push(tri_coords[0]);
    let mut result = MultiPolygon(vec![GeoPoly::new(LineString::new(tri_coords), vec![])]);

    for footprint in footprints {
        if footprint.len() < 3 {
            continue;
        }
        let mut clip_coords: Vec<Coord<f64>> = footprint
            .iter()
            .map(|point| Coord {
                x: point.x,
                y: point.y,
            })
            .collect();
        clip_coords.push(clip_coords[0]);
        let clip = GeoPoly::new(LineString::new(clip_coords), vec![]);
        result = result.difference(&clip);
        if result.0.is_empty() {
            break;
        }
    }

    let mut output = Vec::new();
    for polygon in result {
        append_geo_polygon_as_triangles(&polygon, triangle, &mut output);
    }
    output
}

pub(super) fn append_geo_polygon_as_triangles(
    polygon: &geo::Polygon<f64>,
    source_triangle: [tri00t::Vertex; 3],
    output: &mut Vec<[tri00t::Vertex; 3]>,
) {
    let mut flat = Vec::new();
    let mut holes = Vec::new();

    let exterior: Vec<_> = polygon.exterior().coords().copied().collect();
    append_ring_coords(&exterior, &mut flat);
    for interior in polygon.interiors() {
        let ring: Vec<_> = interior.coords().copied().collect();
        if ring.len().saturating_sub(1) < 3 {
            continue;
        }
        holes.push(flat.len() / 2);
        append_ring_coords(&ring, &mut flat);
    }

    if flat.len() < 6 {
        return;
    }
    let Ok(indices) = earcutr::earcut(&flat, &holes, 2) else {
        return;
    };
    for triangle_indices in indices.chunks_exact(3) {
        let make_vertex = |index: usize| {
            let x = flat[index * 2];
            let y = flat[index * 2 + 1];
            tri00t::Vertex::new(x, y, bary_z(x, y, source_triangle))
        };
        output.push([
            make_vertex(triangle_indices[0]),
            make_vertex(triangle_indices[1]),
            make_vertex(triangle_indices[2]),
        ]);
    }
}

pub(super) fn append_ring_coords(ring: &[geo::Coord<f64>], flat: &mut Vec<f64>) {
    let count = ring.len().saturating_sub(1);
    for coord in &ring[..count] {
        flat.extend([coord.x, coord.y]);
    }
}

pub(super) fn rings_from_boundary_segments(
    segments: &[[tri00t::Vertex; 2]],
) -> Result<Vec<Vec<tri00t::Vertex>>> {
    let segments = unique_boundary_segments(segments);
    let tolerance_sq = 1e-8;
    let mut nodes: Vec<tri00t::Vertex> = Vec::new();
    let mut edges: Vec<(usize, usize)> = Vec::new();

    for segment in &segments {
        let a = boundary_node_index(&mut nodes, segment[0], tolerance_sq);
        let b = boundary_node_index(&mut nodes, segment[1], tolerance_sq);
        if a != b {
            edges.push((a, b));
        }
    }

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

pub(super) fn boundary_node_index(
    nodes: &mut Vec<tri00t::Vertex>,
    vertex: tri00t::Vertex,
    tolerance_sq: f64,
) -> usize {
    if let Some(index) = nodes
        .iter()
        .position(|node| vertices_close_xy(*node, vertex, tolerance_sq))
    {
        return index;
    }
    nodes.push(vertex);
    nodes.len() - 1
}

pub(super) fn sorted_usize_pair(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

pub(super) fn unique_boundary_segments(
    segments: &[[tri00t::Vertex; 2]],
) -> Vec<[tri00t::Vertex; 2]> {
    let tolerance_sq = 1e-8;
    let mut unique: Vec<[tri00t::Vertex; 2]> = Vec::new();
    'segment: for segment in segments {
        for index in 0..unique.len() {
            if segments_match_xy(*segment, unique[index], tolerance_sq) {
                unique.swap_remove(index);
                continue 'segment;
            }
        }
        unique.push(*segment);
    }
    unique
}

pub(super) fn segments_match_xy(
    a: [tri00t::Vertex; 2],
    b: [tri00t::Vertex; 2],
    tolerance_sq: f64,
) -> bool {
    (vertices_close_xy(a[0], b[0], tolerance_sq) && vertices_close_xy(a[1], b[1], tolerance_sq))
        || (vertices_close_xy(a[0], b[1], tolerance_sq)
            && vertices_close_xy(a[1], b[0], tolerance_sq))
}

pub(super) fn vertices_close_xy(a: tri00t::Vertex, b: tri00t::Vertex, tolerance_sq: f64) -> bool {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy <= tolerance_sq
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
    topology_triangles: &'a [[tri00t::Vertex; 3]],
    topology_spatial: &'a crate::model::spatial::TriangleBvh,
    side: TriSurfaceCutSide,
}

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
    topology_triangles: &[[tri00t::Vertex; 3]],
    topology_spatial: &crate::model::spatial::TriangleBvh,
    side: TriSurfaceCutSide,
) -> Result<ClippedShapeMesh> {
    let context = ShapeClipContext {
        topology,
        topology_triangles,
        topology_spatial,
        side,
    };
    let mut output = ShapeClipOutput {
        vertices: Vec::new(),
        faces: Vec::new(),
        topology_segments: Vec::new(),
    };

    for (face_index, face) in shape_faces.iter().copied().enumerate() {
        if cap_mask.get(face_index).copied().unwrap_or(false) {
            continue;
        }

        let triangle = [
            shape_vertices[face[0]],
            shape_vertices[face[1]],
            shape_vertices[face[2]],
        ];
        if triangle_xy_area(triangle).abs() <= 1.0e-10 {
            clip_vertical_shape_triangle_to_topology(triangle, &context, &mut output);
        } else {
            clip_surface_shape_triangle_to_topology(triangle, &context, &mut output);
        }
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

pub(super) fn clip_surface_shape_triangle_to_topology(
    triangle: [tri00t::Vertex; 3],
    context: &ShapeClipContext<'_>,
    output: &mut ShapeClipOutput,
) {
    let bounds = triangle_xy_bounds(triangle);
    for topology_index in
        context
            .topology_spatial
            .xy_bounds_candidate_indices(context.topology, bounds.0, bounds.1)
    {
        let reference = context.topology_triangles[topology_index];
        let overlap = clip_target_triangle_to_reference_xy(triangle, reference);
        if overlap.len() < 3 {
            continue;
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
    }
}

pub(super) fn clip_vertical_shape_triangle_to_topology(
    triangle: [tri00t::Vertex; 3],
    context: &ShapeClipContext<'_>,
    output: &mut ShapeClipOutput,
) {
    let Some((origin, axis, axis_len_sq)) = vertical_triangle_axis(triangle) else {
        return;
    };
    let segment_min = origin.min(origin + axis);
    let segment_max = origin.max(origin + axis);

    for topology_index in context.topology_spatial.xy_bounds_candidate_indices(
        context.topology,
        segment_min,
        segment_max,
    ) {
        let reference = context.topology_triangles[topology_index];
        let Some((t_min, t_max)) = segment_triangle_interval_xy(origin, axis, reference) else {
            continue;
        };
        if t_max - t_min <= 1.0e-10 {
            continue;
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
            continue;
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
    }
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
