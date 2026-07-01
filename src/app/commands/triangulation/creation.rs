use super::*;
use crate::model::geometry::points_coincident;

impl<'a> App<'a> {
    pub(crate) fn create_triangulation_from_objects(
        &mut self,
        name: String,
        object_ids: Vec<ObjectId>,
        surface_type: TriSurfaceType,
    ) -> Result<()> {
        if object_ids.is_empty() {
            anyhow::bail!("No objects selected for triangulation");
        }

        // Only closed polygons are accepted — open lines and points cannot form a surface.
        let mut rings: Vec<Vec<glam::DVec3>> = Vec::new();
        let mut rejected = 0usize;
        for id in &object_ids {
            let Some(obj) = self.scene_document.get_object(*id) else {
                continue;
            };
            match obj {
                Object::Polyline {
                    verts,
                    closed: true,
                    ..
                } if verts.len() >= 3 => {
                    rings.push(verts.iter().map(|v| v.pos).collect());
                }
                _ => {
                    rejected += 1;
                }
            }
        }

        if rings.is_empty() {
            anyhow::bail!(
                "No closed polygons selected — triangulation requires closed polygons with at least 3 vertices"
            );
        }
        if rejected > 0 {
            userspace_warn!(
                "Ignored {} non-polygon object(s) during triangulation",
                rejected
            );
        }

        let (all_verts, all_faces) = if rings.len() == 1 && surface_type == TriSurfaceType::Surface
        {
            // Single ring: triangulate its flat interior (normal pointing up).
            let flip = signed_area_xy(&rings[0]) <= 0.0; // CW ring needs flip for normal-up
            cdt_fill_ring(&rings[0], flip)?
        } else if surface_type == TriSurfaceType::Surface {
            // Nested contours, benches and berms are terrain breaklines, not a
            // Z-sorted loft stack. Triangulate all selected boundaries in one
            // XY CDT so every string edge is preserved and its vertex Z drives
            // the resulting terrain surface.
            cdt_surface_from_breaklines(&rings)?
        } else {
            closed_solid_from_breaklines(&rings)?
        };

        if all_faces.is_empty() {
            anyhow::bail!(
                "Triangulation produced no faces — polygons may be collinear or degenerate"
            );
        }

        userspace_log!(
            "Created triangulation from {} polygon(s), surface type {:?}",
            rings.len(),
            surface_type
        );
        self.finish_generated_triangulation(name, all_verts, all_faces, surface_type)
    }
}

/// Diagnose why breakline edge `edge_index` of `rings[ring_index]` failed to
/// insert as a CDT constraint. Scans every other edge across all rings for a
/// direct geometric conflict (spade doesn't say which edge or why it
/// conflicted) and describes the first one found: a crossing point (with
/// each edge's interpolated Z, to show whether it's even representable by a
/// single-valued terrain), a collinear overlap, or near-but-not-exactly
/// coincident endpoints — the common case when two breaklines were meant to
/// share a boundary but were digitized independently.
fn diagnose_breakline_conflict(
    rings: &[Vec<glam::DVec3>],
    ring_index: usize,
    edge_index: usize,
) -> String {
    let ring = &rings[ring_index];
    let n = ring.len();
    let a = ring[edge_index];
    let b = ring[(edge_index + 1) % n];
    for (other_ring_index, other_ring) in rings.iter().enumerate() {
        let m = other_ring.len();
        for other_edge_index in 0..m {
            if other_ring_index == ring_index && other_edge_index == edge_index {
                continue;
            }
            let c = other_ring[other_edge_index];
            let d = other_ring[(other_edge_index + 1) % m];
            // Edges that legitimately share an endpoint (adjacent edges
            // within a ring, or two rings meeting at a shared vertex) are
            // not conflicts — skip them so a real conflict elsewhere isn't
            // shadowed by this expected topology.
            let shares_endpoint = points_coincident(a, c)
                || points_coincident(a, d)
                || points_coincident(b, c)
                || points_coincident(b, d);
            if shares_endpoint {
                continue;
            }
            if let Some(detail) = describe_edge_conflict(a, b, c, d) {
                return format!(
                    "breakline {ring_index} edge {edge_index} ({a:.3}->{b:.3}) vs breakline {other_ring_index} edge {other_edge_index} ({c:.3}->{d:.3}): {detail}"
                );
            }
        }
    }
    format!(
        "breakline {ring_index} edge {edge_index} ({a:.3}->{b:.3}): no conflicting edge found by direct geometric scan (likely a near-degenerate numerical case)"
    )
}

/// Classify how segment `a->b` conflicts with segment `c->d` in the XY plane,
/// if at all. `None` means these two edges specifically don't touch (the
/// real conflict is with some other edge).
fn describe_edge_conflict(
    a: glam::DVec3,
    b: glam::DVec3,
    c: glam::DVec3,
    d: glam::DVec3,
) -> Option<String> {
    let (a2, b2, c2, d2) = (a.truncate(), b.truncate(), c.truncate(), d.truncate());
    let r = b2 - a2;
    let s = d2 - c2;
    let denom = r.x * s.y - r.y * s.x;
    const NEAR_ZERO: f64 = 1e-9;
    // Strict interior, matching `seg_seg_intersection_2d`: a crossing exactly
    // at an endpoint is a legitimate T-junction (spade handles these fine),
    // not the kind of conflict that fails `try_add_constraint`.
    const EPS: f64 = 1e-8;

    if denom.abs() > NEAR_ZERO {
        let t = ((c2.x - a2.x) * s.y - (c2.y - a2.y) * s.x) / denom;
        let u = ((c2.x - a2.x) * r.y - (c2.y - a2.y) * r.x) / denom;
        if t > EPS && t < 1.0 - EPS && u > EPS && u < 1.0 - EPS {
            return Some(describe_crossing(a, b, c, d, t, u, "cross"));
        }
        if (-EPS..=1.0 + EPS).contains(&t) && (-EPS..=1.0 + EPS).contains(&u) {
            let a_endpoint = !(EPS..=1.0 - EPS).contains(&t);
            let b_endpoint = !(EPS..=1.0 - EPS).contains(&u);
            if a_endpoint != b_endpoint {
                return Some(describe_crossing(a, b, c, d, t, u, "touch"));
            }
        }
        return nearest_endpoint_gap(a, b, c, d);
    }

    // Parallel or anti-parallel: check collinearity, then parameter overlap.
    let r_len = r.length();
    if r_len < 1e-9 {
        return None;
    }
    let cross_ac = r.x * (c2.y - a2.y) - r.y * (c2.x - a2.x);
    let perp_dist = cross_ac.abs() / r_len;
    if perp_dist > 1e-6 {
        return nearest_endpoint_gap(a, b, c, d);
    }
    let t_c = (c2 - a2).dot(r) / r_len.powi(2);
    let t_d = (d2 - a2).dot(r) / r_len.powi(2);
    let (lo, hi) = (t_c.min(t_d), t_c.max(t_d));
    if hi > 1e-6 && lo < 1.0 - 1e-6 {
        Some(format!(
            "collinear and overlapping along the same line for parameter range [{:.3}, {:.3}] of edge A \u{2014} these two breaklines run along the same wall without sharing vertices, so the CDT can't insert both without splitting them",
            lo.max(0.0),
            hi.min(1.0)
        ))
    } else {
        nearest_endpoint_gap(a, b, c, d)
    }
}

fn describe_crossing(
    a: glam::DVec3,
    b: glam::DVec3,
    c: glam::DVec3,
    d: glam::DVec3,
    t: f64,
    u: f64,
    relation: &str,
) -> String {
    let point = a.truncate() + t * (b.truncate() - a.truncate());
    let z_a = a.z + t * (b.z - a.z);
    let z_c = c.z + u * (d.z - c.z);
    let representable = if (z_a - z_c).abs() > 1e-3 {
        "different elevations: not representable by a single-valued terrain surface"
    } else {
        "same elevation: could be split at this point"
    };
    format!(
        "{relation} in XY at ({:.3}, {:.3}); edge A's Z there is {z_a:.3}, edge B's Z there is {z_c:.3} ({representable})",
        point.x, point.y
    )
}

/// If the closest pair of endpoints between the two edges is suspiciously
/// close (but not exactly coincident), report the gap — this is the
/// signature of two breaklines that were meant to share a vertex but were
/// digitized independently and differ by floating-point/snap noise.
fn nearest_endpoint_gap(
    a: glam::DVec3,
    b: glam::DVec3,
    c: glam::DVec3,
    d: glam::DVec3,
) -> Option<String> {
    const NEAR_MISS: f64 = 0.05; // 5cm: plausible "meant to be the same vertex" gap
    [
        (a, c, "A.start~B.start"),
        (a, d, "A.start~B.end"),
        (b, c, "A.end~B.start"),
        (b, d, "A.end~B.end"),
    ]
    .into_iter()
    .map(|(p, q, label)| ((p.truncate() - q.truncate()).length(), (p.z - q.z).abs(), label))
    .filter(|(gap, ..)| *gap < NEAR_MISS)
    .min_by(|x, y| x.0.total_cmp(&y.0))
    .map(|(gap, dz, label)| {
        format!(
            "nearest endpoints ({label}) are {gap:.4} apart in XY (Z differs by {dz:.4}) \u{2014} likely meant to be the same shared vertex but digitized independently"
        )
    })
}

fn conflicting_z_detail(
    a: glam::DVec3,
    b: glam::DVec3,
    c: glam::DVec3,
    d: glam::DVec3,
) -> Option<String> {
    let (a2, b2, c2, d2) = (a.truncate(), b.truncate(), c.truncate(), d.truncate());
    let r = b2 - a2;
    let s = d2 - c2;
    let denom = r.x * s.y - r.y * s.x;
    const EPS: f64 = 1e-8;
    const Z_EPS: f64 = 1e-3;

    if denom.abs() > 1e-9 {
        let t = ((c2.x - a2.x) * s.y - (c2.y - a2.y) * s.x) / denom;
        let u = ((c2.x - a2.x) * r.y - (c2.y - a2.y) * r.x) / denom;
        if (-EPS..=1.0 + EPS).contains(&t) && (-EPS..=1.0 + EPS).contains(&u) {
            let z_a = a.z + t * (b.z - a.z);
            let z_c = c.z + u * (d.z - c.z);
            if (z_a - z_c).abs() > Z_EPS {
                return Some(describe_crossing(a, b, c, d, t, u, "cross"));
            }
        }
        return None;
    }

    let r_len_sq = r.length_squared();
    if r_len_sq < 1e-18 {
        return None;
    }
    let perp_dist = (r.x * (c2.y - a2.y) - r.y * (c2.x - a2.x)).abs() / r_len_sq.sqrt();
    if perp_dist > 1e-6 {
        return None;
    }

    let t_c = (c2 - a2).dot(r) / r_len_sq;
    let t_d = (d2 - a2).dot(r) / r_len_sq;
    let lo = t_c.min(t_d).max(0.0);
    let hi = t_c.max(t_d).min(1.0);
    if hi <= lo + EPS {
        return None;
    }

    for t in [lo, hi] {
        let point = a2 + t * r;
        let u = point_on_segment_parameter(point, c2, d2);
        let z_a = a.z + t * (b.z - a.z);
        let z_c = c.z + u * (d.z - c.z);
        if (z_a - z_c).abs() > Z_EPS {
            return Some(format!(
                "collinear overlap has conflicting elevations near ({:.3}, {:.3}); edge A's Z is {z_a:.3}, edge B's Z is {z_c:.3}",
                point.x, point.y
            ));
        }
    }

    None
}

fn point_on_segment_parameter(point: glam::DVec2, a: glam::DVec2, b: glam::DVec2) -> f64 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-18 {
        0.0
    } else {
        ((point - a).dot(ab) / len_sq).clamp(0.0, 1.0)
    }
}

fn validate_breakline_edge_z(
    rings: &[Vec<glam::DVec3>],
    ring_index: usize,
    edge_index: usize,
) -> Result<()> {
    let ring = &rings[ring_index];
    let n = ring.len();
    let a = ring[edge_index];
    let b = ring[(edge_index + 1) % n];

    for (other_ring_index, other_ring) in rings.iter().enumerate() {
        let m = other_ring.len();
        for other_edge_index in 0..m {
            if other_ring_index == ring_index && other_edge_index == edge_index {
                continue;
            }
            let c = other_ring[other_edge_index];
            let d = other_ring[(other_edge_index + 1) % m];
            if let Some(detail) = conflicting_z_detail(a, b, c, d) {
                anyhow::bail!(
                    "Selected breakline edges intersect in XY at conflicting elevations and cannot form a single-valued terrain surface (breakline {ring_index} edge {edge_index} ({a:.3}->{b:.3}) vs breakline {other_ring_index} edge {other_edge_index} ({c:.3}->{d:.3}): {detail})"
                );
            }
        }
    }

    Ok(())
}

fn interpolate_z_on_edge(a: glam::DVec3, b: glam::DVec3, point: glam::DVec2) -> f64 {
    let ab = b.truncate() - a.truncate();
    let len_sq = ab.length_squared();
    if len_sq < 1e-18 {
        a.z
    } else {
        let t = ((point - a.truncate()).dot(ab) / len_sq).clamp(0.0, 1.0);
        a.z + t * (b.z - a.z)
    }
}

pub(super) fn cdt_surface_from_breaklines(
    rings: &[Vec<glam::DVec3>],
) -> Result<(Vec<tri00t::Vertex>, Vec<[u32; 3]>)> {
    use spade::{ConstrainedDelaunayTriangulation, Point2};

    if rings.is_empty() {
        anyhow::bail!("No breakline rings supplied");
    }

    let mut cdt: ConstrainedDelaunayTriangulation<Point2<f64>> =
        ConstrainedDelaunayTriangulation::new();
    let mut handle_z: HashMap<usize, f64> = HashMap::new();

    for (ring_index, ring) in rings.iter().enumerate() {
        if ring.len() < 3 {
            anyhow::bail!("A selected breakline has fewer than 3 vertices");
        }

        let mut handles = Vec::with_capacity(ring.len());
        for point in ring {
            if !point.is_finite() {
                anyhow::bail!("A selected breakline contains non-finite coordinates");
            }
            let handle = cdt
                .insert(Point2::new(point.x, point.y))
                .map_err(|error| anyhow::anyhow!("CDT insert failed: {error:?}"))?;
            if let Some(existing_z) = handle_z.insert(handle.index(), point.z)
                && (existing_z - point.z).abs() > 1e-7
            {
                anyhow::bail!(
                    "Selected breaklines contain the same XY point at conflicting elevations ({existing_z:.3} and {:.3})",
                    point.z
                );
            }
            handles.push(handle);
        }

        for i in 0..handles.len() {
            let a = handles[i];
            let b = handles[(i + 1) % handles.len()];
            if a == b {
                continue;
            }
            validate_breakline_edge_z(rings, ring_index, i)?;

            let edge_start = ring[i];
            let edge_end = ring[(i + 1) % ring.len()];
            let constraint_edges = cdt.add_constraint_and_split(a, b, |point| point);
            if constraint_edges.is_empty() {
                anyhow::bail!(
                    "Selected breakline edges intersect or overlap in XY and cannot form a terrain surface ({})",
                    diagnose_breakline_conflict(rings, ring_index, i)
                );
            }
            for edge in constraint_edges {
                let edge = cdt.directed_edge(edge);
                for vertex in [edge.from(), edge.to()] {
                    let index = vertex.fix().index();
                    handle_z.entry(index).or_insert_with(|| {
                        let position = vertex.position();
                        interpolate_z_on_edge(
                            edge_start,
                            edge_end,
                            glam::DVec2::new(position.x, position.y),
                        )
                    });
                }
            }
        }
    }

    let mut indexed: Vec<(usize, f64, f64, f64)> = cdt
        .vertices()
        .map(|vertex| {
            let index = vertex.fix().index();
            let position = vertex.position();
            let z = handle_z.get(&index).copied().ok_or_else(|| {
                anyhow::anyhow!(
                    "CDT introduced an elevation-less vertex while resolving breaklines"
                )
            })?;
            Ok((index, position.x, position.y, z))
        })
        .collect::<Result<Vec<_>>>()?;
    indexed.sort_unstable_by_key(|(index, ..)| *index);

    let index_map: HashMap<usize, u32> = indexed
        .iter()
        .enumerate()
        .map(|(output_index, (spade_index, ..))| (*spade_index, output_index as u32))
        .collect();
    let vertices = indexed
        .iter()
        .map(|(_, x, y, z)| tri00t::Vertex::new(*x, *y, *z))
        .collect();

    let mut faces = Vec::new();
    for face in cdt.inner_faces() {
        let face_vertices = face.vertices();
        let positions = face_vertices.map(|vertex| vertex.position());
        let centroid = glam::DVec2::new(
            (positions[0].x + positions[1].x + positions[2].x) / 3.0,
            (positions[0].y + positions[1].y + positions[2].y) / 3.0,
        );
        if !rings
            .iter()
            .any(|ring| crate::model::geometry::point_in_polygon_xy(centroid, ring))
        {
            continue;
        }

        let twice_area = (positions[1].x - positions[0].x) * (positions[2].y - positions[0].y)
            - (positions[1].y - positions[0].y) * (positions[2].x - positions[0].x);
        if twice_area.abs() <= 1e-12 {
            continue;
        }

        let mut triangle = face_vertices.map(|vertex| index_map[&vertex.fix().index()]);
        if twice_area < 0.0 {
            triangle.swap(1, 2);
        }
        faces.push(triangle);
    }

    if faces.is_empty() {
        anyhow::bail!("Constrained surface triangulation produced no faces");
    }

    Ok((vertices, faces))
}

/// Close a constrained terrain surface at each design's outer boundary level.
///
/// For a pit, the terrain is the lower shell and the closure cap lies above it.
/// For a stockpile, the terrain is the upper shell and the closure cap lies
/// below it. Nested rings remain terrain breaklines, not internal walls.
pub(super) fn closed_solid_from_breaklines(
    rings: &[Vec<glam::DVec3>],
) -> Result<(Vec<tri00t::Vertex>, Vec<[u32; 3]>)> {
    let (vertices, mut faces) = cdt_surface_from_breaklines(rings)?;
    let roots = outer_breakline_indices(rings);
    let surface_indices: HashMap<(u64, u64, u64), u32> = vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| {
            (
                (vertex.x.to_bits(), vertex.y.to_bits(), vertex.z.to_bits()),
                index as u32,
            )
        })
        .collect();

    for root_index in roots {
        let ring = &rings[root_index];
        let group_rings: Vec<&Vec<glam::DVec3>> = rings
            .iter()
            .filter(|candidate| {
                std::ptr::eq(*candidate, ring)
                    || crate::model::geometry::point_in_polygon_xy(candidate[0].truncate(), ring)
            })
            .collect();
        let closure_z = ring.iter().map(|point| point.z).sum::<f64>() / ring.len() as f64;
        let outer_z_span = ring
            .iter()
            .map(|point| point.z)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), z| {
                (min.min(z), max.max(z))
            });
        if outer_z_span.1 - outer_z_span.0 > 1e-7 {
            anyhow::bail!(
                "A closed solid requires each outer boundary to have one constant elevation"
            );
        }

        let group_min_z = group_rings
            .iter()
            .flat_map(|group_ring| group_ring.iter())
            .map(|point| point.z)
            .fold(f64::INFINITY, f64::min);
        let group_max_z = group_rings
            .iter()
            .flat_map(|group_ring| group_ring.iter())
            .map(|point| point.z)
            .fold(f64::NEG_INFINITY, f64::max);
        let extends_below = group_min_z < closure_z - 1e-8;
        let extends_above = group_max_z > closure_z + 1e-8;
        if extends_below == extends_above {
            if extends_below {
                anyhow::bail!(
                    "A solid design cannot extend both above and below its outer boundary elevation"
                );
            }
            anyhow::bail!("Each closed solid group requires breaklines at more than one elevation");
        }
        let is_pit = extends_below;

        let boundary_indices: Vec<u32> = ring
            .iter()
            .map(|point| {
                surface_indices
                    .get(&(point.x.to_bits(), point.y.to_bits(), point.z.to_bits()))
                    .copied()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Outer breakline vertex is missing from the constrained surface"
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        // CDT terrain faces initially point upward. A pit's terrain is the
        // bottom of the solid, so its faces must point downward.
        if is_pit {
            for face in &mut faces {
                let centroid = face
                    .iter()
                    .map(|index| {
                        let vertex = vertices[*index as usize];
                        glam::DVec2::new(vertex.x, vertex.y)
                    })
                    .sum::<glam::DVec2>()
                    / 3.0;
                if crate::model::geometry::point_in_polygon_xy(centroid, ring) {
                    face.swap(1, 2);
                }
            }
        }

        let flat: Vec<f64> = ring.iter().flat_map(|point| [point.x, point.y]).collect();
        let cap_triangles = earcutr::earcut(&flat, &[], 2)
            .map_err(|error| anyhow::anyhow!("Failed to triangulate solid closure: {error}"))?;
        for triangle in cap_triangles.chunks_exact(3) {
            let mut face = [
                boundary_indices[triangle[0]],
                boundary_indices[triangle[1]],
                boundary_indices[triangle[2]],
            ];
            let cap_should_point_up = is_pit;
            if (triangle_signed_xy_area(&vertices, face) > 0.0) != cap_should_point_up {
                face.swap(1, 2);
            }
            faces.push(face);
        }
    }

    Ok((vertices, faces))
}
pub(super) fn outer_breakline_indices(rings: &[Vec<glam::DVec3>]) -> Vec<usize> {
    rings
        .iter()
        .enumerate()
        .filter_map(|(index, ring)| {
            let probe = ring[0].truncate();
            let contained = rings.iter().enumerate().any(|(other_index, other)| {
                other_index != index
                    && signed_area_xy(other).abs() > signed_area_xy(ring).abs()
                    && crate::model::geometry::point_in_polygon_xy(probe, other)
            });
            (!contained).then_some(index)
        })
        .collect()
}

pub(super) fn triangle_signed_xy_area(vertices: &[tri00t::Vertex], face: [u32; 3]) -> f64 {
    let a = vertices[face[0] as usize];
    let b = vertices[face[1] as usize];
    let c = vertices[face[2] as usize];
    ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) * 0.5
}

/// Triangulate the interior of a closed ring using CDT.
/// `flip_winding` reverses triangle winding (use to control face normal direction).
pub(super) fn cdt_fill_ring(
    ring: &[glam::DVec3],
    flip_winding: bool,
) -> Result<(Vec<tri00t::Vertex>, Vec<[u32; 3]>)> {
    use spade::{ConstrainedDelaunayTriangulation, Point2};

    if ring.len() < 3 {
        anyhow::bail!("A selected polygon has fewer than 3 vertices");
    }

    let mut cdt: ConstrainedDelaunayTriangulation<Point2<f64>> =
        ConstrainedDelaunayTriangulation::new();
    let mut handles = Vec::new();
    let mut handle_z: HashMap<usize, f64> = HashMap::new();

    for v in ring {
        if !v.is_finite() {
            anyhow::bail!("A selected polygon contains non-finite coordinates");
        }
        let h = cdt
            .insert(Point2::new(v.x, v.y))
            .map_err(|error| anyhow::anyhow!("CDT insert failed: {error:?}"))?;
        if let Some(existing_z) = handle_z.insert(h.index(), v.z)
            && (existing_z - v.z).abs() > 1e-7
        {
            anyhow::bail!(
                "Selected polygon contains the same XY point at conflicting elevations ({existing_z:.3} and {:.3})",
                v.z
            );
        }
        handles.push(h);
    }
    for i in 0..ring.len() {
        let j = (i + 1) % ring.len();
        let (ha, hb) = (handles[i], handles[j]);
        if ha != hb && cdt.try_add_constraint(ha, hb).is_empty() {
            anyhow::bail!(
                "Selected polygon edges intersect or overlap in XY and cannot form a terrain surface"
            );
        }
    }

    let mut indexed: Vec<(usize, f64, f64, f64)> = cdt
        .vertices()
        .map(|v| {
            let idx = v.fix().index();
            let p = v.position();
            (
                idx,
                p.x,
                p.y,
                handle_z.get(&idx).copied().unwrap_or(ring[0].z),
            )
        })
        .collect();
    indexed.sort_unstable_by_key(|(idx, ..)| *idx);
    let verts: Vec<tri00t::Vertex> = indexed
        .iter()
        .map(|(_, x, y, z)| tri00t::Vertex::new(*x, *y, *z))
        .collect();

    // Filter to faces whose centroid lies inside the input ring so that concave
    // polygons don't include CDT faces outside the boundary.
    let ring_xy: Vec<(f64, f64)> = ring.iter().map(|v| (v.x, v.y)).collect();
    let faces: Vec<[u32; 3]> = cdt
        .inner_faces()
        .filter(|f| {
            let vs = f.vertices();
            let cx = (vs[0].position().x + vs[1].position().x + vs[2].position().x) / 3.0;
            let cy = (vs[0].position().y + vs[1].position().y + vs[2].position().y) / 3.0;
            point_in_polygon_xy(cx, cy, &ring_xy)
        })
        .map(|f| {
            let v = f.vertices();
            let [a, b, c] = [
                v[0].fix().index() as u32,
                v[1].fix().index() as u32,
                v[2].fix().index() as u32,
            ];
            if flip_winding { [a, c, b] } else { [a, b, c] }
        })
        .collect();

    if faces.is_empty() {
        anyhow::bail!("Failed to triangulate polygon (may be degenerate or collinear)");
    }
    Ok((verts, faces))
}

/// Signed XY area of a ring (shoelace); positive means CCW viewed from above.
pub(super) fn signed_area_xy(ring: &[glam::DVec3]) -> f64 {
    let n = ring.len();
    (0..n)
        .map(|i| {
            let p = ring[i];
            let q = ring[(i + 1) % n];
            p.x * q.y - q.x * p.y
        })
        .sum::<f64>()
        * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    #[test]
    fn diagnose_reports_crossing_with_differing_elevations() {
        let a = DVec3::new(0.0, 0.0, 10.0);
        let b = DVec3::new(10.0, 10.0, 20.0);
        let c = DVec3::new(0.0, 10.0, 50.0);
        let d = DVec3::new(10.0, 0.0, 60.0);
        let detail = describe_edge_conflict(a, b, c, d).expect("edges cross at (5,5)");
        assert!(detail.contains("cross in XY at (5.000, 5.000)"), "{detail}");
        assert!(detail.contains("not representable"), "{detail}");
    }

    #[test]
    fn diagnose_reports_collinear_overlap() {
        let a = DVec3::new(0.0, 0.0, 0.0);
        let b = DVec3::new(10.0, 0.0, 0.0);
        let c = DVec3::new(5.0, 0.0, 0.0);
        let d = DVec3::new(15.0, 0.0, 0.0);
        let detail = describe_edge_conflict(a, b, c, d).expect("segments overlap along y=0");
        assert!(detail.contains("collinear and overlapping"), "{detail}");
    }

    #[test]
    fn diagnose_reports_near_miss_shared_vertex() {
        let a = DVec3::new(0.0, 0.0, 5.0);
        let b = DVec3::new(10.0, 0.0, 5.0);
        // c starts 1cm from a's endpoint (independent digitizing noise) and
        // runs off to the side without ever crossing a-b's line, so this
        // pair doesn't cross or overlap — only the near-miss endpoint should
        // be flagged.
        let c = DVec3::new(0.01, 0.02, 5.0);
        let d = DVec3::new(0.01, 5.0, 5.0);
        let detail = describe_edge_conflict(a, b, c, d).expect("endpoints are a near miss");
        assert!(detail.contains("A.start~B.start"), "{detail}");
        assert!(detail.contains("0.0224"), "{detail}");
    }

    #[test]
    fn diagnose_reports_endpoint_t_junction() {
        let a = DVec3::new(0.0, 5.0, 10.0);
        let b = DVec3::new(5.0, 0.0, 10.0);
        let c = DVec3::new(0.0, 0.0, 10.0);
        let d = DVec3::new(10.0, 0.0, 10.0);
        let detail = describe_edge_conflict(a, b, c, d).expect("edge endpoint lands on other edge");
        assert!(detail.contains("touch in XY at (5.000, 0.000)"), "{detail}");
        assert!(detail.contains("same elevation"), "{detail}");
    }

    #[test]
    fn diagnose_reports_none_for_unrelated_edges() {
        let a = DVec3::new(0.0, 0.0, 0.0);
        let b = DVec3::new(1.0, 0.0, 0.0);
        let c = DVec3::new(0.0, 100.0, 0.0);
        let d = DVec3::new(1.0, 100.0, 0.0);
        assert!(describe_edge_conflict(a, b, c, d).is_none());
    }

    #[test]
    fn diagnose_breakline_conflict_ignores_adjacent_shared_vertices() {
        // A normal ring: every edge shares a vertex with its neighbors. That
        // must never be reported as a conflict by itself.
        let ring = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(10.0, 0.0, 0.0),
            DVec3::new(10.0, 10.0, 0.0),
            DVec3::new(0.0, 10.0, 0.0),
        ];
        for edge_index in 0..ring.len() {
            let detail = diagnose_breakline_conflict(std::slice::from_ref(&ring), 0, edge_index);
            assert!(
                detail.contains("no conflicting edge found"),
                "edge {edge_index}: {detail}"
            );
        }
    }

    #[test]
    fn diagnose_breakline_conflict_finds_real_crossing_past_the_neighbor() {
        // Ring 0's edge 0 (v0->v1, along x=y) genuinely crosses ring 1's
        // edge 1 at (5,5), at different elevations. Edge 0's ring-0
        // neighbors (sharing a vertex with it) must be skipped rather than
        // reported as a false cross, and ring 1's own non-conflicting edges
        // (0 and 2, scanned first) must not shadow the real conflict.
        let ring0 = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(10.0, 10.0, 10.0),
            DVec3::new(10.0, -10.0, 10.0),
            DVec3::new(0.0, -10.0, 0.0),
        ];
        let ring1 = vec![
            DVec3::new(100.0, 100.0, 999.0),
            DVec3::new(10.0, 0.0, 100.0),
            DVec3::new(0.0, 10.0, 200.0),
        ];
        let rings = vec![ring0, ring1];
        let detail = diagnose_breakline_conflict(&rings, 0, 0);
        assert!(detail.contains("breakline 1 edge 1"), "{detail}");
        assert!(detail.contains("cross in XY at (5.000, 5.000)"), "{detail}");
        assert!(detail.contains("not representable"), "{detail}");
    }

    #[test]
    fn cdt_surface_allows_same_elevation_t_junction_from_pit_design() {
        let ring10 = vec![
            DVec3::new(194184.80804870004, 7954694.601165758, 276.0),
            DVec3::new(194405.8730665064, 7954722.770923374, 276.0),
            DVec3::new(194420.76056342237, 7954736.312696906, 276.0),
            DVec3::new(194454.23337978532, 7954706.282106807, 276.0),
            DVec3::new(194528.97078607292, 7954774.263785397, 276.0),
            DVec3::new(194624.11956229017, 7954779.582434339, 276.0),
            DVec3::new(194768.9848301585, 7954732.851136622, 276.0),
            DVec3::new(194881.47319776443, 7954700.08270376, 276.0),
            DVec3::new(194936.14383465287, 7954714.860250081, 276.0),
            DVec3::new(194954.47283579985, 7954747.731592522, 276.0),
            DVec3::new(194959.1678678599, 7954862.810163006, 276.0),
            DVec3::new(194943.3400341604, 7954913.782532665, 276.0),
            DVec3::new(194906.1137775063, 7954936.205339909, 276.0),
            DVec3::new(194823.6881786337, 7954946.606295587, 276.0),
            DVec3::new(194707.81924784035, 7954936.670062728, 276.0),
            DVec3::new(194443.86456095657, 7954948.020896293, 276.0),
            DVec3::new(194223.3646920947, 7954931.290760769, 276.0),
            DVec3::new(194073.2201442615, 7954891.972778969, 276.0),
            DVec3::new(194044.49549790577, 7954858.126836276, 276.0),
            DVec3::new(194038.02719282577, 7954796.221042206, 276.0),
            DVec3::new(194059.7238864849, 7954739.6131420545, 276.0),
        ];
        let ring11 = vec![
            DVec3::new(194185.70593926773, 7954702.780271191, 276.0),
            DVec3::new(194416.18359982956, 7954732.14945813, 276.0),
            DVec3::new(194420.76056342237, 7954736.312696906, 276.0),
            DVec3::new(194454.23337978532, 7954706.282106807, 276.0),
            DVec3::new(194538.35629049933, 7954782.800907261, 276.0),
            DVec3::new(194625.1591209801, 7954787.653032542, 276.0),
            DVec3::new(194771.3320543625, 7954740.499902129, 276.0),
            DVec3::new(194881.55408042885, 7954708.39166607, 276.0),
            DVec3::new(194930.79960886977, 7954721.702798316, 276.0),
            DVec3::new(194946.55720058328, 7954749.962559397, 276.0),
            DVec3::new(194951.1182262184, 7954861.756537226, 276.0),
            DVec3::new(194936.60315496929, 7954908.501252351, 276.0),
            DVec3::new(194903.4345478952, 7954928.479981072, 276.0),
            DVec3::new(194823.5273835979, 7954938.563145593, 276.0),
            DVec3::new(194707.989860356, 7954928.655332282, 276.0),
            DVec3::new(194443.99579456248, 7954940.007859257, 276.0),
            DVec3::new(194224.6913830506, 7954923.368427475, 276.0),
            DVec3::new(194077.6872139305, 7954884.872808891, 276.0),
            DVec3::new(194052.1951831431, 7954854.835822448, 276.0),
            DVec3::new(194046.18297817255, 7954797.295196548, 276.0),
            DVec3::new(194065.8819988394, 7954745.899338151, 276.0),
        ];

        let result = cdt_surface_from_breaklines(&[ring10, ring11]);
        assert!(result.is_ok(), "{result:?}");
    }

    #[test]
    fn cdt_surface_rejects_t_junction_at_conflicting_elevation() {
        let ring0 = vec![
            DVec3::new(0.0, 0.0, 10.0),
            DVec3::new(10.0, 0.0, 10.0),
            DVec3::new(10.0, 10.0, 10.0),
            DVec3::new(0.0, 10.0, 10.0),
        ];
        let ring1 = vec![
            DVec3::new(5.0, -5.0, 20.0),
            DVec3::new(5.0, 0.0, 20.0),
            DVec3::new(6.0, -5.0, 20.0),
        ];

        let err = cdt_surface_from_breaklines(&[ring0, ring1]).expect_err("conflicting T-junction");
        let message = format!("{err:#}");
        assert!(message.contains("conflicting elevations"), "{message}");
    }
}
