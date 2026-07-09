//! Terrain TIN generation from survey point clouds.

use super::*;
use crate::model::point_cloud::PointCloudId;
use glam::DVec3;

impl<'a> App<'a> {
    /// Build an open XY Delaunay terrain surface from a loaded point cloud on a
    /// background job and register it like any generated triangulation.
    pub(crate) fn run_point_cloud_tin(
        &mut self,
        cloud_id: PointCloudId,
        name: String,
        max_edge: f64,
        max_points: u32,
    ) -> Result<()> {
        let cloud = self
            .point_clouds
            .iter()
            .find(|cloud| cloud.id == cloud_id)
            .ok_or_else(|| anyhow::anyhow!("The selected point cloud is no longer loaded"))?;
        let points = cloud.points.clone();
        let compute = move |cancel: &crate::app::jobs::CancelFlag|
              -> Result<crate::model::triangulation::GeneratedTriangulation> {
            reconstruct_terrain_tin_from_point_cloud(&points, name, max_edge, max_points, cancel)
        };
        let apply =
            move |app: &mut App,
                  result: Result<crate::model::triangulation::GeneratedTriangulation>| {
                match result {
                    Ok(generated) => app.insert_generated_triangulation(generated),
                    Err(error) => {
                        userspace_warn!("Point cloud TIN failed: {error:#}");
                    }
                }
            };
        self.spawn_job(
            "Point cloud TIN...",
            crate::app::jobs::JobKey::Anonymous,
            compute,
            apply,
        );
        Ok(())
    }
}

fn reconstruct_terrain_tin_from_point_cloud(
    points: &[DVec3],
    name: String,
    max_edge: f64,
    max_points: u32,
    cancel: &crate::app::jobs::CancelFlag,
) -> Result<crate::model::triangulation::GeneratedTriangulation> {
    if points.len() < 3 {
        anyhow::bail!("The point cloud has too few points to triangulate a terrain surface");
    }

    let max_points = (max_points as usize).max(3);
    let sampled = spatial_grid_subsample_terrain(points, max_points, cancel)?;
    if sampled.len() < points.len() {
        userspace_log!(
            "Terrain TIN: spatially subsampled {} of {} points",
            sampled.len(),
            points.len()
        );
    }
    reconstruct_terrain_tin(sampled, name, max_edge, cancel)
}

fn reconstruct_terrain_tin(
    sampled: Vec<DVec3>,
    name: String,
    max_edge: f64,
    cancel: &crate::app::jobs::CancelFlag,
) -> Result<crate::model::triangulation::GeneratedTriangulation> {
    use spade::{DelaunayTriangulation, Point2, Triangulation as _};

    if sampled.len() < 3 {
        anyhow::bail!("The point cloud has too few points to triangulate a terrain surface");
    }

    let mut tin: DelaunayTriangulation<Point2<f64>> = DelaunayTriangulation::new();
    let mut handle_z: std::collections::HashMap<usize, (f64, u32)> =
        std::collections::HashMap::new();

    for (index, point) in sampled.iter().enumerate() {
        if index % 4096 == 0 && cancel.is_cancelled() {
            anyhow::bail!("Terrain TIN reconstruction cancelled");
        }
        if !point.is_finite() {
            continue;
        }
        let handle = tin
            .insert(Point2::new(point.x, point.y))
            .map_err(|error| anyhow::anyhow!("Terrain TIN insert failed: {error:?}"))?;
        let entry = handle_z.entry(handle.index()).or_insert((0.0, 0));
        entry.0 += point.z;
        entry.1 += 1;
    }

    if tin.num_vertices() < 3 {
        anyhow::bail!("The point cloud has fewer than 3 unique XY points");
    }

    let mut indexed: Vec<(usize, f64, f64, f64)> = tin
        .vertices()
        .map(|vertex| {
            let index = vertex.fix().index();
            let position = vertex.position();
            let (z_sum, count) = handle_z
                .get(&index)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("Terrain TIN vertex has no source elevation"))?;
            Ok((index, position.x, position.y, z_sum / count.max(1) as f64))
        })
        .collect::<Result<Vec<_>>>()?;
    indexed.sort_unstable_by_key(|(index, ..)| *index);

    let index_map: std::collections::HashMap<usize, u32> = indexed
        .iter()
        .enumerate()
        .map(|(output_index, (spade_index, ..))| (*spade_index, output_index as u32))
        .collect();
    let vertices: Vec<tri00t::Vertex> = indexed
        .iter()
        .map(|(_, x, y, z)| tri00t::Vertex::new(*x, *y, *z))
        .collect();

    let max_edge_sq = (max_edge > 0.0).then_some(max_edge * max_edge);
    let mut faces = Vec::new();
    for face in tin.inner_faces() {
        let face_vertices = face.vertices();
        let positions = face_vertices.map(|vertex| vertex.position());
        let edge_sq = [
            point2_distance_sq(positions[1], positions[0]),
            point2_distance_sq(positions[2], positions[1]),
            point2_distance_sq(positions[0], positions[2]),
        ];
        if max_edge_sq.is_some_and(|limit| edge_sq.iter().any(|distance| *distance > limit)) {
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
        anyhow::bail!("Terrain TIN produced no faces - increase Max edge or use more input points");
    }

    userspace_log!(
        "Terrain TIN: triangulated {} unique XY points into {} faces{}",
        vertices.len(),
        faces.len(),
        if max_edge > 0.0 {
            format!(" (max edge {max_edge:.3})")
        } else {
            " (max edge disabled)".to_owned()
        }
    );
    super::session::build_generated_triangulation(
        name,
        vertices,
        faces,
        TriSurfaceType::Surface,
        crate::model::triangulation::unique_edges,
    )
}

fn point2_distance_sq(a: spade::Point2<f64>, b: spade::Point2<f64>) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy
}

fn spatial_grid_subsample_terrain(
    points: &[DVec3],
    max_points: usize,
    cancel: &crate::app::jobs::CancelFlag,
) -> Result<Vec<DVec3>> {
    if points.len() <= max_points {
        return Ok(points.to_vec());
    }

    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    let mut finite_count = 0usize;
    for (index, point) in points.iter().enumerate() {
        if index % 262_144 == 0 && cancel.is_cancelled() {
            anyhow::bail!("Terrain TIN reconstruction cancelled");
        }
        if point.is_finite() {
            min = min.min(*point);
            max = max.max(*point);
            finite_count += 1;
        }
    }
    if finite_count < 3 {
        anyhow::bail!("The point cloud has fewer than 3 finite points");
    }

    let extent = max - min;
    let area = (extent.x * extent.y).abs();
    if area <= f64::EPSILON {
        return Ok(jittered_subsample(points, max_points));
    }

    let target_cells = (max_points as f64 * 0.75).max(3.0);
    let cell_size = (area / target_cells).sqrt().max(1.0e-9);
    let mut cells: std::collections::HashMap<(i64, i64), (DVec3, u32)> =
        std::collections::HashMap::with_capacity(max_points.min(points.len()));

    for (index, point) in points.iter().enumerate() {
        if index % 262_144 == 0 && cancel.is_cancelled() {
            anyhow::bail!("Terrain TIN reconstruction cancelled");
        }
        if !point.is_finite() {
            continue;
        }
        let key = (
            ((point.x - min.x) / cell_size).floor() as i64,
            ((point.y - min.y) / cell_size).floor() as i64,
        );
        let entry = cells.entry(key).or_insert((DVec3::ZERO, 0));
        entry.0 += *point;
        entry.1 += 1;
    }

    let mut sampled: Vec<DVec3> = cells
        .into_values()
        .filter_map(|(sum, count)| (count > 0).then_some(sum / count as f64))
        .collect();
    if sampled.len() > max_points {
        sampled.sort_by(|a, b| {
            spatial_hash(a)
                .cmp(&spatial_hash(b))
                .then_with(|| a.x.total_cmp(&b.x))
                .then_with(|| a.y.total_cmp(&b.y))
        });
        sampled.truncate(max_points);
    }
    Ok(sampled)
}

fn spatial_hash(point: &DVec3) -> u64 {
    let x = point.x.to_bits();
    let y = point.y.to_bits();
    splitmix64(x ^ y.rotate_left(32))
}

fn jittered_subsample(points: &[DVec3], max_points: usize) -> Vec<DVec3> {
    (0..max_points)
        .map(|bucket| {
            let start = bucket * points.len() / max_points;
            let end = ((bucket + 1) * points.len() / max_points).min(points.len());
            let width = end.saturating_sub(start).max(1);
            points[start + (splitmix64(bucket as u64) as usize % width)]
        })
        .collect()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E3779B97F4A7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^ (value >> 31)
}
