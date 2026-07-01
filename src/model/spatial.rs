//! Spatial acceleration structures for immutable mine geometry.

use glam::{DMat4, DVec2, DVec3};

use crate::model::{Object, PolyVertex, formats::tri00t};

/// BVH over document objects, built once when the scene document changes.
/// Allows `snap_cursor` to find nearby objects in O(log N) instead of O(N).
#[derive(Clone, Debug, Default)]
pub(crate) struct ObjectSnapIndex {
    bboxes: Vec<(DVec3, DVec3)>,
    order: Vec<u32>,
    nodes: Vec<Node>,
}

impl ObjectSnapIndex {
    pub(crate) fn build(objects: &[Object]) -> Self {
        let bboxes: Vec<(DVec3, DVec3)> = objects.iter().map(object_bbox).collect();
        let n = bboxes.len();
        let mut index = Self {
            bboxes,
            order: (0..n as u32).collect(),
            nodes: Vec::new(),
        };
        if n > 0 {
            index.build_node(0, n);
        }
        index
    }

    /// Return object indices (into the original `document.objects()` slice) whose
    /// projected AABB overlaps the cursor region in screen space.
    pub(crate) fn candidates(
        &self,
        view_projection: &DMat4,
        screen: (f32, f32),
        cursor: DVec2,
        threshold: f64,
    ) -> Vec<usize> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::new();
        let mut stack = vec![0usize];
        while let Some(idx) = stack.pop() {
            let node = self.nodes[idx];
            if !projected_box_overlaps(
                node.min,
                node.max,
                view_projection,
                screen,
                cursor,
                threshold,
            ) {
                continue;
            }
            if node.count > 0 {
                let range = node.start as usize..(node.start + node.count) as usize;
                result.extend(self.order[range].iter().map(|&i| i as usize));
            } else {
                stack.push(node.left as usize);
                stack.push(node.right as usize);
            }
        }
        result
    }

    fn build_node(&mut self, start: usize, end: usize) -> u32 {
        let mut min = DVec3::splat(f64::INFINITY);
        let mut max = DVec3::splat(f64::NEG_INFINITY);
        let mut center_min = DVec3::splat(f64::INFINITY);
        let mut center_max = DVec3::splat(f64::NEG_INFINITY);
        for &i in &self.order[start..end] {
            let (bmin, bmax) = self.bboxes[i as usize];
            min = min.min(bmin);
            max = max.max(bmax);
            let center = (bmin + bmax) * 0.5;
            center_min = center_min.min(center);
            center_max = center_max.max(center);
        }
        let node_index = self.nodes.len() as u32;
        self.nodes.push(Node {
            min,
            max,
            left: 0,
            right: 0,
            start: start as u32,
            count: (end - start) as u32,
        });
        if end - start <= 8 {
            return node_index;
        }
        let extent = center_max - center_min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        let middle = start + (end - start) / 2;
        let bboxes = &self.bboxes;
        self.order[start..end].select_nth_unstable_by(middle - start, |&a, &b| {
            let ca = (bboxes[a as usize].0 + bboxes[a as usize].1)[axis];
            let cb = (bboxes[b as usize].0 + bboxes[b as usize].1)[axis];
            ca.total_cmp(&cb)
        });
        let left = self.build_node(start, middle);
        let right = self.build_node(middle, end);
        self.nodes[node_index as usize].left = left;
        self.nodes[node_index as usize].right = right;
        self.nodes[node_index as usize].count = 0;
        node_index
    }
}

fn object_bbox(object: &Object) -> (DVec3, DVec3) {
    match object {
        Object::Point { pos, .. } | Object::Text { pos, .. } => (*pos, *pos),
        Object::Polyline { verts, .. }
        | Object::Road {
            centerline: verts, ..
        } => polyline_bbox(verts),
    }
}

fn polyline_bbox(verts: &[PolyVertex]) -> (DVec3, DVec3) {
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    for v in verts {
        min = min.min(v.pos);
        max = max.max(v.pos);
    }
    if min.x > max.x {
        return (DVec3::ZERO, DVec3::ZERO);
    }
    (min, max)
}

#[derive(Clone, Debug)]
pub(crate) struct TriangleBvh {
    /// World-space reference point subtracted before f32 conversion of node bounds.
    origin: DVec3,
    order: Vec<u32>,
    /// Compact 28-byte nodes in depth-first layout (see `TriNode`).
    nodes: Vec<TriNode>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TriangleHit {
    pub(crate) point: DVec3,
}

/// BVH node for document objects — world-space DVec3 bounds, used by ObjectSnapIndex.
#[derive(Clone, Copy, Debug)]
struct Node {
    min: DVec3,
    max: DVec3,
    left: u32,
    right: u32,
    start: u32,
    count: u32,
}

/// Compact 28-byte BVH node for triangulation meshes.
///
/// Bounds are stored as local f32 relative to `TriangleBvh::origin`.
/// Nodes are laid out in depth-first order so the left child is always
/// at `index + 1`, saving one field over the naive 4-pointer layout.
///
/// Internal nodes (count == 0): `right_child_or_start` = right child index.
/// Leaf nodes    (count  > 0): `right_child_or_start` = start offset into `order`.
#[derive(Clone, Copy, Debug)]
struct TriNode {
    min: [f32; 3],
    max: [f32; 3],
    right_child_or_start: u32,
    count: u32,
}

/// Temporary 40-byte node produced by the parallel recursive build before
/// being compacted into depth-first `TriNode` layout.
#[derive(Clone, Copy)]
struct BuildNode {
    min: [f32; 3],
    max: [f32; 3],
    left: u32,
    right: u32,
    start: u32,
    count: u32,
}

impl TriangleBvh {
    pub(crate) fn build(mesh: &tri00t::Triangulation) -> Self {
        let b = mesh.bounds();
        let origin = DVec3::new(
            (b.min.x + b.max.x) * 0.5,
            (b.min.y + b.max.y) * 0.5,
            (b.min.z + b.max.z) * 0.5,
        );
        // Temporary f32 vertex cache for centroid/bounds computation during build.
        // Freed once the BVH is constructed — not stored in the struct.
        let build_vertices: Vec<[f32; 3]> = mesh
            .vertices()
            .iter()
            .map(|v| {
                [
                    (v.x - origin.x) as f32,
                    (v.y - origin.y) as f32,
                    (v.z - origin.z) as f32,
                ]
            })
            .collect();
        let build_triangles: Vec<[u32; 3]> = mesh
            .face_vertex_indices_iter()
            .map(|face| face.map(|i| i as u32))
            .collect();
        let mut order: Vec<u32> = (0..build_triangles.len() as u32).collect();
        let nodes = if order.is_empty() {
            Vec::new()
        } else {
            let wide = build_triangle_subtree(&mut order, 0, &build_triangles, &build_vertices);
            flatten_to_dfs(&wide)
        };
        Self {
            origin,
            order,
            nodes,
        }
    }

    pub(crate) fn ray_hit(
        &self,
        mesh: &tri00t::Triangulation,
        origin: DVec3,
        direction: DVec3,
    ) -> Option<DVec3> {
        self.ray_hit_details(mesh, origin, direction)
            .map(|hit| hit.point)
    }

    pub(crate) fn ray_hit_details(
        &self,
        mesh: &tri00t::Triangulation,
        origin: DVec3,
        direction: DVec3,
    ) -> Option<TriangleHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let inverse = DVec3::new(
            safe_inverse(direction.x),
            safe_inverse(direction.y),
            safe_inverse(direction.z),
        );
        let mut stack = vec![0usize];
        let mut nearest = f64::INFINITY;
        while let Some(index) = stack.pop() {
            let node = self.nodes[index];
            if !ray_box(
                origin,
                inverse,
                self.node_min(&node),
                self.node_max(&node),
                nearest,
            ) {
                continue;
            }
            if node.count > 0 {
                let start = node.right_child_or_start as usize;
                for &triangle_index in &self.order[start..start + node.count as usize] {
                    if let Some(distance) = ray_triangle(
                        origin,
                        direction,
                        self.triangle(mesh, triangle_index as usize),
                    ) && distance < nearest
                    {
                        nearest = distance;
                    }
                }
            } else {
                stack.push(node.right_child_or_start as usize);
                stack.push(index + 1);
            }
        }
        nearest.is_finite().then_some(TriangleHit {
            point: origin + direction * nearest,
        })
    }

    /// Return triangles whose projected BVH bounds overlap a cursor-sized
    /// screen region. This keeps vertex/edge snapping proportional to nearby
    /// geometry rather than the entire mesh.
    pub(crate) fn screen_candidates(
        &self,
        mesh: &tri00t::Triangulation,
        view_projection: &DMat4,
        screen: (f32, f32),
        cursor: (f32, f32),
        threshold: f32,
    ) -> Vec<[DVec3; 3]> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let cursor = DVec2::new(cursor.0 as f64, cursor.1 as f64);
        let threshold = threshold as f64;
        let mut result = Vec::new();
        let mut stack = vec![0usize];
        while let Some(index) = stack.pop() {
            let node = self.nodes[index];
            if !projected_box_overlaps(
                self.node_min(&node),
                self.node_max(&node),
                view_projection,
                screen,
                cursor,
                threshold,
            ) {
                continue;
            }
            if node.count > 0 {
                let start = node.right_child_or_start as usize;
                result.extend(
                    self.order[start..start + node.count as usize]
                        .iter()
                        .map(|&i| self.triangle(mesh, i as usize)),
                );
            } else {
                stack.push(node.right_child_or_start as usize);
                stack.push(index + 1);
            }
        }
        result
    }

    /// Return triangle indices whose XY bounds overlap the supplied rectangle.
    pub(crate) fn xy_bounds_candidate_indices(
        &self,
        mesh: &tri00t::Triangulation,
        min: DVec2,
        max: DVec2,
    ) -> Vec<usize> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::new();
        let mut stack = vec![0usize];
        while let Some(index) = stack.pop() {
            let node = self.nodes[index];
            let nmin = self.node_min(&node);
            let nmax = self.node_max(&node);
            if nmin.x > max.x || nmax.x < min.x || nmin.y > max.y || nmax.y < min.y {
                continue;
            }
            if node.count > 0 {
                let start = node.right_child_or_start as usize;
                result.extend(
                    self.order[start..start + node.count as usize]
                        .iter()
                        .map(|&i| i as usize)
                        .filter(|&i| {
                            let triangle = self.triangle(mesh, i);
                            let triangle_min = triangle
                                .iter()
                                .fold(DVec2::splat(f64::INFINITY), |bounds, point| {
                                    bounds.min(point.truncate())
                                });
                            let triangle_max = triangle
                                .iter()
                                .fold(DVec2::splat(f64::NEG_INFINITY), |bounds, point| {
                                    bounds.max(point.truncate())
                                });
                            triangle_min.x <= max.x
                                && triangle_max.x >= min.x
                                && triangle_min.y <= max.y
                                && triangle_max.y >= min.y
                        }),
                );
            } else {
                stack.push(node.right_child_or_start as usize);
                stack.push(index + 1);
            }
        }
        result
    }

    fn triangle(&self, mesh: &tri00t::Triangulation, index: usize) -> [DVec3; 3] {
        mesh.face_vertex_indices(index)
            .unwrap_or([0, 0, 0])
            .map(|v| {
                let p = mesh.vertices()[v];
                DVec3::new(p.x, p.y, p.z)
            })
    }

    fn node_min(&self, node: &TriNode) -> DVec3 {
        self.origin + DVec3::new(node.min[0] as f64, node.min[1] as f64, node.min[2] as f64)
    }

    fn node_max(&self, node: &TriNode) -> DVec3 {
        self.origin + DVec3::new(node.max[0] as f64, node.max[1] as f64, node.max[2] as f64)
    }
}

/// Convert the wide arbitrary-indexed `BuildNode` tree into the compact
/// depth-first `TriNode` layout where left child = parent_index + 1.
fn flatten_to_dfs(wide: &[BuildNode]) -> Vec<TriNode> {
    let mut out = Vec::with_capacity(wide.len());
    if !wide.is_empty() {
        dfs_serialize(wide, 0, &mut out);
    }
    out
}

fn dfs_serialize(wide: &[BuildNode], idx: usize, out: &mut Vec<TriNode>) {
    let node = wide[idx];
    let my_idx = out.len();
    if node.count > 0 {
        out.push(TriNode {
            min: node.min,
            max: node.max,
            right_child_or_start: node.start,
            count: node.count,
        });
    } else {
        out.push(TriNode {
            min: node.min,
            max: node.max,
            right_child_or_start: 0,
            count: 0,
        });
        dfs_serialize(wide, node.left as usize, out);
        let right_idx = out.len() as u32;
        out[my_idx].right_child_or_start = right_idx;
        dfs_serialize(wide, node.right as usize, out);
    }
}

/// Minimum subtree size to spawn a rayon parallel task. Below this threshold
/// the recursion finishes serially to avoid thread-pool overhead on tiny slices.
const PARALLEL_MIN_TRIANGLES: usize = 2048;

/// Build a BVH subtree for `order` (a sub-slice of the global order array)
/// and return a `Vec<BuildNode>` whose indices are self-contained (index 0 = root).
///
/// `order_start` is the offset of `order[0]` in the global order array, used
/// to fill leaf `BuildNode::start` correctly.
fn build_triangle_subtree(
    order: &mut [u32],
    order_start: usize,
    triangles: &[[u32; 3]],
    vertices: &[[f32; 3]],
) -> Vec<BuildNode> {
    let n = order.len();

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut center_min = [f32::INFINITY; 3];
    let mut center_max = [f32::NEG_INFINITY; 3];
    for &idx in order.iter() {
        let tri = triangles[idx as usize].map(|v| vertices[v as usize]);
        for vertex in tri {
            for i in 0..3 {
                min[i] = min[i].min(vertex[i]);
                max[i] = max[i].max(vertex[i]);
            }
        }
        let c = centroid_f32(tri);
        for i in 0..3 {
            center_min[i] = center_min[i].min(c[i]);
            center_max[i] = center_max[i].max(c[i]);
        }
    }

    if n <= 8 {
        return vec![BuildNode {
            min,
            max,
            left: 0,
            right: 0,
            start: order_start as u32,
            count: n as u32,
        }];
    }

    let extent = [
        center_max[0] - center_min[0],
        center_max[1] - center_min[1],
        center_max[2] - center_min[2],
    ];
    let axis = if extent[0] >= extent[1] && extent[0] >= extent[2] {
        0
    } else if extent[1] >= extent[2] {
        1
    } else {
        2
    };

    let middle = n / 2;
    order.select_nth_unstable_by(middle, |a, b| {
        centroid_f32(triangles[*a as usize].map(|v| vertices[v as usize]))[axis]
            .total_cmp(&centroid_f32(triangles[*b as usize].map(|v| vertices[v as usize]))[axis])
    });

    let (left_order, right_order) = order.split_at_mut(middle);

    let (mut left_nodes, mut right_nodes) = if n > PARALLEL_MIN_TRIANGLES * 2 {
        rayon::join(
            || build_triangle_subtree(left_order, order_start, triangles, vertices),
            || build_triangle_subtree(right_order, order_start + middle, triangles, vertices),
        )
    } else {
        (
            build_triangle_subtree(left_order, order_start, triangles, vertices),
            build_triangle_subtree(right_order, order_start + middle, triangles, vertices),
        )
    };

    // Merge: parent at [0], left subtree at [1..], right subtree at [1+left.len()..].
    let right_offset = 1 + left_nodes.len() as u32;
    shift_build_node_indices(&mut left_nodes, 1);
    shift_build_node_indices(&mut right_nodes, right_offset);

    let mut result = Vec::with_capacity(1 + left_nodes.len() + right_nodes.len());
    result.push(BuildNode {
        min,
        max,
        left: 1,
        right: right_offset,
        start: order_start as u32,
        count: 0,
    });
    result.extend(left_nodes);
    result.extend(right_nodes);
    result
}

fn shift_build_node_indices(nodes: &mut [BuildNode], offset: u32) {
    for node in nodes.iter_mut() {
        if node.count == 0 {
            node.left += offset;
            node.right += offset;
        }
    }
}

pub(crate) fn projected_box_overlaps(
    min: DVec3,
    max: DVec3,
    view_projection: &DMat4,
    screen: (f32, f32),
    cursor: DVec2,
    threshold: f64,
) -> bool {
    let mut screen_min = DVec2::splat(f64::INFINITY);
    let mut screen_max = DVec2::splat(f64::NEG_INFINITY);
    let mut any = false;
    for x in [min.x, max.x] {
        for y in [min.y, max.y] {
            for z in [min.z, max.z] {
                let clip = *view_projection * DVec3::new(x, y, z).extend(1.0);
                if clip.w.abs() <= f64::EPSILON {
                    continue;
                }
                let ndc = clip.truncate() / clip.w;
                let point = DVec2::new(
                    (ndc.x * 0.5 + 0.5) * screen.0 as f64,
                    (0.5 - ndc.y * 0.5) * screen.1 as f64,
                );
                screen_min = screen_min.min(point);
                screen_max = screen_max.max(point);
                any = true;
            }
        }
    }
    any && cursor.x >= screen_min.x - threshold
        && cursor.x <= screen_max.x + threshold
        && cursor.y >= screen_min.y - threshold
        && cursor.y <= screen_max.y + threshold
}

fn centroid_f32(triangle: [[f32; 3]; 3]) -> [f32; 3] {
    [
        (triangle[0][0] + triangle[1][0] + triangle[2][0]) / 3.0,
        (triangle[0][1] + triangle[1][1] + triangle[2][1]) / 3.0,
        (triangle[0][2] + triangle[1][2] + triangle[2][2]) / 3.0,
    ]
}

fn safe_inverse(value: f64) -> f64 {
    if value.abs() <= f64::EPSILON {
        f64::INFINITY.copysign(value)
    } else {
        value.recip()
    }
}

fn ray_box(origin: DVec3, inverse: DVec3, min: DVec3, max: DVec3, limit: f64) -> bool {
    let t0 = (min - origin) * inverse;
    let t1 = (max - origin) * inverse;
    let near = t0.min(t1).max_element().max(0.0);
    let far = t0.max(t1).min_element().min(limit);
    near <= far
}

fn ray_triangle(origin: DVec3, direction: DVec3, triangle: [DVec3; 3]) -> Option<f64> {
    let edge1 = triangle[1] - triangle[0];
    let edge2 = triangle[2] - triangle[0];
    let p = direction.cross(edge2);
    let determinant = edge1.dot(p);
    if determinant.abs() < 1.0e-12 {
        return None;
    }
    let inverse = determinant.recip();
    let t = origin - triangle[0];
    let u = t.dot(p) * inverse;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = t.cross(edge1);
    let v = direction.dot(q) * inverse;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let distance = edge2.dot(q) * inverse;
    (distance >= 0.0).then_some(distance)
}
