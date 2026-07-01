//! Camera-frustum extraction and AABB intersection, used to skip GPU draw
//! calls for chunks that are entirely outside the current view.

use glam::{Mat4, Vec3, Vec4};

pub(crate) struct Frustum {
    /// Six clip-space half-space planes as `(a, b, c, d)` with the inside
    /// condition `a*x + b*y + c*z + d >= 0`.
    planes: [Vec4; 6],
}

impl Frustum {
    /// Extracts the six frustum planes from a view-projection matrix using
    /// the standard Gribb/Hartmann method, generalized for this project's
    /// D3D-style clip space (`0 <= z <= w`, unlike OpenGL's `-w <= z <= w`).
    pub(crate) fn from_view_proj(view_proj: Mat4) -> Self {
        // `view_proj.transpose().{x,y,z,w}_axis` gives the rows of
        // `view_proj`, since clip = view_proj * v uses v as a column vector.
        let rows = view_proj.transpose();
        let row0 = rows.x_axis;
        let row1 = rows.y_axis;
        let row2 = rows.z_axis;
        let row3 = rows.w_axis;
        Self {
            planes: [
                row3 + row0, // left:   x >= -w
                row3 - row0, // right:  x <=  w
                row3 + row1, // bottom: y >= -w
                row3 - row1, // top:    y <=  w
                row2,        // near:   z >=  0
                row3 - row2, // far:    z <=  w
            ],
        }
    }

    /// `false` only when the AABB is provably entirely on the outside of at
    /// least one plane — conservative, so it never culls a chunk that's
    /// actually (even partially) visible.
    pub(crate) fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        self.planes.iter().all(|plane| {
            let positive = Vec3::new(
                if plane.x >= 0.0 { max.x } else { min.x },
                if plane.y >= 0.0 { max.y } else { min.y },
                if plane.z >= 0.0 { max.z } else { min.z },
            );
            plane.x * positive.x + plane.y * positive.y + plane.z * positive.z + plane.w >= 0.0
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With `view_proj = IDENTITY`, clip == input, so this project's D3D
    /// clip volume is exactly `x,y in [-1, 1]`, `z in [0, 1]`.
    fn identity_frustum() -> Frustum {
        Frustum::from_view_proj(Mat4::IDENTITY)
    }

    #[test]
    fn accepts_box_fully_inside_clip_volume() {
        let frustum = identity_frustum();
        assert!(frustum.intersects_aabb(Vec3::new(-0.5, -0.5, 0.2), Vec3::new(0.5, 0.5, 0.8)));
    }

    #[test]
    fn accepts_box_straddling_a_clip_boundary() {
        let frustum = identity_frustum();
        assert!(frustum.intersects_aabb(Vec3::new(0.5, -0.1, 0.1), Vec3::new(1.5, 0.1, 0.2)));
    }

    #[test]
    fn rejects_box_entirely_outside_on_x() {
        let frustum = identity_frustum();
        assert!(!frustum.intersects_aabb(Vec3::new(2.0, -0.1, 0.1), Vec3::new(3.0, 0.1, 0.2)));
    }

    #[test]
    fn rejects_box_entirely_behind_near_plane() {
        let frustum = identity_frustum();
        assert!(!frustum.intersects_aabb(Vec3::new(-0.1, -0.1, -2.0), Vec3::new(0.1, 0.1, -1.0)));
    }

    #[test]
    fn rejects_box_beyond_far_plane() {
        let frustum = identity_frustum();
        assert!(!frustum.intersects_aabb(Vec3::new(-0.1, -0.1, 1.5), Vec3::new(0.1, 0.1, 2.0)));
    }
}
