pub(crate) mod camera;
pub(crate) mod color;
pub(crate) mod geometry;
pub(crate) mod graphics;
pub(crate) mod pick;
pub(crate) mod query;
pub(crate) mod scene;
pub(crate) mod snap;
pub(crate) mod text;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub(crate) struct Vertex {
    pub(crate) pos: [f32; 3],
    pub(crate) color: [f32; 4],
}

/// Position-only vertex for triangulation surfaces — colour comes from a per-draw uniform.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub(crate) struct SurfaceVertex {
    pub(crate) pos: [f32; 3],
}

/// Block-model surface vertex. Grade is a normalized 0..1 value, or -1 when
/// no numeric colour variable is active for the block.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub(crate) struct BlockModelVertex {
    pub(crate) pos: [f32; 3],
    pub(crate) grade: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub(crate) struct StrokeVertex {
    pub(crate) pos: [f32; 3],
    pub(crate) color: [f32; 4],
    pub(crate) other_pos: [f32; 3],
    pub(crate) offset_px: [f32; 2],
    pub(crate) screen_space: f32,
}
