use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub(crate) struct BackgroundUniform {
    bg_color: [f32; 4],
    _pad: [f32; 4],
}

impl BackgroundUniform {
    pub(crate) fn new() -> Self {
        assert_eq!(std::mem::size_of::<Self>(), 32);
        Self {
            bg_color: [0.0, 0.0, 0.0, 0.0],
            _pad: [0.0; 4],
        }
    }

    pub(crate) fn update_bg_color(&mut self, r: f32, g: f32, b: f32) {
        self.bg_color = [r, g, b, 1.0];
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum RenderItemType {
    Background,
    SurfaceOpaque,
    SurfaceTransparent,
    DocumentFill,
    DocumentStroke,
    SurfaceWireframe,
    Overlay,
}

impl RenderItemType {
    pub(crate) const VARIANTS: &'static [Self] = &[
        Self::Background,
        Self::SurfaceOpaque,
        Self::SurfaceTransparent,
        Self::DocumentFill,
        Self::DocumentStroke,
        Self::SurfaceWireframe,
        Self::Overlay,
    ];

    pub(crate) const fn as_usize(&self) -> usize {
        match self {
            Self::Background => 0,
            Self::SurfaceOpaque => 1,
            Self::SurfaceTransparent => 2,
            Self::DocumentFill => 3,
            Self::DocumentStroke => 4,
            Self::SurfaceWireframe => 5,
            Self::Overlay => 6,
        }
    }
}

pub(crate) struct BackgroundGeometry {
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) index_buffer: wgpu::Buffer,
    pub(crate) index_count: u32,
}

impl BackgroundGeometry {
    pub(crate) fn create(device: &wgpu::Device) -> Self {
        // NDC quad: corners at (-1,-1) to (+1,+1) at z=0
        // 4 vertices, 2 triangles = 6 indices
        let vertices: [[f32; 3]; 4] = [
            [-1.0, -1.0, 0.0], // bottom-left
            [ 1.0, -1.0, 0.0], // bottom-right
            [-1.0,  1.0, 0.0], // top-left
            [ 1.0,  1.0, 0.0], // top-right
        ];
        let indices: [u32; 6] = [0, 2, 1, 1, 2, 3];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Background Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Background Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count: 6,
        }
    }
}
