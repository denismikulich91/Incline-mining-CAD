use std::{sync::Arc, time::Duration};

use anyhow::{Result, anyhow};
use glam::{DMat4, DVec3, DVec4};
use glyphon::{
    Cache, FontSystem, Resolution, SwashCache, TextArea, TextAtlas, TextRenderer, Viewport,
};
use lyon::tessellation::VertexBuffers;
use wgpu::{MultisampleState, util::DeviceExt};
use winit::{
    event::*,
    window::{CursorGrabMode, Window},
};

use crate::{
    Size,
    model::{
        Document, Object, SceneEntityId, block_model::OpenBlockModel,
        triangulation::OpenTriangulation,
    },
    rendering::{
        BlockModelVertex, StrokeVertex, SurfaceVertex, Vertex,
        camera::{
            Camera, CameraController, CameraUniform, FlyCameraController, Projection,
            screen_to_world_on_plane,
        },
        color::linear_to_srgb,
        pick::{PickRecord, TextPickRecord, pick_nearest, pick_text},
        query::SceneQuery,
        scene::{BlockModelGpuCache, EdgeInstance, TriangulationGpuCache},
        snap::SNAP_THRESHOLD_PX,
        text::{CachedTextArea, TextCache, TextSystem},
    },
    ui::{
        Gui,
        state::{CursorMode, EditorState, UiFrameOutput, UiProjectView},
    },
};

use crate::rendering::scene::bounds::scene_bounds;

pub(crate) mod buffers;
pub(crate) mod camera;
pub(crate) mod frame;
pub(crate) mod frustum;
pub(crate) mod init;
pub(crate) mod passes;
pub(crate) mod projections;
pub(crate) mod targets;

pub(super) const TEXT_CACHE_TRIM_INTERVAL_FRAMES: u64 = 300;
pub(super) const TEXT_ATLAS_TRIM_INTERVAL_FRAMES: u64 = 600;
pub(super) const MSAA_SAMPLE_COUNT: u32 = 4;
pub(super) const CAMERA_ROTATE_SENSITIVITY: f64 = 0.003;
pub(super) const REQUESTED_MAX_BUFFER_SIZE: u64 = 2 * 1024 * 1024 * 1024;
pub(super) const YELLOW_HIGHLIGHT_COLOR: [f32; 4] = [1.0, 0.85, 0.0, 1.0];
/// Sizing for editable document geometry.
pub(super) const DOC_LINE_WIDTH: f32 = 1.0;
/// Colour for the in-progress stroke preview (committed segments + rubber band).
pub(super) const PREVIEW_COLOR: [f32; 4] = [0.4, 0.85, 1.0, 1.0];
pub(super) const INVALID_PREVIEW_COLOR: [f32; 4] = [1.0, 0.12, 0.08, 1.0];
pub(super) const MEASUREMENT_COLOR: [f32; 4] = [1.0, 0.82, 0.15, 1.0];
/// Glyph atlas size for document text; world height scales this down.
// Rasterising every label at 256 px and then heavily minifying it produces
// unstable one-pixel fringes without a mipmapped glyph atlas. This remains
// crisp at normal zoom levels while avoiding that extreme minification.
pub(super) const DOC_TEXT_FONT_SIZE: f32 = 64.0;
pub(super) const TEXT_EDIT_INDICATOR_COLOR: [f32; 4] = [0.15, 0.75, 1.0, 1.0];

/// Liang-Barsky segment-vs-AABB test in 2-D screen space.
/// Returns true when the segment [a, b] intersects the rectangle or either endpoint is inside it.
pub(super) fn segment_intersects_rect(
    a: glam::DVec2,
    b: glam::DVec2,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
) -> bool {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let mut t0 = 0.0_f64;
    let mut t1 = 1.0_f64;
    for (p, q) in [
        (-dx, a.x - min_x),
        (dx, max_x - a.x),
        (-dy, a.y - min_y),
        (dy, max_y - a.y),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return false;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
            if t0 > t1 {
                return false;
            }
        }
    }
    true
}

/// Dim an RGBA colour by scaling its alpha channel down toward transparency.
pub(super) fn make_translucent(color: &mut [f32; 4]) {
    color[3] *= 0.3;
}
pub(super) const INITIAL_CAMERA_Z_NEAR: f64 = -1.0e4;
pub(super) const INITIAL_CAMERA_Z_FAR: f64 = 1.0e4;

pub(super) fn text_bounds_corners(
    pos: DVec3,
    content: &str,
    height: f64,
    rotation: f64,
) -> [DVec3; 4] {
    let height = height.abs().max(0.001);
    let width = height * 0.58 * content.chars().count().max(1) as f64;
    let (sin, cos) = rotation.sin_cos();
    let right = DVec3::new(cos, sin, 0.0);
    let down = DVec3::new(sin, -cos, 0.0);
    let padding = height * 0.15;
    let top_left = pos - right * padding - down * padding;
    let top_right = pos + right * (width + padding) - down * padding;
    let bottom_right = top_right + down * (height + padding * 2.0);
    let bottom_left = top_left + down * (height + padding * 2.0);
    [top_left, top_right, bottom_right, bottom_left]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RenderSurfaceError {
    Timeout,
    Occluded,
    Outdated,
    Lost,
    Validation,
}

pub(crate) struct Graphics<'a> {
    // GPU resource owners are declared before device/surface/window so they
    // are dropped first during shutdown.
    pub(super) gui: Gui,
    pub(super) text_system: TextSystem,
    pub(super) surface_render_pipeline: wgpu::RenderPipeline,
    pub(super) transparent_surface_render_pipeline: wgpu::RenderPipeline,
    pub(super) block_model_render_pipeline: wgpu::RenderPipeline,
    pub(super) transparent_block_model_render_pipeline: wgpu::RenderPipeline,
    pub(super) surface_style_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) render_pipeline: wgpu::RenderPipeline,
    pub(super) xray_render_pipeline: wgpu::RenderPipeline,
    pub(super) stroke_render_pipeline: wgpu::RenderPipeline,
    pub(super) edge_render_pipeline: wgpu::RenderPipeline,
    pub(super) edge_style_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) overlay_render_pipeline: wgpu::RenderPipeline,
    pub(super) lyon_vertex_gpu: wgpu::Buffer,
    pub(super) lyon_index_gpu: wgpu::Buffer,
    pub(super) stroke_vertex_gpu: wgpu::Buffer,
    pub(super) stroke_index_gpu: wgpu::Buffer,
    pub(super) overlay_vertex_gpu: wgpu::Buffer,
    pub(super) overlay_index_gpu: wgpu::Buffer,
    pub(super) camera_buffer: wgpu::Buffer,
    pub(super) camera_bind_group: wgpu::BindGroup,
    pub(super) msaa_color: wgpu::Texture,
    pub(super) msaa_view: wgpu::TextureView,
    pub(super) depth_texture: wgpu::Texture,
    pub(super) depth_view: wgpu::TextureView,
    pub(super) surface: wgpu::Surface<'a>,
    pub(super) queue: wgpu::Queue,
    pub(super) device: wgpu::Device,
    pub(super) window: Arc<Window>,
    pub(super) config: wgpu::SurfaceConfiguration,
    pub(super) sample_count: u32,
    pub(super) size: winit::dpi::PhysicalSize<u32>,
    pub(super) lyon_buffer: VertexBuffers<Vertex, u32>,
    pub(super) lyon_vertex_capacity: usize,
    pub(super) lyon_index_capacity: usize,
    pub(super) stroke_vertex_buf: Vec<StrokeVertex>,
    pub(super) stroke_index_buf: Vec<u32>,
    pub(super) stroke_vertex_capacity: usize,
    pub(super) stroke_index_capacity: usize,
    pub(super) overlay_vertex_buf: Vec<StrokeVertex>,
    pub(super) overlay_index_buf: Vec<u32>,
    pub(super) overlay_vertex_capacity: usize,
    pub(super) overlay_index_capacity: usize,
    pub(super) camera: Camera,
    pub(super) camera_uniform: CameraUniform,
    pub(super) camera_controller: CameraController,
    pub(super) fly_camera_controller: FlyCameraController,
    pub(super) projection: Projection,
    pub(super) mouse_pressed: Option<MouseButton>,
    pub(super) fly_mode_enabled: bool,
    pub(super) cached_textareas: Vec<CachedTextArea>,
    pub(super) textarea_depths: Vec<f32>,
    pub(super) frame_index: u64,
    pub(super) geometry_dirty: bool,
    pub(super) cached_document_revision: u64,
    pub(super) cached_bounds_document_revision: u64,
    pub(super) cached_scene_bounds: Option<(DVec3, DVec3)>,
    pub(super) overlay_dirty: bool,
    pub(super) cached_scale_factor: f32,
    pub(super) cached_measurement_state: (bool, Option<DVec3>, Option<DVec3>),
    pub(super) cached_poly_finish_dialog: bool,
    pub(super) pick_records: Vec<PickRecord>,
    pub(super) text_pick_records: Vec<TextPickRecord>,
    pub(super) orbit_marker: Option<DVec3>,
    pub(super) scene_origin: DVec3,
    pub(super) vertical_exaggeration: f64,
    pub(super) triangulation_gpu: TriangulationGpuCache,
    pub(super) block_model_gpu: BlockModelGpuCache,
}

impl<'a> Graphics<'a> {
    pub(crate) fn invalidate_geometry(&mut self) {
        self.geometry_dirty = true;
        self.overlay_dirty = true;
    }

    pub(crate) fn invalidate_overlay(&mut self) {
        self.overlay_dirty = true;
    }

    pub(crate) fn gui_input(&mut self, event: &WindowEvent) -> egui_winit::EventResponse {
        self.gui.handle_event(&self.window, event)
    }

    pub(crate) fn set_fly_mode_enabled(&mut self, enabled: bool) {
        if enabled == self.fly_mode_enabled {
            return;
        }

        if enabled {
            self.camera.sync_angles_from_forward();
            let target_distance = self.camera.position.distance(self.camera.target());
            self.camera.apply_angle_orientation(target_distance);
        } else if self.fly_mode_enabled
            && let Some((min, max)) = self.cached_scene_bounds
        {
            // Free flight can leave the focal point far from the actual model.
            // Orthographic cursor zoom then has to invert an f32 GPU matrix with
            // a huge translation, which creates a dead region around small world
            // coordinates. Re-anchor to the scene while preserving orientation
            // and the pre-fly orthographic scale.
            let center = self.exaggerate_point((min + max) * 0.5);
            self.camera
                .frame_keep_orientation(center, self.projection.zoom.max(1.0e-4));
        }
        self.fly_camera_controller.clear_input();
        self.projection.set_perspective(enabled);
        self.fly_mode_enabled = enabled;
        self.sync_cursor_grab();
    }

    pub(crate) fn release_mouse_capture(&mut self) {
        self.mouse_pressed = None;
        self.camera_controller.end_orbit();
        self.fly_camera_controller.clear_input();
        self.sync_cursor_grab();
    }

    pub(crate) fn needs_continuous_redraw(&self) -> bool {
        self.camera_controller.has_pending_updates()
            || self.fly_camera_controller.has_pending_updates()
    }

    /// Returns true while the user is actively panning or orbiting the camera
    /// (right-mouse drag). Callers can skip expensive per-frame work like snap
    /// queries during camera movement.
    pub(crate) fn is_camera_active(&self) -> bool {
        self.mouse_pressed == Some(MouseButton::Right)
    }

    pub(crate) fn is_fly_camera_active(&self) -> bool {
        self.fly_mode_enabled && self.mouse_pressed == Some(MouseButton::Right)
    }

    pub(crate) fn should_receive_fly_event(&self, event: &WindowEvent) -> bool {
        self.is_fly_camera_active()
            || (self.fly_mode_enabled
                && matches!(
                    event,
                    WindowEvent::MouseInput {
                        button: MouseButton::Right,
                        ..
                    }
                ))
    }

    fn sync_cursor_grab(&self) {
        let fly_active = self.fly_mode_enabled && self.mouse_pressed == Some(MouseButton::Right);
        if fly_active {
            if self.window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                let _ = self.window.set_cursor_grab(CursorGrabMode::Confined);
            }
            self.window.set_cursor_visible(false);
        } else {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
            self.window.set_cursor_visible(true);
        }
    }
}
