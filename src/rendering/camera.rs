use std::time::Duration;

use glam::{DMat4, DQuat, DVec2, DVec3, dcamera};
use winit::{
    dpi::PhysicalPosition,
    event::{MouseButton, MouseScrollDelta},
    keyboard::KeyCode,
};

use crate::Size;

#[derive(Debug)]
pub(crate) struct Camera {
    pub(crate) position: DVec3,
    yaw: f64,
    pitch: f64,
    target: DVec3,
    up: DVec3,
}

impl Camera {
    pub(crate) fn new(position: DVec3, yaw: f64, pitch: f64) -> Self {
        Self {
            position,
            yaw,
            pitch,
            target: DVec3::ZERO,
            up: DVec3::Y,
        }
    }

    pub(crate) fn calc_matrix(&self) -> DMat4 {
        dcamera::rh::view::look_to_mat4(self.position, self.forward(), self.up)
    }

    pub(crate) fn forward(&self) -> DVec3 {
        let to_target = self.target - self.position;
        if to_target.length_squared() > f64::EPSILON {
            return to_target.normalize();
        }

        self.angle_forward()
    }

    fn angle_forward(&self) -> DVec3 {
        let (pitch_sin, pitch_cos) = self.pitch.sin_cos();
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        DVec3::new(pitch_cos * yaw_cos, pitch_cos * yaw_sin, pitch_sin).normalize()
    }

    fn angle_right(&self) -> DVec3 {
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        DVec3::new(-yaw_sin, yaw_cos, 0.0)
    }

    pub(crate) fn apply_angle_orientation(&mut self, target_distance: f64) {
        let forward = self.angle_forward();
        let right = self.angle_right();
        self.up = forward.cross(right).normalize();
        self.target = self.position + forward * target_distance.max(MIN_ORTHO_ZOOM);
    }

    pub(crate) fn sync_angles_from_forward(&mut self) {
        let forward = self.forward();
        let horizontal = forward.x.hypot(forward.y);
        if horizontal > f64::EPSILON {
            self.yaw = forward.y.atan2(forward.x);
        }
        self.pitch = forward.z.atan2(horizontal);
    }

    pub(crate) fn up(&self) -> DVec3 {
        self.up
    }

    pub(crate) fn target(&self) -> DVec3 {
        self.target
    }

    /// Reset to a top-down plan view centred on `center` with the given ortho half-height.
    /// After the next `update_camera` tick, `projection.zoom` will settle to `zoom`.
    /// Set `projection.zoom = zoom` in the same call site for immediate effect.
    pub(crate) fn reset_to_plan_view(&mut self, center: DVec3, zoom: f64) {
        use std::f64::consts::FRAC_PI_2;
        self.yaw = FRAC_PI_2;
        self.pitch = -FRAC_PI_2;
        self.up = DVec3::Y;
        self.target = center;
        // Camera height == zoom so `update_camera` re-derives the same zoom.
        self.position = center + DVec3::Z * zoom.max(MIN_ORTHO_ZOOM);
    }

    /// Centre the current view on `center` without changing its orientation.
    /// The camera distance tracks the orthographic half-height because the
    /// controller derives the projection zoom from that distance each frame.
    pub(crate) fn frame_keep_orientation(&mut self, center: DVec3, zoom: f64) {
        let forward = self.forward();
        self.target = center;
        self.position = center - forward * zoom.max(MIN_ORTHO_ZOOM);
    }

    pub(crate) fn set_target_orientation(
        &mut self,
        forward: DVec3,
        up_hint: DVec3,
        target_distance: f64,
    ) {
        let forward = forward.normalize_or(self.forward());
        let fallback_up = if forward.z.abs() > 0.9 {
            DVec3::Y
        } else {
            DVec3::Z
        };
        let right = {
            let preferred = forward.cross(up_hint);
            if preferred.length_squared() > f64::EPSILON {
                preferred.normalize()
            } else {
                forward.cross(fallback_up).normalize_or(DVec3::X)
            }
        };
        self.up = right.cross(forward).normalize_or(up_hint);
        self.position = self.target - forward * target_distance.max(MIN_ORTHO_ZOOM);
        self.sync_angles_from_forward();
    }
}

const MIN_ORTHO_ZOOM: f64 = 1.0e-4;
pub(crate) const PERSPECTIVE_FOV_Y: f64 = 70.0_f64.to_radians();

pub(crate) struct Projection {
    aspect: f64,
    znear: f64,
    zfar: f64,
    pub(crate) zoom: f64,
    perspective: bool,
}

impl Projection {
    pub(crate) fn new(width: u32, height: u32, znear: f64, zfar: f64) -> Self {
        Self {
            aspect: width as f64 / height as f64,
            znear,
            zfar,
            zoom: 1.0,
            perspective: false,
        }
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f64 / height as f64;
    }

    pub(crate) fn set_symmetric_depth_extent(&mut self, half_depth: f64) {
        let half_depth = half_depth.max(1.0);
        self.znear = -half_depth;
        self.zfar = half_depth;
    }

    pub(crate) fn expand_symmetric_depth_extent(&mut self, half_depth: f64) {
        let current_half_depth = self.znear.abs().max(self.zfar.abs());
        self.set_symmetric_depth_extent(current_half_depth.max(half_depth));
    }

    pub(crate) fn calc_matrix(&self) -> DMat4 {
        if self.perspective {
            let far = self.znear.abs().max(self.zfar.abs()).max(10_000.0);
            let near = (far * 1.0e-4).max(1.0);
            return dcamera::rh::proj::directx::perspective(
                PERSPECTIVE_FOV_Y,
                self.aspect,
                near,
                far,
            );
        }

        let half_w = self.aspect * self.zoom;
        let half_h = self.zoom;
        dcamera::rh::proj::directx::orthographic(
            -half_w, half_w, -half_h, half_h, self.znear, self.zfar,
        )
    }

    pub(crate) fn set_perspective(&mut self, perspective: bool) {
        self.perspective = perspective;
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CameraUniform {
    pub(crate) view_proj: [[f32; 4]; 4],
    cam_forward: [f32; 4],
    cam_position: [f32; 4],
    viewport: [f32; 4],
}

impl CameraUniform {
    pub(crate) fn new() -> Self {
        Self {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            cam_forward: [0.; 4],
            cam_position: [0.; 4],
            viewport: [1.0, 1.0, 0.0, 0.0],
        }
    }

    pub(crate) fn update_view_proj(
        &mut self,
        camera: &Camera,
        projection: &Projection,
        scene_origin: DVec3,
        vertical_exaggeration: f64,
    ) {
        let rebased_view = dcamera::rh::view::look_to_mat4(
            camera.position - scene_origin,
            camera.forward(),
            camera.up,
        );
        let exaggeration = DMat4::from_scale(DVec3::new(1.0, 1.0, vertical_exaggeration));
        self.view_proj = (projection.calc_matrix() * rebased_view * exaggeration)
            .as_mat4()
            .to_cols_array_2d();
        let forward = camera.forward();
        self.cam_forward = [forward.x as f32, forward.y as f32, forward.z as f32, 0.];
        self.cam_position = [
            (camera.position.x - scene_origin.x) as f32,
            (camera.position.y - scene_origin.y) as f32,
            (camera.position.z - scene_origin.z) as f32,
            0.,
        ];
    }

    pub(crate) fn update_viewport(&mut self, width: u32, height: u32) {
        self.viewport = [width.max(1) as f32, height.max(1) as f32, 0.0, 0.0];
    }
}

#[derive(Debug)]
pub(crate) struct CameraController {
    amount_left: f64,
    amount_right: f64,
    amount_forward: f64,
    amount_backward: f64,
    amount_up: f64,
    amount_down: f64,
    pan: DVec2,
    rotate_horizontal: f64,
    rotate_vertical: f64,
    scroll: f64,
    speed: f64,
    zoom_sensitivity: f64,
    rotate_sensitivity: f64,
    pub(crate) mouse_loc: (f32, f32),
    orbit_anchor: Option<DVec3>,
}

impl CameraController {
    pub(crate) fn new(speed: f64, zoom_sensitivity: f64, rotate_sensitivity: f64) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            pan: DVec2::ZERO,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            scroll: 0.0,
            speed,
            zoom_sensitivity,
            rotate_sensitivity,
            mouse_loc: (0., 0.),
            orbit_anchor: None,
        }
    }

    pub(crate) fn begin_orbit(&mut self, anchor: DVec3) {
        self.orbit_anchor = Some(anchor);
    }

    pub(crate) fn end_orbit(&mut self) {
        self.orbit_anchor = None;
    }

    pub(crate) fn process_mouse(
        &mut self,
        mouse_pressed: Option<MouseButton>,
        mouse_dx: f64,
        mouse_dy: f64,
    ) -> bool {
        if let Some(button) = mouse_pressed {
            match button {
                MouseButton::Right => {
                    self.rotate_horizontal += mouse_dx;
                    self.rotate_vertical += mouse_dy;
                    return true;
                }
                MouseButton::Middle => {
                    // Accumulate device deltas until next frame for smooth 1:1 drag.
                    self.pan.y += mouse_dy;
                    self.pan.x += -mouse_dx;
                    return true;
                }
                _ => {}
            }
        } else {
            self.pan = DVec2::ZERO;
        }
        false
    }

    pub(crate) fn process_scroll(&mut self, delta: &MouseScrollDelta) {
        self.scroll += match delta {
            // Assuming a line is about 100 pixels
            MouseScrollDelta::LineDelta(_, scroll) => *scroll as f64 * 100.0,
            MouseScrollDelta::PixelDelta(PhysicalPosition { y: scroll, .. }) => *scroll,
        };
    }

    pub(crate) fn has_pending_updates(&self) -> bool {
        self.amount_left != 0.0
            || self.amount_right != 0.0
            || self.amount_forward != 0.0
            || self.amount_backward != 0.0
            || self.amount_up != 0.0
            || self.amount_down != 0.0
            || self.pan != DVec2::ZERO
            || self.rotate_horizontal != 0.0
            || self.rotate_vertical != 0.0
            || self.scroll != 0.0
    }

    pub(crate) fn update_camera(
        &mut self,
        camera: &mut Camera,
        projection: &mut Projection,
        dt: Duration,
        screen_size: Size,
    ) {
        let dt = dt.as_secs_f64();

        // Calculate direction unit vectors (Z-up: horizontal plane is XY)
        let (yaw_sin, yaw_cos) = camera.yaw.sin_cos();

        let movement_forward = DVec3::new(yaw_cos, yaw_sin, 0.0);
        let movement_right = DVec3::new(-yaw_sin, yaw_cos, 0.0);
        let scrollward = camera.forward();
        let right = scrollward
            .cross(camera.up)
            .normalize_or(movement_right.normalize_or_zero());
        let up = right.cross(scrollward).normalize_or_zero();
        let max_pitch = std::f64::consts::FRAC_PI_2;

        // Movement updates
        let forward_movement = (self.amount_forward - self.amount_backward) * self.speed * dt;
        let right_movement = (self.amount_right - self.amount_left) * self.speed * dt;
        let vertical_movement = (self.amount_up - self.amount_down) * self.speed * dt;

        // Position-based transformations
        let to_target_distance_modifier = (camera.target - camera.position).dot(scrollward).abs();

        camera.position += movement_forward * forward_movement;
        camera.position += movement_right * right_movement;
        camera.position.z += vertical_movement;

        // Update target
        camera.target = camera.position + scrollward * to_target_distance_modifier;

        // Scroll / zoom-like motion
        let mouse_loc_rel = point(self.mouse_loc.0, self.mouse_loc.1, screen_size);
        let zoom_scale = if self.scroll > 0.0 {
            1. - (1. / (1. + self.zoom_sensitivity * self.scroll))
        } else {
            self.zoom_sensitivity * self.scroll
        };
        let zoom_factor = to_target_distance_modifier * zoom_scale;
        let aspect = screen_size.0 as f64 / screen_size.1.max(1.0) as f64;
        let zoom_lateral = (right * mouse_loc_rel.x * aspect + up * mouse_loc_rel.y) * zoom_factor;
        camera.position += scrollward * zoom_factor + zoom_lateral;
        camera.target += zoom_lateral;
        self.scroll = 0.0;

        // Panning motion: map mouse pixels directly to world units for CAD-like drag feel.
        let world_per_pixel = (2.0 * projection.zoom / screen_size.1.max(1.0) as f64).max(0.0);
        let pan_delta = up * self.pan.y * world_per_pixel + right * self.pan.x * world_per_pixel;
        camera.position += pan_delta;
        camera.target += pan_delta;

        self.pan = DVec2::ZERO;

        // Orbit position and target around the selected pivot.
        if self.rotate_horizontal != 0.0 || self.rotate_vertical != 0.0 {
            let pivot = self.orbit_anchor.unwrap_or(camera.target);
            let horizontal_rotate_angle = self.rotate_horizontal * self.rotate_sensitivity;
            let desired_pitch_delta = -self.rotate_vertical * self.rotate_sensitivity;
            let next_pitch = (camera.pitch + desired_pitch_delta).clamp(-max_pitch, max_pitch);
            let vertical_rotate_angle = next_pitch - camera.pitch;
            let horizontal_rotation = DQuat::from_rotation_z(-horizontal_rotate_angle);

            camera.yaw += horizontal_rotate_angle;
            camera.position = horizontal_rotation.mul_vec3(camera.position - pivot) + pivot;
            camera.target = horizontal_rotation.mul_vec3(camera.target - pivot) + pivot;
            camera.up = horizontal_rotation.mul_vec3(camera.up).normalize_or_zero();

            camera.pitch = next_pitch;
            let rotated_scrollward = camera.forward();
            let rotated_right = rotated_scrollward
                .cross(camera.up)
                .normalize_or(right.normalize_or_zero());
            let vertical_rotation = DQuat::from_axis_angle(rotated_right, vertical_rotate_angle);
            camera.position = vertical_rotation.mul_vec3(camera.position - pivot) + pivot;
            camera.target = vertical_rotation.mul_vec3(camera.target - pivot) + pivot;
            camera.up = vertical_rotation.mul_vec3(camera.up).normalize_or_zero();

            self.rotate_horizontal = 0.0;
            self.rotate_vertical = 0.0;
        }

        // Scale orthographic bounds by camera-to-target distance for zoom effect
        let dist = (camera.target - camera.position)
            .dot(camera.forward())
            .abs();
        projection.zoom = dist.max(MIN_ORTHO_ZOOM);
    }
}

#[derive(Debug)]
pub(crate) struct FlyCameraController {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    boost: bool,
    analog_movement: DVec3,
    look_delta: DVec2,
    speed: f64,
    look_sensitivity: f64,
    velocity: DVec3,
    skip_movement_frame: bool,
}

impl FlyCameraController {
    pub(crate) fn new(speed: f64, look_sensitivity: f64) -> Self {
        Self {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            boost: false,
            analog_movement: DVec3::ZERO,
            look_delta: DVec2::ZERO,
            speed,
            look_sensitivity,
            velocity: DVec3::ZERO,
            skip_movement_frame: false,
        }
    }

    pub(crate) fn begin_capture(&mut self) {
        self.clear_input();
        self.skip_movement_frame = true;
    }

    pub(crate) fn process_mouse_motion(&mut self, dx: f64, dy: f64) {
        self.look_delta += DVec2::new(dx, dy);
    }

    pub(crate) fn process_key(&mut self, key: KeyCode, pressed: bool) -> bool {
        match key {
            KeyCode::KeyW => self.forward = pressed,
            KeyCode::KeyS => self.backward = pressed,
            KeyCode::KeyA => self.left = pressed,
            KeyCode::KeyD => self.right = pressed,
            KeyCode::KeyE => self.up = pressed,
            KeyCode::KeyQ => self.down = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.boost = pressed,
            _ => return false,
        }
        true
    }

    pub(crate) fn process_scroll(&mut self, delta: &MouseScrollDelta) {
        let steps = match delta {
            MouseScrollDelta::LineDelta(_, y) => f64::from(*y),
            MouseScrollDelta::PixelDelta(PhysicalPosition { y, .. }) => *y / 100.0,
        };
        self.adjust_speed(steps);
    }

    pub(crate) fn adjust_speed(&mut self, steps: f64) -> f64 {
        self.speed = (self.speed * 1.25_f64.powf(steps)).clamp(0.01, 1.0e9);
        self.speed
    }

    pub(crate) fn clear_input(&mut self) {
        self.forward = false;
        self.backward = false;
        self.left = false;
        self.right = false;
        self.up = false;
        self.down = false;
        self.boost = false;
        self.analog_movement = DVec3::ZERO;
        self.look_delta = DVec2::ZERO;
        self.velocity = DVec3::ZERO;
        self.skip_movement_frame = false;
    }

    pub(crate) fn has_pending_updates(&self) -> bool {
        self.forward
            || self.backward
            || self.left
            || self.right
            || self.up
            || self.down
            || self.analog_movement != DVec3::ZERO
            || self.look_delta != DVec2::ZERO
            || self.velocity != DVec3::ZERO
    }

    pub(crate) fn update_camera(&mut self, camera: &mut Camera, dt: Duration) {
        let target_distance = camera.position.distance(camera.target);
        camera.yaw -= self.look_delta.x * self.look_sensitivity;
        camera.pitch = (camera.pitch - self.look_delta.y * self.look_sensitivity)
            .clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
        camera.apply_angle_orientation(target_distance);
        self.look_delta = DVec2::ZERO;

        let forward_amount =
            f64::from(i8::from(self.forward) - i8::from(self.backward)) + self.analog_movement.y;
        let right_amount =
            f64::from(i8::from(self.left) - i8::from(self.right)) - self.analog_movement.x;
        let vertical_amount =
            f64::from(i8::from(self.up) - i8::from(self.down)) + self.analog_movement.z;
        let input_direction = camera.angle_forward() * forward_amount
            + camera.angle_right() * right_amount
            + DVec3::Z * vertical_amount;
        let speed_multiplier = if self.boost { 3.0 } else { 1.0 };
        let target_velocity = input_direction.normalize_or_zero() * self.speed * speed_multiplier;
        let movement_dt = if self.skip_movement_frame {
            self.skip_movement_frame = false;
            0.0
        } else {
            dt.as_secs_f64().min(0.1)
        };

        // Exponential interpolation gives the same acceleration feel at any
        // frame rate. Releasing movement keys eases cleanly back to rest.
        let response = if target_velocity == DVec3::ZERO {
            12.0
        } else {
            8.0
        };
        let blend = 1.0 - (-response * movement_dt).exp();
        self.velocity = self.velocity.lerp(target_velocity, blend);
        if target_velocity == DVec3::ZERO && self.velocity.length_squared() < 1.0e-8 {
            self.velocity = DVec3::ZERO;
        }
        let movement = self.velocity * movement_dt;

        camera.position += movement;
        camera.target += movement;
    }
}

// Translates a point from pixel coordinates to wgpu NDC coordinates.
// Returns a DVec3 with z=0; only x and y are meaningful for 2D mouse position use.
pub(crate) fn point(x: f32, y: f32, screen: Size) -> DVec3 {
    let scale_x = 2.0 / screen.0 as f64;
    let scale_y = 2.0 / screen.1 as f64;
    let new_x = -1.0 + x as f64 * scale_x;
    let new_y = 1.0 - y as f64 * scale_y;
    DVec3::new(new_x, new_y, 0.)
}

/// Unproject a screen pixel to a world point on the plane `z = plane_z`.
///
/// The projection is orthographic, so every view ray is parallel to the camera
/// `forward` vector: we find the point under the cursor on the camera focal
/// plane, then march along `forward` until it meets the requested Z plane.
/// `zoom` is `Projection::zoom`, which equals the camera-to-target distance.
/// Returns `None` when the view direction is parallel to the plane.
pub(crate) fn screen_to_world_on_plane(
    camera: &Camera,
    zoom: f64,
    aspect: f64,
    screen: Size,
    mouse_px: (f32, f32),
    plane_z: f64,
) -> Option<DVec3> {
    let rel = point(mouse_px.0, mouse_px.1, screen);
    let forward = camera.forward();
    let right = forward.cross(camera.up()).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    let focal =
        camera.position + forward * zoom + right * rel.x * aspect * zoom + up * rel.y * zoom;
    if forward.z.abs() <= f64::EPSILON {
        return None;
    }
    let t = (plane_z - focal.z) / forward.z;
    Some(focal + forward * t)
}
