struct CameraUniform {
    view_proj: mat4x4<f32>,
    cam_forward: vec4<f32>,
    cam_position: vec4<f32>,
    viewport: vec4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct PointCloudStyle {
    // Uniform colour used when `options.y == 0.0` (no per-point colours).
    color: vec4<f32>,
    // x: splat size in physical pixels, y: 1.0 when instance colours are valid.
    options: vec4<f32>,
};
@group(1) @binding(0)
var<uniform> style: PointCloudStyle;

struct PointInput {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// Expands each point instance into a screen-aligned quad (two triangles,
// vertex_index 0..6) sized in physical pixels, mirroring edge.wgsl.
@vertex
fn vs_main(point: PointInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var clip = camera.view_proj * vec4<f32>(point.pos, 1.0);
    let right = vertex_index == 1u || vertex_index == 4u || vertex_index == 5u;
    let up = vertex_index == 2u || vertex_index == 3u || vertex_index == 5u;
    let corner = vec2<f32>(select(-1.0, 1.0, right), select(-1.0, 1.0, up));
    let pixel_to_ndc = vec2<f32>(2.0 / camera.viewport.x, 2.0 / camera.viewport.y);
    let half_size = style.options.x * 0.5;
    clip.x = clip.x + corner.x * half_size * pixel_to_ndc.x * clip.w;
    clip.y = clip.y + corner.y * half_size * pixel_to_ndc.y * clip.w;
    var out: VertexOutput;
    out.clip_position = clip;
    out.color = select(style.color, point.color, style.options.y > 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
