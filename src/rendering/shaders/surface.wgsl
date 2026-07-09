struct CameraUniform {
    view_proj: mat4x4<f32>,
    cam_forward: vec4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct SurfaceStyle {
    color: vec4<f32>,
};
@group(1) @binding(0)
var<uniform> surface_style: SurfaceStyle;

// Scene-origin-relative offset of this chunk's local origin. Vertices are
// stored relative to their chunk's AABB centre so interpolated positions keep
// f32 precision far from the scene origin.
struct SurfaceChunk {
    offset: vec4<f32>,
};
@group(2) @binding(0)
var<uniform> chunk: SurfaceChunk;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    // Flat face normal from the triangle's provoking vertex, unit length and
    // pre-oriented with z >= 0 on the CPU. Using a stored normal instead of
    // dpdx/dpdy of the position avoids derivative cancellation noise when the
    // per-pixel position delta approaches the f32 ULP (static-like speckle,
    // worst close-up in fly mode).
    @location(1) @interpolate(flat) normal: vec3<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.color = surface_style.color;
    out.normal = model.normal;
    out.clip_position = camera.view_proj * vec4<f32>(model.position + chunk.offset.xyz, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = in.normal;
    // Oblique key light from NW at a low sun angle — maximises contrast between
    // faces at different slopes because horizontal faces receive little key light
    // while lit slopes receive full intensity.  A soft fill from the opposite-high
    // direction prevents unlit faces from going fully black.  A small camera
    // headlight keeps faces perpendicular to the view direction readable.
    let key_light = max(dot(normal, normalize(vec3<f32>(-0.60, -0.50, 0.30))), 0.0);
    let fill_light = max(dot(normal, normalize(vec3<f32>(0.40, 0.35, 0.75))), 0.0);
    let view_light = abs(dot(normal, -normalize(camera.cam_forward.xyz)));
    let intensity = 0.04 + 0.10 * view_light + 0.72 * key_light + 0.14 * fill_light;
    return vec4<f32>(in.color.rgb * intensity, in.color.a);
}
