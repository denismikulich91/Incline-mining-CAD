struct CameraUniform {
    view_proj: mat4x4<f32>,
    cam_forward: vec4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct ColorStop {
    color: vec4<f32>,
    // .x holds the stop's position (0..1); rest is padding.
    pos: vec4<f32>,
};
struct BlockModelStyle {
    fallback_color: vec4<f32>,
    // options.x: has grade colour; options.y: active colour stop count.
    options: vec4<f32>,
    stops: array<ColorStop, 12>,
};
@group(1) @binding(0)
var<uniform> block_style: BlockModelStyle;

const VISIBLE_ALPHA_EPSILON: f32 = 0.004;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) grade: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) grade: f32,
    @location(1) local_position: vec3<f32>,
};

fn ramp_rgb(t: f32, stop_count: i32) -> vec3<f32> {
    var before_color = vec4<f32>(0.0);
    var before_pos = 0.0;
    var has_before = false;
    var after_color = vec4<f32>(0.0);
    var after_pos = 0.0;
    var has_after = false;

    for (var i = 0; i < 12; i++) {
        if (i >= stop_count) {
            break;
        }
        let stop = block_style.stops[i];
        if (stop.color.a >= VISIBLE_ALPHA_EPSILON) {
            if (stop.pos.x <= t) {
                before_color = stop.color;
                before_pos = stop.pos.x;
                has_before = true;
            }
            if (!has_after && stop.pos.x >= t) {
                after_color = stop.color;
                after_pos = stop.pos.x;
                has_after = true;
            }
        }
    }

    if (has_before && has_after) {
        let span = max(after_pos - before_pos, 1e-6);
        let k = clamp((t - before_pos) / span, 0.0, 1.0);
        return mix(before_color.rgb, after_color.rgb, k);
    }
    if (has_before) {
        return before_color.rgb;
    }
    if (has_after) {
        return after_color.rgb;
    }
    return vec3<f32>(0.0);
}

// Walks the sorted colour-transfer stops. RGB ignores fully transparent stops
// so hidden handles don't tint the visible ramp, while alpha remains a hard
// interval value so transparent edge markers act as visibility cutoffs.
fn ramp_color(t: f32) -> vec4<f32> {
    let stop_count = max(2, i32(block_style.options.y + 0.5));
    let last_index = stop_count - 1;
    let first = block_style.stops[0];
    let last = block_style.stops[last_index];
    if (t <= first.pos.x) {
        return first.color;
    }
    if (t >= last.pos.x) {
        return last.color;
    }
    for (var i = 0; i < 11; i++) {
        if (i >= last_index) {
            break;
        }
        let a = block_style.stops[i];
        let b = block_style.stops[i + 1];
        if (t >= a.pos.x && t <= b.pos.x) {
            let rgb = ramp_rgb(t, stop_count);
            let alpha = max(a.color.a, b.color.a);
            return vec4<f32>(rgb, alpha);
        }
    }
    return last.color;
}

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.grade = model.grade;
    out.local_position = model.position;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var normal = normalize(cross(dpdx(in.local_position), dpdy(in.local_position)));
    if (dot(normal, camera.cam_forward.xyz) > 0.0) {
        normal = -normal;
    }
    let has_grade = block_style.options.x > 0.5 && in.grade >= 0.0;
    let grade_color = ramp_color(in.grade);
    if (has_grade && grade_color.a < 0.004) {
        discard;
    }
    let rgb = select(block_style.fallback_color.rgb, grade_color.rgb, has_grade);
    let alpha = select(block_style.fallback_color.a, grade_color.a, has_grade);
    let key_light = max(dot(normal, normalize(vec3<f32>(-0.60, -0.50, 0.35))), 0.0);
    let fill_light = max(dot(normal, normalize(vec3<f32>(0.45, 0.35, 0.75))), 0.0);
    let view_light = abs(dot(normal, -normalize(camera.cam_forward.xyz)));
    let intensity = 0.28 + 0.18 * view_light + 0.42 * key_light + 0.12 * fill_light;
    return vec4<f32>(rgb * intensity, alpha);
}
