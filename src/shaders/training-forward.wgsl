@group(0) @binding(0) var<uniform> settings: RenderSettings;

@group(0) @binding(1) var<storage, read> splats : array<Splat>;
@group(0) @binding(2) var<storage, read> sorted_vis_indices : array<u32>;

struct DrawIndirectArgs {
    vertex_count   : u32,
    instance_count : u32,
    first_vertex   : u32,
    first_instance : u32,
};

@group(0) @binding(3) var<storage, read> draw_args : DrawIndirectArgs;

@group(0) @binding(4) var training_color : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(5) var training_alpha : texture_storage_2d<r32float, read_write>;

@compute @workgroup_size(8, 8, 1)
fn training_forward(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(training_color);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let vpx = settings.viewport_x;
    let vpy = settings.viewport_y;
    let pixel = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

    var color = vec3<f32>(0.0, 0.0, 0.0);
    var alpha = 0.0;

    let instance_count = draw_args.instance_count;

    for (var i = 0u; i < instance_count; i = i + 1u) {
        let vis_idx = sorted_vis_indices[i];
        let s = splats[vis_idx];

        let center_px = (s.center_ndc * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5)) * vec2<f32>(vpx, vpy);

        let d = pixel - center_px;
        let a = s.a11;
        let b = s.a12;
        let c = s.a22;
        let t = a * d.x * d.x + 2.0 * b * d.x * d.y + c * d.y * d.y;
        if (t > 9.0) {
            continue;
        }

        let alpha_splat = clamp(s.opacity * exp(-0.5 * t), 0.0, 1.0);
        if (alpha_splat <= 0.0) {
            continue;
        }

        let one_minus_alpha = 1.0 - alpha;
        let contrib_alpha = alpha_splat * one_minus_alpha;
        color += s.color * contrib_alpha;
        alpha += contrib_alpha;

        if (1.0 - alpha < 1e-4) {
            break;
        }
    }

    textureStore(training_color, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(color, 1.0));
    textureStore(training_alpha, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(alpha, 0.0, 0.0, 0.0));
}


