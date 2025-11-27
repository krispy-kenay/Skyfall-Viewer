@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var targetTex : texture_2d<f32>;

@group(0) @binding(2) var training_color : texture_storage_2d<rgba32float, read>;
@group(0) @binding(3) var training_alpha : texture_storage_2d<r32float, read>;

@group(0) @binding(4) var residual_color : texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var residual_alpha : texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn training_loss(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(training_color);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let u = (f32(gid.x) + 0.5) / f32(dims.x);
    let v = (f32(gid.y) + 0.5) / f32(dims.y);
    let uv = vec2<f32>(u, v);

    // Predicted image from training forward pass
    let pred = textureLoad(training_color, vec2<i32>(i32(gid.x), i32(gid.y)), 0).rgb;

    // Target image from dataset
    let targ = textureSampleLevel(targetTex, samp, uv, 0.0).rgb;

    // Simple L2 grad
    let diff = pred - targ;

    // Alpha residual = 0 for now
    let _alpha_pred = textureLoad(training_alpha, vec2<i32>(i32(gid.x), i32(gid.y)), 0).r;
    let alpha_residual = 0.0;

    textureStore(residual_color, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(diff, 0.0));
    textureStore(residual_alpha, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(alpha_residual, 0.0, 0.0, 0.0));
}


