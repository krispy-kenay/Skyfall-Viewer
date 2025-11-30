struct LossParams {
    w_l2 : f32,
    w_l1 : f32,
    _pad0 : f32,
    _pad1 : f32,
};

@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var targetTex : texture_2d<f32>;

@group(0) @binding(2) var training_color : texture_storage_2d<rgba32float, read>;
@group(0) @binding(3) var training_alpha : texture_storage_2d<r32float, read>;

@group(0) @binding(4) var residual_color : texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var residual_alpha : texture_storage_2d<r32float, write>;
@group(0) @binding(6) var<uniform> loss_params : LossParams;

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
    let pred = textureLoad(training_color, vec2<i32>(i32(gid.x), i32(gid.y))).rgb;

    // Target image from dataset
    let targ = textureSampleLevel(targetTex, samp, uv, 0.0).rgb;

    // Simple combined L2 + L1 gradient: grad = w_l2 * (pred - targ) + w_l1 * sign(pred - targ)
    let diff = pred - targ;
    let grad_l2 = diff;
    let grad_l1 = select(-vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), diff >= vec3<f32>(0.0, 0.0, 0.0));
    let grad = loss_params.w_l2 * grad_l2 + loss_params.w_l1 * grad_l1;

    // Alpha residual = 0 for now
    let _alpha_pred = textureLoad(training_alpha, vec2<i32>(i32(gid.x), i32(gid.y))).r;
    let alpha_residual = 0.0;

    textureStore(residual_color, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(grad, 0.0));
    textureStore(residual_alpha, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(alpha_residual, 0.0, 0.0, 0.0));
}


