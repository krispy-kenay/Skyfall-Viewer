struct LossParams {
    w_l2 : f32,
    w_l1 : f32,
    w_ssim : f32,  // D-SSIM weight (structural similarity)
    _pad1 : f32,
};

@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var targetTex : texture_2d<f32>;

@group(0) @binding(2) var training_color : texture_storage_2d<rgba32float, read>;
@group(0) @binding(3) var training_alpha : texture_storage_2d<r32float, read>;

@group(0) @binding(4) var residual_color : texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var residual_alpha : texture_storage_2d<r32float, write>;
@group(0) @binding(6) var<uniform> loss_params : LossParams;

const C1 : f32 = 0.01 * 0.01;
const C2 : f32 = 0.03 * 0.03;

fn samplePred(coord: vec2<i32>, dims: vec2<u32>) -> vec3<f32> {
    let x = clamp(coord.x, 0, i32(dims.x) - 1);
    let y = clamp(coord.y, 0, i32(dims.y) - 1);
    return textureLoad(training_color, vec2<i32>(x, y)).rgb;
}

fn sampleTarg(coord: vec2<i32>, dims: vec2<u32>) -> vec3<f32> {
    let u = (f32(coord.x) + 0.5) / f32(dims.x);
    let v = (f32(coord.y) + 0.5) / f32(dims.y);
    return textureSampleLevel(targetTex, samp, vec2<f32>(u, v), 0.0).rgb;
}

fn computeSSIMGrad(center: vec2<i32>, dims: vec2<u32>) -> vec3<f32> {
    // Compute SSIM using 11x11 window (matches PyTorch reference)
    var mu_x = vec3<f32>(0.0);
    var mu_y = vec3<f32>(0.0);
    let window_size = 11;
    let half_window = 5;
    let n = f32(window_size * window_size);

    // Compute local means (via box filter approximation of Gaussian)
    for (var dy = -half_window; dy <= half_window; dy = dy + 1) {
        for (var dx = -half_window; dx <= half_window; dx = dx + 1) {
            let coord = center + vec2<i32>(dx, dy);
            mu_x = mu_x + samplePred(coord, dims);
            mu_y = mu_y + sampleTarg(coord, dims);
        }
    }
    mu_x = mu_x / n;
    mu_y = mu_y / n;

    // Compute local variances and covariance
    var sigma_x2 = vec3<f32>(0.0);
    var sigma_y2 = vec3<f32>(0.0);
    var sigma_xy = vec3<f32>(0.0);

    for (var dy = -half_window; dy <= half_window; dy = dy + 1) {
        for (var dx = -half_window; dx <= half_window; dx = dx + 1) {
            let coord = center + vec2<i32>(dx, dy);
            let x = samplePred(coord, dims);
            let y = sampleTarg(coord, dims);
            let dx_val = x - mu_x;
            let dy_val = y - mu_y;
            sigma_x2 = sigma_x2 + dx_val * dx_val;
            sigma_y2 = sigma_y2 + dy_val * dy_val;
            sigma_xy = sigma_xy + dx_val * dy_val;
        }
    }
    sigma_x2 = sigma_x2 / n;
    sigma_y2 = sigma_y2 / n;
    sigma_xy = sigma_xy / n;

    // SSIM formula: ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))
    let luminance_num = 2.0 * mu_x * mu_y + C1;
    let luminance_den = mu_x * mu_x + mu_y * mu_y + C1;
    let contrast_num = 2.0 * sigma_xy + C2;
    let contrast_den = sigma_x2 + sigma_y2 + C2;

    let pred = samplePred(center, dims);
    let targ = sampleTarg(center, dims);

    // Compute gradient of (1 - SSIM) loss using chain rule approximation
    // This is a simplified gradient that works reasonably well in practice
    // Full analytical gradient involves complex chain rule through all statistics

    // dL/dSSIM for loss = (1 - SSIM) is simply -1
    // Now we need dSSIM/dpred

    // Approximate gradient based on local statistics and pixel difference
    // This captures the main gradient behavior without full analytical derivation
    let ssim_val = (luminance_num * contrast_num) / (luminance_den * contrast_den);

    // Gradient contribution from luminance term (∂SSIM/∂μ_x affects all pixels in window)
    let dL_dmu_x = -2.0 * (mu_y * luminance_den - luminance_num * mu_x) / (luminance_den * luminance_den);
    let grad_from_mean = dL_dmu_x * contrast_num / contrast_den / n;

    // Gradient contribution from variance/covariance terms
    let pred_centered = pred - mu_x;
    let targ_centered = targ - mu_y;

    // ∂σ_xy/∂pred_pixel ≈ (targ - μ_y) / n
    let dL_dsigma_xy = -2.0 * luminance_num / luminance_den / contrast_den;
    let grad_from_cov = dL_dsigma_xy * targ_centered / n;

    // ∂σ_x^2/∂pred_pixel ≈ 2*(pred - μ_x) / n
    let dL_dsigma_x2 = luminance_num * contrast_num * (-1.0) / (luminance_den * contrast_den * contrast_den);
    let grad_from_var = dL_dsigma_x2 * 2.0 * pred_centered / n;

    // Total gradient (negative because we minimize 1-SSIM)
    return -(grad_from_mean + grad_from_cov + grad_from_var);
}

@compute @workgroup_size(8, 8, 1)
fn training_loss(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(training_color);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let center = vec2<i32>(i32(gid.x), i32(gid.y));
    let u = (f32(gid.x) + 0.5) / f32(dims.x);
    let v = (f32(gid.y) + 0.5) / f32(dims.y);
    let uv = vec2<f32>(u, v);

    // Predicted image from training forward pass
    let pred = textureLoad(training_color, center).rgb;

    // Target image from dataset
    let targ = textureSampleLevel(targetTex, samp, uv, 0.0).rgb;

    // L2 gradient: grad = pred - targ
    let diff = pred - targ;
    let grad_l2 = diff;
    
    // L1 gradient: grad = sign(pred - targ)
    let grad_l1 = sign(diff);

    // Combined gradient
    var grad = loss_params.w_l2 * grad_l2 + loss_params.w_l1 * grad_l1;
    
    // Add SSIM gradient if weight > 0
    if (loss_params.w_ssim > 0.0) {
        let grad_ssim = computeSSIMGrad(center, dims);
        grad = grad + loss_params.w_ssim * grad_ssim;
    }

    // Alpha residual = 0 for now
    let _alpha_pred = textureLoad(training_alpha, center).r;
    let alpha_residual = 0.0;

    textureStore(residual_color, center, vec4<f32>(grad, 0.0));
    textureStore(residual_alpha, center, vec4<f32>(alpha_residual, 0.0, 0.0, 0.0));
}


