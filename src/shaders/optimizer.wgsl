struct OptimizeSettings {
    learning_rate_color  : f32,
    learning_rate_alpha  : f32,
    min_alpha            : f32,
    max_alpha            : f32,
    learning_rate_size   : f32,
    min_size             : f32,
    max_size             : f32,
    target_error         : f32,
};

@group(0) @binding(0)
var<storage, read_write> gaussians : array<Gaussian>;

@group(0) @binding(1)
var<uniform> optimizeSettings : OptimizeSettings;

@group(0) @binding(2)
var samp : sampler;

@group(0) @binding(3)
var targetTex : texture_2d<f32>;

@group(0) @binding(4)
var<storage, read> splats : array<Splat>;

@group(0) @binding(5)
var<storage, read_write> sh_buffer : array<u32>;

@group(0) @binding(6) var<uniform> camera: CameraUniforms;
@group(0) @binding(7) var<uniform> settings: RenderSettings;
@group(0) @binding(8) var<storage, read> density_buffer : array<f32>;
@group(0) @binding(9) var<uniform> active_count : u32;


fn read_f16_coeff(base_word: u32, elem: u32) -> f32 {
    let word_idx = base_word + (elem >> 1u);
    let halves   = unpack2x16float(sh_buffer[word_idx]);
    if ((elem & 1u) == 0u) {
        return halves.x;
    } else {
        return halves.y;
    }
}

fn write_f16_coeff(base_word: u32, elem: u32, value: f32) {
    let word_idx = base_word + (elem >> 1u);
    var halves   = unpack2x16float(sh_buffer[word_idx]);
    if ((elem & 1u) == 0u) {
        halves.x = value;
    } else {
        halves.y = value;
    }
    sh_buffer[word_idx] = pack2x16float(halves);
}

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let base_word = splat_idx * SH_WORDS_PER_POINT;
    let eR = c_idx * 3u + 0u;
    let eG = c_idx * 3u + 1u;
    let eB = c_idx * 3u + 2u;
    return vec3<f32>(
        read_f16_coeff(base_word, eR),
        read_f16_coeff(base_word, eG),
        read_f16_coeff(base_word, eB)
    );
}

fn write_sh_coef(splat_idx: u32, c_idx: u32, value: vec3<f32>) {
    let base_word = splat_idx * SH_WORDS_PER_POINT;
    let eR = c_idx * 3u + 0u;
    let eG = c_idx * 3u + 1u;
    let eB = c_idx * 3u + 2u;
    write_f16_coeff(base_word, eR, value.x);
    write_f16_coeff(base_word, eG, value.y);
    write_f16_coeff(base_word, eB, value.z);
}

fn apply_sh_gradient(splat_idx: u32, coeff_idx: u32, scalar: f32, residual: vec3<f32>, lr: f32) {
    if (abs(scalar) < 1e-5) {
        return;
    }
    var coeff = sh_coef(splat_idx, coeff_idx);
    coeff += lr * scalar * residual;
    coeff = clamp(coeff, vec3<f32>(-2.0), vec3<f32>(2.0));
    write_sh_coef(splat_idx, coeff_idx, coeff);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

fn entropy_grad(alpha : f32) -> f32 {
    let eps = 1e-4;
    let a = clamp(alpha, eps, 1.0 - eps);
    return -log(a) + log(1.0 - a);
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn logit(x: f32) -> f32 {
    let clamped = clamp(x, 1e-4, 1.0 - 1e-4);
    return log(clamped / (1.0 - clamped));
}

fn sign_f(x : f32) -> f32 {
    return select(-1.0, 1.0, x > 0.0);
}

@compute @workgroup_size(256)
fn optimizer(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= active_count) {
        return;
    }
    var g = gaussians[idx];
    var s = splats[idx];

    var p01 = unpack2x16float(g.pos_opacity[0u]);
    var p02 = unpack2x16float(g.pos_opacity[1u]);

    let position_world = vec4<f32>(p01.x, p01.y, p02.x, 1.0);
    let position_camera = camera.view * position_world;
    let position_clip = camera.proj * position_camera;
    let ndc = position_clip.xy / position_clip.w;
    let uv = 0.5 * (ndc + vec2<f32>(1.0, 1.0));

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        let alpha = sigmoid(p02.y);
        let ga = entropy_grad(alpha);
        let mean_dist = density_buffer[idx];
        let typical_mean_dist = 50.0;
        let is_isolated = mean_dist > typical_mean_dist * 2.0;
        
        let isolation_factor = select(1.0, 3.0, is_isolated);
        let alpha_push = -ga * isolation_factor;
        
        let updated_alpha = clamp(
            alpha - optimizeSettings.learning_rate_alpha * alpha_push,
            optimizeSettings.min_alpha,
            optimizeSettings.max_alpha
        );
        p02.y = logit(updated_alpha);

        g.pos_opacity[0u] = pack2x16float(p01);
        g.pos_opacity[1u] = pack2x16float(p02);

        gaussians[idx] = g;
        return;
    }

    let lr_c = optimizeSettings.learning_rate_color;
    let targ = textureSampleLevel(targetTex, samp, uv, 0.0).rgb;

    let viewDir = normalize(-(camera.view * position_world).xyz);
    let x = viewDir.x;
    let y = viewDir.y;
    let z = viewDir.z;

    let sh_deg = u32(settings.sh_deg);
    var shColor = computeColorFromSH(viewDir, idx, sh_deg);

    let base = idx * SH_WORDS_PER_POINT;

    var dc = vec3<f32>(
        read_f16_coeff(base, 0u),
        read_f16_coeff(base, 1u),
        read_f16_coeff(base, 2u)
    );

    dc = mix(dc, targ / vec3<f32>(SH_C0), optimizeSettings.learning_rate_color);
    dc = clamp(dc, vec3<f32>(-2.0), vec3<f32>(2.0));

    var word0 = unpack2x16float(sh_buffer[base + 0u]);
    word0.x = dc.x;
    word0.y = dc.y;
    sh_buffer[base + 0u] = pack2x16float(word0);
    var word1 = unpack2x16float(sh_buffer[base + 1u]);
    word1.x = dc.z;
    sh_buffer[base + 1u] = pack2x16float(word1);

    shColor = computeColorFromSH(viewDir, idx, sh_deg);

    if (sh_deg > 0u) {
        let residual = targ - shColor;

        apply_sh_gradient(idx, 1u, -SH_C1 * y, residual, lr_c);
        apply_sh_gradient(idx, 2u,  SH_C1 * z, residual, lr_c);
        apply_sh_gradient(idx, 3u, -SH_C1 * x, residual, lr_c);

        if (sh_deg > 1u) {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let yz = y * z;
            let xz = x * z;

            apply_sh_gradient(idx, 4u, SH_C2[0] * xy, residual, lr_c);
            apply_sh_gradient(idx, 5u, SH_C2[1] * yz, residual, lr_c);
            apply_sh_gradient(idx, 6u, SH_C2[2] * (2.0 * zz - xx - yy), residual, lr_c);
            apply_sh_gradient(idx, 7u, SH_C2[3] * xz, residual, lr_c);
            apply_sh_gradient(idx, 8u, SH_C2[4] * (xx - yy), residual, lr_c);

            if (sh_deg > 2u) {
                apply_sh_gradient(idx, 9u,  SH_C3[0] * y * (3.0 * xx - yy), residual, lr_c);
                apply_sh_gradient(idx, 10u, SH_C3[1] * xy * z, residual, lr_c);
                apply_sh_gradient(idx, 11u, SH_C3[2] * y * (4.0 * zz - xx - yy), residual, lr_c);
                apply_sh_gradient(idx, 12u, SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy), residual, lr_c);
                apply_sh_gradient(idx, 13u, SH_C3[4] * x * (4.0 * zz - xx - yy), residual, lr_c);
                apply_sh_gradient(idx, 14u, SH_C3[5] * z * (xx - yy), residual, lr_c);
                apply_sh_gradient(idx, 15u, SH_C3[6] * x * (xx - 3.0 * yy), residual, lr_c);
            }
        }

        shColor = computeColorFromSH(viewDir, idx, sh_deg);
    }

    let luminance_weights = vec3<f32>(0.299, 0.587, 0.114);
    let targ_lum = dot(targ, luminance_weights);
    let pred_lum = dot(shColor, luminance_weights);
    let lum_err = clamp(targ_lum - pred_lum, -1.0, 1.0);

    let current_alpha = sigmoid(p02.y);
    
    let mean_dist = density_buffer[idx];
    let typical_mean_dist = 50.0;
    let is_isolated = mean_dist > typical_mean_dist * 2.0;
    
    let isolation_factor = select(1.0, 3.0, is_isolated);
    let isolation_penalty = select(0.0, -0.3 * isolation_factor, is_isolated);
    
    let entropy_push = entropy_grad(current_alpha);
    let entropy_weight = select(0.05, 0.15, is_isolated);
    let alpha_push = lum_err * settings.alpha_scale - entropy_weight * entropy_push + isolation_penalty;
    
    let alpha_lr = select(optimizeSettings.learning_rate_alpha, optimizeSettings.learning_rate_alpha * 2.0, is_isolated);
    let updated_alpha = clamp(
        current_alpha + alpha_lr * alpha_push,
        optimizeSettings.min_alpha,
        optimizeSettings.max_alpha
    );
    p02.y = logit(updated_alpha);

    var s01 = unpack2x16float(g.scale[0u]);
    var s02 = unpack2x16float(g.scale[1u]);

    let sigma_now = exp(clamp(s01.x, -10.0, 10.0));
    let current_pixels = sigma_now / 0.003;
    
    let density_target_pixels = (mean_dist / 3.0) / 0.003;
    
    let color_loss = length(targ - shColor);
    
    let coverage_radius = density_target_pixels / 1024.0;
    let coverage_radius_clamped = clamp(coverage_radius, 1.0 / 1024.0, 8.0 / 1024.0);
    
    let coverage_samples = array<vec2<f32>, 9>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(coverage_radius_clamped, 0.0),
        vec2<f32>(-coverage_radius_clamped, 0.0),
        vec2<f32>(0.0, coverage_radius_clamped),
        vec2<f32>(0.0, -coverage_radius_clamped),
        vec2<f32>(coverage_radius_clamped * 0.707, coverage_radius_clamped * 0.707),
        vec2<f32>(-coverage_radius_clamped * 0.707, coverage_radius_clamped * 0.707),
        vec2<f32>(coverage_radius_clamped * 0.707, -coverage_radius_clamped * 0.707),
        vec2<f32>(-coverage_radius_clamped * 0.707, -coverage_radius_clamped * 0.707)
    );
    
    var coverage_error_sum = 0.0;
    var coverage_count = 0.0;
    
    for (var i = 0u; i < 9u; i++) {
        let sample_uv = uv + coverage_samples[i];
        if (sample_uv.x >= 0.0 && sample_uv.x <= 1.0 && sample_uv.y >= 0.0 && sample_uv.y <= 1.0) {
            let targ_sample = textureSampleLevel(targetTex, samp, sample_uv, 0.0).rgb;
            let sample_lum = dot(targ_sample, vec3<f32>(0.299, 0.587, 0.114));
            let dist_from_center = length(coverage_samples[i]) / coverage_radius_clamped;
            
            if (sample_lum > 0.1 && dist_from_center > (current_pixels / density_target_pixels)) {
                let uncovered_error = sample_lum * dist_from_center;
                coverage_error_sum += uncovered_error;
                coverage_count += 1.0;
            }
        }
    }
    
    let coverage_loss = select(0.0, coverage_error_sum / max(coverage_count, 1.0), coverage_count > 0.0);
    
    let density_deviation = current_pixels - density_target_pixels;
    let density_loss_base = abs(density_deviation) / max(density_target_pixels, 1.0);
    
    let shrink_penalty = select(0.0, (density_target_pixels - current_pixels) / max(density_target_pixels, 1.0) * 2.0, current_pixels < density_target_pixels * 0.5);
    let density_loss = density_loss_base + shrink_penalty;

    let lambda_coverage = 8.0;
    let lambda_density = 0.5;
    
    let total_loss = color_loss + lambda_coverage * coverage_loss + lambda_density * density_loss;
    
    // ----- Gradient Computation (Finite Difference) -----
    let size_delta_ratio = 0.15;
    let perturbed_pixels = current_pixels * (1.0 + size_delta_ratio);
    
    var coverage_loss_large = 0.0;
    if (coverage_count > 0.0) {
        var coverage_error_large_sum = 0.0;
        var coverage_count_large = 0.0;
        for (var i = 0u; i < 9u; i++) {
            let sample_uv = uv + coverage_samples[i];
            if (sample_uv.x >= 0.0 && sample_uv.x <= 1.0 && sample_uv.y >= 0.0 && sample_uv.y <= 1.0) {
                let targ_sample = textureSampleLevel(targetTex, samp, sample_uv, 0.0).rgb;
                let sample_lum = dot(targ_sample, vec3<f32>(0.299, 0.587, 0.114));
                let dist_from_center = length(coverage_samples[i]) / coverage_radius_clamped;
                
                let coverage_ratio_current = current_pixels / max(density_target_pixels, 1.0);
                let coverage_ratio_large = perturbed_pixels / max(density_target_pixels, 1.0);
                
                if (sample_lum > 0.1 && dist_from_center > coverage_ratio_large) {
                    let uncovered_error = sample_lum * dist_from_center;
                    coverage_error_large_sum += uncovered_error;
                    coverage_count_large += 1.0;
                }
            }
        }
        coverage_loss_large = select(0.0, coverage_error_large_sum / max(coverage_count_large, 1.0), coverage_count_large > 0.0);
    }
    
    let density_deviation_large = perturbed_pixels - density_target_pixels;
    let density_loss_large_base = abs(density_deviation_large) / max(density_target_pixels, 1.0);
    let shrink_penalty_large = select(0.0, (density_target_pixels - perturbed_pixels) / max(density_target_pixels, 1.0) * 2.0, perturbed_pixels < density_target_pixels * 0.5);
    let density_loss_large = density_loss_large_base + shrink_penalty_large;
    
    var targ_avg_sum = vec3<f32>(0.0);
    var targ_avg_count = 0.0;
    for (var i = 0u; i < 9u; i++) {
        let sample_uv = uv + coverage_samples[i];
        if (sample_uv.x >= 0.0 && sample_uv.x <= 1.0 && sample_uv.y >= 0.0 && sample_uv.y <= 1.0) {
            targ_avg_sum += textureSampleLevel(targetTex, samp, sample_uv, 0.0).rgb;
            targ_avg_count += 1.0;
        }
    }
    let targ_avg = targ_avg_sum / max(targ_avg_count, 1.0);
    let color_loss_large = length(targ_avg - shColor);
    
    let total_loss_large = color_loss_large + lambda_coverage * coverage_loss_large + lambda_density * density_loss_large;
    let loss_gradient = (total_loss_large - total_loss) / size_delta_ratio;
    
    // ----- Gradient Clipping -----
    let clamped_gradient = clamp(loss_gradient, -3.0, 3.0);
    
    // ----- Size Update (Gradient Descent on Combined Loss) -----
    let size_update = -clamped_gradient * optimizeSettings.learning_rate_size;
    var desired_pixels = current_pixels * (1.0 + size_update);
    
    // ----- Constraints -----
    let density_min = density_target_pixels * 0.8;
    desired_pixels = max(desired_pixels, density_min);
    
    let effective_max = max(optimizeSettings.max_size, density_target_pixels * 5.0);
    desired_pixels = clamp(desired_pixels, optimizeSettings.min_size, effective_max);

    let sigma_target = desired_pixels * 0.003;
    let new_sigma = mix(sigma_now, sigma_target, optimizeSettings.learning_rate_size * 2.0);

    s01.x = log(new_sigma);
    s01.y = log(new_sigma);
    s02.x = log(new_sigma);

    g.scale[0u] = pack2x16float(s01);
    g.scale[1u] = pack2x16float(s02);

    g.pos_opacity[0u] = pack2x16float(p01);
    g.pos_opacity[1u] = pack2x16float(p02);

    gaussians[idx] = g;
}