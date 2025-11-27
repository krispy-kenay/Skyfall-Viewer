struct AdamConfig {
    lr                  : f32,
    beta1               : f32,
    beta2               : f32,
    eps                 : f32,
    bias_correction1_rcp: f32,
    bias_correction2_sqrt_rcp: f32,
    element_count       : u32,
    _pad                : u32,
};

@group(0) @binding(0) var<storage, read_write> param : array<f32>;
@group(0) @binding(1) var<storage, read_write> exp_avg : array<f32>;
@group(0) @binding(2) var<storage, read_write> exp_avg_sq : array<f32>;
@group(0) @binding(3) var<storage, read> grad : array<f32>;
@group(0) @binding(4) var<uniform> cfg : AdamConfig;

@compute @workgroup_size(256)
fn adam_step(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= cfg.element_count) {
        return;
    }

    let g = grad[idx];
    let m1_old = exp_avg[idx];
    let m2_old = exp_avg_sq[idx];

    let m1 = cfg.beta1 * m1_old + (1.0 - cfg.beta1) * g;
    let m2 = cfg.beta2 * m2_old + (1.0 - cfg.beta2) * g * g;

    let denom = sqrt(m2) * cfg.bias_correction2_sqrt_rcp + cfg.eps;
    let step_size = cfg.lr * cfg.bias_correction1_rcp;

    param[idx] = param[idx] - step_size * (m1 / denom);
    exp_avg[idx] = m1;
    exp_avg_sq[idx] = m2;
}


