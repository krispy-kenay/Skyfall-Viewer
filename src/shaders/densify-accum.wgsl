
const GEOM_GRAD_STRIDE : u32 = 5u;
const GEOM_GRAD_CONIC_A : u32 = 0u;
const GEOM_GRAD_CONIC_B : u32 = 1u;
const GEOM_GRAD_CONIC_C : u32 = 2u;
const GEOM_GRAD_MEAN2D_X : u32 = 3u;
const GEOM_GRAD_MEAN2D_Y : u32 = 4u;

const GRAD_ACCUM_STRIDE : u32 = 4u;
const GRAD_ACCUM_X : u32 = 0u;
const GRAD_ACCUM_Y : u32 = 1u;
const GRAD_ACCUM_ABS : u32 = 2u;
const GRAD_ACCUM_DENOM : u32 = 3u;

@group(0) @binding(0) var<storage, read> grad_geom : array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_accum : array<f32>;
@group(0) @binding(2) var<uniform> active_count : u32;

@compute @workgroup_size(256)
fn accumulate_grads(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= active_count) {
        return;
    }
    
    let geom_base = idx * GEOM_GRAD_STRIDE;
    let grad_x = grad_geom[geom_base + GEOM_GRAD_MEAN2D_X];
    let grad_y = grad_geom[geom_base + GEOM_GRAD_MEAN2D_Y];
    
    let grad_a = grad_geom[geom_base + GEOM_GRAD_CONIC_A];
    let grad_b = grad_geom[geom_base + GEOM_GRAD_CONIC_B];
    let grad_c = grad_geom[geom_base + GEOM_GRAD_CONIC_C];
    
    let grad_2d_norm = sqrt(grad_x * grad_x + grad_y * grad_y);
    let grad_abs_norm = sqrt(grad_a * grad_a + grad_b * grad_b + grad_c * grad_c);
    
    let threshold = 1e-10;
    if (grad_2d_norm > threshold || grad_abs_norm > threshold) {
        let accum_base = idx * GRAD_ACCUM_STRIDE;
        
        grad_accum[accum_base + GRAD_ACCUM_X] += grad_2d_norm;
        grad_accum[accum_base + GRAD_ACCUM_Y] = 0.0;  // Not used
        grad_accum[accum_base + GRAD_ACCUM_ABS] += grad_abs_norm;
        grad_accum[accum_base + GRAD_ACCUM_DENOM] += 1.0;
    }
}

