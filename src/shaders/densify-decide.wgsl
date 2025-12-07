const GRAD_ACCUM_STRIDE : u32 = 4u;
const GRAD_ACCUM_X : u32 = 0u;
const GRAD_ACCUM_Y : u32 = 1u;
const GRAD_ACCUM_ABS : u32 = 2u;
const GRAD_ACCUM_DENOM : u32 = 3u;

const ACTION_NONE : u32 = 0u;
const ACTION_CLONE : u32 = 1u;
const ACTION_SPLIT : u32 = 2u;

struct ImportanceStats {
    importance_sum : f32,
    visibility_count : f32,
    max_screen_radius : f32,
    _pad : f32,
};

struct DensifyParams {
    grad_threshold : f32,
    grad_abs_threshold : f32,
    size_threshold : f32,
    opacity_cap : f32,
    active_count : f32,
    min_visibility : f32,
    importance_threshold : f32,
    _pad3 : f32,
};

@group(0) @binding(0) var<storage, read> grad_accum : array<f32>;
@group(0) @binding(1) var<storage, read> scales : array<f32>; 
@group(0) @binding(2) var<storage, read_write> actions : array<u32>;
@group(0) @binding(3) var<storage, read_write> counts : array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params : DensifyParams;
@group(0) @binding(5) var<storage, read> importance_stats : array<ImportanceStats>;

@compute @workgroup_size(256)
fn densify_decide(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    let active_count = u32(params.active_count);
    if (idx >= active_count) {
        return;
    }
    
    let accum_base = idx * GRAD_ACCUM_STRIDE;
    let denom = grad_accum[accum_base + GRAD_ACCUM_DENOM];
    
    if (denom < max(params.min_visibility, 1.0)) {
        actions[idx] = ACTION_NONE;
        return;
    }

    let stat = importance_stats[idx];
    let avg_importance = stat.importance_sum / max(stat.visibility_count, 1.0);
    if (avg_importance < params.importance_threshold) {
        actions[idx] = ACTION_NONE;
        return;
    }
    
    let avg_grad_2d_norm = grad_accum[accum_base + GRAD_ACCUM_X] / denom;
    let avg_grad_abs = grad_accum[accum_base + GRAD_ACCUM_ABS] / denom;
    
    let scale_base = idx * 3u;
    let scale_x = exp(scales[scale_base + 0u]);
    let scale_y = exp(scales[scale_base + 1u]);
    let scale_z = exp(scales[scale_base + 2u]);
    let max_scale = max(max(scale_x, scale_y), scale_z);
    
    var action = ACTION_NONE;
    
    if (max_scale <= params.size_threshold && avg_grad_2d_norm >= params.grad_threshold) {
        action = ACTION_CLONE;
        atomicAdd(&counts[0u], 1u);
        atomicAdd(&counts[2u], 1u);
    }
    else if (max_scale > params.size_threshold && avg_grad_abs >= params.grad_abs_threshold) {
        action = ACTION_SPLIT;
        atomicAdd(&counts[1u], 1u);
        atomicAdd(&counts[2u], 2u);
    }
    
    actions[idx] = action;
}
