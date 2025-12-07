struct ImportanceStats {
    importance_sum : f32,
    visibility_count : f32,
    max_screen_radius : f32,
    _pad : f32,
};

struct PruneDecisionParams {
    min_opacity : f32,
    max_scale : f32,
    max_screen_radius : f32,
    importance_threshold : f32,
    visibility_threshold : f32,
    active_count : u32,
    _pad0 : u32,
    _pad1 : u32,
};

@group(0) @binding(0) var<storage, read> importance_stats : array<ImportanceStats>;
@group(0) @binding(1) var<storage, read> train_opacity : array<f32>;
@group(0) @binding(2) var<storage, read> train_scale : array<f32>;
@group(0) @binding(3) var<storage, read_write> prune_actions : array<u32>;
@group(0) @binding(4) var<storage, read_write> prune_counts : array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params : PruneDecisionParams;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn prune_decide(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.active_count) {
        return;
    }

    let stat = importance_stats[idx];
    let vis = stat.visibility_count;
    let avg_importance = select(0.0, stat.importance_sum / max(vis, 1e-6), vis > 0.0);

    let opacity_logit = train_opacity[idx];
    let opacity = sigmoid(opacity_logit);

    let scale_base = idx * 3u;
    let log_scale_x = train_scale[scale_base + 0u];
    let log_scale_y = train_scale[scale_base + 1u];
    let log_scale_z = train_scale[scale_base + 2u];
    let max_scale = max(max(exp(log_scale_x), exp(log_scale_y)), exp(log_scale_z));

    var should_prune = false;
    if (opacity < params.min_opacity) {
        should_prune = true;
    }
    if (max_scale > params.max_scale) {
        should_prune = true;
    }
    if (stat.max_screen_radius > params.max_screen_radius) {
        should_prune = true;
    }
    if (vis >= params.visibility_threshold && avg_importance < params.importance_threshold) {
        should_prune = true;
    }

    if (should_prune) {
        prune_actions[idx] = 1u;
        atomicAdd(&prune_counts[0], 1u);
    } else {
        prune_actions[idx] = 0u;
    }
}
