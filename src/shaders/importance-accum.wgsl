struct ImportanceStats {
    importance_sum : f32,
    visibility_count : f32,
    max_screen_radius : f32,
    _pad : f32,
};

struct ImportanceAccumParams {
    active_count : u32,
    decay : f32,
    _pad0 : f32,
    _pad1 : f32,
};

@group(0) @binding(0) var<storage, read> gauss_projections : array<GaussProjection>;
@group(0) @binding(1) var<storage, read_write> importance_stats : array<ImportanceStats>;
@group(0) @binding(2) var<uniform> params : ImportanceAccumParams;

@compute @workgroup_size(256)
fn accumulate_importance(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.active_count) {
        return;
    }

    let proj = gauss_projections[idx];
    var stat = importance_stats[idx];

    stat.importance_sum *= params.decay;
    stat.visibility_count *= params.decay;
    stat.max_screen_radius *= params.decay;

    if (proj.tiles_count > 0u) {
        stat.importance_sum += f32(proj.tiles_count);
        stat.visibility_count += 1.0;
        stat.max_screen_radius = max(stat.max_screen_radius, proj.radius);
    }

    importance_stats[idx] = stat;
}
