struct MetricImageParams {
    image_width : u32,
    image_height : u32,
    pixel_count : u32,
    _pad : u32,
};

struct MetricThresholdParams {
    min_value : f32,
    inv_range : f32,
    loss_threshold : f32,
    pixel_count : u32,
};

struct MetricAccumParams {
    pixel_count : u32,
    image_width : u32,
    tile_width : u32,
    tile_height : u32,
    tile_size : u32,
    active_count : u32,
    total_tiles : u32,
    _pad : u32,
};

@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var targetTex : texture_2d<f32>;
@group(0) @binding(2) var training_color : texture_storage_2d<rgba32float, read>;
@group(0) @binding(3) var<storage, read_write> pixel_loss : array<f32>;
@group(0) @binding(4) var<uniform> image_params : MetricImageParams;

@compute @workgroup_size(8, 8, 1)
fn compute_metric_map(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (gid.x >= image_params.image_width || gid.y >= image_params.image_height) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let dims = vec2<f32>(f32(image_params.image_width), f32(image_params.image_height));
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5, 0.5)) / dims;

    let pred = textureLoad(training_color, coord).rgb;
    let targ = textureSampleLevel(targetTex, samp, uv, 0.0).rgb;

    let diff = abs(pred - targ);
    let l1 = (diff.x + diff.y + diff.z) / 3.0;

    let idx = gid.y * image_params.image_width + gid.x;
    pixel_loss[idx] = l1;
}

@group(0) @binding(0) var<storage, read> pixel_loss_read : array<f32>;
@group(0) @binding(1) var<storage, read_write> pixel_flags : array<u32>;
@group(0) @binding(2) var<uniform> threshold_params : MetricThresholdParams;

@compute @workgroup_size(256)
fn threshold_metric_map(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= threshold_params.pixel_count) {
        return;
    }

    let value = (pixel_loss_read[idx] - threshold_params.min_value) * threshold_params.inv_range;
    let flag = select(0u, 1u, value > threshold_params.loss_threshold);
    pixel_flags[idx] = flag;
}

struct TilingParams {
    image_width : u32,
    image_height : u32,
    tile_size : u32,
    tile_width : u32,
    tile_height : u32,
    num_cameras : u32,
};

@group(0) @binding(0) var<storage, read> flagged_pixels : array<u32>;
@group(0) @binding(1) var<storage, read> tile_offsets : array<u32>;
@group(0) @binding(2) var<storage, read> flatten_ids : array<u32>;
@group(0) @binding(3) var<storage, read> gauss_projections : array<GaussProjection>;
@group(0) @binding(4) var<storage, read> last_ids : array<u32>;
@group(0) @binding(5) var<uniform> tiling : TilingParams;
@group(0) @binding(6) var<uniform> accum_params : MetricAccumParams;
@group(0) @binding(7) var<storage, read_write> gaussian_counts : array<atomic<u32>>;

@compute @workgroup_size(256)
fn accumulate_metric_counts(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= accum_params.pixel_count) {
        return;
    }

    if (flagged_pixels[idx] == 0u) {
        return;
    }

    let width = tiling.image_width;
    let x = idx % width;
    let y = idx / width;

    let tile_size_f = f32(tiling.tile_size);
    let tile_x = u32(f32(x) / tile_size_f);
    let tile_y = u32(f32(y) / tile_size_f);

    if (tile_x >= tiling.tile_width || tile_y >= tiling.tile_height) {
        return;
    }

    let tile_id = tile_y * tiling.tile_width + tile_x;
    if (tile_id >= accum_params.total_tiles) {
        return;
    }

    let start = tile_offsets[tile_id];
    let end = tile_offsets[tile_id + 1u];
    if (start >= end) {
        return;
    }

    let last_idx = last_ids[idx];
    if (last_idx < start) {
        return;
    }

    let final_idx = min(last_idx, end - 1u);
    let pixel_center = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);

    for (var i = start; i <= final_idx; i = i + 1u) {
        let g_idx = flatten_ids[i];
        if (g_idx >= accum_params.active_count) {
            continue;
        }

        let gp = gauss_projections[g_idx];
        let d = pixel_center - gp.mean2d;
        let sigma_over_2 = 0.5 * (gp.conic.x * d.x * d.x + gp.conic.z * d.y * d.y) + gp.conic.y * d.x * d.y;
        if (sigma_over_2 < 0.0) {
            continue;
        }

        if (sigma_over_2 > 6.0) {
            continue;
        }

        atomicAdd(&gaussian_counts[g_idx], 1u);
    }
}
