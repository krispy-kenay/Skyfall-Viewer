struct TilingParams {
    image_width : u32,
    image_height : u32,
    tile_size : u32,
    tile_width : u32,
    tile_height : u32,
    num_cameras : u32,
};

struct BackgroundParams {
    color : vec3<f32>,
    _pad  : f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> settings: RenderSettings;
@group(0) @binding(2) var<uniform> tiling : TilingParams;

// Merged per-gaussian projection data (replaces means2d, conics, depths)
@group(0) @binding(3) var<storage, read> gauss_projections : array<GaussProjection>;
@group(0) @binding(4) var<storage, read> tile_offsets : array<u32>;
@group(0) @binding(5) var<storage, read> flatten_ids : array<u32>;

@group(0) @binding(6) var training_color : texture_storage_2d<rgba32float, write>;
@group(0) @binding(7) var training_alpha : texture_storage_2d<r32float, write>;
@group(0) @binding(8) var<storage, read_write> last_ids : array<u32>;

@group(0) @binding(9) var<storage, read> sh_buffer : array<u32>;
@group(0) @binding(10) var<uniform> background : BackgroundParams;
@group(0) @binding(11) var<storage, read> tile_masks : array<u32>;

@group(0) @binding(12) var<storage, read> tiles_cumsum : array<u32>;
@group(0) @binding(13) var<uniform> active_count : u32;

const MAX_SH_COEFFS_TF : u32 = 16u;
const SH_COMPONENTS_TF : u32 = MAX_SH_COEFFS_TF * 3u;

const MIN_ALPHA_THRESHOLD_RCP : f32 = 255.0;
const MIN_ALPHA_THRESHOLD     : f32 = 1.0 / MIN_ALPHA_THRESHOLD_RCP;
const MAX_FRAGMENT_ALPHA      : f32 = 0.999;
const TRANSMITTANCE_THRESHOLD : f32 = 1e-4;

fn read_f16_coeff_tf(base_word: u32, elem: u32) -> f32 {
    let word_idx = base_word + (elem >> 1u);
    let halves   = unpack2x16float(sh_buffer[word_idx]);
    if ((elem & 1u) == 0u) {
        return halves.x;
    } else {
        return halves.y;
    }
}

fn sh_coef_tf(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let base_word = splat_idx * SH_WORDS_PER_POINT;
    let eR = c_idx * 3u + 0u;
    let eG = c_idx * 3u + 1u;
    let eB = c_idx * 3u + 2u;
    return vec3<f32>(
        read_f16_coeff_tf(base_word, eR),
        read_f16_coeff_tf(base_word, eG),
        read_f16_coeff_tf(base_word, eB)
    );
}

fn computeColorFromSH_tf(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef_tf(v_idx, 0u);

    if sh_deg > 0u {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef_tf(v_idx, 1u)
                  + SH_C1 * z * sh_coef_tf(v_idx, 2u)
                  - SH_C1 * x * sh_coef_tf(v_idx, 3u);

        if sh_deg > 1u {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let yz = y * z;
            let xz = x * z;

            result += SH_C2[0] * xy * sh_coef_tf(v_idx, 4u)
                      + SH_C2[1] * yz * sh_coef_tf(v_idx, 5u)
                      + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef_tf(v_idx, 6u)
                      + SH_C2[3] * xz * sh_coef_tf(v_idx, 7u)
                      + SH_C2[4] * (xx - yy) * sh_coef_tf(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef_tf(v_idx, 9u)
                          + SH_C3[1] * xy * z * sh_coef_tf(v_idx, 10u)
                          + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef_tf(v_idx, 11u)
                          + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef_tf(v_idx, 12u)
                          + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef_tf(v_idx, 13u)
                          + SH_C3[5] * z * (xx - yy) * sh_coef_tf(v_idx, 14u)
                          + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef_tf(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return max(vec3<f32>(0.0), result);
}

@compute @workgroup_size(8, 8, 1)
fn training_tiled_forward(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(training_color);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let pixel = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

    var color = vec3<f32>(0.0, 0.0, 0.0);
    var alpha = 0.0;

    let tile_size_f = f32(tiling.tile_size);
    let tile_x = u32(pixel.x / tile_size_f);
    let tile_y = u32(pixel.y / tile_size_f);
    let tile_width = tiling.tile_width;
    let tile_height = tiling.tile_height;

    if (tile_x >= tile_width || tile_y >= tile_height) {
        return;
    }

    let tile_id = tile_y * tile_width + tile_x;
    let n_tiles = tile_width * tile_height;

    var n_isects: u32 = 0u;
    if (tiles_cumsum[active_count - 1u] > 0u) {
        n_isects = tiles_cumsum[active_count - 1u];
    }

    if (tile_id >= n_tiles) {
        return;
    }

    let tile_index = tile_id;
    if (tile_masks[tile_index] == 0u) {
        let bg = background.color;
        textureStore(training_color, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(bg, 1.0));
        textureStore(training_alpha, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let start = tile_offsets[tile_id];

    var end: u32;
    if (tile_id + 1u < n_tiles + 1u) {
        end = tile_offsets[tile_id + 1u];
    } else {
        end = n_isects;
    }

    if (start >= end || start >= n_isects) {
        let bg = background.color;
        textureStore(training_color, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(bg, 1.0));
        textureStore(training_alpha, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let idx1d = gid.y * dims.x + gid.x;
    var last_idx_in_bin : u32 = start;

    for (var i = start; i < end; i = i + 1u) {
        last_idx_in_bin = i;
        let g_idx = flatten_ids[i];

        let gp = gauss_projections[g_idx];
        let center_px = gp.mean2d;
        let a = gp.conic.x;
        let b = gp.conic.y;
        let c = gp.conic.z;

        let d = pixel - center_px;
        let sigma_over_2 = 0.5 * (a * d.x * d.x + c * d.y * d.y) + b * d.x * d.y;
        if (sigma_over_2 < 0.0) {
            continue;
        }

        let position_world = vec4<f32>(gp.position_world, 1.0);
        let viewDir = normalize(-(camera.view * position_world).xyz);
        let sh_deg = u32(settings.sh_deg);

        let opacity = 1.0 / (1.0 + exp(-gp.opacity_logit));

        let vis_color = computeColorFromSH_tf(viewDir, g_idx, sh_deg);

        let gaussian = exp(-sigma_over_2);
        let alpha_local = opacity * gaussian;
        let alpha_splat = min(alpha_local, MAX_FRAGMENT_ALPHA);
        if (alpha_splat < MIN_ALPHA_THRESHOLD) {
            continue;
        }

        let one_minus_alpha = 1.0 - alpha;
        let contrib_alpha = alpha_splat * one_minus_alpha;
        color += vis_color * contrib_alpha;
        alpha += contrib_alpha;

        if (1.0 - alpha < TRANSMITTANCE_THRESHOLD) {
            break;
        }
    }

    let T = 1.0 - alpha;
    let final_color = color + T * background.color;

    textureStore(training_color, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(final_color, 1.0));
    textureStore(training_alpha, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(alpha, 0.0, 0.0, 0.0));
    last_ids[idx1d] = last_idx_in_bin;
}


