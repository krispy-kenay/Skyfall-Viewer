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

@group(0) @binding(3) var<storage, read> sh_buffer : array<u32>;
@group(0) @binding(4) var<storage, read> gauss_projections : array<GaussProjection>;

@group(0) @binding(5) var residual_color : texture_storage_2d<rgba32float, read>;
@group(0) @binding(6) var training_alpha : texture_storage_2d<r32float, read>;

@group(0) @binding(7) var<storage, read> last_ids : array<u32>;
@group(0) @binding(8) var<storage, read> tile_offsets : array<u32>;
@group(0) @binding(9) var<storage, read> flatten_ids : array<u32>;

@group(0) @binding(10) var<uniform> active_count : u32;
@group(0) @binding(11) var<uniform> background : BackgroundParams;
@group(0) @binding(12) var<storage, read> tile_masks : array<u32>;
@group(0) @binding(13) var<storage, read> tiles_cumsum : array<u32>;

@group(0) @binding(14) var<storage, read_write> tile_grads : array<f32>;

const MAX_SH_COEFFS : u32 = 16u;
const SH_COMPONENTS : u32 = MAX_SH_COEFFS * 3u;

const MIN_ALPHA_THRESHOLD_RCP : f32 = 255.0;
const MIN_ALPHA_THRESHOLD     : f32 = 1.0 / MIN_ALPHA_THRESHOLD_RCP;
const MAX_FRAGMENT_ALPHA      : f32 = 0.999;

// Tile grad layout constants
const TILE_GRAD_STRIDE     : u32 = 54u;
const TILE_GRAD_OPACITY    : u32 = 0u;
const TILE_GRAD_GEOM_START : u32 = 1u;
const TILE_GRAD_SH_START   : u32 = 6u;

const FIXED_SCALE : f32 = 65536.0;
const FIXED_SCALE_INV : f32 = 1.0 / 65536.0;

const MAX_SPLATS_PER_BATCH : u32 = 64u;
var<workgroup> wg_grads : array<atomic<i32>, 3456>;

var<workgroup> wg_isect_map : array<u32, 64>;
var<workgroup> wg_splat_map : array<u32, 64>;

fn read_f16_coeff(base_word: u32, elem: u32) -> f32 {
    let word_idx = base_word + (elem >> 1u);
    let halves   = unpack2x16float(sh_buffer[word_idx]);
    if ((elem & 1u) == 0u) { return halves.x; } else { return halves.y; }
}

fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let base_word = splat_idx * SH_WORDS_PER_POINT;
    return vec3<f32>(
        read_f16_coeff(base_word, c_idx * 3u + 0u),
        read_f16_coeff(base_word, c_idx * 3u + 1u),
        read_f16_coeff(base_word, c_idx * 3u + 2u)
    );
}

fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);
    if sh_deg > 0u {
        let x = dir.x; let y = dir.y; let z = dir.z;
        result += -SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);
        if sh_deg > 1u {
            let xx = x * x; let yy = y * y; let zz = z * z;
            let xy = x * y; let yz = y * z; let xz = x * z;
            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u)
                    + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u)
                    + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);
            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u)
                        + SH_C3[1] * xy * z * sh_coef(v_idx, 10u)
                        + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u)
                        + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u)
                        + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u)
                        + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u)
                        + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    return max(vec3<f32>(0.0), result + 0.5);
}

fn wg_atomic_add(local_splat_idx: u32, grad_offset: u32, value: f32) {
    let idx = local_splat_idx * TILE_GRAD_STRIDE + grad_offset;
    let fixed_val = i32(value * FIXED_SCALE);
    atomicAdd(&wg_grads[idx], fixed_val);
}

fn wg_accumulate_sh_grads(local_splat_idx: u32, viewDir: vec3<f32>, res_scaled: vec3<f32>, sh_deg: u32) {
    let x = viewDir.x; let y = viewDir.y; let z = viewDir.z;
    let base = TILE_GRAD_SH_START;
    
    let grad_dc = SH_C0 * res_scaled;
    wg_atomic_add(local_splat_idx, base + 0u, grad_dc.x);
    wg_atomic_add(local_splat_idx, base + 1u, grad_dc.y);
    wg_atomic_add(local_splat_idx, base + 2u, grad_dc.z);

    if (sh_deg > 0u) {
        let g1 = -SH_C1 * y * res_scaled;
        wg_atomic_add(local_splat_idx, base + 3u, g1.x);
        wg_atomic_add(local_splat_idx, base + 4u, g1.y);
        wg_atomic_add(local_splat_idx, base + 5u, g1.z);
        let g2 = SH_C1 * z * res_scaled;
        wg_atomic_add(local_splat_idx, base + 6u, g2.x);
        wg_atomic_add(local_splat_idx, base + 7u, g2.y);
        wg_atomic_add(local_splat_idx, base + 8u, g2.z);
        let g3 = -SH_C1 * x * res_scaled;
        wg_atomic_add(local_splat_idx, base + 9u, g3.x);
        wg_atomic_add(local_splat_idx, base + 10u, g3.y);
        wg_atomic_add(local_splat_idx, base + 11u, g3.z);

        if (sh_deg > 1u) {
            let xx = x * x; let yy = y * y; let zz = z * z;
            let xy = x * y; let yz = y * z; let xz = x * z;
            let g4 = SH_C2[0] * xy * res_scaled;
            wg_atomic_add(local_splat_idx, base + 12u, g4.x);
            wg_atomic_add(local_splat_idx, base + 13u, g4.y);
            wg_atomic_add(local_splat_idx, base + 14u, g4.z);
            let g5 = SH_C2[1] * yz * res_scaled;
            wg_atomic_add(local_splat_idx, base + 15u, g5.x);
            wg_atomic_add(local_splat_idx, base + 16u, g5.y);
            wg_atomic_add(local_splat_idx, base + 17u, g5.z);
            let g6 = SH_C2[2] * (2.0 * zz - xx - yy) * res_scaled;
            wg_atomic_add(local_splat_idx, base + 18u, g6.x);
            wg_atomic_add(local_splat_idx, base + 19u, g6.y);
            wg_atomic_add(local_splat_idx, base + 20u, g6.z);
            let g7 = SH_C2[3] * xz * res_scaled;
            wg_atomic_add(local_splat_idx, base + 21u, g7.x);
            wg_atomic_add(local_splat_idx, base + 22u, g7.y);
            wg_atomic_add(local_splat_idx, base + 23u, g7.z);
            let g8 = SH_C2[4] * (xx - yy) * res_scaled;
            wg_atomic_add(local_splat_idx, base + 24u, g8.x);
            wg_atomic_add(local_splat_idx, base + 25u, g8.y);
            wg_atomic_add(local_splat_idx, base + 26u, g8.z);

            if (sh_deg > 2u) {
                let g9 = SH_C3[0] * y * (3.0 * xx - yy) * res_scaled;
                wg_atomic_add(local_splat_idx, base + 27u, g9.x);
                wg_atomic_add(local_splat_idx, base + 28u, g9.y);
                wg_atomic_add(local_splat_idx, base + 29u, g9.z);
                let g10 = SH_C3[1] * xy * z * res_scaled;
                wg_atomic_add(local_splat_idx, base + 30u, g10.x);
                wg_atomic_add(local_splat_idx, base + 31u, g10.y);
                wg_atomic_add(local_splat_idx, base + 32u, g10.z);
                let g11 = SH_C3[2] * y * (4.0 * zz - xx - yy) * res_scaled;
                wg_atomic_add(local_splat_idx, base + 33u, g11.x);
                wg_atomic_add(local_splat_idx, base + 34u, g11.y);
                wg_atomic_add(local_splat_idx, base + 35u, g11.z);
                let g12 = SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * res_scaled;
                wg_atomic_add(local_splat_idx, base + 36u, g12.x);
                wg_atomic_add(local_splat_idx, base + 37u, g12.y);
                wg_atomic_add(local_splat_idx, base + 38u, g12.z);
                let g13 = SH_C3[4] * x * (4.0 * zz - xx - yy) * res_scaled;
                wg_atomic_add(local_splat_idx, base + 39u, g13.x);
                wg_atomic_add(local_splat_idx, base + 40u, g13.y);
                wg_atomic_add(local_splat_idx, base + 41u, g13.z);
                let g14 = SH_C3[5] * z * (xx - yy) * res_scaled;
                wg_atomic_add(local_splat_idx, base + 42u, g14.x);
                wg_atomic_add(local_splat_idx, base + 43u, g14.y);
                wg_atomic_add(local_splat_idx, base + 44u, g14.z);
                let g15 = SH_C3[6] * x * (xx - 3.0 * yy) * res_scaled;
                wg_atomic_add(local_splat_idx, base + 45u, g15.x);
                wg_atomic_add(local_splat_idx, base + 46u, g15.y);
                wg_atomic_add(local_splat_idx, base + 47u, g15.z);
            }
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn backward_tiles_local(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
    @builtin(workgroup_id) wg_id : vec3<u32>
) {
    let local_idx = lid.y * 16u + lid.x;
    
    let tile_x = wg_id.x;
    let tile_y = wg_id.y;
    let tile_width = tiling.tile_width;
    let tile_height = tiling.tile_height;
    
    if (tile_x >= tile_width || tile_y >= tile_height) {
        return;
    }
    
    let tile_id = tile_y * tile_width + tile_x;
    let n_tiles = tile_width * tile_height;
    
    if (tile_masks[tile_id] == 0u) {
        return;
    }
    
    var n_isects: u32 = 0u;
    if (tiles_cumsum[active_count - 1u] > 0u) {
        n_isects = tiles_cumsum[active_count - 1u];
    }
    if (n_isects == 0u) {
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
        return;
    }
    
    let num_splats_in_tile = end - start;
    
    let dims = textureDimensions(residual_color);
    let px_x = tile_x * tiling.tile_size + lid.x;
    let px_y = tile_y * tiling.tile_size + lid.y;

    let pixel_in_bounds = (px_x < dims.x && px_y < dims.y);

    var res = vec3<f32>(0.0, 0.0, 0.0);
    var alpha_final = 0.0;
    var T_current = 1.0;
    var accum_rec = background.color;
    var bin_final: u32 = 0u;
    var valid_pixel = false;

    if (pixel_in_bounds) {
        let px = f32(px_x) + 0.5;
        let py = f32(px_y) + 0.5;
        let pixel = vec2<f32>(px, py);

        let idx1d = px_y * dims.x + px_x;
        bin_final = last_ids[idx1d];

        res = textureLoad(residual_color, vec2<i32>(i32(px_x), i32(px_y))).rgb;
        alpha_final = textureLoad(training_alpha, vec2<i32>(i32(px_x), i32(px_y))).r;
        T_current = 1.0 - alpha_final;
        accum_rec = background.color;

        valid_pixel = (bin_final >= start);
    }

    let num_batches = (num_splats_in_tile + MAX_SPLATS_PER_BATCH - 1u) / MAX_SPLATS_PER_BATCH;
    
    for (var batch = 0u; batch < num_batches; batch = batch + 1u) {
        let batch_start = start + batch * MAX_SPLATS_PER_BATCH;
        let batch_end = min(batch_start + MAX_SPLATS_PER_BATCH, end);
        let batch_size = batch_end - batch_start;
        
        for (var i = local_idx; i < MAX_SPLATS_PER_BATCH * TILE_GRAD_STRIDE; i = i + 256u) {
            atomicStore(&wg_grads[i], 0);
        }
        
        if (local_idx < batch_size) {
            let isect_idx = batch_start + local_idx;
            wg_isect_map[local_idx] = isect_idx;
            wg_splat_map[local_idx] = flatten_ids[isect_idx];
        }
        
        workgroupBarrier();
        
        if (valid_pixel) {
            let pixel_start = max(batch_start, start);
            let pixel_end = min(batch_end, bin_final + 1u);
            
            if (pixel_start < pixel_end) {
                for (var j = 0u; j < pixel_end - pixel_start; j = j + 1u) {
                    let i = pixel_end - 1u - j;
                    
                    if (i < start || i > bin_final) {
                        continue;
                    }
                    
                    let local_splat_idx = i - batch_start;
                    if (local_splat_idx >= batch_size) {
                        continue;
                    }
                    
                    let g_idx = flatten_ids[i];
                    let gp = gauss_projections[g_idx];
                    
                    if (gp.tiles_count == 0u) {
                        continue;
                    }
                    
                    let center_px = gp.mean2d;
                    let aa = gp.conic.x;
                    let bb = gp.conic.y;
                    let cc = gp.conic.z;

                    let px = f32(px_x) + 0.5;
                    let py = f32(px_y) + 0.5;
                    let pixel = vec2<f32>(px, py);
                    
                    let d = pixel - center_px;
                    let sigma_over_2 = 0.5 * (aa * d.x * d.x + cc * d.y * d.y) + bb * d.x * d.y;
                    if (sigma_over_2 < 0.0) {
                        continue;
                    }
                    
                    let gaussian = exp(-sigma_over_2);
                    let position_world = vec4<f32>(gp.position_world, 1.0);
                    let viewDir = normalize(-(camera.view * position_world).xyz);
                    let sh_deg = u32(settings.sh_deg);
                    let opacity = 1.0 / (1.0 + exp(-gp.opacity_logit));
                    
                    let alpha_local = opacity * gaussian;
                    let alpha_splat = min(alpha_local, MAX_FRAGMENT_ALPHA);
                    if (alpha_splat < MIN_ALPHA_THRESHOLD) {
                        continue;
                    }
                    if (alpha_splat > 0.98) {
                        continue;
                    }
                    
                    let one_minus_alpha = max(1.0 - alpha_splat, 0.0001);
                    let T_before = T_current / one_minus_alpha;
                    let contrib_alpha = alpha_splat * T_before;
                    let res_scaled = res * contrib_alpha;
                    
                    wg_accumulate_sh_grads(local_splat_idx, viewDir, res_scaled, sh_deg);
                    
                    let shColor = computeColorFromSH(viewDir, g_idx, sh_deg);
                    let fg_term = dot(res, shColor);
                    let bg_term = dot(res, accum_rec);
                    let dL_dalpha_splat = (fg_term - bg_term) * T_before;
                    let dL_dlogit = dL_dalpha_splat * alpha_splat * (1.0 - opacity);
                    
                    let dL_dG = dL_dalpha_splat * opacity;
                    let dL_dsigma = dL_dG * (-gaussian);
                    let dL_dconic_a = dL_dsigma * 0.5 * d.x * d.x;
                    let dL_dconic_b = dL_dsigma * d.x * d.y;
                    let dL_dconic_c = dL_dsigma * 0.5 * d.y * d.y;
                    let dL_ddx = dL_dsigma * (aa * d.x + bb * d.y);
                    let dL_ddy = dL_dsigma * (bb * d.x + cc * d.y);
                    
                    wg_atomic_add(local_splat_idx, TILE_GRAD_OPACITY, dL_dlogit);
                    wg_atomic_add(local_splat_idx, TILE_GRAD_GEOM_START + 0u, dL_dconic_a);
                    wg_atomic_add(local_splat_idx, TILE_GRAD_GEOM_START + 1u, dL_dconic_b);
                    wg_atomic_add(local_splat_idx, TILE_GRAD_GEOM_START + 2u, dL_dconic_c);
                    wg_atomic_add(local_splat_idx, TILE_GRAD_GEOM_START + 3u, -dL_ddx);
                    wg_atomic_add(local_splat_idx, TILE_GRAD_GEOM_START + 4u, -dL_ddy);
                    
                    accum_rec = alpha_splat * shColor + (1.0 - alpha_splat) * accum_rec;
                    T_current = T_before;
                }
            }
        }
        
        workgroupBarrier();
        
        for (var s = local_idx; s < batch_size; s = s + 256u) {
            let isect_idx = batch_start + s;
            let grad_base = isect_idx * TILE_GRAD_STRIDE;
            let wg_base = s * TILE_GRAD_STRIDE;
            
            for (var g = 0u; g < TILE_GRAD_STRIDE; g = g + 1u) {
                let fixed_val = atomicLoad(&wg_grads[wg_base + g]);
                tile_grads[grad_base + g] = f32(fixed_val) * FIXED_SCALE_INV;
            }
        }
        
        workgroupBarrier();
    }
}
