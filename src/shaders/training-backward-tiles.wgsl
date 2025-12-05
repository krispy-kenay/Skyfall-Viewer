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

@group(0) @binding(10) var<storage, read_write> grad_sh : array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> grad_opacity : array<atomic<u32>>;

@group(0) @binding(12) var<uniform> active_count : u32;

@group(0) @binding(13) var<storage, read_write> grad_geom : array<atomic<u32>>;

@group(0) @binding(14) var<uniform> background : BackgroundParams;
@group(0) @binding(15) var<storage, read> tile_masks : array<u32>;

@group(0) @binding(16) var<storage, read> tiles_cumsum : array<u32>;

const MAX_SH_COEFFS_TB : u32 = 16u;
const SH_COMPONENTS_TB : u32 = MAX_SH_COEFFS_TB * 3u;

const MIN_ALPHA_THRESHOLD_RCP_TB : f32 = 255.0;
const MIN_ALPHA_THRESHOLD_TB     : f32 = 1.0 / MIN_ALPHA_THRESHOLD_RCP_TB;
const MAX_FRAGMENT_ALPHA_TB      : f32 = 0.999;
const TRANSMITTANCE_THRESHOLD_TB : f32 = 1e-4;


fn atomicAdd_f32(ptr: ptr<storage, atomic<u32>, read_write>, delta: f32) {
    loop {
        let oldBits : u32 = atomicLoad(ptr);
        let oldVal  : f32 = bitcast<f32>(oldBits);
        let newVal  : f32 = oldVal + delta;
        let newBits : u32 = bitcast<u32>(newVal);
        let res = atomicCompareExchangeWeak(ptr, oldBits, newBits);
        if (res.exchanged) {
            break;
        }
    }
}

const GEOM_GRAD_CONIC_A  : u32 = 0u;
const GEOM_GRAD_CONIC_B  : u32 = 1u;
const GEOM_GRAD_CONIC_C  : u32 = 2u;
const GEOM_GRAD_MEAN2D_X : u32 = 3u;
const GEOM_GRAD_MEAN2D_Y : u32 = 4u;
const GEOM_GRAD_STRIDE   : u32 = 5u;

fn read_f16_coeff_tb(base_word: u32, elem: u32) -> f32 {
    let word_idx = base_word + (elem >> 1u);
    let halves   = unpack2x16float(sh_buffer[word_idx]);
    if ((elem & 1u) == 0u) {
        return halves.x;
    } else {
        return halves.y;
    }
}

fn sh_coef_tb(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let base_word = splat_idx * SH_WORDS_PER_POINT;
    let eR = c_idx * 3u + 0u;
    let eG = c_idx * 3u + 1u;
    let eB = c_idx * 3u + 2u;
    return vec3<f32>(
        read_f16_coeff_tb(base_word, eR),
        read_f16_coeff_tb(base_word, eG),
        read_f16_coeff_tb(base_word, eB)
    );
}

fn computeColorFromSH_tb(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef_tb(v_idx, 0u);

    if sh_deg > 0u {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef_tb(v_idx, 1u)
                  + SH_C1 * z * sh_coef_tb(v_idx, 2u)
                  - SH_C1 * x * sh_coef_tb(v_idx, 3u);

        if sh_deg > 1u {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let yz = y * z;
            let xz = x * z;

            result += SH_C2[0] * xy * sh_coef_tb(v_idx, 4u)
                      + SH_C2[1] * yz * sh_coef_tb(v_idx, 5u)
                      + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef_tb(v_idx, 6u)
                      + SH_C2[3] * xz * sh_coef_tb(v_idx, 7u)
                      + SH_C2[4] * (xx - yy) * sh_coef_tb(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef_tb(v_idx, 9u)
                          + SH_C3[1] * xy * z * sh_coef_tb(v_idx, 10u)
                          + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef_tb(v_idx, 11u)
                          + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef_tb(v_idx, 12u)
                          + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef_tb(v_idx, 13u)
                          + SH_C3[5] * z * (xx - yy) * sh_coef_tb(v_idx, 14u)
                          + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef_tb(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return max(vec3<f32>(0.0), result);
}

fn accumulate_sh_grads_tb(idx: u32, viewDir: vec3<f32>, res_scaled: vec3<f32>, sh_deg: u32) {
    let x = viewDir.x;
    let y = viewDir.y;
    let z = viewDir.z;
    let base_sh = idx * SH_COMPONENTS_TB;

    // DC
    let grad_dc = SH_C0 * res_scaled;
    atomicAdd_f32(&grad_sh[base_sh + 0u], grad_dc.x);
    atomicAdd_f32(&grad_sh[base_sh + 1u], grad_dc.y);
    atomicAdd_f32(&grad_sh[base_sh + 2u], grad_dc.z);

    if (sh_deg > 0u) {
        let g1 = -SH_C1 * y * res_scaled;
        let o1 = base_sh + 1u * 3u;
        atomicAdd_f32(&grad_sh[o1 + 0u], g1.x);
        atomicAdd_f32(&grad_sh[o1 + 1u], g1.y);
        atomicAdd_f32(&grad_sh[o1 + 2u], g1.z);

        let g2 = SH_C1 * z * res_scaled;
        let o2 = base_sh + 2u * 3u;
        atomicAdd_f32(&grad_sh[o2 + 0u], g2.x);
        atomicAdd_f32(&grad_sh[o2 + 1u], g2.y);
        atomicAdd_f32(&grad_sh[o2 + 2u], g2.z);

        let g3 = -SH_C1 * x * res_scaled;
        let o3 = base_sh + 3u * 3u;
        atomicAdd_f32(&grad_sh[o3 + 0u], g3.x);
        atomicAdd_f32(&grad_sh[o3 + 1u], g3.y);
        atomicAdd_f32(&grad_sh[o3 + 2u], g3.z);

        if (sh_deg > 1u) {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let yz = y * z;
            let xz = x * z;

            let g4 = SH_C2[0] * xy * res_scaled;
            let o4 = base_sh + 4u * 3u;
            atomicAdd_f32(&grad_sh[o4 + 0u], g4.x);
            atomicAdd_f32(&grad_sh[o4 + 1u], g4.y);
            atomicAdd_f32(&grad_sh[o4 + 2u], g4.z);

            let g5 = SH_C2[1] * yz * res_scaled;
            let o5 = base_sh + 5u * 3u;
            atomicAdd_f32(&grad_sh[o5 + 0u], g5.x);
            atomicAdd_f32(&grad_sh[o5 + 1u], g5.y);
            atomicAdd_f32(&grad_sh[o5 + 2u], g5.z);

            let g6 = SH_C2[2] * (2.0 * zz - xx - yy) * res_scaled;
            let o6 = base_sh + 6u * 3u;
            atomicAdd_f32(&grad_sh[o6 + 0u], g6.x);
            atomicAdd_f32(&grad_sh[o6 + 1u], g6.y);
            atomicAdd_f32(&grad_sh[o6 + 2u], g6.z);

            let g7 = SH_C2[3] * xz * res_scaled;
            let o7 = base_sh + 7u * 3u;
            atomicAdd_f32(&grad_sh[o7 + 0u], g7.x);
            atomicAdd_f32(&grad_sh[o7 + 1u], g7.y);
            atomicAdd_f32(&grad_sh[o7 + 2u], g7.z);

            let g8 = SH_C2[4] * (xx - yy) * res_scaled;
            let o8 = base_sh + 8u * 3u;
            atomicAdd_f32(&grad_sh[o8 + 0u], g8.x);
            atomicAdd_f32(&grad_sh[o8 + 1u], g8.y);
            atomicAdd_f32(&grad_sh[o8 + 2u], g8.z);

            if (sh_deg > 2u) {
                let g9 = SH_C3[0] * y * (3.0 * xx - yy) * res_scaled;
                let o9 = base_sh + 9u * 3u;
                atomicAdd_f32(&grad_sh[o9 + 0u], g9.x);
                atomicAdd_f32(&grad_sh[o9 + 1u], g9.y);
                atomicAdd_f32(&grad_sh[o9 + 2u], g9.z);

                let g10 = SH_C3[1] * xy * z * res_scaled;
                let o10 = base_sh + 10u * 3u;
                atomicAdd_f32(&grad_sh[o10 + 0u], g10.x);
                atomicAdd_f32(&grad_sh[o10 + 1u], g10.y);
                atomicAdd_f32(&grad_sh[o10 + 2u], g10.z);

                let g11 = SH_C3[2] * y * (4.0 * zz - xx - yy) * res_scaled;
                let o11 = base_sh + 11u * 3u;
                atomicAdd_f32(&grad_sh[o11 + 0u], g11.x);
                atomicAdd_f32(&grad_sh[o11 + 1u], g11.y);
                atomicAdd_f32(&grad_sh[o11 + 2u], g11.z);

                let g12 = SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * res_scaled;
                let o12 = base_sh + 12u * 3u;
                atomicAdd_f32(&grad_sh[o12 + 0u], g12.x);
                atomicAdd_f32(&grad_sh[o12 + 1u], g12.y);
                atomicAdd_f32(&grad_sh[o12 + 2u], g12.z);

                let g13 = SH_C3[4] * x * (4.0 * zz - xx - yy) * res_scaled;
                let o13 = base_sh + 13u * 3u;
                atomicAdd_f32(&grad_sh[o13 + 0u], g13.x);
                atomicAdd_f32(&grad_sh[o13 + 1u], g13.y);
                atomicAdd_f32(&grad_sh[o13 + 2u], g13.z);

                let g14 = SH_C3[5] * z * (xx - yy) * res_scaled;
                let o14 = base_sh + 14u * 3u;
                atomicAdd_f32(&grad_sh[o14 + 0u], g14.x);
                atomicAdd_f32(&grad_sh[o14 + 1u], g14.y);
                atomicAdd_f32(&grad_sh[o14 + 2u], g14.z);

                let g15 = SH_C3[6] * x * (xx - 3.0 * yy) * res_scaled;
                let o15 = base_sh + 15u * 3u;
                atomicAdd_f32(&grad_sh[o15 + 0u], g15.x);
                atomicAdd_f32(&grad_sh[o15 + 1u], g15.y);
                atomicAdd_f32(&grad_sh[o15 + 2u], g15.z);
            }
        }
    }
}

@compute @workgroup_size(8, 8, 1)
fn training_tiled_backward(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(residual_color);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let tile_size_f = f32(tiling.tile_size);
    let px = f32(gid.x) + 0.5;
    let py = f32(gid.y) + 0.5;

    let tile_x = u32(px / tile_size_f);
    let tile_y = u32(py / tile_size_f);
    let tile_width = tiling.tile_width;
    let tile_height = tiling.tile_height;

    if (tile_x >= tile_width || tile_y >= tile_height) {
        return;
    }

    let tile_id = tile_y * tile_width + tile_x;
    let n_tiles = tile_width * tile_height;
    if (tile_id >= n_tiles) {
        return;
    }

    let tile_index = tile_id;
    if (tile_masks[tile_index] == 0u) {
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

    let idx1d = gid.y * dims.x + gid.x;
    let bin_final = last_ids[idx1d];

    let res = textureLoad(residual_color, vec2<i32>(i32(gid.x), i32(gid.y))).rgb;
    
    let alpha_final = textureLoad(training_alpha, vec2<i32>(i32(gid.x), i32(gid.y))).r;
    var T_current = 1.0 - alpha_final;
    
    var accum_rec = background.color;
    
    if (bin_final < start) {
        return;
    }
    
    let num_splats = bin_final - start + 1u;
    
    for (var j = 0u; j < num_splats; j = j + 1u) {
        let i = bin_final - j;
        
        if (i < start || i >= end) {
            continue;
        }
        
        let g_idx = flatten_ids[i];

        let gp = gauss_projections[g_idx];
        let center_px = gp.mean2d;
        let aa = gp.conic.x;
        let bb = gp.conic.y;
        let cc = gp.conic.z;

        if (gp.tiles_count == 0u) {
            continue;
        }

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
        let alpha_splat = min(alpha_local, MAX_FRAGMENT_ALPHA_TB);
        if (alpha_splat < MIN_ALPHA_THRESHOLD_TB) {
            continue;
        }

        let one_minus_alpha = max(1.0 - alpha_splat, 0.0001);
        let T_before = T_current / one_minus_alpha;
        
        let contrib_alpha = alpha_splat * T_before;
        let res_scaled = res * contrib_alpha;
        accumulate_sh_grads_tb(g_idx, viewDir, res_scaled, sh_deg);

        let shColor = computeColorFromSH_tb(viewDir, g_idx, sh_deg);
        let fg_term = dot(res, shColor);
        let bg_term = dot(res, accum_rec);
        let dL_dalpha_splat = (fg_term - bg_term) * T_before;

        let dL_dlogit = dL_dalpha_splat * alpha_splat * (1.0 - opacity);
        atomicAdd_f32(&grad_opacity[g_idx], dL_dlogit);

        let dL_dG = dL_dalpha_splat * opacity;
        let dL_dsigma = dL_dG * (-gaussian);
        
        let dL_dconic_a = dL_dsigma * 0.5 * d.x * d.x;
        let dL_dconic_b = dL_dsigma * d.x * d.y;
        let dL_dconic_c = dL_dsigma * 0.5 * d.y * d.y;
        
        let dL_ddx = dL_dsigma * (aa * d.x + bb * d.y);
        let dL_ddy = dL_dsigma * (bb * d.x + cc * d.y);
        let dL_dmean2d_x = -dL_ddx;
        let dL_dmean2d_y = -dL_ddy;
        
        let geom_base = g_idx * GEOM_GRAD_STRIDE;
        atomicAdd_f32(&grad_geom[geom_base + GEOM_GRAD_CONIC_A], dL_dconic_a);
        atomicAdd_f32(&grad_geom[geom_base + GEOM_GRAD_CONIC_B], dL_dconic_b);
        atomicAdd_f32(&grad_geom[geom_base + GEOM_GRAD_CONIC_C], dL_dconic_c);
        atomicAdd_f32(&grad_geom[geom_base + GEOM_GRAD_MEAN2D_X], dL_dmean2d_x);
        atomicAdd_f32(&grad_geom[geom_base + GEOM_GRAD_MEAN2D_Y], dL_dmean2d_y);

        accum_rec = alpha_splat * shColor + (1.0 - alpha_splat) * accum_rec;
        
        T_current = T_before;
    }
}


