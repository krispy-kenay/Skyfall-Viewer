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

@group(0) @binding(3) var<storage, read> gaussians : array<Gaussian>;
@group(0) @binding(4) var<storage, read> sh_buffer : array<u32>;

@group(0) @binding(5) var residual_color : texture_storage_2d<rgba32float, read>;

@group(0) @binding(6) var<storage, read> last_ids : array<u32>;
@group(0) @binding(7) var<storage, read> tile_offsets : array<u32>;
@group(0) @binding(8) var<storage, read> flatten_ids : array<u32>;

@group(0) @binding(9) var<storage, read_write> grad_sh : array<f32>;
@group(0) @binding(10) var<storage, read_write> grad_opacity : array<f32>;

@group(0) @binding(11) var<uniform> active_count : u32;

@group(0) @binding(12) var<storage, read_write> grad_alpha_splat : array<atomic<i32>>;

@group(0) @binding(13) var<uniform> background : BackgroundParams;
@group(0) @binding(14) var<storage, read> tile_masks : array<u32>;

@group(0) @binding(15) var<storage, read> tiles_cumsum : array<u32>;

const MAX_SH_COEFFS_TB : u32 = 16u;
const SH_COMPONENTS_TB : u32 = MAX_SH_COEFFS_TB * 3u;

const MIN_ALPHA_THRESHOLD_RCP_TB : f32 = 255.0;
const MIN_ALPHA_THRESHOLD_TB     : f32 = 1.0 / MIN_ALPHA_THRESHOLD_RCP_TB;
const MAX_FRAGMENT_ALPHA_TB      : f32 = 0.999;
const TRANSMITTANCE_THRESHOLD_TB : f32 = 1e-4;

const GRAD_ALPHA_SCALE_TB : f32 = 1e3;

fn alphaGradToFixed(v: f32) -> i32 {
    return i32(round(v * GRAD_ALPHA_SCALE_TB));
}

fn alphaGradToFloat(v: i32) -> f32 {
    return f32(v) / GRAD_ALPHA_SCALE_TB;
}

fn quat_to_mat3_tiling_bw(q: vec4<f32>) -> mat3x3<f32> {
    let rx = q[0]; let ry = q[1]; let rz = q[2]; let rw = q[3];
    return mat3x3<f32>(
        1.0 - 2.0*(ry * ry + rz * rz),  2.0*(rx * ry - rw * rz),        2.0*(rx * rz + rw * ry),
        2.0*(rx * ry + rw * rz),        1.0 - 2.0*(rx * rx + rz * rz),  2.0*(ry * rz - rw * rx),
        2.0*(rx * rz - rw * ry),        2.0*(ry * rz + rw * rx),        1.0 - 2.0*(rx * rx + ry * ry)
    );
}

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
    grad_sh[base_sh + 0u] += grad_dc.x;
    grad_sh[base_sh + 1u] += grad_dc.y;
    grad_sh[base_sh + 2u] += grad_dc.z;

    if (sh_deg > 0u) {
        let g1 = -SH_C1 * y * res_scaled;
        let o1 = base_sh + 1u * 3u;
        grad_sh[o1 + 0u] += g1.x;
        grad_sh[o1 + 1u] += g1.y;
        grad_sh[o1 + 2u] += g1.z;

        let g2 = SH_C1 * z * res_scaled;
        let o2 = base_sh + 2u * 3u;
        grad_sh[o2 + 0u] += g2.x;
        grad_sh[o2 + 1u] += g2.y;
        grad_sh[o2 + 2u] += g2.z;

        let g3 = -SH_C1 * x * res_scaled;
        let o3 = base_sh + 3u * 3u;
        grad_sh[o3 + 0u] += g3.x;
        grad_sh[o3 + 1u] += g3.y;
        grad_sh[o3 + 2u] += g3.z;

        if (sh_deg > 1u) {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let yz = y * z;
            let xz = x * z;

            let g4 = SH_C2[0] * xy * res_scaled;
            let o4 = base_sh + 4u * 3u;
            grad_sh[o4 + 0u] += g4.x;
            grad_sh[o4 + 1u] += g4.y;
            grad_sh[o4 + 2u] += g4.z;

            let g5 = SH_C2[1] * yz * res_scaled;
            let o5 = base_sh + 5u * 3u;
            grad_sh[o5 + 0u] += g5.x;
            grad_sh[o5 + 1u] += g5.y;
            grad_sh[o5 + 2u] += g5.z;

            let g6 = SH_C2[2] * (2.0 * zz - xx - yy) * res_scaled;
            let o6 = base_sh + 6u * 3u;
            grad_sh[o6 + 0u] += g6.x;
            grad_sh[o6 + 1u] += g6.y;
            grad_sh[o6 + 2u] += g6.z;

            let g7 = SH_C2[3] * xz * res_scaled;
            let o7 = base_sh + 7u * 3u;
            grad_sh[o7 + 0u] += g7.x;
            grad_sh[o7 + 1u] += g7.y;
            grad_sh[o7 + 2u] += g7.z;

            let g8 = SH_C2[4] * (xx - yy) * res_scaled;
            let o8 = base_sh + 8u * 3u;
            grad_sh[o8 + 0u] += g8.x;
            grad_sh[o8 + 1u] += g8.y;
            grad_sh[o8 + 2u] += g8.z;

            if (sh_deg > 2u) {
                let g9 = SH_C3[0] * y * (3.0 * xx - yy) * res_scaled;
                let o9 = base_sh + 9u * 3u;
                grad_sh[o9 + 0u] += g9.x;
                grad_sh[o9 + 1u] += g9.y;
                grad_sh[o9 + 2u] += g9.z;

                let g10 = SH_C3[1] * xy * z * res_scaled;
                let o10 = base_sh + 10u * 3u;
                grad_sh[o10 + 0u] += g10.x;
                grad_sh[o10 + 1u] += g10.y;
                grad_sh[o10 + 2u] += g10.z;

                let g11 = SH_C3[2] * y * (4.0 * zz - xx - yy) * res_scaled;
                let o11 = base_sh + 11u * 3u;
                grad_sh[o11 + 0u] += g11.x;
                grad_sh[o11 + 1u] += g11.y;
                grad_sh[o11 + 2u] += g11.z;

                let g12 = SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * res_scaled;
                let o12 = base_sh + 12u * 3u;
                grad_sh[o12 + 0u] += g12.x;
                grad_sh[o12 + 1u] += g12.y;
                grad_sh[o12 + 2u] += g12.z;

                let g13 = SH_C3[4] * x * (4.0 * zz - xx - yy) * res_scaled;
                let o13 = base_sh + 13u * 3u;
                grad_sh[o13 + 0u] += g13.x;
                grad_sh[o13 + 1u] += g13.y;
                grad_sh[o13 + 2u] += g13.z;

                let g14 = SH_C3[5] * z * (xx - yy) * res_scaled;
                let o14 = base_sh + 14u * 3u;
                grad_sh[o14 + 0u] += g14.x;
                grad_sh[o14 + 1u] += g14.y;
                grad_sh[o14 + 2u] += g14.z;

                let g15 = SH_C3[6] * x * (xx - 3.0 * yy) * res_scaled;
                let o15 = base_sh + 15u * 3u;
                grad_sh[o15 + 0u] += g15.x;
                grad_sh[o15 + 1u] += g15.y;
                grad_sh[o15 + 2u] += g15.z;
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

    var alpha_acc = 0.0;

    for (var i = start; i <= bin_final && i < end; i = i + 1u) {
        let g_idx = flatten_ids[i];

        let gg = gaussians[g_idx];
        let p01 = unpack2x16float(gg.pos_opacity[0u]);
        let p02 = unpack2x16float(gg.pos_opacity[1u]);
        let r01 = unpack2x16float(gg.rot[0u]);
        let r02 = unpack2x16float(gg.rot[1u]);
        let s01 = unpack2x16float(gg.scale[0u]);
        let s02 = unpack2x16float(gg.scale[1u]);

        let position_world = vec4<f32>(p01.x, p01.y, p02.x, 1.0);
        let position_camera = camera.view * position_world;
        var position_clip = camera.proj * position_camera;
        position_clip /= position_clip.w;

        let log_sigma = vec3<f32>(
            clamp(s01.x, -10.0, 10.0),
            clamp(s01.y, -10.0, 10.0),
            clamp(s02.x, -10.0, 10.0)
        );
        let sigma = exp(log_sigma);

        let S = mat3x3<f32>(
            sigma.x, 0.0,    0.0,
            0.0,    sigma.y, 0.0,
            0.0,    0.0,    sigma.z
        );

        let q = normalize(vec4<f32>(r01.y, r02.x, r02.y, r01.x));
        let R = quat_to_mat3_tiling_bw(q);

        let C3D = transpose(S * R) * S * R;

        let t_cam = position_camera.xyz;
        let fx = camera.focal.x;
        let fy = camera.focal.y;
        let J = mat3x3<f32>(
            fx / t_cam.z, 0.0, - (fx * t_cam.x) / (t_cam.z * t_cam.z),
            0.0, fy / t_cam.z, - (fy * t_cam.y) / (t_cam.z * t_cam.z),
            0.0, 0.0, 0.0
        );

        let Rwc = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
        let T = J * Rwc;

        var C2D = T * C3D * transpose(T);

        C2D[0][0] += 0.3;
        C2D[1][1] += 0.3;

        let det = C2D[0][0] * C2D[1][1] - C2D[0][1] * C2D[0][1];
        if (det <= 0.0) {
            continue;
        }

        let det_inv = 1.0 / det;
        let aa = C2D[1][1] * det_inv;
        let bb = -C2D[0][1] * det_inv;
        let cc = C2D[0][0] * det_inv;

        let dims = vec2<f32>(f32(tiling.image_width), f32(tiling.image_height));
        let ndc = position_clip.xy;
        let center_px = 0.5 * (ndc + vec2<f32>(1.0, 1.0)) * dims;

        let pixel = vec2<f32>(px, py);
        let d = pixel - center_px;

        let sigma_over_2 = 0.5 * (aa * d.x * d.x + cc * d.y * d.y) + bb * d.x * d.y;
        if (sigma_over_2 < 0.0) {
            continue;
        }

        let gaussian = exp(-sigma_over_2);

        let viewDir = normalize(-(camera.view * position_world).xyz);
        let sh_deg = u32(settings.sh_deg);

        let base_opacity_logit = p02.y;
        let opacity = 1.0 / (1.0 + exp(-base_opacity_logit));

        let alpha_local = opacity * gaussian;
        let alpha_splat = min(alpha_local, MAX_FRAGMENT_ALPHA_TB);
        if (alpha_splat < MIN_ALPHA_THRESHOLD_TB) {
            continue;
        }

        let one_minus_alpha = 1.0 - alpha_acc;

        let contrib_alpha = alpha_splat * one_minus_alpha;

        let res_scaled = res * contrib_alpha;
        accumulate_sh_grads_tb(g_idx, viewDir, res_scaled, sh_deg);

        let shColor = computeColorFromSH_tb(viewDir, g_idx, sh_deg);
        let fg_term = dot(res, shColor);
        let bg_term = dot(res, background.color);
        let dL_dalpha_splat = (fg_term - bg_term) * one_minus_alpha;

        let dL_dlogit = dL_dalpha_splat * alpha_splat * (1.0 - opacity);
        grad_opacity[g_idx] += dL_dlogit;

        atomicAdd(&grad_alpha_splat[g_idx], alphaGradToFixed(dL_dalpha_splat * gaussian));

        alpha_acc += contrib_alpha;
        if (1.0 - alpha_acc < TRANSMITTANCE_THRESHOLD_TB) {
            break;
        }
    }
}


