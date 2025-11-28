@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> settings: RenderSettings;

@group(0) @binding(2) var<storage, read> gaussians : array<Gaussian>;
@group(0) @binding(3) var<storage, read> sh_buffer : array<u32>;

@group(0) @binding(4) var residual_color : texture_storage_2d<rgba32float, read>;

@group(0) @binding(6) var<storage, read_write> grad_sh : array<f32>;
@group(0) @binding(7) var<storage, read_write> grad_opacity : array<f32>;

@group(0) @binding(8) var<uniform> active_count : u32;

const MAX_SH_COEFFS : u32 = 16u;
const SH_COMPONENTS : u32 = MAX_SH_COEFFS * 3u;

fn read_f16_coeff(base_word: u32, elem: u32) -> f32 {
    let word_idx = base_word + (elem >> 1u);
    let halves   = unpack2x16float(sh_buffer[word_idx]);
    if ((elem & 1u) == 0u) {
        return halves.x;
    } else {
        return halves.y;
    }
}

fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let base_word = splat_idx * SH_WORDS_PER_POINT;
    let eR = c_idx * 3u + 0u;
    let eG = c_idx * 3u + 1u;
    let eB = c_idx * 3u + 2u;
    return vec3<f32>(
        read_f16_coeff(base_word, eR),
        read_f16_coeff(base_word, eG),
        read_f16_coeff(base_word, eB)
    );
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

@compute @workgroup_size(256)
fn training_backward(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= active_count) {
        return;
    }

    let g = gaussians[idx];
    let p01 = unpack2x16float(g.pos_opacity[0u]);
    let p02 = unpack2x16float(g.pos_opacity[1u]);

    let position_world = vec4<f32>(p01.x, p01.y, p02.x, 1.0);
    let position_camera = camera.view * position_world;
    let position_clip = camera.proj * position_camera;
    let ndc = position_clip.xy / position_clip.w;
    let uv = 0.5 * (ndc + vec2<f32>(1.0, 1.0));

    // Skip if outside screen
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return;
    }

    let dims = textureDimensions(residual_color);
    let px = clamp(i32(uv.x * f32(dims.x)), 0, i32(dims.x - 1));
    let py = clamp(i32(uv.y * f32(dims.y)), 0, i32(dims.y - 1));

    // Residual
    let res = textureLoad(residual_color, vec2<i32>(px, py)).rgb;

    let viewDir = normalize(-(camera.view * position_world).xyz);
    let x = viewDir.x;
    let y = viewDir.y;
    let z = viewDir.z;
    let sh_deg = u32(settings.sh_deg);

    let base_sh = idx * SH_COMPONENTS;

    // ---- DC gradient ----
    let grad_dc = SH_C0 * res;
    grad_sh[base_sh + 0u] += grad_dc.x;
    grad_sh[base_sh + 1u] += grad_dc.y;
    grad_sh[base_sh + 2u] += grad_dc.z;

    // ---- Higher-order SH gradients ----
    if (sh_deg > 0u) {
        let g1 = -SH_C1 * y * res;
        let o1 = base_sh + 1u * 3u;
        grad_sh[o1 + 0u] += g1.x;
        grad_sh[o1 + 1u] += g1.y;
        grad_sh[o1 + 2u] += g1.z;

        let g2 = SH_C1 * z * res;
        let o2 = base_sh + 2u * 3u;
        grad_sh[o2 + 0u] += g2.x;
        grad_sh[o2 + 1u] += g2.y;
        grad_sh[o2 + 2u] += g2.z;

        let g3 = -SH_C1 * x * res;
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

            let g4 = SH_C2[0] * xy * res;
            let o4 = base_sh + 4u * 3u;
            grad_sh[o4 + 0u] += g4.x;
            grad_sh[o4 + 1u] += g4.y;
            grad_sh[o4 + 2u] += g4.z;

            let g5 = SH_C2[1] * yz * res;
            let o5 = base_sh + 5u * 3u;
            grad_sh[o5 + 0u] += g5.x;
            grad_sh[o5 + 1u] += g5.y;
            grad_sh[o5 + 2u] += g5.z;

            let g6 = SH_C2[2] * (2.0 * zz - xx - yy) * res;
            let o6 = base_sh + 6u * 3u;
            grad_sh[o6 + 0u] += g6.x;
            grad_sh[o6 + 1u] += g6.y;
            grad_sh[o6 + 2u] += g6.z;

            let g7 = SH_C2[3] * xz * res;
            let o7 = base_sh + 7u * 3u;
            grad_sh[o7 + 0u] += g7.x;
            grad_sh[o7 + 1u] += g7.y;
            grad_sh[o7 + 2u] += g7.z;

            let g8 = SH_C2[4] * (xx - yy) * res;
            let o8 = base_sh + 8u * 3u;
            grad_sh[o8 + 0u] += g8.x;
            grad_sh[o8 + 1u] += g8.y;
            grad_sh[o8 + 2u] += g8.z;

            if (sh_deg > 2u) {
                let g9 = SH_C3[0] * y * (3.0 * xx - yy) * res;
                let o9 = base_sh + 9u * 3u;
                grad_sh[o9 + 0u] += g9.x;
                grad_sh[o9 + 1u] += g9.y;
                grad_sh[o9 + 2u] += g9.z;

                let g10 = SH_C3[1] * xy * z * res;
                let o10 = base_sh + 10u * 3u;
                grad_sh[o10 + 0u] += g10.x;
                grad_sh[o10 + 1u] += g10.y;
                grad_sh[o10 + 2u] += g10.z;

                let g11 = SH_C3[2] * y * (4.0 * zz - xx - yy) * res;
                let o11 = base_sh + 11u * 3u;
                grad_sh[o11 + 0u] += g11.x;
                grad_sh[o11 + 1u] += g11.y;
                grad_sh[o11 + 2u] += g11.z;

                let g12 = SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * res;
                let o12 = base_sh + 12u * 3u;
                grad_sh[o12 + 0u] += g12.x;
                grad_sh[o12 + 1u] += g12.y;
                grad_sh[o12 + 2u] += g12.z;

                let g13 = SH_C3[4] * x * (4.0 * zz - xx - yy) * res;
                let o13 = base_sh + 13u * 3u;
                grad_sh[o13 + 0u] += g13.x;
                grad_sh[o13 + 1u] += g13.y;
                grad_sh[o13 + 2u] += g13.z;

                let g14 = SH_C3[5] * z * (xx - yy) * res;
                let o14 = base_sh + 14u * 3u;
                grad_sh[o14 + 0u] += g14.x;
                grad_sh[o14 + 1u] += g14.y;
                grad_sh[o14 + 2u] += g14.z;

                let g15 = SH_C3[6] * x * (xx - 3.0 * yy) * res;
                let o15 = base_sh + 15u * 3u;
                grad_sh[o15 + 0u] += g15.x;
                grad_sh[o15 + 1u] += g15.y;
                grad_sh[o15 + 2u] += g15.z;
            }
        }
    }

    // ---- Approximate opacity gradient ----
    let shColor = computeColorFromSH(viewDir, idx, sh_deg);
    let grad_alpha = dot(res, shColor);
    grad_opacity[idx] += grad_alpha;
}


