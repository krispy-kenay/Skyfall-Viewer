

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> settings: RenderSettings;
@group(1) @binding(0) var<storage, read> gaussians : array<Gaussian>;
@group(1) @binding(1) var<storage, read> sh_buffer : array<u32>;
@group(1) @binding(2) var<storage, read_write> splats : array<Splat>;

struct DebugProj {
    pos_cam : vec4<f32>,
    clip    : vec4<f32>,
};
@group(1) @binding(3) var<storage, read_write> debug_proj : array<DebugProj>;

//TODO: bind your data here
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;


fn read_f16_coeff(base_word: u32, elem: u32) -> f32 {
    let word_idx = base_word + (elem >> 1u);
    let halves   = unpack2x16float(sh_buffer[word_idx]);
    if ((elem & 1u) == 0u) {
        return halves.x;
    } else {
        return halves.y;
    }
}

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
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

fn float_to_ordered_uint(x: f32) -> u32 {
  let bits = bitcast<u32>(x);
  let mask = select(0x80000000u, 0xFFFFFFFFu, (bits & 0x80000000u) != 0u);
  return bits ^ mask;
}

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let rx = q[0]; let ry = q[1]; let rz = q[2]; let rw = q[3];
    return mat3x3<f32>(
        1.0 - 2.0*(ry * ry + rz * rz),  2.0*(rx * ry - rw * rz),        2.0*(rx * rz + rw * ry),
        2.0*(rx * ry + rw * rz),        1.0 - 2.0*(rx * rx + rz * rz),  2.0*(ry * rz - rw * rx),
        2.0*(rx * rz - rw * ry),        2.0*(ry * rz + rw * rx),        1.0 - 2.0*(rx * rx + ry * ry)
    );
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction
    let num = arrayLength(&gaussians);
    if (idx >= num) { return; }

    // Unpacking
    let g = gaussians[idx];
    let p01 = unpack2x16float(g.pos_opacity[0u]);
    let p02 = unpack2x16float(g.pos_opacity[1u]);
    let r01 = unpack2x16float(g.rot[0u]);
    let r02 = unpack2x16float(g.rot[1u]);
    let s01 = unpack2x16float(g.scale[0u]);
    let s02 = unpack2x16float(g.scale[1u]);

    // Position
    let position_world = vec4<f32>(p01.x, p01.y, p02.x, 1.0);
    let position_camera = camera.view * position_world;
    let viewZ = -position_camera.z;
    var position_clip = camera.proj * position_camera;
    position_clip /= position_clip.w;

    // Debug: record camera-space and clip-space positions for the first few gaussians
    let max_debug = 32u;
    if (idx < max_debug) {
        debug_proj[idx].pos_cam = position_camera;
        debug_proj[idx].clip = position_clip;
    }

    if (abs(position_clip.x) > 1.2 || abs(position_clip.y) > 1.2 || position_camera.z < 0.0) {
        return;
    }
    let opacity = 1.0 / (1.0 + exp(-p02.y));

    // Scale
    let log_sigma = vec3<f32>(clamp(s01.x, -10.0, 10.0), clamp(s01.y, -10.0, 10.0), clamp(s02.x , -10.0, 10.0));
    let sigma = exp(log_sigma);
    
    let S = mat3x3<f32>(
        settings.gaussian_scaling * sigma.x, 0.0, 0.0,
        0.0, settings.gaussian_scaling * sigma.y, 0.0,
        0.0, 0.0, settings.gaussian_scaling * sigma.z
    );

    // Rotation
    let q = normalize(vec4<f32>(r01.y, r02.x, r02.y, r01.x));
    let R = quat_to_mat3(q);    
    
    // Covariance Projection
    let C3D = transpose(S * R) * S * R;

    let t = position_camera.xyz;
    let fx = camera.focal.x;
    let fy = camera.focal.y;
    let J = mat3x3<f32>(
        fx / t.z, 0.0, - (fx * t.x) / (t.z * t.z),
        0.0, fy / t.z, - (fy * t.y) / (t.z * t.z),
        0.0, 0.0, 0.0
    );

    let Rwc = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let T = J * Rwc;

    var C2D = T * C3D * transpose(T);
    C2D[0][0] += 0.3;
    C2D[1][1] += 0.3;

    let det = C2D[0][0] * C2D[1][1] - C2D[0][1] * C2D[0][1];
    if (det <= 0.0) { return; }
    
    let det_inv = 1.0 / det;
    let aa = C2D[1][1] * det_inv;
    let bb = -C2D[0][1] * det_inv;
    let cc = C2D[0][0] * det_inv;

    let viewDir = normalize(-(camera.view * position_world).xyz);
    var color = computeColorFromSH(viewDir, idx, u32(settings.sh_deg));

    // Eigenvalues
    let mid = 0.5 * (C2D[0][0] + C2D[1][1]);
    let disc = sqrt(max(mid * mid - det, 0.0));
    var lambda1 = mid + disc;
    var lambda2 = mid - disc;
    var radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    // Sorting prep
    let visIdx = atomicAdd(&sort_infos.keys_size, 1u);
    sort_depths[visIdx] = float_to_ordered_uint(-position_camera.z);
    sort_indices[visIdx] = visIdx;

    // Write to splats
    splats[visIdx].center_ndc = position_clip.xy;
    splats[visIdx].depth = position_clip.z + 1e-5;
    splats[visIdx].radius = vec2<f32>(radius, radius);
    splats[visIdx].color = color;
    splats[visIdx].a11 = aa;
    splats[visIdx].a12 = bb;
    splats[visIdx].a22 = cc;
    splats[visIdx].opacity = opacity;
    

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if ((visIdx % keys_per_dispatch) == 0u) {
        _ = atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}