@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(0) @binding(1) var<storage, read> gaussians : array<Gaussian>;
@group(0) @binding(2) var<uniform> active_count : u32;

@group(0) @binding(3) var<storage, read_write> grad_alpha_splat : array<atomic<i32>>;

@group(0) @binding(4) var<storage, read_write> grad_position : array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_scale : array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_rotation : array<f32>;

const TRAIN_POS_COMPONENTS : u32 = 3u;
const TRAIN_SCALE_COMPONENTS : u32 = 3u;
const TRAIN_ROT_COMPONENTS : u32 = 4u;

const GRAD_ALPHA_SCALE_GEOM : f32 = 1e3;

fn alphaGradToFloatGeom(v: i32) -> f32 {
    return f32(v) / GRAD_ALPHA_SCALE_GEOM;
}

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let l = dot(v, v);
    if (l > 0.0) {
        let inv = inverseSqrt(l);
        return v * inv;
    }
    return v;
}

fn safe_normalize_bw(v: vec3<f32>, d_out: vec3<f32>) -> vec3<f32> {
    let l = dot(v, v);
    if (l > 0.0) {
        let il = inverseSqrt(l);
        let il3 = il * il * il;
        return il * d_out - il3 * dot(d_out, v) * v;
    }
    return d_out;
}

fn quat_to_rotmat_wxyz(quat: vec4<f32>) -> mat3x3<f32> {
    var w = quat.x;
    var x = quat.y;
    var y = quat.z;
    var z = quat.w;

    let inv_norm = inverseSqrt(x * x + y * y + z * z + w * w);
    x = x * inv_norm;
    y = y * inv_norm;
    z = z * inv_norm;
    w = w * inv_norm;

    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    return mat3x3<f32>(
        (1.0 - 2.0 * (y2 + z2)),
        (2.0 * (xy + wz)),
        (2.0 * (xz - wy)),
        (2.0 * (xy - wz)),
        (1.0 - 2.0 * (x2 + z2)),
        (2.0 * (yz + wx)),
        (2.0 * (xz + wy)),
        (2.0 * (yz - wx)),
        (1.0 - 2.0 * (x2 + y2))
    );
}

fn quat_to_rotmat_vjp_wxyz(quat: vec4<f32>, v_R: mat3x3<f32>) -> vec4<f32> {
    var w = quat.x;
    var x = quat.y;
    var y = quat.z;
    var z = quat.w;

    let inv_norm = inverseSqrt(x * x + y * y + z * z + w * w);
    x = x * inv_norm;
    y = y * inv_norm;
    z = z * inv_norm;
    w = w * inv_norm;

    let v_quat_n = vec4<f32>(
        2.0 * (x * (v_R[1][2] - v_R[2][1]) +
               y * (v_R[2][0] - v_R[0][2]) +
               z * (v_R[0][1] - v_R[1][0])),
        2.0 * (
            -2.0 * x * (v_R[1][1] + v_R[2][2]) +
             y * (v_R[0][1] + v_R[1][0]) +
             z * (v_R[0][2] + v_R[2][0]) +
             w * (v_R[1][2] - v_R[2][1])
        ),
        2.0 * (
             x * (v_R[0][1] + v_R[1][0]) -
             2.0 * y * (v_R[0][0] + v_R[2][2]) +
             z * (v_R[1][2] + v_R[2][1]) +
             w * (v_R[2][0] - v_R[0][2])
        ),
        2.0 * (
             x * (v_R[0][2] + v_R[2][0]) +
             y * (v_R[1][2] + v_R[2][1]) -
             2.0 * z * (v_R[0][0] + v_R[1][1]) +
             w * (v_R[0][1] - v_R[1][0])
        )
    );

    let quat_n = vec4<f32>(w, x, y, z);
    let proj = dot(v_quat_n, quat_n);
    let scale = inv_norm;
    return (v_quat_n - proj * quat_n) * scale;
}

struct QuatScaleVJP {
    v_quat: vec4<f32>,
    v_scale: vec3<f32>,
};

// Combined VJP for precision-half matrix
fn quat_scale_to_preci_half_vjp_wxyz(
    quat: vec4<f32>,
    scale: vec3<f32>,
    R: mat3x3<f32>,
    v_M: mat3x3<f32>
) -> QuatScaleVJP {
    let sx = 1.0 / scale.x;
    let sy = 1.0 / scale.y;
    let sz = 1.0 / scale.z;

    let S = mat3x3<f32>(
        sx, 0.0, 0.0,
        0.0, sy, 0.0,
        0.0, 0.0, sz
    );

    let v_R = v_M * S;
    let v_quat = quat_to_rotmat_vjp_wxyz(quat, v_R);

    var v_scale = vec3<f32>(0.0, 0.0, 0.0);
    v_scale.x += -sx * sx * (R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2]);
    v_scale.y += -sy * sy * (R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2]);
    v_scale.z += -sz * sz * (R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2]);

    return QuatScaleVJP(v_quat, v_scale);
}

@compute @workgroup_size(256)
fn training_backward_geom(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= active_count) {
        return;
    }

    let g = gaussians[idx];
    let p01 = unpack2x16float(g.pos_opacity[0u]);
    let p02 = unpack2x16float(g.pos_opacity[1u]);

    let position_world = vec4<f32>(p01.x, p01.y, p02.x, 1.0);

    let cam_world = camera.view_inv * vec4<f32>(0.0, 0.0, 0.0, 1.0);
    let ray_o = cam_world.xyz;
    let ray_d = normalize(position_world.xyz - ray_o);

    let r01 = unpack2x16float(g.rot[0u]);
    let r02 = unpack2x16float(g.rot[1u]);
    let s01 = unpack2x16float(g.scale[0u]);
    let s02 = unpack2x16float(g.scale[1u]);

    let log_sigma = vec3<f32>(
        clamp(s01.x, -10.0, 10.0),
        clamp(s01.y, -10.0, 10.0),
        clamp(s02.x, -10.0, 10.0)
    );
    let sigma = exp(log_sigma);

    let q_xyzw = vec4<f32>(r01.y, r02.x, r02.y, r01.x);
    let q_wxyz = vec4<f32>(q_xyzw.w, q_xyzw.x, q_xyzw.y, q_xyzw.z);

    let R = quat_to_rotmat_wxyz(q_wxyz);
    let S = mat3x3<f32>(
        1.0 / sigma.x, 0.0, 0.0,
        0.0, 1.0 / sigma.y, 0.0,
        0.0, 0.0, 1.0 / sigma.z
    );
    let M = R * S;
    let Mt = transpose(M);

    let xyz = position_world.xyz;
    let o_minus_mu = ray_o - xyz;
    let gro = Mt * o_minus_mu;
    let grd = Mt * ray_d;
    let grd_n = safe_normalize(grd);
    let gcrod = cross(grd_n, gro);
    let grayDist = dot(gcrod, gcrod);
    let power = -0.5 * grayDist;
    let vis = exp(power);

    let opacity = 1.0 / (1.0 + exp(-p02.y));
    let alpha_geom = opacity * vis;

    if (power <= 0.0 && alpha_geom > (1.0 / 255.0)) {
        let v_alpha = alphaGradToFloatGeom(atomicLoad(&grad_alpha_splat[idx]));
        let v_vis = opacity * v_alpha;
        let v_gradDist = -0.5 * vis * v_vis;

        let v_gcrod = 2.0 * v_gradDist * gcrod;
        let v_grd_n = -cross(v_gcrod, gro);
        let v_gro = cross(v_gcrod, grd_n);
        let v_grd = safe_normalize_bw(grd, v_grd_n);

        let v_Mt = mat3x3<f32>(
            v_grd.x * ray_d.x + v_gro.x * o_minus_mu.x,
            v_grd.x * ray_d.y + v_gro.x * o_minus_mu.y,
            v_grd.x * ray_d.z + v_gro.x * o_minus_mu.z,
            v_grd.y * ray_d.x + v_gro.y * o_minus_mu.x,
            v_grd.y * ray_d.y + v_gro.y * o_minus_mu.y,
            v_grd.y * ray_d.z + v_gro.y * o_minus_mu.z,
            v_grd.z * ray_d.x + v_gro.z * o_minus_mu.x,
            v_grd.z * ray_d.y + v_gro.z * o_minus_mu.y,
            v_grd.z * ray_d.z + v_gro.z * o_minus_mu.z
        );
        let v_o_minus_mu = transpose(Mt) * v_gro;

        let v_mean = -v_o_minus_mu;

        let v_M = transpose(v_Mt);
        let v_qs = quat_scale_to_preci_half_vjp_wxyz(q_wxyz, sigma, R, v_M);

        let v_quat_wxyz = v_qs.v_quat;
        let v_quat_xyzw = vec4<f32>(
            v_quat_wxyz.y,
            v_quat_wxyz.z,
            v_quat_wxyz.w,
            v_quat_wxyz.x
        );

        let v_log_sigma = v_qs.v_scale * sigma;

        let base_pos = idx * TRAIN_POS_COMPONENTS;
        grad_position[base_pos + 0u] += v_mean.x;
        grad_position[base_pos + 1u] += v_mean.y;
        grad_position[base_pos + 2u] += v_mean.z;

        let base_scale = idx * TRAIN_SCALE_COMPONENTS;
        grad_scale[base_scale + 0u] += v_log_sigma.x;
        grad_scale[base_scale + 1u] += v_log_sigma.y;
        grad_scale[base_scale + 2u] += v_log_sigma.z;

        let base_rot = idx * TRAIN_ROT_COMPONENTS;
        grad_rotation[base_rot + 0u] += v_quat_xyzw.x;
        grad_rotation[base_rot + 1u] += v_quat_xyzw.y;
        grad_rotation[base_rot + 2u] += v_quat_xyzw.z;
        grad_rotation[base_rot + 3u] += v_quat_xyzw.w;
    }
}


