@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(0) @binding(1) var<storage, read> gaussians : array<Gaussian>;
@group(0) @binding(2) var<uniform> active_count : u32;

@group(0) @binding(3) var<storage, read> grad_geom_in : array<f32>;

@group(0) @binding(4) var<storage, read_write> grad_position : array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_scale : array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_rotation : array<f32>;

const TRAIN_POS_COMPONENTS : u32 = 3u;
const TRAIN_SCALE_COMPONENTS : u32 = 3u;
const TRAIN_ROT_COMPONENTS : u32 = 4u;

const EPS_2D : f32 = 0.3;

const LIM_X : f32 = 1.3;
const LIM_Y : f32 = 1.3;

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let r = q.w;
    let x = q.x;
    let y = q.y;
    let z = q.z;
    return mat3x3<f32>(
        1.0 - 2.0*(y*y + z*z),  2.0*(x*y - r*z),        2.0*(x*z + r*y),
        2.0*(x*y + r*z),        1.0 - 2.0*(x*x + z*z),  2.0*(y*z - r*x),
        2.0*(x*z - r*y),        2.0*(y*z + r*x),        1.0 - 2.0*(x*x + y*y)
    );
}

// =============================================================================
// MAIN KERNEL
// =============================================================================
@compute @workgroup_size(256)
fn training_backward_geom(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= active_count) {
        return;
    }

    let geom_base = idx * 5u;
    let dL_dconic_a = grad_geom_in[geom_base + 0u];
    let dL_dconic_b = grad_geom_in[geom_base + 1u];
    let dL_dconic_c = grad_geom_in[geom_base + 2u];
    let dL_dmean2d_x = grad_geom_in[geom_base + 3u];
    let dL_dmean2d_y = grad_geom_in[geom_base + 4u];
    
    // Skip if no gradient signal
    let conic_mag = abs(dL_dconic_a) + abs(dL_dconic_b) + abs(dL_dconic_c);
    let mean_mag = abs(dL_dmean2d_x) + abs(dL_dmean2d_y);
    if (conic_mag < 1e-10 && mean_mag < 1e-10) {
        return;
    }

    // =========================================================================
    // Load Gaussian parameters
    // =========================================================================
    let g = gaussians[idx];
    let p01 = unpack2x16float(g.pos_opacity[0u]);
    let p02 = unpack2x16float(g.pos_opacity[1u]);
    let r01 = unpack2x16float(g.rot[0u]);
    let r02 = unpack2x16float(g.rot[1u]);
    let s01 = unpack2x16float(g.scale[0u]);
    let s02 = unpack2x16float(g.scale[1u]);

    let position_world = vec3<f32>(p01.x, p01.y, p02.x);
    
    // =========================================================================
    // Transform to camera space
    // =========================================================================
    let pos_world_h = vec4<f32>(position_world, 1.0);
    let t = (camera.view * pos_world_h).xyz;
    
    if (t.z <= 0.0) {
        return;
    }

    // Focal lengths
    let fx = camera.focal.x;
    let fy = camera.focal.y;
    
    // Clamp to frustum
    let tan_fovx = camera.viewport.x / (2.0 * fx);
    let tan_fovy = camera.viewport.y / (2.0 * fy);
    let limx = LIM_X * tan_fovx;
    let limy = LIM_Y * tan_fovy;
    
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;
    let t_clamped = vec3<f32>(
        clamp(txtz, -limx, limx) * t.z,
        clamp(tytz, -limy, limy) * t.z,
        t.z
    );
    
    let x_grad_mul = select(0.0, 1.0, txtz >= -limx && txtz <= limx);
    let y_grad_mul = select(0.0, 1.0, tytz >= -limy && tytz <= limy);

    // =========================================================================
    // Compute Jacobian of projection
    // =========================================================================
    let tz_inv = 1.0 / t_clamped.z;
    let tz2_inv = tz_inv * tz_inv;
    
    let J = mat3x3<f32>(
        fx * tz_inv, 0.0,         -fx * t_clamped.x * tz2_inv,
        0.0,         fy * tz_inv, -fy * t_clamped.y * tz2_inv,
        0.0,         0.0,         0.0
    );

    // =========================================================================
    // Extract view rotation matrix W (upper-left 3x3 of view matrix)
    // =========================================================================
    let W = mat3x3<f32>(
        camera.view[0].x, camera.view[1].x, camera.view[2].x,
        camera.view[0].y, camera.view[1].y, camera.view[2].y,
        camera.view[0].z, camera.view[1].z, camera.view[2].z
    );

    let T_mat = W * J;

    // =========================================================================
    // Compute 3D covariance from scale and rotation
    // =========================================================================
    let log_sigma = vec3<f32>(
        clamp(s01.x, -10.0, 10.0),
        clamp(s01.y, -10.0, 10.0),
        clamp(s02.x, -10.0, 10.0)
    );
    let sigma = exp(log_sigma);
    
    let q_raw = vec4<f32>(r01.y, r02.x, r02.y, r01.x);
    let q = normalize(q_raw);
    let R = quat_to_mat3(q);
    
    let S = mat3x3<f32>(
        sigma.x, 0.0,     0.0,
        0.0,     sigma.y, 0.0,
        0.0,     0.0,     sigma.z
    );
    let M = S * R;
    
    let cov3D = transpose(M) * M;

    // =========================================================================
    // Compute 2D covariance (for reference)
    // =========================================================================
    var cov2D = transpose(T_mat) * transpose(cov3D) * T_mat;
    
    let a = cov2D[0][0] + EPS_2D;
    let b = cov2D[0][1];
    let c = cov2D[1][1] + EPS_2D;
    
    let det = a * c - b * b;
    if (det <= 0.0) {
        return;
    }

    // =========================================================================
    // STEP 1: dL/dconic -> dL/dcov2D
    // =========================================================================
    let denom2inv = 1.0 / (det * det + 1e-7);
    
    let dL_da = denom2inv * (-c * c * dL_dconic_a + 2.0 * b * c * dL_dconic_b + (det - a * c) * dL_dconic_c);
    let dL_dc = denom2inv * (-a * a * dL_dconic_c + 2.0 * a * b * dL_dconic_b + (det - a * c) * dL_dconic_a);
    let dL_db = denom2inv * 2.0 * (b * c * dL_dconic_a - (det + 2.0 * b * b) * dL_dconic_b + a * b * dL_dconic_c);

    // =========================================================================
    // STEP 2: dL/dcov2D -> dL/dcov3D
    // =========================================================================
    let dL_dcov3D_00 = T_mat[0][0] * T_mat[0][0] * dL_da + T_mat[0][0] * T_mat[1][0] * dL_db + T_mat[1][0] * T_mat[1][0] * dL_dc;
    let dL_dcov3D_11 = T_mat[0][1] * T_mat[0][1] * dL_da + T_mat[0][1] * T_mat[1][1] * dL_db + T_mat[1][1] * T_mat[1][1] * dL_dc;
    let dL_dcov3D_22 = T_mat[0][2] * T_mat[0][2] * dL_da + T_mat[0][2] * T_mat[1][2] * dL_db + T_mat[1][2] * T_mat[1][2] * dL_dc;
    
    let dL_dcov3D_01 = 2.0 * T_mat[0][0] * T_mat[0][1] * dL_da + (T_mat[0][0] * T_mat[1][1] + T_mat[0][1] * T_mat[1][0]) * dL_db + 2.0 * T_mat[1][0] * T_mat[1][1] * dL_dc;
    let dL_dcov3D_02 = 2.0 * T_mat[0][0] * T_mat[0][2] * dL_da + (T_mat[0][0] * T_mat[1][2] + T_mat[0][2] * T_mat[1][0]) * dL_db + 2.0 * T_mat[1][0] * T_mat[1][2] * dL_dc;
    let dL_dcov3D_12 = 2.0 * T_mat[0][2] * T_mat[0][1] * dL_da + (T_mat[0][1] * T_mat[1][2] + T_mat[0][2] * T_mat[1][1]) * dL_db + 2.0 * T_mat[1][1] * T_mat[1][2] * dL_dc;

    let dL_dSigma = mat3x3<f32>(
        dL_dcov3D_00,     0.5 * dL_dcov3D_01, 0.5 * dL_dcov3D_02,
        0.5 * dL_dcov3D_01, dL_dcov3D_11,     0.5 * dL_dcov3D_12,
        0.5 * dL_dcov3D_02, 0.5 * dL_dcov3D_12, dL_dcov3D_22
    );

    // =========================================================================
    // STEP 3: dL/dcov3D -> dL/dM (where cov3D = M^T * M)
    // =========================================================================
    let dL_dM = 2.0 * M * dL_dSigma;
    let dL_dMt = transpose(dL_dM);
    let Rt = transpose(R);

    // =========================================================================
    // STEP 4: dL/dM -> dL/dscale
    // =========================================================================
    var dL_dscale_raw = vec3<f32>(
        dot(Rt[0], dL_dMt[0]),
        dot(Rt[1], dL_dMt[1]),
        dot(Rt[2], dL_dMt[2])
    );

    // =========================================================================
    // STEP 5: dL/dM -> dL/drotation
    // =========================================================================
    var dL_dMt_scaled = dL_dMt;
    dL_dMt_scaled[0] *= sigma.x;
    dL_dMt_scaled[1] *= sigma.y;
    dL_dMt_scaled[2] *= sigma.z;
    
    let r = q.w;
    let x = q.x;
    let y = q.y;
    let z = q.z;
    
    var dL_dq = vec4<f32>(0.0);
    dL_dq.w = 2.0 * z * (dL_dMt_scaled[0][1] - dL_dMt_scaled[1][0]) + 
              2.0 * y * (dL_dMt_scaled[2][0] - dL_dMt_scaled[0][2]) + 
              2.0 * x * (dL_dMt_scaled[1][2] - dL_dMt_scaled[2][1]);
    dL_dq.x = 2.0 * y * (dL_dMt_scaled[1][0] + dL_dMt_scaled[0][1]) + 
              2.0 * z * (dL_dMt_scaled[2][0] + dL_dMt_scaled[0][2]) + 
              2.0 * r * (dL_dMt_scaled[1][2] - dL_dMt_scaled[2][1]) - 
              4.0 * x * (dL_dMt_scaled[2][2] + dL_dMt_scaled[1][1]);
    dL_dq.y = 2.0 * x * (dL_dMt_scaled[1][0] + dL_dMt_scaled[0][1]) + 
              2.0 * r * (dL_dMt_scaled[2][0] - dL_dMt_scaled[0][2]) + 
              2.0 * z * (dL_dMt_scaled[1][2] + dL_dMt_scaled[2][1]) - 
              4.0 * y * (dL_dMt_scaled[2][2] + dL_dMt_scaled[0][0]);
    dL_dq.z = 2.0 * r * (dL_dMt_scaled[0][1] - dL_dMt_scaled[1][0]) + 
              2.0 * x * (dL_dMt_scaled[2][0] + dL_dMt_scaled[0][2]) + 
              2.0 * y * (dL_dMt_scaled[1][2] + dL_dMt_scaled[2][1]) - 
              4.0 * z * (dL_dMt_scaled[1][1] + dL_dMt_scaled[0][0]);

    // =========================================================================
    // STEP 6: dL/dcov2D -> dL/dT -> dL/dmean
    // =========================================================================
    let dL_dT00 = 2.0 * (T_mat[0][0] * cov3D[0][0] + T_mat[0][1] * cov3D[0][1] + T_mat[0][2] * cov3D[0][2]) * dL_da +
                        (T_mat[1][0] * cov3D[0][0] + T_mat[1][1] * cov3D[0][1] + T_mat[1][2] * cov3D[0][2]) * dL_db;
    let dL_dT01 = 2.0 * (T_mat[0][0] * cov3D[1][0] + T_mat[0][1] * cov3D[1][1] + T_mat[0][2] * cov3D[1][2]) * dL_da +
                        (T_mat[1][0] * cov3D[1][0] + T_mat[1][1] * cov3D[1][1] + T_mat[1][2] * cov3D[1][2]) * dL_db;
    let dL_dT02 = 2.0 * (T_mat[0][0] * cov3D[2][0] + T_mat[0][1] * cov3D[2][1] + T_mat[0][2] * cov3D[2][2]) * dL_da +
                        (T_mat[1][0] * cov3D[2][0] + T_mat[1][1] * cov3D[2][1] + T_mat[1][2] * cov3D[2][2]) * dL_db;
    let dL_dT10 = 2.0 * (T_mat[1][0] * cov3D[0][0] + T_mat[1][1] * cov3D[0][1] + T_mat[1][2] * cov3D[0][2]) * dL_dc +
                        (T_mat[0][0] * cov3D[0][0] + T_mat[0][1] * cov3D[0][1] + T_mat[0][2] * cov3D[0][2]) * dL_db;
    let dL_dT11 = 2.0 * (T_mat[1][0] * cov3D[1][0] + T_mat[1][1] * cov3D[1][1] + T_mat[1][2] * cov3D[1][2]) * dL_dc +
                        (T_mat[0][0] * cov3D[1][0] + T_mat[0][1] * cov3D[1][1] + T_mat[0][2] * cov3D[1][2]) * dL_db;
    let dL_dT12 = 2.0 * (T_mat[1][0] * cov3D[2][0] + T_mat[1][1] * cov3D[2][1] + T_mat[1][2] * cov3D[2][2]) * dL_dc +
                        (T_mat[0][0] * cov3D[2][0] + T_mat[0][1] * cov3D[2][1] + T_mat[0][2] * cov3D[2][2]) * dL_db;

    let dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
    let dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
    let dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
    let dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

    let tz3_inv = tz2_inv * tz_inv;
    let dL_dtx_cov = x_grad_mul * (-fx * tz2_inv * dL_dJ02);
    let dL_dty_cov = y_grad_mul * (-fy * tz2_inv * dL_dJ12);
    let dL_dtz_cov = -fx * tz2_inv * dL_dJ00 - fy * tz2_inv * dL_dJ11 +
                     (2.0 * fx * t_clamped.x) * tz3_inv * dL_dJ02 +
                     (2.0 * fy * t_clamped.y) * tz3_inv * dL_dJ12;

    // =========================================================================
    // STEP 7: Combine mean2D gradients with position gradients from covariance
    // =========================================================================
    let dL_dtx_mean = dL_dmean2d_x * fx * tz_inv;
    let dL_dty_mean = dL_dmean2d_y * fy * tz_inv;
    let dL_dtz_mean = -dL_dmean2d_x * fx * t.x * tz2_inv - dL_dmean2d_y * fy * t.y * tz2_inv;

    let dL_dt = vec3<f32>(
        dL_dtx_cov + dL_dtx_mean,
        dL_dty_cov + dL_dty_mean,
        dL_dtz_cov + dL_dtz_mean
    );

    let dL_dpos_world = transpose(W) * dL_dt;

    // =========================================================================
    // STEP 8: Convert scale gradients from sigma to log-sigma
    // =========================================================================
    let dL_dlog_sigma = dL_dscale_raw * sigma;

    // =========================================================================
    // STEP 9: Write gradients to output buffers
    // =========================================================================
    let base_pos = idx * TRAIN_POS_COMPONENTS;
    grad_position[base_pos + 0u] += dL_dpos_world.x;
    grad_position[base_pos + 1u] += dL_dpos_world.y;
    grad_position[base_pos + 2u] += dL_dpos_world.z;

    let base_scale = idx * TRAIN_SCALE_COMPONENTS;
    grad_scale[base_scale + 0u] += dL_dlog_sigma.x;
    grad_scale[base_scale + 1u] += dL_dlog_sigma.y;
    grad_scale[base_scale + 2u] += dL_dlog_sigma.z;

    let base_rot = idx * TRAIN_ROT_COMPONENTS;
    grad_rotation[base_rot + 0u] += dL_dq.x;
    grad_rotation[base_rot + 1u] += dL_dq.y;
    grad_rotation[base_rot + 2u] += dL_dq.z;
    grad_rotation[base_rot + 3u] += dL_dq.w;
}
