struct TilingParams {
    image_width : u32,
    image_height : u32,
    tile_size : u32,
    tile_width : u32,
    tile_height : u32,
    num_cameras : u32,
};

struct ProjectionParams {
    near_plane : f32,
    far_plane : f32,
    radius_clip : f32,
    eps2d : f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> gaussians : array<Gaussian>;
@group(0) @binding(2) var<uniform> active_count : u32;
@group(0) @binding(3) var<uniform> tiling : TilingParams;
@group(0) @binding(4) var<storage, read_write> means2d : array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> radii : array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> depths : array<f32>;
@group(0) @binding(7) var<storage, read_write> tiles_per_gauss : array<u32>;
@group(0) @binding(12) var<storage, read_write> conics : array<vec3<f32>>;
@group(0) @binding(13) var<uniform> proj : ProjectionParams;

@group(0) @binding(8) var<storage, read> tiles_cumsum : array<u32>;
@group(0) @binding(9) var<storage, read_write> isect_ids : array<u32>;
@group(0) @binding(10) var<storage, read_write> flatten_ids : array<u32>;
@group(0) @binding(11) var<storage, read_write> tile_offsets : array<atomic<u32>>;

fn quat_to_mat3_tiling(q: vec4<f32>) -> mat3x3<f32> {
    let rx = q[0]; let ry = q[1]; let rz = q[2]; let rw = q[3];
    return mat3x3<f32>(
        1.0 - 2.0*(ry * ry + rz * rz),  2.0*(rx * ry - rw * rz),        2.0*(rx * rz + rw * ry),
        2.0*(rx * ry + rw * rz),        1.0 - 2.0*(rx * rx + rz * rz),  2.0*(ry * rz - rw * rx),
        2.0*(rx * rz - rw * ry),        2.0*(ry * rz + rw * rx),        1.0 - 2.0*(rx * rx + ry * ry)
    );
}

@compute @workgroup_size(256)
fn project_gaussians_for_tiling(@builtin(global_invocation_id) gid : vec3<u32>) {
    let n = active_count;
    let C = max(tiling.num_cameras, 1u);
    let global_idx = gid.x;
    if (global_idx >= n * C) {
        return;
    }

    let camera_id = global_idx / n;
    let gauss_idx = global_idx - camera_id * n;
    let base = camera_id * n + gauss_idx;

    // Unpack gaussian parameters
    let g = gaussians[gauss_idx];
    let p01 = unpack2x16float(g.pos_opacity[0u]);
    let p02 = unpack2x16float(g.pos_opacity[1u]);
    let r01 = unpack2x16float(g.rot[0u]);
    let r02 = unpack2x16float(g.rot[1u]);
    let s01 = unpack2x16float(g.scale[0u]);
    let s02 = unpack2x16float(g.scale[1u]);

    // Position
    let position_world = vec4<f32>(p01.x, p01.y, p02.x, 1.0);
    let position_camera = camera.view * position_world;
    var position_clip = camera.proj * position_camera;
    position_clip /= position_clip.w;

    if (abs(position_clip.x) > 1.2 || abs(position_clip.y) > 1.2 || position_camera.z <= 0.0) {
        tiles_per_gauss[base] = 0u;
        radii[base] = vec2<f32>(0.0, 0.0);
        depths[base] = 0.0;
        return;
    }

    let depth_cam = position_camera.z;
    if (depth_cam < proj.near_plane || depth_cam > proj.far_plane) {
        tiles_per_gauss[base] = 0u;
        radii[base] = vec2<f32>(0.0, 0.0);
        depths[base] = 0.0;
        return;
    }

    // Scale
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

    // Rotation
    let q = normalize(vec4<f32>(r01.y, r02.x, r02.y, r01.x));
    let R = quat_to_mat3_tiling(q);

    // Covariance projection
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

    C2D[0][0] += proj.eps2d;
    C2D[1][1] += proj.eps2d;

    let det = C2D[0][0] * C2D[1][1] - C2D[0][1] * C2D[0][1];
    if (det <= 0.0) {
        let base = camera_id * n + gauss_idx;
        tiles_per_gauss[base] = 0u;
        radii[base] = vec2<f32>(0.0, 0.0);
        depths[base] = 0.0;
        return;
    }

    let mid = 0.5 * (C2D[0][0] + C2D[1][1]);
    let disc = sqrt(max(mid * mid - det, 0.0));
    let lambda1 = mid + disc;
    let lambda2 = mid - disc;
    let radius_px = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    // Discard gaussians with excessively large or tiny projected radius.
    if (radius_px > proj.radius_clip || radius_px < 1.0) {
        let base = camera_id * n + gauss_idx;
        tiles_per_gauss[base] = 0u;
        radii[base] = vec2<f32>(0.0, 0.0);
        depths[base] = 0.0;
        return;
    }

    // 2D Gaussian conic parameters
    let det_inv = 1.0 / det;
    let aa = C2D[1][1] * det_inv;
    let bb = -C2D[0][1] * det_inv;
    let cc = C2D[0][0] * det_inv;

    // Store projected mean
    let dims = vec2<f32>(f32(tiling.image_width), f32(tiling.image_height));
    let ndc = position_clip.xy;
    let mean_px = 0.5 * (ndc + vec2<f32>(1.0, 1.0)) * dims;
    means2d[base] = mean_px;
    radii[base] = vec2<f32>(radius_px, radius_px);
    depths[base] = depth_cam;
    conics[base] = vec3<f32>(aa, bb, cc);

    // Tile coverage
    let tile_size_f = f32(tiling.tile_size);
    let tile_x = mean_px.x / tile_size_f;
    let tile_y = mean_px.y / tile_size_f;
    let tile_radius_x = radius_px / tile_size_f;
    let tile_radius_y = radius_px / tile_size_f;

    let tw = f32(tiling.tile_width);
    let th = f32(tiling.tile_height);
    let tile_min_x = min(max(0u, u32(floor(tile_x - tile_radius_x))), tiling.tile_width);
    let tile_min_y = min(max(0u, u32(floor(tile_y - tile_radius_y))), tiling.tile_height);
    let tile_max_x = min(max(0u, u32(ceil(tile_x + tile_radius_x))), tiling.tile_width);
    let tile_max_y = min(max(0u, u32(ceil(tile_y + tile_radius_y))), tiling.tile_height);

    let tiles_x = i32(tile_max_x) - i32(tile_min_x);
    let tiles_y = i32(tile_max_y) - i32(tile_min_y);
    if (tiles_x <= 0 || tiles_y <= 0) {
        tiles_per_gauss[base] = 0u;
    } else {
        tiles_per_gauss[base] = u32(tiles_x * tiles_y);
    }
}

const CAMERA_BITS : u32 = 4u;
const TILE_BITS   : u32 = 12u;
const DEPTH_BITS  : u32 = 16u;

const CAMERA_SHIFT : u32 = TILE_BITS + DEPTH_BITS;
const TILE_SHIFT   : u32 = DEPTH_BITS;

const CAMERA_MASK : u32 = (1u << CAMERA_BITS) - 1u;
const TILE_MASK   : u32 = (1u << TILE_BITS) - 1u;
const DEPTH_MASK  : u32 = (1u << DEPTH_BITS) - 1u;

fn encode_isect_key(camera_id: u32, tile_id: u32, depth: f32) -> u32 {
    let depth_clamped = clamp(depth, 0.0, 65535.0);
    let depth_q = u32(depth_clamped) & DEPTH_MASK;
    let cam_q = camera_id & CAMERA_MASK;
    let tile_q = tile_id & TILE_MASK;
    return (cam_q << CAMERA_SHIFT) | (tile_q << TILE_SHIFT) | depth_q;
}

@compute @workgroup_size(256)
fn emit_intersections(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= active_count) {
        return;
    }

    let tiles_count = tiles_per_gauss[idx];
    if (tiles_count == 0u) {
        return;
    }

    var start: u32;
    if (idx == 0u) {
        start = 0u;
    } else {
        start = tiles_cumsum[idx - 1u];
    }

    let mean_px = means2d[idx];
    let radius_px = radii[idx].x;
    let depth_val = depths[idx];

    let tile_size_f = f32(tiling.tile_size);
    let tile_x = mean_px.x / tile_size_f;
    let tile_y = mean_px.y / tile_size_f;
    let tile_radius_x = radius_px / tile_size_f;
    let tile_radius_y = radius_px / tile_size_f;

    let tile_min_x = min(max(0u, u32(floor(tile_x - tile_radius_x))), tiling.tile_width);
    let tile_min_y = min(max(0u, u32(floor(tile_y - tile_radius_y))), tiling.tile_height);
    let tile_max_x = min(max(0u, u32(ceil(tile_x + tile_radius_x))), tiling.tile_width);
    let tile_max_y = min(max(0u, u32(ceil(tile_y + tile_radius_y))), tiling.tile_height);

    var cur = start;
    let width = tiling.tile_width;
    for (var ty = tile_min_y; ty < tile_max_y; ty = ty + 1u) {
        for (var tx = tile_min_x; tx < tile_max_x; tx = tx + 1u) {
            let tile_id = ty * width + tx;
            let key = encode_isect_key(0u, tile_id, depth_val);
            isect_ids[cur] = key;
            flatten_ids[cur] = idx;
            cur = cur + 1u;
        }
    }
}

@compute @workgroup_size(256)
fn build_tile_offsets(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    let n_tiles = tiling.tile_width * tiling.tile_height;
    let num_cameras = max(tiling.num_cameras, 1u);
    let n_tiles_total = n_tiles * num_cameras;
    let n_isects = tiles_cumsum[active_count - 1u];

    if (i >= n_isects) {
        return;
    }

    let key = isect_ids[i];
    let tile_and_cam = key >> TILE_SHIFT;
    let tile_id = tile_and_cam & TILE_MASK;
    let camera_id = tile_and_cam >> TILE_BITS;

    let tile_index = camera_id * n_tiles + tile_id;

    if (tile_index < n_tiles_total) {
        atomicMin(&tile_offsets[tile_index], i);
    }
}
