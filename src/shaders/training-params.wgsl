@group(0) @binding(0) var<storage, read_write> gaussians : array<Gaussian>;
@group(0) @binding(1) var<storage, read_write> sh_buffer : array<u32>;
@group(0) @binding(2) var<storage, read_write> train_opacity : array<f32>;
@group(0) @binding(3) var<storage, read_write> train_sh : array<f32>;
@group(0) @binding(4) var<uniform> active_count : u32;
// Additional trainable parameter buffers (float32, renderer-managed)
@group(0) @binding(5) var<storage, read_write> train_position : array<f32>;   // 3 floats per gaussian
@group(0) @binding(6) var<storage, read_write> train_scale    : array<f32>;   // 3 floats per gaussian (log-sigmas)
@group(0) @binding(7) var<storage, read_write> train_rotation : array<f32>;   // 4 floats per gaussian (quaternion)

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

fn write_f16_coeff(base_word: u32, elem: u32, value: f32) {
    let word_idx = base_word + (elem >> 1u);
    var halves   = unpack2x16float(sh_buffer[word_idx]);
    if ((elem & 1u) == 0u) {
        halves.x = value;
    } else {
        halves.y = value;
    }
    sh_buffer[word_idx] = pack2x16float(halves);
}

@compute @workgroup_size(256)
fn training_init_params(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= active_count) {
        return;
    }

    let g = gaussians[idx];
    let p02 = unpack2x16float(g.pos_opacity[1u]);

    train_opacity[idx] = p02.y;

    // ---- Initialize position (world space) ----
    let p01 = unpack2x16float(g.pos_opacity[0u]);
    let base_pos = idx * 3u;
    train_position[base_pos + 0u] = p01.x; // x
    train_position[base_pos + 1u] = p01.y; // y
    train_position[base_pos + 2u] = p02.x; // z

    // ---- Initialize scale (log-sigmas used in preprocess.wgsl) ----
    let s01 = unpack2x16float(g.scale[0u]);
    let s02 = unpack2x16float(g.scale[1u]);
    let base_scale = idx * 3u;
    train_scale[base_scale + 0u] = s01.x;
    train_scale[base_scale + 1u] = s01.y;
    train_scale[base_scale + 2u] = s02.x;

    // ---- Initialize rotation (quaternion layout matches preprocess.wgsl) ----
    let r01 = unpack2x16float(g.rot[0u]);
    let r02 = unpack2x16float(g.rot[1u]);
    let base_rot = idx * 4u;
    // q = vec4(r01.y, r02.x, r02.y, r01.x)
    train_rotation[base_rot + 0u] = r01.y; // q.x
    train_rotation[base_rot + 1u] = r02.x; // q.y
    train_rotation[base_rot + 2u] = r02.y; // q.z
    train_rotation[base_rot + 3u] = r01.x; // q.w

    // ---- Initialize SH coefficients (already trained) ----
    let base_word = idx * SH_WORDS_PER_POINT;
    let base_sh = idx * SH_COMPONENTS;
    for (var j: u32 = 0u; j < SH_COMPONENTS; j = j + 1u) {
        let v = read_f16_coeff(base_word, j);
        train_sh[base_sh + j] = v;
    }
}

@compute @workgroup_size(256)
fn training_apply_params(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= active_count) {
        return;
    }

    var g = gaussians[idx];
    var p01 = unpack2x16float(g.pos_opacity[0u]);
    var p02 = unpack2x16float(g.pos_opacity[1u]);
    var s01 = unpack2x16float(g.scale[0u]);
    var s02 = unpack2x16float(g.scale[1u]);
    var r01 = unpack2x16float(g.rot[0u]);
    var r02 = unpack2x16float(g.rot[1u]);

    let base_pos = idx * 3u;
    p01.x = train_position[base_pos + 0u];
    p01.y = train_position[base_pos + 1u];
    p02.x = train_position[base_pos + 2u];

    let base_scale = idx * 3u;
    s01.x = train_scale[base_scale + 0u];
    s01.y = train_scale[base_scale + 1u];
    s02.x = train_scale[base_scale + 2u];

    let base_rot = idx * 4u;
    r01.y = train_rotation[base_rot + 0u];
    r02.x = train_rotation[base_rot + 1u];
    r02.y = train_rotation[base_rot + 2u];
    r01.x = train_rotation[base_rot + 3u];

    p02.y = train_opacity[idx];

    g.pos_opacity[0u] = pack2x16float(p01);
    g.pos_opacity[1u] = pack2x16float(p02);
    g.scale[0u] = pack2x16float(s01);
    g.scale[1u] = pack2x16float(s02);
    g.rot[0u] = pack2x16float(r01);
    g.rot[1u] = pack2x16float(r02);
    gaussians[idx] = g;

    let base_word = idx * SH_WORDS_PER_POINT;
    let base_sh = idx * SH_COMPONENTS;
    for (var j: u32 = 0u; j < SH_COMPONENTS; j = j + 1u) {
        let v = train_sh[base_sh + j];
        write_f16_coeff(base_word, j, v);
    }
}


