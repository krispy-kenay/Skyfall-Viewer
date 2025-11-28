@group(0) @binding(0) var<storage, read_write> gaussians : array<Gaussian>;
@group(0) @binding(1) var<storage, read_write> sh_buffer : array<u32>;
@group(0) @binding(2) var<storage, read_write> train_opacity : array<f32>;
@group(0) @binding(3) var<storage, read_write> train_sh : array<f32>;
@group(0) @binding(4) var<uniform> active_count : u32;

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

    p02.y = train_opacity[idx];
    g.pos_opacity[0u] = pack2x16float(p01);
    g.pos_opacity[1u] = pack2x16float(p02);
    gaussians[idx] = g;

    let base_word = idx * SH_WORDS_PER_POINT;
    let base_sh = idx * SH_COMPONENTS;
    for (var j: u32 = 0u; j < SH_COMPONENTS; j = j + 1u) {
        let v = train_sh[base_sh + j];
        write_f16_coeff(base_word, j, v);
    }
}


