@group(0) @binding(0)
var<storage, read> gaussians_in : array<Gaussian>;

@group(0) @binding(1)
var<storage, read> sh_buffer_in : array<u32>;

@group(0) @binding(2)
var<storage, read> density_buffer_in : array<f32>;

@group(0) @binding(3)
var<storage, read_write> gaussians_out : array<Gaussian>;

@group(0) @binding(4)
var<storage, read_write> sh_buffer_out : array<u32>;

@group(0) @binding(5)
var<storage, read_write> density_buffer_out : array<f32>;

@group(0) @binding(6)
var<storage, read_write> new_count : atomic<u32>;

@group(0) @binding(7)
var<uniform> active_count : u32;

@group(0) @binding(8)
var<uniform> split_threshold : f32;
@group(0) @binding(9)
var<uniform> clone_threshold : f32;

@group(0) @binding(10) var<uniform> camera: CameraUniforms;
@group(0) @binding(11) var<uniform> settings: RenderSettings;
@group(0) @binding(12) var samp : sampler;
@group(0) @binding(13) var targetTex : texture_2d<f32>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn read_f16_coeff(base_word: u32, elem: u32) -> f32 {
    let word_idx = base_word + (elem >> 1u);
    let halves   = unpack2x16float(sh_buffer_in[word_idx]);
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
    return max(vec3<f32>(0.), result);
}

fn write_f16_coeff(base_word: u32, elem: u32, value: f32) {
    let word_idx = base_word + (elem >> 1u);
    var halves   = unpack2x16float(sh_buffer_out[word_idx]);
    if ((elem & 1u) == 0u) {
        halves.x = value;
    } else {
        halves.y = value;
    }
    sh_buffer_out[word_idx] = pack2x16float(halves);
}

fn write_sh_coef(splat_idx: u32, c_idx: u32, value: vec3<f32>) {
    let base_word = splat_idx * SH_WORDS_PER_POINT;
    let eR = c_idx * 3u + 0u;
    let eG = c_idx * 3u + 1u;
    let eB = c_idx * 3u + 2u;
    write_f16_coeff(base_word, eR, value.x);
    write_f16_coeff(base_word, eG, value.y);
    write_f16_coeff(base_word, eB, value.z);
}

@compute @workgroup_size(256)
fn densify(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= active_count) {
        return;
    }
    
    let g = gaussians_in[idx];
    let p01 = unpack2x16float(g.pos_opacity[0u]);
    let p02 = unpack2x16float(g.pos_opacity[1u]);
    let s01 = unpack2x16float(g.scale[0u]);
    let s02 = unpack2x16float(g.scale[1u]);
    
    let position_world = vec4<f32>(p01.x, p01.y, p02.x, 1.0);
    let position_camera = camera.view * position_world;
    let position_clip = camera.proj * position_camera;
    let ndc = position_clip.xy / position_clip.w;
    let uv = 0.5 * (ndc + vec2<f32>(1.0, 1.0));
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return;
    }
    
    let mean_dist = density_buffer_in[idx];
    let density_target_pixels = (mean_dist / 3.0) / 0.003;
    let sigma_now = exp(clamp(s01.x, -10.0, 10.0));
    let current_pixels = sigma_now / 0.003;

    let targ = textureSampleLevel(targetTex, samp, uv, 0.0).rgb;
    let viewDir = normalize(-(camera.view * position_world).xyz);
    let sh_deg = u32(settings.sh_deg);
    let shColor = computeColorFromSH(viewDir, idx, sh_deg);
    let error = length(targ - shColor);
    
    let size_ratio = current_pixels / max(density_target_pixels, 1.0);
    let should_split = (size_ratio > split_threshold) && (error > clone_threshold);
    
    if (should_split) {
        let offset_scale = 0.2;
        
        for (var split_idx = 0u; split_idx < 2u; split_idx++) {
            let new_idx = atomicAdd(&new_count, 1u);

            let offset_dir = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), split_idx == 1u);
            let offset = offset_dir * sigma_now * offset_scale;
            let new_pos = position_world.xyz + offset;
            
            var new_g = g;
            var new_p01 = p01;
            var new_p02 = p02;
            new_p01.x = new_pos.x;
            new_p01.y = new_pos.y;
            new_p02.x = new_pos.z;
            
            var new_s01 = s01;
            var new_s02 = s02;
            let log_sigma_new = s01.x - 0.3466;
            new_s01.x = log_sigma_new;
            new_s01.y = log_sigma_new;
            new_s02.x = log_sigma_new;
            
            let current_alpha = sigmoid(p02.y);
            let new_alpha = current_alpha * 0.8;
            new_p02.y = log(new_alpha / (1.0 - new_alpha));
            
            new_g.pos_opacity[0u] = pack2x16float(new_p01);
            new_g.pos_opacity[1u] = pack2x16float(new_p02);
            new_g.scale[0u] = pack2x16float(new_s01);
            new_g.scale[1u] = pack2x16float(new_s02);
            
            gaussians_out[new_idx] = new_g;
            
            let base_in = idx * SH_WORDS_PER_POINT;
            let base_out = new_idx * SH_WORDS_PER_POINT;
            for (var i = 0u; i < SH_WORDS_PER_POINT; i++) {
                sh_buffer_out[base_out + i] = sh_buffer_in[base_in + i];
            }
            
            density_buffer_out[new_idx] = mean_dist;
        }
        
    }
    else if (error > clone_threshold * 1.5) {
        let new_idx = atomicAdd(&new_count, 1u);
        
        let offset_scale = 0.1;
        let offset = vec3<f32>(1.0, 0.0, 0.0) * sigma_now * offset_scale;
        let new_pos = position_world.xyz + offset;
        
        var new_g = g;
        var new_p01 = p01;
        var new_p02 = p02;
        new_p01.x = new_pos.x;
        new_p01.y = new_pos.y;
        new_p02.x = new_pos.z;
        
        let current_alpha = sigmoid(p02.y);
        let new_alpha = current_alpha * 0.9;
        new_p02.y = log(new_alpha / (1.0 - new_alpha));
        
        new_g.pos_opacity[0u] = pack2x16float(new_p01);
        new_g.pos_opacity[1u] = pack2x16float(new_p02);
        
        gaussians_out[new_idx] = new_g;
        
        let base_in = idx * SH_WORDS_PER_POINT;
        let base_out = new_idx * SH_WORDS_PER_POINT;
        for (var i = 0u; i < SH_WORDS_PER_POINT; i++) {
            sh_buffer_out[base_out + i] = sh_buffer_in[base_in + i];
        }
        
        density_buffer_out[new_idx] = mean_dist;
    }
}

