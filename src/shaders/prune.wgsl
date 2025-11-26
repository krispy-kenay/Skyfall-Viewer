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
var<storage, read_write> active_count : atomic<u32>;

@group(0) @binding(7)
var<uniform> prune_threshold : f32;

@group(0) @binding(8)
var<uniform> input_count : u32;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn prune(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= input_count) {
        return;
    }
    
    let g = gaussians_in[idx];
    let p02 = unpack2x16float(g.pos_opacity[1u]);
    let opacity = sigmoid(p02.y);
    
    if (opacity >= prune_threshold) {
        let out_idx = atomicAdd(&active_count, 1u);
        
        gaussians_out[out_idx] = g;
        
        let base_in = idx * SH_WORDS_PER_POINT;
        let base_out = out_idx * SH_WORDS_PER_POINT;
        for (var i = 0u; i < SH_WORDS_PER_POINT; i++) {
            sh_buffer_out[base_out + i] = sh_buffer_in[base_in + i];
        }
        
        density_buffer_out[out_idx] = density_buffer_in[idx];
    }
}

