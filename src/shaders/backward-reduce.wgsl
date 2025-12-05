
const TILE_GRAD_STRIDE     : u32 = 54u;
const TILE_GRAD_OPACITY    : u32 = 0u;
const TILE_GRAD_GEOM_START : u32 = 1u;
const TILE_GRAD_SH_START   : u32 = 6u;

const MAX_SH_COEFFS : u32 = 16u;
const SH_COMPONENTS : u32 = MAX_SH_COEFFS * 3u;

const GEOM_GRAD_STRIDE : u32 = 5u;

@group(0) @binding(0) var<storage, read> tile_grads : array<f32>;
@group(0) @binding(1) var<storage, read> flatten_ids : array<u32>;
@group(0) @binding(2) var<storage, read> tiles_cumsum : array<u32>;
@group(0) @binding(3) var<uniform> active_count : u32;

// Output gradient buffers (atomic accumulation)
@group(0) @binding(4) var<storage, read_write> grad_sh : array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> grad_opacity : array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> grad_geom : array<atomic<u32>>;

fn atomicAdd_f32(ptr: ptr<storage, atomic<u32>, read_write>, delta: f32) {
    loop {
        let oldBits : u32 = atomicLoad(ptr);
        let oldVal  : f32 = bitcast<f32>(oldBits);
        let newVal  : f32 = oldVal + delta;
        let newBits : u32 = bitcast<u32>(newVal);
        let res = atomicCompareExchangeWeak(ptr, oldBits, newBits);
        if (res.exchanged) {
            break;
        }
    }
}

@compute @workgroup_size(256)
fn backward_reduce(@builtin(global_invocation_id) gid : vec3<u32>) {
    var n_isects: u32 = 0u;
    if (active_count > 0u && tiles_cumsum[active_count - 1u] > 0u) {
        n_isects = tiles_cumsum[active_count - 1u];
    }
    
    let isect_idx = gid.x;
    if (isect_idx >= n_isects) {
        return;
    }
    
    let g_idx = flatten_ids[isect_idx];
    
    let grad_base = isect_idx * TILE_GRAD_STRIDE;
    
    let dL_dlogit = tile_grads[grad_base + TILE_GRAD_OPACITY];
    if (abs(dL_dlogit) > 1e-10) {
        atomicAdd_f32(&grad_opacity[g_idx], dL_dlogit);
    }
    
    let geom_base = g_idx * GEOM_GRAD_STRIDE;
    for (var i = 0u; i < 5u; i = i + 1u) {
        let grad_val = tile_grads[grad_base + TILE_GRAD_GEOM_START + i];
        if (abs(grad_val) > 1e-10) {
            atomicAdd_f32(&grad_geom[geom_base + i], grad_val);
        }
    }
    
    let sh_base = g_idx * SH_COMPONENTS;
    for (var i = 0u; i < SH_COMPONENTS; i = i + 1u) {
        let grad_val = tile_grads[grad_base + TILE_GRAD_SH_START + i];
        if (abs(grad_val) > 1e-10) {
            atomicAdd_f32(&grad_sh[sh_base + i], grad_val);
        }
    }
}

