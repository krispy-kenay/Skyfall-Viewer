@group(0) @binding(0)
var<storage, read_write> gaussians : array<Gaussian>;

@group(0) @binding(1)
var<uniform> num_points : u32;

@group(0) @binding(2)
var<storage, read_write> density_buffer : array<f32>;

fn read_position(idx: u32) -> vec3<f32> {
    let g = gaussians[idx];
    let p01 = unpack2x16float(g.pos_opacity[0u]);
    let p02 = unpack2x16float(g.pos_opacity[1u]);
    return vec3<f32>(p01.x, p01.y, p02.x);
}

fn insert_sorted(distances: ptr<function, array<f32, 3>>, new_dist: f32) {
    if (new_dist < (*distances)[0u]) {
        (*distances)[2u] = (*distances)[1u];
        (*distances)[1u] = (*distances)[0u];
        (*distances)[0u] = new_dist;
    } else if (new_dist < (*distances)[1u]) {
        (*distances)[2u] = (*distances)[1u];
        (*distances)[1u] = new_dist;
    } else if (new_dist < (*distances)[2u]) {
        (*distances)[2u] = new_dist;
    }
}

@compute @workgroup_size(256)
fn knn_init(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= num_points) {
        return;
    }

    let my_pos = read_position(idx);
    
    var nearest_dists = array<f32, 3>(
        1e10,
        1e10,
        1e10
    );

    for (var i = 0u; i < num_points; i++) {
        if (i == idx) {
            continue;
        }
        
        let other_pos = read_position(i);
        let dist = length(my_pos - other_pos);
        
        insert_sorted(&nearest_dists, dist);
    }
    let mean_dist = (nearest_dists[0u] + nearest_dists[1u] + nearest_dists[2u]) / 3.0;
    
    density_buffer[idx] = mean_dist;
    
    let initial_sigma = mean_dist / 3.0;
    let log_sigma = log(max(initial_sigma, 1e-6));
    
    var g = gaussians[idx];
    var s01 = unpack2x16float(g.scale[0u]);
    var s02 = unpack2x16float(g.scale[1u]);
    
    s01.x = log_sigma;
    s01.y = log_sigma;
    s02.x = log_sigma;
    
    g.scale[0u] = pack2x16float(s01);
    g.scale[1u] = pack2x16float(s02);
    
    gaussians[idx] = g;
}

