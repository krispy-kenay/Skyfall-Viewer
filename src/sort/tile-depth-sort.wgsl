struct TileConfig {
    tile_size: u32,
    num_tiles_x: u32,
    num_tiles_y: u32,
    total_tiles: u32,
    canvas_width: u32,
    canvas_height: u32,
};

@group(0) @binding(1)
var<storage, read> splats: array<Splat>;

@group(0) @binding(2)
var<storage, read> sort_depths: array<u32>;

@group(1) @binding(0)
var<uniform> tile_config: TileConfig;

@group(2) @binding(0)
var<storage, read_write> tile_keys_hi: array<u32>;

@group(2) @binding(1)
var<storage, read_write> tile_keys_lo: array<u32>;

@group(2) @binding(2)
var<storage, read_write> tile_values: array<u32>;

@group(2) @binding(3)
var<storage, read_write> tile_ranges: array<vec2<u32>>;

@group(3) @binding(0)
var<storage, read> tile_touched: array<u32>;

@group(3) @binding(1)
var<storage, read_write> tile_write_counter: atomic<u32>;

fn get_rect_min_max(center_ndc: vec2<f32>, radius: i32) -> vec4<u32> {
    let canvas_size = vec2<f32>(f32(tile_config.canvas_width), f32(tile_config.canvas_height));
    let center_screen = (center_ndc + 1.0) * canvas_size * 0.5;
    
    let min_x = max(0i, i32(center_screen.x) - radius) / i32(tile_config.tile_size);
    let min_y = max(0i, i32(center_screen.y) - radius) / i32(tile_config.tile_size);
    let max_x = (i32(center_screen.x) + radius + i32(tile_config.tile_size) - 1) / i32(tile_config.tile_size);
    let max_y = (i32(center_screen.y) + radius + i32(tile_config.tile_size) - 1) / i32(tile_config.tile_size);
    
    return vec4<u32>(
        u32(clamp(min_x, 0i, i32(tile_config.num_tiles_x))),
        u32(clamp(min_y, 0i, i32(tile_config.num_tiles_y))),
        u32(clamp(max_x, 0i, i32(tile_config.num_tiles_x))),
        u32(clamp(max_y, 0i, i32(tile_config.num_tiles_y)))
    );
}

@compute @workgroup_size(256, 1, 1)
fn key_generation(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&splats) {
        return;
    }

    let splat = splats[idx];
    let center_ndc = unpack2x16float(splat.center);
    
    let conic_z_and_radius = unpack2x16float(splat.conic[1]);
    let radius = i32(conic_z_and_radius.y);

    if radius <= 0 {
        return;
    }

    let rect = get_rect_min_max(center_ndc, radius);
    
    let num_tiles = (rect.z - rect.x) * (rect.w - rect.y);
    if num_tiles == 0u {
        return;
    }

    let base_pos = atomicAdd(&tile_write_counter, num_tiles);
    var write_pos = base_pos;
    
    for (var y = rect.y; y < rect.w; y = y + 1u) {
        for (var x = rect.x; x < rect.z; x = x + 1u) {
            if write_pos >= arrayLength(&tile_keys_hi) {
                return;
            }
            
            let tile_id = y * tile_config.num_tiles_x + x;
            
            tile_keys_hi[write_pos] = tile_id;
            tile_keys_lo[write_pos] = sort_depths[idx];
            tile_values[write_pos] = idx;
            write_pos = write_pos + 1u;
        }
    }
}

@compute @workgroup_size(256, 1, 1)
fn identify_ranges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    
    let total_pairs = atomicLoad(&tile_write_counter);
    if idx >= total_pairs {
        return;
    }

    let tile_id = tile_keys_hi[idx];

    if idx == 0u {
        tile_ranges[tile_id].x = 0u;
    } else {
        let prev_tile_id = tile_keys_hi[idx - 1u];
        if tile_id != prev_tile_id {
            tile_ranges[prev_tile_id].y = idx;
            tile_ranges[tile_id].x = idx;
        }
    }

    if idx == total_pairs - 1u {
        tile_ranges[tile_id].y = total_pairs;
    }
}