// Tile-based depth sorting pipeline
// Generates tile-sorted indices for per-tile Gaussian rendering

export interface TileDepthSorter {
    tile_size: number;
    num_tiles_x: number;
    num_tiles_y: number;
    max_tile_pairs: number;

    tile_config_buffer: GPUBuffer;
    tile_offsets_buffer: GPUBuffer;
    tile_keys_hi_buffer: GPUBuffer;
    tile_keys_lo_buffer: GPUBuffer;
    tile_values_buffer: GPUBuffer;
    tile_ranges_buffer: GPUBuffer;
    tile_write_counter_buffer: GPUBuffer;

    key_generation_pipeline: GPUComputePipeline;
    key_generation_bind_group_config: GPUBindGroup;
    key_generation_bind_group_output: GPUBindGroup;

    identify_ranges_pipeline: GPUComputePipeline;

    key_generation_pass: (encoder: GPUCommandEncoder, camera: GPUBuffer, splat_data: GPUBuffer, sort_depths: GPUBuffer, tile_offsets: GPUBuffer, tile_touched: GPUBuffer) => void;
    identify_ranges_pass: (encoder: GPUCommandEncoder) => void;
}

const createBuffer = (
    device: GPUDevice,
    label: string,
    size: number,
    usage: GPUBufferUsageFlags,
    data?: ArrayBuffer | ArrayBufferView
): GPUBuffer => {
    const buffer = device.createBuffer({ label, size, usage });
    if (data) device.queue.writeBuffer(buffer, 0, data as ArrayBuffer);
    return buffer;
};

export function create_tile_depth_sorter(
    num_gaussians: number,
    max_tile_pairs: number,
    canvas_width: number,
    canvas_height: number,
    tile_size: number,
    device: GPUDevice,
    shader_code: string
): TileDepthSorter {
    const num_tiles_x = Math.ceil(canvas_width / tile_size);
    const num_tiles_y = Math.ceil(canvas_height / tile_size);

    console.log(`[TileDepthSort] Initialized: ${num_gaussians} gaussians, tile_size=${tile_size}, grid=${num_tiles_x}x${num_tiles_y}, max_pairs=${max_tile_pairs}`);

    const tile_config_buffer = createBuffer(
        device,
        'tile config',
        24,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        new Uint32Array([tile_size, num_tiles_x, num_tiles_y, num_tiles_x * num_tiles_y, canvas_width, canvas_height])
    );

    const tile_offsets_buffer = createBuffer(
        device,
        'tile offsets',
        num_gaussians * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );

    const tile_keys_hi_buffer = createBuffer(
        device,
        'tile keys hi',
        max_tile_pairs * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );

    const tile_keys_lo_buffer = createBuffer(
        device,
        'tile keys lo',
        max_tile_pairs * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );

    const tile_values_buffer = createBuffer(
        device,
        'tile values',
        max_tile_pairs * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );

    const tile_ranges_buffer = createBuffer(
        device,
        'tile ranges',
        num_tiles_x * num_tiles_y * 8,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        new Uint32Array(num_tiles_x * num_tiles_y * 2)
    );

    const tile_write_counter_buffer = createBuffer(
        device,
        'tile write counter',
        4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        new Uint32Array([0])
    );

    const shader_module = device.createShaderModule({ code: shader_code });

    const key_generation_pipeline = device.createComputePipeline({
        label: 'tile key generation',
        layout: 'auto',
        compute: {
            module: shader_module,
            entryPoint: 'key_generation',
        },
    });

    const identify_ranges_pipeline = device.createComputePipeline({
        label: 'tile identify ranges',
        layout: 'auto',
        compute: {
            module: shader_module,
            entryPoint: 'identify_ranges',
        },
    });

    const key_generation_bind_group_config = device.createBindGroup({
        label: 'tile key generation config',
        layout: key_generation_pipeline.getBindGroupLayout(1),
        entries: [
            { binding: 0, resource: { buffer: tile_config_buffer } },
        ],
    });

    const key_generation_bind_group_output = device.createBindGroup({
        label: 'tile key generation output',
        layout: key_generation_pipeline.getBindGroupLayout(2),
        entries: [
            { binding: 0, resource: { buffer: tile_keys_hi_buffer } },
            { binding: 1, resource: { buffer: tile_keys_lo_buffer } },
            { binding: 2, resource: { buffer: tile_values_buffer } },
        ],
    });

    function key_generation_pass(
        encoder: GPUCommandEncoder,
        camera_buffer: GPUBuffer,
        splat_data: GPUBuffer,
        sort_depths_buffer: GPUBuffer,
        tile_offsets_buffer: GPUBuffer,
        tile_touched_buffer: GPUBuffer
    ) {
        device.queue.writeBuffer(tile_write_counter_buffer, 0, new Uint32Array([0]));

        const bg_splats = device.createBindGroup({
            label: 'tile key generation splats (frame)',
            layout: key_generation_pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 1, resource: { buffer: splat_data } },
                { binding: 2, resource: { buffer: sort_depths_buffer } },
            ],
        });

        const bg_counters = device.createBindGroup({
            label: 'tile key generation counters (frame)',
            layout: key_generation_pipeline.getBindGroupLayout(3),
            entries: [
                { binding: 1, resource: { buffer: tile_write_counter_buffer } },
            ],
        });

        const pass = encoder.beginComputePass({ label: 'tile key generation' });
        pass.setPipeline(key_generation_pipeline);
        pass.setBindGroup(0, bg_splats);
        pass.setBindGroup(1, key_generation_bind_group_config);
        pass.setBindGroup(2, key_generation_bind_group_output);
        pass.setBindGroup(3, bg_counters);
        const wg = Math.ceil(num_gaussians / 256);
        pass.dispatchWorkgroups(wg);
        pass.end();
    }

    function identify_ranges_pass(encoder: GPUCommandEncoder) {
        device.queue.writeBuffer(tile_ranges_buffer, 0, new Uint32Array(num_tiles_x * num_tiles_y * 2));

        const bg_keys = device.createBindGroup({
            label: 'tile identify ranges keys (frame)',
            layout: identify_ranges_pipeline.getBindGroupLayout(2),
            entries: [
                { binding: 0, resource: { buffer: tile_keys_hi_buffer } },
                { binding: 3, resource: { buffer: tile_ranges_buffer } },
            ],
        });

        const bg_counters = device.createBindGroup({
            label: 'tile identify ranges counters (frame)',
            layout: identify_ranges_pipeline.getBindGroupLayout(3),
            entries: [
                { binding: 1, resource: { buffer: tile_write_counter_buffer } },
            ],
        });

        const pass = encoder.beginComputePass({ label: 'tile identify ranges' });
        pass.setPipeline(identify_ranges_pipeline);
        pass.setBindGroup(2, bg_keys);
        pass.setBindGroup(3, bg_counters);
        const wg = Math.min(Math.ceil(max_tile_pairs / 256), 65535);
        pass.dispatchWorkgroups(wg);
        pass.end();
    }

    return {
        tile_size,
        num_tiles_x,
        num_tiles_y,
        max_tile_pairs,

        tile_config_buffer,
        tile_offsets_buffer,
        tile_keys_hi_buffer,
        tile_keys_lo_buffer,
        tile_values_buffer,
        tile_ranges_buffer,
        tile_write_counter_buffer,

        key_generation_pipeline,
        key_generation_bind_group_config,
        key_generation_bind_group_output,

        identify_ranges_pipeline,

        key_generation_pass,
        identify_ranges_pass,
    };
}