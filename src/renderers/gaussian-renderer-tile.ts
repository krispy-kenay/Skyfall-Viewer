// Gaussian splatting renderer with tile-based depth sorting
// Complete separate pipeline for per-tile depth ordering

import { PointCloud } from '../utils/load';
import commonWSGL from '../shaders/common-tile.wgsl';
import preprocessTileWGSL from '../shaders/preprocess-tile.wgsl';
import renderWGSL from '../shaders/gaussian-tile.wgsl';
import tileDepthSortWGSL from '../sort/tile-depth-sort.wgsl';
import { get_sorter, c_histogram_block_rows, C } from '../sort/sort';
import { create_tile_depth_sorter } from '../sort/tile-depth-sort';
import { Renderer } from './renderer';
import type { Camera } from '../camera/camera';

export interface GaussianTileAwareRenderer extends Renderer {
    set_gaussian_multiplier: (v: number) => void;
    set_tile_size: (size: number) => void;
}

const createBuffer = (
    device: GPUDevice,
    label: string,
    size: number,
    usage: GPUBufferUsageFlags,
    data?: ArrayBuffer | ArrayBufferView
) => {
    const buffer = device.createBuffer({ label, size, usage });
    if (data) device.queue.writeBuffer(buffer, 0, data as ArrayBuffer);
    return buffer;
};

export default function get_renderer(
    pc: PointCloud,
    device: GPUDevice,
    presentation_format: GPUTextureFormat,
    camera_buffer: GPUBuffer,
    canvas: HTMLCanvasElement,
    camera: Camera,
): GaussianTileAwareRenderer {

    const sorter = get_sorter(pc.num_points, device);

    const nulling_data = new Uint32Array([0]);

    const draw_indirect_buffer = createBuffer(
        device,
        'draw indirect buffer tile',
        4 * Uint32Array.BYTES_PER_ELEMENT,
        GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        new Uint32Array([6, pc.num_points, 0, 0])
    );

    const SPLAT_SIZE = 24;
    const splat_buffer = createBuffer(
        device,
        'splats buffer tile',
        pc.num_points * SPLAT_SIZE,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );

    const settings_buffer = createBuffer(
        device,
        'render settings buffer tile',
        4 * Uint32Array.BYTES_PER_ELEMENT,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        new Float32Array([1.0, pc.sh_deg, 0.0, 0.0])
    );

    let tile_size = 32;
    
    const tile_config_buffer = createBuffer(
        device,
        'tile config',
        24,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        new Uint32Array([
            tile_size,
            Math.ceil(canvas.width / tile_size),
            Math.ceil(canvas.height / tile_size),
            Math.ceil(canvas.width / tile_size) * Math.ceil(canvas.height / tile_size),
            canvas.width,
            canvas.height
        ])
    );

    const tile_touched_buffer = createBuffer(
        device,
        'tile touched',
        pc.num_points * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );

    const max_tile_pairs = Math.min(pc.num_points * 8, 16 * 1024 * 1024);
    const tile_sorter = create_tile_depth_sorter(
        pc.num_points,
        max_tile_pairs,
        canvas.width,
        canvas.height,
        tile_size,
        device,
        commonWSGL + tileDepthSortWGSL
    );

    const preprocess_pipeline = device.createComputePipeline({
        label: 'preprocess tile',
        layout: 'auto',
        compute: {
            module: device.createShaderModule({ code: commonWSGL + preprocessTileWGSL }),
            entryPoint: 'preprocess',
            constants: {
                workgroupSize: C.histogram_wg_size,
                sortKeyPerThread: c_histogram_block_rows,
            },
        },
    });

    const sort_bind_group = device.createBindGroup({
        label: 'sort tile',
        layout: preprocess_pipeline.getBindGroupLayout(2),
        entries: [
            { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
            { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
            { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
            { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
        ],
    });

    const tile_bind_group = device.createBindGroup({
        label: 'tile config bind',
        layout: preprocess_pipeline.getBindGroupLayout(3),
        entries: [
            { binding: 0, resource: { buffer: tile_config_buffer } },
            { binding: 1, resource: { buffer: tile_touched_buffer } },
        ],
    });

    const preprocess_bind_group0 = device.createBindGroup({
        label: 'preprocess 0 tile',
        layout: preprocess_pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: camera_buffer } },
            { binding: 1, resource: { buffer: pc.gaussian_3d_buffer } },
            { binding: 2, resource: { buffer: pc.sh_buffer } },
        ],
    });

    const preprocess_bind_group1 = device.createBindGroup({
        label: 'preprocess 1 tile',
        layout: preprocess_pipeline.getBindGroupLayout(1),
        entries: [
            { binding: 0, resource: { buffer: splat_buffer } },
            { binding: 1, resource: { buffer: settings_buffer } },
        ],
    });

    const render_shader = device.createShaderModule({ code: commonWSGL + renderWGSL });
    const render_pipeline = device.createRenderPipeline({
        label: 'gaussian render tile',
        layout: 'auto',
        vertex: {
            module: render_shader,
            entryPoint: 'vs_main',
        },
        fragment: {
            module: render_shader,
            entryPoint: 'fs_main',
            targets: [{
                format: presentation_format,
                blend: {
                    color: {
                        srcFactor: 'one',
                        dstFactor: 'one-minus-src-alpha',
                        operation: 'add',
                    },
                    alpha: {
                        srcFactor: 'one',
                        dstFactor: 'one-minus-src-alpha',
                        operation: 'add',
                    },
                },
                writeMask: GPUColorWrite.ALL,
            }],
        },
        primitive: {
            topology: 'triangle-list',
        },
    });

    const render_bind_group = device.createBindGroup({
        label: 'gaussian splats tile',
        layout: render_pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: splat_buffer } },
            { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
        ],
    });

    function record_preprocess(encoder: GPUCommandEncoder) {
        device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
        device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);

        const pass = encoder.beginComputePass({ label: 'preprocess tile' });
        pass.setPipeline(preprocess_pipeline);
        pass.setBindGroup(0, preprocess_bind_group0);
        pass.setBindGroup(1, preprocess_bind_group1);
        pass.setBindGroup(2, sort_bind_group);
        pass.setBindGroup(3, tile_bind_group);
        const wg = Math.ceil(pc.num_points / C.histogram_wg_size);
        pass.dispatchWorkgroups(wg);
        pass.end();
    }

    function record_tile_sort(encoder: GPUCommandEncoder) {
        tile_sorter.key_generation_pass(
            encoder,
            camera_buffer,
            splat_buffer,
            sorter.ping_pong[0].sort_depths_buffer,
            tile_sorter.tile_offsets_buffer,
            tile_touched_buffer
        );
        sorter.sort(encoder);
        tile_sorter.identify_ranges_pass(encoder);
    }

    function record_render(encoder: GPUCommandEncoder, texture_view: GPUTextureView) {
        const pass = encoder.beginRenderPass({
            label: 'gaussian render tile',
            colorAttachments: [
                {
                    view: texture_view,
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
        });
        pass.setPipeline(render_pipeline);
        pass.setBindGroup(0, render_bind_group);
        pass.drawIndirect(draw_indirect_buffer, 0);
        pass.end();
    }


    return {
        frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
            record_preprocess(encoder);
            record_tile_sort(encoder);
            encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, draw_indirect_buffer, 4, 4);
            record_render(encoder, texture_view);
        },
        camera_buffer,
        set_gaussian_multiplier(v: number) {
            device.queue.writeBuffer(settings_buffer, 0, new Float32Array([v, pc.sh_deg, 0.0, 0.0]));
        },
        set_tile_size(size: number) {
            tile_size = size;
            const num_tiles_x = Math.ceil(canvas.width / size);
            const num_tiles_y = Math.ceil(canvas.height / size);
            device.queue.writeBuffer(
                tile_config_buffer,
                0,
                new Uint32Array([size, num_tiles_x, num_tiles_y, num_tiles_x * num_tiles_y, canvas.width, canvas.height])
            );
            console.log(`[TileRenderer] Set tile_size=${size}`);
        },
    };
}