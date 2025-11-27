import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import commonWGSL from '../shaders/common.wgsl';
import optimizerWGSL from '../shaders/optimizer.wgsl';
import knnInitWGSL from '../shaders/knn-init.wgsl';
import pruneWGSL from '../shaders/prune.wgsl';
import densifyWGSL from '../shaders/densify.wgsl';
import trainingForwardWGSL from '../shaders/training-forward.wgsl';
import trainingLossWGSL from '../shaders/training-loss.wgsl';
import { get_sorter, C } from '../sort/sort';
import { Renderer } from './renderer';
import { Camera, TrainingCameraData, create_training_camera_uniform_buffer, update_training_camera_uniform_buffer } from '../camera/camera';

export interface GaussianRenderer extends Renderer {
  setGaussianScale(value: number): void;
  resizeViewport(w: number, h: number): void;
  train(numFrames: number): void;
  setTargetTexture(view: GPUTextureView, sampler: GPUSampler): void;
  initializeKnn(encoder: GPUCommandEncoder): void;
  prune(): Promise<number>;
  densify(): Promise<number>;
  setTrainingCameras(cameras: TrainingCameraData[]): void;
  setTrainingImages(images: Array<{ view: GPUTextureView; sampler: GPUSampler }>): void;
}

// ===============================================
// Constants
// ===============================================
const BYTES_PER_SPLAT = 48;
const BYTES_PER_GAUSSIAN = 24;
const BYTES_PER_SH = 96;
const WORKGROUP_SIZE = 256;
const PRUNE_INTERVAL = 70;
const DENSIFY_INTERVAL = 100;

// Float32 training parameter layout
const FLOAT32_BYTES = 4;
const TRAIN_POS_COMPONENTS = 3;
const TRAIN_SCALE_COMPONENTS = 3;
const TRAIN_ROT_COMPONENTS = 4;
const TRAIN_OPACITY_COMPONENTS = 1;
const TRAIN_MAX_SH_COEFFS = 16;
const TRAIN_SH_COMPONENTS = TRAIN_MAX_SH_COEFFS * 3;

const nextPow2 = (n: number) => 1 << Math.ceil(Math.log2(Math.max(1, n)));

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data as BufferSource);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
  canvas: HTMLCanvasElement,
  camera: Camera
): GaussianRenderer {

  // ===============================================
  // State Variables
  // ===============================================
  let trainingCount = 0;
  let trainingIteration = 0;
  let capacity = 0;
  let active_count = 0;
  let pc_data: PointCloud = pc;

  // GPU Buffers
  let splat_buffer: GPUBuffer | null = null;
  let density_buffer: GPUBuffer | null = null;
  let gaussian_buffer: GPUBuffer | null = null;
  let sh_buffer: GPUBuffer | null = null;
  let gaussian_buffer_temp: GPUBuffer | null = null;
  let sh_buffer_temp: GPUBuffer | null = null;
  let density_buffer_temp: GPUBuffer | null = null;
  let active_count_buffer: GPUBuffer | null = null;
  let active_count_uniform_buffer: GPUBuffer | null = null;
  let knn_num_points_buffer: GPUBuffer | null = null;
  let prune_threshold_buffer: GPUBuffer | null = null;
  let prune_input_count_buffer: GPUBuffer | null = null;
  let densify_split_threshold_buffer: GPUBuffer | null = null;
  let densify_clone_threshold_buffer: GPUBuffer | null = null;

  // Training forward render targets
  let training_color_texture: GPUTexture | null = null;
  let training_alpha_texture: GPUTexture | null = null;
  let training_color_view: GPUTextureView | null = null;
  let training_alpha_view: GPUTextureView | null = null;
  let training_forward_bind_group: GPUBindGroup | null = null;
  let training_width = 0;
  let training_height = 0;

  // Training residuals (loss / backward)
  let residual_color_texture: GPUTexture | null = null;
  let residual_alpha_texture: GPUTexture | null = null;
  let residual_color_view: GPUTextureView | null = null;
  let residual_alpha_view: GPUTextureView | null = null;
  let training_loss_bind_group: GPUBindGroup | null = null;

  // Training parameter buffers (float32)
  let train_position_buffer: GPUBuffer | null = null;
  let train_scale_buffer: GPUBuffer | null = null;
  let train_rotation_buffer: GPUBuffer | null = null;
  let train_opacity_buffer: GPUBuffer | null = null;
  let train_sh_buffer: GPUBuffer | null = null;

  // Training gradient buffers (float32)
  let grad_position_buffer: GPUBuffer | null = null;
  let grad_scale_buffer: GPUBuffer | null = null;
  let grad_rotation_buffer: GPUBuffer | null = null;
  let grad_opacity_buffer: GPUBuffer | null = null;
  let grad_sh_buffer: GPUBuffer | null = null;

  // Bind Groups
  let sort_bind_group: GPUBindGroup | null = null;
  let sorted_bg: GPUBindGroup[] = [];
  let render_splat_bg: GPUBindGroup | null = null;
  let preprocess_bg: GPUBindGroup | null = null;
  let optimize_bind_group: GPUBindGroup | null = null;
  let knn_init_bind_group: GPUBindGroup | null = null;
  let prune_bind_group: GPUBindGroup | null = null;
  let densify_bind_group: GPUBindGroup | null = null;

  // External resources
  let targetTextureView: GPUTextureView | null = null;
  let targetSampler: GPUSampler | null = null;
  let sorter: any = null;

  // Training cameras
  let training_cameras: TrainingCameraData[] = [];
  let training_camera_buffers: GPUBuffer[] = [];
  let current_training_camera_index = 0;
  
  // Training images
  let training_images: Array<{ view: GPUTextureView; sampler: GPUSampler }> = [];

  // ===============================================
  // Buffer Creation Functions
  // ===============================================
  function createSettingsBuffer(): GPUBuffer {
    return createBuffer(
      device,
      'gaussian settings',
      32,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      new Float32Array([1.0, pc_data.sh_deg, canvas.width, canvas.height, 1.0, 1.0])
    );
  }

  function createIndirectArgsBuffer(): GPUBuffer {
    return createBuffer(
      device,
      'gaussian indirect args',
      4 * 4,
      GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
      new Uint32Array([6, 0, 0, 0])
    );
  }

  function createOptimizeSettingsBuffer(): GPUBuffer {
    const scene_extent = pc_data.scene_extent || 100.0;
    const typical_mean_dist = scene_extent * 0.05;
    const typical_density_target_pixels = (typical_mean_dist / 3.0) / 0.003;
    const min_size_scaled = 0.1;
    const max_size_scaled = typical_density_target_pixels * 8.0;

    return createBuffer(
      device,
      'optimizer settings',
      4 * 8,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      new Float32Array([
        0.1,   // learning_rate_color
        0.02,   // learning_rate_alpha
        0.02,   // min_alpha
        0.98,   // max_alpha
        0.05,   // learning_rate_size
        min_size_scaled,
        max_size_scaled,
        0.05    // target_error
      ])
    );
  }

  function createSplatBuffer(size: number): GPUBuffer {
    return device.createBuffer({
      label: 'gaussian splat',
      size: size * BYTES_PER_SPLAT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
  }

  function createDensityBuffer(size: number): GPUBuffer {
    return device.createBuffer({
      label: 'gaussian density',
      size: size * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
  }

  function createGaussianBuffer(size: number): GPUBuffer {
    return device.createBuffer({
      label: 'gaussian 3d (renderer-owned)',
      size: size * BYTES_PER_GAUSSIAN,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
  }

  function createSHBuffer(size: number): GPUBuffer {
    return device.createBuffer({
      label: 'sh coefficients (renderer-owned)',
      size: size * BYTES_PER_SH,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
  }

  function createActiveCountBuffers(): { atomic: GPUBuffer; uniform: GPUBuffer } {
    return {
      atomic: device.createBuffer({
        label: 'active count (atomic)',
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      }),
      uniform: device.createBuffer({
        label: 'active count (uniform)',
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      })
    };
  }

  function createPruneThresholdBuffer(): GPUBuffer {
    return createBuffer(
      device,
      'prune threshold',
      4,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      new Float32Array([0.01])
    );
  }

  function createPruneInputCountBuffer(): GPUBuffer {
    return createBuffer(
      device,
      'prune input count',
      4,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      new Uint32Array([0])
    );
  }

  function createDensifyThresholdBuffers(): { split: GPUBuffer; clone: GPUBuffer } {
    return {
      split: createBuffer(
        device,
        'split threshold',
        4,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        new Float32Array([1.6])
      ),
      clone: createBuffer(
        device,
        'clone threshold',
        4,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        new Float32Array([0.1])
      )
    };
  }

  function createKnnNumPointsBuffer(): GPUBuffer {
    return createBuffer(
      device,
      'knn num points',
      4,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      new Uint32Array([active_count])
    );
  }

  // ===============================================
  // Buffer Recreation Functions
  // ===============================================
  function recreateBuffers(newCapacity: number, preserveData: boolean) {
    const old_gaussian = gaussian_buffer;
    const old_sh = sh_buffer;
    const old_density = density_buffer;

    // Create new buffers
    splat_buffer = createSplatBuffer(newCapacity);
    density_buffer = createDensityBuffer(newCapacity);
    gaussian_buffer = createGaussianBuffer(newCapacity);
    sh_buffer = createSHBuffer(newCapacity);
    gaussian_buffer_temp = createGaussianBuffer(newCapacity);
    sh_buffer_temp = createSHBuffer(newCapacity);
    density_buffer_temp = createDensityBuffer(newCapacity);

    // create float32 training parameter buffers
    train_position_buffer = device.createBuffer({
      label: 'train positions (float32)',
      size: newCapacity * TRAIN_POS_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    train_scale_buffer = device.createBuffer({
      label: 'train scales (float32)',
      size: newCapacity * TRAIN_SCALE_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    train_rotation_buffer = device.createBuffer({
      label: 'train rotations (float32)',
      size: newCapacity * TRAIN_ROT_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    train_opacity_buffer = device.createBuffer({
      label: 'train opacities (float32)',
      size: newCapacity * TRAIN_OPACITY_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    train_sh_buffer = device.createBuffer({
      label: 'train SH (float32)',
      size: newCapacity * TRAIN_SH_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // (Re)create float32 training gradient buffers
    grad_position_buffer = device.createBuffer({
      label: 'grad positions (float32)',
      size: newCapacity * TRAIN_POS_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    grad_scale_buffer = device.createBuffer({
      label: 'grad scales (float32)',
      size: newCapacity * TRAIN_SCALE_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    grad_rotation_buffer = device.createBuffer({
      label: 'grad rotations (float32)',
      size: newCapacity * TRAIN_ROT_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    grad_opacity_buffer = device.createBuffer({
      label: 'grad opacities (float32)',
      size: newCapacity * TRAIN_OPACITY_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    grad_sh_buffer = device.createBuffer({
      label: 'grad SH (float32)',
      size: newCapacity * TRAIN_SH_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Create active count buffers if they don't exist
    if (!active_count_buffer || !active_count_uniform_buffer) {
      const counts = createActiveCountBuffers();
      active_count_buffer = counts.atomic;
      active_count_uniform_buffer = counts.uniform;
    }

    // Copy old data if needed
    if (preserveData && old_gaussian && old_sh && active_count > 0) {
      const encoder = device.createCommandEncoder();
      const copy_size = Math.min(active_count, newCapacity);
      encoder.copyBufferToBuffer(old_gaussian, 0, gaussian_buffer!, 0, copy_size * BYTES_PER_GAUSSIAN);
      encoder.copyBufferToBuffer(old_sh, 0, sh_buffer!, 0, copy_size * BYTES_PER_SH);
      if (old_density) {
        encoder.copyBufferToBuffer(old_density, 0, density_buffer!, 0, copy_size * 4);
      }
      device.queue.submit([encoder.finish()]);
    }

    capacity = newCapacity;
  }

  // ===============================================
  // Setter Functions
  // ===============================================
  const settings_buffer = createSettingsBuffer();
  const indirect_args = createIndirectArgsBuffer();
  const optimize_settings_buffer = createOptimizeSettingsBuffer();

  function setGaussianScale(value: number) {
    device.queue.writeBuffer(settings_buffer, 0, new Float32Array([value]));
  }

  function resizeViewport(w: number, h: number) {
    device.queue.writeBuffer(settings_buffer, 2 * 4, new Float32Array([w, h]));
    ensureTrainingRenderTargets(w, h);
    rebuildTrainingForwardBG();
  }

  function setActiveCount(value: number) {
    active_count = value;
    if (active_count_uniform_buffer) {
      device.queue.writeBuffer(active_count_uniform_buffer, 0, new Uint32Array([value]));
    }
    if (indirect_args) {
      device.queue.writeBuffer(indirect_args, 0, new Uint32Array([6, value, 0, 0]));
    }
  }

  function setPruneThreshold(value: number) {
    if (!prune_threshold_buffer) {
      prune_threshold_buffer = createPruneThresholdBuffer();
      rebuildPruneBG();
    }
    device.queue.writeBuffer(prune_threshold_buffer, 0, new Float32Array([value]));
  }

  function setDensifySplitThreshold(value: number) {
    if (!densify_split_threshold_buffer) {
      const buffers = createDensifyThresholdBuffers();
      densify_split_threshold_buffer = buffers.split;
      densify_clone_threshold_buffer = buffers.clone;
      rebuildDensifyBG();
    }
    device.queue.writeBuffer(densify_split_threshold_buffer, 0, new Float32Array([value]));
  }

  function setDensifyCloneThreshold(value: number) {
    if (!densify_clone_threshold_buffer) {
      const buffers = createDensifyThresholdBuffers();
      densify_split_threshold_buffer = buffers.split;
      densify_clone_threshold_buffer = buffers.clone;
      rebuildDensifyBG();
    }
    device.queue.writeBuffer(densify_clone_threshold_buffer, 0, new Float32Array([value]));
  }

  function setTargetTexture(view: GPUTextureView, sampler: GPUSampler) {
    targetTextureView = view;
    targetSampler = sampler;
    rebuildOptimizerBG();
    rebuildDensifyBG();
  }

  // ===============================================
  // Pipeline Creation
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: commonWGSL + '\n' + preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: WORKGROUP_SIZE,
        sortKeyPerThread: (C.histogram_wg_size * C.rs_histogram_block_rows) / WORKGROUP_SIZE,
      },
    },
  });

  const optimizer_pipeline = device.createComputePipeline({
    label: 'optimizer',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: commonWGSL + '\n' + optimizerWGSL }),
      entryPoint: 'optimizer',
    },
  });

  const knn_init_pipeline = device.createComputePipeline({
    label: 'knn-init',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: commonWGSL + '\n' + knnInitWGSL }),
      entryPoint: 'knn_init',
    },
  });

  const prune_pipeline = device.createComputePipeline({
    label: 'prune',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: commonWGSL + '\n' + pruneWGSL }),
      entryPoint: 'prune',
    },
  });

  const densify_pipeline = device.createComputePipeline({
    label: 'densify',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: commonWGSL + '\n' + densifyWGSL }),
      entryPoint: 'densify',
    },
  });

  const render_shader = device.createShaderModule({ code: commonWGSL + '\n' + renderWGSL });

  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian render',
    layout: 'auto',
    vertex: { module: render_shader, entryPoint: 'vs_main' },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{
        format: presentation_format,
        blend: {
          color : { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha : { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
        writeMask: GPUColorWrite.ALL
      }],
    },
    primitive: { topology: 'triangle-list' },
  });

  const training_forward_pipeline = device.createComputePipeline({
    label: 'training forward',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: commonWGSL + '\n' + trainingForwardWGSL }),
      entryPoint: 'training_forward',
    },
  });

  const training_loss_pipeline = device.createComputePipeline({
    label: 'training loss',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: trainingLossWGSL }),
      entryPoint: 'training_loss',
    },
  });

  // ===============================================
  // Bind Group Creation Functions
  // ===============================================
  const preprocess_bg0 = device.createBindGroup({
    label: 'preprocess g0',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: settings_buffer } },
    ],
  });

  const render_settings_bg = device.createBindGroup({
    label: 'gaussian settings',
    layout: render_pipeline.getBindGroupLayout(3),
    entries: [{ binding: 0, resource: { buffer: settings_buffer } }],
  });

  function ensureTrainingRenderTargets(width: number, height: number) {
    if (training_color_texture && training_alpha_texture &&
        residual_color_texture && residual_alpha_texture &&
        training_width === width && training_height === height) {
      return;
    }

    training_color_texture?.destroy();
    training_alpha_texture?.destroy();
    residual_color_texture?.destroy();
    residual_alpha_texture?.destroy();

    training_width = width;
    training_height = height;

    training_color_texture = device.createTexture({
      label: 'training color (rgba32float)',
      size: { width, height },
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    training_alpha_texture = device.createTexture({
      label: 'training alpha (r32float)',
      size: { width, height },
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    training_color_view = training_color_texture.createView();
    training_alpha_view = training_alpha_texture.createView();

    residual_color_texture = device.createTexture({
      label: 'training residual color (rgba32float)',
      size: { width, height },
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    residual_alpha_texture = device.createTexture({
      label: 'training residual alpha (r32float)',
      size: { width, height },
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    residual_color_view = residual_color_texture.createView();
    residual_alpha_view = residual_alpha_texture.createView();
  }

  function rebuildSorterBG() {
    if (!sorter) return;
    sort_bind_group = device.createBindGroup({
      label: 'sort',
      layout: preprocess_pipeline.getBindGroupLayout(2),
      entries: [
        { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
        { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
        { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
        { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
      ],
    });
  }

  function rebuildSortedBG() {
    if (!sorter) return;
    const layout = render_pipeline.getBindGroupLayout(2);
    sorted_bg = [
      device.createBindGroup({
        label: 'sorted indices A',
        layout,
        entries: [{ binding: 0, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } }],
      }),
      device.createBindGroup({
        label: 'sorted indices B',
        layout,
        entries: [{ binding: 0, resource: { buffer: sorter.ping_pong[1].sort_indices_buffer } }],
      }),
    ];
  }

  function rebuildRenderSplatBG() {
    if (!splat_buffer) return;
    render_splat_bg = device.createBindGroup({
      label: 'gaussian splats',
      layout: render_pipeline.getBindGroupLayout(1),
      entries: [{ binding: 0, resource: { buffer: splat_buffer } }],
    });
  }

  function rebuildPreprocessBG() {
    if (!gaussian_buffer || !sh_buffer || !splat_buffer) return;
    preprocess_bg = device.createBindGroup({
      label: 'preprocess g1',
      layout: preprocess_pipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: gaussian_buffer } },
        { binding: 1, resource: { buffer: sh_buffer } },
        { binding: 2, resource: { buffer: splat_buffer } },
      ],
    });
  }

  function rebuildOptimizerBG() {
    if (!gaussian_buffer || !sh_buffer || !splat_buffer || !density_buffer ||
        !active_count_uniform_buffer || !optimize_settings_buffer) {
      return;
    }

    const camera_buffer_to_use = training_camera_buffers.length > 0
      ? training_camera_buffers[current_training_camera_index]
      : camera_buffer;
    
    let texture_view_to_use: GPUTextureView;
    let sampler_to_use: GPUSampler;
    
    if (training_images.length > 0 && current_training_camera_index < training_images.length) {
      texture_view_to_use = training_images[current_training_camera_index].view;
      sampler_to_use = training_images[current_training_camera_index].sampler;
    } else if (targetTextureView && targetSampler) {
      texture_view_to_use = targetTextureView;
      sampler_to_use = targetSampler;
    } else {
      return;
    }
    
    optimize_bind_group = device.createBindGroup({
      label: 'optimizer bind group',
      layout: optimizer_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gaussian_buffer } },
        { binding: 1, resource: { buffer: optimize_settings_buffer } },
        { binding: 2, resource: sampler_to_use },
        { binding: 3, resource: texture_view_to_use },
        { binding: 4, resource: { buffer: splat_buffer } },
        { binding: 5, resource: { buffer: sh_buffer } },
        { binding: 6, resource: { buffer: camera_buffer } },
        // Does not work with training cameras due to missing R and T matrices in Skyfall-GS training dataset, using camera buffer as hacky workaround
        //{ binding: 6, resource: { buffer: camera_buffer_to_use } },
        { binding: 7, resource: { buffer: settings_buffer } },
        { binding: 8, resource: { buffer: density_buffer } },
        { binding: 9, resource: { buffer: active_count_uniform_buffer } },
      ]
    });
  }

  function rebuildKnnInitBG() {
    if (!gaussian_buffer || !density_buffer) return;
    if (knn_num_points_buffer) {
      knn_num_points_buffer.destroy();
    }
    knn_num_points_buffer = createKnnNumPointsBuffer();
    knn_init_bind_group = device.createBindGroup({
      label: 'knn init bind group',
      layout: knn_init_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gaussian_buffer } },
        { binding: 1, resource: { buffer: knn_num_points_buffer } },
        { binding: 2, resource: { buffer: density_buffer } },
      ]
    });
  }

  function rebuildPruneBG() {
    if (!gaussian_buffer || !sh_buffer || !density_buffer || !active_count_buffer ||
        !gaussian_buffer_temp || !sh_buffer_temp || !density_buffer_temp) {
      return;
    }
    if (!prune_threshold_buffer) {
      prune_threshold_buffer = createPruneThresholdBuffer();
    }
    if (!prune_input_count_buffer) {
      prune_input_count_buffer = createPruneInputCountBuffer();
    }
    prune_bind_group = device.createBindGroup({
      label: 'prune bind group',
      layout: prune_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gaussian_buffer } },
        { binding: 1, resource: { buffer: sh_buffer } },
        { binding: 2, resource: { buffer: density_buffer } },
        { binding: 3, resource: { buffer: gaussian_buffer_temp } },
        { binding: 4, resource: { buffer: sh_buffer_temp } },
        { binding: 5, resource: { buffer: density_buffer_temp } },
        { binding: 6, resource: { buffer: active_count_buffer } },
        { binding: 7, resource: { buffer: prune_threshold_buffer } },
        { binding: 8, resource: { buffer: prune_input_count_buffer } },
      ]
    });
  }

  function rebuildDensifyBG() {
    if (!gaussian_buffer || !sh_buffer || !density_buffer || !active_count_buffer ||
        !active_count_uniform_buffer || !targetTextureView || !targetSampler ||
        !gaussian_buffer_temp || !sh_buffer_temp || !density_buffer_temp) {
      return;
    }
    if (!densify_split_threshold_buffer || !densify_clone_threshold_buffer) {
      const buffers = createDensifyThresholdBuffers();
      densify_split_threshold_buffer = buffers.split;
      densify_clone_threshold_buffer = buffers.clone;
    }
    densify_bind_group = device.createBindGroup({
      label: 'densify bind group',
      layout: densify_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gaussian_buffer } },
        { binding: 1, resource: { buffer: sh_buffer } },
        { binding: 2, resource: { buffer: density_buffer } },
        { binding: 3, resource: { buffer: gaussian_buffer_temp } },
        { binding: 4, resource: { buffer: sh_buffer_temp } },
        { binding: 5, resource: { buffer: density_buffer_temp } },
        { binding: 6, resource: { buffer: active_count_buffer } },
        { binding: 7, resource: { buffer: active_count_uniform_buffer } },
        { binding: 8, resource: { buffer: densify_split_threshold_buffer } },
        { binding: 9, resource: { buffer: densify_clone_threshold_buffer } },
        { binding: 10, resource: { buffer: camera_buffer } },
        { binding: 11, resource: { buffer: settings_buffer } },
        { binding: 12, resource: targetSampler },
        { binding: 13, resource: targetTextureView },
      ]
    });
  }

  function rebuildTrainingForwardBG() {
    if (!splat_buffer || !sorter) return;
    ensureTrainingRenderTargets(canvas.width, canvas.height);

    training_forward_bind_group = device.createBindGroup({
      label: 'training forward bind group',
      layout: training_forward_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: settings_buffer } },
        { binding: 1, resource: { buffer: splat_buffer } },
        { binding: 2, resource: { buffer: sorter.ping_pong[sorter.final_out_index].sort_indices_buffer } },
        { binding: 3, resource: { buffer: indirect_args } },
        { binding: 4, resource: training_color_view! },
        { binding: 5, resource: training_alpha_view! },
      ],
    });
  }

  function rebuildTrainingLossBG() {
    if (!training_color_view || !training_alpha_view ||
        !residual_color_view || !residual_alpha_view) {
      return;
    }

    let texture_view_to_use: GPUTextureView;
    let sampler_to_use: GPUSampler;

    if (training_images.length > 0 && current_training_camera_index < training_images.length) {
      texture_view_to_use = training_images[current_training_camera_index].view;
      sampler_to_use = training_images[current_training_camera_index].sampler;
    } else if (targetTextureView && targetSampler) {
      texture_view_to_use = targetTextureView;
      sampler_to_use = targetSampler;
    } else {
      return;
    }

    training_loss_bind_group = device.createBindGroup({
      label: 'training loss bind group',
      layout: training_loss_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sampler_to_use },
        { binding: 1, resource: texture_view_to_use },
        { binding: 2, resource: training_color_view },
        { binding: 3, resource: training_alpha_view },
        { binding: 4, resource: residual_color_view },
        { binding: 5, resource: residual_alpha_view },
      ],
    });
  }

  function rebuildAllBindGroups() {
    rebuildSorterBG();
    rebuildSortedBG();
    rebuildRenderSplatBG();
    rebuildPreprocessBG();
    rebuildOptimizerBG();
    rebuildKnnInitBG();
    rebuildPruneBG();
    rebuildDensifyBG();
    rebuildTrainingForwardBG();
    rebuildTrainingLossBG();
  }

  // ===============================================
  // Capacity Management
  // ===============================================
  function ensureCapacity(requested: number) {
    const want = nextPow2(requested);
    if (want <= capacity) {
      return;
    }
    recreateBuffers(want, capacity > 0);
    sorter = get_sorter(capacity, device);
    rebuildAllBindGroups();
    setActiveCount(active_count);
  }

  function resizeToActiveCount() {
    const target_capacity = nextPow2(Math.max(active_count, 1024));
    if (target_capacity < capacity) {
      recreateBuffers(target_capacity, true);
      sorter = get_sorter(target_capacity, device);
      rebuildAllBindGroups();
    }
  }

  // ===============================================
  // Point Cloud Loading
  // ===============================================
  function reloadPointCloud(new_pc: PointCloud) {
    pc_data = new_pc;
    setActiveCount(new_pc.num_points);
    ensureCapacity(new_pc.num_points);
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      new_pc.gaussian_3d_buffer, 0,
      gaussian_buffer!, 0,
      new_pc.num_points * BYTES_PER_GAUSSIAN
    );
    encoder.copyBufferToBuffer(
      new_pc.sh_buffer, 0,
      sh_buffer!, 0,
      new_pc.num_points * BYTES_PER_SH
    );
    device.queue.submit([encoder.finish()]);

    rebuildPreprocessBG();
  }

  // ===============================================
  // k-NN Initialization
  // ===============================================
  function initializeKnn(encoder: GPUCommandEncoder) {
    if (!knn_init_bind_group) {
      console.warn("k-NN init has no bind group");
      return;
    }
    console.log("Running k-NN initialization for density-aware scales...");
    const num = active_count;
    const num_wg = Math.ceil(num / WORKGROUP_SIZE);

    const pass = encoder.beginComputePass({ label: 'knn init' });
    pass.setPipeline(knn_init_pipeline);
    pass.setBindGroup(0, knn_init_bind_group);
    pass.dispatchWorkgroups(num_wg);
    pass.end();
  }

  // ===============================================
  // Pruning
  // ===============================================
  async function prune(): Promise<number> {
    if (!prune_bind_group || active_count === 0) {
      return active_count;
    }

    ensureCapacity(active_count);
    rebuildPruneBG();
    const safe_input_count = Math.min(active_count, capacity);
    device.queue.writeBuffer(prune_input_count_buffer!, 0, new Uint32Array([safe_input_count]));
    device.queue.writeBuffer(active_count_buffer!, 0, new Uint32Array([0]));

    if (safe_input_count === 0) {
      return active_count;
    }

    const encoder = device.createCommandEncoder();
    const num_wg = Math.ceil(safe_input_count / WORKGROUP_SIZE);
    const pass = encoder.beginComputePass({ label: 'prune' });
    pass.setPipeline(prune_pipeline);
    pass.setBindGroup(0, prune_bind_group);
    pass.dispatchWorkgroups(num_wg);
    pass.end();

    const staging_buffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    encoder.copyBufferToBuffer(active_count_buffer!, 0, staging_buffer, 0, 4);
    device.queue.submit([encoder.finish()]);

    await device.queue.onSubmittedWorkDone();
    await staging_buffer.mapAsync(GPUMapMode.READ);

    const mapped = staging_buffer.getMappedRange();
    const new_count = new Uint32Array(mapped)[0];
    staging_buffer.unmap();
    staging_buffer.destroy();

    const safe_count = Math.min(new_count, capacity);
    if (safe_count > 0) {
      const copy_encoder = device.createCommandEncoder();
      const gaussian_size = safe_count * BYTES_PER_GAUSSIAN;
      const sh_size = safe_count * BYTES_PER_SH;
      const density_size = safe_count * 4;

      copy_encoder.copyBufferToBuffer(gaussian_buffer_temp!, 0, gaussian_buffer!, 0, gaussian_size);
      copy_encoder.copyBufferToBuffer(sh_buffer_temp!, 0, sh_buffer!, 0, sh_size);
      copy_encoder.copyBufferToBuffer(density_buffer_temp!, 0, density_buffer!, 0, density_size);
      device.queue.submit([copy_encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    setActiveCount(safe_count);
    resizeToActiveCount();
    camera.needsPreprocess = true;

    if (safe_count > 0) {
      rebuildKnnInitBG();
      const knn_encoder = device.createCommandEncoder();
      initializeKnn(knn_encoder);
      device.queue.submit([knn_encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    return safe_count;
  }

  // ===============================================
  // Densification (Synchronous - blocks until complete)
  // ===============================================
  async function densify(): Promise<number> {
    if (!densify_bind_group || !targetTextureView || active_count === 0) {
      return active_count;
    }

    ensureCapacity(active_count * 2);
    rebuildDensifyBG();

    setActiveCount(active_count);
    device.queue.writeBuffer(active_count_buffer!, 0, new Uint32Array([0]));

    if (active_count === 0) {
      return active_count;
    }

    const encoder = device.createCommandEncoder();
    const num_wg = Math.ceil(active_count / WORKGROUP_SIZE);
    const pass = encoder.beginComputePass({ label: 'densify' });
    pass.setPipeline(densify_pipeline);
    pass.setBindGroup(0, densify_bind_group);
    pass.dispatchWorkgroups(num_wg);
    pass.end();

    const staging_buffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    encoder.copyBufferToBuffer(active_count_buffer!, 0, staging_buffer, 0, 4);
    device.queue.submit([encoder.finish()]);

    await device.queue.onSubmittedWorkDone();
    await staging_buffer.mapAsync(GPUMapMode.READ);

    const mapped = staging_buffer.getMappedRange();
    const num_new = new Uint32Array(mapped)[0];
    staging_buffer.unmap();
    staging_buffer.destroy();

    const max_new = capacity - active_count;
    const safe_num_new = Math.min(num_new, max_new);

    if (safe_num_new > 0) {
      const copy_encoder = device.createCommandEncoder();
      const gaussian_offset = active_count * BYTES_PER_GAUSSIAN;
      const sh_offset = active_count * BYTES_PER_SH;
      const density_offset = active_count * 4;
      const gaussian_size = safe_num_new * BYTES_PER_GAUSSIAN;
      const sh_size = safe_num_new * BYTES_PER_SH;
      const density_size = safe_num_new * 4;

      copy_encoder.copyBufferToBuffer(gaussian_buffer_temp!, 0, gaussian_buffer!, gaussian_offset, gaussian_size);
      copy_encoder.copyBufferToBuffer(sh_buffer_temp!, 0, sh_buffer!, sh_offset, sh_size);
        copy_encoder.copyBufferToBuffer(density_buffer_temp!, 0, density_buffer!, density_offset, density_size);
      device.queue.submit([copy_encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    const new_active_count = active_count + safe_num_new;
    setActiveCount(new_active_count);
    camera.needsPreprocess = true;

    if (new_active_count > 0) {
      rebuildKnnInitBG();
      const knn_encoder = device.createCommandEncoder();
      initializeKnn(knn_encoder);
      device.queue.submit([knn_encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    return new_active_count;
  }

  // ===============================================
  // Rendering Functions
  // ===============================================
  function render_pass(encoder: GPUCommandEncoder, view: GPUTextureView) {
    if (!render_splat_bg || !sorted_bg.length || !sorter) return;

    const pass = encoder.beginRenderPass({
      label: 'gaussian quad render',
      colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store' }],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(1, render_splat_bg);
    pass.setBindGroup(2, sorted_bg[sorter.final_out_index]);
    pass.setBindGroup(3, render_settings_bg);
    pass.drawIndirect(indirect_args, 0);
    pass.end();
  }

  function run_preprocess(encoder: GPUCommandEncoder) {
    if (!sorter || !preprocess_bg) return;

    const nulling_data = new Uint32Array([0]);
    device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);
    device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);

    const num = active_count;
    const num_wg = Math.ceil(num / WORKGROUP_SIZE);
    const pass = encoder.beginComputePass({ label: 'preprocess' });
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, preprocess_bg0);
    pass.setBindGroup(1, preprocess_bg);
    pass.setBindGroup(2, sort_bind_group!);
    pass.dispatchWorkgroups(num_wg);
    pass.end();
    encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_args, 4, 4);
  }

  function run_training_forward(encoder: GPUCommandEncoder) {
    if (!training_forward_bind_group || !splat_buffer || !sorter) return;

    // Ensure render targets match current viewport
    ensureTrainingRenderTargets(canvas.width, canvas.height);
    rebuildTrainingForwardBG();

    const wgSizeX = 8;
    const wgSizeY = 8;
    const numGroupsX = Math.ceil(canvas.width / wgSizeX);
    const numGroupsY = Math.ceil(canvas.height / wgSizeY);

    const pass = encoder.beginComputePass({ label: 'training forward' });
    pass.setPipeline(training_forward_pipeline);
    pass.setBindGroup(0, training_forward_bind_group);
    pass.dispatchWorkgroups(numGroupsX, numGroupsY);
    pass.end();
  }

  function run_training_loss(encoder: GPUCommandEncoder) {
    if (!training_loss_bind_group) return;

    ensureTrainingRenderTargets(canvas.width, canvas.height);
    rebuildTrainingLossBG();

    const wgSizeX = 8;
    const wgSizeY = 8;
    const numGroupsX = Math.ceil(canvas.width / wgSizeX);
    const numGroupsY = Math.ceil(canvas.height / wgSizeY);

    const pass = encoder.beginComputePass({ label: 'training loss' });
    pass.setPipeline(training_loss_pipeline);
    pass.setBindGroup(0, training_loss_bind_group);
    pass.dispatchWorkgroups(numGroupsX, numGroupsY);
    pass.end();
  }

  // ===============================================
  // Training
  // ===============================================
  function trainStep(encoder: GPUCommandEncoder) {
    if (!optimize_bind_group) {
      console.warn("Optimizer has no bind group");
      return;
    }
    
    if (training_camera_buffers.length > 0) {
      current_training_camera_index = trainingIteration % training_camera_buffers.length;
      rebuildOptimizerBG();
    }
    
    trainingIteration++;
    const cameraInfo = training_camera_buffers.length > 0 
      ? ` (camera ${current_training_camera_index + 1}/${training_camera_buffers.length}${training_images.length > 0 ? `, image ${current_training_camera_index + 1}/${training_images.length}` : ''})` 
      : '';
    console.log(`Training iteration ${trainingIteration}${cameraInfo}...`);
    setActiveCount(active_count);

    const num = active_count;
    const num_wg = Math.ceil(num / WORKGROUP_SIZE);
    const pass = encoder.beginComputePass({ label: 'optimize splats pass' });
    pass.setPipeline(optimizer_pipeline);
    pass.setBindGroup(0, optimize_bind_group);
    pass.dispatchWorkgroups(num_wg);
    pass.end();
  }

  function train(numFrames: number) {
    trainingCount += numFrames;
  }

  // ===============================================
  // Initialization
  // ===============================================
  reloadPointCloud(pc);

  // ===============================================
  // Training Camera Management
  // ===============================================
  function setTrainingCameras(cameras: TrainingCameraData[]) {
    for (const buffer of training_camera_buffers) {
      buffer.destroy();
    }
    
    training_cameras = cameras;
    training_camera_buffers = cameras.map(cam => 
      create_training_camera_uniform_buffer(device, cam)
    );
    
    current_training_camera_index = 0;
    rebuildOptimizerBG();
    
    console.log(`Loaded ${cameras.length} training cameras`);
  }

  function setTrainingImages(images: Array<{ view: GPUTextureView; sampler: GPUSampler }>) {
    training_images = images;
    rebuildOptimizerBG();
    
    console.log(`Loaded ${images.length} training images`);
    if (training_cameras.length > 0 && images.length !== training_cameras.length) {
      console.warn(`Image count (${images.length}) doesn't match camera count (${training_cameras.length}). Some cameras may not have matching images.`);
    }
  }

  // ===============================================
  // Public Interface
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      if (trainingCount > 0) {
        trainStep(encoder);
        trainingCount -= 1;
        camera.needsPreprocess = true;
        const shouldDensify = trainingIteration % DENSIFY_INTERVAL === 0 && densify_bind_group && targetTextureView;
        const shouldPrune = trainingIteration % PRUNE_INTERVAL === 0 && prune_bind_group;
        
        if (shouldDensify && shouldPrune) {
          console.log(`Densifying at iteration ${trainingIteration}...`);
          densify().catch(err => console.error('Densify error:', err));
        } else {
          if (shouldDensify) {
            console.log(`Densifying at iteration ${trainingIteration}...`);
            densify().catch(err => console.error('Densify error:', err));
          }
          if (shouldPrune) {
            console.log(`Pruning at iteration ${trainingIteration}...`);
            prune().catch(err => console.error('Prune error:', err));
          }
        }
      }
      if (camera.needsPreprocess) {
        run_preprocess(encoder);
        if (sorter) {
          sorter.sort(encoder);
        }
        camera.needsPreprocess = false;
      }
      render_pass(encoder, texture_view);
    },
    camera_buffer,
    setGaussianScale,
    resizeViewport,
    train,
    setTargetTexture,
    initializeKnn,
    prune,
    densify,
    setTrainingCameras,
    setTrainingImages,
  };
}
