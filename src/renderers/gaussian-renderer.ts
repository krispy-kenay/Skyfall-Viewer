import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import commonWGSL from '../shaders/common.wgsl';
import knnInitWGSL from '../shaders/knn-init.wgsl';
import pruneWGSL from '../shaders/prune.wgsl';
import densifyWGSL from '../shaders/densify.wgsl';
import trainingForwardTilesWGSL from '../shaders/training-forward-tiles.wgsl';
import trainingLossWGSL from '../shaders/training-loss.wgsl';
import trainingBackwardTilesWGSL from '../shaders/training-backward-tiles.wgsl';
import trainingBackwardGeomWGSL from '../shaders/training-backward-geom.wgsl';
import trainingTilesWGSL from '../shaders/training-tiles.wgsl';
import trainingParamsWGSL from '../shaders/training-params.wgsl';
import adamWGSL from '../shaders/adam.wgsl';
import { get_sorter, C } from '../sort/sort';
import { Renderer } from './renderer';
import { Camera, TrainingCameraData, create_training_camera_uniform_buffer, update_training_camera_uniform_buffer } from '../camera/camera';

export interface GaussianRenderer extends Renderer {
  setGaussianScale(value: number): void;
  resizeViewport(w: number, h: number): void;
  train(numFrames: number): Promise<void>;
  // Debug-only: run a single training step immediately (with its own command submission)
  // and log a small slice of training buffers so we can verify gradients/updates.
  debugTrainOnce?(label?: string): Promise<void>;
  setTargetTexture(view: GPUTextureView, sampler: GPUSampler): void;
  initializeKnn(encoder: GPUCommandEncoder): void;
  prune(): Promise<number>;
  densify(): Promise<number>;
  setTrainingCameras(cameras: TrainingCameraData[]): void;
  setTrainingImages(images: Array<{ view: GPUTextureView; sampler: GPUSampler }>): void;
  buildTrainingTiles(): Promise<void>;
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

// Optimization / training hyperparameters (approximate 3DGS / LichtFeld defaults)
const OPT_ITERS = 30000;
const SH_DEGREE_INTERVAL = 1000; // iterations between SH degree increments

// TEMP: boosted learning rates to make training effects clearly visible while debugging.
// These should be dialed back closer to LichtFeld/3DGS defaults once the pipeline is verified.
const MEANS_LR = 1.6e-4;    // was 1.6e-5
const SHS_LR = 2.5e-2;      // was 2.5e-3
const OPACITY_LR = 5e-1;    // was 5e-2
const SCALING_LR = 5e-2;    // was 5e-3
const ROTATION_LR = 1e-2;   // was 1e-3

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
  let current_sh_degree = 0;

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
  let training_width = 0;
  let training_height = 0;

  // Training residuals (loss / backward)
  let residual_color_texture: GPUTexture | null = null;
  let residual_alpha_texture: GPUTexture | null = null;
  let residual_color_view: GPUTextureView | null = null;
  let residual_alpha_view: GPUTextureView | null = null;
  let training_loss_bind_group: GPUBindGroup | null = null;
  let training_init_params_bind_group: GPUBindGroup | null = null;
  let training_apply_params_bind_group: GPUBindGroup | null = null;
  let training_tiled_forward_bind_group: GPUBindGroup | null = null;
  let training_tiled_backward_bind_group: GPUBindGroup | null = null;
  let training_backward_geom_bind_group: GPUBindGroup | null = null;

  // Adam optimizer state
  let adam_config_buffer: GPUBuffer | null = null;
  let moment_sh_m1_buffer: GPUBuffer | null = null;
  let moment_sh_m2_buffer: GPUBuffer | null = null;
  let moment_opacity_m1_buffer: GPUBuffer | null = null;
  let moment_opacity_m2_buffer: GPUBuffer | null = null;
  let moment_position_m1_buffer: GPUBuffer | null = null;
  let moment_position_m2_buffer: GPUBuffer | null = null;
  let moment_scale_m1_buffer: GPUBuffer | null = null;
  let moment_scale_m2_buffer: GPUBuffer | null = null;
  let moment_rotation_m1_buffer: GPUBuffer | null = null;
  let moment_rotation_m2_buffer: GPUBuffer | null = null;
  let adam_step_counter = 0;

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
  let grad_alpha_splat_buffer: GPUBuffer | null = null;

  // Bind Groups
  let sort_bind_group: GPUBindGroup | null = null;
  let sorted_bg: GPUBindGroup[] = [];
  let render_splat_bg: GPUBindGroup | null = null;
  let preprocess_bg: GPUBindGroup | null = null;
  let knn_init_bind_group: GPUBindGroup | null = null;
  let prune_bind_group: GPUBindGroup | null = null;
  let densify_bind_group: GPUBindGroup | null = null;

  // External resources
  let targetTextureView: GPUTextureView | null = null;
  let targetSampler: GPUSampler | null = null;
  let sorter: any = null;
  let intersect_sorter: any = null;
  let intersect_sort_capacity = 0;

  // Training cameras
  let training_cameras: TrainingCameraData[] = [];
  let training_camera_buffers: GPUBuffer[] = [];
  let current_training_camera_index = 0;
  
  // Training images
  let training_images: Array<{ view: GPUTextureView; sampler: GPUSampler }> = [];

  // Tiling buffers
  const TILE_SIZE = 16;
  let tiling_params_buffer: GPUBuffer | null = null;
  let tile_means2d_buffer: GPUBuffer | null = null;
  let tile_radii_buffer: GPUBuffer | null = null;
  let tile_depths_buffer: GPUBuffer | null = null;
  let tiles_per_gauss_buffer: GPUBuffer | null = null;
  let tile_conics_buffer: GPUBuffer | null = null;

  let training_tiles_project_pipeline: GPUComputePipeline | null = null;
  let tiles_cumsum_buffer: GPUBuffer | null = null;
  let isect_ids_buffer: GPUBuffer | null = null;
  let flatten_ids_buffer: GPUBuffer | null = null;
  let tile_offsets_buffer: GPUBuffer | null = null;
  let num_intersections = 0;

  // Per-pixel last gaussian index for tiled training forward
  let training_last_ids_buffer: GPUBuffer | null = null;
  let tile_sorter: any = null;
  let tile_sort_capacity = 0;

  // Tracking for training tiling state
  let tilesBuilt = false;

  function getCurrentTrainingCameraBuffer(): GPUBuffer {
    // TEMP DEBUG
    // if (training_camera_buffers.length > 0) {
    //   return training_camera_buffers[current_training_camera_index];
    // }
    return camera_buffer;
  }

  // Projection parameters for training / tiling
  let projection_params_buffer: GPUBuffer | null = null;
  let background_params_buffer: GPUBuffer | null = null;
  let tile_mask_buffer: GPUBuffer | null = null;
  let loss_params_buffer: GPUBuffer | null = null;

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

  function createProjectionParamsBuffer(): GPUBuffer {
    const data = new Float32Array([
      0.2,
      10_000,
      512.0,
      0.3,
    ]);
    return createBuffer(
      device,
      'projection params',
      data.byteLength,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      data,
    );
  }

  function createBackgroundParamsBuffer(): GPUBuffer {
    const data = new Float32Array([
      0.0,
      0.0,
      0.0,
      0.0,
    ]);
    return createBuffer(
      device,
      'background params',
      data.byteLength,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      data,
    );
  }

  function createLossParamsBuffer(): GPUBuffer {
    const data = new Float32Array([
      1.0,
      0.0,
      0.0,
      0.0,
    ]);
    return createBuffer(
      device,
      'loss params',
      data.byteLength,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      data,
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

    // create float32 training gradient buffers
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
    grad_alpha_splat_buffer = device.createBuffer({
      label: 'grad alpha_splat (float32)',
      size: newCapacity * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Tiling projection buffers
    tile_means2d_buffer = device.createBuffer({
      label: 'tiling means2d (float32)',
      size: newCapacity * 2 * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    tile_radii_buffer = device.createBuffer({
      label: 'tiling radii (float32)',
      size: newCapacity * 2 * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    tile_depths_buffer = device.createBuffer({
      label: 'tiling depths (float32)',
      size: newCapacity * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    tiles_per_gauss_buffer = device.createBuffer({
      label: 'tiling tiles_per_gauss (u32)',
      size: newCapacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    tile_conics_buffer = device.createBuffer({
      label: 'tiling conics (float32)',
      size: newCapacity * 3 * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    num_intersections = 0;
    isect_ids_buffer = null;
    flatten_ids_buffer = null;
    tile_offsets_buffer = null;

    // Adam moment buffers
    moment_sh_m1_buffer = device.createBuffer({
      label: 'adam m1 SH (float32)',
      size: newCapacity * TRAIN_SH_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    moment_sh_m2_buffer = device.createBuffer({
      label: 'adam m2 SH (float32)',
      size: newCapacity * TRAIN_SH_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    moment_opacity_m1_buffer = device.createBuffer({
      label: 'adam m1 opacity (float32)',
      size: newCapacity * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    moment_opacity_m2_buffer = device.createBuffer({
      label: 'adam m2 opacity (float32)',
      size: newCapacity * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    moment_position_m1_buffer = device.createBuffer({
      label: 'adam m1 position (float32)',
      size: newCapacity * TRAIN_POS_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    moment_position_m2_buffer = device.createBuffer({
      label: 'adam m2 position (float32)',
      size: newCapacity * TRAIN_POS_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    moment_scale_m1_buffer = device.createBuffer({
      label: 'adam m1 scale (float32)',
      size: newCapacity * TRAIN_SCALE_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    moment_scale_m2_buffer = device.createBuffer({
      label: 'adam m2 scale (float32)',
      size: newCapacity * TRAIN_SCALE_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    moment_rotation_m1_buffer = device.createBuffer({
      label: 'adam m1 rotation (float32)',
      size: newCapacity * TRAIN_ROT_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    moment_rotation_m2_buffer = device.createBuffer({
      label: 'adam m2 rotation (float32)',
      size: newCapacity * TRAIN_ROT_COMPONENTS * FLOAT32_BYTES,
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
  projection_params_buffer = createProjectionParamsBuffer();
  background_params_buffer = createBackgroundParamsBuffer();
  loss_params_buffer = createLossParamsBuffer();

  {
    const initialShDegree = 0;
    current_sh_degree = initialShDegree;
    device.queue.writeBuffer(settings_buffer, 4, new Float32Array([initialShDegree]));
  }
  adam_config_buffer = createBuffer(
    device,
    'adam config',
    4 * 8,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  );

  function setGaussianScale(value: number) {
    device.queue.writeBuffer(settings_buffer, 0, new Float32Array([value]));
  }

  function resizeViewport(w: number, h: number) {
    device.queue.writeBuffer(settings_buffer, 2 * 4, new Float32Array([w, h]));
    ensureTrainingRenderTargets(w, h);
    tilesBuilt = false;
  }

  function setActiveCount(value: number) {
    active_count = value;
    if (active_count_uniform_buffer) {
      device.queue.writeBuffer(active_count_uniform_buffer, 0, new Uint32Array([value]));
    }
    if (indirect_args) {
      device.queue.writeBuffer(indirect_args, 0, new Uint32Array([6, value, 0, 0]));
    }
    tilesBuilt = false;
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

  const training_loss_pipeline = device.createComputePipeline({
    label: 'training loss',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: trainingLossWGSL }),
      entryPoint: 'training_loss',
    },
  });

  const training_backward_geom_pipeline = device.createComputePipeline({
    label: 'training backward geom',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: commonWGSL + '\n' + trainingBackwardGeomWGSL }),
      entryPoint: 'training_backward_geom',
    },
  });

  const training_params_module = device.createShaderModule({
    code: commonWGSL + '\n' + trainingParamsWGSL,
  });

  const training_tiles_module = device.createShaderModule({
    code: commonWGSL + '\n' + trainingTilesWGSL,
  });

  training_tiles_project_pipeline = device.createComputePipeline({
    label: 'training tiles project',
    layout: 'auto',
    compute: {
      module: training_tiles_module,
      entryPoint: 'project_gaussians_for_tiling',
    },
  });

  const training_tiles_emit_pipeline = device.createComputePipeline({
    label: 'training tiles emit',
    layout: 'auto',
    compute: {
      module: training_tiles_module,
      entryPoint: 'emit_intersections',
    },
  });

  const training_tiles_offsets_pipeline = device.createComputePipeline({
    label: 'training tiles offsets',
    layout: 'auto',
    compute: {
      module: training_tiles_module,
      entryPoint: 'build_tile_offsets',
    },
  });

  const training_init_params_pipeline = device.createComputePipeline({
    label: 'training params init',
    layout: 'auto',
    compute: {
      module: training_params_module,
      entryPoint: 'training_init_params',
    },
  });

  const training_apply_params_pipeline = device.createComputePipeline({
    label: 'training params apply',
    layout: 'auto',
    compute: {
      module: training_params_module,
      entryPoint: 'training_apply_params',
    },
  });

  const adam_pipeline = device.createComputePipeline({
    label: 'adam optimizer',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: adamWGSL }),
      entryPoint: 'adam_step',
    },
  });

  const training_tiled_forward_pipeline = device.createComputePipeline({
    label: 'training tiled forward',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: commonWGSL + '\n' + trainingForwardTilesWGSL }),
      entryPoint: 'training_tiled_forward',
    },
  });

  const training_tiled_backward_pipeline = device.createComputePipeline({
    label: 'training tiled backward',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: commonWGSL + '\n' + trainingBackwardTilesWGSL }),
      entryPoint: 'training_tiled_backward',
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

    const pixelCount = width * height;
    if (!training_last_ids_buffer || training_last_ids_buffer.size < pixelCount * 4) {
      training_last_ids_buffer?.destroy();
      training_last_ids_buffer = device.createBuffer({
        label: 'training tiled last_ids (u32)',
        size: pixelCount * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }
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

  function ensureTilingParamsBuffer() {
    if (!tiling_params_buffer) {
      tiling_params_buffer = createBuffer(
        device,
        'tiling params',
        6 * 4,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      );
    }
    const tileWidth = Math.ceil(canvas.width / TILE_SIZE);
    const tileHeight = Math.ceil(canvas.height / TILE_SIZE);
    const numCameras = 1;
    const params = new Uint32Array([
      canvas.width,
      canvas.height,
      TILE_SIZE,
      tileWidth,
      tileHeight,
      numCameras,
    ]);
    device.queue.writeBuffer(tiling_params_buffer, 0, params);
  }

  function ensureTileSorter(requiredKeys: number) {
    if (requiredKeys <= 0) {
      tile_sorter = null;
      tile_sort_capacity = 0;
      return;
    }
    if (!tile_sorter || requiredKeys > tile_sort_capacity) {
      tile_sorter = get_sorter(requiredKeys, device);
      tile_sort_capacity = requiredKeys;
    }
  }

  function ensureIntersectionBuffers(requiredIntersections: number) {
    if (requiredIntersections <= 0) {
      num_intersections = 0;
      if (isect_ids_buffer) {
        isect_ids_buffer.destroy();
        isect_ids_buffer = null;
      }
      if (flatten_ids_buffer) {
        flatten_ids_buffer.destroy();
        flatten_ids_buffer = null;
      }
      if (tile_offsets_buffer) {
        tile_offsets_buffer.destroy();
        tile_offsets_buffer = null;
      }
      return;
    }

    const tileWidth = Math.ceil(canvas.width / TILE_SIZE);
    const tileHeight = Math.ceil(canvas.height / TILE_SIZE);
    const tileCount = tileWidth * tileHeight;
    const numCameras = 1;
    const totalTiles = tileCount * numCameras;

    if (!isect_ids_buffer || requiredIntersections > num_intersections) {
      if (isect_ids_buffer) {
        isect_ids_buffer.destroy();
      }
      isect_ids_buffer = device.createBuffer({
        label: 'tiling isect_ids (u32 keys)',
        size: requiredIntersections * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }

    if (!flatten_ids_buffer || requiredIntersections > num_intersections) {
      if (flatten_ids_buffer) {
        flatten_ids_buffer.destroy();
      }
      flatten_ids_buffer = device.createBuffer({
        label: 'tiling flatten_ids (u32)',
        size: requiredIntersections * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }

    if (!tile_offsets_buffer || (totalTiles + 1) * 4 > (tile_offsets_buffer?.size ?? 0)) {
      if (tile_offsets_buffer) {
        tile_offsets_buffer.destroy();
      }
      tile_offsets_buffer = device.createBuffer({
        label: 'tiling tile_offsets (u32)',
        size: (totalTiles + 1) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }

    // Optional per-tile masks; by default all tiles are enabled (1u).
    if (!tile_mask_buffer || totalTiles * 4 > (tile_mask_buffer?.size ?? 0)) {
      tile_mask_buffer?.destroy();
      tile_mask_buffer = device.createBuffer({
        label: 'tiling tile_masks (u32)',
        size: totalTiles * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const masks = new Uint32Array(totalTiles);
      masks.fill(1);
      device.queue.writeBuffer(tile_mask_buffer, 0, masks);
    }

    num_intersections = requiredIntersections;
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

  function rebuildTrainingLossBG() {
    if (!training_color_view || !training_alpha_view ||
        !residual_color_view || !residual_alpha_view) {
      return;
    }

    // Use training images as targets if available; fallback to viewer target.
    let texture_view_to_use: GPUTextureView;
    let sampler_to_use: GPUSampler;

    if (training_images.length > 0 && current_training_camera_index < training_images.length) {
      texture_view_to_use = training_images[current_training_camera_index].view;
      sampler_to_use = training_images[current_training_camera_index].sampler;
    } else if (targetTextureView && targetSampler) {
      // Fallback: viewer camera target
      texture_view_to_use = targetTextureView;
      sampler_to_use = targetSampler;
    } else {
      console.warn('No training images or viewer target texture set for training loss.');
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
        // LossParams: default to pure L2 (w_l2 = 1, w_l1 = 0).
        { binding: 6, resource: { buffer: loss_params_buffer! } },
      ],
    });
  }

  function rebuildTrainingBackwardBG() {
    if (!gaussian_buffer || !grad_sh_buffer || !grad_opacity_buffer ||
        !grad_position_buffer || !grad_scale_buffer || !grad_rotation_buffer ||
        !residual_color_view ||
        !active_count_uniform_buffer) {
      return;
    }

    // Legacy non-tiled backward has been removed in favor of the tiled path.
  }

  function rebuildTrainingBackwardGeomBG() {
    if (!gaussian_buffer ||
        !grad_position_buffer || !grad_scale_buffer || !grad_rotation_buffer ||
        !grad_alpha_splat_buffer || !active_count_uniform_buffer) {
      return;
    }

    training_backward_geom_bind_group = device.createBindGroup({
      label: 'training backward geom bind group',
      layout: training_backward_geom_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: getCurrentTrainingCameraBuffer() } },
        { binding: 1, resource: { buffer: gaussian_buffer } },
        { binding: 2, resource: { buffer: active_count_uniform_buffer } },
        { binding: 3, resource: { buffer: grad_alpha_splat_buffer } },
        { binding: 4, resource: { buffer: grad_position_buffer } },
        { binding: 5, resource: { buffer: grad_scale_buffer } },
        { binding: 6, resource: { buffer: grad_rotation_buffer } },
      ],
    });
  }

  function rebuildTrainingTiledForwardBG() {
    if (!training_tiled_forward_pipeline) return;
    if (!gaussian_buffer || !sh_buffer || !tiling_params_buffer) return;
    if (!tile_means2d_buffer || !tile_conics_buffer) return;
    if (!tile_offsets_buffer || !flatten_ids_buffer || !tile_mask_buffer) return;
    if (!training_color_view || !training_alpha_view || !training_last_ids_buffer) return;
    if (!tiles_cumsum_buffer || !active_count_uniform_buffer) return;

    training_tiled_forward_bind_group = device.createBindGroup({
      label: 'training tiled forward bind group',
      layout: training_tiled_forward_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: getCurrentTrainingCameraBuffer() } },
        { binding: 1, resource: { buffer: settings_buffer } },
        { binding: 2, resource: { buffer: tiling_params_buffer } },
        { binding: 3, resource: { buffer: tile_means2d_buffer } },
        { binding: 4, resource: { buffer: tile_conics_buffer } },
        { binding: 6, resource: { buffer: tile_offsets_buffer } },
        { binding: 7, resource: { buffer: flatten_ids_buffer } },
        { binding: 8, resource: training_color_view },
        { binding: 9, resource: training_alpha_view },
        { binding: 10, resource: { buffer: training_last_ids_buffer } },
        { binding: 11, resource: { buffer: gaussian_buffer } },
        { binding: 12, resource: { buffer: sh_buffer } },
        { binding: 13, resource: { buffer: background_params_buffer! } },
        { binding: 14, resource: { buffer: tile_mask_buffer! } },
        { binding: 15, resource: { buffer: tiles_cumsum_buffer } },
        { binding: 16, resource: { buffer: active_count_uniform_buffer } },
      ],
    });
  }

  function rebuildTrainingTiledBackwardBG() {
    if (!training_tiled_backward_pipeline) return;
    if (!gaussian_buffer || !sh_buffer || !tiling_params_buffer) return;
    if (!residual_color_view || !training_last_ids_buffer) return;
    if (!tile_offsets_buffer || !flatten_ids_buffer || !tile_mask_buffer) return;
    if (!grad_sh_buffer || !grad_opacity_buffer || !grad_alpha_splat_buffer || !active_count_uniform_buffer) return;
    if (!tiles_cumsum_buffer) return;

    training_tiled_backward_bind_group = device.createBindGroup({
      label: 'training tiled backward bind group',
      layout: training_tiled_backward_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: getCurrentTrainingCameraBuffer() } },
        { binding: 1, resource: { buffer: settings_buffer } },
        { binding: 2, resource: { buffer: tiling_params_buffer } },
        { binding: 3, resource: { buffer: gaussian_buffer } },
        { binding: 4, resource: { buffer: sh_buffer } },
        { binding: 5, resource: residual_color_view },
        { binding: 6, resource: { buffer: training_last_ids_buffer } },
        { binding: 7, resource: { buffer: tile_offsets_buffer } },
        { binding: 8, resource: { buffer: flatten_ids_buffer } },
        { binding: 9, resource: { buffer: grad_sh_buffer } },
        { binding: 10, resource: { buffer: grad_opacity_buffer } },
        { binding: 11, resource: { buffer: active_count_uniform_buffer } },
        { binding: 12, resource: { buffer: grad_alpha_splat_buffer } },
        { binding: 13, resource: { buffer: background_params_buffer! } },
        { binding: 14, resource: { buffer: tile_mask_buffer! } },
        { binding: 15, resource: { buffer: tiles_cumsum_buffer } },
      ],
    });
  }

  function rebuildTrainingInitParamsBG() {
    if (!gaussian_buffer || !sh_buffer ||
        !train_opacity_buffer || !train_sh_buffer ||
        !train_position_buffer || !train_scale_buffer || !train_rotation_buffer ||
        !active_count_uniform_buffer) {
      return;
    }

    training_init_params_bind_group = device.createBindGroup({
      label: 'training init params bind group',
      layout: training_init_params_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gaussian_buffer } },
        { binding: 1, resource: { buffer: sh_buffer } },
        { binding: 2, resource: { buffer: train_opacity_buffer } },
        { binding: 3, resource: { buffer: train_sh_buffer } },
        { binding: 4, resource: { buffer: active_count_uniform_buffer } },
        { binding: 5, resource: { buffer: train_position_buffer } },
        { binding: 6, resource: { buffer: train_scale_buffer } },
        { binding: 7, resource: { buffer: train_rotation_buffer } },
      ],
    });
  }

  function rebuildTrainingApplyParamsBG() {
    if (!gaussian_buffer || !sh_buffer ||
        !train_opacity_buffer || !train_sh_buffer ||
        !train_position_buffer || !train_scale_buffer || !train_rotation_buffer ||
        !active_count_uniform_buffer) {
      return;
    }

    training_apply_params_bind_group = device.createBindGroup({
      label: 'training apply params bind group',
      layout: training_apply_params_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gaussian_buffer } },
        { binding: 1, resource: { buffer: sh_buffer } },
        { binding: 2, resource: { buffer: train_opacity_buffer } },
        { binding: 3, resource: { buffer: train_sh_buffer } },
        { binding: 4, resource: { buffer: active_count_uniform_buffer } },
        { binding: 5, resource: { buffer: train_position_buffer } },
        { binding: 6, resource: { buffer: train_scale_buffer } },
        { binding: 7, resource: { buffer: train_rotation_buffer } },
      ],
    });
  }

  function rebuildAllBindGroups() {
    rebuildSorterBG();
    rebuildSortedBG();
    rebuildRenderSplatBG();
    rebuildPreprocessBG();
    rebuildKnnInitBG();
    rebuildPruneBG();
    rebuildDensifyBG();
    rebuildTrainingBackwardGeomBG();
    // Tiled training bind groups are rebuilt lazily when running tiled passes.
    rebuildTrainingInitParamsBG();
    rebuildTrainingApplyParamsBG();
  }

  function run_training_tiles_projection(encoder: GPUCommandEncoder) {
    if (!training_tiles_project_pipeline ||
        !gaussian_buffer ||
        !tile_means2d_buffer || !tile_radii_buffer || !tile_depths_buffer || !tile_conics_buffer ||
        !tiles_per_gauss_buffer ||
        !projection_params_buffer ||
        !active_count_uniform_buffer) {
      return;
    }

    ensureTilingParamsBuffer();

    const bindGroup = device.createBindGroup({
      label: 'training tiles project bind group',
      layout: training_tiles_project_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: getCurrentTrainingCameraBuffer() } },
        { binding: 1, resource: { buffer: gaussian_buffer } },
        { binding: 2, resource: { buffer: active_count_uniform_buffer } },
        { binding: 3, resource: { buffer: tiling_params_buffer! } },
        { binding: 4, resource: { buffer: tile_means2d_buffer } },
        { binding: 5, resource: { buffer: tile_radii_buffer } },
        { binding: 6, resource: { buffer: tile_depths_buffer } },
        { binding: 7, resource: { buffer: tiles_per_gauss_buffer } },
        { binding: 12, resource: { buffer: tile_conics_buffer } },
        { binding: 13, resource: { buffer: projection_params_buffer } },
      ],
    });

    const num = active_count;
    if (num === 0) return;
    const num_wg = Math.ceil(num / WORKGROUP_SIZE);

    const pass = encoder.beginComputePass({ label: 'training tiles project' });
    pass.setPipeline(training_tiles_project_pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(num_wg);
    pass.end();
  }

  async function buildTileIntersections() {
    // Any call to this function is (re)building tiling data for the current state.
    tilesBuilt = false;

    if (!tiles_per_gauss_buffer) {
      return;
    }
    if (active_count === 0) {
      ensureIntersectionBuffers(0);
      return;
    }

    const encoder = device.createCommandEncoder();
    run_training_tiles_projection(encoder);

    const tilesSizeBytes = active_count * 4;
    const staging = device.createBuffer({
      label: 'tiling tiles_per_gauss staging',
      size: tilesSizeBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(tiles_per_gauss_buffer, 0, staging, 0, tilesSizeBytes);
    device.queue.submit([encoder.finish()]);

    await device.queue.onSubmittedWorkDone();
    await staging.mapAsync(GPUMapMode.READ);
    const mapped = staging.getMappedRange();
    const countsView = new Uint32Array(mapped.slice(0, tilesSizeBytes));

    const cumsum = new Uint32Array(active_count);
    let running = 0;
    for (let i = 0; i < active_count; i++) {
      running += countsView[i];
      cumsum[i] = running;
    }
    staging.unmap();
    staging.destroy();

    const totalIntersections = running;

    if (!tiles_cumsum_buffer || tiles_cumsum_buffer.size < active_count * 4) {
      if (tiles_cumsum_buffer) {
        tiles_cumsum_buffer.destroy();
      }
      tiles_cumsum_buffer = device.createBuffer({
        label: 'tiling tiles_cumsum (u32)',
        size: active_count * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    }
    device.queue.writeBuffer(tiles_cumsum_buffer, 0, cumsum);

    ensureIntersectionBuffers(totalIntersections);

    if (totalIntersections === 0) {
      return;
    }

    if (!training_tiles_emit_pipeline ||
        !tile_means2d_buffer || !tile_radii_buffer || !tile_depths_buffer ||
        !tiles_cumsum_buffer ||
        !isect_ids_buffer || !flatten_ids_buffer ||
        !active_count_uniform_buffer) {
      return;
    }

    const emitEncoder = device.createCommandEncoder();
    ensureTilingParamsBuffer();

    const emitBindGroup = device.createBindGroup({
      label: 'training tiles emit bind group',
      layout: training_tiles_emit_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 2, resource: { buffer: active_count_uniform_buffer! } },
        { binding: 3, resource: { buffer: tiling_params_buffer! } },
        { binding: 4, resource: { buffer: tile_means2d_buffer! } },
        { binding: 5, resource: { buffer: tile_radii_buffer! } },
        { binding: 6, resource: { buffer: tile_depths_buffer! } },
        { binding: 7, resource: { buffer: tiles_per_gauss_buffer! } },
        { binding: 8, resource: { buffer: tiles_cumsum_buffer! } },
        { binding: 9, resource: { buffer: isect_ids_buffer! } },
        { binding: 10, resource: { buffer: flatten_ids_buffer! } },
      ],
    });

    const num = active_count;
    const num_wg = Math.ceil(num / WORKGROUP_SIZE);

    {
      const pass = emitEncoder.beginComputePass({ label: 'training tiles emit' });
      pass.setPipeline(training_tiles_emit_pipeline);
      pass.setBindGroup(0, emitBindGroup);
      pass.dispatchWorkgroups(num_wg);
      pass.end();
    }

    device.queue.submit([emitEncoder.finish()]);

    if (totalIntersections <= 0) {
      tilesBuilt = true;
      return;
    }

    if (!intersect_sorter || totalIntersections > intersect_sort_capacity) {
      intersect_sorter = get_sorter(totalIntersections, device);
      intersect_sort_capacity = totalIntersections;
    }

    if (!intersect_sorter) {
      return;
    }

    const sortEncoder = device.createCommandEncoder();
    sortEncoder.copyBufferToBuffer(
      isect_ids_buffer,
      0,
      intersect_sorter.ping_pong[0].sort_depths_buffer,
      0,
      totalIntersections * 4
    );
    sortEncoder.copyBufferToBuffer(
      flatten_ids_buffer,
      0,
      intersect_sorter.ping_pong[0].sort_indices_buffer,
      0,
      totalIntersections * 4
    );

    intersect_sorter.sort(sortEncoder);

    const finalPing = intersect_sorter.ping_pong[intersect_sorter.final_out_index];
    sortEncoder.copyBufferToBuffer(
      finalPing.sort_depths_buffer,
      0,
      isect_ids_buffer,
      0,
      totalIntersections * 4
    );
    sortEncoder.copyBufferToBuffer(
      finalPing.sort_indices_buffer,
      0,
      flatten_ids_buffer,
      0,
      totalIntersections * 4
    );

    device.queue.submit([sortEncoder.finish()]);

    if (!training_tiles_offsets_pipeline) {
      return;
    }

    const offsetsEncoder = device.createCommandEncoder();

    const offsetsBindGroup = device.createBindGroup({
      label: 'training tiles offsets bind group',
      layout: training_tiles_offsets_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 2, resource: { buffer: active_count_uniform_buffer! } },
        { binding: 3, resource: { buffer: tiling_params_buffer! } },
        { binding: 8, resource: { buffer: tiles_cumsum_buffer! } },
        { binding: 9, resource: { buffer: isect_ids_buffer! } },
        { binding: 11, resource: { buffer: tile_offsets_buffer! } },
      ],
    });

    const nTiles = Math.ceil(canvas.width / TILE_SIZE) * Math.ceil(canvas.height / TILE_SIZE);
    const offsets_wg = Math.ceil(Math.max(1, num_intersections, nTiles) / WORKGROUP_SIZE);

    {
      const pass = offsetsEncoder.beginComputePass({ label: 'training tiles offsets' });
      pass.setPipeline(training_tiles_offsets_pipeline);
      pass.setBindGroup(0, offsetsBindGroup);
      pass.dispatchWorkgroups(offsets_wg);
      pass.end();
    }

    device.queue.submit([offsetsEncoder.finish()]);

    tilesBuilt = true;
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
    rebuildTrainingInitParamsBG();
    run_training_init_params(encoder);
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
    tilesBuilt = false;
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
  // Densification
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
    tilesBuilt = false;
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

  function run_training_tiled_forward(encoder: GPUCommandEncoder) {
    ensureTrainingRenderTargets(canvas.width, canvas.height);
    ensureTilingParamsBuffer();

    if (!training_tiles_project_pipeline) {
      console.warn('training_tiles_project_pipeline is null, cannot run tiled training forward');
      return;
    }
    if (!tile_means2d_buffer || !tile_conics_buffer || !tile_depths_buffer ||
        !tile_offsets_buffer || !flatten_ids_buffer) {
      console.warn('Tiling buffers are not ready, call buildTrainingTiles() before tiled training');
      return;
    }

    rebuildTrainingTiledForwardBG();
    if (!training_tiled_forward_bind_group) {
      console.warn('training_tiled_forward_bind_group is null, skipping tiled training_forward dispatch');
      return;
    }

    const wgSizeX = 8;
    const wgSizeY = 8;
    const numGroupsX = Math.ceil(canvas.width / wgSizeX);
    const numGroupsY = Math.ceil(canvas.height / wgSizeY);

    const pass = encoder.beginComputePass({ label: 'training tiled forward' });
    pass.setPipeline(training_tiled_forward_pipeline);
    pass.setBindGroup(0, training_tiled_forward_bind_group);
    pass.dispatchWorkgroups(numGroupsX, numGroupsY);
    pass.end();
  }

  function run_training_loss(encoder: GPUCommandEncoder) {
    ensureTrainingRenderTargets(canvas.width, canvas.height);
    rebuildTrainingLossBG();
    if (!training_loss_bind_group) {
      console.warn('training_loss_bind_group is null, skipping training_loss dispatch');
      return;
    }

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

  function run_training_tiled_backward(encoder: GPUCommandEncoder) {
    ensureTrainingRenderTargets(canvas.width, canvas.height);
    ensureTilingParamsBuffer();

    const num = active_count;
    if (num === 0) {
      console.warn('No active gaussians, skipping tiled training_backward dispatch');
      return;
    }

    if (!tile_offsets_buffer || !flatten_ids_buffer) {
      console.warn('Tiling intersection buffers are not ready, call buildTrainingTiles() before tiled backward');
      return;
    }

    rebuildTrainingTiledBackwardBG();
    if (!training_tiled_backward_bind_group) {
      console.warn('training_tiled_backward_bind_group is null, skipping tiled training_backward dispatch');
      return;
    }

    const wgSizeX = 8;
    const wgSizeY = 8;
    const numGroupsX = Math.ceil(canvas.width / wgSizeX);
    const numGroupsY = Math.ceil(canvas.height / wgSizeY);

    const pass = encoder.beginComputePass({ label: 'training tiled backward' });
    pass.setPipeline(training_tiled_backward_pipeline);
    pass.setBindGroup(0, training_tiled_backward_bind_group);
    pass.dispatchWorkgroups(numGroupsX, numGroupsY);
    pass.end();
  }

  function run_training_backward_geom(encoder: GPUCommandEncoder) {
    const num = active_count;
    if (num === 0) {
      console.warn('No active gaussians, skipping training_backward_geom dispatch');
      return;
    }

    rebuildTrainingBackwardGeomBG();
    if (!training_backward_geom_bind_group) {
      console.warn('training_backward_geom_bind_group is null, skipping training_backward_geom dispatch');
      return;
    }

    const num_wg = Math.ceil(num / WORKGROUP_SIZE);

    const pass = encoder.beginComputePass({ label: 'training backward geom' });
    pass.setPipeline(training_backward_geom_pipeline);
    pass.setBindGroup(0, training_backward_geom_bind_group);
    pass.dispatchWorkgroups(num_wg);
    pass.end();
  }

  function run_adam_on_buffer(
    encoder: GPUCommandEncoder,
    param: GPUBuffer,
    m1: GPUBuffer,
    m2: GPUBuffer,
    grad: GPUBuffer,
    elementCount: number,
    lr: number,
  ) {
    if (!adam_config_buffer) return;
    if (elementCount <= 0) return;

    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;

    const t = Math.max(1, adam_step_counter);
    const bias_correction1 = 1.0 - Math.pow(beta1, t);
    const bias_correction2 = 1.0 - Math.pow(beta2, t);
    const bias_correction1_rcp = 1.0 / bias_correction1;
    const bias_correction2_sqrt_rcp = 1.0 / Math.sqrt(bias_correction2);

    const cfg = new Float32Array([
      lr,
      beta1,
      beta2,
      eps,
      bias_correction1_rcp,
      bias_correction2_sqrt_rcp,
      elementCount,
      0.0,
    ]);
    device.queue.writeBuffer(adam_config_buffer, 0, cfg);

    const bindGroup = device.createBindGroup({
      label: 'adam bind group',
      layout: adam_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: param } },
        { binding: 1, resource: { buffer: m1 } },
        { binding: 2, resource: { buffer: m2 } },
        { binding: 3, resource: { buffer: grad } },
        { binding: 4, resource: { buffer: adam_config_buffer } },
      ],
    });

    const num_wg = Math.ceil(elementCount / WORKGROUP_SIZE);
    const pass = encoder.beginComputePass({ label: 'adam step' });
    pass.setPipeline(adam_pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(num_wg);
    pass.end();
  }

  function run_adam_updates(encoder: GPUCommandEncoder) {
    if (!train_sh_buffer || !grad_sh_buffer || !moment_sh_m1_buffer || !moment_sh_m2_buffer) return;
    if (!train_opacity_buffer || !grad_opacity_buffer || !moment_opacity_m1_buffer || !moment_opacity_m2_buffer) return;
    if (!train_position_buffer || !grad_position_buffer || !moment_position_m1_buffer || !moment_position_m2_buffer) return;
    if (!train_scale_buffer || !grad_scale_buffer || !moment_scale_m1_buffer || !moment_scale_m2_buffer) return;
    if (!train_rotation_buffer || !grad_rotation_buffer || !moment_rotation_m1_buffer || !moment_rotation_m2_buffer) return;

    const sh_elements = active_count * TRAIN_SH_COMPONENTS;
    const opacity_elements = active_count;
    const pos_elements = active_count * TRAIN_POS_COMPONENTS;
    const scale_elements = active_count * TRAIN_SCALE_COMPONENTS;
    const rot_elements = active_count * TRAIN_ROT_COMPONENTS;

    const lr_means = MEANS_LR;
    const lr_sh = SHS_LR;
    const lr_opacity = OPACITY_LR;
    const lr_scale = SCALING_LR;
    const lr_rotation = ROTATION_LR;

    if (sh_elements > 0) {
      run_adam_on_buffer(
        encoder,
        train_sh_buffer,
        moment_sh_m1_buffer,
        moment_sh_m2_buffer,
        grad_sh_buffer,
        sh_elements,
        lr_sh,
      );
    }

    if (opacity_elements > 0) {
      run_adam_on_buffer(
        encoder,
        train_opacity_buffer,
        moment_opacity_m1_buffer,
        moment_opacity_m2_buffer,
        grad_opacity_buffer,
        opacity_elements,
        lr_opacity,
      );
    }

    if (pos_elements > 0) {
      run_adam_on_buffer(
        encoder,
        train_position_buffer,
        moment_position_m1_buffer,
        moment_position_m2_buffer,
        grad_position_buffer,
        pos_elements,
        lr_means,
      );
    }

    if (scale_elements > 0) {
      run_adam_on_buffer(
        encoder,
        train_scale_buffer,
        moment_scale_m1_buffer,
        moment_scale_m2_buffer,
        grad_scale_buffer,
        scale_elements,
        lr_scale,
      );
    }

    if (rot_elements > 0) {
      run_adam_on_buffer(
        encoder,
        train_rotation_buffer,
        moment_rotation_m1_buffer,
        moment_rotation_m2_buffer,
        grad_rotation_buffer,
        rot_elements,
        lr_rotation,
      );
    }

    adam_step_counter += 1;
  }

  function run_training_init_params(encoder: GPUCommandEncoder) {
    if (!training_init_params_bind_group) return;
    const num = active_count;
    if (num === 0) return;
    const num_wg = Math.ceil(num / WORKGROUP_SIZE);

    const pass = encoder.beginComputePass({ label: 'training init params' });
    pass.setPipeline(training_init_params_pipeline);
    pass.setBindGroup(0, training_init_params_bind_group);
    pass.dispatchWorkgroups(num_wg);
    pass.end();
  }

  function run_training_apply_params(encoder: GPUCommandEncoder) {
    if (!training_apply_params_bind_group) {
      console.warn('training_apply_params_bind_group is null, skipping apply dispatch');
      return;
    }
    const num = active_count;
    if (num === 0) {
      console.warn('No active gaussians, skipping apply params dispatch');
      return;
    }
    const num_wg = Math.ceil(num / WORKGROUP_SIZE);
    console.log('[run_training_apply_params] dispatching', num_wg, 'workgroups for', num, 'gaussians');

    const pass = encoder.beginComputePass({ label: 'training apply params' });
    pass.setPipeline(training_apply_params_pipeline);
    pass.setBindGroup(0, training_apply_params_bind_group);
    pass.dispatchWorkgroups(num_wg);
    pass.end();
  }

  // ===============================================
  // Training
  // ===============================================
  async function debugSampleTrainingState(label: string) {
    const sampleCount = Math.min(active_count, 8);
    if (active_count === 0) return;

    if (!train_opacity_buffer || !grad_opacity_buffer || !grad_alpha_splat_buffer) {
      return;
    }

    const sampleBytes = sampleCount * FLOAT32_BYTES;
    const fullBytes = active_count * FLOAT32_BYTES;
    const encoder = device.createCommandEncoder();

    const makeStaging = (dbgLabel: string, size: number) =>
      device.createBuffer({
        label: dbgLabel,
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

    const stTrainOpacity = makeStaging('debug train_opacity', sampleBytes);
    const stGradOpacity = makeStaging('debug grad_opacity_full', fullBytes);
    const stGradAlphaSplat = makeStaging('debug grad_alpha_splat_full', fullBytes);

    encoder.copyBufferToBuffer(train_opacity_buffer, 0, stTrainOpacity, 0, sampleBytes);
    encoder.copyBufferToBuffer(grad_opacity_buffer, 0, stGradOpacity, 0, fullBytes);
    encoder.copyBufferToBuffer(grad_alpha_splat_buffer, 0, stGradAlphaSplat, 0, fullBytes);

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    try {
      // Map and log train_opacity slice
      await stTrainOpacity.mapAsync(GPUMapMode.READ);
      {
        const copy = stTrainOpacity.getMappedRange().slice(0);
        const arr = Array.from(new Float32Array(copy));
        console.log(`[training debug] ${label} train_opacity[0:${sampleCount}] =`, arr);
      }
      stTrainOpacity.unmap();

      // Map and analyze full grad_opacity
      await stGradOpacity.mapAsync(GPUMapMode.READ);
      const gradOpArrFull = new Float32Array(stGradOpacity.getMappedRange().slice(0));
      stGradOpacity.unmap();

      let nonZeroOp = 0;
      let maxAbsOp = 0;
      const examplesOp: Array<{ idx: number; v: number }> = [];
      for (let i = 0; i < active_count; i++) {
        const v = gradOpArrFull[i];
        if (v !== 0) {
          nonZeroOp++;
          const av = Math.abs(v);
          if (av > maxAbsOp) maxAbsOp = av;
          if (examplesOp.length < 5) {
            examplesOp.push({ idx: i, v });
          }
        }
      }
      console.log(
        `[training debug] ${label} grad_opacity stats: nonZero=${nonZeroOp}/${active_count}, maxAbs=${maxAbsOp}, examples=`,
        examplesOp,
      );
      console.log(
        `[training debug] ${label} grad_opacity[0:${sampleCount}] =`,
        Array.from(gradOpArrFull.subarray(0, sampleCount)),
      );

      // Map and analyze full grad_alpha_splat
      await stGradAlphaSplat.mapAsync(GPUMapMode.READ);
      const gradAlphaArrFull = new Float32Array(stGradAlphaSplat.getMappedRange().slice(0));
      stGradAlphaSplat.unmap();

      let nonZeroAlpha = 0;
      let maxAbsAlpha = 0;
      const examplesAlpha: Array<{ idx: number; v: number }> = [];
      for (let i = 0; i < active_count; i++) {
        const v = gradAlphaArrFull[i];
        if (v !== 0) {
          nonZeroAlpha++;
          const av = Math.abs(v);
          if (av > maxAbsAlpha) maxAbsAlpha = av;
          if (examplesAlpha.length < 5) {
            examplesAlpha.push({ idx: i, v });
          }
        }
      }
      console.log(
        `[training debug] ${label} grad_alpha_splat stats: nonZero=${nonZeroAlpha}/${active_count}, maxAbs=${maxAbsAlpha}, examples=`,
        examplesAlpha,
      );
      await Promise.all([
        Promise.resolve(),
        Promise.resolve(),
        Promise.resolve(),
      ]);
    } finally {
      stTrainOpacity.destroy();
      stGradOpacity.destroy();
      stGradAlphaSplat.destroy();
    }
  }

  function zeroGradBuffers() {
    if (!grad_sh_buffer || !grad_opacity_buffer ||
        !grad_position_buffer || !grad_scale_buffer || !grad_rotation_buffer) return;
    const sh_elements = capacity * TRAIN_SH_COMPONENTS;
    const opacity_elements = capacity;
    const pos_elements = capacity * TRAIN_POS_COMPONENTS;
    const scale_elements = capacity * TRAIN_SCALE_COMPONENTS;
    const rot_elements = capacity * TRAIN_ROT_COMPONENTS;
    const alpha_splat_elements = capacity;
    if (sh_elements > 0) {
      device.queue.writeBuffer(grad_sh_buffer, 0, new Float32Array(sh_elements));
    }
    if (opacity_elements > 0) {
      device.queue.writeBuffer(grad_opacity_buffer, 0, new Float32Array(opacity_elements));
    }
    if (pos_elements > 0) {
      device.queue.writeBuffer(grad_position_buffer, 0, new Float32Array(pos_elements));
    }
    if (scale_elements > 0) {
      device.queue.writeBuffer(grad_scale_buffer, 0, new Float32Array(scale_elements));
    }
    if (rot_elements > 0) {
      device.queue.writeBuffer(grad_rotation_buffer, 0, new Float32Array(rot_elements));
    }
    if (alpha_splat_elements > 0 && grad_alpha_splat_buffer) {
      device.queue.writeBuffer(grad_alpha_splat_buffer, 0, new Float32Array(alpha_splat_elements));
    }
  }

  function updateShDegreeSchedule() {
    const maxDeg = pc_data.sh_deg;
    if (maxDeg <= 0) return;

    const scheduledDeg = Math.min(maxDeg, Math.floor(trainingIteration / SH_DEGREE_INTERVAL));
    if (scheduledDeg === current_sh_degree) return;

    current_sh_degree = scheduledDeg;
    const shDegArray = new Float32Array([current_sh_degree]);
    device.queue.writeBuffer(settings_buffer, 4, shDegArray);
    console.log('[SH schedule] degree ->', current_sh_degree);
  }

  function trainStep(encoder: GPUCommandEncoder) {
    if (training_camera_buffers.length > 0) {
      current_training_camera_index = trainingIteration % training_camera_buffers.length;
    }

    trainingIteration++;
    const cameraInfo = training_camera_buffers.length > 0 
      ? ` (camera ${current_training_camera_index + 1}/${training_camera_buffers.length}${training_images.length > 0 ? `, image ${current_training_camera_index + 1}/${training_images.length}` : ''})` 
      : '';
    console.log(`Training iteration ${trainingIteration}${cameraInfo}...`);

    setActiveCount(active_count);
    zeroGradBuffers();

    updateShDegreeSchedule();

    // Forward -> Loss -> Backward -> Adam (tiled path is now canonical)
    run_training_tiled_forward(encoder);
    run_training_loss(encoder);
    run_training_tiled_backward(encoder);
    run_training_backward_geom(encoder);
    run_adam_updates(encoder);
    rebuildTrainingApplyParamsBG();
    run_training_apply_params(encoder);

    camera.needsPreprocess = true;
  }

  async function train(numFrames: number): Promise<void> {
    if (!tilesBuilt) {
      console.log('[train] Building training tiles before scheduling training...');
      try {
        await buildTileIntersections();
      } catch (err) {
        console.error('Error while building training tiles:', err);
        return;
      }
      if (!tilesBuilt) {
        console.warn('Training tiles build did not complete successfully; skipping training request.');
        return;
      }
    }
    trainingCount += numFrames;
  }

  async function debugTrainOnce(label: string = 'debugTrainOnce') : Promise<void> {
    if (active_count === 0) {
      console.warn('[debugTrainOnce] No active gaussians.');
      return;
    }
    if (!tilesBuilt) {
      console.log('[debugTrainOnce] Building training tiles before debug step...');
      try {
        await buildTileIntersections();
      } catch (err) {
        console.error('[debugTrainOnce] Error while building tiles:', err);
        return;
      }
      if (!tilesBuilt) {
        console.warn('[debugTrainOnce] Tiles failed to build; aborting debug step.');
        return;
      }
    }

    const encoder = device.createCommandEncoder();
    trainStep(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await debugSampleTrainingState(label + ` (iter ${trainingIteration})`);
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
    console.log(`Loaded ${cameras.length} training cameras`);
  }

  function setTrainingImages(images: Array<{ view: GPUTextureView; sampler: GPUSampler }>) {
    training_images = images;
    
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
    debugTrainOnce,
    buildTrainingTiles: buildTileIntersections,
  };
}
