import { PointCloud } from '../utils/load';
import { C_SIZE_3D_GAUSSIAN } from '../utils/pointcloud-loader';
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
  debugTrainOnce?(label?: string): Promise<void>;
  debugDumpProjection?(label?: string): Promise<void>;
  debugTrainWithDiagnostics?(): Promise<void>;
  collectGradientDiagnostics?(): Promise<any>;
  tracePixelCompositing?(pixelX: number, pixelY: number): Promise<any>;
  debugTracePixel?(pixelX: number, pixelY: number): Promise<void>;
  getTileCoverageStats?(): any;
  setTargetTexture(view: GPUTextureView, sampler: GPUSampler): void;
  initializeKnn(encoder: GPUCommandEncoder): void;
  initializeKnnAndSync(): Promise<void>;
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

// Training hyperparameters
const SH_DEGREE_INTERVAL = 100;

// Learning rates
const MEANS_LR = 0.0;
const SHS_LR = 2e-1;
const OPACITY_LR = 2e-1;
const SCALING_LR = 1e1;
const ROTATION_LR = 5e-3;

// Training parameter layout
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
  // Combined geometry gradients: [conic_a, conic_b, conic_c, mean2d_x, mean2d_y]
  let grad_geom_buffer: GPUBuffer | null = null;
  const GEOM_GRAD_COMPONENTS = 5;

  // Bind Groups
  let sort_bind_group: GPUBindGroup | null = null;
  let sorted_bg: GPUBindGroup[] = [];
  let render_splat_bg: GPUBindGroup | null = null;
  let preprocess_bg: GPUBindGroup | null = null;
  let knn_init_bind_group: GPUBindGroup | null = null;
  let prune_bind_group: GPUBindGroup | null = null;
  let densify_bind_group: GPUBindGroup | null = null;

  // Debug: projection inspection from preprocess.wgsl
  let debug_projection_buffer: GPUBuffer | null = null;

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
  const GAUSS_PROJECTION_BYTES = 48;
  let tiling_params_buffer: GPUBuffer | null = null;
  let gauss_projections_buffer: GPUBuffer | null = null;

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
  let tilesBuiltForCameraIndex = -1;

  let lastTileCoverageStats: {
    numFrustumPass: number;
    numTileAssigned: number;
    totalIntersections: number;
    avgTilesPerSplat: number;
    maxTilesPerSplat: number;
    tileWidth: number;
    tileHeight: number;
    totalTiles: number;
  } | null = null;

  // Async training loop state
  let isTraining = false;

  function getCurrentTrainingCameraBuffer(): GPUBuffer {
    if (training_camera_buffers.length > 0 &&
        current_training_camera_index >= 0 &&
        current_training_camera_index < training_camera_buffers.length) {
      return training_camera_buffers[current_training_camera_index];
    }
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

    const DEBUG_PROJ_STRIDE_BYTES = 2 * 4 * FLOAT32_BYTES;
    debug_projection_buffer = device.createBuffer({
      label: 'debug projection (pos_cam + clip)',
      size: newCapacity * DEBUG_PROJ_STRIDE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

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
    grad_geom_buffer = device.createBuffer({
      label: 'grad geom [conic_a,b,c, mean2d_x,y] (float32)',
      size: newCapacity * GEOM_GRAD_COMPONENTS * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    gauss_projections_buffer = device.createBuffer({
      label: 'tiling gauss_projections (GaussProjection)',
      size: newCapacity * GAUSS_PROJECTION_BYTES,
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
    if (!debug_projection_buffer) return;
    preprocess_bg = device.createBindGroup({
      label: 'preprocess g1',
      layout: preprocess_pipeline.getBindGroupLayout(1),
      entries: [
        { binding: 0, resource: { buffer: gaussian_buffer } },
        { binding: 1, resource: { buffer: sh_buffer } },
        { binding: 2, resource: { buffer: splat_buffer } },
        { binding: 3, resource: { buffer: debug_projection_buffer } },
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
  }

  function rebuildTrainingBackwardGeomBG() {
    if (!gaussian_buffer ||
        !grad_position_buffer || !grad_scale_buffer || !grad_rotation_buffer ||
        !grad_geom_buffer || !active_count_uniform_buffer) {
      return;
    }

    training_backward_geom_bind_group = device.createBindGroup({
      label: 'training backward geom bind group',
      layout: training_backward_geom_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: getCurrentTrainingCameraBuffer() } },
        { binding: 1, resource: { buffer: gaussian_buffer } },
        { binding: 2, resource: { buffer: active_count_uniform_buffer } },
        { binding: 3, resource: { buffer: grad_geom_buffer } },
        { binding: 4, resource: { buffer: grad_position_buffer } },
        { binding: 5, resource: { buffer: grad_scale_buffer } },
        { binding: 6, resource: { buffer: grad_rotation_buffer } },
      ],
    });
  }

  function rebuildTrainingTiledForwardBG() {
    if (!training_tiled_forward_pipeline) return;
    if (!sh_buffer || !tiling_params_buffer) return;
    if (!gauss_projections_buffer) return;
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
        { binding: 3, resource: { buffer: gauss_projections_buffer } },
        { binding: 4, resource: { buffer: tile_offsets_buffer } },
        { binding: 5, resource: { buffer: flatten_ids_buffer } },
        { binding: 6, resource: training_color_view },
        { binding: 7, resource: training_alpha_view },
        { binding: 8, resource: { buffer: training_last_ids_buffer } },
        { binding: 9, resource: { buffer: sh_buffer } },
        { binding: 10, resource: { buffer: background_params_buffer! } },
        { binding: 11, resource: { buffer: tile_mask_buffer! } },
        { binding: 12, resource: { buffer: tiles_cumsum_buffer } },
        { binding: 13, resource: { buffer: active_count_uniform_buffer } },
      ],
    });
  }

  function rebuildTrainingTiledBackwardBG() {
    if (!training_tiled_backward_pipeline) return;
    if (!sh_buffer || !tiling_params_buffer) return;
    if (!gauss_projections_buffer) return;
    if (!residual_color_view || !training_last_ids_buffer) return;
    if (!tile_offsets_buffer || !flatten_ids_buffer || !tile_mask_buffer) return;
    if (!grad_sh_buffer || !grad_opacity_buffer || !grad_geom_buffer || !active_count_uniform_buffer) return;
    if (!tiles_cumsum_buffer) return;
    if (!training_alpha_view) return;

    training_tiled_backward_bind_group = device.createBindGroup({
      label: 'training tiled backward bind group',
      layout: training_tiled_backward_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: getCurrentTrainingCameraBuffer() } },
        { binding: 1, resource: { buffer: settings_buffer } },
        { binding: 2, resource: { buffer: tiling_params_buffer } },
        { binding: 3, resource: { buffer: sh_buffer } },
        { binding: 4, resource: { buffer: gauss_projections_buffer } },
        { binding: 5, resource: residual_color_view },
        { binding: 6, resource: training_alpha_view },
        { binding: 7, resource: { buffer: training_last_ids_buffer } },
        { binding: 8, resource: { buffer: tile_offsets_buffer } },
        { binding: 9, resource: { buffer: flatten_ids_buffer } },
        { binding: 10, resource: { buffer: grad_sh_buffer } },
        { binding: 11, resource: { buffer: grad_opacity_buffer } },
        { binding: 12, resource: { buffer: active_count_uniform_buffer } },
        { binding: 13, resource: { buffer: grad_geom_buffer } },
        { binding: 14, resource: { buffer: background_params_buffer! } },
        { binding: 15, resource: { buffer: tile_mask_buffer! } },
        { binding: 16, resource: { buffer: tiles_cumsum_buffer } },
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
        !gauss_projections_buffer ||
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
        { binding: 4, resource: { buffer: gauss_projections_buffer } },
        { binding: 5, resource: { buffer: projection_params_buffer } },
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

    if (!gauss_projections_buffer) {
      return;
    }
    if (active_count === 0) {
      ensureIntersectionBuffers(0);
      return;
    }

    const encoder = device.createCommandEncoder();
    run_training_tiles_projection(encoder);

    const projectionsSizeBytes = active_count * GAUSS_PROJECTION_BYTES;
    const staging = device.createBuffer({
      label: 'tiling gauss_projections staging',
      size: projectionsSizeBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(gauss_projections_buffer, 0, staging, 0, projectionsSizeBytes);
    device.queue.submit([encoder.finish()]);

    await device.queue.onSubmittedWorkDone();
    await staging.mapAsync(GPUMapMode.READ);
    const mapped = staging.getMappedRange();
    const dataView = new DataView(mapped);
    const countsView = new Uint32Array(active_count);
    for (let i = 0; i < active_count; i++) {
      countsView[i] = dataView.getUint32(i * GAUSS_PROJECTION_BYTES + 28, true);
    }

    const cumsum = new Uint32Array(active_count);
    let running = 0;
    let numFrustumPass = 0;
    let maxTilesPerSplat = 0;
    for (let i = 0; i < active_count; i++) {
      const tileCount = countsView[i];
      running += tileCount;
      cumsum[i] = running;
      if (tileCount > 0) {
        numFrustumPass++;
        if (tileCount > maxTilesPerSplat) {
          maxTilesPerSplat = tileCount;
        }
      }
    }
    staging.unmap();
    staging.destroy();

    const totalIntersections = running;

    const tileWidth = Math.ceil(canvas.width / TILE_SIZE);
    const tileHeight = Math.ceil(canvas.height / TILE_SIZE);

    lastTileCoverageStats = {
      numFrustumPass,
      numTileAssigned: numFrustumPass,
      totalIntersections,
      avgTilesPerSplat: numFrustumPass > 0 ? totalIntersections / numFrustumPass : 0,
      maxTilesPerSplat,
      tileWidth,
      tileHeight,
      totalTiles: tileWidth * tileHeight,
    };

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
      tilesBuilt = true;
      return;
    }

    if (!training_tiles_emit_pipeline ||
        !gauss_projections_buffer ||
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
        { binding: 4, resource: { buffer: gauss_projections_buffer! } },
        { binding: 6, resource: { buffer: tiles_cumsum_buffer! } },
        { binding: 7, resource: { buffer: isect_ids_buffer! } },
        { binding: 8, resource: { buffer: flatten_ids_buffer! } },
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

    const nTiles = Math.ceil(canvas.width / TILE_SIZE) * Math.ceil(canvas.height / TILE_SIZE);
    const tileOffsetsInit = new Uint32Array(nTiles + 1);
    tileOffsetsInit.fill(totalIntersections);
    device.queue.writeBuffer(tile_offsets_buffer!, 0, tileOffsetsInit);

    await device.queue.onSubmittedWorkDone();

    const offsetsEncoder = device.createCommandEncoder();

    const offsetsBindGroup = device.createBindGroup({
      label: 'training tiles offsets bind group',
      layout: training_tiles_offsets_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 2, resource: { buffer: active_count_uniform_buffer! } },
        { binding: 3, resource: { buffer: tiling_params_buffer! } },
        { binding: 6, resource: { buffer: tiles_cumsum_buffer! } },
        { binding: 7, resource: { buffer: isect_ids_buffer! } },
        { binding: 9, resource: { buffer: tile_offsets_buffer! } },
      ],
    });

    const offsets_wg = Math.ceil(Math.max(1, totalIntersections) / WORKGROUP_SIZE);

    {
      const pass = offsetsEncoder.beginComputePass({ label: 'training tiles offsets' });
      pass.setPipeline(training_tiles_offsets_pipeline);
      pass.setBindGroup(0, offsetsBindGroup);
      pass.dispatchWorkgroups(offsets_wg);
      pass.end();
    }

    device.queue.submit([offsetsEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    const offsetsReadBuffer = device.createBuffer({
      label: 'tile offsets readback',
      size: (nTiles + 1) * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const readEncoder = device.createCommandEncoder();
    readEncoder.copyBufferToBuffer(tile_offsets_buffer!, 0, offsetsReadBuffer, 0, (nTiles + 1) * 4);
    device.queue.submit([readEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await offsetsReadBuffer.mapAsync(GPUMapMode.READ);
    const offsetsData = new Uint32Array(offsetsReadBuffer.getMappedRange().slice(0));
    offsetsReadBuffer.unmap();
    offsetsReadBuffer.destroy();

    for (let t = nTiles - 1; t >= 0; t--) {
      if (offsetsData[t] > offsetsData[t + 1]) {
        offsetsData[t] = offsetsData[t + 1];
      }
    }

    device.queue.writeBuffer(tile_offsets_buffer!, 0, offsetsData);

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
    rebuildKnnInitBG();  // Ensure bind group is current
    if (!knn_init_bind_group) {
      console.warn("k-NN init has no bind group - buffers not ready");
      return;
    }
    console.log(`Running k-NN initialization for ${active_count} gaussians...`);
    const num = active_count;
    const num_wg = Math.ceil(num / WORKGROUP_SIZE);

    const pass = encoder.beginComputePass({ label: 'knn init' });
    pass.setPipeline(knn_init_pipeline);
    pass.setBindGroup(0, knn_init_bind_group);
    pass.dispatchWorkgroups(num_wg);
    pass.end();
  }


  function syncTrainingParams(encoder: GPUCommandEncoder) {
    rebuildTrainingInitParamsBG();
    if (!training_init_params_bind_group) {
      console.warn("training_init_params_bind_group not ready");
      return;
    }
    console.log("Syncing training params from gaussians buffer...");
    run_training_init_params(encoder);
  }
  
  async function initializeKnnAndSync(): Promise<void> {
    const encoder = device.createCommandEncoder();
    initializeKnn(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    
    const syncEncoder = device.createCommandEncoder();
    syncTrainingParams(syncEncoder);
    device.queue.submit([syncEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    
    console.log("k-NN initialization and training param sync complete.");
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
    if (!gauss_projections_buffer || !tile_offsets_buffer || !flatten_ids_buffer) {
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

    if (pos_elements > 0 && lr_means > 0) {
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

    const pass = encoder.beginComputePass({ label: 'training apply params' });
    pass.setPipeline(training_apply_params_pipeline);
    pass.setBindGroup(0, training_apply_params_bind_group);
    pass.dispatchWorkgroups(num_wg);
    pass.end();
  }

  // ===============================================
  // Training
  // ===============================================
  async function debugDumpProjection(label: string = 'proj') : Promise<void> {
    if (!debug_projection_buffer) {
      console.warn('[debugDumpProjection] debug_projection_buffer is null.');
      return;
    }
    if (active_count === 0) {
      console.warn('[debugDumpProjection] No active gaussians.');
      return;
    }

    if (!sorter || !preprocess_bg) {
      console.warn('[debugDumpProjection] sorter or preprocess_bg not ready.');
      return;
    }

    const encoder = device.createCommandEncoder();
    run_preprocess(encoder);

    const maxDebug = Math.min(active_count, 32);
    const DEBUG_PROJ_STRIDE_BYTES = 2 * 4 * FLOAT32_BYTES;
    const bytes = maxDebug * DEBUG_PROJ_STRIDE_BYTES;
    const staging = device.createBuffer({
      label: 'debug projection staging',
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    encoder.copyBufferToBuffer(debug_projection_buffer, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();

    const floats = new Float32Array(copy);
    const perEntry = 8;
    const entries: Array<{ idx: number; pos_cam: number[]; clip: number[] }> = [];
    for (let i = 0; i < maxDebug; i++) {
      const base = i * perEntry;
      const pos_cam = Array.from(floats.subarray(base, base + 4));
      const clip = Array.from(floats.subarray(base + 4, base + 8));
      entries.push({ idx: i, pos_cam, clip });
    }
    console.log(`[debugDumpProjection] ${label} first ${maxDebug} gaussians:`, entries);
  }
  async function debugSampleTrainingState(label: string) {
    const sampleCount = Math.min(active_count, 8);
    if (active_count === 0) return;

    if (!train_opacity_buffer || !grad_opacity_buffer || !grad_geom_buffer) {
      return;
    }

    const sampleBytes = sampleCount * FLOAT32_BYTES;
    const fullScalarBytes = active_count * FLOAT32_BYTES;

    const scaleElements = active_count * TRAIN_SCALE_COMPONENTS;
    const scaleBytes = scaleElements * FLOAT32_BYTES;

    const shComponents = TRAIN_SH_COMPONENTS;
    const shElements = active_count * shComponents;
    const shBytes = shElements * FLOAT32_BYTES;

    const encoder = device.createCommandEncoder();

    const makeStaging = (dbgLabel: string, size: number) =>
      device.createBuffer({
        label: dbgLabel,
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

    const stTrainOpacity = makeStaging('debug train_opacity', sampleBytes);
    const stGradOpacity = makeStaging('debug grad_opacity_full', fullScalarBytes);
    const gradGeomBytes = active_count * GEOM_GRAD_COMPONENTS * FLOAT32_BYTES;
    const stGradGeom = makeStaging('debug grad_geom_full', gradGeomBytes);

    encoder.copyBufferToBuffer(train_opacity_buffer, 0, stTrainOpacity, 0, sampleBytes);
    encoder.copyBufferToBuffer(grad_opacity_buffer, 0, stGradOpacity, 0, fullScalarBytes);
    encoder.copyBufferToBuffer(grad_geom_buffer!, 0, stGradGeom, 0, gradGeomBytes);

    let stTrainScale: GPUBuffer | null = null;
    let stGradScale: GPUBuffer | null = null;
    const haveScaleBuffers = !!train_scale_buffer && !!grad_scale_buffer && scaleElements > 0;
    const haveShBuffers = !!train_sh_buffer && !!grad_sh_buffer && shElements > 0;
    if (haveScaleBuffers) {
      stTrainScale = makeStaging('debug train_scale_full', scaleBytes);
      stGradScale = makeStaging('debug grad_scale_full', scaleBytes);
      encoder.copyBufferToBuffer(train_scale_buffer!, 0, stTrainScale, 0, scaleBytes);
      encoder.copyBufferToBuffer(grad_scale_buffer!, 0, stGradScale, 0, scaleBytes);
    }

    let stTrainSH: GPUBuffer | null = null;
    let stGradSH: GPUBuffer | null = null;
    if (haveShBuffers) {
      stTrainSH = makeStaging('debug train_sh_full', shBytes);
      stGradSH = makeStaging('debug grad_sh_full', shBytes);
      encoder.copyBufferToBuffer(train_sh_buffer!, 0, stTrainSH, 0, shBytes);
      encoder.copyBufferToBuffer(grad_sh_buffer!, 0, stGradSH, 0, shBytes);
    }

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
      const nonZeroIndices: number[] = [];
      for (let i = 0; i < active_count; i++) {
        const v = gradOpArrFull[i];
        if (v !== 0) {
          nonZeroOp++;
          nonZeroIndices.push(i);
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
      
      if (nonZeroIndices.length > 0) {
        const stTrainOpFull = makeStaging('debug train_opacity_full', fullScalarBytes);
        const enc2 = device.createCommandEncoder();
        enc2.copyBufferToBuffer(train_opacity_buffer, 0, stTrainOpFull, 0, fullScalarBytes);
        device.queue.submit([enc2.finish()]);
        await device.queue.onSubmittedWorkDone();
        await stTrainOpFull.mapAsync(GPUMapMode.READ);
        const trainOpFull = new Float32Array(stTrainOpFull.getMappedRange().slice(0));
        stTrainOpFull.unmap();
        stTrainOpFull.destroy();
        
        const samplesAtNonZero: Array<{ idx: number; train_op: number; grad_op: number }> = [];
        for (let k = 0; k < Math.min(5, nonZeroIndices.length); k++) {
          const idx = nonZeroIndices[k];
          samplesAtNonZero.push({
            idx,
            train_op: trainOpFull[idx],
            grad_op: gradOpArrFull[idx],
          });
        }
        console.log(
          `[training debug] ${label} train_opacity at non-zero grad indices =`,
          samplesAtNonZero,
        );
      }

      await stGradGeom.mapAsync(GPUMapMode.READ);
      {
        const geomArr = new Float32Array(stGradGeom.getMappedRange().slice(0));

        let nonZeroConic = 0;
        let nonZeroMean2d = 0;
        let maxAbsConic = 0;
        let maxAbsMean2d = 0;
        const examplesGeom: Array<{ idx: number; conic: [number, number, number]; mean2d: [number, number] }> = [];

        for (let i = 0; i < active_count; i++) {
          const base = i * GEOM_GRAD_COMPONENTS;
          const conic_a = geomArr[base + 0];
          const conic_b = geomArr[base + 1];
          const conic_c = geomArr[base + 2];
          const mean2d_x = geomArr[base + 3];
          const mean2d_y = geomArr[base + 4];

          const conicMag = Math.abs(conic_a) + Math.abs(conic_b) + Math.abs(conic_c);
          const mean2dMag = Math.abs(mean2d_x) + Math.abs(mean2d_y);

          if (conicMag > 1e-10) {
            nonZeroConic++;
            maxAbsConic = Math.max(maxAbsConic, conicMag);
          }
          if (mean2dMag > 1e-10) {
            nonZeroMean2d++;
            maxAbsMean2d = Math.max(maxAbsMean2d, mean2dMag);
          }

          if (examplesGeom.length < 5 && (conicMag > 1e-10 || mean2dMag > 1e-10)) {
            examplesGeom.push({
              idx: i,
              conic: [conic_a, conic_b, conic_c],
              mean2d: [mean2d_x, mean2d_y],
            });
          }
        }

        console.log(
          `[training debug] ${label} grad_geom stats: conic nonZero=${nonZeroConic}/${active_count} maxAbs=${maxAbsConic.toExponential(3)}, ` +
          `mean2d nonZero=${nonZeroMean2d}/${active_count} maxAbs=${maxAbsMean2d.toExponential(3)}`,
        );
        console.log(
          `[training debug] ${label} grad_geom examples =`,
          examplesGeom.map(e => ({
            idx: e.idx,
            conic: e.conic.map(v => v.toExponential(2)),
            mean2d: e.mean2d.map(v => v.toExponential(2)),
          })),
        );
      }
      stGradGeom.unmap();

      if (haveScaleBuffers && stTrainScale && stGradScale) {
        await stTrainScale.mapAsync(GPUMapMode.READ);
        const scaleArr = new Float32Array(stTrainScale.getMappedRange().slice(0));
        stTrainScale.unmap();

        await stGradScale.mapAsync(GPUMapMode.READ);
        const gradScaleArr = new Float32Array(stGradScale.getMappedRange().slice(0));
        stGradScale.unmap();

        const totalScaleEntries = active_count * TRAIN_SCALE_COMPONENTS;
        let minLog = Number.POSITIVE_INFINITY;
        let maxLog = Number.NEGATIVE_INFINITY;
        let sumLog = 0;

        let minSigma = Number.POSITIVE_INFINITY;
        let maxSigma = Number.NEGATIVE_INFINITY;
        let sumSigma = 0;

        let maxAbsGradScale = 0;

        for (let i = 0; i < active_count; i++) {
          const base = i * TRAIN_SCALE_COMPONENTS;
          for (let c = 0; c < TRAIN_SCALE_COMPONENTS; c++) {
            const logv = scaleArr[base + c];
            if (Number.isFinite(logv)) {
              if (logv < minLog) minLog = logv;
              if (logv > maxLog) maxLog = logv;
              sumLog += logv;

              const sigma = Math.exp(logv);
              if (sigma < minSigma) minSigma = sigma;
              if (sigma > maxSigma) maxSigma = sigma;
              sumSigma += sigma;
            }

            const gv = gradScaleArr[base + c];
            const ag = Math.abs(gv);
            if (ag > maxAbsGradScale) maxAbsGradScale = ag;
          }
        }

        const meanLog = totalScaleEntries > 0 ? sumLog / totalScaleEntries : 0;
        const meanSigma = totalScaleEntries > 0 ? sumSigma / totalScaleEntries : 0;

        console.log(
          `[training debug] ${label} train_scale (log_sigma) stats: min=${minLog}, max=${maxLog}, mean=${meanLog}`,
        );
        console.log(
          `[training debug] ${label} train_scale (sigma) stats: min=${minSigma}, max=${maxSigma}, mean=${meanSigma}`,
        );
        console.log(
          `[training debug] ${label} grad_scale stats: maxAbs=${maxAbsGradScale}`,
        );

        const scaleSamples: Array<{ idx: number; log_sigma: number[]; sigma: number[] }> = [];
        const gradScaleSamples: Array<{ idx: number; grad: number[] }> = [];
        const sampleGauss = Math.min(sampleCount, active_count);
        for (let i = 0; i < sampleGauss; i++) {
          const base = i * TRAIN_SCALE_COMPONENTS;
          const ls = [
            scaleArr[base + 0],
            scaleArr[base + 1],
            scaleArr[base + 2],
          ];
          const sig = ls.map((v) => Math.exp(v));
          scaleSamples.push({ idx: i, log_sigma: ls, sigma: sig });

          const gs = [
            gradScaleArr[base + 0],
            gradScaleArr[base + 1],
            gradScaleArr[base + 2],
          ];
          gradScaleSamples.push({ idx: i, grad: gs });
        }
        console.log(
          `[training debug] ${label} train_scale[0:${sampleGauss}] (per-gaussian log_sigma, sigma) =`,
          scaleSamples,
        );
        console.log(
          `[training debug] ${label} grad_scale[0:${sampleGauss}] =`,
          gradScaleSamples,
        );
        
        const nonZeroScaleIndices: number[] = [];
        for (let i = 0; i < active_count; i++) {
          const base = i * TRAIN_SCALE_COMPONENTS;
          if (gradScaleArr[base] !== 0 || gradScaleArr[base+1] !== 0 || gradScaleArr[base+2] !== 0) {
            nonZeroScaleIndices.push(i);
            if (nonZeroScaleIndices.length >= 10) break;
          }
        }
        if (nonZeroScaleIndices.length > 0) {
          const nonZeroScaleSamples: Array<{ idx: number; log_sigma: number[]; grad: number[] }> = [];
          for (const idx of nonZeroScaleIndices) {
            const base = idx * TRAIN_SCALE_COMPONENTS;
            nonZeroScaleSamples.push({
              idx,
              log_sigma: [scaleArr[base], scaleArr[base+1], scaleArr[base+2]],
              grad: [gradScaleArr[base], gradScaleArr[base+1], gradScaleArr[base+2]],
            });
          }
          console.log(
            `[training debug] ${label} train_scale at non-zero grad indices (${nonZeroScaleIndices.length} found) =`,
            nonZeroScaleSamples,
          );
        }
      }

      if (haveShBuffers && stTrainSH && stGradSH) {
        await stTrainSH.mapAsync(GPUMapMode.READ);
        const shArr = new Float32Array(stTrainSH.getMappedRange().slice(0));
        stTrainSH.unmap();

        await stGradSH.mapAsync(GPUMapMode.READ);
        const gradShArr = new Float32Array(stGradSH.getMappedRange().slice(0));
        stGradSH.unmap();

        const sampleGauss = Math.min(sampleCount, active_count);
        const shSamples: Array<{ idx: number; dc: number[]; grad_dc: number[] }> = [];

        for (let i = 0; i < sampleGauss; i++) {
          const base = i * shComponents;
          const dc = [
            shArr[base + 0],
            shArr[base + 1],
            shArr[base + 2],
          ];
          const gdc = [
            gradShArr[base + 0],
            gradShArr[base + 1],
            gradShArr[base + 2],
          ];
          shSamples.push({ idx: i, dc, grad_dc: gdc });
        }

        let maxAbsShGrad = 0;
        for (let i = 0; i < active_count; i++) {
          const base = i * shComponents;
          const gx = Math.abs(gradShArr[base + 0]);
          const gy = Math.abs(gradShArr[base + 1]);
          const gz = Math.abs(gradShArr[base + 2]);
          const gmax = Math.max(gx, gy, gz);
          if (gmax > maxAbsShGrad) maxAbsShGrad = gmax;
        }

        console.log(
          `[training debug] ${label} SH DC grad stats: maxAbs=${maxAbsShGrad}`,
        );
        console.log(
          `[training debug] ${label} SH DC[0:${sampleGauss}] (value, grad) =`,
          shSamples,
        );
      }
    } finally {
      stTrainOpacity.destroy();
      stGradOpacity.destroy();
      stGradGeom.destroy();
      if (stTrainScale) stTrainScale.destroy();
      if (stGradScale) stGradScale.destroy();
      if (stTrainSH) stTrainSH.destroy();
      if (stGradSH) stGradSH.destroy();
    }
  }

  async function debugDumpSplatRadii(label: string) {
    if (active_count === 0) {
      console.warn('[debugDumpSplatRadii] No active gaussians.');
      return;
    }
    if (!splat_buffer) {
      console.warn('[debugDumpSplatRadii] splat_buffer is null.');
      return;
    }
    if (!sorter || !preprocess_bg) {
      console.warn('[debugDumpSplatRadii] sorter or preprocess_bg not ready.');
      return;
    }

    const sampleCount = Math.min(active_count, 32);
    const strideFloats = BYTES_PER_SPLAT / FLOAT32_BYTES;
    const bytes = sampleCount * BYTES_PER_SPLAT;

    const encoder = device.createCommandEncoder();
    run_preprocess(encoder);

    const staging = device.createBuffer({
      label: 'debug splat radii staging',
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(splat_buffer, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();

    const floats = new Float32Array(copy);

    let minR = Number.POSITIVE_INFINITY;
    let maxR = Number.NEGATIVE_INFINITY;
    let sumR = 0;
    let count = 0;
    const samples: Array<{ idx: number; center_ndc: number[]; radius: number[] }> = [];

    for (let i = 0; i < sampleCount; i++) {
      const base = i * strideFloats;
      const center_ndc = [floats[base + 0], floats[base + 1]];
      const radius = [floats[base + 2], floats[base + 3]];
      const r = Math.max(radius[0], radius[1]);
      if (Number.isFinite(r)) {
        if (r < minR) minR = r;
        if (r > maxR) maxR = r;
        sumR += r;
        count++;
      }
      if (samples.length < 16) {
        samples.push({ idx: i, center_ndc, radius });
      }
    }

    const meanR = count > 0 ? sumR / count : 0;
    console.log(
      `[training debug] ${label} splat radius sample stats (screen-space px): min=${minR}, max=${maxR}, mean=${meanR}, sampleCount=${count}`,
    );
    console.log(
      `[training debug] ${label} splat radius samples[0:${samples.length}] =`,
      samples,
    );
  }


  async function collectGradientDiagnostics(): Promise<{
    iteration: number;
    activeCount: number;
    shDeg: number;
    cameraIndex: number;
    gradStats: Record<string, {
      nonZeroCount: number;
      totalCount: number;
      nonZeroPct: number;
      mean: number;
      max: number;
      min: number;
      std: number;
    }>;
    sampleSplats: Array<{
      idx: number;
      opacity: number;
      gradOpacity: number;
      gradAlphaSplat: number;
      scale: number[];
      gradScale: number[];
      shDC: number[];
      gradShDC: number[];
    }>;
  }> {
    const result = {
      iteration: trainingIteration,
      activeCount: active_count,
      shDeg: current_sh_degree,
      cameraIndex: current_training_camera_index,
      gradStats: {} as Record<string, any>,
      sampleSplats: [] as any[],
    };

    if (active_count === 0) return result;

    const makeStaging = (label: string, size: number) =>
      device.createBuffer({
        label,
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

    const computeStats = (arr: Float32Array | Int32Array, scale: number = 1.0) => {
      let sum = 0, sumSq = 0, max = 0, min = Infinity, nonZeroCount = 0;
      for (let i = 0; i < arr.length; i++) {
        const v = (arr[i] as number) * scale;
        const absV = Math.abs(v);
        sum += v;
        sumSq += v * v;
        if (absV > max) max = absV;
        if (absV < min) min = absV;
        if (absV > 1e-15) nonZeroCount++;
      }
      const mean = arr.length > 0 ? sum / arr.length : 0;
      const variance = arr.length > 0 ? (sumSq / arr.length) - (mean * mean) : 0;
      return {
        nonZeroCount,
        totalCount: arr.length,
        nonZeroPct: arr.length > 0 ? (100 * nonZeroCount / arr.length) : 0,
        mean,
        max,
        min: min === Infinity ? 0 : min,
        std: Math.sqrt(Math.max(0, variance)),
      };
    };

    const encoder = device.createCommandEncoder();
    const stagingBuffers: GPUBuffer[] = [];

    const opacityBytes = active_count * FLOAT32_BYTES;
    const geomGradBytes = active_count * GEOM_GRAD_COMPONENTS * FLOAT32_BYTES;
    const scaleBytes = active_count * TRAIN_SCALE_COMPONENTS * FLOAT32_BYTES;
    const rotBytes = active_count * TRAIN_ROT_COMPONENTS * FLOAT32_BYTES;
    const posBytes = active_count * TRAIN_POS_COMPONENTS * FLOAT32_BYTES;
    const shBytes = active_count * TRAIN_SH_COMPONENTS * FLOAT32_BYTES;

    const stGradOpacity = makeStaging('diag grad_opacity', opacityBytes);
    const stGradGeom = makeStaging('diag grad_geom', geomGradBytes);
    const stGradScale = makeStaging('diag grad_scale', scaleBytes);
    const stGradRot = makeStaging('diag grad_rotation', rotBytes);
    const stGradPos = makeStaging('diag grad_position', posBytes);
    const stGradSH = makeStaging('diag grad_sh', shBytes);
    const stTrainOpacity = makeStaging('diag train_opacity', opacityBytes);
    const stTrainScale = makeStaging('diag train_scale', scaleBytes);
    const stTrainSH = makeStaging('diag train_sh', shBytes);

    stagingBuffers.push(stGradOpacity, stGradGeom, stGradScale, stGradRot, stGradPos, stGradSH,
                        stTrainOpacity, stTrainScale, stTrainSH);

    if (grad_opacity_buffer) encoder.copyBufferToBuffer(grad_opacity_buffer, 0, stGradOpacity, 0, opacityBytes);
    if (grad_geom_buffer) encoder.copyBufferToBuffer(grad_geom_buffer, 0, stGradGeom, 0, geomGradBytes);
    if (grad_scale_buffer) encoder.copyBufferToBuffer(grad_scale_buffer, 0, stGradScale, 0, scaleBytes);
    if (grad_rotation_buffer) encoder.copyBufferToBuffer(grad_rotation_buffer, 0, stGradRot, 0, rotBytes);
    if (grad_position_buffer) encoder.copyBufferToBuffer(grad_position_buffer, 0, stGradPos, 0, posBytes);
    if (grad_sh_buffer) encoder.copyBufferToBuffer(grad_sh_buffer, 0, stGradSH, 0, shBytes);
    if (train_opacity_buffer) encoder.copyBufferToBuffer(train_opacity_buffer, 0, stTrainOpacity, 0, opacityBytes);
    if (train_scale_buffer) encoder.copyBufferToBuffer(train_scale_buffer, 0, stTrainScale, 0, scaleBytes);
    if (train_sh_buffer) encoder.copyBufferToBuffer(train_sh_buffer, 0, stTrainSH, 0, shBytes);

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await stGradOpacity.mapAsync(GPUMapMode.READ);
    const gradOpArr = new Float32Array(stGradOpacity.getMappedRange().slice(0));
    stGradOpacity.unmap();
    result.gradStats['opacity'] = computeStats(gradOpArr);

    await stGradGeom.mapAsync(GPUMapMode.READ);
    const gradGeomArr = new Float32Array(stGradGeom.getMappedRange().slice(0));
    stGradGeom.unmap();
    
    const gradConicArr = new Float32Array(active_count * 3);
    for (let i = 0; i < active_count; i++) {
      gradConicArr[i * 3 + 0] = gradGeomArr[i * GEOM_GRAD_COMPONENTS + 0];
      gradConicArr[i * 3 + 1] = gradGeomArr[i * GEOM_GRAD_COMPONENTS + 1];
      gradConicArr[i * 3 + 2] = gradGeomArr[i * GEOM_GRAD_COMPONENTS + 2];
    }
    result.gradStats['conic'] = computeStats(gradConicArr);
    
    const gradMean2dArr = new Float32Array(active_count * 2);
    for (let i = 0; i < active_count; i++) {
      gradMean2dArr[i * 2 + 0] = gradGeomArr[i * GEOM_GRAD_COMPONENTS + 3];
      gradMean2dArr[i * 2 + 1] = gradGeomArr[i * GEOM_GRAD_COMPONENTS + 4];
    }
    result.gradStats['mean2d'] = computeStats(gradMean2dArr);

    await stGradScale.mapAsync(GPUMapMode.READ);
    const gradScaleArr = new Float32Array(stGradScale.getMappedRange().slice(0));
    stGradScale.unmap();
    result.gradStats['scale'] = computeStats(gradScaleArr);

    await stGradRot.mapAsync(GPUMapMode.READ);
    const gradRotArr = new Float32Array(stGradRot.getMappedRange().slice(0));
    stGradRot.unmap();
    result.gradStats['rotation'] = computeStats(gradRotArr);

    await stGradPos.mapAsync(GPUMapMode.READ);
    const gradPosArr = new Float32Array(stGradPos.getMappedRange().slice(0));
    stGradPos.unmap();
    result.gradStats['position'] = computeStats(gradPosArr);

    await stGradSH.mapAsync(GPUMapMode.READ);
    const gradShArr = new Float32Array(stGradSH.getMappedRange().slice(0));
    stGradSH.unmap();
    const gradShDC = new Float32Array(active_count * 3);
    const gradShHigher = new Float32Array(active_count * (TRAIN_SH_COMPONENTS - 3));
    for (let i = 0; i < active_count; i++) {
      const base = i * TRAIN_SH_COMPONENTS;
      gradShDC[i * 3 + 0] = gradShArr[base + 0];
      gradShDC[i * 3 + 1] = gradShArr[base + 1];
      gradShDC[i * 3 + 2] = gradShArr[base + 2];
      for (let j = 3; j < TRAIN_SH_COMPONENTS; j++) {
        gradShHigher[i * (TRAIN_SH_COMPONENTS - 3) + (j - 3)] = gradShArr[base + j];
      }
    }
    result.gradStats['sh_dc'] = computeStats(gradShDC);
    result.gradStats['sh_higher'] = computeStats(gradShHigher);

    await stTrainOpacity.mapAsync(GPUMapMode.READ);
    const trainOpArr = new Float32Array(stTrainOpacity.getMappedRange().slice(0));
    stTrainOpacity.unmap();

    await stTrainScale.mapAsync(GPUMapMode.READ);
    const trainScaleArr = new Float32Array(stTrainScale.getMappedRange().slice(0));
    stTrainScale.unmap();

    await stTrainSH.mapAsync(GPUMapMode.READ);
    const trainShArr = new Float32Array(stTrainSH.getMappedRange().slice(0));
    stTrainSH.unmap();

    const sampleSize = Math.min(50, active_count);
    const indices = new Set<number>();
    
    for (let i = 0; i < Math.min(10, active_count); i++) indices.add(i);
    
    while (indices.size < sampleSize / 2) {
      indices.add(Math.floor(Math.random() * active_count));
    }
    
    const geomGradMag: Array<[number, number]> = [];
    for (let i = 0; i < active_count; i++) {
      const geomBase = i * GEOM_GRAD_COMPONENTS;
      const conicMag = Math.abs(gradGeomArr[geomBase + 0]) + Math.abs(gradGeomArr[geomBase + 1]) + Math.abs(gradGeomArr[geomBase + 2]);
      geomGradMag.push([i, Math.abs(gradOpArr[i]) + conicMag]);
    }
    geomGradMag.sort((a, b) => b[1] - a[1]);
    for (let i = 0; i < Math.min(sampleSize / 2, geomGradMag.length); i++) {
      indices.add(geomGradMag[i][0]);
    }

    for (const idx of indices) {
      const scaleBase = idx * TRAIN_SCALE_COMPONENTS;
      const shBase = idx * TRAIN_SH_COMPONENTS;
      const geomBase = idx * GEOM_GRAD_COMPONENTS;
      result.sampleSplats.push({
        idx,
        opacity: trainOpArr[idx],
        gradOpacity: gradOpArr[idx],
        gradConic: [gradGeomArr[geomBase + 0], gradGeomArr[geomBase + 1], gradGeomArr[geomBase + 2]],
        gradMean2d: [gradGeomArr[geomBase + 3], gradGeomArr[geomBase + 4]],
        scale: [trainScaleArr[scaleBase], trainScaleArr[scaleBase + 1], trainScaleArr[scaleBase + 2]],
        gradScale: [gradScaleArr[scaleBase], gradScaleArr[scaleBase + 1], gradScaleArr[scaleBase + 2]],
        shDC: [trainShArr[shBase], trainShArr[shBase + 1], trainShArr[shBase + 2]],
        gradShDC: [gradShArr[shBase], gradShArr[shBase + 1], gradShArr[shBase + 2]],
      });
    }

    const conicMagOf = (s: any) => Math.abs(s.gradConic[0]) + Math.abs(s.gradConic[1]) + Math.abs(s.gradConic[2]);
    result.sampleSplats.sort((a, b) => 
      (Math.abs(b.gradOpacity) + conicMagOf(b)) - 
      (Math.abs(a.gradOpacity) + conicMagOf(a))
    );

    for (const buf of stagingBuffers) buf.destroy();

    return result;
  }

  function getTileCoverageStats() {
    return lastTileCoverageStats;
  }

  async function tracePixelCompositing(pixelX: number, pixelY: number): Promise<{
    pixel: { x: number; y: number };
    tile: { id: number; x: number; y: number; start: number; end: number };
    forwardTrace: Array<{
      order: number;
      splatIdx: number;
      depth: number;
      gaussianVal: number;
      opacity: number;
      alphaSplat: number;
      color: [number, number, number];
      accumAlpha: number;
      accumColor: [number, number, number];
      transmittance: number;
    }>;
    backwardTrace: Array<{
      order: number;
      splatIdx: number;
      alphaSplat: number;
      T_before: number;
      accum_rec: [number, number, number];
      dL_dalpha_formula: string;
    }>;
    finalColor: [number, number, number];
    finalAlpha: number;
    backgroundUsed: [number, number, number];
    observation: string;
  } | null> {
    if (!tile_offsets_buffer || !flatten_ids_buffer || !gauss_projections_buffer || 
        !gaussian_buffer || !lastTileCoverageStats) {
      console.warn('[tracePixelCompositing] Required buffers not available');
      return null;
    }

    const tileWidth = lastTileCoverageStats.tileWidth;
    const tileHeight = lastTileCoverageStats.tileHeight;
    const tileSize = TILE_SIZE;

    const tileX = Math.floor(pixelX / tileSize);
    const tileY = Math.floor(pixelY / tileSize);
    if (tileX >= tileWidth || tileY >= tileHeight) {
      console.warn('[tracePixelCompositing] Pixel outside tile grid');
      return null;
    }
    const tileId = tileY * tileWidth + tileX;
    const totalTiles = tileWidth * tileHeight;

    const readBuffer = async (buf: GPUBuffer, size: number, name: string): Promise<ArrayBuffer> => {
      const staging = device.createBuffer({
        label: `trace staging ${name}`,
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(buf, 0, staging, 0, size);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      await staging.mapAsync(GPUMapMode.READ);
      const data = staging.getMappedRange().slice(0);
      staging.unmap();
      staging.destroy();
      return data;
    };

    const offsetsData = await readBuffer(tile_offsets_buffer, (totalTiles + 1) * 4, 'tile_offsets');
    const offsets = new Uint32Array(offsetsData);
    const tileStart = offsets[tileId];
    const tileEnd = offsets[tileId + 1];

    if (tileStart >= tileEnd) {
      return {
        pixel: { x: pixelX, y: pixelY },
        tile: { id: tileId, x: tileX, y: tileY, start: tileStart, end: tileEnd },
        forwardTrace: [],
        backwardTrace: [],
        finalColor: [0, 0, 0],
        finalAlpha: 0,
        backgroundUsed: [0, 0, 0],
        observation: 'Empty tile - no splats assigned',
      };
    }

    const flattenData = await readBuffer(flatten_ids_buffer, lastTileCoverageStats.totalIntersections * 4, 'flatten_ids');
    const flattenIds = new Uint32Array(flattenData);

    // Read merged GaussProjection buffer and extract components
    const projData = await readBuffer(gauss_projections_buffer, active_count * GAUSS_PROJECTION_BYTES, 'gauss_projections');
    const projView = new DataView(projData);
    
    const means2d = new Float32Array(active_count * 2);
    const conics = new Float32Array(active_count * 3);
    const depths = new Float32Array(active_count);
    for (let i = 0; i < active_count; i++) {
      const base = i * GAUSS_PROJECTION_BYTES;
      means2d[i * 2 + 0] = projView.getFloat32(base + 0, true);
      means2d[i * 2 + 1] = projView.getFloat32(base + 4, true);
      depths[i] = projView.getFloat32(base + 12, true);
      conics[i * 3 + 0] = projView.getFloat32(base + 16, true);
      conics[i * 3 + 1] = projView.getFloat32(base + 20, true);
      conics[i * 3 + 2] = projView.getFloat32(base + 24, true);
    }

    const gaussianData = await readBuffer(gaussian_buffer, active_count * C_SIZE_3D_GAUSSIAN, 'gaussians');
    const gaussians = new Uint16Array(gaussianData);

    const f16ToF32 = (u16: number): number => {
      const sign = (u16 >> 15) & 1;
      const exp = (u16 >> 10) & 0x1f;
      const frac = u16 & 0x3ff;
      if (exp === 0) {
        return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
      } else if (exp === 31) {
        return frac ? NaN : (sign ? -Infinity : Infinity);
      }
      return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
    };

    const getGaussianParams = (idx: number) => {
      const stride = C_SIZE_3D_GAUSSIAN / 2;
      const base = idx * stride;
      const posX = f16ToF32(gaussians[base + 0]);
      const posY = f16ToF32(gaussians[base + 1]);
      const posZ = f16ToF32(gaussians[base + 2]);
      const opacityLogit = f16ToF32(gaussians[base + 3]);

      return {
        position: [posX, posY, posZ] as [number, number, number],
        opacityLogit,
        opacity: 1 / (1 + Math.exp(-opacityLogit)),
      };
    };

    const pixelCenter = [pixelX + 0.5, pixelY + 0.5];
    const forwardTrace: Array<any> = [];
    let accumColor: [number, number, number] = [0, 0, 0];
    let accumAlpha = 0;

    const MIN_ALPHA = 1 / 255;
    const MAX_ALPHA = 0.999;
    const T_THRESHOLD = 1e-4;

    for (let i = tileStart; i < tileEnd; i++) {
      const splatIdx = flattenIds[i];
      const centerPx = [means2d[splatIdx * 2], means2d[splatIdx * 2 + 1]];
      const a = conics[splatIdx * 3 + 0];
      const b = conics[splatIdx * 3 + 1];
      const c = conics[splatIdx * 3 + 2];
      const depth = depths[splatIdx];

      const dx = pixelCenter[0] - centerPx[0];
      const dy = pixelCenter[1] - centerPx[1];
      const sigmaOver2 = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy;

      if (sigmaOver2 < 0) continue;

      const gaussianVal = Math.exp(-sigmaOver2);
      const params = getGaussianParams(splatIdx);
      const alphaLocal = params.opacity * gaussianVal;
      const alphaSplat = Math.min(alphaLocal, MAX_ALPHA);

      if (alphaSplat < MIN_ALPHA) continue;

      const oneMinusAlpha = 1 - accumAlpha;
      const contribAlpha = alphaSplat * oneMinusAlpha;

      const approxColor: [number, number, number] = [0.5, 0.5, 0.5];

      accumColor = [
        accumColor[0] + approxColor[0] * contribAlpha,
        accumColor[1] + approxColor[1] * contribAlpha,
        accumColor[2] + approxColor[2] * contribAlpha,
      ];
      accumAlpha += contribAlpha;

      forwardTrace.push({
        order: forwardTrace.length,
        splatIdx,
        depth,
        gaussianVal,
        opacity: params.opacity,
        alphaSplat,
        color: approxColor,
        accumAlpha,
        accumColor: [...accumColor] as [number, number, number],
        transmittance: 1 - accumAlpha,
      });

      if (1 - accumAlpha < T_THRESHOLD) break;
    }

    const backwardTrace: Array<any> = [];
    const bgColor: [number, number, number] = [0, 0, 0];
    
    let T_current = 1 - accumAlpha;
    let accum_rec: [number, number, number] = [...bgColor];
    
    for (let j = forwardTrace.length - 1; j >= 0; j--) {
      const entry = forwardTrace[j];
      const alphaSplat = entry.alphaSplat;
      
      const oneMinusAlpha = Math.max(1 - alphaSplat, 0.0001);
      const T_before = T_current / oneMinusAlpha;
      
      backwardTrace.push({
        order: forwardTrace.length - 1 - j,
        splatIdx: entry.splatIdx,
        alphaSplat,
        T_before,
        accum_rec: [...accum_rec] as [number, number, number],
        dL_dalpha_formula: `(color - accum_rec) * T_before = (color - [${accum_rec.map(c => c.toFixed(3)).join(',')}]) * ${T_before.toFixed(4)}`,
      });
      
      const color = entry.color;
      accum_rec = [
        alphaSplat * color[0] + (1 - alphaSplat) * accum_rec[0],
        alphaSplat * color[1] + (1 - alphaSplat) * accum_rec[1],
        alphaSplat * color[2] + (1 - alphaSplat) * accum_rec[2],
      ];
      
      T_current = T_before;
    }

    const T = 1 - accumAlpha;
    const finalColor: [number, number, number] = [
      accumColor[0] + T * bgColor[0],
      accumColor[1] + T * bgColor[1],
      accumColor[2] + T * bgColor[2],
    ];

    let observation = '';
    if (forwardTrace.length === 0) {
      observation = 'No splats contribute to this pixel.';
    } else {
      observation = `${forwardTrace.length} splats contribute. ` +
        `BACKWARD TRAVERSAL IS NOW BACK-TO-FRONT (indices ${forwardTrace[forwardTrace.length-1]?.splatIdx} → ${forwardTrace[0]?.splatIdx}). ` +
        `Uses accum_rec (accumulated color from behind) instead of fixed background. ` +
        `This matches FastGS behavior for correct gradient computation.`;
    }

    return {
      pixel: { x: pixelX, y: pixelY },
      tile: { id: tileId, x: tileX, y: tileY, start: tileStart, end: tileEnd },
      forwardTrace,
      backwardTrace,
      finalColor,
      finalAlpha: accumAlpha,
      backgroundUsed: bgColor,
      observation,
    };
  }

  async function computePerTileStats(): Promise<{
    nonEmptyTiles: number;
    minSplatsPerTile: number;
    maxSplatsPerTile: number;
    avgSplatsPerTile: number;
    maxSplatsTileId: number;
    maxSplatsTileX: number;
    maxSplatsTileY: number;
  } | null> {
    if (!tile_offsets_buffer || !lastTileCoverageStats) return null;
    
    const totalTiles = lastTileCoverageStats.totalTiles;
    const tileWidth = lastTileCoverageStats.tileWidth;
    const bufferSize = (totalTiles + 1) * 4;
    
    const staging = device.createBuffer({
      label: 'tile_offsets staging for debug',
      size: bufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(tile_offsets_buffer, 0, staging, 0, bufferSize);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    
    await staging.mapAsync(GPUMapMode.READ);
    const offsets = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    
    let nonEmptyTiles = 0;
    let minSplats = Infinity;
    let maxSplats = 0;
    let totalSplats = 0;
    let maxSplatsTileId = -1;
    
    for (let t = 0; t < totalTiles; t++) {
      const start = offsets[t];
      const end = offsets[t + 1];
      const count = end - start;
      if (count > 0) {
        nonEmptyTiles++;
        totalSplats += count;
        if (count < minSplats) minSplats = count;
        if (count > maxSplats) {
          maxSplats = count;
          maxSplatsTileId = t;
        }
      }
    }
    
    return {
      nonEmptyTiles,
      minSplatsPerTile: nonEmptyTiles > 0 ? minSplats : 0,
      maxSplatsPerTile: maxSplats,
      avgSplatsPerTile: nonEmptyTiles > 0 ? totalSplats / nonEmptyTiles : 0,
      maxSplatsTileId,
      maxSplatsTileX: maxSplatsTileId >= 0 ? maxSplatsTileId % tileWidth : -1,
      maxSplatsTileY: maxSplatsTileId >= 0 ? Math.floor(maxSplatsTileId / tileWidth) : -1,
    };
  }

  async function debugTileDuplication(): Promise<{
    tiles: Array<{
      tileId: number;
      tileX: number;
      tileY: number;
      numEntries: number;
      numUnique: number;
      duplicationFactor: number;
      firstIndices: number[];
    }>;
    suggestedDebugPixel: { x: number; y: number; tileId: number } | null;
  } | null> {
    if (!tile_offsets_buffer || !flatten_ids_buffer || !lastTileCoverageStats) {
      console.warn('[debugTileDuplication] Required buffers not available');
      return null;
    }

    const totalTiles = lastTileCoverageStats.totalTiles;
    const tileWidth = lastTileCoverageStats.tileWidth;
    const tileSize = TILE_SIZE;

    const offsetsStaging = device.createBuffer({
      label: 'tile_offsets staging for dup check',
      size: (totalTiles + 1) * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc1 = device.createCommandEncoder();
    enc1.copyBufferToBuffer(tile_offsets_buffer, 0, offsetsStaging, 0, (totalTiles + 1) * 4);
    device.queue.submit([enc1.finish()]);
    await device.queue.onSubmittedWorkDone();
    await offsetsStaging.mapAsync(GPUMapMode.READ);
    const offsets = new Uint32Array(offsetsStaging.getMappedRange().slice(0));
    offsetsStaging.unmap();
    offsetsStaging.destroy();

    const tilesToAnalyze: Array<{ tileId: number; start: number; end: number }> = [];
    let maxSplatsTile = -1;
    let maxSplatsCount = 0;
    const nonEmptyTileIds: number[] = [];

    for (let t = 0; t < totalTiles; t++) {
      const start = offsets[t];
      const end = offsets[t + 1];
      const count = end - start;
      if (count > 0) {
        nonEmptyTileIds.push(t);
        if (count > maxSplatsCount) {
          maxSplatsCount = count;
          maxSplatsTile = t;
        }
      }
    }

    if (maxSplatsTile >= 0) {
      tilesToAnalyze.push({
        tileId: maxSplatsTile,
        start: offsets[maxSplatsTile],
        end: offsets[maxSplatsTile + 1],
      });
    }

    const otherTiles = nonEmptyTileIds.filter(t => t !== maxSplatsTile);
    for (let i = 0; i < Math.min(2, otherTiles.length); i++) {
      const randIdx = Math.floor(Math.random() * otherTiles.length);
      const tileId = otherTiles.splice(randIdx, 1)[0];
      tilesToAnalyze.push({
        tileId,
        start: offsets[tileId],
        end: offsets[tileId + 1],
      });
    }

    if (tilesToAnalyze.length === 0) {
      return { tiles: [], suggestedDebugPixel: null };
    }

    const totalIntersections = lastTileCoverageStats.totalIntersections;
    const flattenStaging = device.createBuffer({
      label: 'flatten_ids staging for dup check',
      size: totalIntersections * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc2 = device.createCommandEncoder();
    enc2.copyBufferToBuffer(flatten_ids_buffer, 0, flattenStaging, 0, totalIntersections * 4);
    device.queue.submit([enc2.finish()]);
    await device.queue.onSubmittedWorkDone();
    await flattenStaging.mapAsync(GPUMapMode.READ);
    const flattenIds = new Uint32Array(flattenStaging.getMappedRange().slice(0));
    flattenStaging.unmap();
    flattenStaging.destroy();

    const results: Array<any> = [];
    let suggestedTile: { tileId: number; tileX: number; tileY: number } | null = null;

    for (const tile of tilesToAnalyze) {
      const numEntries = tile.end - tile.start;
      const indices = new Set<number>();
      const firstIndices: number[] = [];

      for (let i = tile.start; i < tile.end; i++) {
        const idx = flattenIds[i];
        indices.add(idx);
        if (firstIndices.length < 10) {
          firstIndices.push(idx);
        }
      }

      const numUnique = indices.size;
      const duplicationFactor = numUnique > 0 ? numEntries / numUnique : 0;
      const tileX = tile.tileId % tileWidth;
      const tileY = Math.floor(tile.tileId / tileWidth);

      results.push({
        tileId: tile.tileId,
        tileX,
        tileY,
        numEntries,
        numUnique,
        duplicationFactor,
        firstIndices,
      });

      if (!suggestedTile && numUnique >= 5 && numUnique <= 200) {
        suggestedTile = { tileId: tile.tileId, tileX, tileY };
      }
    }

    if (!suggestedTile && results.length > 0) {
      const best = results.reduce((a, b) => 
        (a.numUnique > 0 && a.numUnique < b.numUnique) ? a : b
      );
      suggestedTile = { tileId: best.tileId, tileX: best.tileX, tileY: best.tileY };
    }

    const suggestedDebugPixel = suggestedTile ? {
      x: Math.floor((suggestedTile.tileX + 0.5) * tileSize),
      y: Math.floor((suggestedTile.tileY + 0.5) * tileSize),
      tileId: suggestedTile.tileId,
    } : null;

    return { tiles: results, suggestedDebugPixel };
  }

  async function debugInspectTileKeys(): Promise<{
    totalIntersections: number;
    firstKeys: Array<{ idx: number; key: number; tileId: number; depth: number; gaussIdx: number }>;
    tileIdDistribution: Map<number, number>;
    sortedCorrectly: boolean;
    issues: string[];
  } | null> {
    if (!isect_ids_buffer || !flatten_ids_buffer || !lastTileCoverageStats) {
      return null;
    }

    const totalIntersections = lastTileCoverageStats.totalIntersections;
    if (totalIntersections === 0) {
      return { totalIntersections: 0, firstKeys: [], tileIdDistribution: new Map(), sortedCorrectly: true, issues: [] };
    }

    const readBuffer = async (buf: GPUBuffer, size: number): Promise<ArrayBuffer> => {
      const staging = device.createBuffer({
        label: 'key debug staging',
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(buf, 0, staging, 0, size);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      await staging.mapAsync(GPUMapMode.READ);
      const data = staging.getMappedRange().slice(0);
      staging.unmap();
      staging.destroy();
      return data;
    };

    const isectData = await readBuffer(isect_ids_buffer, totalIntersections * 4);
    const isectIds = new Uint32Array(isectData);

    const flattenData = await readBuffer(flatten_ids_buffer, totalIntersections * 4);
    const flattenIds = new Uint32Array(flattenData);

    const TILE_BITS = 12;
    const DEPTH_BITS = 16;
    const TILE_SHIFT = DEPTH_BITS;
    const TILE_MASK = (1 << TILE_BITS) - 1;
    const DEPTH_MASK = (1 << DEPTH_BITS) - 1;

    const decodeKey = (key: number) => {
      const depth = key & DEPTH_MASK;
      const tileAndCam = key >> TILE_SHIFT;
      const tileId = tileAndCam & TILE_MASK;
      return { tileId, depth };
    };

    const firstKeys: Array<any> = [];
    const tileIdDistribution = new Map<number, number>();
    let sortedCorrectly = true;
    let prevKey = 0;
    const issues: string[] = [];

    for (let i = 0; i < totalIntersections; i++) {
      const key = isectIds[i];
      const { tileId, depth } = decodeKey(key);
      const gaussIdx = flattenIds[i];

      tileIdDistribution.set(tileId, (tileIdDistribution.get(tileId) || 0) + 1);

      if (i > 0 && key < prevKey) {
        sortedCorrectly = false;
        if (issues.length < 5) {
          issues.push(`Sort violation at index ${i}: key ${key} < prev ${prevKey}`);
        }
      }
      prevKey = key;

      if (i < 20) {
        firstKeys.push({ idx: i, key, tileId, depth, gaussIdx });
      }
    }

    const tileWidth = lastTileCoverageStats.tileWidth;
    const tileHeight = lastTileCoverageStats.tileHeight;
    const maxValidTileId = tileWidth * tileHeight - 1;

    let invalidTileCount = 0;
    for (const [tileId, count] of tileIdDistribution) {
      if (tileId > maxValidTileId) {
        invalidTileCount += count;
        if (issues.length < 10) {
          issues.push(`Invalid tileId ${tileId} (max valid: ${maxValidTileId}) with ${count} entries`);
        }
      }
    }

    if (invalidTileCount > 0) {
      issues.push(`Total entries with invalid tileId: ${invalidTileCount}`);
    }

    const tile0Count = tileIdDistribution.get(0) || 0;
    if (tile0Count > totalIntersections * 0.9) {
      issues.push(`WARNING: ${tile0Count}/${totalIntersections} (${(100*tile0Count/totalIntersections).toFixed(1)}%) entries are in tile 0!`);
    }

    return { totalIntersections, firstKeys, tileIdDistribution, sortedCorrectly, issues };
  }

  async function debugTrainWithDiagnostics(): Promise<void> {
    console.log('\n========================================');
    console.log('TRAINING DIAGNOSTICS');
    console.log('========================================\n');

    const nextCameraIndex = training_camera_buffers.length > 0
      ? trainingIteration % training_camera_buffers.length
      : 0;

    if (!tilesBuilt || tilesBuiltForCameraIndex !== nextCameraIndex) {
      current_training_camera_index = nextCameraIndex;
      console.log(`Building tiles for camera ${nextCameraIndex}...`);
      await buildTileIntersections();
      tilesBuiltForCameraIndex = nextCameraIndex;
    }

    if (!tilesBuilt) {
      console.error('Failed to build tiles');
      return;
    }

    console.log('--- Tile Coverage Stats ---');
    const tileStats = getTileCoverageStats();
    if (tileStats) {
      console.log(`  Total gaussians: ${active_count}`);
      console.log(`  Frustum pass (assigned to tiles): ${tileStats.numFrustumPass} (${(100 * tileStats.numFrustumPass / active_count).toFixed(2)}%)`);
      console.log(`  Total (gaussian, tile) intersections: ${tileStats.totalIntersections}`);
      console.log(`  Avg tiles per visible splat: ${tileStats.avgTilesPerSplat.toFixed(2)}`);
      console.log(`  Max tiles for single splat: ${tileStats.maxTilesPerSplat}`);
      console.log(`  Tile grid: ${tileStats.tileWidth} x ${tileStats.tileHeight} = ${tileStats.totalTiles} tiles`);
    }
    
    const perTileStats = await computePerTileStats();
    if (perTileStats) {
      console.log(`  Non-empty tiles: ${perTileStats.nonEmptyTiles} (${(100 * perTileStats.nonEmptyTiles / (tileStats?.totalTiles || 1)).toFixed(2)}%)`);
      console.log(`  Splats per non-empty tile: min=${perTileStats.minSplatsPerTile}, max=${perTileStats.maxSplatsPerTile}, avg=${perTileStats.avgSplatsPerTile.toFixed(2)}`);
      if (perTileStats.maxSplatsTileId >= 0) {
        console.log(`  Max-splats tile: id=${perTileStats.maxSplatsTileId} at (${perTileStats.maxSplatsTileX}, ${perTileStats.maxSplatsTileY})`);
      }
    }
    console.log('');

    console.log('--- Tile Duplication Analysis ---');
    const tileDupStats = await debugTileDuplication();
    if (tileDupStats) {
      for (const t of tileDupStats.tiles) {
        const dupWarning = t.duplicationFactor > 1.1 ? ' ⚠️ DUPLICATES!' : '';
        console.log(`  Tile ${t.tileId} (${t.tileX}, ${t.tileY}): entries=${t.numEntries}, unique=${t.numUnique}, dupFactor=${t.duplicationFactor.toFixed(2)}${dupWarning}`);
        console.log(`    First 10 indices: [${t.firstIndices.join(', ')}]`);
      }
      if (tileDupStats.suggestedDebugPixel) {
        const sp = tileDupStats.suggestedDebugPixel;
        console.log('');
        console.log(`  ✓ Suggested debug pixel: (${sp.x}, ${sp.y}) in tile ${sp.tileId}`);
        console.log(`    Run: gaussian_renderer.debugTracePixel(${sp.x}, ${sp.y})`);
      }
    }
    console.log('');

    console.log('--- Intersection Key Analysis ---');
    const keyAnalysis = await debugInspectTileKeys();
    if (keyAnalysis) {
      console.log(`  Total intersections: ${keyAnalysis.totalIntersections}`);
      console.log(`  Sorted correctly: ${keyAnalysis.sortedCorrectly ? '✓ YES' : '✗ NO'}`);
      console.log(`  Unique tile IDs: ${keyAnalysis.tileIdDistribution.size}`);
      
      const sortedTiles = [...keyAnalysis.tileIdDistribution.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);
      console.log('  Top 5 tiles by entry count:');
      for (const [tileId, count] of sortedTiles) {
        const tileX = tileId % (tileStats?.tileWidth || 1);
        const tileY = Math.floor(tileId / (tileStats?.tileWidth || 1));
        console.log(`    Tile ${tileId} (${tileX}, ${tileY}): ${count} entries`);
      }

      if (keyAnalysis.issues.length > 0) {
        console.log('  ⚠️ ISSUES DETECTED:');
        for (const issue of keyAnalysis.issues) {
          console.log(`    - ${issue}`);
        }
      }

      console.log('  First 10 keys (after sort):');
      for (const k of keyAnalysis.firstKeys.slice(0, 10)) {
        console.log(`    [${k.idx}] key=${k.key}, tileId=${k.tileId}, depth=${k.depth}, gaussIdx=${k.gaussIdx}`);
      }
    }
    console.log('');

    const encoder = device.createCommandEncoder();
    trainStep(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    const diag = await collectGradientDiagnostics();

    console.log(`Iteration: ${diag.iteration}`);
    console.log(`Active splats: ${diag.activeCount}`);
    console.log(`SH degree: ${diag.shDeg}`);
    console.log(`Camera index: ${diag.cameraIndex}`);
    console.log('');
    console.log('--- Gradient Statistics ---');
    
    for (const [name, stats] of Object.entries(diag.gradStats)) {
      const s = stats as any;
      console.log(
        `${name.padEnd(12)}: nonzero=${s.nonZeroPct.toFixed(1)}% (${s.nonZeroCount}/${s.totalCount}), ` +
        `mean=${s.mean.toExponential(2)}, max=${s.max.toExponential(2)}, std=${s.std.toExponential(2)}`
      );
    }

    console.log('');
    console.log('--- Top 10 splats by gradient magnitude ---');
    for (const s of diag.sampleSplats.slice(0, 10)) {
      const sig = s.scale.map((v: number) => Math.exp(v).toFixed(4));
      console.log(
        `#${s.idx}: op=${(1/(1+Math.exp(-s.opacity))).toFixed(3)}, ` +
        `gradOp=${s.gradOpacity.toExponential(2)}, gradAlpha=${s.gradAlphaSplat.toExponential(2)}, ` +
        `sigma=[${sig.join(',')}], dc=[${s.shDC.map((v: number) => v.toFixed(3)).join(',')}]`
      );
    }

    console.log('');
    console.log('--- Bottom 10 splats (lowest gradients) ---');
    for (const s of diag.sampleSplats.slice(-10).reverse()) {
      const sig = s.scale.map((v: number) => Math.exp(v).toFixed(4));
      console.log(
        `#${s.idx}: op=${(1/(1+Math.exp(-s.opacity))).toFixed(3)}, ` +
        `gradOp=${s.gradOpacity.toExponential(2)}, gradAlpha=${s.gradAlphaSplat.toExponential(2)}, ` +
        `sigma=[${sig.join(',')}], dc=[${s.shDC.map((v: number) => v.toFixed(3)).join(',')}]`
      );
    }

    const fullDiag = {
      ...diag,
      tileCoverage: tileStats,
      perTileStats,
      tileDuplication: tileDupStats,
      keyAnalysis: keyAnalysis ? {
        sortedCorrectly: keyAnalysis.sortedCorrectly,
        uniqueTileIds: keyAnalysis.tileIdDistribution.size,
        issues: keyAnalysis.issues,
        firstKeys: keyAnalysis.firstKeys,
      } : null,
    };

    console.log('');
    console.log('Full diagnostics object available via: window._lastTrainingDiag');
    console.log('');
    console.log('To trace a specific pixel, run:');
    console.log('  gaussian_renderer.tracePixelCompositing(x, y).then(t => console.log(JSON.stringify(t, null, 2)))');
    console.log('Or use the helper:');
    console.log('  gaussian_renderer.debugTracePixel(x, y)');
    (window as any)._lastTrainingDiag = fullDiag;
  }

  async function debugTracePixel(pixelX: number, pixelY: number): Promise<void> {
    console.log(`\n=== Pixel Trace (${pixelX}, ${pixelY}) ===\n`);
    
    const trace = await tracePixelCompositing(pixelX, pixelY);
    if (!trace) {
      console.log('Failed to trace pixel - tiles may not be built');
      return;
    }

    console.log(`Tile: ${trace.tile.x}, ${trace.tile.y} (id=${trace.tile.id})`);
    console.log(`Tile range in flatten_ids: [${trace.tile.start}, ${trace.tile.end}) = ${trace.tile.end - trace.tile.start} entries`);
    console.log(`Splats that actually contribute: ${trace.forwardTrace.length}`);
    console.log('');

    if (trace.forwardTrace.length === 0) {
      console.log('No splats contribute to this pixel.');
      console.log(`Final color: [${trace.finalColor.map(c => c.toFixed(4)).join(', ')}] (background only)`);
      return;
    }

    console.log('--- Forward Compositing Trace ---');
    console.log('Order | SplatIdx | Depth    | Gaussian | Opacity | AlphaSplat | AccumAlpha | Transmittance');
    console.log('------|----------|----------|----------|---------|------------|------------|-------------');
    for (const entry of trace.forwardTrace) {
      console.log(
        `${String(entry.order).padStart(5)} | ` +
        `${String(entry.splatIdx).padStart(8)} | ` +
        `${entry.depth.toFixed(4).padStart(8)} | ` +
        `${entry.gaussianVal.toFixed(4).padStart(8)} | ` +
        `${entry.opacity.toFixed(3).padStart(7)} | ` +
        `${entry.alphaSplat.toFixed(4).padStart(10)} | ` +
        `${entry.accumAlpha.toFixed(4).padStart(10)} | ` +
        `${entry.transmittance.toFixed(4).padStart(11)}`
      );
    }
    console.log('');

    console.log('--- Backward Traversal (BACK-TO-FRONT with accum_rec) ---');
    console.log('Order | SplatIdx | AlphaSplat | T_before   | accum_rec');
    console.log('------|----------|------------|------------|------------------');
    for (const entry of trace.backwardTrace) {
      const accumRecStr = entry.accum_rec ? 
        `[${entry.accum_rec.map((c: number) => c.toFixed(3)).join(',')}]` : 
        '[N/A]';
      console.log(
        `${String(entry.order).padStart(5)} | ` +
        `${String(entry.splatIdx).padStart(8)} | ` +
        `${entry.alphaSplat.toFixed(4).padStart(10)} | ` +
        `${(entry.T_before || 0).toFixed(4).padStart(10)} | ` +
        `${accumRecStr}`
      );
    }
    console.log('');
    console.log('Formula: dL/dalpha = (color - accum_rec) * T_before');
    console.log('');

    console.log(`Final alpha: ${trace.finalAlpha.toFixed(4)}`);
    console.log(`Final color: [${trace.finalColor.map(c => c.toFixed(4)).join(', ')}]`);
    console.log(`Background: [${trace.backgroundUsed.map(c => c.toFixed(4)).join(', ')}]`);
    console.log('');
    console.log('OBSERVATION:');
    console.log(trace.observation);
    console.log('');

    (window as any)._lastPixelTrace = trace;
    console.log('Full trace object available via: window._lastPixelTrace');
  }

  function zeroGradBuffers() {
    if (!grad_sh_buffer || !grad_opacity_buffer ||
        !grad_position_buffer || !grad_scale_buffer || !grad_rotation_buffer) return;
    const sh_elements = capacity * TRAIN_SH_COMPONENTS;
    const opacity_elements = capacity;
    const pos_elements = capacity * TRAIN_POS_COMPONENTS;
    const scale_elements = capacity * TRAIN_SCALE_COMPONENTS;
    const rot_elements = capacity * TRAIN_ROT_COMPONENTS;
    const geom_grad_elements = capacity * GEOM_GRAD_COMPONENTS;
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
    if (geom_grad_elements > 0 && grad_geom_buffer) {
      device.queue.writeBuffer(grad_geom_buffer, 0, new Float32Array(geom_grad_elements));
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

  async function train(numIterations: number): Promise<void> {
    if (numIterations <= 0) return;
    if (isTraining) {
      console.warn('[train] Training already in progress; ignoring new request.');
      return;
    }

    isTraining = true;
    const startTime = performance.now();
    let lastLogTime = startTime;
    
    try {
      for (let i = 0; i < numIterations; i++) {
        const nextCameraIndex = training_camera_buffers.length > 0
          ? (trainingIteration % training_camera_buffers.length)
          : 0;
        
        if (!tilesBuilt || tilesBuiltForCameraIndex !== nextCameraIndex) {
          current_training_camera_index = nextCameraIndex;
          try {
            await buildTileIntersections();
            tilesBuiltForCameraIndex = nextCameraIndex;
          } catch (err) {
            console.error('[train] Error while rebuilding training tiles:', err);
            break;
          }
          if (!tilesBuilt) {
            console.warn('[train] Training tiles build did not complete successfully; aborting training loop.');
            break;
          }
        }

        const encoder = device.createCommandEncoder();
        trainStep(encoder);
        device.queue.submit([encoder.finish()]);
        
        if ((i + 1) % 10 === 0) {
          await device.queue.onSubmittedWorkDone();
          
          const now = performance.now();
          if ((i + 1) % 50 === 0 || (now - lastLogTime) > 5000) {
            const elapsed = (now - startTime) / 1000;
            const iterPerSec = (i + 1) / elapsed;
            console.log(`[train] Iteration ${i + 1}/${numIterations} (${iterPerSec.toFixed(1)} iter/s)`);
            lastLogTime = now;
          }
          
          await new Promise<void>((resolve) => {
            requestAnimationFrame(() => resolve());
          });
        }
      }
      
      await device.queue.onSubmittedWorkDone();
      const totalTime = (performance.now() - startTime) / 1000;
      console.log(`[train] Completed ${numIterations} iterations in ${totalTime.toFixed(1)}s (${(numIterations/totalTime).toFixed(1)} iter/s)`);
      
    } finally {
      isTraining = false;
    }
  }

  async function debugTrainOnce(label: string = 'debugTrainOnce') : Promise<void> {
    if (active_count === 0) {
      console.warn('[debugTrainOnce] No active gaussians.');
      return;
    }

    const nextCameraIndex = training_camera_buffers.length > 0
      ? trainingIteration % training_camera_buffers.length
      : 0;

    if (!tilesBuilt || tilesBuiltForCameraIndex !== nextCameraIndex) {
      current_training_camera_index = nextCameraIndex;
      
      console.log(`[debugTrainOnce] Building training tiles for camera ${nextCameraIndex}...`);
      try {
        await buildTileIntersections();
        tilesBuiltForCameraIndex = nextCameraIndex;
      } catch (err) {
        console.error('[debugTrainOnce] Error while building tiles:', err);
        return;
      }
      if (!tilesBuilt) {
        console.warn('[debugTrainOnce] Tiles failed to build; aborting debug step.');
        return;
      }
    }

    console.log(`[debugTrainOnce] Starting debug training step at iteration=${trainingIteration + 1}, active_count=${active_count}, camera=${nextCameraIndex}`);

    const encoder = device.createCommandEncoder();
    trainStep(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await debugSampleTrainingState(label + ` (iter ${trainingIteration})`);
    await debugDumpSplatRadii(label + ` (iter ${trainingIteration})`);
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
    initializeKnnAndSync,
    prune,
    densify,
    setTrainingCameras,
    setTrainingImages,
    debugTrainOnce,
    debugDumpProjection,
    debugTrainWithDiagnostics,
    collectGradientDiagnostics,
    tracePixelCompositing,
    debugTracePixel,
    getTileCoverageStats,
    buildTrainingTiles: buildTileIntersections,
  };
}
