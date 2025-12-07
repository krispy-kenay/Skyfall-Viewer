import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import commonWGSL from '../shaders/common.wgsl';
import knnInitWGSL from '../shaders/knn-init.wgsl';
import trainingForwardTilesWGSL from '../shaders/training-forward-tiles.wgsl';
import trainingLossWGSL from '../shaders/training-loss.wgsl';
import trainingBackwardTilesWGSL from '../shaders/training-backward-tiles.wgsl';
import backwardTilesLocalWGSL from '../shaders/backward-tiles-local.wgsl';
import backwardReduceWGSL from '../shaders/backward-reduce.wgsl';
import trainingBackwardGeomWGSL from '../shaders/training-backward-geom.wgsl';
import trainingTilesWGSL from '../shaders/training-tiles.wgsl';
import trainingParamsWGSL from '../shaders/training-params.wgsl';
import adamWGSL from '../shaders/adam.wgsl';
import metricWGSL from '../shaders/metric-map.wgsl';
import { get_sorter, C } from '../sort/sort';
import { Renderer } from './renderer';
import { Camera, TrainingCameraData, create_training_camera_uniform_buffer, update_training_camera_uniform_buffer } from '../camera/camera';
import { TrainingProfiler, setGlobalProfiler } from '../utils/training-profiler';
import { TrainingController, PruneParams, DEFAULT_TRAINING_SCHEDULE } from '../utils/training-controller';
import { DensificationManager, GRAD_ACCUM_STRIDE, GRAD_ACCUM_X, GRAD_ACCUM_ABS, GRAD_ACCUM_DENOM } from '../utils/densification';
import { ImportanceManager } from '../utils/importance';
import densifyAccumWGSL from '../shaders/densify-accum.wgsl';
import densifyDecideWGSL from '../shaders/densify-decide.wgsl';
import importanceAccumWGSL from '../shaders/importance-accum.wgsl';
import pruneDecideWGSL from '../shaders/prune-decide.wgsl';

export interface GaussianRenderer extends Renderer {
  setGaussianScale(value: number): void;
  resizeViewport(w: number, h: number): void;
  train(numFrames: number): Promise<void>;
  stopTraining(): void;
  setTargetTexture(view: GPUTextureView, sampler: GPUSampler): void;
  initializeKnn(encoder: GPUCommandEncoder): void;
  initializeKnnAndSync(): Promise<void>;
  prune(customParams?: PruneParams): Promise<number>;
  resetOpacity(maxOpacity?: number): Promise<void>;
  setTrainingCameras(cameras: TrainingCameraData[]): void;
  setTrainingImages(images: Array<{ view: GPUTextureView; sampler: GPUSampler }>): void;
  buildTrainingTiles(): Promise<void>;
  enableProfiling?(enabled: boolean): void;
  printProfilingReport?(): void;
  profiledTrainStep?(): Promise<void>;
  getTrainingController?(): TrainingController | null;
  getSceneExtent?(): number;
  // Training parameter controls
  getTrainingParams(): TrainingParams;
  setTrainingParams(params: Partial<TrainingParams>): void;
  getTrainingStatus(): TrainingStatus;
  getGaussianCount(): number;
  // Loss weight controls
  getLossWeights(): LossWeights;
  setLossWeights(weights: Partial<LossWeights>): void;
  // Diagnostics
  dumpTrainingDiagnostics(): Promise<void>;
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
const MULTI_VIEW_SAMPLE_COUNT = 10;
const MULTI_VIEW_LOSS_THRESH = 0.1;
const MULTI_VIEW_IMPORTANCE_THRESHOLD = 5;

// Training hyperparameters
const SH_DEGREE_INTERVAL = 1000;

// SH (color) update frequency - only update every N iterations
// FastGS updates SH every 16 iterations for first 15k iters, then less
let SH_UPDATE_INTERVAL = 16;

// Learning rates (mutable for runtime adjustment)
// Based on FastGS defaults, scaled up ~2-3x for faster convergence in 10k iterations
// FastGS: position=0.00016, feature=0.0025, opacity=0.025, scale=0.005, rotation=0.001
let MEANS_LR = 0.00016;    // Position
let SHS_LR = 0.0025;       // SH/color
let OPACITY_LR = 0.025;    // Opacity
let SCALING_LR = 0.005;    // Scale
let ROTATION_LR = 0.001;   // Rotation

// Training hyperparameters interface
export interface TrainingParams {
  lr_means: number;
  lr_sh: number;
  lr_opacity: number;
  lr_scale: number;
  lr_rotation: number;
  sh_update_interval: number;  // Only update SH every N iterations
}

// Loss weight interface
export interface LossWeights {
  w_l2: number;
  w_l1: number;
  w_ssim: number;
}

// Training status interface
export interface TrainingStatus {
  isTraining: boolean;
  currentIteration: number;
  targetIterations: number;
  iterationsPerSecond: number;
  gaussianCount: number;
}

interface MultiViewScores {
  importance: Uint32Array;
  pruning: Float32Array;
  sampleCount: number;
}

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
  let active_count_buffer: GPUBuffer | null = null;
  let active_count_uniform_buffer: GPUBuffer | null = null;
  let knn_num_points_buffer: GPUBuffer | null = null;

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
  let metric_map_bind_group: GPUBindGroup | null = null;
  let metric_threshold_bind_group: GPUBindGroup | null = null;
  let metric_accum_bind_group: GPUBindGroup | null = null;

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

  // Wavefront backward pass buffers and pipelines
  const TILE_GRAD_STRIDE = 54;
  let tile_grads_buffer: GPUBuffer | null = null;
  let backward_tiles_local_pipeline: GPUComputePipeline | null = null;
  let backward_reduce_pipeline: GPUComputePipeline | null = null;
  let backward_tiles_local_bind_group: GPUBindGroup | null = null;
  let backward_reduce_bind_group: GPUBindGroup | null = null;
  let useWavefrontBackward = true;

  // Per-pixel last gaussian index for tiled training forward
  let training_last_ids_buffer: GPUBuffer | null = null;
  let tile_sorter: any = null;
  let tile_sort_capacity = 0;

  // Tracking for training tiling state
  let tilesBuilt = false;
  let tilesBuiltForCameraIndex = -1;
  let renderingSuspended = false;

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
  let stopTrainingRequested = false;
  let trainingTargetIterations = 0;
  let trainingStartTime = 0;
  let trainingIterationsCompleted = 0;

  // Performance profiler
  const profiler = new TrainingProfiler(device);
  setGlobalProfiler(profiler);

  // Training controller for pruning/densification schedule
  let trainingController: TrainingController | null = null;
  let sceneExtent: number = 1.0;  // Will be set when loading point cloud

  // Densification / importance managers and pipelines
  let densificationManager: DensificationManager | null = null;
  let importanceManager: ImportanceManager | null = null;
  let densifyAccumPipeline: GPUComputePipeline | null = null;
  let densifyDecidePipeline: GPUComputePipeline | null = null;
  let densifyAccumBindGroup: GPUBindGroup | null = null;
  let densifyDecideBindGroup: GPUBindGroup | null = null;
  let densifyParamsBuffer: GPUBuffer | null = null;
  let importanceAccumPipeline: GPUComputePipeline | null = null;
  let importanceAccumBindGroup: GPUBindGroup | null = null;
  let importanceAccumParamsBuffer: GPUBuffer | null = null;
  let pruneDecidePipeline: GPUComputePipeline | null = null;
  let pruneDecideBindGroup: GPUBindGroup | null = null;
  let pruneDecideParamsBuffer: GPUBuffer | null = null;
  let cachedMultiViewScores: MultiViewScores | null = null;
  let cachedMultiViewIteration = -1;

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
let metric_image_params_buffer: GPUBuffer | null = null;
let metric_threshold_params_buffer: GPUBuffer | null = null;
let metric_pixel_buffer: GPUBuffer | null = null;
let metric_flag_buffer: GPUBuffer | null = null;
let metric_counts_buffer: GPUBuffer | null = null;
let metric_accum_params_buffer: GPUBuffer | null = null;

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
    // Loss weights: [w_l2, w_l1, w_ssim, _pad]
    // Original 3DGS uses: L = 0.8 * L1 + 0.2 * D-SSIM
    // We'll use L1-heavy mix to avoid blur: w_l2=0.2, w_l1=0.8
    const data = new Float32Array([
      0.0,  // w_l2 - FastGS uses SSIM instead
      0.8,  // w_l1
      0.2,  // w_ssim - FastGS default
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
    tilesBuilt = false;
    tilesBuiltForCameraIndex = -1;

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

  function setTargetTexture(view: GPUTextureView, sampler: GPUSampler) {
    targetTextureView = view;
    targetSampler = sampler;
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
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
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

  const metric_module = device.createShaderModule({ code: commonWGSL + '\n' + metricWGSL });

  const metric_map_pipeline = device.createComputePipeline({
    label: 'metric map',
    layout: 'auto',
    compute: {
      module: metric_module,
      entryPoint: 'compute_metric_map',
    },
  });

  const metric_threshold_pipeline = device.createComputePipeline({
    label: 'metric threshold',
    layout: 'auto',
    compute: {
      module: metric_module,
      entryPoint: 'threshold_metric_map',
    },
  });

  const metric_accum_pipeline = device.createComputePipeline({
    label: 'metric accumulate',
    layout: 'auto',
    compute: {
      module: metric_module,
      entryPoint: 'accumulate_metric_counts',
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

  backward_tiles_local_pipeline = device.createComputePipeline({
    label: 'backward tiles local (wavefront phase 1)',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: commonWGSL + '\n' + backwardTilesLocalWGSL }),
      entryPoint: 'backward_tiles_local',
    },
  });

  backward_reduce_pipeline = device.createComputePipeline({
    label: 'backward reduce (wavefront phase 2)',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: commonWGSL + '\n' + backwardReduceWGSL }),
      entryPoint: 'backward_reduce',
    },
  });

  // Densification pipelines
  densifyAccumPipeline = device.createComputePipeline({
    label: 'densify gradient accumulation',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: densifyAccumWGSL }),
      entryPoint: 'accumulate_grads',
    },
  });
  importanceAccumPipeline = device.createComputePipeline({
    label: 'importance accumulation',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: commonWGSL + '\n' + importanceAccumWGSL }),
      entryPoint: 'accumulate_importance',
    },
  });

  densifyDecidePipeline = device.createComputePipeline({
    label: 'densify decision',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: densifyDecideWGSL }),
      entryPoint: 'densify_decide',
    },
  });
  pruneDecidePipeline = device.createComputePipeline({
    label: 'prune decision',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: pruneDecideWGSL }),
      entryPoint: 'prune_decide',
    },
  });

  // Densification params buffer
  densifyParamsBuffer = device.createBuffer({
    label: 'densify params',
    size: 32,  // 8 floats
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  importanceAccumParamsBuffer = device.createBuffer({
    label: 'importance accum params',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  pruneDecideParamsBuffer = device.createBuffer({
    label: 'prune decide params',
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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

    const pixelByteSize = pixelCount * FLOAT32_BYTES;
    if (!metric_pixel_buffer || metric_pixel_buffer.size < pixelByteSize) {
      metric_pixel_buffer?.destroy();
      metric_pixel_buffer = device.createBuffer({
        label: 'metric pixel loss',
        size: pixelByteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }

    if (!metric_flag_buffer || metric_flag_buffer.size < pixelCount * 4) {
      metric_flag_buffer?.destroy();
      metric_flag_buffer = device.createBuffer({
        label: 'metric pixel flags',
        size: pixelCount * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }

    if (!metric_image_params_buffer) {
      metric_image_params_buffer = device.createBuffer({
        label: 'metric image params',
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    const imageParams = new Uint32Array([width, height, pixelCount, 0]);
    device.queue.writeBuffer(metric_image_params_buffer, 0, imageParams);

    if (!metric_threshold_params_buffer) {
      metric_threshold_params_buffer = device.createBuffer({
        label: 'metric threshold params',
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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
    // CRITICAL FIX: Do NOT destroy buffers if requiredIntersections is 0.
    // The training loop expects these buffers to exist (be non-null) even if empty.
    // If we destroy them, subsequent bind group updates will fail.

    // We'll treat 0 intersections as a valid state that just needs minimal buffer sizes.
    // Just ensure non-negative.
    requiredIntersections = Math.max(0, requiredIntersections);

    const tileWidth = Math.ceil(canvas.width / TILE_SIZE);
    const tileHeight = Math.ceil(canvas.height / TILE_SIZE);
    const tileCount = tileWidth * tileHeight;
    const numCameras = 1;
    const totalTiles = tileCount * numCameras;

    // Ensure buffers are never null. Use a minimum size of 4 bytes (1 u32) 
    // to satisfy WebGPU requirements if requiredIntersections is 0.

    if (!isect_ids_buffer || requiredIntersections > num_intersections) {
      if (isect_ids_buffer) {
        isect_ids_buffer.destroy();
      }
      isect_ids_buffer = device.createBuffer({
        label: 'tiling isect_ids (u32 keys)',
        size: Math.max(requiredIntersections * 4, 4), // Min 4 bytes
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }

    if (!flatten_ids_buffer || requiredIntersections > num_intersections) {
      if (flatten_ids_buffer) {
        flatten_ids_buffer.destroy();
      }
      flatten_ids_buffer = device.createBuffer({
        label: 'tiling flatten_ids (u32)',
        size: Math.max(requiredIntersections * 4, 4), // Min 4 bytes
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }

    const tileGradsSize = requiredIntersections * TILE_GRAD_STRIDE * 4;
    if (!tile_grads_buffer || tileGradsSize > (tile_grads_buffer?.size ?? 0)) {
      if (tile_grads_buffer) {
        tile_grads_buffer.destroy();
      }
      tile_grads_buffer = device.createBuffer({
        label: 'tile_grads (wavefront backward)',
        size: Math.max(tileGradsSize, 256),
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

  function rebuildImportanceAccumBindGroup(): void {
    if (!importanceAccumPipeline || !gauss_projections_buffer || !importanceManager || !importanceAccumParamsBuffer) return;
    const statsBuffer = importanceManager.getStatsBuffer();
    if (!statsBuffer) return;
    importanceAccumBindGroup = device.createBindGroup({
      label: 'importance accum bind group',
      layout: importanceAccumPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gauss_projections_buffer } },
        { binding: 1, resource: { buffer: statsBuffer } },
        { binding: 2, resource: { buffer: importanceAccumParamsBuffer } },
      ],
    });
  }

  function rebuildPruneDecideBindGroup(): void {
    if (!pruneDecidePipeline || !importanceManager || !train_opacity_buffer || !train_scale_buffer || !pruneDecideParamsBuffer) return;
    const statsBuffer = importanceManager.getStatsBuffer();
    const actionsBuffer = importanceManager.getPruneActionsBuffer();
    const countsBuffer = importanceManager.getPruneCountBuffer();
    if (!statsBuffer || !actionsBuffer || !countsBuffer) return;
    pruneDecideBindGroup = device.createBindGroup({
      label: 'prune decide bind group',
      layout: pruneDecidePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: statsBuffer } },
        { binding: 1, resource: { buffer: train_opacity_buffer } },
        { binding: 2, resource: { buffer: train_scale_buffer } },
        { binding: 3, resource: { buffer: actionsBuffer } },
        { binding: 4, resource: { buffer: countsBuffer } },
        { binding: 5, resource: { buffer: pruneDecideParamsBuffer } },
      ],
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

  function rebuildBackwardTilesLocalBG() {
    if (!backward_tiles_local_pipeline) return;
    if (!sh_buffer || !tiling_params_buffer || !gauss_projections_buffer) return;
    if (!residual_color_view || !training_alpha_view || !training_last_ids_buffer) return;
    if (!tile_offsets_buffer || !flatten_ids_buffer || !tile_mask_buffer) return;
    if (!tiles_cumsum_buffer || !active_count_uniform_buffer || !tile_grads_buffer) return;

    backward_tiles_local_bind_group = device.createBindGroup({
      label: 'backward tiles local bind group',
      layout: backward_tiles_local_pipeline.getBindGroupLayout(0),
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
        { binding: 10, resource: { buffer: active_count_uniform_buffer } },
        { binding: 11, resource: { buffer: background_params_buffer! } },
        { binding: 12, resource: { buffer: tile_mask_buffer! } },
        { binding: 13, resource: { buffer: tiles_cumsum_buffer } },
        { binding: 14, resource: { buffer: tile_grads_buffer } },
      ],
    });
  }

  function rebuildBackwardReduceBG() {
    if (!backward_reduce_pipeline) return;
    if (!tile_grads_buffer || !flatten_ids_buffer || !tiles_cumsum_buffer) return;
    if (!grad_sh_buffer || !grad_opacity_buffer || !grad_geom_buffer) return;
    if (!active_count_uniform_buffer) return;

    backward_reduce_bind_group = device.createBindGroup({
      label: 'backward reduce bind group',
      layout: backward_reduce_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tile_grads_buffer } },
        { binding: 1, resource: { buffer: flatten_ids_buffer } },
        { binding: 2, resource: { buffer: tiles_cumsum_buffer } },
        { binding: 3, resource: { buffer: active_count_uniform_buffer } },
        { binding: 4, resource: { buffer: grad_sh_buffer } },
        { binding: 5, resource: { buffer: grad_opacity_buffer } },
        { binding: 6, resource: { buffer: grad_geom_buffer } },
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

  function rebuildMetricMapBindGroup() {
    if (!metric_map_pipeline || !training_color_view || !metric_pixel_buffer || !metric_image_params_buffer) {
      return;
    }
    let targetView: GPUTextureView | null = null;
    let targetSamplerToUse: GPUSampler | null = null;
    if (training_images.length > 0 && current_training_camera_index < training_images.length) {
      targetView = training_images[current_training_camera_index].view;
      targetSamplerToUse = training_images[current_training_camera_index].sampler;
    } else if (targetTextureView && targetSampler) {
      targetView = targetTextureView;
      targetSamplerToUse = targetSampler;
    }
    if (!targetView || !targetSamplerToUse) {
      console.warn('Metric map: no target texture available for comparison.');
      return;
    }
    metric_map_bind_group = device.createBindGroup({
      label: 'metric map bind group',
      layout: metric_map_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: targetSamplerToUse },
        { binding: 1, resource: targetView },
        { binding: 2, resource: training_color_view },
        { binding: 3, resource: { buffer: metric_pixel_buffer } },
        { binding: 4, resource: { buffer: metric_image_params_buffer } },
      ],
    });
  }

  function rebuildMetricThresholdBindGroup() {
    if (!metric_threshold_pipeline || !metric_pixel_buffer || !metric_flag_buffer || !metric_threshold_params_buffer) {
      return;
    }
    metric_threshold_bind_group = device.createBindGroup({
      label: 'metric threshold bind group',
      layout: metric_threshold_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: metric_pixel_buffer } },
        { binding: 1, resource: { buffer: metric_flag_buffer } },
        { binding: 2, resource: { buffer: metric_threshold_params_buffer } },
      ],
    });
  }

  function rebuildMetricAccumBindGroup() {
    ensureTilingParamsBuffer();
    if (!metric_accum_pipeline || !metric_flag_buffer || !tile_offsets_buffer ||
      !flatten_ids_buffer || !gauss_projections_buffer || !training_last_ids_buffer ||
      !tiling_params_buffer || !metric_counts_buffer || !active_count_uniform_buffer) {
      return;
    }
    if (!metric_accum_params_buffer) {
      metric_accum_params_buffer = device.createBuffer({
        label: 'metric accum params',
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    const pixelCount = training_width * training_height;
    const tileWidth = Math.ceil(canvas.width / TILE_SIZE);
    const tileHeight = Math.ceil(canvas.height / TILE_SIZE);
    const totalTiles = tileWidth * tileHeight;
    const accumParams = new Uint32Array([
      pixelCount,
      canvas.width,
      tileWidth,
      tileHeight,
      TILE_SIZE,
      active_count,
      totalTiles,
      0,
    ]);
    device.queue.writeBuffer(metric_accum_params_buffer, 0, accumParams);
    metric_accum_bind_group = device.createBindGroup({
      label: 'metric accumulate bind group',
      layout: metric_accum_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: metric_flag_buffer } },
        { binding: 1, resource: { buffer: tile_offsets_buffer } },
        { binding: 2, resource: { buffer: flatten_ids_buffer } },
        { binding: 3, resource: { buffer: gauss_projections_buffer! } },
        { binding: 4, resource: { buffer: training_last_ids_buffer! } },
        { binding: 5, resource: { buffer: tiling_params_buffer! } },
        { binding: 6, resource: { buffer: metric_accum_params_buffer } },
        { binding: 7, resource: { buffer: metric_counts_buffer! } },
      ],
    });
  }

  function rebuildAllBindGroups() {
    rebuildSorterBG();
    rebuildSortedBG();
    rebuildRenderSplatBG();
    rebuildPreprocessBG();
    rebuildKnnInitBG();
    rebuildTrainingBackwardGeomBG();
    // Tiled training bind groups are rebuilt lazily when running tiled passes.
    rebuildTrainingInitParamsBG();
    rebuildTrainingApplyParamsBG();
    // Densification gradient accumulation bind groups
    rebuildDensifyAccumBindGroup();
    rebuildImportanceAccumBindGroup();
    importanceManager.ensureCapacity(active_count);
    rebuildPruneDecideBindGroup();
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
      // CORRECTION: Even with 0 intersections, we must reset tile_offsets_buffer.
      // Otherwise it contains stale data, causing the renderer to access out-of-bounds intersections.
      if (tile_offsets_buffer) {
        const tileWidth = Math.ceil(canvas.width / TILE_SIZE);
        const tileHeight = Math.ceil(canvas.height / TILE_SIZE);
        const totalTiles = tileWidth * tileHeight;
        const zeroOffsets = new Uint32Array(totalTiles + 1);
        // default init is 0, which is correct for 0 intersections
        device.queue.writeBuffer(tile_offsets_buffer, 0, zeroOffsets);
      }
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

  async function rebuildTilesForCurrentCamera(): Promise<void> {
    tilesBuilt = false;
    tilesBuiltForCameraIndex = -1;
    if (training_camera_buffers.length === 0) {
      return;
    }
    try {
      await buildTileIntersections();
      tilesBuiltForCameraIndex = current_training_camera_index;
    } catch (err) {
      console.error('[tiles] Failed to rebuild training tiles:', err);
      tilesBuilt = false;
      tilesBuiltForCameraIndex = -1;
    }
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
    importanceManager?.ensureCapacity(want);
    importanceManager?.resetStats(want);
    importanceManager?.resetPruneActions(want);
    if (!metric_counts_buffer || metric_counts_buffer.size < want * 4) {
      metric_counts_buffer?.destroy();
      metric_counts_buffer = device.createBuffer({
        label: 'metric gaussian counts',
        size: want * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }
  }

  function resizeToActiveCount() {
    const target_capacity = nextPow2(Math.max(active_count, 1024));
    if (target_capacity < capacity) {
      recreateBuffers(target_capacity, true);
      sorter = get_sorter(target_capacity, device);
      rebuildAllBindGroups();
      importanceManager?.ensureCapacity(target_capacity);
      importanceManager?.resetStats(target_capacity);
      importanceManager?.resetPruneActions(target_capacity);
    }
  }

  function updateSceneExtent(newExtent: number) {
    sceneExtent = newExtent;
    if (trainingController) {
      trainingController.setSceneExtent(newExtent);
    }
    if (densificationManager) {
      densificationManager.setSceneExtent(newExtent);
    }
  }

  // ===============================================
  // Point Cloud Loading
  // ===============================================
  function reloadPointCloud(new_pc: PointCloud) {
    pc_data = new_pc;

    // Update scene extent and training controller
    updateSceneExtent(new_pc.scene_extent ?? 1.0);
    trainingController = new TrainingController(sceneExtent);

    // Initialize densification manager
    densificationManager?.destroy();
    densificationManager = new DensificationManager(device, sceneExtent);
    densificationManager.ensureCapacity(new_pc.num_points);
    importanceManager = new ImportanceManager(device);
    importanceManager.ensureCapacity(new_pc.num_points);

    console.log(`[reloadPointCloud] Scene extent: ${sceneExtent.toFixed(2)}, training controller initialized`);

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

  /**
   * Prune gaussians based on current training controller settings.
   * Uses FastGS-style pruning: opacity + world-scale thresholds.
   * @param customParams Optional custom prune params (overrides controller)
  */
  async function prune(customParams?: PruneParams): Promise<number> {
    if (active_count === 0) {
      return active_count;
    }

    renderingSuspended = true;

    try {
      const multiView = await getMultiViewScores();
      if (!multiView || multiView.sampleCount === 0) {
        console.warn('[prune] Unable to compute multi-view scores; skipping.');
        return active_count;
      }
      const pruned = await applyMultiViewPrune(multiView, active_count);
      if (pruned > 0) {
        console.log(`[prune] Removed ${pruned} gaussians (remaining ${active_count})`);
      } else {
        console.log('[prune] No gaussians removed');
      }
      return active_count;
    } finally {
      renderingSuspended = false;
    }
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

    if (useWavefrontBackward && tile_grads_buffer) {
      run_wavefront_backward(encoder);
    } else {
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
  }

  function run_wavefront_backward(encoder: GPUCommandEncoder) {
    rebuildBackwardTilesLocalBG();
    if (!backward_tiles_local_bind_group) {
      console.warn('backward_tiles_local_bind_group is null, falling back to old backward');
      useWavefrontBackward = false;
      run_training_tiled_backward(encoder);
      return;
    }

    const tilesX = Math.ceil(canvas.width / TILE_SIZE);
    const tilesY = Math.ceil(canvas.height / TILE_SIZE);

    const pass1 = encoder.beginComputePass({ label: 'backward tiles local (phase 1)' });
    pass1.setPipeline(backward_tiles_local_pipeline!);
    pass1.setBindGroup(0, backward_tiles_local_bind_group);
    pass1.dispatchWorkgroups(tilesX, tilesY);
    pass1.end();

    rebuildBackwardReduceBG();
    if (!backward_reduce_bind_group) {
      console.warn('backward_reduce_bind_group is null');
      return;
    }

    const n_isects = lastTileCoverageStats?.totalIntersections ?? 0;
    if (n_isects === 0) {
      return;
    }

    const numReduceGroups = Math.ceil(n_isects / 256);

    const pass2 = encoder.beginComputePass({ label: 'backward reduce (phase 2)' });
    pass2.setPipeline(backward_reduce_pipeline!);
    pass2.setBindGroup(0, backward_reduce_bind_group);
    pass2.dispatchWorkgroups(numReduceGroups);
    pass2.end();
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

    // Only update SH every SH_UPDATE_INTERVAL iterations (FastGS does every 16)
    const shouldUpdateSH = (trainingIteration % SH_UPDATE_INTERVAL) === 0;
    if (sh_elements > 0 && shouldUpdateSH) {
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

  // ===============================================
  // Densification - Gradient Accumulation
  // ===============================================

  function rebuildDensifyAccumBindGroup(): void {
    if (!densifyAccumPipeline || !grad_geom_buffer || !densificationManager) return;

    const gradAccumBuffer = densificationManager.getGradAccumBuffer();
    if (!gradAccumBuffer) return;

    // Create a uniform buffer for active_count
    const activeCountBuffer = device.createBuffer({
      label: 'densify active count',
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(activeCountBuffer, 0, new Uint32Array([active_count]));

    densifyAccumBindGroup = device.createBindGroup({
      label: 'densify accum bind group',
      layout: densifyAccumPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: grad_geom_buffer } },
        { binding: 1, resource: { buffer: gradAccumBuffer } },
        { binding: 2, resource: { buffer: activeCountBuffer } },
      ],
    });
  }

  function rebuildDensifyDecideBindGroup(): void {
    if (!densifyDecidePipeline || !train_scale_buffer || !densificationManager || !densifyParamsBuffer || !importanceManager) return;

    const gradAccumBuffer = densificationManager.getGradAccumBuffer();
    const actionsBuffer = densificationManager.getDensifyOutputBuffer();
    const countsBuffer = densificationManager.getCountBuffer();
    const statsBuffer = importanceManager.getStatsBuffer();
    if (!gradAccumBuffer || !actionsBuffer || !countsBuffer || !statsBuffer) return;

    densifyDecideBindGroup = device.createBindGroup({
      label: 'densify decide bind group',
      layout: densifyDecidePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gradAccumBuffer } },
        { binding: 1, resource: { buffer: train_scale_buffer } },
        { binding: 2, resource: { buffer: actionsBuffer } },
        { binding: 3, resource: { buffer: countsBuffer } },
        { binding: 4, resource: { buffer: densifyParamsBuffer } },
        { binding: 5, resource: { buffer: statsBuffer } },
      ],
    });
  }

  function run_densify_accumulate(encoder: GPUCommandEncoder): void {
    if (!densifyAccumBindGroup || !densifyAccumPipeline || active_count === 0) return;

    const num_wg = Math.ceil(active_count / 256);
    const pass = encoder.beginComputePass({ label: 'densify accumulate' });
    pass.setPipeline(densifyAccumPipeline);
    pass.setBindGroup(0, densifyAccumBindGroup);
    pass.dispatchWorkgroups(num_wg);
    pass.end();
  }

  function run_densify_decide(encoder: GPUCommandEncoder): void {
    if (!densifyDecideBindGroup || !densifyDecidePipeline || !densificationManager || !densifyParamsBuffer) return;
    if (active_count === 0) return;

    // Update params buffer with dynamic visibility gating tied to densification interval
    const paramsData = densificationManager.createDensifyUniformData(active_count);

    // CRITICAL FIX: Scale gradient thresholds to pixel space.
    // The config thresholds (e.g. 0.0002) are based on NDC space (0-1).
    // Our gradients are in pixel space (0-W), so they are ~W/2 times larger.
    const pixelScale = canvas.width / 2.0;
    paramsData[0] *= pixelScale; // gradThreshold
    paramsData[1] *= pixelScale; // gradAbsThreshold

    const interval =
      trainingController?.getConfig().densificationInterval ??
      DEFAULT_TRAINING_SCHEDULE.densificationInterval;
    const dynamicVisibility = Math.max(paramsData[5], Math.max(5, Math.round(interval * 0.35)));
    paramsData[5] = dynamicVisibility;
    device.queue.writeBuffer(densifyParamsBuffer, 0, paramsData);

    // Reset counts
    densificationManager.resetCounts();

    const num_wg = Math.ceil(active_count / 256);
    const pass = encoder.beginComputePass({ label: 'densify decide' });
    pass.setPipeline(densifyDecidePipeline);
    pass.setBindGroup(0, densifyDecideBindGroup);
    pass.dispatchWorkgroups(num_wg);
    pass.end();
  }

  function run_metric_map(encoder: GPUCommandEncoder): void {
    if (!metric_map_pipeline) return;
    rebuildMetricMapBindGroup();
    if (!metric_map_bind_group || training_width === 0 || training_height === 0) return;
    const pass = encoder.beginComputePass({ label: 'metric map' });
    pass.setPipeline(metric_map_pipeline);
    pass.setBindGroup(0, metric_map_bind_group);
    pass.dispatchWorkgroups(Math.ceil(training_width / 8), Math.ceil(training_height / 8));
    pass.end();
  }

  function run_metric_threshold(encoder: GPUCommandEncoder): void {
    if (!metric_threshold_pipeline) return;
    rebuildMetricThresholdBindGroup();
    if (!metric_threshold_bind_group || training_width === 0 || training_height === 0) return;
    const pixelCount = training_width * training_height;
    const numWG = Math.ceil(pixelCount / WORKGROUP_SIZE);
    const pass = encoder.beginComputePass({ label: 'metric threshold' });
    pass.setPipeline(metric_threshold_pipeline);
    pass.setBindGroup(0, metric_threshold_bind_group);
    pass.dispatchWorkgroups(numWG);
    pass.end();
  }

  function run_metric_accumulate(encoder: GPUCommandEncoder): void {
    if (!metric_accum_pipeline) return;
    rebuildMetricAccumBindGroup();
    if (!metric_accum_bind_group || training_width === 0 || training_height === 0) return;
    const pixelCount = training_width * training_height;
    const numWG = Math.ceil(pixelCount / WORKGROUP_SIZE);
    const pass = encoder.beginComputePass({ label: 'metric accumulate' });
    pass.setPipeline(metric_accum_pipeline);
    pass.setBindGroup(0, metric_accum_bind_group);
    pass.dispatchWorkgroups(numWG);
    pass.end();
  }

  function sampleCameraIndices(count: number): number[] {
    const total = training_cameras.length;
    if (count >= total) {
      return Array.from({ length: total }, (_, i) => i);
    }
    const indices = Array.from({ length: total }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    return indices.slice(0, count);
  }

  async function renderCameraForMetrics(cameraIndex: number): Promise<boolean> {
    if (training_cameras.length === 0 || training_images.length === 0) {
      return false;
    }
    current_training_camera_index = cameraIndex;
    if (!tilesBuilt || tilesBuiltForCameraIndex !== cameraIndex) {
      await rebuildTilesForCurrentCamera();
      tilesBuiltForCameraIndex = cameraIndex;
    }
    const encoder = device.createCommandEncoder();
    run_training_tiled_forward(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return true;
  }

  async function computePixelLossStats(): Promise<{ minValue: number; maxValue: number; meanLoss: number } | null> {
    if (!metric_pixel_buffer || !metric_threshold_params_buffer || training_width === 0 || training_height === 0) {
      return null;
    }
    rebuildMetricMapBindGroup();
    if (!metric_map_bind_group) return null;
    const encoder = device.createCommandEncoder();
    run_metric_map(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    const pixelCount = training_width * training_height;
    const pixelData = await readBufferToCPU(metric_pixel_buffer, pixelCount * FLOAT32_BYTES);
    let minValue = Number.POSITIVE_INFINITY;
    let maxValue = Number.NEGATIVE_INFINITY;
    let sum = 0;
    for (let i = 0; i < pixelData.length; i++) {
      const v = pixelData[i];
      if (!Number.isFinite(v)) continue;
      if (v < minValue) minValue = v;
      if (v > maxValue) maxValue = v;
      sum += v;
    }
    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
      return null;
    }
    const meanLoss = pixelCount > 0 ? sum / pixelCount : 0;
    const invRange = maxValue > minValue ? 1.0 / (maxValue - minValue) : 0.0;
    const thresholdBuffer = new ArrayBuffer(16);
    const thresholdFloats = new Float32Array(thresholdBuffer);
    const thresholdInts = new Uint32Array(thresholdBuffer);
    thresholdFloats[0] = minValue;
    thresholdFloats[1] = invRange;
    thresholdFloats[2] = MULTI_VIEW_LOSS_THRESH;
    thresholdInts[3] = training_width * training_height;
    device.queue.writeBuffer(metric_threshold_params_buffer!, 0, thresholdBuffer);
    return { minValue, maxValue, meanLoss };
  }

  async function accumulateMetricCounts(): Promise<Uint32Array | null> {
    if (!metric_counts_buffer) return null;
    const zeroCounts = new Uint32Array(Math.max(active_count, 1));
    device.queue.writeBuffer(metric_counts_buffer, 0, zeroCounts);
    const encoder = device.createCommandEncoder();
    run_metric_threshold(encoder);
    run_metric_accumulate(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return readBufferToCPU_u32(metric_counts_buffer, active_count * 4);
  }

  async function generateMultiViewScores(): Promise<MultiViewScores | null> {
    if (training_cameras.length === 0 || training_images.length === 0 || active_count === 0) {
      return null;
    }
    const sampleCount = Math.min(MULTI_VIEW_SAMPLE_COUNT, training_cameras.length);
    if (sampleCount === 0) {
      return null;
    }
    const indices = sampleCameraIndices(sampleCount);
    const importance = new Uint32Array(active_count);
    const pruning = new Float32Array(active_count);
    for (const idx of indices) {
      const rendered = await renderCameraForMetrics(idx);
      if (!rendered) {
        continue;
      }
      const stats = await computePixelLossStats();
      if (!stats) {
        continue;
      }
      const counts = await accumulateMetricCounts();
      if (!counts) {
        continue;
      }
      for (let i = 0; i < active_count; i++) {
        const c = counts[i];
        importance[i] += c;
        pruning[i] += c * stats.meanLoss;
      }
    }
    return { importance, pruning, sampleCount: indices.length };
  }

  async function getMultiViewScores(): Promise<MultiViewScores | null> {
    if (cachedMultiViewScores && cachedMultiViewIteration === trainingIteration) {
      return cachedMultiViewScores;
    }
    const result = await generateMultiViewScores();
    cachedMultiViewScores = result;
    cachedMultiViewIteration = trainingIteration;
    return result;
  }

  async function applyMultiViewPrune(scores: MultiViewScores, baseCount: number): Promise<number> {
    if (!train_scale_buffer || !train_opacity_buffer) {
      return 0;
    }
    if (active_count === 0 || scores.sampleCount === 0) {
      return 0;
    }
    const prunable = new Array<boolean>(active_count).fill(false);
    const opacities = await readBufferToCPU(train_opacity_buffer, active_count * FLOAT32_BYTES);
    const scales = await readBufferToCPU(train_scale_buffer, active_count * TRAIN_SCALE_COMPONENTS * FLOAT32_BYTES);
    const minOpacity = trainingController?.getConfig().minOpacity ?? DEFAULT_TRAINING_SCHEDULE.minOpacity;
    const scaleThreshold = sceneExtent * 0.1;
    for (let i = 0; i < active_count; i++) {
      const value = 1 / (1 + Math.exp(-opacities[i]));
      if (value < minOpacity) {
        prunable[i] = true;
        continue;
      }
      const scaleBase = i * TRAIN_SCALE_COMPONENTS;
      const maxScale = Math.max(
        Math.exp(scales[scaleBase]),
        Math.exp(scales[scaleBase + 1]),
        Math.exp(scales[scaleBase + 2]),
      );
      if (maxScale > scaleThreshold) {
        prunable[i] = true;
      }
    }

    const limit = Math.min(baseCount, scores.pruning.length);
    let minScore = Number.POSITIVE_INFINITY;
    let maxScore = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < limit; i++) {
      const val = scores.pruning[i];
      if (!Number.isFinite(val)) continue;
      if (val < minScore) minScore = val;
      if (val > maxScore) maxScore = val;
    }
    if (!Number.isFinite(minScore) || !Number.isFinite(maxScore)) {
      return 0;
    }
    const range = Math.max(1e-6, maxScore - minScore);
    const normalizedScores = new Float32Array(active_count);
    for (let i = 0; i < limit; i++) {
      normalizedScores[i] = (scores.pruning[i] - minScore) / range;
    }

    let toRemove = 0;
    for (let i = 0; i < prunable.length; i++) {
      if (prunable[i]) toRemove++;
    }
    const removeBudget = Math.floor(0.5 * toRemove);
    if (removeBudget <= 0) {
      return 0;
    }

    const weights = new Float32Array(active_count);
    let totalWeight = 0;
    for (let i = 0; i < active_count; i++) {
      if (!prunable[i]) continue;
      const score = normalizedScores[i];
      const w = 1.0 / (1e-6 + (1.0 - score));
      weights[i] = w;
      totalWeight += w;
    }
    if (totalWeight === 0) {
      return 0;
    }

    const selected = new Array<boolean>(active_count).fill(false);
    for (let k = 0; k < removeBudget && totalWeight > 0; k++) {
      const target = Math.random() * totalWeight;
      let accum = 0;
      for (let i = 0; i < active_count; i++) {
        const w = weights[i];
        if (w === 0) continue;
        accum += w;
        if (target <= accum) {
          selected[i] = true;
          totalWeight -= w;
          weights[i] = 0;
          break;
        }
      }
    }

    const finalMask = selected.map((chosen, idx) => chosen && prunable[idx]);
    return applyPruneMask(finalMask);
  }

  async function applyPruneMask(mask: boolean[]): Promise<number> {
    if (!train_position_buffer || !train_scale_buffer || !train_rotation_buffer ||
      !train_opacity_buffer || !train_sh_buffer) {
      return 0;
    }
    const removeCount = mask.reduce((sum, flag) => sum + (flag ? 1 : 0), 0);
    if (removeCount === 0 || removeCount >= active_count) {
      return 0;
    }
    const keepCount = active_count - removeCount;
    const positions = await readBufferToCPU(train_position_buffer, active_count * TRAIN_POS_COMPONENTS * FLOAT32_BYTES);
    const scales = await readBufferToCPU(train_scale_buffer, active_count * TRAIN_SCALE_COMPONENTS * FLOAT32_BYTES);
    const rotations = await readBufferToCPU(train_rotation_buffer, active_count * TRAIN_ROT_COMPONENTS * FLOAT32_BYTES);
    const opacities = await readBufferToCPU(train_opacity_buffer, active_count * FLOAT32_BYTES);
    const shs = await readBufferToCPU(train_sh_buffer, active_count * TRAIN_SH_COMPONENTS * FLOAT32_BYTES);
    const params = await readAdamMoments(active_count);

    const newPositions = new Float32Array(keepCount * TRAIN_POS_COMPONENTS);
    const newScales = new Float32Array(keepCount * TRAIN_SCALE_COMPONENTS);
    const newRotations = new Float32Array(keepCount * TRAIN_ROT_COMPONENTS);
    const newOpacities = new Float32Array(keepCount);
    const newSHs = new Float32Array(keepCount * TRAIN_SH_COMPONENTS);
    const newMoments: AdamMoments = {
      opacity_m1: new Float32Array(keepCount),
      opacity_m2: new Float32Array(keepCount),
      pos_m1: new Float32Array(keepCount * TRAIN_POS_COMPONENTS),
      pos_m2: new Float32Array(keepCount * TRAIN_POS_COMPONENTS),
      scale_m1: new Float32Array(keepCount * TRAIN_SCALE_COMPONENTS),
      scale_m2: new Float32Array(keepCount * TRAIN_SCALE_COMPONENTS),
      rot_m1: new Float32Array(keepCount * TRAIN_ROT_COMPONENTS),
      rot_m2: new Float32Array(keepCount * TRAIN_ROT_COMPONENTS),
      sh_m1: new Float32Array(keepCount * TRAIN_SH_COMPONENTS),
      sh_m2: new Float32Array(keepCount * TRAIN_SH_COMPONENTS),
    };

    let writeIdx = 0;
    for (let i = 0; i < active_count; i++) {
      if (mask[i]) continue;
      const posBase = i * TRAIN_POS_COMPONENTS;
      const scaleBase = i * TRAIN_SCALE_COMPONENTS;
      const rotBase = i * TRAIN_ROT_COMPONENTS;
      const shBase = i * TRAIN_SH_COMPONENTS;
      const newPosBase = writeIdx * TRAIN_POS_COMPONENTS;
      const newScaleBase = writeIdx * TRAIN_SCALE_COMPONENTS;
      const newRotBase = writeIdx * TRAIN_ROT_COMPONENTS;
      const newShBase = writeIdx * TRAIN_SH_COMPONENTS;
      newPositions.set(positions.slice(posBase, posBase + TRAIN_POS_COMPONENTS), newPosBase);
      newScales.set(scales.slice(scaleBase, scaleBase + TRAIN_SCALE_COMPONENTS), newScaleBase);
      newRotations.set(rotations.slice(rotBase, rotBase + TRAIN_ROT_COMPONENTS), newRotBase);
      newSHs.set(shs.slice(shBase, shBase + TRAIN_SH_COMPONENTS), newShBase);
      newOpacities[writeIdx] = opacities[i];
      newMoments.opacity_m1[writeIdx] = params.opacity_m1[i];
      newMoments.opacity_m2[writeIdx] = params.opacity_m2[i];
      newMoments.pos_m1.set(params.pos_m1.slice(posBase, posBase + TRAIN_POS_COMPONENTS), newPosBase);
      newMoments.pos_m2.set(params.pos_m2.slice(posBase, posBase + TRAIN_POS_COMPONENTS), newPosBase);
      newMoments.scale_m1.set(params.scale_m1.slice(scaleBase, scaleBase + TRAIN_SCALE_COMPONENTS), newScaleBase);
      newMoments.scale_m2.set(params.scale_m2.slice(scaleBase, scaleBase + TRAIN_SCALE_COMPONENTS), newScaleBase);
      newMoments.rot_m1.set(params.rot_m1.slice(rotBase, rotBase + TRAIN_ROT_COMPONENTS), newRotBase);
      newMoments.rot_m2.set(params.rot_m2.slice(rotBase, rotBase + TRAIN_ROT_COMPONENTS), newRotBase);
      newMoments.sh_m1.set(params.sh_m1.slice(shBase, shBase + TRAIN_SH_COMPONENTS), newShBase);
      newMoments.sh_m2.set(params.sh_m2.slice(shBase, shBase + TRAIN_SH_COMPONENTS), newShBase);
      writeIdx++;
    }

    device.queue.writeBuffer(train_position_buffer!, 0, newPositions);
    device.queue.writeBuffer(train_scale_buffer!, 0, newScales);
    device.queue.writeBuffer(train_rotation_buffer!, 0, newRotations);
    device.queue.writeBuffer(train_opacity_buffer!, 0, newOpacities);
    device.queue.writeBuffer(train_sh_buffer!, 0, newSHs);
    writeAdamMoments(newMoments);

    setActiveCount(keepCount);
    densificationManager?.resetAccumulators(keepCount);
    importanceManager?.resetStats(keepCount);
    importanceManager?.resetPruneActions(keepCount);
    await syncGaussianBufferFromTrainingParams();
    rebuildAllBindGroups();
    rebuildDensifyAccumBindGroup();
    await rebuildTilesForCurrentCamera();
    return removeCount;
  }

  function run_importance_accumulate(encoder: GPUCommandEncoder): void {
    if (!importanceAccumPipeline || !importanceAccumBindGroup || !importanceManager || !importanceAccumParamsBuffer) return;
    if (!gauss_projections_buffer || active_count === 0) return;

    const paramsData = importanceManager.createAccumParams(active_count);
    device.queue.writeBuffer(importanceAccumParamsBuffer, 0, paramsData);

    const num_wg = Math.ceil(active_count / WORKGROUP_SIZE);
    const pass = encoder.beginComputePass({ label: 'importance accumulate' });
    pass.setPipeline(importanceAccumPipeline);
    pass.setBindGroup(0, importanceAccumBindGroup);
    pass.dispatchWorkgroups(num_wg);
    pass.end();
  }

  function run_prune_decide(encoder: GPUCommandEncoder, pruneParams: PruneParams): void {
    if (!pruneDecidePipeline || !pruneDecideBindGroup || !importanceManager || !pruneDecideParamsBuffer) return;
    if (active_count === 0) return;

    const thresholds = {
      minOpacity: pruneParams.minOpacity,
      maxScale: pruneParams.maxScale,
      maxScreenRadius: pruneParams.useScreenSizePruning ? pruneParams.maxScreenRadius : 10000.0,
    };
    const paramsData = importanceManager.createPruneParams(active_count, thresholds);
    device.queue.writeBuffer(pruneDecideParamsBuffer, 0, paramsData);

    const num_wg = Math.ceil(active_count / WORKGROUP_SIZE);
    const pass = encoder.beginComputePass({ label: 'prune decide' });
    pass.setPipeline(pruneDecidePipeline);
    pass.setBindGroup(0, pruneDecideBindGroup);
    pass.dispatchWorkgroups(num_wg);
    pass.end();
  }

  /**
   * Helper to read a GPU buffer to CPU
   */
  async function readBufferToCPU(buffer: GPUBuffer, size: number): Promise<Float32Array> {
    const staging = device.createBuffer({
      label: 'staging read',
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await staging.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    return data;
  }

  /**
   * Helper to read a GPU buffer as Uint32Array
   */
  async function readBufferToCPU_u32(buffer: GPUBuffer, size: number): Promise<Uint32Array> {
    const staging = device.createBuffer({
      label: 'staging read u32',
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await staging.mapAsync(GPUMapMode.READ);
    const data = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    return data;
  }

  /**
   * Build a rotation matrix from quaternion (for split position sampling)
   */
  function quatToMat3(q: number[]): number[][] {
    const [x, y, z, w] = q;
    const x2 = x + x, y2 = y + y, z2 = z + z;
    const xx = x * x2, xy = x * y2, xz = x * z2;
    const yy = y * y2, yz = y * z2, zz = z * z2;
    const wx = w * x2, wy = w * y2, wz = w * z2;

    return [
      [1 - (yy + zz), xy - wz, xz + wy],
      [xy + wz, 1 - (xx + zz), yz - wx],
      [xz - wy, yz + wx, 1 - (xx + yy)],
    ];
  }

  /**
   * Sample a random 3D point from gaussian distribution
   */
  function sampleGaussian3D(mean: number[], std: number[]): number[] {
    // Box-Muller transform for normal distribution
    const u1 = Math.random();
    const u2 = Math.random();
    const u3 = Math.random();
    const u4 = Math.random();

    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
    const z2 = Math.sqrt(-2 * Math.log(u3)) * Math.cos(2 * Math.PI * u4);

    return [
      mean[0] + z0 * std[0],
      mean[1] + z1 * std[1],
      mean[2] + z2 * std[2],
    ];
  }

  interface AdamMoments {
    opacity_m1: Float32Array; opacity_m2: Float32Array;
    pos_m1: Float32Array; pos_m2: Float32Array;
    scale_m1: Float32Array; scale_m2: Float32Array;
    rot_m1: Float32Array; rot_m2: Float32Array;
    sh_m1: Float32Array; sh_m2: Float32Array;
  }

  async function readAdamMoments(count: number): Promise<AdamMoments> {
    const empty = new Float32Array(0);
    if (count === 0) {
      return {
        opacity_m1: empty, opacity_m2: empty,
        pos_m1: empty, pos_m2: empty,
        scale_m1: empty, scale_m2: empty,
        rot_m1: empty, rot_m2: empty,
        sh_m1: empty, sh_m2: empty,
      };
    }

    // Read all moment buffers
    // We can run these in parallel
    const [
      opacity_m1, opacity_m2,
      pos_m1, pos_m2,
      scale_m1, scale_m2,
      rot_m1, rot_m2,
      sh_m1, sh_m2
    ] = await Promise.all([
      readBufferToCPU(moment_opacity_m1_buffer, count * FLOAT32_BYTES),
      readBufferToCPU(moment_opacity_m2_buffer, count * FLOAT32_BYTES),
      readBufferToCPU(moment_position_m1_buffer, count * TRAIN_POS_COMPONENTS * FLOAT32_BYTES),
      readBufferToCPU(moment_position_m2_buffer, count * TRAIN_POS_COMPONENTS * FLOAT32_BYTES),
      readBufferToCPU(moment_scale_m1_buffer, count * TRAIN_SCALE_COMPONENTS * FLOAT32_BYTES),
      readBufferToCPU(moment_scale_m2_buffer, count * TRAIN_SCALE_COMPONENTS * FLOAT32_BYTES),
      readBufferToCPU(moment_rotation_m1_buffer, count * TRAIN_ROT_COMPONENTS * FLOAT32_BYTES),
      readBufferToCPU(moment_rotation_m2_buffer, count * TRAIN_ROT_COMPONENTS * FLOAT32_BYTES),
      readBufferToCPU(moment_sh_m1_buffer, count * TRAIN_SH_COMPONENTS * FLOAT32_BYTES),
      readBufferToCPU(moment_sh_m2_buffer, count * TRAIN_SH_COMPONENTS * FLOAT32_BYTES),
    ]);

    return {
      opacity_m1, opacity_m2,
      pos_m1, pos_m2,
      scale_m1, scale_m2,
      rot_m1, rot_m2,
      sh_m1, sh_m2
    };
  }

  function writeAdamMoments(moments: AdamMoments) {
    if (moment_opacity_m1_buffer) device.queue.writeBuffer(moment_opacity_m1_buffer, 0, moments.opacity_m1);
    if (moment_opacity_m2_buffer) device.queue.writeBuffer(moment_opacity_m2_buffer, 0, moments.opacity_m2);

    if (moment_position_m1_buffer) device.queue.writeBuffer(moment_position_m1_buffer, 0, moments.pos_m1);
    if (moment_position_m2_buffer) device.queue.writeBuffer(moment_position_m2_buffer, 0, moments.pos_m2);

    if (moment_scale_m1_buffer) device.queue.writeBuffer(moment_scale_m1_buffer, 0, moments.scale_m1);
    if (moment_scale_m2_buffer) device.queue.writeBuffer(moment_scale_m2_buffer, 0, moments.scale_m2);

    if (moment_rotation_m1_buffer) device.queue.writeBuffer(moment_rotation_m1_buffer, 0, moments.rot_m1);
    if (moment_rotation_m2_buffer) device.queue.writeBuffer(moment_rotation_m2_buffer, 0, moments.rot_m2);

    if (moment_sh_m1_buffer) device.queue.writeBuffer(moment_sh_m1_buffer, 0, moments.sh_m1);
    if (moment_sh_m2_buffer) device.queue.writeBuffer(moment_sh_m2_buffer, 0, moments.sh_m2);
  }

  function resetAdamMoments(count: number): void {
    if (count <= 0) return;

    const zeroMomentsOpacity = new Float32Array(count);
    const zeroMomentsPos = new Float32Array(count * TRAIN_POS_COMPONENTS);
    const zeroMomentsScale = new Float32Array(count * TRAIN_SCALE_COMPONENTS);
    const zeroMomentsRot = new Float32Array(count * TRAIN_ROT_COMPONENTS);
    const zeroMomentsSH = new Float32Array(count * TRAIN_SH_COMPONENTS);

    if (moment_opacity_m1_buffer) device.queue.writeBuffer(moment_opacity_m1_buffer, 0, zeroMomentsOpacity);
    if (moment_opacity_m2_buffer) device.queue.writeBuffer(moment_opacity_m2_buffer, 0, zeroMomentsOpacity);
    if (moment_position_m1_buffer) device.queue.writeBuffer(moment_position_m1_buffer, 0, zeroMomentsPos);
    if (moment_position_m2_buffer) device.queue.writeBuffer(moment_position_m2_buffer, 0, zeroMomentsPos);
    if (moment_scale_m1_buffer) device.queue.writeBuffer(moment_scale_m1_buffer, 0, zeroMomentsScale);
    if (moment_scale_m2_buffer) device.queue.writeBuffer(moment_scale_m2_buffer, 0, zeroMomentsScale);
    if (moment_rotation_m1_buffer) device.queue.writeBuffer(moment_rotation_m1_buffer, 0, zeroMomentsRot);
    if (moment_rotation_m2_buffer) device.queue.writeBuffer(moment_rotation_m2_buffer, 0, zeroMomentsRot);
    if (moment_sh_m1_buffer) device.queue.writeBuffer(moment_sh_m1_buffer, 0, zeroMomentsSH);
    if (moment_sh_m2_buffer) device.queue.writeBuffer(moment_sh_m2_buffer, 0, zeroMomentsSH);
  }

  /**
   * Perform densification: clone and split gaussians based on accumulated gradients
   * This is done on CPU for simplicity - reads buffers, modifies, writes back.
   */
  async function performDensification(): Promise<{ cloned: number; split: number }> {
    if (!densificationManager || active_count === 0) {
      return { cloned: 0, split: 0 };
    }

    renderingSuspended = true;

    try {
      if (!train_position_buffer || !train_scale_buffer || !train_rotation_buffer ||
        !train_opacity_buffer || !train_sh_buffer) {
        console.warn('[densify] Training buffers not ready');
        return { cloned: 0, split: 0 };
      }

      const multiView = await getMultiViewScores();
      if (!multiView || multiView.sampleCount === 0) {
        densificationManager.resetAccumulators(active_count);
        rebuildDensifyAccumBindGroup();
        return { cloned: 0, split: 0 };
      }

      const gradBuffer = densificationManager.getGradAccumBuffer();
      if (!gradBuffer) {
        return { cloned: 0, split: 0 };
      }

      const gradData = await readBufferToCPU(gradBuffer, active_count * GRAD_ACCUM_STRIDE * FLOAT32_BYTES);
      const positions = await readBufferToCPU(train_position_buffer, active_count * TRAIN_POS_COMPONENTS * FLOAT32_BYTES);
      const scales = await readBufferToCPU(train_scale_buffer, active_count * TRAIN_SCALE_COMPONENTS * FLOAT32_BYTES);
      const rotations = await readBufferToCPU(train_rotation_buffer, active_count * TRAIN_ROT_COMPONENTS * FLOAT32_BYTES);
      const opacities = await readBufferToCPU(train_opacity_buffer, active_count * FLOAT32_BYTES);
      const shs = await readBufferToCPU(train_sh_buffer, active_count * TRAIN_SH_COMPONENTS * FLOAT32_BYTES);
      const params = await readAdamMoments(active_count);

      const config = densificationManager.getConfig();
      const sizeThreshold = densificationManager.getSizeThreshold();
      const actions = new Uint32Array(active_count);
      let cloneCount = 0;
      let splitCount = 0;

      for (let i = 0; i < active_count; i++) {
        const denom = gradData[i * GRAD_ACCUM_STRIDE + GRAD_ACCUM_DENOM];
        if (denom < Math.max(1, config.minVisibility)) {
          continue;
        }

        const avgGrad = gradData[i * GRAD_ACCUM_STRIDE + GRAD_ACCUM_X] / denom;
        const avgGradAbs = gradData[i * GRAD_ACCUM_STRIDE + GRAD_ACCUM_ABS] / denom;
        const scaleBase = i * TRAIN_SCALE_COMPONENTS;
        const maxScale = Math.max(
          Math.exp(scales[scaleBase]),
          Math.exp(scales[scaleBase + 1]),
          Math.exp(scales[scaleBase + 2])
        );
        const importanceCount = multiView.importance[i] / Math.max(1, multiView.sampleCount);

        if (importanceCount <= MULTI_VIEW_IMPORTANCE_THRESHOLD) {
          continue;
        }

        if (maxScale <= sizeThreshold && avgGrad >= config.gradThreshold) {
          actions[i] = 1;
          cloneCount++;
        } else if (maxScale > sizeThreshold && avgGradAbs >= config.gradAbsThreshold) {
          actions[i] = 2;
          splitCount++;
        }
      }

      if (cloneCount === 0 && splitCount === 0) {
        densificationManager.resetAccumulators(active_count);
        rebuildDensifyAccumBindGroup();
        return { cloned: 0, split: 0 };
      }

      const counts = { clones: cloneCount, splits: splitCount };
      const prevCount = active_count;
      const newCount = active_count + cloneCount + splitCount;
      const newPositions = new Float32Array(newCount * TRAIN_POS_COMPONENTS);
      const newScales = new Float32Array(newCount * TRAIN_SCALE_COMPONENTS);
      const newRotations = new Float32Array(newCount * TRAIN_ROT_COMPONENTS);
      const newOpacities = new Float32Array(newCount);
      const newSHs = new Float32Array(newCount * TRAIN_SH_COMPONENTS);
      const newMoments: AdamMoments = {
        opacity_m1: new Float32Array(newCount), opacity_m2: new Float32Array(newCount),
        pos_m1: new Float32Array(newCount * TRAIN_POS_COMPONENTS), pos_m2: new Float32Array(newCount * TRAIN_POS_COMPONENTS),
        scale_m1: new Float32Array(newCount * TRAIN_SCALE_COMPONENTS), scale_m2: new Float32Array(newCount * TRAIN_SCALE_COMPONENTS),
        rot_m1: new Float32Array(newCount * TRAIN_ROT_COMPONENTS), rot_m2: new Float32Array(newCount * TRAIN_ROT_COMPONENTS),
        sh_m1: new Float32Array(newCount * TRAIN_SH_COMPONENTS), sh_m2: new Float32Array(newCount * TRAIN_SH_COMPONENTS),
      };

      let writeIdx = 0;

      for (let i = 0; i < active_count; i++) {
        if (actions[i] === 2) {
          continue;
        }
        const posBase = i * TRAIN_POS_COMPONENTS;
        const scaleBase = i * TRAIN_SCALE_COMPONENTS;
        const rotBase = i * TRAIN_ROT_COMPONENTS;
        const shBase = i * TRAIN_SH_COMPONENTS;
        const newPosBase = writeIdx * TRAIN_POS_COMPONENTS;
        const newScaleBase = writeIdx * TRAIN_SCALE_COMPONENTS;
        const newRotBase = writeIdx * TRAIN_ROT_COMPONENTS;
        const newShBase = writeIdx * TRAIN_SH_COMPONENTS;
        newPositions.set(positions.slice(posBase, posBase + TRAIN_POS_COMPONENTS), newPosBase);
        newScales.set(scales.slice(scaleBase, scaleBase + TRAIN_SCALE_COMPONENTS), newScaleBase);
        newRotations.set(rotations.slice(rotBase, rotBase + TRAIN_ROT_COMPONENTS), newRotBase);
        newSHs.set(shs.slice(shBase, shBase + TRAIN_SH_COMPONENTS), newShBase);
        newOpacities[writeIdx] = opacities[i];
        newMoments.opacity_m1[writeIdx] = params.opacity_m1[i];
        newMoments.opacity_m2[writeIdx] = params.opacity_m2[i];
        newMoments.pos_m1.set(params.pos_m1.slice(posBase, posBase + TRAIN_POS_COMPONENTS), newPosBase);
        newMoments.pos_m2.set(params.pos_m2.slice(posBase, posBase + TRAIN_POS_COMPONENTS), newPosBase);
        newMoments.scale_m1.set(params.scale_m1.slice(scaleBase, scaleBase + TRAIN_SCALE_COMPONENTS), newScaleBase);
        newMoments.scale_m2.set(params.scale_m2.slice(scaleBase, scaleBase + TRAIN_SCALE_COMPONENTS), newScaleBase);
        newMoments.rot_m1.set(params.rot_m1.slice(rotBase, rotBase + TRAIN_ROT_COMPONENTS), newRotBase);
        newMoments.rot_m2.set(params.rot_m2.slice(rotBase, rotBase + TRAIN_ROT_COMPONENTS), newRotBase);
        newMoments.sh_m1.set(params.sh_m1.slice(shBase, shBase + TRAIN_SH_COMPONENTS), newShBase);
        newMoments.sh_m2.set(params.sh_m2.slice(shBase, shBase + TRAIN_SH_COMPONENTS), newShBase);
        writeIdx++;
      }

      for (let i = 0; i < active_count; i++) {
        const action = actions[i];
        if (action === 0) continue;
        const posBase = i * TRAIN_POS_COMPONENTS;
        const scaleBase = i * TRAIN_SCALE_COMPONENTS;
        const rotBase = i * TRAIN_ROT_COMPONENTS;
        const shBase = i * TRAIN_SH_COMPONENTS;
        const pos = [positions[posBase], positions[posBase + 1], positions[posBase + 2]];
        const scale = [scales[scaleBase], scales[scaleBase + 1], scales[scaleBase + 2]];
        const rot = [rotations[rotBase], rotations[rotBase + 1], rotations[rotBase + 2], rotations[rotBase + 3]];
        const sigma = [Math.exp(scale[0]), Math.exp(scale[1]), Math.exp(scale[2])];
        if (action === 1) {
          const newPosBase = writeIdx * TRAIN_POS_COMPONENTS;
          const newScaleBase = writeIdx * TRAIN_SCALE_COMPONENTS;
          const newRotBase = writeIdx * TRAIN_ROT_COMPONENTS;
          const newShBase = writeIdx * TRAIN_SH_COMPONENTS;
          const jitterScale = Math.max(Math.exp(scale[0]), Math.exp(scale[1]), Math.exp(scale[2]));
          const jitterMagnitude = jitterScale * 0.25 * (0.5 + Math.random() * 0.5);
          const jitterDir = [
            (Math.random() * 2 - 1),
            (Math.random() * 2 - 1),
            (Math.random() * 2 - 1),
          ];
          const norm = Math.hypot(jitterDir[0], jitterDir[1], jitterDir[2]) || 1.0;
          const jitter = jitterDir.map((v) => (v / norm) * jitterMagnitude);
          newPositions[newPosBase] = pos[0] + jitter[0];
          newPositions[newPosBase + 1] = pos[1] + jitter[1];
          newPositions[newPosBase + 2] = pos[2] + jitter[2];
          newScales.set(scale, newScaleBase);
          newRotations.set(rot, newRotBase);
          newSHs.set(shs.slice(shBase, shBase + TRAIN_SH_COMPONENTS), newShBase);
          newOpacities[writeIdx] = opacities[i];
          writeIdx++;
        } else if (action === 2) {
          const rotMat = quatToMat3(rot);
          for (let s = 0; s < 2; s++) {
            const localSample = sampleGaussian3D([0, 0, 0], sigma);
            const worldOffset = [
              rotMat[0][0] * localSample[0] + rotMat[0][1] * localSample[1] + rotMat[0][2] * localSample[2],
              rotMat[1][0] * localSample[0] + rotMat[1][1] * localSample[1] + rotMat[1][2] * localSample[2],
              rotMat[2][0] * localSample[0] + rotMat[2][1] * localSample[1] + rotMat[2][2] * localSample[2],
            ];
            const newPosBase = writeIdx * TRAIN_POS_COMPONENTS;
            const newScaleBase = writeIdx * TRAIN_SCALE_COMPONENTS;
            const newRotBase = writeIdx * TRAIN_ROT_COMPONENTS;
            const newShBase = writeIdx * TRAIN_SH_COMPONENTS;
            newPositions[newPosBase] = pos[0] + worldOffset[0];
            newPositions[newPosBase + 1] = pos[1] + worldOffset[1];
            newPositions[newPosBase + 2] = pos[2] + worldOffset[2];
            const scaleReduction = Math.log(0.8 * 2.0);
            newScales[newScaleBase] = scale[0] - scaleReduction;
            newScales[newScaleBase + 1] = scale[1] - scaleReduction;
            newScales[newScaleBase + 2] = scale[2] - scaleReduction;
            newRotations.set(rot, newRotBase);
            newSHs.set(shs.slice(shBase, shBase + TRAIN_SH_COMPONENTS), newShBase);
            newOpacities[writeIdx] = opacities[i];
            writeIdx++;
          }
        }
      }

      ensureCapacity(newCount);
      device.queue.writeBuffer(train_position_buffer!, 0, newPositions);
      device.queue.writeBuffer(train_scale_buffer!, 0, newScales);
      device.queue.writeBuffer(train_rotation_buffer!, 0, newRotations);
      device.queue.writeBuffer(train_opacity_buffer!, 0, newOpacities);
      device.queue.writeBuffer(train_sh_buffer!, 0, newSHs);
      writeAdamMoments(newMoments);
      setActiveCount(newCount);
      importanceManager?.ensureCapacity(newCount);
      importanceManager?.resetStats(newCount);
      importanceManager?.resetPruneActions(newCount);
      await syncGaussianBufferFromTrainingParams();
      rebuildAllBindGroups();
      densificationManager.ensureCapacity(newCount);
      densificationManager.resetAccumulators(newCount);
      rebuildDensifyAccumBindGroup();
      const pruned = await applyMultiViewPrune(multiView, prevCount);
      console.log(`[densify] Completed: ${prevCount} -> ${active_count} gaussians (${cloneCount} cloned, ${splitCount} split, ${pruned} pruned)`);
      await rebuildTilesForCurrentCamera();
      return { cloned: cloneCount, split: splitCount };
    } finally {
      renderingSuspended = false;
    }
  }

  /**
   * Sync the main gaussian buffer and SH buffer from training parameter buffers
   * This rebuilds the packed Float16 format used for rendering
   */
  async function syncGaussianBufferFromTrainingParams(): Promise<void> {
    if (!train_position_buffer || !train_scale_buffer || !train_rotation_buffer ||
      !train_opacity_buffer || !train_sh_buffer || !gaussian_buffer || !sh_buffer ||
      active_count === 0) {
      return;
    }

    // Read training params
    const positions = await readBufferToCPU(train_position_buffer, active_count * TRAIN_POS_COMPONENTS * FLOAT32_BYTES);
    const scales = await readBufferToCPU(train_scale_buffer, active_count * TRAIN_SCALE_COMPONENTS * FLOAT32_BYTES);
    const rotations = await readBufferToCPU(train_rotation_buffer, active_count * TRAIN_ROT_COMPONENTS * FLOAT32_BYTES);
    const opacities = await readBufferToCPU(train_opacity_buffer, active_count * FLOAT32_BYTES);
    const shs = await readBufferToCPU(train_sh_buffer, active_count * TRAIN_SH_COMPONENTS * FLOAT32_BYTES);

    // Import Float16Array for packing
    const { Float16Array } = await import('@petamoriken/float16');

    // Pack into gaussian buffer format
    // Layout per gaussian: pos[3], opacity[1], rot[4], scale[4] = 12 float16s
    const FLOATS_PER_GAUSSIAN = 12;
    const packedData = new Float16Array(active_count * FLOATS_PER_GAUSSIAN);

    for (let i = 0; i < active_count; i++) {
      const outBase = i * FLOATS_PER_GAUSSIAN;
      const posBase = i * TRAIN_POS_COMPONENTS;
      const scaleBase = i * TRAIN_SCALE_COMPONENTS;
      const rotBase = i * TRAIN_ROT_COMPONENTS;

      // Position (3)
      packedData[outBase + 0] = positions[posBase + 0];
      packedData[outBase + 1] = positions[posBase + 1];
      packedData[outBase + 2] = positions[posBase + 2];

      // Opacity (1)
      packedData[outBase + 3] = opacities[i];

      // Rotation (4)
      packedData[outBase + 4] = rotations[rotBase + 0];
      packedData[outBase + 5] = rotations[rotBase + 1];
      packedData[outBase + 6] = rotations[rotBase + 2];
      packedData[outBase + 7] = rotations[rotBase + 3];

      // Scale (4) - stored as log-scale
      packedData[outBase + 8] = scales[scaleBase + 0];
      packedData[outBase + 9] = scales[scaleBase + 1];
      packedData[outBase + 10] = scales[scaleBase + 2];
      packedData[outBase + 11] = 0; // padding or unused
    }

    device.queue.writeBuffer(gaussian_buffer, 0, packedData.buffer);

    // Pack SH buffer (Float32 -> Float16)
    // SH buffer layout: 16 coeffs * 3 channels = 48 float16s per gaussian
    // But stored as pairs in u32 (pack2x16float), so 24 u32s per gaussian
    const SH_FLOATS_PER_GAUSSIAN = TRAIN_SH_COMPONENTS; // 48
    const packedSH = new Float16Array(active_count * SH_FLOATS_PER_GAUSSIAN);

    for (let i = 0; i < active_count; i++) {
      const inBase = i * TRAIN_SH_COMPONENTS;
      const outBase = i * SH_FLOATS_PER_GAUSSIAN;

      for (let j = 0; j < TRAIN_SH_COMPONENTS; j++) {
        packedSH[outBase + j] = shs[inBase + j];
      }
    }

    device.queue.writeBuffer(sh_buffer, 0, packedSH.buffer);
    await device.queue.onSubmittedWorkDone();
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
    // NOTE: current_training_camera_index should be set by the caller before calling trainStep
    // This allows proper coordination with tile building which also needs the camera index
    // All callers (train(), profiledTrainStep(), etc.) now use random selection

    trainingIteration++;

    zeroGradBuffers();

    updateShDegreeSchedule();

    // Forward -> Loss -> Backward -> Adam
    run_training_tiled_forward(encoder);
    run_training_loss(encoder);
    run_training_tiled_backward(encoder);
    run_training_backward_geom(encoder);

    // Update visibility/importance stats and accumulate gradients before Adam modifies them
    run_importance_accumulate(encoder);
    run_densify_accumulate(encoder);

    run_adam_updates(encoder);
    rebuildTrainingApplyParamsBG();
    run_training_apply_params(encoder);

    camera.needsPreprocess = true;
  }

  async function profiledTrainStep(): Promise<void> {
    // Random camera selection (like FastGS) instead of cycling
    if (training_camera_buffers.length > 0) {
      current_training_camera_index = Math.floor(Math.random() * training_camera_buffers.length);
    }

    const stats = lastTileCoverageStats;
    profiler.setStats(
      active_count,
      stats ? stats.totalTiles : 0,
      stats ? stats.totalIntersections : 0
    );

    await profiler.startIteration(trainingIteration);
    trainingIteration++;

    await profiler.startPass('zeroGradBuffers');
    {
      const encoder = device.createCommandEncoder();
      zeroGradBuffers();
      device.queue.submit([encoder.finish()]);
    }
    await profiler.endPass('zeroGradBuffers');

    updateShDegreeSchedule();

    await profiler.startPass('forward');
    {
      const encoder = device.createCommandEncoder();
      run_training_tiled_forward(encoder);
      device.queue.submit([encoder.finish()]);
    }
    await profiler.endPass('forward');

    await profiler.startPass('loss');
    {
      const encoder = device.createCommandEncoder();
      run_training_loss(encoder);
      device.queue.submit([encoder.finish()]);
    }
    await profiler.endPass('loss');

    await profiler.startPass('backward_tiles');
    {
      const encoder = device.createCommandEncoder();
      run_training_tiled_backward(encoder);
      device.queue.submit([encoder.finish()]);
    }
    await profiler.endPass('backward_tiles');

    await profiler.startPass('backward_geom');
    {
      const encoder = device.createCommandEncoder();
      run_training_backward_geom(encoder);
      run_importance_accumulate(encoder);
      run_densify_accumulate(encoder);
      device.queue.submit([encoder.finish()]);
    }
    await profiler.endPass('backward_geom');

    await profiler.startPass('adam');
    {
      const encoder = device.createCommandEncoder();
      run_adam_updates(encoder);
      device.queue.submit([encoder.finish()]);
    }
    await profiler.endPass('adam');

    await profiler.startPass('apply_params');
    {
      const encoder = device.createCommandEncoder();
      rebuildTrainingApplyParamsBG();
      run_training_apply_params(encoder);
      device.queue.submit([encoder.finish()]);
    }
    await profiler.endPass('apply_params');

    await profiler.endIteration();

    camera.needsPreprocess = true;
  }

  async function train(numIterations: number): Promise<void> {
    if (numIterations <= 0) return;
    if (isTraining) {
      console.warn('[train] Training already in progress; ignoring new request.');
      return;
    }

    isTraining = true;
    stopTrainingRequested = false;
    trainingTargetIterations = numIterations;
    trainingStartTime = performance.now();
    trainingIterationsCompleted = 0;

    let lastLogTime = trainingStartTime;

    try {
      for (let i = 0; i < numIterations; i++) {
        // Check for stop request
        if (stopTrainingRequested) {
          break;
        }

        // Random camera selection (like FastGS) instead of cycling
        // This prevents bias from sequential camera ordering patterns
        const nextCameraIndex = training_camera_buffers.length > 0
          ? Math.floor(Math.random() * training_camera_buffers.length)
          : 0;

        // Always set the camera index for this iteration
        current_training_camera_index = nextCameraIndex;

        if (!tilesBuilt || tilesBuiltForCameraIndex !== nextCameraIndex) {
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

        trainingIterationsCompleted = i + 1;

        // Update training controller iteration (after trainStep incremented it)
        if (trainingController) {
          trainingController.setIteration(trainingIteration);
        }

        // Check if we should densify (clone/split based on accumulated gradients)
        if (trainingController && trainingController.shouldDensify()) {
          await device.queue.onSubmittedWorkDone();
          const result = await performDensification();
          if (result.cloned > 0 || result.split > 0) {
            // Tiles need to be rebuilt after densification
            tilesBuilt = false;
            tilesBuiltForCameraIndex = -1;
          }
        }

        // Check if we should prune (based on FastGS schedule)
        if (trainingController && trainingController.shouldPrune()) {
          await device.queue.onSubmittedWorkDone();
          const prevCount = active_count;
          await prune();
          if (active_count !== prevCount) {
            // Tiles need to be rebuilt after pruning
            tilesBuilt = false;
            tilesBuiltForCameraIndex = -1;
          }
        }

        // Check if we should reset opacity (helps break floater dominance)
        if (trainingController && trainingController.shouldResetOpacity()) {
          await device.queue.onSubmittedWorkDone();
          await resetOpacity(trainingController.getOpacityResetValue());
        }

        if ((i + 1) % 10 === 0) {
          await device.queue.onSubmittedWorkDone();

          const now = performance.now();
          if ((i + 1) % 50 === 0 || (now - lastLogTime) > 5000) {
            const elapsed = (now - trainingStartTime) / 1000;
            const iterPerSec = (i + 1) / elapsed;
            lastLogTime = now;
          }

          await new Promise<void>((resolve) => {
            requestAnimationFrame(() => resolve());
          });
        }
      }

      await device.queue.onSubmittedWorkDone();
      const totalTime = (performance.now() - trainingStartTime) / 1000;

    } finally {
      isTraining = false;
      stopTrainingRequested = false;
      camera.needsPreprocess = true;
    }
  }

  function stopTraining() {
    if (isTraining) {
      stopTrainingRequested = true;
      console.log('[train] Stop requested...');
    }
  }

  // ===============================================
  // Opacity Reset (for breaking floater dominance)
  // ===============================================

  /**
   * Reset all opacity values to a maximum cap (like FastGS).
   * This allows splats that were blocked by floaters to "compete" again.
   * FastGS does this every 3000 iterations with a cap of 0.01 (1%).
   * We use 10% since we don't have densification to add new splats.
   * 
   * CRITICAL: Also resets Adam momentum for opacity, scale, AND rotation
   * to prevent old momentum from driving continued floater growth.
   */
  async function resetOpacity(maxOpacity: number = 0.1): Promise<void> {
    if (!train_opacity_buffer || active_count === 0) return;

    // Convert opacity to logit: logit(p) = log(p / (1-p))
    const maxLogit = Math.log(maxOpacity / (1 - maxOpacity));

    // Read current opacity values
    const opacityBytes = active_count * FLOAT32_BYTES;
    const stagingRead = device.createBuffer({
      label: 'opacity reset staging read',
      size: opacityBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(train_opacity_buffer, 0, stagingRead, 0, opacityBytes);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await stagingRead.mapAsync(GPUMapMode.READ);
    const opacityData = new Float32Array(stagingRead.getMappedRange().slice(0));
    stagingRead.unmap();
    stagingRead.destroy();

    // Clamp opacity logits to maxLogit
    let numClamped = 0;
    for (let i = 0; i < active_count; i++) {
      if (opacityData[i] > maxLogit) {
        opacityData[i] = maxLogit;
        numClamped++;
      }
    }

    // Write back clamped values
    device.queue.writeBuffer(train_opacity_buffer, 0, opacityData);

    // CRITICAL: We do NOT reset Adam moments anymore, per alignment with FastGS.
    // Resetting them caused issues where splats lost their "velocity" and couldn't recover.
    // FastGS only modifies the opacity value itself.

    await device.queue.onSubmittedWorkDone();

    console.log(`[opacity reset] Capped ${numClamped}/${active_count} gaussians to max opacity ${(maxOpacity * 100).toFixed(1)}%`);
  }

  // ===============================================
  // Training Parameter Controls
  // ===============================================
  function getTrainingParams(): TrainingParams {
    return {
      lr_means: MEANS_LR,
      lr_sh: SHS_LR,
      lr_opacity: OPACITY_LR,
      lr_scale: SCALING_LR,
      lr_rotation: ROTATION_LR,
      sh_update_interval: SH_UPDATE_INTERVAL,
    };
  }

  function setTrainingParams(params: Partial<TrainingParams>): void {
    if (params.lr_means !== undefined) MEANS_LR = params.lr_means;
    if (params.lr_sh !== undefined) SHS_LR = params.lr_sh;
    if (params.lr_opacity !== undefined) OPACITY_LR = params.lr_opacity;
    if (params.lr_scale !== undefined) SCALING_LR = params.lr_scale;
    if (params.lr_rotation !== undefined) ROTATION_LR = params.lr_rotation;
    if (params.sh_update_interval !== undefined) SH_UPDATE_INTERVAL = params.sh_update_interval;
    console.log(`[Training Params] means=${MEANS_LR}, sh=${SHS_LR}, opacity=${OPACITY_LR}, scale=${SCALING_LR}, rotation=${ROTATION_LR}, sh_interval=${SH_UPDATE_INTERVAL}`);
  }

  function getTrainingStatus(): TrainingStatus {
    const elapsed = isTraining ? (performance.now() - trainingStartTime) / 1000 : 0;
    const iterPerSec = elapsed > 0 ? trainingIterationsCompleted / elapsed : 0;

    return {
      isTraining,
      currentIteration: trainingIteration,
      targetIterations: trainingTargetIterations,
      iterationsPerSecond: iterPerSec,
      gaussianCount: active_count,
    };
  }

  function getGaussianCount(): number {
    return active_count;
  }

  // Diagnostic function to check gradient and parameter statistics
  async function dumpTrainingDiagnostics(): Promise<void> {
    if (!train_scale_buffer || !grad_scale_buffer || !train_opacity_buffer || !grad_opacity_buffer) {
      console.log('[Diagnostics] Buffers not ready');
      return;
    }

    await device.queue.onSubmittedWorkDone();

    // Read scale params and grads
    const scaleSize = active_count * 3 * 4;
    const opacitySize = active_count * 4;

    const scaleStaging = device.createBuffer({
      size: scaleSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const gradScaleStaging = device.createBuffer({
      size: scaleSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const opacityStaging = device.createBuffer({
      size: opacitySize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const gradOpacityStaging = device.createBuffer({
      size: opacitySize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(train_scale_buffer, 0, scaleStaging, 0, scaleSize);
    encoder.copyBufferToBuffer(grad_scale_buffer, 0, gradScaleStaging, 0, scaleSize);
    encoder.copyBufferToBuffer(train_opacity_buffer, 0, opacityStaging, 0, opacitySize);
    encoder.copyBufferToBuffer(grad_opacity_buffer, 0, gradOpacityStaging, 0, opacitySize);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await scaleStaging.mapAsync(GPUMapMode.READ);
    await gradScaleStaging.mapAsync(GPUMapMode.READ);
    await opacityStaging.mapAsync(GPUMapMode.READ);
    await gradOpacityStaging.mapAsync(GPUMapMode.READ);

    const scaleData = new Float32Array(scaleStaging.getMappedRange().slice(0));
    const gradScaleData = new Float32Array(gradScaleStaging.getMappedRange().slice(0));
    const opacityData = new Float32Array(opacityStaging.getMappedRange().slice(0));
    const gradOpacityData = new Float32Array(gradOpacityStaging.getMappedRange().slice(0));

    scaleStaging.unmap();
    gradScaleStaging.unmap();
    opacityStaging.unmap();
    gradOpacityStaging.unmap();
    scaleStaging.destroy();
    gradScaleStaging.destroy();
    opacityStaging.destroy();
    gradOpacityStaging.destroy();

    // Compute statistics
    let scaleMin = Infinity, scaleMax = -Infinity, scaleSum = 0;
    let gradScaleMin = Infinity, gradScaleMax = -Infinity, gradScaleSum = 0, gradScaleNonZero = 0;
    let opacityMin = Infinity, opacityMax = -Infinity;
    let gradOpacityMin = Infinity, gradOpacityMax = -Infinity, gradOpacityNonZero = 0;

    for (let i = 0; i < scaleData.length; i++) {
      const v = scaleData[i];
      scaleMin = Math.min(scaleMin, v);
      scaleMax = Math.max(scaleMax, v);
      scaleSum += v;

      const gv = gradScaleData[i];
      if (Math.abs(gv) > 1e-10) {
        gradScaleMin = Math.min(gradScaleMin, gv);
        gradScaleMax = Math.max(gradScaleMax, gv);
        gradScaleSum += gv;
        gradScaleNonZero++;
      }
    }

    for (let i = 0; i < opacityData.length; i++) {
      const v = opacityData[i];
      opacityMin = Math.min(opacityMin, v);
      opacityMax = Math.max(opacityMax, v);

      const gv = gradOpacityData[i];
      if (Math.abs(gv) > 1e-10) {
        gradOpacityMin = Math.min(gradOpacityMin, gv);
        gradOpacityMax = Math.max(gradOpacityMax, gv);
        gradOpacityNonZero++;
      }
    }

    console.log('========== TRAINING DIAGNOSTICS ==========');
    console.log(`Iteration: ${trainingIteration}, Active Gaussians: ${active_count}`);
    console.log(`Scale (log-sigma): min=${scaleMin.toFixed(4)}, max=${scaleMax.toFixed(4)}, mean=${(scaleSum / scaleData.length).toFixed(4)}`);
    console.log(`Scale Grad: nonzero=${gradScaleNonZero}/${scaleData.length}, min=${gradScaleMin.toFixed(6)}, max=${gradScaleMax.toFixed(6)}`);
    console.log(`Opacity (logit): min=${opacityMin.toFixed(4)}, max=${opacityMax.toFixed(4)}`);
    console.log(`Opacity Grad: nonzero=${gradOpacityNonZero}/${opacityData.length}, min=${gradOpacityMin.toFixed(6)}, max=${gradOpacityMax.toFixed(6)}`);
    console.log(`Current LRs: scale=${SCALING_LR}, opacity=${OPACITY_LR}, sh=${SHS_LR}`);
    console.log(`Loss weights: L2=${currentLossWeights.w_l2}, L1=${currentLossWeights.w_l1}, SSIM=${currentLossWeights.w_ssim}`);
    console.log('===========================================');
  }

  // Loss weight storage
  let currentLossWeights: LossWeights = { w_l2: 0.2, w_l1: 0.8, w_ssim: 0.0 };

  function getLossWeights(): LossWeights {
    return { ...currentLossWeights };
  }

  function setLossWeights(weights: Partial<LossWeights>): void {
    if (weights.w_l2 !== undefined) currentLossWeights.w_l2 = weights.w_l2;
    if (weights.w_l1 !== undefined) currentLossWeights.w_l1 = weights.w_l1;
    if (weights.w_ssim !== undefined) currentLossWeights.w_ssim = weights.w_ssim;

    // Update the GPU buffer
    if (loss_params_buffer) {
      const data = new Float32Array([
        currentLossWeights.w_l2,
        currentLossWeights.w_l1,
        currentLossWeights.w_ssim,
        0.0,
      ]);
      device.queue.writeBuffer(loss_params_buffer, 0, data);
      console.log(`[Loss Weights] L2=${currentLossWeights.w_l2.toFixed(2)}, L1=${currentLossWeights.w_l1.toFixed(2)}, SSIM=${currentLossWeights.w_ssim.toFixed(2)}`);
    }
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
      if (renderingSuspended) {
        return;
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
    initializeKnnAndSync,
    prune,
    resetOpacity,
    setTrainingCameras,
    setTrainingImages,
    buildTrainingTiles: buildTileIntersections,
    enableProfiling: (enabled: boolean) => profiler.setEnabled(enabled),
    printProfilingReport: () => profiler.printReport(),
    profiledTrainStep,
    // Training controller access
    getTrainingController: () => trainingController,
    getSceneExtent: () => sceneExtent,
    // Training parameter controls
    getTrainingParams,
    setTrainingParams,
    getTrainingStatus,
    getGaussianCount,
    stopTraining,
    // Loss weight controls
    getLossWeights,
    setLossWeights,
    // Diagnostics
    dumpTrainingDiagnostics,
  };
}
