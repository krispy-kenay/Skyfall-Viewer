/**
 * Densification Manager - Handles gradient accumulation and clone/split decisions
 * Based on FastGS densification logic
 * 
 * The densification process:
 * 1. Every iteration: accumulate 2D position gradient norms for visible gaussians
 * 2. Every densificationInterval iterations: 
 *    - Compute average gradient per gaussian
 *    - Clone small gaussians with high gradient (under-reconstruction)
 *    - Split large gaussians with high gradient (over-reconstruction)
 *    - Reset accumulators
 */

export interface DensificationConfig {
  // Thresholds
  gradThreshold: number;      // Clone if avg gradient norm >= this (default: 0.0002)
  gradAbsThreshold: number;   // Split if avg abs gradient norm >= this (default: 0.0012)
  denseSizeRatio: number;     // Size threshold = scene_extent * this (default: 0.001)
  minVisibility: number;      // Require at least this many visible frames before densifying
  
  // After densification, cap opacity at this value
  postDensifyOpacityCap: number;  // Default: 0.8
}

export const DEFAULT_DENSIFICATION_CONFIG: DensificationConfig = {
  gradThreshold: 0.002,
  gradAbsThreshold: 0.012,
  denseSizeRatio: 0.001,
  minVisibility: 15,
  postDensifyOpacityCap: 0.8,
};

// Buffer layout for gradient accumulation
// Per gaussian: [grad_2d_norm_accum, unused, grad_abs_norm_accum, denom]
// Note: We accumulate the norm directly (like FastGS), not individual components
export const GRAD_ACCUM_STRIDE = 4;
export const GRAD_ACCUM_X = 0;  // Stores accumulated 2D gradient norm
export const GRAD_ACCUM_Y = 1;  // Unused (kept for buffer layout compatibility)
export const GRAD_ACCUM_ABS = 2;  // Stores accumulated abs gradient norm
export const GRAD_ACCUM_DENOM = 3;  // Count of times gaussian was visible

/**
 * Densification Manager
 * Manages buffers and shaders for gradient accumulation and densification
 */
export class DensificationManager {
  private device: GPUDevice;
  private config: DensificationConfig;
  private sceneExtent: number;
  
  // Buffers
  private gradAccumBuffer: GPUBuffer | null = null;
  private densifyOutputBuffer: GPUBuffer | null = null;  // Clone/split decisions
  private densifyCountBuffer: GPUBuffer | null = null;   // Number of clones/splits
  
  // Pipelines
  private accumPipeline: GPUComputePipeline | null = null;
  private densifyPipeline: GPUComputePipeline | null = null;
  
  // Bind group layouts
  private accumBindGroupLayout: GPUBindGroupLayout | null = null;
  private densifyBindGroupLayout: GPUBindGroupLayout | null = null;
  
  // Current capacity
  private capacity: number = 0;
  
  constructor(device: GPUDevice, sceneExtent: number, config: Partial<DensificationConfig> = {}) {
    this.device = device;
    this.sceneExtent = sceneExtent;
    this.config = { ...DEFAULT_DENSIFICATION_CONFIG, ...config };
  }
  
  /**
   * Initialize or resize buffers for the given gaussian count
   */
  ensureCapacity(gaussianCount: number): void {
    if (this.capacity >= gaussianCount && this.gradAccumBuffer) {
      return;
    }
    
    // Add some headroom for new gaussians from densification
    const newCapacity = Math.max(gaussianCount * 2, 1024);
    
    // Clean up old buffers
    this.gradAccumBuffer?.destroy();
    this.densifyOutputBuffer?.destroy();
    this.densifyCountBuffer?.destroy();
    
    // Gradient accumulation buffer
    // Per gaussian: [grad_2d_norm_accum, unused, grad_abs_norm_accum, denom]
    this.gradAccumBuffer = this.device.createBuffer({
      label: 'densification grad accum',
      size: newCapacity * GRAD_ACCUM_STRIDE * 4,  // 4 floats per gaussian
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    
    // Densification output buffer
    // Per gaussian: action (0=none, 1=clone, 2=split)
    this.densifyOutputBuffer = this.device.createBuffer({
      label: 'densification output',
      size: newCapacity * 4,  // 1 u32 per gaussian
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    
    // Count buffer: [num_clones, num_splits, total_new_gaussians, 0]
    // Note: MAP_READ can only be combined with COPY_DST, so we use staging buffer for readback
    this.densifyCountBuffer = this.device.createBuffer({
      label: 'densification counts',
      size: 16,  // 4 u32s
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    
    this.capacity = newCapacity;
    
    // Zero the buffers
    const zeroData = new Float32Array(newCapacity * GRAD_ACCUM_STRIDE);
    this.device.queue.writeBuffer(this.gradAccumBuffer, 0, zeroData);
    
    const zeroOutput = new Uint32Array(newCapacity);
    this.device.queue.writeBuffer(this.densifyOutputBuffer, 0, zeroOutput);
    
    const zeroCounts = new Uint32Array(4);
    this.device.queue.writeBuffer(this.densifyCountBuffer, 0, zeroCounts);
  }
  
  /**
   * Reset gradient accumulators (called after densification)
   */
  resetAccumulators(gaussianCount: number): void {
    if (!this.gradAccumBuffer) return;
    
    const zeroData = new Float32Array(gaussianCount * GRAD_ACCUM_STRIDE);
    this.device.queue.writeBuffer(this.gradAccumBuffer, 0, zeroData);
  }
  
  /**
   * Get the gradient accumulation buffer
   */
  getGradAccumBuffer(): GPUBuffer | null {
    return this.gradAccumBuffer;
  }
  
  /**
   * Get the densification output buffer
   */
  getDensifyOutputBuffer(): GPUBuffer | null {
    return this.densifyOutputBuffer;
  }
  
  /**
   * Get the count buffer
   */
  getCountBuffer(): GPUBuffer | null {
    return this.densifyCountBuffer;
  }
  
  /**
   * Get current config
   */
  getConfig(): DensificationConfig {
    return this.config;
  }
  
  /**
   * Update config
   */
  setConfig(config: Partial<DensificationConfig>): void {
    this.config = { ...this.config, ...config };
  }
  
  /**
   * Update scene extent
   */
  setSceneExtent(extent: number): void {
    this.sceneExtent = extent;
  }
  
  /**
   * Get the size threshold for clone vs split decision
   */
  getSizeThreshold(): number {
    return this.sceneExtent * this.config.denseSizeRatio;
  }
  
  /**
   * Create uniform buffer data for densification shader
   */
  createDensifyUniformData(gaussianCount: number): Float32Array {
    return new Float32Array([
      this.config.gradThreshold,
      this.config.gradAbsThreshold,
      this.getSizeThreshold(),
      this.config.postDensifyOpacityCap,
      gaussianCount,
      this.config.minVisibility,
      0,
      0,
    ]);
  }
  
  /**
   * Read back the clone/split counts after densification
   */
  async readCounts(): Promise<{ clones: number; splits: number; total: number }> {
    if (!this.densifyCountBuffer) {
      return { clones: 0, splits: 0, total: 0 };
    }
    
    // Create staging buffer for readback
    const stagingBuffer = this.device.createBuffer({
      label: 'densify count staging',
      size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.densifyCountBuffer, 0, stagingBuffer, 0, 16);
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const data = new Uint32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();
    stagingBuffer.destroy();
    
    return {
      clones: data[0],
      splits: data[1],
      total: data[2],
    };
  }
  
  /**
   * Reset the count buffer
   */
  resetCounts(): void {
    if (!this.densifyCountBuffer) return;
    const zeroCounts = new Uint32Array(4);
    this.device.queue.writeBuffer(this.densifyCountBuffer, 0, zeroCounts);
  }
  
  /**
   * Clean up resources
   */
  destroy(): void {
    this.gradAccumBuffer?.destroy();
    this.densifyOutputBuffer?.destroy();
    this.densifyCountBuffer?.destroy();
    this.gradAccumBuffer = null;
    this.densifyOutputBuffer = null;
    this.densifyCountBuffer = null;
    this.capacity = 0;
  }
}
