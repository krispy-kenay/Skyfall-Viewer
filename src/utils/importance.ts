export interface ImportanceConfig {
  decay: number;
  densifyImportanceThreshold: number;
  pruneImportanceThreshold: number;
  pruneVisibilityThreshold: number;
}

export const DEFAULT_IMPORTANCE_CONFIG: ImportanceConfig = {
  decay: 0.98,
  densifyImportanceThreshold: 2.0,
  pruneImportanceThreshold: 0.5,
  pruneVisibilityThreshold: 5,
};

const FLOAT32_BYTES = 4;
export const IMPORTANCE_STATS_STRIDE = 4;

export class ImportanceManager {
  private device: GPUDevice;
  private config: ImportanceConfig;
  private capacity = 0;
  private statsBuffer: GPUBuffer | null = null;
  private pruneActionBuffer: GPUBuffer | null = null;
  private pruneCountBuffer: GPUBuffer | null = null;

  constructor(device: GPUDevice, config: Partial<ImportanceConfig> = {}) {
    this.device = device;
    this.config = { ...DEFAULT_IMPORTANCE_CONFIG, ...config };
  }

  ensureCapacity(gaussianCount: number): void {
    if (this.capacity >= gaussianCount && this.statsBuffer) return;

    const newCapacity = Math.max(gaussianCount * 2, 1024);

    this.statsBuffer?.destroy();
    this.pruneActionBuffer?.destroy();
    this.pruneCountBuffer?.destroy();

    this.statsBuffer = this.device.createBuffer({
      label: 'importance stats buffer',
      size: newCapacity * IMPORTANCE_STATS_STRIDE * FLOAT32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    this.pruneActionBuffer = this.device.createBuffer({
      label: 'prune actions',
      size: newCapacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    this.pruneCountBuffer = this.device.createBuffer({
      label: 'prune counts',
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    this.capacity = newCapacity;
    this.resetStats(newCapacity);
    this.resetPruneActions(newCapacity);
  }

  getStatsBuffer(): GPUBuffer | null {
    return this.statsBuffer;
  }

  getPruneActionsBuffer(): GPUBuffer | null {
    return this.pruneActionBuffer;
  }

  getPruneCountBuffer(): GPUBuffer | null {
    return this.pruneCountBuffer;
  }

  getConfig(): ImportanceConfig {
    return this.config;
  }

  setConfig(config: Partial<ImportanceConfig>): void {
    this.config = { ...this.config, ...config };
  }

  createAccumParams(activeCount: number): Float32Array {
    return new Float32Array([
      activeCount,
      this.config.decay,
      0,
      0,
    ]);
  }

  createPruneParams(activeCount: number, thresholds: {
    minOpacity: number;
    maxScale: number;
    maxScreenRadius: number;
  }): Float32Array {
    return new Float32Array([
      thresholds.minOpacity,
      thresholds.maxScale,
      thresholds.maxScreenRadius,
      this.config.pruneImportanceThreshold,
      this.config.pruneVisibilityThreshold,
      activeCount,
      0,
      0,
    ]);
  }

  resetStats(gaussianCount: number): void {
    if (!this.statsBuffer || gaussianCount <= 0) return;
    const zeroData = new Float32Array(gaussianCount * IMPORTANCE_STATS_STRIDE);
    this.device.queue.writeBuffer(this.statsBuffer, 0, zeroData);
  }

  resetPruneActions(gaussianCount: number): void {
    if (this.pruneActionBuffer && gaussianCount > 0) {
      this.device.queue.writeBuffer(this.pruneActionBuffer, 0, new Uint32Array(gaussianCount));
    }
    if (this.pruneCountBuffer) {
      this.device.queue.writeBuffer(this.pruneCountBuffer, 0, new Uint32Array(4));
    }
  }

  async readPruneCounts(): Promise<{ removed: number }> {
    if (!this.pruneCountBuffer) return { removed: 0 };

    const staging = this.device.createBuffer({
      label: 'prune count staging',
      size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.pruneCountBuffer, 0, staging, 0, 16);
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    await staging.mapAsync(GPUMapMode.READ);
    const data = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    return { removed: data[0] };
  }
}
