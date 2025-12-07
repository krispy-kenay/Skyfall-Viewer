
export interface TrainingScheduleConfig {
  // Densification schedule
  densifyFromIter: number;
  densifyUntilIter: number;
  densificationInterval: number;

  // Pruning schedule
  pruneFromIter: number;
  pruneUntilIter: number;
  pruneInterval: number;

  // Opacity reset
  opacityResetInterval: number;
  opacityResetValue: number;

  // Pruning thresholds
  minOpacity: number;
  maxScreenSize: number;
  maxScaleRatio: number;
  maxScaleAbsolute: number;
}

export const DEFAULT_TRAINING_SCHEDULE: TrainingScheduleConfig = {
  // Densification
  densifyFromIter: 500,
  densifyUntilIter: 15000,
  densificationInterval: 100,

  // Pruning
  pruneFromIter: 500,
  pruneUntilIter: 25000,
  pruneInterval: 100,

  // Opacity reset
  opacityResetInterval: 3000,
  opacityResetValue: 0.01,

  // Pruning thresholds
  minOpacity: 0.005,
  maxScreenSize: 20.0,
  maxScaleRatio: 0.0,
  maxScaleAbsolute: 1.0,
};

// ===============================================
// Pruning Parameters (passed to shader)
// ===============================================
export interface PruneParams {
  minOpacity: number;
  maxScale: number;
  maxScreenRadius: number;
  useScreenSizePruning: boolean;
}

// ===============================================
// Training Controller
// ===============================================
export class TrainingController {
  private config: TrainingScheduleConfig;
  private sceneExtent: number;
  private iteration: number = 0;

  constructor(sceneExtent: number, config: Partial<TrainingScheduleConfig> = {}) {
    this.config = { ...DEFAULT_TRAINING_SCHEDULE, ...config };
    this.sceneExtent = sceneExtent;
  }

  /**
   * Update the current iteration count
   */
  setIteration(iteration: number): void {
    this.iteration = iteration;
  }

  /**
   * Get the current iteration
   */
  getIteration(): number {
    return this.iteration;
  }

  /**
   * Check if pruning should run at current iteration
   */
  shouldPrune(): boolean {
    const iter = this.iteration;
    const cfg = this.config;

    if (iter < cfg.pruneFromIter || iter > cfg.pruneUntilIter) {
      return false;
    }

    return iter % cfg.pruneInterval === 0;
  }

  /**
   * Check if densification should run at current iteration
   */
  shouldDensify(): boolean {
    const iter = this.iteration;
    const cfg = this.config;

    if (iter < cfg.densifyFromIter || iter > cfg.densifyUntilIter) {
      return false;
    }

    return iter % cfg.densificationInterval === 0;
  }

  /**
   * Check if opacity should be reset at current iteration
   */
  shouldResetOpacity(): boolean {
    const iter = this.iteration;
    const cfg = this.config;

    if (iter === 0) return false;

    // Only reset during densification phase
    if (iter < cfg.densifyFromIter || iter > cfg.densifyUntilIter) {
      return false;
    }

    return iter % cfg.opacityResetInterval === 0;
  }

  /**
   * Get pruning parameters for the current iteration
   */
  getPruneParams(): PruneParams {
    const cfg = this.config;
    const iter = this.iteration;

    const useScreenSizePruning = iter > cfg.opacityResetInterval;

    let maxScale: number;
    if (cfg.maxScaleAbsolute > 0) {
      maxScale = cfg.maxScaleAbsolute;
    } else if (cfg.maxScaleRatio > 0) {
      maxScale = this.sceneExtent * cfg.maxScaleRatio;
    } else {
      maxScale = 1e6;
    }

    return {
      minOpacity: cfg.minOpacity,
      maxScale,
      maxScreenRadius: useScreenSizePruning ? cfg.maxScreenSize : 10000.0,
      useScreenSizePruning,
    };
  }

  /**
   * Get the opacity reset value
   */
  getOpacityResetValue(): number {
    return this.config.opacityResetValue;
  }

  /**
   * Get current config
   */
  getConfig(): TrainingScheduleConfig {
    return this.config;
  }

  /**
   * Update scene extent
   */
  setSceneExtent(extent: number): void {
    this.sceneExtent = extent;
  }

  /**
   * Get scene extent
   */
  getSceneExtent(): number {
    return this.sceneExtent;
  }

  /**
   * Get a summary of what should happen at the current iteration
   */
  getIterationActions(): {
    shouldPrune: boolean;
    shouldDensify: boolean;
    shouldResetOpacity: boolean;
    pruneParams: PruneParams | null;
  } {
    const shouldPrune = this.shouldPrune();
    return {
      shouldPrune,
      shouldDensify: this.shouldDensify(),
      shouldResetOpacity: this.shouldResetOpacity(),
      pruneParams: shouldPrune ? this.getPruneParams() : null,
    };
  }

  /**
   * Log current state for debugging
   */
  logState(): void {
    const actions = this.getIterationActions();
    console.log(`[TrainingController] Iteration ${this.iteration}:`);
    console.log(`  - Scene extent: ${this.sceneExtent.toFixed(2)}`);
    console.log(`  - Should prune: ${actions.shouldPrune}`);
    console.log(`  - Should densify: ${actions.shouldDensify}`);
    console.log(`  - Should reset opacity: ${actions.shouldResetOpacity}`);
    if (actions.pruneParams) {
      console.log(`  - Prune params:`);
      console.log(`    - minOpacity: ${actions.pruneParams.minOpacity}`);
      console.log(`    - maxScale: ${actions.pruneParams.maxScale.toFixed(4)}`);
      console.log(`    - maxScreenRadius: ${actions.pruneParams.maxScreenRadius}`);
      console.log(`    - useScreenSizePruning: ${actions.pruneParams.useScreenSizePruning}`);
    }
  }
}

// ===============================================
// Helper to create prune uniform buffer data
// ===============================================
export function createPruneUniformData(params: PruneParams): Float32Array {
  const flags = params.useScreenSizePruning ? 1.0 : 0.0;
  return new Float32Array([
    params.minOpacity,
    params.maxScale,
    params.maxScreenRadius,
    flags,
  ]);
}

