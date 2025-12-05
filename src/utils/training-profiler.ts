
export interface PassTiming {
  name: string;
  durationMs: number;
}

export interface TrainingProfile {
  iteration: number;
  totalMs: number;
  passes: PassTiming[];
  splatsProcessed: number;
  tilesProcessed: number;
  intersections: number;
}

export interface AggregateProfile {
  iterationCount: number;
  avgTotalMs: number;
  avgPassTimes: Map<string, number>;
  minTotalMs: number;
  maxTotalMs: number;
  splatsProcessed: number;
  tilesProcessed: number;
  intersections: number;
}

export class TrainingProfiler {
  private device: GPUDevice;
  private enabled: boolean = false;
  private profiles: TrainingProfile[] = [];
  private currentPasses: PassTiming[] = [];
  private passStartTime: number = 0;
  private iterationStartTime: number = 0;
  private currentIteration: number = 0;
  
  // Stats for current profile
  private splatsProcessed: number = 0;
  private tilesProcessed: number = 0;
  private intersections: number = 0;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (enabled) {
      console.log('[Profiler] Enabled - NOTE: This adds GPU sync overhead!');
      this.profiles = [];
    } else {
      console.log('[Profiler] Disabled');
    }
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  setStats(splats: number, tiles: number, intersections: number): void {
    this.splatsProcessed = splats;
    this.tilesProcessed = tiles;
    this.intersections = intersections;
  }

  async startIteration(iteration: number): Promise<void> {
    if (!this.enabled) return;
    
    await this.device.queue.onSubmittedWorkDone();
    
    this.currentIteration = iteration;
    this.currentPasses = [];
    this.iterationStartTime = performance.now();
  }

  async startPass(name: string): Promise<void> {
    if (!this.enabled) return;
    
    await this.device.queue.onSubmittedWorkDone();
    this.passStartTime = performance.now();
  }

  async endPass(name: string): Promise<void> {
    if (!this.enabled) return;
    
    await this.device.queue.onSubmittedWorkDone();
    
    const durationMs = performance.now() - this.passStartTime;
    this.currentPasses.push({ name, durationMs });
  }

  async endIteration(): Promise<void> {
    if (!this.enabled) return;
    
    await this.device.queue.onSubmittedWorkDone();
    
    const totalMs = performance.now() - this.iterationStartTime;
    
    this.profiles.push({
      iteration: this.currentIteration,
      totalMs,
      passes: [...this.currentPasses],
      splatsProcessed: this.splatsProcessed,
      tilesProcessed: this.tilesProcessed,
      intersections: this.intersections,
    });
    
    if (this.profiles.length > 100) {
      this.profiles.shift();
    }
  }

  getLastProfile(): TrainingProfile | null {
    return this.profiles.length > 0 ? this.profiles[this.profiles.length - 1] : null;
  }

  getAggregateProfile(lastN: number = 10): AggregateProfile | null {
    if (this.profiles.length === 0) return null;
    
    const samples = this.profiles.slice(-lastN);
    const n = samples.length;
    
    const passTotals = new Map<string, number>();
    let totalSum = 0;
    let minTotal = Infinity;
    let maxTotal = 0;
    
    for (const profile of samples) {
      totalSum += profile.totalMs;
      minTotal = Math.min(minTotal, profile.totalMs);
      maxTotal = Math.max(maxTotal, profile.totalMs);
      
      for (const pass of profile.passes) {
        passTotals.set(pass.name, (passTotals.get(pass.name) || 0) + pass.durationMs);
      }
    }
    
    const avgPassTimes = new Map<string, number>();
    for (const [name, total] of passTotals) {
      avgPassTimes.set(name, total / n);
    }
    
    const lastProfile = samples[samples.length - 1];
    
    return {
      iterationCount: n,
      avgTotalMs: totalSum / n,
      avgPassTimes,
      minTotalMs: minTotal,
      maxTotalMs: maxTotal,
      splatsProcessed: lastProfile.splatsProcessed,
      tilesProcessed: lastProfile.tilesProcessed,
      intersections: lastProfile.intersections,
    };
  }

  printReport(lastN: number = 10): void {
    const agg = this.getAggregateProfile(lastN);
    if (!agg) {
      console.log('[Profiler] No data collected yet');
      return;
    }
    
    console.log('\n========== TRAINING PERFORMANCE PROFILE ==========');
    console.log(`Samples: ${agg.iterationCount} iterations`);
    console.log(`Splats: ${agg.splatsProcessed.toLocaleString()}`);
    console.log(`Tiles: ${agg.tilesProcessed.toLocaleString()}`);
    console.log(`Intersections: ${agg.intersections.toLocaleString()}`);
    console.log(`\nTotal iteration time: ${agg.avgTotalMs.toFixed(2)}ms avg (${agg.minTotalMs.toFixed(2)}-${agg.maxTotalMs.toFixed(2)}ms range)`);
    console.log(`Theoretical max: ${(1000 / agg.avgTotalMs).toFixed(1)} iter/s`);
    console.log('\n--- Pass Breakdown ---');
    
    const sortedPasses = [...agg.avgPassTimes.entries()].sort((a, b) => b[1] - a[1]);
    
    const totalPassTime = sortedPasses.reduce((sum, [, t]) => sum + t, 0);
    
    for (const [name, avgMs] of sortedPasses) {
      const pct = (avgMs / totalPassTime * 100).toFixed(1);
      const bar = '█'.repeat(Math.round(avgMs / totalPassTime * 30));
      console.log(`  ${name.padEnd(25)} ${avgMs.toFixed(2).padStart(8)}ms (${pct.padStart(5)}%) ${bar}`);
    }
    
    console.log('\n--- Overhead Analysis ---');
    const overhead = agg.avgTotalMs - totalPassTime;
    console.log(`  Sync/overhead: ${overhead.toFixed(2)}ms (${(overhead / agg.avgTotalMs * 100).toFixed(1)}%)`);
    console.log('====================================================\n');
  }

  clear(): void {
    this.profiles = [];
    this.currentPasses = [];
  }
}

let globalProfiler: TrainingProfiler | null = null;

export function setGlobalProfiler(profiler: TrainingProfiler): void {
  globalProfiler = profiler;
}

export function getGlobalProfiler(): TrainingProfiler | null {
  return globalProfiler;
}

