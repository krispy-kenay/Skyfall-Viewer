export interface GradientStats {
  mean: number;
  max: number;
  min: number;
  std: number;
  nonZeroCount: number;
  totalCount: number;
  histogram: number[];
}

export interface TrainingDebugReport {
  iteration: number;
  timestamp: number;
  
  gradients: {
    sh_dc: GradientStats;
    sh_higher: GradientStats;
    opacity: GradientStats;
    position: GradientStats;
    scale: GradientStats;
    rotation: GradientStats;
    alpha_splat: GradientStats;
  };
  
  pixelDebug?: {
    pixelX: number;
    pixelY: number;
    numContributingSplats: number;
    forwardTrace: Array<{
      splatIndex: number;
      depth: number;
      alpha: number;
      color: [number, number, number];
      accumAlpha: number;
      accumColor: [number, number, number];
    }>;
    backwardTrace: Array<{
      splatIndex: number;
      dL_dcolor: [number, number, number];
      dL_dalpha: number;
    }>;
  };
  
  sampledSplats?: Array<{
    index: number;
    position: [number, number, number];
    scale: [number, number, number];
    opacity: number;
    shDC: [number, number, number];
    gradPosition: [number, number, number];
    gradScale: [number, number, number];
    gradOpacity: number;
    gradShDC: [number, number, number];
    gradConic: [number, number, number];
    gradMean2d: [number, number];
  }>;
}

export function computeGradientStats(data: Float32Array): GradientStats {
  const totalCount = data.length;
  if (totalCount === 0) {
    return {
      mean: 0, max: 0, min: 0, std: 0,
      nonZeroCount: 0, totalCount: 0,
      histogram: new Array(20).fill(0)
    };
  }
  
  let sum = 0;
  let sumSq = 0;
  let max = -Infinity;
  let min = Infinity;
  let nonZeroCount = 0;
  
  const histogram = new Array(22).fill(0);
  
  for (let i = 0; i < totalCount; i++) {
    const v = data[i];
    const absV = Math.abs(v);
    
    sum += v;
    sumSq += v * v;
    if (absV > max) max = absV;
    if (absV < min) min = absV;
    if (absV > 1e-15) nonZeroCount++;
    
    if (absV < 1e-10) {
      histogram[0]++;
    } else if (absV >= 1e10) {
      histogram[21]++;
    } else {
      const bin = Math.floor(Math.log10(absV) + 10) + 1;
      histogram[Math.min(21, Math.max(0, bin))]++;
    }
  }
  
  const mean = sum / totalCount;
  const variance = (sumSq / totalCount) - (mean * mean);
  const std = Math.sqrt(Math.max(0, variance));
  
  return { mean, max, min, std, nonZeroCount, totalCount, histogram };
}

export function formatGradientStats(name: string, stats: GradientStats): string {
  const nonZeroPct = (100 * stats.nonZeroCount / stats.totalCount).toFixed(2);
  return `${name.padEnd(12)}: mean=${stats.mean.toExponential(2)}, max=${stats.max.toExponential(2)}, ` +
         `std=${stats.std.toExponential(2)}, nonzero=${nonZeroPct}% (${stats.nonZeroCount}/${stats.totalCount})`;
}

export function formatDebugReport(report: TrainingDebugReport): string {
  const lines: string[] = [];
  lines.push(`\n=== Training Debug Report (iter ${report.iteration}) ===`);
  lines.push(`Timestamp: ${new Date(report.timestamp).toISOString()}`);
  lines.push('');
  lines.push('--- Gradient Statistics ---');
  
  for (const [name, stats] of Object.entries(report.gradients)) {
    lines.push(formatGradientStats(name, stats));
  }
  
  if (report.pixelDebug) {
    const pd = report.pixelDebug;
    lines.push('');
    lines.push(`--- Pixel Debug (${pd.pixelX}, ${pd.pixelY}) ---`);
    lines.push(`Contributing splats: ${pd.numContributingSplats}`);
    
    if (pd.forwardTrace.length > 0) {
      lines.push('Forward trace:');
      for (const t of pd.forwardTrace.slice(0, 10)) {
        lines.push(`  splat ${t.splatIndex}: depth=${t.depth.toFixed(3)}, alpha=${t.alpha.toFixed(4)}, ` +
                   `color=[${t.color.map(c => c.toFixed(3)).join(',')}], accumAlpha=${t.accumAlpha.toFixed(4)}`);
      }
      if (pd.forwardTrace.length > 10) {
        lines.push(`  ... and ${pd.forwardTrace.length - 10} more`);
      }
    }
    
    if (pd.backwardTrace.length > 0) {
      lines.push('Backward trace:');
      for (const t of pd.backwardTrace.slice(0, 10)) {
        lines.push(`  splat ${t.splatIndex}: dL_dalpha=${t.dL_dalpha.toExponential(2)}, ` +
                   `dL_dcolor=[${t.dL_dcolor.map(c => c.toExponential(2)).join(',')}]`);
      }
    }
  }
  
  if (report.sampledSplats && report.sampledSplats.length > 0) {
    lines.push('');
    lines.push(`--- Sampled Splats (${report.sampledSplats.length}) ---`);
    
    const sorted = [...report.sampledSplats].sort((a, b) => {
      const magA = Math.abs(a.gradOpacity) + Math.abs(a.gradAlphaSplat);
      const magB = Math.abs(b.gradOpacity) + Math.abs(b.gradAlphaSplat);
      return magB - magA;
    });
    
    lines.push('Top 5 by opacity+alpha gradient:');
    for (const s of sorted.slice(0, 5)) {
      lines.push(`  #${s.index}: pos=[${s.position.map(p => p.toFixed(2)).join(',')}], ` +
                 `scale=[${s.scale.map(sc => sc.toFixed(3)).join(',')}], opacity=${s.opacity.toFixed(3)}, ` +
                 `gradOp=${s.gradOpacity.toExponential(2)}, gradAlpha=${s.gradAlphaSplat.toExponential(2)}`);
    }
    
    lines.push('Bottom 5 (lowest gradients):');
    for (const s of sorted.slice(-5).reverse()) {
      lines.push(`  #${s.index}: pos=[${s.position.map(p => p.toFixed(2)).join(',')}], ` +
                 `scale=[${s.scale.map(sc => sc.toFixed(3)).join(',')}], opacity=${s.opacity.toFixed(3)}, ` +
                 `gradOp=${s.gradOpacity.toExponential(2)}, gradAlpha=${s.gradAlphaSplat.toExponential(2)}`);
    }
  }
  
  lines.push('');
  return lines.join('\n');
}

export function downloadDebugReport(report: TrainingDebugReport): void {
  const json = JSON.stringify(report, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `training_debug_iter${report.iteration}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

