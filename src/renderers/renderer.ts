import type { PointCloud } from '../utils/load';
import { Pane } from 'tweakpane';
import * as TweakpaneFileImportPlugin from 'tweakpane-plugin-file-import';
import { default as get_renderer_gaussian, GaussianRenderer, TrainingParams, LossWeights } from './gaussian-renderer';
import { default as get_renderer_pointcloud } from './point-cloud-renderer';
import { Camera, TrainingCameraData } from '../camera/camera';
import { CameraControl } from '../camera/camera-control';
import { time, timeReturn } from '../utils/simple-console';
import { loadColmapDatasetScene } from '../utils/dataset-loader';

export interface Renderer {
  frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => void,
  camera_buffer: GPUBuffer,
}

export default async function init(
  canvas: HTMLCanvasElement,
  context: GPUCanvasContext,
  device: GPUDevice
) {
  let dataset_loaded = false;
  let isBenchmarking = false;
  let benchmarkSum = 0;
  let benchmarkFrames = 0;
  let renderers: { pointcloud?: Renderer, gaussian?: Renderer } = {};
  let gaussian_renderer: GaussianRenderer | undefined;
  let pointcloud_renderer: Renderer | undefined;
  let renderer: Renderer | undefined;
  let training_cameras_cache: TrainingCameraData[] = [];
  let currentTrainingCameraIndex = 0;
  let currentPointCloud: PointCloud | undefined;

  const camera = new Camera(canvas, device);
  const control = new CameraControl(camera);

  const observer = new ResizeObserver(() => {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    camera.on_update_canvas();
  });
  observer.observe(canvas);

  const presentation_format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentation_format,
    alphaMode: 'opaque',
  });

  const params = {
    fps: 0.0,
    gaussian_multiplier: 1,
    renderer: 'pointcloud',
    benchFrames: 300,
  };

  const pane = new Pane({
    title: 'Config',
    expanded: true,
  });
  pane.registerPlugin(TweakpaneFileImportPlugin);
  {
    pane.addMonitor(params, 'fps', {
      readonly: true
    });
  }
  {
    const f = pane.addFolder({ title: '3DGS / COLMAP Dataset' });
    f.addButton({ title: 'Load Dataset Folder' }).on('click', async () => {
      // @ts-ignore - File System Access API
      if (!window.showDirectoryPicker) {
        alert('Your browser does not support folder selection. Please use a Chromium-based browser (Chrome, Edge, etc.).');
        return;
      }

      try {
        // @ts-ignore
        const dirHandle: FileSystemDirectoryHandle = await window.showDirectoryPicker();

        const dataset = await loadColmapDatasetScene(dirHandle, device);

        currentPointCloud = dataset.pointCloud;

        pointcloud_renderer = get_renderer_pointcloud(dataset.pointCloud, device, presentation_format, camera.uniform_buffer);
        gaussian_renderer = get_renderer_gaussian(dataset.pointCloud, device, presentation_format, camera.uniform_buffer, canvas, camera);
        (window as any).gaussian_renderer = gaussian_renderer;
        renderers = {
          pointcloud: pointcloud_renderer,
          gaussian: gaussian_renderer,
        };
        renderer = renderers[params.renderer];

        training_cameras_cache = dataset.trainingCameras;
        if (gaussian_renderer) {
          gaussian_renderer.setTrainingCameras(dataset.trainingCameras);
          gaussian_renderer.setTrainingImages(dataset.trainingImages);

          console.log("Initializing k-NN density-aware scales...");
          await gaussian_renderer.initializeKnnAndSync();
          console.log("k-NN initialization complete.");

          // Sync GUI with renderer's training params
          syncTrainingParamsFromRenderer();
          updateTrainingStatus();
        }

        dataset_loaded = true;
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          console.error('Failed to load COLMAP/3DGS dataset:', err);
          alert(`Failed to load COLMAP/3DGS dataset: ${(err as Error).message}`);
        }
      }
    });
  }
  {
    pane.addInput(params, 'renderer', {
      options: {
        pointcloud: 'pointcloud',
        gaussian: 'gaussian',
      }
    }).on('change', (e) => {
      renderer = renderers[e.value];
    });
  }
  {
    pane.addInput(
      params,
      'gaussian_multiplier',
      { min: 0, max: 1.5 }
    ).on('change', (e) => {
      clearTimeout(scaleDebounce);
      scaleDebounce = setTimeout(() => gaussian_renderer.setGaussianScale(e.value), 25);
    });
    let scaleDebounce: any = null;
  }
  // Training controls
  let trainingFolder: any;
  const trainingParams = {
    // Control params
    iterations: 1000,
    // Status display
    statusText: 'Not started',
    iterPerSec: 0,
    gaussianCount: 0,
    // Learning rates
    lr_sh: 0.0025,
    lr_opacity: 0.025,
    lr_scale: 0.005,
    lr_rotation: 0.001,
    lr_means: 0.00016,
    sh_update_interval: 16,  // Only update SH every N iterations
    // Pruning settings
    pruneFromIter: 500,
    pruneInterval: 100,
    maxScale: 1.0,
    minOpacity: 0.005,
    // Opacity reset settings
    opacityResetInterval: 3000,
    opacityResetValue: 0.01,
    // Loss weights
    loss_l2: 0.0,
    loss_l1: 0.8,
    loss_ssim: 0.2,
  };

  // Status update interval
  let statusUpdateInterval: number | null = null;

  function updateTrainingStatus() {
    if (!gaussian_renderer) return;

    const status = gaussian_renderer.getTrainingStatus();
    trainingParams.gaussianCount = status.gaussianCount;
    trainingParams.iterPerSec = status.iterationsPerSecond;

    if (status.isTraining) {
      trainingParams.statusText = `Training: ${status.currentIteration} / ${status.targetIterations}`;
    } else if (status.currentIteration > 0) {
      trainingParams.statusText = `Done: ${status.currentIteration} iterations`;
    } else {
      trainingParams.statusText = 'Not started';
    }

    pane.refresh();
  }

  function startStatusUpdates() {
    if (statusUpdateInterval) return;
    statusUpdateInterval = window.setInterval(updateTrainingStatus, 200);
  }

  function stopStatusUpdates() {
    if (statusUpdateInterval) {
      window.clearInterval(statusUpdateInterval);
      statusUpdateInterval = null;
    }
  }

  {
    trainingFolder = pane.addFolder({ title: 'Training' });

    // Status displays
    trainingFolder.addMonitor(trainingParams, 'statusText', {
      label: 'Status',
    });
    trainingFolder.addMonitor(trainingParams, 'iterPerSec', {
      label: 'Iter/s',
      format: (v) => v.toFixed(1),
    });
    trainingFolder.addMonitor(trainingParams, 'gaussianCount', {
      label: 'Gaussians',
      format: (v) => v.toLocaleString(),
    });

    trainingFolder.addSeparator();

    // Iteration control
    trainingFolder.addInput(trainingParams, 'iterations', {
      label: 'Iterations',
      min: 1,
      max: 50000,
      step: 100,
    });

    // Training buttons
    trainingFolder.addButton({ title: 'Start Training' }).on('click', async () => {
      if (!gaussian_renderer) return;
      startStatusUpdates();
      await gaussian_renderer.train(trainingParams.iterations);
      updateTrainingStatus();
    });

    trainingFolder.addButton({ title: 'Stop Training' }).on('click', () => {
      if (gaussian_renderer) {
        gaussian_renderer.stopTraining();
      }
    });

    trainingFolder.addSeparator();

    // Learning rate controls
    const lrFolder = trainingFolder.addFolder({ title: 'Learning Rates', expanded: false });

    lrFolder.addInput(trainingParams, 'lr_sh', {
      label: 'SH (color)',
      min: 0.001,
      max: 0.1,
      step: 0.001,
    }).on('change', (e) => {
      if (gaussian_renderer) {
        gaussian_renderer.setTrainingParams({ lr_sh: e.value });
      }
    });

    lrFolder.addInput(trainingParams, 'lr_opacity', {
      label: 'Opacity',
      min: 0.005,
      max: 0.5,
      step: 0.005,
    }).on('change', (e) => {
      if (gaussian_renderer) {
        gaussian_renderer.setTrainingParams({ lr_opacity: e.value });
      }
    });

    lrFolder.addInput(trainingParams, 'lr_scale', {
      label: 'Scale',
      min: 0.001,
      max: 0.1,
      step: 0.001,
    }).on('change', (e) => {
      if (gaussian_renderer) {
        gaussian_renderer.setTrainingParams({ lr_scale: e.value });
      }
    });

    lrFolder.addInput(trainingParams, 'lr_rotation', {
      label: 'Rotation',
      min: 0.0005,
      max: 0.05,
      step: 0.0005,
    }).on('change', (e) => {
      if (gaussian_renderer) {
        gaussian_renderer.setTrainingParams({ lr_rotation: e.value });
      }
    });

    lrFolder.addInput(trainingParams, 'lr_means', {
      label: 'Position',
      min: 0.0000,
      max: 0.001,
      step: 0.00001,
    }).on('change', (e) => {
      if (gaussian_renderer) {
        gaussian_renderer.setTrainingParams({ lr_means: e.value });
      }
    });

    lrFolder.addInput(trainingParams, 'sh_update_interval', {
      label: 'Color Freq',
      min: 1,
      max: 100,
      step: 1,
    }).on('change', (e) => {
      if (gaussian_renderer) {
        gaussian_renderer.setTrainingParams({ sh_update_interval: e.value });
      }
    });

    // Loss function controls
    const lossFolder = trainingFolder.addFolder({ title: 'Loss Function', expanded: false });

    lossFolder.addInput(trainingParams, 'loss_l1', {
      label: 'L1 Weight',
      min: 0,
      max: 1.0,
      step: 0.05,
    }).on('change', (e) => {
      if (gaussian_renderer) {
        gaussian_renderer.setLossWeights({ w_l1: e.value });
      }
    });

    lossFolder.addInput(trainingParams, 'loss_l2', {
      label: 'L2 Weight',
      min: 0,
      max: 1.0,
      step: 0.05,
    }).on('change', (e) => {
      if (gaussian_renderer) {
        gaussian_renderer.setLossWeights({ w_l2: e.value });
      }
    });

    lossFolder.addInput(trainingParams, 'loss_ssim', {
      label: 'SSIM Weight',
      min: 0,
      max: 1.0,
      step: 0.05,
    }).on('change', (e) => {
      if (gaussian_renderer) {
        gaussian_renderer.setLossWeights({ w_ssim: e.value });
      }
    });

    // Pruning controls
    const pruneFolder = trainingFolder.addFolder({ title: 'Pruning', expanded: false });

    pruneFolder.addInput(trainingParams, 'pruneFromIter', {
      label: 'Start After',
      min: 0,
      max: 5000,
      step: 100,
    }).on('change', (e) => {
      if (gaussian_renderer) {
        const controller = gaussian_renderer.getTrainingController?.();
        if (controller) {
          const config = controller.getConfig();
          config.pruneFromIter = e.value;
        }
      }
    });

    pruneFolder.addInput(trainingParams, 'pruneInterval', {
      label: 'Interval',
      min: 10,
      max: 500,
      step: 10,
    }).on('change', (e) => {
      if (gaussian_renderer) {
        const controller = gaussian_renderer.getTrainingController?.();
        if (controller) {
          const config = controller.getConfig();
          config.pruneInterval = e.value;
        }
      }
    });

    pruneFolder.addInput(trainingParams, 'maxScale', {
      label: 'Max Scale',
      min: 0.1,
      max: 10.0,
      step: 0.1,
    }).on('change', (e) => {
      if (gaussian_renderer) {
        const controller = gaussian_renderer.getTrainingController?.();
        if (controller) {
          const config = controller.getConfig();
          config.maxScaleAbsolute = e.value;
        }
      }
    });

    pruneFolder.addInput(trainingParams, 'minOpacity', {
      label: 'Min Opacity',
      min: 0.001,
      max: 0.1,
      step: 0.001,
    }).on('change', (e) => {
      if (gaussian_renderer) {
        const controller = gaussian_renderer.getTrainingController?.();
        if (controller) {
          const config = controller.getConfig();
          config.minOpacity = e.value;
        }
      }
    });

    pruneFolder.addButton({ title: 'Prune Now' }).on('click', async () => {
      if (gaussian_renderer) {
        await gaussian_renderer.prune();
        updateTrainingStatus();
      }
    });

    pruneFolder.addSeparator();

    // Opacity reset controls
    pruneFolder.addInput(trainingParams, 'opacityResetInterval', {
      label: 'Reset Interval',
      min: 500,
      max: 10000,
      step: 500,
    }).on('change', (e) => {
      if (gaussian_renderer) {
        const controller = gaussian_renderer.getTrainingController?.();
        if (controller) {
          const config = controller.getConfig();
          config.opacityResetInterval = e.value;
        }
      }
    });

    pruneFolder.addInput(trainingParams, 'opacityResetValue', {
      label: 'Reset Max α',
      min: 0.01,
      max: 0.8,
      step: 0.01,
    }).on('change', (e) => {
      if (gaussian_renderer) {
        const controller = gaussian_renderer.getTrainingController?.();
        if (controller) {
          const config = controller.getConfig();
          config.opacityResetValue = e.value;
        }
      }
    });

    pruneFolder.addButton({ title: 'Reset Opacity Now' }).on('click', async () => {
      if (gaussian_renderer) {
        await gaussian_renderer.resetOpacity(trainingParams.opacityResetValue);
        updateTrainingStatus();
      }
    });

    trainingFolder.addSeparator();

    // Camera control
    trainingFolder.addButton({ title: 'Cycle Training Camera' }).on('click', () => {
      if (training_cameras_cache.length === 0) {
        console.warn('No training cameras loaded. Please load a transforms_train.json file first.');
        return;
      }
      const index = currentTrainingCameraIndex % training_cameras_cache.length;
      const trainingCam = training_cameras_cache[index];
      camera.set_from_training_camera(trainingCam);
      console.log(
        `Set viewer camera to training camera ${index + 1}/${training_cameras_cache.length} (${trainingCam.file_path})`,
      );
      currentTrainingCameraIndex = (currentTrainingCameraIndex + 1) % training_cameras_cache.length;
      verifyTrainingCameraFrustumCPU(trainingCam, currentPointCloud)
        .catch(err => console.error('Frustum verification error:', err));
    });
  }

  // Sync training params from renderer when it's loaded
  function syncTrainingParamsFromRenderer() {
    if (!gaussian_renderer) return;

    const params = gaussian_renderer.getTrainingParams();
    trainingParams.lr_sh = params.lr_sh;
    trainingParams.lr_opacity = params.lr_opacity;
    trainingParams.lr_scale = params.lr_scale;
    trainingParams.lr_rotation = params.lr_rotation;
    trainingParams.lr_means = params.lr_means;
    trainingParams.sh_update_interval = params.sh_update_interval;

    const controller = gaussian_renderer.getTrainingController?.();
    if (controller) {
      const config = controller.getConfig();
      trainingParams.pruneFromIter = config.pruneFromIter;
      trainingParams.pruneInterval = config.pruneInterval;
      trainingParams.maxScale = config.maxScaleAbsolute;
      trainingParams.minOpacity = config.minOpacity;
      trainingParams.opacityResetInterval = config.opacityResetInterval;
      trainingParams.opacityResetValue = config.opacityResetValue;
    }

    // Sync loss weights
    const lossWeights = gaussian_renderer.getLossWeights();
    trainingParams.loss_l1 = lossWeights.w_l1;
    trainingParams.loss_l2 = lossWeights.w_l2;
    trainingParams.loss_ssim = lossWeights.w_ssim;

    pane.refresh();
  }
  {
    const f = pane.addFolder({ title: 'Benchmark' });
    f.addButton({ title: 'Run Benchmark' }).on('click', () => {
      runBenchmark();
    });
  }

  function runBenchmark() {
    isBenchmarking = true;
    benchmarkSum = 0;
    benchmarkFrames = 0;
  }

  async function verifyTrainingCameraFrustumCPU(
    trainingCam: TrainingCameraData,
    pc: PointCloud | undefined,
  ): Promise<void> {
    if (!pc) {
      console.warn('[frustum verify] No point cloud loaded.');
      return;
    }

    const gaussBuffer = pc.gaussian_3d_buffer;
    const numPoints = pc.num_points;
    if (numPoints === 0) {
      console.warn('[frustum verify] Point cloud is empty.');
      return;
    }

    const bytes = gaussBuffer.size;
    const staging = device.createBuffer({
      label: 'frustum verify staging',
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(gaussBuffer, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();

    const strideHalf = (24) / 2;
    const f16 = new (window as any).Float16Array(copy) as Float32Array;

    const view = trainingCam.view as Float32Array;
    const w = trainingCam.viewport[0];
    const h = trainingCam.viewport[1];
    const fx = trainingCam.focal[0];
    const fy = trainingCam.focal[1];
    const cx = trainingCam.center_px ? trainingCam.center_px[0] : w * 0.5;
    const cy = trainingCam.center_px ? trainingCam.center_px[1] : h * 0.5;

    const sampleStep = Math.max(1, Math.floor(numPoints / 50000));
    let samples = 0;
    let inFront = 0;
    let inImage = 0;

    for (let i = 0; i < numPoints; i += sampleStep) {
      const base = i * strideHalf;
      const x = f16[base + 0];
      const y = f16[base + 1];
      const z = f16[base + 2];

      const Xc = view[0] * x + view[4] * y + view[8] * z + view[12];
      const Yc = view[1] * x + view[5] * y + view[9] * z + view[13];
      const Zc = view[2] * x + view[6] * y + view[10] * z + view[14];

      samples++;
      if (Zc <= 0) {
        continue;
      }
      inFront++;

      const x_img = fx * Xc / Zc + cx;
      const y_img = fy * Yc / Zc + cy;
      const inside = x_img >= 0 && x_img < w && y_img >= 0 && y_img < h;
      if (inside) inImage++;
    }

    const fracFront = samples > 0 ? inFront / samples : 0;
    const fracImage = samples > 0 ? inImage / samples : 0;
    console.log(
      `[frustum verify] trainingCam ${trainingCam.file_path} samples=${samples} ` +
      `in_front=${inFront} (frac=${fracFront.toFixed(3)}), ` +
      `in_image=${inImage} (frac=${fracImage.toFixed(3)})`,
    );
  }

  function frame() {
    if (dataset_loaded) {
      params.fps = 1.0 / timeReturn() * 1000.0;
      time();
      const encoder = device.createCommandEncoder();
      const texture_view = context.getCurrentTexture().createView();
      renderer.frame(encoder, texture_view);
      device.queue.submit([encoder.finish()]);

      if (isBenchmarking) {
        benchmarkSum += params.fps;
        benchmarkFrames++;
        if (benchmarkFrames >= params.benchFrames) {
          isBenchmarking = false;
          console.log(`Average FPS: ${benchmarkSum / params.benchFrames}`);
        }
      }
    }
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}
