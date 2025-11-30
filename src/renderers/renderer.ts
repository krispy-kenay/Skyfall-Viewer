import type { PointCloud } from '../utils/load';
import { Pane } from 'tweakpane';
import * as TweakpaneFileImportPlugin from 'tweakpane-plugin-file-import';
import { default as get_renderer_gaussian, GaussianRenderer } from './gaussian-renderer';
import { default as get_renderer_pointcloud } from './point-cloud-renderer';
import { Camera, load_camera_presets, load_satellite_camera, TrainingCameraData, load_all_training_cameras_from_transforms, CameraPreset} from '../camera/camera';
import { CameraControl } from '../camera/camera-control';
import { time, timeReturn } from '../utils/simple-console';
import { loadColmapDatasetScene } from '../utils/dataset-loader';
import { vec3, mat4 } from 'wgpu-matrix';

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
  let cameras;
  let satellite;
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
  
  async function loadImageToTexture(device: GPUDevice, source: File) {
    let blob: Blob;
    blob = source;
    const imageBitmap = await createImageBitmap(blob);
  
    const texture = device.createTexture({
      size: [imageBitmap.width, imageBitmap.height, 1],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
  
    device.queue.copyExternalImageToTexture(
      { source: imageBitmap },
      { texture },
      [imageBitmap.width, imageBitmap.height]
    );
  
    return {
      texture,
      view: texture.createView(),
      sampler: device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
      })
    };
  }
  

  // Tweakpane: easily adding tweak control for parameters.
  const params = {
    fps: 0.0,
    gaussian_multiplier: 1,
    renderer: 'pointcloud',
    ply_file: '',
    cam_file: '',
    img_file: '',
    benchFrames: 300,
  };

  const pane = new Pane({
    title: 'Config',
    expanded: true,
  });
  pane.registerPlugin(TweakpaneFileImportPlugin);
  {
    pane.addMonitor(params, 'fps', {
      readonly:true
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

          const initEncoder = device.createCommandEncoder();
          gaussian_renderer.initializeKnn(initEncoder);
          device.queue.submit([initEncoder.finish()]);
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
      {min: 0, max: 1.5}
    ).on('change', (e) => {
      //TODO: Bind constants to the gaussian renderer.
      clearTimeout(scaleDebounce);
      scaleDebounce = setTimeout(() => gaussian_renderer.setGaussianScale(e.value), 25);
    });
    let scaleDebounce: any = null;
  }
  {
    const f = pane.addFolder({ title: 'Training' });
    f.addButton({ title: 'Train' }).on('click', () => {
      if (gaussian_renderer) gaussian_renderer.train(1);
    });
    f.addButton({ title: 'Train x50' }).on('click', () => {
      if (gaussian_renderer) gaussian_renderer.train(50);
    });
    f.addButton({ title: 'Set Viewer to Random Training Camera' }).on('click', () => {
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
  {
    const f = pane.addFolder({ title: 'Benchmark' });
    f.addButton({ title: 'Run Benchmark' }).on('click', () => {
      runBenchmark();
    });
  }
  

  document.addEventListener('keydown', (event) => {
    switch(event.key) {
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
        const i = parseInt(event.key);
        console.log(`set to camera preset ${i}`);
        camera.set_preset(cameras[i]);
        break;
    }
  });

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
  const proj = trainingCam.proj as Float32Array;
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

      const Xc =
        view[0] * x + view[4] * y + view[8] * z + view[12];
      const Yc =
        view[1] * x + view[5] * y + view[9] * z + view[13];
      const Zc =
        view[2] * x + view[6] * y + view[10] * z + view[14];
      const Wc =
        view[3] * x + view[7] * y + view[11] * z + view[15];

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
      params.fps=1.0/timeReturn()*1000.0;
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
