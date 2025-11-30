import type { PointCloud } from '../utils/load';
import { Pane } from 'tweakpane';
import * as TweakpaneFileImportPlugin from 'tweakpane-plugin-file-import';
import { default as get_renderer_gaussian, GaussianRenderer } from './gaussian-renderer';
import { default as get_renderer_pointcloud } from './point-cloud-renderer';
import { Camera, load_camera_presets, load_satellite_camera, TrainingCameraData, load_all_training_cameras_from_transforms, CameraPreset} from '../camera/camera';
import { CameraControl } from '../camera/camera-control';
import { time, timeReturn } from '../utils/simple-console';
import { loadBasicPlyDataset, loadSatelliteDatasetScene, loadColmapDatasetScene } from '../utils/dataset-loader';
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
  let ply_file_loaded = false; 
  let cam_file_loaded = true;
  let img_file_loaded = false;
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
    const f = pane.addFolder({ title: 'Satellite Dataset' });
    f.addButton({ title: 'Load Dataset Folder' }).on('click', async () => {
      // @ts-ignore - File System Access API
      if (!window.showDirectoryPicker) {
        alert('Your browser does not support folder selection. Please use a Chromium-based browser (Chrome, Edge, etc.).');
        return;
      }

      try {
        // @ts-ignore
        const dirHandle: FileSystemDirectoryHandle = await window.showDirectoryPicker();

        const dataset = await loadSatelliteDatasetScene(dirHandle, device);

        currentPointCloud = dataset.pointCloud;

        pointcloud_renderer = get_renderer_pointcloud(dataset.pointCloud, device, presentation_format, camera.uniform_buffer);
        gaussian_renderer = get_renderer_gaussian(dataset.pointCloud, device, presentation_format, camera.uniform_buffer, canvas, camera);
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

        ply_file_loaded = true;
        cam_file_loaded = dataset.trainingCameras.length > 0;
        img_file_loaded = dataset.trainingImages.length > 0;
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          console.error('Failed to load satellite dataset:', err);
          alert(`Failed to load satellite dataset: ${(err as Error).message}`);
        }
      }
    });
  }
  {
    const f = pane.addFolder({ title: 'COLMAP Dataset' });
    f.addButton({ title: 'Load COLMAP Dataset' }).on('click', async () => {
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

        ply_file_loaded = true;
        cam_file_loaded = dataset.trainingCameras.length > 0;
        img_file_loaded = dataset.trainingImages.length > 0;
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          console.error('Failed to load COLMAP dataset:', err);
          alert(`Failed to load COLMAP dataset: ${(err as Error).message}`);
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
    pane.addInput(params, 'ply_file', {
      view: 'file-input',
      lineCount: 3,
      filetypes: ['.ply'],
      invalidFiletypeMessage: "We can't accept those filetypes!"
    })
    .on('change', async (file) => {
      const uploadedFile = file.value as string | null;
      if (uploadedFile) {
        const dataset = await loadBasicPlyDataset(
          { plyFile: uploadedFile },
          device
        );

        currentPointCloud = dataset.pointCloud;

        pointcloud_renderer = get_renderer_pointcloud(dataset.pointCloud, device, presentation_format, camera.uniform_buffer);
        gaussian_renderer = get_renderer_gaussian(dataset.pointCloud, device, presentation_format, camera.uniform_buffer, canvas, camera);
        renderers = {
          pointcloud: pointcloud_renderer,
          gaussian: gaussian_renderer,
        };
        renderer = renderers[params.renderer];

        if (gaussian_renderer && dataset.trainingCameras.length) {
          gaussian_renderer.setTrainingCameras(dataset.trainingCameras);
        }
        if (gaussian_renderer) {
          const initEncoder = device.createCommandEncoder();
          gaussian_renderer.initializeKnn(initEncoder);
          device.queue.submit([initEncoder.finish()]);
        }

        ply_file_loaded = true;
      } else {
        ply_file_loaded = false;
      }
    });
  }
  {
    pane.addInput(params, 'cam_file', {
      view: 'file-input',
      lineCount: 3,
      filetypes: ['.json'],
      invalidFiletypeMessage: "We can't accept those filetypes!"
    })
    .on('change', async (file) => {
      const uploadedFile = file.value;
      if (uploadedFile) {
        try {

          const trainingCameras = await load_all_training_cameras_from_transforms(uploadedFile);
          if (trainingCameras.length > 0) {
            console.log(`Loaded ${trainingCameras.length} training cameras from transforms file`);
            training_cameras_cache = trainingCameras;
            if (gaussian_renderer) {
              gaussian_renderer.setTrainingCameras(trainingCameras);
            } else {
              console.warn('Gaussian renderer not initialized yet. Load a PLY file first.');
            }
            cam_file_loaded = true;
          } else {
            console.warn('No cameras found in transforms file');
            cam_file_loaded = false;
          }
        } catch (error) {
          console.log('Not a transforms file, trying camera preset format...');
          try {
            const cam = await load_camera_presets(uploadedFile);
            if (cam.length > 0) {
              camera.set_preset(cam[0]);
              training_cameras_cache = [];
              if (gaussian_renderer) {
                gaussian_renderer.setTrainingCameras([]);
                gaussian_renderer.setTrainingImages([]);
              }
              cam_file_loaded = true;
            } else {
              cam_file_loaded = false;
            }
          } catch (presetError) {
            console.error('Failed to load cameras:', presetError);
            cam_file_loaded = false;
          }
        }
      } else {
        cam_file_loaded = false;
        training_cameras_cache = [];
        if (gaussian_renderer) {
          gaussian_renderer.setTrainingCameras([]);
          gaussian_renderer.setTrainingImages([]);
        }
      }
    });
  }
  {
    pane.addInput(params, 'img_file', {
      view: 'file-input',
      lineCount: 3,
      filetypes: ['.jpg', '.png'],
      invalidFiletypeMessage: "We can't accept those filetypes!"
    })
    .on('change', async (file) => {
      const uploadedFile = file.value as unknown as File | null;
      if (uploadedFile) {
        const { texture, view, sampler } = await loadImageToTexture(device, uploadedFile);
        if (gaussian_renderer) {
          gaussian_renderer.setTargetTexture(view, sampler);
        }
        img_file_loaded = true;
      }else{
        img_file_loaded = false;
      }
    })
  }
  {
    const f = pane.addFolder({ title: 'Training Images' });
    f.addButton({ title: 'Load Images Folder' }).on('click', async () => {
      if (!gaussian_renderer) {
        alert('Please load a PLY file first to initialize the renderer.');
        return;
      }
      // @ts-ignore - File System Access API
      if (!window.showDirectoryPicker) {
        alert('Your browser does not support folder selection. Please use a Chromium-based browser (Chrome, Edge, etc.).');
        return;
      }

      try {
        // @ts-ignore
        const dirHandle: FileSystemDirectoryHandle = await window.showDirectoryPicker();
        const files = new Map<string, File>();

        const scanDir = async (handle: FileSystemDirectoryHandle, path: string) => {
          // @ts-ignore
          for await (const [name, entry] of handle.entries()) {
            const entryPath = path ? `${path}/${name}` : name;
            if (entry.kind === 'file') {
              const file = await entry.getFile();
              const lowerName = name.toLowerCase();
              const lowerFullPath = entryPath.toLowerCase();
              if (lowerName.endsWith('.jpg') || lowerName.endsWith('.jpeg') || 
                  lowerName.endsWith('.png') || lowerName.endsWith('.webp')) {
                files.set(lowerFullPath, file);
                files.set(lowerName, file);
              }
            } else if (entry.kind === 'directory') {
              await scanDir(entry, entryPath);
            }
          }
        };

        await scanDir(dirHandle, '');
        const normalizePath = (path: string) => {
          return path.replace(/\\/g, '/')
                     .replace(/^\.\//, '')
                     .toLowerCase()
                     .trim();
        };
        if (training_cameras_cache.length === 0) {
          alert('Please load training cameras first (transforms_train.json) before loading images.');
          return;
        }
        
        console.log('Matching images to cameras by file_path...');
        console.log(`Found ${files.size} image files in folder`);
        console.log(`Matching to ${training_cameras_cache.length} cameras`);
        
        const imageResources: Array<{ view: GPUTextureView; sampler: GPUSampler }> = [];
        let matched = 0;
        let unmatched = 0;

        for (let i = 0; i < training_cameras_cache.length; i++) {
          const camera = training_cameras_cache[i];
          const cameraFilePath = camera.file_path;
          
          if (!cameraFilePath) {
            console.warn(`Camera ${i + 1} has no file_path, skipping image match`);
            unmatched++;
            continue;
          }
          const normalizedCameraPath = normalizePath(cameraFilePath);
          const cameraFileName = normalizedCameraPath.split('/').pop() || '';
          let imageFile: File | undefined = undefined;
          imageFile = files.get(normalizedCameraPath);
          if (!imageFile && cameraFileName) {
            imageFile = files.get(cameraFileName);
          }
          if (!imageFile && cameraFileName) {
            const baseName = cameraFileName.replace(/\.(jpg|jpeg|png|webp)$/i, '');
            for (const [path, file] of files.entries()) {
              const pathBaseName = path.replace(/\.(jpg|jpeg|png|webp)$/i, '');
              if (pathBaseName === baseName || path.endsWith(baseName)) {
                imageFile = file;
                break;
              }
            }
          }
          
          if (imageFile) {
            try {
              const { texture, view, sampler } = await loadImageToTexture(device, imageFile);
              imageResources.push({ view, sampler });
              matched++;
              console.log(`Matched camera ${i + 1} (${cameraFilePath}) to image: ${imageFile.name}`);
            } catch (err) {
              console.error(`Failed to load image ${imageFile.name} for camera ${i + 1}:`, err);
              unmatched++;
            }
          } else {
            console.warn(`No matching image found for camera ${i + 1} with path: ${cameraFilePath}`);
            unmatched++;
          }
        }

        if (imageResources.length === 0) {
          console.warn('No images were successfully loaded.');
          img_file_loaded = false;
          return;
        }

        if (gaussian_renderer) {
          gaussian_renderer.setTrainingImages(imageResources);
        }

        img_file_loaded = matched > 0;
        console.log(`Loaded ${matched} training images from folder "${dirHandle.name}".${unmatched > 0 ? ` ${unmatched} images failed to load.` : ''}`);
        
        if (matched > 0) {
          alert(`Successfully loaded ${matched} images. Make sure training cameras are loaded and match the image order.`);
        }
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          console.error('Failed to load image folder:', err);
          alert(`Failed to load images: ${(err as Error).message}`);
        }
        img_file_loaded = false;
      }
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
      const randomIndex = Math.floor(Math.random() * training_cameras_cache.length);
      const trainingCam = training_cameras_cache[randomIndex];

      camera.set_from_training_camera(trainingCam);
      console.log(`Set viewer camera to training camera ${randomIndex + 1}/${training_cameras_cache.length}`);
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

  function frame() {
    if (ply_file_loaded && cam_file_loaded) {
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
