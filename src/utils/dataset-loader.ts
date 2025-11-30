import type { PointCloud } from './load';
import { load as loadPly } from './load';
import type { TrainingCameraData } from '../camera/camera';
import { load_all_training_cameras_from_transforms } from '../camera/camera';
import { loadColmapDatasetScene as loadColmapSceneInternal } from './colmap-dataset';

export interface TrainingImageResource {
  view: GPUTextureView;
  sampler: GPUSampler;
}

export interface SceneDataset {
  pointCloud: PointCloud;
  trainingCameras: TrainingCameraData[];
  trainingImages: TrainingImageResource[];
}

export interface BasicPlyInputs {
  plyFile: string;
  cameraFile?: string;
  imageFile?: File | null;
}

export async function loadBasicPlyDataset(
  inputs: BasicPlyInputs,
  device: GPUDevice
): Promise<SceneDataset> {
  const pointCloud = await loadPly(inputs.plyFile, device);

  let trainingCameras: TrainingCameraData[] = [];
  if (inputs.cameraFile) {
    try {
      trainingCameras = await load_all_training_cameras_from_transforms(inputs.cameraFile);
    } catch {
      trainingCameras = [];
    }
  }

  const trainingImages: TrainingImageResource[] = [];
  if (inputs.imageFile) {
    const imageBitmap = await createImageBitmap(inputs.imageFile);
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
    trainingImages.push({
      view: texture.createView(),
      sampler: device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
      }),
    });
  }

  return {
    pointCloud,
    trainingCameras,
    trainingImages,
  };
}

export async function loadSatelliteDatasetScene(
  dirHandle: any,
  device: GPUDevice
): Promise<SceneDataset> {
  console.warn(
    'loadSatelliteDatasetScene is deprecated. Please use loadColmapDatasetScene with a COLMAP-style dataset folder.',
  );
  const dataset = await loadColmapSceneInternal(dirHandle, device);
  return dataset;
}

export async function loadColmapDatasetScene(
  dirHandle: any,
  device: GPUDevice
): Promise<SceneDataset> {
  return await loadColmapSceneInternal(dirHandle, device);
}


