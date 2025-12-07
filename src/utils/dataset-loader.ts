import type { PointCloud } from './load';
import { load as loadPly } from './load';
import { loadColmapDatasetScene as loadColmapSceneInternal } from './colmap-dataset';

export interface TrainingImageResource {
  view: GPUTextureView;
  sampler: GPUSampler;
  width: number;
  height: number;
}

export interface SceneDataset {
  pointCloud: PointCloud;
  trainingCameras: import('../camera/camera').TrainingCameraData[];
  trainingImages: TrainingImageResource[];
}


export async function loadPlyFile(
  plyFile: string,
  device: GPUDevice
): Promise<PointCloud> {
  return await loadPly(plyFile, device);
}

export async function loadColmapDatasetScene(
  dirHandle: any,
  device: GPUDevice
): Promise<SceneDataset> {
  return await loadColmapSceneInternal(dirHandle, device);
}
