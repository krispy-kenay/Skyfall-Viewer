import type { PointCloud } from './load';
import { load as loadPly } from './load';
import { loadColmapDatasetScene as loadColmapSceneInternal } from './colmap-dataset';

export interface TrainingImageResource {
  view: GPUTextureView;
  sampler: GPUSampler;
}

export interface SceneDataset {
  pointCloud: PointCloud;
  trainingCameras: import('../camera/camera').TrainingCameraData[];
  trainingImages: TrainingImageResource[];
}

/**
 * Load a full gaussian splat .ply file.
 * Supports both standard 3DGS PLY format and simple xyz+rgb point clouds.
 */
export async function loadPlyFile(
  plyFile: string,
  device: GPUDevice
): Promise<PointCloud> {
  return await loadPly(plyFile, device);
}

/**
 * Load a COLMAP-style dataset folder with sparse reconstruction.
 * Expected structure:
 *   /sparse/0/cameras.bin
 *   /sparse/0/images.bin
 *   /sparse/0/points3D.bin
 *   /images/...
 */
export async function loadColmapDatasetScene(
  dirHandle: any,
  device: GPUDevice
): Promise<SceneDataset> {
  return await loadColmapSceneInternal(dirHandle, device);
}
