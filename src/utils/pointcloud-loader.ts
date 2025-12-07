import { Float16Array } from '@petamoriken/float16';
import type { PointCloud } from './load';

// Shared helper for both pointcloud and gaussian splat format .ply files

const C_SIZE_F16 = 2;
const MAX_NUM_COEFFS = 16;

const SH_C0 = 0.28209479177387814;

function RGB2SH(rgb: number): number {
  return (rgb - 0.5) / SH_C0;
}

export const C_SIZE_3D_GAUSSIAN =
  3 * C_SIZE_F16 + 
  C_SIZE_F16 +
  4 * C_SIZE_F16 +
  4 * C_SIZE_F16;

export const C_SIZE_SH_COEF =
  3 * MAX_NUM_COEFFS * C_SIZE_F16;

export interface XYZRGB {
  xyz: Float32Array;
  rgb: Uint8Array;
}

export function createPointCloudFromXYZRGB(
  { xyz, rgb }: XYZRGB,
  device: GPUDevice
): PointCloud {
  const num_points = xyz.length / 3;

  // gaussian_3d_buffer
  const gaussian_3d_buffer = device.createBuffer({
    label: '3d gaussians',
    size: num_points * C_SIZE_3D_GAUSSIAN,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const gaussian = new Float16Array(gaussian_3d_buffer.getMappedRange());

  // SH coefficients buffer
  const sh_buffer = device.createBuffer({
    label: 'sh coefficients',
    size: num_points * C_SIZE_SH_COEF,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const sh = new Float16Array(sh_buffer.getMappedRange());

  const defaultOpacity = -2.197;
  const defaultSigma = 0.01;
  const defaultScale = Math.log(defaultSigma);
  const defaultRot = [0, 0, 0, 1];

  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;

  for (let i = 0; i < num_points; i++) {
    const px = xyz[i * 3 + 0];
    const py = xyz[i * 3 + 1];
    const pz = xyz[i * 3 + 2];

    const o = i * (C_SIZE_3D_GAUSSIAN / C_SIZE_F16);
    const output_offset = i * MAX_NUM_COEFFS * 3;

    const r = rgb[i * 3 + 0] / 255.0;
    const g = rgb[i * 3 + 1] / 255.0;
    const b = rgb[i * 3 + 2] / 255.0;
    sh[output_offset + 0] = RGB2SH(r);
    sh[output_offset + 1] = RGB2SH(g);
    sh[output_offset + 2] = RGB2SH(b);
    for (let k = 3; k < MAX_NUM_COEFFS * 3; ++k) {
      sh[output_offset + k] = 0.0;
    }

    gaussian[o + 0] = px;
    gaussian[o + 1] = py;
    gaussian[o + 2] = pz;

    minX = Math.min(minX, px);
    maxX = Math.max(maxX, px);
    minY = Math.min(minY, py);
    maxY = Math.max(maxY, py);
    minZ = Math.min(minZ, pz);
    maxZ = Math.max(maxZ, pz);

    gaussian[o + 3] = defaultOpacity;
    gaussian[o + 4] = defaultRot[0];
    gaussian[o + 5] = defaultRot[1];
    gaussian[o + 6] = defaultRot[2];
    gaussian[o + 7] = defaultRot[3];
    gaussian[o + 8] = defaultScale;
    gaussian[o + 9] = defaultScale;
    gaussian[o + 10] = defaultScale;
  }

  gaussian_3d_buffer.unmap();
  sh_buffer.unmap();

  const extentX = maxX - minX;
  const extentY = maxY - minY;
  const extentZ = maxZ - minZ;
  const scene_extent = Math.max(extentX, extentY, extentZ);

  const centerX = (minX + maxX) / 2.0;
  const centerY = (minY + maxY) / 2.0;
  const centerZ = (minZ + maxZ) / 2.0;

  return {
    num_points,
    sh_deg: 0,
    gaussian_3d_buffer,
    sh_buffer,
    scene_extent,
    bounding_box: {
      min: { x: minX, y: minY, z: minZ },
      max: { x: maxX, y: maxY, z: maxZ },
      center: { x: centerX, y: centerY, z: centerZ },
      extent: { x: extentX, y: extentY, z: extentZ },
    },
  } as PointCloud;
}


