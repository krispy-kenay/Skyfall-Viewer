import { mat4, vec2 } from 'wgpu-matrix';
import type { TrainingCameraData } from '../camera/camera';
import { get_pinhole_projection_matrix, principal_point_to_ndc } from '../camera/camera';
import type { PointCloud } from './load';
import { createPointCloudFromXYZRGB, XYZRGB } from './pointcloud-loader';

export interface SatelliteDataset {
  pointCloud: PointCloud;
  trainingCameras: TrainingCameraData[];
  trainingImages: Array<{ view: GPUTextureView; sampler: GPUSampler }>;
}

async function readTextFromDir(
  dirHandle: any,
  relativePath: string
): Promise<string> {
  const parts = relativePath.replace(/^[./]+/, '').split('/');
  let dir = dirHandle;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i];
    // @ts-ignore
    dir = await dir.getDirectoryHandle(part);
  }
  const fileName = parts[parts.length - 1];
  // @ts-ignore
  const fileHandle = await dir.getFileHandle(fileName);
  const file = await fileHandle.getFile();
  return await file.text();
}

async function loadImageToTextureFromDir(
  dirHandle: any,
  relativePath: string,
  device: GPUDevice
): Promise<{ view: GPUTextureView; sampler: GPUSampler }> {
  const parts = relativePath.replace(/^[./]+/, '').split('/');
  let dir = dirHandle;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i];
    // @ts-ignore
    dir = await dir.getDirectoryHandle(part);
  }
  const fileName = parts[parts.length - 1];
  // @ts-ignore
  const fileHandle = await dir.getFileHandle(fileName);
  const file = await fileHandle.getFile();

  const imageBitmap = await createImageBitmap(file);

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
    view: texture.createView(),
    sampler: device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    }),
  };
}
// Parse COLMAP style points file
function parsePoints3D(text: string): XYZRGB {
  const lines = text.split(/\r?\n/);
  const xyz: number[] = [];
  const rgb: number[] = [];

  for (const line of lines) {
    if (!line || line.startsWith('#')) continue;
    const parts = line.trim().split(/\s+/);
    if (parts.length < 8) continue;
    const x = parseFloat(parts[1]);
    const y = parseFloat(parts[2]);
    const z = parseFloat(parts[3]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

    const r = parseInt(parts[4], 10);
    const g = parseInt(parts[5], 10);
    const b = parseInt(parts[6], 10);

    xyz.push(x, y, z);
    rgb.push(
      Math.max(0, Math.min(255, r)),
      Math.max(0, Math.min(255, g)),
      Math.max(0, Math.min(255, b))
    );
  }

  return {
    xyz: new Float32Array(xyz),
    rgb: new Uint8Array(rgb),
  };
}

function createPointCloudFromXYZRGB(
  xyz: Float32Array,
  rgb: Uint8Array,
  device: GPUDevice
): PointCloud {
  const num_points = xyz.length / 3;
  const max_num_coefs = 16;

  const c_size_float = 2;
  const c_size_3d_gaussian =
    3 * c_size_float +
    c_size_float +
    4 * c_size_float +
    4 * c_size_float;

  const c_size_sh_coef =
    3 * max_num_coefs * c_size_float;

  const gaussian_3d_buffer = device.createBuffer({
    label: 'satellite 3d gaussians',
    size: num_points * c_size_3d_gaussian,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const gaussian = new Float16Array(gaussian_3d_buffer.getMappedRange());

  const sh_buffer = device.createBuffer({
    label: 'satellite sh coeffs',
    size: num_points * c_size_sh_coef,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const sh = new Float16Array(sh_buffer.getMappedRange());

  const defaultOpacity = 1.0;
  const defaultScale = 0.01;
  const defaultRot = [0, 0, 0, 1];

  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;

  for (let i = 0; i < num_points; i++) {
    const px = xyz[i * 3 + 0];
    const py = xyz[i * 3 + 1];
    const pz = xyz[i * 3 + 2];

    const o = i * (c_size_3d_gaussian / c_size_float);
    const output_offset = i * max_num_coefs * 3;

    const r = rgb[i * 3 + 0] / 255.0;
    const g = rgb[i * 3 + 1] / 255.0;
    const b = rgb[i * 3 + 2] / 255.0;
    sh[output_offset + 0] = r;
    sh[output_offset + 1] = g;
    sh[output_offset + 2] = b;
    for (let k = 3; k < max_num_coefs * 3; ++k) {
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
  };
}

function applyRTFixIfPresent(
  xyz: Float32Array,
  contents: any
): Float32Array {
  if (!('R' in contents) || !('T' in contents)) {
    return xyz;
  }
  const Rfix: number[][] = contents['R'];
  const Tfix: number[] = contents['T'];
  if (!Array.isArray(Rfix) || !Array.isArray(Tfix)) return xyz;

  const out = new Float32Array(xyz.length);
  for (let i = 0; i < xyz.length / 3; i++) {
    const x = xyz[i * 3 + 0];
    const y = xyz[i * 3 + 1];
    const z = xyz[i * 3 + 2];

    const rx =
      Rfix[0][0] * x + Rfix[0][1] * y + Rfix[0][2] * z - Tfix[0];
    const ry =
      Rfix[1][0] * x + Rfix[1][1] * y + Rfix[1][2] * z - Tfix[1];
    const rz =
      Rfix[2][0] * x + Rfix[2][1] * y + Rfix[2][2] * z - Tfix[2];

    out[i * 3 + 0] = rx;
    out[i * 3 + 1] = ry;
    out[i * 3 + 2] = rz;
  }
  return out;
}

function normalizeXYZ(xyz: Float32Array): {
  xyzNorm: Float32Array;
  scale: number;
  zMin: number;
} {
  const num = xyz.length / 3;
  const norms: number[] = new Array(num);
  for (let i = 0; i < num; i++) {
    const x = xyz[i * 3 + 0];
    const y = xyz[i * 3 + 1];
    const z = xyz[i * 3 + 2];
    norms[i] = Math.hypot(x, y, z);
  }
  const sorted = norms.slice().sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.floor(0.99 * (sorted.length - 1)));
  const radius = Math.max(1e-3, sorted[idx]);
  const scale = 256 / radius;

  const xyzScaled = new Float32Array(xyz.length);
  for (let i = 0; i < num; i++) {
    xyzScaled[i * 3 + 0] = xyz[i * 3 + 0] * scale;
    xyzScaled[i * 3 + 1] = xyz[i * 3 + 1] * scale;
    xyzScaled[i * 3 + 2] = xyz[i * 3 + 2] * scale;
  }

  const zs: number[] = new Array(num);
  for (let i = 0; i < num; i++) {
    zs[i] = xyzScaled[i * 3 + 2];
  }
  const zsSorted = zs.slice().sort((a, b) => a - b);
  const idxZ = Math.min(zsSorted.length - 1, Math.floor(0.01 * (zsSorted.length - 1)));
  const zMin = zsSorted[idxZ];

  const xyzNorm = new Float32Array(xyz.length);
  for (let i = 0; i < num; i++) {
    xyzNorm[i * 3 + 0] = xyzScaled[i * 3 + 0];
    xyzNorm[i * 3 + 1] = xyzScaled[i * 3 + 1];
    xyzNorm[i * 3 + 2] = xyzScaled[i * 3 + 2] - zMin;
  }

  return { xyzNorm, scale, zMin };
}

export async function loadSatelliteDataset(
  dirHandle: any,
  device: GPUDevice
): Promise<SatelliteDataset> {
  const transformsText = await readTextFromDir(dirHandle, 'transforms_train.json');
  const transforms = JSON.parse(transformsText);
  const frames: any[] = transforms.frames || [];

  const ptsText = await readTextFromDir(dirHandle, 'points3D.txt');
  const parsed = parsePoints3D(ptsText);

  const xyzRT = applyRTFixIfPresent(parsed.xyz, transforms);

  const { xyzNorm, scale, zMin } = normalizeXYZ(xyzRT);

  console.log(
    `Satellite normalization: num_points=${xyzNorm.length / 3}, ` +
    `scale=${scale.toFixed(4)}, zMin=${zMin.toFixed(4)}`
  );

  const pointCloud = createPointCloudFromXYZRGB(xyzNorm, parsed.rgb, device);

  const w: number = transforms.w;
  const h: number = transforms.h;

  const znear = 0.01;
  const zfar = 4_000_000;

  const cameras: TrainingCameraData[] = [];
  const images: Array<{ view: GPUTextureView; sampler: GPUSampler }> = [];

  const hasFix = 'R' in transforms && 'T' in transforms;
  const c2wKey = hasFix ? 'transform_matrix_rotated' : 'transform_matrix';

  for (const frame of frames) {
    const fl_x: number = frame.fl_x;
    const fl_y: number = frame.fl_y;
    const cx: number   = frame.cx;
    const cy: number   = frame.cy;

    const tm: number[][] = frame[c2wKey] || frame.transform_matrix;

    const c2w = mat4.create(
      tm[0][0], tm[1][0], tm[2][0], tm[3][0],
      tm[0][1], tm[1][1], tm[2][1], tm[3][1],
      tm[0][2], tm[1][2], tm[2][2], tm[3][2],
      tm[0][3], tm[1][3], tm[2][3], tm[3][3],
    );

    c2w[12] *= scale;
    c2w[13] *= scale;
    c2w[14] *= scale;
    c2w[14] -= zMin;

    const view = mat4.invert(c2w);
    const view_inv = mat4.copy(c2w);

    const proj = get_pinhole_projection_matrix(znear, zfar, fl_x, fl_y, cx, cy, w, h);
    const proj_inv = mat4.inverse(proj);

    const focal = vec2.create(fl_x, fl_y);
    const [cx_ndc, cy_ndc] = principal_point_to_ndc(cx, cy, w, h);
    const center = vec2.create(cx_ndc, cy_ndc);
    const center_px = vec2.create(cx, cy);
    const viewport = vec2.create(w, h);
    const imgSize = vec2.create(w, h);

    cameras.push({
      view,
      view_inv,
      proj,
      proj_inv,
      viewport,
      focal,
      center,
      center_px,
      imgSize,
      file_path: frame.file_path || '',
    });

    const imgPath: string = frame.file_path;
    const tex = await loadImageToTextureFromDir(dirHandle, imgPath, device);
    images.push(tex);
  }

  return {
    pointCloud,
    trainingCameras: cameras,
    trainingImages: images,
  };
}


