import { Float16Array } from '@petamoriken/float16';
import { mat4, vec2 } from 'wgpu-matrix';
import type { TrainingCameraData } from '../camera/camera';
import { get_pinhole_projection_matrix, principal_point_to_ndc } from '../camera/camera';
import type { PointCloud } from './load';
import type { XYZRGB } from './pointcloud-loader';

export interface ColmapDataset {
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

async function readBinaryFromDir(
  dirHandle: any,
  relativePath: string
): Promise<ArrayBuffer> {
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
  return await file.arrayBuffer();
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

// Parse COLMAP points3D.txt file
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

// Parse COLMAP cameras.txt file
interface CameraParams {
  camera_id: number;
  model: string;
  width: number;
  height: number;
  params: number[];
}

function parseCameras(text: string): Map<number, CameraParams> {
  const cameras = new Map<number, CameraParams>();
  const lines = text.split(/\r?\n/);

  for (const line of lines) {
    if (!line || line.startsWith('#')) continue;
    const parts = line.trim().split(/\s+/);
    if (parts.length < 4) continue;

    const camera_id = parseInt(parts[0], 10);
    const model = parts[1];
    const width = parseInt(parts[2], 10);
    const height = parseInt(parts[3], 10);
    const params: number[] = [];

    for (let i = 4; i < parts.length; i++) {
      params.push(parseFloat(parts[i]));
    }

    cameras.set(camera_id, {
      camera_id,
      model,
      width,
      height,
      params,
    });
  }

  return cameras;
}

// Parse COLMAP images.txt file
interface ImageData {
  image_id: number;
  qw: number;
  qx: number;
  qy: number;
  qz: number;
  tx: number;
  ty: number;
  tz: number;
  camera_id: number;
  name: string;
}

function parseImages(text: string): ImageData[] {
  const images: ImageData[] = [];
  const lines = text.split(/\r?\n/);
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    if (!line || line.startsWith('#')) {
      i++;
      continue;
    }

    const parts = line.trim().split(/\s+/);
    if (parts.length < 10) {
      i++;
      continue;
    }

    const image_id = parseInt(parts[0], 10);
    const qw = parseFloat(parts[1]);
    const qx = parseFloat(parts[2]);
    const qy = parseFloat(parts[3]);
    const qz = parseFloat(parts[4]);
    const tx = parseFloat(parts[5]);
    const ty = parseFloat(parts[6]);
    const tz = parseFloat(parts[7]);
    const camera_id = parseInt(parts[8], 10);
    const name = parts.slice(9).join(' ');

    images.push({
      image_id,
      qw,
      qx,
      qy,
      qz,
      tx,
      ty,
      tz,
      camera_id,
      name,
    });

    i += 2;
  }

  return images;
}

// Convert quaternion to rotation matrix
function quaternionToRotationMatrix(qw: number, qx: number, qy: number, qz: number): number[] {
  // Normalize quaternion
  const len = Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
  if (len < 1e-8) {
    throw new Error('Zero-length quaternion');
  }

  const w = qw / len;
  const x = qx / len;
  const y = qy / len;
  const z = qz / len;

  // Build rotation matrix 
  return [
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
    2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
    2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)
  ];
}

// Convert COLMAP camera-to-world pose to matrix
function colmapPoseToC2W(qw: number, qx: number, qy: number, qz: number, tx: number, ty: number, tz: number): Float32Array {
  const R = quaternionToRotationMatrix(qw, qx, qy, qz);
  
  const c2w = mat4.create(
    R[0], R[1], R[2], tx, 
    R[3], R[4], R[5], ty, 
    R[6], R[7], R[8], tz,  
    0, 0, 0, 1
  );

  return c2w;
}

function extractCameraParams(camera: CameraParams): { fl_x: number; fl_y: number; cx: number; cy: number } {

  if (camera.model === 'SIMPLE_RADIAL') {
    const [f, cx, cy] = camera.params;
    return { fl_x: f, fl_y: f, cx, cy };
  } else if (camera.model === 'PINHOLE') {
    const [fx, fy, cx, cy] = camera.params;
    return { fl_x: fx, fl_y: fy, cx, cy };
  } else if (camera.model === 'RADIAL') {
    const [f, cx, cy] = camera.params;
    return { fl_x: f, fl_y: f, cx, cy };
  } else if (camera.model === 'OPENCV') {
    const [fx, fy, cx, cy] = camera.params;
    return { fl_x: fx, fl_y: fy, cx, cy };
  } else {
    // Default: assume first param is focal length
    const f = camera.params[0] || camera.width;
    return { fl_x: f, fl_y: f, cx: camera.width / 2, cy: camera.height / 2 };
  }
}

// Binary COLMAP parsers
function parseCamerasBinary(buffer: ArrayBuffer): Map<number, CameraParams> {
  const cameras = new Map<number, CameraParams>();
  const view = new DataView(buffer);
  let offset = 0;

  // Read number of cameras
  const numCameras = view.getBigUint64(offset, true);
  offset += 8;

  for (let i = 0; i < Number(numCameras); i++) {
    const camera_id = view.getUint32(offset, true);
    offset += 4;

    const model_id = view.getInt32(offset, true);
    offset += 4;

    const width = view.getBigUint64(offset, true);
    offset += 8;

    const height = view.getBigUint64(offset, true);
    offset += 8;

    // Model names by ID 
    const modelNames = ['', 'SIMPLE_PINHOLE', 'PINHOLE', 'SIMPLE_RADIAL', 'RADIAL', 'OPENCV', 'OPENCV_FISHEYE', 'FULL_OPENCV', 'FOV', 'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE', 'THIN_PRISM_FISHEYE'];
    const modelParamCounts = [0, 3, 4, 4, 5, 8, 8, 12, 5, 4, 5, 12];

    const model = modelNames[model_id] || 'PINHOLE';
    const numParams = modelParamCounts[model_id] || 4;

    const params: number[] = [];
    for (let j = 0; j < numParams; j++) {
      params.push(view.getFloat64(offset, true));
      offset += 8;
    }

    cameras.set(camera_id, {
      camera_id,
      model,
      width: Number(width),
      height: Number(height),
      params,
    });
  }

  return cameras;
}

function parseImagesBinary(buffer: ArrayBuffer): ImageData[] {
  const images: ImageData[] = [];
  const view = new DataView(buffer);
  let offset = 0;

  // Read number of registered images
  const numRegImages = view.getBigUint64(offset, true);
  offset += 8;

  for (let i = 0; i < Number(numRegImages); i++) {
    const image_id = view.getUint32(offset, true);
    offset += 4;

    const qw = view.getFloat64(offset, true);
    offset += 8;
    const qx = view.getFloat64(offset, true);
    offset += 8;
    const qy = view.getFloat64(offset, true);
    offset += 8;
    const qz = view.getFloat64(offset, true);
    offset += 8;

    const tx = view.getFloat64(offset, true);
    offset += 8;
    const ty = view.getFloat64(offset, true);
    offset += 8;
    const tz = view.getFloat64(offset, true);
    offset += 8;

    const camera_id = view.getUint32(offset, true);
    offset += 4;

    // Read image name 
    let name = '';
    while (offset < buffer.byteLength) {
      const char = view.getUint8(offset);
      offset += 1;
      if (char === 0) break;
      name += String.fromCharCode(char);
    }

    // Read number of 2D points
    const numPoints2D = view.getBigUint64(offset, true);
    offset += 8;

    // Skip 2D points 
    offset += Number(numPoints2D) * (8 + 8 + 8);

    images.push({
      image_id,
      qw,
      qx,
      qy,
      qz,
      tx,
      ty,
      tz,
      camera_id,
      name,
    });
  }

  return images;
}

function parsePoints3DBinary(buffer: ArrayBuffer): XYZRGB {
  const view = new DataView(buffer);
  let offset = 0;

  // Read number of points
  const numPoints = view.getBigUint64(offset, true);
  offset += 8;

  const xyz: number[] = [];
  const rgb: number[] = [];

  for (let i = 0; i < Number(numPoints); i++) {
    offset += 8;

    const x = view.getFloat64(offset, true);
    offset += 8;
    const y = view.getFloat64(offset, true);
    offset += 8;
    const z = view.getFloat64(offset, true);
    offset += 8;

    const r = view.getUint8(offset);
    offset += 1;
    const g = view.getUint8(offset);
    offset += 1;
    const b = view.getUint8(offset);
    offset += 1;

    offset += 8;

    // Read track length and skip track elements
    const trackLength = view.getBigUint64(offset, true);
    offset += 8;
    offset += Number(trackLength) * (4 + 4); // image_id + point2D_idx 

    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
      xyz.push(x, y, z);
      rgb.push(
        Math.max(0, Math.min(255, r)),
        Math.max(0, Math.min(255, g)),
        Math.max(0, Math.min(255, b))
      );
    }
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
    label: 'colmap 3d gaussians',
    size: num_points * c_size_3d_gaussian,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const gaussian = new Float16Array(gaussian_3d_buffer.getMappedRange());

  const sh_buffer = device.createBuffer({
    label: 'colmap sh coeffs',
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

export async function loadColmapDataset(
  dirHandle: any,
  device: GPUDevice
): Promise<ColmapDataset> {
  // Try to find COLMAP files - first try binary, then text
  let cameras: Map<number, CameraParams>;
  let images: ImageData[];
  let parsed: XYZRGB;

  // Try binary format first
  try {
    console.log('Attempting to load binary COLMAP files...');
    const camerasBuffer = await readBinaryFromDir(dirHandle, 'sparse/0/cameras.bin');
    const imagesBuffer = await readBinaryFromDir(dirHandle, 'sparse/0/images.bin');
    const pointsBuffer = await readBinaryFromDir(dirHandle, 'sparse/0/points3D.bin');

    cameras = parseCamerasBinary(camerasBuffer);
    images = parseImagesBinary(imagesBuffer);
    parsed = parsePoints3DBinary(pointsBuffer);
    console.log('Successfully loaded binary COLMAP files');
  } catch (binaryError) {
    console.log('Binary files not found, trying text format...', binaryError);

    // Fall back to text format
    let camerasText: string;
    let imagesText: string;
    let pointsText: string;

    try {
      camerasText = await readTextFromDir(dirHandle, 'sparse/cameras.txt');
      imagesText = await readTextFromDir(dirHandle, 'sparse/images.txt');
      pointsText = await readTextFromDir(dirHandle, 'sparse/points3D.txt');
    } catch {
      try {
        camerasText = await readTextFromDir(dirHandle, 'sparse/0/cameras.txt');
        imagesText = await readTextFromDir(dirHandle, 'sparse/0/images.txt');
        pointsText = await readTextFromDir(dirHandle, 'sparse/0/points3D.txt');
      } catch {
        // Try root directory
        camerasText = await readTextFromDir(dirHandle, 'cameras.txt');
        imagesText = await readTextFromDir(dirHandle, 'images.txt');
        pointsText = await readTextFromDir(dirHandle, 'points3D.txt');
      }
    }

    cameras = parseCameras(camerasText);
    images = parseImages(imagesText);
    parsed = parsePoints3D(pointsText);
    console.log('Successfully loaded text COLMAP files');
  }

  // Normalize point cloud
  const { xyzNorm, scale, zMin } = normalizeXYZ(parsed.xyz);

  console.log(
    `COLMAP normalization: num_points=${xyzNorm.length / 3}, ` +
    `scale=${scale.toFixed(4)}, zMin=${zMin.toFixed(4)}`
  );

  const pointCloud = createPointCloudFromXYZRGB(xyzNorm, parsed.rgb, device);

  const znear = 0.01;
  const zfar = 4_000_000;

  const trainingCameras: TrainingCameraData[] = [];
  const trainingImages: Array<{ view: GPUTextureView; sampler: GPUSampler }> = [];

  // Process each image
  for (const image of images) {
    const camera = cameras.get(image.camera_id);
    if (!camera) {
      console.warn(`Camera ${image.camera_id} not found for image ${image.name}`);
      continue;
    }

    // Extract camera parameters
    const { fl_x, fl_y, cx, cy } = extractCameraParams(camera);
    const w = camera.width;
    const h = camera.height;

    // Convert COLMAP pose to camera-to-world matrix
    const c2w = colmapPoseToC2W(image.qw, image.qx, image.qy, image.qz, image.tx, image.ty, image.tz);

    // Apply normalization scale to translation
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

    trainingCameras.push({
      view,
      view_inv,
      proj,
      proj_inv,
      viewport,
      focal,
      center,
      center_px,
      imgSize,
      file_path: image.name,
    });

    try {
      const imgPath = `images/${image.name}`;
      const tex = await loadImageToTextureFromDir(dirHandle, imgPath, device);
      trainingImages.push(tex);
    } catch (err) {
      console.warn(`Failed to load image ${image.name}:`, err);
      const placeholderTexture = device.createTexture({
        size: [w, h, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      trainingImages.push({
        view: placeholderTexture.createView(),
        sampler: device.createSampler({
          magFilter: 'linear',
          minFilter: 'linear',
          addressModeU: 'clamp-to-edge',
          addressModeV: 'clamp-to-edge',
        }),
      });
    }
  }

  console.log(`Loaded ${trainingCameras.length} cameras and ${trainingImages.length} images from COLMAP dataset`);

  return {
    pointCloud,
    trainingCameras,
    trainingImages,
  };
}

