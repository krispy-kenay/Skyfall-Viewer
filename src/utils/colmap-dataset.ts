import { mat4, vec2 } from 'wgpu-matrix';
import type { TrainingCameraData } from '../camera/camera';
import { get_pinhole_projection_matrix, principal_point_to_ndc } from '../camera/camera';
import type { XYZRGB } from './pointcloud-loader';
import { createPointCloudFromXYZRGB } from './pointcloud-loader';
import type { SceneDataset, TrainingImageResource } from './dataset-loader';

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
): Promise<TrainingImageResource> {
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

function readUint64(dv: DataView, offset: { value: number }): number {
  const v = dv.getBigUint64(offset.value, true);
  offset.value += 8;
  return Number(v);
}

function readInt64(dv: DataView, offset: { value: number }): number {
  const v = dv.getBigInt64(offset.value, true);
  offset.value += 8;
  return Number(v);
}

function readUint32(dv: DataView, offset: { value: number }): number {
  const v = dv.getUint32(offset.value, true);
  offset.value += 4;
  return v;
}

function readInt32(dv: DataView, offset: { value: number }): number {
  const v = dv.getInt32(offset.value, true);
  offset.value += 4;
  return v;
}

function readFloat64(dv: DataView, offset: { value: number }): number {
  const v = dv.getFloat64(offset.value, true);
  offset.value += 8;
  return v;
}

function readString(dv: DataView, offset: { value: number }): string {
  const chars: number[] = [];
  while (offset.value < dv.byteLength) {
    const c = dv.getUint8(offset.value++);
    if (c === 0) break;
    chars.push(c);
  }
  return new TextDecoder().decode(new Uint8Array(chars));
}

interface ColmapCamera {
  id: number;
  modelId: number;
  width: number;
  height: number;
  params: number[];
}

interface ColmapImage {
  id: number;
  qvec: [number, number, number, number];
  tvec: [number, number, number];
  cameraId: number;
  name: string;
}

function parseColmapCamerasBin(buffer: ArrayBuffer): Map<number, ColmapCamera> {
  const dv = new DataView(buffer);
  const offset = { value: 0 };
  const numCameras = readUint64(dv, offset);
  const cameras = new Map<number, ColmapCamera>();

  for (let i = 0; i < numCameras; i++) {
    const cameraId = readUint32(dv, offset);
    const modelId = readInt32(dv, offset);
    const width = readUint64(dv, offset);
    const height = readUint64(dv, offset);

    const params: number[] = [
      readFloat64(dv, offset),
      readFloat64(dv, offset),
      readFloat64(dv, offset),
      readFloat64(dv, offset),
    ];

    cameras.set(cameraId, { id: cameraId, modelId, width, height, params });
  }

  return cameras;
}

function parseColmapImagesBin(buffer: ArrayBuffer): ColmapImage[] {
  const dv = new DataView(buffer);
  const offset = { value: 0 };
  const numImages = readUint64(dv, offset);
  const images: ColmapImage[] = [];

  for (let i = 0; i < numImages; i++) {
    const imageId = readUint32(dv, offset);
    const qvec: [number, number, number, number] = [
      readFloat64(dv, offset),
      readFloat64(dv, offset),
      readFloat64(dv, offset),
      readFloat64(dv, offset),
    ];
    const tvec: [number, number, number] = [
      readFloat64(dv, offset),
      readFloat64(dv, offset),
      readFloat64(dv, offset),
    ];
    const cameraId = readUint32(dv, offset);
    const name = readString(dv, offset);

    const numPoints2D = readUint64(dv, offset);
    for (let j = 0; j < numPoints2D; j++) {
      offset.value += 8 * 2;
      offset.value += 8;
    }

    images.push({ id: imageId, qvec, tvec, cameraId, name });
  }

  return images;
}

function parseColmapPoints3DBin(buffer: ArrayBuffer): XYZRGB {
  const dv = new DataView(buffer);
  const offset = { value: 0 };
  const numPoints = readUint64(dv, offset);

  const xyz: number[] = [];
  const rgb: number[] = [];

  for (let i = 0; i < numPoints; i++) {
    readUint64(dv, offset);
    const x = readFloat64(dv, offset);
    const y = readFloat64(dv, offset);
    const z = readFloat64(dv, offset);

    const r = dv.getUint8(offset.value++);
    const g = dv.getUint8(offset.value++);
    const b = dv.getUint8(offset.value++);

    offset.value += 8;

    const trackLen = readUint64(dv, offset);
    offset.value += trackLen * (4 + 4);

    xyz.push(x, y, z);
    rgb.push(r, g, b);
  }

  return {
    xyz: new Float32Array(xyz),
    rgb: new Uint8Array(rgb),
  };
}

function qvecToRotMat(qw: number, qx: number, qy: number, qz: number): number[] {
  const xx = qx * qx;
  const yy = qy * qy;
  const zz = qz * qz;
  const xy = qx * qy;
  const xz = qx * qz;
  const yz = qy * qz;
  const wx = qw * qx;
  const wy = qw * qy;
  const wz = qw * qz;

  return [
    1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
    2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
    2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy),
  ];
}

export async function loadColmapDatasetScene(
  dirHandle: any,
  device: GPUDevice
): Promise<SceneDataset> {
  const sparsePrefix = 'sparse/0/';
  const camerasBuf = await readBinaryFromDir(dirHandle, sparsePrefix + 'cameras.bin');
  const imagesBuf = await readBinaryFromDir(dirHandle, sparsePrefix + 'images.bin');
  const pointsBuf = await readBinaryFromDir(dirHandle, sparsePrefix + 'points3D.bin');

  const camerasMap = parseColmapCamerasBin(camerasBuf);
  const imagesMeta = parseColmapImagesBin(imagesBuf);
  const points = parseColmapPoints3DBin(pointsBuf);

  const pointCloud = createPointCloudFromXYZRGB(points, device);

  const trainingCameras: TrainingCameraData[] = [];
  const trainingImages: TrainingImageResource[] = [];

  const znear = 0.01;
  const zfar = 4_000_000;

  for (const img of imagesMeta) {
    const cam = camerasMap.get(img.cameraId);
    if (!cam || cam.params.length < 4) {
      continue;
    }
    const fx = cam.params[0];
    const fy = cam.params[1];
    const cx = cam.params[2];
    const cy = cam.params[3];
    const w = cam.width;
    const h = cam.height;

    const [qw, qx, qy, qz] = img.qvec;
    const R = qvecToRotMat(qw, qx, qy, qz);
    const t = img.tvec;

    const cX =
      -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]);
    const cY =
      -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]);
    const cZ =
      -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]);

    const c2w = mat4.create(
      R[0], R[1], R[2], 0,
      R[3], R[4], R[5], 0,
      R[6], R[7], R[8], 0,
      cX,   cY,   cZ,   1,
    );

    const view = mat4.invert(c2w);
    const view_inv = mat4.copy(c2w);

    const proj = get_pinhole_projection_matrix(znear, zfar, fx, fy, cx, cy, w, h);
    const proj_inv = mat4.inverse(proj);

    const focal = vec2.create(fx, fy);
    const [cx_ndc, cy_ndc] = principal_point_to_ndc(cx, cy, w, h);
    const center = vec2.create(cx_ndc, cy_ndc);
    const center_px = vec2.create(cx, cy);
    const viewport = vec2.create(w, h);
    const imgSize = vec2.create(w, h);

    const file_path = img.name;
    const tex = await loadImageToTextureFromDir(dirHandle, `images/${file_path}`, device);

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
      file_path,
    });
    trainingImages.push(tex);
  }

  return {
    pointCloud,
    trainingCameras,
    trainingImages,
  };
}


