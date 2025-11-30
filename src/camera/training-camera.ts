import { Mat4, mat4, Vec2, vec2 } from 'wgpu-matrix';
import { create_camera_uniform_buffer, c_size_camera_uniform } from './camera-uniform';

export interface TrainingCameraData {
  view: Mat4;     // world -> camera
  view_inv: Mat4; // camera -> world
  proj: Mat4;     // projection matrix
  proj_inv: Mat4; // inverse projection matrix
  viewport: Vec2; // (w, h)
  focal: Vec2;    // (fx, fy) in pixels
  center: Vec2;   // normalized (cx, cy) in clip space
  center_px?: Vec2; // optional raw principal point in pixels
  imgSize: Vec2;  // (w, h)
  file_path: string; // image file path for reference
}

export function principal_point_to_ndc(
  cx: number,
  cy: number,
  w: number,
  h: number
): [number, number] {
  const ndcX = (cx - w * 0.5) / (w * 0.5);
  const ndcY = 1 - (cy - h * 0.5) / (h * 0.5);
  return [ndcX, ndcY];
}

// Helper function to create pinhole projection matrix from camera intrinsics
export function get_pinhole_projection_matrix(
  znear: number,
  zfar: number,
  fx: number,
  fy: number,
  cx: number,
  cy: number,
  w: number,
  h: number
): Mat4 {
  const sx = 2.0 * fx / w;
  const sy = -2.0 * fy / h;
  const ox = 2.0 * cx / w - 1.0;
  const oy = 1.0 - 2.0 * cy / h;

  const proj = mat4.create(
    sx,  0,  ox, 0,
    0,  sy,  oy, 0,
    0,   0,   1, 0,
    0,   0,   1, 0,
  );
  mat4.transpose(proj, proj);
  return proj;
}

export function create_training_camera_uniform_buffer(
  device: GPUDevice,
  cameraData: TrainingCameraData
): GPUBuffer {
  const buffer = create_camera_uniform_buffer(device);
  update_training_camera_uniform_buffer(device, buffer, cameraData);
  return buffer;
}

export function update_training_camera_uniform_buffer(
  device: GPUDevice,
  buffer: GPUBuffer,
  cameraData: TrainingCameraData
): void {
  const data = new Float32Array(c_size_camera_uniform / Float32Array.BYTES_PER_ELEMENT);
  let offset = 0;

  data.set(cameraData.view, offset);
  offset += 16;
  data.set(cameraData.view_inv, offset);
  offset += 16;
  data.set(cameraData.proj, offset);
  offset += 16;
  data.set(cameraData.proj_inv, offset);
  offset += 16;
  data.set(cameraData.viewport, offset);
  offset += 2;
  data.set(cameraData.focal, offset);
  offset += 2;

  device.queue.writeBuffer(buffer, 0, data);
}

export async function load_training_camera_from_transforms(
  file: string
): Promise<TrainingCameraData> {
  const blob = new Blob([file]);
  const arrayBuffer = await blob.arrayBuffer();
  const text = new TextDecoder().decode(arrayBuffer);
  const json = JSON.parse(text);

  const frame = json.frames[0];

  const fl_x: number = frame.fl_x;
  const fl_y: number = frame.fl_y;
  const cx: number   = frame.cx;
  const cy: number   = frame.cy;

  const w: number = json.w;
  const h: number = json.h;

  const tm: number[][] = frame.transform_matrix;

  const c2w = mat4.create(
    tm[0][0], tm[1][0], tm[2][0], tm[3][0],
    tm[0][1], tm[1][1], tm[2][1], tm[3][1],
    tm[0][2], tm[1][2], tm[2][2], tm[3][2],
    tm[0][3], tm[1][3], tm[2][3], tm[3][3],
  );

  const view = mat4.invert(c2w);
  const view_inv = mat4.copy(c2w);

  const znear = 0.01;
  const zfar = 4_000_000;
  const proj = get_pinhole_projection_matrix(znear, zfar, fl_x, fl_y, cx, cy, w, h);
  const proj_inv = mat4.inverse(proj);

  const focal  = vec2.create(fl_x, fl_y);
  const [cx_ndc, cy_ndc] = principal_point_to_ndc(cx, cy, w, h);
  const center = vec2.create(cx_ndc, cy_ndc);
  const center_px = vec2.create(cx, cy);
  const viewport = vec2.create(w, h);
  const imgSize = vec2.create(w, h);

  return {
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
  };
}

export async function load_all_training_cameras_from_transforms(
  file: string
): Promise<TrainingCameraData[]> {
  const blob = new Blob([file]);
  const arrayBuffer = await blob.arrayBuffer();
  const text = new TextDecoder().decode(arrayBuffer);
  const json = JSON.parse(text);

  const w: number = json.w;
  const h: number = json.h;
  const znear = 0.01;
  const zfar = 4_000_000;

  const hasRT = Array.isArray(json.R) && Array.isArray(json.T);
  const R_fix = hasRT ? json.R : null;
  const T_fix = hasRT ? json.T : null;
  const c2wKey = hasRT ? 'transform_matrix_rotated' : 'transform_matrix';

  const cameras: TrainingCameraData[] = [];

  for (const frame of json.frames) {
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

    if (!hasRT) {
      c2w[12] *= 1;
      c2w[13] *= 1;
      c2w[14] *= 1;
    }

    const view = mat4.invert(c2w);
    const view_inv = mat4.copy(c2w);

    const proj = get_pinhole_projection_matrix(znear, zfar, fl_x, fl_y, cx, cy, w, h);
    const proj_inv = mat4.inverse(proj);

    const focal  = vec2.create(fl_x, fl_y);
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
  }

  return cameras;
}


