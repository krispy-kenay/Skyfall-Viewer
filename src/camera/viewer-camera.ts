import { Mat3, mat3, Mat4, mat4, Vec3, vec3, Vec2, vec2 } from 'wgpu-matrix';
import { log } from '../utils/simple-console';
import { c_size_camera_uniform, create_camera_uniform_buffer } from './camera-uniform';
import type { TrainingCameraData } from './training-camera';

interface CameraJson {
  id: number;
  img_name: string;
  width: number;
  height: number;
  position: number[];
  rotation: number[][];
  fx: number;
  fy: number;
}

function focal2fov(focal: number, pixels: number): number {
  return 2 * Math.atan(pixels / (2 * focal));
}

function get_view_matrix(r: Mat4, t: Vec3): Mat4 {
  const minus_t = vec3.mulScalar(t, -1);
  return mat4.translate(r, minus_t);
}

function get_projection_matrix(znear: number, zfar: number, fov_x: number, fov_y: number) {
  const tan_half_fov_y = Math.tan(fov_y / 2.);
  const tan_half_fov_x = Math.tan(fov_x / 2.);

  const top = tan_half_fov_y * znear;
  const bottom = -top;
  const right = tan_half_fov_x * znear;
  const left = -right;

  const p = mat4.create();
  p[0] = 2.0 * znear / (right - left);
  p[5] = -2.0 * znear / (top - bottom);
  p[2] = (right + left) / (right - left);
  p[6] = (top + bottom) / (top - bottom);
  p[14] = 1.0;
  p[10] = zfar / (zfar - znear);
  p[11] = -(zfar * znear) / (zfar - znear);
  mat4.transpose(p, p);
  return p;
}

export interface CameraPreset {
  position: Vec3;
  rotation: Mat4;
}

export async function load_camera_presets(file: string): Promise<CameraPreset[]> {
  const blob = new Blob([file]);
  const arrayBuffer = await new Promise<ArrayBuffer>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (event) => resolve(event.target!.result as ArrayBuffer);
    reader.onerror = reject;
    reader.readAsArrayBuffer(blob);
  });
  const text = new TextDecoder().decode(arrayBuffer);
  const json = JSON.parse(text);
  log(`loaded cameras count: ${json.length}`);

  return json.map((j: CameraJson): CameraPreset => {
    const position = vec3.clone(j.position);
    const rotation = mat4.fromMat3(mat3.create(...j.rotation.flat()));
    return { position, rotation };
  });
}

export async function load_satellite_camera(file: string): Promise<CameraPreset[]> {
  const blob = new Blob([file]);
  const arrayBuffer = await blob.arrayBuffer();
  const text = new TextDecoder().decode(arrayBuffer);
  const json = JSON.parse(text);

  const K = json.K;
  const W2C = json.W2C;
  const imgSize = json.img_size;

  const EngineBasis = mat4.create(
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, -1, 0,
    0, 0, 0, 1,
  );
  const W2C_engine = mat4.multiply(EngineBasis, W2C);
  const C2W_engine = mat4.invert(W2C_engine);

  const position = vec3.create(C2W_engine[12], C2W_engine[13], C2W_engine[14]);

  const rotation = mat4.copy(C2W_engine);
  rotation[12] = rotation[13] = rotation[14] = 0;

  const fx = K[0];
  const fy = K[5];

  const width = imgSize[0];
  const height = imgSize[1];

  const fovX = 2 * Math.atan(width / (2 * fx));
  const fovY = 2 * Math.atan(height / (2 * fy));
  void fovX;
  void fovY;

  return [{ position, rotation }];
}

const intermediate_float_32_array = new Float32Array(
  c_size_camera_uniform / Float32Array.BYTES_PER_ELEMENT
);

export class ViewerCamera {
  constructor(
    public readonly canvas: HTMLCanvasElement,
    private readonly device: GPUDevice,
  ) {
    this.uniform_buffer = create_camera_uniform_buffer(device);
    this.on_update_canvas();
  }

  on_update_canvas(): void {
    if (this.trainingCameraLock) {
      return;
    }
    this.viewport[0] = this.canvas.width;
    this.viewport[1] = this.canvas.height;

    if (!this.customProjection) {
      const focal = 0.5 * this.canvas.height / Math.tan(this.fovY * 0.5);
      this.focal[0] = focal;
      this.focal[1] = focal;
      this.fovX = focal2fov(focal, this.canvas.width);
    }

    this.update_buffer();
  }

  readonly uniform_buffer: GPUBuffer;

  public needsPreprocess = true;

  position = vec3.create(0, 0, 100);
  rotation = mat4.fromMat3(mat3.create(
    1, 0, 0,
    0, 1, 0,
    0, 0, -1,
  ));
  public fovY: number = 45 / 180 * Math.PI;
  public fovX: number;
  public focal: Vec2 = vec2.create();
  public viewport: Vec2 = vec2.create();
  private customProjection: Mat4 | null = null;
  private customProjectionInv: Mat4 | null = null;
  private trainingCameraLock: TrainingCameraData | null = null;

  public view_matrix: Mat4 = mat4.identity();
  public proj_matrix: Mat4 = mat4.identity();

  look = vec3.create(0, 0, 1);
  up = vec3.create(0, 1, 0);
  right = vec3.create(1, 0, 0);

  update_buffer(): void {
    let offset = 0;

    if (this.trainingCameraLock) {
      this.write_training_camera(this.trainingCameraLock);
      return;
    }

    this.view_matrix = get_view_matrix(this.rotation, this.position);
    if (this.customProjection) {
      this.proj_matrix = mat4.copy(this.customProjection, this.proj_matrix);
    } else {
      this.proj_matrix = get_projection_matrix(0.01, 4_000_000, this.fovX, this.fovY);
    }

    const inv_view_matrix = mat4.inverse(this.view_matrix);
    const proj_inv_matrix = this.customProjectionInv
      ? mat4.copy(this.customProjectionInv, mat4.create())
      : mat4.inverse(this.proj_matrix);
    vec3.transformMat4Upper3x3(vec3.create(0, 0, 1), inv_view_matrix, this.look);
    vec3.normalize(this.look, this.look);

    vec3.cross(this.up, this.look, this.right);
    vec3.normalize(this.right, this.right);

    intermediate_float_32_array.set(this.view_matrix, offset);
    offset += 16;
    intermediate_float_32_array.set(inv_view_matrix, offset);
    offset += 16;
    intermediate_float_32_array.set(this.proj_matrix, offset);
    offset += 16;
    intermediate_float_32_array.set(proj_inv_matrix, offset);
    offset += 16;
    intermediate_float_32_array.set(this.viewport, offset);
    offset += 2;
    intermediate_float_32_array.set(this.focal, offset);
    offset += 2;

    this.device.queue.writeBuffer(this.uniform_buffer, 0, intermediate_float_32_array);
    this.needsPreprocess = true;
  }

  set_preset(preset: CameraPreset): void {
    this.clear_custom_projection();
    this.unlock_training_camera();
    vec3.copy(preset.position, this.position);
    mat4.copy(preset.rotation, this.rotation);
    this.update_buffer();
  }

  set_from_training_camera(trainingCamera: TrainingCameraData): void {
    this.trainingCameraLock = trainingCamera;
    this.clear_custom_projection();
    console.log('[viewer-camera] Locking onto training camera', trainingCamera.file_path);
    this.write_training_camera(trainingCamera);
  }

  clear_custom_projection(): void {
    this.customProjection = null;
    this.customProjectionInv = null;
  }

  unlock_training_camera(): void {
    this.trainingCameraLock = null;
  }

  is_training_camera_locked(): boolean {
    return this.trainingCameraLock !== null;
  }

  private write_training_camera(trainingCamera: TrainingCameraData): void {
    const { view, view_inv, proj, proj_inv, viewport, focal } = trainingCamera;

    mat4.copy(view, this.view_matrix);
    vec3.copy(vec3.create(view_inv[12], view_inv[13], view_inv[14]), this.position);
    mat4.copy(view, this.rotation);
    this.rotation[12] = 0;
    this.rotation[13] = 0;
    this.rotation[14] = 0;
    this.rotation[15] = 1;

    mat4.copy(proj, this.proj_matrix);
    vec2.copy(viewport, this.viewport);
    vec2.copy(focal, this.focal);

    const fx = focal[0];
    const fy = focal[1];
    const w = viewport[0];
    const h = viewport[1];
    this.fovX = focal2fov(fx, w);
    this.fovY = focal2fov(fy, h);

    const data = new Float32Array(c_size_camera_uniform / Float32Array.BYTES_PER_ELEMENT);
    let offset = 0;
    data.set(view, offset); offset += 16;
    data.set(view_inv, offset); offset += 16;
    data.set(proj, offset); offset += 16;
    data.set(proj_inv, offset); offset += 16;
    data.set(viewport, offset); offset += 2;
    data.set(focal, offset); offset += 2;

    this.device.queue.writeBuffer(this.uniform_buffer, 0, data);
    console.log('[viewer-camera] Applied training camera', {
      position: Array.from(this.position),
      view: Array.from(this.view_matrix),
      proj: Array.from(this.proj_matrix),
      viewport: Array.from(this.viewport),
      focal: Array.from(this.focal),
    });
    this.needsPreprocess = true;
  }
}


