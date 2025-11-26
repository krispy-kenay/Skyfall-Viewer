import { Mat4, Vec2 } from 'wgpu-matrix';

// Base camera layout for both viewer and training cameras

const c_size_vec2 = 4 * 2;
const c_size_mat4 = 4 * 16;

export const c_size_camera_uniform = 4 * c_size_mat4 + 2 * c_size_vec2;

export interface CameraUniform {
  view_matrix: Mat4;
  view_inv_matrix: Mat4;
  proj_matrix: Mat4;
  proj_inv_matrix: Mat4;
  viewport: Vec2;
  focal: Vec2;
}

export function create_camera_uniform_buffer(device: GPUDevice): GPUBuffer {
  return device.createBuffer({
    label: 'camera uniform',
    size: c_size_camera_uniform,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}


