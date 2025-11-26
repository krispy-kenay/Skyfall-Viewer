import { vec3, mat3, mat4, quat } from 'wgpu-matrix';
import { Camera } from './camera';

export class CameraControl {
  element: HTMLCanvasElement;
  constructor(private camera: Camera) {
    this.register_element(camera.canvas);
  }

  register_element(value: HTMLCanvasElement) {
    if (this.element && this.element != value) {
      this.element.removeEventListener('pointerdown', this.downCallback.bind(this));
      this.element.removeEventListener('pointermove', this.moveCallback.bind(this));
      this.element.removeEventListener('pointerup', this.upCallback.bind(this));
      this.element.removeEventListener('wheel', this.wheelCallback.bind(this));
    }

    this.element = value;
    this.element.addEventListener('pointerdown', this.downCallback.bind(this));
    this.element.addEventListener('pointermove', this.moveCallback.bind(this));
    this.element.addEventListener('pointerup', this.upCallback.bind(this));
    this.element.addEventListener('wheel', this.wheelCallback.bind(this));
    this.element.addEventListener('contextmenu', (e) => { e.preventDefault(); });
  }

  private panning = false;
  private rotating = false;
  private lastX: number;
  private lastY: number;
  private zoomSpeed = 0.1;
  private panSpeed = 0.002;
  private rotateSpeed = 0.003;

  downCallback(event: PointerEvent) {
    if (!event.isPrimary) {
      return;
    }
    this.camera.unlock_training_camera();
    this.panning = event.button === 0;
    this.rotating = event.button === 2;
    this.lastX = event.pageX;
    this.lastY = event.pageY;
  }
  moveCallback(event: PointerEvent) {
    if (!(this.rotating || this.panning)) {
      return;
    }

    const dx = event.pageX - this.lastX;
    const dy = event.pageY - this.lastY;
    this.lastX = event.pageX;
    this.lastY = event.pageY;

    if (this.panning) this.pan(dx, dy);
    if (this.rotating) this.rotate(dx, dy);
  }
  upCallback(_: PointerEvent) {
    this.rotating = false;
    this.panning = false;
  }
  wheelCallback(event: WheelEvent) {
    event.preventDefault();
    this.camera.unlock_training_camera();
    this.zoom(event.deltaY);
  }

  private zoom(deltaY: number) {
    // Zoom in and out for map view
    this.camera.position[2] *= 1 + deltaY * this.zoomSpeed * 0.01;
    this.camera.position[2] = Math.max(0.1, this.camera.position[2]);
    this.camera.update_buffer();
  }

  private rotate(dx: number, dy: number) {
    // Rotate the map view
    const yaw = -dx * this.rotateSpeed;
    const pitch = Math.max(-0.8, Math.min(0.8, -dy * this.rotateSpeed));

    const rot = mat4.identity();
    mat4.rotateZ(rot, yaw, rot);
    mat4.rotateX(rot, pitch, rot);
    mat4.mul(rot, this.camera.rotation, this.camera.rotation);
    this.camera.update_buffer();
  }

  private pan(dx: number, dy: number) {
    // Pan the map view
    dx = -dx;
    dy = -dy;
    const scale = this.panSpeed * this.camera.position[2];
    const right = vec3.fromValues(1, 0, 0);
    const forward = vec3.fromValues(0, 1, 0);
    const move = vec3.create();
    vec3.scale(right, dx * scale, move);
    vec3.add(move, vec3.scale(forward, dy * scale, forward), move);

    vec3.add(this.camera.position, move, this.camera.position);
    this.camera.update_buffer();
  }
};