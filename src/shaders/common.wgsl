const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);
const SH_WORDS_PER_POINT : u32 = 24u;

struct CameraUniforms {
  view: mat4x4<f32>,
  view_inv: mat4x4<f32>,
  proj: mat4x4<f32>,
  proj_inv: mat4x4<f32>,
  viewport: vec2<f32>,
  focal: vec2<f32>,
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
    viewport_x: f32,
    viewport_y: f32,
    alpha_scale: f32,
    gaussian_mode: f32,
};

struct Gaussian {
  pos_opacity: array<u32, 2>,
  rot:         array<u32, 2>,
  scale:       array<u32, 2>,
};

struct Splat {
    //TODO: store information for 2D splat rendering
    center_ndc : vec2<f32>,
    radius : vec2<f32>,
    depth : f32,
    color : vec3<f32>,
    a11 : f32,
    a12 : f32,
    a22 : f32,
    opacity: f32,
};

struct GaussProjection {
    mean2d: vec2<f32>,
    radius: f32,
    depth: f32,
    conic: vec3<f32>,
    tiles_count: u32,
};