@group(3) @binding(0) var<uniform> settings: RenderSettings;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) pixel_px : vec2<f32>,
    @location(1) center_px : vec2<f32>,
    @location(2) color     : vec3<f32>,
    @location(3) conicA    : vec3<f32>,
    @location(4) weight    : f32,
};

@group(1) @binding(0) var<storage, read> splats : array<Splat>;
@group(2) @binding(0) var<storage, read> sorted_vis_indices : array<u32>;

@vertex
fn vs_main(
    @builtin(instance_index) iid: u32,
    @builtin(vertex_index) vid: u32
) -> VertexOutput {
    var out: VertexOutput;

    let visIdx = sorted_vis_indices[iid];
    let s = splats[visIdx];
    let center_ndc = s.center_ndc;
    let vpx = settings.viewport_x;
    let vpy = settings.viewport_y;

    let center_px = (center_ndc * vec2<f32>(0.5, -0.5) + 0.5) * vec2<f32>(vpx, vpy);

    let corners_px = array<vec2<f32>, 6>(
        vec2<f32>(-s.radius.x, -s.radius.y),
        vec2<f32>( s.radius.x, -s.radius.y),
        vec2<f32>(-s.radius.x,  s.radius.y),
        vec2<f32>(-s.radius.x,  s.radius.y),
        vec2<f32>( s.radius.x, -s.radius.y),
        vec2<f32>( s.radius.x,  s.radius.y)
    );

    let pixel_px = center_px + corners_px[vid % 6u];

    let ndc = vec2<f32>(
        (pixel_px.x / vpx) * 2.0 - 1.0,
        1.0 - (pixel_px.y / vpy) * 2.0
    );

    out.position = vec4<f32>(ndc, 0.0, 1.0);
    out.pixel_px = pixel_px;
    out.center_px = center_px;
    out.color = s.color;
    out.conicA = vec3<f32>(s.a11, s.a12, s.a22);
    out.weight = s.opacity;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = in.pixel_px - in.center_px;
    let a = in.conicA.x;
    let b = in.conicA.y;
    let c = in.conicA.z;

    let sigma_over_2 = 0.5 * (a * d.x * d.x + c * d.y * d.y) + b * d.x * d.y;
    if (sigma_over_2 < 0.0) { discard; }
    let gaussian = exp(-sigma_over_2);
    let alpha = clamp(in.weight * gaussian, 0.0, 1.0);
    
    return vec4<f32>(in.color, alpha);
}