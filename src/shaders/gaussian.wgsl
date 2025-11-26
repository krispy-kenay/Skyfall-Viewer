@group(3) @binding(0) var<uniform> settings: RenderSettings;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) center_px : vec2<f32>,
    @location(1) half_px   : vec2<f32>,
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
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;

    let visIdx = sorted_vis_indices[iid];
    let s = splats[visIdx];
    let center = s.center_ndc;
    let vpx = settings.viewport_x;
    let vpy = settings.viewport_y;

    let radius_ndc = vec2<f32>(2.0 * s.radius.x / vpx, 2.0 * s.radius.y / vpy);
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-radius_ndc.x, -radius_ndc.y),
        vec2<f32>( radius_ndc.x, -radius_ndc.y),
        vec2<f32>(-radius_ndc.x,  radius_ndc.y),
        vec2<f32>(-radius_ndc.x,  radius_ndc.y),
        vec2<f32>( radius_ndc.x, -radius_ndc.y),
        vec2<f32>( radius_ndc.x,  radius_ndc.y)
    );

    let half_px = max(
    vec2<f32>(3.0, 3.0),
    vec2<f32>(
        s.radius.x * 0.5 * settings.viewport_x,
        s.radius.y * 0.5 * settings.viewport_y
    ));

    out.position = vec4<f32>(center + corners[vid % 6u], 0.0, 1.0);
    out.center_px = (center * vec2<f32>(0.5, -0.5) + 0.5) * vec2<f32>(vpx, vpy);
    out.half_px = half_px;
    out.color = s.color;
    out.conicA = vec3<f32>(s.a11, s.a12, s.a22);
    out.weight = s.opacity;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = in.position.xy - in.center_px;
    let a = in.conicA.x;
    let b = in.conicA.y;
    let c = in.conicA.z;
    let t = a * d.x * d.x + 2.0 * b * d.x * d.y + c * d.y * d.y;
    if (t > 9.0) { discard; }
    let alpha = clamp(in.weight * exp(-0.5 * t), 0.0, 1.0);
    
    return vec4<f32>(in.color, alpha);
}