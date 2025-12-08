struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) rgba: vec4<f32>,
    @location(1) opacity: f32,
    @location(2) center: vec2<f32>,
    @location(3) conic: vec3<f32>,
};

@group(0) @binding(0)
var<storage, read> splats : array<Splat>;
@group(0) @binding(1)
var<storage, read> sort_indices : array<u32>;

const NORMALS: array<vec2<f32>,6> = array<vec2<f32>,6>(
    vec2<f32>(-1.0f, -1.0f),
    vec2<f32>( 1.0f, -1.0f),
    vec2<f32>( 1.0f,  1.0f),
    vec2<f32>(-1.0f, -1.0f),
    vec2<f32>( 1.0f,  1.0f),
    vec2<f32>(-1.0f,  1.0f),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;
    let index = sort_indices[instance_index];
    let splat = splats[index];

    let center = unpack2x16float(splat.center);
    let size = unpack2x16float(splat.size);
    let rg = unpack2x16float(splat.rgba[0]);
    let ba = unpack2x16float(splat.rgba[1]);
    let conic_xy = unpack2x16float(splat.conic[0]);
    let conic_zw = unpack2x16float(splat.conic[1]);

    let conic = vec3<f32>(conic_xy, conic_zw.x);
    let radius = conic_zw.y;
    let offset = NORMALS[vertex_index % 6u] * radius / size;

    out.position = vec4<f32>(center + offset, 0.0f, 1.0f);
    out.rgba = vec4<f32>(rg, ba.x, 1.0f);
    out.opacity = ba.y;
    out.center = vec2<f32>(center.x + 1.0f, -center.y + 1.0f) * size - 1.0f;
    out.conic = conic;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let x = in.conic.x;
    let y = in.conic.y;
    let z = in.conic.z;

    let offset = in.position.xy - in.center;
    let rx = offset.x;
    let ry = offset.y;

    let prod = x * rx * rx + z * ry * ry + 2.0f * y * rx * ry;

    if (prod <= 0.0f) {
        discard;
    }

    let opacity = clamp(in.opacity * exp(-0.5f * prod), 0.0f, 0.99f);
    return in.rgba * opacity;
}