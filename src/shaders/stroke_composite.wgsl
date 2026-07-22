#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var resolved_live_layers: texture_2d_array<f32>;

struct LiveLayer {
    canvas_layer: u32,
    texture_layer: u32,
    opacity: f32,
    stroke_count: u32,
}

@group(1) @binding(1) var<storage, read> live_layers: array<LiveLayer>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) live_layer_index: u32,
}

const FULLSCREEN_TRIANGLE: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0),
);

@vertex
fn vertex_composite(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    return VertexOutput(
        vec4(FULLSCREEN_TRIANGLE[vertex_index], 0.0, 1.0),
        instance_index,
    );
}

@fragment
fn fragment_composite(in: VertexOutput) -> @location(0) vec4<f32> {
    let layer = live_layers[in.live_layer_index];
    let dimensions = textureDimensions(resolved_live_layers);
    let local = in.clip_position.xy - view.viewport.xy;
    if local.x < 0.0
        || local.y < 0.0
        || local.x >= f32(dimensions.x)
        || local.y >= f32(dimensions.y)
    {
        discard;
    }
    return textureLoad(
        resolved_live_layers,
        vec2<i32>(local),
        i32(layer.texture_layer),
        0,
    ) * layer.opacity;
}
