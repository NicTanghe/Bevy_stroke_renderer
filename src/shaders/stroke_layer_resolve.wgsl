struct LiveStroke {
    mask_layer: u32,
    canvas_layer: u32,
    model_and_deposition: u32,
    padding: u32,
}

struct LiveLayer {
    canvas_layer: u32,
    texture_layer: u32,
    opacity: f32,
    stroke_count: u32,
}

@group(0) @binding(0) var live_stroke_masks: texture_2d_array<f32>;
@group(0) @binding(1) var cached_live_layers: texture_2d_array<f32>;
@group(0) @binding(2) var<storage, read> live_strokes: array<LiveStroke>;
@group(0) @binding(3) var<storage, read> live_layers: array<LiveLayer>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) live_layer_index: u32,
}

const FULLSCREEN_TRIANGLE: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0),
);

fn rgba8_checkpoint(value: vec4<f32>) -> vec4<f32> {
    return unpack4x8unorm(pack4x8unorm(clamp(value, vec4(0.0), vec4(1.0))));
}

@vertex
fn vertex_resolve(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    return VertexOutput(
        vec4(FULLSCREEN_TRIANGLE[vertex_index], 0.0, 1.0),
        instance_index,
    );
}

@fragment
fn fragment_resolve(in: VertexOutput) -> @location(0) vec4<f32> {
    let layer = live_layers[in.live_layer_index];
    let pixel = vec2<i32>(in.clip_position.xy);
    var destination = textureLoad(
        cached_live_layers,
        pixel,
        i32(layer.texture_layer),
        0,
    );

    for (var index = 0u; index < layer.stroke_count; index += 1u) {
        let stroke = live_strokes[index];
        if stroke.canvas_layer != layer.canvas_layer {
            continue;
        }
        let source = textureLoad(
            live_stroke_masks,
            pixel,
            i32(stroke.mask_layer),
            0,
        );
        let erase = (stroke.model_and_deposition & 0x80000000u) != 0u;
        if erase {
            destination *= 1.0 - source.a;
        } else {
            destination = source + destination * (1.0 - source.a);
        }
        // cached_live_layers is an RGBA8 checkpoint. Preserve that same
        // whole-stroke checkpoint between every concurrently live stroke so
        // persistent replay produces the identical color after retirement.
        destination = rgba8_checkpoint(destination);
    }
    return destination;
}
