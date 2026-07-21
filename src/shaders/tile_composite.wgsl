#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;

struct TileInstance {
    slot: u32,
    tile_x: i32,
    tile_y: i32,
    tile_size: u32,
    opacity: f32,
}

@group(1) @binding(0) var<storage, read> tile_pixels: array<u32>;
@group(1) @binding(1) var<storage, read> tile_instances: array<TileInstance>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) @interpolate(flat) slot: u32,
    @location(2) @interpolate(flat) tile_size: u32,
    @location(3) @interpolate(flat) opacity: f32,
}

const QUAD: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0),
);

@vertex
fn vertex_tile(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let instance = tile_instances[instance_index];
    let local = QUAD[vertex_index];
    let size = f32(instance.tile_size);
    let world = vec2(f32(instance.tile_x), f32(instance.tile_y)) * size + local * size;
    return VertexOutput(
        view.clip_from_world * vec4(world, 0.0, 1.0),
        local,
        instance.slot,
        instance.tile_size,
        instance.opacity,
    );
}

@fragment
fn fragment_tile(in: VertexOutput) -> @location(0) vec4<f32> {
    let size = in.tile_size;
    let pixel = min(vec2<u32>(in.local * f32(size)), vec2(size - 1u));
    let index = in.slot * size * size + pixel.y * size + pixel.x;
    return unpack4x8unorm(tile_pixels[index]) * in.opacity;
}
