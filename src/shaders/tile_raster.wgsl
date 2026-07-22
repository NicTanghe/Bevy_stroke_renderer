#import hamerons_stroke_render::stroke_geometry as geometry

struct RgbaMaterial {
    color: vec4<f32>,
}

struct TileJob {
    slot: u32,
    replay_start: u32,
    replay_count: u32,
    tile_size: u32,
    origin_x: i32,
    origin_y: i32,
    padding_0: u32,
    padding_1: u32,
}

struct ReplayStroke {
    segment_index_start: u32,
    segment_count: u32,
    material: u32,
    model_and_deposition: u32,
}

@group(0) @binding(2) var<storage, read> rgba_materials: array<RgbaMaterial>;
@group(0) @binding(3) var<storage, read> tile_jobs: array<TileJob>;
@group(0) @binding(4) var<storage, read> replay_strokes: array<ReplayStroke>;
@group(0) @binding(5) var<storage, read> replay_segment_indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> tile_pixels: array<u32>;

fn rgba8_checkpoint(value: vec4<f32>) -> vec4<f32> {
    return unpack4x8unorm(pack4x8unorm(clamp(value, vec4(0.0), vec4(1.0))));
}

// One invocation owns one destination pixel. It replays strokes in document
// order and takes Max coverage across every segment of each stroke before a
// single deposition. That prevents translucent joins from darkening.
@compute @workgroup_size(8, 8, 1)
fn rasterize_tile(@builtin(global_invocation_id) id: vec3<u32>) {
    let job = tile_jobs[id.z];
    if id.x >= job.tile_size || id.y >= job.tile_size {
        return;
    }

    let pixel = vec2(
        f32(job.origin_x) + f32(id.x) + 0.5,
        f32(job.origin_y) + f32(id.y) + 0.5,
    );
    var destination = vec4(0.0);

    for (var replay_index = 0u; replay_index < job.replay_count; replay_index += 1u) {
        let replay = replay_strokes[job.replay_start + replay_index];
        let model = replay.model_and_deposition & 0x7fffffffu;
        if model != 0u {
            continue;
        }

        var amount = 0.0;
        for (var segment_offset = 0u; segment_offset < replay.segment_count; segment_offset += 1u) {
            let segment_index = replay_segment_indices[replay.segment_index_start + segment_offset];
            amount = max(amount, geometry::segment_body_amount(pixel, segment_index));
            amount = max(amount, geometry::segment_cap_amount(pixel, segment_index));
        }

        let material = rgba_materials[replay.material].color;
        let erase = (replay.model_and_deposition & 0x80000000u) != 0u;
        if erase {
            destination *= 1.0 - clamp(amount * material.a, 0.0, 1.0);
        } else {
            let source = material * clamp(amount, 0.0, 1.0);
            destination = source + destination * (1.0 - source.a);
        }
        // The live path starts each new stroke from the completed RGBA8 tile.
        // Use the same whole-stroke checkpoint during replay so a stroke does
        // not change color when it moves from the live overlay into this tile.
        destination = rgba8_checkpoint(destination);
    }

    let tile_area = job.tile_size * job.tile_size;
    let pixel_index = job.slot * tile_area + id.y * job.tile_size + id.x;
    tile_pixels[pixel_index] = pack4x8unorm(clamp(destination, vec4(0.0), vec4(1.0)));
}
