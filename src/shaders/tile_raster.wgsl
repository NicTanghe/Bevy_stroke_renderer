struct StrokePoint {
    position: vec2<f32>,
    half_width: f32,
    flow: f32,
    orientation: vec2<f32>,
    twist_radians: f32,
    aspect_ratio: f32,
}

struct StrokeSegment {
    start: u32,
    end: u32,
    material: u32,
    model_and_deposition: u32,
    layer: u32,
}

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

@group(0) @binding(0) var<storage, read> stroke_points: array<StrokePoint>;
@group(0) @binding(1) var<storage, read> stroke_segments: array<StrokeSegment>;
@group(0) @binding(2) var<storage, read> rgba_materials: array<RgbaMaterial>;
@group(0) @binding(3) var<storage, read> tile_jobs: array<TileJob>;
@group(0) @binding(4) var<storage, read> replay_strokes: array<ReplayStroke>;
@group(0) @binding(5) var<storage, read> replay_segment_indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> tile_pixels: array<u32>;

const SMOOTH_JOIN_MIN_DOT: f32 = 0.75;

fn rotate_vector(vector: vec2<f32>, angle: f32) -> vec2<f32> {
    let sine = sin(angle);
    let cosine = cos(angle);
    return vec2(
        cosine * vector.x - sine * vector.y,
        sine * vector.x + cosine * vector.y,
    );
}

fn point_major_axis(point: StrokePoint) -> vec2<f32> {
    var major_axis = point.orientation;
    if length(major_axis) <= 0.0001 {
        major_axis = vec2(0.0, 1.0);
    } else {
        major_axis = normalize(major_axis);
    }
    return rotate_vector(major_axis, point.twist_radians);
}

fn ellipse_support_radius(point: StrokePoint, axis: vec2<f32>) -> f32 {
    let minor_radius = max(point.half_width, 0.0001);
    let major_radius = minor_radius * max(point.aspect_ratio, 1.0);
    let major_axis = point_major_axis(point);
    let minor_axis = vec2(-major_axis.y, major_axis.x);
    let major_component = dot(axis, major_axis) * major_radius;
    let minor_component = dot(axis, minor_axis) * minor_radius;
    return sqrt(major_component * major_component + minor_component * minor_component);
}

fn ellipse_amount(pixel: vec2<f32>, point: StrokePoint) -> f32 {
    let minor_radius = max(point.half_width, 0.0001);
    let major_radius = minor_radius * max(point.aspect_ratio, 1.0);
    let major_axis = point_major_axis(point);
    let minor_axis = vec2(-major_axis.y, major_axis.x);
    let offset = pixel - point.position;
    let normalized_distance = length(vec2(
        dot(offset, major_axis) / major_radius,
        dot(offset, minor_axis) / minor_radius,
    ));
    let signed_distance = (normalized_distance - 1.0) * minor_radius;
    let coverage = clamp(0.5 - signed_distance, 0.0, 1.0);
    return coverage * clamp(point.flow, 0.0, 1.0);
}

fn segment_tangent(segment: StrokeSegment) -> vec2<f32> {
    let delta = stroke_points[segment.end].position - stroke_points[segment.start].position;
    if length(delta) <= 0.0001 {
        return vec2(0.0);
    }
    return normalize(delta);
}

fn join_tangent(incoming: vec2<f32>, outgoing: vec2<f32>) -> vec2<f32> {
    if length(incoming) <= 0.0001 {
        return outgoing;
    }
    if length(outgoing) <= 0.0001 {
        return incoming;
    }
    let joined = incoming + outgoing;
    if length(joined) <= 0.0001 {
        return outgoing;
    }
    return normalize(joined);
}

fn join_is_smooth(incoming: vec2<f32>, outgoing: vec2<f32>) -> bool {
    return length(incoming) > 0.0001
        && length(outgoing) > 0.0001
        && dot(normalize(incoming), normalize(outgoing)) >= SMOOTH_JOIN_MIN_DOT;
}

fn segment_body_amount(pixel: vec2<f32>, segment_index: u32) -> f32 {
    let segment = stroke_segments[segment_index];
    let point_a = stroke_points[segment.start];
    let point_b = stroke_points[segment.end];
    let delta = point_b.position - point_a.position;
    let length_squared = dot(delta, delta);
    if length_squared <= 0.000001 {
        return 0.0;
    }

    let tangent = delta * inverseSqrt(length_squared);
    var start_tangent = tangent;
    var end_tangent = tangent;
    if segment_index > 0u {
        let previous = stroke_segments[segment_index - 1u];
        if previous.end == segment.start {
            let incoming = segment_tangent(previous);
            if join_is_smooth(incoming, tangent) {
                start_tangent = join_tangent(incoming, tangent);
            }
        }
    }
    if segment_index + 1u < arrayLength(&stroke_segments) {
        let next = stroke_segments[segment_index + 1u];
        if next.start == segment.end {
            let outgoing = segment_tangent(next);
            if join_is_smooth(tangent, outgoing) {
                end_tangent = join_tangent(tangent, outgoing);
            }
        }
    }

    // Adjacent bodies use opposite halves of the same smoothed cross-section,
    // so an internal data point is not a hard round join.
    if dot(pixel - point_a.position, start_tangent) < 0.0
        || dot(pixel - point_b.position, end_tangent) > 0.0
    {
        return 0.0;
    }

    let t = clamp(dot(pixel - point_a.position, delta) / length_squared, 0.0, 1.0);
    let start_normal = vec2(-start_tangent.y, start_tangent.x);
    let end_normal = vec2(-end_tangent.y, end_tangent.x);
    var normal = mix(start_normal, end_normal, t);
    if length(normal) <= 0.0001 {
        normal = vec2(-tangent.y, tangent.x);
    } else {
        normal = normalize(normal);
    }
    let center = mix(point_a.position, point_b.position, t);
    let support_radius = mix(
        ellipse_support_radius(point_a, start_normal),
        ellipse_support_radius(point_b, end_normal),
        t,
    );
    let signed_distance = abs(dot(pixel - center, normal)) - support_radius;
    let coverage = clamp(0.5 - signed_distance, 0.0, 1.0);
    return coverage * clamp(mix(point_a.flow, point_b.flow, t), 0.0, 1.0);
}

fn segment_cap_amount(
    pixel: vec2<f32>,
    segment_index: u32,
) -> f32 {
    let segment = stroke_segments[segment_index];
    let point = stroke_points[segment.end];
    let offset = pixel - point.position;
    let incoming_delta = point.position - stroke_points[segment.start].position;
    let incoming_length = length(incoming_delta);
    var outgoing_delta = vec2(0.0);
    var outgoing_length = 0.0;

    if segment_index + 1u < arrayLength(&stroke_segments) {
        let next_segment = stroke_segments[segment_index + 1u];
        if next_segment.start == segment.end {
            outgoing_delta = stroke_points[next_segment.end].position - point.position;
            outgoing_length = length(outgoing_delta);
        }
    }

    if incoming_length > 0.0001
        && outgoing_length > 0.0001
        && join_is_smooth(incoming_delta, outgoing_delta)
    {
        return 0.0;
    }
    if incoming_length > 0.0001 && dot(offset, incoming_delta / incoming_length) <= 0.0 {
        return 0.0;
    }
    if outgoing_length > 0.0001 && dot(offset, outgoing_delta / outgoing_length) >= 0.0 {
        return 0.0;
    }

    return ellipse_amount(pixel, point);
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
            amount = max(amount, segment_body_amount(pixel, segment_index));
            amount = max(amount, segment_cap_amount(pixel, segment_index));
        }

        let material = rgba_materials[replay.material].color;
        let erase = (replay.model_and_deposition & 0x80000000u) != 0u;
        if erase {
            destination *= 1.0 - clamp(amount * material.a, 0.0, 1.0);
        } else {
            let source = material * clamp(amount, 0.0, 1.0);
            destination = source + destination * (1.0 - source.a);
        }
    }

    let tile_area = job.tile_size * job.tile_size;
    let pixel_index = job.slot * tile_area + id.y * job.tile_size + id.x;
    tile_pixels[pixel_index] = pack4x8unorm(clamp(destination, vec4(0.0), vec4(1.0)));
}
