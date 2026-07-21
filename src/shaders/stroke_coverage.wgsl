#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;

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

struct CanvasLayer {
    opacity: f32,
    visible: u32,
    padding: vec2<u32>,
}

@group(1) @binding(0) var<storage, read> stroke_points: array<StrokePoint>;
@group(1) @binding(1) var<storage, read> stroke_segments: array<StrokeSegment>;
@group(1) @binding(2) var<storage, read> rgba_materials: array<RgbaMaterial>;
@group(1) @binding(3) var<storage, read> canvas_layers: array<CanvasLayer>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) shape_local: vec2<f32>,
    @location(1) flow: f32,
    @location(2) @interpolate(flat) material: u32,
    @location(3) @interpolate(flat) shape: u32,
    @location(4) cap_offset_px: vec2<f32>,
    @location(5) @interpolate(flat) incoming_tangent: vec2<f32>,
    @location(6) @interpolate(flat) outgoing_tangent: vec2<f32>,
    @location(7) @interpolate(flat) cap_neighbors: u32,
    @location(8) radius_px: f32,
    @location(9) document_unit_px: f32,
    @location(10) @interpolate(flat) layer: u32,
}

const SEGMENT_VERTICES: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2(-1.0, 0.0),
    vec2(-1.0, 1.0),
    vec2( 1.0, 1.0),
    vec2(-1.0, 0.0),
    vec2( 1.0, 1.0),
    vec2( 1.0, 0.0),
);

const CAP_VERTICES: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2(-1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2( 1.0, -1.0),
);

// Coverage reaches zero half a document unit beyond the nominal radius. Keep
// an additional physical-pixel guard so triangle rasterization cannot clip the
// transition at unusual DPI scales.
const ANTIALIAS_GUARD_PX: f32 = 1.0;
// Averaged endpoint normals are safe only while a segment remains an
// unambiguous left/right strip. Sharper turns use a bounded round join instead.
const SMOOTH_JOIN_MIN_DOT: f32 = 0.75;

fn world_to_clip(position: vec2<f32>) -> vec4<f32> {
    return view.clip_from_world * vec4(position, 0.0, 1.0);
}

fn clip_to_screen(clip: vec4<f32>, resolution: vec2<f32>) -> vec2<f32> {
    return resolution * (0.5 * clip.xy / clip.w + 0.5);
}

fn screen_to_clip(screen: vec2<f32>, source: vec4<f32>, resolution: vec2<f32>) -> vec4<f32> {
    return vec4(source.w * ((2.0 * screen) / resolution - 1.0), source.z, source.w);
}

fn document_radius_to_screen(position: vec2<f32>, radius: f32, resolution: vec2<f32>) -> f32 {
    let center = world_to_clip(position);
    let offset = world_to_clip(position + vec2(radius, 0.0));
    return distance(clip_to_screen(center, resolution), clip_to_screen(offset, resolution));
}

fn rotate_vector(vector: vec2<f32>, angle: f32) -> vec2<f32> {
    let sine = sin(angle);
    let cosine = cos(angle);
    return vec2(
        cosine * vector.x - sine * vector.y,
        sine * vector.x + cosine * vector.y,
    );
}

fn point_major_axis(point: StrokePoint, resolution: vec2<f32>) -> vec2<f32> {
    var world_axis = point.orientation;
    if length(world_axis) <= 0.0001 {
        world_axis = vec2(0.0, 1.0);
    } else {
        world_axis = normalize(world_axis);
    }
    world_axis = rotate_vector(world_axis, point.twist_radians);
    let center = clip_to_screen(world_to_clip(point.position), resolution);
    let axis_point = clip_to_screen(world_to_clip(point.position + world_axis), resolution);
    let screen_axis = axis_point - center;
    if length(screen_axis) <= 0.0001 {
        return vec2(0.0, 1.0);
    }
    return normalize(screen_axis);
}

fn ellipse_support_radius(
    point: StrokePoint,
    axis: vec2<f32>,
    minor_radius: f32,
    resolution: vec2<f32>,
) -> f32 {
    let major_axis = point_major_axis(point, resolution);
    let minor_axis = vec2(-major_axis.y, major_axis.x);
    let major_radius = minor_radius * max(point.aspect_ratio, 1.0);
    let major_component = dot(axis, major_axis) * major_radius;
    let minor_component = dot(axis, minor_axis) * minor_radius;
    return sqrt(major_component * major_component + minor_component * minor_component);
}

fn segment_tangent(point_a: StrokePoint, point_b: StrokePoint, resolution: vec2<f32>) -> vec2<f32> {
    let screen_a = clip_to_screen(world_to_clip(point_a.position), resolution);
    let screen_b = clip_to_screen(world_to_clip(point_b.position), resolution);
    let delta = screen_b - screen_a;
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

@vertex
fn vertex(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let resolution = view.viewport.zw;

    if vertex_index < 6u {
        let segment = stroke_segments[instance_index];
        let point_a = stroke_points[segment.start];
        let point_b = stroke_points[segment.end];
        let local = SEGMENT_VERTICES[vertex_index];
        let clip_a = world_to_clip(point_a.position);
        let clip_b = world_to_clip(point_b.position);
        let screen_a = clip_to_screen(clip_a, resolution);
        let screen_b = clip_to_screen(clip_b, resolution);
        var tangent = segment_tangent(point_a, point_b, resolution);
        if length(tangent) <= 0.0001 {
            tangent = vec2(1.0, 0.0);
        }
        var start_tangent = tangent;
        var end_tangent = tangent;

        if instance_index > 0u {
            let previous_segment = stroke_segments[instance_index - 1u];
            if previous_segment.end == segment.start {
                let previous_point = stroke_points[previous_segment.start];
                let incoming = segment_tangent(previous_point, point_a, resolution);
                if join_is_smooth(incoming, tangent) {
                    start_tangent = join_tangent(incoming, tangent);
                }
            }
        }
        if instance_index + 1u < arrayLength(&stroke_segments) {
            let next_segment = stroke_segments[instance_index + 1u];
            if next_segment.start == segment.end {
                let next_point = stroke_points[next_segment.end];
                let outgoing = segment_tangent(point_b, next_point, resolution);
                if join_is_smooth(tangent, outgoing) {
                    end_tangent = join_tangent(tangent, outgoing);
                }
            }
        }

        let start_normal = vec2(-start_tangent.y, start_tangent.x);
        let end_normal = vec2(-end_tangent.y, end_tangent.x);
        let normal = mix(start_normal, end_normal, local.y);
        let minor_a = document_radius_to_screen(point_a.position, point_a.half_width, resolution);
        let minor_b = document_radius_to_screen(point_b.position, point_b.half_width, resolution);
        let width_a = ellipse_support_radius(point_a, start_normal, minor_a, resolution);
        let width_b = ellipse_support_radius(point_b, end_normal, minor_b, resolution);
        let width = max(mix(width_a, width_b, local.y), 0.0001);
        let document_unit_a = document_radius_to_screen(point_a.position, 1.0, resolution);
        let document_unit_b = document_radius_to_screen(point_b.position, 1.0, resolution);
        let document_unit_px = max(mix(document_unit_a, document_unit_b, local.y), 0.0001);
        let outer_width = width + 0.5 * document_unit_px + ANTIALIAS_GUARD_PX;
        let source_clip = mix(clip_a, clip_b, local.y);
        let screen = mix(screen_a, screen_b, local.y) + normal * local.x * outer_width;

        return VertexOutput(
            screen_to_clip(screen, source_clip, resolution),
            // Keep the lateral distance in pixels. A normalized distance is
            // nonlinear when the two endpoint radii differ, so interpolating
            // it across the quad's triangles exposes their diagonal.
            vec2(local.x * outer_width, 0.0),
            mix(point_a.flow, point_b.flow, local.y),
            segment.material,
            0u,
            vec2(0.0),
            vec2(0.0),
            vec2(0.0),
            0u,
            width,
            document_unit_px,
            segment.layer,
        );
    }

    // Every segment contributes its end point exactly once. The initial dot is
    // a zero-length segment, so this also covers the stroke's first point
    // without double-depositing both ends of every later segment.
    let segment = stroke_segments[instance_index];
    let point = stroke_points[segment.end];
    let local = CAP_VERTICES[vertex_index - 6u];
    let clip = world_to_clip(point.position);
    let center = clip_to_screen(clip, resolution);
    var incoming_tangent = vec2(0.0);
    var outgoing_tangent = vec2(0.0);
    var cap_neighbors = 0u;

    let point_a = stroke_points[segment.start];
    let incoming_delta = center - clip_to_screen(world_to_clip(point_a.position), resolution);
    let incoming_length = length(incoming_delta);
    if incoming_length > 0.0001 {
        incoming_tangent = incoming_delta / incoming_length;
        cap_neighbors |= 1u;
    }

    if instance_index + 1u < arrayLength(&stroke_segments) {
        let next_segment = stroke_segments[instance_index + 1u];
        if next_segment.start == segment.end {
            let next_point = stroke_points[next_segment.end];
            let outgoing_delta =
                clip_to_screen(world_to_clip(next_point.position), resolution) - center;
            let outgoing_length = length(outgoing_delta);
            if outgoing_length > 0.0001 {
                outgoing_tangent = outgoing_delta / outgoing_length;
                cap_neighbors |= 2u;
            }
        }
    }

    let minor_radius = max(
        document_radius_to_screen(point.position, point.half_width, resolution),
        0.0001,
    );
    let major_radius = minor_radius * max(point.aspect_ratio, 1.0);
    let major_axis = point_major_axis(point, resolution);
    let minor_axis = vec2(-major_axis.y, major_axis.x);
    let document_unit_px = max(document_radius_to_screen(point.position, 1.0, resolution), 0.0001);
    let guard = 0.5 * document_unit_px + ANTIALIAS_GUARD_PX;
    let outer_major = major_radius + guard;
    let outer_minor = minor_radius + guard;
    let cap_offset = major_axis * local.x * outer_major + minor_axis * local.y * outer_minor;
    let screen = center + cap_offset;

    return VertexOutput(
        screen_to_clip(screen, clip, resolution),
        vec2(
            local.x * outer_major / major_radius,
            local.y * outer_minor / minor_radius,
        ),
        point.flow,
        segment.material,
        1u,
        cap_offset,
        incoming_tangent,
        outgoing_tangent,
        cap_neighbors,
        minor_radius,
        document_unit_px,
        segment.layer,
    );
}

// Coverage is geometry-only. Paint models consume this scalar in their own
// deposition entry points, which lets future pigment and wetness pipelines reuse
// the same points, segment metadata, caps, and active-tip bridge.
fn evaluate_coverage(
    shape_local: vec2<f32>,
    shape: u32,
    radius_px: f32,
    document_unit_px: f32,
) -> f32 {
    var signed_distance_px = abs(shape_local.x) - radius_px;
    if shape == 1u {
        signed_distance_px = (length(shape_local) - 1.0) * radius_px;
    }
    return clamp(0.5 - signed_distance_px / document_unit_px, 0.0, 1.0);
}

fn deposit_rgba(material: RgbaMaterial, coverage: f32, flow: f32) -> vec4<f32> {
    let amount = clamp(coverage * flow, 0.0, 1.0);
    return material.color * amount;
}

@fragment
fn fragment_rgba(in: VertexOutput) -> @location(0) vec4<f32> {
    let layer = canvas_layers[in.layer];
    if layer.visible == 0u || layer.opacity <= 0.0 {
        return vec4(0.0);
    }
    let coverage = evaluate_coverage(
        in.shape_local,
        in.shape,
        in.radius_px,
        in.document_unit_px,
    );
    if in.shape == 1u {
        // Internal points are connected by bodies that share the same smoothed
        // cross-section. Drawing a tiny round cap at every raw direction
        // change creates the radial "twig" comb on curved strokes.
        if (in.cap_neighbors & 3u) == 3u
            && join_is_smooth(in.incoming_tangent, in.outgoing_tangent)
        {
            return vec4(0.0);
        }
        let covered_by_incoming_body =
            (in.cap_neighbors & 1u) != 0u
                && dot(in.cap_offset_px, in.incoming_tangent) <= 0.0;
        let covered_by_outgoing_body =
            (in.cap_neighbors & 2u) != 0u
                && dot(in.cap_offset_px, in.outgoing_tangent) >= 0.0;
        if covered_by_incoming_body || covered_by_outgoing_body {
            return vec4(0.0);
        }
    }
    return deposit_rgba(rgba_materials[in.material], coverage, in.flow) * layer.opacity;
}
