#import bevy_render::view::View
#import hamerons_stroke_render::stroke_geometry as geometry

@group(1) @binding(0) var<uniform> view: View;

struct RgbaMaterial {
    color: vec4<f32>,
}

struct CanvasLayer {
    opacity: f32,
    visible: u32,
    padding: vec2<u32>,
}

@group(0) @binding(2) var<storage, read> rgba_materials: array<RgbaMaterial>;
@group(0) @binding(3) var<storage, read> canvas_layers: array<CanvasLayer>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec2<f32>,
    @location(1) @interpolate(flat) shape: u32,
    @location(2) @interpolate(flat) segment_index: u32,
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

// The fragment snaps to the persistent canvas pixel center. One extra
// document unit plus a physical pixel keeps that snapped sample inside the
// conservative body/cap bounds at every zoom level.
const RASTER_GUARD_PX: f32 = 1.0;

fn world_to_clip(position: vec2<f32>) -> vec4<f32> {
    return view.clip_from_world * vec4(position, 0.0, 1.0);
}

fn clip_to_screen(clip: vec4<f32>, resolution: vec2<f32>) -> vec2<f32> {
    return resolution * (0.5 * clip.xy / clip.w + 0.5);
}

fn screen_to_clip(screen: vec2<f32>, source: vec4<f32>, resolution: vec2<f32>) -> vec4<f32> {
    return vec4(source.w * ((2.0 * screen) / resolution - 1.0), source.z, source.w);
}

fn screen_to_world(screen: vec2<f32>, resolution: vec2<f32>) -> vec2<f32> {
    let world = view.world_from_clip * vec4((2.0 * screen) / resolution - 1.0, 0.0, 1.0);
    return world.xy / world.w;
}

fn document_radius_to_screen(position: vec2<f32>, radius: f32, resolution: vec2<f32>) -> f32 {
    let center = world_to_clip(position);
    let offset = world_to_clip(position + vec2(radius, 0.0));
    return distance(clip_to_screen(center, resolution), clip_to_screen(offset, resolution));
}

fn screen_point_major_axis(
    point: geometry::StrokePoint,
    resolution: vec2<f32>,
) -> vec2<f32> {
    let world_axis = geometry::point_major_axis(point);
    let center = clip_to_screen(world_to_clip(point.position), resolution);
    let axis_point = clip_to_screen(world_to_clip(point.position + world_axis), resolution);
    let screen_axis = axis_point - center;
    if length(screen_axis) <= 0.0001 {
        return vec2(0.0, 1.0);
    }
    return normalize(screen_axis);
}

fn screen_ellipse_support_offset(
    point: geometry::StrokePoint,
    axis: vec2<f32>,
    minor_radius: f32,
    resolution: vec2<f32>,
) -> vec2<f32> {
    let major_axis = screen_point_major_axis(point, resolution);
    let minor_axis = vec2(-major_axis.y, major_axis.x);
    let major_radius = minor_radius * max(point.aspect_ratio, 1.0);
    let major_projection = dot(axis, major_axis);
    let minor_projection = dot(axis, minor_axis);
    let support_radius = sqrt(
        major_projection * major_projection * major_radius * major_radius
            + minor_projection * minor_projection * minor_radius * minor_radius,
    );
    if support_radius <= 0.0001 {
        return vec2(0.0);
    }
    // The support point of a rotated ellipse is generally not `axis` times
    // its support radius; it also has a tangent component. Returning the real
    // support point keeps the body edge attached to the ellipse cap.
    return (
        major_axis * (major_projection * major_radius * major_radius)
            + minor_axis * (minor_projection * minor_radius * minor_radius)
    ) / support_radius;
}

fn screen_segment_tangent(
    point_a: geometry::StrokePoint,
    point_b: geometry::StrokePoint,
    resolution: vec2<f32>,
) -> vec2<f32> {
    let screen_a = clip_to_screen(world_to_clip(point_a.position), resolution);
    let screen_b = clip_to_screen(world_to_clip(point_b.position), resolution);
    let delta = screen_b - screen_a;
    if length(delta) <= 0.0001 {
        return vec2(0.0);
    }
    return normalize(delta);
}

@vertex
fn vertex(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let resolution = view.viewport.zw;
    let segment = geometry::stroke_segments[instance_index];

    if vertex_index < 6u {
        let point_a = geometry::stroke_points[segment.start];
        let point_b = geometry::stroke_points[segment.end];
        let local = SEGMENT_VERTICES[vertex_index];
        let clip_a = world_to_clip(point_a.position);
        let clip_b = world_to_clip(point_b.position);
        let screen_a = clip_to_screen(clip_a, resolution);
        let screen_b = clip_to_screen(clip_b, resolution);
        var tangent = screen_segment_tangent(point_a, point_b, resolution);
        if length(tangent) <= 0.0001 {
            tangent = vec2(1.0, 0.0);
        }
        let normal = vec2(-tangent.y, tangent.x);
        let side_axis = normal * local.x;
        let minor_a = document_radius_to_screen(point_a.position, point_a.half_width, resolution);
        let minor_b = document_radius_to_screen(point_b.position, point_b.half_width, resolution);
        let support_a = screen_ellipse_support_offset(point_a, side_axis, minor_a, resolution);
        let support_b = screen_ellipse_support_offset(point_b, side_axis, minor_b, resolution);
        let support_offset = mix(support_a, support_b, local.y);
        let document_unit_a = document_radius_to_screen(point_a.position, 1.0, resolution);
        let document_unit_b = document_radius_to_screen(point_b.position, 1.0, resolution);
        let document_unit_px = max(mix(document_unit_a, document_unit_b, local.y), 0.0001);
        let guard = document_unit_px + RASTER_GUARD_PX;
        let source_clip = mix(clip_a, clip_b, local.y);
        let screen = mix(screen_a, screen_b, local.y)
            + support_offset
            + side_axis * guard;

        return VertexOutput(
            screen_to_clip(screen, source_clip, resolution),
            screen_to_world(screen, resolution),
            0u,
            instance_index,
        );
    }

    // Every segment contributes its end point exactly once. The initial dot is
    // a zero-length segment, so this also covers the stroke's first point.
    let point = geometry::stroke_points[segment.end];
    let local = CAP_VERTICES[vertex_index - 6u];
    let clip = world_to_clip(point.position);
    let center = clip_to_screen(clip, resolution);
    let minor_radius = max(
        document_radius_to_screen(point.position, point.half_width, resolution),
        0.0001,
    );
    let major_radius = minor_radius * max(point.aspect_ratio, 1.0);
    let major_axis = screen_point_major_axis(point, resolution);
    let minor_axis = vec2(-major_axis.y, major_axis.x);
    let document_unit_px = max(document_radius_to_screen(point.position, 1.0, resolution), 0.0001);
    let guard = document_unit_px + RASTER_GUARD_PX;
    let outer_major = major_radius + guard;
    let outer_minor = minor_radius + guard;
    let screen = center
        + major_axis * local.x * outer_major
        + minor_axis * local.y * outer_minor;

    return VertexOutput(
        screen_to_clip(screen, clip, resolution),
        screen_to_world(screen, resolution),
        1u,
        instance_index,
    );
}

fn primitive_sample(in: VertexOutput) -> geometry::StrokeSample {
    let document_pixel = floor(in.world_position) + vec2(0.5);
    if in.shape == 1u {
        return geometry::segment_exposed_cap_sample(document_pixel, in.segment_index);
    }
    return geometry::segment_body_sample(document_pixel, in.segment_index);
}

@fragment
fn fragment_amount(in: VertexOutput) -> @location(0) vec4<f32> {
    let segment = geometry::stroke_segments[in.segment_index];
    let layer = canvas_layers[segment.layer];
    if layer.visible == 0u || layer.opacity <= 0.0 {
        discard;
    }

    // Persistent tiles own integer document pixels. Evaluating the live mask
    // at that same center makes the stroke invariant across the handoff.
    let sample = primitive_sample(in);
    if sample.coverage <= 0.0 {
        discard;
    }
    // MAX blending unions premultiplied deposition directly. A later weak
    // sample can never punch a bright hole into an earlier strong overlap.
    // Layer opacity belongs to the resolved layer after stroke composition.
    return rgba_materials[segment.material].color
        * geometry::stroke_sample_amount(sample);
}
