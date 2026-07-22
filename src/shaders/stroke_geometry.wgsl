#define_import_path hamerons_stroke_render::stroke_geometry

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

struct StrokeSample {
    coverage: f32,
    flow: f32,
}

const INVALID_STROKE_DISTANCE: f32 = 1.0e30;

@group(0) @binding(0) var<storage, read> stroke_points: array<StrokePoint>;
@group(0) @binding(1) var<storage, read> stroke_segments: array<StrokeSegment>;

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

fn empty_stroke_sample() -> StrokeSample {
    return StrokeSample(
        0.0,
        0.0,
    );
}

fn ellipse_sample(pixel: vec2<f32>, point: StrokePoint) -> StrokeSample {
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
    return StrokeSample(
        coverage,
        clamp(point.flow, 0.0, 1.0),
    );
}

fn stroke_sample_amount(sample: StrokeSample) -> f32 {
    return sample.coverage * sample.flow;
}

fn interpolate_ellipse_axis(
    start_axis_input: vec2<f32>,
    end_axis_input: vec2<f32>,
    amount: f32,
) -> vec2<f32> {
    var from_axis = start_axis_input;
    if length(from_axis) <= 0.0001 {
        from_axis = vec2(0.0, 1.0);
    } else {
        from_axis = normalize(from_axis);
    }
    var to_axis = end_axis_input;
    if length(to_axis) <= 0.0001 {
        to_axis = from_axis;
    } else {
        to_axis = normalize(to_axis);
    }
    // An ellipse axis is undirected. Align the two representations before
    // interpolation so a half-turn does not collapse through a zero vector.
    if dot(from_axis, to_axis) < 0.0 {
        to_axis = -to_axis;
    }
    let interpolated = mix(from_axis, to_axis, amount);
    if length(interpolated) <= 0.0001 {
        return from_axis;
    }
    return normalize(interpolated);
}

fn interpolate_stroke_point(
    start_point: StrokePoint,
    end_point: StrokePoint,
    amount: f32,
) -> StrokePoint {
    // Only the final undirected major axis affects an ellipse. Interpolating
    // that axis directly also handles orientation and twist crossing their
    // wrap points without introducing a transient flip.
    let major_axis = interpolate_ellipse_axis(
        point_major_axis(start_point),
        point_major_axis(end_point),
        amount,
    );
    return StrokePoint(
        mix(start_point.position, end_point.position, amount),
        mix(start_point.half_width, end_point.half_width, amount),
        mix(start_point.flow, end_point.flow, amount),
        major_axis,
        0.0,
        mix(start_point.aspect_ratio, end_point.aspect_ratio, amount),
    );
}

fn ellipse_metric_projection(
    pixel: vec2<f32>,
    origin: vec2<f32>,
    delta: vec2<f32>,
    point: StrokePoint,
) -> f32 {
    let minor_radius = max(point.half_width, 0.0001);
    let major_radius = minor_radius * max(point.aspect_ratio, 1.0);
    let major_axis = point_major_axis(point);
    let minor_axis = vec2(-major_axis.y, major_axis.x);
    let offset = pixel - origin;
    let metric_offset = vec2(
        dot(offset, major_axis) / major_radius,
        dot(offset, minor_axis) / minor_radius,
    );
    let metric_delta = vec2(
        dot(delta, major_axis) / major_radius,
        dot(delta, minor_axis) / minor_radius,
    );
    let denominator = dot(metric_delta, metric_delta);
    if denominator <= 0.000001 {
        return 0.0;
    }
    return dot(metric_offset, metric_delta) / denominator;
}

fn segment_body_projection(pixel: vec2<f32>, segment_index: u32) -> f32 {
    let segment = stroke_segments[segment_index];
    let point_a = stroke_points[segment.start];
    let point_b = stroke_points[segment.end];
    let delta = point_b.position - point_a.position;
    let length_squared = dot(delta, delta);
    if length_squared <= 0.000001 {
        return INVALID_STROKE_DISTANCE;
    }

    var amount = clamp(dot(pixel - point_a.position, delta) / length_squared, 0.0, 1.0);
    for (var iteration = 0u; iteration < 2u; iteration += 1u) {
        let point = interpolate_stroke_point(point_a, point_b, clamp(amount, 0.0, 1.0));
        amount = ellipse_metric_projection(pixel, point_a.position, delta, point);
    }
    return amount;
}

fn segment_body_sample(pixel: vec2<f32>, segment_index: u32) -> StrokeSample {
    let segment = stroke_segments[segment_index];
    let amount = segment_body_projection(pixel, segment_index);
    if amount < 0.0 || amount > 1.0 {
        return empty_stroke_sample();
    }
    let point_a = stroke_points[segment.start];
    let point_b = stroke_points[segment.end];
    return ellipse_sample(pixel, interpolate_stroke_point(point_a, point_b, amount));
}

fn segment_cap_sample(pixel: vec2<f32>, segment_index: u32) -> StrokeSample {
    let segment = stroke_segments[segment_index];
    return ellipse_sample(pixel, stroke_points[segment.end]);
}

fn segment_exposed_cap_sample(pixel: vec2<f32>, segment_index: u32) -> StrokeSample {
    let segment = stroke_segments[segment_index];
    let cap_sample = segment_cap_sample(pixel, segment_index);
    if cap_sample.coverage <= 0.0 {
        return empty_stroke_sample();
    }

    // A cap owns only the angular region outside its incoming and outgoing
    // centerline half-planes. This leaves a rear cap at stroke start, the outer
    // wedge at a turn, and a forward cap at stroke end. It never leaves a full
    // internal ellipse behind to stamp a pressure value across later segments.
    let current_start = stroke_points[segment.start].position;
    let current_end = stroke_points[segment.end].position;
    if distance(current_start, current_end) > 0.0001
        && segment_body_projection(pixel, segment_index) <= 1.0
    {
        return empty_stroke_sample();
    }
    let next_index = segment_index + 1u;
    if next_index < arrayLength(&stroke_segments) {
        let next_segment = stroke_segments[next_index];
        if next_segment.start == segment.end {
            let next_start = stroke_points[next_segment.start].position;
            let next_end = stroke_points[next_segment.end].position;
            if distance(next_start, next_end) > 0.0001
                && segment_body_projection(pixel, next_index) >= 0.0
            {
                return empty_stroke_sample();
            }
        }
    }
    return cap_sample;
}
