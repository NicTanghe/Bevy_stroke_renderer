use std::{collections::HashMap, time::Instant};

use bevy_camera::{Camera, Camera2d, RenderTarget};
use bevy_ecs::{
    entity::Entity,
    message::MessageReader,
    query::With,
    resource::Resource,
    schedule::SystemSet,
    system::{Query, Res, ResMut},
};
use bevy_input::{
    pen::{PenAction, PenButton, PenData, PenId, PenInput, PenPressure, PenToolKind},
    ButtonState,
};
use bevy_math::{ops, Rect, Vec2};
use bevy_transform::components::GlobalTransform;
use bevy_window::{PrimaryWindow, Window, WindowRef};

use crate::{
    BrushProfile, BrushSizeSpace, StrokeDeltaBatch, StrokeId, StrokePoint, StrokePointResampler,
    StrokeRendererSettings, StrokeStore, StrokeTelemetry,
};

#[derive(Resource, Default)]
pub(crate) struct PenContacts(HashMap<PenId, ActivePenContact>);

/// Logical viewport regions that must not begin or extend canvas strokes.
#[derive(Resource, Default)]
pub struct StrokeInputBlocker {
    regions: Vec<Rect>,
}

impl StrokeInputBlocker {
    /// Replaces the blocked viewport regions for the current frame.
    pub fn set_regions(&mut self, regions: impl IntoIterator<Item = Rect>) {
        self.regions.clear();
        self.regions.extend(regions);
    }

    /// Returns whether a logical viewport point belongs to application chrome.
    pub fn blocks(&self, point: Vec2) -> bool {
        self.regions.iter().any(|region| region.contains(point))
    }
}

/// Ordering label for application chrome that must block pen collection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, SystemSet)]
pub enum StrokeInputSystems {
    /// Converts pen input into document geometry.
    Collect,
}

#[derive(Clone, Copy)]
struct ActivePenContact {
    stroke: StrokeId,
    generation: u64,
    resampler: StrokePointResampler,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn collect_pen_strokes(
    mut messages: MessageReader<PenInput>,
    settings: Res<StrokeRendererSettings>,
    cameras: Query<(&Camera, &RenderTarget, &GlobalTransform), With<Camera2d>>,
    windows: Query<&Window>,
    primary_windows: Query<Entity, With<PrimaryWindow>>,
    mut contacts: ResMut<PenContacts>,
    blockers: Res<StrokeInputBlocker>,
    mut store: ResMut<StrokeStore>,
    mut batch: ResMut<StrokeDeltaBatch>,
    telemetry: Res<StrokeTelemetry>,
) {
    let started = Instant::now();
    let mut message_count = 0usize;
    batch.begin_frame();
    store.begin_input_batch();
    let primary_window = primary_windows.iter().next();

    for message in messages.read() {
        message_count += 1;
        match &message.action {
            PenAction::Button {
                button: PenButton::Contact,
                state: ButtonState::Pressed,
                data,
            } => {
                if let Some(mut previous) = contacts.0.remove(&message.pen.device) {
                    finish_contact(&mut previous, &mut store, &mut batch);
                    if let Some(delta) = store.end_stroke(previous.stroke) {
                        batch.push(delta);
                    }
                }

                if message
                    .pen
                    .position
                    .is_some_and(|position| blockers.blocks(position))
                {
                    continue;
                }

                let profile = profile_for_tool(&settings, message.pen.tool);
                if let Some(point) =
                    make_point(message, data, profile, &cameras, &windows, primary_window)
                {
                    let (stroke, delta) = store.begin_stroke(point, profile);
                    contacts.0.insert(
                        message.pen.device,
                        ActivePenContact {
                            stroke,
                            generation: store.generation(),
                            resampler: StrokePointResampler::new(point),
                        },
                    );
                    batch.push(delta);
                }
            }
            PenAction::Moved(data) => {
                if message
                    .pen
                    .position
                    .is_some_and(|position| blockers.blocks(position))
                {
                    if let Some(mut contact) = contacts.0.remove(&message.pen.device)
                        && contact.generation == store.generation()
                    {
                        finish_contact(&mut contact, &mut store, &mut batch);
                        if let Some(delta) = store.end_stroke(contact.stroke) {
                            batch.push(delta);
                        }
                    }
                    continue;
                }
                let Some(contact) = contacts.0.get_mut(&message.pen.device) else {
                    continue;
                };
                let profile = profile_for_tool(&settings, message.pen.tool);
                if let Some(point) =
                    make_point(message, data, profile, &cameras, &windows, primary_window)
                {
                    if contact.generation == store.generation() {
                        let stroke = contact.stroke;
                        contact.resampler.push(point, |point| {
                            append_sample(stroke, point, &mut store, &mut batch);
                        });
                    } else {
                        let (stroke, delta) = store.begin_stroke(point, profile);
                        *contact = ActivePenContact {
                            stroke,
                            generation: store.generation(),
                            resampler: StrokePointResampler::new(point),
                        };
                        batch.push(delta);
                    }
                }
            }
            PenAction::Button {
                button: PenButton::Contact,
                state: ButtonState::Released,
                data,
            } => {
                let Some(mut contact) = contacts.0.remove(&message.pen.device) else {
                    continue;
                };
                if contact.generation != store.generation() {
                    continue;
                }
                let profile = profile_for_tool(&settings, message.pen.tool);
                if !message
                    .pen
                    .position
                    .is_some_and(|position| blockers.blocks(position))
                    && let Some(point) =
                        make_point(message, data, profile, &cameras, &windows, primary_window)
                    && let release_point =
                        release_tip_point(point.position, contact.resampler.latest_input())
                {
                    let stroke = contact.stroke;
                    contact.resampler.push(release_point, |point| {
                        append_sample(stroke, point, &mut store, &mut batch);
                    });
                }
                finish_contact(&mut contact, &mut store, &mut batch);
                if let Some(delta) = store.end_stroke(contact.stroke) {
                    batch.push(delta);
                }
            }
            PenAction::Left => {
                if let Some(mut contact) = contacts.0.remove(&message.pen.device)
                    && contact.generation == store.generation()
                {
                    finish_contact(&mut contact, &mut store, &mut batch);
                    if let Some(delta) = store.end_stroke(contact.stroke) {
                        batch.push(delta);
                    }
                }
            }
            PenAction::Entered
            | PenAction::Button {
                button: PenButton::Barrel | PenButton::Other(_),
                ..
            } => {}
        }
    }

    telemetry.record_input_batch(
        message_count,
        started.elapsed(),
        store.points().len(),
        store.segments().len(),
    );
}

fn release_tip_point(position: Vec2, last_contact_point: StrokePoint) -> StrokePoint {
    StrokePoint {
        position,
        ..last_contact_point
    }
}

fn append_sample(
    stroke: StrokeId,
    point: StrokePoint,
    store: &mut StrokeStore,
    batch: &mut StrokeDeltaBatch,
) {
    if let Some(delta) = store.append_point(stroke, point) {
        batch.push(delta);
    }
}

fn finish_contact(
    contact: &mut ActivePenContact,
    store: &mut StrokeStore,
    batch: &mut StrokeDeltaBatch,
) {
    let stroke = contact.stroke;
    contact.resampler.finish(|point| {
        append_sample(stroke, point, store, batch);
    });
}

fn profile_for_tool(settings: &StrokeRendererSettings, tool: PenToolKind) -> BrushProfile {
    if tool == PenToolKind::Eraser {
        settings.eraser
    } else {
        settings.pen
    }
}

fn make_point(
    message: &PenInput,
    data: &PenData,
    profile: BrushProfile,
    cameras: &Query<(&Camera, &RenderTarget, &GlobalTransform), With<Camera2d>>,
    windows: &Query<&Window>,
    primary_window: Option<Entity>,
) -> Option<StrokePoint> {
    let viewport_position = message.pen.position?;
    let (position, document_units_per_logical_pixel) = cameras
        .iter()
        .filter(|(camera, target, _)| {
            camera.is_active && camera_targets_window(target, message.pen.window, primary_window)
        })
        .find_map(|(camera, _, transform)| {
            let position = camera
                .viewport_to_world_2d(transform, viewport_position)
                .ok()?;
            let neighbor = camera
                .viewport_to_world_2d(transform, viewport_position + Vec2::X)
                .or_else(|_| camera.viewport_to_world_2d(transform, viewport_position - Vec2::X))
                .ok()?;
            Some((position, position.distance(neighbor).max(f32::EPSILON)))
        })?;
    let scale_factor = windows
        .get(message.pen.window)
        .map_or(1.0, Window::scale_factor);
    let pressure = match data.pressure {
        Some(PenPressure::Normalized(value)) => value as f32,
        Some(PenPressure::Calibrated { .. }) | None => 1.0,
    };

    let tilt_degrees = brush_tilt_degrees(data);
    let footprint = profile.footprint(pressure, tilt_degrees);
    let document_scale = match profile.size_space {
        BrushSizeSpace::Document => 1.0,
        BrushSizeSpace::Screen => document_units_per_logical_pixel / scale_factor.max(0.01),
    };

    Some(StrokePoint {
        position,
        half_width: footprint.half_size.y * document_scale,
        aspect_ratio: footprint.half_size.x / footprint.half_size.y,
        flow: footprint.flow,
        orientation: brush_orientation(data),
        twist_radians: brush_twist_radians(data),
    })
}

fn brush_tilt_degrees(data: &PenData) -> f32 {
    if let Some(angle) = data.angle {
        return (std::f64::consts::FRAC_PI_2 - angle.altitude).to_degrees() as f32;
    }
    data.tilt.map_or(0.0, |tilt| {
        tilt_surface_projection(tilt.x, tilt.y)
            .length()
            .atan()
            .to_degrees()
    })
}

fn camera_targets_window(
    target: &RenderTarget,
    window: Entity,
    primary_window: Option<Entity>,
) -> bool {
    match target {
        RenderTarget::Window(WindowRef::Primary) => primary_window == Some(window),
        RenderTarget::Window(WindowRef::Entity(entity)) => *entity == window,
        _ => false,
    }
}

fn brush_orientation(data: &PenData) -> Vec2 {
    if let Some(angle) = data.angle {
        let azimuth = angle.azimuth as f32;
        let (sin, cos) = ops::sin_cos(azimuth);
        // Winit azimuth increases clockwise in screen coordinates. Bevy's
        // document space has positive Y upward, so screen Y must be mirrored.
        return Vec2::new(cos, -sin);
    }
    if let Some(tilt) = data.tilt {
        return tilt_surface_projection(tilt.x, tilt.y).normalize_or(Vec2::Y);
    }
    Vec2::Y
}

fn tilt_surface_projection(x_degrees: i8, y_degrees: i8) -> Vec2 {
    // Winit reports two plane angles, not the components of one angle vector.
    // Their tangents are the projected tool-axis components on the tablet.
    // Positive tablet Y is down-screen, while positive document Y is upward.
    Vec2::new(
        (x_degrees as f32).to_radians().tan(),
        -(y_degrees as f32).to_radians().tan(),
    )
}

fn brush_twist_radians(data: &PenData) -> f32 {
    // Winit twist is clockwise; positive mathematical rotation in the stroke
    // shaders is counter-clockwise in Bevy world coordinates.
    data.twist
        .map_or(0.0, |degrees| -(degrees as f32).to_radians())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_input::pen::{PenAngle, PenTilt};

    #[test]
    fn release_position_does_not_import_zero_pressure_footprint() {
        let last_contact_point = StrokePoint {
            position: Vec2::new(10.0, 20.0),
            half_width: 4.5,
            aspect_ratio: 1.8,
            flow: 0.73,
            orientation: Vec2::new(0.6, 0.8),
            twist_radians: 0.4,
        };
        let released = release_tip_point(Vec2::new(12.0, 23.0), last_contact_point);

        assert_eq!(released.position, Vec2::new(12.0, 23.0));
        assert_eq!(released.half_width, last_contact_point.half_width);
        assert_eq!(released.aspect_ratio, last_contact_point.aspect_ratio);
        assert_eq!(released.flow, last_contact_point.flow);
        assert_eq!(released.orientation, last_contact_point.orientation);
        assert_eq!(released.twist_radians, last_contact_point.twist_radians);
    }

    #[test]
    fn pen_angles_are_converted_from_screen_to_world_coordinates() {
        let from_tilt = PenData {
            tilt: Some(PenTilt { x: 30, y: 40 }),
            ..Default::default()
        };
        let projected = Vec2::new(30.0_f32.to_radians().tan(), -40.0_f32.to_radians().tan());
        assert!(brush_orientation(&from_tilt).abs_diff_eq(projected.normalize_or_zero(), 0.0001));
        assert!(
            (brush_tilt_degrees(&from_tilt) - projected.length().atan().to_degrees()).abs()
                < 0.0001
        );

        let from_angle = PenData {
            angle: Some(PenAngle {
                altitude: core::f64::consts::FRAC_PI_4,
                azimuth: core::f64::consts::FRAC_PI_2,
            }),
            twist: Some(90),
            ..Default::default()
        };
        assert!(brush_orientation(&from_angle).abs_diff_eq(Vec2::NEG_Y, 0.0001));
        assert!((brush_twist_radians(&from_angle) + core::f32::consts::FRAC_PI_2).abs() < 0.0001);
    }
}
