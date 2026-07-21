use alloc::collections::BTreeSet;
use std::{collections::HashMap, time::Instant};

use bevy_ecs::resource::Resource;
use bevy_math::{ops, Rect, UVec2, Vec2};

use crate::effects::EffectGraph;
use crate::tiles::{TileInvalidation, TileInvalidationCause, TileKey};

mod persistence;
pub use persistence::{
    CheckpointRequest, DocumentCheckpointManager, DocumentCompatibilityIssue, DocumentIoError,
    DocumentSaveReport, LoadedStrokeDocument, DOCUMENT_SCHEMA_VERSION,
};

/// Stable identity of a stroke for the lifetime of a [`StrokeDocument`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StrokeId(pub u64);

/// Identifies a registered paint model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PaintModelId(pub u32);

/// Identifies a material owned by a paint model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PaintMaterialId(pub u32);

/// A paint model and one of its immutable material records.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PaintMaterialRef {
    /// Model that interprets the material payload.
    pub model: PaintModelId,
    /// Paint-model implementation version required to replay the material.
    pub model_version: u32,
    /// Material within that model.
    pub material: PaintMaterialId,
}

/// Versioned, model-owned payload retained by the authoritative document.
#[derive(Clone, Debug, PartialEq)]
pub enum PaintMaterialPayload {
    /// Built-in premultiplied linear RGBA recipe.
    PremultipliedLinearRgba([f32; 4]),
    /// Opaque pigment/model data with a persisted schema version.
    ModelData {
        /// Payload schema interpreted by the selected model version.
        schema_version: u32,
        /// Immutable coefficients, pigment identities, or recipe data.
        bytes: Vec<u8>,
    },
}

/// Immutable paint recipe keyed by model, implementation version, and material id.
#[derive(Clone, Debug, PartialEq)]
pub struct PaintMaterialRecord {
    /// Stable reference stored by stroke metadata.
    pub reference: PaintMaterialRef,
    /// Native model payload required for deterministic regeneration.
    pub payload: PaintMaterialPayload,
}

/// Document-owned library of immutable color and future pigment recipes.
#[derive(Clone, Debug, Default)]
pub struct PaintMaterialLibrary {
    records: HashMap<PaintMaterialRef, PaintMaterialRecord>,
}

impl PaintMaterialLibrary {
    /// Adds a record, rejecting an attempt to mutate an existing identity.
    pub fn add(&mut self, record: PaintMaterialRecord) -> Result<(), &'static str> {
        if let Some(existing) = self.records.get(&record.reference) {
            return (existing == &record)
                .then_some(())
                .ok_or("paint material identities are immutable");
        }
        self.records.insert(record.reference, record);
        Ok(())
    }

    /// Retrieves a versioned material recipe.
    pub fn get(&self, reference: PaintMaterialRef) -> Option<&PaintMaterialRecord> {
        self.records.get(&reference)
    }

    /// Iterates all recipes without exposing mutation of existing records.
    pub fn iter(&self) -> impl Iterator<Item = &PaintMaterialRecord> {
        self.records.values()
    }
}

/// The built-in premultiplied-linear RGBA model identifier.
pub const RGBA_PAINT_MODEL_ID: PaintModelId = PaintModelId(0);

/// Ordering guarantee required when replaying depositions into a material surface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DepositionOrdering {
    /// Preserve document order because existing material state may affect the result.
    #[default]
    Strict,
    /// Compatible depositions may be regrouped when a model guarantees equivalence.
    CommutativeAssociative,
}

/// Operation a paint model applies beneath paint-model-independent coverage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum DepositionMode {
    /// Add the selected paint material.
    #[default]
    Normal = 0,
    /// Remove material according to the selected paint model.
    Erase = 1,
}

/// Storage representation requested by a paint-model surface plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaintPlaneFormat {
    /// Four normalized 8-bit channels. Display conversion happens after effects.
    Rgba8Unorm,
    /// Four half-float channels for high-dynamic-range paint state.
    Rgba16Float,
    /// One 32-bit floating-point channel, suitable for masks or wetness.
    R32Float,
    /// A model-defined packed representation with an explicit byte cost.
    Custom {
        /// Explicit packed cost used by the common allocator.
        bytes_per_pixel: u8,
    },
}

impl PaintPlaneFormat {
    /// Number of persistent bytes required for one pixel of this plane.
    pub const fn bytes_per_pixel(self) -> u64 {
        match self {
            Self::Rgba8Unorm | Self::R32Float => 4,
            Self::Rgba16Float => 8,
            Self::Custom { bytes_per_pixel } => bytes_per_pixel as u64,
        }
    }
}

/// Deterministic value used when allocating or regenerating a plane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PaintPlaneClearValue {
    /// All stored channels begin at zero.
    #[default]
    Zero,
    /// Packed normalized RGBA clear value.
    Rgba8([u8; 4]),
}

/// Describes one native surface plane owned by a paint model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaintPlaneDescriptor {
    /// Stable shader-facing plane name.
    pub name: &'static str,
    /// Human-readable meaning of the stored values.
    pub semantic: &'static str,
    /// Native storage format used for allocation and memory accounting.
    pub format: PaintPlaneFormat,
    /// Value restored before deterministic tile replay.
    pub clear: PaintPlaneClearValue,
}

/// Describes a paint model without coupling stroke geometry to its channels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaintModelDescriptor {
    /// Stable model identifier stored in stroke metadata.
    pub id: PaintModelId,
    /// Version persisted beside paint recipes and document commands.
    pub version: u32,
    /// Human-readable model name.
    pub name: &'static str,
    /// Native planes required by deposition, effects, and display resolve.
    pub persistent_planes: Vec<PaintPlaneDescriptor>,
    /// Temporary planes needed by model-local mixing or display resolve.
    pub scratch_planes: Vec<PaintPlaneDescriptor>,
    /// Whether the scheduler may reorder compatible deposition work.
    pub deposition_ordering: DepositionOrdering,
}

/// Registry boundary used by RGBA now and multi-plane pigment models later.
#[derive(Resource, Default)]
pub struct PaintModelRegistry {
    models: HashMap<PaintModelId, PaintModelDescriptor>,
}

impl PaintModelRegistry {
    /// Registers or replaces a model descriptor.
    pub fn register(&mut self, descriptor: PaintModelDescriptor) {
        self.models.insert(descriptor.id, descriptor);
    }

    /// Returns the descriptor for `id`.
    pub fn get(&self, id: PaintModelId) -> Option<&PaintModelDescriptor> {
        self.models.get(&id)
    }

    /// Iterates over registered descriptors.
    pub fn iter(&self) -> impl Iterator<Item = &PaintModelDescriptor> {
        self.models.values()
    }
}

/// Stable identity of a non-destructive canvas effect implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EffectId(pub u32);

/// Surface domain in which an effect operates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EffectDomain {
    /// Native material planes declared by the selected paint model.
    MaterialSurface,
    /// Premultiplied linear RGBA after paint-model display resolve.
    LinearDisplayRgba,
}

/// Spatial dependency used by tile invalidation and halo allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EffectInfluence {
    /// Each output pixel reads a bounded neighborhood.
    Local {
        /// Maximum read radius in physical pixels.
        radius_px: u32,
    },
    /// Output may depend on the complete surface.
    Global,
}

/// Descriptor consumed by the effect graph without knowing paint channels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectDescriptor {
    /// Stable effect implementation identifier.
    pub id: EffectId,
    /// Version persisted with effect parameters.
    pub implementation_version: u32,
    /// Native material or resolved display-color domain.
    pub domain: EffectDomain,
    /// Tile dependency expansion required by the effect.
    pub influence: EffectInfluence,
    /// Scratch planes leased while this effect executes.
    pub scratch_planes: Vec<PaintPlaneDescriptor>,
}

/// Registry boundary for pass-through, blur, drying, and later material effects.
#[derive(Resource, Default)]
pub struct EffectRegistry {
    effects: HashMap<EffectId, EffectDescriptor>,
}

impl EffectRegistry {
    /// Registers or replaces an effect implementation descriptor.
    pub fn register(&mut self, descriptor: EffectDescriptor) {
        self.effects.insert(descriptor.id, descriptor);
    }

    /// Returns the descriptor for `id`.
    pub fn get(&self, id: EffectId) -> Option<&EffectDescriptor> {
        self.effects.get(&id)
    }

    /// Iterates over registered effect implementations.
    pub fn iter(&self) -> impl Iterator<Item = &EffectDescriptor> {
        self.effects.values()
    }
}

/// Premultiplied linear RGBA material consumed by live and cached deposition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RgbaMaterial {
    /// Premultiplied linear RGBA channels.
    pub premultiplied_linear_rgba: [f32; 4],
}

impl RgbaMaterial {
    /// Creates a material from straight linear RGBA channels.
    pub fn from_linear_rgba(rgba: [f32; 4]) -> Self {
        let alpha = rgba[3].clamp(0.0, 1.0);
        Self {
            premultiplied_linear_rgba: [rgba[0] * alpha, rgba[1] * alpha, rgba[2] * alpha, alpha],
        }
    }
}

/// The first production paint model: premultiplied linear RGBA deposition.
#[derive(Resource, Clone)]
pub struct RgbaPaintModel {
    materials: Vec<RgbaMaterial>,
    revision: u64,
}

impl Default for RgbaPaintModel {
    fn default() -> Self {
        Self {
            materials: vec![
                RgbaMaterial::from_linear_rgba([0.015, 0.02, 0.03, 1.0]),
                RgbaMaterial::from_linear_rgba([1.0, 1.0, 1.0, 1.0]),
            ],
            revision: 1,
        }
    }
}

impl RgbaPaintModel {
    /// Version of the RGBA deposition and material contract.
    pub const MODEL_VERSION: u32 = 1;
    /// Default dark ink material.
    pub const DEFAULT_INK: PaintMaterialId = PaintMaterialId(0);
    /// Opaque removal strength used by the eraser preset.
    pub const DEFAULT_ERASER: PaintMaterialId = PaintMaterialId(1);

    /// Returns all material records.
    pub fn materials(&self) -> &[RgbaMaterial] {
        &self.materials
    }

    /// Returns the monotonically increasing material-table revision.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Appends an immutable material record and returns its identifier.
    pub fn add_material(&mut self, material: RgbaMaterial) -> PaintMaterialId {
        let id = PaintMaterialId(self.materials.len() as u32);
        self.materials.push(material);
        self.revision = self.revision.wrapping_add(1);
        id
    }

    pub(crate) fn sync_from_document(&mut self, library: &PaintMaterialLibrary) {
        let mut records: Vec<_> = library
            .iter()
            .filter(|record| {
                record.reference.model == RGBA_PAINT_MODEL_ID
                    && record.reference.model_version == Self::MODEL_VERSION
            })
            .filter_map(|record| match &record.payload {
                PaintMaterialPayload::PremultipliedLinearRgba(rgba) => Some((
                    record.reference.material.0 as usize,
                    RgbaMaterial {
                        premultiplied_linear_rgba: *rgba,
                    },
                )),
                PaintMaterialPayload::ModelData { .. } => None,
            })
            .collect();
        records.sort_by_key(|(id, _)| *id);
        let Some(required) = records.last().map(|(id, _)| id + 1) else {
            return;
        };
        let mut changed = false;
        if self.materials.len() < required {
            self.materials.resize(
                required,
                RgbaMaterial {
                    premultiplied_linear_rgba: [0.0; 4],
                },
            );
            changed = true;
        }
        for (id, material) in records {
            if self.materials[id] != material {
                self.materials[id] = material;
                changed = true;
            }
        }
        if changed {
            self.revision = self.revision.wrapping_add(1);
        }
    }

    pub(crate) fn descriptor() -> PaintModelDescriptor {
        PaintModelDescriptor {
            id: RGBA_PAINT_MODEL_ID,
            version: Self::MODEL_VERSION,
            name: "Premultiplied linear RGBA",
            persistent_planes: vec![PaintPlaneDescriptor {
                name: "rgba",
                semantic: "premultiplied linear display color and opacity",
                format: PaintPlaneFormat::Rgba8Unorm,
                clear: PaintPlaneClearValue::Zero,
            }],
            scratch_planes: Vec::new(),
            deposition_ordering: DepositionOrdering::Strict,
        }
    }
}

/// Pressure-sensitive brush configuration used while collecting points.
#[derive(Clone, Copy, Debug)]
pub struct BrushProfile {
    /// Paint model and material stored in segment metadata.
    pub paint: PaintMaterialRef,
    /// Normal paint or model-specific material removal.
    pub deposition: DepositionMode,
    /// Full-pressure diameter in the selected size space.
    pub diameter: f32,
    /// Coordinate space in which `diameter` is interpreted.
    pub size_space: BrushSizeSpace,
    /// Minimum pressure-to-diameter ratio.
    pub minimum_diameter_ratio: f32,
    /// Exponent applied to normalized pressure.
    pub pressure_gamma: f32,
    /// Per-sample deposition strength.
    pub flow: f32,
}

/// Pressure- and tilt-derived brush footprint before viewport scaling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrushFootprint {
    /// Major/minor radii in the brush profile's configured size space.
    pub half_size: Vec2,
    /// Per-sample deposition strength after the feather-light pressure response.
    pub flow: f32,
}

impl BrushProfile {
    pub(crate) fn half_width(self, pressure: f32) -> f32 {
        let pressure = ops::powf(pressure, self.pressure_gamma.max(0.01));
        let ratio = self.minimum_diameter_ratio.clamp(0.0, 1.0);
        let diameter = self.diameter.max(0.25) * (ratio + (1.0 - ratio) * pressure);
        0.5 * diameter
    }

    /// Evaluates the shared pressure, tilt, and flow response used by input and previews.
    /// Pressure is consumed exactly as supplied by winit's normalized pen event.
    pub fn footprint(self, pressure: f32, tilt_degrees: f32) -> BrushFootprint {
        const OVAL_START_DEGREES: f32 = 35.0;
        const OVAL_FULL_DEGREES: f32 = 75.0;
        const LIGHT_PRESSURE_END: f32 = 0.15;
        const MINIMUM_LIGHT_FLOW: f32 = 0.18;

        let radius = self.half_width(pressure);
        let oval = if tilt_degrees <= OVAL_START_DEGREES {
            0.0
        } else if tilt_degrees >= OVAL_FULL_DEGREES {
            1.0
        } else {
            let value =
                (tilt_degrees - OVAL_START_DEGREES) / (OVAL_FULL_DEGREES - OVAL_START_DEGREES);
            value * value * (3.0 - 2.0 * value)
        };
        let pressure_flow = if pressure <= 0.0 {
            MINIMUM_LIGHT_FLOW
        } else if pressure >= LIGHT_PRESSURE_END {
            1.0
        } else {
            let value = pressure / LIGHT_PRESSURE_END;
            let eased = value * value * (3.0 - 2.0 * value);
            MINIMUM_LIGHT_FLOW + (1.0 - MINIMUM_LIGHT_FLOW) * eased
        };

        BrushFootprint {
            half_size: Vec2::new(radius * (1.0 + 1.55 * oval), radius * (1.0 - 0.35 * oval)),
            flow: self.flow * pressure_flow,
        }
    }
}

/// Unit convention for brush diameter before it becomes stored document geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BrushSizeSpace {
    /// Diameter is measured directly in document-coordinate units.
    Document,
    /// Diameter is measured in physical viewport pixels at sample time.
    #[default]
    Screen,
}

/// Runtime configuration for pen collection and persistent GPU buffers.
#[derive(Resource, Clone, Debug)]
pub struct StrokeRendererSettings {
    /// Profile used by pen, brush, pencil, and airbrush tools.
    pub pen: BrushProfile,
    /// Profile used by an eraser-end tablet tool.
    pub eraser: BrushProfile,
    /// Initial point-buffer capacity, grown geometrically when exceeded.
    pub initial_point_capacity: usize,
    /// Initial segment-metadata capacity, grown geometrically when exceeded.
    pub initial_segment_capacity: usize,
    /// Width and height of one persistent canvas content tile.
    pub tile_size: u32,
    /// Aggregate persistent GPU cache budget across every model plane.
    pub gpu_cache_budget_bytes: u64,
    /// Tile regeneration budget while one or more strokes are active.
    pub max_dirty_tiles_while_drawing: u32,
    /// Tile regeneration budget while the pen is idle.
    pub max_dirty_tiles_while_idle: u32,
    /// Effect/scratch work budget while drawing.
    pub max_effect_tiles_while_drawing: u32,
    /// Effect/scratch work budget while idle.
    pub max_effect_tiles_while_idle: u32,
    /// Emit one aggregate telemetry record per second when enabled.
    pub log_diagnostics: bool,
}

impl Default for StrokeRendererSettings {
    fn default() -> Self {
        Self {
            pen: BrushProfile {
                paint: PaintMaterialRef {
                    model: RGBA_PAINT_MODEL_ID,
                    model_version: RgbaPaintModel::MODEL_VERSION,
                    material: RgbaPaintModel::DEFAULT_INK,
                },
                deposition: DepositionMode::Normal,
                diameter: 8.0,
                size_space: BrushSizeSpace::Screen,
                minimum_diameter_ratio: 0.12,
                pressure_gamma: 0.72,
                flow: 1.0,
            },
            eraser: BrushProfile {
                paint: PaintMaterialRef {
                    model: RGBA_PAINT_MODEL_ID,
                    model_version: RgbaPaintModel::MODEL_VERSION,
                    material: RgbaPaintModel::DEFAULT_ERASER,
                },
                deposition: DepositionMode::Erase,
                diameter: 28.0,
                size_space: BrushSizeSpace::Screen,
                minimum_diameter_ratio: 0.35,
                pressure_gamma: 0.72,
                flow: 1.0,
            },
            initial_point_capacity: 16 * 1024,
            initial_segment_capacity: 16 * 1024,
            tile_size: 256,
            gpu_cache_budget_bytes: 256 * 1024 * 1024,
            max_dirty_tiles_while_drawing: 8,
            max_dirty_tiles_while_idle: 32,
            max_effect_tiles_while_drawing: 4,
            max_effect_tiles_while_idle: 16,
            log_diagnostics: true,
        }
    }
}

/// Geometry-only sample. Paint channels are referenced by [`StrokeSegment`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrokePoint {
    /// Position in 2D world coordinates.
    pub position: Vec2,
    /// Half-width in document coordinates.
    pub half_width: f32,
    /// Major-to-minor footprint ratio. One is circular.
    pub aspect_ratio: f32,
    /// Deposition amount before the paint model interprets it.
    pub flow: f32,
    /// Projected brush-major-axis orientation.
    pub orientation: Vec2,
    /// Tool-axis rotation in radians.
    pub twist_radians: f32,
}

/// Converts irregular device samples into a smooth, adaptively sampled curve.
///
/// Consecutive raw points become midpoint quadratic curves. Their derivatives
/// agree at every shared midpoint, so the stored centerline has no hard corner
/// at a platform sample. Position, pressure-derived width/flow, tilt shape, and
/// twist all follow the same curve without altering platform pressure values.
#[derive(Clone, Copy, Debug)]
pub struct StrokePointResampler {
    last_input: StrokePoint,
    curve_start: StrokePoint,
}

impl StrokePointResampler {
    /// Starts a resampler after `first` has already been appended to a stroke.
    pub fn new(first: StrokePoint) -> Self {
        Self {
            last_input: first,
            curve_start: first,
        }
    }

    /// Observes one raw point and emits the completed smooth curve prefix.
    pub fn push(&mut self, point: StrokePoint, mut emit: impl FnMut(StrokePoint)) -> usize {
        let control = self.last_input;
        let curve_end = interpolate_stroke_point(control, point, 0.5);
        let emitted = emit_quadratic_curve(self.curve_start, control, curve_end, &mut emit);
        self.curve_start = curve_end;
        self.last_input = point;
        emitted
    }

    /// Completes the pending half-curve and emits the exact latest input point.
    pub fn finish(&mut self, mut emit: impl FnMut(StrokePoint)) -> bool {
        let emitted = emit_quadratic_curve(
            self.curve_start,
            self.last_input,
            self.last_input,
            &mut emit,
        );
        self.curve_start = self.last_input;
        emitted > 0
    }

    /// Most recent unmodified platform sample observed by the resampler.
    pub fn latest_input(&self) -> StrokePoint {
        self.last_input
    }
}

fn stroke_point_spacing(point: StrokePoint) -> f32 {
    // Four percent of the diameter bounds curvature error for very large
    // brushes. Raw midpoints are always retained, while this spacing only adds
    // samples when the platform leaves a larger gap.
    (point.half_width * 0.08).max(0.7)
}

fn emit_quadratic_curve(
    start: StrokePoint,
    control: StrokePoint,
    end: StrokePoint,
    emit: &mut impl FnMut(StrokePoint),
) -> usize {
    if start == control && control == end {
        return 0;
    }

    let approximate_length =
        start.position.distance(control.position) + control.position.distance(end.position);
    let spacing = stroke_point_spacing(start)
        .min(stroke_point_spacing(control))
        .min(stroke_point_spacing(end));
    let steps = (approximate_length / spacing).ceil().max(1.0) as usize;
    for step in 1..=steps {
        let amount = step as f32 / steps as f32;
        emit(quadratic_stroke_point(start, control, end, amount));
    }
    steps
}

fn quadratic_stroke_point(
    start: StrokePoint,
    control: StrokePoint,
    end: StrokePoint,
    amount: f32,
) -> StrokePoint {
    let first = interpolate_stroke_point(start, control, amount);
    let second = interpolate_stroke_point(control, end, amount);
    interpolate_stroke_point(first, second, amount)
}

fn interpolate_stroke_point(from: StrokePoint, to: StrokePoint, amount: f32) -> StrokePoint {
    if amount <= 0.0 {
        return from;
    }
    if amount >= 1.0 {
        return to;
    }
    StrokePoint {
        position: from.position.lerp(to.position, amount),
        half_width: from.half_width + (to.half_width - from.half_width) * amount,
        aspect_ratio: from.aspect_ratio + (to.aspect_ratio - from.aspect_ratio) * amount,
        flow: from.flow + (to.flow - from.flow) * amount,
        orientation: interpolate_ellipse_axis(from.orientation, to.orientation, amount),
        twist_radians: interpolate_angle(from.twist_radians, to.twist_radians, amount),
    }
}

fn interpolate_ellipse_axis(from: Vec2, to: Vec2, amount: f32) -> Vec2 {
    let from = from.normalize_or(Vec2::Y);
    let mut to = to.normalize_or(from);
    // An ellipse axis is undirected: v and -v describe the same footprint.
    // Align the representations before interpolation to avoid a zero vector at
    // the half-way point.
    if from.dot(to) < 0.0 {
        to = -to;
    }
    from.lerp(to, amount).normalize_or(from)
}

fn interpolate_angle(from: f32, to: f32, amount: f32) -> f32 {
    let delta = (to - from + core::f32::consts::PI).rem_euclid(core::f32::consts::TAU)
        - core::f32::consts::PI;
    from + delta * amount
}

/// Append-only GPU-facing metadata for one procedural segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StrokeSegment {
    /// First point index.
    pub start: u32,
    /// Second point index; equal to `start` for an initial contact dot.
    pub end: u32,
    /// Paint model that owns deposition for this segment.
    pub model: PaintModelId,
    /// Material interpreted by `model`.
    pub material: PaintMaterialId,
    /// Deposition operation interpreted by `model`.
    pub deposition: DepositionMode,
    /// Stable canvas layer used by live compositing and durable replay.
    pub layer: LayerId,
}

/// Lightweight stroke range metadata retained on the CPU.
#[derive(Clone, Debug)]
pub struct StrokeMetadata {
    /// Stable stroke identity.
    pub id: StrokeId,
    /// Index of the contact point.
    pub first_point: u32,
    /// Number of points currently in the stroke.
    pub point_count: u32,
    /// First segment in the append-only geometry buffer.
    pub first_segment: u32,
    /// Number of procedural segments owned by the stroke.
    pub segment_count: u32,
    /// Canvas layer receiving this stroke.
    pub layer: LayerId,
    /// Immutable brush style captured when contact began.
    pub brush: BrushProfile,
    /// Bounds including brush radius and two antialias pixels.
    pub bounds: Rect,
    /// Document revision that must be present in every affected tile.
    pub revision: u64,
    /// Current document/cache lifecycle state.
    pub state: StrokeState,
    /// Whether the vector stroke participates in replay.
    pub visible: bool,
    /// Whether contact has ended.
    pub complete: bool,
    /// Tiles whose commit controls active-to-cached handoff.
    pub affected_tiles: Vec<TileKey>,
}

/// Stable layer identity carried by document commands and tile keys.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerId(pub u32);

/// Initial layer compositing contract. More modes can be registered later.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LayerCompositeMode {
    /// Premultiplied source-over compositing.
    #[default]
    Normal,
}

/// Persistent canvas-layer metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct CanvasLayer {
    /// Stable identity stored by strokes and tile keys.
    pub id: LayerId,
    /// Human-readable layer label.
    pub name: String,
    /// Layer opacity applied during display resolve.
    pub opacity: f32,
    /// Layer compositing operation.
    pub composite: LayerCompositeMode,
    /// Hidden layers stay in the document but are not replayed.
    pub visible: bool,
}

impl Default for CanvasLayer {
    fn default() -> Self {
        Self {
            id: LayerId(0),
            name: "Paint".into(),
            opacity: 1.0,
            composite: LayerCompositeMode::Normal,
            visible: true,
        }
    }
}

/// Lifecycle of a vector stroke relative to the persistent tile cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrokeState {
    /// Contact is active and always renders through the low-latency overlay.
    Active,
    /// Contact ended, but at least one affected tile has not committed.
    PendingCache,
    /// Every affected tile contains this stroke's required revision.
    Cached,
    /// An undo or clear command excludes this stroke from replay.
    Hidden,
}

#[derive(Clone, Copy, Debug)]
struct ActiveStroke {
    last_point: u32,
    metadata_index: usize,
    profile: BrushProfile,
}

#[derive(Clone, Debug)]
pub(crate) struct TileStrokeIndex {
    pub stroke_index: usize,
    pub segments: Vec<u32>,
}

/// An append-only change recorded during the most recent input batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrokeDelta {
    /// A contact began and created an initial dot segment.
    Begin {
        /// New stroke.
        stroke: StrokeId,
        /// Appended point index.
        point: u32,
        /// Appended segment index.
        segment: u32,
    },
    /// A contact appended a point and bridge segment.
    Append {
        /// Changed stroke.
        stroke: StrokeId,
        /// Appended point index.
        point: u32,
        /// Appended segment index.
        segment: u32,
    },
    /// A contact ended.
    End {
        /// Completed stroke.
        stroke: StrokeId,
    },
}

/// Deltas produced by one Bevy input update; it is replaced rather than queued.
#[derive(Resource, Default, Debug)]
pub struct StrokeDeltaBatch {
    frame: u64,
    deltas: Vec<StrokeDelta>,
}

impl StrokeDeltaBatch {
    /// Input-batch sequence number.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Changes produced in this input batch.
    pub fn deltas(&self) -> &[StrokeDelta] {
        &self.deltas
    }

    pub(crate) fn begin_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        self.deltas.clear();
    }

    pub(crate) fn push(&mut self, delta: StrokeDelta) {
        self.deltas.push(delta);
    }
}

#[derive(Clone, Debug)]
enum HistoryCommand {
    AddStroke(usize),
    Clear(Vec<usize>),
    Resize {
        before: UVec2,
        after: UVec2,
    },
    AddLayer {
        layer: CanvasLayer,
        index: usize,
    },
    LayerVisibility {
        layer: LayerId,
        before: bool,
        after: bool,
    },
    LayerOpacity {
        layer: LayerId,
        before: f32,
        after: f32,
    },
    LayerMove {
        layer: LayerId,
        before: usize,
        after: usize,
    },
    LayerRename {
        layer: LayerId,
        before: String,
        after: String,
    },
}

/// Authoritative, editable vector document.
///
/// Geometry remains append-only so render extraction stays proportional to new
/// input. Visibility, history, the spatial index, and tile revisions determine
/// what is replayed into the derived GPU cache.
#[derive(Resource)]
pub struct StrokeDocument {
    points: Vec<StrokePoint>,
    segments: Vec<StrokeSegment>,
    strokes: Vec<StrokeMetadata>,
    stroke_lookup: HashMap<StrokeId, usize>,
    active: HashMap<StrokeId, ActiveStroke>,
    overlay_strokes: BTreeSet<usize>,
    spatial_index: HashMap<TileKey, Vec<TileStrokeIndex>>,
    invalidations: Vec<TileInvalidation>,
    undo: Vec<HistoryCommand>,
    redo: Vec<HistoryCommand>,
    layers: Vec<CanvasLayer>,
    active_layer: LayerId,
    next_layer: u32,
    layer_revision: u64,
    effects: EffectGraph,
    paint_materials: PaintMaterialLibrary,
    next_stroke: u64,
    revision: u64,
    contact_generation: u64,
    geometry_generation: u64,
    tile_size: u32,
    effect_radius_px: u32,
    canvas_size: UVec2,
    input_batch_started: Option<Instant>,
    latest_sample_received: Option<Instant>,
}

impl Default for StrokeDocument {
    fn default() -> Self {
        let mut paint_materials = PaintMaterialLibrary::default();
        for (material, rgba) in RgbaPaintModel::default().materials().iter().enumerate() {
            paint_materials
                .add(PaintMaterialRecord {
                    reference: PaintMaterialRef {
                        model: RGBA_PAINT_MODEL_ID,
                        model_version: RgbaPaintModel::MODEL_VERSION,
                        material: PaintMaterialId(material as u32),
                    },
                    payload: PaintMaterialPayload::PremultipliedLinearRgba(
                        rgba.premultiplied_linear_rgba,
                    ),
                })
                .expect("built-in RGBA material identities are unique");
        }
        Self {
            points: Vec::new(),
            segments: Vec::new(),
            strokes: Vec::new(),
            stroke_lookup: HashMap::new(),
            active: HashMap::new(),
            overlay_strokes: BTreeSet::new(),
            spatial_index: HashMap::new(),
            invalidations: Vec::new(),
            undo: Vec::new(),
            redo: Vec::new(),
            layers: vec![CanvasLayer::default()],
            active_layer: LayerId(0),
            next_layer: 1,
            layer_revision: 1,
            effects: EffectGraph::default(),
            paint_materials,
            next_stroke: 0,
            revision: 0,
            contact_generation: 0,
            geometry_generation: 0,
            tile_size: 256,
            effect_radius_px: 0,
            canvas_size: UVec2::new(4096, 4096),
            input_batch_started: None,
            latest_sample_received: None,
        }
    }
}

impl StrokeDocument {
    /// All geometry samples in append order.
    pub fn points(&self) -> &[StrokePoint] {
        &self.points
    }

    /// All procedural segments in append order.
    pub fn segments(&self) -> &[StrokeSegment] {
        &self.segments
    }

    /// CPU metadata for recorded strokes.
    pub fn strokes(&self) -> &[StrokeMetadata] {
        &self.strokes
    }

    /// Current document revision. Every editing command increments it.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Layers retained by the authoritative document.
    pub fn layers(&self) -> &[CanvasLayer] {
        &self.layers
    }

    /// Layer receiving newly started contacts.
    pub fn active_layer(&self) -> LayerId {
        self.active_layer
    }

    /// Looks up persistent layer metadata by stable identity.
    pub fn layer(&self, id: LayerId) -> Option<&CanvasLayer> {
        self.layers.iter().find(|layer| layer.id == id)
    }

    /// Returns the bottom-to-top composite position of a stable layer.
    pub fn layer_index(&self, id: LayerId) -> Option<usize> {
        self.layers.iter().position(|layer| layer.id == id)
    }

    /// Selects the layer that receives future strokes.
    pub fn set_active_layer(&mut self, id: LayerId) -> bool {
        if self.layer(id).is_none() {
            return false;
        }
        self.active_layer = id;
        true
    }

    /// Creates a normal layer above the current top layer and selects it.
    pub fn add_layer(&mut self, name: impl Into<String>) -> LayerId {
        let id = LayerId(self.next_layer);
        self.next_layer = self.next_layer.wrapping_add(1);
        let layer = CanvasLayer {
            id,
            name: name.into(),
            opacity: 1.0,
            composite: LayerCompositeMode::Normal,
            visible: true,
        };
        let index = self.layers.len();
        self.layers.push(layer.clone());
        self.active_layer = id;
        self.layer_revision = self.layer_revision.wrapping_add(1);
        self.bump_revision();
        self.undo.push(HistoryCommand::AddLayer { layer, index });
        self.redo.clear();
        id
    }

    /// Changes whether a layer participates in live and cached compositing.
    pub fn set_layer_visibility(&mut self, id: LayerId, visible: bool) -> bool {
        let Some(index) = self.layer_index(id) else {
            return false;
        };
        let before = self.layers[index].visible;
        if before == visible {
            return true;
        }
        self.layers[index].visible = visible;
        self.layer_revision = self.layer_revision.wrapping_add(1);
        self.bump_revision();
        self.undo.push(HistoryCommand::LayerVisibility {
            layer: id,
            before,
            after: visible,
        });
        self.redo.clear();
        true
    }

    /// Sets normal-layer opacity in the inclusive `0..=1` range.
    pub fn set_layer_opacity(&mut self, id: LayerId, opacity: f32) -> bool {
        let Some(index) = self.layer_index(id) else {
            return false;
        };
        let opacity = opacity.clamp(0.0, 1.0);
        let before = self.layers[index].opacity;
        if before == opacity {
            return true;
        }
        self.layers[index].opacity = opacity;
        self.layer_revision = self.layer_revision.wrapping_add(1);
        self.bump_revision();
        self.undo.push(HistoryCommand::LayerOpacity {
            layer: id,
            before,
            after: opacity,
        });
        self.redo.clear();
        true
    }

    /// Renames a layer without changing its stable identity.
    pub fn rename_layer(&mut self, id: LayerId, name: impl Into<String>) -> bool {
        let Some(index) = self.layer_index(id) else {
            return false;
        };
        let name = name.into();
        let before = self.layers[index].name.clone();
        if before == name {
            return true;
        }
        self.layers[index].name = name.clone();
        self.layer_revision = self.layer_revision.wrapping_add(1);
        self.bump_revision();
        self.undo.push(HistoryCommand::LayerRename {
            layer: id,
            before,
            after: name,
        });
        self.redo.clear();
        true
    }

    /// Moves a layer to a bottom-to-top composite index.
    pub fn move_layer(&mut self, id: LayerId, index: usize) -> bool {
        let Some(before) = self.layer_index(id) else {
            return false;
        };
        let after = index.min(self.layers.len().saturating_sub(1));
        if before == after {
            return true;
        }
        self.move_layer_untracked(id, after);
        self.layer_revision = self.layer_revision.wrapping_add(1);
        self.bump_revision();
        self.undo.push(HistoryCommand::LayerMove {
            layer: id,
            before,
            after,
        });
        self.redo.clear();
        true
    }

    pub(crate) fn layer_revision(&self) -> u64 {
        self.layer_revision
    }

    /// Versioned non-destructive effect graph stored with the document.
    pub fn effects(&self) -> &EffectGraph {
        &self.effects
    }

    /// Immutable color/pigment recipes required to regenerate this document.
    pub fn paint_materials(&self) -> &PaintMaterialLibrary {
        &self.paint_materials
    }

    /// Adds a new immutable material recipe to the document.
    pub fn add_paint_material(&mut self, record: PaintMaterialRecord) -> Result<(), &'static str> {
        self.paint_materials.add(record)
    }

    pub(crate) fn sync_rgba_materials(&mut self, model: &RgbaPaintModel) {
        for (index, material) in model.materials().iter().enumerate() {
            let _ = self.paint_materials.add(PaintMaterialRecord {
                reference: PaintMaterialRef {
                    model: RGBA_PAINT_MODEL_ID,
                    model_version: RgbaPaintModel::MODEL_VERSION,
                    material: PaintMaterialId(index as u32),
                },
                payload: PaintMaterialPayload::PremultipliedLinearRgba(
                    material.premultiplied_linear_rgba,
                ),
            });
        }
    }

    /// Replaces the document effect graph. The tile scheduler applies its radius.
    pub fn set_effects(&mut self, effects: EffectGraph) {
        if effects.revision() == self.effects.revision() && effects.nodes() == self.effects.nodes()
        {
            return;
        }
        self.effects = effects;
        self.bump_revision();
        self.invalidate_all(TileInvalidationCause::EffectDependency, false);
    }

    /// Logical canvas extent. Resizing never rewrites vector coordinates.
    pub fn canvas_size(&self) -> UVec2 {
        self.canvas_size
    }

    /// Number of history commands that can currently be undone.
    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    /// Number of history commands that can currently be redone.
    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }

    /// Generation used by the input adapter to abandon contacts after clear.
    pub fn generation(&self) -> u64 {
        self.contact_generation
    }

    /// Generation used by extraction to request a full geometry rebuild.
    pub(crate) fn geometry_generation(&self) -> u64 {
        self.geometry_generation
    }

    /// Returns `true` while any contact is active.
    pub fn is_drawing(&self) -> bool {
        !self.active.is_empty()
    }

    /// Replaces this document after a validated load while forcing every
    /// disposable CPU/GPU cache to recognize the new authoritative identity.
    pub fn replace_loaded(&mut self, mut loaded: StrokeDocument) {
        loaded.geometry_generation = self.geometry_generation.wrapping_add(1);
        loaded.contact_generation = self.contact_generation.wrapping_add(1);
        *self = loaded;
    }

    /// Adds an undoable clear command without destroying vector history.
    pub fn clear(&mut self) {
        let active_indices: Vec<_> = self
            .active
            .values()
            .map(|active| active.metadata_index)
            .collect();
        for index in active_indices {
            self.overlay_strokes.remove(&index);
            let stroke = &mut self.strokes[index];
            stroke.visible = false;
            stroke.complete = true;
            stroke.state = StrokeState::Hidden;
        }
        self.active.clear();
        self.contact_generation = self.contact_generation.wrapping_add(1);

        let changed: Vec<_> = self
            .strokes
            .iter()
            .enumerate()
            .filter_map(|(index, stroke)| (stroke.visible && stroke.complete).then_some(index))
            .collect();
        if !changed.is_empty() {
            self.bump_revision();
            for &index in &changed {
                self.set_visibility(index, false);
            }
            self.invalidate_indices(&changed, TileInvalidationCause::Clear, false);
            self.undo.push(HistoryCommand::Clear(changed));
            self.redo.clear();
        }
        let now = Instant::now();
        self.input_batch_started = Some(now);
        self.latest_sample_received = Some(now);
    }

    /// Reverts the most recent completed stroke, clear, or resize command.
    pub fn undo(&mut self) -> bool {
        let Some(command) = self.undo.pop() else {
            return false;
        };
        self.bump_revision();
        match &command {
            HistoryCommand::AddStroke(index) => {
                self.set_visibility(*index, false);
                self.invalidate_indices(
                    core::slice::from_ref(index),
                    TileInvalidationCause::UndoRedo,
                    false,
                );
            }
            HistoryCommand::Clear(indices) => {
                for &index in indices {
                    self.set_visibility(index, true);
                }
                self.invalidate_indices(indices, TileInvalidationCause::UndoRedo, false);
            }
            HistoryCommand::Resize { before, .. } => {
                self.canvas_size = *before;
                self.invalidate_all(TileInvalidationCause::Resize, false);
            }
            HistoryCommand::AddLayer { layer, .. } => {
                self.layers.retain(|candidate| candidate.id != layer.id);
                if self.active_layer == layer.id {
                    self.active_layer = self.layers.last().map_or(LayerId(0), |layer| layer.id);
                }
                self.layer_revision = self.layer_revision.wrapping_add(1);
            }
            HistoryCommand::LayerVisibility { layer, before, .. } => {
                if let Some(index) = self.layer_index(*layer) {
                    self.layers[index].visible = *before;
                    self.layer_revision = self.layer_revision.wrapping_add(1);
                }
            }
            HistoryCommand::LayerOpacity { layer, before, .. } => {
                if let Some(index) = self.layer_index(*layer) {
                    self.layers[index].opacity = *before;
                    self.layer_revision = self.layer_revision.wrapping_add(1);
                }
            }
            HistoryCommand::LayerMove { layer, before, .. } => {
                self.move_layer_untracked(*layer, *before);
                self.layer_revision = self.layer_revision.wrapping_add(1);
            }
            HistoryCommand::LayerRename { layer, before, .. } => {
                if let Some(index) = self.layer_index(*layer) {
                    self.layers[index].name = before.clone();
                    self.layer_revision = self.layer_revision.wrapping_add(1);
                }
            }
        }
        self.redo.push(command);
        true
    }

    /// Reapplies the most recently undone command.
    pub fn redo(&mut self) -> bool {
        let Some(command) = self.redo.pop() else {
            return false;
        };
        self.bump_revision();
        match &command {
            HistoryCommand::AddStroke(index) => {
                self.set_visibility(*index, true);
                self.invalidate_indices(
                    core::slice::from_ref(index),
                    TileInvalidationCause::UndoRedo,
                    false,
                );
            }
            HistoryCommand::Clear(indices) => {
                for &index in indices {
                    self.set_visibility(index, false);
                }
                self.invalidate_indices(indices, TileInvalidationCause::UndoRedo, false);
            }
            HistoryCommand::Resize { after, .. } => {
                self.canvas_size = *after;
                self.invalidate_all(TileInvalidationCause::Resize, false);
            }
            HistoryCommand::AddLayer { layer, index } => {
                let index = (*index).min(self.layers.len());
                self.layers.insert(index, layer.clone());
                self.active_layer = layer.id;
                self.layer_revision = self.layer_revision.wrapping_add(1);
            }
            HistoryCommand::LayerVisibility { layer, after, .. } => {
                if let Some(index) = self.layer_index(*layer) {
                    self.layers[index].visible = *after;
                    self.layer_revision = self.layer_revision.wrapping_add(1);
                }
            }
            HistoryCommand::LayerOpacity { layer, after, .. } => {
                if let Some(index) = self.layer_index(*layer) {
                    self.layers[index].opacity = *after;
                    self.layer_revision = self.layer_revision.wrapping_add(1);
                }
            }
            HistoryCommand::LayerMove { layer, after, .. } => {
                self.move_layer_untracked(*layer, *after);
                self.layer_revision = self.layer_revision.wrapping_add(1);
            }
            HistoryCommand::LayerRename { layer, after, .. } => {
                if let Some(index) = self.layer_index(*layer) {
                    self.layers[index].name = after.clone();
                    self.layer_revision = self.layer_revision.wrapping_add(1);
                }
            }
        }
        self.undo.push(command);
        true
    }

    /// Changes the logical canvas extent and invalidates viewport-derived tiles.
    pub fn resize(&mut self, size: UVec2) -> bool {
        let size = size.max(UVec2::ONE);
        if size == self.canvas_size {
            return false;
        }
        let before = self.canvas_size;
        self.canvas_size = size;
        self.bump_revision();
        self.invalidate_all(TileInvalidationCause::Resize, false);
        self.undo.push(HistoryCommand::Resize {
            before,
            after: size,
        });
        self.redo.clear();
        true
    }

    /// Invalidates all derived GPU state while retaining authoritative history.
    pub fn regenerate_after_device_loss(&mut self) {
        self.geometry_generation = self.geometry_generation.wrapping_add(1);
        self.bump_revision();
        self.invalidate_all(TileInvalidationCause::DeviceLoss, false);
        for (index, stroke) in self.strokes.iter_mut().enumerate() {
            if stroke.visible && stroke.complete {
                stroke.state = StrokeState::PendingCache;
                self.overlay_strokes.remove(&index);
                stroke.revision = self.revision;
            }
        }
    }

    /// Updates tile identity when renderer configuration changes.
    pub(crate) fn set_tile_size(&mut self, tile_size: u32) {
        let tile_size = tile_size.max(1);
        if tile_size == self.tile_size {
            return;
        }
        self.tile_size = tile_size;
        self.rebuild_spatial_index();
        self.bump_revision();
        self.invalidate_all(TileInvalidationCause::Resize, false);
    }

    /// Applies the largest enabled local-effect radius to future invalidations.
    pub(crate) fn set_effect_radius(&mut self, radius_px: u32) {
        if self.effect_radius_px == radius_px {
            return;
        }
        self.effect_radius_px = radius_px;
        self.rebuild_spatial_index();
        self.bump_revision();
        self.invalidate_all(TileInvalidationCause::EffectDependency, false);
    }

    pub(crate) fn invalidate_cache_layout(&mut self) {
        self.bump_revision();
        self.invalidate_all(TileInvalidationCause::CacheLayout, false);
        for (index, stroke) in self.strokes.iter_mut().enumerate() {
            if stroke.visible && stroke.complete {
                stroke.state = StrokeState::PendingCache;
                stroke.revision = self.revision;
                self.overlay_strokes.remove(&index);
            }
        }
    }

    pub(crate) fn begin_input_batch(&mut self) {
        self.input_batch_started = None;
    }

    pub(crate) fn input_batch_started(&self) -> Option<Instant> {
        self.input_batch_started
    }

    pub(crate) fn latest_sample_received(&self) -> Option<Instant> {
        self.latest_sample_received
    }

    /// Starts a renderer-neutral stroke and returns its identity and append delta.
    ///
    /// Synthetic benchmarks and replay sources may call this directly instead of
    /// producing platform input messages.
    pub fn begin_stroke(
        &mut self,
        point: StrokePoint,
        profile: BrushProfile,
    ) -> (StrokeId, StrokeDelta) {
        let id = StrokeId(self.next_stroke);
        self.next_stroke = self.next_stroke.wrapping_add(1);

        let layer = self.active_layer;
        let point_index = self.push_point(point);
        let segment_index = self.push_segment(point_index, point_index, profile, layer);
        let metadata_index = self.strokes.len();
        let bounds = point_bounds(point);
        self.strokes.push(StrokeMetadata {
            id,
            first_point: point_index,
            point_count: 1,
            first_segment: segment_index,
            segment_count: 1,
            layer,
            brush: profile,
            bounds,
            revision: self.revision,
            state: StrokeState::Active,
            visible: true,
            complete: false,
            affected_tiles: Vec::new(),
        });
        self.stroke_lookup.insert(id, metadata_index);
        self.overlay_strokes.insert(metadata_index);
        self.active.insert(
            id,
            ActiveStroke {
                last_point: point_index,
                metadata_index,
                profile,
            },
        );
        (
            id,
            StrokeDelta::Begin {
                stroke: id,
                point: point_index,
                segment: segment_index,
            },
        )
    }

    /// Appends one geometry sample, or returns `None` for an unknown or exact duplicate sample.
    pub fn append_point(&mut self, id: StrokeId, point: StrokePoint) -> Option<StrokeDelta> {
        let active = *self.active.get(&id)?;
        if self.points.get(active.last_point as usize) == Some(&point) {
            return None;
        }

        let point_index = self.push_point(point);
        let layer = self.strokes[active.metadata_index].layer;
        let segment_index =
            self.push_segment(active.last_point, point_index, active.profile, layer);
        let active = self.active.get_mut(&id)?;
        active.last_point = point_index;
        let stroke = &mut self.strokes[active.metadata_index];
        stroke.point_count += 1;
        stroke.segment_count += 1;
        stroke.bounds = stroke.bounds.union(point_bounds(point));
        Some(StrokeDelta::Append {
            stroke: id,
            point: point_index,
            segment: segment_index,
        })
    }

    /// Completes an active stroke, returning `None` when it is no longer active.
    pub fn end_stroke(&mut self, id: StrokeId) -> Option<StrokeDelta> {
        let active = self.active.remove(&id)?;
        self.bump_revision();
        let index = active.metadata_index;
        {
            let stroke = &mut self.strokes[index];
            stroke.complete = true;
            stroke.state = StrokeState::PendingCache;
            stroke.revision = self.revision;
            stroke.affected_tiles = tiles_for_rect(
                stroke.bounds.inflate(self.effect_radius_px as f32),
                stroke.layer,
                self.tile_size,
            );
        }
        self.index_stroke(index);
        let tiles = self.strokes[index].affected_tiles.clone();
        self.invalidate_tiles(tiles, TileInvalidationCause::StrokeCommit, true);
        self.undo.push(HistoryCommand::AddStroke(index));
        self.redo.clear();
        Some(StrokeDelta::End { stroke: id })
    }

    /// Returns metadata by stable identity.
    pub fn stroke(&self, id: StrokeId) -> Option<&StrokeMetadata> {
        self.stroke_lookup
            .get(&id)
            .and_then(|&index| self.strokes.get(index))
    }

    #[cfg(test)]
    pub(crate) fn overlay_segment_ranges(&self, output: &mut Vec<core::ops::Range<u32>>) {
        output.clear();
        for &index in &self.overlay_strokes {
            let stroke = &self.strokes[index];
            if stroke.visible
                && self
                    .layer(stroke.layer)
                    .is_some_and(|layer| layer.visible && layer.opacity > 0.0)
            {
                output.push(
                    stroke.first_segment..stroke.first_segment.saturating_add(stroke.segment_count),
                );
            }
        }
    }

    pub(crate) fn overlay_layer_segment_ranges(
        &self,
        output: &mut Vec<(LayerId, core::ops::Range<u32>)>,
    ) {
        output.clear();
        for &index in &self.overlay_strokes {
            let stroke = &self.strokes[index];
            if stroke.visible
                && self
                    .layer(stroke.layer)
                    .is_some_and(|layer| layer.visible && layer.opacity > 0.0)
            {
                output.push((
                    stroke.layer,
                    stroke.first_segment..stroke.first_segment.saturating_add(stroke.segment_count),
                ));
            }
        }
    }

    pub(crate) fn refresh_cache_handoffs(
        &mut self,
        mut tile_is_ready: impl FnMut(TileKey, u64) -> bool,
    ) {
        let ready: Vec<_> = self
            .overlay_strokes
            .iter()
            .copied()
            .filter(|&index| {
                let stroke = &self.strokes[index];
                stroke.complete
                    && stroke.visible
                    && stroke
                        .affected_tiles
                        .iter()
                        .all(|&key| tile_is_ready(key, stroke.revision))
            })
            .collect();
        for index in ready {
            self.overlay_strokes.remove(&index);
            self.strokes[index].state = StrokeState::Cached;
        }
    }

    pub(crate) fn drain_invalidations(&mut self, output: &mut Vec<TileInvalidation>) {
        output.clear();
        output.append(&mut self.invalidations);
    }

    pub(crate) fn tile_strokes(&self, key: TileKey) -> &[TileStrokeIndex] {
        self.spatial_index.get(&key).map_or(&[], Vec::as_slice)
    }

    pub(crate) fn tile_has_visible_strokes(&self, key: TileKey) -> bool {
        if !self.layer(key.layer).is_some_and(|layer| layer.visible) {
            return false;
        }
        self.tile_strokes(key).iter().any(|entry| {
            self.strokes[entry.stroke_index].visible && self.strokes[entry.stroke_index].complete
        })
    }

    fn move_layer_untracked(&mut self, id: LayerId, index: usize) {
        let Some(current) = self.layer_index(id) else {
            return;
        };
        let layer = self.layers.remove(current);
        self.layers.insert(index.min(self.layers.len()), layer);
    }

    fn push_point(&mut self, point: StrokePoint) -> u32 {
        let index = u32::try_from(self.points.len()).expect("stroke point limit exceeded");
        self.points.push(point);
        let now = Instant::now();
        self.input_batch_started.get_or_insert(now);
        self.latest_sample_received = Some(now);
        index
    }

    fn push_segment(&mut self, start: u32, end: u32, profile: BrushProfile, layer: LayerId) -> u32 {
        let index = u32::try_from(self.segments.len()).expect("stroke segment limit exceeded");
        self.segments.push(StrokeSegment {
            start,
            end,
            model: profile.paint.model,
            material: profile.paint.material,
            deposition: profile.deposition,
            layer,
        });
        index
    }

    fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    fn set_visibility(&mut self, index: usize, visible: bool) {
        let stroke = &mut self.strokes[index];
        stroke.visible = visible;
        stroke.revision = self.revision;
        if visible {
            stroke.state = StrokeState::PendingCache;
            self.overlay_strokes.insert(index);
        } else {
            stroke.state = StrokeState::Hidden;
            self.overlay_strokes.remove(&index);
        }
    }

    fn index_stroke(&mut self, index: usize) {
        let stroke = &self.strokes[index];
        let layer = stroke.layer;
        let first = stroke.first_segment;
        let end = first.saturating_add(stroke.segment_count);
        for segment_index in first..end {
            let segment = self.segments[segment_index as usize];
            let a = self.points[segment.start as usize];
            let b = self.points[segment.end as usize];
            let bounds = point_bounds(a).union(point_bounds(b));
            for key in tiles_for_rect(bounds, layer, self.tile_size) {
                let entries = self.spatial_index.entry(key).or_default();
                if let Some(entry) = entries.last_mut()
                    && entry.stroke_index == index
                {
                    entry.segments.push(segment_index);
                } else {
                    entries.push(TileStrokeIndex {
                        stroke_index: index,
                        segments: vec![segment_index],
                    });
                }
            }
        }
    }

    fn rebuild_spatial_index(&mut self) {
        self.spatial_index.clear();
        for index in 0..self.strokes.len() {
            if !self.strokes[index].complete {
                continue;
            }
            let bounds = self.strokes[index].bounds;
            let layer = self.strokes[index].layer;
            self.strokes[index].affected_tiles = tiles_for_rect(
                bounds.inflate(self.effect_radius_px as f32),
                layer,
                self.tile_size,
            );
            self.index_stroke(index);
        }
    }

    fn invalidate_indices(
        &mut self,
        indices: &[usize],
        cause: TileInvalidationCause,
        allow_stale_display: bool,
    ) {
        let mut tiles = BTreeSet::new();
        for &index in indices {
            tiles.extend(self.strokes[index].affected_tiles.iter().copied());
        }
        self.invalidate_tiles(tiles, cause, allow_stale_display);
    }

    fn invalidate_all(&mut self, cause: TileInvalidationCause, allow_stale_display: bool) {
        let mut tiles = BTreeSet::new();
        for stroke in &self.strokes {
            if stroke.complete {
                tiles.extend(stroke.affected_tiles.iter().copied());
            }
        }
        self.invalidate_tiles(tiles, cause, allow_stale_display);
    }

    fn invalidate_tiles(
        &mut self,
        tiles: impl IntoIterator<Item = TileKey>,
        cause: TileInvalidationCause,
        allow_stale_display: bool,
    ) {
        self.invalidations
            .extend(tiles.into_iter().map(|key| TileInvalidation {
                key,
                revision: self.revision,
                cause,
                allow_stale_display,
                halo_radius_px: self.effect_radius_px,
            }));
    }
}

/// Compatibility name retained for the Phase 1 API.
pub type StrokeStore = StrokeDocument;

fn point_bounds(point: StrokePoint) -> Rect {
    Rect::from_center_size(point.position, Vec2::ZERO)
        .inflate(point.half_width.max(0.0) * point.aspect_ratio.max(1.0) + 2.0)
}

pub(crate) fn tiles_for_rect(bounds: Rect, layer: LayerId, tile_size: u32) -> Vec<TileKey> {
    let tile_size = tile_size.max(1) as f32;
    let min_x = (bounds.min.x / tile_size).floor() as i32;
    let min_y = (bounds.min.y / tile_size).floor() as i32;
    let max_x = (bounds.max.x / tile_size).floor() as i32;
    let max_y = (bounds.max.y / tile_size).floor() as i32;
    let mut tiles =
        Vec::with_capacity(((max_x - min_x + 1).max(0) * (max_y - min_y + 1).max(0)) as usize);
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            tiles.push(TileKey::new(layer, x, y));
        }
    }
    tiles
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f32) -> StrokePoint {
        StrokePoint {
            position: Vec2::new(x, 2.0),
            half_width: 3.0,
            aspect_ratio: 1.0,
            flow: 1.0,
            orientation: Vec2::Y,
            twist_radians: 0.0,
        }
    }

    fn profile() -> BrushProfile {
        StrokeRendererSettings::default().pen
    }

    #[test]
    fn brush_stays_round_until_tilt_threshold_then_becomes_oval() {
        let brush = profile();
        let upright = brush.footprint(1.0, 0.0);
        let threshold = brush.footprint(1.0, 35.0);
        let tilted = brush.footprint(1.0, 60.0);
        assert_eq!(upright.half_size.x, upright.half_size.y);
        assert_eq!(threshold.half_size.x, threshold.half_size.y);
        assert!(tilted.half_size.x > tilted.half_size.y);
    }

    #[test]
    fn only_feather_light_pressure_reduces_flow() {
        let brush = profile();
        assert_eq!(brush.footprint(0.15, 0.0).flow, brush.flow);
        assert_eq!(brush.footprint(0.8, 0.0).flow, brush.flow);
        assert!(brush.footprint(0.03, 0.0).flow < brush.flow);
    }

    #[test]
    fn resampler_builds_midpoint_curves_and_finishes_at_the_exact_tip() {
        let first = StrokePoint {
            half_width: 100.0,
            ..point(0.0)
        };
        let mut resampler = StrokePointResampler::new(first);
        let mut output = Vec::new();

        for x in [2.0, 4.0] {
            resampler.push(
                StrokePoint {
                    position: Vec2::new(x, first.position.y),
                    ..first
                },
                |point| output.push(point),
            );
        }

        assert_eq!(output.len(), 2);
        assert_eq!(output[0].position.x, 1.0);
        assert_eq!(output[1].position.x, 3.0);
        assert!(resampler.finish(|point| output.push(point)));
        assert_eq!(output.last().unwrap().position.x, 4.0);
    }

    #[test]
    fn midpoint_quadratics_are_tangent_continuous() {
        let first = point(0.0);
        let middle = StrokePoint {
            position: Vec2::new(3.0, 5.0),
            ..first
        };
        let midpoint = interpolate_stroke_point(first, middle, 0.5);
        let first_end_derivative = 2.0 * (midpoint.position - first.position);
        let second_start_derivative = 2.0 * (middle.position - midpoint.position);
        assert!(first_end_derivative.distance(second_start_derivative) < 0.0001);
    }

    #[test]
    fn resampler_interpolates_every_footprint_channel() {
        let first = StrokePoint {
            position: Vec2::ZERO,
            half_width: 10.0,
            aspect_ratio: 1.0,
            flow: 0.2,
            orientation: Vec2::X,
            twist_radians: core::f32::consts::PI - 0.1,
        };
        let last = StrokePoint {
            position: Vec2::new(10.0, 0.0),
            half_width: 20.0,
            aspect_ratio: 2.0,
            flow: 0.8,
            orientation: Vec2::Y,
            twist_radians: -core::f32::consts::PI + 0.1,
        };
        let mut resampler = StrokePointResampler::new(first);
        let mut output = Vec::new();
        resampler.push(last, |point| output.push(point));

        let interpolated = *output.last().unwrap();
        assert_eq!(interpolated.position.x, 5.0);
        assert_eq!(interpolated.half_width, 15.0);
        assert_eq!(interpolated.aspect_ratio, 1.5);
        assert_eq!(interpolated.flow, 0.5);
        assert!(interpolated.orientation.x > 0.7);
        assert!(interpolated.orientation.y > 0.7);
        assert!(interpolated.twist_radians > first.twist_radians);

        assert!(resampler.finish(|point| output.push(point)));
        assert_eq!(*output.last().unwrap(), last);
    }

    #[test]
    fn ellipse_axis_interpolation_treats_opposite_vectors_as_equivalent() {
        let axis = interpolate_ellipse_axis(Vec2::X, Vec2::NEG_X, 0.5);
        assert_eq!(axis, Vec2::X);
    }

    #[test]
    fn stroke_storage_is_append_only_and_bridges_the_tip() {
        let mut store = StrokeStore::default();
        let (stroke, begin) = store.begin_stroke(point(0.0), profile());
        let append = store.append_point(stroke, point(1.0)).unwrap();
        store.end_stroke(stroke).unwrap();

        assert!(matches!(begin, StrokeDelta::Begin { segment: 0, .. }));
        assert!(matches!(append, StrokeDelta::Append { segment: 1, .. }));
        assert_eq!(store.points().len(), 2);
        assert_eq!(store.segments()[0].start, store.segments()[0].end);
        assert_eq!(store.segments()[1].end, 1);
        assert!(store.strokes()[0].complete);
    }

    #[test]
    fn new_strokes_use_stable_active_layers_and_layer_edits_are_undoable() {
        let mut document = StrokeDocument::default();
        let upper = document.add_layer("Upper");
        let (stroke, _) = document.begin_stroke(point(0.0), profile());
        document.end_stroke(stroke).unwrap();
        assert_eq!(document.stroke(stroke).unwrap().layer, upper);
        assert!(document
            .stroke(stroke)
            .unwrap()
            .affected_tiles
            .iter()
            .all(|key| key.layer == upper));

        document.set_layer_opacity(upper, 0.4);
        document.set_layer_visibility(upper, false);
        assert!(!document.layer(upper).unwrap().visible);
        assert!(document.undo());
        assert!(document.layer(upper).unwrap().visible);
        assert!(document.undo());
        assert_eq!(document.layer(upper).unwrap().opacity, 1.0);
        assert!(document.redo());
        assert_eq!(document.layer(upper).unwrap().opacity, 0.4);
    }

    #[test]
    fn normal_layer_order_changes_without_rewriting_stroke_geometry() {
        let mut document = StrokeDocument::default();
        let point_count = document.points().len();
        let second = document.add_layer("Second");
        let third = document.add_layer("Third");
        assert_eq!(document.layer_index(second), Some(1));
        assert_eq!(document.layer_index(third), Some(2));
        assert!(document.move_layer(third, 0));
        assert_eq!(document.layer_index(third), Some(0));
        assert_eq!(document.points().len(), point_count);
        assert!(document.undo());
        assert_eq!(document.layer_index(third), Some(2));
    }

    #[test]
    fn device_recovery_never_expands_completed_history_into_the_live_overlay() {
        let mut document = StrokeDocument::default();
        for x in 0..32 {
            let (stroke, _) = document.begin_stroke(point(x as f32), profile());
            document.end_stroke(stroke).unwrap();
        }
        document.regenerate_after_device_loss();
        let mut ranges = Vec::new();
        document.overlay_segment_ranges(&mut ranges);
        assert!(ranges.is_empty());
        assert!(document
            .strokes()
            .iter()
            .all(|stroke| stroke.state == StrokeState::PendingCache));
    }

    #[test]
    fn rgba_material_is_premultiplied() {
        let material = RgbaMaterial::from_linear_rgba([0.8, 0.4, 0.2, 0.5]);
        assert_eq!(material.premultiplied_linear_rgba, [0.4, 0.2, 0.1, 0.5]);
    }

    #[test]
    fn versioned_pigment_payload_identity_is_immutable() {
        let mut document = StrokeDocument::default();
        let reference = PaintMaterialRef {
            model: PaintModelId(91),
            model_version: 4,
            material: PaintMaterialId(12),
        };
        let record = PaintMaterialRecord {
            reference,
            payload: PaintMaterialPayload::ModelData {
                schema_version: 2,
                bytes: vec![3, 1, 4, 1, 5],
            },
        };
        document.add_paint_material(record.clone()).unwrap();
        document.add_paint_material(record.clone()).unwrap();
        assert_eq!(document.paint_materials().get(reference), Some(&record));

        let changed = PaintMaterialRecord {
            payload: PaintMaterialPayload::ModelData {
                schema_version: 2,
                bytes: vec![9],
            },
            ..record
        };
        assert!(document.add_paint_material(changed).is_err());
    }

    #[test]
    fn document_rgba_recipes_restore_their_exact_runtime_material_ids() {
        let mut document = StrokeDocument::default();
        let reference = PaintMaterialRef {
            model: RGBA_PAINT_MODEL_ID,
            model_version: RgbaPaintModel::MODEL_VERSION,
            material: PaintMaterialId(7),
        };
        let rgba = [0.2, 0.1, 0.05, 0.4];
        document
            .add_paint_material(PaintMaterialRecord {
                reference,
                payload: PaintMaterialPayload::PremultipliedLinearRgba(rgba),
            })
            .unwrap();
        let mut runtime = RgbaPaintModel::default();
        runtime.sync_from_document(document.paint_materials());
        assert_eq!(runtime.materials()[7].premultiplied_linear_rgba, rgba);
    }

    #[test]
    fn duplicate_samples_do_not_grow_the_queue() {
        let mut store = StrokeStore::default();
        let (stroke, _) = store.begin_stroke(point(0.0), profile());
        assert!(store.append_point(stroke, point(0.0)).is_none());
        assert_eq!(store.points().len(), 1);
        assert_eq!(store.segments().len(), 1);
    }

    #[test]
    fn thousand_sample_synthetic_batch_keeps_the_exact_tip() {
        let mut store = StrokeStore::default();
        let (stroke, _) = store.begin_stroke(point(0.0), profile());
        for index in 1_u16..=1_000 {
            store.append_point(stroke, point(f32::from(index))).unwrap();
        }

        assert_eq!(store.points().len(), 1_001);
        assert_eq!(store.segments().len(), 1_001);
        assert_eq!(store.segments().last().unwrap().end, 1_000);
        assert_eq!(store.points().last().unwrap().position.x, 1_000.0);
    }

    #[test]
    fn undo_redo_preserves_vector_history_and_invalidates_the_same_tiles() {
        let mut document = StrokeDocument::default();
        let (stroke, _) = document.begin_stroke(point(0.0), profile());
        document.append_point(stroke, point(300.0)).unwrap();
        document.end_stroke(stroke).unwrap();
        let point_count = document.points().len();
        let affected = document.stroke(stroke).unwrap().affected_tiles.clone();
        let committed_revision = document.revision();

        let mut initial = Vec::new();
        document.drain_invalidations(&mut initial);
        assert_eq!(
            initial.iter().map(|item| item.key).collect::<BTreeSet<_>>(),
            affected.iter().copied().collect()
        );

        assert!(document.undo());
        assert!(document.revision() > committed_revision);
        assert_eq!(document.points().len(), point_count);
        assert!(!document.stroke(stroke).unwrap().visible);
        let mut undone = Vec::new();
        document.drain_invalidations(&mut undone);
        assert_eq!(
            undone.iter().map(|item| item.key).collect::<BTreeSet<_>>(),
            affected.iter().copied().collect()
        );

        assert!(document.redo());
        assert!(document.stroke(stroke).unwrap().visible);
        assert_eq!(document.points().len(), point_count);
    }

    #[test]
    fn active_overlay_retires_only_after_every_tile_revision_is_ready() {
        let mut document = StrokeDocument::default();
        let (stroke, _) = document.begin_stroke(point(0.0), profile());
        document.append_point(stroke, point(300.0)).unwrap();
        document.end_stroke(stroke).unwrap();
        let required = document.stroke(stroke).unwrap().affected_tiles.clone();
        let revision = document.stroke(stroke).unwrap().revision;
        let mut ranges = Vec::new();
        document.overlay_segment_ranges(&mut ranges);
        assert!(!ranges.is_empty());

        let first = required[0];
        document.refresh_cache_handoffs(|key, ready_revision| {
            key == first && ready_revision == revision
        });
        document.overlay_segment_ranges(&mut ranges);
        assert!(!ranges.is_empty());

        document.refresh_cache_handoffs(|_, ready_revision| ready_revision == revision);
        document.overlay_segment_ranges(&mut ranges);
        assert!(ranges.is_empty());
        assert_eq!(document.stroke(stroke).unwrap().state, StrokeState::Cached);
    }

    #[test]
    fn tile_spatial_index_contains_only_local_segments() {
        let mut document = StrokeDocument::default();
        let (stroke, _) = document.begin_stroke(point(0.0), profile());
        for x in 1..=1_000 {
            document.append_point(stroke, point(x as f32)).unwrap();
        }
        document.end_stroke(stroke).unwrap();

        let left = document.tile_strokes(TileKey::new(LayerId(0), 0, 0));
        let right = document.tile_strokes(TileKey::new(LayerId(0), 3, 0));
        assert_eq!(left.len(), 1);
        assert_eq!(right.len(), 1);
        assert!(left[0].segments.len() < document.strokes()[0].segment_count as usize);
        assert!(right[0].segments.len() < document.strokes()[0].segment_count as usize);
    }
}
