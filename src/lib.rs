#![doc = "Low-latency GPU stroke rendering and persistent vector canvas for Hamerons."]

extern crate alloc;

mod diagnostics;
mod effects;
mod input;
mod model;
mod render;
mod tiles;

pub use diagnostics::{StrokeTelemetry, StrokeTelemetrySnapshot};
pub use effects::{
    EffectGraph, EffectNode, EffectNodeId, ScratchPlaneLease, ScratchPlanePool,
    PASS_THROUGH_EFFECT_ID,
};
pub use input::{StrokeInputBlocker, StrokeInputSystems};
pub use model::{
    BrushFootprint, BrushProfile, BrushSizeSpace, CanvasLayer, CheckpointRequest, DepositionMode,
    DepositionOrdering, DocumentCheckpointManager, DocumentCompatibilityIssue, DocumentIoError,
    DocumentSaveReport, EffectDescriptor, EffectDomain, EffectId, EffectInfluence, EffectRegistry,
    LayerCompositeMode, LayerId, LoadedStrokeDocument, PaintMaterialId, PaintMaterialLibrary,
    PaintMaterialPayload, PaintMaterialRecord, PaintMaterialRef, PaintModelDescriptor,
    PaintModelId, PaintModelRegistry, PaintPlaneClearValue, PaintPlaneDescriptor, PaintPlaneFormat,
    RgbaMaterial, RgbaPaintModel, StrokeDelta, StrokeDeltaBatch, StrokeDocument, StrokeId,
    StrokeMetadata, StrokePoint, StrokePointResampler, StrokeRendererSettings, StrokeSegment,
    StrokeState, StrokeStore, DOCUMENT_SCHEMA_VERSION, RGBA_PAINT_MODEL_ID,
};
pub use tiles::{
    neighbor_halo, CanvasTileCache, TileCacheState, TileCacheStats, TileKey, TilePlaneAllocation,
    TileSurfaceLayout,
};

use bevy_app::{App, Plugin, PostUpdate, PreUpdate, Startup, Update};
use bevy_asset::embedded_asset;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_input::InputSystems;

use diagnostics::log_stroke_diagnostics;
use input::{collect_pen_strokes, PenContacts};
use tiles::{maintain_canvas_tiles, TileFeedback};

/// Collects first-class pen input into append-only stroke geometry.
#[derive(Default)]
pub struct StrokeInputPlugin;

impl Plugin for StrokeInputPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<StrokeRendererSettings>()
            .init_resource::<StrokeStore>()
            .init_resource::<StrokeDeltaBatch>()
            .init_resource::<StrokeTelemetry>()
            .init_resource::<PenContacts>()
            .init_resource::<StrokeInputBlocker>()
            .add_systems(
                PreUpdate,
                collect_pen_strokes
                    .after(InputSystems)
                    .in_set(StrokeInputSystems::Collect),
            )
            .add_systems(Update, log_stroke_diagnostics);
    }
}

/// Installs the procedural 2D coverage and RGBA deposition pipeline.
#[derive(Default)]
pub struct StrokeRenderPlugin;

impl Plugin for StrokeRenderPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/stroke_coverage.wgsl");
        embedded_asset!(app, "shaders/tile_raster.wgsl");
        embedded_asset!(app, "shaders/tile_composite.wgsl");
        app.init_resource::<RgbaPaintModel>()
            .init_resource::<PaintModelRegistry>()
            .init_resource::<EffectRegistry>()
            .init_resource::<CanvasTileCache>()
            .init_resource::<TileFeedback>()
            .init_resource::<DocumentCheckpointManager>()
            .add_systems(Startup, register_rgba_paint_model)
            .add_systems(Update, sync_rgba_document_materials);
        app.add_systems(
            PostUpdate,
            maintain_canvas_tiles
                .after(bevy_transform::TransformSystems::Propagate)
                .after(bevy_camera::CameraUpdateSystems),
        );
        render::build_render_app(app);
    }
}

/// Complete Phase 3 plugin: live ink, persistent tiles, layers, and durable documents.
#[derive(Default)]
pub struct HameronsStrokeRenderPlugin;

impl Plugin for HameronsStrokeRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((StrokeInputPlugin, StrokeRenderPlugin));
    }
}

/// Short compatibility name for [`HameronsStrokeRenderPlugin`].
pub use HameronsStrokeRenderPlugin as HameronsStrokePlugin;

/// Specification name for the first-class pen input adapter.
pub use StrokeInputPlugin as PenStrokeAdapterPlugin;

fn register_rgba_paint_model(
    mut paint_models: bevy_ecs::system::ResMut<PaintModelRegistry>,
    mut effect_registry: bevy_ecs::system::ResMut<EffectRegistry>,
) {
    paint_models.register(RgbaPaintModel::descriptor());
    effect_registry.register(effects::pass_through_descriptor());
}

fn sync_rgba_document_materials(
    mut document: bevy_ecs::system::ResMut<StrokeDocument>,
    mut rgba: bevy_ecs::system::ResMut<RgbaPaintModel>,
) {
    rgba.sync_from_document(document.paint_materials());
    document.sync_rgba_materials(&rgba);
}
