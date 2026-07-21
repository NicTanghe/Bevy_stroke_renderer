use std::{mem::size_of, ops::Range};

#[cfg(test)]
use std::time::Duration;

use bevy_app::App;
use bevy_asset::{AssetServer, Handle};
use bevy_camera::Camera2d;
use bevy_core_pipeline::core_2d::{Transparent2d, CORE_2D_DEPTH_FORMAT};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    name::Name,
    prelude::With,
    query::ROQueryItem,
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{lifetimeless::SRes, Commands, Local, Query, Res, ResMut, SystemParamItem},
};
use bevy_math::{FloatOrd, Vec2, Vec4};
use bevy_render::{
    render_phase::{
        AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
        RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
    },
    render_resource::{
        binding_types::storage_buffer_read_only, BindGroup, BindGroupEntries,
        BindGroupLayoutDescriptor, BindGroupLayoutEntries, BlendState, BufferUsages,
        CachedRenderPipelineId, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
        DepthStencilState, FragmentState, MultisampleState, PipelineCache, PrimitiveState,
        RawBufferVec, RenderPipelineDescriptor, ShaderStages, ShaderType,
        SpecializedRenderPipeline, SpecializedRenderPipelines, StencilFaceState, StencilState,
        VertexState,
    },
    renderer::{RenderAdapterInfo, RenderDevice, RenderGraph, RenderQueue},
    sync_world::{MainEntity, TemporaryRenderEntity},
    view::{ExtractedView, Msaa},
    Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::Shader;
use bevy_sprite_render::{
    init_mesh_2d_pipeline, Mesh2dPipeline, Mesh2dPipelineKey, SetMesh2dViewBindGroup,
};
use bytemuck::{Pod, Zeroable};
use tracing::{info, warn};

use crate::{
    LayerId, RgbaPaintModel, StrokePoint, StrokeRendererSettings, StrokeSegment, StrokeStore,
    StrokeTelemetry,
};

mod tiles;

pub(crate) fn build_render_app(app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        warn!("StrokeRenderPlugin could not find RenderApp; add it after Bevy's RenderPlugin");
        return;
    };

    render_app
        .add_render_command::<Transparent2d, DrawStrokeOverlay>()
        .add_render_command::<Transparent2d, tiles::DrawTileCanvas>()
        .init_resource::<ExtractedStrokeDelta>()
        .init_resource::<tiles::ExtractedTileBatch>()
        .init_gpu_resource::<StrokeGpuBuffers>()
        .init_gpu_resource::<tiles::TileGpuCache>()
        .init_gpu_resource::<SpecializedRenderPipelines<StrokePipeline>>()
        .init_gpu_resource::<SpecializedRenderPipelines<tiles::TileCompositePipeline>>()
        .add_systems(
            RenderStartup,
            (
                init_stroke_bind_group_layout,
                init_stroke_pipeline
                    .after(init_stroke_bind_group_layout)
                    .after(init_mesh_2d_pipeline),
                tiles::init_tile_pipelines.after(init_mesh_2d_pipeline),
                log_render_adapter,
            ),
        )
        .add_systems(
            ExtractSchedule,
            (extract_stroke_delta, tiles::extract_tile_batch),
        )
        .add_systems(
            Render,
            (
                prepare_stroke_buffers.in_set(RenderSystems::PrepareResources),
                tiles::prepare_tile_buffers
                    .after(prepare_stroke_buffers)
                    .in_set(RenderSystems::PrepareResources),
                prepare_stroke_bind_group.in_set(RenderSystems::PrepareBindGroups),
                tiles::prepare_tile_bind_groups
                    .after(prepare_stroke_bind_group)
                    .in_set(RenderSystems::PrepareBindGroups),
                tiles::queue_tile_canvas.in_set(RenderSystems::Queue),
                queue_stroke_overlay.in_set(RenderSystems::Queue),
            ),
        )
        .add_systems(
            RenderGraph,
            tiles::rasterize_tiles.before(bevy_core_pipeline::schedule::camera_driver),
        );
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, ShaderType)]
pub(super) struct GpuStrokePoint {
    position: Vec2,
    half_width: f32,
    flow: f32,
    orientation: Vec2,
    twist_radians: f32,
    aspect_ratio: f32,
}

impl From<&StrokePoint> for GpuStrokePoint {
    fn from(point: &StrokePoint) -> Self {
        Self {
            position: point.position,
            half_width: point.half_width,
            flow: point.flow,
            orientation: point.orientation,
            twist_radians: point.twist_radians,
            aspect_ratio: point.aspect_ratio,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, ShaderType)]
pub(super) struct GpuStrokeSegment {
    start: u32,
    end: u32,
    material: u32,
    model_and_deposition: u32,
    layer: u32,
}

impl From<&StrokeSegment> for GpuStrokeSegment {
    fn from(segment: &StrokeSegment) -> Self {
        Self {
            start: segment.start,
            end: segment.end,
            material: segment.material.0,
            model_and_deposition: (segment.model.0 & 0x7fff_ffff)
                | ((segment.deposition as u32) << 31),
            layer: segment.layer.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, ShaderType)]
pub(super) struct GpuRgbaMaterial {
    color: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, ShaderType)]
struct GpuCanvasLayer {
    opacity: f32,
    visible: u32,
    padding: [u32; 2],
}

#[derive(Resource, Default)]
struct ExtractedStrokeDelta {
    reset: bool,
    points: Vec<GpuStrokePoint>,
    segments: Vec<GpuStrokeSegment>,
    materials: Vec<GpuRgbaMaterial>,
    materials_changed: bool,
    layers: Vec<GpuCanvasLayer>,
    layers_changed: bool,
    overlay_ranges: Vec<(LayerId, Range<u32>)>,
    initial_point_capacity: usize,
    initial_segment_capacity: usize,
    telemetry: StrokeTelemetry,
}

#[derive(Default)]
struct ExtractionState {
    geometry_generation: Option<u64>,
    point_count: usize,
    segment_count: usize,
    material_revision: Option<u64>,
    layer_revision: Option<u64>,
}

#[derive(Component)]
struct StrokeOverlayRenderEntity {
    layer: LayerId,
    order: u32,
    layer_count: u32,
}

#[derive(Component)]
pub(super) struct TileCanvasRenderEntity {
    pub(super) instances: Range<u32>,
    pub(super) order: u32,
    pub(super) layer_count: u32,
}

fn extract_stroke_delta(
    mut commands: Commands,
    store: Extract<Res<StrokeStore>>,
    materials: Extract<Res<RgbaPaintModel>>,
    settings: Extract<Res<StrokeRendererSettings>>,
    telemetry: Extract<Res<StrokeTelemetry>>,
    mut state: Local<ExtractionState>,
    mut delta: ResMut<ExtractedStrokeDelta>,
) {
    let reset = state.geometry_generation != Some(store.geometry_generation())
        || state.point_count > store.points().len()
        || state.segment_count > store.segments().len();
    let point_start = if reset { 0 } else { state.point_count };
    let segment_start = if reset { 0 } else { state.segment_count };

    delta.points.clear();
    delta.points.extend(
        store.points()[point_start..]
            .iter()
            .map(GpuStrokePoint::from),
    );
    delta.segments.clear();
    delta.segments.extend(
        store.segments()[segment_start..]
            .iter()
            .map(GpuStrokeSegment::from),
    );

    let material_changed = state.material_revision != Some(materials.revision());
    if material_changed {
        delta.materials.clear();
        delta.materials.extend(
            materials
                .materials()
                .iter()
                .map(|material| GpuRgbaMaterial {
                    color: Vec4::from_array(material.premultiplied_linear_rgba),
                }),
        );
    }

    let layers_changed = state.layer_revision != Some(store.layer_revision());
    if layers_changed {
        delta.layers.clear();
        let layer_count = store
            .layers()
            .iter()
            .map(|layer| layer.id.0 as usize + 1)
            .max()
            .unwrap_or(1);
        delta.layers.resize(
            layer_count,
            GpuCanvasLayer {
                opacity: 0.0,
                visible: 0,
                padding: [0; 2],
            },
        );
        for layer in store.layers() {
            delta.layers[layer.id.0 as usize] = GpuCanvasLayer {
                opacity: layer.opacity,
                visible: u32::from(layer.visible),
                padding: [0; 2],
            };
        }
    }

    state.geometry_generation = Some(store.geometry_generation());
    state.point_count = store.points().len();
    state.segment_count = store.segments().len();
    state.material_revision = Some(materials.revision());
    state.layer_revision = Some(store.layer_revision());

    if (reset || point_start < store.points().len())
        && let Some(newest) = store.latest_sample_received()
    {
        let oldest = store.input_batch_started().unwrap_or(newest);
        telemetry.record_extraction_ages(oldest.elapsed(), newest.elapsed());
    }

    store.overlay_layer_segment_ranges(&mut delta.overlay_ranges);
    delta.reset = reset;
    delta.materials_changed = material_changed;
    delta.layers_changed = layers_changed;
    delta.initial_point_capacity = settings.initial_point_capacity;
    delta.initial_segment_capacity = settings.initial_segment_capacity;
    delta.telemetry = telemetry.clone();

    if !store.segments().is_empty() {
        // Transparent2d stores phase items by render entity. One entity per
        // layer lets cached tiles and the live overlay interleave in document
        // order without one phase item replacing another.
        let layer_count = store.layers().len() as u32;
        for (order, layer) in store.layers().iter().enumerate() {
            commands.spawn((
                Name::new(format!("HameronsStrokeOverlayLayer{}", layer.id.0)),
                StrokeOverlayRenderEntity {
                    layer: layer.id,
                    order: order as u32,
                    layer_count,
                },
                MainEntity::from(Entity::PLACEHOLDER),
                TemporaryRenderEntity::default(),
            ));
        }
    }
}

#[derive(Resource)]
pub(super) struct StrokeGpuBuffers {
    pub(super) points: RawBufferVec<GpuStrokePoint>,
    pub(super) segments: RawBufferVec<GpuStrokeSegment>,
    pub(super) materials: RawBufferVec<GpuRgbaMaterial>,
    layers: RawBufferVec<GpuCanvasLayer>,
    bind_group: Option<BindGroup>,
    bind_group_dirty: bool,
    overlay_ranges: Vec<(LayerId, Range<u32>)>,
    pub(super) reallocation_generation: u64,
}

impl Default for StrokeGpuBuffers {
    fn default() -> Self {
        let mut points = RawBufferVec::new(BufferUsages::STORAGE);
        points.set_label(Some("Hamerons stroke points"));
        let mut segments = RawBufferVec::new(BufferUsages::STORAGE);
        segments.set_label(Some("Hamerons stroke segment metadata"));
        let mut materials = RawBufferVec::new(BufferUsages::STORAGE);
        materials.set_label(Some("Hamerons RGBA paint materials"));
        let mut layers = RawBufferVec::new(BufferUsages::STORAGE);
        layers.set_label(Some("Hamerons canvas layers"));
        Self {
            points,
            segments,
            materials,
            layers,
            bind_group: None,
            bind_group_dirty: true,
            overlay_ranges: Vec::new(),
            reallocation_generation: 0,
        }
    }
}

#[derive(Default)]
struct BufferUpdate {
    uploaded_bytes: usize,
    reallocated: bool,
}

fn append_buffer<T: Pod + Copy>(
    buffer: &mut RawBufferVec<T>,
    values: &[T],
    reset: bool,
    minimum_capacity: usize,
    device: &RenderDevice,
    queue: &RenderQueue,
) -> BufferUpdate {
    if reset {
        buffer.clear();
    }
    let first_new = buffer.len();
    for value in values {
        buffer.push(*value);
    }

    if buffer.is_empty() {
        return BufferUpdate::default();
    }

    let needs_buffer = buffer.buffer().is_none();
    let needs_growth = buffer.len() > buffer.capacity();
    let reallocated = needs_buffer || needs_growth;
    if reallocated {
        let requested = buffer.len().max(minimum_capacity).max(1);
        buffer.reserve(requested.next_power_of_two(), device);
    }

    if reset || reallocated {
        buffer.write_buffer(device, queue);
        BufferUpdate {
            uploaded_bytes: buffer.len() * size_of::<T>(),
            reallocated,
        }
    } else if first_new < buffer.len() {
        buffer
            .write_buffer_range(queue, first_new..buffer.len())
            .expect("preallocated stroke buffer rejected an in-capacity append");
        BufferUpdate {
            uploaded_bytes: (buffer.len() - first_new) * size_of::<T>(),
            reallocated: false,
        }
    } else {
        BufferUpdate::default()
    }
}

fn replace_buffer<T: Pod + Copy>(
    buffer: &mut RawBufferVec<T>,
    values: &[T],
    device: &RenderDevice,
    queue: &RenderQueue,
) -> BufferUpdate {
    buffer.clear();
    for value in values {
        buffer.push(*value);
    }
    if buffer.is_empty() {
        return BufferUpdate::default();
    }
    let reallocated = buffer.buffer().is_none() || buffer.len() > buffer.capacity();
    if reallocated {
        buffer.reserve(buffer.len().max(4).next_power_of_two(), device);
    }
    buffer.write_buffer(device, queue);
    BufferUpdate {
        uploaded_bytes: buffer.len() * size_of::<T>(),
        reallocated,
    }
}

fn prepare_stroke_buffers(
    delta: Res<ExtractedStrokeDelta>,
    mut buffers: ResMut<StrokeGpuBuffers>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let points = append_buffer(
        &mut buffers.points,
        &delta.points,
        delta.reset,
        delta.initial_point_capacity,
        &device,
        &queue,
    );
    let segments = append_buffer(
        &mut buffers.segments,
        &delta.segments,
        delta.reset,
        delta.initial_segment_capacity,
        &device,
        &queue,
    );
    let materials = if delta.materials_changed {
        replace_buffer(&mut buffers.materials, &delta.materials, &device, &queue)
    } else {
        BufferUpdate::default()
    };
    let layers = if delta.layers_changed {
        replace_buffer(&mut buffers.layers, &delta.layers, &device, &queue)
    } else {
        BufferUpdate::default()
    };

    let reallocations = u64::from(points.reallocated)
        + u64::from(segments.reallocated)
        + u64::from(materials.reallocated)
        + u64::from(layers.reallocated);
    buffers.bind_group_dirty |= reallocations > 0 || delta.layers_changed;
    if reallocations > 0 {
        buffers.reallocation_generation = buffers.reallocation_generation.wrapping_add(1);
    }
    buffers.overlay_ranges.clear();
    buffers
        .overlay_ranges
        .extend(delta.overlay_ranges.iter().cloned());
    delta.telemetry.record_upload(
        points.uploaded_bytes
            + segments.uploaded_bytes
            + materials.uploaded_bytes
            + layers.uploaded_bytes,
        reallocations,
    );
}

#[derive(Resource)]
struct StrokeBindGroupLayout(BindGroupLayoutDescriptor);

fn init_stroke_bind_group_layout(mut commands: Commands) {
    commands.insert_resource(StrokeBindGroupLayout(BindGroupLayoutDescriptor::new(
        "Hamerons stroke storage layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                storage_buffer_read_only::<GpuStrokePoint>(false),
                storage_buffer_read_only::<GpuStrokeSegment>(false),
                storage_buffer_read_only::<GpuRgbaMaterial>(false),
                storage_buffer_read_only::<GpuCanvasLayer>(false),
            ),
        ),
    )));
}

fn prepare_stroke_bind_group(
    mut buffers: ResMut<StrokeGpuBuffers>,
    layout: Res<StrokeBindGroupLayout>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
) {
    if !buffers.bind_group_dirty {
        return;
    }
    let (Some(points), Some(segments), Some(materials), Some(layers)) = (
        buffers.points.binding(),
        buffers.segments.binding(),
        buffers.materials.binding(),
        buffers.layers.binding(),
    ) else {
        return;
    };

    buffers.bind_group = Some(device.create_bind_group(
        "Hamerons stroke storage bind group",
        &pipeline_cache.get_bind_group_layout(&layout.0),
        &BindGroupEntries::sequential((points, segments, materials, layers)),
    ));
    buffers.bind_group_dirty = false;
}

#[derive(Clone, Resource)]
struct StrokePipeline {
    mesh_pipeline: Mesh2dPipeline,
    storage_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

fn init_stroke_pipeline(
    mut commands: Commands,
    mesh_pipeline: Res<Mesh2dPipeline>,
    layout: Res<StrokeBindGroupLayout>,
    asset_server: Res<AssetServer>,
) {
    commands.insert_resource(StrokePipeline {
        mesh_pipeline: mesh_pipeline.clone(),
        storage_layout: layout.0.clone(),
        shader: asset_server.load("embedded://hamerons_stroke_render/shaders/stroke_coverage.wgsl"),
    });
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct StrokePipelineKey {
    mesh_key: Mesh2dPipelineKey,
}

impl SpecializedRenderPipeline for StrokePipeline {
    type Key = StrokePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("Hamerons procedural stroke pipeline".into()),
            layout: vec![
                self.mesh_pipeline.view_layout.clone(),
                self.storage_layout.clone(),
            ],
            vertex: VertexState {
                shader: self.shader.clone(),
                entry_point: Some("vertex".into()),
                shader_defs: Vec::new(),
                buffers: Vec::new(),
                constants: Vec::new(),
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                entry_point: Some("fragment_rgba".into()),
                shader_defs: Vec::new(),
                targets: vec![Some(ColorTargetState {
                    format: key.mesh_key.target_format(),
                    blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                constants: Vec::new(),
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: CORE_2D_DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::Always),
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState {
                count: key.mesh_key.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            ..Default::default()
        }
    }
}

type DrawStrokeOverlay = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    SetStrokeBindGroup<1>,
    DrawStrokes,
);

struct SetStrokeBindGroup<const I: usize>;

impl<const I: usize, P: PhaseItem> RenderCommand<P> for SetStrokeBindGroup<I> {
    type Param = SRes<StrokeGpuBuffers>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        _view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _entity: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        buffers: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let buffers = buffers.into_inner();
        let Some(bind_group) = &buffers.bind_group else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(I, bind_group, &[]);
        RenderCommandResult::Success
    }
}

struct DrawStrokes;

impl<P: PhaseItem> RenderCommand<P> for DrawStrokes {
    type Param = SRes<StrokeGpuBuffers>;
    type ViewQuery = ();
    type ItemQuery = &'static StrokeOverlayRenderEntity;

    fn render<'w>(
        _item: &P,
        _view: ROQueryItem<'w, '_, Self::ViewQuery>,
        entity: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        buffers: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let buffers = buffers.into_inner();
        if buffers.overlay_ranges.is_empty() {
            return RenderCommandResult::Success;
        }
        let Some(renderer) = entity else {
            return RenderCommandResult::Skip;
        };
        for (layer, range) in &buffers.overlay_ranges {
            if *layer == renderer.layer {
                pass.draw(0..6, range.clone());
            }
        }
        for (layer, range) in &buffers.overlay_ranges {
            if *layer == renderer.layer {
                pass.draw(6..12, range.clone());
            }
        }
        RenderCommandResult::Success
    }
}

#[allow(clippy::too_many_arguments)]
fn queue_stroke_overlay(
    draw_functions: Res<DrawFunctions<Transparent2d>>,
    pipeline: Res<StrokePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<StrokePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    buffers: Res<StrokeGpuBuffers>,
    renderers: Query<(Entity, &MainEntity, &StrokeOverlayRenderEntity)>,
    mut phases: ResMut<ViewSortedRenderPhases<Transparent2d>>,
    views: Query<(&ExtractedView, &Msaa), With<Camera2d>>,
) {
    if buffers.overlay_ranges.is_empty() || buffers.bind_group.is_none() {
        return;
    }
    let draw_function = draw_functions
        .read()
        .get_id::<DrawStrokeOverlay>()
        .expect("stroke draw command was not registered");

    for (view, msaa) in &views {
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let mesh_key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples())
            | Mesh2dPipelineKey::from_target_format(view.target_format);
        let pipeline_id: CachedRenderPipelineId =
            pipelines.specialize(&pipeline_cache, &pipeline, StrokePipelineKey { mesh_key });

        for (render_entity, main_entity, renderer) in &renderers {
            if !buffers
                .overlay_ranges
                .iter()
                .any(|(layer, _)| *layer == renderer.layer)
            {
                continue;
            }
            phase.add_transient(Transparent2d {
                entity: (render_entity, *main_entity),
                draw_function,
                pipeline: pipeline_id,
                sort_key: layer_phase_sort_key(renderer.order, renderer.layer_count, true),
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                extracted_index: usize::MAX,
                indexed: false,
            });
        }
    }
}

pub(super) fn layer_phase_sort_key(order: u32, layer_count: u32, overlay: bool) -> FloatOrd {
    let layer_count = layer_count.max(1);
    let rank = (order.saturating_mul(2) + u32::from(overlay) + 1) as f32
        / (layer_count.saturating_mul(2) + 1) as f32;
    FloatOrd(f32::MAX * 0.5 + f32::MAX * 0.25 * rank)
}

fn log_render_adapter(adapter: Res<RenderAdapterInfo>) {
    info!(
        target: "hamerons_stroke_render::latency",
        adapter = %adapter.name,
        backend = ?adapter.backend,
        device_type = ?adapter.device_type,
        "GPU stroke renderer initialized"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_layouts_remain_shader_compatible() {
        assert_eq!(size_of::<GpuStrokePoint>(), 32);
        assert_eq!(size_of::<GpuStrokeSegment>(), 20);
        assert_eq!(size_of::<GpuRgbaMaterial>(), 16);
        assert_eq!(size_of::<GpuCanvasLayer>(), 16);

        let packed = GpuStrokeSegment::from(&StrokeSegment {
            start: 0,
            end: 1,
            model: crate::PaintModelId(7),
            material: crate::PaintMaterialId(3),
            deposition: crate::DepositionMode::Erase,
            layer: crate::LayerId(5),
        });
        assert_eq!(packed.model_and_deposition, 0x8000_0007);
        assert_eq!(packed.layer, 5);
    }

    #[test]
    fn extracted_age_duration_fits_telemetry() {
        let telemetry = StrokeTelemetry::default();
        telemetry.record_extraction_ages(Duration::from_millis(5), Duration::from_millis(3));
        assert_eq!(
            telemetry.snapshot().extracted_sample_age,
            Duration::from_millis(3)
        );
        assert_eq!(
            telemetry.snapshot().oldest_extracted_sample_age,
            Duration::from_millis(5)
        );
    }

    #[test]
    fn cached_and_live_batches_interleave_in_layer_order() {
        let lower_tile = layer_phase_sort_key(0, 3, false);
        let lower_live = layer_phase_sort_key(0, 3, true);
        let middle_tile = layer_phase_sort_key(1, 3, false);
        let middle_live = layer_phase_sort_key(1, 3, true);
        let upper_tile = layer_phase_sort_key(2, 3, false);
        assert!(lower_tile < lower_live);
        assert!(lower_live < middle_tile);
        assert!(middle_tile < middle_live);
        assert!(middle_live < upper_tile);
    }

    #[test]
    fn stroke_shader_is_valid_wgsl() {
        let source = include_str!("../shaders/stroke_coverage.wgsl").replacen(
            "#import bevy_render::view::View",
            "struct View { clip_from_world: mat4x4<f32>, viewport: vec4<f32>, }",
            1,
        );
        let module =
            naga::front::wgsl::parse_str(&source).expect("stroke shader must parse as WGSL");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("stroke shader must pass Naga validation");
    }
}
