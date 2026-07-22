use std::{
    collections::{HashMap, HashSet},
    mem::size_of,
    ops::Range,
};

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
        binding_types::{storage_buffer_read_only, texture_2d_array},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        BlendComponent, BlendFactor, BlendOperation, BlendState, BufferUsages,
        CachedRenderPipelineId, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
        DepthStencilState, Extent3d, FragmentState, LoadOp, MultisampleState, Operations,
        PipelineCache, PrimitiveState, RawBufferVec, RenderPassColorAttachment,
        RenderPassDescriptor, RenderPipelineDescriptor, ShaderStages, ShaderType,
        SpecializedRenderPipeline, SpecializedRenderPipelines, StencilFaceState, StencilState,
        StoreOp, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
        TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension, VertexState,
    },
    renderer::{RenderAdapterInfo, RenderContext, RenderDevice, RenderGraph, RenderQueue},
    sync_world::{MainEntity, TemporaryRenderEntity},
    view::{ExtractedView, Msaa, ViewUniformOffset},
    Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::Shader;
use bevy_sprite_render::{
    init_mesh_2d_pipeline, Mesh2dPipeline, Mesh2dPipelineKey, Mesh2dViewBindGroup,
    SetMesh2dViewBindGroup,
};
use bytemuck::{Pod, Zeroable};
use tracing::{info, warn};

use crate::{
    LayerId, RgbaPaintModel, StrokePoint, StrokeRendererSettings, StrokeSegment, StrokeStore,
    StrokeTelemetry,
};

// Live coverage is premultiplied before it is resolved over the cached layer.
// An 8-bit mask cannot represent low-alpha dark colors accurately: its RGB
// channels round to zero while alpha remains nonzero, effectively turning the
// source into translucent black. Keep each stroke mask in floating point, then
// quantize only after that whole stroke is resolved. The persistent tile replay
// uses the same per-stroke RGBA8 checkpoints, so retiring a live stroke cannot
// change its color.
const LIVE_MASK_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
const RESOLVED_LAYER_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

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
        .init_resource::<QueuedLiveLayers>()
        .init_gpu_resource::<StrokeGpuBuffers>()
        .init_gpu_resource::<LiveStrokeMasks>()
        .init_gpu_resource::<tiles::TileGpuCache>()
        .init_gpu_resource::<SpecializedRenderPipelines<StrokePipeline>>()
        .init_gpu_resource::<SpecializedRenderPipelines<tiles::TileCompositePipeline>>()
        .add_systems(
            RenderStartup,
            (
                init_stroke_bind_group_layout,
                init_stroke_pipeline
                    .after(init_stroke_bind_group_layout)
                    .after(init_mesh_2d_pipeline)
                    .after(tiles::init_tile_pipelines),
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
                prepare_live_stroke_masks
                    .after(prepare_stroke_buffers)
                    .in_set(RenderSystems::PrepareResources),
                tiles::prepare_tile_buffers
                    .after(prepare_stroke_buffers)
                    .in_set(RenderSystems::PrepareResources),
                prepare_stroke_bind_group.in_set(RenderSystems::PrepareBindGroups),
                prepare_live_stroke_mask_bind_groups
                    .after(prepare_live_stroke_masks)
                    .in_set(RenderSystems::PrepareBindGroups),
                tiles::prepare_tile_bind_groups
                    .after(prepare_stroke_bind_group)
                    .in_set(RenderSystems::PrepareBindGroups),
                queue_stroke_overlay.in_set(RenderSystems::Queue),
                tiles::queue_tile_canvas
                    .after(queue_stroke_overlay)
                    .in_set(RenderSystems::Queue),
            ),
        )
        .add_systems(
            RenderGraph,
            (
                tiles::rasterize_tiles,
                rasterize_live_stroke_masks,
                rasterize_live_layers,
            )
                .chain()
                .before(bevy_core_pipeline::schedule::camera_driver),
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, ShaderType)]
struct GpuLiveStroke {
    mask_layer: u32,
    canvas_layer: u32,
    model_and_deposition: u32,
    padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, ShaderType)]
struct GpuLiveLayer {
    canvas_layer: u32,
    texture_layer: u32,
    opacity: f32,
    stroke_count: u32,
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
    pub(super) layer: LayerId,
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
        // layer lets the resolved active-layer surface replace that layer's
        // cached tile draw without disturbing document layer order.
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

struct LiveTextureArray {
    _texture: Texture,
    array_view: TextureView,
    layer_views: Vec<TextureView>,
}

struct LiveStrokeMaskView {
    width: u32,
    height: u32,
    stroke_capacity: u32,
    layer_capacity: u32,
    stroke_masks: LiveTextureArray,
    cached_layers: LiveTextureArray,
    resolved_layers: LiveTextureArray,
    resolve_bind_group: Option<BindGroup>,
    composite_bind_group: Option<BindGroup>,
}

#[derive(Resource)]
struct LiveStrokeMasks {
    views: HashMap<Entity, LiveStrokeMaskView>,
    active_layers: Vec<LayerId>,
    live_strokes: RawBufferVec<GpuLiveStroke>,
    live_layers: RawBufferVec<GpuLiveLayer>,
    warned_required_strokes: Option<u32>,
}

impl Default for LiveStrokeMasks {
    fn default() -> Self {
        let mut live_strokes = RawBufferVec::new(BufferUsages::STORAGE);
        live_strokes.set_label(Some("Hamerons live stroke resolve metadata"));
        let mut live_layers = RawBufferVec::new(BufferUsages::STORAGE);
        live_layers.set_label(Some("Hamerons live layer resolve metadata"));
        Self {
            views: HashMap::new(),
            active_layers: Vec::new(),
            live_strokes,
            live_layers,
            warned_required_strokes: None,
        }
    }
}

impl LiveStrokeMasks {
    fn active_layer_slot(&self, layer: LayerId) -> Option<u32> {
        self.active_layers
            .iter()
            .position(|candidate| *candidate == layer)
            .map(|index| index as u32)
    }

    fn has_active_layer(&self, layer: LayerId) -> bool {
        self.active_layers.contains(&layer)
    }
}

#[derive(Resource, Default)]
struct QueuedLiveLayers {
    by_view: HashMap<Entity, HashSet<LayerId>>,
}

impl QueuedLiveLayers {
    fn clear(&mut self) {
        self.by_view.clear();
    }

    fn insert(&mut self, view: Entity, layer: LayerId) {
        self.by_view.entry(view).or_default().insert(layer);
    }

    fn contains(&self, view: Entity, layer: LayerId) -> bool {
        self.by_view
            .get(&view)
            .is_some_and(|layers| layers.contains(&layer))
    }
}

fn create_live_texture_array(
    device: &RenderDevice,
    label: &'static str,
    width: u32,
    height: u32,
    layers: u32,
    format: TextureFormat,
) -> LiveTextureArray {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let array_view = texture.create_view(&TextureViewDescriptor {
        label: Some(label),
        dimension: Some(TextureViewDimension::D2Array),
        base_array_layer: 0,
        array_layer_count: Some(layers),
        ..Default::default()
    });
    let layer_views = (0..layers)
        .map(|layer| {
            texture.create_view(&TextureViewDescriptor {
                label: Some(label),
                dimension: Some(TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
        .collect();
    LiveTextureArray {
        _texture: texture,
        array_view,
        layer_views,
    }
}

fn prepare_live_stroke_masks(
    mut masks: ResMut<LiveStrokeMasks>,
    buffers: Res<StrokeGpuBuffers>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    views: Query<(Entity, &ExtractedView), With<Camera2d>>,
) {
    let active_views: HashSet<_> = views.iter().map(|(entity, _)| entity).collect();
    masks
        .views
        .retain(|entity, _| active_views.contains(entity));

    let maximum_layers = device.limits().max_texture_array_layers.max(1);
    let required_strokes = u32::try_from(buffers.overlay_ranges.len())
        .unwrap_or(u32::MAX)
        .max(1);
    let requested_stroke_capacity = required_strokes
        .checked_next_power_of_two()
        .unwrap_or(maximum_layers)
        .min(maximum_layers);
    if required_strokes > maximum_layers && masks.warned_required_strokes != Some(required_strokes)
    {
        warn!(
            required_strokes,
            maximum_layers, "live stroke mask array reached the device layer limit"
        );
        masks.warned_required_strokes = Some(required_strokes);
    } else if required_strokes <= maximum_layers {
        masks.warned_required_strokes = None;
    }

    let mut active_layers = Vec::new();
    let mut live_strokes = Vec::new();
    for (mask_layer, (layer, segment_range)) in buffers
        .overlay_ranges
        .iter()
        .take(maximum_layers as usize)
        .enumerate()
    {
        if !active_layers.contains(layer) {
            active_layers.push(*layer);
        }
        let model_and_deposition = buffers
            .segments
            .values()
            .get(segment_range.start as usize)
            .map_or(0, |segment| segment.model_and_deposition);
        live_strokes.push(GpuLiveStroke {
            mask_layer: mask_layer as u32,
            canvas_layer: layer.0,
            model_and_deposition,
            padding: 0,
        });
    }
    active_layers.truncate(maximum_layers as usize);
    let live_layers: Vec<_> = active_layers
        .iter()
        .enumerate()
        .map(|(texture_layer, layer)| GpuLiveLayer {
            canvas_layer: layer.0,
            texture_layer: texture_layer as u32,
            opacity: buffers
                .layers
                .values()
                .get(layer.0 as usize)
                .map_or(0.0, |metadata| metadata.opacity),
            stroke_count: live_strokes.len() as u32,
        })
        .collect();
    let required_layers = u32::try_from(active_layers.len())
        .unwrap_or(u32::MAX)
        .max(1);
    let requested_layer_capacity = required_layers
        .checked_next_power_of_two()
        .unwrap_or(maximum_layers)
        .min(maximum_layers);
    let stroke_update = replace_buffer(&mut masks.live_strokes, &live_strokes, &device, &queue);
    let layer_update = replace_buffer(&mut masks.live_layers, &live_layers, &device, &queue);
    if stroke_update.reallocated || layer_update.reallocated {
        for view in masks.views.values_mut() {
            view.resolve_bind_group = None;
            view.composite_bind_group = None;
        }
    }
    masks.active_layers = active_layers;

    for (entity, view) in &views {
        let width = view.viewport.z.max(1);
        let height = view.viewport.w.max(1);
        let recreate = masks.views.get(&entity).is_none_or(|mask| {
            mask.width != width
                || mask.height != height
                || mask.stroke_capacity < requested_stroke_capacity
                || mask.layer_capacity < requested_layer_capacity
        });
        if !recreate {
            continue;
        }

        masks.views.insert(
            entity,
            LiveStrokeMaskView {
                width,
                height,
                stroke_capacity: requested_stroke_capacity,
                layer_capacity: requested_layer_capacity,
                stroke_masks: create_live_texture_array(
                    &device,
                    "Hamerons live per-stroke union masks",
                    width,
                    height,
                    requested_stroke_capacity,
                    LIVE_MASK_FORMAT,
                ),
                cached_layers: create_live_texture_array(
                    &device,
                    "Hamerons cached active-layer surfaces",
                    width,
                    height,
                    requested_layer_capacity,
                    RESOLVED_LAYER_FORMAT,
                ),
                resolved_layers: create_live_texture_array(
                    &device,
                    "Hamerons resolved active-layer surfaces",
                    width,
                    height,
                    requested_layer_capacity,
                    RESOLVED_LAYER_FORMAT,
                ),
                resolve_bind_group: None,
                composite_bind_group: None,
            },
        );
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
    layer_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

#[derive(Resource)]
struct StrokeMaskPipeline {
    pipeline: CachedRenderPipelineId,
}

#[derive(Resource)]
struct LiveLayerPipelines {
    tile_pipeline: CachedRenderPipelineId,
    resolve_pipeline: CachedRenderPipelineId,
    resolve_layout: BindGroupLayoutDescriptor,
}

fn init_stroke_pipeline(
    mut commands: Commands,
    mesh_pipeline: Res<Mesh2dPipeline>,
    layout: Res<StrokeBindGroupLayout>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    tile_pipeline: Res<tiles::TileCompositePipeline>,
) {
    let coverage_shader =
        asset_server.load("embedded://hamerons_stroke_render/shaders/stroke_coverage.wgsl");
    let mask_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("Hamerons live stroke union-mask pipeline".into()),
        layout: vec![layout.0.clone(), mesh_pipeline.view_layout.clone()],
        vertex: VertexState {
            shader: coverage_shader.clone(),
            entry_point: Some("vertex".into()),
            shader_defs: Vec::new(),
            buffers: Vec::new(),
            constants: Vec::new(),
        },
        fragment: Some(FragmentState {
            shader: coverage_shader,
            entry_point: Some("fragment_rgba".into()),
            shader_defs: Vec::new(),
            targets: vec![Some(ColorTargetState {
                format: LIVE_MASK_FORMAT,
                blend: Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Max,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Max,
                    },
                }),
                write_mask: ColorWrites::ALL,
            })],
            constants: Vec::new(),
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        ..Default::default()
    });
    commands.insert_resource(StrokeMaskPipeline {
        pipeline: mask_pipeline,
    });

    let live_tile_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("Hamerons cached active-layer surface pipeline".into()),
        layout: vec![
            mesh_pipeline.view_layout.clone(),
            tile_pipeline.layout.clone(),
        ],
        vertex: VertexState {
            shader: tile_pipeline.shader.clone(),
            entry_point: Some("vertex_tile".into()),
            shader_defs: Vec::new(),
            buffers: Vec::new(),
            constants: Vec::new(),
        },
        fragment: Some(FragmentState {
            shader: tile_pipeline.shader.clone(),
            entry_point: Some("fragment_tile_unscaled".into()),
            shader_defs: Vec::new(),
            targets: vec![Some(ColorTargetState {
                format: RESOLVED_LAYER_FORMAT,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            constants: Vec::new(),
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        ..Default::default()
    });

    let resolve_layout = BindGroupLayoutDescriptor::new(
        "Hamerons active-layer resolve layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d_array(TextureSampleType::Float { filterable: false }),
                texture_2d_array(TextureSampleType::Float { filterable: false }),
                storage_buffer_read_only::<GpuLiveStroke>(false),
                storage_buffer_read_only::<GpuLiveLayer>(false),
            ),
        ),
    );
    let resolve_shader =
        asset_server.load("embedded://hamerons_stroke_render/shaders/stroke_layer_resolve.wgsl");
    let resolve_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("Hamerons active-layer resolve pipeline".into()),
        layout: vec![resolve_layout.clone()],
        vertex: VertexState {
            shader: resolve_shader.clone(),
            entry_point: Some("vertex_resolve".into()),
            shader_defs: Vec::new(),
            buffers: Vec::new(),
            constants: Vec::new(),
        },
        fragment: Some(FragmentState {
            shader: resolve_shader,
            entry_point: Some("fragment_resolve".into()),
            shader_defs: Vec::new(),
            targets: vec![Some(ColorTargetState {
                format: RESOLVED_LAYER_FORMAT,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            constants: Vec::new(),
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        ..Default::default()
    });
    commands.insert_resource(LiveLayerPipelines {
        tile_pipeline: live_tile_pipeline,
        resolve_pipeline,
        resolve_layout,
    });

    let layer_layout = BindGroupLayoutDescriptor::new(
        "Hamerons resolved live-layer texture layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d_array(TextureSampleType::Float { filterable: false }),
                storage_buffer_read_only::<GpuLiveLayer>(false),
            ),
        ),
    );
    commands.insert_resource(StrokePipeline {
        mesh_pipeline: mesh_pipeline.clone(),
        layer_layout,
        shader: asset_server
            .load("embedded://hamerons_stroke_render/shaders/stroke_composite.wgsl"),
    });
}

fn prepare_live_stroke_mask_bind_groups(
    mut masks: ResMut<LiveStrokeMasks>,
    pipeline: Res<StrokePipeline>,
    layer_pipelines: Res<LiveLayerPipelines>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
) {
    let (Some(live_strokes), Some(live_layers)) = (
        masks.live_strokes.buffer().cloned(),
        masks.live_layers.buffer().cloned(),
    ) else {
        return;
    };
    let resolve_layout = pipeline_cache.get_bind_group_layout(&layer_pipelines.resolve_layout);
    let composite_layout = pipeline_cache.get_bind_group_layout(&pipeline.layer_layout);
    for mask in masks.views.values_mut() {
        if mask.resolve_bind_group.is_none() {
            mask.resolve_bind_group = Some(device.create_bind_group(
                "Hamerons active-layer resolve bind group",
                &resolve_layout,
                &BindGroupEntries::sequential((
                    &mask.stroke_masks.array_view,
                    &mask.cached_layers.array_view,
                    live_strokes.as_entire_buffer_binding(),
                    live_layers.as_entire_buffer_binding(),
                )),
            ));
        }
        if mask.composite_bind_group.is_none() {
            mask.composite_bind_group = Some(device.create_bind_group(
                "Hamerons resolved live-layer composite bind group",
                &composite_layout,
                &BindGroupEntries::sequential((
                    &mask.resolved_layers.array_view,
                    live_layers.as_entire_buffer_binding(),
                )),
            ));
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct StrokePipelineKey {
    mesh_key: Mesh2dPipelineKey,
}

impl SpecializedRenderPipeline for StrokePipeline {
    type Key = StrokePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("Hamerons resolved active-layer composite pipeline".into()),
            layout: vec![
                self.mesh_pipeline.view_layout.clone(),
                self.layer_layout.clone(),
            ],
            vertex: VertexState {
                shader: self.shader.clone(),
                entry_point: Some("vertex_composite".into()),
                shader_defs: Vec::new(),
                buffers: Vec::new(),
                constants: Vec::new(),
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                entry_point: Some("fragment_composite".into()),
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
    SetStrokeMaskBindGroup<1>,
    DrawStrokes,
);

struct SetStrokeMaskBindGroup<const I: usize>;

impl<const I: usize, P: PhaseItem> RenderCommand<P> for SetStrokeMaskBindGroup<I> {
    type Param = SRes<LiveStrokeMasks>;
    type ViewQuery = Entity;
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _entity: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        masks: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let masks = masks.into_inner();
        let Some(bind_group) = masks
            .views
            .get(&view)
            .and_then(|mask| mask.composite_bind_group.as_ref())
        else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(I, bind_group, &[]);
        RenderCommandResult::Success
    }
}

struct DrawStrokes;

impl<P: PhaseItem> RenderCommand<P> for DrawStrokes {
    type Param = SRes<LiveStrokeMasks>;
    type ViewQuery = ();
    type ItemQuery = &'static StrokeOverlayRenderEntity;

    fn render<'w>(
        _item: &P,
        _view: ROQueryItem<'w, '_, Self::ViewQuery>,
        entity: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        masks: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let masks = masks.into_inner();
        let Some(renderer) = entity else {
            return RenderCommandResult::Skip;
        };
        if let Some(layer_slot) = masks.active_layer_slot(renderer.layer) {
            pass.draw(0..3, layer_slot..layer_slot + 1);
        }
        RenderCommandResult::Success
    }
}

type LiveStrokeMaskViewQuery<'w> = (Entity, &'w ViewUniformOffset, &'w Mesh2dViewBindGroup);
type LiveStrokeMaskViewFilter = (With<ExtractedView>, With<Camera2d>);

fn rasterize_live_stroke_masks(
    mut render_context: RenderContext,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<StrokeMaskPipeline>,
    buffers: Res<StrokeGpuBuffers>,
    masks: Res<LiveStrokeMasks>,
    views: Query<LiveStrokeMaskViewQuery, LiveStrokeMaskViewFilter>,
) {
    if buffers.overlay_ranges.is_empty() {
        return;
    }
    let (Some(render_pipeline), Some(stroke_bind_group)) = (
        pipeline_cache.get_render_pipeline(pipeline.pipeline),
        buffers.bind_group.as_ref(),
    ) else {
        return;
    };

    for (view_entity, view_uniform, view_bind_group) in &views {
        let Some(mask) = masks.views.get(&view_entity) else {
            continue;
        };
        for (mask_layer, (_, segment_range)) in buffers
            .overlay_ranges
            .iter()
            .take(mask.stroke_masks.layer_views.len())
            .enumerate()
        {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_render_pass(&RenderPassDescriptor {
                        label: Some("Hamerons live stroke union-mask pass"),
                        color_attachments: &[Some(RenderPassColorAttachment {
                            view: &mask.stroke_masks.layer_views[mask_layer],
                            depth_slice: None,
                            resolve_target: None,
                            ops: Operations {
                                load: LoadOp::Clear(Default::default()),
                                store: StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
            pass.set_pipeline(render_pipeline);
            pass.set_bind_group(0, stroke_bind_group, &[]);
            pass.set_bind_group(1, &view_bind_group.value, &[view_uniform.offset]);
            pass.draw(0..6, segment_range.clone());
            pass.draw(6..12, segment_range.clone());
        }
    }
}

fn rasterize_live_layers(
    mut render_context: RenderContext,
    pipeline_cache: Res<PipelineCache>,
    pipelines: Res<LiveLayerPipelines>,
    tile_cache: Res<tiles::TileGpuCache>,
    masks: Res<LiveStrokeMasks>,
    views: Query<LiveStrokeMaskViewQuery, LiveStrokeMaskViewFilter>,
    tile_renderers: Query<&TileCanvasRenderEntity>,
) {
    if masks.active_layers.is_empty() {
        return;
    }
    let tile_pipeline = pipeline_cache.get_render_pipeline(pipelines.tile_pipeline);
    let resolve_pipeline = pipeline_cache.get_render_pipeline(pipelines.resolve_pipeline);

    for (view_entity, view_uniform, view_bind_group) in &views {
        let Some(mask_view) = masks.views.get(&view_entity) else {
            continue;
        };
        for (layer_slot, layer) in masks.active_layers.iter().enumerate() {
            let mut cached_pass =
                render_context
                    .command_encoder()
                    .begin_render_pass(&RenderPassDescriptor {
                        label: Some("Hamerons cached active-layer surface pass"),
                        color_attachments: &[Some(RenderPassColorAttachment {
                            view: &mask_view.cached_layers.layer_views[layer_slot],
                            depth_slice: None,
                            resolve_target: None,
                            ops: Operations {
                                load: LoadOp::Clear(Default::default()),
                                store: StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
            if let (Some(tile_pipeline), Some(tile_bind_group), Some(renderer)) = (
                tile_pipeline,
                tile_cache.composite_bind_group.as_ref(),
                tile_renderers
                    .iter()
                    .find(|renderer| renderer.layer == *layer),
            ) {
                cached_pass.set_pipeline(tile_pipeline);
                cached_pass.set_bind_group(0, &view_bind_group.value, &[view_uniform.offset]);
                cached_pass.set_bind_group(1, tile_bind_group, &[]);
                cached_pass.draw(0..6, renderer.instances.clone());
            }
            drop(cached_pass);

            let (Some(resolve_pipeline), Some(resolve_bind_group)) =
                (resolve_pipeline, mask_view.resolve_bind_group.as_ref())
            else {
                continue;
            };
            let mut resolve_pass =
                render_context
                    .command_encoder()
                    .begin_render_pass(&RenderPassDescriptor {
                        label: Some("Hamerons active-layer stroke resolve pass"),
                        color_attachments: &[Some(RenderPassColorAttachment {
                            view: &mask_view.resolved_layers.layer_views[layer_slot],
                            depth_slice: None,
                            resolve_target: None,
                            ops: Operations {
                                load: LoadOp::Clear(Default::default()),
                                store: StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
            resolve_pass.set_pipeline(resolve_pipeline);
            resolve_pass.set_bind_group(0, resolve_bind_group, &[]);
            let layer_slot = layer_slot as u32;
            resolve_pass.draw(0..3, layer_slot..layer_slot + 1);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn queue_stroke_overlay(
    draw_functions: Res<DrawFunctions<Transparent2d>>,
    pipeline: Res<StrokePipeline>,
    mask_pipeline: Res<StrokeMaskPipeline>,
    layer_pipelines: Res<LiveLayerPipelines>,
    mut pipelines: ResMut<SpecializedRenderPipelines<StrokePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    buffers: Res<StrokeGpuBuffers>,
    masks: Res<LiveStrokeMasks>,
    mut queued_live_layers: ResMut<QueuedLiveLayers>,
    renderers: Query<(Entity, &MainEntity, &StrokeOverlayRenderEntity)>,
    mut phases: ResMut<ViewSortedRenderPhases<Transparent2d>>,
    views: Query<(Entity, &ExtractedView, &Msaa), With<Camera2d>>,
) {
    queued_live_layers.clear();
    let mask_pipeline_ready = pipeline_cache
        .get_render_pipeline(mask_pipeline.pipeline)
        .is_some();
    let layer_pipelines_ready = pipeline_cache
        .get_render_pipeline(layer_pipelines.tile_pipeline)
        .is_some()
        && pipeline_cache
            .get_render_pipeline(layer_pipelines.resolve_pipeline)
            .is_some();
    let can_draw = !masks.active_layers.is_empty()
        && buffers.bind_group.is_some()
        && mask_pipeline_ready
        && layer_pipelines_ready;
    let draw_function = can_draw.then(|| {
        draw_functions
            .read()
            .get_id::<DrawStrokeOverlay>()
            .expect("stroke draw command was not registered")
    });

    for (view_entity, view, msaa) in &views {
        let mesh_key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples())
            | Mesh2dPipelineKey::from_target_format(view.target_format);
        let pipeline_id: CachedRenderPipelineId =
            pipelines.specialize(&pipeline_cache, &pipeline, StrokePipelineKey { mesh_key });
        let Some(draw_function) = draw_function else {
            continue;
        };
        if pipeline_cache.get_render_pipeline(pipeline_id).is_none()
            || masks.views.get(&view_entity).is_none_or(|mask| {
                mask.resolve_bind_group.is_none() || mask.composite_bind_group.is_none()
            })
        {
            continue;
        }
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        for (render_entity, main_entity, renderer) in &renderers {
            if !masks.has_active_layer(renderer.layer) {
                continue;
            }
            phase.add_transient(Transparent2d {
                entity: (render_entity, *main_entity),
                draw_function,
                pipeline: pipeline_id,
                sort_key: layer_phase_sort_key(renderer.order, renderer.layer_count),
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                extracted_index: usize::MAX,
                indexed: false,
            });
            queued_live_layers.insert(view_entity, renderer.layer);
        }
    }
}

pub(super) fn layer_phase_sort_key(order: u32, layer_count: u32) -> FloatOrd {
    let layer_count = layer_count.max(1);
    let rank = order.saturating_add(1) as f32 / layer_count.saturating_add(1) as f32;
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
fn expand_stroke_geometry_for_test(source: &str) -> String {
    let geometry = include_str!("../shaders/stroke_geometry.wgsl").replacen(
        "#define_import_path hamerons_stroke_render::stroke_geometry",
        "",
        1,
    );
    source
        .replacen(
            "#import hamerons_stroke_render::stroke_geometry as geometry",
            &geometry,
            1,
        )
        .replace("geometry::", "")
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
        assert_eq!(size_of::<GpuLiveStroke>(), 16);
        assert_eq!(size_of::<GpuLiveLayer>(), 16);

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
    fn layer_phase_sort_keys_follow_document_order() {
        let lower = layer_phase_sort_key(0, 3);
        let middle = layer_phase_sort_key(1, 3);
        let upper = layer_phase_sort_key(2, 3);
        assert!(lower < middle);
        assert!(middle < upper);
    }

    #[test]
    fn stroke_shader_is_valid_wgsl() {
        let source = expand_stroke_geometry_for_test(include_str!(
            "../shaders/stroke_coverage.wgsl"
        ))
        .replacen(
            "#import bevy_render::view::View",
            "struct View { clip_from_world: mat4x4<f32>, world_from_clip: mat4x4<f32>, viewport: vec4<f32>, }",
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

    #[test]
    fn stroke_composite_shader_is_valid_wgsl() {
        let source = include_str!("../shaders/stroke_composite.wgsl").replacen(
            "#import bevy_render::view::View",
            "struct View { viewport: vec4<f32>, }",
            1,
        );
        let module =
            naga::front::wgsl::parse_str(&source).expect("stroke composite must parse as WGSL");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("stroke composite must pass Naga validation");
    }

    #[test]
    fn stroke_layer_resolve_shader_is_valid_wgsl() {
        let source = include_str!("../shaders/stroke_layer_resolve.wgsl");
        let module =
            naga::front::wgsl::parse_str(source).expect("stroke layer resolve must parse as WGSL");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("stroke layer resolve must pass Naga validation");
    }

    #[test]
    fn layer_opacity_is_applied_after_live_source_over() {
        let cached = Vec4::new(0.12, 0.04, 0.03, 0.7);
        let source = Vec4::new(0.18, 0.03, 0.02, 0.25);
        let opacity = 0.4;

        let resolved = source + cached * (1.0 - source.w);
        let correct = resolved * opacity;
        let old_live_order = source * opacity + cached * opacity * (1.0 - source.w * opacity);

        assert!(correct.abs_diff_eq(Vec4::new(0.108, 0.024, 0.017, 0.31), 0.0001));
        assert!(!correct.abs_diff_eq(old_live_order, 0.0001));
    }

    #[test]
    fn dark_translucent_live_ink_quantizes_only_after_each_stroke_resolves() {
        fn unorm8(value: f32) -> f32 {
            (value.clamp(0.0, 1.0) * 255.0).round() / 255.0
        }

        let ink = Vec4::new(0.015, 0.02, 0.03, 1.0);
        let amount = 0.13;
        let source = ink * amount;
        let cached_ink = ink.map(unorm8);

        // Resolving the whole stroke in floating point before its RGBA8
        // checkpoint keeps an opaque patch unchanged.
        let resolved = source + cached_ink * (1.0 - source.w);
        let live_result = resolved.map(unorm8);
        assert_eq!(live_result, cached_ink);

        // Quantizing the premultiplied source first loses its dark red channel
        // while alpha survives. That was the visible oval band in the live
        // stroke even though the released tile had the correct color.
        let lossy_source = source.map(unorm8);
        let lossy_result = (lossy_source + cached_ink * (1.0 - lossy_source.w)).map(unorm8);
        assert!(lossy_result.x < live_result.x);

        assert_eq!(LIVE_MASK_FORMAT, TextureFormat::Rgba16Float);
        assert_eq!(RESOLVED_LAYER_FORMAT, TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn live_handoff_matches_persistent_per_stroke_checkpoints() {
        fn unorm8(value: f32) -> f32 {
            (value.clamp(0.0, 1.0) * 255.0).round() / 255.0
        }

        fn deposit(destination: Vec4, source: Vec4) -> Vec4 {
            (source + destination * (1.0 - source.w)).map(unorm8)
        }

        let ink = Vec4::new(0.015, 0.02, 0.03, 1.0);
        let source = ink * 0.13;

        // The live path begins with the completed RGBA8 tile, then resolves
        // the current stroke. Persistent replay must checkpoint at that same
        // whole-stroke boundary before it continues with the current stroke.
        let cached_after_first_stroke = deposit(Vec4::ZERO, source);
        let live_after_second_stroke = deposit(cached_after_first_stroke, source);

        let mut persistent_replay = Vec4::ZERO;
        for stroke in [source, source] {
            persistent_replay = deposit(persistent_replay, stroke);
        }
        assert_eq!(live_after_second_stroke, persistent_replay);

        // Quantizing only once after replaying every stroke creates a different
        // color and was the visible change when the live stroke retired.
        let float_replay = source + source * (1.0 - source.w);
        assert_ne!(float_replay.map(unorm8), live_after_second_stroke);
    }
}
