use alloc::borrow::Cow;
use std::mem::size_of;

use bevy_asset::{AssetServer, Handle};
use bevy_camera::Camera2d;
use bevy_core_pipeline::core_2d::{Transparent2d, CORE_2D_DEPTH_FORMAT};
use bevy_ecs::{
    entity::Entity,
    name::Name,
    prelude::With,
    query::ROQueryItem,
    resource::Resource,
    system::{lifetimeless::SRes, Commands, Query, Res, ResMut, SystemParamItem},
};
use bevy_render::{
    render_phase::{
        DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand, RenderCommandResult,
        SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
    },
    render_resource::{
        binding_types::{storage_buffer, storage_buffer_read_only},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BlendState,
        Buffer, BufferDescriptor, BufferUsages, CachedComputePipelineId, CachedRenderPipelineId,
        ColorTargetState, ColorWrites, CompareFunction, ComputePassDescriptor,
        ComputePipelineDescriptor, DepthBiasState, DepthStencilState, FragmentState,
        MultisampleState, PipelineCache, PrimitiveState, RawBufferVec, RenderPipelineDescriptor,
        ShaderStages, ShaderType, SpecializedRenderPipeline, SpecializedRenderPipelines,
        StencilFaceState, StencilState, VertexState,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    sync_world::{MainEntity, TemporaryRenderEntity},
    view::{ExtractedView, Msaa},
    Extract,
};
use bevy_shader::Shader;
use bevy_sprite_render::{Mesh2dPipeline, Mesh2dPipelineKey, SetMesh2dViewBindGroup};
use bytemuck::{Pod, Zeroable};

use super::{
    layer_phase_sort_key, replace_buffer, GpuRgbaMaterial, GpuStrokePoint, GpuStrokeSegment,
    QueuedLiveLayers, StrokeGpuBuffers, TileCanvasRenderEntity,
};
use crate::{
    tiles::{TileDisplayInstance, TileFeedback, TileRasterJob},
    CanvasTileCache, TileKey,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, ShaderType)]
struct GpuTileJob {
    slot: u32,
    replay_start: u32,
    replay_count: u32,
    tile_size: u32,
    origin_x: i32,
    origin_y: i32,
    padding_0: u32,
    padding_1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, ShaderType)]
struct GpuReplayStroke {
    segment_index_start: u32,
    segment_count: u32,
    material: u32,
    model_and_deposition: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, ShaderType)]
struct GpuTileInstance {
    slot: u32,
    tile_x: i32,
    tile_y: i32,
    tile_size: u32,
    opacity: f32,
}

#[derive(Resource, Default)]
pub(super) struct ExtractedTileBatch {
    serial: u64,
    tile_size: u32,
    jobs: Vec<GpuTileJob>,
    replay: Vec<GpuReplayStroke>,
    segment_indices: Vec<u32>,
    instances: Vec<GpuTileInstance>,
    completions: Vec<(TileKey, u64)>,
    presentations: Vec<(TileKey, u64)>,
    required_slots: u32,
    feedback: TileFeedback,
}

pub(super) fn extract_tile_batch(
    mut commands: Commands,
    cache: Extract<Res<CanvasTileCache>>,
    feedback: Extract<Res<TileFeedback>>,
    mut extracted: ResMut<ExtractedTileBatch>,
) {
    let batch = cache.batch();
    extracted.serial = batch.serial;
    extracted.tile_size = batch.tile_size;
    extracted.jobs.clear();
    extracted.replay.clear();
    extracted.segment_indices.clear();
    extracted.instances.clear();
    extracted.completions.clear();
    extracted.presentations.clear();
    extracted.required_slots = 0;
    extracted.feedback = feedback.clone();

    for job in &batch.jobs {
        let replay_start = extracted.replay.len() as u32;
        for stroke in &job.replay {
            let segment_index_start = extracted.segment_indices.len() as u32;
            extracted
                .segment_indices
                .extend_from_slice(&stroke.segment_indices);
            extracted.replay.push(GpuReplayStroke {
                segment_index_start,
                segment_count: stroke.segment_indices.len() as u32,
                material: stroke.material.0,
                model_and_deposition: (stroke.model.0 & 0x7fff_ffff)
                    | ((stroke.deposition as u32) << 31),
            });
        }
        extracted
            .jobs
            .push(gpu_job(job, replay_start, batch.tile_size));
        extracted.completions.push((job.key, job.revision));
        extracted.required_slots = extracted.required_slots.max(job.slot.saturating_add(1));
        let _halo_dependencies_are_declared = job.halo.len();
    }
    // A bind group cannot bind an absent replay buffer. Blank-tile jobs still
    // dispatch with replay_count zero and safely ignore this sentinel.
    if !extracted.jobs.is_empty() && extracted.replay.is_empty() {
        extracted.replay.push(GpuReplayStroke::zeroed());
    }
    if !extracted.jobs.is_empty() && extracted.segment_indices.is_empty() {
        extracted.segment_indices.push(0);
    }
    for instance in &batch.display {
        extracted
            .instances
            .push(gpu_instance(instance, batch.tile_size));
        extracted
            .presentations
            .push((instance.key, instance.revision));
        extracted.required_slots = extracted
            .required_slots
            .max(instance.slot.saturating_add(1));
    }
    let mut start = 0usize;
    while start < batch.display.len() {
        let first = batch.display[start];
        let mut end = start + 1;
        while end < batch.display.len() && batch.display[end].key.layer == first.key.layer {
            end += 1;
        }
        commands.spawn((
            Name::new(format!("HameronsTileCanvasLayer{}", first.key.layer.0)),
            TileCanvasRenderEntity {
                layer: first.key.layer,
                instances: start as u32..end as u32,
                order: first.layer_order,
                layer_count: first.layer_count,
            },
            MainEntity::from(Entity::PLACEHOLDER),
            TemporaryRenderEntity::default(),
        ));
        start = end;
    }
}

fn gpu_job(job: &TileRasterJob, replay_start: u32, tile_size: u32) -> GpuTileJob {
    let size = tile_size.max(1) as i64;
    GpuTileJob {
        slot: job.slot,
        replay_start,
        replay_count: job.replay.len() as u32,
        tile_size: tile_size.max(1),
        origin_x: (i64::from(job.key.x) * size).clamp(i32::MIN as i64, i32::MAX as i64) as i32,
        origin_y: (i64::from(job.key.y) * size).clamp(i32::MIN as i64, i32::MAX as i64) as i32,
        padding_0: 0,
        padding_1: 0,
    }
}

fn gpu_instance(instance: &TileDisplayInstance, tile_size: u32) -> GpuTileInstance {
    GpuTileInstance {
        slot: instance.slot,
        tile_x: instance.key.x,
        tile_y: instance.key.y,
        tile_size: tile_size.max(1),
        opacity: instance.opacity,
    }
}

#[derive(Resource)]
pub(super) struct TileGpuCache {
    jobs: RawBufferVec<GpuTileJob>,
    replay: RawBufferVec<GpuReplayStroke>,
    segment_indices: RawBufferVec<u32>,
    instances: RawBufferVec<GpuTileInstance>,
    pixels: Option<Buffer>,
    pixel_capacity_slots: u32,
    compute_bind_group: Option<BindGroup>,
    pub(super) composite_bind_group: Option<BindGroup>,
    bind_groups_dirty: bool,
    job_count: u32,
    instance_count: u32,
    processed_serial: Option<u64>,
    reset_requested: bool,
    device_reset_pending: bool,
    reported_capacity: Option<(u32, u32)>,
    stroke_buffer_generation: Option<u64>,
}

impl Default for TileGpuCache {
    fn default() -> Self {
        let mut jobs = RawBufferVec::new(BufferUsages::STORAGE);
        jobs.set_label(Some("Hamerons tile regeneration jobs"));
        let mut replay = RawBufferVec::new(BufferUsages::STORAGE);
        replay.set_label(Some("Hamerons tile stroke replay metadata"));
        let mut instances = RawBufferVec::new(BufferUsages::STORAGE);
        instances.set_label(Some("Hamerons tile display instances"));
        Self {
            jobs,
            replay,
            segment_indices: {
                let mut buffer = RawBufferVec::new(BufferUsages::STORAGE);
                buffer.set_label(Some("Hamerons tile-local segment indices"));
                buffer
            },
            instances,
            pixels: None,
            pixel_capacity_slots: 0,
            compute_bind_group: None,
            composite_bind_group: None,
            bind_groups_dirty: true,
            job_count: 0,
            instance_count: 0,
            processed_serial: None,
            reset_requested: false,
            device_reset_pending: true,
            reported_capacity: None,
            stroke_buffer_generation: None,
        }
    }
}

pub(super) fn prepare_tile_buffers(
    extracted: Res<ExtractedTileBatch>,
    mut cache: ResMut<TileGpuCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let tile_bytes = u64::from(extracted.tile_size.max(1)).pow(2) * size_of::<u32>() as u64;
    let max_binding = device.limits().max_storage_buffer_binding_size;
    let max_slots = (max_binding / tile_bytes).max(1).min(u64::from(u32::MAX)) as u32;
    let reported_capacity = (extracted.tile_size, max_slots);
    if cache.reported_capacity != Some(reported_capacity) {
        extracted.feedback.capacity(max_slots);
        cache.reported_capacity = Some(reported_capacity);
    }
    if cache.device_reset_pending {
        extracted.feedback.device_reset();
        cache.device_reset_pending = false;
        cache.reset_requested = true;
    }

    if cache.pixels.is_none() && extracted.jobs.is_empty() && !extracted.instances.is_empty() {
        cache.instance_count = 0;
        if !cache.reset_requested {
            extracted.feedback.device_reset();
            cache.reset_requested = true;
        }
        return;
    }

    if !extracted.jobs.is_empty() {
        cache.reset_requested = false;
    }
    let jobs = replace_buffer(&mut cache.jobs, &extracted.jobs, &device, &queue);
    let replay = replace_buffer(&mut cache.replay, &extracted.replay, &device, &queue);
    let segment_indices = replace_buffer(
        &mut cache.segment_indices,
        &extracted.segment_indices,
        &device,
        &queue,
    );
    let instances = replace_buffer(&mut cache.instances, &extracted.instances, &device, &queue);
    cache.bind_groups_dirty |= jobs.reallocated
        || replay.reallocated
        || segment_indices.reallocated
        || instances.reallocated;

    if extracted.required_slots > cache.pixel_capacity_slots {
        let new_capacity = extracted.required_slots.next_power_of_two().min(max_slots);
        if new_capacity < extracted.required_slots {
            for &(key, revision) in &extracted.completions {
                extracted.feedback.retry(key, revision);
            }
            cache.job_count = 0;
            cache.instance_count = 0;
            cache.processed_serial = Some(extracted.serial);
            return;
        }
        let size = u64::from(new_capacity) * tile_bytes;
        let new_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Hamerons packed RGBA tile pixels"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(old) = &cache.pixels {
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(old, 0, &new_buffer, 0, old.size());
            queue.submit([encoder.finish()]);
        }
        cache.pixels = Some(new_buffer);
        cache.pixel_capacity_slots = new_capacity;
        cache.bind_groups_dirty = true;
    }

    cache.job_count = extracted.jobs.len() as u32;
    cache.instance_count = if cache.pixels.is_some() {
        extracted.instances.len() as u32
    } else {
        0
    };
}

#[derive(Resource)]
pub(super) struct TileComputePipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

#[derive(Clone, Resource)]
pub(super) struct TileCompositePipeline {
    pub(super) mesh_pipeline: Mesh2dPipeline,
    pub(super) layout: BindGroupLayoutDescriptor,
    pub(super) shader: Handle<Shader>,
}

pub(super) fn init_tile_pipelines(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    mesh_pipeline: Res<Mesh2dPipeline>,
) {
    let compute_layout = BindGroupLayoutDescriptor::new(
        "Hamerons tile compute layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only::<GpuStrokePoint>(false),
                storage_buffer_read_only::<GpuStrokeSegment>(false),
                storage_buffer_read_only::<GpuRgbaMaterial>(false),
                storage_buffer_read_only::<GpuTileJob>(false),
                storage_buffer_read_only::<GpuReplayStroke>(false),
                storage_buffer_read_only::<u32>(false),
                storage_buffer::<u32>(false),
            ),
        ),
    );
    let raster_shader =
        asset_server.load("embedded://hamerons_stroke_render/shaders/tile_raster.wgsl");
    let compute_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("Hamerons tile coverage and RGBA deposition".into()),
        layout: vec![compute_layout.clone()],
        shader: raster_shader,
        entry_point: Some(Cow::Borrowed("rasterize_tile")),
        ..Default::default()
    });
    commands.insert_resource(TileComputePipeline {
        layout: compute_layout,
        pipeline: compute_pipeline,
    });

    let composite_layout = BindGroupLayoutDescriptor::new(
        "Hamerons tile composite layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                storage_buffer_read_only::<u32>(false),
                storage_buffer_read_only::<GpuTileInstance>(false),
            ),
        ),
    );
    commands.insert_resource(TileCompositePipeline {
        mesh_pipeline: mesh_pipeline.clone(),
        layout: composite_layout,
        shader: asset_server.load("embedded://hamerons_stroke_render/shaders/tile_composite.wgsl"),
    });
}

pub(super) fn prepare_tile_bind_groups(
    strokes: Res<StrokeGpuBuffers>,
    mut cache: ResMut<TileGpuCache>,
    compute: Res<TileComputePipeline>,
    composite: Res<TileCompositePipeline>,
    pipeline_cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
) {
    if cache.stroke_buffer_generation != Some(strokes.reallocation_generation) {
        cache.bind_groups_dirty = true;
    }
    if !cache.bind_groups_dirty {
        return;
    }
    let compute_bind_group = {
        let (
            Some(points),
            Some(segments),
            Some(materials),
            Some(jobs),
            Some(replay),
            Some(segment_indices),
            Some(pixels),
        ) = (
            strokes.points.binding(),
            strokes.segments.binding(),
            strokes.materials.binding(),
            cache.jobs.binding(),
            cache.replay.binding(),
            cache.segment_indices.binding(),
            cache.pixels.as_ref(),
        )
        else {
            return;
        };
        Some(device.create_bind_group(
            "Hamerons tile compute bind group",
            &pipeline_cache.get_bind_group_layout(&compute.layout),
            &BindGroupEntries::sequential((
                points,
                segments,
                materials,
                jobs,
                replay,
                segment_indices,
                pixels.as_entire_buffer_binding(),
            )),
        ))
    };
    let composite_bind_group = match (cache.pixels.as_ref(), cache.instances.binding()) {
        (Some(pixels), Some(instances)) => Some(device.create_bind_group(
            "Hamerons tile composite bind group",
            &pipeline_cache.get_bind_group_layout(&composite.layout),
            &BindGroupEntries::sequential((pixels.as_entire_buffer_binding(), instances)),
        )),
        _ => None,
    };
    cache.compute_bind_group = compute_bind_group;
    cache.composite_bind_group = composite_bind_group;
    cache.stroke_buffer_generation = Some(strokes.reallocation_generation);
    cache.bind_groups_dirty = false;
}

pub(super) fn rasterize_tiles(
    mut render_context: RenderContext,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<TileComputePipeline>,
    extracted: Res<ExtractedTileBatch>,
    mut cache: ResMut<TileGpuCache>,
) {
    if cache.processed_serial == Some(extracted.serial) {
        return;
    }
    cache.processed_serial = Some(extracted.serial);
    if cache.job_count == 0 {
        return;
    }
    let (Some(pipeline), Some(bind_group)) = (
        pipeline_cache.get_compute_pipeline(pipeline.pipeline),
        cache.compute_bind_group.as_ref(),
    ) else {
        for &(key, revision) in &extracted.completions {
            extracted.feedback.retry(key, revision);
        }
        return;
    };

    let workgroups = extracted.tile_size.max(1).div_ceil(8);
    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("Hamerons persistent tile regeneration"),
            ..Default::default()
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(workgroups, workgroups, cache.job_count);
    drop(pass);

    for &(key, revision) in &extracted.completions {
        extracted.feedback.complete(key, revision);
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) struct TileCompositePipelineKey {
    mesh_key: Mesh2dPipelineKey,
}

impl SpecializedRenderPipeline for TileCompositePipeline {
    type Key = TileCompositePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("Hamerons persistent tile composite pipeline".into()),
            layout: vec![self.mesh_pipeline.view_layout.clone(), self.layout.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                entry_point: Some("vertex_tile".into()),
                shader_defs: Vec::new(),
                buffers: Vec::new(),
                constants: Vec::new(),
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                entry_point: Some("fragment_tile".into()),
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
                bias: DepthBiasState::default(),
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

pub(super) type DrawTileCanvas = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    SetTileBindGroup<1>,
    DrawTiles,
);

pub(super) struct SetTileBindGroup<const I: usize>;

impl<const I: usize, P: PhaseItem> RenderCommand<P> for SetTileBindGroup<I> {
    type Param = SRes<TileGpuCache>;
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        _view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _entity: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        cache: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let cache = cache.into_inner();
        let Some(bind_group) = &cache.composite_bind_group else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(I, bind_group, &[]);
        RenderCommandResult::Success
    }
}

pub(super) struct DrawTiles;

impl<P: PhaseItem> RenderCommand<P> for DrawTiles {
    type Param = ();
    type ViewQuery = ();
    type ItemQuery = &'static TileCanvasRenderEntity;

    fn render<'w>(
        _item: &P,
        _view: ROQueryItem<'w, '_, Self::ViewQuery>,
        entity: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        _cache: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(renderer) = entity else {
            return RenderCommandResult::Skip;
        };
        pass.draw(0..6, renderer.instances.clone());
        RenderCommandResult::Success
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn queue_tile_canvas(
    draw_functions: Res<DrawFunctions<Transparent2d>>,
    pipeline: Res<TileCompositePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<TileCompositePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    cache: Res<TileGpuCache>,
    extracted: Res<ExtractedTileBatch>,
    queued_live_layers: Res<QueuedLiveLayers>,
    renderers: Query<(Entity, &MainEntity, &TileCanvasRenderEntity)>,
    mut phases: ResMut<ViewSortedRenderPhases<Transparent2d>>,
    views: Query<(Entity, &ExtractedView, &Msaa), With<Camera2d>>,
) {
    // Specialize for every active 2D view even while the canvas is empty. If
    // specialization only starts once the first completed tile is displayable,
    // the live overlay can retire while this pipeline is still compiling and
    // leave a one-frame hole after the first stroke.
    let can_draw = cache.instance_count > 0 && cache.composite_bind_group.is_some();
    let draw_function = can_draw.then(|| {
        draw_functions
            .read()
            .get_id::<DrawTileCanvas>()
            .expect("tile draw command was not registered")
    });
    let mut queued_for_presentation = false;
    for (view_entity, view, msaa) in &views {
        let mesh_key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples())
            | Mesh2dPipelineKey::from_target_format(view.target_format);
        let pipeline_id: CachedRenderPipelineId = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            TileCompositePipelineKey { mesh_key },
        );
        let Some(draw_function) = draw_function else {
            continue;
        };
        if pipeline_cache.get_render_pipeline(pipeline_id).is_none() {
            continue;
        }
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        for (render_entity, main_entity, renderer) in &renderers {
            if queued_live_layers.contains(view_entity, renderer.layer) {
                // The same cached tile range was drawn into the resolved live
                // layer before the camera pass, so it was still presented.
                queued_for_presentation = true;
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
            queued_for_presentation = true;
        }
    }
    if queued_for_presentation {
        for &(key, revision) in &extracted.presentations {
            extracted.feedback.presented(key, revision);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_gpu_layouts_match_wgsl_storage_structs() {
        assert_eq!(size_of::<GpuTileJob>(), 32);
        assert_eq!(size_of::<GpuReplayStroke>(), 16);
        assert_eq!(size_of::<GpuTileInstance>(), 20);
    }

    #[test]
    fn tile_shaders_parse_and_validate() {
        let raster = super::super::expand_stroke_geometry_for_test(include_str!(
            "../shaders/tile_raster.wgsl"
        ));
        let module = naga::front::wgsl::parse_str(&raster).expect("tile raster shader must parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("tile raster shader must validate");

        let composite = include_str!("../shaders/tile_composite.wgsl").replacen(
            "#import bevy_render::view::View",
            "struct View { clip_from_world: mat4x4<f32>, viewport: vec4<f32>, }",
            1,
        );
        let module =
            naga::front::wgsl::parse_str(&composite).expect("tile composite shader must parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("tile composite shader must validate");
    }

    #[test]
    fn deposition_mode_uses_the_same_high_bit_as_live_ink() {
        let packed = (9 & 0x7fff_ffff) | ((crate::DepositionMode::Erase as u32) << 31);
        assert_eq!(packed, 0x8000_0009);
    }
}
