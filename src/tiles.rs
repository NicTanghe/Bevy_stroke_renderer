use alloc::{collections::VecDeque, sync::Arc};
use std::{
    collections::{HashMap, HashSet},
    sync::{Mutex, PoisonError},
};

use bevy_camera::{Camera, Camera2d};
use bevy_ecs::{
    query::With,
    resource::Resource,
    system::{Query, Res, ResMut},
};
use bevy_math::{Rect, Vec2};
use bevy_transform::components::GlobalTransform;

use crate::{
    DepositionMode, EffectInfluence, EffectRegistry, LayerId, PaintMaterialId,
    PaintModelDescriptor, PaintModelId, PaintModelRegistry, PaintPlaneDescriptor, ScratchPlanePool,
    StrokeDocument, StrokeRendererSettings, StrokeTelemetry,
};

/// Signed document-space tile identity, independent of paint-model plane formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileKey {
    /// Canvas layer owning the tile.
    pub layer: LayerId,
    /// Signed horizontal tile coordinate.
    pub x: i32,
    /// Signed vertical tile coordinate.
    pub y: i32,
}

impl TileKey {
    /// Creates a key in stable document coordinates.
    pub const fn new(layer: LayerId, x: i32, y: i32) -> Self {
        Self { layer, x, y }
    }
}

/// Why a tile revision became stale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileInvalidationCause {
    /// A newly completed stroke may display the old tile beneath its overlay.
    StrokeCommit,
    /// Undo or redo changed vector visibility.
    UndoRedo,
    /// Clear removed visible vector records.
    Clear,
    /// Canvas or viewport-dependent resources changed.
    Resize,
    /// Effect parameters, topology, or neighborhood dependencies changed.
    EffectDependency,
    /// Paint-model plane layout or tile size changed.
    CacheLayout,
    /// The render device was recreated and all derived resources were lost.
    DeviceLoss,
}

/// Revisioned document-to-cache invalidation record.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TileInvalidation {
    pub key: TileKey,
    pub revision: u64,
    pub cause: TileInvalidationCause,
    pub allow_stale_display: bool,
    pub halo_radius_px: u32,
}

/// Allocation state of one logical tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileCacheState {
    /// Needs a persistent slot and deterministic replay.
    Dirty,
    /// Replay was extracted and awaits render-world completion.
    Scheduled,
    /// Its display revision is current.
    Ready,
    /// No persistent slot is resident; vector history can regenerate it.
    Evicted,
}

/// Public snapshot of aggregate cache behavior.
#[derive(Clone, Copy, Debug, Default)]
pub struct TileCacheStats {
    /// Tiles currently occupying persistent slots.
    pub resident_tiles: u64,
    /// Tiles awaiting replay.
    pub dirty_tiles: u64,
    /// Tiles replayed since startup.
    pub rasterized_tiles: u64,
    /// Clean tiles evicted since startup.
    pub evicted_tiles: u64,
    /// Evicted tiles later regenerated.
    pub regenerated_tiles: u64,
    /// Bytes represented by all resident paint-model planes.
    pub persistent_bytes: u64,
    /// Current transient scratch usage.
    pub scratch_bytes: u64,
    /// High-water transient scratch usage.
    pub scratch_high_water_bytes: u64,
    /// Tiles dirtied directly by stroke bounds.
    pub stroke_invalidations: u64,
    /// Additional neighborhood tiles dirtied by effect expansion.
    pub effect_expanded_invalidations: u64,
}

/// Per-plane accounting derived entirely from a model descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TilePlaneAllocation {
    /// Stable plane name.
    pub name: &'static str,
    /// Bytes consumed by one content tile in this plane.
    pub bytes_per_tile: u64,
}

/// Descriptor-derived persistent surface layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileSurfaceLayout {
    /// Paint-model implementation owning the planes.
    pub model: PaintModelId,
    /// Persisted implementation version.
    pub model_version: u32,
    /// Content tile edge length.
    pub tile_size: u32,
    /// Every persistent model plane.
    pub planes: Vec<TilePlaneAllocation>,
    /// Sum of all plane costs for one resident tile.
    pub bytes_per_tile: u64,
}

impl TileSurfaceLayout {
    /// Builds allocation metadata without assuming RGBA or a fixed channel count.
    pub fn from_model(model: &PaintModelDescriptor, tile_size: u32) -> Self {
        let pixels = u64::from(tile_size.max(1)).pow(2);
        let planes: Vec<_> = model
            .persistent_planes
            .iter()
            .map(|plane| TilePlaneAllocation {
                name: plane.name,
                bytes_per_tile: plane.format.bytes_per_pixel() * pixels,
            })
            .collect();
        let bytes_per_tile = planes.iter().map(|plane| plane.bytes_per_tile).sum();
        Self {
            model: model.id,
            model_version: model.version,
            tile_size: tile_size.max(1),
            planes,
            bytes_per_tile,
        }
    }
}

/// One strict-order stroke replay record used by the RGBA compute backend.
#[derive(Clone, Debug)]
pub(crate) struct TileReplayStroke {
    pub segment_indices: Vec<u32>,
    pub material: PaintMaterialId,
    pub model: PaintModelId,
    pub deposition: DepositionMode,
}

/// Complete regeneration request for a single tile.
#[derive(Clone, Debug)]
pub(crate) struct TileRasterJob {
    pub key: TileKey,
    pub slot: u32,
    pub revision: u64,
    pub replay: Vec<TileReplayStroke>,
    pub halo: Vec<TileKey>,
}

/// A resident tile ready for one batched display pass.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TileDisplayInstance {
    pub key: TileKey,
    pub slot: u32,
    pub revision: u64,
    pub opacity: f32,
    pub layer_order: u32,
    pub layer_count: u32,
}

/// Immutable per-frame handoff from the tile scheduler to extraction.
#[derive(Clone, Debug, Default)]
pub(crate) struct TileWorkBatch {
    pub serial: u64,
    pub tile_size: u32,
    pub jobs: Vec<TileRasterJob>,
    pub display: Vec<TileDisplayInstance>,
}

#[derive(Clone, Copy, Debug)]
enum TileRenderFeedback {
    Complete { key: TileKey, revision: u64 },
    Presented { key: TileKey, revision: u64 },
    Retry { key: TileKey, revision: u64 },
    DeviceReset,
    Capacity { max_slots: u32 },
}

/// Lock-minimal completion channel shared between the main and render worlds.
#[derive(Resource, Clone, Default)]
pub(crate) struct TileFeedback(Arc<Mutex<Vec<TileRenderFeedback>>>);

impl TileFeedback {
    pub(crate) fn complete(&self, key: TileKey, revision: u64) {
        self.push(TileRenderFeedback::Complete { key, revision });
    }

    pub(crate) fn retry(&self, key: TileKey, revision: u64) {
        self.push(TileRenderFeedback::Retry { key, revision });
    }

    pub(crate) fn presented(&self, key: TileKey, revision: u64) {
        self.push(TileRenderFeedback::Presented { key, revision });
    }

    pub(crate) fn device_reset(&self) {
        self.push(TileRenderFeedback::DeviceReset);
    }

    pub(crate) fn capacity(&self, max_slots: u32) {
        self.push(TileRenderFeedback::Capacity { max_slots });
    }

    fn push(&self, feedback: TileRenderFeedback) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(feedback);
    }

    fn drain(&self, output: &mut Vec<TileRenderFeedback>) {
        let mut feedback = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        output.append(&mut feedback);
    }
}

#[derive(Clone, Debug)]
struct TileRecord {
    state: TileCacheState,
    desired_revision: u64,
    display_revision: Option<u64>,
    presented_revision: Option<u64>,
    display_allowed: bool,
    slot: Option<u32>,
    visible: bool,
    last_visible_frame: u64,
    halo_radius_px: u32,
    was_evicted: bool,
}

/// Descriptor-driven persistent tile allocator and bounded replay scheduler.
#[derive(Resource)]
pub struct CanvasTileCache {
    document_generation: Option<u64>,
    layout: Option<TileSurfaceLayout>,
    budget_bytes: u64,
    records: HashMap<TileKey, TileRecord>,
    dirty: VecDeque<TileKey>,
    queued: HashSet<TileKey>,
    free_slots: Vec<u32>,
    next_slot: u32,
    slot_limit: u32,
    frame: u64,
    batch: TileWorkBatch,
    stats: TileCacheStats,
    invalidations: Vec<TileInvalidation>,
    feedback: Vec<TileRenderFeedback>,
    visible_keys: HashSet<TileKey>,
    scratch_pool: ScratchPlanePool,
}

impl Default for CanvasTileCache {
    fn default() -> Self {
        Self {
            document_generation: None,
            layout: None,
            budget_bytes: 256 * 1024 * 1024,
            records: HashMap::new(),
            dirty: VecDeque::with_capacity(512),
            queued: HashSet::with_capacity(512),
            free_slots: Vec::new(),
            next_slot: 0,
            slot_limit: u32::MAX,
            frame: 0,
            batch: TileWorkBatch::default(),
            stats: TileCacheStats::default(),
            invalidations: Vec::with_capacity(512),
            feedback: Vec::with_capacity(64),
            visible_keys: HashSet::new(),
            scratch_pool: ScratchPlanePool::with_budget(256 * 1024 * 1024),
        }
    }
}

impl CanvasTileCache {
    /// Current descriptor-derived surface layout.
    pub fn layout(&self) -> Option<&TileSurfaceLayout> {
        self.layout.as_ref()
    }

    /// Aggregate allocator and replay counters.
    pub fn stats(&self) -> TileCacheStats {
        self.stats
    }

    /// Current state for a logical tile.
    pub fn tile_state(&self, key: TileKey) -> Option<TileCacheState> {
        self.records.get(&key).map(|record| record.state)
    }

    pub(crate) fn batch(&self) -> &TileWorkBatch {
        &self.batch
    }

    fn reset_records(&mut self) {
        self.records.clear();
        self.dirty.clear();
        self.queued.clear();
        self.free_slots.clear();
        self.next_slot = 0;
        self.batch.jobs.clear();
        self.batch.display.clear();
        self.stats.resident_tiles = 0;
        self.stats.persistent_bytes = 0;
    }

    fn configure(
        &mut self,
        descriptor: &PaintModelDescriptor,
        tile_size: u32,
        budget_bytes: u64,
    ) -> bool {
        let layout = TileSurfaceLayout::from_model(descriptor, tile_size);
        let changed = self.layout.as_ref() != Some(&layout) || self.budget_bytes != budget_bytes;
        if !changed {
            return false;
        }
        self.layout = Some(layout);
        self.budget_bytes = budget_bytes;
        self.reset_records();
        self.scratch_pool = ScratchPlanePool::with_budget(budget_bytes);
        true
    }

    fn apply_feedback(&mut self, feedback: &TileFeedback) -> bool {
        self.feedback.clear();
        feedback.drain(&mut self.feedback);
        let mut device_reset = false;
        let feedback_items = core::mem::take(&mut self.feedback);
        for item in feedback_items {
            match item {
                TileRenderFeedback::Complete { key, revision } => {
                    if let Some(record) = self.records.get_mut(&key)
                        && record.desired_revision == revision
                        && record.state == TileCacheState::Scheduled
                    {
                        record.state = TileCacheState::Ready;
                        record.display_revision = Some(revision);
                        record.display_allowed = true;
                        self.stats.rasterized_tiles += 1;
                        if record.was_evicted {
                            record.was_evicted = false;
                            self.stats.regenerated_tiles += 1;
                        }
                    }
                }
                TileRenderFeedback::Presented { key, revision } => {
                    if let Some(record) = self.records.get_mut(&key)
                        && record.desired_revision == revision
                        && record.state == TileCacheState::Ready
                        && record
                            .display_revision
                            .is_some_and(|display| display >= revision)
                    {
                        record.presented_revision = Some(
                            record
                                .presented_revision
                                .map_or(revision, |presented| presented.max(revision)),
                        );
                    }
                }
                TileRenderFeedback::Retry { key, revision } => {
                    if self
                        .records
                        .get(&key)
                        .is_some_and(|record| record.desired_revision == revision)
                    {
                        self.records.get_mut(&key).unwrap().state = TileCacheState::Dirty;
                        self.enqueue(key);
                    }
                }
                TileRenderFeedback::DeviceReset => device_reset = true,
                TileRenderFeedback::Capacity { max_slots } => {
                    self.slot_limit = max_slots.max(1);
                }
            }
        }
        device_reset
    }

    fn invalidate(&mut self, invalidation: TileInvalidation) {
        match invalidation.cause {
            TileInvalidationCause::StrokeCommit => self.stats.stroke_invalidations += 1,
            TileInvalidationCause::EffectDependency if invalidation.halo_radius_px > 0 => {
                self.stats.effect_expanded_invalidations += 1;
            }
            _ => {}
        }
        let record = self.records.entry(invalidation.key).or_insert(TileRecord {
            state: TileCacheState::Dirty,
            desired_revision: invalidation.revision,
            display_revision: None,
            presented_revision: None,
            display_allowed: false,
            slot: None,
            visible: false,
            last_visible_frame: 0,
            halo_radius_px: invalidation.halo_radius_px,
            was_evicted: false,
        });
        if invalidation.revision >= record.desired_revision {
            record.desired_revision = invalidation.revision;
            record.state = TileCacheState::Dirty;
            record.halo_radius_px = invalidation.halo_radius_px;
            record.display_allowed &= invalidation.allow_stale_display;
        }
        self.enqueue(invalidation.key);
    }

    fn enqueue(&mut self, key: TileKey) {
        if self.queued.insert(key) {
            self.dirty.push_back(key);
        }
    }

    fn update_visibility(&mut self, document: &StrokeDocument, bounds: Option<Rect>) {
        self.frame = self.frame.wrapping_add(1);
        self.visible_keys.clear();
        for record in self.records.values_mut() {
            record.visible = false;
        }

        let (Some(layout), Some(bounds)) = (&self.layout, bounds) else {
            return;
        };
        let size = layout.tile_size as f32;
        let min_x = (bounds.min.x / size).floor() as i32;
        let min_y = (bounds.min.y / size).floor() as i32;
        let max_x = (bounds.max.x / size).floor() as i32;
        let max_y = (bounds.max.y / size).floor() as i32;
        for layer in document.layers().iter().filter(|layer| layer.visible) {
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let key = TileKey::new(layer.id, x, y);
                    if document.tile_has_visible_strokes(key) || self.records.contains_key(&key) {
                        self.visible_keys.insert(key);
                        if let Some(record) = self.records.get_mut(&key) {
                            record.visible = true;
                            record.last_visible_frame = self.frame;
                            if record.state == TileCacheState::Evicted {
                                record.state = TileCacheState::Dirty;
                                self.enqueue(key);
                            }
                        } else if document.tile_has_visible_strokes(key) {
                            self.invalidate(TileInvalidation {
                                key,
                                revision: document.revision(),
                                cause: TileInvalidationCause::DeviceLoss,
                                allow_stale_display: false,
                                halo_radius_px: 0,
                            });
                            if let Some(record) = self.records.get_mut(&key) {
                                record.visible = true;
                                record.last_visible_frame = self.frame;
                            }
                        }
                    }
                }
            }
        }
    }

    fn schedule(
        &mut self,
        document: &StrokeDocument,
        budget: u32,
        effect_budget: u32,
        scratch_planes: &[PaintPlaneDescriptor],
    ) {
        self.batch.serial = self.batch.serial.wrapping_add(1);
        self.batch.tile_size = self.layout.as_ref().map_or(256, |layout| layout.tile_size);
        self.batch.jobs.clear();
        self.batch.display.clear();

        for (&key, record) in &self.records {
            if record.visible
                && record.display_allowed
                && record.display_revision.is_some()
                && let Some(slot) = record.slot
            {
                self.batch.display.push(TileDisplayInstance {
                    key,
                    slot,
                    revision: record.display_revision.unwrap(),
                    opacity: document.layer(key.layer).map_or(0.0, |layer| layer.opacity),
                    layer_order: document
                        .layer_index(key.layer)
                        .expect("displayable tile must belong to a document layer")
                        as u32,
                    layer_count: document.layers().len() as u32,
                });
            }
        }
        self.batch.display.sort_by_key(|instance| {
            (
                document
                    .layer_index(instance.key.layer)
                    .unwrap_or(usize::MAX),
                instance.key.y,
                instance.key.x,
            )
        });

        for prefer_visible in [true, false] {
            while self.batch.jobs.len() < budget as usize {
                let Some(key) = self.take_next_dirty(prefer_visible) else {
                    break;
                };
                let Some(slot) = self.ensure_slot(key) else {
                    self.enqueue(key);
                    break;
                };
                let record = &self.records[&key];
                let replay = document
                    .tile_strokes(key)
                    .iter()
                    .filter_map(|entry| {
                        let stroke = &document.strokes()[entry.stroke_index];
                        (stroke.visible && stroke.complete).then_some(TileReplayStroke {
                            segment_indices: entry.segments.clone(),
                            material: stroke.brush.paint.material,
                            model: stroke.brush.paint.model,
                            deposition: stroke.brush.deposition,
                        })
                    })
                    .collect();
                let revision = record.desired_revision;
                let halo = neighbor_halo(key, record.halo_radius_px, self.batch.tile_size);
                self.records.get_mut(&key).unwrap().state = TileCacheState::Scheduled;
                self.batch.jobs.push(TileRasterJob {
                    key,
                    slot,
                    revision,
                    replay,
                    halo,
                });
            }
        }
        self.stats.dirty_tiles = self
            .records
            .values()
            .filter(|record| {
                matches!(
                    record.state,
                    TileCacheState::Dirty | TileCacheState::Scheduled
                )
            })
            .count() as u64;

        let available_scratch = self
            .budget_bytes
            .saturating_sub(self.stats.persistent_bytes);
        self.scratch_pool.set_budget(available_scratch);
        let effect_tiles = (self.batch.jobs.len() as u32).min(effect_budget);
        if effect_tiles > 0
            && let Some(lease) =
                self.scratch_pool
                    .lease(scratch_planes, self.batch.tile_size, effect_tiles)
        {
            self.stats.scratch_bytes = self.scratch_pool.leased_bytes();
            self.stats.scratch_high_water_bytes = self.scratch_pool.high_water_bytes();
            self.scratch_pool.release(lease);
            self.stats.scratch_bytes = self.scratch_pool.leased_bytes();
        }
    }

    fn take_next_dirty(&mut self, prefer_visible: bool) -> Option<TileKey> {
        let attempts = self.dirty.len();
        for _ in 0..attempts {
            let key = self.dirty.pop_front()?;
            self.queued.remove(&key);
            let Some(record) = self.records.get(&key) else {
                continue;
            };
            if record.state != TileCacheState::Dirty {
                continue;
            }
            if record.visible == prefer_visible {
                return Some(key);
            }
            self.enqueue(key);
        }
        None
    }

    fn ensure_slot(&mut self, key: TileKey) -> Option<u32> {
        if let Some(slot) = self.records.get(&key)?.slot {
            return Some(slot);
        }
        let bytes_per_tile = self.layout.as_ref()?.bytes_per_tile.max(1);
        let can_grow = self.next_slot < self.slot_limit
            && (self.next_slot as u64 + 1).saturating_mul(bytes_per_tile) <= self.budget_bytes;
        let slot = if let Some(slot) = self.free_slots.pop() {
            slot
        } else if can_grow {
            let slot = self.next_slot;
            self.next_slot = self.next_slot.saturating_add(1);
            slot
        } else {
            let evict = self
                .records
                .iter()
                .filter(|(candidate_key, record)| {
                    **candidate_key != key
                        && !record.visible
                        && record.state == TileCacheState::Ready
                        && record.slot.is_some()
                })
                .min_by_key(|(_, record)| record.last_visible_frame)
                .map(|(&key, _)| key)?;
            let record = self.records.get_mut(&evict).unwrap();
            let slot = record.slot.take().unwrap();
            record.state = TileCacheState::Evicted;
            record.display_revision = None;
            record.presented_revision = None;
            record.display_allowed = false;
            record.was_evicted = true;
            self.stats.evicted_tiles += 1;
            slot
        };
        self.records.get_mut(&key)?.slot = Some(slot);
        self.stats.resident_tiles = self
            .records
            .values()
            .filter(|record| record.slot.is_some())
            .count() as u64;
        self.stats.persistent_bytes = self.stats.resident_tiles.saturating_mul(bytes_per_tile);
        Some(slot)
    }

    fn tile_is_ready(&self, key: TileKey, revision: u64) -> bool {
        self.records.get(&key).is_some_and(|record| {
            record.state == TileCacheState::Ready
                && record.display_allowed
                && record.slot.is_some()
                && record
                    .display_revision
                    .is_some_and(|display| display >= revision)
        })
    }
}

/// Returns every neighboring tile needed to provide a radius-sized source halo.
pub fn neighbor_halo(key: TileKey, radius_px: u32, tile_size: u32) -> Vec<TileKey> {
    if radius_px == 0 {
        return Vec::new();
    }
    let radius_tiles = radius_px.div_ceil(tile_size.max(1)) as i32;
    let side = radius_tiles * 2 + 1;
    let mut result = Vec::with_capacity((side * side - 1) as usize);
    for y in -radius_tiles..=radius_tiles {
        for x in -radius_tiles..=radius_tiles {
            if x != 0 || y != 0 {
                result.push(TileKey::new(key.layer, key.x + x, key.y + y));
            }
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn maintain_canvas_tiles(
    mut document: ResMut<StrokeDocument>,
    mut cache: ResMut<CanvasTileCache>,
    settings: Res<StrokeRendererSettings>,
    models: Res<PaintModelRegistry>,
    effects: Res<EffectRegistry>,
    feedback: Res<TileFeedback>,
    telemetry: Res<StrokeTelemetry>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
) {
    let model_id = settings.pen.paint.model;
    let Some(model) = models.get(model_id) else {
        return;
    };

    if cache.document_generation != Some(document.geometry_generation()) {
        cache.reset_records();
        cache.document_generation = Some(document.geometry_generation());
        document.invalidate_cache_layout();
    }

    document.set_tile_size(settings.tile_size);
    let influence = document.effects().combined_influence(&effects);
    match influence {
        EffectInfluence::Local { radius_px } => document.set_effect_radius(radius_px),
        EffectInfluence::Global => document.set_effect_radius(0),
    }

    if cache.configure(model, settings.tile_size, settings.gpu_cache_budget_bytes) {
        document.invalidate_cache_layout();
    }
    if cache.apply_feedback(&feedback) {
        cache.reset_records();
        document.regenerate_after_device_loss();
        cache.document_generation = Some(document.geometry_generation());
    }

    document.drain_invalidations(&mut cache.invalidations);
    let invalidations = core::mem::take(&mut cache.invalidations);
    for invalidation in invalidations {
        cache.invalidate(invalidation);
    }

    let visible_bounds = camera_visible_bounds(&cameras);
    cache.update_visibility(&document, visible_bounds);
    let mut scratch_planes = model.scratch_planes.clone();
    for node in document
        .effects()
        .nodes()
        .iter()
        .filter(|node| node.enabled)
    {
        if let Some(descriptor) = effects.get(node.effect) {
            scratch_planes.extend(descriptor.scratch_planes.iter().cloned());
        }
    }
    let budget = if document.is_drawing() {
        settings.max_dirty_tiles_while_drawing
    } else {
        settings.max_dirty_tiles_while_idle
    };
    let effect_budget = if document.is_drawing() {
        settings.max_effect_tiles_while_drawing
    } else {
        settings.max_effect_tiles_while_idle
    };
    cache.schedule(&document, budget, effect_budget, &scratch_planes);
    document.refresh_cache_handoffs(|key, revision| cache.tile_is_ready(key, revision));
    telemetry.record_tiles(document.revision(), cache.stats());
}

fn camera_visible_bounds(
    cameras: &Query<(&Camera, &GlobalTransform), With<Camera2d>>,
) -> Option<Rect> {
    let mut bounds = Rect::EMPTY;
    let mut found = false;
    for (camera, transform) in cameras.iter().filter(|(camera, _)| camera.is_active) {
        let Some(viewport) = camera.logical_viewport_rect() else {
            continue;
        };
        for point in [
            viewport.min,
            Vec2::new(viewport.max.x, viewport.min.y),
            viewport.max,
            Vec2::new(viewport.min.x, viewport.max.y),
        ] {
            if let Ok(world) = camera.viewport_to_world_2d(transform, point) {
                bounds = bounds.union_point(world);
                found = true;
            }
        }
    }
    found.then_some(bounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DepositionOrdering, PaintPlaneClearValue, PaintPlaneDescriptor, PaintPlaneFormat};

    fn mock_two_plane_model() -> PaintModelDescriptor {
        PaintModelDescriptor {
            id: PaintModelId(77),
            version: 3,
            name: "mock pigment",
            persistent_planes: vec![
                PaintPlaneDescriptor {
                    name: "absorption",
                    semantic: "mock absorption",
                    format: PaintPlaneFormat::Rgba16Float,
                    clear: PaintPlaneClearValue::Zero,
                },
                PaintPlaneDescriptor {
                    name: "concentration",
                    semantic: "mock concentration",
                    format: PaintPlaneFormat::R32Float,
                    clear: PaintPlaneClearValue::Zero,
                },
            ],
            scratch_planes: vec![PaintPlaneDescriptor {
                name: "mix_scratch",
                semantic: "mock scratch",
                format: PaintPlaneFormat::R32Float,
                clear: PaintPlaneClearValue::Zero,
            }],
            deposition_ordering: DepositionOrdering::Strict,
        }
    }

    #[test]
    fn mock_two_plane_model_drives_allocation_and_budget() {
        let layout = TileSurfaceLayout::from_model(&mock_two_plane_model(), 64);
        assert_eq!(layout.planes.len(), 2);
        assert_eq!(layout.bytes_per_tile, 64 * 64 * (8 + 4));

        let mut cache = CanvasTileCache::default();
        assert!(cache.configure(&mock_two_plane_model(), 64, layout.bytes_per_tile * 2));
        cache.invalidate(TileInvalidation {
            key: TileKey::new(LayerId(0), 0, 0),
            revision: 1,
            cause: TileInvalidationCause::StrokeCommit,
            allow_stale_display: false,
            halo_radius_px: 0,
        });
        assert_eq!(cache.ensure_slot(TileKey::new(LayerId(0), 0, 0)), Some(0));
        assert_eq!(cache.stats().persistent_bytes, layout.bytes_per_tile);
    }

    #[test]
    fn nonzero_radius_requests_complete_neighbor_halo() {
        let center = TileKey::new(LayerId(2), -3, 8);
        let halo = neighbor_halo(center, 257, 256);
        assert_eq!(halo.len(), 24);
        assert!(halo.contains(&TileKey::new(LayerId(2), -5, 6)));
        assert!(halo.contains(&TileKey::new(LayerId(2), -1, 10)));
        assert!(!halo.contains(&center));
    }

    #[test]
    fn cache_handoff_uses_a_complete_displayable_tile_without_double_presenting() {
        let model = mock_two_plane_model();
        let bytes = TileSurfaceLayout::from_model(&model, 16).bytes_per_tile;
        let mut cache = CanvasTileCache::default();
        cache.configure(&model, 16, bytes);
        let key = TileKey::new(LayerId(0), 0, 0);
        cache.invalidate(TileInvalidation {
            key,
            revision: 7,
            cause: TileInvalidationCause::StrokeCommit,
            allow_stale_display: false,
            halo_radius_px: 0,
        });
        assert_eq!(cache.ensure_slot(key), Some(0));
        cache.records.get_mut(&key).unwrap().state = TileCacheState::Scheduled;

        let feedback = TileFeedback::default();
        feedback.complete(key, 7);
        assert!(!cache.apply_feedback(&feedback));
        assert_eq!(cache.tile_state(key), Some(TileCacheState::Ready));
        assert!(cache.tile_is_ready(key, 7));

        feedback.presented(key, 7);
        assert!(!cache.apply_feedback(&feedback));
        assert!(cache.tile_is_ready(key, 7));
    }

    #[test]
    fn display_batch_respects_document_layer_order_and_opacity() {
        let model = crate::RgbaPaintModel::descriptor();
        let bytes = TileSurfaceLayout::from_model(&model, 16).bytes_per_tile * 4;
        let mut cache = CanvasTileCache::default();
        cache.configure(&model, 16, bytes);
        let mut document = StrokeDocument::default();
        let lower = document.active_layer();
        let upper = document.add_layer("Upper");
        document.set_layer_opacity(upper, 0.35);

        for (slot, layer) in [lower, upper].into_iter().enumerate() {
            let key = TileKey::new(layer, 0, 0);
            cache.invalidate(TileInvalidation {
                key,
                revision: 1,
                cause: TileInvalidationCause::StrokeCommit,
                allow_stale_display: false,
                halo_radius_px: 0,
            });
            let record = cache.records.get_mut(&key).unwrap();
            record.state = TileCacheState::Ready;
            record.display_revision = Some(1);
            record.display_allowed = true;
            record.visible = true;
            record.slot = Some(slot as u32);
        }

        cache.schedule(&document, 0, 0, &[]);
        assert_eq!(cache.batch.display.len(), 2);
        assert_eq!(cache.batch.display[0].key.layer, lower);
        assert_eq!(cache.batch.display[0].opacity, 1.0);
        assert_eq!(cache.batch.display[1].key.layer, upper);
        assert_eq!(cache.batch.display[1].opacity, 0.35);

        document.move_layer(upper, 0);
        cache.schedule(&document, 0, 0, &[]);
        assert_eq!(cache.batch.display[0].key.layer, upper);
    }

    #[test]
    fn eviction_only_reuses_clean_offscreen_slots() {
        let model = mock_two_plane_model();
        let bytes = TileSurfaceLayout::from_model(&model, 16).bytes_per_tile;
        let mut cache = CanvasTileCache::default();
        cache.configure(&model, 16, bytes);
        let first = TileKey::new(LayerId(0), 0, 0);
        let second = TileKey::new(LayerId(0), 1, 0);
        cache.invalidate(TileInvalidation {
            key: first,
            revision: 1,
            cause: TileInvalidationCause::StrokeCommit,
            allow_stale_display: false,
            halo_radius_px: 0,
        });
        assert_eq!(cache.ensure_slot(first), Some(0));
        {
            let record = cache.records.get_mut(&first).unwrap();
            record.state = TileCacheState::Ready;
            record.display_revision = Some(1);
        }
        cache.invalidate(TileInvalidation {
            key: second,
            revision: 2,
            cause: TileInvalidationCause::StrokeCommit,
            allow_stale_display: false,
            halo_radius_px: 0,
        });
        assert_eq!(cache.ensure_slot(second), Some(0));
        assert_eq!(cache.tile_state(first), Some(TileCacheState::Evicted));
        assert_eq!(cache.stats().evicted_tiles, 1);
    }
}
