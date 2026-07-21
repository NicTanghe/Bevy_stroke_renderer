use alloc::sync::Arc;
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use bevy_ecs::{
    resource::Resource,
    system::{Local, Res},
};
use tracing::info;

use crate::StrokeRendererSettings;
use crate::TileCacheStats;

#[derive(Default)]
struct StrokeTelemetryInner {
    pen_messages: AtomicU64,
    input_batches: AtomicU64,
    last_batch_depth: AtomicU64,
    max_batch_depth: AtomicU64,
    input_cpu_nanos: AtomicU64,
    resident_points: AtomicU64,
    resident_segments: AtomicU64,
    oldest_extracted_sample_age_micros: AtomicU64,
    extracted_sample_age_micros: AtomicU64,
    uploaded_bytes_last_frame: AtomicU64,
    uploaded_bytes_total: AtomicU64,
    gpu_reallocations: AtomicU64,
    document_revision: AtomicU64,
    resident_tiles: AtomicU64,
    dirty_tiles: AtomicU64,
    rasterized_tiles: AtomicU64,
    evicted_tiles: AtomicU64,
    regenerated_tiles: AtomicU64,
    tile_cache_bytes: AtomicU64,
    scratch_high_water_bytes: AtomicU64,
}

/// Lock-free telemetry shared by the main and render worlds.
#[derive(Resource, Clone, Default)]
pub struct StrokeTelemetry(Arc<StrokeTelemetryInner>);

/// One coherent read of [`StrokeTelemetry`].
#[derive(Clone, Copy, Debug, Default)]
pub struct StrokeTelemetrySnapshot {
    /// Pen messages consumed since startup.
    pub pen_messages: u64,
    /// Input updates containing zero or more messages.
    pub input_batches: u64,
    /// Messages consumed in the latest input update.
    pub last_batch_depth: u64,
    /// Largest message batch observed.
    pub max_batch_depth: u64,
    /// CPU time spent collecting the latest message batch.
    pub input_cpu: Duration,
    /// Points retained by the authoritative document and live overlay buffer.
    pub resident_points: u64,
    /// Segments retained by the authoritative document and GPU replay buffer.
    pub resident_segments: u64,
    /// Age of the newest sample when it reached render extraction.
    pub extracted_sample_age: Duration,
    /// Age of the oldest sample in the extracted input batch.
    pub oldest_extracted_sample_age: Duration,
    /// Bytes uploaded by the latest prepare step.
    pub uploaded_bytes_last_frame: u64,
    /// Bytes uploaded since startup.
    pub uploaded_bytes_total: u64,
    /// Number of geometric GPU-buffer growth operations.
    pub gpu_reallocations: u64,
    /// Latest authoritative document revision.
    pub document_revision: u64,
    /// Persistent tiles occupying paint-model plane slots.
    pub resident_tiles: u64,
    /// Dirty or scheduled tiles awaiting a committed revision.
    pub dirty_tiles: u64,
    /// Total tile regeneration jobs completed.
    pub rasterized_tiles: u64,
    /// Clean tiles removed by the memory budget.
    pub evicted_tiles: u64,
    /// Evicted tiles regenerated from vector history.
    pub regenerated_tiles: u64,
    /// Aggregate resident bytes across all persistent model planes.
    pub tile_cache_bytes: u64,
    /// Peak descriptor-driven transient scratch usage.
    pub scratch_high_water_bytes: u64,
}

impl StrokeTelemetry {
    /// Reads all counters using relaxed ordering; values are diagnostic only.
    pub fn snapshot(&self) -> StrokeTelemetrySnapshot {
        let inner = &self.0;
        StrokeTelemetrySnapshot {
            pen_messages: inner.pen_messages.load(Ordering::Relaxed),
            input_batches: inner.input_batches.load(Ordering::Relaxed),
            last_batch_depth: inner.last_batch_depth.load(Ordering::Relaxed),
            max_batch_depth: inner.max_batch_depth.load(Ordering::Relaxed),
            input_cpu: Duration::from_nanos(inner.input_cpu_nanos.load(Ordering::Relaxed)),
            resident_points: inner.resident_points.load(Ordering::Relaxed),
            resident_segments: inner.resident_segments.load(Ordering::Relaxed),
            extracted_sample_age: Duration::from_micros(
                inner.extracted_sample_age_micros.load(Ordering::Relaxed),
            ),
            oldest_extracted_sample_age: Duration::from_micros(
                inner
                    .oldest_extracted_sample_age_micros
                    .load(Ordering::Relaxed),
            ),
            uploaded_bytes_last_frame: inner.uploaded_bytes_last_frame.load(Ordering::Relaxed),
            uploaded_bytes_total: inner.uploaded_bytes_total.load(Ordering::Relaxed),
            gpu_reallocations: inner.gpu_reallocations.load(Ordering::Relaxed),
            document_revision: inner.document_revision.load(Ordering::Relaxed),
            resident_tiles: inner.resident_tiles.load(Ordering::Relaxed),
            dirty_tiles: inner.dirty_tiles.load(Ordering::Relaxed),
            rasterized_tiles: inner.rasterized_tiles.load(Ordering::Relaxed),
            evicted_tiles: inner.evicted_tiles.load(Ordering::Relaxed),
            regenerated_tiles: inner.regenerated_tiles.load(Ordering::Relaxed),
            tile_cache_bytes: inner.tile_cache_bytes.load(Ordering::Relaxed),
            scratch_high_water_bytes: inner.scratch_high_water_bytes.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn record_input_batch(
        &self,
        message_count: usize,
        elapsed: Duration,
        point_count: usize,
        segment_count: usize,
    ) {
        let depth = message_count as u64;
        self.0.pen_messages.fetch_add(depth, Ordering::Relaxed);
        self.0.input_batches.fetch_add(1, Ordering::Relaxed);
        self.0.last_batch_depth.store(depth, Ordering::Relaxed);
        self.0.max_batch_depth.fetch_max(depth, Ordering::Relaxed);
        self.0.input_cpu_nanos.store(
            elapsed.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        self.0
            .resident_points
            .store(point_count as u64, Ordering::Relaxed);
        self.0
            .resident_segments
            .store(segment_count as u64, Ordering::Relaxed);
    }

    pub(crate) fn record_extraction_ages(&self, oldest: Duration, newest: Duration) {
        self.0.oldest_extracted_sample_age_micros.store(
            oldest.as_micros().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        self.0.extracted_sample_age_micros.store(
            newest.as_micros().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    pub(crate) fn record_upload(&self, bytes: usize, reallocations: u64) {
        self.0
            .uploaded_bytes_last_frame
            .store(bytes as u64, Ordering::Relaxed);
        self.0
            .uploaded_bytes_total
            .fetch_add(bytes as u64, Ordering::Relaxed);
        self.0
            .gpu_reallocations
            .fetch_add(reallocations, Ordering::Relaxed);
    }

    pub(crate) fn record_tiles(&self, revision: u64, stats: TileCacheStats) {
        self.0.document_revision.store(revision, Ordering::Relaxed);
        self.0
            .resident_tiles
            .store(stats.resident_tiles, Ordering::Relaxed);
        self.0
            .dirty_tiles
            .store(stats.dirty_tiles, Ordering::Relaxed);
        self.0
            .rasterized_tiles
            .store(stats.rasterized_tiles, Ordering::Relaxed);
        self.0
            .evicted_tiles
            .store(stats.evicted_tiles, Ordering::Relaxed);
        self.0
            .regenerated_tiles
            .store(stats.regenerated_tiles, Ordering::Relaxed);
        self.0
            .tile_cache_bytes
            .store(stats.persistent_bytes, Ordering::Relaxed);
        self.0
            .scratch_high_water_bytes
            .store(stats.scratch_high_water_bytes, Ordering::Relaxed);
    }
}

pub(crate) fn log_stroke_diagnostics(
    settings: Res<StrokeRendererSettings>,
    telemetry: Res<StrokeTelemetry>,
    mut last_log: Local<Option<Instant>>,
) {
    if !settings.log_diagnostics {
        return;
    }

    let now = Instant::now();
    if last_log.is_some_and(|last| now.duration_since(last) < Duration::from_secs(1)) {
        return;
    }
    *last_log = Some(now);

    let sample = telemetry.snapshot();
    info!(
        target: "hamerons_stroke_render::latency",
        pen_messages = sample.pen_messages,
        batch_depth = sample.last_batch_depth,
        max_batch_depth = sample.max_batch_depth,
        input_cpu_us = sample.input_cpu.as_micros() as u64,
        oldest_extracted_sample_age_us = sample.oldest_extracted_sample_age.as_micros() as u64,
        extracted_sample_age_us = sample.extracted_sample_age.as_micros() as u64,
        points = sample.resident_points,
        segments = sample.resident_segments,
        upload_bytes = sample.uploaded_bytes_last_frame,
        upload_total_bytes = sample.uploaded_bytes_total,
        gpu_reallocations = sample.gpu_reallocations,
        document_revision = sample.document_revision,
        resident_tiles = sample.resident_tiles,
        dirty_tiles = sample.dirty_tiles,
        rasterized_tiles = sample.rasterized_tiles,
        evicted_tiles = sample.evicted_tiles,
        cache_bytes = sample.tile_cache_bytes,
        "aggregate pen-to-GPU stroke telemetry"
    );
}
