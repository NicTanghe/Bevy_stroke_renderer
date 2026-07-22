use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    thread::{self, JoinHandle},
};

use bevy_ecs::resource::Resource;
use bevy_math::{FloatExt, Rect, UVec2, Vec2};
use png::{BitDepth, ColorType, Compression};
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};
use zip::{result::ZipError, write::SimpleFileOptions, CompressionMethod, ZipArchive, ZipWriter};

use super::*;
use crate::{EffectNode, EffectNodeId};

/// Current authoritative payload schema stored inside Hamerons `.kra` files.
pub const DOCUMENT_SCHEMA_VERSION: u32 = 3;

const KRA_MIMETYPE: &[u8] = b"application/x-krita";
const DOCUMENT_ENTRY: &str = "hamerons/document.ron";
const MAX_DOCUMENT_BYTES: u64 = 512 * 1024 * 1024;
const FORMAT_ID: &str = "org.hamerons.stroke-document";
const KRITA_IMAGE_NAME: &str = "unnamed";
const KRITA_TILE_SIZE: u32 = 64;
const PAPER_SRGB: [u8; 4] = [248, 247, 244, 255];
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A model/effect implementation needed by a loaded document is unavailable.
/// The authoritative opaque records remain intact so the file can be resaved.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DocumentCompatibilityIssue {
    PaintModelUnavailable {
        model: PaintModelId,
        required_version: u32,
    },
    PaintModelVersionMismatch {
        model: PaintModelId,
        required_version: u32,
        available_version: u32,
    },
    EffectUnavailable {
        effect: EffectId,
        required_version: u32,
    },
    EffectVersionMismatch {
        effect: EffectId,
        required_version: u32,
        available_version: u32,
    },
}

/// Successfully loaded document plus non-destructive compatibility diagnostics.
pub struct LoadedStrokeDocument {
    pub document: StrokeDocument,
    pub source_schema_version: u32,
    pub migrated_from: Option<u32>,
    pub compatibility_issues: Vec<DocumentCompatibilityIssue>,
}

/// Result metadata for an atomic save or background checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentSaveReport {
    pub path: PathBuf,
    pub revision: u64,
    pub bytes_written: u64,
}

/// Durable document read/write failures. Loading never mutates the current document.
#[derive(Debug)]
pub enum DocumentIoError {
    Io(io::Error),
    Archive(String),
    Encode(String),
    InvalidDocument(String),
    UnsupportedSchema { found: u32, supported: u32 },
    ActiveContacts,
    BackgroundWorkerPanicked,
}

impl fmt::Display for DocumentIoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "document I/O failed: {error}"),
            Self::Archive(error) => write!(formatter, "invalid .kra archive: {error}"),
            Self::Encode(error) => write!(formatter, "document encoding failed: {error}"),
            Self::InvalidDocument(error) => write!(formatter, "invalid document: {error}"),
            Self::UnsupportedSchema { found, supported } => write!(
                formatter,
                "document schema {found} is newer than supported schema {supported}"
            ),
            Self::ActiveContacts => {
                formatter.write_str("cannot checkpoint while a drawing contact is active")
            }
            Self::BackgroundWorkerPanicked => {
                formatter.write_str("background checkpoint worker panicked")
            }
        }
    }
}

impl std::error::Error for DocumentIoError {}

impl From<io::Error> for DocumentIoError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<ZipError> for DocumentIoError {
    fn from(error: ZipError) -> Self {
        Self::Archive(error.to_string())
    }
}

/// Result of asking the bounded checkpoint service to start work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointRequest {
    Started,
    Busy,
    Unchanged,
}

/// Single-flight background serializer. At most one complete document snapshot
/// is retained by a worker, preventing an unbounded autosave queue.
#[derive(Resource, Default)]
pub struct DocumentCheckpointManager {
    in_flight: Option<JoinHandle<Result<DocumentSaveReport, DocumentIoError>>>,
    in_flight_path: Option<PathBuf>,
    in_flight_revision: Option<u64>,
    last_saved: Option<(PathBuf, u64)>,
}

impl DocumentCheckpointManager {
    pub fn request(
        &mut self,
        document: &StrokeDocument,
        path: impl AsRef<Path>,
    ) -> Result<CheckpointRequest, DocumentIoError> {
        if self.in_flight.is_some() {
            return Ok(CheckpointRequest::Busy);
        }
        let path = path.as_ref().to_path_buf();
        if self
            .last_saved
            .as_ref()
            .is_some_and(|(saved_path, revision)| {
                saved_path == &path && *revision == document.revision
            })
        {
            return Ok(CheckpointRequest::Unchanged);
        }
        let archive = ArchiveDocument::from_document(document)?;
        let revision = archive.revision;
        let worker_path = path.clone();
        self.in_flight = Some(thread::spawn(move || {
            save_archive_atomic(archive, &worker_path)
        }));
        self.in_flight_path = Some(path);
        self.in_flight_revision = Some(revision);
        Ok(CheckpointRequest::Started)
    }

    pub fn is_busy(&self) -> bool {
        self.in_flight.is_some()
    }

    pub fn poll(&mut self) -> Option<Result<DocumentSaveReport, DocumentIoError>> {
        if !self
            .in_flight
            .as_ref()
            .is_some_and(|worker| worker.is_finished())
        {
            return None;
        }
        let result = self
            .in_flight
            .take()
            .expect("finished worker checked above")
            .join()
            .map_err(|_| DocumentIoError::BackgroundWorkerPanicked)
            .and_then(|result| result);
        if let Ok(report) = &result {
            self.last_saved = Some((report.path.clone(), report.revision));
        }
        self.in_flight_path = None;
        self.in_flight_revision = None;
        Some(result)
    }
}

impl StrokeDocument {
    /// Atomically writes the authoritative document into a Krita `.kra` ZIP
    /// container. The previous valid file remains untouched on pre-rename errors.
    pub fn save_kra(&self, path: impl AsRef<Path>) -> Result<DocumentSaveReport, DocumentIoError> {
        save_archive_atomic(ArchiveDocument::from_document(self)?, path.as_ref())
    }

    /// Loads and validates a Hamerons `.kra` without mutating an existing document.
    pub fn load_kra(
        path: impl AsRef<Path>,
        paint_models: &PaintModelRegistry,
        effects: &EffectRegistry,
    ) -> Result<LoadedStrokeDocument, DocumentIoError> {
        let mut archive = read_archive(path.as_ref())?;
        let source_schema_version = archive.schema_version;
        let migrated_from = migrate_archive(&mut archive)?;
        let document = archive.into_document()?;
        let compatibility_issues = document.compatibility_issues(paint_models, effects);
        Ok(LoadedStrokeDocument {
            document,
            source_schema_version,
            migrated_from,
            compatibility_issues,
        })
    }

    /// Reports unavailable/newer implementations while retaining every opaque record.
    pub fn compatibility_issues(
        &self,
        paint_models: &PaintModelRegistry,
        effects: &EffectRegistry,
    ) -> Vec<DocumentCompatibilityIssue> {
        let mut issues = HashSet::new();
        for reference in self
            .paint_materials
            .iter()
            .map(|record| record.reference)
            .chain(self.strokes.iter().map(|stroke| stroke.brush.paint))
        {
            match paint_models.get(reference.model) {
                None => {
                    issues.insert(DocumentCompatibilityIssue::PaintModelUnavailable {
                        model: reference.model,
                        required_version: reference.model_version,
                    });
                }
                Some(model) if model.version != reference.model_version => {
                    issues.insert(DocumentCompatibilityIssue::PaintModelVersionMismatch {
                        model: reference.model,
                        required_version: reference.model_version,
                        available_version: model.version,
                    });
                }
                Some(_) => {}
            }
        }
        for node in self.effects.nodes() {
            match effects.get(node.effect) {
                None => {
                    issues.insert(DocumentCompatibilityIssue::EffectUnavailable {
                        effect: node.effect,
                        required_version: node.implementation_version,
                    });
                }
                Some(effect) if effect.implementation_version != node.implementation_version => {
                    issues.insert(DocumentCompatibilityIssue::EffectVersionMismatch {
                        effect: node.effect,
                        required_version: node.implementation_version,
                        available_version: effect.implementation_version,
                    });
                }
                Some(_) => {}
            }
        }
        let mut issues: Vec<_> = issues.into_iter().collect();
        issues.sort_by_key(compatibility_sort_key);
        issues
    }
}

fn compatibility_sort_key(issue: &DocumentCompatibilityIssue) -> (u8, u32, u32) {
    match issue {
        DocumentCompatibilityIssue::PaintModelUnavailable {
            model,
            required_version,
        } => (0, model.0, *required_version),
        DocumentCompatibilityIssue::PaintModelVersionMismatch {
            model,
            required_version,
            ..
        } => (1, model.0, *required_version),
        DocumentCompatibilityIssue::EffectUnavailable {
            effect,
            required_version,
        } => (2, effect.0, *required_version),
        DocumentCompatibilityIssue::EffectVersionMismatch {
            effect,
            required_version,
            ..
        } => (3, effect.0, *required_version),
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct ArchiveDocument {
    format: String,
    schema_version: u32,
    canvas_size: [u32; 2],
    tile_size: u32,
    revision: u64,
    next_stroke: u64,
    active_layer: u32,
    next_layer: u32,
    points: Vec<ArchivePoint>,
    segments: Vec<ArchiveSegment>,
    strokes: Vec<ArchiveStroke>,
    layers: Vec<ArchiveLayer>,
    materials: Vec<ArchiveMaterial>,
    effects: ArchiveEffects,
    undo: Vec<ArchiveHistoryCommand>,
    redo: Vec<ArchiveHistoryCommand>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct ArchivePoint {
    position: [f32; 2],
    half_width: f32,
    #[serde(default = "default_aspect_ratio")]
    aspect_ratio: f32,
    flow: f32,
    orientation: [f32; 2],
    twist_radians: f32,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct ArchiveSegment {
    start: u32,
    end: u32,
    model: u32,
    material: u32,
    deposition: u8,
    layer: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ArchiveStroke {
    id: u64,
    first_point: u32,
    point_count: u32,
    first_segment: u32,
    segment_count: u32,
    layer: u32,
    brush: ArchiveBrush,
    bounds: [f32; 4],
    visible: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct ArchiveBrush {
    model: u32,
    model_version: u32,
    material: u32,
    deposition: u8,
    diameter: f32,
    size_space: u8,
    minimum_diameter_ratio: f32,
    pressure_gamma: f32,
    flow: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct ArchiveLayer {
    id: u32,
    name: String,
    opacity: f32,
    composite: u8,
    visible: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ArchiveMaterial {
    model: u32,
    model_version: u32,
    material: u32,
    payload: ArchiveMaterialPayload,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum ArchiveMaterialPayload {
    PremultipliedLinearRgba([f32; 4]),
    ModelData { schema_version: u32, bytes: Vec<u8> },
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct ArchiveEffects {
    revision: u64,
    next_node: u64,
    nodes: Vec<ArchiveEffectNode>,
    dependencies: Vec<(u64, Vec<u64>)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ArchiveEffectNode {
    id: u64,
    effect: u32,
    implementation_version: u32,
    enabled: bool,
    parameters: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum ArchiveHistoryCommand {
    AddStroke(u64),
    Clear(Vec<u64>),
    Resize {
        before: [u32; 2],
        after: [u32; 2],
    },
    AddLayer {
        layer: ArchiveLayer,
        index: usize,
    },
    LayerVisibility {
        layer: u32,
        before: bool,
        after: bool,
    },
    LayerOpacity {
        layer: u32,
        before: f32,
        after: f32,
    },
    LayerMove {
        layer: u32,
        before: usize,
        after: usize,
    },
    LayerRename {
        layer: u32,
        before: String,
        after: String,
    },
}

impl ArchiveDocument {
    fn from_document(document: &StrokeDocument) -> Result<Self, DocumentIoError> {
        if !document.active.is_empty() || document.strokes.iter().any(|stroke| !stroke.complete) {
            return Err(DocumentIoError::ActiveContacts);
        }
        let mut materials: Vec<_> = document.paint_materials.iter().collect();
        materials.sort_by_key(|record| {
            (
                record.reference.model.0,
                record.reference.model_version,
                record.reference.material.0,
            )
        });
        let (effect_revision, next_node, effect_nodes, dependencies) =
            document.effects.persistence_parts();
        let mut dependencies: Vec<_> = dependencies
            .iter()
            .map(|(node, dependencies)| {
                let mut dependencies: Vec<_> = dependencies.iter().map(|id| id.0).collect();
                dependencies.sort_unstable();
                (node.0, dependencies)
            })
            .collect();
        dependencies.sort_by_key(|(node, _)| *node);

        Ok(Self {
            format: FORMAT_ID.into(),
            schema_version: DOCUMENT_SCHEMA_VERSION,
            canvas_size: document.canvas_size.to_array(),
            tile_size: document.tile_size,
            revision: document.revision,
            next_stroke: document.next_stroke,
            active_layer: document.active_layer.0,
            next_layer: document.next_layer,
            points: document
                .points
                .iter()
                .copied()
                .map(ArchivePoint::from)
                .collect(),
            segments: document
                .segments
                .iter()
                .copied()
                .map(ArchiveSegment::from)
                .collect(),
            strokes: document.strokes.iter().map(ArchiveStroke::from).collect(),
            layers: document.layers.iter().map(ArchiveLayer::from).collect(),
            materials: materials.into_iter().map(ArchiveMaterial::from).collect(),
            effects: ArchiveEffects {
                revision: effect_revision,
                next_node,
                nodes: effect_nodes.iter().map(ArchiveEffectNode::from).collect(),
                dependencies,
            },
            undo: document
                .undo
                .iter()
                .map(|command| ArchiveHistoryCommand::from_command(command, &document.strokes))
                .collect(),
            redo: document
                .redo
                .iter()
                .map(|command| ArchiveHistoryCommand::from_command(command, &document.strokes))
                .collect(),
        })
    }

    fn into_document(self) -> Result<StrokeDocument, DocumentIoError> {
        validate_archive(&self)?;
        let points: Vec<_> = self.points.into_iter().map(StrokePoint::from).collect();
        let segments: Vec<_> = self.segments.into_iter().map(StrokeSegment::from).collect();
        let layers: Vec<_> = self.layers.into_iter().map(CanvasLayer::from).collect();
        let effects = self.effects.into_graph()?;
        let mut paint_materials = PaintMaterialLibrary::default();
        for material in self.materials {
            paint_materials
                .add(material.into_record())
                .map_err(|error| DocumentIoError::InvalidDocument(error.into()))?;
        }
        let mut strokes = Vec::with_capacity(self.strokes.len());
        let mut stroke_lookup = HashMap::with_capacity(self.strokes.len());
        for archived in self.strokes {
            let index = strokes.len();
            let id = StrokeId(archived.id);
            stroke_lookup.insert(id, index);
            strokes.push(StrokeMetadata {
                id,
                first_point: archived.first_point,
                point_count: archived.point_count,
                first_segment: archived.first_segment,
                segment_count: archived.segment_count,
                layer: LayerId(archived.layer),
                brush: archived.brush.into_brush(),
                bounds: Rect::new(
                    archived.bounds[0],
                    archived.bounds[1],
                    archived.bounds[2],
                    archived.bounds[3],
                ),
                revision: self.revision.max(1),
                state: if archived.visible {
                    StrokeState::PendingCache
                } else {
                    StrokeState::Hidden
                },
                visible: archived.visible,
                complete: true,
                affected_tiles: Vec::new(),
            });
        }
        let undo = restore_history(self.undo, &stroke_lookup)?;
        let redo = restore_history(self.redo, &stroke_lookup)?;
        let max_stroke = strokes
            .iter()
            .map(|stroke| stroke.id.0.saturating_add(1))
            .max()
            .unwrap_or(0);
        let max_layer = layers
            .iter()
            .map(|layer| layer.id.0.saturating_add(1))
            .max()
            .unwrap_or(1);
        let mut document = StrokeDocument {
            points,
            segments,
            strokes,
            stroke_lookup,
            active: HashMap::new(),
            overlay_strokes: BTreeSet::new(),
            spatial_index: HashMap::new(),
            invalidations: Vec::new(),
            undo,
            redo,
            layers,
            active_layer: LayerId(self.active_layer),
            next_layer: self.next_layer.max(max_layer),
            layer_revision: 1,
            effects,
            paint_materials,
            next_stroke: self.next_stroke.max(max_stroke),
            revision: self.revision.max(1),
            contact_generation: 1,
            geometry_generation: 1,
            tile_size: self.tile_size.max(1),
            effect_radius_px: 0,
            canvas_size: UVec2::from_array(self.canvas_size).max(UVec2::ONE),
            input_batch_started: None,
            latest_sample_received: None,
        };
        document.rebuild_spatial_index();
        document.invalidate_all(TileInvalidationCause::CacheLayout, false);
        Ok(document)
    }
}

impl From<StrokePoint> for ArchivePoint {
    fn from(point: StrokePoint) -> Self {
        Self {
            position: point.position.to_array(),
            half_width: point.half_width,
            aspect_ratio: point.aspect_ratio,
            flow: point.flow,
            orientation: point.orientation.to_array(),
            twist_radians: point.twist_radians,
        }
    }
}

impl From<ArchivePoint> for StrokePoint {
    fn from(point: ArchivePoint) -> Self {
        Self {
            position: Vec2::from_array(point.position),
            half_width: point.half_width,
            aspect_ratio: point.aspect_ratio,
            flow: point.flow,
            orientation: Vec2::from_array(point.orientation),
            twist_radians: point.twist_radians,
        }
    }
}

impl From<StrokeSegment> for ArchiveSegment {
    fn from(segment: StrokeSegment) -> Self {
        Self {
            start: segment.start,
            end: segment.end,
            model: segment.model.0,
            material: segment.material.0,
            deposition: segment.deposition as u8,
            layer: segment.layer.0,
        }
    }
}

impl From<ArchiveSegment> for StrokeSegment {
    fn from(segment: ArchiveSegment) -> Self {
        Self {
            start: segment.start,
            end: segment.end,
            model: PaintModelId(segment.model),
            material: PaintMaterialId(segment.material),
            deposition: deposition_mode(segment.deposition),
            layer: LayerId(segment.layer),
        }
    }
}

impl From<&StrokeMetadata> for ArchiveStroke {
    fn from(stroke: &StrokeMetadata) -> Self {
        Self {
            id: stroke.id.0,
            first_point: stroke.first_point,
            point_count: stroke.point_count,
            first_segment: stroke.first_segment,
            segment_count: stroke.segment_count,
            layer: stroke.layer.0,
            brush: ArchiveBrush::from(stroke.brush),
            bounds: [
                stroke.bounds.min.x,
                stroke.bounds.min.y,
                stroke.bounds.max.x,
                stroke.bounds.max.y,
            ],
            visible: stroke.visible,
        }
    }
}

impl From<BrushProfile> for ArchiveBrush {
    fn from(brush: BrushProfile) -> Self {
        Self {
            model: brush.paint.model.0,
            model_version: brush.paint.model_version,
            material: brush.paint.material.0,
            deposition: brush.deposition as u8,
            diameter: brush.diameter,
            size_space: match brush.size_space {
                BrushSizeSpace::Document => 0,
                BrushSizeSpace::Screen => 1,
            },
            minimum_diameter_ratio: brush.minimum_diameter_ratio,
            pressure_gamma: brush.pressure_gamma,
            flow: brush.flow,
        }
    }
}

impl ArchiveBrush {
    fn into_brush(self) -> BrushProfile {
        BrushProfile {
            paint: PaintMaterialRef {
                model: PaintModelId(self.model),
                model_version: self.model_version,
                material: PaintMaterialId(self.material),
            },
            deposition: deposition_mode(self.deposition),
            diameter: self.diameter,
            size_space: if self.size_space == 0 {
                BrushSizeSpace::Document
            } else {
                BrushSizeSpace::Screen
            },
            minimum_diameter_ratio: self.minimum_diameter_ratio,
            pressure_gamma: self.pressure_gamma,
            flow: self.flow,
        }
    }
}

impl From<&CanvasLayer> for ArchiveLayer {
    fn from(layer: &CanvasLayer) -> Self {
        Self {
            id: layer.id.0,
            name: layer.name.clone(),
            opacity: layer.opacity,
            composite: 0,
            visible: layer.visible,
        }
    }
}

impl From<ArchiveLayer> for CanvasLayer {
    fn from(layer: ArchiveLayer) -> Self {
        Self {
            id: LayerId(layer.id),
            name: layer.name,
            opacity: layer.opacity,
            composite: LayerCompositeMode::Normal,
            visible: layer.visible,
        }
    }
}

impl From<&PaintMaterialRecord> for ArchiveMaterial {
    fn from(record: &PaintMaterialRecord) -> Self {
        Self {
            model: record.reference.model.0,
            model_version: record.reference.model_version,
            material: record.reference.material.0,
            payload: match &record.payload {
                PaintMaterialPayload::PremultipliedLinearRgba(rgba) => {
                    ArchiveMaterialPayload::PremultipliedLinearRgba(*rgba)
                }
                PaintMaterialPayload::ModelData {
                    schema_version,
                    bytes,
                } => ArchiveMaterialPayload::ModelData {
                    schema_version: *schema_version,
                    bytes: bytes.clone(),
                },
            },
        }
    }
}

impl ArchiveMaterial {
    fn into_record(self) -> PaintMaterialRecord {
        PaintMaterialRecord {
            reference: PaintMaterialRef {
                model: PaintModelId(self.model),
                model_version: self.model_version,
                material: PaintMaterialId(self.material),
            },
            payload: match self.payload {
                ArchiveMaterialPayload::PremultipliedLinearRgba(rgba) => {
                    PaintMaterialPayload::PremultipliedLinearRgba(rgba)
                }
                ArchiveMaterialPayload::ModelData {
                    schema_version,
                    bytes,
                } => PaintMaterialPayload::ModelData {
                    schema_version,
                    bytes,
                },
            },
        }
    }
}

impl From<&EffectNode> for ArchiveEffectNode {
    fn from(node: &EffectNode) -> Self {
        Self {
            id: node.id.0,
            effect: node.effect.0,
            implementation_version: node.implementation_version,
            enabled: node.enabled,
            parameters: node.parameters.clone(),
        }
    }
}

impl ArchiveEffects {
    fn into_graph(self) -> Result<EffectGraph, DocumentIoError> {
        let nodes = self
            .nodes
            .into_iter()
            .map(|node| EffectNode {
                id: EffectNodeId(node.id),
                effect: EffectId(node.effect),
                implementation_version: node.implementation_version,
                enabled: node.enabled,
                parameters: node.parameters,
            })
            .collect();
        let dependencies = self
            .dependencies
            .into_iter()
            .map(|(node, dependencies)| {
                (
                    EffectNodeId(node),
                    dependencies.into_iter().map(EffectNodeId).collect(),
                )
            })
            .collect();
        EffectGraph::from_persistence_parts(self.revision, self.next_node, nodes, dependencies)
            .map_err(|error| DocumentIoError::InvalidDocument(error.into()))
    }
}

impl ArchiveHistoryCommand {
    fn from_command(command: &HistoryCommand, strokes: &[StrokeMetadata]) -> Self {
        match command {
            HistoryCommand::AddStroke(index) => Self::AddStroke(strokes[*index].id.0),
            HistoryCommand::Clear(indices) => {
                Self::Clear(indices.iter().map(|index| strokes[*index].id.0).collect())
            }
            HistoryCommand::Resize { before, after } => Self::Resize {
                before: before.to_array(),
                after: after.to_array(),
            },
            HistoryCommand::AddLayer { layer, index } => Self::AddLayer {
                layer: ArchiveLayer::from(layer),
                index: *index,
            },
            HistoryCommand::LayerVisibility {
                layer,
                before,
                after,
            } => Self::LayerVisibility {
                layer: layer.0,
                before: *before,
                after: *after,
            },
            HistoryCommand::LayerOpacity {
                layer,
                before,
                after,
            } => Self::LayerOpacity {
                layer: layer.0,
                before: *before,
                after: *after,
            },
            HistoryCommand::LayerMove {
                layer,
                before,
                after,
            } => Self::LayerMove {
                layer: layer.0,
                before: *before,
                after: *after,
            },
            HistoryCommand::LayerRename {
                layer,
                before,
                after,
            } => Self::LayerRename {
                layer: layer.0,
                before: before.clone(),
                after: after.clone(),
            },
        }
    }
}

fn restore_history(
    commands: Vec<ArchiveHistoryCommand>,
    lookup: &HashMap<StrokeId, usize>,
) -> Result<Vec<HistoryCommand>, DocumentIoError> {
    commands
        .into_iter()
        .map(|command| {
            Ok(match command {
                ArchiveHistoryCommand::AddStroke(id) => {
                    HistoryCommand::AddStroke(*lookup.get(&StrokeId(id)).ok_or_else(|| {
                        DocumentIoError::InvalidDocument(
                            "history references an unknown stroke".into(),
                        )
                    })?)
                }
                ArchiveHistoryCommand::Clear(ids) => HistoryCommand::Clear(
                    ids.into_iter()
                        .map(|id| {
                            lookup.get(&StrokeId(id)).copied().ok_or_else(|| {
                                DocumentIoError::InvalidDocument(
                                    "clear history references an unknown stroke".into(),
                                )
                            })
                        })
                        .collect::<Result<_, _>>()?,
                ),
                ArchiveHistoryCommand::Resize { before, after } => HistoryCommand::Resize {
                    before: UVec2::from_array(before),
                    after: UVec2::from_array(after),
                },
                ArchiveHistoryCommand::AddLayer { layer, index } => HistoryCommand::AddLayer {
                    layer: CanvasLayer::from(layer),
                    index,
                },
                ArchiveHistoryCommand::LayerVisibility {
                    layer,
                    before,
                    after,
                } => HistoryCommand::LayerVisibility {
                    layer: LayerId(layer),
                    before,
                    after,
                },
                ArchiveHistoryCommand::LayerOpacity {
                    layer,
                    before,
                    after,
                } => HistoryCommand::LayerOpacity {
                    layer: LayerId(layer),
                    before,
                    after,
                },
                ArchiveHistoryCommand::LayerMove {
                    layer,
                    before,
                    after,
                } => HistoryCommand::LayerMove {
                    layer: LayerId(layer),
                    before,
                    after,
                },
                ArchiveHistoryCommand::LayerRename {
                    layer,
                    before,
                    after,
                } => HistoryCommand::LayerRename {
                    layer: LayerId(layer),
                    before,
                    after,
                },
            })
        })
        .collect()
}

fn deposition_mode(value: u8) -> DepositionMode {
    if value == DepositionMode::Erase as u8 {
        DepositionMode::Erase
    } else {
        DepositionMode::Normal
    }
}

fn migrate_archive(archive: &mut ArchiveDocument) -> Result<Option<u32>, DocumentIoError> {
    if archive.schema_version > DOCUMENT_SCHEMA_VERSION {
        return Err(DocumentIoError::UnsupportedSchema {
            found: archive.schema_version,
            supported: DOCUMENT_SCHEMA_VERSION,
        });
    }
    if archive.schema_version == 0 {
        return Err(DocumentIoError::InvalidDocument(
            "missing document schema version".into(),
        ));
    }
    let source = archive.schema_version;
    if archive.schema_version == 1 {
        if archive.layers.is_empty() {
            archive.layers.push(ArchiveLayer {
                id: 0,
                name: "Paint".into(),
                opacity: 1.0,
                composite: 0,
                visible: true,
            });
        }
        archive.active_layer = archive.layers.first().map_or(0, |layer| layer.id);
        archive.next_layer = archive
            .layers
            .iter()
            .map(|layer| layer.id.saturating_add(1))
            .max()
            .unwrap_or(1);
        archive.schema_version = 2;
    }
    if archive.schema_version == 2 {
        archive.schema_version = 3;
    }
    Ok((source != archive.schema_version).then_some(source))
}

fn validate_archive(archive: &ArchiveDocument) -> Result<(), DocumentIoError> {
    let invalid = |message: &str| DocumentIoError::InvalidDocument(message.into());
    if archive.format != FORMAT_ID {
        return Err(invalid("unrecognized Hamerons document payload"));
    }
    if archive.schema_version != DOCUMENT_SCHEMA_VERSION {
        return Err(invalid(
            "document migration did not reach the current schema",
        ));
    }
    if archive.canvas_size.contains(&0) || archive.tile_size == 0 {
        return Err(invalid("canvas and tile dimensions must be nonzero"));
    }
    let mut layer_ids = HashSet::new();
    if archive.layers.is_empty()
        || archive.layers.iter().any(|layer| {
            !layer_ids.insert(layer.id)
                || !layer.opacity.is_finite()
                || !(0.0..=1.0).contains(&layer.opacity)
                || layer.composite != 0
        })
    {
        return Err(invalid(
            "layers contain invalid ids, opacity, or blend modes",
        ));
    }
    if !layer_ids.contains(&archive.active_layer) {
        return Err(invalid("active layer does not exist"));
    }
    if archive.points.iter().any(|point| {
        !point.position.into_iter().all(f32::is_finite)
            || !point.half_width.is_finite()
            || point.half_width < 0.0
            || !point.aspect_ratio.is_finite()
            || point.aspect_ratio < 1.0
            || !point.flow.is_finite()
            || !point.orientation.into_iter().all(f32::is_finite)
            || !point.twist_radians.is_finite()
    }) {
        return Err(invalid("stroke points contain non-finite geometry"));
    }
    if archive.segments.iter().any(|segment| {
        segment.start as usize >= archive.points.len()
            || segment.end as usize >= archive.points.len()
            || segment.deposition > 1
            || !layer_ids.contains(&segment.layer)
    }) {
        return Err(invalid(
            "segments contain invalid point, layer, or deposition ids",
        ));
    }
    let materials: HashSet<_> = archive
        .materials
        .iter()
        .map(|material| (material.model, material.model_version, material.material))
        .collect();
    if materials.len() != archive.materials.len() {
        return Err(invalid("material identities are duplicated"));
    }
    let mut stroke_ids = HashSet::new();
    for stroke in &archive.strokes {
        let point_end = stroke.first_point as usize + stroke.point_count as usize;
        let segment_end = stroke.first_segment as usize + stroke.segment_count as usize;
        if !stroke_ids.insert(stroke.id)
            || point_end > archive.points.len()
            || segment_end > archive.segments.len()
            || !layer_ids.contains(&stroke.layer)
            || !stroke.bounds.into_iter().all(f32::is_finite)
            || stroke.bounds[0] > stroke.bounds[2]
            || stroke.bounds[1] > stroke.bounds[3]
            || !stroke.brush.is_valid()
            || !materials.contains(&(
                stroke.brush.model,
                stroke.brush.model_version,
                stroke.brush.material,
            ))
        {
            return Err(invalid(
                "stroke metadata or material references are invalid",
            ));
        }
        if archive.segments[stroke.first_segment as usize..segment_end]
            .iter()
            .any(|segment| {
                segment.layer != stroke.layer
                    || segment.model != stroke.brush.model
                    || segment.material != stroke.brush.material
                    || segment.deposition != stroke.brush.deposition
            })
        {
            return Err(invalid("stroke segment metadata disagrees with its owner"));
        }
    }
    Ok(())
}

const fn default_aspect_ratio() -> f32 {
    1.0
}

impl ArchiveBrush {
    fn is_valid(&self) -> bool {
        self.deposition <= 1
            && self.size_space <= 1
            && self.diameter.is_finite()
            && self.minimum_diameter_ratio.is_finite()
            && self.pressure_gamma.is_finite()
            && self.flow.is_finite()
    }
}

fn save_archive_atomic(
    archive: ArchiveDocument,
    path: &Path,
) -> Result<DocumentSaveReport, DocumentIoError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| DocumentIoError::InvalidDocument("save path has no file name".into()))?
        .to_string_lossy();
    let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp_path = parent.join(format!(
        ".{file_name}.{}.{}.tmp",
        std::process::id(),
        sequence
    ));
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .read(true)
        .open(&temp_path)?;
    let revision = archive.revision;
    let result = write_kra(file, &archive).and_then(|mut file| {
        file.flush()?;
        file.sync_all()?;
        Ok(())
    });
    if let Err(error) = result {
        let _ = fs::remove_file(&temp_path);
        return Err(error);
    }
    if let Err(error) = fs::rename(&temp_path, path) {
        let _ = fs::remove_file(&temp_path);
        return Err(DocumentIoError::Io(error));
    }
    let bytes_written = fs::metadata(path)?.len();
    if let Ok(directory) = File::open(parent) {
        let _ = directory.sync_all();
    }
    Ok(DocumentSaveReport {
        path: path.to_path_buf(),
        revision,
        bytes_written,
    })
}

fn write_kra(file: File, archive: &ArchiveDocument) -> Result<File, DocumentIoError> {
    let payload = ron::ser::to_string_pretty(archive, PrettyConfig::new())
        .map_err(|error| DocumentIoError::Encode(error.to_string()))?;
    let manifest = maindoc_xml(archive);
    let krita_layers = rasterize_krita_layers(archive);
    let merged = encode_composite_png(archive, &krita_layers, None)?;
    let preview = encode_composite_png(archive, &krita_layers, Some(256))?;
    let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    let mut writer = ZipWriter::new(file);
    writer.start_file("mimetype", stored)?;
    writer.write_all(KRA_MIMETYPE)?;
    writer.start_file("maindoc.xml", stored)?;
    writer.write_all(manifest.as_bytes())?;
    writer.start_file("mergedimage.png", stored)?;
    writer.write_all(&merged)?;
    writer.start_file("preview.png", stored)?;
    writer.write_all(&preview)?;
    for layer in &krita_layers {
        writer.start_file(
            format!("{KRITA_IMAGE_NAME}/layers/{}", layer.filename),
            stored,
        )?;
        writer.write_all(&paint_device_bytes(&layer.tiles))?;
        writer.start_file(
            format!("{KRITA_IMAGE_NAME}/layers/{}.defaultpixel", layer.filename),
            stored,
        )?;
        writer.write_all(&[0, 0, 0, 0])?;
    }
    writer.start_file(DOCUMENT_ENTRY, stored)?;
    writer.write_all(payload.as_bytes())?;
    writer.finish().map_err(DocumentIoError::from)
}

fn read_archive(path: &Path) -> Result<ArchiveDocument, DocumentIoError> {
    let file = File::open(path)?;
    let mut zip = ZipArchive::new(file)?;
    let mut mimetype = Vec::new();
    zip.by_name("mimetype")?.read_to_end(&mut mimetype)?;
    if mimetype != KRA_MIMETYPE {
        return Err(DocumentIoError::Archive(
            "archive does not declare the Krita MIME type".into(),
        ));
    }
    let mut entry = zip.by_name(DOCUMENT_ENTRY)?;
    if entry.size() > MAX_DOCUMENT_BYTES {
        return Err(DocumentIoError::Archive(
            "authoritative document payload exceeds the safety limit".into(),
        ));
    }
    let mut payload = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut payload)?;
    ron::de::from_bytes(&payload).map_err(|error| DocumentIoError::Encode(error.to_string()))
}

struct KritaLayerRaster {
    id: u32,
    filename: String,
    tiles: BTreeMap<(u32, u32), Vec<[f32; 4]>>,
}

fn rgba8_checkpoint(pixel: &mut [f32; 4]) {
    for channel in pixel {
        *channel = (channel.clamp(0.0, 1.0) * 255.0).round() / 255.0;
    }
}

fn rasterize_krita_layers(archive: &ArchiveDocument) -> Vec<KritaLayerRaster> {
    let materials: HashMap<_, _> = archive
        .materials
        .iter()
        .filter_map(|material| match material.payload {
            ArchiveMaterialPayload::PremultipliedLinearRgba(rgba) => Some((
                (material.model, material.model_version, material.material),
                rgba,
            )),
            ArchiveMaterialPayload::ModelData { .. } => None,
        })
        .collect();

    archive
        .layers
        .iter()
        .enumerate()
        .map(|(layer_index, layer)| {
            let mut tile_strokes: BTreeMap<(u32, u32), Vec<usize>> = BTreeMap::new();
            for (stroke_index, stroke) in archive.strokes.iter().enumerate() {
                if stroke.visible && stroke.layer == layer.id {
                    for tile in stroke_krita_tiles(archive, stroke) {
                        tile_strokes.entry(tile).or_default().push(stroke_index);
                    }
                }
            }
            let tiles = tile_strokes
                .into_iter()
                .filter_map(|(tile, strokes)| {
                    let pixels = rasterize_krita_tile(archive, &materials, tile, &strokes);
                    pixels
                        .iter()
                        .any(|pixel| pixel[3] > 0.0)
                        .then_some((tile, pixels))
                })
                .collect();
            KritaLayerRaster {
                id: layer.id,
                filename: format!("layer{}", layer_index + 2),
                tiles,
            }
        })
        .collect()
}

fn stroke_krita_tiles(archive: &ArchiveDocument, stroke: &ArchiveStroke) -> BTreeSet<(u32, u32)> {
    let width = archive.canvas_size[0] as f32;
    let height = archive.canvas_size[1] as f32;
    let min_x = stroke.bounds[0] + width * 0.5;
    let max_x = stroke.bounds[2] + width * 0.5;
    let min_y = height * 0.5 - stroke.bounds[3];
    let max_y = height * 0.5 - stroke.bounds[1];
    if max_x < 0.0 || max_y < 0.0 || min_x >= width || min_y >= height {
        return BTreeSet::new();
    }
    let min_x = min_x.max(0.0).floor() as u32 / KRITA_TILE_SIZE;
    let min_y = min_y.max(0.0).floor() as u32 / KRITA_TILE_SIZE;
    let max_x = max_x.min(width - 1.0).ceil() as u32 / KRITA_TILE_SIZE;
    let max_y = max_y.min(height - 1.0).ceil() as u32 / KRITA_TILE_SIZE;
    let mut output = BTreeSet::new();
    for tile_y in min_y..=max_y {
        for tile_x in min_x..=max_x {
            output.insert((tile_x, tile_y));
        }
    }
    output
}

fn rasterize_krita_tile(
    archive: &ArchiveDocument,
    materials: &HashMap<(u32, u32, u32), [f32; 4]>,
    tile: (u32, u32),
    stroke_indices: &[usize],
) -> Vec<[f32; 4]> {
    let mut pixels = vec![[0.0; 4]; (KRITA_TILE_SIZE * KRITA_TILE_SIZE) as usize];
    let origin_x = tile.0 * KRITA_TILE_SIZE;
    let origin_y = tile.1 * KRITA_TILE_SIZE;
    let canvas_width = archive.canvas_size[0];
    let canvas_height = archive.canvas_size[1];

    for &stroke_index in stroke_indices {
        let stroke = &archive.strokes[stroke_index];
        let Some(&material) = materials.get(&(
            stroke.brush.model,
            stroke.brush.model_version,
            stroke.brush.material,
        )) else {
            continue;
        };
        let segment_end = stroke.first_segment + stroke.segment_count;
        for local_y in 0..KRITA_TILE_SIZE {
            let canvas_y = origin_y + local_y;
            if canvas_y >= canvas_height {
                continue;
            }
            for local_x in 0..KRITA_TILE_SIZE {
                let canvas_x = origin_x + local_x;
                if canvas_x >= canvas_width {
                    continue;
                }
                let document_point = Vec2::new(
                    canvas_x as f32 + 0.5 - canvas_width as f32 * 0.5,
                    canvas_height as f32 * 0.5 - canvas_y as f32 - 0.5,
                );
                let mut amount = 0.0f32;
                for segment_index in stroke.first_segment..segment_end {
                    amount = amount.max(krita_segment_body_amount(
                        document_point,
                        segment_index,
                        &archive.segments,
                        &archive.points,
                    ));
                    amount = amount.max(krita_segment_cap_amount(
                        document_point,
                        segment_index,
                        &archive.segments,
                        &archive.points,
                    ));
                }
                amount = amount.clamp(0.0, 1.0);
                let index = (local_y * KRITA_TILE_SIZE + local_x) as usize;
                let destination = &mut pixels[index];
                if stroke.brush.deposition == DepositionMode::Erase as u8 {
                    let remaining = 1.0 - (amount * material[3]).clamp(0.0, 1.0);
                    for channel in &mut *destination {
                        *channel *= remaining;
                    }
                } else {
                    let source = material.map(|channel| channel * amount);
                    for channel in 0..4 {
                        destination[channel] =
                            source[channel] + destination[channel] * (1.0 - source[3]);
                    }
                }
                // Match the GPU tile replay and the live-to-completed handoff:
                // one RGBA8 checkpoint after each complete stroke deposition.
                rgba8_checkpoint(destination);
            }
        }
    }
    pixels
}

fn krita_point_major_axis(point: ArchivePoint) -> Vec2 {
    let major_axis = Vec2::from_array(point.orientation).normalize_or(Vec2::Y);
    let (sine, cosine) = point.twist_radians.sin_cos();
    Vec2::new(
        cosine * major_axis.x - sine * major_axis.y,
        sine * major_axis.x + cosine * major_axis.y,
    )
}

fn krita_ellipse_amount(pixel: Vec2, point: ArchivePoint) -> f32 {
    let minor_radius = point.half_width.max(0.0001);
    let major_radius = minor_radius * point.aspect_ratio.max(1.0);
    let major_axis = krita_point_major_axis(point);
    let minor_axis = Vec2::new(-major_axis.y, major_axis.x);
    let offset = pixel - Vec2::from_array(point.position);
    let normalized_distance = Vec2::new(
        offset.dot(major_axis) / major_radius,
        offset.dot(minor_axis) / minor_radius,
    )
    .length();
    let signed_distance = (normalized_distance - 1.0) * minor_radius;
    let coverage = (0.5 - signed_distance).clamp(0.0, 1.0);
    coverage * point.flow.clamp(0.0, 1.0)
}

fn krita_interpolate_ellipse_axis(from: Vec2, to: Vec2, amount: f32) -> Vec2 {
    let from = from.normalize_or(Vec2::Y);
    let mut to = to.normalize_or(from);
    if from.dot(to) < 0.0 {
        to = -to;
    }
    from.lerp(to, amount).normalize_or(from)
}

fn krita_interpolate_stroke_point(
    from: ArchivePoint,
    to: ArchivePoint,
    amount: f32,
) -> ArchivePoint {
    let major_axis = krita_interpolate_ellipse_axis(
        krita_point_major_axis(from),
        krita_point_major_axis(to),
        amount,
    );
    ArchivePoint {
        position: Vec2::from_array(from.position)
            .lerp(Vec2::from_array(to.position), amount)
            .to_array(),
        half_width: from.half_width.lerp(to.half_width, amount),
        aspect_ratio: from.aspect_ratio.lerp(to.aspect_ratio, amount),
        flow: from.flow.lerp(to.flow, amount),
        orientation: major_axis.to_array(),
        twist_radians: 0.0,
    }
}

fn krita_ellipse_metric_projection(
    pixel: Vec2,
    origin: Vec2,
    delta: Vec2,
    point: ArchivePoint,
) -> f32 {
    let minor_radius = point.half_width.max(0.0001);
    let major_radius = minor_radius * point.aspect_ratio.max(1.0);
    let major_axis = krita_point_major_axis(point);
    let minor_axis = Vec2::new(-major_axis.y, major_axis.x);
    let offset = pixel - origin;
    let metric_offset = Vec2::new(
        offset.dot(major_axis) / major_radius,
        offset.dot(minor_axis) / minor_radius,
    );
    let metric_delta = Vec2::new(
        delta.dot(major_axis) / major_radius,
        delta.dot(minor_axis) / minor_radius,
    );
    let denominator = metric_delta.length_squared();
    if denominator <= 0.000_001 {
        return 0.0;
    }
    metric_offset.dot(metric_delta) / denominator
}

fn krita_segment_body_amount(
    pixel: Vec2,
    segment_index: u32,
    segments: &[ArchiveSegment],
    points: &[ArchivePoint],
) -> f32 {
    let segment = segments[segment_index as usize];
    let point_a = points[segment.start as usize];
    let point_b = points[segment.end as usize];
    let position_a = Vec2::from_array(point_a.position);
    let position_b = Vec2::from_array(point_b.position);
    let delta = position_b - position_a;
    let length_squared = delta.length_squared();
    if length_squared <= 0.000_001 {
        return 0.0;
    }
    let mut amount = ((pixel - position_a).dot(delta) / length_squared).clamp(0.0, 1.0);
    for _ in 0..2 {
        let point = krita_interpolate_stroke_point(point_a, point_b, amount);
        amount = krita_ellipse_metric_projection(pixel, position_a, delta, point).clamp(0.0, 1.0);
    }
    if amount <= 0.0 || amount >= 1.0 {
        return 0.0;
    }
    krita_ellipse_amount(
        pixel,
        krita_interpolate_stroke_point(point_a, point_b, amount),
    )
}

fn krita_segment_cap_amount(
    pixel: Vec2,
    segment_index: u32,
    segments: &[ArchiveSegment],
    points: &[ArchivePoint],
) -> f32 {
    let segment = segments[segment_index as usize];
    krita_ellipse_amount(pixel, points[segment.end as usize])
}

fn paint_device_bytes(tiles: &BTreeMap<(u32, u32), Vec<[f32; 4]>>) -> Vec<u8> {
    let mut output = format!(
        "VERSION 2\nTILEWIDTH {KRITA_TILE_SIZE}\nTILEHEIGHT {KRITA_TILE_SIZE}\nPIXELSIZE 4\nDATA {}\n",
        tiles.len()
    )
    .into_bytes();
    for (&(tile_x, tile_y), pixels) in tiles {
        let mut planar = Vec::with_capacity((KRITA_TILE_SIZE * KRITA_TILE_SIZE * 4) as usize);
        for channel in [2, 1, 0, 3] {
            for pixel in pixels {
                planar.push(if channel == 3 {
                    unit_to_byte(pixel[3])
                } else if pixel[3] > 0.0 {
                    linear_to_srgb_byte(pixel[channel] / pixel[3])
                } else {
                    0
                });
            }
        }
        let payload_size = planar.len() + 1;
        output.extend_from_slice(
            format!(
                "{},{},LZF,{payload_size}\n",
                tile_x * KRITA_TILE_SIZE,
                tile_y * KRITA_TILE_SIZE
            )
            .as_bytes(),
        );
        output.push(0); // Krita's RAW_DATA_FLAG inside a version-2 LZF tile record.
        output.extend_from_slice(&planar);
    }
    output
}

fn encode_composite_png(
    archive: &ArchiveDocument,
    layers: &[KritaLayerRaster],
    maximum_dimension: Option<u32>,
) -> Result<Vec<u8>, DocumentIoError> {
    let source_width = archive.canvas_size[0];
    let source_height = archive.canvas_size[1];
    let scale = maximum_dimension.map_or(1.0, |maximum| {
        (maximum as f32 / source_width.max(source_height) as f32).min(1.0)
    });
    let width = ((source_width as f32 * scale).round() as u32).max(1);
    let height = ((source_height as f32 * scale).round() as u32).max(1);
    let mut output = Vec::new();
    let mut encoder = png::Encoder::new(&mut output, width, height);
    encoder.set_color(ColorType::Rgba);
    encoder.set_depth(BitDepth::Eight);
    encoder.set_compression(Compression::Fast);
    let mut writer = encoder
        .write_header()
        .map_err(|error| DocumentIoError::Encode(error.to_string()))?;
    {
        let mut stream = writer
            .stream_writer()
            .map_err(|error| DocumentIoError::Encode(error.to_string()))?;
        let mut row = vec![0; width as usize * 4];
        for y in 0..height {
            for x in 0..width {
                let source_x =
                    ((x as f32 + 0.5) * source_width as f32 / width as f32).floor() as u32;
                let source_y =
                    ((y as f32 + 0.5) * source_height as f32 / height as f32).floor() as u32;
                let pixel = composite_pixel(
                    archive,
                    layers,
                    source_x.min(source_width - 1),
                    source_y.min(source_height - 1),
                );
                row[x as usize * 4..x as usize * 4 + 4].copy_from_slice(&pixel);
            }
            stream.write_all(&row)?;
        }
        stream
            .finish()
            .map_err(|error| DocumentIoError::Encode(error.to_string()))?;
    }
    drop(writer);
    Ok(output)
}

fn composite_pixel(
    archive: &ArchiveDocument,
    layers: &[KritaLayerRaster],
    x: u32,
    y: u32,
) -> [u8; 4] {
    let mut destination = PAPER_SRGB.map(|channel| channel as f32 / 255.0);
    for layer in &archive.layers {
        if !layer.visible || layer.opacity <= 0.0 {
            continue;
        }
        let Some(raster) = layers.iter().find(|raster| raster.id == layer.id) else {
            continue;
        };
        let tile = (x / KRITA_TILE_SIZE, y / KRITA_TILE_SIZE);
        let Some(pixels) = raster.tiles.get(&tile) else {
            continue;
        };
        let local_x = x % KRITA_TILE_SIZE;
        let local_y = y % KRITA_TILE_SIZE;
        let source = pixels[(local_y * KRITA_TILE_SIZE + local_x) as usize];
        let alpha = (source[3] * layer.opacity).clamp(0.0, 1.0);
        if alpha <= 0.0 {
            continue;
        }
        for channel in 0..3 {
            let straight_linear = source[channel] / source[3].max(f32::EPSILON);
            let straight_srgb = linear_to_srgb(straight_linear);
            destination[channel] = straight_srgb * alpha + destination[channel] * (1.0 - alpha);
        }
    }
    [
        unit_to_byte(destination[0]),
        unit_to_byte(destination[1]),
        unit_to_byte(destination[2]),
        255,
    ]
}

fn linear_to_srgb(linear: f32) -> f32 {
    let linear = linear.clamp(0.0, 1.0);
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

fn linear_to_srgb_byte(linear: f32) -> u8 {
    unit_to_byte(linear_to_srgb(linear))
}

fn unit_to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn maindoc_xml(archive: &ArchiveDocument) -> String {
    let mut xml = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <!DOCTYPE DOC PUBLIC '-//KDE//DTD krita 2.0//EN' 'http://www.calligra.org/DTD/krita-2.0.dtd'>\n\
         <DOC xmlns=\"http://www.calligra.org/DTD/krita\" syntaxVersion=\"2\" editor=\"Krita\">\n\
          <IMAGE width=\"{}\" height=\"{}\" name=\"{KRITA_IMAGE_NAME}\" mime=\"application/x-kra\" colorspacename=\"RGBA\" profile=\"sRGB built-in\" x-res=\"100\" y-res=\"100\" description=\"Hamerons stroke document\">\n\
           <layers>\n",
        archive.canvas_size[0], archive.canvas_size[1]
    );
    for (index, layer) in archive.layers.iter().enumerate().rev() {
        xml.push_str(&format!(
            "    <layer name=\"{}\" filename=\"layer{}\" nodetype=\"paintlayer\" x=\"0\" y=\"0\" colorspacename=\"RGBA\" compositeop=\"normal\" opacity=\"{}\" visible=\"{}\" locked=\"0\" collapsed=\"0\" channelflags=\"\" channellockflags=\"\" uuid=\"{}\"{}/>\n",
            escape_xml(&layer.name),
            index + 2,
            (layer.opacity * 255.0).round() as u8,
            if layer.visible { "1" } else { "0" },
            layer_uuid(layer.id),
            if layer.id == archive.active_layer {
                " selected=\"true\""
            } else {
                ""
            },
        ));
    }
    xml.push_str(
        "   </layers>\n  <ProjectionBackgroundColor ColorData=\"AAAAAA==\"/>\n  </IMAGE>\n </DOC>\n",
    );
    xml
}

fn layer_uuid(id: u32) -> String {
    format!("{{00000000-0000-4000-8000-{:012x}}}", id as u64 + 1)
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{effects::pass_through_descriptor, RgbaPaintModel, StrokeRendererSettings};

    fn point(x: f32, y: f32) -> StrokePoint {
        StrokePoint {
            position: Vec2::new(x, y),
            half_width: 3.5,
            aspect_ratio: 1.75,
            flow: 0.75,
            orientation: Vec2::Y,
            twist_radians: 0.25,
        }
    }

    fn registries() -> (PaintModelRegistry, EffectRegistry) {
        let mut models = PaintModelRegistry::default();
        models.register(RgbaPaintModel::descriptor());
        let mut effects = EffectRegistry::default();
        effects.register(pass_through_descriptor());
        (models, effects)
    }

    fn raster_point(x: f32, y: f32) -> ArchivePoint {
        ArchivePoint::from(StrokePoint {
            position: Vec2::new(x, y),
            half_width: 4.0,
            aspect_ratio: 1.0,
            flow: 0.25,
            orientation: Vec2::Y,
            twist_radians: 0.0,
        })
    }

    #[test]
    fn raster_geometry_sweeps_ellipses_without_join_spurs() {
        let segment = ArchiveSegment {
            start: 0,
            end: 1,
            ..Default::default()
        };
        let next = ArchiveSegment {
            start: 1,
            end: 2,
            ..Default::default()
        };
        let straight = [
            raster_point(0.0, 0.0),
            raster_point(10.0, 0.0),
            raster_point(20.0, 0.0),
        ];
        let straight_segments = [segment, next];

        assert_eq!(
            krita_segment_body_amount(Vec2::new(-1.0, 0.0), 0, &straight_segments, &straight),
            0.0
        );
        assert_eq!(
            krita_segment_body_amount(Vec2::new(5.0, 0.0), 0, &straight_segments, &straight),
            0.25
        );
        assert_eq!(
            krita_segment_cap_amount(Vec2::new(10.0, 2.0), 0, &straight_segments, &straight),
            0.25
        );

        let corner = [
            raster_point(0.0, 0.0),
            raster_point(10.0, 0.0),
            raster_point(10.0, 10.0),
        ];
        assert!(
            krita_segment_cap_amount(Vec2::new(11.0, -1.0), 0, &straight_segments, &corner) > 0.24
        );
        let outside_corner =
            krita_segment_body_amount(Vec2::new(11.0, -1.0), 0, &straight_segments, &corner).max(
                krita_segment_body_amount(Vec2::new(11.0, -1.0), 1, &straight_segments, &corner),
            );
        assert_eq!(outside_corner, 0.0);
        assert_eq!(
            krita_segment_cap_amount(Vec2::new(30.0, -20.0), 0, &straight_segments, &corner),
            0.0
        );

        let tilted_point = |x| ArchivePoint {
            position: [x, 0.0],
            half_width: 2.0,
            aspect_ratio: 5.0,
            flow: 0.25,
            orientation: Vec2::from_angle(core::f32::consts::FRAC_PI_4).to_array(),
            twist_radians: 0.0,
        };
        let tilted = [tilted_point(0.0), tilted_point(20.0)];
        let tilted_segments = [ArchiveSegment {
            start: 0,
            end: 1,
            ..Default::default()
        }];

        // A rotated ellipse's upper support point is shifted along the path.
        // Treating the support radius as a purely vertical radius used to
        // cover this pixel and create a triangular point at every join.
        let old_spur = Vec2::new(0.0, 6.5);
        assert_eq!(
            krita_segment_body_amount(old_spur, 0, &tilted_segments, &tilted).max(
                krita_segment_cap_amount(old_spur, 0, &tilted_segments, &tilted)
            ),
            0.0
        );
        assert!(
            krita_segment_body_amount(Vec2::new(10.0, 6.5), 0, &tilted_segments, &tilted,) > 0.15
        );
    }

    #[test]
    fn kra_round_trip_preserves_ids_layers_materials_effects_and_history() {
        let mut document = StrokeDocument::default();
        let upper = document.add_layer("Upper & Ink");
        document.set_layer_opacity(upper, 0.625);
        let (stroke, _) =
            document.begin_stroke(point(1.0, 2.0), StrokeRendererSettings::default().pen);
        document.append_point(stroke, point(20.0, 9.0)).unwrap();
        document.end_stroke(stroke).unwrap();
        document.undo();

        let path = std::env::temp_dir().join(format!(
            "hamerons-roundtrip-{}-{}.kra",
            std::process::id(),
            TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        document.save_kra(&path).unwrap();
        let (models, effects) = registries();
        let mut loaded = StrokeDocument::load_kra(&path, &models, &effects).unwrap();
        fs::remove_file(&path).unwrap();

        assert!(loaded.compatibility_issues.is_empty());
        assert_eq!(loaded.document.active_layer(), upper);
        assert_eq!(loaded.document.layers(), document.layers());
        assert_eq!(loaded.document.points(), document.points());
        assert_eq!(loaded.document.segments(), document.segments());
        assert_eq!(loaded.document.strokes()[0].id, stroke);
        assert_eq!(loaded.document.redo_len(), 1);
        assert!(loaded.document.redo());
        assert!(loaded.document.stroke(stroke).unwrap().visible);
    }

    #[test]
    fn unsupported_versions_are_reported_without_losing_opaque_payloads() {
        let mut document = StrokeDocument::default();
        let reference = PaintMaterialRef {
            model: PaintModelId(77),
            model_version: 9,
            material: PaintMaterialId(4),
        };
        let record = PaintMaterialRecord {
            reference,
            payload: PaintMaterialPayload::ModelData {
                schema_version: 3,
                bytes: vec![9, 8, 7, 6],
            },
        };
        document.add_paint_material(record.clone()).unwrap();
        let (models, effects) = registries();
        let issues = document.compatibility_issues(&models, &effects);
        assert!(matches!(
            issues.as_slice(),
            [DocumentCompatibilityIssue::PaintModelUnavailable {
                model: PaintModelId(77),
                required_version: 9
            }]
        ));
        assert_eq!(document.paint_materials().get(reference), Some(&record));
    }

    #[test]
    fn schema_one_payloads_migrate_to_explicit_layer_state() {
        let document = StrokeDocument::default();
        let mut archive = ArchiveDocument::from_document(&document).unwrap();
        archive.schema_version = 1;
        archive.active_layer = 99;
        archive.next_layer = 0;
        assert_eq!(migrate_archive(&mut archive).unwrap(), Some(1));
        assert_eq!(archive.schema_version, DOCUMENT_SCHEMA_VERSION);
        assert_eq!(archive.active_layer, 0);
        assert_eq!(archive.next_layer, 1);
        archive.into_document().unwrap();
    }

    #[test]
    fn failed_save_leaves_the_previous_valid_kra_intact() {
        let mut document = StrokeDocument::default();
        let path = std::env::temp_dir().join(format!(
            "hamerons-atomic-{}-{}.kra",
            std::process::id(),
            TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        document.save_kra(&path).unwrap();
        let (stroke, _) =
            document.begin_stroke(point(4.0, 5.0), StrokeRendererSettings::default().pen);
        assert!(matches!(
            document.save_kra(&path),
            Err(DocumentIoError::ActiveContacts)
        ));
        document.end_stroke(stroke).unwrap();

        let (models, effects) = registries();
        let loaded = StrokeDocument::load_kra(&path, &models, &effects).unwrap();
        fs::remove_file(&path).unwrap();
        assert!(loaded.document.strokes().is_empty());
    }

    #[test]
    fn background_checkpoints_are_single_flight_and_skip_unchanged_revisions() {
        let document = StrokeDocument::default();
        let path = std::env::temp_dir().join(format!(
            "hamerons-checkpoint-{}-{}.kra",
            std::process::id(),
            TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let mut checkpoints = DocumentCheckpointManager::default();
        assert_eq!(
            checkpoints.request(&document, &path).unwrap(),
            CheckpointRequest::Started
        );
        assert_eq!(
            checkpoints.request(&document, &path).unwrap(),
            CheckpointRequest::Busy
        );
        let report = loop {
            if let Some(result) = checkpoints.poll() {
                break result.unwrap();
            }
            thread::yield_now();
        };
        assert_eq!(report.revision, document.revision());
        assert_eq!(
            checkpoints.request(&document, &path).unwrap(),
            CheckpointRequest::Unchanged
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn kra_contains_krita_native_manifest_layers_and_previews() {
        let mut document = StrokeDocument::default();
        document.resize(UVec2::new(128, 96));
        let (stroke, _) =
            document.begin_stroke(point(0.0, 0.0), StrokeRendererSettings::default().pen);
        document.end_stroke(stroke);
        let path = std::env::temp_dir().join(format!(
            "hamerons-krita-native-{}-{}.kra",
            std::process::id(),
            TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        document.save_kra(&path).unwrap();

        let mut zip = ZipArchive::new(File::open(&path).unwrap()).unwrap();
        let mut manifest = String::new();
        zip.by_name("maindoc.xml")
            .unwrap()
            .read_to_string(&mut manifest)
            .unwrap();
        assert!(manifest.contains("syntaxVersion=\"2\""));
        assert!(manifest.contains("mime=\"application/x-kra\""));
        assert!(manifest.contains("colorspacename=\"RGBA\""));
        assert!(manifest.contains("filename=\"layer2\""));

        let mut layer = Vec::new();
        zip.by_name("unnamed/layers/layer2")
            .unwrap()
            .read_to_end(&mut layer)
            .unwrap();
        assert!(layer.starts_with(b"VERSION 2\nTILEWIDTH 64\nTILEHEIGHT 64\nPIXELSIZE 4\nDATA "));
        assert!(layer.windows(5).any(|window| window == b",LZF,"));
        assert_eq!(
            zip.by_name("unnamed/layers/layer2.defaultpixel")
                .unwrap()
                .size(),
            4
        );
        for entry in ["mergedimage.png", "preview.png"] {
            let mut signature = [0; 8];
            zip.by_name(entry)
                .unwrap()
                .read_exact(&mut signature)
                .unwrap();
            assert_eq!(signature, *b"\x89PNG\r\n\x1a\n");
        }
        drop(zip);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn compute_completion_is_not_serialized_as_a_live_overlay_recovery_strategy() {
        let mut document = StrokeDocument::default();
        let (stroke, _) =
            document.begin_stroke(point(0.0, 0.0), StrokeRendererSettings::default().pen);
        document.end_stroke(stroke).unwrap();
        let archive = ArchiveDocument::from_document(&document).unwrap();
        let restored = archive.into_document().unwrap();
        let mut ranges = Vec::new();
        restored.overlay_segment_ranges(&mut ranges);
        assert!(ranges.is_empty());
        assert_eq!(
            restored.stroke(stroke).unwrap().state,
            StrokeState::PendingCache
        );
    }
}
