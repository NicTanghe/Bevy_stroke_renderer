# GPU paint architecture

This document defines the boundaries that every implementation phase must
preserve. The purpose is not to lock in a particular pigment equation or blur
kernel. It is to make those later choices additive.

## Design invariants

1. The CPU document is authoritative; GPU surfaces are derived caches.
2. Stroke geometry describes contact and coverage, never a concrete color
   channel layout.
3. Paint models own deposition, erase semantics, native planes, material
   recipes, and display resolve.
4. Effects are versioned graph nodes with explicit spatial dependencies.
5. All persistent and transient GPU allocations participate in known budgets.
6. Active ink has a bounded low-latency path independent of historical size.
7. Cache eviction and device loss affect performance, not document correctness.
8. Stored ids and versioned payloads are stable across save/load.

## Ownership

```text
hamerons_stroke_render
├── input adapter
├── authoritative paint document
├── stroke geometry and history
├── paint/effect registries
├── tile scheduler and cache policy
├── render extraction and GPU passes
└── diagnostics

drawing_test (or another consumer)
├── application window and camera
├── tool controls and product UI
└── save/load policy

Hamerons_bevy
├── first-class pen events
├── windows, cameras, schedules, and ECS
├── render device/queue and render graph
└── generic 2D compositing primitives
```

The painting subsystem is this standalone crate. Keep it outside the Bevy fork
so applications can reuse it without making paint models or document formats
part of the engine.

## End-to-end data flow

```text
pen/mouse event
    │
    ▼
input adapter ── current camera/viewport/DPI ──► document-space sample
    │
    ▼
append-only StrokeDocument
    ├── active range ──────────────────────────► live coverage overlay
    ├── history / visibility / layers
    ├── spatial index ─────────────────────────► dirty tile replay list
    ├── immutable paint material recipes
    └── effect graph
                                                 │
                                                 ▼
                                      paint-model deposition
                                                 │ native planes
                                                 ▼
                                         effect graph passes
                                                 │
                                                 ▼
                                  resolve to premultiplied linear RGBA
                                                 │
                                                 ▼
                                      instanced tile compositor
```

The live overlay and persistent replay must share coverage math and material
semantics. Otherwise a stroke changes shape or color when it moves into the
cache.

## Module layout

A practical target layout is:

```text
src/
  paint/
    mod.rs
    input.rs
    document.rs
    geometry.rs
    materials.rs
    effects.rs
    tiles.rs
    diagnostics.rs
    render/
      mod.rs
      live_overlay.rs
      tile_raster.rs
      effects.rs
      composite.rs
    shaders/
      coverage.wgsl
      rgba_deposition.wgsl
      tile_composite.wgsl
```

If this moves into a crate, keep the same internal boundaries and expose a
small plugin plus document/editing API.

## Authoritative document

The document contains append-only geometry arrays and mutable metadata:

```text
StrokeDocument
  points[]
  segments[]
  strokes[]
  layers[]
  material_library
  effect_graph
  history
  tile_spatial_index
  revision counters
  canvas extent
```

Append-only geometry keeps extraction proportional to new input. Undo does not
remove geometry; it changes stroke visibility and document revision. A compact
or archival rewrite can be a later explicit operation.

Each completed stroke records its point and segment ranges, bounds, layer,
paint model/version/material, deposition mode, visibility, and revision. The
spatial index stores tile-local references while preserving document order.

History commands should name stable strokes/layers and carry the smallest state
needed to invert the edit. `clear` is a visibility command over existing
strokes. Resize changes logical bounds without transforming coordinates.

## Coordinate and brush-size policy

Positions stored in the document use document coordinates. Camera pan/zoom and
window resize must not rewrite them.

Choose brush-size semantics explicitly:

- document-space size scales on screen with camera zoom and is naturally stable
  in a tiled document;
- screen-space size is converted at sampling time and then becomes fixed
  document geometry.

If a product requires old screen-space marks to remain the same physical width
after later zoom changes, that is a different semantic and requires
view-dependent rasterization. Do not imply that behavior merely by naming a
stored brush `Screen`.

The input adapter should use the current render target and camera transform,
support nonuniform scale, and avoid assuming the primary window when an event
names another window.

## Geometry and coverage

`StrokePoint` contains position, half-width, flow, orientation, and twist.
`StrokeSegment` references point indices and a paint material/model plus
deposition mode.

Coverage output is a scalar mask. It may be generated with procedural segment
quads and cap primitives in the live path, and with compute in tile replay.
Both implementations use the same definitions for:

- pressure curve and minimum diameter;
- tilted ellipse orientation and aspect;
- contact dots;
- segment interpolation;
- joins and caps;
- antialiasing;
- per-stroke maximum/union behavior.

Do not source-over every overlapping triangle from the same stroke. First form
the intended per-stroke coverage, then invoke deposition once for that stroke
and pixel.

## Paint model interface

RGBA is one implementation of a general paint model.

```text
PaintModelDescriptor
  id
  implementation_version
  persistent_planes[]
  scratch_planes[]
  deposition_ordering
  deposition_pipeline
  erase_pipeline
  display_resolve_pipeline
```

Every plane descriptor specifies a stable semantic name, format, bytes per
pixel, and deterministic clear value. The common allocator must create all
declared planes, not merely include them in accounting. Deposition and resolve
bind plane sets through descriptor/pipeline-specific layouts.

The built-in RGBA model can use a premultiplied linear RGBA surface. Normal
deposition uses source-over semantics. Erase reduces stored material/alpha; it
must not deposit the current paper color.

A model declares strict deposition ordering by default. It may opt into
regrouping only when it proves commutative and associative behavior.

## Material library

A stroke references:

```text
(paint_model_id, model_version, material_id)
```

The document owns an immutable record for that identity. RGBA records contain
premultiplied linear values. Pigment records contain opaque, versioned model
data such as pigment coefficients and recipe proportions.

Once an identity is used, changing its payload is forbidden. Editing a color or
pigment recipe creates a new material id. This guarantees deterministic replay
and prevents a load from silently substituting a process-global palette.

## Persistent tile cache

The cache key is independent of plane format:

```text
TileKey { layer, x, y }
```

Each resident tile maps to one slot in every persistent plane required by its
paint model. A tile record tracks:

- desired document/effect revision;
- displayed revision;
- dirty/scheduled/ready/evicted state;
- display permission for a stale revision;
- visibility and last-visible frame;
- plane slot allocation;
- halo requirements.

Use one aggregate memory budget:

```text
persistent bytes = sum(resident slots × bytes per tile for every plane)
transient bytes  = live overlay + replay jobs + effect scratch + resolve scratch
```

Geometric buffer growth may reserve more than the logical content. Budget
reserved capacity, not only occupied slots, or cap allocation to a predictable
page/slot scheme.

Dirty scheduling priorities are normally:

1. visible tile needed for correctness;
2. visible tile whose stale content is allowed;
3. offscreen dirty tile;
4. prefetch candidate.

Use separate per-frame budgets while drawing and idle. GPU timings should later
adapt those budgets, but fixed conservative limits are acceptable initially.

## Deterministic tile replay

For a dirty tile:

1. collect visible intersecting strokes from the spatial index;
2. retain strict document order;
3. clear every persistent plane to its declared value;
4. generate per-stroke coverage for tile pixels;
5. apply paint-model deposition once per stroke/pixel;
6. run material-domain effects in graph order;
7. resolve the model to linear premultiplied RGBA;
8. run display-domain effects;
9. publish the tile revision after render completion is safely ordered.

Do not scan all document segments for each tile. Avoid a compute design that
loops every segment for every pixel without binning; tile-local indexing is the
minimum, and larger workloads may require sub-tile bins or segment worklists.

Indexing a very long stroke entirely on contact release can stall input. Update
bounds and spatial bins incrementally or schedule completion work under a CPU
budget.

## Live-to-cache state machine

```text
Active
  │ contact release
  ▼
PendingCache ── dirty affected tiles, keep live overlay
  │
  ├── newer edit arrives ──► update desired revisions / reschedule
  │
  └── every required tile reaches matching revision
                         ▼
                       Cached ── retire overlay range
```

An old ready tile can remain visible under a newly completed stroke when the
overlay supplies the semantic difference. Undo, clear, effects, and some layer
changes cannot always use stale content; those invalidations must hide it or
use another bounded preview strategy.

Device recovery and large history edits must not mark every historical stroke
as a live overlay. That makes overlay cost proportional to document history.

## Effect graph

An effect node stores stable identity, effect/version identity, enabled state,
parameters, and dependency edges. The registry describes execution:

```text
EffectDescriptor
  id
  implementation_version
  domain: MaterialSurface | LinearDisplayRgba
  influence: Local(radius) | Global
  scratch_planes[]
  pipeline
```

The graph is acyclic and executes in stable topological order. Influence is a
dependency-path property. For a chain, an output's required source radius is
the sum of local radii along the relevant path, taking the maximum among
alternative paths. Any global dependency makes the downstream output global.

Local tile work carries explicit halo keys/ranges into the render world. A
radius that crosses a tile boundary must read the neighbor's matching upstream
revision. The scheduler either schedules those dependencies or retains an
older downstream output until they are ready.

Scratch allocation represents real GPU textures/buffers and remains leased
through encoding/submission safety. Effects do not mutate vector history.

## Blur placement

The first blur should be a display-domain effect:

```text
paint planes → model resolve → linear RGBA blur → layer composite
```

Use a separable kernel or another bounded-radius implementation. The effect
declares its radius and two scratch passes if required. Sampling must include
tile halos, and premultiplied color must remain premultiplied through filtering.

Later diffusion or wet paint behavior belongs in `MaterialSurface` and may
update wetness/pigment planes before resolve. It uses the same graph and halo
scheduler.

## Pigment model placement

Pigment mixing replaces the RGBA deposition/resolve implementation while
reusing input, geometry, coverage, document history, tile allocation policy,
effects, and compositing.

A possible model owns planes for pigment state, thickness/coverage, wetness,
and substrate interaction. Its material payload contains versioned pigment
coefficients or references. On deposition it reads existing state, mixes in the
new recipe according to coverage/flow, and writes native planes. Its display
resolve calculates linear RGBA/reflectance for Bevy.

Do not put pigment coefficients in `StrokePoint`, special-case pigment tile
keys, or make the compositor sample pigment planes directly. Those choices
would create the architectural rewrite this design is intended to avoid.

## Layer and compositor boundary

Each model resolves a tile to premultiplied linear RGBA. The common compositor
therefore needs only tile position/size, resolved surface slot, layer order,
opacity, and blend mode. This permits RGBA and pigment layers in one document.

Start with one normal layer, then multiple normal layers. Add specialized blend
modes only after ordering and invalidation are correct.

## Device loss and cache-layout changes

GPU resource generations are distinct from document revisions. On device loss:

1. discard buffer/texture handles and in-flight cache state;
2. recreate pipelines and descriptor-driven plane pools;
3. upload authoritative geometry/material data;
4. prioritize visible tile regeneration;
5. retain document/history/material/effect identities unchanged.

A tile-size, format, or paint-model-version change follows the same derived
cache rebuild path. No change should require rewriting stroke geometry.

## Diagnostics

Expose a snapshot that includes:

- received/appended/coalesced input samples;
- newest sample age at extraction;
- point/segment/material buffer lengths and capacities;
- live overlay ranges and draw calls;
- dirty/scheduled/ready/evicted/resident tile counts;
- rasterized and regenerated tiles;
- persistent reserved bytes per plane;
- transient/scratch current and high-water bytes;
- invalidation causes and effect-expanded tile counts;
- CPU indexing/scheduling time;
- GPU coverage, deposition, effect, resolve, and composite timings;
- device reset and allocation failure counts.

Diagnostics must describe actual submitted work. Scheduling a pass or leasing
accounting bytes is not equivalent to completing GPU work.

## Serialization rules

Persist document data, not cache slots or device handles. At minimum store:

- schema version and canvas coordinate convention;
- points, segments, strokes, visibility, and layer records;
- paint model ids/versions and immutable material payloads;
- effect graph nodes, versions, parameters, and dependencies;
- canvas extent and document revision metadata needed for migration.

On load, validate ids, ranges, graph acyclicity, payload sizes, and registered
implementation compatibility before allocating GPU resources. Unknown models
or effects should keep their opaque records so the file can be reopened without
data loss, while rendering a clear unsupported-content indication.

## Known implementation traps

- Treating multi-plane descriptors as memory accounting while allocating only
  one RGBA buffer.
- Dropping effect halo metadata before render execution.
- Using the largest single effect radius instead of composing dependency paths.
- Treating global effects as radius zero for later edits.
- Reporting tile completion when work is merely encoded and can be superseded.
- Drawing every historical stroke in the live overlay after clear undo, resize,
  or device recovery.
- Doing all long-stroke spatial indexing synchronously on release.
- Making eraser preview paint the paper color.
- Ignoring per-plane clear values.
- Letting geometric GPU capacity exceed the configured aggregate budget.
- Testing only allocator metadata for a mock paint model instead of rendering it
  through deposition, eviction, regeneration, and resolve.
