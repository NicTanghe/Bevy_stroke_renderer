# GPU paint implementation roadmap

Status snapshot: 2026-07-20

## Goal

Replace `drawing_test`'s CPU raster loop and per-tile image uploads with a GPU
paint engine that keeps pen ink under the pointer, supports persistent editing,
and can later add blur, wet-media effects, and pigment-based mixing without a
document-format or renderer rewrite.

The roadmap deliberately separates four kinds of work:

1. input and stroke geometry;
2. coverage generation;
3. paint-model-specific deposition;
4. effects and display resolve.

That separation is the main future-proofing requirement. Coverage describes
where a tool touched the surface. Deposition decides what that contact does to
the selected paint model. Effects transform material or display surfaces.
Display resolve converts any paint model to premultiplied linear RGBA for Bevy.

## Current baseline

`drawing_test` currently maintains a 1600 by 1000 CPU RGBA buffer. It stamps
pressure- and tilt-dependent ellipses into RAM, marks 256 by 256 tiles dirty,
and copies those tiles into Bevy `Image` assets. It also disables pipelined
rendering and requests a one-frame, non-vsynced surface queue in its default
low-latency mode.

This baseline is useful for visual reference and input testing, but its cost is
proportional to the pixels touched on the CPU and the bytes uploaded. Long or
large strokes can therefore fall behind the pen. Presentation tuning cannot
remove that paint-generation backlog.

## Phase summary

| Phase | Outcome | Required for fast painting? | Status in `drawing_test` |
| --- | --- | --- | --- |
| 0 | Measured and stable input/render baseline | Yes | Implemented in the standalone engine/test app |
| 1 | Low-latency live strokes generated and drawn on the GPU | Yes | Implemented in the standalone engine/test app |
| 2 | Persistent editable GPU tile canvas | No for a paint demo; yes for an editor | Implemented in the standalone engine/test app |
| 3 | Durable documents, layers, and production cache behavior | No | Implemented in the standalone engine/test app |
| 4 | Non-destructive blur and other effect passes | No | Architectural contracts reserved |
| 5 | Multi-plane pigment deposition and mixing | No | Architectural contracts reserved |
| 6 | Product hardening and platform/performance qualification | For release | Planned |

If the only goal is responsive painting, implement through and including
Phase 1. Phase 2 is where marks become an editable, regenerable canvas rather
than a live overlay or flattened result.

## Phase 0: stabilize and measure the baseline

### Scope

- Preserve first-class pen identity, pressure, tilt, twist, tool kind, barrel
  state, contact transitions, and window identity from the input event.
- Convert each sample into document coordinates exactly once.
- Use the current frame's camera, viewport, DPI scale, and window dimensions.
- Keep mouse input as a clearly identified fallback, not as the pen data model.
- Retain the low-latency presentation policy documented in
  `pen-latency-mode.md`.
- Add counters and timings before replacing the rasterizer.

### Measurements

Record at least:

- input events and generated samples per frame;
- coalesced or dropped samples;
- input-to-extraction age;
- points and segments appended;
- CPU paint time and uploaded bytes in the baseline;
- CPU frame p50, p95, and p99 while drawing;
- render frame time and presentation mode;
- adapter name, backend, and device limits.

Use a repeatable dense spiral and a long diagonal stroke as fixtures. Keep a
saved screenshot or pixel hash from the CPU implementation for visual
comparison, but do not require identical antialiasing.

### Completion gate

- Coordinate mapping remains correct during resize, zoom, pan, and DPI change.
- A contact press, move sequence, and release always produce one stroke.
- Pen and mouse contacts cannot accidentally join each other's strokes.
- The baseline measurements are reproducible in debug and release builds.

## Phase 1: live GPU stroke overlay

Phase 1 is the minimum implementation for fast painting.

### Data model

Store append-only, geometry-only points:

```text
StrokePoint
  position: document-space vec2
  half_width: document units
  flow: 0..1
  orientation: projected brush-major-axis vec2
  twist: radians
```

Store segments separately. A segment references two point indices plus stable
paint material/model identity and a deposition mode. A first contact dot is a
segment whose endpoints reference the same point. Geometry must not embed RGBA
channels; that would force a rewrite for pigments.

Each active contact owns a stroke id and a contiguous point/segment range.
Append new samples only. Do not rebuild or upload the complete stroke on every
move.

### GPU representation

- Use persistent storage buffers for points, segments, and material records.
- Upload only the newly appended suffix each frame.
- Grow buffers geometrically and report growth events.
- Generate segment coverage procedurally in the vertex/fragment stages or a
  compute pass. Do not create a CPU mesh for every dab.
- Draw compatible active ranges in no more than two draw calls per view: bodies
  and caps/contact dots.
- Render the overlay after the persistent canvas so it is always current.
- Cull ranges that cannot intersect the view.

The coverage shader owns pressure width, tilt ellipse orientation, caps, and
antialiasing. It must expose a paint-independent coverage value. RGBA blending
is the first deposition/display path, not part of the geometry contract.

### Stroke-internal accumulation

Overlapping triangles or adjacent dabs from the same stroke must not darken a
pixel merely because the tessellation overlaps. Use a per-stroke coverage mask
or a procedural representation that evaluates the stroke footprint once per
pixel. Flow can then be applied once to the resulting coverage.

### Input-to-render schedule

```text
PreUpdate: read pen events and append document samples
Extract:   copy only new point/segment/material data
Prepare:   grow/write persistent GPU buffers when necessary
Queue:     queue the active overlay after the canvas
Render:    draw procedural coverage
```

Avoid a one-frame staging resource in `Update` when the event can be consumed
and appended in `PreUpdate`. Camera and viewport state used for conversion must
already describe the frame being drawn.

### Completion gate

- Ink remains visually under the pointer during the repeatable stress stroke.
- Steady drawing performs no per-sample GPU allocation and no full-buffer
  upload.
- A long stroke does not increase draw-call count with point count.
- Pressure, tilt, contact dots, joins, caps, and eraser preview are correct.
- Transparent paint and erase preview agree with the eventual persistent
  result, not only with a white paper background.
- The live path stays within the agreed p95 frame and sample-age budgets.

## Phase 2: persistent editable GPU tile canvas

Phase 2 keeps Phase 1 as a low-latency overlay and adds an authoritative vector
document plus a derived, bounded GPU tile cache.

### Authoritative document

The CPU document owns:

- append-only point and segment arrays;
- stable stroke ids and metadata;
- visibility and completion state;
- immutable, versioned paint material recipes;
- layer identity;
- undo and redo commands;
- canvas extent;
- the effect graph configuration;
- a spatial index from document tiles to stroke-local segment indices.

The GPU canvas is disposable. Undo, redo, clear, resize, cache eviction, device
loss, and a format change regenerate pixels by replaying the document. Never
make cached pixels the only copy of an edit.

### Tile cache

- Use document-space tile keys such as `(layer, x, y)`.
- Start with 256 by 256 content tiles, but keep the size configurable.
- Allocate persistent planes from the active paint model descriptor.
- Keep one aggregate persistent GPU memory budget across all planes.
- Track `dirty`, `scheduled`, `ready`, and `evicted` states with desired and
  displayed revision numbers.
- Prefer visible dirty tiles. Limit work per frame, with a smaller budget while
  drawing and a larger idle budget.
- Evict only clean tiles. Prefer least-recently-visible tiles.
- Regenerate an evicted tile deterministically from the spatial index.
- Batch ready tiles into one instanced composite pass per compatible layer/model.

For each dirty tile, clear every model plane to its declared clear value and
replay intersecting visible strokes in strict document order unless the paint
model explicitly guarantees commutative and associative deposition.

### Active-to-cached handoff

An active stroke is drawn by the Phase 1 overlay. On release it becomes
`pending-cache`; affected tiles are dirtied while the overlay remains visible.
Only after the render work for the matching tile revision is complete may the
overlay range be retired. Queue submission order alone is not a substitute for
matching revisions.

When editing old history or rebuilding after device loss, do not send the
entire document through the live overlay. That turns one million cached points
back into one million live points and defeats Phase 2. Show valid stale tiles
where semantically safe, prioritize visible regeneration, or display a bounded
recovery state.

### Editing commands

- `undo` and `redo` change visibility and dirty only affected tiles;
- `clear` is undoable and does not destroy vector history;
- `resize` changes logical bounds without rewriting coordinates;
- cache-layout changes invalidate derived resources, not the document;
- device loss drops all GPU resources and regenerates from the document.

### Completion gate

- Completed strokes survive after the live overlay is retired.
- Undo, redo, clear, resize, eviction, and device loss regenerate the same
  visible result.
- Cached frame cost is independent of total historical point count.
- A one-million-point document remains interactive after visible tiles are
  ready.
- Memory remains within the configured persistent plus transient budgets.
- Tile seams are absent for ordinary RGBA paint.
- Long-stroke release/indexing work is incremental or budgeted; releasing the
  pen must not create a large main-thread spike.

## Phase 3: durable documents, layers, and cache hardening

### Scope

- Define a versioned on-disk document schema for geometry, materials, models,
  effects, layers, and history policy.
- Save atomically and load without changing stable ids or material recipes.
- Implement multiple normal layers before advanced blend modes.
- Add layer visibility, opacity, ordering, and bounded composite batching.
- Add background saving or checkpointing for very large documents.
- Define migration behavior for unavailable or newer paint/effect versions.
- Harden cache behavior under zoom, pan, resize, memory pressure, and repeated
  device recreation.
- Add thumbnail or flattened-preview data only as a convenience; it cannot
  replace authoritative vector/material records.
- use .kra for on disk saving and loading
### Completion gate

- Save/load/regenerate produces the same document and visible output.
- Layer edits invalidate only their downstream composite dependencies.
- Unsupported model/effect versions fail visibly and non-destructively.
- Recovery work is bounded and never expands the live overlay to full history.

## Phase 4: non-destructive effects, including blur

Effects are graph nodes, not special cases in the stroke shader.

### Required contracts

Each effect declares:

- stable effect id and implementation version;
- versioned parameter bytes;
- execution domain: native material planes or resolved linear RGBA;
- local read radius or global influence;
- persistent outputs, if any;
- transient scratch-plane formats;
- dependencies on other effect nodes.

A local effect expands tile invalidation and requests neighbor halos. Influence
must be accumulated along each dependency path: two chained radius-10 passes
can require a radius-20 source region. A global node invalidates all dependent
output whenever any relevant source changes.

The scheduler must retain halo information through render extraction, lease
real GPU scratch resources for the lifetime of the pass, execute nodes in
topological order, and retire scratch only after submission is safe. Counting
scratch bytes without executing the corresponding pass is not an implementation.

Blur should initially operate on resolved premultiplied linear RGBA. A later
material-domain diffusion or wetness effect can use native pigment planes
without changing the graph, invalidation, or scheduler contracts.

### Completion gate

- Radius-zero pass-through runs through the production graph path.
- Local blur has no tile seams and invalidates every required neighbor.
- Chained local effects compose influence correctly.
- Effect parameter changes rerun only downstream nodes/tiles.
- Scratch high-water use remains inside the shared transient budget.
- Disabling an effect restores the deterministic pre-effect result.

## Phase 5: pigment-based paint and mixing

Pigment paint is a new paint model, not a replacement stroke engine.

### Paint-model contract

Each paint model declares:

- stable model id and implementation version;
- one or more persistent surface planes and exact formats;
- clear values for every plane;
- scratch planes required for deposition, mixing, drying, or resolve;
- material recipe schema and immutable versioned payloads;
- whether deposition order is strict or may be regrouped;
- deposition and erase entry points;
- display resolve to premultiplied linear RGBA.

The first pigment model may store coefficients or concentrations appropriate to
the selected physical approximation, for example absorption/scattering terms,
coverage or thickness, binder/wetness, and substrate state. The exact model can
be chosen later because strokes reference an opaque versioned material recipe,
not hard-coded RGBA.

Mixing is state-dependent deposition. It normally requires strict document
order and reads the existing material planes before writing new state. Drying,
diffusion, granulation, and color resolve can be model-domain effect nodes.
Display compositing still consumes resolved linear RGBA, so cameras and Bevy's
2D pipeline remain independent of pigment internals.

### Completion gate

- A mock two-plane model works end to end: allocation, clear, deposition,
  eviction, regeneration, effects, and display resolve.
- Pigment recipe identities survive save/load unchanged.
- Mixed output is deterministic for the same document and implementation
  version.
- Erasing removes model material rather than painting the paper color.
- RGBA and pigment documents can share the stroke, tile, history, effect, and
  compositor infrastructure.

## Phase 6: product hardening

- Qualify supported native and Web targets.
- Measure representative integrated and discrete GPUs.
- Add adaptive work budgets using CPU and GPU timing feedback.
- Handle very small storage-buffer limits and maximum texture dimensions.
- Add out-of-memory degradation and user-visible recovery.
- Validate color management and export behavior.
- Run long-session tests for id wrap, cache churn, undo depth, and resource leaks.
- Document public APIs and migration policy.

Release readiness requires the complete validation matrix, not only a successful
desktop debug build.

## Repository boundary

The reusable implementation belongs in the standalone
`hamerons_stroke_render` crate. `drawing_test` is an integration consumer, and
the local `Hamerons_bevy` fork should expose generally useful
pen/window/rendering primitives only. This keeps experimental paint models and
document formats out of the engine fork while allowing multiple applications
to share and test the paint engine.
