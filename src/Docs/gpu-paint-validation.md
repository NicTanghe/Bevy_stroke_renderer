# GPU paint validation plan

This plan turns the roadmap's completion gates into repeatable checks. Record
the GPU, backend, resolution, presentation mode, build profile, and commit for
every performance result.

## Test fixtures

Maintain deterministic fixtures for:

- a contact dot at several pressures;
- a straight line with changing pressure;
- a tight spiral with heavy self-overlap;
- a long diagonal crossing many tiles;
- a tilted-ellipse stroke whose orientation rotates;
- transparent paint over transparent and colored surfaces;
- erase over opaque, transparent, and pigment material;
- strokes exactly on tile boundaries;
- a one-million-point historical document;
- repeated clear/undo/redo and resize sequences.

Where exact pixels are not portable across GPUs, compare bounded error in a
linear color space and separately validate structural properties such as no
seams, correct alpha, stable bounds, and deterministic replay on the same
backend.

## Phase 0 checks

### Input lifecycle

- Enter, press, move, release, and leave produce one stable contact identity.
- Pressure-only contact starts are supported when a device omits an explicit
  contact press.
- Pen re-entry into another window in the same frame preserves the replacement
  contact.
- Enabling pen handling at runtime creates state on the first in-range event,
  even when the original enter event is no longer available.
- Pen, eraser end, barrel erase, and mouse fallback remain distinct.

### Coordinate mapping

- Resize the window in the same frame as a pen move.
- Change DPI or move between monitors while hovering/drawing where supported.
- Pan, zoom, rotate, and nonuniformly scale the camera.
- Use a non-primary window/render target.
- Verify both document-space and sample-time screen-size brush policies.

### Baseline telemetry

- Capture CPU paint time, tile uploads, uploaded bytes, event counts, and sample
  age for the standard fixtures.
- Confirm the telemetry path adds negligible release-build overhead when
  disabled.

## Phase 1 checks

### Geometry correctness

- Initial contact renders a dot without requiring movement.
- Zero-length and extremely short segments remain finite.
- Pressure and width interpolation are continuous.
- Tilt orientation and twist remain stable near zero tilt.
- Caps and joins have no pinholes or spikes.
- Self-overlap does not darken merely because segment geometry overlaps.

### Material correctness

- Normal RGBA deposition is premultiplied and linear.
- Transparent paint matches a CPU reference within tolerance.
- Erase reduces stored/displayed material rather than painting the background.
- Active paint and erase previews match the Phase 2 cached result over nonwhite
  content.

### Incremental behavior

- After warm-up, one new sample writes only the new point/segment suffix.
- Persistent buffers grow geometrically and do not allocate per sample.
- Live draw calls stay bounded as point count grows.
- Simultaneous contacts batch where compatible and never share stroke state.

### Latency and performance

- Run the spiral for at least 30 seconds in release mode.
- Record CPU frame p50/p95/p99 and latest-sample age.
- Confirm no monotonically growing distance between pointer and visible tip.
- Repeat with low-latency and vsync modes; label tearing as a presentation
  tradeoff rather than a raster bug.

## Phase 2 checks

### Document/history

- Stroke ids and point/segment ranges remain stable after undo/redo.
- Clear is undoable and retains geometry.
- Redo is discarded after a new edit.
- Resize changes logical extent without rewriting coordinates.
- Invalid range, missing material, and unsupported version loads fail safely.

### Spatial indexing

- Every segment appears in every tile its inflated bounds intersect.
- Tile replay lists preserve document order.
- Long strokes are indexed incrementally or under a measurable budget.
- Effect influence expands invalidation independently of base segment bins.

### Cache lifecycle

- Exercise every transition among dirty, scheduled, ready, and evicted.
- Revisions prevent stale completion from retiring a newer overlay.
- Visible dirty tiles outrank offscreen work.
- Drawing and idle budgets are enforced separately.
- Only clean tiles are evicted.
- Re-entering an evicted region regenerates identical output.
- Reserved capacity plus scratch remains within the configured budget.

### Handoff and editing

- A released stroke remains visible continuously until matching tiles are ready.
- No frame shows both cached and overlay versions with doubled opacity.
- Undo/redo/clear hide semantically invalid stale tiles.
- Undoing a large clear does not put all historical strokes in the live overlay.
- Device recovery cost is based on visible tile work, not full live geometry.

### Tile rendering

- Tile clears honor every declared plane clear value.
- Coverage/deposition replay is deterministic in strict document order.
- All edge/corner boundary fixtures have no visible seams.
- The compositor draws ready tiles in bounded instanced batches.
- Canvas extent clips or excludes tiles outside the logical canvas.

### Scale test

1. Load or generate one million points.
2. Wait until visible tiles are ready.
3. Record steady idle and drawing frame times.
4. Pan through cached and evicted regions.
5. Undo and redo recent strokes.
6. Force the configured cache budget low enough to churn.

Pass criteria: cached-frame and new-live-stroke costs do not scale with the
total historical point count. Regeneration may scale with visible dirty content
but remains inside its per-frame work budget.

### Device recovery

- Trigger renderer/device recreation using the platform test mechanism.
- Verify document, ids, materials, history, and effects are unchanged.
- Confirm geometry buffers and visible tiles regenerate.
- Repeat during an active contact and during pending-cache handoff.

## Phase 3 checks

### Save/load

- Round-trip a document with hidden strokes, redo history policy, layers,
  multiple materials, and effect parameters.
- Stable model/material/effect ids and opaque payload bytes remain identical.
- Regenerated pixels match the pre-save result on the same backend.
- Simulate interrupted save and confirm the last valid document survives.
- Load an older schema through each supported migration.
- Load unknown model/effect versions without destroying opaque data.

### Layers

- Reorder, hide, change opacity, and edit normal layers.
- Confirm only downstream composite dependencies invalidate.
- Mix RGBA and mock-model layers through the common resolved compositor.

## Phase 4 checks

### Effect graph

- Reject missing dependency ids, self-dependencies, and cycles.
- Produce a stable topological order.
- Enabling, disabling, parameter edits, and topology edits increment revisions.
- Only downstream nodes/tiles rerun.

### Influence and halos

- Radius-zero pass-through uses the real render path.
- Radius-1 and radius-larger-than-tile effects request exact halos.
- Two chained local radii accumulate along the dependency path.
- Parallel branches use the maximum accumulated path.
- A global dependency invalidates all downstream output after every relevant
  edit.
- Neighbor tiles must be at the required upstream revision before publication.

### Blur

- Blur a single bright pixel on a transparent surface.
- Blur across each tile edge and corner; compare with an untiled reference.
- Preserve premultiplied alpha without dark or bright fringes.
- Change radius while tiles are in flight and reject stale completion.
- Constrain scratch budget and verify work defers without corruption.

## Phase 5 checks

Create a mock two-plane paint model before implementing pigment math. It must
use non-RGBA clear values, strict state-dependent deposition, scratch space, and
a display resolve.

Run it end to end through:

- material creation and document persistence;
- live preview or a documented model-correct fallback;
- persistent plane allocation and clear;
- deposition in document order;
- local material effect with halos;
- display resolve and common tile composite;
- eviction and byte-identical deterministic regeneration;
- device recovery;
- save/load with stable opaque recipe payload;
- low-memory deferral.

Allocator-only tests do not satisfy this gate.

For the pigment implementation, additionally validate:

- known single-pigment swatches against the model reference;
- order-dependent wet mixing fixtures;
- dry-over-wet and wet-over-dry behavior if supported;
- erase against pigment thickness/concentration rather than paper-color paint;
- deterministic output for identical model/version/parameters;
- documented tolerances across GPU vendors.

## Phase 6 platform matrix

For every supported target:

- compile the application and paint crate from a clean checkout;
- run unit and render tests where the target permits;
- validate storage-buffer/texture limits and fallback behavior;
- test window resize, suspend/resume, surface recreation, and device loss;
- record presentation capabilities and latency policy;
- run multi-window input and drag/drop smoke tests when those engine paths are
  part of the shipped application.

External forks or path dependencies must be available from a clean checkout.
Do not treat a developer's sibling directory as a reproducible dependency.

## Suggested commands

Once the paint engine is owned by this repository, keep a narrow fast loop and
a broader release gate. Adapt package names to the final crate layout.

```bash
cargo fmt --all -- --check
cargo test
cargo clippy --all-targets -- -D warnings
cargo run --release
```

GPU fixtures that need a real adapter should be a separate explicitly invoked
test/example so headless CI failures remain understandable. Always run the
release stress fixture on physical hardware before marking a performance phase
complete.
