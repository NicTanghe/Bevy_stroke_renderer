# Drawing engine documentation

The paint-engine design and reusable implementation belong to the standalone
`hamerons_stroke_render` crate. The Bevy fork supplies input, windowing, and
rendering primitives; applications such as `drawing_test` supply product UI and
configure the library.

- [GPU paint roadmap](gpu-paint-roadmap.md) defines the phases, their scope, and
  their completion gates.
- [GPU paint architecture](gpu-paint-architecture.md) defines the boundaries
  that let RGBA paint grow into blur, wet-media behavior, and pigment mixing
  without replacing the stroke or tile systems.
- [GPU paint validation](gpu-paint-validation.md) lists the correctness,
  performance, recovery, and future-extension tests for each phase.
- [Pen drawing latency mode](pen-latency-mode.md) records the presentation
  policy used while drawing.
- [Phase 3 document format](phase3-document-format.md) records the durable
  `.kra` contract, migration, compatibility, and atomic-save behavior.

These documents are the source of truth for implementation sequencing. A phase
is complete only when its completion gate and relevant validation cases pass.
