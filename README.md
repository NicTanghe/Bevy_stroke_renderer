# hamerons_stroke_render

Standalone low-latency GPU stroke-rendering library for the local Hamerons
Bevy fork. It owns stroke history, GPU buffers, persistent canvas tiles, paint
deposition, effects, normal layers, durable `.kra` documents, background
checkpoints, and diagnostics; applications such as `drawing_test`
provide the window, camera, controls, and product UI.

This checkout uses the following local layout:

```text
dev/
├── Hamerons_bevy/
└── drawings_main/
    ├── hamerons_stroke_render/
    └── stroke_drawing_test/
```

The included test app consumes the local crate with:

```toml
hamerons_stroke_render = { path = "../hamerons_stroke_render" }
```

See [the drawing-engine documentation](src/Docs/README.md) for the roadmap,
architecture, validation requirements, and pen-latency policy.

Phase 3 persistence writes the Krita MIME entry first, a syntax-version-2
`maindoc.xml`, sparse native Krita paint layers, merged/preview PNGs, and the
complete authoritative vector/material/effect/history payload at
`hamerons/document.ron`. Saves use a same-directory temporary file plus atomic
rename; load validation completes before an application replaces its current
document.
# Bevy_stroke_renderer
