# Phase 3 document format

Hamerons documents use the `.kra` extension and the ZIP container conventions
used by Krita. `mimetype` is the first, uncompressed entry and contains
`application/x-krita`. `maindoc.xml` uses Krita syntax version 2 and exposes
canvas, color-space, and normal-layer metadata. Every layer also has a native
sparse Krita paint-device entry under `unnamed/layers/`; `mergedimage.png` and
`preview.png` contain flattened display copies. The authoritative Hamerons
record is `hamerons/document.ron`.

The authoritative record contains:

- schema and format identity;
- canvas and tile dimensions;
- append-only points and segments;
- stable stroke, layer, model, material, and effect identities;
- immutable versioned material payloads, including opaque model bytes;
- normal-layer order, visibility, opacity, names, and the active layer;
- effect nodes, opaque parameters, and dependency edges;
- undo and redo commands expressed with stable identities.

Schema version 3 is current. Version 1 migrates to explicit active-layer and
next-layer state; version 2 migrates circular stroke points to the explicit
major/minor aspect contract. Newer schemas fail before the application replaces
its open document. Missing or mismatched model/effect implementations produce
visible compatibility issues while their opaque records remain available for a
later resave; cached rendering skips implementations it cannot execute.

The native paint layers are regenerated from authoritative vector geometry at
save time. They make the result visible and editable when opened by Krita;
Hamerons itself reloads the lossless vector payload rather than those derived
raster copies.

Saving creates a same-directory, uniquely named temporary archive, flushes and
synchronizes it, then atomically renames it over the destination. Failure before
the rename leaves the previous valid document intact. The checkpoint manager
allows one background serialization at a time and rejects unbounded queues or
snapshots taken during an active contact.

GPU tiles, presentation feedback, live-overlay membership, spatial bins, and
other derived state are never persisted. Loading rebuilds the spatial index and
regenerates visible tiles within the configured per-frame budgets. It does not
send completed history through the live overlay.
