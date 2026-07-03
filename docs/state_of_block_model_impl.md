# State of the Block Model Implementation

Assessment date: 2026-07-01 9:18 AM AWST

Ranked by severity: correctness bugs first, then missing-but-expected features,
then performance/scale concerns, then polish/UX.

## 1. Data model correctness

- **Fixed.** Sub-block geometry with no explicit `__lower_x/y/z` /
  `__upper_x/y/z` variables no longer silently falls back to a uniform grid.
  `block_bounds()` now returns an error when the metadata's `is_irregular`
  flag is set but those bound variables are missing, instead of discarding
  the sub-celling schema and rendering wrong geometry with no indication.
  (`src/model/formats/bmf.rs`, see `block_bounds`)
- **Partially fixed.** `local_to_world` now composes bearing (Z), dip (X),
  and plunge (Y) rotations per Maptek's documented bearing/dip/plunge
  convention, instead of ignoring dip/plunge entirely. No sample `.bmf` in
  this repo has non-zero dip/plunge, so the dip/plunge portion of the
  transform is unverified against real data; loading a model with non-zero
  dip/plunge now logs a `userspace_warn!` (`BmfModel::has_verified_rotation`)
  instead of silently mis-placing it. Validate against a real tilted model
  when one becomes available. (`src/model/formats/bmf.rs`, see
  `local_to_world`/`rotation_matrix`)
- **Fixed.** Metadata root selection is now anchored to the file header's
  primary table pointer (offset `0x18`): the live root's `00 02` pages are
  identified by choosing the parsed candidate root whose maximum page offset
  is nearest before that table pointer. In repo samples the root sits exactly
  one page before the table; a company file has also been observed with
  unrelated pages between the root and table. The old whole-file heuristic
  scoring is now only a fallback for files where the pointer is
  missing/invalid or no parsed root sits before it, and that fallback path
  logs a `userspace_warn!` so it's no longer silent.
  (`src/model/formats/bmf.rs`, see `parse_bmf_metadata_root`)
- **Fixed.** The `chunk.try_into().unwrap()` calls in numeric/page-table
  decoding now go through a `read_chunk` helper that returns a `BmfError`
  instead of panicking. (These specific call sites were always fed
  `chunks_exact`-sized slices so couldn't actually panic today, but the
  helper removes the sharp edge for future refactors.)
- **Not changed — no current caller.** Storage is still a flat
  `Vec<BlockBounds>` with no spatial index. There is currently no CPU
  spatial query against block bounds anywhere in the app (no per-block
  picking, no section-extraction feature); the one full scan
  (`create_ore_triangulation`'s value filter) has to visit every block
  regardless of indexing, and neighbor lookups in `boundary_mesh_from_blocks`
  already use a hash map, not a linear scan. Building an octree/grid now
  would be speculative infrastructure for a query pattern that doesn't exist
  yet — revisit once picking or section extraction against block models is
  actually implemented. (`src/model/block_model.rs`)

## 2. File format support

- Only Vulcan BMF (`TBMS2.0`) + companion `.bdf` import is supported. No
  export path exists at all.
- **Fixed.** Numeric decode now also covers `double` (page kind `00 03`,
  256×`f64le`/page) and `short` (page kind `00 0a`,
  1024×`i16le`/page). `double` is confirmed against `value0`/`value1`/`value2`
  in `test/Vulcan/bmf_bdf/shawmodel_reg.bmf`; `short` is covered by a
  synthetic regression test because company-file metadata reported plain
  `short` variables. Categorical still covers only
  `namedbyte`/`namedshort` (no other categorical encoding has been observed
  in any sample file). Variables of a type this reader still can't decode
  are no longer invisible: `BmfModel::unsupported_variables()` reports them,
  block-model load now logs a `userspace_warn!` listing their names/types,
  and the block-model UI's Variables table marks them "(unsupported)" in
  orange instead of showing a blank type-looking row.
  (`src/model/formats/bmf.rs`, see `is_numeric_type`,
  `unsupported_variables`; `src/app/commands/block_model.rs`;
  `src/ui/elements/block_model.rs`)
- No caching: every read of a variable fully re-walks the page table and
  re-decodes the entire variable from raw bytes with no memoization.
  (`src/model/formats/bmf.rs:271-320`)
- `from_bytes` reads the entire file into memory; there is no
  incremental/streaming read path. (`src/model/formats/bmf.rs:97`)

## 3. Rendering

- **Known limitation.** Each block is triangulated independently (no
  shared-vertex/greedy meshing: a block's 8 corners are always its own
  vertices) and all six faces are emitted per rendered block. This preserves
  internal block faces for transparent grade inspection and for
  hidden-empty-value holes, but dense regular grids still carry substantial
  interior overdraw. True shared-vertex/greedy meshing (deduplicating
  vertices, merging coplanar faces into larger quads only when that cannot
  hide inspectable blocks) is not implemented.
  (`src/rendering/scene/gpu_cache.rs`, see `build_block_model_surface_chunks`)
- **Fixed.** Block model surface chunks now carry a scene-relative AABB
  (`CachedSurfaceChunk::bounds_min/max`), and `render_scene_pass` extracts
  the camera's view-frustum planes each frame (`Frustum::from_view_proj`,
  `src/rendering/graphics/frustum.rs`) to skip the GPU draw call for any
  chunk entirely outside the current view, for both opaque and translucent
  block models. This doesn't reduce what's *uploaded* to the GPU (chunks
  are still built/cached for the whole model up front) — it only skips
  `draw_indexed` calls for off-screen chunks each frame. No LOD (reduced
  detail at distance) is implemented.
  (`src/rendering/graphics/passes.rs`, `src/rendering/graphics/frustum.rs`)
- **Fixed.** An attribute/legend-colour switch alone (translucency
  unchanged) no longer walks blocks or reallocates
  GPU buffers: chunk geometry (positions, indices, which faces are visible)
  is cached CPU-side alongside a per-vertex-group block-index list
  (`CachedBlockModelSurfaceChunk`), so only `grade` needs recomputing and
  `queue.write_buffer`-ing into the existing vertex buffer.
  Translucency toggles (which change chunking/pipeline) and new/changed
  block models still do a full rebuild.
  (`src/rendering/scene/gpu_cache.rs`, see `recolor_block_model_surface_chunks`,
  `BlockModelGpuCache::sync`)
- **Fixed.** Transparent block models are still ordered relative to each
  other by whole-model bounds, but each model's own chunks are now also
  sorted back-to-front by chunk centroid before drawing, so overlapping
  translucent blocks within one model blend in the correct order.
  (`src/rendering/graphics/passes.rs`, transparent block model draw loop)
- **Fixed.** `BmfModel` now precomputes its bearing/dip/plunge rotation
  matrix once at construction (`orientation` is fixed after parsing), instead
  of redoing the trig on every `local_to_world` call — previously that was 8
  times per block on every bounds query. (`src/model/formats/bmf.rs`, see
  `compute_rotation_matrix`, `BmfModel::rotation`)

## 4. UI/UX

- **Fixed.** A gradient colour-scale legend now appears bottom-center of the
  3D viewport whenever a visible block model has an active grade-colour
  variable with a usable render range: the selected block model takes
  priority, otherwise the first visible one with an active variable. The
  gradient bar is drawn in Rust (`ramp_color` in
  `src/ui/widgets/viewport.rs`) matching the shader's `ramp_color`
  (`src/rendering/shaders/block_model.wgsl`) stop-for-stop — the two must be
  kept in sync by hand if the ramp ever changes. Annotation density scales
  with the bar's width (which itself scales with the viewport): narrow bars
  label only the start/end, medium bars add the midpoint, wide bars label
  the start, quartiles, and end. No units are shown, since `BmfVariable`
  doesn't carry a units field distinct from its free-text description.
  The legend also exposes a `Hide empty` checkbox, defaulting on, which hides
  blocks whose active colour variable is unset/default/sentinel-valued instead
  of drawing them with the model fallback colour.
  (`src/ui/widgets/viewport.rs`, see `ColorScaleLegend`; `src/ui/mod.rs`,
  canvas overlays)
- **Fixed.** The block value table and the new colour-scale legend both used
  to (or would have) re-decoded a numeric variable's raw bytes from scratch
  on every UI frame — the table for every visible variable across all
  blocks, regardless of which page was in view. Decoded values are now
  cached per `(BlockModelId, variable name)` in `EditorState` and reused
  across frames, invalidated only when the block model itself closes. This
  doesn't address the underlying lack of decode caching inside
  `BmfModel::numeric_values` itself (still noted below) — a *different*
  variable, or the *first* frame after opening the table/switching the
  legend's variable, still pays the full decode cost once.
  (`src/ui/elements/block_model.rs`, see `decoded_numeric_variable`,
  `active_color_scale`; `src/ui/state.rs`, `block_model_variable_cache`)
- No cross-section/slice view; no live cutoff-grade filter in the 3D view
  (filtering only exists as a one-shot "Create Ore Triangulation" export).
- No cell/value editing — block models are explicitly excluded from the
  generic property-editing dialog. (`src/ui/dialogs/editing.rs:59`)

## 5. Commands/error handling

- No undo/redo integration: `close_block_model`, `remove_block_model`,
  `toggle_block_model_visible` mutate state directly with no undo-stack
  hooks, even though the project has an undo system used elsewhere.
  (`src/app/commands/block_model.rs:164-199`)
- Background load errors are only logged via `userspace_warn!`, with no
  user-facing retry or detail beyond a toast. (`src/app/commands/block_model.rs:151`)
- `boundary_mesh_from_blocks`/`add_block_face` key adjacency by
  float-quantized bounds, which is fragile for non-uniform/irregular
  sub-block boundaries. (`src/app/commands/block_model.rs:298-393,410-412`)

## 6. Integration gaps

- Block model ↔ triangulation/solids integration is one-directional and
  destructive only: "Create Ore Triangulation" builds a boundary mesh from
  a value filter. There is no reverse operation — no "constrain/clip block
  model to an existing pit-shell triangulation," no live pit-shell cutoff
  filter, and no re-import of the generated triangulation as a live filter
  reference. (`src/app/commands/block_model.rs:243-275`)

## Suggested priority order

1. ~~Fix sub-block schema fallback and rotation/dip-plunge handling~~ — done;
   dip/plunge rotation is implemented but unverified against real tilted
   samples (none exist in this repo).
2. ~~Add decode caching for BMF variables~~ — done at the UI layer (table +
   legend reuse a per-model/per-variable cache instead of re-decoding every
   frame); `BmfModel::numeric_values` itself still has no internal
   memoization, so a variable is still fully re-walked the first time
   anything asks for it.
3. Shared-vertex meshing + frustum culling/LOD for rendering scale.
4. ~~Legend~~ — done; cross-section and live cutoff-filter UI remain.
5. Undo/redo wiring and bidirectional block-model/solid integration.
