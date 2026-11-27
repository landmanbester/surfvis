---
type: reference
title: Chi-squared browser
description: How surfvis serve links surfchi2 output to on-demand baseline waterfalls, the waterfall's correction / Stokes / flag-version controls, and the xarray-ms facts it depends on.
tags: [web, fastapi, htmx, xarray-ms, zarr]
timestamp: 2026-11-27
last_verified_commit: 2cca15c
---

# Chi-squared browser

Source: `src/surfvis/web/` (store, msdata, render, app), `src/surfvis/cli/serve.py`,
and the zarr writer at the end of `src/surfvis/core/chi2.py`.

**Not a Stimela cab, and never will be.** A long-running GUI is not a batch
task. `serve` carries no `@stimela_cab` decorator, and `parse_module` skips
undecorated functions, so `generate-cabs` emits nothing for it.

## Data flow

```
surfvis chi2 --dataout chi2.zarr   →  chi2.zarr        (numbers, not pictures)
                                          ↓ store.py
surfvis serve --data chi2.zarr     →  heatmap grid + histogram
                                          ↓ click  → msdata.py
                                      canvas waterfall (read from the MS)
```

The MS's visibilities are read only when a waterfall is requested. Browsing the
grid reads the zarr plus two small MS subtables, `FIELD::NAME` for the field
menu and `ANTENNA::POSITION` for baseline lengths (both cached; if the MS has
moved they are left out rather than failing the page).

`Chi2Store.matrix` reads one product's `(antenna, antenna)` matrix, and groups
are opened with `cache=False`, so xarray never keeps a whole variable.

## The zarr contract

One group per `(field, spw, scan)`, holding that scan's totals:

```
chi2.zarr/field{F}/spw{S}/scan{N}
  chi2, counts        (product, antenna1, antenna2), summed over the scan
  attrs               ms, rcol, wcol, fcol, mcol, gtable, corr_labels, antenna_names,
                      autocorrelations (whether the scan has any; absent before c26cd15),
                      created (unix time chi2 started, before it read FLAG; absent before
                      step 4 of #16, when the zarr directory's mtime stands in),
                      recomputed_cells, recomputed_at (once serve has recomputed any)
  computed            (antenna1, antenna2) unix time each cell was computed; added by
                      serve's first recompute, absent until then (created stands in)
```

The browser looks at one field, spw and scan at a time. `--ntimes` and
`--nfreqs` chunk the work in `chi2` (and its per-chunk PNGs), not the data
product: stores used to keep every `(time bin, freq bin)` chunk, with bin
bounds to outline the clicked chunk on the waterfall. That made the heatmap
a fragment of a scan, and `chi2` held every scan's dense cube in memory
until the end. Stores in the old layout still open: `store._scan_totals`
sums their bins one time bin at a time.

Two things are easy to get wrong here:

- **`product` holds labels** -- `XX`, `YY` for the default `--use-corrs`, then
  any `--stokes` (`I`, `V`). They are resolved from `POLARIZATION::CORR_TYPE`,
  not offsets into `--use-corrs`, and the waterfall forms the same label from
  the MS. Stores written before products existed have a `corr` axis instead;
  `Chi2Store.products` refuses them with a message to rerun `chi2`.

`surfchisq_products_plot` returns `(chi2, counts)`, each
`(nproducts, nant, nant)`, and does not say which chunk it was.
`core/chi2.py` therefore keys futures by `(ds, t, f)` in the parent --
`futures[fut] = (i, t, f)` -- and adds each into its scan's total.

## xarray-ms facts this depends on

Verified against the `0.4.0a*` line pinned in `pyproject.toml` (see
`tests/fixtures/ms.py`):

- Variables come out as `(time, baseline_id, frequency, polarization)`, which
  *is* the waterfall shape. No row-to-grid pivot is needed, unlike dask-ms.
- **The default partition schema is not what you want.** It is
  `OBSERVATION_ID/PROCESSOR_ID/DATA_DESC_ID/OBS_MODE_ID` — no `FIELD_ID`, no
  `SCAN_NUMBER`. `msdata.PARTITION_SCHEMA` is
  `("FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER")`, matching `pfb-imaging`'s
  default and surfchi2's `group_cols` exactly. One partition is then one
  (field, spw, scan), so a partition's time axis *is* the scan and the chunk
  can be addressed by integer ids via `partition_key` — no name lookup, no
  boolean filtering over `scan_name`.
- **xarray-ms is pinned to a commit of its `write-support` branch**
  (`0.4.0-alpha.16`, unreleased); see CLAUDE.md for why and when to unpin.
- **Column names are MSv4 inside, MSv2 on screen.** xarray-ms's standard data groups are `base`
  (`DATA` → `VISIBILITY`), `corrected` (`CORRECTED_DATA` →
  `VISIBILITY_CORRECTED`) and `model` (`MODEL_DATA` → `VISIBILITY_MODEL`),
  all sharing `WEIGHT` (from `WEIGHT_SPECTRUM`) and `FLAG`.
  `DataGroup.resolve()` translates those MSv2 names and passes everything else
  through, so both spellings work on a URL; `msdata.msv2_name()` translates
  back for the menus (`available_columns`) and the plot title. It must never map
  `CORRECTED_DATA` to `VISIBILITY`: it once did, and choosing
  `CORRECTED_DATA` plotted `DATA`.
- **The data groups are constants; never open the datatree to learn them.**
  `msdata.data_group()` used to read `data_groups` from `open_datatree`,
  which opens every partition: on a real MS the first waterfall sat there for
  over ten minutes. A test pins that a waterfall never calls `open_datatree`.
  Per-scan reads use `open_dataset(partition_key=...)`.
- Columns that are *not* part of the MSv2 standard are surfaced automatically as
  secondary variables under their own names — which is exactly why `RESIDUAL` is
  plottable -- only once their cells are defined.
- **Reversed baselines are converted on read** (xarray-ms #201). Baselines are
  presented upper-triangular; a row stored as `ANTENNA1 > ANTENNA2` is
  conjugated with its cross-hands swapped (FLAG and WEIGHT: cross-hands
  swapped). So the waterfall follows xarray-ms's labels (`msdata.view_order`)
  for gains, cross-hand labels and the title. xarray-ms cannot orient a
  non-standard column, and drops it from any partition with a reversed row
  unless told: `_open_partition` passes `baseline_orientation` declaring every
  non-standard 2-D column (`_orientations`, from the table descriptor)
  `hermitian`, which is right for visibilities, weights and flags per
  channel and correlation.
- **`time` is decoded to unix seconds** (`attrs["format"] == "unix"`). Gain
  tables and flag versions are keyed on the MS's MJD-second `TIME`, so
  `msdata.ms_times()` adds `MJD_UNIX_OFFSET` back before matching.
- The partition's `polarization` coordinate already holds labels (`XX` ...),
  which is what `msdata.labels()` returns.
- Baselines are addressed by antenna *name*, via the `baseline_antenna1_name` /
  `baseline_antenna2_name` coordinates, hence the names stored in the zarr.
- **xarray-ms lays a regular (time, baseline) grid and fills gaps with NaN.**
  Those slots hold no data, so `waterfall()` folds `~isfinite` into the flag
  mask rather than letting them punch holes in the colour scale.

- **xarray-ms warns on every open** (imputed metadata, column shapes, UVW
  frame, irregular grids). `msdata` calls `utils.msv4.ignore_xarray_ms_warnings()`
  at module scope, the same filter set as pfb-imaging.

## Waterfall controls: product, model, gains, flag version

The waterfall panel carries six controls: product, column, model, gain table,
flag version, quantity. They recompute **the waterfall only**. The heatmap is
what `chi2` measured, so the controls default to how it was formed (the zarr's
`rcol`/`mcol`/`gtable` attrs and the selected product), and the panel shows a
note once they no longer describe the same quantity. Recomputing the heatmap
on the fly was ruled out: it is every baseline across the whole scan, which
at MeerKAT scale is far too much to read per click.

The plotted quantity is `inv(G_p) column inv(G_q)^H - model` formed into the
product, via `utils/products.form` -- the same function `chi2` uses, so the two
cannot drift apart. The menus list MSv2 names, as `chi2` records them; a
column given by its MSv4 name is translated back, so it is never listed twice.

- **Gain table** is a menu, not free text: the term `chi2` used (the zarr's
  `gtable` attr), any passed as `serve --gtable` (repeatable), and the terms in
  `*.qc` directories next to the MS (`gains.find_terms`, re-scanned per
  request). Anything else in the query string is refused, with a message that
  does not reveal whether the path exists: the server has no authentication,
  and `--host 0.0.0.0` is common on compute nodes. Gains are read only for the
  antennas in the partition and apply on-grid only (see
  [chi2-pipeline.md](chi2-pipeline.md)). The term is cached on the
  directory's mtime, so a re-solve is picked up.
- **Antenna order follows the MS.** The grid is symmetric, so a cell can ask
  for `(q, p)` when the MS stores `(p, q)`. `msdata.waterfall` reads the
  stored row and swaps the names to match it before building gain indices,
  cross-hand labels and the title. Applying `G_p` and `G_q` from the request
  to the `(q, p)` row used to corrupt half the grid's waterfalls whenever
  gains were in play.
- **Flag version** lists CASA's `<ms>.flagversions/FLAG_VERSION_LIST`
  (`name : comment` lines) where `flags.<name>` exists. Version tables are
  row-aligned with the main table, so the baseline's rows are found with an
  arcae TaQL query (`Table.from_taql`, `ROWID()`) on the main table and the
  same row numbers read from the version; `FLAG_ROW` is ORed in and grid slots
  with no row are flagged. arcae returns these boolean columns as `uint8`, and
  wants a writable row-index array. The web layer needs no python-casacore.
  Only listed names are accepted, which also keeps the name from being a path.
- **The column menu lists complex columns only.** The column is corrected and
  has a model subtracted, which means nothing for `WEIGHT`.
- **Problems are reported in the panel**, not as a broken image: the
  `/waterfall` route runs `msdata.check()` (columns exist, product formable,
  gains load and align, version exists) before emitting the `<img>`. If the
  PNG still fails, the page fetches it again and shows the route's `detail`.
- **Choices survive a change of baseline.** The route marks each control whose
  value differs from the heatmap's default with `class="keep"`, and
  `static/app.js` adds those to a cell click's parameters in
  `htmx:configRequest` (an `hx-include` would warn on every click while
  nothing is marked). Only *changed* controls are
  carried, so a product left at its default follows the heatmap's product
  when that changes, while an explicit choice sticks.

## The grid

The matrix holds each baseline twice, as `(p, q)` and `(q, p)`, so the header
counts the upper triangle. Diagonal cells are auto-correlations, which `chi2`
does not measure, but whose waterfalls are worth seeing when flagging (RFI
and a misbehaving receiver show there first). When the scan's
`autocorrelations` attr is true, or absent (an older zarr), the diagonal is
hatched and clickable; otherwise it carries no `hx-get`. An older zarr over
an MS without autos gets a panel saying so (`_has_auto`, from
`msdata.baseline_rows`) rather than a broken plot.

For the waterfall to have them, `_open_partition` passes `auto_corrs=True`:
xarray-ms drops auto-correlations by default. On an MS without any, xarray-ms
pads one baseline per antenna as missing (NaN, flagged); cross baselines and
the time grid are unchanged, and everything looks baselines up by name
(`baseline_index`), never by position. An auto's panel names it "m004
auto-correlation", shows no χ²/dof and no "no longer shows what the heatmap
measured" note (the heatmap measured nothing there), and defaults the model
to none: a predicted model's auto-correlations mean nothing when hunting
RFI. Because of that different default, a model carried in from a
cross-correlation would look like a default on the auto's panel and be
dropped; `static/carry.mjs` holds it back and hands it on, unless the model
is changed on the auto panel. Every other control follows the panel exactly,
so resetting one on an auto takes effect (merging everything once let a
reset flag version come back). `app.js` is a module for that import.
Flagging an auto works as on any baseline; its rows are never reversed.

Fields are listed by `FIELD::NAME` (`msdata.field_names`, arcae), with the
`FIELD_ID` as the option value; the zarr stays keyed by ID. If the MS cannot be
read the menu falls back to IDs.

The grid and the histogram sit in two equal columns of one CSS grid, and the
histogram PNG is square, so the two render at the same size.

The waterfall title carries the baseline length: the distance between the two
`ANTENNA::POSITION` rows (ITRF metres), read once per MS with arcae
(`msdata.baseline_length`). Its time axis is labelled in UTC; pixels stay in
slot coordinates, since flagging selections use them.

The heatmap ships one JSON blob (`#grid-data`: antenna names, the χ²/dof
matrix, antenna positions) rather than attributes on every cell. `app.js`
reads it for:

- **the hover tooltip** at the cursor, mirrored in the readout under the grid:
  names with antenna numbers (the grid index is the `ANTENNA` row, since
  `chi2` records every antenna), χ²/dof, baseline length. A custom element,
  not `title=`, which shows only after a delay;
- **arrow-key navigation**: up/down change antenna1, left/right antenna2,
  skipping cells that plot nothing (the diagonal, unless it plots
  auto-correlations);
- **following the selection.** The selected baseline is held by antenna
  *names*, since another scan can hold a different antenna set. When the
  heatmap re-renders (field, scan, bin or product) and a waterfall is open, the
  same baseline is re-fetched there, with the carried settings. This hooks
  `htmx:afterSettle`, not `afterSwap`: htmx has not wired the new cells' `hx-get`
  at `afterSwap`, so a click there does nothing.

Antenna names label the grid: slanted at 60° along the top, horizontal down the
side. They sit in a gutter and band around the square (`.grid-frame`), sized to
the cell pitch with container query units (`100cqw / --n`); the histogram pads
by the same amounts, on its right, so the two squares stay equal.

While an htmx request is in flight, `#heatmap` and `#waterfall` show a spinner
over their dimmed content (the children are dimmed, not the region, so the
spinner stays opaque). Without it, the first click showed nothing in the
waterfall area: the per-PNG spinner only exists once a panel has rendered.

**Save as PNG** renders `<body>` in the browser with `static/html-to-image.js`
(html-to-image 1.11.11, MIT, vendored like htmx so a node without internet
still works). It goes through an SVG `foreignObject`, so the page comes out as
shown, scrolled-off parts included, at `pixelRatio: 2`. The library re-fetches
every `<img>` it embeds; for the waterfall that would mean reading the MS
again, so `savePng` first swaps each loaded image's `src` for a data URL of its
decoded pixels (the original is kept in `data-src`). Printing is no substitute:
browsers drop background colours by default, and every grid cell is one.

The histogram's axis covers the data; the χ²/dof = 1 reference is drawn only
when it falls inside, since forcing it in left most decades empty on
uncalibrated data.

## Loading feedback

Three layers, because the slow part is not always an htmx request:

- `#busy`, a bar across the top, is the `hx-indicator` for every htmx request
  (set once on `<body>`, inherited). It also stays lit while any `.plot.loading`
  exists (`body:has(...)`), since the waterfall's first block is the slow part and its
  panel is usually below the fold when a cell is clicked.
- The region a request will replace (`#heatmap`, `#waterfall`, `#selectors`)
  is dimmed between `htmx:beforeRequest` and `htmx:afterRequest`.
- The histogram PNG is not an htmx request. Its `<img data-plot>` sits in a
  `.plot.loading` wrapper showing a spinner; `app.js` `fetch`es it on
  `htmx:afterSettle` and sets a blob URL, so a failure's JSON detail arrives
  with the response (an `onerror` handler used to re-request the URL to read
  it). The canvas waterfall fetches its own blocks and clears its wrapper's
  `.loading` when the first one is drawn.
- Cells inherit `hx-sync="this:replace"` from `.grid`: a new click or arrow
  key aborts the previous waterfall request, so the outlined cell and the
  waterfall cannot disagree. The heatmap is loaded once, by the selector
  form's `load` trigger.
- The scan select re-renders the selectors (`/scans`) like field and spw:
  each scan has its own time and frequency bins. Bin, product and scale are
  kept where the new scan has them.

Static URLs carry `?v=<digest of the static files>`. `StaticFiles` sends no
`Cache-Control`, so browsers cached the stylesheet heuristically and served
the old one with new HTML after an upgrade: no indicators, an unsized
histogram.

The selected cell keeps an outline (`.selected`) across heatmap re-renders.

## Manual flagging (write path)

Design: issue #16. Source: `web/flagging.py`, the `/flag*` routes in `app.py`,
and the selection tool in `static/wf/` (see "Selection on the canvas"). Terms
(selection, pending selection, edit, scope, backup) are in `GLOSSARY.md`.

- **Selections are polygons** in the waterfall's (time slot, channel) data
  coordinates; a rectangle is four vertices. `rasterise` selects a cell when
  its centre `(i, j)` is inside, by even-odd, with half-open edges, so a
  rectangle around slots `t0..t1` and channels `c0..c1` has corners at
  `t - 0.5` / `c + 0.5`. The server's mask is what is written; a client preview
  must use the same rule.
- **`POST /flag`** takes `{field, spw, scan, p, q, product, vertices,
  correlations, baselines, antenna, gtable, flagversion, dry_run}`: the
  polygon plus the panel's settings it was drawn on. `baselines` is the
  scope (step 4): `baseline` (the default, the panel's), `antenna` (every
  baseline of `antenna`, an index that must be `p` or `q`, else 400; its
  auto-correlation included) or `all` (every row of the scan, autos too).
  Every baseline of a scan shares its time grid and channels, so the same
  mask applies to each; wide scopes stay inside the scan.
  `msdata.scope_rows` finds their rows and marks each `reversed` when it is
  stored the other way round from how xarray-ms presents its baseline. `correlations` is `all` (the default: one flagged
  correlation should flag them all) or `product`, the correlations the plotted
  product is formed from, by the waterfall's own rule
  (`msdata.product_correlations`): XX and YY for I, and all four for
  anything corrected by a full-Jones gain table. An unset `gtable` means the
  term `chi2` used, as on `/waterfall`; it must be on the allow-list (403),
  and a term that is not a gain table is a 400. A non-empty `flagversion`
  is refused with 409, dry runs too: the picture showed a saved version, not
  the FLAG the write builds on. Rows are found with
  `msdata.baseline_rows` (shared with the flag-version reader) and mapped to
  slots with `msdata.slot_index`. It returns the baseline's rows stored either
  way round plus a `reversed` mask; on reversed rows the chosen correlations
  are permuted by `msdata.reversed_correlations` (XY ↔ YX, RL ↔ LR), since
  `corr_idx` is in the waterfall's (converted) order. The flag-version
  reader applies the same permutation, so a saved version and the live FLAG
  agree. A test flags XY on a baseline stored both ways at alternate times.
  `FLAG_ROW`
  is only ever set (`FLAG_ROW | all(FLAG)`); recomputing it as `all(FLAG)`
  used to unflag rows flagged through FLAG_ROW alone. 403 without
  `--allow-flagging`; 400 for an empty or unformable selection; 422 for
  fewer than three vertices or a negative antenna index; 503 while the backup
  is being written.
- **Backup.** Before any write, FLAG and FLAG_ROW are saved as a CASA flag
  version, `surfvis_before_<UTC to the microsecond>` (seconds alone let two
  sessions collide). `serve --allow-flagging` starts it in a background
  thread at startup: copying the column takes minutes on a large MS (about
  335 M cells/s, so ~50 min for 8 h of 32k-channel MeerKAT data), far too long
  for one HTTP request. Writes answer 503 until it is done, `/flag/state`
  reports `backup_state`, and a failed backup disables writing. `Flagger`
  used directly takes it synchronously on the first write. It is written via `Table.from_descriptor` with the
  source's column descriptors, copied one `DATA_DESC_ID` run at a time in
  ~256 MB chunks. Keep arcae's `_c_order` hint in the descriptor: without it
  a tiled FLAG column's fixed cell shape is read transposed and the copy fails.
  TaQL `SELECT ... GIVING ... AS PLAIN` was rejected: it copies every subtable
  and labels the result a Measurement Set.
- **Undo** (`POST /flag/undo {"id": n}`) clears exactly the cells edit `n`
  turned on, reading the rows as they are now, and recomputes FLAG_ROW only
  for rows that edit flagged. The stack is shared by every tab, so `n` must
  be the most recent edit (409 otherwise, and when empty): a button showing
  an older edit cannot undo another tab's newer one. Edit ids are
  microseconds since the epoch, so they are unique across restarts too.
  Apply, undo and `/flag/state` report each edit's `Edit.summary()`: id,
  description, field, spw, scan, `antennas` (the panel's baseline, in the
  waterfall's order), `scope`, `scope_antennas`, `baselines` and cell
  counts. One selection is one edit and one undo, whatever its scope. An
  edit records the cells it changed, not row snapshots: snapshots both wiped
  later writes by `flag-chi2`, CASA or another server, and cost ~59 MB per
  MeerKAT baseline-scan. It holds them as a bit mask over (its rows, the
  selection's channel range, correlation), zlib-compressed per chunk of
  rows (`Edit.changed`): cell indices cost 12 bytes per cell, hundreds of GB
  for a full-band rectangle over every baseline of a MeerKAT scan, and the
  mask of a polygon compresses to almost nothing. Reads and writes go in
  chunks of `CHUNK_CELLS` (2^24) and touch only the selection's channels
  (arcae takes a channel index); FLAG_ROW needs the rest of a row, read only
  for rows the selection leaves fully flagged within its channels. A test
  pins that a 37-cell chunk size writes what one chunk does. FLAG_ROW is set
  only on rows the edit changed: a row already fully flagged in FLAG but not
  FLAG_ROW (as many flaggers leave one) used to get FLAG_ROW, which undo
  then kept, across a whole scan for scope `all`. An apply that would change
  nothing is refused (400). A journal in step 3's format (cell indices) is
  still recovered.
- **Dry runs on wide scopes.** A full-band rectangle over every baseline of
  a MeerKAT scan is ~1e11 cells: reading FLAG to count it takes minutes,
  twice (dry run, then apply). Above `DRY_RUN_CELLS` (2^26) a dry run
  returns the cells selected with `newly_flagged: null` and the button
  reads "Flag up to N cells"; the apply reports the real count. Dry runs do
  not take the `Flagger` lock, and the browser aborts a superseded one.
  `baselines` counts those with rows in the selected slots. The stack is per process; the backup version is the durable
  fallback. `GET /flag/state` lists the edits. Apply and undo share one
  idempotent operation, `_put_cells(table, edit, on)`.
- **Crash safety: the journal.** casacore flushes a TiledShapeStMan (where
  real MSs keep FLAG) by truncating its header file `table.f<n>` (`O_TRUNC`)
  and writing it again, and fsyncs neither the header nor the tiles (strace).
  Killed in between -- reproduced with strace fault injection through the
  real `Flagger` -- the header is empty and neither casacore nor arcae opens
  the MS. A flag write leaves that header byte-identical (tiles are rewritten
  in place), so a copy taken just before the write restores it exactly. So
  `Flagger._write`: (1) writes `<ms>.flagversions/surfvis_journal/` -- the
  edit's rows, changed cells and FLAG_ROW rows, plus copies of the
  `table.f<n>` files (up to 64 MB, i.e. headers, not SSM data) of the storage
  managers holding FLAG and FLAG_ROW -- fsynced, built under a temporary name
  and renamed into place; (2) writes; (3) fsyncs every file of those storage
  managers and the MS directory; (4) retires the journal (renamed aside, then
  deleted). Even with a read handle held open (as xarray-ms does), casacore's
  flush happens inside `putcol`, never after `close` returns, so the window
  is fully covered. `flagging.recover(ms)` runs at `serve --allow-flagging`
  startup, before the backup: if the MS does not open it puts the saved
  headers back (keeping the files as found in `found/`), then clears the
  journalled cells -- which rolls back an apply and completes an undo -- and
  is itself safe to interrupt. A write that raises in-process runs the same
  recovery at once; if that fails too, `Flagger.broken` is set and writes are
  refused until a restart. 25 random `kill -9`s during a stream of edits (7
  of them mid-write) left every MS openable and no edit half-applied.
- **Audit log.** Every apply, undo and recovery rollback appends one JSON
  line to `<ms>.flagversions/surfvis_flags.jsonl` (fsynced), so a session's
  manual flags can be reviewed or replayed. An apply records time, user@host,
  edit id, backup, field/spw/scan, scope and its antennas, the panel's
  baseline, product, correlations, vertices, the slot and channel ranges and
  their UTC/MJD times and frequencies, and cell counts; undo and rollback
  name the id. The line is written after the storage managers are fsynced
  and before the journal is retired, and the journal records the edit's id
  and whether it was an apply or an undo: recovery from a crash after the
  line appends `rollback` (or completes the undo and appends `undo`), so the
  log never claims an edit the MS does not hold. A failed log write fails the
  edit and puts it back.
- **Stale cells.** Staleness is per cell. `Chi2Store.cell_times` is when each
  cell's χ² was computed: the group's `computed` array, else its `created`,
  else (an older zarr) the group directory's mtime, which is when `chi2`
  wrote it, so an edit made while it ran is missed. Not the root's mtime:
  every recompute re-consolidates at the root, which used to clear every
  other scan's stale marks. An audit line's `unix` is stamped when the write
  is complete (FLAG written and fsynced), not when the edit was built: a
  recompute that started during a slow write read the old FLAG, yet was
  newer than the edit's stamp, and its cells passed as fresh. `flagging.stale_cells` walks the
  audit log: a cell is stale if an edit touching it was applied after the
  cell's time and is still applied, or applied before and undone (or rolled
  back) after it; an apply and its undo both after cancel. The diagonal is
  never stale (autos have no χ²). All of this relies on `chi2` and `serve`
  sharing a clock. `GET /stale` returns `{edits, cells (upper triangle),
  recomputed, can_recompute, why_not}`, also embedded in `#grid-data`;
  `static/stale.mjs` turns it into markers (both triangles), the note, the
  button label and the provenance line; `app.js` applies them on every
  heatmap render and fetches `/stale` again on `surfvis:flagged`, which
  `flagtool.mjs` dispatches after an apply or undo. Reading the log rather
  than the process's stack means edits by other sessions and from before a
  restart count. A read-only `serve` shows them too.
- **Recompute** (step 5): `POST /recompute {field, spw, scan}` forms the
  scan's stale cells again and writes them into the zarr, in a background
  thread, one job per server (`GET /recompute/state`: state, cells,
  rows_done/rows_total, error). Needs `--allow-flagging` (403): it writes.
  409 when nothing is stale, a job is running, or the group cannot be
  recomputed (`Chi2Store.recomputable`: per-chunk stores, or attrs missing
  `rcol`/`wcol`/`fcol`/`corr_labels`). `web/recompute.py::cells` follows
  `chi2`'s recipe, not the waterfall's: the zarr's columns, model, gains and
  products, read by row with arcae in chunks, formed by
  `utils/products.form`, `|r|^2 w` over samples unflagged in **FLAG only**
  (FLAG_ROW ignored, as `chi2` does), rows as stored (a baseline stored both
  ways contributes both orientations, as in `chi2`), and `1/σ²` computed in
  the column's float32 for `SIGMA_SPECTRUM` (float64 differed from `chi2` by
  1e-8). Tests pin agreement with `chi2` to 1e-9 for plain residuals, gains
  with Stokes I and V, random weights, `SIGMA_SPECTRUM` and reversed rows.
  Every product is recomputed for the stale cells; the cost is the reads.
  `Chi2Store.write_cells` sets both triangles of `chi2` and `counts`, stamps
  the cells' `computed` with the time the job **started** (before it read
  FLAG, so an edit made meanwhile stays stale; taken under the `Flagger`
  lock, so never mid-write), bumps `recomputed_cells`, records
  `recomputed_ms` (serve's `--ms` need not be the zarr's `ms`),
  and re-consolidates. Adding `computed` goes through xarray (for its
  dimension names), whose append mode replaced the group's attrs: they are
  saved through zarr and put back. `computed` is written last, so a kill
  mid-write leaves the cells stale, to be recomputed again (a heatmap read
  between the `chi2` and `counts` writes can see a mixed cell, still marked
  stale). The browser polls the job by its id (another tab's job or a
  restart ends the poll and re-enables the button), then fetches the
  heatmap and swaps it in itself, dropping it if the user has moved to
  another scan: an htmx swap would follow the waterfall, and a flag to
  suppress that leaked when the request failed. `--imagesout` PNGs are not touched.
- **Never open a possibly broken MS in the server process.** `_opens()`
  probes in a child process: a failed arcae open of the broken MS left a read
  lock in casacore's in-process lock registry (`FileLocker::acquire` waits on
  a condition variable, not fcntl), and the recovery write then hung, in 5 of
  8 runs.
- **Exit waits for a write in flight.** An atexit handler takes each live
  `Flagger`'s lock (up to 120 s): FastAPI runs the write in a daemon worker
  thread, which Ctrl-C does not interrupt but interpreter exit would kill.
- **The backup is written as `.flags.<name>.partial`**, fsynced and renamed,
  and only then listed in `FLAG_VERSION_LIST`, so a backup cut short never
  looks usable.
- A write is wrapped in `msdata.writing(ms)` and clears nothing: no cache
  holds FLAG (see "Plane cache"), so the next waterfall reads the new flags
  by row (about 0.03 s on com08) and keeps the open partition and the cached
  plane. On exit `writing` records the MS's new storage fingerprint as seen,
  so `_open` does not take the write for another program's rewrite and
  rebuild the scan (it did, 24 s per edit on the ESO137-006 MS). If the
  fingerprint had already moved before the write, that change is left for
  `_open` to find. `_open`'s fingerprint check takes the same lock, so a
  request never sees a half-done write of ours.
- `Flagger` serialises apply/undo with a lock: FastAPI runs sync routes in a
  thread pool.

## Profiling (`serve --profile DIR`)

`web/profiling.py`. Off by default, and then a `span()` costs one context
variable lookup. With a directory:

- `ProfileMiddleware` gives each request a span list in a `ContextVar`.
  Starlette copies the context into the worker thread a sync route runs in, so
  `span()` / `@timed()` anywhere in `msdata`, `store` or `app` lands on the
  request that caused it. Spans nest (indent = depth) and are listed in the
  order they *finished*, so a parent follows its children.
- Each response carries `Server-Timing` (the browser's network panel draws
  it); one block per request goes to stderr and a JSON line to
  `DIR/spans.jsonl`. `python -m surfvis.web.profiling DIR` totals by span.
- Routes are wrapped in `@profiling.cprofiled`, which runs cProfile in the
  route's own thread (cProfile sees only the thread that enabled it) and dumps
  `DIR/<time>-<route>.prof` when the request took over `SLOW` (1 s). One
  profiler at a time: an overlapping request just isn't cProfiled.

Spans sit on the cache *misses* (`@timed` below `@lru_cache`), so a hit does
not appear at all. The expensive ones to watch: `open_partition`,
`scan_rows: query` (one TaQL query per scan), and the column reads.

Requests are only printed when they finish, so a watchdog thread reports any
request running longer than `STUCK` (10 s), every 10 s, with the spans it is
inside; `kill -USR1 <pid>` dumps every thread's stack via `faulthandler`.
`templates.TemplateResponse` is wrapped as a `render template` span. cProfile
roughly doubles pure-Python time (Jinja especially), so read totals from a
`--profile` run as upper bounds.

First finding, on com08: the first `/waterfall` hung for over ten minutes in
`data_group` → `open_datatree` (see "xarray-ms facts" above), and opened it
twice. Fixed by not opening the datatree at all.

Second round, same MS (10.9 M rows, 64 antennas, 9 secondary columns):

| Cost | Where | Now |
|---|---|---|
| 4.6 s per heatmap | `render.cell_colours` called the matplotlib norm per cell, through numpy masked arrays | one vectorised call; a test pins it matches `to_hex(cmap(norm(v)))` cell for cell |
| 7.5 s x 2 per waterfall | `baseline_rows` ran a TaQL query over the whole main table, twice when the click was in the unstored order | `scan_rows()` reads (ANTENNA1, ANTENNA2, ROWID, TIME) for the scan once; `baseline_rows` selects from it |
| 37 s per scan | `open_partition`: 22.6 s building xarray-ms's structure (once per MS; 15 s of it partitioning the rows), 14.4 s `secondary_variables` row-shape `np.unique` per column (every scan) | hidden: `/heatmap` warms its scan in the background |

**Scan rows** are keyed on `table.dat`'s mtime and size, not the storage
fingerprint, and `invalidate()` leaves them: rows only move when `table.dat` does.

**Warm-up.** `/heatmap` calls `msdata.warm()`: one worker thread opens the
partition and reads its scan rows while the heatmap is on screen. Only the
latest request runs (a queued one that was overtaken is skipped), so flicking
through scans does not queue reads. `create_app(warm=False)` turns it off.
With profiling on it prints `[profile] warmed field F spw S scan N: ...`.

**Coalescing.** `msdata._Coalescer.hold(key, what)` gives `_open`,
`scan_rows` and `cached_plane` a per-key lock, so a request that arrives while
the warm-up is reading waits for it instead of repeating it; the wait shows as
`<what>: wait` (only when it actually waited).

**Third round: the row-shape check is gone upstream.** On the full
ESO137-006 MS (10,876,320 rows, 64 antennas, 2016 cross-correlation baselines,
1024 channels, 2 correlations, 24 scans of about 453,600 rows, and 16 secondary
columns, all `TiledColumnStMan` FixedShape), cProfile of reopening one scan
with 0.4.0a12 gave 25.0 s, 24.6 s of it `np.unique` in `secondary_variables`.
(The 14.4 s above was an earlier, unexplained figure.) xarray-ms #183 replaced
the sort with a comparison against the first row and reads one row's shape
for FixedShape columns: with 0.5.14, which carries the same code, the reopen
took 0.06 s and the first open 23.1 s instead of 48.2 s. surfvis now pins the
write-support commit that carries it.

**What a waterfall costs now** (com08, ESO137-006, profiled 2026-11-26): a
plane-cache hit 26 ms; a new baseline in a warmed scan about 5.8 s, all of it
the two complex column reads (about 2.9 s each; FLAG 0.05 s); changing
model, flag version or product about 2.7-3 s (one read). The reads are slow
because of the MS's tiling, not surfvis: `CORRECTED_DATA` is a
`TiledColumnStMan` with tiles of 2 corr x 16 chan x 4096 rows (1 MB),
`MODEL_DATA` 2 x 2 x 4096, and one baseline's rows are 2016 rows apart, so a
baseline over a scan touches thousands of tiles to return 3.7 MB. The
warm-up costs 23.7 s to open the first partition (xarray-ms's structure, once
per MS) and 9.4 s for the scan's row query.

**Flag writes no longer reopen.** The flagger used to call `invalidate()`
after every write, which cleared xarray-ms's multiton, so the next waterfall
rebuilt its structure and reopened the scan -- about 24 s on that MS. Now a
write, an undo or a switch of flag version costs one FLAG read for the
baseline: 0.03 s warm, 0.23 s cold, plus 0.01 s to open the table (com08).

## Colour scaling

Default is **log**. chi-squared/dof spans orders of magnitude, and on a linear
scale one bad baseline renders every other cell black — which defeats the point
of looking at the grid. The same applies to the histogram: linear bins collapse
the bulk into a single bar, so log bins follow the log norm. `robust` (2nd-98th
percentile) and `full` remain available; note that `robust` is degenerate when
there are few antenna pairs, since the 98th percentile lands inside the
outliers.

## The fork deadlock

`core/chi2.py` builds its `ProcessPoolExecutor` with an explicit **spawn**
context. `dask.config.set(pool=ThreadPool(...))` has already started threads by
that point, and forking a process with live threads deadlocks the children —
the parent sits in `as_completed` forever. This was reproducible on every run
before the fix.

Spawned workers re-import matplotlib, so `utils/plotting.py` forces `Agg`.
Otherwise a forwarded `DISPLAY` selects TkAgg, and when the X connection drops
(a closed SSH session under `screen`) every worker dies on an XIO error and the
parent raises `BrokenProcessPool`.

## Testing

`tests/fixtures/ms.py` writes a real Measurement Set with python-casacore, small
enough to build in well under a second. Three subtables have to be populated or
xarray-ms refuses to construct the MSv4 view: `FEED` (validated against
`ANTENNA1`/`ANTENNA2`), `STATE`, and the `FIELD` direction columns
(`PHASE_DIR`/`REFERENCE_DIR`/`DELAY_DIR`). The fixture plants an inflated
residual on one baseline, and the tests follow that outlier all the way to the
rendered grid. Its columns are mutually consistent --
`DATA = G_p MODEL_DATA G_q^H + RESIDUAL` and
`CORRECTED_DATA = inv(G_p) DATA inv(G_q)^H`, with diagonal gains varying in
time, channel and antenna -- so the corrections are checked exactly.
`write_gain_table` writes those gains as a QuartiCal term and
`add_flag_version` writes a CASA flag version.

Tests skip cleanly without the `[full,web]` extras; `tests/conftest.py` carries
the `needs_ms` / `needs_web` markers.

## Canvas waterfall

The page no longer shows `/waterfall.png` (the route is kept, unused by the
page). `templates/_waterfall.html` renders a `.wf` root with a `<canvas>`;
`app.js` imports `static/wf/widget.mjs` on `htmx:afterSettle` of `#waterfall`
and calls `mount(root)`.

- **Route:** `GET /waterfall.bin` takes the PNG route's query plus
  `t0,t1,c0,c1,max_t,max_c,meta`. The plane is reduced to at most
  `max_t x max_c` cells (`planes.py`); the first request sets `meta=true` and
  also returns slot times and channel frequencies.
- **Wire format:** a JSON header (version, nt, nc, window, tstep, cstep, vmin,
  vmax, quantity, colormap, title, rect) followed by float32 values and uint8
  flag fractions; decoded by `static/wf/format.mjs`.
- **Modules** (`static/wf/`, node-tested via `tests/test_js.py`): `format`,
  `luts`, `colormap` (paint), `view` (View, Latest, niceTicks, wheelFactor),
  `widget`, and for flagging `select` (rasterise, dragRectangle),
  `selection` (the gesture state machine) and `flagtool` (DOM wiring, not
  node-tested).
- **Interactions:** wheel zooms about the cursor, shift-drag or middle-drag
  pans, `0` or Reset view resets, hover reads time, frequency, value and flag
  percentage. After 150 ms of rest the visible window is refetched at screen
  resolution; `Latest` discards stale responses. "show flags" repaints locally.
  Plain left-drag and click belong to the selection tool (below), and only
  exist with `--allow-flagging`.

### Selection on the canvas

Step 3 of #16. With `--allow-flagging` the panel carries `data-flag` (the
JSON `/flag` needs: field, spw, scan, p, q, product, gtable, flagversion) and
`data-names`, and `widget.mjs` imports `flagtool.mjs`.

- **Gestures** (`selection.mjs`, plain left button only): drag makes a
  rectangle snapped outwards to the cells it touched (`dragRectangle`); clicks
  add polygon vertices; clicking the first vertex (8 px), double-clicking or
  Enter closes it. A closed shape is a **pending selection**; only a second
  Enter or the button applies it. Backspace drops a vertex, Esc cancels, a
  new press replaces a pending selection -- except the second press of a
  double-click on the first vertex (500 ms, 8 px), which once threw away the
  polygon it had just closed.
- **Never one key press from drawing to writing.** Key auto-repeat (`e.repeat`)
  and modified keys are ignored: holding Enter used to close a polygon and,
  once its count arrived, apply it. A canvas press blurs the focused control
  (pointerdown's `preventDefault` otherwise left focus on, say, the
  checkbox, which swallowed Enter and Esc), and buttons blur after a click,
  so Enter cannot also click a focused button.
- **Preview and count.** `select.mjs::rasterise` is a port of the server's
  rule returning row spans (a full mask is 15 MB at 32k channels); the same
  expression in the same order, pinned by 206 cases Python writes to
  `tests/js/rasterise_cases.json` (thirds, tenths and integers nudged by
  1e-3 catch a reordered interpolation, which plain integer cases did not).
  Counts come from a `/flag` dry run, which also returns the correlation
  names for the label. No cells, or a failed count, leaves nothing to
  confirm. Changing "all correlations" or "baselines", or undoing,
  re-counts. A wide scope's label adds `· N baselines`, and the undo label
  names the scope.
- **Scope menu** (`.wf-scope`): "this one", "every baseline of" each of the
  panel's antennas (one on an auto panel) and "all baselines". It has no
  `name`, so it is neither part of the panel's form nor carried: every
  panel starts at "this one", so a wide scope cannot follow you to the next
  baseline. A carried pending selection is still carried; it is the scope
  that resets.
- **After apply or undo** the waterfall re-fetches in place (`api.reload`),
  keeping zoom and colours; an undo of another baseline only reports.
- **Carrying.** A pending selection is module state in `flagtool.mjs`, carried
  to the next panel of the same (field, spw, scan) and shown pending there
  (with a fresh count); it is kept after an apply, so arrow, Enter, arrow,
  Enter flags one patch across baselines. Esc or a new gesture discards it;
  a new scan drops it. While a shape is being drawn the panel has
  `data-drawing="1"` and `app.js` ignores the arrow keys.
- **Disabled** while the flags menu shows a saved version (and the server
  refuses such requests); confirming waits for the backup, polled from
  `/flag/state` every 3 s.
  Layers added through `api.addLayer(fn)` draw clipped to the plot area.
- **Cache:** static files are served `Cache-Control: no-cache`
  (`RevalidatedStaticFiles`), because imports between ES modules cannot carry
  the `?v=` digest.
- Save as PNG captures the canvas too (a canvas is rasterised by html-to-image).
- **Wheel units:** `wheelFactor` normalises `deltaY` by `deltaMode` (Firefox
  reports lines, not pixels) and clamps each event, so zoom feels alike across
  browsers. The widget aborts a superseded fetch, skips a refetch when the
  block on screen was fetched with the same window and bin counts (so the first
  load makes one request), and drops a drag when pointer capture is lost.

### Plane cache

`msdata.cached_plane` keeps the last four formed baseline planes (an `lru_cache`)
so zooming and panning re-slice memory instead of re-reading the MS. The key is
the MS's generation (bumped by `_open` when the storage fingerprint moves for a
reason other than our own flag write), the gain term's mtime, and the baseline and
settings (field, spw, scan, antennas, product, column, model, gtable). **FLAG is
not in the entry.** The entry's flags are only what the data implies (grid gaps,
gain flags); every call reads FLAG, or the flag version, by row and ORs it in.
That is exact: `products.form` flags a product wherever any correlation it
reads is flagged, and the plane's values do not depend on flags. The antennas are put in xarray-ms's order first (`view_order`), so
the two mirrored cells of the heatmap share an entry: keyed on the clicked
order, the second cell re-read the MS (5.8 s on com08). It is **not** keyed on quantity or on the heatmap's product, so
another quantity, or the same baseline reached from another heatmap, hits the
cache, and so does a flag edit or another flag version. `invalidate(ms)` clears
it, for data-column rewrites. An entry holds the plane as complex64 and the flags as
bool, `values` is dropped: about 9 bytes per cell, so 450 x 32768 is ~130 MB and
four entries ~0.5 GB. Colour limits are cached per quantity and flag version on
the entry (`Waterfall.limits`), so colours do not move on zoom or on a flag
edit (the edit is what changes on screen), and are not recomputed.

Concurrent misses for the same plane (two tabs, a quick double click) wait on
a per-key lock (`_Coalescer`) for the first read rather than each reading the MS;
`lru_cache` does not coalesce them. A miss still peaks at roughly 1-2 GB at
MeerKAT scale while the plane is formed; the waterfall passes `weight=None`
to `products.form`, so unit weights are a broadcast view rather than a
float64 array per correlation.

The widget registers its document-level listeners with one
`AbortController` and tears them down (with its `ResizeObserver` and any
in-flight fetch) on the `htmx:beforeSwap` that replaces the panel. Until the
next keypress, an old panel used to stay reachable from a `keydown` listener,
holding its blocks and offscreen canvas.
`planes._binned` skips its padded copy when the window divides evenly.
