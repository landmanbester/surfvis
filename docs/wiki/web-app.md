---
type: reference
title: Chi-squared browser
description: How surfvis serve links surfchi2 output to on-demand baseline waterfalls, the waterfall's correction / Stokes / flag-version controls, and the xarray-ms facts it depends on.
tags: [web, fastapi, htmx, xarray-ms, zarr]
timestamp: 2026-11-05
last_verified_commit: 00cccac
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
  attrs               ms, rcol, wcol, fcol, mcol, gtable, corr_labels, antenna_names
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
- **Column names are MSv4, resolved via `data_groups`.** `DATA` is exposed as
  `VISIBILITY` and `WEIGHT_SPECTRUM` as `WEIGHT`. Do not hardcode that: read
  `node.ds.attrs["data_groups"][group]["correlated_data"]` / `["weight"]` /
  `["flag"]`, as `pfb-imaging`'s `core/imager.py` does. `msdata.DataGroup`
  wraps this and `DataGroup.resolve()` translates the MSv2 names the data group
  maps (`DATA`, `WEIGHT_SPECTRUM`, `FLAG`), so both spellings work in the UI.
  It must translate **nothing else**: it once mapped `CORRECTED_DATA` to
  `VISIBILITY` too, so choosing `CORRECTED_DATA` plotted `DATA`.
- **`data_groups` is attached by `open_datatree`, not `open_dataset`.** Hence
  `msdata.data_group()` opens the tree once and caches it, while the per-chunk
  reads use `open_dataset(partition_key=...)`.
- Columns that are *not* part of the MSv2 standard are surfaced automatically as
  secondary variables under their own names — which is exactly why `RESIDUAL` is
  plottable. Standard columns other than `DATA`/`WEIGHT_SPECTRUM` are dropped,
  except `CORRECTED_DATA`, `MODEL_DATA`, `FLOAT_DATA` and
  `CORRECTED_WEIGHT_SPECTRUM`. Those four appear under their own names --
  only once their cells are defined, so an empty `MODEL_DATA` is invisible.
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
cannot drift apart. `chi2`'s recorded `DATA` is resolved to `VISIBILITY` before
it reaches the menus, or the column would be listed twice.

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
counts the upper triangle. Diagonal cells are autocorrelations, which `chi2`
does not measure: they carry no `hx-get`, since the waterfall route would
answer them with a 404 that renders as a broken image.

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
  skipping the diagonal;
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

Design: issue #16. Source: `web/flagging.py`, the `/flag*` routes in `app.py`.
The canvas waterfall is in place, but it has no selection tool yet (step 3 of
#16), so the routes are driven by JSON.

- **Selections are polygons** in the waterfall's (time slot, channel) data
  coordinates; a rectangle is four vertices. `rasterise` selects a cell when
  its centre `(i, j)` is inside, by even-odd, with half-open edges, so a
  rectangle around slots `t0..t1` and channels `c0..c1` has corners at
  `t - 0.5` / `c + 0.5`. The server's mask is what is written; a client preview
  must use the same rule.
- **`POST /flag`** takes `{field, spw, scan, p, q, product, vertices,
  correlations, dry_run}`. `correlations` is `product` (those the plotted
  product is formed from: XX and YY for I) or `all`. Rows are found with
  `msdata.baseline_rows` (arcae TaQL, shared with the flag-version reader) and
  mapped to slots with `msdata.slot_index`, in the antenna order the MS
  stores (`msdata.stored_order`): a baseline stored both ways in one
  partition is two baselines, and flagging one leaves the other. `FLAG_ROW`
  is only ever set (`FLAG_ROW | all(FLAG)`); recomputing it as `all(FLAG)`
  used to unflag rows flagged through FLAG_ROW alone. 403 without
  `--allow-flagging`; 400 for an empty or unformable selection; 422 for
  fewer than three vertices or a negative antenna index; 503 while the backup
  is being written.
- **xarray-ms mislabels reversed baselines.** A baseline stored as (2, 0) is
  labelled (0, 2) by `baseline_antenna1/2_name`, but its data stay V_20,
  unconjugated. So the waterfall and the flag writer take the antenna order
  from the MS (`stored_order`), never from those labels: gains, cross-hand
  labels and rows all follow the data.
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
- **Undo** (`POST /flag/undo`) clears exactly the cells the last edit turned
  on, reading the rows as they are now, and recomputes FLAG_ROW only for rows
  that edit flagged; 409 when empty. An edit records the changed cells'
  indices, not row snapshots: snapshots both wiped later writes by
  `flag-chi2`, CASA or another server, and cost ~59 MB per MeerKAT
  baseline-scan. A write that fails half-way is put back before the error is
  raised. The stack is per process; the backup version is the durable
  fallback. `GET /flag/state` lists the edits.
- After a write `msdata.invalidate(ms)` drops the caches, so the next
  waterfall shows the new flags even within one mtime tick. Writes go through
  a separate arcae handle in the same process as xarray-ms's read handle;
  casacore shares the underlying table, and the tests confirm reads see writes.
- `Flagger` serialises apply/undo with a lock: FastAPI runs sync routes in a
  thread pool.

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
  `luts`, `colormap` (paint), `view` (View, Latest, niceTicks, wheelFactor), `widget`.
- **Interactions:** wheel zooms about the cursor, shift-drag or middle-drag
  pans, `0` or Reset view resets, hover reads time, frequency, value and flag
  percentage. After 150 ms of rest the visible window is refetched at screen
  resolution; `Latest` discards stale responses. "show flags" repaints locally.
  Plain left-drag and click are unbound, reserved for flagging polygons.
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
the MS fingerprint (storage mtimes), the gain term's mtime, and the baseline and
settings (field, spw, scan, antennas, product, column, model, gtable, flag
version). It is **not** keyed on quantity or on the heatmap's product, so
another quantity, or the same baseline reached from another heatmap, hits the
cache. `invalidate(ms)` clears it (the flagger
calls it after a write). An entry holds the plane as complex64 and the flags as
bool, `values` is dropped: about 9 bytes per cell, so 450 x 32768 is ~130 MB and
four entries ~0.5 GB. Colour limits are cached per quantity on the entry
(`Waterfall.limits`), so colours do not move on zoom and are not recomputed.

Concurrent misses for the same plane (two tabs, a quick double click) wait on
a per-key lock for the first read rather than each reading the MS;
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
