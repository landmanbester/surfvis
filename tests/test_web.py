"""Tests for the chi-squared browser.

The fixture MS plants an inflated residual on one baseline; every test here
checks that the pipeline carries that outlier through to something a human
would see.
"""

import pytest

from tests.conftest import needs_ms, needs_web

# Skip the whole module rather than fail collection on a lightweight install.
np = pytest.importorskip("numpy")


@needs_ms
def test_chi2_writes_a_readable_zarr(chi2_zarr):
    """chi2 --dataout produces a store the web layer can open."""
    from surfvis.web.store import Chi2Store

    store = Chi2Store(chi2_zarr["zarr"])
    assert store.scans == [(0, 0, 1), (0, 0, 2)]
    assert store.fields == [0]
    assert store.spws(0) == [0]

    # Diagonal correlations of a 4-correlation MS, resolved from [0, -1].
    assert store.products(0, 0, 1) == ["XX", "YY"]
    # One total per scan: the time and frequency chunks only split the work.
    assert store.group(0, 0, 1).chi2.dims == ("product", "antenna1", "antenna2")


@needs_ms
def test_hot_baseline_dominates_its_scan(chi2_zarr):
    """The planted bad baseline is the worst pair, by a wide margin."""
    from surfvis.web.store import Chi2Store, Selection

    store = Chi2Store(chi2_zarr["zarr"])
    matrix = store.matrix(Selection(0, 0, 1, "XX")).chi2_dof

    assert matrix.shape == (8, 8)
    assert np.isnan(np.diag(matrix)).all(), "auto-correlations carry no chi-squared"

    p, q = np.unravel_index(np.nanargmax(matrix), matrix.shape)
    assert {int(p), int(q)} == set(chi2_zarr["ms"]["hot"])
    assert matrix[p, q] > 100 * np.nanmedian(matrix)
    assert matrix[p, q] == matrix[q, p], "the matrix is symmetric"


@needs_ms
def test_heatmap_is_the_whole_scan(chi2_zarr):
    """Each cell is chi-squared/dof over every time and channel of the scan, however chi2 chunked it."""
    from casacore.tables import table

    from surfvis.web.store import Chi2Store, Selection

    store = Chi2Store(chi2_zarr["zarr"])
    matrix = store.matrix(Selection(0, 0, 1, "XX"))
    with table(chi2_zarr["ms"]["path"], ack=False) as t:
        sel = t.query("SCAN_NUMBER == 1")
        a1, a2 = sel.getcol("ANTENNA1"), sel.getcol("ANTENNA2")
        r, w = sel.getcol("RESIDUAL")[..., 0], sel.getcol("WEIGHT_SPECTRUM")[..., 0]
    for p, q in ((0, 1), (1, 5), (3, 7)):
        rows = (a1 == p) & (a2 == q)
        want = (np.abs(r[rows]) ** 2 * w[rows]).sum() / r[rows].size
        np.testing.assert_allclose(matrix.chi2_dof[p, q], want, rtol=1e-5)
    assert matrix.rcol == "RESIDUAL"


@needs_web
def test_older_stores_with_bins_are_summed(tmp_path):
    """Stores written before per-scan totals kept every chunk; the reader sums them."""
    import xarray as xr

    from surfvis.web.store import Chi2Store, Selection

    rng = np.random.default_rng(0)
    shape = (3, 2, 1, 4, 4)
    chi2, counts = rng.uniform(1, 2, shape), rng.integers(1, 5, shape).astype(float)
    dims = ("time_bin", "freq_bin", "product", "antenna1", "antenna2")
    xr.Dataset(
        {"chi2": (dims, chi2), "counts": (dims, counts)},
        coords={"product": ["XX"], "time_bin": np.arange(3), "freq_bin": np.arange(2)},
        attrs={"ms": "x.ms", "rcol": "RESIDUAL", "antenna_names": ["a", "b", "c", "d"]},
    ).to_zarr(tmp_path / "old.zarr", group="field0/spw0/scan1")
    got = Chi2Store(tmp_path / "old.zarr").matrix(Selection(0, 0, 1, "XX")).chi2_dof
    np.testing.assert_allclose(got, chi2.sum(axis=(0, 1))[0] / counts.sum(axis=(0, 1))[0])


@needs_web
def test_waterfall_reads_the_right_baseline(chi2_zarr):
    """The hot baseline reads back brighter than a quiet one."""
    from surfvis.web import msdata

    ms = chi2_zarr["ms"]["path"]
    p, q = chi2_zarr["ms"]["hot"]
    names = chi2_zarr["ms"]["antenna_names"]

    hot = msdata.waterfall(ms, 0, 0, 1, names[p], names[q], "XX", "RESIDUAL")
    cold = msdata.waterfall(ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")

    assert hot.values.shape == (12, 32), "one scan of times by all channels"
    assert np.median(hot.values) > 10 * np.median(cold.values)


@needs_web
def test_waterfall_accepts_either_antenna_order(chi2_zarr):
    """Clicking (p, q) or (q, p) in the grid finds the same baseline."""
    from surfvis.web import msdata

    ms = chi2_zarr["ms"]["path"]
    names = chi2_zarr["ms"]["antenna_names"]
    forward = msdata.waterfall(ms, 0, 0, 1, names[1], names[5], "XX", "RESIDUAL")
    reverse = msdata.waterfall(ms, 0, 0, 1, names[5], names[1], "XX", "RESIDUAL")
    assert np.array_equal(forward.values, reverse.values)


@needs_web
@pytest.mark.parametrize("product", ["XX", "XY", "V"])
def test_reversed_cell_applies_the_gains_to_the_stored_baseline(chi2_zarr, gain_table, product):
    """A lower-triangle cell asks for (q, p); gains, cross-hands and title must follow the MS's (p, q) row.

    The gains were applied in the requested order to the stored row, which
    corrupted half the grid's waterfalls whenever a gain table was in play.
    """
    from surfvis.web import msdata

    ms = chi2_zarr["ms"]["path"]
    names = chi2_zarr["ms"]["antenna_names"]
    kw = {"column": "DATA", "model": "MODEL_DATA", "gtable": gain_table}
    for quantity in ("real", "imag"):
        forward = msdata.waterfall(ms, 0, 0, 1, names[2], names[6], product, quantity=quantity, **kw)
        reverse = msdata.waterfall(ms, 0, 0, 1, names[6], names[2], product, quantity=quantity, **kw)
        np.testing.assert_array_equal(reverse.values, forward.values)
    assert (reverse.antenna1, reverse.antenna2) == (names[2], names[6])


@needs_web
def test_waterfall_carries_the_baseline_length(chi2_zarr):
    """The title's baseline length is the distance between ANTENNA::POSITION rows."""
    from casacore.tables import table

    from surfvis.web import msdata

    ms = chi2_zarr["ms"]["path"]
    names = chi2_zarr["ms"]["antenna_names"]
    with table(f"{ms}::ANTENNA", ack=False) as ant:
        positions = ant.getcol("POSITION")
    wf = msdata.waterfall(ms, 0, 0, 1, names[1], names[5], "XX", "RESIDUAL")
    assert wf.length == pytest.approx(np.linalg.norm(positions[1] - positions[5]))
    assert wf.length > 0

    # Slot times for the UTC axis: unix seconds, one per row, increasing.
    with table(ms, ack=False) as main:
        mjd = np.unique(main.query("SCAN_NUMBER==1").getcol("TIME"))
    assert wf.times.shape == (wf.values.shape[0],)
    np.testing.assert_allclose(wf.times, mjd - msdata.MJD_UNIX_OFFSET)


@needs_web
def test_residual_is_plottable_but_flag_is_not(chi2_zarr):
    """RESIDUAL is non-standard, so xarray-ms surfaces it; FLAG is the overlay."""
    from surfvis.web import msdata

    columns = msdata.available_columns(chi2_zarr["ms"]["path"], 0, 0, 1)
    assert "RESIDUAL" in columns
    assert "VISIBILITY" in columns
    assert "FLAG" not in columns


@needs_web
def test_msv2_column_names_resolve_to_msv4(chi2_zarr):
    """DATA is VISIBILITY and WEIGHT_SPECTRUM is WEIGHT in the MSv4 view."""
    from surfvis.web import msdata

    group = msdata.data_group(chi2_zarr["ms"]["path"])
    assert group.correlated_data == "VISIBILITY"
    assert group.weight == "WEIGHT"

    assert group.resolve("DATA") == "VISIBILITY"
    assert group.resolve("WEIGHT_SPECTRUM") == "WEIGHT"
    assert group.resolve("FLAG") == "FLAG"
    # Non-standard columns are not renamed by the MSv4 view.
    assert group.resolve("RESIDUAL") == "RESIDUAL"
    # Nor are the standard columns xarray-ms keeps as secondary variables.
    assert group.resolve("CORRECTED_DATA") == "CORRECTED_DATA"
    assert group.resolve("MODEL_DATA") == "MODEL_DATA"


@needs_web
def test_data_column_is_plottable_by_its_msv2_name(chi2_zarr):
    """Asking for DATA must work even though the variable is called VISIBILITY."""
    from surfvis.web import msdata

    names = chi2_zarr["ms"]["antenna_names"]
    wf = msdata.waterfall(chi2_zarr["ms"]["path"], 0, 0, 1, names[0], names[1], "XX", "DATA")
    assert wf.column == "VISIBILITY"


@needs_web
def test_each_partition_is_one_scan(chi2_zarr):
    """SCAN_NUMBER is in the partition schema, so a partition's time axis is the scan."""
    from surfvis.web import msdata

    names = chi2_zarr["ms"]["antenna_names"]
    one = msdata.waterfall(chi2_zarr["ms"]["path"], 0, 0, 1, names[0], names[1], "XX", "RESIDUAL")
    two = msdata.waterfall(chi2_zarr["ms"]["path"], 0, 0, 2, names[0], names[1], "XX", "RESIDUAL")
    assert one.values.shape == two.values.shape == (12, 32)
    assert not np.array_equal(one.values, two.values)


@needs_web
@pytest.mark.parametrize("quantity", ["amp", "phase", "real", "imag"])
def test_every_quantity_renders(chi2_zarr, quantity):
    from surfvis.web import msdata

    names = chi2_zarr["ms"]["antenna_names"]
    wf = msdata.waterfall(chi2_zarr["ms"]["path"], 0, 0, 1, names[0], names[1], "XX", "RESIDUAL", quantity=quantity)
    assert np.isfinite(wf.values).all()


@needs_web
def test_log_scale_separates_the_outlier(chi2_zarr):
    """A linear scale buries the bulk; the log default must not."""
    from surfvis.web import render
    from surfvis.web.store import Chi2Store, Selection

    store = Chi2Store(chi2_zarr["zarr"])
    matrix = store.matrix(Selection(0, 0, 1, "XX")).chi2_dof

    norm, vmin, vmax = render.make_norm(matrix, "log")
    colours = render.cell_colours(matrix, norm)
    assert vmin > 0 and vmax > vmin
    # The bulk must not all collapse onto a single colour.
    flat = [c for row in colours for c in row if c != render.NAN_COLOUR]
    assert len(set(flat)) > 5

    with pytest.raises(ValueError):
        render.make_norm(matrix, "nonsense")


@needs_web
def test_routes(chi2_zarr):
    """Every endpoint answers, and a missing chunk 404s rather than 500s."""
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web.app import create_app

    client = TestClient(create_app(chi2_zarr["zarr"]))

    assert client.get("/healthz").json()["scans"] == 2
    index = client.get("/")
    assert index.status_code == 200
    # Fields are listed by FIELD::NAME, keyed by FIELD_ID.
    assert '<option value="0" title="FIELD_ID 0" selected>testfield</option>' in index.text
    # Static URLs are versioned, so an upgrade cannot pair new HTML with a cached stylesheet.
    assert "/static/style.css?v=" in index.text
    # Save as PNG renders the page in the browser with the vendored html-to-image.
    assert 'id="save-png"' in index.text
    # Static files revalidate: ES module imports cannot carry the ?v= version.
    assert client.get("/static/wf/widget.mjs").headers["cache-control"] == "no-cache"
    assert client.get("/static/html-to-image.js").status_code == 200

    heatmap = client.get("/heatmap?field=0&spw=0&scan=1&c=XX")
    assert heatmap.status_code == 200
    # One clickable cell per ordered antenna pair; the diagonal is not a baseline.
    assert heatmap.text.count('hx-get="/waterfall') == 64 - 8
    assert heatmap.text.count('class="auto"') == 8
    assert "28 baselines" in heatmap.text
    # Antenna names label both axes: once along the top, once down the side.
    names = chi2_zarr["ms"]["antenna_names"]
    assert heatmap.text.count(f"<span><b>{names[3]}</b></span>") == 1
    assert heatmap.text.count(f"<span>{names[3]}</span>") == 1
    # The hover readout and keyboard navigation read the grid from one JSON blob.
    import json

    blob = heatmap.text.split('<script type="application/json" id="grid-data">')[1].split("</script>")[0]
    grid = json.loads(blob)
    assert len(grid["names"]) == 8 and len(grid["values"]) == 8 and len(grid["positions"]) == 8
    assert grid["values"][1][5] == max(v for row in grid["values"] for v in row if v is not None)

    png = client.get("/histogram.png?field=0&spw=0&scan=1&c=XX")
    assert png.headers["content-type"] == "image/png"
    assert png.content[:8] == b"\x89PNG\r\n\x1a\n"

    wf = client.get("/waterfall.png?field=0&spw=0&scan=1&c=XX&p=1&q=5&column=RESIDUAL&quantity=amp")
    assert wf.headers["content-type"] == "image/png"

    assert client.get("/heatmap?field=0&spw=0&scan=99&c=XX").status_code == 404
    assert client.get("/heatmap?field=0&spw=0&scan=1&c=XX&scale=nope").status_code == 400

    # Bad indices are client errors, never a 500 and never a silent wrap to the end.
    assert client.get("/heatmap?field=0&spw=0&scan=1&c=ZZ").status_code == 404
    assert client.get("/histogram.png?field=0&spw=0&scan=1&c=XX&scale=nope").status_code == 400
    waterfall = "/waterfall?field=0&spw=0&scan=1&c=XX"
    assert client.get(waterfall + "&p=99&q=1").status_code == 404
    assert client.get(waterfall + "&p=-1&q=1").status_code == 422
    assert client.get(waterfall.replace("/waterfall?", "/waterfall.png?") + "&p=1&q=99").status_code == 404

    # The heatmap loads once; plots are fetched by app.js, so failures are read from the one response.
    assert index.text.count('hx-get="/heatmap"') == 1 and 'hx-trigger="load"' not in index.text
    assert 'data-plot="/histogram.png?' in heatmap.text and 'src="/histogram.png' not in heatmap.text
    # Cells share one sync scope: a new click aborts the last waterfall request.
    assert '<div class="grid" hx-sync="this:replace">' in heatmap.text


@needs_web
def test_changing_scan_keeps_product_and_scale(chi2_zarr):
    """The heatmap is a whole scan; the scan select re-renders the selectors, keeping product and scale."""
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web.app import create_app

    client = TestClient(create_app(chi2_zarr["zarr"]))
    index = client.get("/").text
    assert '<select name="scan"\n            hx-get="/scans"' in index
    assert 'name="t"' not in index and 'name="f"' not in index

    page = client.get("/scans?field=0&spw=0&scan=2&c=YY&scale=robust").text
    assert '<option value="2" selected>2</option>' in page
    assert '<option value="YY" selected>' in page and '<option value="robust" selected>' in page
    # A product the scan does not have falls back to the first.
    assert '<option value="XX" selected>' in client.get("/scans?field=0&spw=0&scan=2&c=V").text
    assert client.get("/scans?field=7&spw=0").status_code == 404


@needs_web
def test_only_offered_gain_tables_are_opened(chi2_zarr, gain_table, tmp_path):
    """The query string cannot name an arbitrary path; the server has no authentication."""
    pytest.importorskip("httpx2")
    import shutil
    from pathlib import Path

    from fastapi.testclient import TestClient

    from surfvis.web.app import create_app

    query = "field=0&spw=0&scan=1&c=XX&p=1&q=5&column=RESIDUAL"

    # chi2_zarr was made without gains, and gain_table is not next to its MS.
    client = TestClient(create_app(chi2_zarr["zarr"]))
    for path in (gain_table, "/etc", "/nowhere/NET"):
        panel = client.get(f"/waterfall?{query}&gtable={path}").text
        assert "not one this server offers" in panel and "waterfall.bin?" not in panel
        assert "holds no" not in panel and "No gain term" not in panel
    assert client.get(f"/waterfall.png?{query}&gtable={gain_table}").status_code == 403
    assert client.get(f"/waterfall.bin?{query}&gtable={gain_table}").status_code == 403

    # Passed to serve --gtable: offered, and applied.
    given = TestClient(create_app(chi2_zarr["zarr"], gtables=[gain_table]))
    panel = given.get(f"/waterfall?{query}").text
    assert f'<option value="{gain_table}"' in panel
    assert given.get(f"/waterfall.png?{query}&gtable={gain_table}").status_code == 200

    # Found in a *.qc directory next to the MS: offered without being named.
    ms = tmp_path / "moved.ms"
    shutil.copytree(chi2_zarr["ms"]["path"], ms)
    shutil.copytree(Path(gain_table).parent, tmp_path / "solve.qc")
    nearby = TestClient(create_app(chi2_zarr["zarr"], ms=ms))
    assert f'<option value="{tmp_path}/solve.qc/NET"' in nearby.get(f"/waterfall?{query}").text
    assert nearby.get(f"/waterfall.png?{query}&gtable={tmp_path}/solve.qc/NET").status_code == 200


@needs_web
def test_reload_builds_the_app_from_the_environment(chi2_zarr, monkeypatch):
    """serve --reload hands uvicorn an import string, so the options travel in the environment."""
    from surfvis.web import app as web_app

    monkeypatch.setenv(web_app.ENV_DATA, str(chi2_zarr["zarr"]))
    monkeypatch.setenv(web_app.ENV_MS, "")
    monkeypatch.setenv(web_app.ENV_GTABLES, "")
    assert web_app.app_from_env().title == "surfvis"


@needs_web
def test_a_matrix_does_not_cache_the_group(chi2_zarr):
    """Reading one product must not pull, or keep, every product (or an older store's every bin)."""
    from surfvis.web.store import Chi2Store, Selection

    store = Chi2Store(chi2_zarr["zarr"])
    store.matrix(Selection(0, 0, 1, "XX"))
    ds = store.group(0, 0, 1)
    assert type(ds.chi2.variable._data).__name__ != "MemoryCachedArray"
    assert not isinstance(ds.chi2.variable._data, np.ndarray)


@needs_web
def test_waterfall_sees_flags_written_after_the_first_read(chi2_zarr, tmp_path):
    """A long-running server must not pin the flags it read first.

    xarray-ms caches its tables in a process-global multiton whose TTL resets
    on every access, and a real MS keeps FLAG in a TiledShapeStMan whose tile
    cache belongs to that table object. Together they mean ``surfvis serve``
    renders the flags as they were when it opened the partition, for the life
    of the process, however often the MS is re-flagged underneath it.
    """
    import subprocess
    import sys

    from surfvis.web import msdata
    from tests.fixtures.ms import copy_with_tiled_flags

    ms = copy_with_tiled_flags(chi2_zarr["ms"]["path"], str(tmp_path / "reflagged.ms"))
    names = chi2_zarr["ms"]["antenna_names"]

    before = msdata.waterfall(ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")
    assert not before.flags[:, 7].any(), "fixture channel is unflagged to start with"

    # The flagger is another program -- skarabina, CASA -- so the write has to
    # come from another process. Writing through this process's own casacore
    # handle would invalidate its table cache and hide the bug.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from casacore.tables import table; "
            "t = table(sys.argv[1], readonly=False, ack=False); "
            "f = t.getcol('FLAG'); f[:, 7, :] = True; t.putcol('FLAG', f); t.close()",
            ms,
        ],
        check=True,
    )

    after = msdata.waterfall(ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")
    assert after.flags[:, 7].all(), "waterfall is still showing the flags from the first read"


@needs_ms
def test_each_correlation_gets_its_own_chi2(chi2_zarr):
    """Each product's slice is computed from that product alone.

    surfchi2 used to hand every correlation's task all of --use-corrs, so the
    per-correlation images were identical and the pol selector did nothing.
    """
    from surfvis.web.store import Chi2Store, Selection

    store = Chi2Store(chi2_zarr["zarr"])
    xx = store.matrix(Selection(0, 0, 1, "XX")).chi2_dof
    yy = store.matrix(Selection(0, 0, 1, "YY")).chi2_dof
    assert not np.allclose(xx, yy, equal_nan=True)


@needs_ms
def test_residual_formed_on_the_fly_matches_the_residual_column(chi2_zarr, chi2_corrected):
    """inv(G) DATA inv(G)^H - MODEL, with propagated weights, is the RESIDUAL chi-squared."""
    from surfvis.web.store import Chi2Store, Selection

    plain = Chi2Store(chi2_zarr["zarr"])
    formed = Chi2Store(chi2_corrected["zarr"])
    for product in ("XX", "YY"):
        want = plain.matrix(Selection(0, 0, 2, product)).chi2_dof
        got = formed.matrix(Selection(0, 0, 2, product)).chi2_dof
        np.testing.assert_allclose(got, want, rtol=1e-3, equal_nan=True)

    attrs = formed.group(0, 0, 1).attrs
    assert (attrs["rcol"], attrs["mcol"], attrs["gtable"]) == ("DATA", "MODEL_DATA", chi2_corrected["gtable"])


@needs_ms
def test_stokes_chi2_is_normalised_and_finds_the_bad_baseline(chi2_corrected):
    """Stokes weights are exact inverse variances, so chi2/dof matches the correlations'."""
    from surfvis.web.store import Chi2Store, Selection

    store = Chi2Store(chi2_corrected["zarr"])
    assert store.products(0, 0, 1) == ["XX", "YY", "I", "V"]

    xx = store.matrix(Selection(0, 0, 1, "XX")).chi2_dof
    for product in ("I", "V"):
        matrix = store.matrix(Selection(0, 0, 1, product)).chi2_dof
        p, q = np.unravel_index(np.nanargmax(matrix), matrix.shape)
        assert {int(p), int(q)} == set(chi2_corrected["ms"]["hot"])
        # Complex unit-variance noise gives chi2/dof ~ 2 for XX and for Stokes alike.
        assert 0.75 < np.nanmedian(matrix) / np.nanmedian(xx) < 1.33
    assert (chi2_corrected["imgs"] / "field0" / "spw0" / "scan1" / "combined_V.png").exists()


@needs_ms
def test_chi2_refuses_stokes_it_cannot_form(tmp_path):
    """Stokes V of linear feeds needs XY and YX; an XX,YY-only MS must fail before any work."""
    from surfvis.core.chi2 import chi2
    from tests.fixtures.ms import make_ms

    info = make_ms(str(tmp_path / "dual.ms"), nant=4, nchan=4, ncorr=2, ntime=2, nscan=1)
    # Outputs of an earlier run must survive a run that fails validation.
    earlier = [tmp_path / "c.zarr" / "field0" / "kept", tmp_path / "i" / "field0" / "kept"]
    for path in earlier:
        path.parent.mkdir(parents=True)
        path.write_text("from an earlier run")
    with pytest.raises(ValueError, match="Cannot form"):
        chi2(info["path"], dataout=tmp_path / "c.zarr", imagesout=tmp_path / "i", nthreads=1, stokes=["V"])
    assert all(path.exists() for path in earlier)


@needs_ms
@pytest.mark.parametrize("which", ["dataout", "imagesout"])
def test_chi2_refuses_to_delete_what_it_did_not_write(tmp_path, which):
    """Outputs are deleted and recreated, so a mistyped path must be refused, not removed."""
    from surfvis.core.chi2 import chi2
    from tests.fixtures.ms import make_ms

    info = make_ms(str(tmp_path / "small.ms"), nant=4, nchan=4, ncorr=2, ntime=2, nscan=1)
    precious = tmp_path / "work" / "notes.txt"
    precious.parent.mkdir()
    precious.write_text("keep me")
    outputs = {"dataout": tmp_path / "c.zarr", "imagesout": tmp_path / "i", which: precious.parent}
    with pytest.raises(ValueError, match="does not look like an earlier surfvis chi2 output"):
        chi2(info["path"], nthreads=1, **outputs)
    assert precious.read_text() == "keep me"
    # An MS given by mistake is refused too.
    with pytest.raises(ValueError, match="refusing"):
        chi2(info["path"], nthreads=1, **{**outputs, which: tmp_path / "small.ms"})


@needs_web
def test_corrected_data_is_its_own_column(chi2_zarr):
    """CORRECTED_DATA used to resolve to VISIBILITY, silently plotting DATA."""
    from surfvis.web import msdata

    ms = chi2_zarr["ms"]["path"]
    names = chi2_zarr["ms"]["antenna_names"]
    data = msdata.waterfall(ms, 0, 0, 1, names[0], names[1], "XX", "DATA")
    corrected = msdata.waterfall(ms, 0, 0, 1, names[0], names[1], "XX", "CORRECTED_DATA")
    assert corrected.column == "CORRECTED_DATA"
    assert not np.allclose(data.values, corrected.values)
    assert "CORRECTED_DATA" in msdata.available_columns(ms, 0, 0, 1)


@needs_web
@pytest.mark.parametrize("quantity", ["real", "imag"])
def test_waterfall_forms_the_corrected_residual(chi2_zarr, gain_table, quantity):
    """DATA corrected by the gains, minus MODEL_DATA, is RESIDUAL in the corrected frame."""
    from surfvis.web import msdata

    info = chi2_zarr["ms"]
    names = info["antenna_names"]
    p, q = 2, 6
    formed = msdata.waterfall(
        info["path"], 0, 0, 2, names[p], names[q], "XX", "DATA", quantity, model="MODEL_DATA", gtable=gain_table
    )
    raw = msdata.waterfall(info["path"], 0, 0, 2, names[p], names[q], "XX", "RESIDUAL", "real")
    raw_i = msdata.waterfall(info["path"], 0, 0, 2, names[p], names[q], "XX", "RESIDUAL", "imag")
    g = info["gains"][info["ntime"] :]
    want = (raw.values + 1j * raw_i.values) / (g[:, :, p, 0] * np.conj(g[:, :, q, 0]))
    want = want.real if quantity == "real" else want.imag
    np.testing.assert_allclose(formed.values, want, atol=1e-4)
    assert formed.column == "corrected VISIBILITY - MODEL_DATA"


@needs_web
def test_gains_follow_the_stored_order_not_xarray_ms_labels(chi2_zarr, gain_table, tmp_path):
    """xarray-ms labels a baseline stored as (6, 2) as (2, 6) without conjugating it.

    The data are V_62, so the correction is inv(g_6) V inv(g_2)^H, whichever
    order the cell asked for and whatever xarray-ms calls it.
    """
    import shutil

    from casacore.tables import table

    from surfvis.web import msdata

    info = chi2_zarr["ms"]
    names = info["antenna_names"]
    ms = str(tmp_path / "reversed.ms")
    shutil.copytree(info["path"], ms)
    with table(ms, readonly=False, ack=False) as t:
        a1, a2 = t.getcol("ANTENNA1"), t.getcol("ANTENNA2")
        bl = np.flatnonzero((a1 == 2) & (a2 == 6))
        a1[bl], a2[bl] = 6, 2
        t.putcol("ANTENNA1", a1)
        t.putcol("ANTENNA2", a2)
        scan = t.getcol("SCAN_NUMBER")[bl]
        data = t.getcol("DATA")[bl][scan == 2][..., 0]
        model = t.getcol("MODEL_DATA")[bl][scan == 2][..., 0]

    g = info["gains"][info["ntime"] :]
    want = data / (g[:, :, 6, 0] * np.conj(g[:, :, 2, 0])) - model
    for request in ((names[2], names[6]), (names[6], names[2])):
        parts = [
            msdata.waterfall(ms, 0, 0, 2, *request, "XX", "DATA", q, model="MODEL_DATA", gtable=gain_table)
            for q in ("real", "imag")
        ]
        np.testing.assert_allclose(parts[0].values + 1j * parts[1].values, want, atol=1e-4)
        assert (parts[0].antenna1, parts[0].antenna2) == (names[6], names[2])


@needs_web
def test_stokes_waterfall_combines_correlations(chi2_zarr):
    from surfvis.web import msdata

    ms = chi2_zarr["ms"]["path"]
    names = chi2_zarr["ms"]["antenna_names"]
    assert msdata.available_products(ms, 0, 0, 1) == ["XX", "XY", "YX", "YY", "I", "Q", "U", "V"]

    def wf(product):
        return msdata.waterfall(ms, 0, 0, 1, names[0], names[1], product, "RESIDUAL", "real").values

    np.testing.assert_allclose(wf("I"), 0.5 * (wf("XX") + wf("YY")), atol=1e-5)
    np.testing.assert_allclose(wf("Q"), 0.5 * (wf("XX") - wf("YY")), atol=1e-5)


@needs_web
def test_gains_off_the_data_grid_are_refused(chi2_zarr, gain_table, tmp_path):
    """On-grid only: a solution interval coarser than the data is an error, not a guess."""
    import xarray as xr

    from surfvis.web import msdata

    coarse = tmp_path / "coarse.qc" / "NET"
    for n in range(2):
        ds = xr.open_zarr(f"{gain_table}/NET_{n}", chunks=None).load()
        ds.isel(gain_time=slice(0, None, 2)).to_zarr(coarse / f"NET_{n}", mode="w")

    names = chi2_zarr["ms"]["antenna_names"]
    with pytest.raises(ValueError, match="on-grid"):
        msdata.waterfall(chi2_zarr["ms"]["path"], 0, 0, 1, names[0], names[1], "XX", "RESIDUAL", gtable=str(coarse))
    problems = msdata.check(chi2_zarr["ms"]["path"], 0, 0, 1, "XX", "RESIDUAL", gtable=str(coarse))
    assert any("on-grid" in p for p in problems)


@needs_web
def test_flag_versions_are_listed_and_overlaid(chi2_zarr, tmp_path):
    """A saved CASA flag version replaces FLAG in the overlay, and the live column is untouched."""
    import shutil

    from casacore.tables import table

    from surfvis.web import msdata
    from tests.fixtures.ms import add_flag_version

    ms = str(tmp_path / "versioned.ms")
    shutil.copytree(chi2_zarr["ms"]["path"], ms)
    assert msdata.flag_versions(ms) == []

    t = table(ms, ack=False)
    flags = t.getcol("FLAG")
    t.close()
    flags[:, 3, :] = True
    add_flag_version(ms, "before_tricolour", flags, "flags autosave")
    assert msdata.flag_versions(ms) == ["before_tricolour"]

    names = chi2_zarr["ms"]["antenna_names"]
    live = msdata.waterfall(ms, 0, 0, 1, names[0], names[1], "XX", "RESIDUAL")
    saved = msdata.waterfall(ms, 0, 0, 1, names[0], names[1], "XX", "RESIDUAL", flagversion="before_tricolour")
    assert not live.flags[:, 3].any()
    assert saved.flags[:, 3].all()
    assert not saved.flags[:, 2].any()

    with pytest.raises(KeyError):
        msdata.waterfall(ms, 0, 0, 1, names[0], names[1], "XX", "RESIDUAL", flagversion="../../etc")


@needs_web
def test_waterfall_controls(chi2_corrected):
    """The panel defaults to how the heatmap was formed, and reports problems instead of a broken image."""
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web.app import create_app

    client = TestClient(create_app(chi2_corrected["zarr"]))
    base = "/waterfall?field=0&spw=0&scan=1&c=I&p=1&q=5"

    panel = client.get(base)
    assert panel.status_code == 200
    assert f'value="{chi2_corrected["gtable"]}"' in panel.text
    assert '<option value="MODEL_DATA" selected>' in panel.text
    # chi2 recorded DATA; the menu knows it as VISIBILITY and must not list both.
    assert '<option value="VISIBILITY" selected>' in panel.text
    assert 'value="DATA"' not in panel.text
    assert (
        'name="model"' in panel.text
        and '<option value="WEIGHT"' not in panel.text.split('name="model"')[1].split("</select>")[0]
    )
    assert "no longer shows" not in panel.text
    # The column is corrected and has a model subtracted, so WEIGHT is not offered.
    assert '<option value="WEIGHT"' not in panel.text
    assert 'data-src="/waterfall.bin?' in panel.text and "<canvas" in panel.text
    assert 'class="keep"' not in panel.text

    png = client.get(base.replace("/waterfall?", "/waterfall.png?"))
    assert png.headers["content-type"] == "image/png"

    changed = client.get(base + "&product=V&column=DATA&model=MODEL_DATA&gtable=" + chi2_corrected["gtable"])
    assert "no longer shows" in changed.text
    # Only the controls moved off the heatmap's setting are carried to the next baseline.
    assert '<select name="product" class="keep">' in changed.text
    assert '<select name="model">' in changed.text
    assert 'name="gtable" class="keep"' not in changed.text

    broken = client.get(base + "&gtable=/nowhere/NET")
    assert "gain table" in broken.text
    assert "waterfall.bin?" not in broken.text


@needs_web
def test_plane_cache_reuses_and_invalidates(chi2_zarr, monkeypatch):
    """Zooming re-slices a cached plane; new settings or an invalidate re-read the MS."""
    from surfvis.web import msdata

    ms = chi2_zarr["ms"]["path"]
    names = chi2_zarr["ms"]["antenna_names"]
    msdata.invalidate(ms)
    calls = []
    real = msdata.waterfall
    monkeypatch.setattr(msdata, "waterfall", lambda *a, **k: calls.append(1) or real(*a, **k))

    a = msdata.cached_plane(ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")
    b = msdata.cached_plane(ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")
    assert a is b and len(calls) == 1
    assert np.iscomplexobj(a.plane) and a.plane.shape == a.flags.shape == (12, 32)
    assert a.freqs.shape == (32,) and a.times.shape == (12,)

    msdata.cached_plane(ms, 0, 0, 1, names[0], names[2], "YY", "RESIDUAL")
    assert len(calls) == 2
    msdata.invalidate(ms)
    msdata.cached_plane(ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")
    assert len(calls) == 3


@needs_web
def test_waterfall_bin(chi2_zarr):
    """Overview, window, clamping, caps and errors of the canvas data route."""
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web import msdata
    from surfvis.web.app import create_app
    from surfvis.web.planes import decode

    client = TestClient(create_app(chi2_zarr["zarr"]))
    names = chi2_zarr["ms"]["antenna_names"]
    base = "/waterfall.bin?field=0&spw=0&scan=1&c=XX&p=1&q=5&column=RESIDUAL"

    r = client.get(base + "&meta=true")
    assert r.status_code == 200 and r.headers["content-type"] == "application/octet-stream"
    assert r.headers["cache-control"] == "no-store"
    head, values, flags = decode(r.content)
    assert (head["nt"], head["nc"], head["tstep"], head["cstep"]) == (12, 32, 1, 1)
    assert (head["ntime"], head["nchan"]) == (12, 32) and "rect" not in head
    assert len(head["times"]) == 12 and len(head["freqs"]) == 32
    assert head["colormap"] == "viridis" and head["vmin"] < head["vmax"]
    assert head["title"].startswith(f"{names[1]}-{names[5]}")
    wf = msdata.waterfall(chi2_zarr["ms"]["path"], 0, 0, 1, names[1], names[5], "XX", "RESIDUAL")
    ok = ~wf.flags
    np.testing.assert_allclose(values[ok], wf.values[ok], rtol=1e-5)

    small, _, _ = decode(client.get(base + "&max_t=4&max_c=5").content)
    assert (small["tstep"], small["cstep"], small["nt"], small["nc"]) == (3, 7, 4, 5)
    assert "times" not in small
    assert (small["vmin"], small["vmax"]) == (head["vmin"], head["vmax"])  # colours do not move on zoom

    win, wvalues, _ = decode(client.get(base + "&t0=2&t1=5&c0=10&c1=20").content)
    assert (win["nt"], win["nc"], win["t0"], win["c1"]) == (3, 10, 2, 20)
    np.testing.assert_array_equal(wvalues, values[2:5, 10:20])

    clamped, _, _ = decode(client.get(base + "&t0=-5&t1=999&max_t=100000").content)
    assert (clamped["t0"], clamped["t1"], clamped["nt"]) == (0, 12, 12)

    phase, _, _ = decode(client.get(base + "&quantity=phase").content)
    assert phase["colormap"] == "twilight" and phase["quantity"] == "phase"

    assert client.get(base + "&t0=5&t1=5").status_code == 400
    assert client.get(base + "&c0=40").status_code == 400
    assert client.get(base + "&quantity=nope").status_code == 400
    assert client.get(base.replace("RESIDUAL", "NOPE")).status_code == 404
    assert client.get(base.replace("p=1", "p=99")).status_code == 404
    assert client.get(base.replace("p=1", "p=-1")).status_code == 422
    # A lower-triangle cell draws the baseline the way the MS stores it.
    reverse, rvalues, _ = decode(client.get(base.replace("p=1&q=5", "p=5&q=1")).content)
    assert reverse["title"].startswith(f"{names[1]}-{names[5]}")
    np.testing.assert_array_equal(rvalues, values)


@needs_web
def test_waterfall_bin_reads_the_ms_once_across_heatmaps(chi2_zarr, monkeypatch):
    """The same baseline and settings from another heatmap product reuse the cached plane."""
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web import msdata
    from surfvis.web.app import create_app
    from surfvis.web.planes import decode

    msdata.invalidate(chi2_zarr["ms"]["path"])
    calls = []
    real = msdata.waterfall
    monkeypatch.setattr(msdata, "waterfall", lambda *a, **k: calls.append(1) or real(*a, **k))
    client = TestClient(create_app(chi2_zarr["zarr"]))
    base = "/waterfall.bin?field=0&spw=0&scan=1&p=1&q=5&product=XX&column=RESIDUAL"

    first, _, _ = decode(client.get(base + "&c=XX").content)
    second, _, _ = decode(client.get(base + "&c=YY&quantity=phase").content)
    assert (first["quantity"], second["quantity"]) == ("amp", "phase")
    assert len(calls) == 1
    cached = msdata.cached_plane(
        chi2_zarr["ms"]["path"], 0, 0, 1, *chi2_zarr["ms"]["antenna_names"][1:6:4], "XX", "RESIDUAL"
    )
    assert cached.plane.dtype == np.complex64 and cached.values is None
