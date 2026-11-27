"""Step 5 of manual flagging: recomputing stale heatmap cells in place.

A recomputed cell must be what a fresh ``chi2`` run would write for it, by
the zarr's own recipe (columns, model, gains, products, weights), so it can
sit among the cells chi2 wrote.
"""

import json
import re
import shutil
import time

import pytest

from tests.conftest import needs_web

np = pytest.importorskip("numpy")


def _rect(t0, t1, c0, c1):
    return [(t0 - 0.5, c0 - 0.5), (t0 - 0.5, c1 + 0.5), (t1 + 0.5, c1 + 0.5), (t1 + 0.5, c0 - 0.5)]


def _zarr_cells(zarr, field, spw, scan):
    from surfvis.web.store import _open_group

    _open_group.cache_clear()
    ds = _open_group(str(zarr), field, spw, scan)
    return ds.chi2.values, ds.counts.values, dict(ds.attrs), [str(p) for p in ds["product"].values]


def _all_pairs(nant):
    return [(p, q) for p in range(nant) for q in range(p + 1, nant)]


def _recomputed_matrix(ms, zarr, field, spw, scan):
    """Every cross cell recomputed by recompute.cells, as (nprod, nant, nant)."""
    from surfvis.web import recompute

    chi2, counts, attrs, products = _zarr_cells(zarr, field, spw, scan)
    nant = chi2.shape[-1]
    pairs = _all_pairs(nant)
    c2, cn = recompute.cells(ms, attrs, products, field, spw, scan, pairs)
    out2, outn = np.zeros_like(chi2), np.zeros_like(counts)
    for k, (p, q) in enumerate(pairs):
        out2[:, p, q] = out2[:, q, p] = c2[:, k]
        outn[:, p, q] = outn[:, q, p] = cn[:, k]
    return (out2, outn), (chi2, counts)


@needs_web
@pytest.mark.parametrize("fixture", ["chi2_zarr", "chi2_corrected"])
def test_recomputed_cells_are_what_chi2_wrote(fixture, request):
    """Plain residuals, and DATA - MODEL_DATA corrected by gains with Stokes I and V, chunked in time and freq."""
    made = request.getfixturevalue(fixture)
    for scan in (1, 2):
        (c2, cn), (z2, zn) = _recomputed_matrix(made["ms"]["path"], made["zarr"], 0, 0, scan)
        np.testing.assert_array_equal(cn, zn)
        np.testing.assert_allclose(c2, z2, rtol=1e-9, atol=1e-12)


@needs_web
def test_weights_sigma_and_reversed_rows_follow_chi2(tiny_ms, tmp_path):
    """Random weights, SIGMA_SPECTRUM as the weight column, a baseline stored the other way round, scattered flags."""
    from casacore.tables import makearrcoldesc, maketabdesc, table

    from surfvis.core.chi2 import chi2

    ms = str(tmp_path / "weights.ms")
    shutil.copytree(tiny_ms["path"], ms)
    rng = np.random.default_rng(3)
    with table(ms, readonly=False, ack=False) as t:
        shape = t.getcol("WEIGHT_SPECTRUM").shape
        t.putcol("WEIGHT_SPECTRUM", rng.uniform(0.2, 3.0, shape).astype(np.float32))
        if "SIGMA_SPECTRUM" not in t.colnames():
            t.addcols(maketabdesc(makearrcoldesc("SIGMA_SPECTRUM", 0.0, ndim=2, valuetype="float")))
        t.putcol("SIGMA_SPECTRUM", rng.uniform(0.5, 2.0, shape).astype(np.float32))
        flag = t.getcol("FLAG")
        flag |= rng.random(flag.shape) < 0.1
        t.putcol("FLAG", flag)
        a1, a2 = t.getcol("ANTENNA1"), t.getcol("ANTENNA2")
        flip = np.flatnonzero((a1 == 2) & (a2 == 6))[::2]
        a1[flip], a2[flip] = 6, 2
        t.putcol("ANTENNA1", a1)
        t.putcol("ANTENNA2", a2)
    for wcol in ("WEIGHT_SPECTRUM", "SIGMA_SPECTRUM"):
        out = tmp_path / wcol
        chi2(ms, rcol="CORRECTED_DATA", wcol=wcol, dataout=out / "z", imagesout=out / "i", nthreads=1, ntimes=5)
        (c2, cn), (z2, zn) = _recomputed_matrix(ms, out / "z", 0, 0, 1)
        np.testing.assert_array_equal(cn, zn)
        np.testing.assert_allclose(c2, z2, rtol=1e-9, atol=1e-12)


@needs_web
def test_stale_cells_compare_each_cells_time(tmp_path):
    """A cell is stale when an edit touching it is newer, or was undone after the cell was computed."""
    from surfvis.web import flagging

    ms = str(tmp_path / "x.ms")
    names = ["a", "b", "c", "d"]
    t = 1000.0

    def log(event, unix, **kw):
        entry = {"event": event, "unix": unix, "field": 0, "spw": 0, "scan": 1, **kw}
        flagging.append_log(ms, entry)

    log("apply", t + 1, id=1, scope="baseline", antennas=["a", "b"])
    log("apply", t + 2, id=2, scope="antenna", antennas=["c"])
    log("undo", t + 3, id=2)
    log("apply", t + 4, id=3, scope="all", antennas=[])
    log("rollback", t + 5, id=3)
    computed = np.full((4, 4), t)
    edits, stale = flagging.stale_cells(ms, 0, 0, 1, computed, names)
    assert edits == 1 and stale[0, 1] and stale[1, 0] and stale.sum() == 2  # 2 and 3 cancelled out

    computed[:] = t + 2.5  # after 1 and 2 were applied, before 2 was undone
    edits, stale = flagging.stale_cells(ms, 0, 0, 1, computed, names)
    off = np.arange(4) != 2
    assert edits == 1 and stale[2, off].all() and stale[off, 2].all() and not stale[2, 2] and not stale[0, 1]

    computed[0, 1] = computed[1, 0] = t  # one cell older than edit 1
    edits, stale = flagging.stale_cells(ms, 0, 0, 1, computed, names)
    assert edits == 2 and stale[0, 1]
    assert flagging.stale_cells(ms, 0, 0, 2, computed, names)[0] == 0


@pytest.fixture
def served(chi2_zarr, tmp_path):
    """A private MS and zarr, and a flagging server over them."""
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web.app import create_app
    from tests.test_flagging import _wait_for_backup

    ms = str(tmp_path / "flagme.ms")
    zarr = tmp_path / "chi2.zarr"
    shutil.copytree(chi2_zarr["ms"]["path"], ms)
    shutil.copytree(chi2_zarr["zarr"], zarr)
    client = TestClient(create_app(zarr, ms=ms, allow_flagging=True, warm=False))
    _wait_for_backup(client)
    return {"client": client, "ms": ms, "zarr": zarr, "tmp": tmp_path}


def _wait_for_job(client, timeout=60.0):
    deadline = time.monotonic() + timeout
    while (job := client.get("/recompute/state").json())["state"] == "running":
        assert time.monotonic() < deadline, "recompute did not finish"
        time.sleep(0.05)
    return job


@needs_web
def test_recompute_writes_what_a_fresh_chi2_would(served, chi2_zarr):
    from surfvis.core.chi2 import chi2

    client, ms, zarr = served["client"], served["ms"], served["zarr"]
    where = {"field": 0, "spw": 0, "scan": 1}
    q = "field=0&spw=0&scan=1"
    before2, beforen, before_attrs, _ = _zarr_cells(zarr, 0, 0, 1)
    body = {**where, "p": 1, "q": 3, "product": "XX", "vertices": _rect(1, 4, 2, 20)}
    assert client.post("/flag", json={**body, "baselines": "antenna", "antenna": 3}).status_code == 200
    stale = client.get(f"/stale?{q}").json()
    assert stale["edits"] == 1 and len(stale["cells"]) == 7 and stale["can_recompute"]
    assert all(3 in cell for cell in stale["cells"])
    assert "Recompute 7 stale cells" in client.get(f"/heatmap?{q}&c=XX").text

    job = client.post("/recompute", json=where).json()
    assert job["state"] == "running" and job["cells"] == 7
    assert _wait_for_job(client)["state"] == "done"
    stale = client.get(f"/stale?{q}").json()
    assert stale["edits"] == 0 and stale["cells"] == [] and stale["recomputed"] == 7
    html = client.get(f"/heatmap?{q}&c=XX").text
    assert "7 cells recomputed by serve since chi2" in html and 'id="stale-note" hidden' in html

    # The same numbers as chi2 run afresh on the flagged MS, with the zarr's settings.
    fresh = served["tmp"] / "fresh"
    chi2(ms, dataout=fresh / "z", imagesout=fresh / "i", nthreads=1, nfreqs=16, ntimes=6)
    after2, aftern, after_attrs, _ = _zarr_cells(zarr, 0, 0, 1)
    fresh2, freshn, _, _ = _zarr_cells(fresh / "z", 0, 0, 1)
    np.testing.assert_array_equal(aftern, freshn)
    np.testing.assert_allclose(after2, fresh2, rtol=1e-9)
    # Only antenna 3's cells changed; everything chi2 recorded is still there.
    changed = np.argwhere((after2 != before2).any(axis=0))
    assert len(changed) and all(3 in pq for pq in changed.tolist())
    for key in ("ms", "rcol", "created", "autocorrelations", "antenna_names", "corr_labels"):
        assert after_attrs[key] == before_attrs[key] or np.array_equal(after_attrs[key], before_attrs[key])

    # An undo after the recompute makes them stale again: the recomputed cells counted those flags.
    edit = client.get("/flag/state").json()["edits"][-1]
    client.post("/flag/undo", json={"id": edit["id"]})
    assert len(client.get(f"/stale?{q}").json()["cells"]) == 7
    client.post("/recompute", json=where)
    _wait_for_job(client)
    np.testing.assert_allclose(_zarr_cells(zarr, 0, 0, 1)[0], before2, rtol=1e-9)
    assert client.get(f"/stale?{q}").json()["recomputed"] == 14


@needs_web
def test_recompute_refusals(served, chi2_zarr):
    import zarr as zarr_
    from fastapi.testclient import TestClient

    from surfvis.web.app import create_app

    client, ms, zarr = served["client"], served["ms"], served["zarr"]
    where = {"field": 0, "spw": 0, "scan": 1}
    assert client.post("/recompute", json=where).status_code == 409  # nothing stale
    body = {**where, "p": 0, "q": 2, "product": "XX", "vertices": _rect(1, 2, 2, 4)}
    client.post("/flag", json=body)

    readonly = TestClient(create_app(zarr, ms=ms, warm=False))
    assert readonly.post("/recompute", json=where).status_code == 403
    stale = readonly.get("/stale?field=0&spw=0&scan=1").json()
    assert stale["edits"] == 1 and not stale["can_recompute"]
    html = readonly.get("/heatmap?field=0&spw=0&scan=1&c=XX").text
    assert re.search(r'id="recompute"[^>]*\shidden', html) and "re-run chi2" in html

    # An output that does not say how chi2 formed the residual cannot be recomputed.
    root = zarr_.open_group(str(zarr), mode="r+")
    del root["field0/spw0/scan1"].attrs["wcol"]
    zarr_.consolidate_metadata(str(zarr))
    from surfvis.web.store import _open_group

    _open_group.cache_clear()
    assert client.post("/recompute", json=where).status_code == 409
    assert "re-run chi2" in client.get("/heatmap?field=0&spw=0&scan=1&c=XX").text


@needs_web
def test_an_edit_made_while_recomputing_stays_stale(served):
    """The cells are stamped with the recompute's start, before it read FLAG."""
    from surfvis.web import flagging
    from surfvis.web.store import Chi2Store

    client, ms, zarr = served["client"], served["ms"], served["zarr"]
    store = Chi2Store(zarr)
    nprod = len(store.products(0, 0, 1))
    started = time.time()
    body = {"field": 0, "spw": 0, "scan": 1, "p": 0, "q": 2, "product": "XX", "vertices": _rect(1, 2, 2, 4)}
    client.post("/flag", json=body)  # after the recompute began reading
    store.write_cells(0, 0, 1, [(0, 2)], np.ones((nprod, 1)), np.ones((nprod, 1)), started)
    stale = client.get("/stale?field=0&spw=0&scan=1").json()
    assert stale["cells"] == [[0, 2]] and stale["recomputed"] == 1
    entries = [json.loads(line) for line in open(flagging.audit_path(ms))]
    assert entries[-1]["unix"] > started
