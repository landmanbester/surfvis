"""Manual flagging writes: exact cells, the backup flag version, undo and the routes.

Needs a Measurement Set, and works on a copy so the session fixtures stay
pristine. Which cells a polygon covers is tested in ``test_rasterise.py``.
"""

import os
import shutil

import pytest

from tests.conftest import needs_web

np = pytest.importorskip("numpy")


def _rect(t0, t1, c0, c1):
    """The four-vertex polygon around slots t0..t1 and channels c0..c1, inclusive."""
    return [(t0 - 0.5, c0 - 0.5), (t0 - 0.5, c1 + 0.5), (t1 + 0.5, c1 + 0.5), (t1 + 0.5, c0 - 0.5)]


@pytest.fixture
def writable_ms(chi2_zarr, tmp_path):
    """A private copy of the fixture MS, free to be flagged."""
    path = str(tmp_path / "flagme.ms")
    shutil.copytree(chi2_zarr["ms"]["path"], path)
    return path


def _flags(ms: str) -> tuple[np.ndarray, np.ndarray]:
    from casacore.tables import table

    with table(ms, ack=False) as t:
        return t.getcol("FLAG"), t.getcol("FLAG_ROW")


def _wait_for_backup(client, timeout: float = 30.0) -> dict:
    """serve takes the backup in the background at startup; writes wait for it."""
    import time

    deadline = time.monotonic() + timeout
    while (state := client.get("/flag/state").json())["backup_state"] == "running":
        assert time.monotonic() < deadline, "backup did not finish"
        time.sleep(0.05)
    return state


def _baseline_rows(ms: str, a: int, b: int, scan: int) -> np.ndarray:
    from casacore.tables import table

    with table(ms, ack=False) as t:
        a1, a2, sc = t.getcol("ANTENNA1"), t.getcol("ANTENNA2"), t.getcol("SCAN_NUMBER")
    return np.flatnonzero((a1 == a) & (a2 == b) & (sc == scan))


@needs_web
def test_flag_writes_exactly_the_selected_cells(chi2_zarr, writable_ms):
    """Rows, channels and correlations outside the polygon are untouched."""
    from surfvis.web import msdata
    from surfvis.web.flagging import Flagger

    names = chi2_zarr["ms"]["antenna_names"]
    before, before_row = _flags(writable_ms)
    rows = _baseline_rows(writable_ms, 0, 2, scan=1)

    flagger = Flagger(writable_ms)
    out = flagger.apply(0, 0, 1, names[0], names[2], np.array(_rect(3, 5, 10, 14)), corr_idx=[0, 3])

    after, after_row = _flags(writable_ms)
    expected = before.copy()
    # The baseline's rows in scan 1 are its slots, in time order.
    sel = np.ix_(rows[3:6], np.arange(10, 15), [0, 3])
    expected[sel] = True
    np.testing.assert_array_equal(after, expected)
    np.testing.assert_array_equal(after_row, before_row | expected.all(axis=(1, 2)))
    assert out["cells"] == 3 * 5 * 2
    assert out["newly_flagged"] == int((~before[sel] & ~before_row[rows[3:6], None, None]).sum())

    # The browser sees the write at once.
    wf = msdata.waterfall(writable_ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")
    assert wf.flags[3:6, 10:15].all()


@needs_web
def test_first_write_saves_a_backup_version(chi2_zarr, writable_ms):
    """Before the first write, FLAG is saved as a CASA flag version; later writes reuse it."""
    from surfvis.web import msdata
    from surfvis.web.flagging import Flagger

    names = chi2_zarr["ms"]["antenna_names"]
    before, before_row = _flags(writable_ms)
    flagger = Flagger(writable_ms)
    first = flagger.apply(0, 0, 1, names[0], names[2], np.array(_rect(0, 1, 0, 3)), corr_idx=[0])
    second = flagger.apply(0, 0, 1, names[1], names[3], np.array(_rect(0, 1, 0, 3)), corr_idx=[0])

    assert first["backup"] == second["backup"] and first["backup"].startswith("surfvis_before_")
    assert first["backup"] in msdata.flag_versions(writable_ms)

    from casacore.tables import table

    with table(f"{writable_ms}.flagversions/flags.{first['backup']}", ack=False) as t:
        np.testing.assert_array_equal(t.getcol("FLAG"), before)
        np.testing.assert_array_equal(t.getcol("FLAG_ROW"), before_row)


@needs_web
def test_undo_restores_flags_in_reverse_order(chi2_zarr, writable_ms):
    from surfvis.web.flagging import Flagger

    names = chi2_zarr["ms"]["antenna_names"]
    original, original_row = _flags(writable_ms)
    flagger = Flagger(writable_ms)
    first = flagger.apply(0, 0, 1, names[0], names[2], np.array(_rect(0, 4, 0, 31)), corr_idx=[0, 1, 2, 3])
    middle, _ = _flags(writable_ms)
    second = flagger.apply(0, 0, 1, names[0], names[2], np.array(_rect(5, 8, 2, 6)), corr_idx=[0])
    assert second["id"] == first["id"] + 1

    assert flagger.undo().id == second["id"]
    np.testing.assert_array_equal(_flags(writable_ms)[0], middle)
    assert flagger.undo().id == first["id"]
    after, after_row = _flags(writable_ms)
    np.testing.assert_array_equal(after, original)
    np.testing.assert_array_equal(after_row, original_row)
    assert flagger.undo() is None


@needs_web
def test_dry_run_counts_without_writing(chi2_zarr, writable_ms):
    import os

    from surfvis.web.flagging import Flagger

    names = chi2_zarr["ms"]["antenna_names"]
    before, _ = _flags(writable_ms)
    flagger = Flagger(writable_ms)
    out = flagger.apply(0, 0, 1, names[0], names[2], np.array(_rect(0, 2, 0, 3)), corr_idx=[0], dry_run=True)
    assert out["id"] is None and out["cells"] == 3 * 4 and out["backup"] is None
    np.testing.assert_array_equal(_flags(writable_ms)[0], before)
    assert not os.path.exists(f"{writable_ms}.flagversions") or not any(
        name.startswith("flags.surfvis_") for name in os.listdir(f"{writable_ms}.flagversions")
    )
    assert flagger.edits == []


@needs_web
def test_flag_routes(chi2_zarr, writable_ms):
    """Read-only by default; with --allow-flagging the routes flag, report and undo."""
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web.app import create_app

    body = {"field": 0, "spw": 0, "scan": 1, "p": 0, "q": 2, "product": "I", "vertices": _rect(1, 2, 4, 9)}

    readonly = TestClient(create_app(chi2_zarr["zarr"], ms=writable_ms))
    assert readonly.post("/flag", json=body).status_code == 403
    assert readonly.get("/flag/state").json() == {"enabled": False}

    client = TestClient(create_app(chi2_zarr["zarr"], ms=writable_ms, allow_flagging=True))
    assert client.get("/healthz").json()["flagging"] is True
    _wait_for_backup(client)
    before, _ = _flags(writable_ms)

    # Stokes I is formed from XX and YY, so those are the correlations flagged.
    out = client.post("/flag", json={**body, "correlations": "product"}).json()
    assert out["cells"] == 2 * 6 * 2
    after, _ = _flags(writable_ms)
    rows = _baseline_rows(writable_ms, 0, 2, scan=1)
    changed = np.argwhere(after != before)
    assert set(changed[:, 0]) <= set(rows[1:3]) and set(changed[:, 2]) <= {0, 3}
    assert after[np.ix_(rows[1:3], np.arange(4, 10), [0, 3])].all()

    # All correlations is the default: one flagged correlation flags them all.
    allcorr = client.post("/flag", json={**body, "dry_run": True}).json()
    assert allcorr["cells"] == 2 * 6 * 4

    state = client.get("/flag/state").json()
    assert state["enabled"] and len(state["edits"]) == 1 and state["backup"]

    assert client.post("/flag/undo", json={"id": out["id"]}).json()["undone"] == out["id"]
    np.testing.assert_array_equal(_flags(writable_ms)[0], before)
    assert client.post("/flag/undo", json={"id": out["id"]}).status_code == 409

    assert client.post("/flag", json={**body, "vertices": [(0, 0), (1, 1)]}).status_code == 422
    # A negative antenna index must not wrap round to another baseline.
    assert client.post("/flag", json={**body, "p": -1}).status_code == 422
    assert client.post("/flag", json={**body, "p": 99}).status_code == 404
    assert client.post("/flag", json={**body, "vertices": _rect(50, 60, 0, 3)}).status_code == 400
    # Linear feeds cannot form RR.
    assert client.post("/flag", json={**body, "product": "RR", "correlations": "product"}).status_code == 400


@needs_web
def test_a_selection_drawn_over_a_saved_flag_version_is_refused(chi2_zarr, writable_ms):
    """The picture shows a version's flags; the write would build on FLAG. Refuse, even a dry run."""
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web.app import create_app

    client = TestClient(create_app(chi2_zarr["zarr"], ms=writable_ms, allow_flagging=True))
    _wait_for_backup(client)
    before, _ = _flags(writable_ms)
    body = {"field": 0, "spw": 0, "scan": 1, "p": 0, "q": 2, "product": "XX", "vertices": _rect(1, 2, 4, 9)}
    for dry_run in (True, False):
        r = client.post("/flag", json={**body, "flagversion": "older", "dry_run": dry_run})
        assert r.status_code == 409 and "FLAG (current)" in r.json()["detail"]
    np.testing.assert_array_equal(_flags(writable_ms)[0], before)


@needs_web
def test_product_correlations_follow_full_jones_gains(chi2_zarr, writable_ms, tmp_path):
    """Corrected XX with full-Jones gains is formed from all four correlations, so all four are flagged."""
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web.app import create_app
    from tests.fixtures.ms import write_gain_table

    full = write_gain_table(chi2_zarr["ms"], str(tmp_path / "full.qc"), full=True)
    client = TestClient(create_app(chi2_zarr["zarr"], ms=writable_ms, gtables=[full], allow_flagging=True))
    _wait_for_backup(client)
    body = {
        "field": 0,
        "spw": 0,
        "scan": 1,
        "p": 0,
        "q": 2,
        "product": "XX",
        "vertices": _rect(1, 2, 4, 9),
        "correlations": "product",
        "dry_run": True,
    }
    assert client.post("/flag", json=body).json()["cells"] == 2 * 6
    assert client.post("/flag", json={**body, "gtable": full}).json()["cells"] == 2 * 6 * 4
    # Only gain tables the server offers.
    assert client.post("/flag", json={**body, "gtable": "/etc"}).status_code == 403
    # A term that is offered but is not a gain table is a 400, not a 500, and writes nothing.
    import xarray as xr

    bad = tmp_path / "bad.qc" / "NET"
    xr.Dataset(attrs={"FIELD_ID": 0, "DATA_DESC_ID": 0, "SCAN_NUMBER": 1}).to_zarr(str(bad / "NET_0"), mode="w")
    lenient = TestClient(create_app(chi2_zarr["zarr"], ms=writable_ms, gtables=[str(bad)], allow_flagging=True))
    _wait_for_backup(lenient)
    before, _ = _flags(writable_ms)
    r = lenient.post("/flag", json={**body, "gtable": str(bad), "dry_run": False})
    assert r.status_code == 400 and "not a gain term" in r.json()["detail"]
    np.testing.assert_array_equal(_flags(writable_ms)[0], before)


@needs_web
def test_undo_reverts_only_the_edit_it_names(chi2_zarr, writable_ms):
    """Another tab's newer edit is not undone by a button that named an older one."""
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web.app import create_app

    client = TestClient(create_app(chi2_zarr["zarr"], ms=writable_ms, allow_flagging=True))
    _wait_for_backup(client)
    body = {"field": 0, "spw": 0, "scan": 1, "p": 0, "q": 2, "product": "XX", "vertices": _rect(1, 2, 4, 9)}
    first = client.post("/flag", json=body).json()
    assert (
        first["description"]
        == f"{chi2_zarr['ms']['antenna_names'][0]}-{chi2_zarr['ms']['antenna_names'][2]} field 0 spw 0 scan 1"
    )
    assert (first["field"], first["spw"], first["scan"]) == (0, 0, 1)
    assert first["antennas"] == [chi2_zarr["ms"]["antenna_names"][0], chi2_zarr["ms"]["antenna_names"][2]]
    second = client.post("/flag", json={**body, "q": 3}).json()
    after_both, _ = _flags(writable_ms)

    stale = client.post("/flag/undo", json={"id": first["id"]})
    assert stale.status_code == 409 and str(second["id"]) in stale.json()["detail"]
    np.testing.assert_array_equal(_flags(writable_ms)[0], after_both)
    assert client.post("/flag/undo").status_code == 422  # the id is required

    assert client.post("/flag/undo", json={"id": second["id"]}).json()["undone"] == second["id"]
    state = client.get("/flag/state").json()
    assert [e["id"] for e in state["edits"]] == [first["id"]]
    assert state["edits"][0]["antennas"] == first["antennas"] and state["edits"][0]["scan"] == 1

    # A button left over from before a restart cannot name an edit made after it.
    restarted = TestClient(create_app(chi2_zarr["zarr"], ms=writable_ms, allow_flagging=True))
    _wait_for_backup(restarted)
    fresh = restarted.post("/flag", json={**body, "q": 4}).json()
    assert fresh["id"] not in (first["id"], second["id"])
    assert restarted.post("/flag/undo", json={"id": first["id"]}).status_code == 409


@needs_web
def test_an_unset_gtable_means_the_one_chi2_used(chi2_zarr, tmp_path):
    """As on /waterfall: a request that leaves gtable out gets the heatmap's term, not none."""
    pytest.importorskip("httpx2")
    import shutil

    from fastapi.testclient import TestClient

    from surfvis.core.chi2 import chi2
    from surfvis.web.app import create_app
    from tests.fixtures.ms import write_gain_table

    ms = str(tmp_path / "fj.ms")
    shutil.copytree(chi2_zarr["ms"]["path"], ms)
    full = write_gain_table(chi2_zarr["ms"], str(tmp_path / "full.qc"), full=True)
    out = tmp_path / "chi2fj"
    chi2(ms, rcol="DATA", mcol="MODEL_DATA", gtable=full, dataout=out / "chi2.zarr", imagesout=out / "imgs", nthreads=1)
    client = TestClient(create_app(out / "chi2.zarr", ms=ms, allow_flagging=True))
    _wait_for_backup(client)
    body = {
        "field": 0,
        "spw": 0,
        "scan": 1,
        "p": 0,
        "q": 2,
        "product": "XX",
        "vertices": _rect(1, 2, 4, 9),
        "correlations": "product",
        "dry_run": True,
    }
    assert client.post("/flag", json=body).json()["cells"] == 2 * 6 * 4
    assert client.post("/flag", json={**body, "gtable": ""}).json()["cells"] == 2 * 6


@needs_web
def test_backup_of_a_tiled_flag_column(chi2_zarr, tmp_path):
    """A real MS keeps FLAG in a TiledShapeStMan; the backup must copy it, shape intact."""
    from casacore.tables import table

    from surfvis.web.flagging import save_flag_version
    from tests.fixtures.ms import copy_with_tiled_flags

    ms = copy_with_tiled_flags(chi2_zarr["ms"]["path"], str(tmp_path / "tiled.ms"))
    with table(ms, ack=False) as t:
        assert t.getdminfo("FLAG")["TYPE"] == "TiledShapeStMan"
        before = t.getcol("FLAG")
    save_flag_version(ms, "probe", "a probe")
    with table(f"{ms}.flagversions/flags.probe", ack=False) as v:
        np.testing.assert_array_equal(v.getcol("FLAG"), before)
    with pytest.raises(FileExistsError):
        save_flag_version(ms, "probe", "again")


@needs_web
def test_cached_plane_sees_a_flag_write(chi2_zarr, writable_ms):
    """A plane cached before a flag write must not be served after it."""
    from surfvis.web import msdata
    from surfvis.web.flagging import Flagger

    names = chi2_zarr["ms"]["antenna_names"]
    before = msdata.cached_plane(writable_ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")
    assert not before.flags[2:4, 5:9].all()
    Flagger(writable_ms).apply(0, 0, 1, names[0], names[2], np.array(_rect(2, 3, 5, 8)), corr_idx=[0])
    after = msdata.cached_plane(writable_ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL")
    assert after.flags[2:4, 5:9].all()


@needs_web
def test_a_flag_write_reuses_the_open_scan_and_the_cached_plane(chi2_zarr, writable_ms, monkeypatch):
    """Flagging, undoing and switching flag version re-read FLAG only.

    Each used to clear every cache, and the next waterfall rebuilt the
    partition: 25 s per scan on a MeerKAT MS, after every edit.
    """
    from surfvis.web import msdata
    from surfvis.web.flagging import Flagger, save_flag_version

    names = chi2_zarr["ms"]["antenna_names"]
    formed = []
    real = msdata._formed
    monkeypatch.setattr(msdata, "_formed", lambda *a, **k: formed.append(1) or real(*a, **k))

    def plane(**kw):
        return msdata.cached_plane(writable_ms, 0, 0, 1, names[0], names[2], "XX", "RESIDUAL", **kw)

    before = plane()
    opened = msdata._open_partition.cache_info().misses
    flagger = Flagger(writable_ms)
    flagger.apply(0, 0, 1, names[0], names[2], np.array(_rect(2, 3, 5, 8)), corr_idx=[0])
    flagged = plane()
    assert flagged.flags[2:4, 5:9].all() and flagged.plane is before.plane
    # Colours hold still across an edit, so the edit is what changes on screen.
    assert flagged.limits is before.limits
    flagger.undo()
    np.testing.assert_array_equal(plane().flags, before.flags)

    save_flag_version(writable_ms, "edited", "a probe")
    version = plane(flagversion="edited")
    np.testing.assert_array_equal(version.flags, before.flags)
    assert version.plane is before.plane and version.limits is not before.limits

    assert msdata._open_partition.cache_info().misses == opened
    assert len(formed) == 1


@needs_web
def test_flag_row_is_only_ever_set(chi2_zarr, writable_ms):
    """A row flagged through FLAG_ROW alone stays flagged when part of it is flagged again."""
    from casacore.tables import table

    from surfvis.web.flagging import Flagger

    names = chi2_zarr["ms"]["antenna_names"]
    rows = _baseline_rows(writable_ms, 0, 2, scan=1)
    with table(writable_ms, readonly=False, ack=False) as t:
        flag_row = t.getcol("FLAG_ROW")
        flag_row[rows[2]] = True
        t.putcol("FLAG_ROW", flag_row)

    flagger = Flagger(writable_ms)
    flagger.apply(0, 0, 1, names[0], names[2], np.array(_rect(2, 3, 0, 3)), corr_idx=[0])
    after, after_row = _flags(writable_ms)
    assert after_row[rows[2]], "FLAG_ROW was cleared"
    np.testing.assert_array_equal(after_row, flag_row)

    flagger.undo()
    np.testing.assert_array_equal(_flags(writable_ms)[1], flag_row)


@needs_web
def test_undo_keeps_flags_written_since(chi2_zarr, writable_ms):
    """Undo clears the cells the edit set, not what flag-chi2, CASA or anyone else wrote afterwards."""
    from casacore.tables import table

    from surfvis.web.flagging import Flagger

    names = chi2_zarr["ms"]["antenna_names"]
    rows = _baseline_rows(writable_ms, 0, 2, scan=1)
    original, _ = _flags(writable_ms)
    flagger = Flagger(writable_ms)
    out = flagger.apply(0, 0, 1, names[0], names[2], np.array(_rect(0, 4, 0, 9)), corr_idx=[0, 3])
    # Only the cells it changed are kept for undo, not whole rows.
    assert sum(i.size for i in flagger.edits[-1].changed) == 3 * out["newly_flagged"]

    with table(writable_ms, readonly=False, ack=False) as t:
        flag = t.getcol("FLAG")
        flag[rows[1], 20:30, :] = True  # another writer, same rows, other channels
        flag[rows[2], 5, 0] = True  # and a cell the edit already set
        t.putcol("FLAG", flag)

    flagger.undo()
    after, _ = _flags(writable_ms)
    expected = original.copy()
    expected[rows[1], 20:30, :] = True
    np.testing.assert_array_equal(after, expected)


@needs_web
def test_a_failed_write_is_put_back(chi2_zarr, writable_ms, monkeypatch):
    """A write that raises after FLAG went to disk is rolled back, and flagging carries on."""
    import arcae

    from surfvis.web import flagging
    from surfvis.web.flagging import Flagger

    names = chi2_zarr["ms"]["antenna_names"]
    before, before_row = _flags(writable_ms)
    real = arcae.table

    class Failing:
        failed = False

        def __init__(self, t):
            self._t = t

        def __getattr__(self, name):
            return getattr(self._t, name)

        def putcol(self, column, *args, **kwargs):
            out = self._t.putcol(column, *args, **kwargs)
            if column == "FLAG" and not Failing.failed:
                Failing.failed = True
                raise OSError("disk full")
            return out

    flagger = _ready(Flagger(writable_ms))
    monkeypatch.setattr(
        arcae, "table", lambda *a, **k: Failing(real(*a, **k)) if k.get("readonly") is False else real(*a, **k)
    )
    with pytest.raises(OSError, match="disk full"):
        flagger.apply(0, 0, 1, names[0], names[2], np.array(_rect(1, 2, 0, 3)), corr_idx=[0])
    monkeypatch.setattr(arcae, "table", real)

    after, after_row = _flags(writable_ms)
    np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(after_row, before_row)
    assert flagger.edits == [] and flagger.broken is None
    assert flagging.journal_path(writable_ms) is None
    flagger.apply(0, 0, 1, names[0], names[2], np.array(_rect(1, 2, 0, 3)), corr_idx=[0])
    assert _flags(writable_ms)[0][_baseline_rows(writable_ms, 0, 2, scan=1)[1:3], 0:4, 0].all()


def _ready(flagger):
    """A Flagger whose backup counts as taken, for tests about writes."""
    flagger.backup = "pretend"
    flagger._backup_done.set()
    return flagger


# Run in a child process: flag through Flagger, but die inside the FLAG write
# the way a kill -9 lands between casacore truncating the tiled storage
# manager's header and writing it again -- the header left empty, nothing
# closed or fsynced.
_KILLED_MID_WRITE = """
import os, sys
import numpy as np
import arcae
from surfvis.web import flagging
from surfvis.web.flagging import Flagger

ms, header, p, q = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
real = arcae.table


class Dies:
    def __init__(self, t):
        self._t = t

    def __getattr__(self, name):
        return getattr(self._t, name)

    def putcol(self, column, *args, **kwargs):
        self._t.putcol(column, *args, **kwargs)
        if column == "FLAG":
            os.truncate(os.path.join(ms, header), 0)
            os._exit(9)


arcae.table = lambda *a, **k: Dies(real(*a, **k)) if k.get("readonly") is False else real(*a, **k)
f = Flagger(ms)
f.backup = "pretend"
f._backup_done.set()
rect = [(0.5, -0.5), (0.5, 9.5), (3.5, 9.5), (3.5, -0.5)]  # slots 1-3, channels 0-9
f.apply(0, 0, 1, p, q, np.array(rect), corr_idx=[0, 1, 2, 3])
"""


@needs_web
def test_a_write_killed_half_way_cannot_brick_the_ms(chi2_zarr, tmp_path):
    """serve --allow-flagging after a kill mid-write: the MS opens again and the edit is rolled back.

    casacore truncates and rewrites a TiledShapeStMan's header on every flush
    and fsyncs neither it nor the tiles. Killed in between, the MS would not
    open at all; the journal holds a copy of the header and the edit.
    """
    import subprocess
    import sys
    from pathlib import Path

    pytest.importorskip("httpx2")
    from casacore.tables import table
    from fastapi.testclient import TestClient

    from surfvis.web import flagging
    from surfvis.web.app import create_app
    from tests.fixtures.ms import copy_with_tiled_flags

    ms = copy_with_tiled_flags(chi2_zarr["ms"]["path"], str(tmp_path / "killed.ms"))
    with table(ms, ack=False) as t:
        header = f"table.f{t.getdminfo('FLAG')['SEQNR']}"
    assert os.path.exists(os.path.join(ms, header + "_TSM1")), "FLAG is in a tiled storage manager"
    before, before_row = _flags(ms)
    names = chi2_zarr["ms"]["antenna_names"]

    root = str(Path(__file__).resolve().parents[1])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(Path(root) / "src"), root, os.environ.get("PYTHONPATH", "")]),
    }
    child = subprocess.run([sys.executable, "-c", _KILLED_MID_WRITE, ms, header, names[0], names[2]], env=env)
    assert child.returncode == 9
    assert os.path.getsize(os.path.join(ms, header)) == 0
    assert not flagging._opens(ms), "the emulated kill should have bricked the MS"
    assert flagging.journal_path(ms) is not None

    readonly = TestClient(create_app(chi2_zarr["zarr"], ms=ms))
    assert readonly.get("/flag/state").json() == {"enabled": False}
    assert flagging.journal_path(ms) is not None, "recovery is only done with --allow-flagging"

    client = TestClient(create_app(chi2_zarr["zarr"], ms=ms, allow_flagging=True))
    state = _wait_for_backup(client)
    assert "rolled back" in state["recovered"] and header in state["recovered"]
    assert flagging._opens(ms) and flagging.journal_path(ms) is None
    after, after_row = _flags(ms)
    np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(after_row, before_row)
    # The backup was taken after the rollback, and flagging works.
    assert state["backup_state"] == "done" and state["broken"] is None
    body = {"field": 0, "spw": 0, "scan": 1, "p": 0, "q": 2, "product": "XX", "vertices": _rect(1, 2, 4, 9)}
    assert client.post("/flag", json=body).status_code == 200
    assert flagging.recover(ms) is None


@needs_web
def test_recovery_is_idempotent_and_leaves_newer_flags(chi2_zarr, writable_ms):
    """A journal is cleared by clearing its cells: rerunning it, or flags set elsewhere since, are safe."""
    from casacore.tables import table

    from surfvis.web import flagging
    from surfvis.web.flagging import Edit, _write_journal

    rows = _baseline_rows(writable_ms, 0, 2, scan=1)[:2]
    edit = Edit(
        id=1,
        rows=rows.astype(np.int64),
        corr_idx=[0],
        changed=(np.array([0, 1], np.int32), np.array([5, 5], np.int32), np.array([0, 0], np.int32)),
        row_set=np.array([], np.int64),
        cells=2,
        newly_flagged=2,
        description="by hand",
    )
    with table(writable_ms, readonly=False, ack=False) as t:
        flag = t.getcol("FLAG")
        flag[rows, 5, 0] = True  # what the interrupted edit wrote
        flag[rows, 6, 1] = True  # what someone else wrote
        t.putcol("FLAG", flag)
    for _ in range(2):
        _write_journal(writable_ms, edit, [])
        assert "by hand" in flagging.recover(writable_ms)
        after, _ = _flags(writable_ms)
        assert not after[rows, 5, 0].any() and after[rows, 6, 1].all()


@needs_web
def test_rows_out_of_order_missing_and_stored_reversed(chi2_zarr, tmp_path):
    """Rows not in time order, with gaps, for a baseline the MS stores as (2, 0): exactly the selected cells."""
    from casacore.tables import table

    from surfvis.web.flagging import Flagger

    names = chi2_zarr["ms"]["antenna_names"]
    ms = str(tmp_path / "shuffled.ms")
    with table(chi2_zarr["ms"]["path"], ack=False) as t:
        t.selectrows(np.random.default_rng(1).permutation(t.nrows())).copy(ms, deep=True).close()
    with table(ms, readonly=False, ack=False) as t:
        a1, a2, times = t.getcol("ANTENNA1"), t.getcol("ANTENNA2"), t.getcol("TIME")
        bl = np.flatnonzero((a1 == 0) & (a2 == 2))
        a1[bl], a2[bl] = 2, 0
        t.putcol("ANTENNA1", a1)
        t.putcol("ANTENNA2", a2)
        grid = np.unique(times)
        t.removerows(sorted(bl[np.isin(times[bl], grid[[5, 6]])].tolist()))

    before, _ = _flags(ms)
    # Asked for in either order, the stored (2, 0) baseline is the one flagged.
    out = Flagger(ms).apply(0, 0, 1, names[0], names[2], np.array(_rect(3, 8, 10, 12)), corr_idx=[1, 2])
    after, _ = _flags(ms)
    with table(ms, ack=False) as t:
        a1, a2, times = t.getcol("ANTENNA1"), t.getcol("ANTENNA2"), t.getcol("TIME")
    sel = np.flatnonzero((a1 == 2) & (a2 == 0) & np.isin(times, grid[[3, 4, 7, 8]]))
    expected = before.copy()
    expected[np.ix_(sel, np.arange(10, 13), [1, 2])] = True
    np.testing.assert_array_equal(after, expected)
    assert out["cells"] == 4 * 3 * 2


@needs_web
def test_a_cross_hand_on_a_reversed_baseline_lands_on_its_partner(chi2_zarr, tmp_path):
    """XY of the (0, 2) waterfall is YX of the rows stored as (2, 0): V_20 = V_02^H.

    The baseline is stored both ways round, at alternate times. Flagging XY
    must set XY on the forward rows and YX on the reversed ones, and both the
    live FLAG (converted by xarray-ms) and a saved flag version (converted
    here) must read back as XY.
    """
    from casacore.tables import table

    from surfvis.web import msdata
    from surfvis.web.flagging import Flagger
    from tests.fixtures.ms import add_flag_version

    names = chi2_zarr["ms"]["antenna_names"]
    ms = str(tmp_path / "mixed.ms")
    shutil.copytree(chi2_zarr["ms"]["path"], ms)
    with table(ms, readonly=False, ack=False) as t:
        a1, a2, times = t.getcol("ANTENNA1"), t.getcol("ANTENNA2"), t.getcol("TIME")
        grid = np.unique(times)
        bl = np.flatnonzero((a1 == 0) & (a2 == 2) & np.isin(times, grid[::2]))
        a1[bl], a2[bl] = 2, 0
        t.putcol("ANTENNA1", a1)
        t.putcol("ANTENNA2", a2)

    def flags(product, **kw):
        return msdata.waterfall(ms, 0, 0, 1, names[0], names[2], product, "RESIDUAL", **kw).flags

    xy_before, yx_before = flags("XY"), flags("YX")
    before, _ = _flags(ms)
    ntime = xy_before.shape[0]
    # Asked for as (2, 0); presented, and flagged, as (0, 2).
    Flagger(ms).apply(0, 0, 1, names[2], names[0], np.array(_rect(0, ntime - 1, 10, 12)), corr_idx=[1])

    after, _ = _flags(ms)
    with table(ms, ack=False) as t:
        a1, a2, scan = t.getcol("ANTENNA1"), t.getcol("ANTENNA2"), t.getcol("SCAN_NUMBER")
    forward = np.flatnonzero((a1 == 0) & (a2 == 2) & (scan == 1))
    reverse = np.flatnonzero((a1 == 2) & (a2 == 0) & (scan == 1))
    assert forward.size and reverse.size
    expected = before.copy()
    expected[np.ix_(forward, np.arange(10, 13), [1])] = True
    expected[np.ix_(reverse, np.arange(10, 13), [2])] = True
    np.testing.assert_array_equal(after, expected)

    want = xy_before.copy()
    want[:, 10:13] = True
    np.testing.assert_array_equal(flags("XY"), want)
    np.testing.assert_array_equal(flags("YX"), yx_before)
    add_flag_version(ms, "saved", after, "after the edit")
    np.testing.assert_array_equal(flags("XY", flagversion="saved"), want)
    np.testing.assert_array_equal(flags("YX", flagversion="saved"), yx_before)


@needs_web
def test_writes_wait_for_the_background_backup(chi2_zarr, writable_ms, monkeypatch):
    """serve starts the backup at startup; /flag answers 503 until it is written, then works."""
    pytest.importorskip("httpx2")
    import threading

    from fastapi.testclient import TestClient

    from surfvis.web import flagging
    from surfvis.web.app import create_app

    release = threading.Event()
    real = flagging.save_flag_version

    def slow(*args, **kwargs):
        release.wait(30)
        return real(*args, **kwargs)

    monkeypatch.setattr(flagging, "save_flag_version", slow)
    client = TestClient(create_app(chi2_zarr["zarr"], ms=writable_ms, allow_flagging=True))
    body = {"field": 0, "spw": 0, "scan": 1, "p": 0, "q": 2, "product": "XX", "vertices": _rect(1, 2, 4, 9)}

    assert client.get("/flag/state").json()["backup_state"] == "running"
    pending = client.post("/flag", json=body)
    assert pending.status_code == 503 and "backup" in pending.json()["detail"]
    # Counting does not need the backup.
    assert client.post("/flag", json={**body, "dry_run": True}).status_code == 200

    release.set()
    state = _wait_for_backup(client)
    assert state["backup_state"] == "done" and state["backup"].startswith("surfvis_before_")
    assert client.post("/flag", json=body).status_code == 200


@needs_web
def test_backups_from_two_sessions_do_not_collide(chi2_zarr, writable_ms):
    """Backup names carry microseconds: two servers, or a quick restart, used to clash and 500."""
    from surfvis.web import msdata
    from surfvis.web.flagging import Flagger

    names = chi2_zarr["ms"]["antenna_names"]
    backups = [
        Flagger(writable_ms).apply(0, 0, 1, names[0], names[2], np.array(_rect(0, 0, 0, 0)), corr_idx=[0])["backup"]
        for _ in range(2)
    ]
    assert backups[0] != backups[1]
    assert set(backups) <= set(msdata.flag_versions(writable_ms))
