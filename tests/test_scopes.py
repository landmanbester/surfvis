"""Step 4 of manual flagging: wider scopes, the audit log and stale heatmaps.

A selection drawn on one baseline's waterfall can be applied to every
baseline of one of its antennas (auto-correlation included) or to every
baseline in the scan; each application is one edit, undone in one step,
and recorded in the audit log, from which the heatmap learns which cells
its chi-squared no longer describes.
"""

import json
import shutil

import pytest

from tests.conftest import needs_web

np = pytest.importorskip("numpy")


def _rect(t0, t1, c0, c1):
    """The four-vertex polygon around slots t0..t1 and channels c0..c1, inclusive."""
    return [(t0 - 0.5, c0 - 0.5), (t0 - 0.5, c1 + 0.5), (t1 + 0.5, c1 + 0.5), (t1 + 0.5, c0 - 0.5)]


@pytest.fixture(scope="module")
def scoped(tmp_path_factory):
    """An MS with auto-correlations and chi2 over it, for every test to copy."""
    from surfvis.core.chi2 import chi2
    from tests.fixtures.ms import make_ms

    out = tmp_path_factory.mktemp("scopes")
    ms = make_ms(str(out / "scopes.ms"), autos=True)
    chi2(
        ms["path"],
        rcol="CORRECTED_DATA",
        mcol="MODEL_DATA",
        dataout=out / "chi2.zarr",
        imagesout=out / "imgs",
        nthreads=1,
    )
    return {"zarr": out / "chi2.zarr", "ms": ms}


@pytest.fixture
def ms(scoped, tmp_path):
    path = str(tmp_path / "flagme.ms")
    shutil.copytree(scoped["ms"]["path"], path)
    return path


def _columns(ms):
    from casacore.tables import table

    with table(ms, ack=False) as t:
        return {c: t.getcol(c) for c in ("FLAG", "FLAG_ROW", "ANTENNA1", "ANTENNA2", "SCAN_NUMBER", "TIME")}


def _flagger(ms):
    from surfvis.web.flagging import Flagger

    flagger = Flagger(ms)
    flagger.backup = "pretend"
    flagger._backup_done.set()
    return flagger


def _expected(before, cols, rows_mask, slots, chans, corrs):
    """FLAG with the cells of ``rows_mask`` rows in ``slots`` x ``chans`` x ``corrs`` set."""
    grid = np.unique(cols["TIME"][cols["SCAN_NUMBER"] == 1])
    rows = np.flatnonzero(rows_mask & (cols["SCAN_NUMBER"] == 1) & np.isin(cols["TIME"], grid[slots]))
    expected = before.copy()
    expected[np.ix_(rows, chans, corrs)] = True
    return expected, rows


@needs_web
def test_an_antenna_scope_flags_every_baseline_of_it_and_its_auto(ms):
    cols = _columns(ms)
    names = [f"m{k:03d}" for k in range(8)]
    out = _flagger(ms).apply(
        0, 0, 1, names[2], names[4], np.array(_rect(3, 5, 10, 12)), [0, 3], scope="antenna", antenna=names[4]
    )
    after = _columns(ms)["FLAG"]
    involves = (cols["ANTENNA1"] == 4) | (cols["ANTENNA2"] == 4)
    expected, rows = _expected(cols["FLAG"], cols, involves, [3, 4, 5], np.arange(10, 13), [0, 3])
    np.testing.assert_array_equal(after, expected)
    assert out["baselines"] == 8  # seven crosses and the auto
    assert out["scope"] == "antenna" and out["scope_antennas"] == ["m004"]
    assert out["description"].startswith("every baseline of m004 ")
    assert out["cells"] == rows.size * 3 * 2
    assert ((cols["ANTENNA1"][rows] == 4) & (cols["ANTENNA2"][rows] == 4)).any(), "the auto is included"


@needs_web
def test_all_baselines_is_one_edit_and_one_undo(ms):
    cols = _columns(ms)
    flagger = _flagger(ms)
    out = flagger.apply(0, 0, 1, "m000", "m001", np.array(_rect(0, 1, 0, 31)), [0, 1, 2, 3], scope="all")
    after = _columns(ms)
    expected, rows = _expected(
        cols["FLAG"], cols, np.ones(cols["FLAG"].shape[0], bool), [0, 1], np.arange(32), [0, 1, 2, 3]
    )
    np.testing.assert_array_equal(after["FLAG"], expected)
    assert out["baselines"] == 8 * 7 // 2 + 8
    # Every channel of those rows is now flagged, so FLAG_ROW is set on them and nowhere else.
    np.testing.assert_array_equal(np.flatnonzero(after["FLAG_ROW"] & ~cols["FLAG_ROW"]), rows)

    assert flagger.undo(out["id"]).id == out["id"]
    restored = _columns(ms)
    np.testing.assert_array_equal(restored["FLAG"], cols["FLAG"])
    np.testing.assert_array_equal(restored["FLAG_ROW"], cols["FLAG_ROW"])
    assert flagger.undo() is None


@needs_web
def test_undo_leaves_flag_row_of_rows_it_did_not_change(ms):
    """A row fully flagged in FLAG but not FLAG_ROW, as many flaggers leave one, is left exactly as found."""
    from casacore.tables import table

    with table(ms, readonly=False, ack=False) as t:
        flag = t.getcol("FLAG")
        flag[13] = True
        t.putcol("FLAG", flag)
    cols = _columns(ms)
    assert not cols["FLAG_ROW"][13]
    grid = np.unique(cols["TIME"][cols["SCAN_NUMBER"] == 1])
    slot = int(np.searchsorted(grid, cols["TIME"][13]))
    flagger = _flagger(ms)
    flagger.apply(0, 0, 1, "m000", "m001", np.array(_rect(slot, slot, 0, 31)), [0, 1, 2, 3], scope="all")
    assert not _columns(ms)["FLAG_ROW"][13]
    flagger.undo()
    after = _columns(ms)
    np.testing.assert_array_equal(after["FLAG"], cols["FLAG"])
    np.testing.assert_array_equal(after["FLAG_ROW"], cols["FLAG_ROW"])


@needs_web
def test_counts_and_refusals(ms, monkeypatch):
    """Baselines count those the selection reaches; large dry runs skip FLAG; a no-op is refused."""
    from casacore.tables import table

    from surfvis.web import flagging

    with table(ms, readonly=False, ack=False) as t:
        a1, a2, times, scans = t.getcol("ANTENNA1"), t.getcol("ANTENNA2"), t.getcol("TIME"), t.getcol("SCAN_NUMBER")
        grid = np.unique(times[scans == 1])
        gone = np.flatnonzero((a1 == 2) & (a2 == 6) & np.isin(times, grid[[3, 4]]))
        t.removerows(gone.tolist())
    flagger = _flagger(ms)
    dry = flagger.apply(
        0, 0, 1, "m002", "m006", np.array(_rect(3, 4, 0, 5)), [0], dry_run=True, scope="antenna", antenna="m006"
    )
    assert dry["baselines"] == 7, "m002-m006 has no rows in those slots"

    exact = flagger.apply(0, 0, 1, "m000", "m001", np.array(_rect(0, 5, 0, 31)), [0, 3], dry_run=True, scope="all")
    monkeypatch.setattr(flagging, "DRY_RUN_CELLS", 100)
    rough = flagger.apply(0, 0, 1, "m000", "m001", np.array(_rect(0, 5, 0, 31)), [0, 3], dry_run=True, scope="all")
    assert rough["newly_flagged"] is None and rough["cells"] == exact["cells"] and exact["newly_flagged"] is not None

    flagger.apply(0, 0, 1, "m000", "m001", np.array(_rect(1, 2, 3, 4)), [0])
    with pytest.raises(ValueError, match="Nothing new"):
        flagger.apply(0, 0, 1, "m000", "m001", np.array(_rect(1, 2, 3, 4)), [0])
    assert len(flagger.edits) == 1


@needs_web
def test_a_journal_from_before_step_4_is_recovered(ms):
    """A journal listing cell indices, as step 3 wrote them, still rolls back."""
    import json as json_
    import os

    from surfvis.web import flagging

    cols = _columns(ms)
    rows = np.flatnonzero((cols["ANTENNA1"] == 0) & (cols["ANTENNA2"] == 1))[:2].astype(np.int64)
    flagging._fsync  # noqa: B018 - the module is importable
    with __import__("casacore.tables", fromlist=["table"]).table(ms, readonly=False, ack=False) as t:
        flag = t.getcol("FLAG")
        flag[rows, 7, 1] = True
        t.putcol("FLAG", flag)
    path = os.path.join(flagging._versions_root(ms), flagging.JOURNAL)
    os.makedirs(os.path.join(path, "headers"))
    np.savez(
        os.path.join(path, "edit.npz"),
        rows=rows,
        changed_row=np.array([0, 1], np.int32),
        changed_chan=np.array([7, 7], np.int32),
        changed_corr=np.array([1, 1], np.int32),
        row_set=np.array([], np.int64),
    )
    with open(os.path.join(path, "journal.json"), "w") as meta:
        json_.dump({"ms": ms, "headers": [], "description": "old"}, meta)
    assert "old" in flagging.recover(ms)
    np.testing.assert_array_equal(_columns(ms)["FLAG"], cols["FLAG"])


@needs_web
def test_chunks_and_channel_ranges_write_the_same_cells(ms, tmp_path, monkeypatch):
    """Rows go in chunks and only the selection's channels are read; neither changes what is written."""
    from surfvis.web import flagging

    other = str(tmp_path / "other.ms")
    shutil.copytree(ms, other)
    original = _columns(other)["FLAG"]
    poly = np.array([(0.2, 3.1), (7.6, 20.4), (10.4, 5.2), (4.0, 26.9)])  # irregular, many channel runs
    big = _flagger(ms).apply(0, 0, 1, "m001", "m003", poly, [1, 2], scope="all")
    monkeypatch.setattr(flagging, "CHUNK_CELLS", 37)  # a few rows per chunk, one row when a row is larger
    small_flagger = _flagger(other)
    small = small_flagger.apply(0, 0, 1, "m001", "m003", poly, [1, 2], scope="all")
    assert len(small_flagger.edits[-1].changed) > 10
    assert (big["cells"], big["newly_flagged"]) == (small["cells"], small["newly_flagged"])
    np.testing.assert_array_equal(_columns(ms)["FLAG"], _columns(other)["FLAG"])
    small_flagger.undo()
    np.testing.assert_array_equal(_columns(other)["FLAG"], original)


@needs_web
def test_flag_row_needs_the_channels_outside_the_selection(ms):
    """A row is fully flagged by a narrow selection only when the rest of it already was."""
    from casacore.tables import table

    cols = _columns(ms)
    rows = np.flatnonzero((cols["ANTENNA1"] == 0) & (cols["ANTENNA2"] == 5) & (cols["SCAN_NUMBER"] == 1))
    grid = np.unique(cols["TIME"][cols["SCAN_NUMBER"] == 1])
    slot = np.searchsorted(grid, cols["TIME"][rows])
    full_outside = rows[slot == 2]
    with table(ms, readonly=False, ack=False) as t:
        flag = t.getcol("FLAG")
        flag[full_outside] = True
        flag[full_outside, 8:12] = False  # all but the channels the selection covers
        t.putcol("FLAG", flag)
    flagger = _flagger(ms)
    flagger.apply(0, 0, 1, "m000", "m005", np.array(_rect(2, 3, 8, 11)), [0, 1, 2, 3], scope="antenna", antenna="m005")
    after = _columns(ms)
    newly = np.flatnonzero(after["FLAG_ROW"] & ~cols["FLAG_ROW"])
    np.testing.assert_array_equal(newly, full_outside)
    flagger.undo()
    assert not _columns(ms)["FLAG_ROW"][full_outside].any()


@needs_web
def test_all_baselines_swaps_cross_hands_on_reversed_rows(scoped, tmp_path):
    """XY of the waterfall is YX on rows stored as (q, p), whichever baseline they belong to."""
    from casacore.tables import table

    ms = str(tmp_path / "reversed.ms")
    shutil.copytree(scoped["ms"]["path"], ms)
    with table(ms, readonly=False, ack=False) as t:
        a1, a2 = t.getcol("ANTENNA1"), t.getcol("ANTENNA2")
        flip = np.flatnonzero((a1 == 1) & (a2 == 6))
        a1[flip], a2[flip] = 6, 1
        t.putcol("ANTENNA1", a1)
        t.putcol("ANTENNA2", a2)
    cols = _columns(ms)
    _flagger(ms).apply(0, 0, 1, "m002", "m003", np.array(_rect(4, 4, 7, 7)), [1], scope="all")
    after = _columns(ms)["FLAG"]
    changed = np.argwhere(after != cols["FLAG"])
    reversed_rows = np.isin(changed[:, 0], flip)
    assert reversed_rows.any() and (~reversed_rows).any()
    assert (changed[reversed_rows, 2] == 2).all(), "YX on the reversed rows"
    assert (changed[~reversed_rows, 2] == 1).all(), "XY everywhere else"


@needs_web
def test_the_audit_log_records_applies_undos_and_rollbacks(ms):
    from surfvis.web import flagging

    flagger = _flagger(ms)
    first = flagger.apply(0, 0, 1, "m001", "m002", np.array(_rect(1, 2, 3, 4)), [0], product="XX")
    second = flagger.apply(
        0, 0, 1, "m001", "m002", np.array(_rect(5, 6, 3, 4)), [0, 3], scope="antenna", antenna="m002"
    )
    flagger.undo(second["id"])
    entries = [json.loads(line) for line in open(flagging.audit_path(ms))]
    assert [e["event"] for e in entries] == ["apply", "apply", "undo"]
    apply = entries[0]
    assert apply["id"] == first["id"] and apply["scope"] == "baseline" and apply["antennas"] == ["m001", "m002"]
    assert apply["correlations"] == ["XX"] and apply["product"] == "XX" and apply["backup"] == "pretend"
    assert apply["slots"] == [1, 2] and apply["channels"] == [3, 4] and apply["cells"] == first["cells"]
    assert apply["times_utc"][0].endswith("Z") and len(apply["freqs_hz"]) == 2 and "@" in apply["user"]
    assert entries[1]["antennas"] == ["m002"] and entries[2]["id"] == second["id"]
    assert [e["id"] for e in flagging.outstanding(ms, 0, 0, 1, since=0)] == [first["id"]]
    assert flagging.outstanding(ms, 0, 0, 2, since=0) == []
    assert flagging.outstanding(ms, 0, 0, 1, since=entries[-1]["unix"]) == []
    # chi2 between an apply and its undo saw flags FLAG no longer holds: stale too.
    between = (entries[1]["unix"] + entries[2]["unix"]) / 2
    assert [e["id"] for e in flagging.outstanding(ms, 0, 0, 1, since=between)] == [second["id"]]

    # An edit interrupted after it was logged is rolled back at recovery, and the log says so.
    edit = flagger.edits[-1]
    flagging._write_journal(ms, edit, [], "apply")
    assert "rolled back" in flagging.recover(ms)
    assert json.loads(open(flagging.audit_path(ms)).readlines()[-1])["event"] == "rollback"
    assert flagging.outstanding(ms, 0, 0, 1, since=0) == []


@needs_web
def test_the_heatmap_marks_what_edits_made_stale(scoped, ms):
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web.app import create_app
    from tests.test_flagging import _wait_for_backup

    client = TestClient(create_app(scoped["zarr"], ms=ms, allow_flagging=True, warm=False))
    _wait_for_backup(client)
    query = "field=0&spw=0&scan=1"
    assert client.get(f"/stale?{query}").json() == {"edits": 0, "all": False, "antennas": [], "baselines": []}
    assert 'id="stale-note" hidden' in client.get(f"/heatmap?{query}&c=XX").text

    body = {"field": 0, "spw": 0, "scan": 1, "p": 1, "q": 3, "product": "XX", "vertices": _rect(1, 2, 4, 9)}
    one = client.post("/flag", json=body).json()
    client.post("/flag", json={**body, "baselines": "antenna", "antenna": 3})
    stale = client.get(f"/stale?{query}").json()
    assert stale == {"edits": 2, "all": False, "antennas": ["m003"], "baselines": [["m001", "m003"]]}
    html = client.get(f"/heatmap?{query}&c=XX").text
    assert "predates 2 manual edits" in html
    blob = html.split('id="grid-data">')[1].split("</script>")[0]
    assert json.loads(blob)["stale"] == stale
    assert client.get("/stale?field=0&spw=0&scan=2").json()["edits"] == 0

    # Undo takes the newest edit out; the antenna that remains is only the baseline's.
    client.post("/flag/undo", json={"id": client.get("/flag/state").json()["edits"][-1]["id"]})
    assert client.get(f"/stale?{query}").json()["baselines"] == [["m001", "m003"]]
    assert client.get(f"/stale?{query}").json()["antennas"] == []
    client.post("/flag", json={**body, "baselines": "all"})
    assert client.get(f"/stale?{query}").json()["all"] is True
    assert one["scope"] == "baseline"


@needs_web
def test_a_scope_reaches_only_the_panels_antennas(scoped, ms):
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web.app import create_app
    from tests.test_flagging import _wait_for_backup

    client = TestClient(create_app(scoped["zarr"], ms=ms, allow_flagging=True, warm=False))
    _wait_for_backup(client)
    body = {"field": 0, "spw": 0, "scan": 1, "p": 1, "q": 3, "product": "XX", "vertices": _rect(1, 2, 4, 9)}
    assert client.post("/flag", json={**body, "baselines": "antenna", "antenna": 5}).status_code == 400
    assert client.post("/flag", json={**body, "baselines": "antenna"}).status_code == 400
    assert client.post("/flag", json={**body, "baselines": "everything"}).status_code == 422
    dry = client.post("/flag", json={**body, "baselines": "all", "dry_run": True}).json()
    assert dry["baselines"] == 36 and dry["id"] is None
    panel = client.get("/waterfall?field=0&spw=0&scan=1&c=XX&p=1&q=3").text
    assert 'value="antenna:1">every baseline of m001' in panel and 'value="antenna:3">every baseline of m003' in panel
    auto = client.get("/waterfall?field=0&spw=0&scan=1&c=XX&p=4&q=4").text
    assert auto.count('value="antenna:') == 1 and "every baseline of m004" in auto
