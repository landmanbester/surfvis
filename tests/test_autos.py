"""Auto-correlations: no chi-squared, but a waterfall from the heatmap's diagonal.

``chi2`` skips them and records whether a scan has any; ``serve`` makes the
diagonal clickable only then (or when an older zarr does not say).
"""

import shutil

import pytest

from tests.conftest import needs_web

np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def autos_zarr(tmp_path_factory):
    """An MS with auto-correlations, and chi2 of CORRECTED_DATA - MODEL_DATA over it."""
    from surfvis.core.chi2 import chi2
    from tests.fixtures.ms import make_ms

    out = tmp_path_factory.mktemp("autos")
    ms = make_ms(str(out / "autos.ms"), autos=True)
    chi2(
        ms["path"],
        rcol="CORRECTED_DATA",
        mcol="MODEL_DATA",
        dataout=out / "chi2.zarr",
        imagesout=out / "imgs",
        nthreads=1,
    )
    return {"zarr": out / "chi2.zarr", "ms": ms}


def _client(zarr, ms, **kwargs):
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web.app import create_app

    return TestClient(create_app(zarr, ms=ms, warm=False, **kwargs))


def _diagonal_cells(html: str, nant: int) -> int:
    return sum(f"&p={p}&q={p}" in html for p in range(nant))


def _forget_autos(zarr, tmp_path):
    """A copy of a chi2 zarr as written before chi2 recorded auto-correlations."""
    import zarr as zarr_

    old = tmp_path / "old.zarr"
    shutil.copytree(zarr, old)
    root = zarr_.open_group(str(old), mode="r+")
    for scan in (1, 2):
        del root[f"field0/spw0/scan{scan}"].attrs["autocorrelations"]
    zarr_.consolidate_metadata(str(old))  # xarray reads the consolidated copy
    return old


@needs_web
def test_chi2_records_autos_and_leaves_the_diagonal_empty(autos_zarr, chi2_zarr):
    from surfvis.web.store import Chi2Store, Selection

    with_autos = Chi2Store(autos_zarr["zarr"]).matrix(Selection(0, 0, 1, "XX"))
    assert with_autos.autos is True
    assert np.isnan(np.diag(with_autos.chi2_dof)).all()
    assert np.isfinite(with_autos.chi2_dof[~np.eye(len(with_autos.antenna_names), dtype=bool)]).all()
    assert Chi2Store(chi2_zarr["zarr"]).matrix(Selection(0, 0, 1, "XX")).autos is False


@needs_web
def test_the_diagonal_is_clickable_when_the_ms_has_autos(autos_zarr, chi2_zarr, tmp_path):
    query = "/heatmap?field=0&spw=0&scan=1&c=XX"
    nant = autos_zarr["ms"]["nant"]
    html = _client(autos_zarr["zarr"], autos_zarr["ms"]["path"]).get(query).text
    assert _diagonal_cells(html, nant) == nant and "diagonal: auto-correlations" in html

    html = _client(chi2_zarr["zarr"], chi2_zarr["ms"]["path"]).get(query).text
    assert _diagonal_cells(html, nant) == 0 and "diagonal" not in html

    # An older zarr does not say, so the diagonal is offered and the panel explains.
    old = _forget_autos(chi2_zarr["zarr"], tmp_path)
    client = _client(old, chi2_zarr["ms"]["path"])
    assert _diagonal_cells(client.get(query).text, nant) == nant
    panel = client.get("/waterfall?field=0&spw=0&scan=1&c=XX&p=3&q=3").text
    assert "m003 has no auto-correlation in scan 1" in panel and "waterfall.bin?" not in panel


@needs_web
def test_an_auto_waterfall_shows_the_column_without_the_model(autos_zarr):
    from casacore.tables import table

    from surfvis.web import planes

    ms = autos_zarr["ms"]
    client = _client(autos_zarr["zarr"], ms["path"])
    panel = client.get("/waterfall?field=0&spw=0&scan=1&c=XX&p=2&q=2").text
    assert "m002 auto-correlation" in panel and "&chi;<sup>2</sup>/dof" not in panel
    # The heatmap subtracted MODEL_DATA; an auto defaults to no model, and that is not a mismatch.
    assert '<option value="" selected>none</option>' in panel.split('name="model"')[1].split("</select>")[0]
    assert "no longer shows" not in panel

    header, values, flags = planes.decode(
        client.get("/waterfall.bin?field=0&spw=0&scan=1&c=XX&p=2&q=2&max_t=4096&max_c=4096").content
    )
    assert header["title"].startswith("m002 auto ")
    with table(ms["path"], ack=False) as t:
        sel = t.query("ANTENNA1 == 2 && ANTENNA2 == 2 && SCAN_NUMBER == 1", sortlist="TIME")
        expected = np.abs(sel.getcol("CORRECTED_DATA")[..., 0])
    assert not flags.any()
    np.testing.assert_allclose(values, expected, rtol=1e-5)

    # A cross-correlation on the same server still defaults to the heatmap's model.
    cross = client.get("/waterfall?field=0&spw=0&scan=1&c=XX&p=1&q=2").text
    assert 'value="MODEL_DATA" selected' in cross


@needs_web
def test_flagging_an_auto_touches_only_its_rows(autos_zarr, tmp_path):
    from casacore.tables import table

    ms = str(tmp_path / "flagme.ms")
    shutil.copytree(autos_zarr["ms"]["path"], ms)
    client = _client(autos_zarr["zarr"], ms, allow_flagging=True)
    while client.get("/flag/state").json()["backup_state"] == "running":
        pass

    with table(ms, ack=False) as t:
        before = t.getcol("FLAG")
        a1, a2, scans = t.getcol("ANTENNA1"), t.getcol("ANTENNA2"), t.getcol("SCAN_NUMBER")
    vertices = [(1.5, 3.5), (1.5, 9.5), (4.5, 9.5), (4.5, 3.5)]  # slots 2-4, channels 4-9
    out = client.post(
        "/flag", json={"field": 0, "spw": 0, "scan": 1, "p": 4, "q": 4, "product": "XX", "vertices": vertices}
    ).json()
    assert out["newly_flagged"] == 3 * 6 * 4
    assert out["description"].startswith("m004 auto ")

    with table(ms, ack=False) as t:
        after = t.getcol("FLAG")
    rows = np.flatnonzero((a1 == 4) & (a2 == 4) & (scans == 1))
    changed = np.unique(np.argwhere(after != before)[:, 0])
    np.testing.assert_array_equal(changed, rows[2:5])
    assert after[np.ix_(rows[2:5], np.arange(4, 10))].all()
