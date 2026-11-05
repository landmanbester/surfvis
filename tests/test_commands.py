"""Smoke tests for the plotting, chi-squared and flagging commands.

Neither had any coverage: `test_roundtrip.py` only proves the CLI and the cab
agree, which says nothing about whether the command runs. These exercise the
core functions against the fixture MS.
"""

import shutil

import pytest

from tests.conftest import needs_ms

np = pytest.importorskip("numpy")


@needs_ms
def test_summary_reports_every_antenna(tiny_ms, capsys):
    """summary prints one row per antenna, per field and per spectral window."""
    from surfvis.core.summary import summary

    summary(tiny_ms["path"])
    out = capsys.readouterr().out

    for name in tiny_ms["antenna_names"]:
        assert name in out
    assert "testfield" in out
    assert "/ANTENNA" in out and "/FIELD" in out and "/SPECTRAL_WINDOW" in out


@needs_ms
def test_surf_writes_one_png_per_baseline(tiny_ms, tmp_path):
    """surf renders the baseline it was asked for, and names it after the pair."""
    from surfvis.core.surf import surf

    opdir = tmp_path / "plots"
    surf(tiny_ms["path"], datacolumn="DATA", field=0, i=0, j=1, opdir=opdir, plot="amp")

    pngs = sorted(opdir.glob("*.png"))
    assert len(pngs) == 1
    assert "baseline_0_1" in pngs[0].name
    assert "_amp" in pngs[0].name
    assert pngs[0].stat().st_size > 0


@needs_ms
def test_surf_rejects_an_unknown_quantity(tiny_ms, tmp_path):
    """Stimela calls the core function directly, so it validates for itself."""
    from surfvis.core.surf import surf

    with pytest.raises(ValueError, match="must be one of"):
        surf(tiny_ms["path"], opdir=tmp_path / "plots", plot="nonsense")


@needs_ms
def test_flag_chi2_flags_the_bad_baseline_in_place(tiny_ms, tmp_path):
    """The planted residual is flagged, and quiet baselines are left alone."""
    from casacore.tables import table

    from surfvis.core.flag_chi2 import flag_chi2

    ms = tmp_path / "flagme.ms"
    shutil.copytree(tiny_ms["path"], ms)

    with table(str(ms), readonly=True, ack=False) as t:
        before = t.getcol("FLAG")
    assert not before.any(), "fixture starts unflagged"

    flag_chi2(ms, rcol="RESIDUAL", wcol="WEIGHT_SPECTRUM", flag_above=5.0, nthreads=1)

    with table(str(ms), readonly=True, ack=False) as t:
        after = t.getcol("FLAG")
        ant1, ant2 = t.getcol("ANTENNA1"), t.getcol("ANTENNA2")
        flag_row = t.getcol("FLAG_ROW")

    p, q = tiny_ms["hot"]
    hot_rows = (ant1 == p) & (ant2 == q)
    assert after[hot_rows].all(), "every hot-baseline visibility should be flagged"
    # Noise alone flags ~16% at this threshold on the fixture; a kernel that
    # flagged most good data must fail here, not just one that flagged all of it.
    assert after[~hot_rows].mean() < 0.25, "quiet baselines should stay mostly unflagged"
    assert flag_row[hot_rows].all(), "FLAG_ROW follows FLAG"


@needs_ms
def test_flag_chi2_respects_protected_antennas(tiny_ms, tmp_path):
    """--respect-ants leaves every baseline touching those antennas untouched."""
    from casacore.tables import table

    from surfvis.core.flag_chi2 import flag_chi2

    ms = tmp_path / "respect.ms"
    shutil.copytree(tiny_ms["path"], ms)
    p, q = tiny_ms["hot"]

    flag_chi2(ms, rcol="RESIDUAL", flag_above=5.0, nthreads=1, respect_ants=[p])

    with table(str(ms), readonly=True, ack=False) as t:
        after = t.getcol("FLAG")
        ant1, ant2 = t.getcol("ANTENNA1"), t.getcol("ANTENNA2")

    touching = (ant1 == p) | (ant2 == p)
    assert not after[touching].any(), "protected antenna's baselines must be untouched"


@needs_ms
def test_chi2_writes_per_correlation_and_combined_images(tiny_ms, tmp_path):
    """One image per (time bin, freq bin, correlation) and one per scan, through the spawn pool.

    Each correlation's image is computed from that correlation alone: the old
    code gave every correlation's task all of them, so c0 and c1 were identical.
    """
    import matplotlib.pyplot as plt

    from surfvis.core.chi2 import chi2

    out = tmp_path / "imgs"
    chi2(tiny_ms["path"], imagesout=out, dataout=tmp_path / "zarr", nthreads=2)

    for scan in range(1, tiny_ms["nscan"] + 1):
        scandir = out / "field0" / "spw0" / f"scan{scan}"
        assert (scandir / "combined.png").stat().st_size > 0
        # Default chunking: one time bin, one freq bin (nfreqs=128 > 32 channels), XX and YY.
        assert sorted(p.name for p in scandir.glob("t*.png")) == ["t0_f0_XX.png", "t0_f0_YY.png"]
        # Compare the plotted pixels, not the bytes: the titles ("XX" vs "YY")
        # differ even when the data are identical.
        c0, c1 = (plt.imread(scandir / f"t0_f0_{c}.png") for c in ("XX", "YY"))
        top = c0.shape[0] // 8
        assert not np.array_equal(c0[top:], c1[top:]), "per-correlation images hold the same data"


@needs_ms
def test_chi2_survives_an_antenna_missing_from_part_of_a_scan(tiny_ms, tmp_path):
    """Chunks that lack the highest antenna still add into the per-scan totals."""
    from casacore.tables import table

    from surfvis.core.chi2 import chi2

    ms = tmp_path / "gappy.ms"
    shutil.copytree(tiny_ms["path"], ms)
    last = tiny_ms["nant"] - 1
    with table(str(ms), readonly=False, ack=False) as t:
        ant1, ant2, scan, time = t.getcol("ANTENNA1"), t.getcol("ANTENNA2"), t.getcol("SCAN_NUMBER"), t.getcol("TIME")
        late = time > np.unique(time[scan == 1])[tiny_ms["ntime"] // 2 - 1]
        t.removerows(np.flatnonzero((scan == 1) & late & ((ant1 == last) | (ant2 == last))))

    out = tmp_path / "imgs"
    chi2(ms, imagesout=out, dataout=tmp_path / "zarr", nthreads=2, ntimes=3)

    assert (out / "field0" / "spw0" / "scan1" / "combined.png").stat().st_size > 0
