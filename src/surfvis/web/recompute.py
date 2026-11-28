"""Recompute stale heatmap cells from the Measurement Set, by chi2's recipe.

A cell of the chi2 zarr is the sum, over one scan, of ``|r|^2 w`` over the
unflagged samples of one baseline and product, and their count. ``chi2``
forms ``r`` from the columns its attrs name (``rcol`` minus ``mcol``,
corrected by ``gtable``), weights it by ``wcol`` (``1/sigma^2`` for
``SIGMA_SPECTRUM``), and counts only FLAG, never FLAG_ROW. Recomputing
reads exactly those columns, by row with arcae, and forms the products
with the same ``utils/products.form``, so a recomputed cell agrees with a
fresh ``chi2`` run on the same flags to rounding (the sums are added in a
different order). The waterfall's controls play no part.

Like ``chi2``, rows are taken as stored: a baseline stored both ways round
contributes both orientations to its cell.
"""

from collections.abc import Callable

import numpy as np

from surfvis.utils import products as products_mod

# Visibility cells per read: four columns of this many complex/float cells.
CHUNK_CELLS = 2**23


def cells(
    ms: str,
    attrs: dict,
    products: list[str],
    field: int,
    spw: int,
    scan: int,
    pairs: list[tuple[int, int]],
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Chi-squared and counts of the given baselines, as ``chi2`` would write them.

    Args:
        ms: The Measurement Set.
        attrs: The zarr group's attrs: ``rcol``, ``wcol``, ``fcol``, ``mcol``,
            ``gtable``, ``corr_labels``, ``antenna_names``.
        products: The group's products, in its order.
        field: FIELD_ID.
        spw: DATA_DESC_ID.
        scan: SCAN_NUMBER.
        pairs: Antenna index pairs ``(p, q)``, ``p != q``, each unordered.
        progress: Called with (rows done, rows in total) after each chunk.

    Returns:
        ``(chi2, counts)``, each ``(nproducts, len(pairs))``.

    Raises:
        ValueError: if the attrs do not say how chi2 formed the residual.
    """
    import arcae

    from surfvis.utils import gains as gains_mod
    from surfvis.web import msdata

    missing = [k for k in ("rcol", "wcol", "fcol", "corr_labels", "antenna_names") if k not in attrs]
    if missing:
        raise ValueError(f"The chi2 output does not record {', '.join(missing)}; rerun surfvis chi2")
    rcol, wcol, fcol = str(attrs["rcol"]), str(attrs["wcol"]), str(attrs["fcol"])
    mcol, gtable = str(attrs.get("mcol", "") or ""), str(attrs.get("gtable", "") or "")
    labels = [str(c) for c in attrs["corr_labels"]]
    names = [str(n) for n in attrs["antenna_names"]]
    nant = len(names)

    scan_ = msdata.scan_rows(ms, field, spw, scan)
    times = np.unique(scan_.times)  # chi2's grid: every TIME of the scan
    lo = np.minimum(scan_.antenna1, scan_.antenna2)
    hi = np.maximum(scan_.antenna1, scan_.antenna2)
    code = {min(p, q) * nant + max(p, q): k for k, (p, q) in enumerate(pairs)}
    which = np.array([code.get(c, -1) for c in (lo * nant + hi).tolist()], dtype=np.int64)
    take = (which >= 0) & (lo != hi)
    rows, which = scan_.rows[take].copy(), which[take]
    a1, a2, row_times = scan_.antenna1[take], scan_.antenna2[take], scan_.times[take]

    chi2 = np.zeros((len(products), len(pairs)))
    counts = np.zeros((len(products), len(pairs)))
    if rows.size == 0:
        return chi2, counts

    gains = None
    full = False
    if gtable:
        freqs = msdata._open(ms, field, spw, scan).frequency.values
        gains = gains_mod.load(gtable, field, spw, scan, times, freqs, names)
        full = gains.full
    read = products_mod.needed(products, labels, full_jones=full)
    read_idx = np.array([labels.index(c) for c in read], dtype=np.int64)

    table = arcae.table(ms)
    try:
        nchan = int(np.asarray(table.getcol(fcol, index=(rows[:1],))).shape[1])
        step = max(1, CHUNK_CELLS // (nchan * len(labels)))
        for i0 in range(0, rows.size, step):
            part = rows[i0 : i0 + step].copy()

            def col(name, part=part):
                return np.asarray(table.getcol(name, index=(part,)))[..., read_idx]

            vis = col(rcol)
            weight = col(wcol)
            if wcol == "SIGMA_SPECTRUM":
                # In the column's own precision, as chi2 does.
                with np.errstate(divide="ignore"):
                    weight = 1.0 / weight**2
            flag = col(fcol).astype(bool)
            model = col(mcol) if mcol else None
            sub = slice(i0, i0 + part.size)
            tidx = gains.time_index(row_times[sub], times) if gains is not None else None
            v, w, f = products_mod.form(
                products, read, vis, weight, flag, a1[sub], a2[sub], tidx=tidx, gains=gains, model=model
            )
            good = ~f
            contrib = np.where(good, (v.real**2 + v.imag**2) * w, 0.0).sum(axis=1)  # (rows, products)
            for k in range(len(products)):
                np.add.at(chi2[k], which[sub], contrib[:, k])
                np.add.at(counts[k], which[sub], good[..., k].sum(axis=1))
            if progress is not None:
                progress(min(i0 + step, rows.size), rows.size)
    finally:
        table.close()
    return chi2, counts
