"""Read the chi-squared zarr written by ``surfvis chi2 --dataout``."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from surfvis.web.profiling import timed


@dataclass(frozen=True)
class Selection:
    """One heatmap: a (field, spw, scan) and a product."""

    field: int
    spw: int
    scan: int
    product: str


class Chi2Store:
    """Lazy reader over the chi-squared zarr hierarchy."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"No chi-squared dataset at {self.path}")
        self._scans = _discover_scans(self.path)
        if not self._scans:
            raise ValueError(f"{self.path} contains no field*/spw*/scan* groups")

    @property
    def scans(self) -> list[tuple[int, int, int]]:
        """Every (field, spw, scan) in the store, in sorted order."""
        return self._scans

    @property
    def fields(self) -> list[int]:
        return sorted({f for f, _, _ in self._scans})

    def spws(self, field: int) -> list[int]:
        return sorted({s for f, s, _ in self._scans if f == field})

    def scan_numbers(self, field: int, spw: int) -> list[int]:
        return sorted({n for f, s, n in self._scans if f == field and s == spw})

    def group(self, field: int, spw: int, scan: int) -> xr.Dataset:
        """Open one (field, spw, scan) group."""
        return _open_group(str(self.path), field, spw, scan)

    def matrix(self, sel: Selection) -> "Matrix":
        """The antenna-by-antenna chi-squared/dof matrix of one scan and product."""
        ds = self.group(sel.field, sel.spw, sel.scan)
        hit = np.where(ds["product"].values.astype(str) == sel.product)[0]
        if hit.size == 0:
            raise KeyError(f"No product {sel.product!r}; have {list(ds['product'].values)}")
        chi2, counts = _scan_totals(ds, int(hit[0]))

        with np.errstate(invalid="ignore", divide="ignore"):
            chi2_dof = np.where(counts > 0, chi2 / np.where(counts > 0, counts, 1), np.nan)

        return Matrix(
            selection=sel,
            chi2_dof=chi2_dof,
            antenna_names=list(ds.attrs["antenna_names"]),
            ms=str(ds.attrs["ms"]),
            rcol=str(ds.attrs["rcol"]),
            mcol=str(ds.attrs.get("mcol", "")),
            gtable=str(ds.attrs.get("gtable", "")),
            autos=_autos(ds.attrs.get("autocorrelations")),
        )

    def attrs(self, field: int, spw: int, scan: int) -> dict:
        """The attributes ``chi2`` recorded on a (field, spw, scan) group."""
        return dict(self.group(field, spw, scan).attrs)

    def cell_times(self, field: int, spw: int, scan: int) -> np.ndarray:
        """When each cell's chi-squared was computed, unix seconds, ``(antenna1, antenna2)``.

        ``computed`` once ``serve`` has recomputed any cell; before that the
        group's ``created`` (when chi2 started), or for an older zarr the
        group directory's mtime (when chi2 finished writing it, so edits made
        while it ran are missed). The group's own directory, not the root's:
        every recompute re-consolidates at the root.
        """
        ds = self.group(field, spw, scan)
        if "computed" in ds:
            return np.asarray(ds["computed"].values, dtype=float)
        since = ds.attrs.get("created")
        if since is None:
            since = os.path.getmtime(self.path / f"field{field}" / f"spw{spw}" / f"scan{scan}")
        nant = len(ds.attrs["antenna_names"])
        return np.full((nant, nant), float(since))

    def recomputable(self, field: int, spw: int, scan: int) -> str | None:
        """Why this group's cells cannot be recomputed, or None if they can."""
        ds = self.group(field, spw, scan)
        if "time_bin" in ds.dims or "product" not in ds.coords:
            return "This chi2 output predates per-scan totals; rerun surfvis chi2 to recompute cells."
        missing = [k for k in ("rcol", "wcol", "fcol", "corr_labels") if k not in ds.attrs]
        if missing:
            return f"This chi2 output does not record {', '.join(missing)}; rerun surfvis chi2 to recompute cells."
        return None

    def write_cells(
        self,
        field: int,
        spw: int,
        scan: int,
        pairs: list[tuple[int, int]],
        chi2: np.ndarray,
        counts: np.ndarray,
        started: float,
        ms: str | None = None,
    ) -> None:
        """Replace the given cells' chi-squared and counts (both triangles), stamped ``started``.

        Args:
            pairs: Antenna index pairs ``(p, q)``.
            chi2: ``(nproducts, len(pairs))``, in the group's product order.
            counts: Same shape.
            started: When the recompute began, before it read FLAG: the cells'
                new ``computed`` time.
            ms: The Measurement Set the cells were read from, recorded as
                ``recomputed_ms`` (serve's ``--ms`` need not be chi2's ``ms``).
        """
        import xarray as xr_

        key = f"field{field}/spw{spw}/scan{scan}"
        times = self.cell_times(field, spw, scan)
        ds = self.group(field, spw, scan)
        # chi2's attrs must survive adding a variable; some xarray versions'
        # append mode has replaced a group's attrs with the dataset's.
        saved = dict(zarr.open_group(str(self.path), mode="r")[key].attrs)
        if "computed" not in ds:
            # Through xarray, so the new variable carries its dimension names.
            coords = {"antenna1": ds["antenna1"].values, "antenna2": ds["antenna2"].values}
            xr_.Dataset({"computed": (("antenna1", "antenna2"), times)}, coords=coords).to_zarr(
                str(self.path), group=key, mode="a", consolidated=False
            )
            zarr.consolidate_metadata(str(self.path))  # zarr opens groups through it
        group = zarr.open_group(str(self.path), mode="r+")[key]
        group.attrs.update({k: v for k, v in saved.items() if k not in group.attrs})
        c2, cn = np.array(group["chi2"][:]), np.array(group["counts"][:])
        for k, (p, q) in enumerate(pairs):
            for a, b in ((p, q), (q, p)):
                c2[:, a, b] = chi2[:, k]
                cn[:, a, b] = counts[:, k]
                times[a, b] = started
        group["chi2"][:] = c2
        group["counts"][:] = cn
        group["computed"][:] = times
        group.attrs["recomputed_cells"] = int(group.attrs.get("recomputed_cells", 0)) + len(pairs)
        group.attrs["recomputed_at"] = float(started)
        if ms is not None:
            group.attrs["recomputed_ms"] = os.path.abspath(ms)
        zarr.consolidate_metadata(str(self.path))
        _open_group.cache_clear()

    def products(self, field: int, spw: int, scan: int) -> list[str]:
        """The products (``XX``, ``I``, ...) a scan has."""
        ds = self.group(field, spw, scan)
        if "product" not in ds.coords:
            raise ValueError(f"{self.path} predates products (has no 'product' axis); rerun surfvis chi2")
        return [str(v) for v in ds["product"].values]


@timed("zarr: scan totals")
def _scan_totals(ds: xr.Dataset, k: int) -> tuple[np.ndarray, np.ndarray]:
    """One product's chi-squared and counts summed over the scan.

    ``chi2`` now writes these sums directly. Older stores kept every (time
    bin, freq bin) chunk; those are summed one time bin at a time, so the
    whole cube is never in memory (with ``--ntimes 1`` on a long scan it is
    gigabytes).
    """
    if "time_bin" not in ds.dims:
        return ds.chi2.isel(product=k).values, ds.counts.isel(product=k).values
    chi2 = counts = 0.0
    for t in range(ds.sizes["time_bin"]):
        idx = {"time_bin": t, "product": k}
        chi2 = chi2 + ds.chi2.isel(idx).values.sum(axis=0)
        counts = counts + ds.counts.isel(idx).values.sum(axis=0)
    return chi2, counts


@dataclass(frozen=True)
class Matrix:
    """One scan's chi-squared/dof matrix for one product, plus the context to plot it."""

    selection: Selection
    chi2_dof: np.ndarray
    antenna_names: list[str]
    ms: str
    rcol: str
    # How chi2 formed the quantity; empty when not used.
    mcol: str = ""
    gtable: str = ""
    # Whether the scan has auto-correlations; None for a zarr from before
    # chi2 recorded it.
    autos: bool | None = None

    @property
    def finite(self) -> np.ndarray:
        """The non-NaN values, i.e. the antenna pairs that had unflagged data."""
        return self.chi2_dof[~np.isnan(self.chi2_dof)]

    @property
    def nbaselines(self) -> int:
        """Baselines with unflagged data. The grid holds each one twice, as (p, q) and (q, p)."""
        valid = ~np.isnan(self.chi2_dof)
        return int(np.triu(valid | valid.T, k=1).sum())


def _autos(value) -> bool | None:
    """The ``autocorrelations`` attribute, which a zarr store may hand back as a numpy bool."""
    return None if value is None else bool(value)


@lru_cache(maxsize=32)
@timed("zarr: open group")
def _open_group(path: str, field: int, spw: int, scan: int) -> xr.Dataset:
    # chunks=None reads eagerly into numpy. Without it xarray returns
    # dask-backed arrays, which would make dask a hard requirement of the web
    # extra for no benefit -- a single chunk's matrix is a few hundred KB and
    # every consumer here calls .values immediately. cache=False keeps xarray
    # from holding a whole variable in memory once any slice of it is read.
    return xr.open_dataset(path, engine="zarr", group=f"field{field}/spw{spw}/scan{scan}", chunks=None, cache=False)


def _discover_scans(path: Path) -> list[tuple[int, int, int]]:
    """Walk the zarr hierarchy for field*/spw*/scan* groups."""
    root = zarr.open_group(str(path), mode="r")
    found = []
    for fname in root:
        if not fname.startswith("field"):
            continue
        for sname in root[fname]:
            if not sname.startswith("spw"):
                continue
            for nname in root[fname][sname]:
                if not nname.startswith("scan"):
                    continue
                found.append(
                    (
                        int(fname.removeprefix("field")),
                        int(sname.removeprefix("spw")),
                        int(nname.removeprefix("scan")),
                    )
                )
    return sorted(found)
