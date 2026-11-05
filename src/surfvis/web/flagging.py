"""Manual flagging from the waterfall: polygon selections written to FLAG.

A selection is a polygon in the waterfall's data coordinates -- (time slot,
channel), fractional -- and a rectangle is just a four-vertex polygon, so there
is one code path. The server rasterises it (:func:`rasterise`) and that mask is
what gets written, so a client-side preview, the write and any log agree on
which cells were flagged.

Writes are guarded three ways. ``serve`` must be started with
``--allow-flagging``. Before the first write of a session the current FLAG is
saved as a CASA flag version (:func:`save_flag_version`), restorable with
CASA's ``flagmanager``. And every write records the cells it turned on, so
:meth:`Flagger.undo` clears exactly those and nothing else.

Design and plan: https://github.com/ratt-ru/surfvis/issues/16
"""

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

# Rows per read/write when copying FLAG for a backup: bounds memory on a large
# MS (a MeerKAT row is up to 32k channels x 4 correlations).
BACKUP_CHUNK_BYTES = 256 * 2**20


def rasterise(vertices: np.ndarray, ntime: int, nchan: int) -> np.ndarray:
    """Cells of a (time slot, channel) grid whose centres lie inside a polygon.

    Cell ``(i, j)`` has its centre at ``(i, j)`` -- the waterfall draws slot
    ``i`` from ``i - 0.5`` to ``i + 0.5`` -- so a rectangle with corners
    ``(t0 - 0.5, c0 - 0.5)`` and ``(t1 + 0.5, c1 + 0.5)`` selects exactly slots
    ``t0..t1`` and channels ``c0..c1``. Inside is decided by the even-odd rule,
    so a self-intersecting polygon behaves predictably. Edges are half-open (a
    centre exactly on a lower or left edge is inside, on an upper or right edge
    outside), so two polygons sharing an edge never both claim a cell.

    Args:
        vertices: ``(k, 2)`` array of ``(slot, channel)`` vertices, ``k >= 3``,
            in order around the polygon. It is closed implicitly.
        ntime: Number of time slots in the grid.
        nchan: Number of channels in the grid.

    Returns:
        Boolean mask of shape ``(ntime, nchan)``.

    Raises:
        ValueError: if there are fewer than three vertices or any is not finite.
    """
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
        raise ValueError(f"A polygon needs at least three (slot, channel) vertices, got shape {verts.shape}")
    if not np.isfinite(verts).all():
        raise ValueError("Polygon vertices must be finite")

    mask = np.zeros((ntime, nchan), dtype=bool)
    t0, t1 = verts[:, 0], np.roll(verts[:, 0], -1)
    c0, c1 = verts[:, 1], np.roll(verts[:, 1], -1)
    lo = max(int(np.ceil(verts[:, 0].min())), 0)
    hi = min(int(np.ceil(verts[:, 0].max())), ntime)
    for i in range(lo, hi):
        # Edges spanning this row's centre line, half-open in time so a vertex
        # on the line is counted once.
        spans = (np.minimum(t0, t1) <= i) & (i < np.maximum(t0, t1))
        if not spans.any():
            continue
        x = c0[spans] + (i - t0[spans]) * (c1[spans] - c0[spans]) / (t1[spans] - t0[spans])
        x.sort()
        for left, right in zip(x[0::2], x[1::2], strict=True):
            start = max(int(np.ceil(left)), 0)
            stop = min(int(np.ceil(right)), nchan)
            if start < stop:
                mask[i, start:stop] = True
    return mask


def save_flag_version(ms: str, name: str, comment: str) -> str:
    """Copy FLAG and FLAG_ROW to a new CASA flag version of ``ms``.

    Same layout CASA uses and :func:`surfvis.web.msdata.flag_versions` reads:
    a plain ``<ms>.flagversions/flags.<name>`` table, row-aligned with the
    main table, plus a ``name : comment`` line in ``FLAG_VERSION_LIST``.
    Copied in row chunks of one ``DATA_DESC_ID`` at a time, since FLAG's shape
    varies between spectral windows and a whole column may not fit in memory.

    Returns:
        The version's name.

    Raises:
        FileExistsError: if a version of that name already exists.
    """
    import arcae
    from arcae.lib.arrow_tables import Table

    root = f"{ms.rstrip('/')}.flagversions"
    path = f"{root}/flags.{name}"
    if os.path.exists(path):
        raise FileExistsError(f"Flag version {name!r} already exists at {path}")
    os.makedirs(root, exist_ok=True)

    src = arcae.table(ms)
    try:
        desc = src.tabledesc()
        nrow = src.nrow()
        ddid = np.asarray(src.getcol("DATA_DESC_ID"))
        # Keep the descriptor as arcae reports it, storage manager and all. Its
        # ``_c_order`` hint says a fixed cell shape is in C order; dropping it
        # transposes the shape of a tiled FLAG column.
        table_desc = {col: desc[col] for col in ("FLAG", "FLAG_ROW")}
        dest = Table.from_descriptor(path, table_desc=table_desc)
        try:
            dest.addrows(nrow)
            for start, stop in _runs(ddid):
                width = int(np.prod(src.getcol("FLAG", index=(np.arange(start, start + 1),)).shape[1:]))
                step = max(1, BACKUP_CHUNK_BYTES // max(width, 1))
                for lo in range(start, stop, step):
                    rows = np.arange(lo, min(lo + step, stop), dtype=np.int64)
                    dest.putcol("FLAG", np.asarray(src.getcol("FLAG", index=(rows,)), dtype=bool), index=(rows,))
                    dest.putcol(
                        "FLAG_ROW", np.asarray(src.getcol("FLAG_ROW", index=(rows,)), dtype=bool), index=(rows,)
                    )
        finally:
            dest.close()
    finally:
        src.close()

    with open(f"{root}/FLAG_VERSION_LIST", "a") as listing:
        listing.write(f"{name} : {comment}\n")
    return name


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    """``[start, stop)`` ranges of consecutive equal values."""
    if values.size == 0:
        return []
    edges = np.flatnonzero(np.diff(values)) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [values.size]])
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


class BackupPendingError(RuntimeError):
    """Writes wait until the backup flag version is complete."""


@dataclass
class Edit:
    """One applied selection: exactly the cells it changed, so undo touches nothing else."""

    id: int
    rows: np.ndarray
    corr_idx: list[int]
    # Indices into (rows, channel, correlation) of the FLAG cells this edit
    # turned on. Cells already flagged are not recorded, so undo leaves them be.
    changed: tuple[np.ndarray, np.ndarray, np.ndarray]
    # Positions in ``rows`` whose FLAG_ROW this edit turned on.
    row_set: np.ndarray
    cells: int
    newly_flagged: int
    description: str


@dataclass
class Flagger:
    """Applies and undoes flag edits on one Measurement Set, for one server session.

    The undo stack lives in this process only; the backup flag version taken
    before the first write is the durable fallback. An edit records only the
    cells it changed, so undo leaves writes made since by anything else --
    ``flag-chi2``, CASA, another server -- alone.

    The backup copies the whole FLAG column, which on a large MS takes minutes.
    :meth:`start_backup` runs it in a background thread (``serve`` does this
    at startup) and writes wait for it; without that, the first write takes it
    synchronously.
    """

    ms: str
    edits: list[Edit] = field(default_factory=list)
    backup: str | None = None
    backup_error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _backup_done: threading.Event = field(default_factory=threading.Event)
    _backup_thread: threading.Thread | None = None
    _next_id: int = 1

    @property
    def backup_state(self) -> str:
        """``none`` (not started), ``running``, ``done`` or ``failed``."""
        if self.backup_error is not None:
            return "failed"
        if self._backup_done.is_set():
            return "done"
        return "running" if self._backup_thread is not None else "none"

    def start_backup(self) -> None:
        """Save FLAG as a flag version in a background thread; writes wait for it."""
        if self._backup_thread is not None or self._backup_done.is_set():
            return
        self._backup_thread = threading.Thread(target=self._take_backup, name="surfvis-flag-backup", daemon=True)
        self._backup_thread.start()

    def _take_backup(self) -> None:
        try:
            # Microseconds, so two servers (or a quick restart) never collide.
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            self.backup = save_flag_version(
                self.ms, f"surfvis_before_{stamp}", "surfvis serve: FLAG before manual flagging"
            )
        except Exception as exc:  # noqa: BLE001 - reported through backup_state
            self.backup_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._backup_done.set()

    def _require_backup(self) -> None:
        if self.backup_state == "none":
            self._take_backup()  # no background thread: take it now
        if not self._backup_done.is_set():
            raise BackupPendingError("The backup flag version is still being written; try again shortly")
        if self.backup_error is not None:
            raise RuntimeError(f"No backup flag version, so flagging is disabled: {self.backup_error}")

    def apply(
        self,
        field_id: int,
        spw: int,
        scan: int,
        name1: str,
        name2: str,
        vertices: np.ndarray,
        corr_idx: list[int],
        dry_run: bool = False,
    ) -> dict:
        """Flag the cells of one baseline inside a polygon.

        Args:
            field_id: FIELD_ID.
            spw: DATA_DESC_ID.
            scan: SCAN_NUMBER.
            name1: First antenna's name.
            name2: Second antenna's name.
            vertices: ``(k, 2)`` polygon in (time slot, channel) coordinates of
                the waterfall, slots counted on the partition's time grid.
            corr_idx: Correlations (positions in the MS's FLAG) to flag.
            dry_run: Count the cells without writing anything.

        Returns:
            ``id`` (None for a dry run), ``cells`` selected, ``newly_flagged``
            (selected cells that were not already flagged), and ``backup``.

        Raises:
            KeyError: if the baseline has no rows in the partition.
            ValueError: if the polygon is malformed or selects no cells.
            BackupPendingError: if the backup flag version is still being written.
        """
        from surfvis.web import msdata

        if not corr_idx:
            raise ValueError("No correlations to flag")
        if not dry_run:
            self._require_backup()
        with self._lock:
            ds = msdata._open(self.ms, field_id, spw, scan)
            grid = msdata.ms_times(ds)
            nchan = ds.sizes["frequency"]
            mask = rasterise(vertices, grid.size, nchan)
            if not mask.any():
                raise ValueError("The selection contains no cells")

            # The rows the waterfall shows: the baseline in the order the MS stores it.
            name1, name2 = msdata.stored_order(self.ms, field_id, spw, scan, name1, name2)
            rows, times = msdata.baseline_rows(self.ms, field_id, spw, scan, name1, name2)
            if rows.size == 0:
                raise KeyError(f"No rows for {name1}-{name2} in field {field_id}, spw {spw}, scan {scan}")
            slot, hit = msdata.slot_index(grid, times)
            touched = hit & mask[slot].any(axis=1)
            rows, slot = rows[touched], slot[touched]
            if rows.size == 0:
                raise ValueError("The selection covers no time slots with data for this baseline")

            import arcae

            table = arcae.table(self.ms, readonly=dry_run)
            try:
                flag = np.asarray(table.getcol("FLAG", index=(rows,)), dtype=bool)
                flag_row = np.asarray(table.getcol("FLAG_ROW", index=(rows,)), dtype=bool)
                # (row, channel) cells to set, broadcast over the chosen correlations.
                cells = np.zeros(flag.shape, dtype=bool)
                cells[..., corr_idx] = mask[slot][..., None]
                already = flag | flag_row[:, None, None]
                summary = {
                    "cells": int(cells.sum()),
                    "newly_flagged": int((cells & ~already).sum()),
                    "backup": self.backup,
                }
                if dry_run:
                    return {"id": None, **summary}

                changed = cells & ~flag
                new = flag | changed
                # Only ever set: a row flagged through FLAG_ROW alone stays flagged.
                new_row = flag_row | new.all(axis=(1, 2))
                edit = Edit(
                    id=self._next_id,
                    rows=rows,
                    corr_idx=list(corr_idx),
                    changed=tuple(i.astype(np.int32) for i in np.nonzero(changed)),
                    row_set=np.flatnonzero(new_row & ~flag_row),
                    cells=summary["cells"],
                    newly_flagged=summary["newly_flagged"],
                    description=f"{name1}-{name2} field {field_id} spw {spw} scan {scan}",
                )
                try:
                    table.putcol("FLAG", new, index=(rows,))
                    table.putcol("FLAG_ROW", new_row, index=(rows,))
                except Exception:
                    # Put back what this edit wrote, so a half-written edit is not left behind unrecorded.
                    try:
                        table.putcol("FLAG", flag, index=(rows,))
                        table.putcol("FLAG_ROW", flag_row, index=(rows,))
                    except Exception:  # noqa: BLE001 - keep the edit so undo can still try
                        self._record(edit)
                    raise
            finally:
                table.close()
                if not dry_run:
                    msdata.invalidate(self.ms)

            self._record(edit)
            return {"id": edit.id, **summary}

    def _record(self, edit: Edit) -> None:
        self._next_id += 1
        self.edits.append(edit)

    def undo(self) -> Edit | None:
        """Clear the cells the most recent edit set. None if there is nothing to undo.

        Reads the rows as they are now and clears only the recorded cells, so
        flags written since by anything else survive. FLAG_ROW is recomputed
        only for rows this edit flagged.
        """
        import arcae

        from surfvis.web import msdata

        with self._lock:
            if not self.edits:
                return None
            edit = self.edits[-1]
            table = arcae.table(self.ms, readonly=False)
            try:
                flag = np.array(table.getcol("FLAG", index=(edit.rows,)), dtype=bool)
                flag[edit.changed] = False
                table.putcol("FLAG", flag, index=(edit.rows,))
                if edit.row_set.size:
                    flag_row = np.array(table.getcol("FLAG_ROW", index=(edit.rows,)), dtype=bool)
                    flag_row[edit.row_set] = flag[edit.row_set].all(axis=(1, 2))
                    table.putcol("FLAG_ROW", flag_row, index=(edit.rows,))
            finally:
                table.close()
                msdata.invalidate(self.ms)
            self.edits.pop()
            return edit
