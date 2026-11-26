"""Manual flagging from the waterfall: polygon selections written to FLAG.

A selection is a polygon in the waterfall's data coordinates -- (time slot,
channel), fractional -- and a rectangle is just a four-vertex polygon, so there
is one code path. The server rasterises it (:func:`rasterise`) and that mask is
what gets written, so a client-side preview, the write and any log agree on
which cells were flagged.

Writes are guarded four ways. ``serve`` must be started with
``--allow-flagging``. Before the first write of a session the current FLAG is
saved as a CASA flag version (:func:`save_flag_version`), restorable with
CASA's ``flagmanager``. Every write records the cells it turned on, so
:meth:`Flagger.undo` clears exactly those and nothing else. And every write
is journalled, so a crash part-way through cannot leave the MS unopenable or
an edit half-applied (see :func:`recover`).

Why the journal: when casacore flushes a TiledShapeStMan -- which is where
real Measurement Sets keep FLAG -- it truncates the storage manager's header
file (``table.f<n>``) and writes it again, and fsyncs neither the header nor
the tiles. A process killed between the truncate and the write leaves an
empty header, and the MS no longer opens. A flag write never changes that
header (it only rewrites tiles in place), so a copy taken just before the
write restores it exactly.

Design and plan: https://github.com/ratt-ru/surfvis/issues/16
"""

import atexit
import filecmp
import json
import os
import shutil
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

# Rows per read/write when copying FLAG for a backup: bounds memory on a large
# MS (a MeerKAT row is up to 32k channels x 4 correlations).
BACKUP_CHUNK_BYTES = 256 * 2**20

# Selections are drawn on a waterfall a few thousand cells across; a vertex
# beyond this is a crafted request, and would overflow the edge arithmetic.
MAX_COORDINATE = 1e9

# The write journal, in ``<ms>.flagversions/``. Present only while a write is
# in progress, or after one was interrupted.
JOURNAL = "surfvis_journal"
# Storage-manager files above this size hold column data written in place
# (a StandardStMan's buckets), not a header casacore rewrites; they are not
# copied into the journal.
HEADER_LIMIT = 64 * 2**20


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
        ValueError: if there are fewer than three vertices, or any is not finite
            or beyond ``MAX_COORDINATE``.
    """
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
        raise ValueError(f"A polygon needs at least three (slot, channel) vertices, got shape {verts.shape}")
    if not np.isfinite(verts).all():
        raise ValueError("Polygon vertices must be finite")
    if np.abs(verts).max() > MAX_COORDINATE:
        # Far beyond any waterfall, and big enough to overflow the edge arithmetic.
        raise ValueError(f"Polygon vertices must lie within +/-{MAX_COORDINATE:g} of the grid")

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

    root = _versions_root(ms)
    path = f"{root}/flags.{name}"
    if os.path.exists(path):
        raise FileExistsError(f"Flag version {name!r} already exists at {path}")
    os.makedirs(root, exist_ok=True)
    # Written under a hidden name and renamed when complete and on disk, so a
    # backup cut short never passes for a usable flag version.
    partial = f"{root}/.flags.{name}.partial"
    shutil.rmtree(partial, ignore_errors=True)

    src = arcae.table(ms)
    try:
        desc = src.tabledesc()
        nrow = src.nrow()
        ddid = np.asarray(src.getcol("DATA_DESC_ID"))
        # Keep the descriptor as arcae reports it, storage manager and all. Its
        # ``_c_order`` hint says a fixed cell shape is in C order; dropping it
        # transposes the shape of a tiled FLAG column.
        table_desc = {col: desc[col] for col in ("FLAG", "FLAG_ROW")}
        dest = Table.from_descriptor(partial, table_desc=table_desc)
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

    for entry in os.listdir(partial):
        if os.path.isfile(os.path.join(partial, entry)):
            _fsync(os.path.join(partial, entry))
    _fsync(partial)
    os.rename(partial, path)
    _fsync(root)
    with open(f"{root}/FLAG_VERSION_LIST", "a") as listing:
        listing.write(f"{name} : {comment}\n")
        listing.flush()
        os.fsync(listing.fileno())
    return name


def _versions_root(ms: str) -> str:
    return f"{ms.rstrip('/')}.flagversions"


def _fsync(path: str) -> None:
    """Push a file's (or a directory's entries') writes to disk."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _storage_files(ms: str, table) -> tuple[list[str], list[str]]:
    """The files of the storage managers holding FLAG and FLAG_ROW.

    Returns:
        ``(headers, files)``: the ``table.f<n>`` files small enough to be
        headers, which go in the journal, and every file of those storage
        managers (``table.f<n>`` and ``table.f<n>_TSM*``), which are fsynced
        after a write.
    """
    seqs = sorted(
        int(info["SEQNR"]) for info in table.getdminfo().values() if {"FLAG", "FLAG_ROW"} & set(info.get("COLUMNS", ()))
    )
    names = os.listdir(ms)
    headers, files = [], []
    for seq in seqs:
        base = f"table.f{seq}"
        files += sorted(n for n in names if n == base or n.startswith(base + "_"))
        if base in names and os.path.getsize(os.path.join(ms, base)) <= HEADER_LIMIT:
            headers.append(base)
    return headers, files


def _write_journal(ms: str, edit: "Edit", headers: list[str]) -> None:
    """Record an edit and copy the storage-manager headers, durably, before a write touches the MS.

    Built under a temporary name and renamed into place, so a journal is
    either complete or absent.
    """
    root = _versions_root(ms)
    os.makedirs(root, exist_ok=True)
    tmp = os.path.join(root, f".{JOURNAL}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "headers"))
    for name in headers:
        shutil.copyfile(os.path.join(ms, name), os.path.join(tmp, "headers", name))
        _fsync(os.path.join(tmp, "headers", name))
    np.savez(
        os.path.join(tmp, "edit.npz"),
        rows=edit.rows,
        changed_row=edit.changed[0],
        changed_chan=edit.changed[1],
        changed_corr=edit.changed[2],
        row_set=edit.row_set,
    )
    with open(os.path.join(tmp, "journal.json"), "w") as meta:
        json.dump({"ms": os.path.abspath(ms), "headers": headers, "description": edit.description}, meta)
    for name in ("edit.npz", "journal.json"):
        _fsync(os.path.join(tmp, name))
    _fsync(os.path.join(tmp, "headers"))
    _fsync(tmp)
    os.rename(tmp, os.path.join(root, JOURNAL))
    _fsync(root)


def _clear_journal(ms: str) -> None:
    """Retire the journal: renamed aside first, so a crash mid-delete never leaves a partial one."""
    root = _versions_root(ms)
    done = os.path.join(root, f".{JOURNAL}.done")
    shutil.rmtree(done, ignore_errors=True)
    os.rename(os.path.join(root, JOURNAL), done)
    _fsync(root)
    shutil.rmtree(done, ignore_errors=True)


def journal_path(ms: str) -> str | None:
    """The journal an interrupted write left behind, or None."""
    path = os.path.join(_versions_root(ms), JOURNAL)
    return path if os.path.isdir(path) else None


# Run by _opens in a child process.
_PROBE = """
import sys
import numpy as np
import arcae

table = arcae.table(sys.argv[1])
try:
    if table.nrow():
        first = (np.arange(1, dtype=np.int64),)
        table.getcol("FLAG", index=first)
        table.getcol("FLAG_ROW", index=first)
finally:
    table.close()
"""


def _opens(ms: str) -> bool:
    """Whether the MS opens and its FLAG and FLAG_ROW can be read.

    Tried in a child process. A failed open inside this one can leave a
    casacore read lock held in casacore's in-process lock registry, and the
    next write then waits for it forever.
    """
    import subprocess

    try:
        probe = subprocess.run([sys.executable, "-c", _PROBE, ms], capture_output=True, timeout=600, check=False)
    except subprocess.TimeoutExpired:
        return False
    return probe.returncode == 0


def _put_cells(table, edit: "Edit", on: bool) -> None:
    """Set (``on``) or clear an edit's cells, reading the rows as they are now.

    Setting turns on FLAG_ROW for the rows the edit recorded; clearing
    recomputes it for those rows from FLAG. Either is idempotent, which is
    what lets recovery simply run the clear again.
    """
    flag = np.array(table.getcol("FLAG", index=(edit.rows,)), dtype=bool)
    flag[edit.changed] = on
    table.putcol("FLAG", flag, index=(edit.rows,))
    if edit.row_set.size:
        flag_row = np.array(table.getcol("FLAG_ROW", index=(edit.rows,)), dtype=bool)
        flag_row[edit.row_set] = True if on else flag[edit.row_set].all(axis=(1, 2))
        table.putcol("FLAG_ROW", flag_row, index=(edit.rows,))


def recover(ms: str) -> str | None:
    """Undo whatever a write interrupted by a crash, kill or power cut left behind.

    Run before anything else opens the MS for writing (``serve
    --allow-flagging`` does so at startup). If the MS no longer opens, the
    storage-manager headers saved in the journal are put back (the files as
    found are kept in the journal's ``found/``); then the interrupted edit's
    cells are cleared, which rolls back an apply and completes an undo. Safe to
    run again if it is itself interrupted.

    Returns:
        What was done, or None when there was no journal.

    Raises:
        RuntimeError: if the MS still does not open with the headers restored.
    """
    import arcae

    path = journal_path(ms)
    root = _versions_root(ms)
    for stale in (f".{JOURNAL}.tmp", f".{JOURNAL}.done"):
        shutil.rmtree(os.path.join(root, stale), ignore_errors=True)
    if path is None:
        return None
    with open(os.path.join(path, "journal.json")) as meta_file:
        meta = json.load(meta_file)
    saved = np.load(os.path.join(path, "edit.npz"))
    edit = Edit(
        id=0,
        rows=saved["rows"].astype(np.int64),
        corr_idx=[],
        changed=(saved["changed_row"], saved["changed_chan"], saved["changed_corr"]),
        row_set=saved["row_set"],
        cells=0,
        newly_flagged=0,
        description=meta["description"],
    )

    restored = []
    if not _opens(ms):
        os.makedirs(os.path.join(path, "found"), exist_ok=True)
        for name in meta["headers"]:
            snapshot, current = os.path.join(path, "headers", name), os.path.join(ms, name)
            if os.path.exists(current) and filecmp.cmp(snapshot, current, shallow=False):
                continue
            if os.path.exists(current):
                shutil.copyfile(current, os.path.join(path, "found", name))
            staged = current + ".surfvis-restore"
            shutil.copyfile(snapshot, staged)
            _fsync(staged)
            os.replace(staged, current)
            restored.append(name)
        _fsync(ms)
        if not _opens(ms):
            raise RuntimeError(
                f"{ms} does not open, even with the storage-manager headers saved before the interrupted "
                f"flag write put back. The journal at {path} holds those headers (headers/) and the files "
                "as they were found (found/)."
            )

    table = arcae.table(ms, readonly=False)
    try:
        headers, files = _storage_files(ms, table)
        _put_cells(table, edit, on=False)
    finally:
        table.close()
    for name in files:
        _fsync(os.path.join(ms, name))
    _fsync(ms)
    _clear_journal(ms)
    done = f"rolled back an interrupted flag edit ({meta['description']})"
    return done + (f"; restored {', '.join(restored)}, which the interruption had broken" if restored else "")


# Flaggers with a write possibly in flight, waited for at interpreter exit.
_LIVE: "weakref.WeakSet[Flagger]" = weakref.WeakSet()


@atexit.register
def _wait_for_writes() -> None:
    """Let a write that is in progress finish before the process exits.

    FastAPI runs the write in a worker thread, which Ctrl-C does not
    interrupt, but which would die with the interpreter. atexit handlers run
    while such daemon threads are still alive.
    """
    for flagger in list(_LIVE):
        if flagger._lock.acquire(timeout=120):
            flagger._lock.release()


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


class StaleUndoError(LookupError):
    """The edit an undo named is no longer the most recent one."""


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
    # Where: the partition and the baseline as the waterfall presents it.
    field: int = -1
    spw: int = -1
    scan: int = -1
    antennas: tuple[str, str] = ("", "")

    def summary(self) -> dict:
        """What a client needs to name, find and undo this edit."""
        return {
            "id": self.id,
            "description": self.description,
            "field": self.field,
            "spw": self.spw,
            "scan": self.scan,
            "antennas": list(self.antennas),
            "cells": self.cells,
            "newly_flagged": self.newly_flagged,
        }


@dataclass(eq=False)  # compared and hashed by identity: one per server
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
    # Microseconds since the epoch: ids stay unique across restarts, so an undo
    # button left over from a previous server cannot name a new edit.
    _next_id: int = field(default_factory=lambda: time.time_ns() // 1000)
    # Set when a write failed and could not be put back; writes are refused.
    broken: str | None = None
    # What recover() did at startup, if anything.
    recovered: str | None = None

    def __post_init__(self) -> None:
        _LIVE.add(self)

    def recover(self) -> str | None:
        """Roll back a write a previous session was killed in the middle of; see :func:`recover`."""
        with self._lock:
            self.recovered = recover(self.ms)
            return self.recovered

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
            corr_idx: Correlations to flag, as positions in the partition's
                ``polarization`` axis (the MS's FLAG for rows not stored reversed).
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

            # The rows the waterfall shows: the baseline as xarray-ms presents it,
            # stored either way round.
            name1, name2 = msdata.view_order(ds, name1, name2)
            rows, times, flipped = msdata.baseline_rows(self.ms, field_id, spw, scan, name1, name2)
            if rows.size == 0:
                raise KeyError(f"No rows for {name1}-{name2} in field {field_id}, spw {spw}, scan {scan}")
            slot, hit = msdata.slot_index(grid, times)
            touched = hit & mask[slot].any(axis=1)
            rows, slot, flipped = rows[touched], slot[touched], flipped[touched]
            if rows.size == 0:
                raise ValueError("The selection covers no time slots with data for this baseline")

            import arcae

            table = arcae.table(self.ms)
            try:
                flag = np.asarray(table.getcol("FLAG", index=(rows,)), dtype=bool)
                flag_row = np.asarray(table.getcol("FLAG_ROW", index=(rows,)), dtype=bool)
            finally:
                table.close()
            # (row, channel) cells to set, broadcast over the chosen correlations.
            cells = np.zeros(flag.shape, dtype=bool)
            cells[..., corr_idx] = mask[slot][..., None]
            if flipped.any():
                # corr_idx is in the waterfall's order; a reversed row stores
                # the cross-hands the other way round.
                labels = [str(p) for p in ds.polarization.values]
                cells[flipped] = cells[flipped][..., msdata.reversed_correlations(labels)]
            already = flag | flag_row[:, None, None]
            summary = {
                "cells": int(cells.sum()),
                "newly_flagged": int((cells & ~already).sum()),
                "backup": self.backup,
            }
            if dry_run:
                return {"id": None, **summary}

            changed = cells & ~flag
            # Only ever set: a row flagged through FLAG_ROW alone stays flagged.
            new_row = flag_row | (flag | changed).all(axis=(1, 2))
            edit = Edit(
                id=self._next_id,
                rows=rows,
                corr_idx=list(corr_idx),
                changed=tuple(i.astype(np.int32) for i in np.nonzero(changed)),
                row_set=np.flatnonzero(new_row & ~flag_row),
                cells=summary["cells"],
                newly_flagged=summary["newly_flagged"],
                description=f"{name1}-{name2} field {field_id} spw {spw} scan {scan}",
                field=field_id,
                spw=spw,
                scan=scan,
                antennas=(name1, name2),
            )
            self._write(edit, on=True)
            self._record(edit)
            return {**edit.summary(), **summary}

    def _write(self, edit: Edit, on: bool) -> None:
        """Set or clear an edit's cells, journalled; on failure, put the MS back as it was before the edit.

        The journal is on disk before the first byte of the MS changes and is
        retired only once the storage managers' files are fsynced, so from a
        crash anywhere in between :func:`recover` restores a broken header and
        rolls the edit back.

        Raises:
            RuntimeError: if an earlier failure could not be put back.
        """
        import arcae

        from surfvis.web import msdata

        if self.broken is not None:
            raise RuntimeError(self.broken)
        # Nothing cached holds FLAG, so the write needs no re-open (see msdata.writing).
        with msdata.writing(self.ms):
            table = arcae.table(self.ms, readonly=False)
            try:
                headers, files = _storage_files(self.ms, table)
                _write_journal(self.ms, edit, headers)
            except BaseException:
                table.close()
                raise
            try:
                try:
                    _put_cells(table, edit, on)
                finally:
                    table.close()
                for name in files:
                    _fsync(os.path.join(self.ms, name))
                _fsync(self.ms)
                _clear_journal(self.ms)
            except BaseException as exc:
                self._put_back(exc)
                if self.broken is None:
                    exc.rolled_back = True  # the edit's cells are clear
                raise

    def _put_back(self, exc: BaseException) -> None:
        """Recover from a failed write in place; if that fails too, refuse further writes."""
        try:
            recover(self.ms)
        except Exception as again:  # noqa: BLE001 - reported, and writes stop
            self.broken = (
                f"A flag write failed ({type(exc).__name__}: {exc}) and putting it back failed too "
                f"({type(again).__name__}: {again}). Flagging is disabled; restart surfvis serve with "
                f"--allow-flagging to roll it back from {journal_path(self.ms)}."
            )
            print(f"surfvis: {self.broken}", file=sys.stderr, flush=True)

    def _record(self, edit: Edit) -> None:
        self._next_id += 1
        self.edits.append(edit)

    def undo(self, edit_id: int | None = None) -> Edit | None:
        """Clear the cells the most recent edit set. None if there is nothing to undo.

        Reads the rows as they are now and clears only the recorded cells, so
        flags written since by anything else survive. FLAG_ROW is recomputed
        only for rows this edit flagged.

        Args:
            edit_id: The edit the caller means to undo. The stack is shared by
                every tab on the server, so a button showing an older edit
                must not undo a newer one made elsewhere.

        Raises:
            StaleUndoError: if ``edit_id`` is not the most recent edit.
        """
        with self._lock:
            if not self.edits:
                return None
            edit = self.edits[-1]
            if edit_id is not None and edit_id != edit.id:
                raise StaleUndoError(
                    f"That edit is no longer the most recent: {edit.description} "
                    f"({edit.newly_flagged} cells) was flagged since, perhaps in another tab"
                )
            try:
                self._write(edit, on=False)
            except Exception as exc:
                # Putting a failed undo back clears the edit's cells, which is the undo.
                if getattr(exc, "rolled_back", False):
                    self.edits.pop()
                raise
            self.edits.pop()
            return edit
