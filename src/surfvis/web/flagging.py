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
import contextlib
import filecmp
import functools
import getpass
import json
import os
import shutil
import socket
import sys
import threading
import time
import weakref
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

# Rows per read/write when copying FLAG for a backup: bounds memory on a large
# MS (a MeerKAT row is up to 32k channels x 4 correlations).
BACKUP_CHUNK_BYTES = 256 * 2**20

# FLAG cells per read or write of a selection: a selection over every
# baseline of a MeerKAT scan can cover billions, so they go in row chunks.
CHUNK_CELLS = 2**24

# A dry run reads FLAG to count the cells a selection would newly flag only
# up to this many cells (a second or two); beyond it, as for a wide scope on
# a large MS, it counts the cells selected and leaves ``newly_flagged`` None.
DRY_RUN_CELLS = 2**26

# Where a selection applies, besides its correlations (see GLOSSARY.md).
SCOPES = ("baseline", "antenna", "all")

# The audit log, in ``<ms>.flagversions/``: one JSON object per line for
# every edit applied, undone or rolled back, by any session.
AUDIT_LOG = "surfvis_flags.jsonl"

# MJD seconds of the unix epoch.
_MJD_UNIX = 3506716800.0

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


def _write_journal(ms: str, edit: "Edit", headers: list[str], op: str) -> None:
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
    blobs = [np.frombuffer(blob, np.uint8) for _, _, blob in edit.changed]
    np.savez(
        os.path.join(tmp, "edit.npz"),
        rows=edit.rows,
        row_set=edit.row_set,
        chan=np.asarray(edit.chan, np.int64),
        ncorr=np.int64(edit.ncorr),
        bounds=np.asarray([(i0, i1) for i0, i1, _ in edit.changed], np.int64).reshape(-1, 2),
        offsets=np.cumsum([0] + [b.size for b in blobs]).astype(np.int64),
        blob=np.concatenate(blobs) if blobs else np.zeros(0, np.uint8),
    )
    with open(os.path.join(tmp, "journal.json"), "w") as meta:
        json.dump(
            {
                "ms": os.path.abspath(ms),
                "headers": headers,
                "description": edit.description,
                "id": edit.id,
                "op": op,
            },
            meta,
        )
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
import os, sys

# Die with the server, however it dies (Linux). Done here, not in a preexec_fn,
# which is unsafe in the server's threaded process.
if sys.platform.startswith("linux"):
    import ctypes

    ctypes.CDLL(None).prctl(1, 9)  # PR_SET_PDEATHSIG, SIGKILL
    if os.getppid() != int(sys.argv[2]):
        sys.exit(1)  # the server died before that took effect

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
        probe = subprocess.run(
            [sys.executable, "-c", _PROBE, ms, str(os.getpid())],
            capture_output=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return probe.returncode == 0


def _put_cells(table, edit: "Edit", on: bool) -> None:
    """Set (``on``) or clear an edit's cells, reading the rows as they are now.

    Setting turns on FLAG_ROW for the rows the edit recorded; clearing
    recomputes it for those rows from FLAG. Either is idempotent, which is
    what lets recovery simply run the clear again.
    """
    c0, c1 = edit.chan
    chans = np.arange(c0, c1, dtype=np.int64)
    for i0, i1, blob in edit.changed:
        changed = _unpack(blob, (i1 - i0, c1 - c0, edit.ncorr))
        if not changed.any():
            continue
        rows = edit.rows[i0:i1].copy()
        flag = np.array(table.getcol("FLAG", index=(rows, chans)), dtype=bool)
        if changed.shape[-1] < flag.shape[-1]:  # an old journal's mask stops at its last correlation
            changed = np.pad(changed, ((0, 0), (0, 0), (0, flag.shape[-1] - changed.shape[-1])))
        flag[changed] = on
        table.putcol("FLAG", flag, index=(rows, chans))
    if edit.row_set.size:
        rows = edit.rows[edit.row_set].copy()
        if on:
            flag_row = np.ones(rows.size, dtype=bool)
        else:
            flag_row = np.zeros(rows.size, dtype=bool)
            for i0, i1 in _row_chunks(rows.size, table.getcol("FLAG", index=(rows[:1],)).shape[1:]):
                part = rows[i0:i1].copy()
                flag_row[i0:i1] = np.asarray(table.getcol("FLAG", index=(part,)), dtype=bool).all(axis=(1, 2))
        table.putcol("FLAG_ROW", flag_row, index=(rows,))


def _row_chunks(nrows: int, cell_shape: tuple[int, ...]) -> list[tuple[int, int]]:
    """``[start, stop)`` row ranges of at most :data:`CHUNK_CELLS` cells each."""
    step = max(1, CHUNK_CELLS // max(1, int(np.prod(cell_shape))))
    return [(i0, min(i0 + step, nrows)) for i0 in range(0, nrows, step)]


def _pack(mask: np.ndarray) -> bytes:
    """A boolean mask as compressed bits: an edit's cells are mostly one block, so this is small."""
    return zlib.compress(np.packbits(mask, axis=None).tobytes(), 1)


def _unpack(blob: bytes, shape: tuple[int, ...]) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(zlib.decompress(blob), np.uint8), count=int(np.prod(shape)))
    return bits.reshape(shape).astype(bool)


def _load_edit(path: str, meta: dict) -> "Edit":
    """The edit a journal recorded."""
    saved = np.load(os.path.join(path, "edit.npz"))
    if "blob" not in saved:  # a journal written before step 4 of #16: cell indices
        return _load_old_edit(saved, meta)
    blob, offsets = saved["blob"], saved["offsets"]
    changed = [
        (int(i0), int(i1), blob[offsets[k] : offsets[k + 1]].tobytes()) for k, (i0, i1) in enumerate(saved["bounds"])
    ]
    return Edit(
        id=int(meta.get("id", 0)),
        rows=saved["rows"].astype(np.int64),
        chan=(int(saved["chan"][0]), int(saved["chan"][1])),
        ncorr=int(saved["ncorr"]),
        changed=changed,
        row_set=saved["row_set"],
        cells=0,
        newly_flagged=0,
        description=meta["description"],
    )


def _load_old_edit(saved, meta: dict) -> "Edit":
    """An edit from a journal that listed its cells as (row, channel, correlation) indices."""
    rows = saved["rows"].astype(np.int64)
    r, c, k = saved["changed_row"], saved["changed_chan"], saved["changed_corr"]
    if c.size:
        c0, c1, ncorr = int(c.min()), int(c.max()) + 1, int(k.max()) + 1
    else:
        c0, c1, ncorr = 0, 1, 1
    mask = np.zeros((rows.size, c1 - c0, ncorr), dtype=bool)
    mask[r, c - c0, k] = True
    return Edit(
        id=0,
        rows=rows,
        chan=(c0, c1),
        ncorr=ncorr,
        changed=[(0, rows.size, _pack(mask))] if rows.size else [],
        row_set=saved["row_set"],
        cells=0,
        newly_flagged=0,
        description=meta["description"],
    )


def audit_path(ms: str) -> str:
    """The audit log of ``ms`` (which may not exist yet)."""
    return os.path.join(_versions_root(ms), AUDIT_LOG)


def _who() -> str:
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name in some containers
        user = "unknown"
    return f"{user}@{socket.gethostname()}"


def _utc(unix: float) -> str:
    return datetime.fromtimestamp(unix, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def log_entry(event: str, **fields) -> dict:
    """An audit-log line: what happened, when and by whom."""
    now = time.time()
    return {"event": event, "time": _utc(now), "unix": now, "user": _who(), **fields}


def append_log(ms: str, entry: dict) -> None:
    """Append one line to the audit log, durably."""
    path = audit_path(ms)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as log:
        log.write(json.dumps(entry) + "\n")
        log.flush()
        os.fsync(log.fileno())


def read_log(ms: str) -> list[dict]:
    """Every entry in the audit log, oldest first; lines that do not parse are skipped."""
    path = audit_path(ms)
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return []
    return _read_log(path, stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=4)
def _read_log(path: str, mtime_ns: int, size: int) -> list[dict]:
    entries = []
    with open(path) as log:
        for line in log:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
    return entries


def _edit_cells(entry: dict, names: list[str]) -> np.ndarray:
    """The heatmap cells an applied edit touched, ``(nant, nant)``, both triangles."""
    n = len(names)
    index = {name: k for k, name in enumerate(names)}
    cells = np.zeros((n, n), dtype=bool)
    scope = entry.get("scope", "baseline")
    antennas = [index[a] for a in entry.get("antennas", []) if a in index]
    if scope == "all":
        cells[:] = True
    elif scope == "antenna":
        for p in antennas:
            cells[p, :] = cells[:, p] = True
    elif len(antennas) == 2:
        p, q = antennas
        cells[p, q] = cells[q, p] = True
    return cells


def stale_cells(
    ms: str, field_id: int, spw: int, scan: int, computed: np.ndarray, names: list[str]
) -> tuple[int, np.ndarray]:
    """Which cells of a scan's heatmap no longer describe FLAG, and how many edits make them so.

    A cell is stale when an edit touching it was applied after the cell's
    chi-squared was computed and is still applied, or was applied before
    and undone (or rolled back) after: the chi-squared counted flags FLAG
    no longer holds. An edit applied and undone since cancels out.

    Args:
        computed: When each cell was computed, unix seconds, ``(nant, nant)``.
        names: The heatmap's antenna names, in grid order.

    Returns:
        The number of edits making any cell stale, and the ``(nant, nant)``
        stale mask, never true on the diagonal.
    """
    entries = read_log(ms)
    gone = {e.get("id"): float(e.get("unix", 0)) for e in entries if e.get("event") in ("undo", "rollback")}
    stale = np.zeros(computed.shape, dtype=bool)
    edits = 0
    for e in entries:
        if e.get("event") != "apply" or (e.get("field"), e.get("spw"), e.get("scan")) != (field_id, spw, scan):
            continue
        applied = float(e.get("unix", 0))
        undone = gone.get(e.get("id"))
        touched = _edit_cells(e, names)
        np.fill_diagonal(touched, False)  # auto-correlations have no chi-squared to go stale
        if undone is None:
            hit = touched & (computed < applied)
        else:
            hit = touched & (computed >= applied) & (computed < undone)
        if hit.any():
            edits += 1
            stale |= hit
    return edits, stale


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
    edit = _load_edit(path, meta)

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
    # An interrupted apply is rolled back and an interrupted undo completed;
    # the audit log says which, so the edit no longer counts as applied.
    if meta.get("id"):
        try:
            event = "undo" if meta.get("op") == "undo" else "rollback"
            append_log(ms, log_entry(event, id=meta["id"], description=meta["description"], at_recovery=True))
        except OSError as exc:
            print(f"surfvis: could not record the recovery in {audit_path(ms)}: {exc}", file=sys.stderr, flush=True)
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


class MSLockedError(RuntimeError):
    """Another process holds a lock on the MS, so a write would wait for it without limit."""


# Seconds a write waits for another process to release the MS before refusing.
LOCK_WAIT = 5.0


def lock_holder(ms: str, exclusive: bool = True) -> int | None:
    """The pid of another process whose casacore lock on ``ms`` stands in the way, or None.

    ``exclusive`` asks on behalf of a write, which any other holder blocks;
    otherwise on behalf of a read, which only another process's write lock
    blocks.

    casacore locks a table with fcntl locks on ``table.lock``, and a writer
    waits for every other holder without limit -- including an idle
    python-casacore or CASA session that has merely read the table, since
    auto-locking only gives a lock up while its holder is busy with the
    table. ``F_GETLK`` asks who would block an exclusive lock without taking
    one, and never reports this process's own locks. Linux only (the
    ``struct flock`` layout is platform-specific); elsewhere, None.
    """
    import fcntl
    import struct

    if not sys.platform.startswith("linux"):
        return None
    layout = "hhxxxxqqixxxx"  # struct flock on Linux: type, whence, start, len, pid
    try:
        fd = os.open(os.path.join(ms, "table.lock"), os.O_RDONLY)
    except OSError:
        return None
    try:
        probe = struct.pack(layout, fcntl.F_WRLCK if exclusive else fcntl.F_RDLCK, os.SEEK_SET, 0, 0, 0)
        kind, _, _, _, pid = struct.unpack(layout, fcntl.fcntl(fd, fcntl.F_GETLK, probe))
    finally:
        os.close(fd)
    return None if kind == fcntl.F_UNLCK else pid


def _process(pid: int) -> str:
    """``pid`` and its command line, as far as /proc tells."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        cmd = ""
    if len(cmd) > 120:
        cmd = cmd[:117] + "..."
    return f"pid {pid}" + (f" ({cmd})" if cmd else "")


def wait_unlocked(ms: str, wait: float | None = None, exclusive: bool = True) -> None:
    """Return once no other process holds a lock on ``ms``; refuse after ``wait`` (:data:`LOCK_WAIT`) seconds.

    A busy reader releases between its reads, so this polls rather than
    giving up on the first look.

    Raises:
        MSLockedError: naming the process still holding it.
    """
    deadline = time.monotonic() + (LOCK_WAIT if wait is None else wait)
    while (pid := lock_holder(ms, exclusive)) is not None:
        if time.monotonic() >= deadline:
            raise MSLockedError(
                f"The Measurement Set is locked by another process, {_process(pid)}. "
                "Close the table there (or wait for it to finish) and try again; nothing was written."
            )
        time.sleep(0.1)


class BackupPendingError(RuntimeError):
    """Writes wait until the backup flag version is complete."""


class StaleUndoError(LookupError):
    """The edit an undo named is no longer the most recent one."""


@dataclass
class Edit:
    """One applied selection: exactly the cells it changed, so undo touches nothing else."""

    id: int
    # The rows the selection reaches: its scope's rows in the slots it covers.
    rows: np.ndarray
    # The selection's channel range, [start, stop): the only channels read or written.
    chan: tuple[int, int]
    ncorr: int
    # The FLAG cells this edit turned on, over (rows, channel range,
    # correlation), as (start, stop, packed mask) per chunk of rows. Cells
    # already flagged are not recorded, so undo leaves them be.
    changed: list[tuple[int, int, bytes]]
    # Positions in ``rows`` whose FLAG_ROW this edit turned on.
    row_set: np.ndarray
    cells: int
    newly_flagged: int
    description: str
    # Where: the partition and the baseline of the panel it was drawn on.
    field: int = -1
    spw: int = -1
    scan: int = -1
    antennas: tuple[str, str] = ("", "")
    # Which baselines: ``baseline`` (the panel's), ``antenna`` (every baseline
    # of ``scope_antennas[0]``) or ``all`` (every baseline in the scan).
    scope: str = "baseline"
    scope_antennas: tuple[str, ...] = ()
    baselines: int = 1

    def summary(self) -> dict:
        """What a client needs to name, find and undo this edit."""
        return {
            "id": self.id,
            "description": self.description,
            "field": self.field,
            "spw": self.spw,
            "scan": self.scan,
            "antennas": list(self.antennas),
            "scope": self.scope,
            "scope_antennas": list(self.scope_antennas),
            "baselines": self.baselines,
            "cells": self.cells,
            "newly_flagged": self.newly_flagged,
        }


def _describe(scope: str, members: tuple[str, ...], field_id: int, spw: int, scan: int) -> str:
    if scope == "all":
        where = "all baselines"
    elif scope == "antenna":
        where = f"every baseline of {members[0]}"
    else:
        a, b = members
        where = f"{a} auto" if a == b else f"{a}-{b}"
    return f"{where} field {field_id} spw {spw} scan {scan}"


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
        scope: str = "baseline",
        antenna: str | None = None,
        product: str = "",
    ) -> dict:
        """Flag the cells inside a polygon drawn on one baseline's waterfall, over a scope.

        Args:
            field_id: FIELD_ID.
            spw: DATA_DESC_ID.
            scan: SCAN_NUMBER.
            name1: First antenna's name of the panel's baseline.
            name2: Second antenna's name.
            vertices: ``(k, 2)`` polygon in (time slot, channel) coordinates of
                the waterfall, slots counted on the partition's time grid,
                which every baseline of the scan shares.
            corr_idx: Correlations to flag, as positions in the partition's
                ``polarization`` axis (the MS's FLAG for rows not stored reversed).
            dry_run: Count the cells without writing anything.
            scope: ``baseline`` (the panel's), ``antenna`` (every baseline of
                ``antenna``, its auto-correlation included) or ``all``.
            antenna: For the antenna scope: one of the panel's two antennas.
            product: The product the panel plotted, for the audit log.

        Returns:
            ``id`` (None for a dry run), ``cells`` selected, ``newly_flagged``
            (selected cells that were not already flagged), ``baselines``
            reached, and ``backup``.

        Raises:
            KeyError: if the baseline has no rows in the partition.
            ValueError: if the polygon is malformed, selects no cells, or the
                scope is not one of :data:`SCOPES`.
            BackupPendingError: if the backup flag version is still being written.
        """
        from surfvis.web import msdata

        if scope not in SCOPES:
            raise ValueError(f"Unknown scope {scope!r}; expected one of {', '.join(SCOPES)}")
        if not corr_idx:
            raise ValueError("No correlations to flag")
        if not dry_run:
            self._require_backup()
        # Another process's lock would hang the reads below as well as the
        # write; a dry run only reads, so only a writer stands in its way.
        wait_unlocked(self.ms, exclusive=not dry_run)
        # A dry run only reads, so it need not wait for (or hold up) a write.
        with contextlib.nullcontext() if dry_run else self._lock:
            ds = msdata._open(self.ms, field_id, spw, scan)
            grid = msdata.ms_times(ds)
            nchan = ds.sizes["frequency"]
            mask = rasterise(vertices, grid.size, nchan)
            if not mask.any():
                raise ValueError("The selection contains no cells")

            # The rows the selection reaches. A baseline is as xarray-ms
            # presents it; rows stored the other way round are marked reversed.
            name1, name2 = msdata.view_order(ds, name1, name2)
            if scope == "baseline":
                members: tuple[str, ...] = (name1, name2)
                rows, times, flipped = msdata.baseline_rows(self.ms, field_id, spw, scan, name1, name2)
                pair = np.zeros(rows.size, dtype=np.int64)
                if rows.size == 0:
                    raise KeyError(f"No rows for {name1}-{name2} in field {field_id}, spw {spw}, scan {scan}")
            else:
                if scope == "antenna" and antenna not in (name1, name2):
                    raise ValueError(f"The antenna scope must name {name1} or {name2}, not {antenna!r}")
                members = (antenna,) if scope == "antenna" else ()
                rows, times, flipped, pair = msdata.scope_rows(self.ms, field_id, spw, scan, ds, members or None)
            slots = mask.any(axis=1)
            slot, hit = msdata.slot_index(grid, times)
            touched = hit & slots[slot]
            rows, slot, flipped = rows[touched], slot[touched], flipped[touched]
            if rows.size == 0:
                raise ValueError("The selection covers no time slots with data for this baseline")
            # The baselines with rows the selection reaches, not every one of the scope.
            nbaselines = int(np.unique(pair[touched]).size)

            # Only the selection's channel range is read and written.
            chans_hit = np.flatnonzero(mask.any(axis=0))
            c0, c1 = int(chans_hit[0]), int(chans_hit[-1]) + 1
            chans = np.arange(c0, c1, dtype=np.int64)
            sub = mask[:, c0:c1]
            labels = [str(p) for p in ds.polarization.values]
            ncorr = len(labels)
            corr_sel = np.zeros(ncorr, dtype=bool)
            corr_sel[list(corr_idx)] = True
            # corr_idx is in the waterfall's order; a reversed row stores
            # the cross-hands the other way round.
            perm = msdata.reversed_correlations(labels) if flipped.any() else None
            outside = np.ones(nchan, dtype=bool)
            outside[c0:c1] = False

            import arcae

            ncells = int(mask.sum(axis=1)[slot].sum()) * len(set(corr_idx))
            if dry_run and ncells > DRY_RUN_CELLS:
                return {
                    "id": None,
                    "cells": ncells,
                    "newly_flagged": None,
                    "baselines": nbaselines,
                    "scope": scope,
                    "backup": self.backup,
                }
            nnew = 0
            touched_any = False
            changed_chunks: list[tuple[int, int, bytes]] = []
            row_set: list[np.ndarray] = []
            table = arcae.table(self.ms)
            try:
                flag_row = np.asarray(table.getcol("FLAG_ROW", index=(rows,)), dtype=bool)
                for i0, i1 in _row_chunks(rows.size, (c1 - c0, ncorr)):
                    part = rows[i0:i1].copy()
                    flag = np.asarray(table.getcol("FLAG", index=(part, chans)), dtype=bool)
                    cells = sub[slot[i0:i1]][..., None] & corr_sel
                    fl = flipped[i0:i1]
                    if perm is not None and fl.any():
                        cells[fl] = cells[fl][..., perm]
                    fr = flag_row[i0:i1]
                    nnew += int((cells & ~(flag | fr[:, None, None])).sum())
                    if dry_run:
                        continue
                    changed = cells & ~flag
                    changed_chunks.append((i0, i1, _pack(changed)))
                    # Rows the edit leaves fully flagged get FLAG_ROW; only ever set.
                    # Only rows it changed: a row already fully flagged in FLAG
                    # is left as found, so undo restores its FLAG_ROW exactly.
                    full = np.flatnonzero(~fr & changed.any(axis=(1, 2)) & (flag | changed).all(axis=(1, 2)))
                    if full.size and outside.any():
                        keep = np.zeros(full.size, dtype=bool)
                        for j0, j1 in _row_chunks(full.size, (nchan, ncorr)):
                            whole = np.asarray(table.getcol("FLAG", index=(part[full[j0:j1]].copy(),)), dtype=bool)
                            keep[j0:j1] = whole[:, outside].all(axis=(1, 2))
                        full = full[keep]
                    row_set.append(i0 + full)
                    touched_any = touched_any or bool(changed.any())
            finally:
                table.close()
            summary = {
                "cells": ncells,
                "newly_flagged": nnew,
                "baselines": nbaselines,
                "scope": scope,
                "backup": self.backup,
            }
            if dry_run:
                return {"id": None, **summary}
            if not touched_any:
                raise ValueError("Nothing new to flag: every selected cell is flagged already")

            description = _describe(scope, members, field_id, spw, scan)
            edit = Edit(
                id=self._next_id,
                rows=rows,
                chan=(c0, c1),
                ncorr=ncorr,
                changed=changed_chunks,
                row_set=np.concatenate(row_set) if row_set else np.zeros(0, np.int64),
                cells=ncells,
                newly_flagged=nnew,
                description=description,
                field=field_id,
                spw=spw,
                scan=scan,
                antennas=(name1, name2),
                scope=scope,
                scope_antennas=members,
                baselines=nbaselines,
            )
            covered = np.flatnonzero(slots)
            t0, t1 = float(grid[covered[0]]), float(grid[covered[-1]])
            freqs = ds.frequency.values
            entry = log_entry(
                "apply",
                id=edit.id,
                description=description,
                ms=os.path.abspath(self.ms),
                backup=self.backup,
                field=field_id,
                spw=spw,
                scan=scan,
                scope=scope,
                antennas=list(members),
                panel=[name1, name2],
                baselines=nbaselines,
                rows=int(rows.size),
                product=product,
                correlations=[labels[i] for i in corr_idx],
                vertices=[[float(t), float(c)] for t, c in np.asarray(vertices, dtype=float)],
                slots=[int(covered[0]), int(covered[-1])],
                channels=[c0, c1 - 1],
                times_utc=[_utc(t0 - _MJD_UNIX), _utc(t1 - _MJD_UNIX)],
                times_mjd=[t0, t1],
                freqs_hz=[float(freqs[c0]), float(freqs[c1 - 1])],
                cells=ncells,
                newly_flagged=nnew,
            )
            self._write(edit, on=True, entry=entry)
            self._record(edit)
            return {**edit.summary(), **summary}

    def _write(self, edit: Edit, on: bool, entry: dict | None = None) -> None:
        """Set or clear an edit's cells, journalled; on failure, put the MS back as it was before the edit.

        The journal is on disk before the first byte of the MS changes and is
        retired only once the storage managers' files are fsynced, so from a
        crash anywhere in between :func:`recover` restores a broken header and
        rolls the edit back. ``entry`` goes into the audit log before the
        journal is retired: a crash after it is recorded by recovery as a
        rollback, so the log never claims an edit the MS does not hold.

        Raises:
            MSLockedError: if another process holds the MS; nothing was written.
            RuntimeError: if an earlier failure could not be put back.
        """
        import arcae

        from surfvis.web import msdata

        if self.broken is not None:
            raise RuntimeError(self.broken)
        # Before anything is journalled: a write would otherwise wait on
        # another process's lock without limit, with the request hanging.
        wait_unlocked(self.ms)
        # Nothing cached holds FLAG, so the write needs no re-open (see msdata.writing).
        with msdata.writing(self.ms):
            table = arcae.table(self.ms, readonly=False)
            try:
                headers, files = _storage_files(self.ms, table)
                _write_journal(self.ms, edit, headers, "apply" if on else "undo")
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
                if entry is not None:
                    # Stamped now, with FLAG written and synced: anything that
                    # read FLAG before this moment (a recompute, chi2) is older.
                    now = time.time()
                    append_log(self.ms, {**entry, "unix": now, "time": _utc(now)})
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
            MSLockedError: if another process holds the MS; nothing was written.
        """
        wait_unlocked(self.ms)
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
                self._write(edit, on=False, entry=log_entry("undo", id=edit.id, description=edit.description))
            except Exception as exc:
                # Putting a failed undo back clears the edit's cells, which is the undo.
                if getattr(exc, "rolled_back", False):
                    self.edits.pop()
                raise
            self.edits.pop()
            return edit
