"""Where a request's time goes: named spans per request, and cProfile dumps of slow ones.

Off unless ``surfvis serve --profile DIR`` turns it on, and then:

- every response carries a ``Server-Timing`` header, which the browser's
  network panel draws as a breakdown of that request;
- one line per request goes to stderr and to ``DIR/spans.jsonl``, listing the
  spans in the order they finished, indented by nesting;
- a request slower than :data:`SLOW` seconds also leaves a cProfile dump,
  ``DIR/<time>-<route>.prof``, for whatever the spans did not anticipate
  (read it with ``python -m pstats`` or snakeviz).

Spans are recorded through a context variable, which Starlette copies into the
worker thread a sync route runs in, so ``span()`` deep inside ``msdata`` lands
on the request that caused it. With profiling off a span costs one context
variable lookup.
"""

import cProfile
import functools
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Seconds; requests slower than this leave a cProfile dump.
SLOW = 1.0

_DIR: Path | None = None
_REQUEST: ContextVar["_Request | None"] = ContextVar("surfvis_profile_request", default=None)
_DEPTH: ContextVar[int] = ContextVar("surfvis_profile_depth", default=0)
# cProfile cannot run two profilers at once on 3.12+ (sys.monitoring), and
# overlapping ones would muddle each other anyway: one request at a time.
_CPROFILE = threading.Lock()


@dataclass
class _Request:
    spans: list[tuple[str, float, int]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def enable(directory: str | Path | None) -> None:
    """Turn profiling on, writing to ``directory`` (created if missing); None turns it off.

    Process-wide, like the caches it measures: the last app built decides.
    """
    global _DIR
    _DIR = None if directory is None else Path(directory)
    if _DIR is not None:
        _DIR.mkdir(parents=True, exist_ok=True)


def enabled() -> bool:
    return _DIR is not None


@contextmanager
def span(name: str):
    """Time the block as ``name`` on the current request, if profiling is on."""
    request = _REQUEST.get()
    if request is None:
        yield
        return
    depth = _DEPTH.get()
    token = _DEPTH.set(depth + 1)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _DEPTH.reset(token)
        with request.lock:
            request.spans.append((name, elapsed, depth))


def timed(name: str):
    """Decorator form of :func:`span`."""

    def wrap(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            with span(name):
                return func(*args, **kwargs)

        return inner

    return wrap


def cprofiled(func):
    """Run a sync route under cProfile when profiling is on; dump it if it was slow.

    Applied to the route function itself because a sync route runs in a worker
    thread, and cProfile only sees the thread that enabled it.
    """

    @functools.wraps(func)
    def inner(*args, **kwargs):
        if _DIR is None or not _CPROFILE.acquire(blocking=False):
            return func(*args, **kwargs)
        profiler = cProfile.Profile()
        start = time.perf_counter()
        try:
            profiler.enable()
            try:
                return func(*args, **kwargs)
            finally:
                profiler.disable()
                if time.perf_counter() - start > SLOW:
                    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
                    profiler.dump_stats(_DIR / f"{stamp}-{func.__name__}.prof")
        finally:
            _CPROFILE.release()

    return inner


class ProfileMiddleware:
    """ASGI middleware that collects a request's spans and reports them."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or _DIR is None or scope["path"].startswith("/static"):
            await self.app(scope, receive, send)
            return

        request = _Request()
        token = _REQUEST.set(request)
        start = time.perf_counter()
        status = 0

        async def send_with_timing(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                total = time.perf_counter() - start
                message.setdefault("headers", [])
                message["headers"] = list(message["headers"]) + [(b"server-timing", _server_timing(request, total))]
            await send(message)

        try:
            await self.app(scope, receive, send_with_timing)
        finally:
            _REQUEST.reset(token)
            _report(scope, status, time.perf_counter() - start, request)


def _server_timing(request: _Request, total: float) -> bytes:
    with request.lock:
        spans = list(request.spans)
    # Header names are tokens: no spaces or punctuation besides - and _.
    entries = [
        f"{k}-{''.join(c if c.isalnum() else '_' for c in name)};dur={1e3 * s:.1f}"
        for k, (name, s, _) in enumerate(spans)
    ]
    entries.append(f"total;dur={1e3 * total:.1f}")
    return ", ".join(entries).encode("latin-1")


def _report(scope, status: int, total: float, request: _Request) -> None:
    with request.lock:
        spans = list(request.spans)
    path = scope["path"] + ("?" + scope["query_string"].decode() if scope.get("query_string") else "")
    lines = [f"[profile] {scope['method']} {path} -> {status} in {total:.3f} s"]
    lines += [f"[profile]   {'  ' * depth}{name}: {s:.3f} s" for name, s, depth in spans]
    print("\n".join(lines), file=sys.stderr, flush=True)
    record = {
        "time": time.time(),
        "method": scope["method"],
        "path": scope["path"],
        "query": scope.get("query_string", b"").decode(),
        "status": status,
        "total": total,
        "spans": [{"name": n, "seconds": s, "depth": d} for n, s, d in spans],
    }
    if _DIR is not None:
        with open(_DIR / "spans.jsonl", "a") as out:
            out.write(json.dumps(record) + "\n")


def summarise(directory: str | Path) -> str:
    """Total and worst time per span name across a ``spans.jsonl``, slowest first."""
    totals: dict[str, list[float]] = {}
    for line in open(Path(directory) / "spans.jsonl"):
        record = json.loads(line)
        for s in record["spans"]:
            totals.setdefault(s["name"], []).append(s["seconds"])
    rows = sorted(totals.items(), key=lambda kv: -sum(kv[1]))
    out = [f"{'span':<40} {'calls':>6} {'total s':>9} {'max s':>9}"]
    out += [f"{name:<40} {len(v):>6} {sum(v):>9.3f} {max(v):>9.3f}" for name, v in rows]
    return "\n".join(out)


if __name__ == "__main__":  # python -m surfvis.web.profiling DIR
    print(summarise(sys.argv[1] if len(sys.argv) > 1 else os.getcwd()))
