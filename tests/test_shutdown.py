"""``surfvis serve`` leaves nothing running when it is interrupted.

A dropped ssh connection sends SIGHUP; Ctrl-C sends SIGINT; a scheduler or
``kill`` sends SIGTERM. Each must stop the server within seconds, even with a
request still reading the MS, and only after a flag write in progress has
finished.
"""

import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

from tests.conftest import needs_web

# The real serve command, with one route that never finishes in time: what a
# waterfall read of a large MS looks like to the server.
SERVE = """
import sys, time
from pathlib import Path
import surfvis.web.app as web

real = web.create_app

def create_app(*args, **kwargs):
    app = real(*args, **kwargs)

    @app.get("/slow")
    def slow():
        time.sleep(300)
        return "late"

    return app

web.create_app = create_app
from surfvis.cli.serve import serve

serve(data=Path(sys.argv[1]), ms=Path(sys.argv[2]), port=int(sys.argv[3]), allow_flagging=True)
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str, timeout: float) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status


def _start(chi2_zarr, tmp_path) -> tuple[subprocess.Popen, str]:
    ms = str(tmp_path / "serve.ms")
    shutil.copytree(chi2_zarr["ms"]["path"], ms)
    port = _free_port()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    proc = subprocess.Popen(
        [sys.executable, "-c", SERVE, str(chi2_zarr["zarr"]), ms, str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,  # its own process group, as a login shell gives it
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            if _get(base + "/flag/state", 1) == 200:
                return proc, base
        except OSError:
            time.sleep(0.2)
    proc.kill()
    raise AssertionError(f"serve did not start: {proc.stderr.read().decode()}")


def _group(pgid: int) -> list[int]:
    """Live processes in process group ``pgid`` (Linux /proc): what a hangup would have to reach."""
    out = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/stat") as f:
                    fields = f.read().rsplit(")", 1)[1].split()
                if int(fields[2]) == pgid and fields[0] != "Z":
                    out.append(int(entry))
            except (OSError, IndexError, ValueError):
                pass
    return out


@needs_web
@pytest.mark.parametrize("sig", [signal.SIGHUP, signal.SIGINT, signal.SIGTERM], ids=["hup", "int", "term"])
def test_serve_exits_promptly_with_a_request_in_flight(chi2_zarr, tmp_path, sig):
    """A busy request thread does not keep the process alive, whichever signal stops it."""
    proc, base = _start(chi2_zarr, tmp_path)
    try:
        threading.Thread(target=lambda: _swallow(base + "/slow"), daemon=True).start()
        time.sleep(1.0)  # the request is now sleeping in a worker thread
        os.killpg(proc.pid, sig)
        started = time.monotonic()
        code = proc.wait(timeout=30)
        assert time.monotonic() - started < 15
        assert code == 0, proc.stderr.read().decode()
        assert _group(proc.pid) == []
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def _swallow(url: str) -> None:
    try:
        _get(url, 60)
    except OSError:
        pass


@needs_web
def test_shutdown_finishes_a_flag_write_and_stops_a_recompute(chi2_zarr, tmp_path, monkeypatch):
    """What runs before the exit: a write in progress completes, a recompute writes nothing."""
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from surfvis.web import flagging, recompute, store
    from surfvis.web.app import create_app
    from tests.test_flagging import _rect, _wait_for_backup

    ms = str(tmp_path / "flagme.ms")
    zarr = tmp_path / "chi2.zarr"
    shutil.copytree(chi2_zarr["ms"]["path"], ms)
    shutil.copytree(chi2_zarr["zarr"], zarr)
    app = create_app(zarr, ms=ms, allow_flagging=True, warm=False)
    client = TestClient(app)
    _wait_for_backup(client)
    where = {"field": 0, "spw": 0, "scan": 1}
    body = {**where, "p": 1, "q": 3, "product": "XX", "vertices": _rect(1, 4, 2, 20)}
    assert client.post("/flag", json=body).status_code == 200  # something stale to recompute

    # A recompute that reads slowly, chunk by chunk, until told to stop.
    def slow_cells(*args, progress=None, **kwargs):
        for done in range(1000):
            time.sleep(0.05)
            progress(done, 1000)
        raise AssertionError("never stopped")

    writes = []
    monkeypatch.setattr(recompute, "cells", lambda *a: slow_cells(progress=a[-1]))
    monkeypatch.setattr(store.Chi2Store, "write_cells", lambda *a, **k: writes.append(1))
    assert client.post("/recompute", json=where).json()["state"] == "running"

    # A flag write that takes a while.
    real_write = flagging.Flagger._write

    def slow_write(self, *args, **kwargs):
        time.sleep(1.5)
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(flagging.Flagger, "_write", slow_write)
    result = {}
    writer = threading.Thread(
        target=lambda: result.update(r=client.post("/flag", json={**body, "vertices": _rect(6, 8, 2, 20)}))
    )
    writer.start()
    time.sleep(0.5)  # the write is under way

    started = time.monotonic()
    app.state.shutdown(timeout=30)
    assert time.monotonic() - started >= 0.5  # it waited for the write
    writer.join(timeout=5)
    assert result["r"].status_code == 200
    deadline = time.monotonic() + 5
    while (job := client.get("/recompute/state").json())["state"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert job["state"] == "failed" and "shutting down" in job["error"] and writes == []
