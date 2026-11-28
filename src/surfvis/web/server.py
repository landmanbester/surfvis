"""Run the app under uvicorn, and leave nothing behind when interrupted.

``surfvis serve`` usually runs over ssh, so it has to stop cleanly on a
dropped connection (SIGHUP) as well as on Ctrl-C (SIGINT) and ``kill``
(SIGTERM). uvicorn alone does not: it ignores SIGHUP, whose default action
kills the process mid-write, and on the others it waits without limit for
open requests and then the interpreter waits for every non-daemon thread
(anyio's request workers, the scan warm-up), so one waterfall read of a large
MS keeps a "stopped" server alive for minutes.

Here every one of those signals starts uvicorn's graceful shutdown, open
requests get :data:`GRACE` seconds, the app's own shutdown runs (a flag write
in progress finishes; a recompute stops before it writes), and the process
then exits without waiting for threads that are only reading. A flag write
is waited for however long it takes, with a message; a signal during the
app's shutdown exits at once, and the interrupted write is what the journal
recovers at the next start. Under ``nohup`` (SIGHUP ignored) a hangup is
still ignored: surviving the dropped connection is what was asked for.
"""

import os
import signal
import sys
from collections.abc import Callable

# Seconds open requests get to finish once shutdown starts.
GRACE = 3.0

SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def run(app, host: str, port: int) -> None:
    """Serve ``app`` until a signal, then shut down and exit the process.

    Never returns: it ends the process with ``os._exit``, which is the only way
    past a worker thread still blocked in a read.
    """
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, timeout_graceful_shutdown=GRACE))
    stopping = False

    def on_signal(sig, frame):
        # uvicorn installs its own handlers for SIGINT and SIGTERM while it
        # serves and re-raises the signal here afterwards; SIGHUP it leaves
        # alone. Either way, and in the moments before uvicorn takes over,
        # asking it to exit is right (and harmless once it has stopped).
        if stopping:
            _exit(1)
        server.handle_exit(signal.SIGTERM if sig == signal.SIGHUP else sig, frame)

    for sig in SIGNALS:
        if sig == signal.SIGHUP and signal.getsignal(sig) is signal.SIG_IGN:
            continue  # nohup
        signal.signal(sig, on_signal)
    code = 0
    try:
        server.run()
    except BaseException:  # noqa: BLE001 - nothing may stop the exit below
        code = 1
    stopping = True
    shutdown: Callable[[], None] | None = getattr(app.state, "shutdown", None)
    if shutdown is not None:
        try:
            shutdown()
        except BaseException:  # noqa: BLE001
            code = 1
    _exit(code)


def _exit(code: int) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):  # a hung-up terminal
            pass
    os._exit(code)
