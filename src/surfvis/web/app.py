"""FastAPI application wiring the chi-squared store to waterfall plots."""

import hashlib
import os
import sys
import threading
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from surfvis.utils import gains as gains_mod
from surfvis.web import flagging, msdata, planes, profiling, recompute, render
from surfvis.web.flagging import BackupPendingError, Flagger, StaleUndoError
from surfvis.web.store import Chi2Store, Selection

HERE = Path(__file__).parent

# How `surfvis serve --reload` hands its options to the worker uvicorn
# re-imports: reload needs an import string, not an app object.
ENV_DATA, ENV_MS, ENV_GTABLES = "SURFVIS_SERVE_DATA", "SURFVIS_SERVE_MS", "SURFVIS_SERVE_GTABLES"
ENV_FLAGGING = "SURFVIS_SERVE_ALLOW_FLAGGING"
ENV_PROFILE = "SURFVIS_SERVE_PROFILE"


class FlagRequest(BaseModel):
    """A polygon on one baseline's waterfall, in its (time slot, channel) coordinates."""

    field: int
    spw: int
    scan: int
    # Non-negative: a negative index would wrap and flag another baseline.
    p: int = Field(ge=0)
    q: int = Field(ge=0)
    product: str
    vertices: list[tuple[float, float]] = Field(min_length=3)
    # "all" (the default: one flagged correlation should flag them all) or
    # "product": the correlations the plotted product is formed from.
    correlations: Literal["product", "all"] = "all"
    # Which baselines: the panel's, every baseline of one of its antennas
    # (``antenna``, an index like p and q, must be one of them), or every
    # baseline in the scan.
    baselines: Literal["baseline", "antenna", "all"] = "baseline"
    antenna: int | None = Field(default=None, ge=0)
    # The panel's settings the selection was drawn on. A gain table can change
    # which correlations a product reads; a flag version means the picture is
    # not FLAG, so a selection drawn on it is refused.
    # Unset means the term chi2 used, as on /waterfall.
    gtable: str | None = None
    flagversion: str = ""
    dry_run: bool = False


class RecomputeRequest(BaseModel):
    """The scan whose stale cells to recompute."""

    field: int
    spw: int
    scan: int


class UndoRequest(BaseModel):
    """The edit the undo button named."""

    id: int


class RevalidatedStaticFiles(StaticFiles):
    """Static files the browser must revalidate (by ETag) on every use.

    The ``?v=`` digest busts caches for files the page names, but imports
    between ES modules under ``static/wf/`` cannot carry it.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def app_from_env() -> FastAPI:
    """The app :func:`create_app` builds, configured from the environment."""
    gtables = [g for g in os.environ.get(ENV_GTABLES, "").split(os.pathsep) if g]
    return create_app(
        os.environ[ENV_DATA],
        os.environ.get(ENV_MS) or None,
        gtables=gtables,
        allow_flagging=bool(os.environ.get(ENV_FLAGGING)),
        profile=os.environ.get(ENV_PROFILE) or None,
    )


def create_app(
    data: str | Path,
    ms: str | Path | None = None,
    gtables: list[str] | None = None,
    allow_flagging: bool = False,
    profile: str | Path | None = None,
    warm: bool = True,
) -> FastAPI:
    """Build the app around one chi-squared dataset.

    Args:
        data: Path to the zarr written by ``surfvis chi2 --dataout``.
        ms: Measurement Set to read waterfalls from. Defaults to the path
            recorded in the dataset, which is right unless it has moved.
        gtables: Gain terms the waterfall may apply, besides the one ``chi2``
            used and any found in ``*.qc`` directories next to the MS. No
            other path is opened: the server has no authentication, so a
            query string must not be able to name arbitrary directories.
        allow_flagging: Enable the routes that write FLAG. Off by default:
            the browser is read-only unless asked otherwise.
        profile: Directory to write request timings and cProfile dumps of
            slow requests to (see :mod:`surfvis.web.profiling`). Off if None.
        warm: Open a scan and read its rows in the background while its
            heatmap is on screen, so the first waterfall does not wait for
            that (on a large MS, the better part of a minute).

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    profiling.enable(profile)
    store = Chi2Store(data)
    app = FastAPI(title="surfvis")
    app.add_middleware(profiling.ProfileMiddleware)
    app.mount("/static", RevalidatedStaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    # StaticFiles sends no Cache-Control, so browsers cache by heuristic and
    # keep serving an old stylesheet after an upgrade. A version in the URL
    # makes every change a new resource.
    templates.env.globals["static_version"] = _static_version(HERE / "static")
    # Jinja renders inside TemplateResponse; the heatmap's thousands of cells make that a real cost.
    templates.TemplateResponse = profiling.timed("render template")(templates.TemplateResponse)

    first = store.scans[0]
    ms_path = str(ms) if ms is not None else store.matrix(Selection(*first, store.products(*first)[0])).ms

    recorded = sorted({store.attrs(*key).get("gtable", "") for key in store.scans} - {""})

    @profiling.timed("gain_tables")
    def _gain_tables() -> dict[str, str]:
        """Offered gain terms, as {real path: spelling shown}. Re-scanned per request, so new solves appear."""
        listed = list(recorded) + list(gtables or []) + gains_mod.find_terms(str(Path(ms_path).resolve().parent))
        offered = {}
        for spec in listed:
            offered.setdefault(os.path.realpath(gains_mod.term_path(spec)), spec)
        return offered

    def _allowed(gtable: str) -> bool:
        return not gtable or os.path.realpath(gains_mod.term_path(gtable)) in _gain_tables()

    @profiling.timed("store.matrix")
    def _matrix(field, spw, scan, c):
        try:
            return store.matrix(Selection(field=field, spw=spw, scan=scan, product=c))
        except (KeyError, IndexError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=f"No such scan or product: {exc}") from exc

    def _baseline(matrix, p, q) -> tuple[str, str]:
        names = matrix.antenna_names
        if not (p < len(names) and q < len(names)):
            raise HTTPException(status_code=404, detail=f"No antenna pair ({p}, {q}); there are {len(names)} antennas")
        return names[p], names[q]

    def _selectors(request, field, spw, scan, c="", scale="log"):
        corrs = store.products(field, spw, scan)
        return {
            "store": store,
            "field_names": msdata.field_names(ms_path),
            "field": field,
            "spw": spw,
            "scan": scan,
            "corrs": corrs,
            # Keep the product and scale across a change of scan where they still exist.
            "c": c if c in corrs else corrs[0],
            "scale": scale if scale in render.SCALES else "log",
        }

    @app.get("/", response_class=HTMLResponse)
    @profiling.cprofiled
    def index(request: Request):
        field, spw, scan = store.scans[0]
        return templates.TemplateResponse(
            request, "index.html", {"ms": ms_path, **_selectors(request, field, spw, scan)}
        )

    @app.get("/scans", response_class=HTMLResponse)
    @profiling.cprofiled
    def scans(
        request: Request,
        field: int = Query(...),
        spw: int = Query(...),
        scan: int | None = Query(None),
        c: str = Query(""),
        scale: str = Query("log"),
    ):
        """Re-render the selectors when field, spw or scan changes: each scan has its own products."""
        if field not in store.fields:
            raise HTTPException(status_code=404, detail=f"No field {field}")
        spws = store.spws(field)
        if spw not in spws:
            spw = spws[0]
        numbers = store.scan_numbers(field, spw)
        if scan not in numbers:
            scan = numbers[0]
        return templates.TemplateResponse(request, "_selectors.html", _selectors(request, field, spw, scan, c, scale))

    @app.get("/heatmap", response_class=HTMLResponse)
    @profiling.cprofiled
    def heatmap(
        request: Request,
        field: int = Query(...),
        spw: int = Query(...),
        scan: int = Query(...),
        c: str = Query(...),
        scale: str = Query("log"),
    ):
        matrix = _matrix(field, spw, scan, c)
        if warm:
            msdata.warm(ms_path, field, spw, scan)

        try:
            norm, vmin, vmax = render.make_norm(matrix.chi2_dof, scale)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        colours = render.cell_colours(matrix.chi2_dof, norm)
        finite = matrix.finite
        # What the hover readout and keyboard navigation need, sent once
        # rather than as attributes on every cell.
        grid = {
            "names": matrix.antenna_names,
            "values": [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in matrix.chi2_dof],
            "positions": msdata.antenna_positions(ms_path, matrix.antenna_names),
            "stale": _stale(field, spw, scan),
            "where": {"field": field, "spw": spw, "scan": scan},
            "query": urlencode({"field": field, "spw": spw, "scan": scan, "c": c, "scale": scale}),
        }
        return templates.TemplateResponse(
            request,
            "_heatmap.html",
            {
                "matrix": matrix,
                "colours": colours,
                "names": matrix.antenna_names,
                "nant": len(matrix.antenna_names),
                "vmin": vmin,
                "vmax": vmax,
                "scale": scale,
                "nvalid": matrix.nbaselines,
                "worst": float(np.nanmax(matrix.chi2_dof)) if finite.size else float("nan"),
                "median": float(np.median(finite)) if finite.size else float("nan"),
                "query": urlencode({"field": field, "spw": spw, "scan": scan, "c": c}),
                "scales": render.SCALES,
                "product": c,
                "grid": grid,
                "stale": grid["stale"],
                # The diagonal plots auto-correlations; unknown (None) for an older zarr, so offered.
                "autos": matrix.autos is not False,
            },
        )

    @app.get("/histogram.png")
    @profiling.cprofiled
    def histogram_png(
        field: int = Query(...),
        spw: int = Query(...),
        scan: int = Query(...),
        c: str = Query(...),
        scale: str = Query("log"),
    ):
        matrix = _matrix(field, spw, scan, c)
        try:
            norm, _, _ = render.make_norm(matrix.chi2_dof, scale)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        with profiling.span("render histogram"):
            png = render.histogram_png(matrix.chi2_dof, norm)
        return Response(png, media_type="image/png")

    def _stale_mask(field: int, spw: int, scan: int) -> tuple[int, np.ndarray]:
        """Edits making cells of a scan stale, and the cells (``flagging.stale_cells``)."""
        names = list(store.group(field, spw, scan).attrs["antenna_names"])
        edits, mask = flagging.stale_cells(ms_path, field, spw, scan, store.cell_times(field, spw, scan), names)
        return edits, mask

    def _stale(field: int, spw: int, scan: int) -> dict:
        """What the heatmap shows about stale cells and recomputing them."""
        edits, mask = _stale_mask(field, spw, scan)
        attrs = store.attrs(field, spw, scan)
        reason = store.recomputable(field, spw, scan) if flagger is not None else "Recomputing needs --allow-flagging"
        cells = [[int(p), int(q)] for p, q in zip(*np.nonzero(np.triu(mask)), strict=True)]
        return {
            "edits": edits,
            "cells": cells,
            "recomputed": int(attrs.get("recomputed_cells", 0)),
            "can_recompute": reason is None,
            "why_not": reason,
        }

    def _has_auto(field: int, spw: int, scan: int, name: str) -> bool:
        """Whether the MS holds ``name``'s auto-correlation in this scan."""
        try:
            return msdata.baseline_rows(ms_path, field, spw, scan, name, name)[0].size > 0
        except KeyError:  # an antenna the MS's ANTENNA table does not name
            return False
        except Exception:  # noqa: BLE001 - an unreadable MS; check() reports it as for any baseline
            return True

    def _resolve(column: str) -> str:
        """MSv2 name for a column, as the menus list them, whichever name it was given by."""
        return msdata.msv2_name(column)

    def _waterfall_settings(matrix, product, column, model, gtable, flagversion, quantity, auto=False):
        """Fill unset waterfall controls from how chi2 formed the heatmap.

        ``None`` means "not given", so the heatmap's setting applies; an empty
        string is a deliberate "none" (no model, no gains). An
        auto-correlation (``auto``) has no heatmap value to match, and a
        model's auto-correlations mean nothing when hunting RFI, so its
        model defaults to none.
        """
        if model is None:
            model = "" if auto else matrix.mcol
        return {
            "product": product or matrix.selection.product,
            "column": _resolve(column or matrix.rcol),
            "model": _resolve(model),
            "gtable": matrix.gtable if gtable is None else gtable.strip(),
            "flagversion": flagversion or "",
            "quantity": quantity,
        }

    @app.get("/waterfall", response_class=HTMLResponse)
    @profiling.cprofiled
    def waterfall(
        request: Request,
        field: int = Query(...),
        spw: int = Query(...),
        scan: int = Query(...),
        c: str = Query(...),
        p: int = Query(..., ge=0),
        q: int = Query(..., ge=0),
        product: str = Query(""),
        column: str = Query(""),
        model: str | None = Query(None),
        gtable: str | None = Query(None),
        flagversion: str = Query(""),
        quantity: str = Query("amp"),
    ):
        """Return the controls and <img> wrapper; the browser then fetches the PNG.

        The controls drive the waterfall only. The heatmap is what ``chi2``
        computed, so the panel says when the two no longer describe the same
        quantity.
        """
        matrix = _matrix(field, spw, scan, c)
        name1, name2 = _baseline(matrix, p, q)
        auto = name1 == name2
        settings = _waterfall_settings(matrix, product, column, model, gtable, flagversion, quantity, auto)
        base = {"field": field, "spw": spw, "scan": scan, "c": c, "p": p, "q": q}
        offered = _gain_tables()
        if auto and matrix.autos is None and not _has_auto(field, spw, scan, name1):
            # Only a zarr from before chi2 recorded autos offers the diagonal without
            # them; when it says, the scan query (seconds on a large MS) is not needed.
            problems = [f"{name1} has no auto-correlation in scan {scan} of this Measurement Set"]
        elif not _allowed(settings["gtable"]):
            problems = [
                f"gain table {settings['gtable']!r} is not one this server offers; "
                "pass it to surfvis serve with --gtable, or put it in a *.qc directory next to the MS"
            ]
        else:
            problems = msdata.check(
                ms_path,
                field,
                spw,
                scan,
                settings["product"],
                settings["column"],
                model=settings["model"],
                gtable=settings["gtable"],
                flagversion=settings["flagversion"],
            )
        defaults = _waterfall_settings(matrix, "", "", None, None, "", "amp", auto)
        # Controls the user moved off the heatmap's setting. Clicking another
        # cell sends these along, so a choice survives a change of baseline.
        keep = {key for key, value in settings.items() if value != defaults[key]}
        return templates.TemplateResponse(
            request,
            "_waterfall.html",
            {
                "base": base,
                "query": urlencode({**base, **settings}),
                "name1": name1,
                "name2": name2,
                "auto": auto,
                "gtables": list(offered.values()),
                "value": float(matrix.chi2_dof[p, q]),
                # The column is corrected and has a model subtracted, so only
                # visibility-like columns make sense; WEIGHT does not.
                "columns": _columns(ms_path, field, spw, scan, complex_only=True),
                "models": _columns(ms_path, field, spw, scan, complex_only=True),
                "products": _products(ms_path, field, spw, scan),
                "flagversions": msdata.flag_versions(ms_path),
                "quantities": msdata.QUANTITIES,
                "s": settings,
                "matrix": matrix,
                "keep": keep,
                # The heatmap measured nothing on its diagonal, so there is nothing to differ from.
                "differs": not auto and bool(keep - {"quantity"}),
                "problems": problems,
                # What a selection drawn on this panel is posted with (see /flag).
                "flagging": flagger is not None,
                # The antennas a selection's scope can widen to, as (index, name).
                "scope_antennas": list(dict.fromkeys((matrix.antenna_names.index(n), n) for n in (name1, name2))),
                "flag_context": {
                    "field": field,
                    "spw": spw,
                    "scan": scan,
                    "p": p,
                    "q": q,
                    "product": settings["product"],
                    "gtable": settings["gtable"],
                    "flagversion": settings["flagversion"],
                },
            },
        )

    @app.get("/waterfall.png")
    @profiling.cprofiled
    def waterfall_png(
        field: int = Query(...),
        spw: int = Query(...),
        scan: int = Query(...),
        c: str = Query(...),
        p: int = Query(..., ge=0),
        q: int = Query(..., ge=0),
        product: str = Query(""),
        column: str = Query(""),
        model: str | None = Query(None),
        gtable: str | None = Query(None),
        flagversion: str = Query(""),
        quantity: str = Query("amp"),
    ):
        matrix = _matrix(field, spw, scan, c)
        name1, name2 = _baseline(matrix, p, q)
        settings = _waterfall_settings(matrix, product, column, model, gtable, flagversion, quantity, name1 == name2)
        if not _allowed(settings["gtable"]):
            raise HTTPException(status_code=403, detail="That gain table is not one this server offers.")
        try:
            wf = msdata.waterfall(
                ms_path,
                field=field,
                spw=spw,
                scan=scan,
                name1=name1,
                name2=name2,
                rect=None,
                **settings,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        with profiling.span("render png"):
            png = render.waterfall_png(wf)
        return Response(png, media_type="image/png")

    @app.get("/waterfall.bin")
    @profiling.cprofiled
    def waterfall_bin(
        field: int = Query(...),
        spw: int = Query(...),
        scan: int = Query(...),
        c: str = Query(...),
        p: int = Query(..., ge=0),
        q: int = Query(..., ge=0),
        product: str = Query(""),
        column: str = Query(""),
        model: str | None = Query(None),
        gtable: str | None = Query(None),
        flagversion: str = Query(""),
        quantity: str = Query("amp"),
        t0: int = Query(0),
        t1: int | None = Query(None),
        c0: int = Query(0),
        c1: int | None = Query(None),
        max_t: int = Query(1024),
        max_c: int = Query(1024),
        meta: bool = Query(False),
    ):
        """A window of one baseline's waterfall, reduced to at most max_t x max_c bins, for the canvas."""
        if quantity not in msdata.QUANTITIES:
            raise HTTPException(status_code=400, detail=f"quantity must be one of {msdata.QUANTITIES}")
        matrix = _matrix(field, spw, scan, c)
        name1, name2 = _baseline(matrix, p, q)
        settings = _waterfall_settings(matrix, product, column, model, gtable, flagversion, quantity, name1 == name2)
        settings.pop("quantity")
        if not _allowed(settings["gtable"]):
            raise HTTPException(status_code=403, detail="That gain table is not one this server offers.")
        try:
            wf = msdata.cached_plane(
                ms_path,
                field,
                spw,
                scan,
                name1,
                name2,
                **settings,
            )
        except (KeyError, IndexError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        ntime, nchan = wf.plane.shape
        t0, t1 = max(t0, 0), min(ntime if t1 is None else t1, ntime)
        c0, c1 = max(c0, 0), min(nchan if c1 is None else c1, nchan)
        if t0 >= t1 or c0 >= c1:
            raise HTTPException(status_code=400, detail=f"Empty window: slots {t0}-{t1}, channels {c0}-{c1}")
        cap = planes.MAX_BINS
        with profiling.span("reduce_block"):
            block = planes.reduce_block(
                wf.plane, wf.flags, quantity, t0, t1, c0, c1, min(max(max_t, 1), cap), min(max(max_c, 1), cap)
            )
        if quantity not in wf.limits:  # cached with the plane; colours do not move on zoom
            with profiling.span("colour_limits"):
                wf.limits[quantity] = planes.colour_limits(wf.plane, wf.flags, quantity)
        vmin, vmax = wf.limits[quantity]
        if wf.antenna1 == wf.antenna2:
            baseline = f"{wf.antenna1} auto"
        else:
            baseline = f"{wf.antenna1}-{wf.antenna2}" + (f" ({wf.length:.0f} m)" if wf.length is not None else "")
        header = {
            "t0": t0,
            "t1": t1,
            "c0": c0,
            "c1": c1,
            "tstep": block.tstep,
            "cstep": block.cstep,
            "ntime": ntime,
            "nchan": nchan,
            "vmin": vmin,
            "vmax": vmax,
            "quantity": quantity,
            "colormap": "twilight" if quantity == "phase" else "viridis",
            "title": f"{baseline}   scan {wf.scan}   {wf.column}   {quantity}   {wf.product}",
        }
        if meta:
            header["times"] = [float(x) for x in wf.times] if wf.times is not None else None
            header["freqs"] = [float(x) for x in wf.freqs] if wf.freqs is not None else None
        with profiling.span("encode"):
            body = planes.encode(header, block)
        return Response(
            body,
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    flagger = Flagger(ms_path) if allow_flagging else None
    if flagger is not None:
        # Before the backup, so the backup is of a consistent FLAG.
        if flagger.recover():
            print(f"surfvis: {flagger.recovered}", file=sys.stderr, flush=True)
        # Copying FLAG takes minutes on a large MS; do it now, not inside the first /flag request.
        flagger.start_backup()
    elif (journal := flagging.journal_path(ms_path)) is not None:
        print(
            f"surfvis: a flag write to {ms_path} was interrupted (journal at {journal}); "
            "restart with --allow-flagging to roll it back",
            file=sys.stderr,
            flush=True,
        )

    def _flagger() -> Flagger:
        if flagger is None:
            raise HTTPException(status_code=403, detail="Flagging is disabled; start serve with --allow-flagging")
        return flagger

    @app.post("/flag")
    @profiling.cprofiled
    def flag(req: FlagRequest):
        """Flag the cells inside a polygon on one baseline, or count them with ``dry_run``."""
        flagger_ = _flagger()
        if req.flagversion:
            raise HTTPException(
                status_code=409,
                detail=f"This waterfall shows flag version {req.flagversion!r}, not FLAG; "
                "switch flags to FLAG (current) to flag.",
            )
        try:
            group = store.group(req.field, req.spw, req.scan)
        except (KeyError, IndexError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        gtable = str(group.attrs.get("gtable", "")) if req.gtable is None else req.gtable.strip()
        if not _allowed(gtable):
            raise HTTPException(status_code=403, detail="That gain table is not one this server offers.")
        try:
            names = list(group.attrs["antenna_names"])
            labels = msdata.labels(ms_path, req.field, req.spw, req.scan)
            if req.correlations == "all":
                corr_idx = list(range(len(labels)))
            else:
                corr_idx = msdata.product_correlations(ms_path, req.field, req.spw, req.scan, req.product, gtable)
            antenna = None
            if req.baselines == "antenna":
                if req.antenna not in (req.p, req.q):
                    raise ValueError("The antenna scope must name one of the panel's two antennas")
                antenna = names[req.antenna]
            result = flagger_.apply(
                req.field,
                req.spw,
                req.scan,
                names[req.p],
                names[req.q],
                np.asarray(req.vertices, dtype=float),
                corr_idx,
                dry_run=req.dry_run,
                scope=req.baselines,
                antenna=antenna,
                product=req.product,
            )
            return {**result, "correlations": [labels[i] for i in corr_idx]}
        except BackupPendingError as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "10"}) from exc
        except (KeyError, IndexError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/flag/undo")
    def flag_undo(req: UndoRequest):
        """Restore the cells the named edit changed, if it is still the most recent."""
        try:
            edit = _flagger().undo(req.id)
        except StaleUndoError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if edit is None:
            raise HTTPException(status_code=409, detail="Nothing to undo")
        return {"undone": edit.id, **edit.summary()}

    @app.get("/stale")
    def stale(field: int = Query(...), spw: int = Query(...), scan: int = Query(...)):
        """Which baselines of a scan have edits newer than its χ², from the audit log."""
        try:
            return _stale(field, spw, scan)
        except (KeyError, IndexError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # One recompute at a time per server, in a background thread: an "all
    # baselines" edit means re-reading the whole scan, minutes on a large MS.
    job: dict = {"state": "idle"}
    job_lock = threading.Lock()

    def _run_recompute(field: int, spw: int, scan: int, pairs: list[tuple[int, int]], started: float) -> None:

        def progress(done: int, total: int) -> None:
            job.update(rows_done=done, rows_total=total)

        try:
            attrs = store.attrs(field, spw, scan)
            products = store.products(field, spw, scan)
            chi2, counts = recompute.cells(ms_path, attrs, products, field, spw, scan, pairs, progress)
            store.write_cells(field, spw, scan, pairs, chi2, counts, started, ms=ms_path)
            job.update(state="done", finished=time.time())
        except Exception as exc:  # noqa: BLE001 - reported through /recompute/state
            job.update(state="failed", error=f"{type(exc).__name__}: {exc}", finished=time.time())

    @app.post("/recompute")
    def recompute_start(req: RecomputeRequest):
        """Recompute a scan's stale cells by chi2's recipe and write them into the zarr, in the background."""
        _flagger()  # writing the zarr is writing too: --allow-flagging
        try:
            reason = store.recomputable(req.field, req.spw, req.scan)
            if reason is not None:
                raise HTTPException(status_code=409, detail=reason)
            _, mask = _stale_mask(req.field, req.spw, req.scan)
        except (KeyError, IndexError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        pairs = [(int(p), int(q)) for p, q in zip(*np.nonzero(np.triu(mask)), strict=True)]
        if not pairs:
            raise HTTPException(status_code=409, detail="Nothing in this scan is stale")
        with job_lock:
            if job.get("state") == "running":
                raise HTTPException(status_code=409, detail="A recompute is already running")
            # Before FLAG is read, and never inside a flag write or undo (which
            # hold the lock until their audit line is stamped): edits logged
            # after this stay stale.
            with flagger._lock:
                started = time.time()
            job.clear()
            job.update(
                id=time.time_ns() // 1000,  # microseconds: exact as a JS number
                state="running",
                field=req.field,
                spw=req.spw,
                scan=req.scan,
                cells=len(pairs),
                rows_done=0,
                rows_total=0,
            )
        threading.Thread(
            target=_run_recompute,
            args=(req.field, req.spw, req.scan, pairs, started),
            name="surfvis-recompute",
            daemon=True,
        ).start()
        return dict(job)

    @app.get("/recompute/state")
    def recompute_state():
        return dict(job)

    @app.get("/flag/state")
    def flag_state():
        if flagger is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "backup": flagger.backup,
            "backup_state": flagger.backup_state,
            "backup_error": flagger.backup_error,
            "recovered": flagger.recovered,
            "broken": flagger.broken,
            "edits": [e.summary() for e in flagger.edits],
        }

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "scans": len(store.scans), "ms": ms_path, "flagging": flagger is not None}

    return app


def _static_version(root: Path) -> str:
    """Short digest of the static files' names, sizes and mtimes."""
    stats = sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in root.iterdir() if p.is_file())
    return hashlib.sha1(repr(stats).encode()).hexdigest()[:10]


def _columns(ms: str, field: int, spw: int, scan: int, complex_only: bool = False) -> list[str]:
    """Plottable columns, or an empty list if the MS cannot be opened."""
    try:
        return msdata.available_columns(ms, field, spw, scan, complex_only=complex_only)
    except Exception:
        return []


def _products(ms: str, field: int, spw: int, scan: int) -> list[str]:
    """Correlations and formable Stokes parameters, or an empty list if the MS cannot be opened."""
    try:
        return msdata.available_products(ms, field, spw, scan)
    except Exception:
        return []
