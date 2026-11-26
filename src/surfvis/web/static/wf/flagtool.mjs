// Flagging from the waterfall (serve --allow-flagging): draw a selection, see
// the cells it covers and how many it would newly flag, confirm, undo.
//
// The gestures live in selection.mjs, the cell rule in select.mjs (the
// server's rasterise, ported); this file wires them to the canvas, to /flag
// and to the buttons. The server's mask is what gets written: the shading
// is a preview and the counts come from a dry run.

import { rasterise } from "./select.mjs";
import { Selection, confirmLabel, undoLabel, sameBaseline } from "./selection.mjs";

const FILL = "rgba(224, 48, 150, 0.35)";
const LINE = "#e03096";
const BACKUP_POLL_MS = 3000;

// A pending selection outlives its panel: changing baseline within a scan
// carries it to the next waterfall (same grid), so the same patch can be
// flagged on several baselines with arrow, Enter, arrow, Enter. A new scan
// discards it. Module state, since every panel imports this one module.
let carried = null; // {scan: "field/spw/scan", vertices}

const scanKey = (ctx) => `${ctx.field}/${ctx.spw}/${ctx.scan}`;

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data = null;
  try {
    data = await res.json();
  } catch (_) {}
  if (!res.ok) {
    let detail = data?.detail ?? `HTTP ${res.status}`;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg).join("; ");
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return data;
}

export function attach(root, api) {
  const ctx = JSON.parse(root.dataset.flag);
  const [name1, name2] = JSON.parse(root.dataset.names);
  const canvas = api.canvas;
  const view = api.view;
  const sel = new Selection(view.ntime, view.nchan);
  const allCorr = root.querySelector(".wf-allcorr");
  const applyBtn = root.querySelector(".wf-apply");
  const undoBtn = root.querySelector(".wf-undo");
  const status = root.querySelector(".wf-flagstatus");
  const life = api.life;

  let spans = null; // the pending selection's cells
  let dry = null; // the server's count for it
  let dryFor = 0; // which dry run is the latest
  let busy = false; // a write in flight
  let backupReady = false;
  let disabled = null; // why flagging cannot happen here, or null
  let latest = null; // the most recent edit on the server
  let pollTimer = null;

  const say = (text, error = false) => {
    status.textContent = text || "";
    status.classList.toggle("error", error);
  };

  if (ctx.flagversion) {
    disabled = `Showing flag version ${ctx.flagversion}: switch flags to FLAG (current) to flag.`;
  }

  // Data and screen coordinates of a pointer event.
  const point = (e) => ({ ...api.toData(e.offsetX, e.offsetY), x: e.offsetX, y: e.offsetY });

  function body(extra = {}) {
    return {
      ...ctx,
      vertices: sel.vertices,
      correlations: allCorr.checked ? "all" : "product",
      ...extra,
    };
  }

  function refreshButtons() {
    const pending = sel.mode === "pending" && !disabled;
    applyBtn.hidden = !pending;
    if (pending) {
      applyBtn.textContent = dry ? confirmLabel(dry) : "Counting cells…";
      applyBtn.disabled = busy || !dry || !dry.newly_flagged || !backupReady;
      applyBtn.title = backupReady ? "Enter" : "Waiting for the backup flag version";
    }
    undoBtn.hidden = !latest;
    if (latest) {
      undoBtn.textContent = undoLabel(latest);
      undoBtn.disabled = busy;
    }
  }

  // A selection just closed: shade it, count it, remember it for the next baseline.
  async function onPending() {
    spans = rasterise(sel.vertices, view.ntime, view.nchan);
    dry = null;
    carried = { scan: scanKey(ctx), vertices: sel.vertices.map((v) => [...v]) };
    refreshButtons();
    api.redraw();
    if (!spans.length) {
      say("The selection covers no cells.");
      return;
    }
    const id = ++dryFor;
    try {
      const out = await postJSON("/flag", body({ dry_run: true }));
      if (id !== dryFor || sel.mode !== "pending") return;
      dry = out;
      say(backupReady ? "Enter to flag, Esc to cancel." : "");
    } catch (err) {
      if (id === dryFor) say(err.message, true);
    }
    refreshButtons();
  }

  function onCleared() {
    spans = null;
    dry = null;
    dryFor++;
    refreshButtons();
    api.redraw();
  }

  async function apply() {
    if (busy || sel.mode !== "pending" || !dry?.newly_flagged || !backupReady || disabled) return;
    busy = true;
    refreshButtons();
    say("Flagging…");
    try {
      const out = await postJSON("/flag", body());
      latest = out;
      sel.clear();
      spans = null;
      dry = null;
      say(`Flagged ${out.newly_flagged.toLocaleString("en-US")} new cells.`);
      await api.reload();
    } catch (err) {
      say(err.message, true);
      if (err.status === 503) pollBackup();
    } finally {
      busy = false;
      refreshButtons();
      api.redraw();
    }
  }

  async function undo() {
    if (busy || !latest) return;
    busy = true;
    refreshButtons();
    const edit = latest;
    say("Undoing…");
    try {
      await postJSON("/flag/undo", { id: edit.id });
      say(`Undid ${edit.description}.`);
      if (sameBaseline(edit, ctx, name1, name2)) await api.reload();
    } catch (err) {
      say(err.message, true);
    } finally {
      busy = false;
      await loadState();
    }
  }

  async function loadState() {
    try {
      const res = await fetch("/flag/state");
      const state = await res.json();
      if (!state.enabled) {
        disabled = "Flagging is disabled on this server.";
      } else if (state.broken) {
        disabled = state.broken;
      } else if (state.backup_state === "failed") {
        disabled = `No backup flag version, so flagging is disabled: ${state.backup_error}`;
      }
      backupReady = state.backup_state === "done";
      latest = state.edits?.length ? state.edits[state.edits.length - 1] : null;
      if (disabled) say(disabled, true);
      else if (!backupReady) {
        say("Saving backup flag version… (you can draw; flagging waits for it)");
        pollBackup();
      }
    } catch (err) {
      say(`Could not read the flagging state: ${err.message}`, true);
    }
    refreshButtons();
  }

  function pollBackup() {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(async () => {
      if (!root.isConnected) return;
      await loadState();
      if (backupReady && !disabled) say(sel.mode === "pending" ? "Backup saved. Enter to flag." : "Backup saved.");
    }, BACKUP_POLL_MS);
  }

  // Drawing: the pending selection's cells, then outlines and vertices.
  api.addLayer((g) => {
    const win = api.window;
    if (spans && win) {
      g.fillStyle = FILL;
      for (const [i, a, b] of spans) {
        if (i + 0.5 < win.t0 || i - 0.5 > win.t1 || b - 0.5 < win.c0 || a - 0.5 > win.c1) continue;
        const p = api.toScreen(i - 0.5, a - 0.5);
        const q = api.toScreen(i + 0.5, b - 0.5);
        g.fillRect(p.x, q.y, q.x - p.x, p.y - q.y);
      }
    }
    const outline = sel.mode === "dragging" ? sel.dragRect : sel.vertices;
    if (!outline || !outline.length) return;
    g.strokeStyle = LINE;
    g.lineWidth = 1.5;
    g.setLineDash(sel.mode === "pending" ? [] : [5, 3]);
    g.beginPath();
    outline.forEach(([t, c], k) => {
      const p = api.toScreen(t, c);
      if (k) g.lineTo(p.x, p.y);
      else g.moveTo(p.x, p.y);
    });
    if (sel.mode === "drawing" && sel.cursor) g.lineTo(sel.cursor.x, sel.cursor.y);
    if (sel.mode !== "drawing") g.closePath();
    g.stroke();
    g.setLineDash([]);
    if (sel.mode === "drawing") {
      g.fillStyle = LINE;
      sel.vertices.forEach(([t, c], k) => {
        const p = api.toScreen(t, c);
        g.beginPath();
        g.arc(p.x, p.y, k === 0 ? 4 : 2.5, 0, 2 * Math.PI);
        g.fill();
      });
    }
  });

  // Input. Plain left button only: shift- and middle-drag are the widget's pan.
  const ours = (e) => e.button === 0 && !e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey;
  canvas.addEventListener(
    "pointerdown",
    (e) => {
      if (disabled || busy || !ours(e) || !api.inPlot(e.offsetX, e.offsetY)) return;
      e.preventDefault();
      canvas.setPointerCapture(e.pointerId);
      const wasPending = sel.mode === "pending";
      sel.down(point(e));
      if (wasPending) onCleared();
    },
    { signal: life },
  );
  canvas.addEventListener(
    "pointermove",
    (e) => {
      if (sel.mode === "idle" && !sel.press) return;
      sel.move(point(e));
      api.redraw();
    },
    { signal: life },
  );
  canvas.addEventListener(
    "pointerup",
    (e) => {
      if (!sel.press || e.button !== 0) return;
      const what = sel.up(point(e), api.toScreen);
      if (what === "pending") onPending();
      api.redraw();
    },
    { signal: life },
  );
  canvas.addEventListener(
    "dblclick",
    (e) => {
      if (disabled) return;
      e.preventDefault();
      if (sel.doubleClick(api.toScreen) === "pending") onPending();
      api.redraw();
    },
    { signal: life },
  );
  document.addEventListener(
    "keydown",
    (e) => {
      if (!root.isConnected || e.target.closest("input, select, textarea, button")) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && !e.shiftKey) {
        if (sel.mode === "drawing" || sel.mode === "dragging") return;
        e.preventDefault();
        undo();
        return;
      }
      const what = sel.key(e.key);
      if (!what) return;
      e.preventDefault();
      if (what === "pending") onPending();
      else if (what === "apply") apply();
      else if (what === "cleared") {
        carried = null;
        onCleared();
        say("");
      }
      api.redraw();
    },
    { signal: life },
  );
  applyBtn.addEventListener("click", apply, { signal: life });
  undoBtn.addEventListener("click", undo, { signal: life });
  allCorr.addEventListener(
    "change",
    () => {
      if (sel.mode === "pending") onPending();
    },
    { signal: life },
  );
  life.addEventListener("abort", () => clearTimeout(pollTimer));

  loadState().then(() => {
    // A selection carried over from the previous baseline of this scan.
    if (!disabled && carried && carried.scan === scanKey(ctx)) {
      sel.setPending(carried.vertices);
      onPending();
    } else if (carried && carried.scan !== scanKey(ctx)) {
      carried = null;
    }
  });
}
