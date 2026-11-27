import { Carry } from "./carry.mjs";
import { jobText, recomputedNote, recomputeLabel, staleCells, staleNote } from "./stale.mjs";

// Behaviour htmx does not cover: loading feedback, the selected baseline,
// the hover readout, keyboard navigation, and carrying waterfall settings
// from one baseline to the next.

// The selected baseline, by antenna names rather than grid indices: a
// different scan can hold a different set of antennas.
let selected = null;

function gridData() {
  const node = document.getElementById("grid-data");
  return node ? JSON.parse(node.textContent) : null;
}

function cellFor(names) {
  const data = gridData();
  if (!data || !names) return null;
  const p = data.names.indexOf(names[0]);
  const q = data.names.indexOf(names[1]);
  if (p < 0 || q < 0) return null;
  // Null for the diagonal unless it plots auto-correlations: only then does it carry data-pq.
  return document.querySelector(`.grid i[data-pq="${p},${q}"]`);
}

function pq(cell) {
  return cell.dataset.pq.split(",").map(Number);
}

function markSelected() {
  document.querySelectorAll(".grid i.selected").forEach((c) => c.classList.remove("selected"));
  cellFor(selected)?.classList.add("selected");
}

// Dim whatever a request is about to replace, so a click visibly did something.
document.body.addEventListener("htmx:beforeRequest", (e) => e.detail.target.classList.add("loading"));
document.body.addEventListener("htmx:afterRequest", (e) => e.detail.target.classList.remove("loading"));

// A cell click carries the waterfall controls the user moved off the
// heatmap's defaults, so a choice survives a change of baseline (carry.mjs).
const carry = new Carry();
document.body.addEventListener("htmx:afterSettle", (e) => {
  if (e.detail.target.id !== "waterfall") return;
  const panel = e.detail.target.querySelector(".panel");
  if (!panel) return carry.reset();
  const keep = Object.fromEntries([...panel.querySelectorAll(".keep")].map((el) => [el.name, el.value]));
  carry.settled(keep, Boolean(panel.dataset.auto));
});

document.body.addEventListener("change", (e) => {
  const panel = e.target.closest("#waterfall .panel");
  if (panel && e.target.name) carry.changed(e.target.name, Boolean(panel.dataset.auto));
});

document.body.addEventListener("htmx:configRequest", (e) => {
  if (!e.detail.elt.matches(".grid i[data-pq]")) return;
  Object.assign(e.detail.parameters, carry.params);
});

document.body.addEventListener("click", (e) => {
  const cell = e.target.closest(".grid i[data-pq]");
  if (!cell) return;
  const data = gridData();
  const [p, q] = pq(cell);
  selected = [data.names[p], data.names[q]];
  markSelected();
});

document.body.addEventListener("htmx:afterSettle", (e) => {
  if (e.detail.target.id !== "heatmap") return;
  markSelected();
  // A redraw after a recompute: the same scan, so the waterfall stays as it is.
  if (refreshing) {
    refreshing = false;
    return;
  }
  // A new field, scan, bin or product: follow the selected baseline into it,
  // rather than leave a waterfall from somewhere else on screen.
  if (!selected || !document.querySelector("#waterfall .panel")) return;
  const cell = cellFor(selected);
  if (cell) {
    cell.click();
  } else {
    const hint = document.createElement("p");
    hint.className = "hint";
    const what = selected[0] === selected[1] ? `${selected[0]}'s auto-correlation` : `${selected[0]} \u00d7 ${selected[1]}`;
    hint.textContent = `${what} has no data here. Click a cell to plot another baseline.`;
    document.getElementById("waterfall").replaceChildren(hint);
    carry.reset(); // no panel, so nothing to carry, as before a first click
  }
});

// Cells whose chi-squared predates manual edits get a marker, and the header
// says how many and offers to recompute them (stale.mjs). Rendered with the
// heatmap, and fetched again whenever the waterfall applies or undoes an edit.
function markStale(stale) {
  const data = gridData();
  if (!data) return;
  const cells = staleCells(stale);
  const n = data.names.length;
  document.querySelectorAll(".grid i").forEach((cell, k) => {
    cell.classList.toggle("stale", cells.has(`${Math.floor(k / n)},${k % n}`));
  });
  const note = document.getElementById("stale-note");
  if (note) {
    note.querySelector(".stale-text").textContent = staleNote(stale);
    note.hidden = !staleNote(stale);
    const button = document.getElementById("recompute");
    button.textContent = recomputeLabel(stale);
    button.hidden = !button.textContent;
  }
  const done = document.getElementById("recomputed");
  if (done) {
    done.textContent = recomputedNote(stale);
    done.hidden = !done.textContent;
  }
}

const sameScan = (a, b) => a && b && a.field === b.field && a.spw === b.spw && a.scan === b.scan;

async function refreshStale() {
  const where = gridData()?.where;
  if (!where) return;
  try {
    const res = await fetch(`/stale?${new URLSearchParams(where)}`);
    if (!res.ok) return;
    const stale = await res.json();
    // Only if the heatmap on screen is still the one asked about.
    if (sameScan(gridData()?.where, where)) markStale(stale);
  } catch (_) {}
}

document.body.addEventListener("htmx:afterSettle", (e) => {
  if (e.detail.target.id === "heatmap") markStale(gridData()?.stale);
});
document.addEventListener("surfvis:flagged", refreshStale);

// Recompute: start the server's background job, show its progress, and
// redraw the heatmap when it is done (without re-requesting the waterfall).
let refreshing = false;
function recomputeStatus(text, error = false) {
  const status = document.querySelector("#stale-note .recompute-status");
  if (!status) return;
  status.textContent = text;
  status.classList.toggle("error", error);
}

async function pollRecompute(where) {
  for (;;) {
    await new Promise((r) => setTimeout(r, 1000));
    let job;
    try {
      job = await (await fetch("/recompute/state")).json();
    } catch (err) {
      return recomputeStatus(`Lost track of the recompute: ${err.message}`, true);
    }
    if (!sameScan(gridData()?.where, where)) return; // the user moved on; the job carries on
    recomputeStatus(jobText(job), job.state === "failed");
    if (job.state === "failed") return;
    if (job.state === "done") {
      const query = gridData()?.query;
      if (!query) return;
      refreshing = true;
      htmx.ajax("GET", `/heatmap?${query}`, { target: "#heatmap" });
      return;
    }
  }
}

document.body.addEventListener("click", async (e) => {
  const button = e.target.closest("#recompute");
  if (!button) return;
  const where = gridData()?.where;
  if (!where) return;
  button.disabled = true;
  try {
    const res = await fetch("/recompute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(where),
    });
    const job = await res.json();
    if (!res.ok) {
      recomputeStatus(job.detail || `HTTP ${res.status}`, true);
      button.disabled = false;
      return;
    }
    recomputeStatus(jobText(job));
    pollRecompute(where);
  } catch (err) {
    recomputeStatus(err.message, true);
    button.disabled = false;
  }
});

// The waterfall panel is drawn by static/wf/widget.mjs; mount it whenever the panel is (re)rendered.
document.body.addEventListener("htmx:afterSettle", (e) => {
  if (e.detail.target.id !== "waterfall") return;
  const root = e.detail.target.querySelector(".wf[data-src]");
  if (!root) return;
  import(root.dataset.module).then((m) => m.mount(root));
});

// Hover: a tooltip at the cursor and the readout under the grid, both naming
// the antennas with their numbers (the grid index is the ANTENNA row).
function describe(cell) {
  const data = gridData();
  const [p, q] = pq(cell);
  const value = data.values[p][q];
  if (p === q) return [`${data.names[p]} (${p})`, "auto-correlation"];
  const names = `${data.names[p]} (${p}) \u00d7 ${data.names[q]} (${q})`;
  let stats = `\u03c7\u00b2/dof ${value === null ? "flagged" : value.toFixed(2)}`;
  if (data.positions) {
    const [a, b] = [data.positions[p], data.positions[q]];
    stats += ` \u00b7 ${Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]).toFixed(0)} m`;
  }
  return [names, stats];
}

let readoutDefault = null;
function showReadout(cell) {
  const readout = document.getElementById("readout");
  if (!readout) return;
  readoutDefault ??= readout.textContent;
  readout.textContent = cell ? describe(cell).join(" \u00b7 ") : readoutDefault;
}

document.body.addEventListener("mousemove", (e) => {
  const tip = document.getElementById("tip");
  const cell = e.target.closest(".grid i[data-pq]");
  showReadout(cell);
  if (!tip) return;
  if (!cell) {
    tip.hidden = true;
    return;
  }
  tip.textContent = describe(cell).join("\n");
  tip.hidden = false;
  // Below-right of the cursor, flipped when it would leave the viewport.
  const pad = 14;
  const { width, height } = tip.getBoundingClientRect();
  const x = e.clientX + pad + width > window.innerWidth ? e.clientX - pad - width : e.clientX + pad;
  const y = e.clientY + pad + height > window.innerHeight ? e.clientY - pad - height : e.clientY + pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
});

// Arrow keys move the selection: up/down change antenna1 (the row),
// left/right antenna2 (the column), skipping cells that plot nothing (the
// diagonal, unless the MS has auto-correlations).
document.addEventListener("keydown", (e) => {
  const steps = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
  if (!(e.key in steps) || e.target.closest("input, select, textarea")) return;
  // Mid-selection on the waterfall: moving on would drop the half-drawn shape.
  if (document.querySelector('#waterfall .wf[data-drawing="1"]')) return;
  const current = cellFor(selected);
  const data = gridData();
  if (!current || !data) return;
  e.preventDefault();
  const n = data.names.length;
  let [p, q] = pq(current);
  const [dp, dq] = steps[e.key];
  let next = null;
  for (let step = 0; step < n && !next; step++) {
    p = (p + dp + n) % n;
    q = (q + dq + n) % n;
    next = document.querySelector(`.grid i[data-pq="${p},${q}"]`);
  }
  if (!next || next === current) return;
  next.click();
  showReadout(next);
});

// PNGs are fetched here rather than by an <img src>, so a failure's JSON detail
// arrives with the response. Re-requesting the URL to read it would redo the
// whole (expensive) MS read.
async function loadPlot(img) {
  const wrap = img.parentElement;
  try {
    const response = await fetch(img.dataset.plot);
    if (!response.ok) {
      let detail = "Could not render this plot.";
      try {
        detail = (await response.json()).detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }
    const url = URL.createObjectURL(await response.blob());
    img.addEventListener("load", () => URL.revokeObjectURL(url), { once: true });
    img.src = url;
    await img.decode().catch(() => {});
    wrap.classList.remove("loading");
  } catch (err) {
    if (!wrap.isConnected) return; // replaced by a newer selection meanwhile
    const p = document.createElement("p");
    p.className = "problems";
    p.textContent = err.message;
    wrap.replaceWith(p);
  }
}

document.body.addEventListener("htmx:afterSettle", (e) => {
  e.detail.target.querySelectorAll("img[data-plot]:not([src])").forEach(loadPlot);
});

// Save the whole page as a PNG, including what is scrolled out of view, which
// a screenshot cannot do. html-to-image (vendored) renders the DOM through an
// SVG foreignObject, so the grid, labels and controls come out as shown.
function pageName() {
  const pick = (name) => {
    const el = document.querySelector(`#selectors select[name="${name}"]`);
    return el ? el.selectedOptions[0]?.textContent.trim() : "";
  };
  const parts = ["surfvis", pick("field"), `scan${pick("scan")}`, pick("c")];
  if (selected && document.querySelector("#waterfall .panel")) {
    parts.push(selected[0] === selected[1] ? `${selected[0]}-auto` : `${selected[0]}-${selected[1]}`);
  }
  return parts.filter(Boolean).join("_").replace(/[^A-Za-z0-9_.+-]+/g, "") + ".png";
}

async function savePng(button) {
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "Saving\u2026";
  try {
    // The library re-downloads every <img> it embeds; for the waterfall that
    // would mean reading the MS again. Hand it the already-decoded pixels.
    for (const img of document.querySelectorAll("img")) {
      if (!img.complete || !img.naturalWidth || img.src.startsWith("data:")) continue;
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext("2d").drawImage(img, 0, 0);
      img.dataset.src = img.src;
      img.src = canvas.toDataURL("image/png");
    }
    const body = document.body;
    const url = await htmlToImage.toPng(body, {
      backgroundColor: getComputedStyle(body).backgroundColor,
      width: body.scrollWidth,
      height: body.scrollHeight,
      pixelRatio: 2,
      skipFonts: true,
      filter: (node) => !["busy", "tip", "save-png"].includes(node.id),
    });
    const a = document.createElement("a");
    a.href = url;
    a.download = pageName();
    a.click();
  } catch (err) {
    alert(`Could not save the page: ${err}`);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

document.getElementById("save-png")?.addEventListener("click", (e) => savePng(e.currentTarget));
