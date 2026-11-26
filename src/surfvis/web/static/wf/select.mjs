// Which cells a selection covers, for the pending selection's shading.
//
// A port of surfvis.web.flagging.rasterise: a cell is covered when its
// centre (i, j) is inside the polygon by the even-odd rule, with half-open
// edges. The server's mask is what gets written; this must agree with it
// cell for cell, so the arithmetic is the same expression in the same order
// (tests/js/rasterise.test.mjs checks it against cases Python pins).

// Covered cells as row spans: [[slot, firstChannel, stopChannel], ...], stop
// exclusive, in increasing slot then channel order. Spans rather than a mask:
// a 32k-channel waterfall makes a full mask 15 MB for one stroke.
export function rasterise(vertices, ntime, nchan) {
  if (!Array.isArray(vertices) || vertices.length < 3) {
    throw new Error(`A polygon needs at least three (slot, channel) vertices, got ${vertices?.length ?? 0}`);
  }
  for (const [t, c] of vertices) {
    if (!Number.isFinite(t) || !Number.isFinite(c)) throw new Error("Polygon vertices must be finite");
  }
  const n = vertices.length;
  let tmin = Infinity;
  let tmax = -Infinity;
  for (const [t] of vertices) {
    tmin = Math.min(tmin, t);
    tmax = Math.max(tmax, t);
  }
  const lo = Math.max(Math.ceil(tmin), 0);
  const hi = Math.min(Math.ceil(tmax), ntime);
  const spans = [];
  for (let i = lo; i < hi; i++) {
    const xs = [];
    for (let k = 0; k < n; k++) {
      const [t0, c0] = vertices[k];
      const [t1, c1] = vertices[(k + 1) % n];
      // Edges spanning this row's centre line, half-open in time so a vertex
      // on the line is counted once.
      if (Math.min(t0, t1) <= i && i < Math.max(t0, t1)) {
        xs.push(c0 + ((i - t0) * (c1 - c0)) / (t1 - t0));
      }
    }
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const start = Math.max(Math.ceil(xs[k]), 0);
      const stop = Math.min(Math.ceil(xs[k + 1]), nchan);
      if (start < stop) spans.push([i, start, stop]);
    }
  }
  return merge(spans);
}

// Adjacent or overlapping spans on one row become one (Python's mask does not
// distinguish them, so neither may the comparison).
function merge(spans) {
  const out = [];
  for (const s of spans) {
    const last = out[out.length - 1];
    if (last && last[0] === s[0] && s[1] <= last[2]) last[2] = Math.max(last[2], s[2]);
    else out.push([...s]);
  }
  return out;
}

// Number of cells in a list of spans.
export function countCells(spans) {
  let n = 0;
  for (const [, a, b] of spans) n += b - a;
  return n;
}

// The four-vertex polygon around the cells a drag from (t0, c0) to (t1, c1)
// touches, in data coordinates: corners snap outwards to cell edges, so the
// rectangle covers every cell the drag passed over.
export function dragRectangle(a, b, ntime, nchan) {
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  const t0 = clamp(Math.round(Math.min(a.t, b.t)), 0, ntime - 1);
  const t1 = clamp(Math.round(Math.max(a.t, b.t)), 0, ntime - 1);
  const c0 = clamp(Math.round(Math.min(a.c, b.c)), 0, nchan - 1);
  const c1 = clamp(Math.round(Math.max(a.c, b.c)), 0, nchan - 1);
  return [
    [t0 - 0.5, c0 - 0.5],
    [t0 - 0.5, c1 + 0.5],
    [t1 + 0.5, c1 + 0.5],
    [t1 + 0.5, c0 - 0.5],
  ];
}
