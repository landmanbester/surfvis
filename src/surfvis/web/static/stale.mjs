// What the heatmap shows about stale cells, from the server's /stale
// summary (also embedded in #grid-data). Tested in tests/js/stale.test.mjs.
//
// stale: {edits, cells: [[p, q]] (upper triangle), recomputed, can_recompute, why_not}

// The cells to mark, as "p,q" keys of the grid (both triangles: it is symmetric).
export function staleCells(stale) {
  const cells = new Set();
  if (!stale || !stale.edits) return cells;
  for (const [p, q] of stale.cells || []) {
    cells.add(`${p},${q}`);
    cells.add(`${q},${p}`);
  }
  return cells;
}

const plural = (n, word) => `${n.toLocaleString("en-US")} ${word}${n === 1 ? "" : "s"}`;

// The header's note, or "" when nothing is stale.
export function staleNote(stale) {
  if (!stale || !stale.edits) return "";
  const why = stale.can_recompute ? "" : " (re-run chi2)";
  return `χ² predates ${plural(stale.edits, "manual edit")} in this scan; marked cells are stale${why}`;
}

// The recompute button's label, or "" when there is nothing it can do.
export function recomputeLabel(stale) {
  if (!stale || !stale.edits || !stale.can_recompute || !stale.cells?.length) return "";
  return `Recompute ${plural(stale.cells.length, "stale cell")}`;
}

// The provenance line, or "".
export function recomputedNote(stale) {
  return stale?.recomputed ? `${plural(stale.recomputed, "cell")} recomputed by serve since chi2` : "";
}

// Progress of a /recompute/state job, in words.
export function jobText(job) {
  if (!job || job.state === "idle") return "";
  if (job.state === "failed") return `Recompute failed: ${job.error}`;
  if (job.state === "done") return "Recomputed.";
  if (!job.rows_total) return `Recomputing ${plural(job.cells, "cell")}…`;
  return `Recomputing ${plural(job.cells, "cell")}… ${Math.floor((100 * job.rows_done) / job.rows_total)}%`;
}
