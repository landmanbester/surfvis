// Which heatmap cells manual edits have made stale, from the server's /stale
// summary of the audit log. Tested in tests/js/stale.test.mjs.
//
// stale: {edits, all, antennas: [name], baselines: [[name, name]]}
// names: the heatmap's antenna names, in grid order.

// The cells to mark, as "p,q" keys of the grid (both triangles: it is symmetric).
export function staleCells(stale, names) {
  const cells = new Set();
  if (!stale || !stale.edits) return cells;
  const n = names.length;
  const index = new Map(names.map((name, k) => [name, k]));
  const both = (p, q) => {
    cells.add(`${p},${q}`);
    cells.add(`${q},${p}`);
  };
  if (stale.all) {
    for (let p = 0; p < n; p++) for (let q = 0; q < n; q++) cells.add(`${p},${q}`);
    return cells;
  }
  for (const name of stale.antennas || []) {
    const p = index.get(name);
    if (p === undefined) continue;
    for (let q = 0; q < n; q++) both(p, q);
  }
  for (const [a, b] of stale.baselines || []) {
    const p = index.get(a);
    const q = index.get(b);
    if (p !== undefined && q !== undefined) both(p, q);
  }
  return cells;
}

// The heatmap header's note, or "" when nothing is stale.
export function staleNote(stale) {
  if (!stale || !stale.edits) return "";
  const s = stale.edits === 1 ? "" : "s";
  return `χ² predates ${stale.edits} manual edit${s} in this scan; marked cells are stale (re-run chi2)`;
}
