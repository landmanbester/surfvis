import { test } from "node:test";
import assert from "node:assert/strict";

import { jobText, recomputedNote, recomputeLabel, staleCells, staleNote } from "../../src/surfvis/web/static/stale.mjs";

const stale = (over = {}) => ({ edits: 1, cells: [[1, 3]], recomputed: 0, can_recompute: true, why_not: null, ...over });

test("nothing stale marks nothing and offers nothing", () => {
  const none = stale({ edits: 0, cells: [] });
  assert.equal(staleCells(none).size, 0);
  assert.equal(staleCells(null).size, 0);
  assert.equal(staleNote(none), "");
  assert.equal(recomputeLabel(none), "");
});

test("cells are marked in both triangles", () => {
  assert.deepEqual([...staleCells(stale({ cells: [[1, 3], [0, 2]] }))].sort(), ["0,2", "1,3", "2,0", "3,1"]);
});

test("the note counts edits, and says to re-run chi2 only when the server cannot recompute", () => {
  assert.match(staleNote(stale()), /1 manual edit in this scan; marked cells are stale$/);
  assert.match(staleNote(stale({ edits: 3, can_recompute: false })), /3 manual edits .*\(re-run chi2\)$/);
});

test("the button counts cells and is absent when recomputing is not possible", () => {
  assert.equal(recomputeLabel(stale({ cells: [[0, 1], [0, 2]] })), "Recompute 2 stale cells");
  assert.equal(recomputeLabel(stale()), "Recompute 1 stale cell");
  assert.equal(recomputeLabel(stale({ can_recompute: false })), "");
});

test("provenance and progress", () => {
  assert.equal(recomputedNote(stale({ recomputed: 12 })), "12 cells recomputed by serve since chi2");
  assert.equal(recomputedNote(stale()), "");
  assert.equal(jobText({ state: "running", cells: 5, rows_done: 0, rows_total: 0 }), "Recomputing 5 cells…");
  assert.equal(jobText({ state: "running", cells: 5, rows_done: 30, rows_total: 120 }), "Recomputing 5 cells… 25%");
  assert.equal(jobText({ state: "failed", error: "OSError: x" }), "Recompute failed: OSError: x");
  assert.equal(jobText({ state: "idle" }), "");
});
