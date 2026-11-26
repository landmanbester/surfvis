import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { rasterise, countCells, dragRectangle } from "../../src/surfvis/web/static/wf/select.mjs";

// Written and pinned by tests/test_rasterise.py from surfvis.web.flagging.rasterise,
// so the pending selection's shading is the server's mask, cell for cell.
const cases = JSON.parse(readFileSync(new URL("./rasterise_cases.json", import.meta.url), "utf8"));

test("agrees with the Python rasteriser on every pinned case", () => {
  assert.ok(cases.length > 40);
  for (const c of cases) {
    assert.deepEqual(rasterise(c.vertices, c.ntime, c.nchan), c.spans, c.name);
  }
});

test("counts cells in spans", () => {
  assert.equal(countCells([[0, 2, 5], [3, 0, 1]]), 4);
});

test("a drag covers every cell it passed over, whichever way it went", () => {
  const rect = dragRectangle({ t: 4.2, c: 7.6 }, { t: 1.6, c: 2.4 }, 10, 12);
  assert.deepEqual(rect, [[1.5, 1.5], [1.5, 8.5], [4.5, 8.5], [4.5, 1.5]]);
  const spans = rasterise(rect, 10, 12);
  assert.equal(countCells(spans), 3 * 7); // slots 2-4, channels 2-8
  // Clamped to the plane.
  assert.deepEqual(dragRectangle({ t: -3, c: -3 }, { t: 30, c: 30 }, 10, 12), [[-0.5, -0.5], [-0.5, 11.5], [9.5, 11.5], [9.5, -0.5]]);
});

test("rejects degenerate polygons", () => {
  assert.throws(() => rasterise([[0, 0], [1, 1]], 4, 4));
  assert.throws(() => rasterise([[0, 0], [1, NaN], [2, 0]], 4, 4));
});
