import { test } from "node:test";
import assert from "node:assert/strict";

import { staleCells, staleNote } from "../../src/surfvis/web/static/stale.mjs";

const names = ["m000", "m001", "m002", "m003"];

test("nothing stale marks nothing", () => {
  assert.equal(staleCells({ edits: 0, all: false, antennas: [], baselines: [] }, names).size, 0);
  assert.equal(staleCells(null, names).size, 0);
  assert.equal(staleNote({ edits: 0 }), "");
});

test("a baseline marks both triangles", () => {
  const cells = staleCells({ edits: 1, all: false, antennas: [], baselines: [["m001", "m003"]] }, names);
  assert.deepEqual([...cells].sort(), ["1,3", "3,1"]);
});

test("an antenna marks its row, its column and its auto", () => {
  const cells = staleCells({ edits: 1, all: false, antennas: ["m002"], baselines: [] }, names);
  assert.equal(cells.size, 2 * 4 - 1);
  assert.ok(cells.has("2,2") && cells.has("0,2") && cells.has("2,3"));
  assert.ok(!cells.has("0,1"));
});

test("all baselines marks every cell", () => {
  assert.equal(staleCells({ edits: 2, all: true, antennas: [], baselines: [] }, names).size, 16);
});

test("antennas not in this scan's grid are ignored", () => {
  const cells = staleCells({ edits: 2, all: false, antennas: ["m009"], baselines: [["m000", "m009"]] }, names);
  assert.equal(cells.size, 0);
});

test("the note counts edits", () => {
  assert.match(staleNote({ edits: 1 }), /1 manual edit in/);
  assert.match(staleNote({ edits: 3 }), /3 manual edits in/);
});
