import { test } from "node:test";
import assert from "node:assert/strict";
import { Selection, confirmLabel, undoLabel, sameBaseline, scopeParams, DRAG_PX } from "../../src/surfvis/web/static/wf/selection.mjs";

// One cell per 10 px, slot axis pointing down the screen (as near enough for these tests).
const toScreen = (t, c) => ({ x: c * 10, y: t * 10 });
const at = (t, c) => ({ t, c, ...toScreen(t, c) });

test("a drag makes a pending rectangle on release", () => {
  const s = new Selection(10, 12);
  s.down(at(1.2, 2.2));
  s.move(at(3.8, 6.9));
  assert.equal(s.mode, "dragging");
  assert.deepEqual(s.dragRect, [[0.5, 1.5], [0.5, 7.5], [4.5, 7.5], [4.5, 1.5]]);
  assert.equal(s.up(at(3.8, 6.9), toScreen), "pending");
  assert.deepEqual(s.vertices, [[0.5, 1.5], [0.5, 7.5], [4.5, 7.5], [4.5, 1.5]]);
});

test("a press that barely moves is a click, not a drag", () => {
  const s = new Selection(10, 12);
  s.down(at(2, 2));
  s.move({ t: 2, c: 2, x: 20 + DRAG_PX - 1, y: 20 });
  assert.equal(s.mode, "idle");
  assert.equal(s.up(at(2, 2), toScreen), "drawing");
  assert.deepEqual(s.vertices, [[2, 2]]);
});

test("clicks add vertices and clicking the first closes the polygon; Enter then applies", () => {
  const s = new Selection(10, 12);
  for (const p of [at(1, 1), at(1, 6), at(5, 6)]) {
    s.down(p);
    assert.equal(s.up(p, toScreen), "drawing");
  }
  s.down(at(1.3, 1.2)); // within CLOSE_PX of the first vertex
  assert.equal(s.up(at(1.3, 1.2), toScreen), "pending");
  assert.equal(s.vertices.length, 3);
  assert.equal(s.key("Enter"), "apply");
});

test("Enter closes a polygon and only a second Enter applies", () => {
  const s = new Selection(10, 12);
  for (const p of [at(1, 1), at(1, 6), at(5, 6)]) {
    s.down(p);
    s.up(p, toScreen);
  }
  assert.equal(s.key("Enter"), "pending");
  assert.equal(s.key("Enter"), "apply");
});

test("Enter does nothing to a polygon of fewer than three vertices", () => {
  const s = new Selection(10, 12);
  for (const p of [at(1, 1), at(1, 6)]) {
    s.down(p);
    s.up(p, toScreen);
  }
  assert.equal(s.key("Enter"), null);
  assert.equal(s.mode, "drawing");
});

test("a double-click closes, dropping the vertex its second click added", () => {
  const s = new Selection(10, 12);
  for (const p of [at(1, 1), at(1, 6), at(5, 6), at(5, 6)]) {
    s.down(p);
    s.up(p, toScreen);
  }
  assert.equal(s.doubleClick(toScreen), "pending");
  assert.deepEqual(s.vertices, [[1, 1], [1, 6], [5, 6]]);
});

test("Backspace drops the last vertex; Esc cancels anything", () => {
  const s = new Selection(10, 12);
  for (const p of [at(1, 1), at(1, 6)]) {
    s.down(p);
    s.up(p, toScreen);
  }
  assert.equal(s.key("Backspace"), "drawing");
  assert.deepEqual(s.vertices, [[1, 1]]);
  assert.equal(s.key("Backspace"), "cleared");
  assert.equal(s.mode, "idle");
  assert.equal(s.key("Escape"), null); // nothing to cancel: let the key through

  s.setPending([[0, 0], [0, 3], [3, 3]]);
  assert.equal(s.key("Backspace"), null);
  assert.equal(s.key("Escape"), "cleared");
  assert.equal(s.mode, "idle");
});

test("double-clicking the first vertex closes the polygon once and keeps it", () => {
  const s = new Selection(10, 12);
  for (const p of [at(1, 1), at(1, 6), at(5, 6)]) {
    s.down(p, 0);
    s.up(p, toScreen, 0);
  }
  s.down(at(1, 1), 1000);
  assert.equal(s.up(at(1, 1), toScreen, 1000), "pending");
  // The double-click's second press and the dblclick event change nothing.
  assert.equal(s.down(at(1, 1), 1150), false);
  assert.equal(s.up(at(1, 1), toScreen, 1150), null);
  assert.equal(s.doubleClick(toScreen), null);
  assert.equal(s.mode, "pending");
  assert.equal(s.vertices.length, 3);
  // A press later, or elsewhere, is a new gesture.
  assert.equal(s.down(at(1, 1), 2000), true);
  assert.equal(s.mode, "idle");
});

test("a new press replaces a pending selection", () => {
  const s = new Selection(10, 12);
  s.setPending([[0, 0], [0, 3], [3, 3]]);
  s.down(at(7, 7));
  assert.equal(s.mode, "idle");
  assert.equal(s.up(at(7, 7), toScreen), "drawing");
  assert.deepEqual(s.vertices, [[7, 7]]);
});

test("a drag that never touches the plane selects nothing", () => {
  const s = new Selection(10, 12);
  s.down(at(-5, 2));
  s.move(at(-2, 3));
  assert.equal(s.up(at(-2, 3), toScreen), null);
  assert.equal(s.mode, "idle");
});

test("labels", () => {
  assert.equal(
    confirmLabel({ cells: 1500, newly_flagged: 1234, correlations: ["XX", "YY"] }),
    "Flag 1,234 new cells (1,500 selected) on XX, YY",
  );
  assert.match(confirmLabel({ cells: 10, newly_flagged: 0, correlations: ["XX"] }), /^Nothing new to flag/);
  assert.equal(undoLabel({ antennas: ["m000", "m001"], scan: 3, newly_flagged: 1234 }), "Undo m000–m001, scan 3, 1,234 cells");
  assert.equal(undoLabel({ antennas: ["m004", "m004"], scan: 1, newly_flagged: 72 }), "Undo m004 auto, scan 1, 72 cells");
});

test("an edit is on the panel's baseline in either order, in the same partition", () => {
  const edit = { field: 0, spw: 0, scan: 1, antennas: ["m000", "m002"] };
  const ctx = { field: 0, spw: 0, scan: 1 };
  assert.ok(sameBaseline(edit, ctx, "m002", "m000"));
  assert.ok(!sameBaseline(edit, { ...ctx, scan: 2 }, "m000", "m002"));
  assert.ok(!sameBaseline(edit, ctx, "m000", "m003"));
});

test("wider scopes: labels, which panels an edit covers, and the /flag fields", () => {
  const antenna = { antennas: ["m000", "m002"], scope: "antenna", scope_antennas: ["m002"], scan: 1, newly_flagged: 9, field: 0, spw: 0 };
  const all = { ...antenna, scope: "all", scope_antennas: [] };
  assert.equal(undoLabel(antenna), "Undo every baseline of m002, scan 1, 9 cells");
  assert.equal(undoLabel(all), "Undo all baselines, scan 1, 9 cells");
  const ctx = { field: 0, spw: 0, scan: 1 };
  assert.ok(sameBaseline(antenna, ctx, "m002", "m005"));
  assert.ok(sameBaseline(antenna, ctx, "m002", "m002"));
  assert.ok(!sameBaseline(antenna, ctx, "m001", "m005"));
  assert.ok(sameBaseline(all, ctx, "m001", "m005"));
  assert.ok(!sameBaseline(all, { ...ctx, scan: 2 }, "m001", "m005"));
  assert.match(confirmLabel({ cells: 20, newly_flagged: 20, correlations: ["XX"], baselines: 28 }), /· 28 baselines$/);
  assert.doesNotMatch(confirmLabel({ cells: 20, newly_flagged: 20, correlations: ["XX"], baselines: 1 }), /baselines/);
  // Too large for the server to read FLAG while counting: the cells selected, and still confirmable.
  assert.equal(
    confirmLabel({ cells: 2e9, newly_flagged: null, correlations: ["XX"], baselines: 2016 }),
    "Flag up to 2,000,000,000 cells on XX · 2,016 baselines",
  );
  assert.deepEqual(scopeParams("baseline"), { baselines: "baseline" });
  assert.deepEqual(scopeParams("antenna:4"), { baselines: "antenna", antenna: 4 });
  assert.deepEqual(scopeParams("all"), { baselines: "all" });
});
