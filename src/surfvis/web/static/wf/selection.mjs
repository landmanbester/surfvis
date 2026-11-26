// The selection tool's state, without the DOM: which gesture is under way and
// what a key or pointer event does to it. flagtool.mjs feeds it events and
// acts on what it returns. Tested in tests/js/selection.test.mjs.
//
// Gestures (plain left button; shift- and middle-drag stay the widget's pan):
//   drag                -> rectangle, pending on release
//   click, click, ...   -> polygon vertices
//   click first vertex, double-click or Enter -> close the polygon: pending
//   Enter on a pending selection -> "apply" (never the same press that closed it)
//   Backspace           -> drop the last vertex
//   Esc                 -> cancel whatever is under way
// Vertices are in data coordinates (fractional slot, channel), so zooming or
// panning mid-gesture is harmless; distances that decide a gesture are in
// screen pixels, measured through the toScreen the caller passes.

import { dragRectangle } from "./select.mjs";

// A press that moves further than this is a drag, not a click.
export const DRAG_PX = 4;
// A click this close to the first vertex closes the polygon.
export const CLOSE_PX = 8;

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

export class Selection {
  constructor(ntime, nchan) {
    this.ntime = ntime;
    this.nchan = nchan;
    this.clear();
  }

  // idle | drawing (polygon under way) | dragging (rectangle under way) | pending
  clear() {
    this.mode = "idle";
    this.vertices = [];
    this.cursor = null; // {t, c, x, y}: the pointer, for the rubber band
    this.press = null; // where the button went down, until it comes up
  }

  // A closed selection, ready for confirmation (e.g. carried over from another baseline).
  setPending(vertices) {
    this.clear();
    this.mode = "pending";
    this.vertices = vertices.map(([t, c]) => [t, c]);
  }

  // The rectangle a drag in progress would make, or null.
  get dragRect() {
    if (this.mode !== "dragging" || !this.press || !this.cursor) return null;
    return dragRectangle(this.press, this.cursor, this.ntime, this.nchan);
  }

  down(pt) {
    if (this.mode === "pending") this.clear(); // a new gesture replaces an unconfirmed one
    this.press = pt;
    this.cursor = pt;
  }

  move(pt) {
    this.cursor = pt;
    if (this.press && this.mode === "idle" && dist(this.press, pt) > DRAG_PX) this.mode = "dragging";
  }

  // Returns "pending" when a selection closed, "drawing" when a vertex was
  // added, null otherwise. toScreen maps (t, c) to {x, y} as currently drawn.
  up(pt, toScreen) {
    const press = this.press;
    this.press = null;
    this.cursor = pt;
    if (!press) return null;
    if (this.mode === "dragging") {
      const rect = dragRectangle(press, pt, this.ntime, this.nchan);
      if (!rect) {
        this.clear();
        return null;
      }
      this.setPending(rect);
      return "pending";
    }
    if (this.mode === "drawing" && this.vertices.length >= 3) {
      const [t, c] = this.vertices[0];
      if (dist(toScreen(t, c), pt) <= CLOSE_PX) return this.close();
    }
    this.vertices.push([pt.t, pt.c]);
    this.mode = "drawing";
    return "drawing";
  }

  // The browser sends click, click, dblclick: the second click added a
  // vertex on top of the first, which is dropped before closing.
  doubleClick(toScreen) {
    if (this.mode !== "drawing") return null;
    const n = this.vertices.length;
    if (n >= 2) {
      const a = toScreen(...this.vertices[n - 2]);
      const b = toScreen(...this.vertices[n - 1]);
      if (dist(a, b) <= DRAG_PX) this.vertices.pop();
    }
    return this.vertices.length >= 3 ? this.close() : null;
  }

  // Returns "pending", "apply", "cleared", "drawing" or null (key not used).
  key(name) {
    if (name === "Escape") {
      if (this.mode === "idle") return null;
      this.clear();
      return "cleared";
    }
    if (name === "Enter") {
      if (this.mode === "drawing" && this.vertices.length >= 3) return this.close();
      if (this.mode === "pending") return "apply";
      return null;
    }
    if (name === "Backspace" && this.mode === "drawing") {
      this.vertices.pop();
      if (!this.vertices.length) this.clear();
      return this.mode === "drawing" ? "drawing" : "cleared";
    }
    return null;
  }

  close() {
    this.mode = "pending";
    this.press = null;
    return "pending";
  }
}

// The confirm button's label, from a dry run's answer.
export function confirmLabel(dry) {
  const n = (v) => v.toLocaleString("en-US");
  const on = dry.correlations?.length ? ` on ${dry.correlations.join(", ")}` : "";
  if (!dry.newly_flagged) return `Nothing new to flag (${n(dry.cells)} selected, all flagged already)`;
  return `Flag ${n(dry.newly_flagged)} new cells (${n(dry.cells)} selected)${on}`;
}

// The undo button's label for an edit summary from the server.
export function undoLabel(edit) {
  const [a, b] = edit.antennas;
  return `Undo ${a}–${b}, scan ${edit.scan}, ${edit.newly_flagged.toLocaleString("en-US")} cells`;
}

// Whether an edit is on the baseline a panel shows, whichever order the panel names it in.
export function sameBaseline(edit, ctx, name1, name2) {
  if (edit.field !== ctx.field || edit.spw !== ctx.spw || edit.scan !== ctx.scan) return false;
  const [a, b] = edit.antennas;
  return (a === name1 && b === name2) || (a === name2 && b === name1);
}
