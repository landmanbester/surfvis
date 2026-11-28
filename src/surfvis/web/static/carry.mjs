// Which waterfall controls a heatmap click hands the next panel, so a choice
// survives a change of baseline. Tested in tests/js/carry.test.mjs.
//
// The server marks the controls a panel moved off the heatmap's defaults
// (.keep); those are what is carried. An auto-correlation's panel defaults
// the model to none where a cross-correlation's defaults to the heatmap's,
// so a model carried in from a cross-correlation can look like a default
// there and would be dropped. It is held back and handed on, unless the
// model is changed on the auto panel itself. Every other control follows
// the panel exactly, so resetting one anywhere takes effect.

export class Carry {
  constructor() {
    this.reset();
  }

  // Nothing carried: no panel is on screen.
  reset() {
    this.params = {};
    this.crossModel = undefined; // the model the last cross-correlation kept
  }

  // A panel settled. keep: its .keep controls as {name: value}.
  settled(keep, auto) {
    this.params = { ...keep };
    if (!auto) {
      this.crossModel = keep.model;
    } else if (!("model" in keep) && this.crossModel !== undefined) {
      this.params.model = this.crossModel;
    }
  }

  // The user changed a control on the panel on screen.
  changed(name, auto) {
    if (auto && name === "model") this.crossModel = undefined;
  }
}
