import { test } from "node:test";
import assert from "node:assert/strict";
import { Carry } from "../../src/surfvis/web/static/carry.mjs";

test("a cross-correlation panel carries exactly what it keeps", () => {
  const c = new Carry();
  c.settled({ quantity: "phase" }, false);
  assert.deepEqual(c.params, { quantity: "phase" });
  c.settled({}, false);
  assert.deepEqual(c.params, {});
});

test("a model chosen on a cross-correlation survives passing through an auto", () => {
  const c = new Carry();
  c.settled({ model: "" }, false); // none, off the heatmap's MODEL_DATA
  c.settled({}, true); // none is the auto's default, so its panel does not keep it
  assert.deepEqual(c.params, { model: "" });
  c.settled({}, true); // and through a second auto
  assert.deepEqual(c.params, { model: "" });
});

test("resetting a control on an auto panel takes effect", () => {
  const c = new Carry();
  c.settled({ quantity: "phase", flagversion: "v1" }, false);
  c.settled({ quantity: "phase", flagversion: "v1" }, true);
  c.changed("quantity", true);
  c.settled({ flagversion: "v1" }, true);
  c.changed("flagversion", true);
  c.settled({}, true);
  assert.deepEqual(c.params, {});
});

test("a model changed on an auto panel replaces the one carried in", () => {
  const c = new Carry();
  c.settled({ model: "" }, false);
  c.changed("model", true);
  c.settled({ model: "MODEL_DATA" }, true);
  assert.deepEqual(c.params, { model: "MODEL_DATA" });
  c.changed("model", true); // and back to the auto's default: nothing carried
  c.settled({}, true);
  assert.deepEqual(c.params, {});
});

test("the heatmap's model comes back after an auto when none was chosen", () => {
  const c = new Carry();
  c.settled({}, false);
  c.settled({}, true);
  assert.deepEqual(c.params, {});
});

test("reset forgets everything", () => {
  const c = new Carry();
  c.settled({ model: "", gtable: "g.qc" }, false);
  c.reset();
  c.settled({}, true);
  assert.deepEqual(c.params, {});
});
