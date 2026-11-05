import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { decode } from "../../src/surfvis/web/static/wf/format.mjs";

// Written by planes.encode (tests/test_planes.py pins the bytes), so this
// checks the Python encoder against the JS decoder.
test("decodes the block planes.encode wrote", () => {
  const file = readFileSync(new URL("./golden.bin", import.meta.url));
  const buf = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
  const { header, values, flags } = decode(buf);
  assert.equal(header.title, "m000-m002 χ²");
  assert.equal(header.nt, 2);
  assert.equal(header.nc, 3);
  assert.deepEqual(Array.from(values.slice(0, 2)), [1.5, -2.25]);
  assert.ok(Number.isNaN(values[2]));
  assert.deepEqual(Array.from(values.slice(3)), [0, 3e6, Math.fround(-1e-7)]);
  assert.deepEqual(Array.from(flags), [0, 255, 128, 1, 0, 254]);
});
