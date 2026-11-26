"""Polygon rasterisation for manual flagging: which cells a selection covers.

numpy only, so these run wherever numpy is installed -- no Measurement Set.
"""

import json
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from surfvis.web.flagging import rasterise  # noqa: E402


def _rect(t0, t1, c0, c1):
    """The four-vertex polygon around slots t0..t1 and channels c0..c1, inclusive."""
    return [(t0 - 0.5, c0 - 0.5), (t0 - 0.5, c1 + 0.5), (t1 + 0.5, c1 + 0.5), (t1 + 0.5, c0 - 0.5)]


def test_rectangle_selects_exactly_its_cells():
    mask = rasterise(np.array(_rect(2, 4, 3, 7)), 10, 12)
    expected = np.zeros((10, 12), bool)
    expected[2:5, 3:8] = True
    np.testing.assert_array_equal(mask, expected)


def test_vertex_order_and_direction_do_not_matter():
    rect = np.array(_rect(1, 3, 0, 5))
    np.testing.assert_array_equal(rasterise(rect, 8, 8), rasterise(rect[::-1], 8, 8))
    np.testing.assert_array_equal(rasterise(rect, 8, 8), rasterise(np.roll(rect, 2, axis=0), 8, 8))


def test_triangle_uses_cell_centres():
    # Right triangle with legs just outside slot 0 and channel 0 and its
    # hypotenuse on c = 3.5 - t: centre (i, j) is inside when i + j <= 3.
    mask = rasterise(np.array([(-0.5, -0.5), (-0.5, 4.0), (4.0, -0.5)]), 6, 6)
    i, j = np.indices((6, 6))
    np.testing.assert_array_equal(mask, (i + j) < 3.5)


def test_self_intersecting_polygon_is_even_odd():
    # A bow tie: two triangles meeting at (5, 5); the crossing point is not
    # double-counted and the two lobes are both inside.
    bow = np.array([(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)])
    mask = rasterise(bow, 11, 11)
    assert mask[1, 5] and mask[9, 5]  # inside the upper and lower lobes
    assert not mask[5, 1] and not mask[5, 9]  # the side notches are outside


def test_polygon_beyond_the_grid_is_clipped():
    mask = rasterise(np.array(_rect(-5, 2, 30, 40)), 6, 34)
    expected = np.zeros((6, 34), bool)
    expected[0:3, 30:34] = True
    np.testing.assert_array_equal(mask, expected)


def test_rasterise_rejects_degenerate_input():
    with pytest.raises(ValueError):
        rasterise(np.array([(0.0, 0.0), (1.0, 1.0)]), 4, 4)
    with pytest.raises(ValueError):
        rasterise(np.array([(0.0, 0.0), (1.0, np.nan), (2.0, 0.0)]), 4, 4)
    # Big enough to overflow the edge arithmetic: rejected, not a 500.
    with pytest.raises(ValueError, match="within"):
        rasterise(np.array([(0.0, 0.0), (10.0, 1e308), (10.0, -1e308)]), 12, 12)


CASES = Path(__file__).parent / "js" / "rasterise_cases.json"


def _cases() -> list[dict]:
    """Polygons for the JS port to agree with: the shapes above, then random ones.

    Random vertices sit on integers, half-integers and arbitrary values, so
    centres on edges and vertices on centre lines both occur.
    """
    rng = np.random.default_rng(16)
    shapes = [
        ("rectangle", _rect(2, 4, 3, 7), 10, 12),
        ("triangle", [(-0.5, -0.5), (-0.5, 4.0), (4.0, -0.5)], 6, 6),
        ("bow tie", [(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)], 11, 11),
        ("clipped", _rect(-5, 2, 30, 40), 6, 34),
        ("vertex on a centre line", [(1.0, 0.2), (3.0, 4.7), (5.0, 0.2), (3.0, 2.0)], 7, 6),
    ]
    for k in range(40):
        n = int(rng.integers(3, 9))
        grid = rng.choice([1.0, 0.5, 0.0], p=[0.3, 0.3, 0.4])
        v = rng.uniform(-2, 14, size=(n, 2))
        if grid:
            v = np.round(v / grid) * grid
        shapes.append((f"random {k}", v.tolist(), 12, 12))
    # Near ties, where the crossing lands within rounding of a cell centre and
    # only the same arithmetic in the same order agrees: thirds and tenths are
    # not exact in binary, and integers nudged by 1e-3 sit next to the ties.
    for k in range(160):
        n = int(rng.integers(3, 12))
        kind = k % 4
        v = rng.uniform(-4, 24, size=(n, 2))
        if kind == 0:
            v = np.round(v * 3) / 3
        elif kind == 1:
            v = np.round(v * 10) / 10
        elif kind == 2:
            v = np.round(v) + rng.choice([-1e-3, 0.0, 1e-3], size=v.shape)
        else:
            v = np.round(v * 2) / 2 + rng.choice([-1e-9, 0.0, 1e-9], size=v.shape)
        shapes.append((f"near tie {k}", v.tolist(), 20, 20))
    # Found by fuzzing: an edge reordering of the interpolation gets row 16 wrong.
    shapes.append(("mined", [[13.5, -4.0], [19.0, 18.5], [22.5, 15.0], [16.0, 0.0], [-4.5, -3.0]], 20, 20))
    out = []
    for name, vertices, ntime, nchan in shapes:
        mask = rasterise(np.array(vertices, dtype=float), ntime, nchan)
        spans = []
        for i, row in enumerate(mask):
            edges = np.flatnonzero(np.diff(np.concatenate([[0], row.astype(np.int8), [0]])))
            spans += [[i, int(a), int(b)] for a, b in zip(edges[0::2], edges[1::2], strict=True)]
        vertices = [[float(t), float(c)] for t, c in vertices]
        out.append({"name": name, "vertices": vertices, "ntime": ntime, "nchan": nchan, "spans": spans})
    return out


def test_js_cases_match_python():
    """tests/js/rasterise_cases.json is what this rasteriser says; the JS port is checked against it.

    Regenerate with SURFVIS_REGEN_JS_CASES=1 after a deliberate change to the rule.
    """
    cases = _cases()
    if os.environ.get("SURFVIS_REGEN_JS_CASES"):
        CASES.write_text(json.dumps(cases, indent=None, separators=(",", ":")) + "\n")
    assert json.loads(CASES.read_text()) == cases
