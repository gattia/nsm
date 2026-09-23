"""``remove_overlapping_points`` must remove exactly the points inside >= 2 surfaces.

Its predecessor tested a sign SUM (``total == -2``), which is equivalent to the count
test only at exactly two surfaces: enumerated by execution during the Aug 2026 audit, it
removed nothing at 3 or 5 surfaces and only the inside-3-of-4 patterns at 4 — while never
removing a point it should have kept. At exactly two surfaces the two tests agree on
every sign pattern, which is why the committed regression baselines (two-surface data)
do not move with this fix. See ``docs/KNOWN_ISSUES.md`` § History.
"""

import itertools
from types import SimpleNamespace

import torch

from NSM.datasets.sdf_dataset import MultiSurfaceSDFSamples


def _run(rows):
    data = {
        "gt_sdf": torch.tensor(rows, dtype=torch.float32),
        "xyz": torch.zeros(len(rows), 3),
    }
    shim = SimpleNamespace()
    out, removed = MultiSurfaceSDFSamples.remove_overlapping_points(shim, data)
    return out, int(removed)


def test_every_sign_pattern_keeps_a_point_iff_it_is_inside_fewer_than_two():
    """Inside, on and outside for up to five surfaces. Exactly 0 is not inside."""
    for n_surfaces in (2, 3, 4, 5):
        rows = list(itertools.product((-0.5, 0.0, 0.5), repeat=n_surfaces))
        out, removed = _run(rows)
        expected = [row for row in rows if sum(v < 0 for v in row) < 2]
        assert removed == len(rows) - len(expected)
        assert [tuple(v) for v in out["gt_sdf"].tolist()] == [
            tuple(map(float, r)) for r in expected
        ]
        assert out["xyz"].shape[0] == len(expected)
