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

import pytest
import torch

from NSM.datasets.sdf_dataset import MultiSurfaceSDFSamples


def _run(rows):
    data = {
        "gt_sdf": torch.tensor(rows, dtype=torch.float32),
        "xyz": torch.zeros(len(rows), 3),
    }
    shim = SimpleNamespace(verbose=False)
    out, removed = MultiSurfaceSDFSamples.remove_overlapping_points(shim, data)
    return out, int(removed)


@pytest.mark.parametrize("n_surfaces", [2, 3, 4, 5])
def test_every_sign_pattern_keeps_iff_inside_fewer_than_two(n_surfaces):
    # Every combination of inside / on-surface / outside, one synthetic point each.
    # On-surface (exactly 0) is not "inside": only strictly negative SDFs count.
    rows = list(itertools.product((-0.5, 0.0, 0.5), repeat=n_surfaces))
    out, removed = _run(rows)

    expected_keep = [row for row in rows if sum(v < 0 for v in row) < 2]
    assert removed == len(rows) - len(expected_keep)
    kept = [tuple(v) for v in out["gt_sdf"].tolist()]
    assert kept == [tuple(float(v) for v in row) for row in expected_keep]
    assert out["xyz"].shape[0] == len(expected_keep)


def test_nan_columns_do_not_count_as_inside():
    # A None surface is stored as an all-NaN column and must be ignored by the count.
    rows = [(-0.5, -0.5, float("nan")), (-0.5, 0.5, float("nan"))]
    out, removed = _run(rows)
    assert removed == 1
    assert out["gt_sdf"].shape[0] == 1
    assert out["xyz"].shape[0] == 1


def test_fewer_than_two_real_surfaces_is_a_no_op():
    rows = [(-0.5, float("nan")), (-0.5, float("nan"))]
    out, removed = _run(rows)
    assert removed == 0
    assert out["gt_sdf"].shape[0] == 2
