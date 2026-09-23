"""``remove_overlapping_points`` must remove exactly the points inside >= 2 surfaces.

A sign-sum test (``total == -2``) agrees with the count on every sign pattern at exactly
two surfaces, so only three or more surfaces tell the two apart. At 3 or 5 surfaces the sum
removes nothing; at 4 it removes only the inside-3-of-4 patterns. The two-surface
regression baselines cannot see the difference. See ``docs/KNOWN_ISSUES.md`` § History 5.
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
    """
    Fails if ``MultiSurfaceSDFSamples.remove_overlapping_points`` keeps a point inside two or
    more of 2-5 surfaces, drops one inside fewer, or counts an SDF of exactly 0 as inside
    (KNOWN_ISSUES History 5).

    The method runs unbound on an empty ``SimpleNamespace``: it reads nothing from ``self``.
    """
    for n_surfaces in (2, 3, 4, 5):
        rows = list(itertools.product((-0.5, 0.0, 0.5), repeat=n_surfaces))
        out, removed = _run(rows)
        expected = [row for row in rows if sum(v < 0 for v in row) < 2]
        assert removed == len(rows) - len(expected)
        assert [tuple(v) for v in out["gt_sdf"].tolist()] == [
            tuple(map(float, r)) for r in expected
        ]
        assert out["xyz"].shape[0] == len(expected)
