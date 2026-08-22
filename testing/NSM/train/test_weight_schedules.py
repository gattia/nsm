"""``cyclic_anneal_linear`` must stay finite for runs shorter than its cycle count.

``floor(n_epochs / n_cycles)`` was 0 for ``n_epochs < 5``, so ``epoch % 0`` returned NaN
and the NaN regularization weight silently NaN'd the entire training loss — the run
completed and exited 0. Hit live while building ``test_default_config_trains.py``, whose
two-epoch run under the shipped ``code_cyclic_anneal: True`` produced ``loss: [nan, nan]``
with no error. Degenerate runs now clamp the cycle length to one epoch, which pins the
weight at ``min_``; any run with ``n_epochs >= n_cycles`` is bit-identical.
"""

import numpy as np

from NSM.train.utils import cyclic_anneal_linear


def test_runs_shorter_than_the_cycle_count_stay_finite_at_min():
    for epoch in (1, 2):
        weight = cyclic_anneal_linear(epoch, n_epochs=2)
        assert np.isfinite(weight)
        assert weight == 0  # the default min_


def test_runs_with_enough_epochs_are_unchanged():
    # n_epochs=10, n_cycles=5 -> cycle_length 2; ratio 0.5 ramps to max_ mid-cycle.
    # Values verified against the pre-fix implementation, which this path never touches.
    assert [cyclic_anneal_linear(e, 10) for e in (0, 1, 2, 3)] == [0.0, 1.0, 0.0, 1.0]
