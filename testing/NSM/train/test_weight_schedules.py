"""``cyclic_anneal_linear`` must stay finite for runs shorter than its cycle count.

``floor(n_epochs / n_cycles)`` was 0 for ``n_epochs < 5``, so ``epoch % 0`` returned NaN
and the NaN regularization weight silently NaN'd the entire training loss — the run
completed and exited 0. Hit live while building ``test_default_config_trains.py``, whose
two-epoch run under the shipped ``code_cyclic_anneal: True`` produced ``loss: [nan, nan]``
with no error. Degenerate runs now clamp the cycle length to one epoch, which pins the
weight at ``min_``; any run with ``n_epochs >= n_cycles`` is bit-identical.
"""

from NSM.train.utils import cyclic_anneal_linear


def test_the_cycle_is_finite_for_short_runs_and_unchanged_for_long_ones():
    """Runs with at least ``n_cycles`` epochs are bit-identical to before the fix."""
    assert [cyclic_anneal_linear(epoch, n_epochs=2) for epoch in (1, 2)] == [0, 0]
    assert [cyclic_anneal_linear(e, 10) for e in (0, 1, 2, 3)] == [0.0, 1.0, 0.0, 1.0]
