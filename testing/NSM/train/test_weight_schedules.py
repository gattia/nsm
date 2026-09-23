"""``cyclic_anneal_linear``, the regularization weight behind ``code_cyclic_anneal``.

The cycle length is ``floor(n_epochs / n_cycles)``. For a run shorter than ``n_cycles``
that is 0, and ``epoch % 0`` is NaN. A NaN weight NaNs the whole training loss, and the run
still exits 0. Such runs get a one-epoch cycle instead, which pins the weight at ``min_``.
"""

from NSM.train.utils import cyclic_anneal_linear


def test_the_cycle_is_finite_for_short_runs_and_unchanged_for_long_ones():
    """
    Fails if ``cyclic_anneal_linear`` returns anything but 0 for a run shorter than
    ``n_cycles``, or changes a 10-epoch run's first four weights, 0, 1, 0, 1.

    The 10-epoch case pins runs with ``n_epochs >= n_cycles``, which the one-epoch clamp
    must leave unchanged.
    """
    assert [cyclic_anneal_linear(epoch, n_epochs=2) for epoch in (1, 2)] == [0, 0]
    assert [cyclic_anneal_linear(e, 10) for e in (0, 1, 2, 3)] == [0.0, 1.0, 0.0, 1.0]
