"""
Plan §8.0.R — the parameter surface: is each parameter read on every path that accepts it?

Cross-module by construction, which is why this is one file rather than an addition to a
per-module suite: the class the slice chases is one *frame above* any single module, at
the point where a config key becomes a constructor argument. `models/loader.py`,
`train/train_deep_sdf.py`, `utils.py`, `datasets/sdf_dataset.py` and
`reconstruct/latent_fit.py` each hold one instance and none holds the pattern.

Three kinds of assertion live here and they are deliberately mixed:

* **Pins for sites that are already closed.** §8.0.K deferred five `reconstruct_latent`
  parameters to this slice and then fixed them in its own review rounds without updating
  the row, so the slice was scheduled to repair three numbers that had already moved. A
  test is the difference between a fix and a fix nobody can tell from a deferral.
* **Strict xfails for what this slice changes.** Each is retired by the commit named in
  its reason, and its docstring then records what was measured before the fix.
* **Plain tests for what this slice deliberately does not change** — the evidence
  §8.0.S and `SCOPE` §2.6 were missing, asserted where it cannot rot.
"""

import inspect

import pytest
import torch

from NSM.datasets.sdf_dataset import MultiSurfaceSDFSamples, SDFSamples
from NSM.reconstruct.latent_fit import _decode, reconstruct_latent
from NSM.reconstruct.utils import refuse_unknown_kwargs
from NSM.train.train_deep_sdf import _code_regularization_loss
from NSM.utils import get_latent_vecs

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _TinyTriplanarInterface(torch.nn.Module):
    """A decoder with `TriplanarDecoder`'s keyword forward interface, on the CPU."""

    def __init__(self, latent_size=4):
        super().__init__()
        self.net = torch.nn.Linear(latent_size + 3, 1)

    def forward(self, x=None, latent=None, xyz=None, epoch=None):
        if latent is not None:
            x = torch.cat([latent.expand(xyz.shape[0], -1), xyz], dim=1)
        return self.net(x)


class _TinyMlpInterface(torch.nn.Module):
    """A decoder with `deep_sdf.Decoder`'s positional forward interface."""

    def __init__(self, latent_size=4):
        super().__init__()
        self.net = torch.nn.Linear(latent_size + 3, 1)

    def forward(self, input_, epoch=None):
        return self.net(input_)


def _fit_kwargs(**overrides):
    torch.manual_seed(0)
    kwargs = dict(
        decoders=_TinyTriplanarInterface(),
        num_iterations=1,
        latent_size=4,
        xyz=torch.rand(32, 3),
        sdf_gt=torch.rand(32, 1),
        pts_surface=torch.zeros(32, dtype=torch.long),
        device="cpu",
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# (a) The five sites §8.0.K deferred here, all closed before the slice began
# ---------------------------------------------------------------------------


class TestGradClipReachesTheModelOnly:
    """
    Carrier (b), unchanged by this slice and re-verified rather than transcribed.
    `KNOWN_ISSUES` § Open holds the epoch-level measurement and the maintainer's ruling
    that the repair needs a training experiment, not a patch.
    """

    def test_the_latent_embedding_is_not_among_the_model_parameters(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 1))
        latent_vecs = get_latent_vecs(
            5,
            {
                "latent_size": 8,
                "variational": False,
                "code_regularization_type_prior": "spherical",
                "latent_bound": None,
                "code_init_std_dev": 0.01,
            },
        )

        clipped = {id(p) for p in model.parameters()}

        assert clipped, "the fixture must have parameters for the disjointness to mean anything"
        assert not any(id(p) in clipped for p in latent_vecs.parameters())


# ---------------------------------------------------------------------------
# (c) Config values that divide by zero, or are missing
# ---------------------------------------------------------------------------


class TestZeroAndMissingConfigValuesRefuse:
    """
    Carrier (c): four inputs that raise from arithmetic rather than from validation.

    None is a § History entry -- each has always crashed, so nobody holds a result from
    it (`CLAUDE.md` § Numerical-behaviour changes). What makes them worth a slice is
    *where* they crash: `code_regularization_warmup: 0` surfaces from inside the batch
    split loop, several hundred steps into a run, as `ZeroDivisionError: division by
    zero` naming nothing.

    The correct spelling of "off" was already written in the docstring beside each one.
    The fix moves that sentence into the refusal, which is where a caller reads it.

    Were four strict xfails. Measured before the fix: ``ZeroDivisionError: integer
    division or modulo by zero`` (Step), ``float division by zero`` (Warmup),
    ``division by zero`` (the code-regularization warmup) and ``KeyError:
    'additional_checkpoints'`` -- four messages, none of which names the config key that
    produced it or what to set instead.
    """

    def test_code_regularization_refuses_a_zero_warmup(self):
        config = {
            "code_regularization_type_prior": "spherical",
            "code_regularization_warmup": 0,
            "code_regularization_weight": 1e-4,
            "variational": False,
            "code_cyclic_anneal": False,
            "n_epochs": 100,
        }
        with pytest.raises(ValueError, match="code_regularization_weight"):
            _code_regularization_loss(
                batch_vecs=torch.randn(4, 8),
                mu=None,
                logvar=None,
                num_sdf_samples=100,
                epoch=0,
                config=config,
            )


# ---------------------------------------------------------------------------
# Sites this slice deliberately does not change, asserted so they cannot rot
# ---------------------------------------------------------------------------


class TestUpgradeCachedLayoutKeepsItsCachePath:
    """
    The sweep's one candidate on the function surface, and running it retired that too.

    `SDFSamples._upgrade_cached_layout` never reads the `cache_path` it is handed, so the
    per-function predicate reports it as accepted-and-never-read and #20's standing
    remedy says delete it. Deleting it breaks the override: `MultiSurfaceSDFSamples`
    names the file in the warning it emits before asking for a delete-and-rebuild, which
    is the one message that says *which* subject's cache was beyond repair.

    Was two strict xfails, retired by measurement rather than by a fix. It is the same
    discriminator as `TestPolymorphicConformanceIsNotTheAcceptedAndIgnoredClass` and the
    reason that class exists: **a polymorphic hook's parameters have to be judged across
    every implementation, not per function.** The sweep is per function, so the
    discriminator is not optional -- it is the second half of the predicate. After it, the
    accepted-and-never-read class is **empty** on NSM's documented function surface.
    """

    @pytest.mark.parametrize("cls", [SDFSamples, MultiSurfaceSDFSamples])
    def test_the_hook_still_accepts_the_path_one_implementation_names(self, cls):
        assert "cache_path" in inspect.signature(cls._upgrade_cached_layout).parameters

    def test_the_subclass_is_the_implementation_that_reads_it(self):
        base = inspect.getsource(SDFSamples._upgrade_cached_layout)
        override = inspect.getsource(MultiSurfaceSDFSamples._upgrade_cached_layout)

        assert "cache_path" not in base.split('"""')[-1]
        assert "cache_path" in override.split('"""')[-1]
