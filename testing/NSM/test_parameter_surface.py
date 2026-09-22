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
import json
import os

import pytest
import torch

import NSM
from NSM.datasets.sdf_dataset import MultiSurfaceSDFSamples, SDFSamples
from NSM.models.loader import (
    _get_deepsdf_params,
    _get_implicit_params,
    _get_triplanar_params,
)
from NSM.models.triplanar import TriplanarDecoder
from NSM.reconstruct.latent_fit import _decode, reconstruct_latent
from NSM.reconstruct.utils import refuse_unknown_kwargs
from NSM.train.train_deep_sdf import _code_regularization_loss
from NSM.utils import (
    StepLearningRateSchedule,
    WarmupLearningRateSchedule,
    get_checkpoints,
    get_latent_vecs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _TinyTriplanarInterface(torch.nn.Module):
    """A decoder with `TriplanarDecoder`'s keyword forward interface, on the CPU."""

    def __init__(self, latent_size=4):
        super().__init__()
        self.net = torch.nn.Linear(latent_size + 3, 1)

    def forward(self, x=None, latent=None, xyz=None, epoch=None, verbose=False):
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


#: A config that both `_get_triplanar_params` and `_get_deepsdf_params` accept, so the
#: three translators can be compared on one input. Only the keys under test vary.
BOTH_MODEL_TYPES = {
    "latent_size": 16,
    "layer_dimensions": [8, 8],
    "padding": 0.35,
    "conv_norm_type": "layer",
    "conv_activation": None,
    "conv_hidden_dims": [8],
    "sdf_hidden_dims": [8],
    "sdf_latent_size": 8,
}


# ---------------------------------------------------------------------------
# (a) The five sites §8.0.K deferred here, all closed before the slice began
# ---------------------------------------------------------------------------


class TestTheDeferredSitesAreClosed:
    """
    §8.0.K measured five `reconstruct_latent` parameters that were named on one path and
    read on another, deferred all five to this slice, and then closed four of them in its
    own review round 2 (`5f1dbf7`) and the fifth through §8.0.J's kwargs refusal
    (`63209df`). The row kept the deferral and lost the fix, so R was scheduled to
    rediscover three numbers that had already moved -- `(0.005, 10, 100)` against a
    requested `(1.0, 3, 7)`, the "200x step size".

    These pins are the slice's product for carrier (a): a closed site that no test
    describes is indistinguishable from an open one at the next sweep.
    """

    def test_hybrid_optimizer_refuses_an_optimizer_name_it_will_not_consult(self):
        with pytest.raises(ValueError, match="optimizer_name is not consulted"):
            reconstruct_latent(**_fit_kwargs(hybrid_optimizer=True, optimizer_name="lbfgs"))

    def test_the_lbfgs_triple_is_read_on_the_non_hybrid_path(self, monkeypatch):
        """`lr` stood in for `lbfgs_lr` here, at a config's usual 0.005 -- a 200x step."""
        seen = {}
        real = torch.optim.LBFGS

        def spy(params, **kwargs):
            seen.update(kwargs)
            return real(params, **kwargs)

        monkeypatch.setattr(torch.optim, "LBFGS", spy)
        reconstruct_latent(
            **_fit_kwargs(
                optimizer_name="lbfgs",
                lr=0.005,
                lbfgs_lr=1.0,
                lbfgs_max_iter=3,
                lbfgs_history_size=7,
            )
        )

        assert (seen["lr"], seen["max_iter"], seen["history_size"]) == (1.0, 3, 7)

    def test_reconstruct_mesh_refuses_log_wandb_step_by_name(self):
        """
        `reconstruct_latent` names it and `reconstruct_mesh` has never forwarded it. The
        refusal names the parameter rather than reporting an anonymous unknown key,
        because "it never reached the fit even when accepted" is the part a caller who
        set it needs to read.
        """
        with pytest.raises(TypeError, match="log_wandb_step"):
            refuse_unknown_kwargs(
                {"log_wandb_step": 5},
                function_name="reconstruct_mesh",
                deprecated=frozenset({"batch_size_latent_recon"}),
            )


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

    @pytest.mark.parametrize("interval", [0, -1])
    def test_a_step_schedule_refuses_a_non_positive_interval(self, interval):
        with pytest.raises(ValueError, match="Factor"):
            StepLearningRateSchedule(initial=0.001, interval=interval, factor=0.5)

    @pytest.mark.parametrize("length", [0, -1])
    def test_a_warmup_schedule_refuses_a_non_positive_length(self, length):
        with pytest.raises(ValueError, match="Constant"):
            WarmupLearningRateSchedule(initial=0.0, warmed_up=0.001, length=length)

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

    def test_get_checkpoints_names_the_key_and_the_remedy(self):
        with pytest.raises(KeyError, match=r"\[\]"):
            get_checkpoints({"checkpoint_epochs": 100, "n_epochs": 1000})

    def test_the_schedules_still_compute_their_documented_formula(self):
        """The refusals must not move a rate anyone is training against."""
        step = StepLearningRateSchedule(initial=0.001, interval=500, factor=0.5)
        warm = WarmupLearningRateSchedule(initial=0.0, warmed_up=0.001, length=100)

        assert step.get_learning_rate(0) == pytest.approx(0.001)
        assert step.get_learning_rate(1000) == pytest.approx(0.00025)
        assert warm.get_learning_rate(50) == pytest.approx(0.0005)
        assert warm.get_learning_rate(500) == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Sites this slice deliberately does not change, asserted so they cannot rot
# ---------------------------------------------------------------------------


class TestTheEvidenceForSlicesThatOwnTheFix:
    """
    A finding whose remedy belongs to another slice, recorded as a passing test rather
    than a sentence, because the sentence is what goes stale.

    This class held a second one until §8.0.P: a `two_stage` config carried a misspelled
    architecture key into `TriplanarDecoder`'s unread `**kwargs`, which was §8.0.S item
    (4)'s only *config* path to the swallow. The `two_stage` model type is deleted, so
    that path is gone -- and item (4) is back to having no config route, which is worth
    knowing before S re-derives one.
    """

    def test_the_implicit_translator_ignores_the_activation_both_siblings_read(self):
        """
        A third independent way the `implicit` model type is unreachable as configured,
        alongside the two `SCOPE` §2.6 already records. `block_type: "linear"` builds
        `LinearBlockFactory()` at its `nn.ReLU` default whatever `activation` says.

        Not fixed here: §2.6 rules that any fix folds into the registration pathway, and
        patching one translator in isolation is what that ruling exists to prevent.
        """
        _, params = _get_implicit_params(
            {
                "latent_dim": 8,
                "hidden_dim": 8,
                "num_layers": 2,
                "block_type": "linear",
                "activation": "sin",
            }
        )

        assert params["block_factory"].activation_cls is torch.nn.ReLU
        assert _get_deepsdf_params(dict(BOTH_MODEL_TYPES, activation="sin"))[1]["activation"] == (
            "sin"
        )


class TestPolymorphicConformanceIsNotTheAcceptedAndIgnoredClass:
    """
    Four of the five parameters the sweep found accepted-and-never-read are the same
    `epoch`, on the decoders and on the schedules, and deleting any of them would break
    the caller: `train_epoch` calls every decoder as `model(inputs, epoch=epoch)` and
    `adjust_learning_rate` calls every schedule as `get_learning_rate(epoch)`.

    **The discriminator is the sibling**, and it is what #20's standing remedy -- delete
    the parameter -- has always needed to be safe. This test is that discriminator,
    executed.
    """

    def test_a_sibling_implementation_reads_the_epoch_the_others_ignore(self):
        from NSM.models.deep_sdf import Decoder
        from NSM.models.modulated_periodic_activations import ImplicitDecoder

        for cls in (Decoder, TriplanarDecoder, ImplicitDecoder):
            assert "epoch" in inspect.signature(cls.forward).parameters

        source = inspect.getsource(Decoder.forward)

        assert "epoch" in source, "the sibling that reads it is what makes the rest conformance"


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


class TestDecodeDispatchesOnTheForwardInterface:
    """
    Carrier (d). `_decode` inspects the decoder's signature on every call, inside
    `_recon_loss`'s optimization loop -- 20.5 us against 5.00 ms for one decode+backward
    at 10,000 points and 56.79 ms at 100,000, both CPU, so it is loop-invariant overhead
    rather than a defect.

    No wall-clock assertion: §7.4 ruled that a timing bound inside the suite it measures
    is self-referential and goes red on a shared runner for reasons unrelated to the code.
    What is pinned is the behaviour the hoist must not change -- both interfaces still
    dispatch, and to the same numbers.
    """

    def test_both_forward_interfaces_produce_the_same_values(self):
        torch.manual_seed(0)
        keyword = _TinyTriplanarInterface()
        positional = _TinyMlpInterface()
        positional.net.load_state_dict(keyword.net.state_dict())
        latent = torch.randn(1, 4)
        xyz = torch.rand(16, 3)

        assert torch.allclose(_decode(keyword, latent, xyz), _decode(positional, latent, xyz))

    def test_the_positional_interface_is_reached_at_all(self):
        """
        The MLP arm raised `TypeError: forward() got an unexpected keyword argument
        'latent'` on every subject before #105, on `main` and on the pre-refactor ref
        alike; production ships only triplanar, so nothing noticed.
        """
        result = _decode(_TinyMlpInterface(), torch.randn(1, 4), torch.rand(16, 3))

        assert result.shape == (16, 1)


# ---------------------------------------------------------------------------
# The predicate, not the data: what a substring sweep over the sources misses
# ---------------------------------------------------------------------------


def test_the_shipped_config_is_the_one_this_slice_measured():
    """
    A guard on the guard. The literal sweep in `test_default_config_sync` is only as
    honest as the file it reads, and §8.0.N's version of it passed six keys this slice
    reports. If the shipped config is replaced wholesale, that test's exception list is
    describing a file that no longer exists.
    """
    path = os.path.join(os.path.dirname(NSM.__file__), "configs", "default_config.json")
    with open(path, encoding="utf-8") as handle:
        shipped = json.load(handle)

    assert shipped["model_type"] == "triplanar"
    assert "conv_norm_type" in shipped and "padding" in shipped
