"""
``reconstruct_latent``: the parameters it takes, the values it accepts, the loss it
returns, and the helpers it is built from (``NSM/reconstruct/latent_fit.py``).
"""

import ast
import inspect
import logging

import numpy as np
import pytest
import torch

import NSM.reconstruct.latent_fit as latent_fit
import NSM.reconstruct.main as recon_main
from NSM.reconstruct import reconstruct_latent
from NSM.reconstruct.main import (
    latent_norm_penalty,
    project_latent,
    reconstruct_latent_decoders_type_check,
    reconstruct_latent_get_lr_update_freq,
    reconstruct_latent_preprocess_sdf_gt,
    reconstruct_latent_pts_surface_type_check,
    reconstruct_latent_sdf_gt_type_check,
)


class LinearDecoder(torch.nn.Module):
    """``pts[:, 0] + latent.sum()`` per surface: differentiable in the latent, no parameters."""

    def __init__(self, surfaces=1):
        super().__init__()
        self.surfaces = surfaces

    def forward(self, x=None, latent=None, xyz=None, epoch=None):
        pts = xyz if xyz is not None else x[:, -3:]
        return (pts[:, :1] + latent.sum()).repeat(1, self.surfaces)


class RecordingDecoder(LinearDecoder):
    """Records a fingerprint of every point set it is evaluated on."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.draws = []

    def forward(self, x=None, latent=None, xyz=None, epoch=None):
        self.draws.append(round(float((xyz if xyz is not None else x[:, -3:]).sum()), 9))
        return super().forward(x=x, latent=latent, xyz=xyz, epoch=epoch)


class LegacyInterfaceDecoder(torch.nn.Module):
    """``deep_sdf.Decoder``'s interface: one concatenated tensor. Same function as above."""

    def __init__(self, latent_size=8):
        super().__init__()
        self.latent_size = latent_size

    def forward(self, input_, epoch=None):
        latent, pts = input_[:, : self.latent_size], input_[:, self.latent_size :]
        return pts[:, :1] + latent[0].sum()


def fit_kwargs(n_pts=64, **overrides):
    """A three-step CPU fit, seeded."""
    torch.manual_seed(7)
    kwargs = dict(
        num_iterations=3,
        latent_size=8,
        xyz=torch.rand(n_pts, 3),
        sdf_gt=torch.rand(n_pts, 1),
        pts_surface=[0] * n_pts,
        device="cpu",
    )
    kwargs.update(overrides)
    return kwargs


def fit(decoder=None, **overrides):
    return reconstruct_latent(decoders=decoder or LinearDecoder(), **fit_kwargs(**overrides))


class TestUnknownKeywordsAreRefused:
    def test_a_misspelled_parameter_raises(self):
        """
        Fails if ``reconstruct_latent`` accepts a misspelled keyword or the removed
        ``max_batch_size`` instead of raising ``TypeError`` naming it (KNOWN_ISSUES History 20).
        """
        for wrong in (
            "num_iteration",
            "latent_reg_wieght",
            "clamp_distance",
            "lattent_size",
            "optimiser_name",
            "n_iterations",
            "lr_",
            "max_batch_size",  # removed in v0.4.0; use n_samples_per_chunk (#75)
        ):
            with pytest.raises(TypeError, match=wrong):
                fit(**{wrong: 999})

    def test_reconstruct_mesh_passes_only_parameters_this_signature_names(self):
        """
        Fails if the ``reconstruct_inputs`` dict in ``reconstruct_mesh`` forwards a key that
        ``reconstruct_latent`` does not name.

        The keys are read from ``main.py`` with ``ast``, by the local name
        ``reconstruct_inputs``, so a new key is checked too.
        """
        tree = ast.parse(open(recon_main.__file__, encoding="utf-8").read())
        dict_node = next(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", None) == "reconstruct_inputs"
        )
        passed = {key.value for key in dict_node.keys}
        assert passed and passed <= set(inspect.signature(reconstruct_latent).parameters)


def test_pts_surface_is_required_and_a_positional_shift_is_refused():
    """
    Fails if ``reconstruct_latent`` gives ``pts_surface`` a default, or accepts ``None``, a
    string or a float for it.

    ``pts_surface`` follows ``sdf_gt``. An old six-argument positional call puts ``loss_type``
    or ``lr`` in its slot, and the type check must refuse it.
    """
    kwargs = fit_kwargs()
    del kwargs["pts_surface"]
    with pytest.raises(TypeError, match="pts_surface"):
        reconstruct_latent(decoders=LinearDecoder(), **kwargs)
    for value in (None, "l1", 5e-4):
        with pytest.raises(ValueError, match="pts_surface"):
            fit(pts_surface=value)


class TestUnknownValuesAreRefusedWhereTheyAreNamed:
    """``reconstruct_latent`` folds case because NSM's trainer spells ``"Adam"``."""

    def test_an_unknown_value_names_its_parameter_and_case_is_folded(self):
        """
        Fails if ``reconstruct_latent`` accepts an unknown or ``None`` ``optimizer_name``,
        ``loss_type`` or ``convergence`` without a ``ValueError`` naming the parameter, or
        stops folding case (KNOWN_ISSUES History 23).

        ``"Recon_Loss"`` must fit exactly as ``"recon_loss"``, the mode
        ``default_config.json`` ships.
        """
        for parameter, value in (
            ("optimizer_name", "sgd"),
            ("loss_type", "l1_smooth"),
            ("convergence", "banana"),
            ("convergence", None),
        ):
            with pytest.raises(ValueError, match=parameter):
                fit(**{parameter: value})
        for parameter, value in (
            ("optimizer_name", "Adam"),
            ("optimizer_name", "LBFGS"),
            ("loss_type", "L1"),
            ("loss_type", "L1_LOG"),
        ):
            assert isinstance(fit(**{parameter: value})[1], torch.Tensor)

        lower, upper = fit(convergence="recon_loss"), fit(convergence="Recon_Loss")
        assert torch.equal(lower[1], upper[1]) and float(lower[0]) == float(upper[0])

    def test_a_flag_that_is_not_a_bool_is_refused(self):
        """
        Fails if ``reconstruct_latent`` accepts an ``l2reg`` or ``log_wandb`` that is not a
        bool (#116).

        Both are read with ``is True``, so ``l2reg=1`` gave a latent loss of 0 and
        ``log_wandb=1`` logged nothing.
        """
        for name in ("l2reg", "log_wandb"):
            for value in (1, "yes", None):
                with pytest.raises(TypeError, match=name):
                    fit(**{name: value})

    def test_hybrid_mode_refuses_an_optimizer_name_it_will_not_consult(self):
        """
        Fails if ``reconstruct_latent(hybrid_optimizer=True, optimizer_name="lbfgs")`` runs
        instead of raising ``ValueError`` (KNOWN_ISSUES History 22).

        Hybrid mode runs Adam then LBFGS and reads ``optimizer_name`` nowhere.
        """
        with pytest.raises(ValueError, match="optimizer_name is not consulted"):
            fit(hybrid_optimizer=True, optimizer_name="lbfgs", adam_iterations=2)


class TestTheReturnedLossIsALoss:
    def test_each_convergence_mode_returns_a_real_loss(self):
        """
        Fails if ``reconstruct_latent``, in any convergence mode, returns its float sentinel
        instead of a tensor loss or raises when no loss falls below 100
        (KNOWN_ISSUES History 21).

        Ground truth scaled by 1000 keeps every loss above 100, the old sentinel. The test
        also fails if ``num_iterations=0`` returns other than ``inf`` and a ``(1, L)`` latent,
        or if ``overall_loss`` patience never stops the fit.
        """
        for convergence in ("recon_loss", "overall_loss", "num_iterations"):
            loss, latent = fit(convergence=convergence)
            assert isinstance(loss, torch.Tensor) and float(loss) != 100
            loss, latent = fit(convergence=convergence, sdf_gt=torch.rand(64, 1) * 1000)
            assert torch.isfinite(latent).all()

        loss, latent = fit(num_iterations=0)
        assert loss == float("inf") and latent.shape == (1, 8)

        # Patience runs out long before 100 steps on this problem.
        patient = RecordingDecoder()
        fit(patient, num_iterations=100, convergence="overall_loss", convergence_patience=5)
        assert len(patient.draws) < 100

    def test_the_best_step_s_latent_is_returned(self):
        """
        Fails if ``reconstruct_latent`` under ``convergence="recon_loss"`` or
        ``"overall_loss"`` returns a latent other than the one its returned loss was measured
        on, such as the latent after that step, the initial one or the last one
        (KNOWN_ISSUES History 32). kneepipeline fits both shipped models with ``recon_loss``.

        At ``lr=0.05`` the fit overshoots a constant target, so the initial, best, next and
        last latents all differ.
        """

        class LatentRecorder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.latents = []

            def forward(self, x=None, latent=None, xyz=None, epoch=None):
                self.latents.append(latent.detach().clone())
                return xyz[:, :1] * 0 + latent.sum()

        for convergence in ("recon_loss", "overall_loss"):
            decoder = LatentRecorder()
            loss, latent = fit(
                decoder,
                num_iterations=40,
                latent_size=4,
                sdf_gt=torch.full((64, 1), 0.3),
                lr=0.05,
                n_lr_updates=0,
                convergence=convergence,
                convergence_patience=5,
            )
            losses = [abs(float(seen.sum()) - 0.3) for seen in decoder.latents]
            best = losses.index(min(losses))
            assert 0 < best < len(losses) - 2, convergence
            assert float(loss) == pytest.approx(losses[best], abs=1e-6)
            assert torch.equal(latent[0], decoder.latents[best]), convergence


_ADJUST_LEARNING_RATE = latent_fit.adjust_learning_rate


def _learning_rates_seen(monkeypatch, **overrides):
    seen = []

    def spy(**kwargs):
        _ADJUST_LEARNING_RATE(**kwargs)
        seen.append(round(kwargs["optimizer"].param_groups[0]["lr"], 12))

    monkeypatch.setattr(latent_fit, "adjust_learning_rate", spy)
    fit(lr=1e-2, n_lr_updates=2, lr_update_factor=10, **overrides)
    return seen


class TestTheLearningRateScheduleSpansThePhaseItSteps:
    def test_hybrid_mode_applies_the_updates_asked_for(self, monkeypatch):
        """
        Fails if hybrid ``reconstruct_latent`` takes its LR decay interval from
        ``num_iterations`` instead of ``adam_iterations``, so the Adam phase decays more than
        asked (KNOWN_ISSUES History 22).

        With the interval from ``num_iterations=10``, 100 Adam steps took 11 decays and ended
        at exactly 0.0.
        """
        expected = _learning_rates_seen(monkeypatch, num_iterations=100)
        hybrid = _learning_rates_seen(
            monkeypatch,
            num_iterations=10,
            hybrid_optimizer=True,
            adam_iterations=100,
            lbfgs_iterations=0,
        )
        assert len(hybrid) == len(expected) == 100
        assert sorted(set(hybrid), reverse=True) == sorted(set(expected), reverse=True)
        assert sorted(set(hybrid), reverse=True) == [1e-2, 1e-3]


class TestTheDeferredSitesAreClosed:
    def test_the_lbfgs_triple_is_read_on_the_non_hybrid_path(self, monkeypatch):
        """
        Fails if ``reconstruct_latent(optimizer_name="lbfgs")`` builds LBFGS with ``lr``
        instead of ``lbfgs_lr``, or ignores ``lbfgs_max_iter`` or ``lbfgs_history_size``.

        At a config's usual ``lr`` of 0.005 and ``lbfgs_lr`` of 1.0, that step is 200x too
        small.
        """
        seen = {}
        real = torch.optim.LBFGS

        def spy(params, **kwargs):
            seen.update(kwargs)
            return real(params, **kwargs)

        monkeypatch.setattr(torch.optim, "LBFGS", spy)
        fit(
            num_iterations=1,
            optimizer_name="lbfgs",
            lr=0.005,
            lbfgs_lr=1.0,
            lbfgs_max_iter=3,
            lbfgs_history_size=7,
        )
        assert (seen["lr"], seen["max_iter"], seen["history_size"]) == (1.0, 3, 7)


class TestTheDrawIsPerEvaluation:
    """
    The sample draw runs on every loss evaluation, not once per step. Adam evaluates once
    per step; LBFGS evaluates several times, and each evaluation redraws.

    A draw once per step measures worse. L-BFGS assumes a fixed objective, but the redraw is
    also how the fit covers the cloud. Over 20 problems at 12,000 decoder evaluations, LBFGS,
    median held-out error:

    | sampling ratio | per evaluation | per step | cloud seen |
    |---|---|---|---|
    | 1.6% | 0.034 | 0.202 | 96% vs 82% |
    | 5%   | 0.007 | 0.029 | 95% vs 41% |
    | 20%  | 0.0049 | 0.0054 | 99% vs 37% |
    | 50%  | 0.0042 | 0.0045 | 100% vs 50% |

    The gap tracks coverage. Resetting LBFGS's curvature history at each draw made it worse
    (13/20 diverged against 7/20). A deterministic draw with good coverage would answer the
    line-search objection; that is a sampling-strategy change.

    The full cloud measures best of all, and ``n_samples_per_chunk`` (#75) makes it
    affordable:

    | regime | median | diverged | cloud seen |
    |---|---|---|---|
    | full cloud | 0.0038 | 0/20 | 100% |
    | per-evaluation redraw, 5% | 0.0066 | 2/20 | 95% |
    | per-step redraw, 5% | 0.115 | 12/20 | 47% |
    | per-step without replacement, 5% | 0.056 | 11/20 | 54% |
    """

    def test_lbfgs_redraws_within_a_step_and_adam_draws_once(self):
        """
        Fails if ``reconstruct_latent`` stops redrawing samples on every LBFGS loss
        evaluation, evaluates Adam more than once per step, or subsamples when ``n_samples``
        equals the cloud.
        """
        lbfgs = RecordingDecoder()
        fit(lbfgs, n_pts=100, num_iterations=1, optimizer_name="lbfgs", n_samples=50)
        assert len(lbfgs.draws) > 1 and len(set(lbfgs.draws)) == len(lbfgs.draws)

        adam = RecordingDecoder()
        fit(adam, n_pts=100, num_iterations=4, optimizer_name="adam", n_samples=50)
        assert len(adam.draws) == len(set(adam.draws)) == 4

        full = RecordingDecoder()
        fit(full, n_pts=60, num_iterations=1, optimizer_name="lbfgs", n_samples=60)
        assert len(set(full.draws)) == 1

    def test_a_subsampled_lbfgs_fit_says_so(self, caplog):
        """
        Fails if ``reconstruct_latent`` stops warning that an LBFGS or hybrid fit draws a
        subsample, including for ``optimizer_name="LBFGS"``.

        The upper-case spelling checks that the guard runs after the case fold.
        """
        for options in (
            {"optimizer_name": "lbfgs"},
            {"optimizer_name": "LBFGS"},
            {"hybrid_optimizer": True, "adam_iterations": 1},
        ):
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="NSM"):
                fit(n_pts=100, num_iterations=1, n_samples=50, **options)
            assert "n_samples_per_chunk" in caplog.text, options

    def test_a_multi_surface_draw_is_balanced_and_the_guard_says_what_it_draws(self, caplog):
        """
        Fails if ``_samples_per_surface`` stops holding every surface to the smallest one's
        count, or the LBFGS warning reports the budget instead of the planned draw and the
        reachable maximum (KNOWN_ISSUES History 24).

        On 300 and 90 points, a budget of 390 draws 90 from each surface. The fit's budget of
        100 draws 50 from each, and 180 of the 390 points is the most a balanced draw reaches.
        """
        pts_surface = torch.tensor([0] * 300 + [1] * 90)
        assert latent_fit._samples_per_surface(
            n_samples=390, pts_surface=pts_surface, n_surfaces=2
        ) == [90, 90]

        sdf = torch.rand(390, 1)
        with caplog.at_level(logging.WARNING, logger="NSM"):
            reconstruct_latent(
                decoders=LinearDecoder(surfaces=2),
                num_iterations=1,
                latent_size=8,
                xyz=torch.rand(390, 3),
                sdf_gt=[sdf, sdf.clone()],
                pts_surface=pts_surface,
                n_samples=100,
                optimizer_name="lbfgs",
                device="cpu",
            )
        assert "draws 100 points" in caplog.text
        assert "180 of 390 points" in caplog.text


def test_the_lbfgs_closure_does_not_retain_its_graph():
    """
    Fails if any call in ``latent_fit.py`` passes ``retain_graph=``, or a two-step LBFGS fit
    returns a non-finite loss or latent.

    Each closure call builds its own graph, so none is backwarded twice. Retaining kept a dead
    graph resident: on a T4 at 60,000 points, 2265 MiB against 1240 MiB, for a bit-identical
    latent. The source is scanned for the call keyword, not the text, because a comment names
    the flag.
    """
    tree = ast.parse(open(latent_fit.__file__, encoding="utf-8").read())
    assert not [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "retain_graph"
    ]
    loss, latent = fit(n_pts=80, num_iterations=2, optimizer_name="lbfgs")
    assert torch.isfinite(latent).all() and np.isfinite(float(loss))


class TestChunkedForwardAndBackward:
    """
    #75: ``n_samples_per_chunk`` splits one step's forward and backward and accumulates the
    gradient on the latent. On a T4 at 200,000 points (latent 256, 8x512 decoder), through
    ``reconstruct_latent``: 4128 MiB at the default, 623 MiB at 30,000 per chunk, with the
    loss agreeing to 1.2e-07. A backward per chunk is the design because retaining every
    chunk's graph saved 17% and freeing each saved 85%. Measuring that needs a GPU; the
    gradient and the default are what is asserted.
    """

    def test_a_chunked_step_has_the_unchunked_gradient(self, monkeypatch):
        """
        Fails if ``reconstruct_latent``'s chunked step weights a chunk other than by its share
        of the points, or ``_recon_loss`` stops returning a per-point mean, so the latent
        gradient Adam steps on differs from the unchunked one.

        The gradient is read at Adam's first step. 97 points leave a ragged last chunk at
        sizes 10 and 32. Adam's first update is about ``lr`` times the gradient's sign, so the
        fitted latent alone hides a wrong weight.
        """
        gradients = []

        class RecordingAdam(torch.optim.Adam):
            def step(self, closure=None):
                gradients.append(self.param_groups[0]["params"][0].grad.clone())
                return super().step(closure)

        monkeypatch.setattr(torch.optim, "Adam", RecordingAdam)

        def gradient(chunk):
            torch.manual_seed(5)
            reconstruct_latent(
                decoders=LinearDecoder(surfaces=2),
                num_iterations=1,
                latent_size=8,
                xyz=torch.rand(97, 3),
                sdf_gt=[torch.rand(97, 1) for _ in range(2)],
                pts_surface=[0] * 97,
                clamp_dist=0.1,
                n_samples_per_chunk=chunk,
                device="cpu",
            )
            return gradients.pop()

        unchunked = gradient(None)
        for chunk in (10, 32, 200):
            assert float((unchunked - gradient(chunk)).abs().max() / unchunked.abs().max()) < 1e-6

    def test_the_default_is_the_unchunked_path_and_a_chunked_fit_lands_close(self):
        """
        Fails if ``reconstruct_mesh``'s ``n_samples_per_chunk_latent_recon`` stops defaulting
        to ``None``, the default fit differs at all from ``n_samples_per_chunk=None``, or a
        chunked ``reconstruct_latent`` fit moves the latent more than 1e-6 (#75).

        Chunking changes the summation order, so a chunked fit is close, not identical.
        """

        def run(**overrides):
            torch.manual_seed(3)
            xyz, sdf_gt = torch.rand(97, 3), [torch.rand(97, 1) for _ in range(2)]
            return reconstruct_latent(
                decoders=LinearDecoder(surfaces=2),
                num_iterations=4,
                latent_size=8,
                xyz=xyz,
                sdf_gt=sdf_gt,
                pts_surface=[i % 2 for i in range(97)],
                n_samples=60,
                clamp_dist=0.1,
                device="cpu",
                **overrides,
            )

        default, explicit, chunked = (
            run(),
            run(n_samples_per_chunk=None),
            run(n_samples_per_chunk=25),
        )
        assert torch.equal(default[1], explicit[1]) and float(default[0]) == float(explicit[0])
        assert torch.allclose(default[1], chunked[1], atol=1e-6)
        named = inspect.signature(recon_main.reconstruct_mesh).parameters
        assert named["n_samples_per_chunk_latent_recon"].default is None


def test_both_decoder_forward_interfaces_fit_the_same_latent():
    """
    Fails if ``_decode`` stops dispatching between the keyword ``forward(latent=, xyz=)`` and
    ``deep_sdf.Decoder``'s concatenated ``forward(input_)``, or concatenates other than
    ``[latent, xyz]``.
    """
    seen = []

    class Recording(LinearDecoder):
        def forward(self, x=None, latent=None, xyz=None, epoch=None):
            seen.append(x is None and latent is not None and xyz is not None)
            return super().forward(x=x, latent=latent, xyz=xyz)

    keyword = fit(Recording())
    legacy = fit(LegacyInterfaceDecoder(latent_size=8))
    assert seen and all(seen)
    assert float(keyword[0]) == float(legacy[0]) and torch.equal(keyword[1], legacy[1])


def test_each_decoder_reads_its_own_slice_of_the_ground_truth():
    """
    Fails if ``_recon_loss`` indexes ``sdf_gt`` per decoder instead of by the running
    ``surface_offset``, so the second decoder re-reads surfaces 0 and 1
    (KNOWN_ISSUES History 5).

    Only surfaces 2 and 3 differ between the two calls.
    """

    class TwoSurfaces(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, x=None, latent=None, xyz=None, epoch=None):
            base = xyz[:, :1] * 0.01 + latent.sum() * self.scale
            return torch.cat([base, 2.0 * base], dim=1)

    def loss(values):
        torch.manual_seed(0)
        return float(
            reconstruct_latent(
                decoders=[TwoSurfaces(1.0), TwoSurfaces(-1.0)],
                num_iterations=3,
                latent_size=8,
                xyz=torch.rand(50, 3),
                sdf_gt=[torch.full((50, 1), v) for v in values],
                pts_surface=[0] * 50,
                device="cpu",
            )[0]
        )

    assert loss((0.1, 0.2, 0.3, 0.4)) != loss((0.1, 0.2, 5.0, -5.0))


class TestTheHelpers:
    def test_the_type_checks(self):
        """
        Fails if ``reconstruct_latent_sdf_gt_type_check`` stops wrapping or copying its input,
        ``reconstruct_latent_pts_surface_type_check`` stops converting a list or array, or any
        of the three type checks accepts an ``int``.

        ``sdf_gt`` refuses a string by pointing at ``reconstruct_mesh``, and other types with
        a bare ``Exception``; the test pins that type. The copied list holds the caller's
        tensors. The decoders check also refuses a list holding a non-module.
        """
        sdf = torch.zeros(5, 1)
        assert reconstruct_latent_sdf_gt_type_check(sdf) == [sdf]
        array = np.zeros((5, 1))
        assert reconstruct_latent_sdf_gt_type_check(array)[0] is array
        caller = [sdf, None]
        checked = reconstruct_latent_sdf_gt_type_check(caller)
        assert checked == caller and checked is not caller and checked[0] is sdf
        with pytest.raises(Exception, match="Invalid sdf_gt type") as excinfo:
            reconstruct_latent_sdf_gt_type_check(42)
        assert type(excinfo.value) is Exception
        with pytest.raises(Exception, match="reconstruct_mesh instead"):
            fit(sdf_gt="a path")

        for value in ([0, 0, 1], np.array([0, 0, 1])):
            assert reconstruct_latent_pts_surface_type_check(value, "cpu").tolist() == [0, 0, 1]
        tensor = torch.tensor([0, 1])
        assert reconstruct_latent_pts_surface_type_check(tensor, "cpu") is tensor
        with pytest.raises(ValueError, match="pts_surface must be"):
            reconstruct_latent_pts_surface_type_check(42, "cpu")

        decoder = torch.nn.Linear(2, 1)
        assert reconstruct_latent_decoders_type_check(decoder) == [decoder]
        assert reconstruct_latent_decoders_type_check([decoder]) == [decoder]
        for bad, match in (([decoder, 42], "list of torch.nn.Module"), (42, "torch.nn.Module")):
            with pytest.raises(ValueError, match=match):
                reconstruct_latent_decoders_type_check(bad)

    def test_the_lr_update_frequency_and_the_sdf_preprocess(self):
        """
        Fails if ``reconstruct_latent_get_lr_update_freq`` stops treating 0 or ``None`` as
        never or flooring at 1, or ``reconstruct_latent_preprocess_sdf_gt`` stops clamping
        to ``clamp_dist``, drops a ``None`` surface, or clamps when ``clamp_dist`` is ``None``.

        Never is an interval one past the loop bound: 101 for 100 iterations.
        """
        assert [reconstruct_latent_get_lr_update_freq(n, 100) for n in (0, None, 4, 7, 200)] == [
            101,
            101,
            25,
            14,
            1,
        ]

        clamped = reconstruct_latent_preprocess_sdf_gt(
            [torch.tensor([-2.0, -0.05, 0.05, 2.0]), None], clamp_dist=0.1, device="cpu"
        )
        assert clamped[0].tolist() == pytest.approx([-0.1, -0.05, 0.05, 0.1])
        assert clamped[1] is None
        untouched = reconstruct_latent_preprocess_sdf_gt([torch.tensor([-2.0])], None, "cpu")
        assert untouched[0].tolist() == [-2.0]

    def test_project_latent_clamps_the_norm_in_place(self):
        """
        Fails if ``project_latent`` stops rescaling the latent in place into ``[min, max]`` or
        onto a single target, or accepts a three-element or string ``latent_norm``.
        """

        def norm_after(start, spec):
            latent = torch.zeros(1, 4)
            latent[0, 0] = start
            assert project_latent(latent, spec) is None
            return latent.norm().item()

        assert norm_after(5.0, (1.0, 2.0)) == pytest.approx(2.0)
        assert norm_after(0.5, (1.0, 2.0)) == pytest.approx(1.0)
        assert norm_after(1.5, (1.0, 2.0)) == pytest.approx(1.5)
        assert norm_after(5.0, 3.0) == pytest.approx(3.0)
        for bad in ((1.0, 2.0, 3.0), "1.0"):
            with pytest.raises(ValueError, match="latent_norm must be"):
                norm_after(1.5, bad)

    def test_latent_norm_penalty(self):
        """
        Fails if ``latent_norm_penalty`` changes its quadratic or Huber values or its
        ``penalty_weight`` scaling, stops computing quadratic for a single-target
        ``"barrier"``, or accepts an unknown ``penalty_type``.

        Huber's delta is 10% of the range.
        """

        def penalty(norm, target, **kwargs):
            latent = torch.zeros(1, 4)
            latent[0, 0] = norm
            return float(latent_norm_penalty(latent, target, **kwargs))

        assert penalty(1.5, (1.0, 2.0)) == 0.0
        assert penalty(0.5, (1.0, 2.0)) == pytest.approx(0.25)
        assert penalty(3.0, (1.0, 2.0)) == pytest.approx(1.0)
        assert penalty(3.0, (1.0, 2.0), penalty_weight=0.5) == pytest.approx(0.5)
        assert penalty(1.5, (1.0, 2.0), penalty_type="huber") == 0.0
        assert penalty(2.05, (1.0, 2.0), penalty_type="huber") == pytest.approx(0.00125, rel=1e-5)
        assert penalty(2.5, (1.0, 2.0), penalty_type="huber") == pytest.approx(0.045, rel=1e-5)
        assert penalty(3.0, 2.0) == pytest.approx(1.0)
        assert penalty(3.0, 2.0, penalty_type="barrier") == penalty(3.0, 2.0)
        for target in (2.0, (1.0, 2.0)):
            with pytest.raises(ValueError, match="Unknown penalty_type"):
                penalty(1.5, target, penalty_type="cubic")


class TestBarrierNormPenalty:
    def test_outside_the_range_raises_and_inside_is_finite(self):
        """
        Fails if ``latent_norm_penalty(penalty_type="barrier")`` returns NaN instead of
        raising outside ``(min, max)``, is non-finite inside it, or ``reconstruct_latent``
        with that range starts fitting instead of raising (KNOWN_ISSUES History 8).

        ``torch.ones(1, 256) * 0.01`` has norm 0.16, where ``latent_init_std=0.01`` starts a
        256-dim latent.
        """
        for latent in (torch.ones(1, 256) * 0.01, torch.ones(1, 16)):
            with pytest.raises(ValueError, match="barrier"):
                latent_norm_penalty(latent, (0.5, 1.0), penalty_type="barrier")
        inside = torch.ones(1, 16) * (0.75 / 4.0)
        assert torch.isfinite(latent_norm_penalty(inside, (0.5, 1.0), penalty_type="barrier"))
        with pytest.raises(ValueError, match="barrier"):
            fit(latent_norm=(0.5, 1.0), norm_penalty_type="barrier")
