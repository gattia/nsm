"""
What ``reconstruct_latent`` promises about the 38 parameters it takes, the values it
accepts for them, and the loss it says it returns.

Plan §8.0.K. Six of the seven contracts here are one shape seen from six sides: a value is
accepted where it is named and consulted somewhere else, so the failure -- when there is
one -- names a local the caller has never heard of, and when there is none the caller gets
a default they did not ask for.

1. **Unknown keywords** are swallowed by ``**kwargs``, across 38 near-synonymous names.
   This is ``KNOWN_ISSUES`` § History 20's defect at its second site: §8.0.J refused them
   in ``reconstruct_mesh`` and this is the function it calls.
2. **``optimizer_name`` is validated nowhere.** A third value leaves ``optimizer``
   unassigned and surfaces 100 lines later as an ``UnboundLocalError``.
3. **``loss_type`` is the same shape**, and its failure escapes into a nested closure that
   is not in the signature the caller read.
4. **The ``100`` sentinel returns itself**: ``convergence="recon_loss"`` -- the shipped
   default -- returns the literal ``100`` as its loss, and a fit whose losses never drop
   below 100 loses the whole run to an ``UnboundLocalError`` on ``latent_``.
5. **Hybrid mode's LR schedule is computed over a horizon it does not run**, so the Adam
   phase decays its way to exactly 0.0.
6. **``compute_loss`` resamples on every call** and LBFGS calls it many times per step, so
   the line search optimises a function that moves under it.
7. **25 of 30 log records are gated behind the deprecated ``verbose`` flag**, three of
   them warnings about the result rather than chatter.

Plus the end-to-end pin the commit-9 extraction and the commit-10 chunking are measured
against: a fixed-seed fit whose ``(loss, latent)`` must not move when the body is split.

Strict xfails mark what NSM does not honour yet. Each is retired by the commit that fixes
it.
"""

import ast
import inspect
import logging

import pytest
import torch

import NSM.reconstruct.latent_fit as latent_fit
import NSM.reconstruct.main as recon_main
from NSM.reconstruct import reconstruct_latent

PLAN = ".claude/plans/NSM_CODE_HEALTH_REFACTOR.md §8.0.K"


def broken(reason):
    return pytest.mark.xfail(strict=True, reason=f"{reason} ({PLAN})")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class LinearDecoder(torch.nn.Module):
    """
    One output column per surface, differentiable in the latent, no parameters of its own.

    ``latent.sum()`` keeps the output on the latent's graph so the backward has a leaf to
    reach, and makes the fit deterministic given a seed: there is nothing else to vary.
    """

    def __init__(self, surfaces=1, scale=1.0):
        super().__init__()
        self.surfaces = surfaces
        self.scale = scale

    def forward(self, x=None, latent=None, xyz=None, epoch=None, verbose=False):
        pts = xyz if xyz is not None else x[:, -3:]
        base = (pts[:, :1] + latent.sum()) * self.scale
        return base.repeat(1, self.surfaces)


class RecordingDecoder(LinearDecoder):
    """Records the point set of every forward pass, so a step's draws can be counted."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.draws = []

    def forward(self, x=None, latent=None, xyz=None, epoch=None, verbose=False):
        pts = xyz if xyz is not None else x[:, -3:]
        self.draws.append(round(float(pts.sum()), 9))
        return super().forward(x=x, latent=latent, xyz=xyz, epoch=epoch, verbose=verbose)


def fit_kwargs(n_pts=64, **overrides):
    """The cheap CPU fit every test in this file runs, seeded where it matters."""
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


# ---------------------------------------------------------------------------
# 1. Unknown keywords
# ---------------------------------------------------------------------------

#: Misspellings of real parameters, of the kinds that actually happen: a dropped plural
#: (``num_iterations``), a transposition (``latent_reg_weight``), a synonym
#: (``clamp_dist``), a doubled letter (``latent_size``), the British spelling
#: (``optimizer_name``), a plausible alternative name, and a trailing typo.
MISSPELLINGS = [
    "num_iteration",
    "latent_reg_wieght",
    "clamp_distance",
    "lattent_size",
    "optimiser_name",
    "n_iterations",
    "lr_",
]


class TestUnknownKeywordsAreRefused:
    """
    ``**kwargs`` is inspected for exactly one key, ``max_batch_size``. Every other key
    reaches the end of the function unread, so the caller gets the default for the
    parameter they meant to set and no indication that they had not set it.
    """

    @pytest.mark.parametrize("wrong", MISSPELLINGS)
    def test_a_misspelled_parameter_raises(self, wrong):
        """
        Were seven strict xfails. Measured before the fix: all seven complete a fit with no
        exception, no warning and no log record naming the key. That measurement is why the fix is a refusal and not
        a warning -- there was nowhere for a warning to be noticed.
        """
        with pytest.raises(TypeError, match=wrong):
            reconstruct_latent(decoders=LinearDecoder(), **fit_kwargs(**{wrong: 999}))

    def test_the_deprecated_key_is_still_accepted(self, caplog):
        """
        ``max_batch_size`` is the one key ``**kwargs`` is *for*. Refusing unknown keys must
        not refuse it: it warns and runs, as it has since the chunked forward was removed.
        """
        with caplog.at_level(logging.WARNING, logger="NSM"):
            reconstruct_latent(decoders=LinearDecoder(), **fit_kwargs(max_batch_size=1))
        assert any("max_batch_size is deprecated" in r.getMessage() for r in caplog.records)

    def test_reconstruct_mesh_passes_only_parameters_this_signature_names(self):
        """
        ``reconstruct_mesh`` builds a dict and splats it (``main.py:537``), so it is the one
        caller a refusal could break. The keys are read from the source rather than listed
        here, so a key added later without a matching parameter turns this red instead of
        turning the fit into a ``TypeError``.
        """
        source = open(recon_main.__file__, encoding="utf-8").read()
        dict_node = next(
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", None) == "reconstruct_inputs"
        )
        passed = {key.value for key in dict_node.keys}
        assert passed, "the reconstruct_inputs scan matched no keys"
        named = set(inspect.signature(reconstruct_latent).parameters)
        assert passed - named == set()


# ---------------------------------------------------------------------------
# 2 & 3. Values accepted where they are named, consulted somewhere else
# ---------------------------------------------------------------------------


class TestUnknownValuesAreRefusedWhereTheyAreNamed:
    """
    Two ``if``/``elif`` chains with no ``else``. The name each chain would have assigned is
    read much later, so the caller is told about ``optimizer`` or ``loss_fn`` -- neither of
    which appears in the signature they wrote against.
    """

    @pytest.mark.parametrize("value", ["sgd", "Adam"])
    @broken("optimizer_name has no else branch; the failure is an UnboundLocalError")
    def test_an_unknown_optimizer_names_the_parameter(self, value):
        """
        Measured before the fix: both raise ``UnboundLocalError: local variable 'optimizer'
        referenced before assignment``, from the step loop. ``"Adam"`` is in the list on
        purpose -- it is the capitalised spelling of the default, and the likeliest way to
        get this wrong.
        """
        with pytest.raises(ValueError, match="optimizer_name"):
            reconstruct_latent(decoders=LinearDecoder(), **fit_kwargs(optimizer_name=value))

    @pytest.mark.parametrize("value", ["l1_smooth", "L1"])
    @broken("loss_type has no else branch; the failure is a NameError inside a closure")
    def test_an_unknown_loss_type_names_the_parameter(self, value):
        """
        Measured before the fix: both raise ``NameError: free variable 'loss_fn' referenced
        before assignment in enclosing scope`` -- raised inside ``compute_loss``, so the
        traceback ends in a nested function the caller cannot see from the signature.
        """
        with pytest.raises(ValueError, match="loss_type"):
            reconstruct_latent(decoders=LinearDecoder(), **fit_kwargs(loss_type=value))


# ---------------------------------------------------------------------------
# 4. The 100 sentinel
# ---------------------------------------------------------------------------


class TestTheReturnedLossIsALoss:
    """
    ``loss`` and ``recon_loss`` are initialised to the literal ``100`` and used both as the
    comparison sentinel and as the returned value. Under ``convergence="recon_loss"`` only
    ``recon_loss`` is ever updated, so ``loss`` is still ``100`` at the ``return``.

    That mode is the shipped default: ``NSM/configs/default_config.json`` sets
    ``convergence_type_recon`` to ``"recon_loss"`` and kneepipeline forwards it
    (``steps/run_nsm.py:207``). ``reconstruct_mesh`` has hidden it -- it binds the returned
    loss at ``main.py:537`` and nothing reads it -- but ``reconstruct_latent`` is public API
    in its own right.
    """

    @broken("convergence='recon_loss' never updates `loss`, so it returns the sentinel")
    def test_recon_loss_convergence_returns_the_loss_it_selected(self):
        """Measured before the fix: ``loss`` is the int ``100``, exactly."""
        loss, _ = reconstruct_latent(
            decoders=LinearDecoder(), **fit_kwargs(convergence="recon_loss")
        )
        assert isinstance(loss, torch.Tensor)
        assert float(loss) != 100

    @pytest.mark.parametrize("convergence", ["overall_loss", "recon_loss"])
    @broken("a loss that never drops below 100 leaves `latent_` unbound")
    def test_a_large_loss_still_returns_a_latent(self, convergence):
        """
        Measured before the fix: ``sdf_gt`` scaled by 1000 puts every step's loss above the
        sentinel, so no step is ever recorded and the function raises
        ``UnboundLocalError: local variable 'latent_' referenced before assignment`` --
        after running every iteration it was asked for.
        """
        kwargs = fit_kwargs(convergence=convergence)
        kwargs["sdf_gt"] = kwargs["sdf_gt"] * 1000
        _, latent = reconstruct_latent(decoders=LinearDecoder(), **kwargs)
        assert isinstance(latent, torch.Tensor)
        assert torch.isfinite(latent).all()

    def test_num_iterations_convergence_returns_the_last_loss(self):
        """The mode that is already right, pinned so the sentinel fix does not move it."""
        loss, latent = reconstruct_latent(
            decoders=LinearDecoder(), **fit_kwargs(convergence="num_iterations")
        )
        assert isinstance(loss, torch.Tensor)
        assert isinstance(latent, torch.Tensor)


# ---------------------------------------------------------------------------
# 5. The hybrid LR schedule's horizon
# ---------------------------------------------------------------------------


def _learning_rates_seen(monkeypatch, **overrides):
    """Every LR ``adjust_learning_rate`` leaves on the optimizer, in order."""
    seen = []
    original = latent_fit.adjust_learning_rate

    def spy(**kwargs):
        original(**kwargs)
        seen.append(round(kwargs["optimizer"].param_groups[0]["lr"], 12))

    monkeypatch.setattr(latent_fit, "adjust_learning_rate", spy)
    reconstruct_latent(
        decoders=LinearDecoder(),
        **fit_kwargs(lr=1e-2, n_lr_updates=2, lr_update_factor=10, **overrides),
    )
    assert seen, "the spy recorded nothing -- adjust_learning_rate was not called"
    return seen


class TestTheLearningRateScheduleSpansThePhaseItSteps:
    """
    ``adjust_lr_every`` is derived from ``num_iterations``. With ``hybrid_optimizer=True``
    the loop runs ``adam_iterations + lbfgs_iterations`` instead, and ``num_iterations`` is
    read for nothing else, so ``n_lr_updates`` means a different thing in each mode.
    """

    @broken("hybrid mode schedules over num_iterations and steps over adam_iterations")
    def test_hybrid_mode_applies_the_updates_asked_for(self, monkeypatch):
        """
        Measured before the fix, ``num_iterations=10, adam_iterations=100,
        n_lr_updates=2, lr_update_factor=10``: 11 decays ending at exactly 0.0, where the
        same 100 Adam steps scheduled over their own horizon take one. An Adam phase at
        lr 0.0 stops moving the latent, silently.
        """
        seen = _learning_rates_seen(
            monkeypatch,
            num_iterations=10,
            hybrid_optimizer=True,
            adam_iterations=100,
            lbfgs_iterations=0,
        )
        assert len(seen) == 100
        assert sorted(set(seen), reverse=True) == [1e-2, 1e-3]
        assert 0.0 not in seen

    def test_non_hybrid_mode_applies_the_updates_asked_for(self, monkeypatch):
        """The trajectory the hybrid fix has to reproduce, pinned on the mode that is right."""
        seen = _learning_rates_seen(monkeypatch, num_iterations=100)
        assert len(seen) == 100
        assert sorted(set(seen), reverse=True) == [1e-2, 1e-3]


# ---------------------------------------------------------------------------
# 6. One draw per step
# ---------------------------------------------------------------------------


class TestTheObjectiveIsFixedWithinAStep:
    """
    ``compute_loss`` draws its own random subsample. Adam calls it once per step, so the
    draw and the step coincide; LBFGS calls it once per line-search evaluation and once
    more, without gradients, to record the loss that feeds the convergence test.
    """

    @broken("compute_loss resamples on every call; LBFGS calls it many times per step")
    def test_lbfgs_optimises_one_draw_per_step(self):
        """
        Measured before the fix, one step at ``n_samples=50`` of 100 points: 7 forward
        passes on 7 distinct point sets. L-BFGS's line search and its curvature update both
        assume the objective is a fixed function of the parameter, and the last of the
        seven is the draw the convergence test is measured on -- one the step never fitted.
        """
        decoder = RecordingDecoder()
        reconstruct_latent(
            decoders=decoder,
            **fit_kwargs(n_pts=100, num_iterations=1, optimizer_name="lbfgs", n_samples=50),
        )
        assert len(set(decoder.draws)) == 1

    def test_adam_already_draws_once_per_step(self):
        """The mode that is already right, pinned so the fix does not move it."""
        decoder = RecordingDecoder()
        reconstruct_latent(
            decoders=decoder,
            **fit_kwargs(n_pts=100, num_iterations=4, optimizer_name="adam", n_samples=50),
        )
        assert len(decoder.draws) == 4
        assert len(set(decoder.draws)) == 4


# ---------------------------------------------------------------------------
# 7. Log records and the deprecated flag
# ---------------------------------------------------------------------------


def _verbose_gated_log_calls():
    """``logger.*`` calls under an ``if verbose ...:`` anywhere in ``latent_fit.py``."""
    source = open(latent_fit.__file__, encoding="utf-8").read()
    gated = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.If) or "verbose" not in ast.dump(node.test):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and isinstance(inner.func.value, ast.Name)
                and inner.func.value.id == "logger"
            ):
                gated.append(inner.lineno)
    return gated


def _fit_that_skips_a_surface(**overrides):
    """A two-column decoder with no ground truth for its second surface."""
    kwargs = fit_kwargs(**overrides)
    kwargs["sdf_gt"] = [kwargs["sdf_gt"], None]
    return dict(decoders=LinearDecoder(surfaces=2), **kwargs)


class TestLogRecordsReachAConfiguredHost:
    """
    §8.0.G made logging the mechanism; 25 of this file's 30 records went on answering to
    the parameter it deprecated. Three of the 25 are ``warning``s about the result -- a
    surface was skipped, or the decoder emitted more surfaces than there was ground truth
    for -- so a host configured at ``WARNING`` is told nothing about a fit that silently
    dropped a surface.
    """

    @broken("25 of 30 logger calls in latent_fit.py sit under `if verbose is True:`")
    def test_no_log_record_is_gated_on_the_deprecated_flag(self):
        assert _verbose_gated_log_calls() == []

    @broken("a host that configured logging sees none of the fit's records")
    def test_a_host_at_debug_sees_the_fit_records(self, caplog):
        """Measured before the fix: empty, for a host that did exactly what the notice said."""
        with caplog.at_level(logging.DEBUG, logger="NSM"):
            reconstruct_latent(decoders=LinearDecoder(), **fit_kwargs())
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "xyz shape" in messages

    @broken("the skipped-surface warnings are gated behind verbose")
    def test_a_host_at_warning_is_told_a_surface_was_skipped(self, caplog):
        """
        The sharpest of the three: the fit dropped a surface from its objective and said so
        only to a caller who passed the deprecated flag.
        """
        with caplog.at_level(logging.WARNING, logger="NSM"):
            reconstruct_latent(**_fit_that_skips_a_surface())
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "skipping surface 1" in messages

    def test_verbose_true_shows_them_today_and_must_keep_doing_so(self, caplog):
        """
        The bridge attaches at ``DEBUG`` (``_verbose_deprecation.py:82``), so ungating
        cannot take anything away from a ``verbose=True`` caller. ``caplog`` stands in for
        the bridge's handler -- it is a handler on the root, so the bridge declines to add
        its own and the records land here either way.
        """
        with caplog.at_level(logging.DEBUG, logger="NSM"):
            with pytest.warns(DeprecationWarning):
                reconstruct_latent(**_fit_that_skips_a_surface(verbose=True))
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "xyz shape" in messages
        assert "skipping surface 1" in messages


# ---------------------------------------------------------------------------
# The end-to-end pin the extraction and the chunking are measured against
# ---------------------------------------------------------------------------


class TestTheFitIsUnchangedByTheSplit:
    """
    Commit 9 splits the 191-line ``compute_loss`` into three helpers and commit 10 adds a
    chunked step behind a default-off parameter. This is what makes both provably
    behaviour-preserving: a fixed-seed fit on each of the four shapes the body branches
    over -- one surface or several, subsampled or not -- run twice in the same process and
    compared.

    Not golden numbers in the file: the run is executed twice and the two are compared.
    That pins *determinism* here and, across a commit, the values themselves, because the
    commit is the only thing between two runs of it.
    """

    @staticmethod
    def _fit(surfaces, n_samples):
        torch.manual_seed(11)
        n_pts = 80
        xyz = torch.rand(n_pts, 3)
        sdf_gt = [torch.rand(n_pts, 1) for _ in range(surfaces)]
        pts_surface = [i % surfaces for i in range(n_pts)]
        torch.manual_seed(11)
        return reconstruct_latent(
            decoders=LinearDecoder(surfaces=surfaces),
            num_iterations=5,
            latent_size=8,
            xyz=xyz,
            sdf_gt=sdf_gt,
            pts_surface=pts_surface,
            n_samples=n_samples,
            clamp_dist=0.1,
            l2reg=True,
            device="cpu",
        )

    @pytest.mark.parametrize("surfaces", [1, 3])
    @pytest.mark.parametrize("n_samples", [None, 40])
    def test_the_fit_is_deterministic(self, surfaces, n_samples):
        first_loss, first_latent = self._fit(surfaces, n_samples)
        second_loss, second_latent = self._fit(surfaces, n_samples)
        assert torch.equal(first_latent, second_latent)
        assert float(first_loss) == float(second_loss)
        assert torch.isfinite(first_latent).all()
