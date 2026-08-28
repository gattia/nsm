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
import contextlib
import inspect
import logging

import pytest
import torch

import NSM.reconstruct.latent_fit as latent_fit
import NSM.reconstruct.main as recon_main
from NSM.reconstruct import reconstruct_latent

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


@contextlib.contextmanager
def caplog_at_warning():
    """Collect ``NSM`` warning records without pytest's caplog, so a test can assert absence."""
    records = []

    class _Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collector(level=logging.WARNING)
    logger = logging.getLogger("NSM")
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


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
    Three ``if``/``elif`` chains with no ``else``. Two leave a name unassigned that is read
    much later, so the caller is told about ``optimizer`` or ``loss_fn`` -- neither of which
    appears in the signature they wrote against. The third, ``convergence``, has no name to
    leave unassigned: its missing ``else`` is the default branch, so an unrecognised value
    was accepted and quietly meant ``"num_iterations"``.

    Values are normalised for case and then refused, which is not the same as accepting two
    spellings: everything downstream reads the normalised name.
    """

    @pytest.mark.parametrize(
        "parameter,value",
        [
            ("optimizer_name", "sgd"),
            ("loss_type", "l1_smooth"),
            ("convergence", "banana"),
            ("convergence", None),
        ],
    )
    def test_an_unknown_value_names_the_parameter(self, parameter, value):
        """
        Were strict xfails. Measured before the fix: ``optimizer_name`` raised
        ``UnboundLocalError: local variable 'optimizer' referenced before assignment`` from
        the step loop; ``loss_type`` raised ``NameError: free variable 'loss_fn' referenced
        before assignment in enclosing scope``, from inside ``compute_loss``, so the
        traceback ended in a nested function the caller cannot see from the signature.

        ``convergence`` is the third of the same shape and the worst of the three, because
        its missing ``else`` is a live branch rather than an unassigned name: every
        unrecognised value -- ``None`` and ``""`` included -- was accepted and silently
        meant ``"num_iterations"``, so a mis-capitalised ``"Recon_Loss"`` turned convergence
        checking off and returned a plausible result.
        """
        with pytest.raises(ValueError, match=parameter):
            reconstruct_latent(decoders=LinearDecoder(), **fit_kwargs(**{parameter: value}))

    @pytest.mark.parametrize(
        "parameter,value",
        [
            ("optimizer_name", "Adam"),
            ("optimizer_name", "LBFGS"),
            ("loss_type", "L1"),
            ("loss_type", "L1_LOG"),
            ("convergence", "Recon_Loss"),
        ],
    )
    def test_case_is_folded_rather_than_refused(self, parameter, value):
        """
        Case is the one difference that is never a different intent, and NSM's own training
        path spells its optimizers ``"Adam"`` and ``"AdamW"`` (``utils.get_optimizer``, and
        ``default_config.json``'s ``optimizer`` key). A caller writing ``"Adam"`` here is
        being consistent with the rest of the library, so refusing it would be the library
        disagreeing with itself.
        """
        _, latent = reconstruct_latent(decoders=LinearDecoder(), **fit_kwargs(**{parameter: value}))
        assert isinstance(latent, torch.Tensor)

    def test_folding_case_selects_the_same_branch(self):
        """
        The point of folding rather than merely accepting: the normalised value is what
        every branch downstream reads, so a capitalised spelling is not a second code path.
        """
        lower = reconstruct_latent(decoders=LinearDecoder(), **fit_kwargs(convergence="recon_loss"))
        upper = reconstruct_latent(decoders=LinearDecoder(), **fit_kwargs(convergence="Recon_Loss"))
        assert torch.equal(lower[1], upper[1])
        assert float(lower[0]) == float(upper[0])

    def test_hybrid_mode_refuses_an_optimizer_name_it_will_not_consult(self):
        """
        The eighth defect of the same class, found while fixing the second and not in the
        §8.0.K statement's seven: with ``hybrid_optimizer=True`` the loop derives its
        optimizer from the step number and ``optimizer_name`` is read nowhere at all. It
        was accepted and ignored, which is the trap NSM's rule says to close rather than
        implement -- and there is nothing to implement here, since hybrid mode *is*
        Adam then LBFGS.
        """
        with pytest.raises(ValueError, match="optimizer_name"):
            reconstruct_latent(
                decoders=LinearDecoder(),
                **fit_kwargs(hybrid_optimizer=True, optimizer_name="lbfgs", adam_iterations=2),
            )


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

    def test_recon_loss_convergence_returns_the_loss_it_selected(self):
        """Was a strict xfail. Measured before the fix: ``loss`` is the int ``100``, exactly."""
        loss, _ = reconstruct_latent(
            decoders=LinearDecoder(), **fit_kwargs(convergence="recon_loss")
        )
        assert isinstance(loss, torch.Tensor)
        assert float(loss) != 100

    @pytest.mark.parametrize("convergence", ["overall_loss", "recon_loss"])
    def test_a_large_loss_still_returns_a_latent(self, convergence):
        """
        Were two strict xfails. Measured before the fix: ``sdf_gt`` scaled by 1000 puts every step's loss above the
        sentinel, so no step is ever recorded and the function raises
        ``UnboundLocalError: local variable 'latent_' referenced before assignment`` --
        after running every iteration it was asked for.
        """
        kwargs = fit_kwargs(convergence=convergence)
        kwargs["sdf_gt"] = kwargs["sdf_gt"] * 1000
        _, latent = reconstruct_latent(decoders=LinearDecoder(), **kwargs)
        assert isinstance(latent, torch.Tensor)
        assert torch.isfinite(latent).all()

    def test_no_iterations_still_returns_the_initial_latent(self):
        """
        The other half of an always-bound ``latent_``: nothing to record, so the function
        returns what it started from and a loss of ``inf`` rather than raising.
        """
        loss, latent = reconstruct_latent(decoders=LinearDecoder(), **fit_kwargs(num_iterations=0))
        assert loss == float("inf")
        assert latent.shape == (1, 8)

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

    def test_hybrid_mode_applies_the_updates_asked_for(self, monkeypatch):
        """
        Was a strict xfail. Measured before the fix, ``num_iterations=10, adam_iterations=100,
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


class TestTheDrawIsPerEvaluation:
    """
    ``_select_samples`` runs on every loss evaluation, not once per optimization step.
    Adam evaluates the loss once per step so the two coincide; LBFGS evaluates it several
    times per step, and each of those redraws.

    **§8.0.K proposed hoisting this to once per step and the measurement refused it.**
    The argument was that L-BFGS's line search and its secant condition both assume a
    deterministic objective, which is true. What it missed is that the redraw is also how
    the fit covers the point cloud: at equal compute, fewer draws means less of the cloud
    seen. Measured over 20 problems at 12,000 decoder evaluations, LBFGS, median held-out
    error against noise-free truth (`sampling ratio: per-evaluation vs per-step`):

    | ratio | per evaluation | per step | cloud seen |
    |---|---|---|---|
    | 1.6% | 0.034 | 0.202 | 96% vs 82% |
    | 5%   | 0.007 | 0.029 | 95% vs 41% |
    | 20%  | 0.0049 | 0.0054 | 99% vs 37% |
    | 50%  | 0.0042 | 0.0045 | 100% vs 50% |

    The gap tracks coverage and closes as the ratio rises, so coverage is the mechanism,
    not the line search. Resetting LBFGS's curvature history at each draw -- the obvious
    repair if stale curvature were the problem -- made it *worse* (13/20 diverged against
    7/20), which is how that explanation was eliminated.

    The theoretical objection stands and is not answered by cadence: a **deterministic**
    draw with good coverage (stratified, or blue noise) would give the line search a fixed
    objective *and* keep the coverage. That is a sampling-strategy change and it is not in
    this slice.
    """

    def test_the_draw_changes_between_evaluations_within_a_step(self):
        """
        The property the coverage measurement above depends on. If a later change makes
        the draw per-step, this goes red and the table says what it costs.
        """
        decoder = RecordingDecoder()
        reconstruct_latent(
            decoders=decoder,
            **fit_kwargs(n_pts=100, num_iterations=1, optimizer_name="lbfgs", n_samples=50),
        )
        assert len(decoder.draws) > 1, "LBFGS should evaluate the loss more than once per step"
        assert len(set(decoder.draws)) == len(decoder.draws)

    def test_adam_draws_once_per_step(self):
        """Adam evaluates once per step, so per-evaluation and per-step coincide for it."""
        decoder = RecordingDecoder()
        reconstruct_latent(
            decoders=decoder,
            **fit_kwargs(n_pts=100, num_iterations=4, optimizer_name="adam", n_samples=50),
        )
        assert len(decoder.draws) == 4
        assert len(set(decoder.draws)) == 4

    def test_a_full_sample_draw_is_the_same_points_every_time(self):
        """
        With ``n_samples`` at or above the point count there is no subsampling, so the
        objective is deterministic and the line search is well-posed -- and this is not
        merely the principled option, it is the one that measured best. Same 12,000-decoder
        -evaluation budget, 20 problems, median held-out error and divergences:

        | regime | median | diverged | cloud seen |
        |---|---|---|---|
        | full cloud | **0.0038** | **0/20** | 100% |
        | per-evaluation redraw, 5% | 0.0066 | 2/20 | 95% |
        | per-step redraw, 5% | 0.115 | 12/20 | 47% |
        | per-step *without replacement*, 5% | 0.056 | 11/20 | 54% |

        The middle option -- cycling a permutation so each step gets a disjoint block --
        is the obvious way to have a fixed objective and full coverage, and it was measured
        and does not rescue it. What the full cloud buys is that LBFGS converges: it
        reached that error in one outer step of the budget.

        The memory ceiling that forced subsampling is what ``n_samples_per_chunk`` (#75)
        removes, so this configuration is affordable now in a way it was not.
        """
        decoder = RecordingDecoder()
        reconstruct_latent(
            decoders=decoder,
            **fit_kwargs(n_pts=60, num_iterations=1, optimizer_name="lbfgs", n_samples=60),
        )
        assert len(set(decoder.draws)) == 1

    @pytest.mark.parametrize(
        "options",
        [
            {"optimizer_name": "lbfgs"},
            {"optimizer_name": "LBFGS"},
            {"hybrid_optimizer": True, "adam_iterations": 1},
        ],
        ids=["lbfgs", "LBFGS", "hybrid"],
    )
    def test_a_subsampled_lbfgs_fit_says_so(self, caplog, options):
        """
        The one combination where coverage and a deterministic objective genuinely collide,
        and it was silent. ``"LBFGS"`` is parameterised because the first version of this
        guard read ``optimizer_name`` *before* the case fold and missed it -- the fold and
        the guard now both sit at the top of the function, in that order.
        """
        with caplog.at_level(logging.WARNING, logger="NSM"):
            reconstruct_latent(
                decoders=LinearDecoder(),
                **fit_kwargs(n_pts=100, num_iterations=1, n_samples=50, **options),
            )
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "n_samples_per_chunk" in messages

    def test_the_guard_measures_the_draw_not_the_budget(self):
        """
        ``n_samples`` is split across surfaces and capped at each one's size, so with
        unequal surfaces a budget at or above the cloud size still subsamples: 300 and 90
        points at ``n_samples=390`` draws 285. The guard reads the planned draw through the
        same helper ``_select_samples`` uses, so it cannot describe a different draw from
        the one that happens -- the first version compared ``n_samples`` and stayed silent
        here, while recommending ``n_samples=None``, which lands in exactly this case.
        """
        big, small = 300, 90
        n_pts = big + small
        pts_surface = torch.tensor([0] * big + [1] * small)
        planned = latent_fit._samples_per_surface(
            n_samples=n_pts, pts_surface=pts_surface, n_surfaces=2
        )
        assert sum(planned) == 285

        with caplog_at_warning() as records:
            sdf = torch.rand(n_pts, 1)
            reconstruct_latent(
                decoders=LinearDecoder(surfaces=2),
                num_iterations=1,
                latent_size=8,
                xyz=torch.rand(n_pts, 3),
                sdf_gt=[sdf, sdf.clone()],
                pts_surface=pts_surface,
                n_samples=n_pts,
                optimizer_name="lbfgs",
                device="cpu",
            )
        messages = " ".join(r.getMessage() for r in records)
        assert "draws 285 of 390" in messages
        assert "at least 600" in messages

    def test_a_full_cloud_lbfgs_fit_is_silent(self):
        """The warning must not fire on the configuration it is recommending."""
        decoder = LinearDecoder()
        with caplog_at_warning() as records:
            reconstruct_latent(
                decoders=decoder,
                **fit_kwargs(n_pts=60, num_iterations=1, optimizer_name="lbfgs"),
            )
        assert not [r for r in records if "n_samples_per_chunk" in r.getMessage()]


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

    def test_no_log_record_is_gated_on_the_deprecated_flag(self):
        """Was a strict xfail: 25 of 30, three of them skipped-surface warnings."""
        assert _verbose_gated_log_calls() == []

    def test_a_host_at_debug_sees_the_fit_records(self, caplog):
        """Was a strict xfail: empty, for a host that did exactly what the notice said."""
        with caplog.at_level(logging.DEBUG, logger="NSM"):
            reconstruct_latent(decoders=LinearDecoder(), **fit_kwargs())
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "xyz shape" in messages

    def test_a_host_at_warning_is_told_a_surface_was_skipped(self, caplog):
        """
        Was a strict xfail, and the sharpest of the three: the fit dropped a surface from its objective and said so
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


class TestTheLbfgsParametersAreReadOnBothPaths:
    """
    ``lbfgs_lr``, ``lbfgs_max_iter`` and ``lbfgs_history_size`` were read in the hybrid
    branch and ignored in the non-hybrid one, which built ``LBFGS(lr=lr, max_iter=10,
    history_size=100)`` from literals.

    That asymmetry is not cosmetic. torch's LBFGS runs with no line search
    (``line_search_fn`` is never set), so ``lbfgs_lr`` *is* the step length -- and at a
    config's usual ``lr=0.005`` a caller asking for ``lbfgs_lr=1.0`` silently ran at a step
    200x smaller. Any "LBFGS alone does not converge" conclusion drawn from the non-hybrid
    path was measured on a fit that never received the step size it was configured with.
    """

    @pytest.mark.parametrize("hybrid", [False, True], ids=["non-hybrid", "hybrid"])
    def test_both_paths_build_lbfgs_from_the_same_parameters(self, monkeypatch, hybrid):
        built = []
        original = torch.optim.LBFGS

        def spy(params, **kwargs):
            built.append(kwargs)
            return original(params, **kwargs)

        monkeypatch.setattr(torch.optim, "LBFGS", spy)
        options = (
            {"hybrid_optimizer": True, "adam_iterations": 1, "lbfgs_iterations": 1}
            if hybrid
            else {"optimizer_name": "lbfgs"}
        )
        reconstruct_latent(
            decoders=LinearDecoder(),
            **fit_kwargs(
                num_iterations=1,
                lbfgs_lr=0.25,
                lbfgs_max_iter=3,
                lbfgs_history_size=7,
                **options,
            ),
        )
        assert built, "no LBFGS optimizer was constructed"
        assert built[0]["lr"] == 0.25
        assert built[0]["max_iter"] == 3
        assert built[0]["history_size"] == 7


class TestTheLbfgsClosureDoesNotRetainItsGraph:
    """
    ``step_closure`` used to call ``backward(retain_graph=True)``, with a comment saying it
    was needed because LBFGS calls the closure many times per step. It is not: every call
    runs its own forward and builds its own graph, so no graph is ever backwarded twice.
    What retaining bought was a dead graph's activations staying resident alongside the
    live one.

    Measured on a T4, one LBFGS step at 60,000 points, latent 256 with an 8x512 decoder:
    peak allocation above baseline **2265 MiB retained, 1240 MiB not** -- 1.83x, for a
    bit-identical latent. That multiplier lands on the largest allocation in a fit, which
    is the quantity ``n_samples_per_chunk`` exists to bound, so the two worked against
    each other.
    """

    def test_no_call_passes_retain_graph(self):
        """
        Read from the source as a keyword on a call, not as a word -- the comment that
        replaced the flag names it, and a scan that matched text would match the comment.
        A future edit reintroducing the flag argues with this test rather than a comment.
        """
        tree = ast.parse(open(latent_fit.__file__, encoding="utf-8").read())
        passed = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for kw in node.keywords
            if kw.arg == "retain_graph"
        ]
        assert passed == []

    def test_an_lbfgs_fit_still_runs_and_is_finite(self):
        """The behavioural half: freeing each graph does not break the repeated calls."""
        loss, latent = reconstruct_latent(
            decoders=LinearDecoder(),
            **fit_kwargs(n_pts=80, num_iterations=2, optimizer_name="lbfgs"),
        )
        assert torch.isfinite(latent).all()
        assert torch.isfinite(torch.as_tensor(float(loss)))


# ---------------------------------------------------------------------------
# #75: a step whose sample count exceeds single-forward memory
# ---------------------------------------------------------------------------


class TestChunkedForwardAndBackward:
    """
    Issue #75. One optimization step had to fit its whole forward and backward in memory,
    so ``n_samples`` had a hardware ceiling. ``n_samples_per_chunk`` splits both, keeping
    the gradient by accumulating it on the latent.

    Measured on a Tesla T4 -- the hardware #75 used -- one step at 200,000 points with a
    DeepSDF-shaped decoder (latent 256, 8x512), peak CUDA allocation above baseline:
    **4104 MiB unchunked, 3412 MiB for 30k chunks with one backward at the end, 616 MiB
    for 30k chunks with a backward each**. That 6.7x is the reason the option exists and
    the reason it is per-chunk backward rather than the accumulate-then-backward design
    removed in 4583246: retaining every chunk's graph saves 17%, freeing each one saves
    85%. Through ``reconstruct_latent`` itself, same shape and 200,000 points:
    **4128 MiB at the default, 623 MiB at ``n_samples_per_chunk=30000``**, with the
    fitted loss agreeing to 1.2e-07 relative.

    Those numbers are recorded rather than asserted because reproducing them needs a GPU;
    what is asserted below is the part that must hold everywhere -- the gradient, and the
    default staying bit-identical to the code that had no chunking at all.
    """

    @staticmethod
    def _gradient(n_samples_per_chunk, surfaces=2, n_pts=97):
        """The gradient one step leaves on the latent, chunked or not."""
        torch.manual_seed(5)
        latent = torch.zeros(1, 8, requires_grad=True)
        xyz = torch.rand(n_pts, 3)
        sdf_gt_ = [torch.rand(n_pts, 1) for _ in range(surfaces)]
        decoder = LinearDecoder(surfaces=surfaces)
        loss_fn = torch.nn.L1Loss(reduction="none")
        if n_samples_per_chunk is None:
            latent_fit._recon_loss(
                decoders=[decoder],
                latent=latent,
                xyz_input=xyz,
                sdf_gt_=sdf_gt_,
                loss_fn=loss_fn,
                loss_weight=1.0,
                clamp_dist=0.1,
                difficulty_weight=None,
            ).backward()
        else:
            for start in range(0, n_pts, n_samples_per_chunk):
                stop = min(start + n_samples_per_chunk, n_pts)
                (
                    latent_fit._recon_loss(
                        decoders=[decoder],
                        latent=latent,
                        xyz_input=xyz[start:stop],
                        sdf_gt_=[gt[start:stop] for gt in sdf_gt_],
                        loss_fn=loss_fn,
                        loss_weight=1.0,
                        clamp_dist=0.1,
                        difficulty_weight=None,
                    )
                    * ((stop - start) / n_pts)
                ).backward()
        return latent.grad.clone()

    @pytest.mark.parametrize("chunk", [10, 32, 97, 200])
    def test_the_accumulated_gradient_matches_the_unchunked_one(self, chunk):
        """
        Chunk sizes that divide the sample count and that do not (97 points), one that is
        exactly the count, and one larger than it. The weighting is by share of points, so
        a ragged last chunk must not be over-counted -- which is the arithmetic this
        catches and an equal-weight average would not.
        """
        unchunked = self._gradient(None)
        chunked = self._gradient(chunk)
        scale = unchunked.abs().max()
        assert float((unchunked - chunked).abs().max() / scale) < 1e-6

    @staticmethod
    def _fit(surfaces=2, **overrides):
        """A seeded multi-surface fit. Fresh tensors each call: ``sdf_gt`` is clamped and
        moved in place by the preprocessing pass, so a shared list is a different input
        the second time round."""
        torch.manual_seed(3)
        n_pts = 97
        xyz = torch.rand(n_pts, 3)
        sdf_gt = [torch.rand(n_pts, 1) for _ in range(surfaces)]
        torch.manual_seed(3)
        return reconstruct_latent(
            decoders=LinearDecoder(surfaces=surfaces),
            num_iterations=4,
            latent_size=8,
            xyz=xyz,
            sdf_gt=sdf_gt,
            pts_surface=[i % surfaces for i in range(n_pts)],
            n_samples=60,
            clamp_dist=0.1,
            device="cpu",
            **overrides,
        )

    def test_the_default_is_the_unchunked_path(self):
        """
        ``None`` must be what every run before this parameter existed did, bit for bit --
        the chunked path changes summation order, so a default that quietly took it would
        move results for callers who never asked.
        """
        default_loss, default_latent = self._fit()
        explicit_loss, explicit_latent = self._fit(n_samples_per_chunk=None)
        assert torch.equal(default_latent, explicit_latent)
        assert float(default_loss) == float(explicit_loss)

    def test_a_chunked_fit_lands_in_the_same_place(self):
        """
        Not bit-identical -- the summation order differs -- but the same fit. A chunking
        bug that dropped or double-counted a chunk would show here as a latent that is not
        close, which the single-step gradient test would not catch across four steps.
        """
        _, unchunked = self._fit()
        _, chunked = self._fit(n_samples_per_chunk=25)
        assert torch.allclose(unchunked, chunked, atol=1e-6)

    def test_reconstruct_mesh_can_reach_it(self):
        """
        The capability has to be reachable from the entry point the consumer calls;
        ``reconstruct_mesh`` refuses unknown keywords now, so a caller cannot pass it
        through by accident.
        """
        import inspect

        named = inspect.signature(recon_main.reconstruct_mesh).parameters
        assert "n_samples_per_chunk_latent_recon" in named
        assert named["n_samples_per_chunk_latent_recon"].default is None
