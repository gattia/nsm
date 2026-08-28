"""
What ``train_epoch`` promises about the config it reads, the epoch it runs, and the
metrics it returns.

Plan §8.0.L. Four of the seven contracts here are §8.0.K's shape seen from four sides: a
value is checked, or a loop bounded, somewhere other than where the value is named, so the
failure -- when there is one -- names a local or a tensor size the caller has never heard
of, and when there is none the caller gets an epoch they did not ask for.

1. **``batch_split`` is the number of splits asked for, not the number produced.**
   ``torch.chunk(t, k)`` returns at most ``k`` pieces; the split loop is
   ``range(config["batch_split"])``. Second site in the same module, in the eval warm-up
   whose docstring says it unpacks the batch exactly the way ``train_epoch`` does (#42).
2. **The logged latent-norm statistics are the last split's.** ``KNOWN_ISSUES`` § History
   12 (#59) at its second site: that fix moved ``=`` to ``+=`` on the batch loop and left
   the split loop underneath it.
3. **``multi_object_overlap`` is refused 174 lines after it is named**, as a bare
   ``Exception``, after the backward pass.
4. **``train_epoch`` is the third entry point to the gated eikonal loss** and the only one
   that does not raise. Its refusal is asserted beside the other two, in
   ``testing/NSM/test_losses.py``.
5. **``surface_weighting``'s length check is a bare ``assert``**, in the innermost loop,
   alongside an epoch-constant weight normalisation.
6. **``samples_per_object_per_batch`` restates the dataset's ``subsample``** and nothing
   checks the two agree.
7. **20 ``logger.debug`` records sit behind 8 ``config["verbose"]`` gates** -- §8.0.G's
   residue in its config-key form.

Plus the invariance matrix the commit-10 and commit-11 extractions are measured against:
the epoch's ``log_dict`` must be self-consistent and must not depend on ``batch_split``,
across surface counts, priors and the variational branch.

Strict xfails mark what NSM does not honour yet. Each is retired by the commit that fixes
it.
"""

import contextlib
import logging
import math
import pathlib
import subprocess
import sys
import textwrap

import pytest
import torch

from NSM.train.train_deep_sdf import _schedule_free_eval_warmup, train_epoch
from NSM.utils import get_latent_vecs, get_learning_rate_schedules, get_optimizer

#: The repo root, so the ``-O`` subprocess below can import NSM the way the suite does.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

LATENT_SIZE = 4
N_SUBJECTS = 4
N_SAMPLES = 8

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class LinearDecoder(torch.nn.Module):
    """
    One output column per surface, read from the concatenated ``[latent | xyz]`` input
    ``train_epoch`` builds. Linear so that a second derivative exists -- the eikonal path
    is reachable here, which is the point of contract 4.
    """

    def __init__(self, surfaces=2, latent_size=LATENT_SIZE):
        super().__init__()
        self.linear = torch.nn.Linear(latent_size + 3, surfaces)

    def forward(self, x, epoch=None):
        return self.linear(x)


class TinyDataset(torch.utils.data.Dataset):
    """``(sample_dict, index)`` in the shape ``MultiSurfaceSDFSamples`` yields."""

    def __init__(self, surfaces=2, n_subjects=N_SUBJECTS, n_samples=N_SAMPLES, seed=0):
        generator = torch.Generator().manual_seed(seed)
        self.xyz = torch.rand(n_subjects, n_samples, 3, generator=generator)
        self.gt_sdf = torch.rand(n_subjects, n_samples, surfaces, generator=generator) - 0.5
        self.n_subjects = n_subjects

    def __len__(self):
        return self.n_subjects

    def __getitem__(self, index):
        return {"xyz": self.xyz[index], "gt_sdf": self.gt_sdf[index]}, index


def learning_rate_schedule(model_lr=1e-3, latent_lr=1e-3):
    """Two constant schedules, one per ``Target``, in the form the config declares them."""
    return [
        {"Target": "model", "Type": "Constant", "Value": model_lr},
        {"Target": "latent", "Type": "Constant", "Value": latent_lr},
    ]


def epoch_config(model_lr=1e-3, latent_lr=1e-3, **overrides):
    """Every key ``train_epoch`` reads, at values that make one cheap CPU epoch run."""
    config = {
        "optimizer": "Adam",
        "weight_decay": 1e-4,
        "device": "cpu",
        "batch_split": 1,
        "samples_per_object_per_batch": N_SAMPLES,
        "enforce_minmax": False,
        "clamp_dist": 1.0,
        "surface_accuracy_e": None,
        "surface_accuracy_schedule": "linear",
        "surface_accuracy_cooldown": None,
        "sample_difficulty_weight": None,
        "sample_difficulty_weight_schedule": "linear",
        "sample_difficulty_cooldown": None,
        "code_regularization": True,
        "code_regularization_type_prior": "identity",
        "code_regularization_weight": 1e-4,
        "code_regularization_warmup": 2,
        "code_cyclic_anneal": False,
        "n_epochs": 4,
        "grad_clip": None,
        "verbose": False,
        "log_latent": None,
        "latent_size": LATENT_SIZE,
        "latent_bound": 10,
        "latent_init_std": 0.01,
        "latent_init_normal": True,
        "variational": False,
        "LearningRateSchedule": learning_rate_schedule(model_lr, latent_lr),
    }
    config.update(overrides)
    config["lr_schedules"] = get_learning_rate_schedules(config)
    return config


def epoch_inputs(surfaces=2, objects_per_batch=2, seed=42, **overrides):
    """``(model, data_loader, latent_vecs, optimizer, config)`` for one seeded epoch."""
    config = epoch_config(**overrides)
    dataset = TinyDataset(surfaces=surfaces)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=objects_per_batch, shuffle=False)
    torch.manual_seed(seed)
    model = LinearDecoder(surfaces=surfaces)
    latent_vecs = get_latent_vecs(len(dataset), config)
    optimizer = get_optimizer(
        model,
        latent_vecs,
        lr_schedules=config["lr_schedules"],
        optimizer=config["optimizer"],
        weight_decay=config["weight_decay"],
    )
    return model, data_loader, latent_vecs, optimizer, config


def run_epoch(surfaces=2, objects_per_batch=2, seed=42, epoch=1, **overrides):
    """One epoch's ``log_dict``, from a fresh seeded model and embedding."""
    model, data_loader, latent_vecs, optimizer, config = epoch_inputs(
        surfaces=surfaces, objects_per_batch=objects_per_batch, seed=seed, **overrides
    )
    return train_epoch(
        model, data_loader, latent_vecs, optimizer, config, epoch=epoch, n_surfaces=surfaces
    )


@contextlib.contextmanager
def records_at(level):
    """
    Collect ``NSM`` records emitted *at* ``level`` -- not at or above it -- without
    pytest's caplog, so absence is assertable and the four ``logger.info`` summaries at the
    end of the epoch do not join the ``DEBUG`` set under test.
    """
    collected = []

    class _Collector(logging.Handler):
        def emit(self, record):
            if record.levelno == level:
                collected.append(record)

    handler = _Collector(level=level)
    logger = logging.getLogger("NSM")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield collected
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


# ---------------------------------------------------------------------------
# 1. batch_split is the number of splits that ran
# ---------------------------------------------------------------------------

#: ``batch_split`` values that ``torch.chunk`` cannot honour on this fixture's 16-row
#: batch (2 objects x 8 samples), so ``range(config["batch_split"])`` walks past the end.
UNHONOURED_SPLITS = [5, 7, 9, 11]

#: Values ``torch.chunk`` returns exactly, which is why the defect has stood.
HONOURED_SPLITS = [1, 2, 3, 4, 6, 8, 16]


class TestBatchSplitIsTheNumberOfSplitsThatRan:
    """
    ``torch.chunk(t, k)`` splits into pieces of ``ceil(len / k)`` and stops when the tensor
    runs out, so it returns *at most* ``k``. The split loop assumes exactly ``k``.
    """

    @pytest.mark.parametrize("splits", UNHONOURED_SPLITS + HONOURED_SPLITS)
    def test_torch_chunk_returns_at_most_the_requested_count(self, splits):
        """The premise, asserted rather than assumed: it is a torch contract, not ours."""
        produced = len(torch.chunk(torch.zeros(N_SUBJECTS // 2 * N_SAMPLES, 3), splits))
        assert produced <= splits
        assert (produced == splits) is (splits in HONOURED_SPLITS)

    @pytest.mark.parametrize("splits", UNHONOURED_SPLITS + HONOURED_SPLITS)
    def test_every_batch_split_completes_an_epoch(self, splits):
        """
        Were four strict xfails. Before the fix ``batch_split`` 5, 7, 9 and 11 raised
        ``IndexError: tuple index out of range`` from ``latent_vecs(indices[split_idx])``
        -- naming a local tuple, not the config key the caller set -- while 3 and 6 on the
        same batch completed.
        """
        assert run_epoch(batch_split=splits)["loss"] > 0

    @pytest.mark.parametrize("splits", UNHONOURED_SPLITS + HONOURED_SPLITS)
    def test_the_epoch_loss_does_not_depend_on_batch_split(self, splits):
        """
        ``batch_split`` exists to bound memory, and ``chunk`` partitions the batch, so every
        split count covers the same samples exactly once. It held before the fix for the
        values that ran, which is what says the fix makes the unhonoured values *correct*
        rather than merely uncrashed: they are now on the same number, not on a new one.
        """
        assert run_epoch(batch_split=splits)["loss"] == pytest.approx(
            run_epoch(batch_split=1)["loss"], rel=1e-6
        )


class TestTheEvalWarmupSharesTheSplitBound:
    """
    ``_schedule_free_eval_warmup`` bounds its loop the same way, in the helper whose
    docstring says it unpacks the batch "exactly the way ``train_epoch`` does". #42 is the
    issue about those two drifting apart; this is them still in step, on a defect.
    """

    class _StubScheduleFree:
        """Only ``eval()`` is reached before the loop the test is about."""

        def eval(self):
            pass

    @pytest.mark.parametrize("splits", [1, 2] + UNHONOURED_SPLITS)
    def test_the_warmup_runs_at_every_batch_split(self, splits):
        """Were four strict xfails, retired by the same one-line change at both sites."""
        model, data_loader, latent_vecs, _, config = epoch_inputs(batch_split=splits)
        config["optimizer"] = "schedule_free_AdamW"
        _schedule_free_eval_warmup(
            model, latent_vecs, data_loader, self._StubScheduleFree(), config, epoch=1
        )


# ---------------------------------------------------------------------------
# 2. The latent-norm statistics
# ---------------------------------------------------------------------------


class TestTheLatentNormStatsAreTheEpochMean:
    """
    ``mean_vec_length`` and ``std_vec_length`` are computed inside the split loop and
    accumulated outside it, so whichever split ran last is the one that counts.

    This is ``KNOWN_ISSUES`` § History 12 (#59) at its second site. That fix changed ``=``
    to ``+=`` on the *batch* loop -- ``test_training_regression.TestLatentNormLogging``
    pins it there, at ``batch_split`` 1, where the split loop runs once and the defect is
    invisible.
    """

    @pytest.mark.parametrize("stat", ["mean_vec_length", "std_vec_length"])
    @pytest.mark.parametrize("splits", [2, 4, 6, 8, 16])
    def test_the_stats_do_not_depend_on_batch_split(self, stat, splits):
        """
        Were six strict xfails. Measured before the fix on a 4-subject fixture:
        ``mean_vec_length`` walked 0.1445 / 0.2026 / 0.3201 for ``batch_split`` 1 / 2 / 4
        and ``std_vec_length`` collapsed to 0.0, while ``loss`` over the same three runs
        was invariant to 1.5e-08. A memory knob must not move a reported number.
        """
        assert run_epoch(batch_split=splits)[stat] == pytest.approx(
            run_epoch(batch_split=1)[stat], rel=1e-6
        )

    @pytest.mark.parametrize("splits", [6, 16])
    def test_the_std_is_never_nan(self, splits):
        """
        Were two strict xfails, and the sharpest form of the defect: ``torch.std`` over a
        single row is NaN, and a split loop that reached a one-row chunk put that NaN
        straight into the epoch's wandb payload. Measured before the fix on this fixture,
        ``batch_split`` 6 and 16 both reported ``nan``; 2, 4 and 8 reported exactly 0.0,
        which is the same defect being merely wrong instead of unusable. Computing the
        statistic over the whole batch -- what the metric has always meant -- removes both.
        """
        assert not math.isnan(run_epoch(batch_split=splits)["std_vec_length"])

    def test_the_mean_is_the_mean_over_every_subject(self):
        """
        Was a strict xfail. The latent learning rate is 0, so the embedding cannot move
        during the epoch and the expected value is computable from it directly --
        § History 12's test does the same at the batch level. One batch of all four
        subjects, split four ways, gives one subject per split: pre-fix the reported value
        was the *last* subject's norm.
        """
        model, data_loader, latent_vecs, optimizer, config = epoch_inputs(
            objects_per_batch=N_SUBJECTS, batch_split=N_SUBJECTS, latent_lr=0.0
        )
        expected = torch.norm(latent_vecs.weight.data, dim=1).mean().item()
        log = train_epoch(model, data_loader, latent_vecs, optimizer, config, epoch=1, n_surfaces=2)
        assert log["mean_vec_length"] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. multi_object_overlap
# ---------------------------------------------------------------------------


class TestMultiObjectOverlapIsRefusedWhereItIsNamed:
    """
    ``config.get("multi_object_overlap", False) is True`` raises ``Exception("Not
    implemented yet")`` from inside the innermost loop -- after ``optimizer.zero_grad()``,
    after the forward pass, and after every earlier batch of the epoch has already
    ``step()``ped.
    """

    def test_it_raises_not_implemented_naming_the_parameter(self):
        """Was a strict xfail: a bare ``Exception("Not implemented yet")``."""
        with pytest.raises(NotImplementedError, match="multi_object_overlap"):
            run_epoch(multi_object_overlap=True)

    def test_it_raises_before_the_first_batch_is_fetched(self):
        """
        Was a strict xfail. A data loader that refuses to be iterated: if the refusal is
        where the parameter is named, nothing asks it for a batch. Before the fix it was
        asked, and the epoch ran a full forward and backward before failing.
        """

        class _Unfetchable:
            def __iter__(self):
                raise AssertionError("the batch loop was entered")

            def __len__(self):
                return 1

        model, _, latent_vecs, optimizer, config = epoch_inputs(multi_object_overlap=True)
        with pytest.raises(NotImplementedError, match="multi_object_overlap"):
            train_epoch(
                model, _Unfetchable(), latent_vecs, optimizer, config, epoch=1, n_surfaces=2
            )


# ---------------------------------------------------------------------------
# 5. surface_weighting
# ---------------------------------------------------------------------------


class TestSurfaceWeightingIsValidatedOnce:
    """
    ``assert len(config["surface_weighting"]) == n_surfaces`` sits in the innermost loop:
    it disappears under ``python -O``, it carries no message, and it and the weight
    normalisation it guards are epoch-constant yet re-evaluated once per split per batch.
    """

    @pytest.mark.parametrize("weighting", [[1, 1, 1], [1], [3, 1, 99]])
    def test_a_mismatched_weighting_names_both_lengths(self, weighting):
        """Was a strict xfail: a bare ``AssertionError`` with no text, and only one case."""
        with pytest.raises(ValueError, match="surface_weighting"):
            run_epoch(surface_weighting=weighting)

    def test_a_mismatched_weighting_is_still_refused_under_O(self, tmp_path):
        """
        Was a strict xfail. ``python -O`` strips ``assert``, and what was behind this one
        was not a crash.

        ``weights_sum`` was taken over the whole declared list while ``weights_total`` was
        ``n_surfaces``, so a list one entry too long rescaled every weight it *did* use.
        Measured under ``-O`` on this fixture, 2 surfaces: unweighted and ``[1, 1]`` both
        give 0.2752, ``[1, 1, 1]`` gave **0.1835** and ``[3, 1, 99]`` gave **0.0109** --
        the third entry is never indexed and still moved the loss by two orders of
        magnitude. A list one entry too *short* raised ``IndexError: list index out of
        range``, which named nothing either.

        Run in a subprocess because ``-O`` is an interpreter flag, not a runtime setting:
        the suite cannot be under it and test it at the same time.
        """
        script = tmp_path / "under_O.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {str(pathlib.Path(__file__).parent)!r})
                sys.path.insert(0, {str(REPO_ROOT)!r})
                from test_train_epoch_internals import run_epoch
                assert __debug__ is False, "-O did not take"
                try:
                    run_epoch(surface_weighting=[1, 1, 1])
                except Exception as exc:
                    print(type(exc).__name__)
                else:
                    print("ACCEPTED")
                """
            ),
            encoding="utf-8",
        )
        result = subprocess.run([sys.executable, "-O", str(script)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() != "ACCEPTED", result.stdout

    @pytest.mark.parametrize("weighting", [None, [1, 1], [3, 1]])
    def test_hoisting_the_weights_may_not_move_the_loss(self, weighting):
        """
        The weights are a pure function of the config, so computing them once per epoch
        instead of once per split is inert. Asserted against ``batch_split`` rather than
        against a stored number, so it holds at both ends of the change.
        """
        overrides = {} if weighting is None else {"surface_weighting": weighting}
        assert run_epoch(**overrides)["loss"] == pytest.approx(
            run_epoch(batch_split=4, **overrides)["loss"], rel=1e-6
        )


# ---------------------------------------------------------------------------
# 6. samples_per_object_per_batch
# ---------------------------------------------------------------------------


class TestSamplesPerObjectPerBatchMatchesItsBatch:
    """
    The trainer rebuilds the per-object sample count from the config
    (``indices.repeat(1, config["samples_per_object_per_batch"])``) when the batch it was
    handed already carries it as ``sdf_data["xyz"].shape[1]``. The two are declared in
    different objects -- one a ``MultiSurfaceSDFSamples`` constructor argument
    (``subsample``), one a config key -- so nothing but a check can hold them together.
    """

    @pytest.mark.parametrize("declared", [4, 7, 16])
    def test_a_disagreement_names_the_config_key(self, declared):
        """
        Were two strict xfails. It has never been silent, which is why this is a message
        defect and not a numerical one: before the fix a ``RuntimeError`` from
        ``torch.cat`` naming tensor sizes, 35 lines below the config read.
        """
        with pytest.raises(ValueError, match="samples_per_object_per_batch"):
            run_epoch(samples_per_object_per_batch=declared)

    @pytest.mark.parametrize("declared", [4, 7, 16])
    def test_the_warmup_refuses_it_too(self, declared):
        """
        The second site, for the same reason as the split bound: the warm-up rebuilds the
        index tensor the same way, so a config the trainer refuses must not be one the
        warm-up accepts.
        """
        model, data_loader, latent_vecs, _, config = epoch_inputs(
            samples_per_object_per_batch=declared
        )
        config["optimizer"] = "schedule_free_AdamW"
        with pytest.raises(ValueError, match="samples_per_object_per_batch"):
            _schedule_free_eval_warmup(
                model,
                latent_vecs,
                data_loader,
                TestTheEvalWarmupSharesTheSplitBound._StubScheduleFree(),
                config,
                epoch=1,
            )


# ---------------------------------------------------------------------------
# 7. The verbose-gated records
# ---------------------------------------------------------------------------

#: The message templates ``train_epoch`` emits at ``DEBUG`` for one 2-surface epoch.
#: Written out rather than counted so that commit 9's deletions are visible in this file's
#: diff, and so a record added later without a reason turns this red.
DEBUG_TEMPLATES = {
    "sdf index size: %s",
    "xyz data size: %s",
    "sdf gt size: %s",
    "len sdf_gt %s",
    "len sdf_gt chunks: %s",
    "len xyz chunks %s",
    "Split idx:  %s",
    "model dtype %s",
    "inputs dtype %s",
    "len pred_sdf %s",
    "split idx %s",
    "surf idx %s",
    "%s",
    "pred_sdf shape %s",
    "unsqueezed pred_sdf shape %s",
    "sdf_gt shape %s",
    "l1 losses: %s",
    "l1 loss: %s",
}


class TestTheDebugRecordsAreNotGatedOnAConfigKey:
    """
    All 20 records are already ``logger.debug``, so the gate only ever subtracted. The
    shipped ``NSM/configs/default_config.json`` sets ``verbose: true``, which left the
    gate permanently open and the *level* doing the filtering; a config with
    ``verbose: false`` hid all 20 from a host that configured ``DEBUG`` and asked for
    them. ``config["verbose"]`` is still read -- ``_run_validation`` forwards it -- so
    ungating these did not turn the key into an accepted-and-ignored one.
    """

    def test_the_record_set_under_verbose_true_is_what_it_is(self):
        """
        The before-and-after pin for commit 8: ungating may not change what a host with
        ``verbose: true`` sees. Commit 9 then deletes four of these templates, and that
        deletion shows up here rather than in a count.
        """
        with records_at(logging.DEBUG) as collected:
            run_epoch(verbose=True)
        assert {record.msg for record in collected} == DEBUG_TEMPLATES

    def test_a_host_at_debug_sees_them_without_the_config_key(self):
        """Was a strict xfail: the set was empty, whatever the host had configured."""
        with records_at(logging.DEBUG) as collected:
            run_epoch(verbose=False)
        assert {record.msg for record in collected} == DEBUG_TEMPLATES


# ---------------------------------------------------------------------------
# The extraction pin
# ---------------------------------------------------------------------------

PRIORS = ["identity", "spherical", "kld_diagonal"]


class TestTheLogDictIsSelfConsistent:
    """
    What commits 10 and 11 may not change. The end-to-end arithmetic of the production
    path is pinned by ``testing/NSM/regression`` against committed baselines; this covers
    the configurations those baselines do not reach -- surface counts, priors, the
    variational branch, and ``batch_split`` above 1 -- with identities rather than stored
    numbers, so nothing here needs regenerating when commits 3-7 legitimately move a value.
    """

    @pytest.mark.parametrize("surfaces", [1, 2, 3])
    @pytest.mark.parametrize("prior", PRIORS)
    def test_the_loss_is_its_reported_parts(self, surfaces, prior):
        """``loss`` is the L1 term plus the regularization term, and nothing else."""
        log = run_epoch(surfaces=surfaces, code_regularization_type_prior=prior)
        assert log["loss"] == pytest.approx(
            log["l1_loss"] + log["latent_code_regularization_loss"], rel=1e-6
        )

    @pytest.mark.parametrize("surfaces", [1, 2, 3])
    def test_the_l1_term_is_the_mean_of_its_per_surface_parts(self, surfaces):
        """
        True under uniform weighting, which is every configuration except an explicit
        ``surface_weighting``: the per-surface records are taken before weighting and the
        total after.
        """
        log = run_epoch(surfaces=surfaces)
        parts = [log[f"l1_loss_{index}"] for index in range(surfaces)]
        assert log["l1_loss"] == pytest.approx(sum(parts) / surfaces, rel=1e-6)

    def test_an_explicit_weighting_is_the_documented_exception(self):
        """
        The one case where the identity above does not hold, asserted so that it is a
        stated exception rather than a discovered one. Measured on this fixture:
        ``[3, 1]`` gives an ``l1_loss`` below the unweighted mean of its parts.
        """
        log = run_epoch(surface_weighting=[3, 1])
        parts = [log["l1_loss_0"], log["l1_loss_1"]]
        assert log["l1_loss"] != pytest.approx(sum(parts) / 2, rel=1e-6)

    @pytest.mark.parametrize("variational", [False, True])
    @pytest.mark.parametrize("prior", PRIORS)
    def test_the_documented_keys_are_all_there(self, variational, prior):
        """
        The flat dict the docstring describes, for a run with no load timing and no
        ``log_latent``: the six scalars plus one per surface.
        """
        log = run_epoch(variational=variational, code_regularization_type_prior=prior)
        assert set(log) == {
            "loss",
            "epoch_time_s",
            "l1_loss",
            "latent_code_regularization_loss",
            "mean_vec_length",
            "std_vec_length",
            "l1_loss_0",
            "l1_loss_1",
        }

    @pytest.mark.parametrize("surfaces", [1, 2])
    @pytest.mark.parametrize("splits", [2, 4])
    def test_the_loss_survives_the_split_loop_unchanged(self, surfaces, splits):
        """The seam commit 10 cuts: chunking is a memory strategy, not an arithmetic one."""
        assert run_epoch(surfaces=surfaces, batch_split=splits)["loss"] == pytest.approx(
            run_epoch(surfaces=surfaces)["loss"], rel=1e-6
        )
