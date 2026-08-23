"""
End-to-end training regression: 8 epochs, CPU, fixed seed, asserted against baselines.

What this pins, in decreasing order of how much it matters:

1. **The per-param-group learning rate at every epoch.** This is the assertion that would
   have caught the schedule swap in ``docs/KNOWN_ISSUES.md`` section 1. The two
   schedules used here differ in Interval and Factor as well as Initial, so transposing
   them inverts the run rather than nudging it -- and ``TestDeliberateBreak`` transposes
   them and watches the baselines fail.
2. The loss trajectory across all 8 epochs.
3. The final latent norms.

``testing/NSM/test_lr_schedules.py`` covers the schedule-to-group mapping in isolation.
This module is the other half of that: the mapping surviving a real training loop.
"""

import copy
import json
import os
import types

import pytest
import torch
from _harness import (
    LATENT_NORM_ATOL,
    LOSS_RTOL,
    LR_SCHEDULE,
    MIN_HEADROOM,
    N_EPOCHS,
    REGENERATE_DECODER_ENV,
    REGENERATE_ENV,
    build_model,
    headroom,
    platform_matches,
    provenance,
    quiet,
    regenerating,
    regenerating_decoder,
    run_training,
    training_config,
)

from NSM.train.train_deep_sdf import train_epoch
from NSM.utils import get_latent_vecs, get_learning_rate_schedules, get_optimizer


def test_baselines_are_not_being_regenerated():
    """
    A regeneration run must not read as a passing run. If this is the only red test, the
    harness rewrote its baselines instead of checking them.
    """
    assert not regenerating(), (
        f"{REGENERATE_ENV} is set: baselines were REWRITTEN, not checked. "
        f"Unset it and re-run to verify."
    )


def test_the_reconstruction_decoder_is_not_being_regenerated():
    """
    The same rule for the other committed artifact, and a separate test so its own message
    is the one that prints. A run that rewrote ``assets/reconstruction_decoder.pt`` checked
    nothing about reconstruction: every baseline it compared against was fitted to whatever
    weights it had just written.
    """
    assert not regenerating_decoder(), (
        f"{REGENERATE_DECODER_ENV} is set: the reconstruction decoder was RETRAINED and "
        f"rewritten, so the reconstruction baselines were compared against a decoder this "
        f"run produced. Unset it and re-run to verify -- and if the decoder really changed, "
        f"the reconstruction baselines have to be regenerated too."
    )


class TestBaselinePlatformPin:
    """
    The numeric baselines are pinned to Linux-x86_64 and skip elsewhere
    (``_harness.platform_matches``). A skip mechanism nobody has exercised is how a harness
    turns into a blanket skip without anyone noticing, so it is tested directly.
    """

    def test_the_committed_baselines_are_linux_x86_64(self, training_baseline):
        if regenerating():
            pytest.skip("baselines are being rewritten")
        assert training_baseline.generated_on["platform"] == "Linux-x86_64"

    def test_the_numeric_baselines_are_actually_running_here(self, training_baseline):
        """On the pinned platform, nothing may be skipped."""
        if regenerating():
            pytest.skip("baselines are being rewritten")
        if platform_matches(training_baseline.generated_on):
            assert training_baseline.values, "baseline file is empty"
        else:
            pytest.skip(f"not the pinned platform: {training_baseline.generated_on}")

    def test_a_foreign_platform_skips_numeric_checks_but_not_portable_ones(self):
        foreign = {"platform": "Neverland-vax", "torch": "0.0", "numpy": "0.0", "python": "0.0"}
        assert not platform_matches(foreign)
        assert platform_matches(provenance())
        assert platform_matches({})

    def test_a_torch_change_is_not_skipped(self):
        """
        The asymmetry that matters: a dependency bump that moves numbers must go red, not
        quietly skip. Only the OS/architecture gates.
        """
        same_platform_new_torch = dict(provenance(), torch="99.0.0")
        assert platform_matches(same_platform_new_torch)


class TestLearningRateTrajectory:
    def test_every_epoch_has_a_recorded_learning_rate(self, training_run):
        assert [r["epoch"] for r in training_run["records"]] == list(range(1, N_EPOCHS + 1))

    def test_param_groups_are_named_and_targeted(self, training_run):
        first = training_run["records"][0]
        assert sorted(first["lrs"]) == ["latent", "model_0"]
        assert first["targets"] == {"latent": "latent", "model_0": "model"}

    def test_per_epoch_learning_rates_match_baseline(self, training_run, training_baseline):
        """
        Exact equality, no tolerance: these are ``Initial * Factor ** (epoch // Interval)``
        in Python floats and are identical on every platform. Any movement is a real change
        to the LR path.
        """
        observed = {
            name: [r["lrs"][name] for r in training_run["records"]]
            for name in ("model_0", "latent")
        }
        training_baseline.check(
            "learning_rates_per_epoch", observed, rtol=0.0, atol=0.0, portable=True
        )

    def test_learning_rates_follow_their_declared_schedule(self, training_run):
        """
        Independent of the baseline: recompute both Step schedules from the config and
        confirm each group got its own rather than the other one's.
        """
        by_target = {entry["Target"]: entry for entry in LR_SCHEDULE}
        for name, target in (("model_0", "model"), ("latent", "latent")):
            spec = by_target[target]
            for record in training_run["records"]:
                expected = spec["Initial"] * spec["Factor"] ** (record["epoch"] // spec["Interval"])
                assert record["lrs"][name] == pytest.approx(expected), (
                    f"epoch {record['epoch']}: group {name!r} ran at {record['lrs'][name]}, "
                    f"but its {target!r} schedule says {expected}"
                )

    def test_the_two_schedules_are_distinguishable(self, training_run):
        """
        The guard under every other assertion here: if the two schedules produced the same
        rates, swapping them would be undetectable and this module would pin nothing.
        """
        model_lrs = [r["lrs"]["model_0"] for r in training_run["records"]]
        latent_lrs = [r["lrs"]["latent"] for r in training_run["records"]]
        assert model_lrs != latent_lrs
        assert len(set(model_lrs)) > 1 and len(set(latent_lrs)) > 1, "neither schedule decayed"


class TestLossTrajectory:
    def test_loss_trajectory_matches_baseline(self, training_run, training_baseline):
        training_baseline.check(
            "loss_trajectory", [r["loss"] for r in training_run["records"]], rtol=LOSS_RTOL
        )

    def test_loss_components_match_baseline(self, training_run, training_baseline):
        training_baseline.check(
            "l1_loss_trajectory",
            [r["l1_loss"] for r in training_run["records"]],
            rtol=LOSS_RTOL,
        )
        training_baseline.check(
            "code_reg_loss_trajectory",
            [r["code_reg_loss"] for r in training_run["records"]],
            rtol=LOSS_RTOL,
        )

    def test_training_actually_reduces_the_loss(self, training_run):
        """A run frozen at its initial loss would match a frozen baseline forever."""
        losses = [r["loss"] for r in training_run["records"]]
        assert losses[-1] < losses[0], f"loss did not fall over {N_EPOCHS} epochs: {losses}"


class TestLatentCodes:
    def test_final_latent_norms_match_baseline(self, training_run, training_baseline):
        training_baseline.check(
            "final_latent_norms", training_run["records"][-1]["latent_norms"], atol=LATENT_NORM_ATOL
        )

    def test_latent_norm_trajectory_matches_baseline(self, training_run, training_baseline):
        training_baseline.check(
            "latent_norm_trajectory",
            [r["latent_norms"] for r in training_run["records"]],
            atol=LATENT_NORM_ATOL,
        )

    def test_one_latent_per_training_object(self, training_run, training_dataset):
        assert len(training_run["records"][-1]["latent_norms"]) == len(training_dataset)


class TestCheckpoints:
    def test_checkpoints_written_at_the_expected_epochs(self, training_run):
        directory = training_run["config"]["experiment_directory"]
        models = sorted(int(f.split(".")[0]) for f in os.listdir(os.path.join(directory, "model")))
        latents = sorted(
            int(f.split(".")[0]) for f in os.listdir(os.path.join(directory, "latent_codes"))
        )
        # save_frequency=4 with checkpoint_epochs=8 -> epochs 4 and 8.
        assert models == [4, 8]
        assert latents == models

    def test_checkpoint_carries_param_group_targets(self, training_run):
        path = os.path.join(training_run["config"]["experiment_directory"], "model", "8.pth")
        checkpoint = torch.load(path, weights_only=False)
        groups = checkpoint["optimizer"]["param_groups"]
        assert [g["name"] for g in groups] == ["latent", "model_0"]
        assert [g["target"] for g in groups] == ["latent", "model"]

    def test_saved_model_params_config_records_the_targets(self, training_run):
        path = os.path.join(
            training_run["config"]["experiment_directory"], "model_params_config.json"
        )
        with open(path) as f:
            saved = json.load(f)
        assert [e["Target"] for e in saved["LearningRateSchedule"]] == ["model", "latent"]
        assert saved["mesh_names"] == ["bone", "cart"]


class TestDeliberateBreak:
    """
    A regression harness nobody has seen fail is not evidence of anything.

    These re-run training with the two learning-rate ``Target`` labels transposed -- the
    exact shape of the bug that started this work -- and assert that the baselines above
    reject the result. They are the reason those baselines can be trusted.

    The two tolerance-based rejections also assert ``MIN_HEADROOM``, so how far outside the
    tolerance the break lands is measured on every run instead of being written down once.
    The learning-rate baseline is compared exactly (``rtol=atol=0``) and has no headroom to
    measure.
    """

    @pytest.fixture(scope="class")
    def swapped_schedule_run(self, training_dataset, tmp_path_factory):
        config = training_config(tmp_path_factory.mktemp("swapped_lr"))
        swapped = copy.deepcopy(config["LearningRateSchedule"])
        swapped[0]["Target"], swapped[1]["Target"] = swapped[1]["Target"], swapped[0]["Target"]
        config["LearningRateSchedule"] = swapped
        records, _ = run_training(config, build_model(config), training_dataset)
        return records

    def test_swapping_lr_targets_changes_the_learning_rates(
        self, training_run, swapped_schedule_run
    ):
        good = [r["lrs"]["model_0"] for r in training_run["records"]]
        broken = [r["lrs"]["model_0"] for r in swapped_schedule_run]
        assert good != broken, "swapping the two LR Targets left the model rate unchanged"

    def test_swapping_lr_targets_fails_the_learning_rate_baseline(
        self, swapped_schedule_run, training_baseline
    ):
        if regenerating():
            pytest.skip("baselines are being rewritten")
        observed = {
            name: [r["lrs"][name] for r in swapped_schedule_run] for name in ("model_0", "latent")
        }
        with pytest.raises(AssertionError, match="differs from baseline"):
            training_baseline.check(
                "learning_rates_per_epoch", observed, rtol=0.0, atol=0.0, portable=True
            )

    def test_swapping_lr_targets_fails_the_loss_baseline(
        self, swapped_schedule_run, training_baseline
    ):
        """
        The stronger claim: the swap is not merely visible in the recorded rates, it moves
        the numbers the model produces -- which is what a run on disk would show.
        """
        if regenerating():
            pytest.skip("baselines are being rewritten")
        losses = [r["loss"] for r in swapped_schedule_run]
        with pytest.raises(AssertionError, match="differs from baseline"):
            training_baseline.check("loss_trajectory", losses, rtol=LOSS_RTOL)

        measured = headroom(training_baseline, "loss_trajectory", losses, rtol=LOSS_RTOL)
        assert measured >= MIN_HEADROOM, (
            f"the LR swap moves the loss trajectory only {measured:.1f}x LOSS_RTOL "
            f"({LOSS_RTOL}), under the MIN_HEADROOM of {MIN_HEADROOM}x. Widen the break, "
            f"never the tolerance."
        )

    def test_swapping_lr_targets_fails_the_latent_baseline(
        self, swapped_schedule_run, training_baseline
    ):
        if regenerating():
            pytest.skip("baselines are being rewritten")
        norms = swapped_schedule_run[-1]["latent_norms"]
        with pytest.raises(AssertionError, match="differs from baseline"):
            training_baseline.check("final_latent_norms", norms, atol=LATENT_NORM_ATOL)

        measured = headroom(training_baseline, "final_latent_norms", norms, atol=LATENT_NORM_ATOL)
        assert measured >= MIN_HEADROOM, (
            f"the LR swap moves the final latent norms only {measured:.1f}x "
            f"LATENT_NORM_ATOL ({LATENT_NORM_ATOL}), under the MIN_HEADROOM of "
            f"{MIN_HEADROOM}x. Widen the break, never the tolerance."
        )


class TestClampedPredictionGradients:
    """
    Characterization of a live gradient path, not a complaint.

    With ``enforce_minmax``, ``train_epoch`` clamps the PREDICTION as well as the target
    (``train_deep_sdf.py:401``), and ``torch.clamp`` passes no gradient outside its
    bounds. Every sample the decoder predicts outside ``+/-clamp_dist`` therefore
    contributes exactly zero gradient, however wrong it is.

    That makes ``clamp_dist`` a training-dynamics knob and not just a target transform,
    which is not what its name or the docs suggest. The harness uses 1.0, the value both
    shipped ShapeMedKnee configs use; the shipped ``default_config.json`` uses 0.1.
    """

    def test_clamped_predictions_receive_no_gradient(self):
        prediction = torch.tensor([2.0], requires_grad=True)
        torch.nn.L1Loss()(torch.clamp(prediction, -0.1, 0.1), torch.tensor([0.05])).backward()
        assert prediction.grad.item() == 0.0

    def test_unclamped_predictions_still_receive_gradient(self):
        """The other half, so the test above cannot pass because clamp broke entirely."""
        prediction = torch.tensor([0.05], requires_grad=True)
        torch.nn.L1Loss()(torch.clamp(prediction, -0.1, 0.1), torch.tensor([0.0])).backward()
        assert prediction.grad.item() != 0.0

    def test_fraction_of_dead_samples_at_init_matches_baseline(
        self, training_run, training_baseline
    ):
        """
        How much of the training signal ``clamp_dist=0.1`` discards before the first step,
        measured on a freshly built decoder. Baselined because it is a property of the
        initialization, and a change to weight init or ``final_activation`` moves it.
        """
        model = build_model(training_run["config"])
        model.eval()
        torch.manual_seed(0)
        points = torch.rand(2048, 3) * 2 - 1
        inputs = torch.cat([torch.zeros(2048, training_run["config"]["latent_size"]), points], 1)
        with torch.no_grad():
            predictions = model(inputs)

        fraction_dead = (predictions.abs() > 0.1).float().mean().item()
        assert fraction_dead > 0.25, (
            f"only {fraction_dead:.1%} of a fresh decoder's predictions fall outside "
            f"+/-0.1 -- the clamp no longer discards a meaningful share of the signal"
        )
        training_baseline.check("fraction_dead_at_init_clamp_0_1", fraction_dead, atol=1e-6)


class TestTrainerContract:
    def test_train_deep_sdf_returns_its_history(self, training_run):
        """
        The public entry point returns what wandb would have seen (#28) — one entry per
        epoch, carrying the log payload plus ``epoch``/``lrs``/``targets``/
        ``latent_norms``. This is what lets ``run_training`` read the run instead of
        wrapping ``train_epoch``.
        """
        history = training_run["returned"]
        assert [entry["epoch"] for entry in history] == list(range(1, N_EPOCHS + 1))
        for entry in history:
            assert {"loss", "l1_loss", "lrs", "targets", "latent_norms"} <= entry.keys()

    def test_mesh_names_are_carried_but_never_reach_the_model(self, training_run):
        """
        ``mesh_names`` is validated against ``objects_per_decoder`` and written to
        ``model_params_config.json``, and nothing in ``NSM/models/`` reads it -- which is
        why surface identity in ``reconstruct_mesh``'s result is positional. See
        ``test_reconstruction_regression.TestSurfaceOrderContract``.
        """
        assert training_run["config"]["mesh_names"] == ["bone", "cart"]
        assert not hasattr(training_run["model"], "mesh_names")


# ---------------------------------------------------------------------------
# §8.0.D characterization: resume, schedule_free, latent-norm logging
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def resume_source_run(training_dataset, tmp_path_factory):
    """A 2-epoch run checkpointing at every epoch — the source the resume tests load."""
    config = training_config(tmp_path_factory.mktemp("resume_source"))
    config.update({"n_epochs": 2, "checkpoint_epochs": 1})
    model = build_model(config)
    records, _ = run_training(config, model, training_dataset)
    return {"config": config, "records": records}


def _assert_same_state(state, checkpoint_state):
    assert state.keys() == checkpoint_state.keys()
    for key in state:
        assert torch.equal(state[key], checkpoint_state[key]), f"weights differ at {key!r}"


class TestResumeContract:
    """
    ``resume_epoch`` names the last *completed* epoch: its checkpoint is loaded and the
    loop continues at ``resume_epoch + 1``. The load guard and the loop boundary must
    share that convention — they did not, and ``resume_epoch=1`` used to skip epoch 1
    while loading nothing (#49, ``docs/KNOWN_ISSUES.md`` § History 11).
    """

    def test_resume_epoch_0_runs_every_epoch(self, resume_source_run):
        assert [r["epoch"] for r in resume_source_run["records"]] == [1, 2]

    def test_resume_from_a_later_checkpoint_loads_it(self, resume_source_run, training_dataset):
        """
        ``resume_epoch=2`` with ``n_epochs=2`` trains nothing — it only loads — so the
        model must leave ``train_deep_sdf`` carrying exactly the epoch-2 checkpoint's
        weights rather than the fresh init it walked in with (a different seed, so a
        skipped load cannot pass by accident).
        """
        source_dir = resume_source_run["config"]["experiment_directory"]
        config = training_config(source_dir)
        config.update({"n_epochs": 2, "checkpoint_epochs": 1, "resume_epoch": 2})
        model = build_model(config, seed=7)
        records, _ = run_training(config, model, training_dataset)

        assert records == []
        checkpoint = torch.load(os.path.join(source_dir, "model", "2.pth"), weights_only=False)
        _assert_same_state(model.state_dict(), checkpoint["model"])

    def test_resume_epoch_1_loads_the_epoch_1_checkpoint(self, resume_source_run, training_dataset):
        """Same contract as the test above, one epoch earlier — the boundary #49 names."""
        source_dir = resume_source_run["config"]["experiment_directory"]
        config = training_config(source_dir)
        config.update({"n_epochs": 1, "checkpoint_epochs": 1, "resume_epoch": 1})
        model = build_model(config, seed=7)
        records, _ = run_training(config, model, training_dataset)

        assert records == []
        checkpoint = torch.load(os.path.join(source_dir, "model", "1.pth"), weights_only=False)
        _assert_same_state(model.state_dict(), checkpoint["model"])


class TestScheduleFreeRuns:
    """
    The eval warm-up must unpack the batch the way ``train_epoch`` does. It used to hand
    the decoder the raw dataloader item (``model(batch)`` where ``batch`` is
    ``(sdf_data, indices)``), so every schedule_free run died with a ``TypeError`` at its
    first checkpoint or validation epoch — which every run reaches (#42).

    ``schedulefree`` is not installed in ``nsm-dev``, and the defect was in the trainer's
    warm-up, not in schedulefree — so the optimizer is stubbed: AdamW plus the
    ``train()``/``eval()`` mode switches, the entire interface the trainer uses. What the
    stub does not exercise is schedulefree's numerics, which ``docs/KNOWN_ISSUES.md`` §1
    already records as needing retuning.
    """

    def test_a_schedule_free_run_survives_its_first_checkpoint_epoch(
        self, training_dataset, tmp_path_factory, monkeypatch
    ):
        import NSM.utils

        class _StubAdamWScheduleFree(torch.optim.AdamW):
            def train(self):
                pass

            def eval(self):
                pass

        monkeypatch.setattr(
            NSM.utils,
            "schedulefree",
            types.SimpleNamespace(AdamWScheduleFree=_StubAdamWScheduleFree),
        )

        config = training_config(tmp_path_factory.mktemp("schedule_free_run"))
        config.update({"optimizer": "schedule_free_AdamW", "n_epochs": 2, "checkpoint_epochs": 1})
        model = build_model(config)
        records, _ = run_training(config, model, training_dataset)

        assert [r["epoch"] for r in records] == [1, 2]


class TestLatentNormLogging:
    """
    ``train_epoch``'s logged latent-norm stats must be accumulated over the epoch, like
    every accumulator around them. They used to be assigned (``=`` for ``+=``), which
    made the logged value the *last batch's* stat over the batch count — wrong by
    ~×n_batches on every wandb run since the metric existed (#59,
    ``docs/KNOWN_ISSUES.md`` § History 12). Weights and gradients were never affected.
    """

    def test_logged_mean_vec_length_is_the_epoch_mean_not_the_last_batch(
        self, training_dataset, tmp_path_factory
    ):
        """
        The latent LR is set to 0 so the embedding cannot move during the epoch: the
        expected value is then exactly the mean over batches of each batch's mean latent
        norm, computable from the embedding directly. ``shuffle=False`` makes the batch
        composition (``[s0, s1], [s2]``) part of the arithmetic rather than of the seed.
        Pre-fix the logged value was ``norm(s2) / 2`` — the singleton last batch over
        the batch count — the issue's ×n_batches observation in miniature.
        """
        config = training_config(tmp_path_factory.mktemp("latent_norm_log"))
        for entry in config["LearningRateSchedule"]:
            if entry["Target"] == "latent":
                entry["Initial"] = 0.0
        config["log_latent"] = None
        config["lr_schedules"] = get_learning_rate_schedules(config)

        data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=2, shuffle=False)
        model = build_model(config)
        torch.manual_seed(0)
        latent_vecs = get_latent_vecs(len(training_dataset), config)
        optimizer = get_optimizer(
            model,
            latent_vecs,
            lr_schedules=config["lr_schedules"],
            optimizer=config["optimizer"],
            weight_decay=config["weight_decay"],
        )

        norms = torch.norm(latent_vecs.weight.data, dim=1)
        expected = ((norms[0] + norms[1]) / 2 + norms[2]) / 2

        with quiet():
            log = train_epoch(model, data_loader, latent_vecs, optimizer, config, epoch=1)

        assert log["mean_vec_length"] == pytest.approx(expected.item(), rel=1e-6)
