"""
End-to-end training regression: 8 epochs, CPU, fixed seed, asserted against baselines.

What this pins, in decreasing order of how much it matters:

1. **The per-param-group learning rate at every epoch.** This is the assertion that would
   have caught the schedule swap in ``docs/KNOWN_ISSUES_HISTORY.md`` section 1. The two
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

import pytest
import torch
from _harness import (
    LR_SCHEDULE,
    N_EPOCHS,
    REGENERATE_ENV,
    build_model,
    platform_matches,
    provenance,
    regenerating,
    run_training,
    training_config,
)

#: Sized from the deliberate break, not from taste. Transposing the two LR targets moves
#: the loss trajectory by 155% at its widest (7.7% at its narrowest, epoch 1) and the final
#: latent norms by 4.3e-2, so these leave two to three orders of magnitude of headroom for a
#: different BLAS while still catching the break outright.
LOSS_RTOL = 1e-3
LATENT_ATOL = 1e-4


def test_baselines_are_not_being_regenerated():
    """
    A regeneration run must not read as a passing run. If this is the only red test, the
    harness rewrote its baselines instead of checking them.
    """
    assert not regenerating(), (
        f"{REGENERATE_ENV} is set: baselines were REWRITTEN, not checked. "
        f"Unset it and re-run to verify."
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
            "final_latent_norms", training_run["records"][-1]["latent_norms"], atol=LATENT_ATOL
        )

    def test_latent_norm_trajectory_matches_baseline(self, training_run, training_baseline):
        training_baseline.check(
            "latent_norm_trajectory",
            [r["latent_norms"] for r in training_run["records"]],
            atol=LATENT_ATOL,
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
        with pytest.raises(AssertionError, match="differs from baseline"):
            training_baseline.check(
                "loss_trajectory", [r["loss"] for r in swapped_schedule_run], rtol=LOSS_RTOL
            )

    def test_swapping_lr_targets_fails_the_latent_baseline(
        self, swapped_schedule_run, training_baseline
    ):
        if regenerating():
            pytest.skip("baselines are being rewritten")
        with pytest.raises(AssertionError, match="differs from baseline"):
            training_baseline.check(
                "final_latent_norms", swapped_schedule_run[-1]["latent_norms"], atol=LATENT_ATOL
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
    @pytest.mark.xfail(strict=True, reason="worklist #11: train_deep_sdf returns nothing")
    def test_train_deep_sdf_returns_its_history(self, training_run):
        """
        No loss history is observable from the public entry point, so ``run_training`` has
        to wrap ``train_epoch`` to see anything. When this goes green that wrapper can go.
        """
        assert training_run["returned"] is not None

    def test_mesh_names_are_carried_but_never_reach_the_model(self, training_run):
        """
        ``mesh_names`` is validated against ``objects_per_decoder`` and written to
        ``model_params_config.json``, and nothing in ``NSM/models/`` reads it -- which is
        why surface identity in ``reconstruct_mesh``'s result is positional. See
        ``test_reconstruction_regression.TestSurfaceOrderContract``.
        """
        assert training_run["config"]["mesh_names"] == ["bone", "cart"]
        assert not hasattr(training_run["model"], "mesh_names")
