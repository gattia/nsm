"""
End-to-end training regression: 8 epochs, CPU, fixed seed, asserted against baselines.

What this pins, most important first:

1. **The learning rate of each param group at every epoch.** This would have caught the
   schedule swap in ``docs/KNOWN_ISSUES.md`` §1. The two schedules differ in Interval and
   Factor as well as Initial, so transposing them inverts the run, and
   ``TestDeliberateBreak`` transposes them and watches the baselines fail.
2. The loss trajectory and its components.
3. The latent norms.
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
    regenerating,
    regenerating_decoder,
    run_training,
    training_config,
)


def test_baselines_are_not_being_regenerated():
    """A regeneration run must not read as a passing run."""
    assert not regenerating(), f"{REGENERATE_ENV} is set: baselines were REWRITTEN, not checked"


def test_the_reconstruction_decoder_is_not_being_regenerated():
    """
    A run that rewrote ``assets/reconstruction_decoder.pt`` compared every reconstruction
    baseline against weights it had just written. If the decoder really changed, the
    reconstruction baselines must be regenerated too.
    """
    assert not regenerating_decoder(), f"{REGENERATE_DECODER_ENV} is set: the decoder was RETRAINED"


class TestBaselinePlatformPin:
    """
    Numeric baselines are pinned to Linux-x86_64 and skip elsewhere. A torch or numpy change
    on the same platform goes red: that is what the harness exists to report.
    """

    def test_the_gate_skips_only_a_foreign_platform(self, training_baseline):
        assert platform_matches(provenance()) and platform_matches({})
        assert platform_matches(dict(provenance(), torch="99.0.0"))
        assert not platform_matches({"platform": "Neverland-vax"})
        if not regenerating():
            assert training_baseline.generated_on["platform"] == "Linux-x86_64"
            if platform_matches(training_baseline.generated_on):
                assert training_baseline.values, "baseline file is empty"


class TestLearningRateTrajectory:
    def test_per_epoch_learning_rates_match_baseline_and_their_own_schedule(
        self, training_run, training_baseline
    ):
        """
        Exact, with no tolerance: ``Initial * Factor ** (epoch // Interval)`` in Python floats
        is identical everywhere. Also recomputed from the config independently of the
        baseline, and the two schedules must differ, or a swap would be invisible.
        """
        records = training_run["records"]
        assert [r["epoch"] for r in records] == list(range(1, N_EPOCHS + 1))
        assert records[0]["targets"] == {"latent": "latent", "model_0": "model"}
        observed = {name: [r["lrs"][name] for r in records] for name in ("model_0", "latent")}
        training_baseline.check("learning_rates_per_epoch", observed, portable=True)

        by_target = {entry["Target"]: entry for entry in LR_SCHEDULE}
        for name, target in (("model_0", "model"), ("latent", "latent")):
            spec = by_target[target]
            for record in records:
                expected = spec["Initial"] * spec["Factor"] ** (record["epoch"] // spec["Interval"])
                assert record["lrs"][name] == pytest.approx(expected), (record["epoch"], name)
        assert observed["model_0"] != observed["latent"]
        assert len(set(observed["model_0"])) > 1 and len(set(observed["latent"])) > 1


class TestLossAndLatents:
    def test_the_trajectories_match_baseline(self, training_run, training_baseline):
        records = training_run["records"]
        for key, field in (
            ("loss_trajectory", "loss"),
            ("l1_loss_trajectory", "l1_loss"),
            ("code_reg_loss_trajectory", "code_reg_loss"),
        ):
            training_baseline.check(key, [r[field] for r in records], rtol=LOSS_RTOL)
        norms = [r["latent_norms"] for r in records]
        training_baseline.check("latent_norm_trajectory", norms, atol=LATENT_NORM_ATOL)
        training_baseline.check("final_latent_norms", norms[-1], atol=LATENT_NORM_ATOL)

    def test_training_reduces_the_loss(self, training_run, training_dataset):
        """A run frozen at its initial loss would match a frozen baseline forever."""
        losses = [r["loss"] for r in training_run["records"]]
        assert losses[-1] < losses[0], losses
        assert len(training_run["records"][-1]["latent_norms"]) == len(training_dataset)


def test_the_run_leaves_checkpoints_and_a_config_that_record_the_targets(training_run):
    """
    ``save_frequency=4`` with ``checkpoint_epochs=8`` saves at epochs 4 and 8. The
    returned history is what wandb would have seen (#28). ``mesh_names`` is recorded but
    no model reads it, which is why surface identity in ``reconstruct_mesh`` is positional.
    """
    directory = training_run["config"]["experiment_directory"]
    for sub in ("model", "latent_codes"):
        epochs = sorted(int(f.split(".")[0]) for f in os.listdir(os.path.join(directory, sub)))
        assert epochs == [4, 8], sub

    groups = torch.load(os.path.join(directory, "model", "8.pth"), weights_only=False)["optimizer"][
        "param_groups"
    ]
    assert [(g["name"], g["target"]) for g in groups] == [
        ("latent", "latent"),
        ("model_0", "model"),
    ]

    with open(os.path.join(directory, "model_params_config.json"), encoding="utf-8") as f:
        saved = json.load(f)
    assert [e["Target"] for e in saved["LearningRateSchedule"]] == ["model", "latent"]
    assert saved["mesh_names"] == ["bone", "cart"]
    assert not hasattr(training_run["model"], "mesh_names")

    history = training_run["returned"]
    assert [entry["epoch"] for entry in history] == list(range(1, N_EPOCHS + 1))
    assert all({"loss", "l1_loss", "lrs", "targets", "latent_norms"} <= e.keys() for e in history)


class TestDeliberateBreak:
    """
    A harness nobody has seen fail is not evidence. This transposes the two ``Target`` labels,
    the bug that started this work, and asserts the baselines reject the result by at least
    ``MIN_HEADROOM`` times their tolerance. The learning rates are compared exactly.
    """

    @pytest.fixture(scope="class")
    def swapped(self, training_dataset, tmp_path_factory):
        config = training_config(tmp_path_factory.mktemp("swapped_lr"))
        schedule = copy.deepcopy(config["LearningRateSchedule"])
        schedule[0]["Target"], schedule[1]["Target"] = schedule[1]["Target"], schedule[0]["Target"]
        config["LearningRateSchedule"] = schedule
        return run_training(config, build_model(config), training_dataset)[0]

    def test_swapping_lr_targets_fails_every_baseline(self, swapped, training_baseline):
        if regenerating():
            pytest.skip("baselines are being rewritten")
        rates = {name: [r["lrs"][name] for r in swapped] for name in ("model_0", "latent")}
        with pytest.raises(AssertionError, match="differs from baseline"):
            training_baseline.check("learning_rates_per_epoch", rates, portable=True)

        for key, observed, tolerance in (
            ("loss_trajectory", [r["loss"] for r in swapped], dict(rtol=LOSS_RTOL)),
            ("final_latent_norms", swapped[-1]["latent_norms"], dict(atol=LATENT_NORM_ATOL)),
        ):
            with pytest.raises(AssertionError, match="differs from baseline"):
                training_baseline.check(key, observed, **tolerance)
            measured = headroom(training_baseline, key, observed, **tolerance)
            assert measured >= MIN_HEADROOM, (
                f"the swap moves {key} only {measured:.1f}x its tolerance, under the "
                f"MIN_HEADROOM of {MIN_HEADROOM}x. Widen the break, never the tolerance."
            )


class TestClampedPredictionGradients:
    """
    With ``enforce_minmax``, ``train_epoch`` clamps the prediction as well as the target,
    and ``torch.clamp`` passes no gradient outside its bounds. So every sample predicted
    outside ``+/-clamp_dist`` contributes no gradient however wrong it is, and
    ``clamp_dist`` is a training-dynamics knob. The harness uses 1.0, as both shipped
    ShapeMedKnee configs do; ``default_config.json``'s DeepSDF value is 0.1.
    """

    def test_fraction_of_dead_samples_at_init_matches_baseline(
        self, training_run, training_baseline
    ):
        """How much of the signal ``clamp_dist=0.1`` discards before the first step."""
        model = build_model(training_run["config"])
        model.eval()
        torch.manual_seed(0)
        inputs = torch.cat(
            [torch.zeros(2048, training_run["config"]["latent_size"]), torch.rand(2048, 3) * 2 - 1],
            1,
        )
        with torch.no_grad():
            fraction_dead = (model(inputs).abs() > 0.1).float().mean().item()
        assert fraction_dead > 0.25, f"only {fraction_dead:.1%} fall outside +/-0.1"
        training_baseline.check("fraction_dead_at_init_clamp_0_1", fraction_dead, atol=1e-6)


class TestResumeContract:
    """
    ``resume_epoch`` names the last completed epoch: its checkpoint is loaded and the loop
    continues after it. ``resume_epoch=1`` used to skip epoch 1 and load nothing (#49,
    § History 11).
    """

    def test_resuming_loads_exactly_the_named_checkpoint(self, training_dataset, tmp_path_factory):
        """
        With ``n_epochs`` equal to ``resume_epoch`` nothing trains, so the model must leave
        carrying that checkpoint's weights. It starts from a different seed, so a skipped
        load cannot pass.
        """
        source = training_config(tmp_path_factory.mktemp("resume_source"))
        source.update({"n_epochs": 2, "checkpoint_epochs": 1})
        records, _ = run_training(source, build_model(source), training_dataset)
        assert [r["epoch"] for r in records] == [1, 2]

        directory = source["experiment_directory"]
        for resume_epoch in (1, 2):
            config = training_config(directory)
            config.update(
                {"n_epochs": resume_epoch, "checkpoint_epochs": 1, "resume_epoch": resume_epoch}
            )
            model = build_model(config, seed=7)
            assert run_training(config, model, training_dataset)[0] == []
            saved = torch.load(
                os.path.join(directory, "model", f"{resume_epoch}.pth"), weights_only=False
            )["model"]
            state = model.state_dict()
            assert state.keys() == saved.keys()
            assert all(torch.equal(state[key], saved[key]) for key in state), resume_epoch


def test_a_schedule_free_run_survives_its_first_checkpoint(
    training_dataset, tmp_path_factory, monkeypatch
):
    """
    #42: the eval warm-up handed the decoder the raw dataloader item, so every schedule_free
    run died at its first checkpoint. ``schedulefree`` is not installed here, so it is
    stubbed: AdamW plus the ``train()``/``eval()`` switches the trainer uses.
    """
    import NSM.utils

    class StubAdamWScheduleFree(torch.optim.AdamW):
        def train(self):
            pass

        def eval(self):
            pass

    monkeypatch.setattr(
        NSM.utils, "schedulefree", types.SimpleNamespace(AdamWScheduleFree=StubAdamWScheduleFree)
    )
    config = training_config(tmp_path_factory.mktemp("schedule_free_run"))
    config.update({"optimizer": "schedule_free_AdamW", "n_epochs": 2, "checkpoint_epochs": 1})
    records, _ = run_training(config, build_model(config), training_dataset)
    assert [r["epoch"] for r in records] == [1, 2]
