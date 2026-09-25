"""
End-to-end training regression: 8 epochs, CPU, fixed seed, asserted against baselines.

What this pins, most important first:

1. **The learning rate of each param group at every epoch**, which catches the schedule
   swap in ``docs/KNOWN_ISSUES.md`` §1. The two schedules differ in Interval and Factor as
   well as Initial, so transposing them inverts the run. ``TestDeliberateBreak`` transposes
   them and asserts the baselines fail.
2. The loss trajectory and its components.
3. The latent norms.
"""

import copy
import json
import os
import types

import numpy as np
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
    """Fails if ``NSM_REGENERATE_BASELINES`` is set, so a run that rewrote baselines reads red."""
    assert not regenerating(), f"{REGENERATE_ENV} is set: baselines were REWRITTEN, not checked"


def test_the_reconstruction_decoder_is_not_being_regenerated():
    """
    Fails if ``NSM_REGENERATE_RECON_DECODER`` is set, so a run that retrained the committed
    decoder reads red.

    Such a run rewrote ``assets/reconstruction_decoder.pt`` and compared every reconstruction
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
        """
        Fails if the harness's ``platform_matches`` skips baselines on a torch version change
        or applies them on a foreign platform, or ``training.json`` was generated off
        Linux-x86_64 or is empty.
        """
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
        Fails if ``train_deep_sdf`` runs an epoch with a param group's LR other than its own
        ``Target`` schedule's ``Initial * Factor ** (epoch // Interval)``, as when the model
        and latent schedules are swapped (KNOWN_ISSUES History 1).

        The baseline comparison is exact, with no tolerance: that expression in Python floats
        is identical everywhere, so it runs on every platform. The rates are also recomputed
        from the config. The two schedules must differ and each must change within the run,
        or a swap would be invisible.
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
        """
        Fails if ``train_deep_sdf``'s 8-epoch loss, L1, regularization or per-subject
        latent-norm trajectory moves from baseline by more than ``LOSS_RTOL`` or
        ``LATENT_NORM_ATOL``.

        The baselines were generated on Linux-x86_64, and this test skips elsewhere.
        """
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
        """
        Fails if ``train_deep_sdf``'s last-epoch loss is not below its first, or the history's
        ``latent_norms`` stops holding one entry per subject.

        A run frozen at its initial loss would match a baseline regenerated from it forever.
        This test runs on every platform.
        """
        losses = [r["loss"] for r in training_run["records"]]
        assert losses[-1] < losses[0], losses
        assert len(training_run["records"][-1]["latent_norms"]) == len(training_dataset)


def test_the_run_leaves_checkpoints_and_a_config_that_record_the_targets(training_run):
    """
    Fails if ``train_deep_sdf`` checkpoints at epochs other than 4 and 8, saves param groups
    without ``name`` and ``target``, writes ``model_params_config.json`` without ``Target``
    or ``mesh_names``, or returns a history missing an epoch or key (#28).

    ``save_frequency=4`` with ``checkpoint_epochs=8`` saves at epochs 4 and 8. The returned
    history is what wandb would have seen. ``mesh_names`` is recorded in the config and not
    on the model, which is why surface identity in ``reconstruct_mesh`` is positional.
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
    which reproduces the schedule swap in ``docs/KNOWN_ISSUES.md`` §1, and asserts the
    baselines reject the result by at least ``MIN_HEADROOM`` times their tolerance. The
    learning rates are compared exactly.
    """

    @pytest.fixture(scope="class")
    def swapped(self, training_dataset, tmp_path_factory):
        config = training_config(tmp_path_factory.mktemp("swapped_lr"))
        schedule = copy.deepcopy(config["LearningRateSchedule"])
        schedule[0]["Target"], schedule[1]["Target"] = schedule[1]["Target"], schedule[0]["Target"]
        config["LearningRateSchedule"] = schedule
        return run_training(config, build_model(config), training_dataset)[0]

    def test_swapping_lr_targets_fails_every_baseline(self, swapped, training_baseline):
        """
        Fails if ``LOSS_RTOL`` or ``LATENT_NORM_ATOL`` is widened, or the baselines lose
        sensitivity, until the swapped run moves the loss trajectory or the final latent
        norms by less than ``MIN_HEADROOM`` times the tolerance.

        It guards the tolerances. ``TestLearningRateTrajectory`` catches the swap itself.
        """
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
        """
        Fails if the fraction of a fresh seed-42 ``TriplanarDecoder``'s zero-latent
        predictions outside +/-0.1 moves from baseline or falls to 25% or below.

        The measurement behind the ``enforce_minmax`` entry in ``docs/KNOWN_ISSUES.md``
        § Open: how much of the signal ``clamp_dist=0.1`` discards before the first step. It
        never calls ``train_epoch``, so fixing the clamp does not turn it red.
        """
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
    continues after it.
    """

    def test_a_resumed_run_continues_the_uninterrupted_one(
        self, training_dataset, tmp_path_factory, monkeypatch
    ):
        """
        Fails if ``train_deep_sdf`` resumed at epoch 1 of 2 trains any epoch but 2, or
        trains it differently from the uninterrupted run: in its losses, learning rates,
        latent norms or final weights (#49; KNOWN_ISSUES History 11).

        Skipping the model, optimizer or latent restore fails it. The resumed model starts
        from a different seed. Checkpoints hold no random state (#119), so the test records
        it at each save and restores it after the resume.
        """
        import NSM.train.train_deep_sdf as trainer

        random_state = {}
        save_checkpoint = trainer._save_checkpoint
        resume_from_checkpoint = trainer._resume_from_checkpoint

        def save_and_record(config, epoch, *args):
            random_state[epoch] = (torch.get_rng_state(), np.random.get_state())
            save_checkpoint(config, epoch, *args)

        def resume_and_restore(config, *args):
            resume_from_checkpoint(config, *args)
            if config["resume_epoch"]:
                torch_state, numpy_state = random_state[config["resume_epoch"]]
                torch.set_rng_state(torch_state)
                np.random.set_state(numpy_state)

        monkeypatch.setattr(trainer, "_save_checkpoint", save_and_record)
        monkeypatch.setattr(trainer, "_resume_from_checkpoint", resume_and_restore)

        config = training_config(tmp_path_factory.mktemp("resume"))
        config.update({"n_epochs": 2, "checkpoint_epochs": 1})
        uninterrupted = build_model(config)
        records, _ = run_training(copy.deepcopy(config), uninterrupted, training_dataset)
        assert [r["epoch"] for r in records] == [1, 2]

        config["resume_epoch"] = 1
        resumed = build_model(config, seed=7)
        assert run_training(config, resumed, training_dataset)[0] == records[1:]
        weights = zip(uninterrupted.state_dict().values(), resumed.state_dict().values())
        assert all(torch.equal(a, b) for a, b in weights)


def test_a_schedule_free_run_survives_its_first_checkpoint(
    training_dataset, tmp_path_factory, monkeypatch
):
    """
    Fails if a ``schedule_free_AdamW`` run through ``train_deep_sdf`` raises at a checkpoint
    epoch, in the eval warm-up or the save, or stops short of ``n_epochs`` (#42).

    ``schedulefree`` is not installed here, so it is stubbed: AdamW plus the
    ``train()``/``eval()`` switches the trainer uses.
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
