"""
Regression tests for learning-rate schedule -> optimizer param-group mapping.

Motivating bug: ``get_optimizer`` ordered param groups ``[latent, model...]`` but
``adjust_learning_rate`` assigned ``lr_schedules[i]`` to ``param_groups[i]``, so from
epoch 1 onward the model and latent schedules were applied swapped. Every Adam/AdamW run
from May 2023 to Jul 2026 is affected; ``schedule_free_*`` runs are not.

Reported by Dr. Katherine Wolcott (Florida Museum of Natural History), 2026-07-10.
"""

import copy
import warnings

import pytest
import torch

from NSM.train.utils import add_plain_lr_to_config
from NSM.utils import (
    LR_CONVENTION_LEGACY,
    LR_CONVENTION_V2,
    adjust_learning_rate,
    get_learning_rate_schedules,
    get_optimizer,
    rename_optimizer_param_groups,
    restore_optimizer_param_group_names,
    save_model,
)

MODEL_LR = 0.01
LATENT_LR = 0.001


def make_config(model_lr=MODEL_LR, latent_lr=LATENT_LR, convention=LR_CONVENTION_V2, **extra):
    """Config whose entries are ordered [model, latent] under the v2 convention."""
    config = {
        "LearningRateSchedule": [
            {"Type": "Step", "Initial": model_lr, "Interval": 10, "Factor": 0.5},
            {"Type": "Step", "Initial": latent_lr, "Interval": 10, "Factor": 0.5},
        ],
        "optimizer": "Adam",
    }
    if convention is not None:
        config["lr_schedule_convention"] = convention
    config.update(extra)
    return config


def make_model(n_features=4):
    return torch.nn.Linear(n_features, 1)


def make_latents(n=3, dim=4):
    return torch.nn.Embedding(n, dim)


def build(config, models=None):
    """Build schedules + optimizer from a config, as the train loop does."""
    models = models if models is not None else make_model()
    schedules = get_learning_rate_schedules(config)
    optimizer = get_optimizer(
        models, make_latents(), lr_schedules=schedules, optimizer=config["optimizer"]
    )
    return schedules, optimizer


def lrs_by_name(optimizer):
    return {group["name"]: group["lr"] for group in optimizer.param_groups}


class TestNamedParamGroups:
    def test_get_optimizer_names_groups(self):
        _, optimizer = build(make_config())
        assert [g["name"] for g in optimizer.param_groups] == ["latent", "model_0"]

    def test_multiple_models_get_indexed_names(self):
        _, optimizer = build(make_config(), models=[make_model(), make_model()])
        assert [g["name"] for g in optimizer.param_groups] == ["latent", "model_0", "model_1"]

    def test_initial_lrs_are_correct_before_any_epoch(self):
        _, optimizer = build(make_config())
        assert lrs_by_name(optimizer) == {"latent": LATENT_LR, "model_0": MODEL_LR}


class TestScheduleMapping:
    """The core regression: mapping must be by name, never by position."""

    def test_lrs_not_swapped_after_first_epoch(self):
        schedules, optimizer = build(make_config())
        adjust_learning_rate(schedules, optimizer, epoch=1)

        lrs = lrs_by_name(optimizer)
        assert lrs["model_0"] == pytest.approx(MODEL_LR)
        assert lrs["latent"] == pytest.approx(LATENT_LR)

    def test_mapping_survives_reversed_group_order(self):
        """Position must not matter, even if group order is reversed."""
        schedules, optimizer = build(make_config())
        optimizer.param_groups.reverse()
        adjust_learning_rate(schedules, optimizer, epoch=1)

        lrs = lrs_by_name(optimizer)
        assert lrs["model_0"] == pytest.approx(MODEL_LR)
        assert lrs["latent"] == pytest.approx(LATENT_LR)

    def test_decay_applies_to_correct_group(self):
        schedules, optimizer = build(make_config())
        adjust_learning_rate(schedules, optimizer, epoch=10)  # one interval -> factor 0.5

        lrs = lrs_by_name(optimizer)
        assert lrs["model_0"] == pytest.approx(MODEL_LR * 0.5)
        assert lrs["latent"] == pytest.approx(LATENT_LR * 0.5)

    def test_all_model_groups_share_the_model_schedule(self):
        schedules, optimizer = build(make_config(), models=[make_model(), make_model()])
        adjust_learning_rate(schedules, optimizer, epoch=1)

        lrs = lrs_by_name(optimizer)
        assert lrs["model_0"] == pytest.approx(MODEL_LR)
        assert lrs["model_1"] == pytest.approx(MODEL_LR)
        assert lrs["latent"] == pytest.approx(LATENT_LR)

    def test_classification_heads_group_uses_model_schedule(self):
        schedules, optimizer = build(make_config())
        optimizer.add_param_group(
            {"name": "classification_heads", "params": make_model().parameters(), "lr": 0.0}
        )
        adjust_learning_rate(schedules, optimizer, epoch=1)

        assert lrs_by_name(optimizer)["classification_heads"] == pytest.approx(MODEL_LR)

    def test_unnamed_group_raises(self):
        schedules, optimizer = build(make_config())
        del optimizer.param_groups[0]["name"]

        with pytest.raises(KeyError, match="no recognized 'name'"):
            adjust_learning_rate(schedules, optimizer, epoch=1)

    def test_too_few_schedules_raises(self):
        _, optimizer = build(make_config())
        schedules = get_learning_rate_schedules(make_config())

        with pytest.raises(ValueError, match="at least 2 lr_schedules"):
            adjust_learning_rate(schedules[:1], optimizer, epoch=1)


class TestMigrationGuard:
    """A pre-fix config run on fixed code must fail loudly, not silently swap."""

    def test_missing_convention_raises_for_adam(self):
        with pytest.raises(ValueError, match="lr_schedule_convention"):
            get_learning_rate_schedules(make_config(convention=None))

    def test_missing_convention_raises_for_adamw(self):
        with pytest.raises(ValueError, match="lr_schedule_convention"):
            get_learning_rate_schedules(make_config(convention=None, optimizer="AdamW"))

    def test_error_message_offers_both_migration_paths(self):
        with pytest.raises(ValueError) as exc:
            get_learning_rate_schedules(make_config(convention=None))

        message = str(exc.value)
        assert LR_CONVENTION_LEGACY in message
        assert LR_CONVENTION_V2 in message

    def test_schedule_free_needs_no_convention(self):
        """schedule_free_* skipped adjust_learning_rate, so it was never affected."""
        schedules = get_learning_rate_schedules(
            make_config(convention=None, optimizer="schedule_free_AdamW")
        )
        assert schedules[0].get_learning_rate(0) == pytest.approx(MODEL_LR)
        assert schedules[1].get_learning_rate(0) == pytest.approx(LATENT_LR)

    def test_unknown_convention_raises(self):
        with pytest.raises(ValueError, match="Unknown lr_schedule_convention"):
            get_learning_rate_schedules(make_config(convention="something_else"))

    def test_fewer_than_two_entries_raises(self):
        config = make_config()
        config["LearningRateSchedule"] = config["LearningRateSchedule"][:1]

        with pytest.raises(ValueError, match="at least 2 LearningRateSchedule"):
            get_learning_rate_schedules(config)

    def test_legacy_convention_swaps_into_canonical_order(self):
        """A legacy config lists [latent, model]; canonical order is [model, latent]."""
        legacy = make_config(
            model_lr=LATENT_LR,  # legacy entry 0 held the LATENT lr
            latent_lr=MODEL_LR,  # legacy entry 1 held the MODEL lr
            convention=LR_CONVENTION_LEGACY,
        )
        schedules = get_learning_rate_schedules(legacy)

        assert schedules[0].get_learning_rate(0) == pytest.approx(MODEL_LR)
        assert schedules[1].get_learning_rate(0) == pytest.approx(LATENT_LR)


def buggy_lrs_for_old_config(config, epoch):
    """
    Reproduce the PRE-FIX runtime mapping: schedules assigned to groups by position,
    against get_optimizer's [latent, model] group order.
    """
    specs = config["LearningRateSchedule"]
    initials = [s["Initial"] for s in specs]
    factors = [s["Factor"] for s in specs]
    intervals = [s["Interval"] for s in specs]

    def lr(i):
        return initials[i] * (factors[i] ** (epoch // intervals[i]))

    # param_groups[0] is latent, param_groups[1] is model -- assigned lr_schedules[0]/[1]
    return {"latent": lr(0), "model_0": lr(1)}


class TestHistoricalEquivalence:
    """
    The migration promise: an old config + legacy_swapped on fixed code must produce
    exactly the learning rates that config produced under the buggy code.
    """

    @pytest.mark.parametrize("epoch", [1, 5, 10, 25, 100])
    def test_legacy_swapped_reproduces_pre_fix_behaviour(self, epoch):
        # An untouched historical config, entries in their original order.
        old_config = {
            "LearningRateSchedule": [
                {"Type": "Step", "Initial": 0.005, "Interval": 16, "Factor": 0.95},
                {"Type": "Step", "Initial": 0.0001, "Interval": 1000, "Factor": 0.1},
            ],
            "optimizer": "Adam",
        }
        expected = buggy_lrs_for_old_config(old_config, epoch)

        migrated = dict(old_config, lr_schedule_convention=LR_CONVENTION_LEGACY)
        schedules, optimizer = build(migrated)
        adjust_learning_rate(schedules, optimizer, epoch=epoch)

        actual = lrs_by_name(optimizer)
        assert actual["latent"] == pytest.approx(expected["latent"])
        assert actual["model_0"] == pytest.approx(expected["model_0"])

    def test_v2_differs_from_legacy_on_the_same_config(self):
        """Sanity: the two conventions really are different, so the guard earns its keep."""
        config = {
            "LearningRateSchedule": [
                {"Type": "Step", "Initial": 0.005, "Interval": 16, "Factor": 0.95},
                {"Type": "Step", "Initial": 0.0001, "Interval": 1000, "Factor": 0.1},
            ],
            "optimizer": "Adam",
        }
        v2 = get_learning_rate_schedules(dict(config, lr_schedule_convention=LR_CONVENTION_V2))
        legacy = get_learning_rate_schedules(
            dict(config, lr_schedule_convention=LR_CONVENTION_LEGACY)
        )

        assert v2[0].get_learning_rate(0) != legacy[0].get_learning_rate(0)


class TestCheckpointResume:
    def test_names_survive_loading_a_post_fix_checkpoint(self):
        """state_dict() retains 'name', so new checkpoints need no restoration."""
        _, optimizer = build(make_config())
        state = optimizer.state_dict()
        assert all("name" in group for group in state["param_groups"])

        _, fresh = build(make_config())
        fresh.load_state_dict(state)

        assert [g.get("name") for g in fresh.param_groups] == ["latent", "model_0"]

    def test_names_are_lost_loading_a_pre_fix_checkpoint(self):
        """
        The behaviour that makes restoration necessary: load_state_dict adopts the
        CHECKPOINT's param-group metadata, so a pre-fix checkpoint (no names) strips the
        names off a freshly built, correctly named optimizer.
        """
        _, optimizer = build(make_config())
        old_state = copy.deepcopy(optimizer.state_dict())
        for group in old_state["param_groups"]:
            group.pop("name", None)

        _, fresh = build(make_config())
        fresh.load_state_dict(old_state)

        assert all(g.get("name") is None for g in fresh.param_groups)

    def test_restore_from_saved_names(self):
        _, optimizer = build(make_config())
        checkpoint = {"optimizer_group_names": ["latent", "model_0"]}

        _, fresh = build(make_config())
        fresh.load_state_dict(optimizer.state_dict())
        restore_optimizer_param_group_names(fresh, checkpoint, n_model_groups=1)

        assert [g["name"] for g in fresh.param_groups] == ["latent", "model_0"]

    def test_restore_falls_back_and_warns_for_old_checkpoints(self):
        _, optimizer = build(make_config())
        optimizer.load_state_dict(optimizer.state_dict())

        with pytest.warns(UserWarning, match="saved before Aug 2026"):
            restore_optimizer_param_group_names(optimizer, {}, n_model_groups=1)

        assert [g["name"] for g in optimizer.param_groups] == ["latent", "model_0"]

    def test_restore_rejects_length_mismatch(self):
        _, optimizer = build(make_config())

        with pytest.raises(ValueError, match="length mismatch"):
            restore_optimizer_param_group_names(
                optimizer, {"optimizer_group_names": ["latent"]}, n_model_groups=1
            )

    def test_rename_rejects_wrong_group_count(self):
        _, optimizer = build(make_config())

        with pytest.raises(ValueError, match="Expected 3 optimizer param groups"):
            rename_optimizer_param_groups(optimizer, n_model_groups=2)

    def test_resume_applies_correct_lrs_after_restore(self):
        """End-to-end: resume must not resurrect the swap."""
        schedules, optimizer = build(make_config())

        _, fresh = build(make_config())
        fresh.load_state_dict(optimizer.state_dict())
        restore_optimizer_param_group_names(
            fresh, {"optimizer_group_names": ["latent", "model_0"]}, n_model_groups=1
        )
        adjust_learning_rate(schedules, fresh, epoch=1)

        lrs = lrs_by_name(fresh)
        assert lrs["model_0"] == pytest.approx(MODEL_LR)
        assert lrs["latent"] == pytest.approx(LATENT_LR)


class TestSaveModel:
    def test_save_model_persists_group_names(self, tmp_path):
        _, optimizer = build(make_config())
        model = make_model()
        config = {"experiment_directory": str(tmp_path)}

        save_model(config, epoch=1, decoder=model, optimizer=optimizer)

        checkpoint = torch.load(tmp_path / "model" / "1.pth", weights_only=False)
        assert checkpoint["optimizer_group_names"] == ["latent", "model_0"]

    def test_save_model_rejects_unnamed_groups(self, tmp_path):
        _, optimizer = build(make_config())
        del optimizer.param_groups[0]["name"]
        config = {"experiment_directory": str(tmp_path)}

        with pytest.raises(ValueError, match="must have a 'name'"):
            save_model(config, epoch=1, decoder=make_model(), optimizer=optimizer)

    def test_save_model_without_optimizer_still_works(self, tmp_path):
        config = {"experiment_directory": str(tmp_path)}
        save_model(config, epoch=1, decoder=make_model(), optimizer=None)

        checkpoint = torch.load(tmp_path / "model" / "1.pth", weights_only=False)
        assert checkpoint["optimizer_group_names"] is None


class TestPlainLrLogging:
    """
    add_plain_lr_to_config flattens the schedules into scalar keys for wandb. Those
    labels must follow the config's convention, or an experiment tracker records the
    model LR under 'latent_lr_initial' and vice versa.
    """

    def test_v2_labels_match_entry_order(self):
        config = add_plain_lr_to_config(make_config(convention=LR_CONVENTION_V2))

        assert config["model_lr_initial"] == pytest.approx(MODEL_LR)
        assert config["latent_lr_initial"] == pytest.approx(LATENT_LR)

    def test_legacy_labels_are_reversed(self):
        # legacy config: entry 0 held the latent LR, entry 1 held the model LR
        config = add_plain_lr_to_config(
            make_config(model_lr=LATENT_LR, latent_lr=MODEL_LR, convention=LR_CONVENTION_LEGACY)
        )

        assert config["model_lr_initial"] == pytest.approx(MODEL_LR)
        assert config["latent_lr_initial"] == pytest.approx(LATENT_LR)

    def test_logged_lrs_agree_with_optimizer(self):
        """The logged values must be the ones actually applied, under both conventions."""
        for convention, model_lr, latent_lr in (
            (LR_CONVENTION_V2, MODEL_LR, LATENT_LR),
            (LR_CONVENTION_LEGACY, LATENT_LR, MODEL_LR),
        ):
            raw = make_config(model_lr=model_lr, latent_lr=latent_lr, convention=convention)
            logged = add_plain_lr_to_config(dict(raw))
            schedules, optimizer = build(raw)
            adjust_learning_rate(schedules, optimizer, epoch=1)

            lrs = lrs_by_name(optimizer)
            assert lrs["model_0"] == pytest.approx(logged["model_lr_initial"])
            assert lrs["latent"] == pytest.approx(logged["latent_lr_initial"])

    def test_explicit_indices_still_override(self):
        config = add_plain_lr_to_config(make_config(), idx_model=1, idx_latent=0)

        assert config["model_lr_initial"] == pytest.approx(LATENT_LR)


class TestShippedConfigs:
    def test_default_config_json_declares_convention_and_loads(self):
        import json
        import os

        import NSM

        path = os.path.join(os.path.dirname(NSM.__file__), "configs", "default_config.json")
        with open(path) as f:
            config = json.load(f)

        assert config["lr_schedule_convention"] == LR_CONVENTION_V2

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            schedules = get_learning_rate_schedules(config)

        # index 0 = model, and the shipped model LR is the larger of the two
        assert schedules[0].get_learning_rate(0) > schedules[1].get_learning_rate(0)
