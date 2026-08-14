"""
Regression tests for learning-rate schedule -> optimizer param-group mapping.

Motivating bug: ``get_optimizer`` ordered param groups ``[latent, model...]`` but
``adjust_learning_rate`` assigned ``lr_schedules[i]`` to ``param_groups[i]``, so the model
and latent schedules were applied swapped for the whole of every affected run. Every
Adam/AdamW run from May 2023 to Aug 2026 is affected; ``schedule_free_*`` runs are not.

Reported by Dr. Katherine Wolcott (Florida Museum of Natural History), 2026-07-10.

Both orderings the bug depended on are now non-positional: param groups carry ``name``,
and schedule entries carry ``Target``.
"""

import copy
import warnings

import pytest
import torch

from NSM.train.utils import add_plain_lr_to_config
from NSM.utils import (
    LR_TARGET_LATENT,
    LR_TARGET_MODEL,
    adjust_learning_rate,
    get_learning_rate_schedules,
    get_optimizer,
    rename_optimizer_param_groups,
    resolve_schedule_targets,
    restore_optimizer_param_group_names,
    save_model,
)

MODEL_LR = 0.01
LATENT_LR = 0.001


def make_config(model_lr=MODEL_LR, latent_lr=LATENT_LR, targets=("model", "latent"), **extra):
    """
    Config with one entry per target.

    ``targets`` sets the Target of entry 0 and entry 1 respectively; ``None`` in either
    slot omits the key from that entry. The LR values stay attached to their *role*, not
    their position, so reordering ``targets`` reorders the entries too.
    """
    specs = {
        "model": {"Type": "Step", "Initial": model_lr, "Interval": 10, "Factor": 0.5},
        "latent": {"Type": "Step", "Initial": latent_lr, "Interval": 10, "Factor": 0.5},
    }
    # An unlabelled entry still needs content; fall back to whichever role is unclaimed.
    unclaimed = [role for role in ("model", "latent") if role not in targets]
    entries = []
    for target in targets:
        role = target if target in specs else unclaimed.pop(0)
        entry = dict(specs[role])
        if target is not None:
            entry["Target"] = target
        entries.append(entry)

    config = {"LearningRateSchedule": entries, "optimizer": "Adam"}
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


class TestTargetResolution:
    """Target decides the mapping. Position must never be consulted."""

    def test_targets_resolved_in_entry_order(self):
        targets = resolve_schedule_targets(make_config()["LearningRateSchedule"])
        assert targets == [LR_TARGET_MODEL, LR_TARGET_LATENT]

    def test_entry_order_does_not_change_the_mapping(self):
        """The whole point: listing latent first must change nothing."""
        forward = get_learning_rate_schedules(make_config(targets=("model", "latent")))
        reversed_ = get_learning_rate_schedules(make_config(targets=("latent", "model")))

        for schedules in (forward, reversed_):
            assert schedules[0].get_learning_rate(0) == pytest.approx(MODEL_LR)
            assert schedules[1].get_learning_rate(0) == pytest.approx(LATENT_LR)

    def test_reordered_entries_reach_the_right_groups(self):
        config = make_config(targets=("latent", "model"))
        schedules, optimizer = build(config)
        adjust_learning_rate(schedules, optimizer, epoch=1)

        lrs = lrs_by_name(optimizer)
        assert lrs["model_0"] == pytest.approx(MODEL_LR)
        assert lrs["latent"] == pytest.approx(LATENT_LR)

    def test_the_whole_schedule_travels_with_its_target(self):
        """Not just Initial -- Interval and Factor must follow the target too."""
        config = {
            "optimizer": "Adam",
            "LearningRateSchedule": [
                {
                    "Target": "latent",
                    "Type": "Step",
                    "Initial": 0.005,
                    "Interval": 16.7,
                    "Factor": 0.95,
                },
                {
                    "Target": "model",
                    "Type": "Step",
                    "Initial": 0.0001,
                    "Interval": 1000,
                    "Factor": 0.1,
                },
            ],
        }
        model_sched, latent_sched = get_learning_rate_schedules(config)

        # At epoch 1000 the model has taken one x0.1 step; the latent has decayed smoothly.
        assert model_sched.get_learning_rate(1000) == pytest.approx(0.0001 * 0.1)
        assert latent_sched.get_learning_rate(1000) == pytest.approx(0.005 * 0.95 ** (1000 // 16.7))


class TestMigrationGuard:
    """An un-annotated config must fail loudly, never fall back to a positional guess."""

    def test_missing_target_raises(self):
        with pytest.raises(ValueError, match="must declare 'Target'"):
            get_learning_rate_schedules(make_config(targets=(None, None)))

    def test_partially_annotated_config_raises(self):
        """Half-migrated is the dangerous case -- it looks done at a glance."""
        with pytest.raises(ValueError, match="not migrated"):
            get_learning_rate_schedules(make_config(targets=("model", None)))

    def test_schedule_free_also_requires_targets(self):
        """One rule for every optimizer; get_optimizer was positional for these too."""
        with pytest.raises(ValueError, match="must declare 'Target'"):
            get_learning_rate_schedules(
                make_config(targets=(None, None), optimizer="schedule_free_AdamW")
            )

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="unknown Target"):
            get_learning_rate_schedules(make_config(targets=("model", "decoder")))

    def test_duplicate_targets_raise(self):
        with pytest.raises(ValueError, match="exactly once each"):
            get_learning_rate_schedules(make_config(targets=("model", "model")))

    def test_wrong_entry_count_raises(self):
        config = make_config()
        config["LearningRateSchedule"] = config["LearningRateSchedule"][:1]

        with pytest.raises(ValueError, match="exactly 2 LearningRateSchedule"):
            get_learning_rate_schedules(config)

    def test_error_quotes_the_historical_mapping_for_adam(self):
        """Adam ran through adjust_learning_rate: entry 0 drove the latents."""
        with pytest.raises(ValueError) as exc:
            get_learning_rate_schedules(make_config(targets=(None, None), optimizer="AdamW"))

        message = str(exc.value)
        assert "entry 0 -> latent, entry 1 -> model" in message

    def test_error_quotes_the_opposite_mapping_for_schedule_free(self):
        """schedule_free kept get_optimizer's assignment: entry 0 drove the model."""
        with pytest.raises(ValueError) as exc:
            get_learning_rate_schedules(
                make_config(targets=(None, None), optimizer="schedule_free_AdamW")
            )

        message = str(exc.value)
        assert "entry 0 -> model, entry 1 -> latent" in message

    def test_error_includes_paste_ready_json(self):
        with pytest.raises(ValueError) as exc:
            get_learning_rate_schedules(make_config(targets=(None, None)))

        message = str(exc.value)
        assert '"Target": "latent"' in message
        assert '"Target": "model"' in message
        assert '"LearningRateSchedule"' in message


def buggy_lrs_for_old_config(specs, epoch):
    """
    Reproduce the PRE-FIX runtime mapping: schedules assigned to groups by position,
    against get_optimizer's [latent, model] group order.
    """

    def lr(i):
        return specs[i]["Initial"] * (specs[i]["Factor"] ** (epoch // specs[i]["Interval"]))

    # param_groups[0] is latent, param_groups[1] is model -- assigned lr_schedules[0]/[1]
    return {"latent": lr(0), "model_0": lr(1)}


#: The real ShapeMedKnee_2024 schedules, entries in their original (un-annotated) order.
#: Deliberately a config whose two entries differ in Type/Interval/Factor and not just
#: Initial -- annotating it wrongly inverts the run rather than perturbing it.
SHAPEMEDKNEE_2024_SPECS = [
    {
        "Type": "Step",
        "Initial": 0.005,
        "Interval": 16.666666666666668,
        "Factor": 0.9523809523809523,
    },
    {"Type": "Step", "Initial": 0.0001, "Interval": 1000, "Factor": 0.1},
]


class TestHistoricalEquivalence:
    """
    The migration promise: annotating a pre-fix Adam/AdamW config with the historical
    targets (entry 0 -> latent, entry 1 -> model) must reproduce exactly the learning
    rates that config produced under the buggy code.
    """

    @pytest.mark.parametrize("epoch", [1, 5, 100, 500, 1000, 1500, 2000])
    def test_annotating_with_historical_targets_reproduces_pre_fix_behaviour(self, epoch):
        expected = buggy_lrs_for_old_config(SHAPEMEDKNEE_2024_SPECS, epoch)

        migrated = {
            "optimizer": "AdamW",
            "LearningRateSchedule": [
                dict(SHAPEMEDKNEE_2024_SPECS[0], Target=LR_TARGET_LATENT),
                dict(SHAPEMEDKNEE_2024_SPECS[1], Target=LR_TARGET_MODEL),
            ],
        }
        schedules, optimizer = build(migrated)
        adjust_learning_rate(schedules, optimizer, epoch=epoch)

        actual = lrs_by_name(optimizer)
        assert actual["latent"] == pytest.approx(expected["latent"])
        assert actual["model_0"] == pytest.approx(expected["model_0"])

    def test_the_opposite_annotation_really_is_a_different_run(self):
        """Sanity: the guard earns its keep only if getting this wrong matters."""
        historical = get_learning_rate_schedules(
            {
                "optimizer": "AdamW",
                "LearningRateSchedule": [
                    dict(SHAPEMEDKNEE_2024_SPECS[0], Target=LR_TARGET_LATENT),
                    dict(SHAPEMEDKNEE_2024_SPECS[1], Target=LR_TARGET_MODEL),
                ],
            }
        )
        inverted = get_learning_rate_schedules(
            {
                "optimizer": "AdamW",
                "LearningRateSchedule": [
                    dict(SHAPEMEDKNEE_2024_SPECS[0], Target=LR_TARGET_MODEL),
                    dict(SHAPEMEDKNEE_2024_SPECS[1], Target=LR_TARGET_LATENT),
                ],
            }
        )

        # 50x apart at epoch 0 -- this is an inversion, not a perturbation.
        assert historical[0].get_learning_rate(0) == pytest.approx(0.0001)
        assert inverted[0].get_learning_rate(0) == pytest.approx(0.005)


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
    labels must follow each entry's Target, or an experiment tracker records the model LR
    under 'latent_lr_initial' and vice versa.
    """

    def test_labels_follow_target_not_position(self):
        config = add_plain_lr_to_config(make_config(targets=("model", "latent")))

        assert config["model_lr_initial"] == pytest.approx(MODEL_LR)
        assert config["latent_lr_initial"] == pytest.approx(LATENT_LR)

    def test_labels_survive_reordered_entries(self):
        config = add_plain_lr_to_config(make_config(targets=("latent", "model")))

        assert config["model_lr_initial"] == pytest.approx(MODEL_LR)
        assert config["latent_lr_initial"] == pytest.approx(LATENT_LR)

    def test_logged_lrs_agree_with_optimizer(self):
        """The logged values must be the ones actually applied, in either entry order."""
        for targets in (("model", "latent"), ("latent", "model")):
            raw = make_config(targets=targets)
            logged = add_plain_lr_to_config(copy.deepcopy(raw))
            schedules, optimizer = build(raw)
            adjust_learning_rate(schedules, optimizer, epoch=1)

            lrs = lrs_by_name(optimizer)
            assert lrs["model_0"] == pytest.approx(logged["model_lr_initial"])
            assert lrs["latent"] == pytest.approx(logged["latent_lr_initial"])

    def test_full_schedule_is_logged_not_just_initial(self):
        config = add_plain_lr_to_config(make_config(targets=("latent", "model")))

        assert config["model_lr_type"] == "Step"
        assert config["model_lr_update_interval"] == 10
        assert config["model_lr_update_factor"] == pytest.approx(0.5)

    def test_explicit_indices_still_override(self):
        config = add_plain_lr_to_config(make_config(), idx_model=1, idx_latent=0)

        assert config["model_lr_initial"] == pytest.approx(LATENT_LR)


def load_shipped_default_config():
    import json
    import os

    import NSM

    path = os.path.join(os.path.dirname(NSM.__file__), "configs", "default_config.json")
    with open(path) as f:
        return json.load(f)


class TestShippedConfigs:
    def test_default_config_json_annotates_targets_and_loads(self):
        config = load_shipped_default_config()

        targets = [entry["Target"] for entry in config["LearningRateSchedule"]]
        assert sorted(targets) == [LR_TARGET_LATENT, LR_TARGET_MODEL]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            schedules = get_learning_rate_schedules(config)

        # index 0 = model, and the shipped model LR is the larger of the two
        assert schedules[0].get_learning_rate(0) > schedules[1].get_learning_rate(0)

    def test_generated_default_config_annotates_targets(self, tmp_path, monkeypatch):
        # NB: importing this module writes ./default_config.json as a side effect, so run
        # the import from a tmp cwd rather than littering the repo root.
        monkeypatch.chdir(tmp_path)
        from NSM.configs.generate_sdf_default_config import config

        targets = [entry["Target"] for entry in config["LearningRateSchedule"]]
        assert sorted(targets) == [LR_TARGET_LATENT, LR_TARGET_MODEL]

    def test_saved_config_round_trips_without_migration(self):
        """
        save_model_params writes the config verbatim, so a post-fix run's saved
        model_params_config.json already carries its targets and resumes without edits.
        """
        import json

        saved = json.loads(json.dumps(load_shipped_default_config()))

        assert resolve_schedule_targets(saved["LearningRateSchedule"]) == [
            LR_TARGET_MODEL,
            LR_TARGET_LATENT,
        ]
