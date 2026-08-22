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
    resolve_schedule_targets,
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

    def test_groups_declare_their_target(self):
        _, optimizer = build(make_config(), models=[make_model(), make_model()])

        assert [g["target"] for g in optimizer.param_groups] == [
            LR_TARGET_LATENT,
            LR_TARGET_MODEL,
            LR_TARGET_MODEL,
        ]


class TestConstantScheduleLogging:
    """#48: ``get_learning_rate_schedules`` accepts Constant entries, but the logging
    helper read ``Initial`` unconditionally and raised ``KeyError('Initial')`` on them.
    """

    def test_constant_entries_do_not_crash_the_logging_helper(self):
        config = make_config()
        config["LearningRateSchedule"] = [
            {"Target": "model", "Type": "Constant", "Value": 0.005},
            {"Target": "latent", "Type": "Constant", "Value": 0.001},
        ]
        config = add_plain_lr_to_config(config)
        assert config["model_lr_initial"] == 0.005
        assert config["latent_lr_initial"] == 0.001
        assert config["model_lr_type"] == "Constant"


class TestWeightDecayIsForwarded:
    """#47: the Adam branch dropped ``weight_decay`` while AdamW passed it.

    torch stamps the constructor's ``weight_decay`` into every param group, which is
    where a silent drop is observable. ``schedule_free_AdamW`` always forwarded the
    argument and is not constructible here (schedulefree is absent from the dev env).
    """

    @pytest.mark.parametrize("name", ["Adam", "AdamW"])
    def test_every_group_carries_the_configured_decay(self, name):
        config = make_config()
        schedules = get_learning_rate_schedules(config)
        optimizer = get_optimizer(
            make_model(),
            make_latents(),
            lr_schedules=schedules,
            optimizer=name,
            weight_decay=0.123,
        )
        assert [g["weight_decay"] for g in optimizer.param_groups] == [0.123, 0.123]


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
            {
                "name": "classification_heads",
                "target": LR_TARGET_MODEL,
                "params": make_model().parameters(),
                "lr": 0.0,
            }
        )
        adjust_learning_rate(schedules, optimizer, epoch=1)

        assert lrs_by_name(optimizer)["classification_heads"] == pytest.approx(MODEL_LR)

    def test_group_without_a_target_raises(self):
        schedules, optimizer = build(make_config())
        del optimizer.param_groups[0]["target"]

        with pytest.raises(KeyError, match="no known 'target'"):
            adjust_learning_rate(schedules, optimizer, epoch=1)

    def test_name_is_a_label_and_not_load_bearing(self):
        """Renaming a group must not change which schedule drives it."""
        schedules, optimizer = build(make_config())
        for group in optimizer.param_groups:
            group["name"] = "something_else"
        adjust_learning_rate(schedules, optimizer, epoch=1)

        by_target = {g["target"]: g["lr"] for g in optimizer.param_groups}
        assert by_target[LR_TARGET_MODEL] == pytest.approx(MODEL_LR)
        assert by_target[LR_TARGET_LATENT] == pytest.approx(LATENT_LR)


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
            assert schedules[LR_TARGET_MODEL].get_learning_rate(0) == pytest.approx(MODEL_LR)
            assert schedules[LR_TARGET_LATENT].get_learning_rate(0) == pytest.approx(LATENT_LR)

    def test_schedules_are_returned_keyed_by_target(self):
        """A mapping, not a list -- there is no index for a caller to get wrong."""
        schedules = get_learning_rate_schedules(make_config())

        assert set(schedules) == {LR_TARGET_MODEL, LR_TARGET_LATENT}

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
        schedules = get_learning_rate_schedules(config)

        # At epoch 1000 the model has taken one x0.1 step; the latent has decayed smoothly.
        assert schedules[LR_TARGET_MODEL].get_learning_rate(1000) == pytest.approx(0.0001 * 0.1)
        assert schedules[LR_TARGET_LATENT].get_learning_rate(1000) == pytest.approx(
            0.005 * 0.95 ** (1000 // 16.7)
        )


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
        """A typo and a duplicate share one check; the message shows what was given."""
        with pytest.raises(ValueError, match="exactly once each") as exc:
            get_learning_rate_schedules(make_config(targets=("model", "decoder")))

        assert "'decoder'" in str(exc.value)

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

    def test_schedule_free_error_warns_that_reproducing_may_be_wrong(self):
        """
        A schedule_free config's values were almost always tuned under Adam, where the
        mapping was the opposite -- so faithfully reproducing the run reproduces the
        mismatch. The message has to say so.
        """
        with pytest.raises(ValueError) as exc:
            get_learning_rate_schedules(
                make_config(targets=(None, None), optimizer="schedule_free_AdamW")
            )

        message = str(exc.value)
        assert "CAUTION" in message
        assert "may not be what you want" in message

    def test_adam_error_has_no_schedule_free_caution(self):
        with pytest.raises(ValueError) as exc:
            get_learning_rate_schedules(make_config(targets=(None, None)))

        assert "CAUTION" not in str(exc.value)

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
        assert historical[LR_TARGET_MODEL].get_learning_rate(0) == pytest.approx(0.0001)
        assert inverted[LR_TARGET_MODEL].get_learning_rate(0) == pytest.approx(0.005)


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
        Why resuming a pre-fix checkpoint is refused: load_state_dict adopts the
        CHECKPOINT's param-group metadata, so a checkpoint with no names strips them off a
        freshly built, correctly named optimizer. Group identity is unrecoverable.
        """
        _, optimizer = build(make_config())
        old_state = copy.deepcopy(optimizer.state_dict())
        for group in old_state["param_groups"]:
            group.pop("name", None)

        _, fresh = build(make_config())
        fresh.load_state_dict(old_state)

        assert all(g.get("name") is None for g in fresh.param_groups)

    def test_resume_from_post_fix_checkpoint_applies_correct_lrs(self):
        """End-to-end: a resume that IS supported must not resurrect the swap."""
        schedules, optimizer = build(make_config())

        _, fresh = build(make_config())
        fresh.load_state_dict(optimizer.state_dict())
        adjust_learning_rate(schedules, fresh, epoch=1)

        lrs = lrs_by_name(fresh)
        assert lrs["model_0"] == pytest.approx(MODEL_LR)
        assert lrs["latent"] == pytest.approx(LATENT_LR)

    def test_untargeted_groups_are_rejected(self):
        """
        The train loop refuses a pre-fix checkpoint at load time. adjust_learning_rate is
        only the backstop -- it is skipped for schedule_free_*, which would otherwise run
        to the first checkpoint save before failing.
        """
        schedules, optimizer = build(make_config())
        for group in optimizer.param_groups:
            group.pop("target")

        with pytest.raises(KeyError, match="no known 'target'"):
            adjust_learning_rate(schedules, optimizer, epoch=1)


class TestSaveModel:
    def test_saved_optimizer_state_carries_group_names(self, tmp_path):
        """
        No separate names key is stored: state_dict() retains 'name' itself, which is what
        makes the round-trip work.
        """
        _, optimizer = build(make_config())
        config = {"experiment_directory": str(tmp_path)}

        save_model(config, epoch=1, decoder=make_model(), optimizer=optimizer)

        checkpoint = torch.load(tmp_path / "model" / "1.pth", weights_only=False)
        assert "optimizer_group_names" not in checkpoint
        assert [g["name"] for g in checkpoint["optimizer"]["param_groups"]] == [
            "latent",
            "model_0",
        ]

    def test_save_model_rejects_untargeted_groups(self, tmp_path):
        _, optimizer = build(make_config())
        del optimizer.param_groups[0]["target"]
        config = {"experiment_directory": str(tmp_path)}

        with pytest.raises(ValueError, match="must declare a 'target'"):
            save_model(config, epoch=1, decoder=make_model(), optimizer=optimizer)

    def test_save_model_without_optimizer_writes_real_none(self, tmp_path):
        """
        None, not the string "None". The string is truthy, so the natural presence check
        `if checkpoint["optimizer"]:` would pass and then feed a str to load_state_dict.
        """
        config = {"experiment_directory": str(tmp_path)}
        save_model(config, epoch=1, decoder=make_model(), optimizer=None)

        checkpoint = torch.load(tmp_path / "model" / "1.pth", weights_only=False)
        assert checkpoint["optimizer"] is None
        assert not checkpoint["optimizer"]


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

        # The 647-derived default puts the larger LR on the LATENTS. Deliberate, and
        # worth pinning because it looks backwards: that is what the shipped models
        # actually trained under (History §1 — AdamW's entry 0 historically drove the
        # latents), and their hyperparameters were tuned for exactly that mapping.
        model_lr = schedules[LR_TARGET_MODEL].get_learning_rate(0)
        latent_lr = schedules[LR_TARGET_LATENT].get_learning_rate(0)
        assert latent_lr > model_lr

    def test_generated_default_config_annotates_targets(self):
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

        # Resolution must honour each entry's own declaration, whatever order the
        # shipped file happens to list them in.
        assert resolve_schedule_targets(saved["LearningRateSchedule"]) == [
            entry["Target"] for entry in saved["LearningRateSchedule"]
        ]
        assert sorted(resolve_schedule_targets(saved["LearningRateSchedule"])) == [
            LR_TARGET_LATENT,
            LR_TARGET_MODEL,
        ]
