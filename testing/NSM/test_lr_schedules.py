"""
Learning-rate schedules reach optimizer param groups by ``Target``, never by position.

``get_optimizer`` orders groups ``[latent, model...]``, so giving ``lr_schedules[i]`` to
``param_groups[i]`` swaps the two schedules (``docs/KNOWN_ISSUES.md`` § History 1).
"""

import copy
import json

import pytest
import torch

from NSM.train.utils import add_plain_lr_to_config
from NSM.utils import (
    LR_TARGET_LATENT,
    LR_TARGET_MODEL,
    StepLearningRateSchedule,
    WarmupLearningRateSchedule,
    adjust_learning_rate,
    get_learning_rate_schedules,
    get_optimizer,
    resolve_schedule_targets,
    save_model,
)

MODEL_LR = 0.01
LATENT_LR = 0.001


def make_config(targets=("model", "latent"), optimizer="Adam"):
    """
    One Step entry per role. ``targets`` sets entry 0's and entry 1's ``Target``; ``None``
    omits the key. Each LR stays with its role, so reordering ``targets`` reorders entries.
    """
    specs = {
        "model": {"Type": "Step", "Initial": MODEL_LR, "Interval": 10, "Factor": 0.5},
        "latent": {"Type": "Step", "Initial": LATENT_LR, "Interval": 10, "Factor": 0.5},
    }
    unclaimed = [role for role in ("model", "latent") if role not in targets]
    entries = []
    for target in targets:
        entry = dict(specs[target if target in specs else unclaimed.pop(0)])
        if target is not None:
            entry["Target"] = target
        entries.append(entry)
    return {"LearningRateSchedule": entries, "optimizer": optimizer}


def build(config, models=None, weight_decay=0.0):
    """Schedules and optimizer from a config, as the train loop builds them."""
    schedules = get_learning_rate_schedules(config)
    optimizer = get_optimizer(
        models if models is not None else torch.nn.Linear(4, 1),
        torch.nn.Embedding(3, 4),
        lr_schedules=schedules,
        optimizer=config["optimizer"],
        weight_decay=weight_decay,
    )
    return schedules, optimizer


def lrs_by_name(optimizer):
    return {group["name"]: group["lr"] for group in optimizer.param_groups}


class TestParamGroups:
    def test_groups_are_named_and_targeted(self):
        """
        Fails if ``get_optimizer`` changes the ``[latent, model_0, model_1]`` group order,
        drops a group's ``target``, or seeds a group's initial rate from the other schedule.

        The order is kept because ``load_state_dict`` matches a checkpoint's groups by
        position.
        """
        _, optimizer = build(make_config(), models=[torch.nn.Linear(4, 1) for _ in range(2)])
        groups = optimizer.param_groups
        assert [g["name"] for g in groups] == ["latent", "model_0", "model_1"]
        assert [g["target"] for g in groups] == [LR_TARGET_LATENT] + [LR_TARGET_MODEL] * 2
        assert lrs_by_name(optimizer) == {
            "latent": LATENT_LR,
            "model_0": MODEL_LR,
            "model_1": MODEL_LR,
        }

    @pytest.mark.parametrize("name", ["Adam", "AdamW"])
    def test_weight_decay_reaches_every_group(self, name):
        """Fails if ``get_optimizer``'s Adam or AdamW branch drops ``weight_decay`` (#47)."""
        _, optimizer = build(make_config(optimizer=name), weight_decay=0.123)
        assert [g["weight_decay"] for g in optimizer.param_groups] == [0.123, 0.123]


class TestScheduleMapping:
    @pytest.mark.parametrize("targets", [("model", "latent"), ("latent", "model")])
    def test_each_group_follows_its_targets_schedule(self, targets):
        """
        Fails if ``adjust_learning_rate`` picks a group's schedule by position or name, or
        ``get_learning_rate_schedules`` keys schedules by entry position, instead of by
        ``target`` (KNOWN_ISSUES History 1).

        Both entry orders run. The groups are reversed, one is renamed, and an extra
        ``model`` group is added, so only ``target`` identifies a group's schedule.
        """
        schedules, optimizer = build(
            make_config(targets), models=[torch.nn.Linear(4, 1) for _ in range(2)]
        )
        optimizer.add_param_group(
            {
                "name": "classification_heads",
                "target": LR_TARGET_MODEL,
                "params": torch.nn.Linear(4, 1).parameters(),
                "lr": 0.0,
            }
        )
        optimizer.param_groups.reverse()
        optimizer.param_groups[0]["name"] = "renamed"

        adjust_learning_rate(schedules, optimizer, epoch=10)  # one interval: factor 0.5
        by_target = {}
        for group in optimizer.param_groups:
            by_target.setdefault(group["target"], set()).add(group["lr"])
        assert by_target == {LR_TARGET_MODEL: {MODEL_LR * 0.5}, LR_TARGET_LATENT: {LATENT_LR * 0.5}}

    def test_a_group_without_a_target_raises(self):
        """
        Fails if ``adjust_learning_rate`` skips or defaults a param group that has no
        ``target`` instead of raising ``KeyError``.
        """
        schedules, optimizer = build(make_config())
        del optimizer.param_groups[0]["target"]
        with pytest.raises(KeyError, match="no known 'target'"):
            adjust_learning_rate(schedules, optimizer, epoch=1)


class TestMigrationGuard:
    """A config that does not declare ``Target`` on both entries raises. It never guesses."""

    @pytest.mark.parametrize(
        "targets, optimizer, match",
        [
            ((None, None), "Adam", "must declare 'Target'"),
            (("model", None), "Adam", "not migrated"),
            ((None, None), "schedule_free_AdamW", "must declare 'Target'"),
            (("model", "decoder"), "Adam", "exactly once each"),
            (("model", "model"), "Adam", "exactly once each"),
        ],
    )
    def test_an_unmigrated_config_raises(self, targets, optimizer, match):
        """
        Fails if ``get_learning_rate_schedules`` infers a target from entry position for an
        unannotated or half-annotated config, or accepts an unknown or repeated ``Target``,
        instead of raising ``ValueError``.

        The ``must declare`` and ``not migrated`` matches come from the migration message.
        Update them when ``NSM/_lr_migration.py`` is deleted.
        """
        with pytest.raises(ValueError, match=match):
            get_learning_rate_schedules(make_config(targets, optimizer))

    def test_one_entry_raises(self):
        """
        Fails if ``get_learning_rate_schedules`` accepts a one-entry ``LearningRateSchedule``,
        or refuses it without saying that exactly 2 entries are expected.
        """
        config = make_config()
        config["LearningRateSchedule"] = config["LearningRateSchedule"][:1]
        with pytest.raises(ValueError, match="exactly 2 LearningRateSchedule"):
            get_learning_rate_schedules(config)

    def test_the_message_prints_each_optimizers_historical_mapping(self):
        """
        Fails if ``migration_error`` gives AdamW and schedule_free configs the same historical
        mapping, omits the paste-ready annotated entries, or puts the ``CAUTION`` on the
        wrong family.

        The ``CAUTION`` is for schedule_free alone: those configs were usually tuned under
        Adam, so reproducing one may reproduce the mismatch. Delete this test with
        ``NSM/_lr_migration.py``.
        """
        messages = {}
        for optimizer in ("AdamW", "schedule_free_AdamW"):
            with pytest.raises(ValueError) as exc:
                get_learning_rate_schedules(make_config((None, None), optimizer))
            messages[optimizer] = str(exc.value)

        assert "entry 0 -> latent, entry 1 -> model" in messages["AdamW"]
        assert "entry 0 -> model, entry 1 -> latent" in messages["schedule_free_AdamW"]
        assert "CAUTION" in messages["schedule_free_AdamW"]
        assert "CAUTION" not in messages["AdamW"]
        assert '"Target": "latent"' in messages["AdamW"]


#: The real ShapeMedKnee_2024 schedules, in their original un-annotated order. The entries
#: differ in Interval and Factor, so a wrong annotation inverts the run.
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
    Annotating a pre-fix Adam/AdamW config with entry 0 -> latent and entry 1 -> model
    reproduces exactly the learning rates the buggy code gave it.
    """

    def test_the_historical_annotation_reproduces_the_pre_fix_rates(self):
        """
        Fails if annotating ShapeMedKnee_2024's schedules entry 0 -> latent, entry 1 -> model
        gives a group a different rate from the pre-fix positional code at any epoch from 1
        to 2000 (KNOWN_ISSUES History 1).
        """

        def pre_fix(index, epoch):
            spec = SHAPEMEDKNEE_2024_SPECS[index]
            return spec["Initial"] * spec["Factor"] ** (epoch // spec["Interval"])

        migrated = {
            "optimizer": "AdamW",
            "LearningRateSchedule": [
                dict(SHAPEMEDKNEE_2024_SPECS[0], Target=LR_TARGET_LATENT),
                dict(SHAPEMEDKNEE_2024_SPECS[1], Target=LR_TARGET_MODEL),
            ],
        }
        for epoch in (1, 5, 100, 500, 1000, 1500, 2000):
            schedules, optimizer = build(migrated)
            adjust_learning_rate(schedules, optimizer, epoch=epoch)
            lrs = lrs_by_name(optimizer)
            assert lrs["latent"] == pytest.approx(pre_fix(0, epoch))
            assert lrs["model_0"] == pytest.approx(pre_fix(1, epoch))

        # The opposite annotation is a different run: 50x apart at epoch 0.
        inverted = copy.deepcopy(migrated["LearningRateSchedule"])
        inverted[0]["Target"], inverted[1]["Target"] = LR_TARGET_MODEL, LR_TARGET_LATENT
        schedules = get_learning_rate_schedules({"LearningRateSchedule": inverted})
        assert schedules[LR_TARGET_MODEL].get_learning_rate(0) == pytest.approx(0.005)


class TestCheckpoints:
    def test_a_saved_optimizer_resumes_with_its_names_and_targets(self, tmp_path):
        """
        Fails if the optimizer state ``save_model`` writes loses a group's ``name`` or
        ``target``, so ``adjust_learning_rate`` cannot schedule the optimizer it restores.

        The checkpoint relies on torch's ``state_dict()`` keeping custom group keys, and
        saves no separate names key.
        """
        schedules, optimizer = build(make_config())
        save_model(
            {"experiment_directory": str(tmp_path)},
            epoch=1,
            decoder=torch.nn.Linear(4, 1),
            optimizer=optimizer,
        )
        checkpoint = torch.load(tmp_path / "model" / "1.pth", weights_only=False)

        _, resumed = build(make_config())
        resumed.load_state_dict(checkpoint["optimizer"])
        adjust_learning_rate(schedules, resumed, epoch=1)
        assert lrs_by_name(resumed) == {"latent": LATENT_LR, "model_0": MODEL_LR}

    def test_save_model_refuses_an_untargeted_group(self, tmp_path):
        """
        Fails if ``save_model`` writes a checkpoint from an optimizer whose param group has
        no ``target``, instead of raising ``ValueError``.
        """
        _, optimizer = build(make_config())
        del optimizer.param_groups[0]["target"]
        with pytest.raises(ValueError, match="must declare a 'target'"):
            save_model(
                {"experiment_directory": str(tmp_path)},
                epoch=1,
                decoder=torch.nn.Linear(4, 1),
                optimizer=optimizer,
            )

    def test_no_optimizer_is_saved_as_none(self, tmp_path):
        """
        Fails if ``save_model(optimizer=None)`` stores anything but ``None`` under
        ``"optimizer"``.

        The string ``"None"`` is truthy, so ``if checkpoint["optimizer"]:`` would pass.
        """
        save_model(
            {"experiment_directory": str(tmp_path)},
            epoch=1,
            decoder=torch.nn.Linear(4, 1),
            optimizer=None,
        )
        assert torch.load(tmp_path / "model" / "1.pth", weights_only=False)["optimizer"] is None


class TestLoggedLearningRates:
    """``add_plain_lr_to_config`` flattens the schedules into scalar keys for wandb."""

    @pytest.mark.parametrize("targets", [("model", "latent"), ("latent", "model")])
    def test_labels_follow_target_and_match_the_applied_rates(self, targets):
        """
        Fails if ``add_plain_lr_to_config`` labels the ``model_lr_*`` and ``latent_lr_*`` keys
        by entry position instead of ``Target``, so a logged rate is the other group's.
        """
        raw = make_config(targets)
        logged = add_plain_lr_to_config(copy.deepcopy(raw))
        schedules, optimizer = build(raw)
        adjust_learning_rate(schedules, optimizer, epoch=1)

        assert logged["model_lr_initial"] == lrs_by_name(optimizer)["model_0"] == MODEL_LR
        assert logged["latent_lr_initial"] == lrs_by_name(optimizer)["latent"] == LATENT_LR
        assert logged["model_lr_type"] == "Step"
        assert logged["model_lr_update_interval"] == 10
        assert logged["model_lr_update_factor"] == pytest.approx(0.5)

    def test_constant_entries_are_logged(self):
        """
        Fails if ``add_plain_lr_to_config`` raises ``KeyError`` on a Constant entry, or does
        not log its ``Value`` as ``*_lr_initial`` (#48).
        """
        config = add_plain_lr_to_config(
            {
                "LearningRateSchedule": [
                    {"Target": "model", "Type": "Constant", "Value": 0.005},
                    {"Target": "latent", "Type": "Constant", "Value": 0.001},
                ]
            }
        )
        assert (config["model_lr_initial"], config["latent_lr_initial"]) == (0.005, 0.001)
        assert config["model_lr_type"] == "Constant"


class TestScheduleTypes:
    def test_non_positive_intervals_and_lengths_refuse_by_name(self):
        """
        Fails if ``StepLearningRateSchedule`` or ``WarmupLearningRateSchedule`` accepts a
        non-positive ``interval`` or ``length`` instead of raising a ``ValueError`` that names
        the replacement (``Factor`` or ``Constant``), or computes a wrong decay or ramp.
        """
        for interval in (0, -1):
            with pytest.raises(ValueError, match="Factor"):
                StepLearningRateSchedule(initial=0.001, interval=interval, factor=0.5)
        for length in (0, -1):
            with pytest.raises(ValueError, match="Constant"):
                WarmupLearningRateSchedule(initial=0.0, warmed_up=0.001, length=length)

        step = StepLearningRateSchedule(initial=0.001, interval=500, factor=0.5)
        warm = WarmupLearningRateSchedule(initial=0.0, warmed_up=0.001, length=100)
        assert step.get_learning_rate(1000) == pytest.approx(0.00025)
        assert warm.get_learning_rate(50) == pytest.approx(0.0005)
        assert warm.get_learning_rate(500) == pytest.approx(0.001)


def test_the_shipped_default_config_declares_both_targets():
    """
    Fails if ``default_config.json`` or ``generate_sdf_default_config.config`` lacks a valid
    ``Target`` on either entry, or the shipped file stops giving the latents the larger
    initial LR (KNOWN_ISSUES History 1).

    The larger latent LR looks backwards and is deliberate: the shipped models trained under
    AdamW's historical mapping, and their values were tuned for it.
    """
    import NSM
    from NSM.configs.generate_sdf_default_config import config as generated

    path = f"{NSM.__path__[0]}/configs/default_config.json"
    with open(path, encoding="utf-8") as handle:
        shipped = json.load(handle)

    for config in (shipped, generated):
        entries = config["LearningRateSchedule"]
        assert resolve_schedule_targets(entries) == [e["Target"] for e in entries]
        assert sorted(e["Target"] for e in entries) == [LR_TARGET_LATENT, LR_TARGET_MODEL]

    schedules = get_learning_rate_schedules(shipped)
    assert schedules[LR_TARGET_LATENT].get_learning_rate(0) > schedules[
        LR_TARGET_MODEL
    ].get_learning_rate(0)
