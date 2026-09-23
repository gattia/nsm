"""
``NSM/utils.py`` apart from the LR path (``test_lr_schedules.py``): the
``model_params_config.json`` record a run writes, and the helpers around it.
"""

import copy
import json
import logging
import pathlib
import re

import pytest
import torch

from NSM.train.utils import add_plain_lr_to_config
from NSM.utils import (
    ConstantLearningRateSchedule,
    LearningRateSchedule,
    LogAnnealLearningRateSchedule,
    StepLearningRateSchedule,
    WarmupLearningRateSchedule,
    clear_gpu_cache,
    filter_non_jsonable,
    get_checkpoints,
    get_learning_rate_schedules,
    is_jsonable,
    save_model_params,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "NSM" / "configs" / "default_config.json"
SUBJECTS = ["/data/subj001_bone.vtk", "/data/subj002_bone.vtk"]


@pytest.fixture
def shipped_config(tmp_path):
    """``default_config.json`` verbatim, pointed at a scratch experiment directory."""
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["experiment_directory"] = str(tmp_path / "experiment")
    return config


@pytest.fixture
def running_config(shipped_config):
    """``shipped_config`` plus the three keys ``train_deep_sdf`` writes back into it."""
    config = add_plain_lr_to_config(shipped_config)
    config["checkpoints"] = get_checkpoints(config)
    config["lr_schedules"] = get_learning_rate_schedules(config)
    return config


def saved_record(config):
    path = pathlib.Path(config["experiment_directory"]) / "model_params_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


class TestTheRecordNamesItsSubjects:
    """
    ``save_model_params``' ``list_mesh_paths`` argument is the live dataset's paths. A
    config key of the same name used to overwrite it (``KNOWN_ISSUES.md`` § History 26).
    """

    def test_the_argument_wins_over_the_shipped_null(self, shipped_config):
        assert shipped_config["list_mesh_paths"] is None, "the shipped default changed"
        save_model_params(config=shipped_config, list_mesh_paths=SUBJECTS)
        assert saved_record(shipped_config)["list_mesh_paths"] == SUBJECTS

        bare = {"experiment_directory": shipped_config["experiment_directory"] + "_bare"}
        save_model_params(config=bare, list_mesh_paths=SUBJECTS)
        assert saved_record(bare)["list_mesh_paths"] == SUBJECTS

    def test_a_previous_runs_list_is_replaced_and_reported(self, shipped_config, caplog):
        """Re-training from a saved config used to record the previous run's subjects."""
        shipped_config["list_mesh_paths"] = ["/data/PREVIOUS_RUN.vtk"]
        with caplog.at_level(logging.WARNING, logger="NSM.utils"):
            save_model_params(config=shipped_config, list_mesh_paths=SUBJECTS)
        assert saved_record(shipped_config)["list_mesh_paths"] == SUBJECTS
        assert "model_params_config.json" in caplog.text


class TestWriteOnce:
    """
    The first write wins: the file records the configuration that produced the
    checkpoints. A later call with a different config names what differs (#50).
    """

    def test_a_diverging_second_call_names_exactly_the_keys_that_differ(
        self, running_config, caplog
    ):
        save_model_params(config=running_config, list_mesh_paths=SUBJECTS)
        caplog.clear()

        diverged = copy.copy(running_config)
        diverged.update({"latent_size": 999, "n_epochs": 7, "objects_per_decoder": 4})
        with caplog.at_level(logging.WARNING, logger="NSM.utils"):
            save_model_params(config=diverged, list_mesh_paths=SUBJECTS)

        assert saved_record(running_config)["latent_size"] == running_config["latent_size"]
        # Single-word keys (device, seed, padding) could appear in any wording of the
        # message, so only multi-token keys are searched for.
        named = {
            key
            for key in running_config
            if "_" in key and re.search(rf"\b{re.escape(key)}\b", caplog.text)
        }
        assert named == {"latent_size", "n_epochs", "objects_per_decoder"}

    def test_later_checkpoints_of_a_healthy_run_say_nothing(self, running_config, caplog):
        """
        The first write reports the dropped ``lr_schedules`` once. Every later checkpoint
        writes the same config and must stay quiet.
        """
        save_model_params(config=running_config, list_mesh_paths=SUBJECTS)
        assert "lr_schedules" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="NSM.utils"):
            for _ in range(3):
                save_model_params(config=running_config, list_mesh_paths=SUBJECTS)
        assert caplog.records == []


class TestNonSerialisableValues:
    def test_a_real_run_drops_only_the_schedule_objects(self, running_config):
        """A dropped key is absent from the record, not ``null``."""
        assert {k for k, v in running_config.items() if not is_jsonable(v)} == {"lr_schedules"}
        save_model_params(config=running_config, list_mesh_paths=SUBJECTS)
        assert "lr_schedules" not in saved_record(running_config)

    def test_the_filter_is_shallow_and_a_cycle_is_not_jsonable(self):
        """One bad leaf drops the whole nested value. A cycle answers False, not raises."""
        nested = {"keep": 1, "drop": {"fine": 2, "not_fine": object()}}
        assert filter_non_jsonable(nested) == {"keep": 1}

        cyclic = {}
        cyclic["self"] = cyclic
        assert is_jsonable(cyclic) is False


class TestScheduleClasses:
    def test_a_subclass_that_forgets_to_override_refuses(self):
        """The base body was ``pass``, so the ``None`` failed later inside ``torch.optim``."""

        class Forgot(LearningRateSchedule):
            pass

        with pytest.raises(NotImplementedError, match="Forgot"):
            Forgot().get_learning_rate(0)

        for schedule in (
            ConstantLearningRateSchedule(1e-3),
            StepLearningRateSchedule(1e-3, 500, 0.5),
            WarmupLearningRateSchedule(1e-5, 1e-3, 100),
            LogAnnealLearningRateSchedule(1e-3, 1e-5, 1000),
        ):
            assert isinstance(schedule.get_learning_rate(0), float)

    def test_two_entry_keys_that_do_not_match_their_parameter(self):
        """
        Warmup's ``Final`` feeds a parameter called ``warmed_up``. LogAnneal ignores any
        ``Length`` on its entry and takes its horizon from the top-level ``n_epochs``.
        """
        config = {
            "n_epochs": 100,
            "LearningRateSchedule": [
                {"Target": "model", "Type": "Warmup", "Initial": 0.0, "Final": 0.5, "Length": 10},
                {
                    "Target": "latent",
                    "Type": "LogAnneal",
                    "Initial": 1e-2,
                    "Final": 1e-4,
                    "Length": 7,
                },
            ],
        }
        schedules = get_learning_rate_schedules(config)
        assert schedules["model"].get_learning_rate(10) == 0.5
        assert schedules["latent"].n_epochs == 100
        assert schedules["latent"].get_learning_rate(100) == pytest.approx(1e-4)


def test_clear_gpu_cache_accepts_a_string_or_a_device():
    """Configs hold ``"cpu"``; a Python caller holds ``torch.device``, which used to raise."""
    for device in ("cpu", torch.device("cpu")):
        with pytest.warns(UserWarning, match="Not clearing cache"):
            clear_gpu_cache(device)


class TestCheckpointList:
    def test_a_repeated_checkpoint_is_listed_twice_and_changes_nothing(self):
        """
        Both trainer reads are membership tests, so the duplicate cannot save a checkpoint
        twice. Deduplicating would change a value recorded in ``model_params_config.json``.
        """
        config = {"checkpoint_epochs": 10, "n_epochs": 30, "additional_checkpoints": [10, 25]}
        checkpoints = get_checkpoints(config)
        assert checkpoints == [10, 10, 20, 25, 30]

    def test_a_missing_additional_checkpoints_names_the_remedy(self):
        with pytest.raises(KeyError, match=r"additional_checkpoints.*\[\]"):
            get_checkpoints({"checkpoint_epochs": 10, "n_epochs": 30})
