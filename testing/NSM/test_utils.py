"""
What ``NSM/utils.py`` promises about the record a run leaves on disk, and about the
handful of helpers around it.

Plan §8.0.M. The LR path has its own file (``test_lr_schedules.py``); everything else in
the module is here, and until this file existed that was everything else in the module.

Three of the seven contracts are one shape -- ``model_params_config.json`` is assembled
from a dict built in the wrong order, and nothing reports what the assembly discarded:

1. **``save_model_params``' ``list_mesh_paths`` argument loses to the config key of the
   same name.** ``dict_save.update(config)`` runs after the argument is placed, and the
   shipped ``default_config.json`` carries ``"list_mesh_paths": null``, so a default run
   records no mesh list at all.
2. **The write-once refusal is silent** (#50). Deliberate provenance protection, but a
   re-configured run in the same directory leaves a file describing a different model from
   the checkpoints beside it, and says nothing.
3. **``filter_non_jsonable`` drops keys with no account of them** (#50). Measured on a real
   run's config the drop set is exactly ``{"lr_schedules"}``, which is what decides where
   the log belongs: at the write, once per run, not on every checkpoint.

The other four are independent:

4. **``LearningRateSchedule.get_learning_rate`` returns ``None``** -- the base body is
   ``pass``, so a subclass that forgets to override it fails inside ``torch.optim``.
5. **``print_gpu_memory`` prints nothing and has no caller** -- §8.0.G converted its
   ``print`` calls and left the name.
6. **``clear_gpu_cache`` raises on a ``torch.device``**, which is the form a Python caller
   holds; both in-repo callers pass the JSON string.
7. **``is_jsonable`` does not catch the error a cycle raises**, so a predicate that
   answers True or False propagates ``ValueError`` instead.

Strict xfails mark what NSM does not honour yet. Each is retired by the commit that fixes
it. ``get_checkpoints``' duplicate entry is characterized rather than xfailed: the statement
rules it inert and out of the slice, and this file is where that ruling is checked.
"""

import copy
import json
import logging
import pathlib
import re

import pytest
import torch

import NSM.utils
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
    get_latent_vecs,
    get_learning_rate_schedules,
    get_optimizer,
    is_jsonable,
    save_model_params,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: The config a run actually starts from. Read rather than hand-written, so a change to
#: the shipped defaults reaches these assertions instead of going unnoticed beside them.
DEFAULT_CONFIG_PATH = REPO_ROOT / "NSM" / "configs" / "default_config.json"

SUBJECTS = ["/data/subj001_bone.vtk", "/data/subj002_bone.vtk"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def shipped_config(tmp_path):
    """``default_config.json`` verbatim, pointed at a scratch experiment directory."""
    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["experiment_directory"] = str(tmp_path / "experiment")
    return config


@pytest.fixture
def running_config(shipped_config):
    """
    ``shipped_config`` plus the three keys ``train_deep_sdf`` writes back into it.

    Lines 121-123 of the trainer, in order: the flattened logging LRs, the checkpoint
    list, and the schedule objects. The third is the only value in a real run that
    ``filter_non_jsonable`` has anything to do.
    """
    config = add_plain_lr_to_config(shipped_config)
    config["checkpoints"] = get_checkpoints(config)
    config["lr_schedules"] = get_learning_rate_schedules(config)
    return config


def saved_record(config):
    """The ``model_params_config.json`` a config's experiment directory holds."""
    path = pathlib.Path(config["experiment_directory"]) / "model_params_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Which subjects trained the model
# ---------------------------------------------------------------------------


class TestTheRecordNamesItsSubjects:
    """
    ``list_mesh_paths`` is the argument, and the argument is the live dataset's paths.

    ``ARCHITECTURE`` §7's "parameter accepted and silently ignored" in a variant the row
    does not have: the shadow is not a local rebind or a ``**kwargs`` hole but a config key
    with the same name. It is also the case where the class's usual remedy inverts --
    ``CLAUDE.md`` says delete the ignored parameter, and here the parameter is the
    authoritative source and the config's copy is the stale artifact.
    """

    def test_the_shipped_default_records_the_subjects_it_was_given(self, shipped_config):
        """
        Was a strict xfail, and the measurement that made this the slice's first defect
        rather than #50's footnote.

        ``default_config.json`` carries ``"list_mesh_paths": null``. Every run started from
        it records ``null`` -- not a stale list, no list -- in the file ``load_model``,
        ``examples/load_trained_model.py`` and both consumer scripts read.
        """
        assert shipped_config["list_mesh_paths"] is None, "the shipped default changed"

        save_model_params(config=shipped_config, list_mesh_paths=SUBJECTS)

        assert saved_record(shipped_config)["list_mesh_paths"] == SUBJECTS

    def test_a_config_from_a_previous_run_does_not_win(self, shipped_config, caplog):
        """
        Was a strict xfail. The second reachable case, and the one that recorded a
        plausible wrong answer rather than an obviously empty one.

        A real ``model_params_config.json`` carries this key --
        ``generate_sdf_default_config.py``'s header lists it among the machine paths it had
        to sanitize out of the ``647_nsm_femur_v0.0.1`` config -- so re-training from a
        saved config records the previous run's subjects against this run's checkpoints.
        """
        shipped_config["list_mesh_paths"] = ["/data/PREVIOUS_RUN.vtk"]

        with caplog.at_level(logging.WARNING, logger="NSM.utils"):
            save_model_params(config=shipped_config, list_mesh_paths=SUBJECTS)

        assert saved_record(shipped_config)["list_mesh_paths"] == SUBJECTS
        assert "model_params_config.json" in caplog.text

    def test_a_config_without_the_key_records_the_argument(self, tmp_path):
        """The control: with no shadow there is no defect, and that must stay true."""
        config = {"experiment_directory": str(tmp_path / "experiment")}

        save_model_params(config=config, list_mesh_paths=SUBJECTS)

        assert saved_record(config)["list_mesh_paths"] == SUBJECTS

    def test_no_regression_test_reads_the_saved_value(self):
        """
        Why this went unnoticed: the three assertions that read the file check other keys.

        ``test_training_regression`` checks the LR targets and ``mesh_names``;
        ``test_model_roundtrip`` checks that the architecture keys are present -- which
        ``null`` satisfies, because the key *is* present. Delete this test when a
        regression test starts reading ``list_mesh_paths``; until then it records that the
        coverage which exists is coverage of something else.
        """
        regression = REPO_ROOT / "testing" / "NSM" / "regression"
        readers = [
            path
            for path in regression.glob("test_*.py")
            if "model_params_config.json" in path.read_text(encoding="utf-8")
        ]

        assert sorted(p.name for p in readers) == [
            "test_model_roundtrip.py",
            "test_training_regression.py",
        ]
        assert not any("list_mesh_paths" in p.read_text(encoding="utf-8") for p in readers)


# ---------------------------------------------------------------------------
# 2. Write-once
# ---------------------------------------------------------------------------


class TestWriteOnce:
    """
    First write wins, and #50 asks whether that is provenance protection or a bug.

    §8.0.M rules it protection: the file records the configuration that produced the
    checkpoints, and overwriting it on every checkpoint of a resumed run would replace that
    record with a description of a run that only partly happened. What is a bug is the
    silence, and that is what changes.
    """

    def test_the_second_call_does_not_overwrite(self, tmp_path):
        """The contract that stays. ``_save_checkpoint`` calls this on every checkpoint."""
        config = {"experiment_directory": str(tmp_path / "experiment"), "lr": 0.001}
        save_model_params(config=config, list_mesh_paths=SUBJECTS)

        diverged = dict(config, lr=0.9999)
        save_model_params(config=diverged, list_mesh_paths=SUBJECTS)

        assert saved_record(config)["lr"] == 0.001

    def test_the_refusal_names_what_diverges_from_it(self, tmp_path, caplog):
        """
        Was a strict xfail. Measured before the fix: ``lr=0.9999`` left ``0.001`` on disk
        and logged nothing.

        The learning rate is the harmless version. ``load_model`` rebuilds the architecture
        from this file, so the key that matters is one like ``latent_size``: the refusal has
        to name whichever key it is.
        """
        config = {"experiment_directory": str(tmp_path / "experiment"), "lr": 0.001}
        save_model_params(config=config, list_mesh_paths=SUBJECTS)

        diverged = dict(config, lr=0.9999)
        with caplog.at_level(logging.WARNING, logger="NSM.utils"):
            save_model_params(config=diverged, list_mesh_paths=SUBJECTS)

        assert any(record.levelno == logging.WARNING for record in caplog.records)
        assert "lr" in caplog.text

    def test_the_report_names_every_diverging_key_and_no_other(self, running_config, caplog):
        """
        Was a strict xfail. Three diverging keys out of the shipped config's 123: the report is a set, not a
        sample, and it must not sweep in the 120 that agree.

        Only multi-token keys are searched for. Sixteen of the 123 are ordinary words --
        ``device``, ``cache``, ``seed``, ``padding``, ``verbose`` -- which any wording of
        the message could contain innocently, so scanning for those would assert the prose
        rather than the report.
        """
        save_model_params(config=running_config, list_mesh_paths=SUBJECTS)
        caplog.clear()  # the first write reports its own dropped keys; this is the second

        diverged = copy.copy(running_config)
        diverged.update({"latent_size": 999, "n_epochs": 7, "objects_per_decoder": 4})
        with caplog.at_level(logging.WARNING, logger="NSM.utils"):
            save_model_params(config=diverged, list_mesh_paths=SUBJECTS)

        named = {
            key
            for key in running_config
            if "_" in key and re.search(rf"\b{re.escape(key)}\b", caplog.text)
        }
        assert named == {"latent_size", "n_epochs", "objects_per_decoder"}

    def test_an_agreeing_second_call_says_nothing(self, running_config, caplog):
        """
        The quiet case, and the reason the report is a diff rather than a notice: every
        checkpoint after the first calls this with the same config, and a warning that
        fires on every healthy run is one people learn to skip past.
        """
        save_model_params(config=running_config, list_mesh_paths=SUBJECTS)
        caplog.clear()  # the first write reports its own dropped keys; this is the second

        with caplog.at_level(logging.WARNING, logger="NSM.utils"):
            save_model_params(config=running_config, list_mesh_paths=SUBJECTS)

        assert caplog.records == []


# ---------------------------------------------------------------------------
# 3. Values the record cannot hold
# ---------------------------------------------------------------------------


class TestNonSerialisableValues:
    """``filter_non_jsonable`` discards, and #50 asks it to say what."""

    def test_a_real_runs_config_drops_exactly_the_schedule_objects(self, running_config):
        """
        The measurement that decides where the log belongs.

        The reported symptom is real, but its production instance is benign and singular:
        of the 123 keys a run carries, the only one that cannot be serialised is
        ``lr_schedules``, the dict of schedule objects the trainer writes back into the
        caller's config at ``train_deep_sdf:123`` -- and its source, ``LearningRateSchedule``,
        is on disk regardless. So the log rides with the write, once per run, and a warning
        on every checkpoint would be noise around a key nobody set.
        """
        dropped = {key for key, value in running_config.items() if not is_jsonable(value)}

        assert dropped == {"lr_schedules"}

    def test_a_nested_dict_is_dropped_whole_for_one_bad_leaf(self):
        """
        The filter is shallow: ``is_jsonable`` runs on the whole value, so one object in a
        nested dict takes every serialisable sibling with it. Documented rather than fixed
        -- a deep filter would write a value the caller never set.
        """
        nested = {"keep": 1, "drop": {"fine": 2, "also_fine": 3, "not_fine": object()}}

        assert filter_non_jsonable(nested) == {"keep": 1}

    def test_the_dropped_keys_are_named_where_the_record_is_written(self, running_config, caplog):
        """Was a strict xfail: the keys left the record with nothing said about them."""
        with caplog.at_level(logging.WARNING, logger="NSM.utils"):
            save_model_params(config=running_config, list_mesh_paths=SUBJECTS)

        assert "lr_schedules" in caplog.text

    def test_the_drop_is_reported_once_per_run_and_not_once_per_checkpoint(
        self, running_config, caplog
    ):
        """
        Why the log rides with the write. ``_save_checkpoint`` calls this on every
        checkpoint and the drop set is fixed by the config, so reporting it at the filter
        would put a warning naming ``lr_schedules`` -- a key the trainer inserted itself --
        into every healthy run, once per checkpoint.
        """
        save_model_params(config=running_config, list_mesh_paths=SUBJECTS)
        assert "lr_schedules" in caplog.text, "the write should have reported the drop"
        caplog.clear()

        with caplog.at_level(logging.WARNING, logger="NSM.utils"):
            for _ in range(3):
                save_model_params(config=running_config, list_mesh_paths=SUBJECTS)

        assert caplog.records == []

    def test_a_dropped_key_is_absent_from_the_record_rather_than_null(self, running_config):
        """
        Not the same thing as ``list_mesh_paths``' ``null``, and worth keeping apart: a
        dropped key leaves no entry at all, so a consumer's ``in`` test is the honest check
        and ``.get()`` cannot confuse "unserialisable" with "set to nothing".
        """
        save_model_params(config=running_config, list_mesh_paths=SUBJECTS)

        assert "lr_schedules" not in saved_record(running_config)

    def test_a_cycle_is_not_jsonable_rather_than_an_exception(self):
        """
        Was a strict xfail. A predicate answers True or False; this one used to propagate
        ``ValueError: Circular reference detected`` out of the checkpoint write.
        """
        cyclic = {}
        cyclic["self"] = cyclic

        assert is_jsonable(cyclic) is False


# ---------------------------------------------------------------------------
# 4. The schedule base class
# ---------------------------------------------------------------------------


class TestTheScheduleBaseClass:
    """
    The LR path's founding shape at another site: a value travels one function too far
    before anything looks at it.
    """

    def test_a_subclass_that_forgets_to_override_refuses(self):
        """Was a strict xfail: the base body was ``pass``, so it returned ``None``."""

        class Forgot(LearningRateSchedule):
            pass

        with pytest.raises(NotImplementedError, match="Forgot"):
            Forgot().get_learning_rate(0)

    def test_the_none_it_returns_today_reaches_the_optimizer(self):
        """
        Where the failure lands without the refusal, and why the message is useless:
        ``TypeError: unsupported operand type(s) for /: 'NoneType' and 'float'``, naming
        neither the schedule class, nor the param group, nor the config entry behind it.

        This asserts the *consequence*, not the defect, so it survives the fix: with the
        refusal in place a schedule can no longer hand ``None`` to a param group, and the
        only way to get one there is the manual assignment below.
        """
        model, latents = torch.nn.Linear(3, 1), get_latent_vecs(
            2, {"latent_size": 4, "latent_bound": 1}
        )
        schedules = {
            "model": ConstantLearningRateSchedule(1e-3),
            "latent": ConstantLearningRateSchedule(1e-3),
        }
        optimizer = get_optimizer(model, latents, schedules)
        for group in optimizer.param_groups:
            group["lr"] = None

        model(torch.zeros(1, 3)).sum().backward()
        with pytest.raises(TypeError):
            optimizer.step()

    @pytest.mark.parametrize(
        "schedule",
        [
            ConstantLearningRateSchedule(1e-3),
            StepLearningRateSchedule(1e-3, 500, 0.5),
            WarmupLearningRateSchedule(1e-5, 1e-3, 100),
            LogAnnealLearningRateSchedule(1e-3, 1e-5, 1000),
        ],
        ids=["constant", "step", "warmup", "loganneal"],
    )
    def test_every_shipped_subclass_returns_a_number(self, schedule):
        """The plain half of the pin: the refusal must not reach a real schedule."""
        assert isinstance(schedule.get_learning_rate(0), float)

    def test_warmup_reads_the_entrys_final_into_a_parameter_called_warmed_up(self):
        """
        A config-key trap the docstrings have to name: the entry says ``Final`` and the
        constructor parameter is ``warmed_up``, so the two vocabularies for one value meet
        only inside ``get_learning_rate_schedules``.
        """
        config = {
            "LearningRateSchedule": [
                {"Target": "model", "Type": "Warmup", "Initial": 0.0, "Final": 0.5, "Length": 10},
                {"Target": "latent", "Type": "Constant", "Value": 1e-3},
            ]
        }

        assert get_learning_rate_schedules(config)["model"].get_learning_rate(10) == 0.5

    def test_loganneal_takes_its_horizon_from_n_epochs_and_not_from_its_entry(self):
        """
        The second trap: every other type is built from its own entry, and ``LogAnneal``
        reaches past it for the top-level ``n_epochs``. A ``Length`` on the entry is read by
        nothing.
        """
        config = {
            "n_epochs": 100,
            "LearningRateSchedule": [
                {
                    "Target": "model",
                    "Type": "LogAnneal",
                    "Initial": 1e-2,
                    "Final": 1e-4,
                    "Length": 7,
                },
                {"Target": "latent", "Type": "Constant", "Value": 1e-3},
            ],
        }

        schedule = get_learning_rate_schedules(config)["model"]

        assert schedule.n_epochs == 100
        assert schedule.get_learning_rate(100) == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# 5-6. The GPU helpers
# ---------------------------------------------------------------------------


class TestGpuHelpers:
    def test_print_gpu_memory_is_gone(self):
        """
        §8.0.G converted its four ``print`` calls to ``logger.info`` and left the name.
        Before the deletion there was one occurrence in the code and the docs, its own
        ``def``, which is what made the fix a deletion rather than a rename. Kept after it
        so a later import cannot quietly bring the name back. ``testing/`` is outside the
        sweep because this file names the symbol in prose; the frozen-name list below is
        what covers the suite.
        """
        searched = [REPO_ROOT / d for d in ("NSM", "examples", "docs")]
        hits = [
            f"{path.relative_to(REPO_ROOT)}"
            for root in searched
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".md"}
            and "print_gpu_memory" in path.read_text(encoding="utf-8")
        ]

        assert hits == []

    def test_clear_gpu_cache_no_ops_on_a_cpu_string(self):
        """Both in-repo callers pass ``config["device"]``, which comes from JSON."""
        with pytest.warns(UserWarning, match="Not clearing cache"):
            clear_gpu_cache("cpu")

    def test_clear_gpu_cache_accepts_a_torch_device(self):
        """
        Was a strict xfail: ``"cuda" in device`` needs a str, so the object form raised
        ``TypeError: argument of type 'torch.device' is not iterable``. This is the form a
        Python caller holds, as opposed to the form JSON produces.
        """
        with pytest.warns(UserWarning, match="Not clearing cache"):
            clear_gpu_cache(torch.device("cpu"))


# ---------------------------------------------------------------------------
# 7. Ruled inert: the duplicate checkpoint
# ---------------------------------------------------------------------------


class TestCheckpointList:
    """
    §8.0.M rules the duplicate out of the slice, and this is where that ruling is checked
    rather than asserted in prose. Deduplicating would change a value recorded in
    ``model_params_config.json`` for appearance; the tests below are what say there is
    nothing behind the appearance.
    """

    def test_an_additional_checkpoint_that_repeats_a_regular_one_appears_twice(self):
        config = {"checkpoint_epochs": 10, "n_epochs": 30, "additional_checkpoints": [10, 25]}

        assert get_checkpoints(config) == [10, 10, 20, 25, 30]

    def test_the_duplicate_changes_nothing_either_consumer_sees(self):
        """
        Both reads are membership tests -- ``train_deep_sdf:179`` and ``:184`` -- so the
        repeated entry cannot save a checkpoint twice or shift an epoch boundary.
        """
        config = {"checkpoint_epochs": 10, "n_epochs": 30, "additional_checkpoints": [10, 25]}
        checkpoints = get_checkpoints(config)

        for epoch in range(0, 32):
            assert (epoch in checkpoints) == (epoch in sorted(set(checkpoints)))

    def test_a_config_without_additional_checkpoints_raises(self):
        """
        Fail-fast at ``train_deep_sdf:122``, before anything is written. Giving the key a
        default would make its absence mean "none", which is a config-schema decision and
        §8.0.N's.
        """
        with pytest.raises(KeyError, match="additional_checkpoints"):
            get_checkpoints({"checkpoint_epochs": 10, "n_epochs": 30})


# ---------------------------------------------------------------------------
# The module's public surface
# ---------------------------------------------------------------------------


#: Every public name bound in ``NSM.utils``. Frozen the way
#: ``test_train_import_compat`` freezes the trainer's: removing one is a deliberate,
#: changelogged decision, and adding one should be visible in a diff. ``honour_verbose``
#: is a leaked re-import from ``._verbose_deprecation`` and is listed for the same reason
#: that file lists ``os`` and ``torch`` -- because it is importable, not because it is API.
PUBLIC_NAMES = [
    "ConstantLearningRateSchedule",
    "LATENT_GROUP_NAME",
    "LR_TARGETS",
    "LR_TARGET_KEY",
    "LR_TARGET_LATENT",
    "LR_TARGET_MODEL",
    "LearningRateSchedule",
    "LogAnnealLearningRateSchedule",
    "MODEL_GROUP_PREFIX",
    "PARAM_GROUP_TARGET_KEY",
    "StepLearningRateSchedule",
    "WarmupLearningRateSchedule",
    "adjust_learning_rate",
    "clear_gpu_cache",
    "filter_non_jsonable",
    "get_checkpoints",
    "get_latent_vecs",
    "get_learning_rate_schedules",
    "get_optimizer",
    "honour_verbose",
    "is_jsonable",
    "resolve_schedule_targets",
    "save_latent_vectors",
    "save_model",
    "save_model_params",
]


def test_the_modules_public_names_are_frozen():
    """Imported modules and the logger are bindings too; only NSM's own names are listed."""
    imported = {"json", "logging", "math", "os", "warnings", "torch", "schedulefree"}
    bound = {
        name
        for name in dir(NSM.utils)
        if not name.startswith("_") and name not in imported and name != "logger"
    }

    assert sorted(bound) == PUBLIC_NAMES
