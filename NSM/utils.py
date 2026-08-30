"""Learning-rate schedules, optimizer construction, and checkpoint I/O.

The file §1.2 of the code-health plan calls its exhibit: it held the positional
mapping that applied the model schedule to the latents and the latent schedule to
the model on every Adam/AdamW run from May 2023 to Aug 2026
(``docs/KNOWN_ISSUES.md`` §1).

What replaced it is one vocabulary spanning config and optimizer, and it is the
thing to know before touching anything here: a schedule entry declares
``Target`` (``"model"`` or ``"latent"``), a param group carries ``target``, and
``adjust_learning_rate`` is a dict lookup between them. **There is no positional
indexing anywhere in the LR path**, and entry order is ignored. Param groups also
carry ``name`` (``latent``, ``model_0``, ...), which is a human label only --
nothing dispatches on it, and several groups may share one target.

``NSM/_lr_migration.py`` is the transitional half of that change and says when to
delete it.
"""

import json
import logging
import math
import os
import warnings

import torch

from ._verbose_deprecation import honour_verbose

logger = logging.getLogger(__name__)

try:
    import schedulefree
except ImportError:
    warnings.warn("schedulefree not found, skipping import")
    schedulefree = None

# Human-readable labels for optimizer param groups. These are for logs and debugging
# only -- nothing dispatches on them. Scheduling is driven by the group's "target"
# (see LR_TARGET_KEY below), which is what makes group ORDER irrelevant.
LATENT_GROUP_NAME = "latent"
MODEL_GROUP_PREFIX = "model_"

# Key under which a param group records which schedule drives it. Same vocabulary as a
# config entry's "Target", so there is one set of names spanning config and optimizer.
PARAM_GROUP_TARGET_KEY = "target"


class LearningRateSchedule:
    """
    A learning rate as a function of epoch, for one target (``model`` or ``latent``).

    Subclasses are built from one ``config["LearningRateSchedule"]`` entry by
    :func:`get_learning_rate_schedules`, which reads the entry's ``Type`` to choose the
    class and its ``Target`` to decide which param groups the result drives. The entry's
    key names and the constructor's parameter names are two vocabularies for the same
    values and do not always match; each subclass below names its own.
    """

    def get_learning_rate(self, epoch):
        """
        The rate for ``epoch``. Every subclass implements this; the base refuses.

        Not ``pass``: a ``None`` returned from here is assigned straight into
        ``param_group["lr"]`` by :func:`adjust_learning_rate` and surfaces two calls later
        at ``optimizer.step()`` as ``unsupported operand type(s) for /: 'NoneType' and
        'float'``, naming neither the schedule, nor the group, nor the config entry.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_learning_rate(epoch). Every "
            f"LearningRateSchedule subclass must return the rate for an epoch."
        )


class ConstantLearningRateSchedule(LearningRateSchedule):
    """``Type: "Constant"``. One rate for the whole run, from the entry's ``Value``."""

    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):  # noqa: D102 - see the class docstring
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    """
    ``Type: "Step"``. ``initial * factor ** (epoch // interval)``.

    From the entry's ``Initial``, ``Interval`` and ``Factor``. ``interval`` of 0 is a
    ``ZeroDivisionError``; "no decay" is ``Factor: 1``, not ``Interval: 0``.
    """

    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):  # noqa: D102 - see the class docstring

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    """
    ``Type: "Warmup"``. Linear from ``initial`` to ``warmed_up`` over ``length`` epochs,
    then flat.

    Note the two names for the destination rate: the config entry calls it ``Final`` and
    this class calls it ``warmed_up``. ``length`` of 0 is a ``ZeroDivisionError``.
    """

    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):  # noqa: D102 - see the class docstring
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


class LogAnnealLearningRateSchedule(LearningRateSchedule):
    """
    ``Type: "LogAnneal"``. Geometric decay from ``initial`` to ``final`` at ``n_epochs``.

    The only schedule whose horizon comes from outside its own entry: ``n_epochs`` is the
    **top-level config key**, so the run length and the anneal length cannot disagree --
    and a ``Length`` on a ``LogAnneal`` entry is read by nothing.
    """

    def __init__(self, initial, final, n_epochs):
        self.initial = initial
        self.final = final
        self.n_epochs = n_epochs

    def get_learning_rate(self, epoch):  # noqa: D102 - see the class docstring
        return self.initial * math.exp(math.log(self.final / self.initial) * epoch / self.n_epochs)


# Per-entry key naming the parameter group a LearningRateSchedule entry drives.
# Entry ORDER carries no meaning. See resolve_schedule_targets() for why this is
# mandatory rather than inferred from position.
LR_TARGET_KEY = "Target"

#: Drives the model/decoder param groups (``model_0``, ``model_1``, ...) and any other
#: group that declares this target (NSM itself creates none beyond the decoders).
LR_TARGET_MODEL = "model"

#: Drives the latent-code param group.
LR_TARGET_LATENT = "latent"

LR_TARGETS = (LR_TARGET_MODEL, LR_TARGET_LATENT)


def resolve_schedule_targets(schedule_specs, optimizer="Adam"):
    """
    Return the target of each ``LearningRateSchedule`` entry, in entry order.

    Exactly two entries, one targeting ``"model"`` and one ``"latent"``. Position is never
    consulted -- this is the only thing that decides which schedule drives which parameter
    group, and it reads only the declared target.

    ``optimizer`` only tailors the migration message: Adam/AdamW and ``schedule_free_*``
    runs migrate to opposite annotations.

    Raises
    ------
    ValueError
        If there are not exactly two entries, if either omits ``Target``, or if the two do
        not cover ``model`` and ``latent`` exactly once.
    """
    if len(schedule_specs) != 2:
        raise ValueError(
            f"Expected exactly 2 LearningRateSchedule entries, one per target; got "
            f"{len(schedule_specs)}. (Extras were silently ignored before Aug 2026.)"
        )

    targets = [spec.get(LR_TARGET_KEY) for spec in schedule_specs]

    if None in targets:
        # One-time migration help, in its own module so it can be deleted wholesale.
        # Imported here rather than at module scope to keep that a one-line removal
        # (and because _lr_migration imports the LR_TARGET_* constants from here).
        from NSM._lr_migration import migration_error

        raise migration_error(
            schedule_specs,
            optimizer,
            problem=(
                f"No entry declares '{LR_TARGET_KEY}'."
                if all(target is None for target in targets)
                # The dangerous case: half-annotated looks migrated at a glance.
                else f"Entry {targets.index(None)} is missing '{LR_TARGET_KEY}' while the "
                f"other declares it. A partially annotated config is not migrated."
            ),
        )

    if sorted(targets) != sorted(LR_TARGETS):
        raise ValueError(
            f"LearningRateSchedule must target '{LR_TARGET_MODEL}' and "
            f"'{LR_TARGET_LATENT}' exactly once each; got {targets!r}."
        )

    return targets


def get_learning_rate_schedules(config):
    """
    Build learning-rate schedule objects from ``config``, keyed by target.

    Returns a ``{target: schedule}`` mapping, not a list: entries are matched to groups by
    their declared ``Target``, and there is deliberately no ordering anywhere in the LR
    path for a caller to get wrong.

    Returns
    -------
    dict
        Keys are :data:`LR_TARGET_MODEL` and :data:`LR_TARGET_LATENT`; values are
        :class:`LearningRateSchedule` instances.

    Raises
    ------
    ValueError
        If the entries do not each declare a valid ``Target`` covering both the model and
        the latent codes exactly once (see :func:`resolve_schedule_targets`).
    """
    schedule_specs = config["LearningRateSchedule"]
    targets = resolve_schedule_targets(schedule_specs, optimizer=config.get("optimizer", "Adam"))

    schedules = []

    for schedule_spec in schedule_specs:

        if schedule_spec["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_spec["Initial"],
                    schedule_spec["Interval"],
                    schedule_spec["Factor"],
                )
            )
        elif schedule_spec["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_spec["Initial"],
                    schedule_spec["Final"],
                    schedule_spec["Length"],
                )
            )
        elif schedule_spec["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_spec["Value"]))

        elif schedule_spec["Type"] == "LogAnneal":
            schedules.append(
                LogAnnealLearningRateSchedule(
                    schedule_spec["Initial"],
                    schedule_spec["Final"],
                    config["n_epochs"],
                )
            )

        else:
            raise ValueError(
                'no known learning rate schedule of type "{}"'.format(schedule_spec["Type"])
            )

    return dict(zip(targets, schedules))


@honour_verbose
def adjust_learning_rate(lr_schedules, optimizer, epoch, verbose=False):
    """
    Set each optimizer param group's learning rate for ``epoch``.

    ``lr_schedules`` is the ``{target: schedule}`` mapping from
    :func:`get_learning_rate_schedules`; each group names its own target. Nothing here
    depends on the order of either, which is the point: the previous implementation
    assigned ``lr_schedules[i]`` to ``param_groups[i]``, and because ``get_optimizer``
    orders the groups ``[latent, model...]`` the two schedules were applied swapped for
    the whole of every affected run. See ``docs/KNOWN_ISSUES.md``.

    Several groups may share a target -- every decoder takes the model schedule, as
    would any extra group declaring ``target="model"`` (NSM itself creates none).

    Raises
    ------
    KeyError
        If a param group declares no known target, which in practice means the optimizer
        state came from a checkpoint saved before Aug 2026 (the train loop rejects those
        at load time, so this is a backstop).
    """
    logger.debug("optimizer param groups:  %s", optimizer.param_groups)
    logger.debug("lr_schedules:  %s", lr_schedules)

    for param_group in optimizer.param_groups:
        target = param_group.get(PARAM_GROUP_TARGET_KEY)
        if target not in lr_schedules:
            raise KeyError(
                f"optimizer param_group {param_group.get('name')!r} declares no known "
                f"'{PARAM_GROUP_TARGET_KEY}' (got {target!r}; expected one of "
                f"{sorted(lr_schedules)}). Build the optimizer with get_optimizer(); if "
                f"its state came from a checkpoint saved before Aug 2026, that checkpoint "
                f"cannot be resumed."
            )
        param_group["lr"] = lr_schedules[target].get_learning_rate(epoch)


def save_latent_vectors(config, epoch, latent_vec, latent_codes_subdir="latent_codes"):
    """
    Write the latent embedding to ``{experiment_directory}/{subdir}/{epoch}.pth``.

    One file per checkpoint epoch, ``{"epoch": ..., "latent_codes": state_dict}``, read
    back by ``train_deep_sdf`` on resume. The whole embedding is saved, so with
    ``variational: true`` each row is the ``2 * latent_size`` mean-and-log-variance pair
    :func:`get_latent_vecs` allocated.
    """
    filename = f"{epoch}.pth"
    folder_save = os.path.join(config["experiment_directory"], latent_codes_subdir)
    if not os.path.exists(folder_save):
        os.makedirs(folder_save, exist_ok=True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(folder_save, filename),
    )


def save_model(config, epoch, decoder, model_subdir="model", optimizer=None):
    """
    Save a decoder checkpoint.

    Param-group targets need no special handling: ``optimizer.state_dict()`` retains
    custom group keys and ``load_state_dict()`` restores them. They are still validated
    here, so a checkpoint can never be written from an optimizer whose groups have lost
    the target that schedules them.
    """
    if type(decoder) not in (list, tuple):
        decoder = [decoder]

    filename = f"{epoch}.pth"

    # None, not the string "None" this used to write. The string is truthy, so the natural
    # `if checkpoint["optimizer"]:` reads as "state present" and hands load_state_dict a str.
    optimizer_state = None
    if optimizer is not None:
        if any(
            group.get(PARAM_GROUP_TARGET_KEY) not in LR_TARGETS for group in optimizer.param_groups
        ):
            raise ValueError(
                f"Every optimizer param group must declare a '{PARAM_GROUP_TARGET_KEY}' "
                f"of {list(LR_TARGETS)} before saving. Build the optimizer with "
                f"get_optimizer(); if its state came from a checkpoint saved before "
                f"Aug 2026, that checkpoint cannot be resumed."
            )
        optimizer_state = optimizer.state_dict()

    for decoder_idx, decoder_ in enumerate(decoder):
        if len(decoder) > 1:
            model_subdir_ = model_subdir + f"_{decoder_idx}"
        else:
            model_subdir_ = model_subdir

        folder_save = os.path.join(config["experiment_directory"], model_subdir_)
        if not os.path.exists(folder_save):
            os.makedirs(folder_save, exist_ok=True)

        dict_ = {
            "epoch": epoch,
            "model": decoder_.state_dict(),
            "optimizer": optimizer_state,
        }

        torch.save(
            dict_,
            os.path.join(folder_save, filename),
        )


def _diverging_keys(path_existing, dict_new):
    """
    The keys on which ``dict_new`` disagrees with the record already at ``path_existing``.

    Added, removed and changed in one sorted list: all three mean the same thing to the
    reader of a file that is not going to be rewritten.
    """
    with open(path_existing) as f:
        existing = json.load(f)

    shared = set(existing) & set(dict_new)
    return sorted(
        (set(existing) ^ set(dict_new)) | {key for key in shared if existing[key] != dict_new[key]}
    )


def save_model_params(config, list_mesh_paths):
    """
    Write ``model_params_config.json``, the record of what produced this experiment.

    ``load_model``, ``examples/load_trained_model.py`` and both consumer scripts rebuild a
    model from this file (``docs/SCOPE.md`` §5), so it is the run's public contract and not
    a log. Called on every checkpoint.

    **First write wins, deliberately.** The file records the configuration that produced
    the weights; a resumed or re-configured run must not replace that record with a
    description of a run that only partly happened. When the current config disagrees with
    what is on disk, every diverging key is named in a ``WARNING`` -- the refusal used to
    be silent, which is the first half of #50.

    ``list_mesh_paths`` is applied **after** ``config``, so the dataset that is training
    wins over any ``list_mesh_paths`` the config carries; the shipped ``default_config.json``
    carries ``None`` and a config round-tripped from an earlier run carries that run's
    subjects. See ``docs/KNOWN_ISSUES.md`` § History 26.

    Values ``json.dumps`` cannot encode are omitted and named in a ``WARNING`` at the write
    -- the second half of #50. In a normal run that is ``lr_schedules`` alone, which the
    trainer derived from the ``LearningRateSchedule`` entries this file does carry.
    """
    if not os.path.exists(config["experiment_directory"]):
        os.makedirs(config["experiment_directory"], exist_ok=True)

    path_save = os.path.join(config["experiment_directory"], "model_params_config.json")

    dict_save = dict(config)
    dict_save["list_mesh_paths"] = list_mesh_paths  # after the merge, so the data wins
    dropped = sorted(key for key, value in dict_save.items() if not is_jsonable(value))
    dict_save = filter_non_jsonable(dict_save)

    if os.path.exists(path_save):
        diverging = _diverging_keys(path_save, dict_save)
        if diverging:
            logger.warning(
                "%s already exists and is not being rewritten: it records the "
                "configuration that produced the weights stored alongside it. %d value(s) "
                "in the current config disagree with what it holds: %s.",
                path_save,
                len(diverging),
                ", ".join(diverging),
            )
        return

    superseded = config.get("list_mesh_paths")
    if superseded is not None and superseded != list_mesh_paths:
        logger.warning(
            "config declares %d mesh path(s) and the dataset supplies %d; recording the "
            "dataset's. A config carrying this key usually came from an earlier run's "
            "model_params_config.json, whose subject list is not this run's.",
            len(superseded),
            len(list_mesh_paths),
        )

    # At the write, not at the filter: once per run rather than once per checkpoint.
    if dropped:
        logger.warning(
            "%d config value(s) cannot be JSON-encoded and are omitted from %s: %s. The "
            "filter is shallow, so a nested value is omitted whole.",
            len(dropped),
            os.path.basename(path_save),
            ", ".join(dropped),
        )

    with open(path_save, "w") as f:
        json.dump(dict_save, f, indent=4)


def get_checkpoints(config):
    """
    The epochs at which the trainer writes a checkpoint, ascending.

    Every ``checkpoint_epochs``-th epoch up to and including ``n_epochs``, plus everything
    in ``additional_checkpoints`` -- which is required, not defaulted, so a config missing
    it raises here rather than silently checkpointing on the regular cadence only.

    An ``additional_checkpoints`` entry that repeats a regular epoch appears twice in the
    result. Both consumers test membership (``train_deep_sdf``'s epoch loop), so the
    duplicate cannot save an epoch twice; it is left in rather than deduplicated because
    the list is recorded verbatim in ``model_params_config.json``.
    """
    checkpoints = list(
        range(
            config["checkpoint_epochs"],
            config["n_epochs"] + 1,
            config["checkpoint_epochs"],
        )
    )

    for checkpoint in config["additional_checkpoints"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    return checkpoints


def get_latent_vecs(num_objects, config):
    """Build the per-object latent embedding the trainer optimizes.

    Variational contract (deliberate, per maintainer review 2026-08-22): with
    ``variational: true`` the embedding width doubles to ``2 * latent_size`` because
    each row stores the VAE mean and log-variance; the decoder's latent is still
    ``latent_size``, so that is the correct value for ``model_params_config.json``
    and for downstream consumers. In the same mode the hard ``latent_bound``
    (``max_norm``) is superseded: training swaps the regularizer to KLD
    (``train_deep_sdf.train_epoch``), and the hardcoded 1000 below is deliberately
    "effectively unbounded", not an ignored config value.
    """
    if ("variational" in config) and (config["variational"] is True):
        latent_size = config["latent_size"] * 2
        latent_bound = 1000
    else:
        latent_size = config["latent_size"]
        latent_bound = config["latent_bound"]

    lat_vecs = torch.nn.Embedding(num_objects, latent_size, max_norm=latent_bound)

    if ("latent_init_normal" in config) and (config["latent_init_normal"] is True):
        torch.nn.init.normal_(
            lat_vecs.weight.data,
            0.0,
            config["latent_init_std"] / math.sqrt(latent_size),
        )

    return lat_vecs


def get_optimizer(model, latent_vecs, lr_schedules, optimizer="Adam", weight_decay=0.0001):
    """
    Build the optimizer with TARGETED parameter groups.

    ``lr_schedules`` is the ``{target: schedule}`` mapping from
    :func:`get_learning_rate_schedules`.

    Each group carries a ``target`` naming the schedule that drives it, plus a ``name``
    that is a human label only. Groups are still emitted ``[latent, model_0, ...]`` for
    checkpoint compatibility with existing runs, but nothing reads that order.
    """
    if type(model) not in (list, tuple):
        model = [model]

    list_params = [
        {
            "name": LATENT_GROUP_NAME,
            PARAM_GROUP_TARGET_KEY: LR_TARGET_LATENT,
            "params": latent_vecs.parameters(),
            "lr": lr_schedules[LR_TARGET_LATENT].get_learning_rate(0),
        }
    ]
    for idx, model_ in enumerate(model):
        list_params.append(
            {
                "name": f"{MODEL_GROUP_PREFIX}{idx}",
                PARAM_GROUP_TARGET_KEY: LR_TARGET_MODEL,
                "params": model_.parameters(),
                "lr": lr_schedules[LR_TARGET_MODEL].get_learning_rate(0),
            }
        )

    if optimizer == "Adam":
        optimizer = torch.optim.Adam(list_params, weight_decay=weight_decay)
    elif optimizer == "AdamW":
        optimizer = torch.optim.AdamW(list_params, weight_decay=weight_decay)
    elif optimizer == "schedule_free_AdamW":
        if schedulefree is None:
            raise ImportError("schedulefree not imported, because not installed")
        optimizer = schedulefree.AdamWScheduleFree(list_params, weight_decay=weight_decay)
    elif optimizer == "schedule_free_SGD":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    return optimizer


def is_jsonable(x):
    """Whether ``json.dumps`` can encode ``x``. Answers, never raises."""
    try:
        json.dumps(x)
        return True
    # ValueError is what a cycle raises ("Circular reference detected"). A predicate that
    # answers True or False must not propagate it out of a checkpoint write.
    except (TypeError, OverflowError, ValueError):
        return False


def filter_non_jsonable(dict_obj):
    """
    ``dict_obj`` without the values ``json.dumps`` cannot encode.

    **Shallow**: the test runs on each whole value, so a nested dict holding one
    unencodable leaf is dropped entire, serialisable siblings and all. Callers that need
    to tell the user what left the record should compute the dropped set with
    :func:`is_jsonable` before calling this -- :func:`save_model_params` does.
    """
    return {k: v for k, v in dict_obj.items() if is_jsonable(v)}


def clear_gpu_cache(device):
    """
    Empty the allocator cache for ``device``, which may be a string or a ``torch.device``.

    Warns and does nothing for anything that is neither CUDA nor MPS; the trainer calls
    this once an epoch, so a CPU run gets the warning once, from Python's default
    once-per-location filter.
    """
    # str() so a torch.device works as well as the JSON string the trainer passes:
    # `"cuda" in torch.device("cuda")` is a TypeError, not a False.
    device = str(device)

    if "cuda" in device:
        torch.cuda.empty_cache()
    elif "mps" in device:
        torch.mps.empty_cache()
    else:
        warnings.warn("Not clearing cache because not cuda or mps (apple metal)")
