import json
import math
import os
import warnings

import torch

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
CLASSIFICATION_HEADS_GROUP_NAME = "classification_heads"

# Key under which a param group records which schedule drives it. Same vocabulary as a
# config entry's "Target", so there is one set of names spanning config and optimizer.
PARAM_GROUP_TARGET_KEY = "target"


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


class LogAnnealLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, final, n_epochs):
        self.initial = initial
        self.final = final
        self.n_epochs = n_epochs

    def get_learning_rate(self, epoch):
        return self.initial * math.exp(math.log(self.final / self.initial) * epoch / self.n_epochs)


# Per-entry key naming the parameter group a LearningRateSchedule entry drives.
# Entry ORDER carries no meaning. See resolve_schedule_targets() for why this is
# mandatory rather than inferred from position.
LR_TARGET_KEY = "Target"

#: Drives the model/decoder param groups (``model_0``, ``model_1``, ...) and, when
#: present, ``classification_heads``.
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


def adjust_learning_rate(lr_schedules, optimizer, epoch, verbose=False):
    """
    Set each optimizer param group's learning rate for ``epoch``.

    ``lr_schedules`` is the ``{target: schedule}`` mapping from
    :func:`get_learning_rate_schedules`; each group names its own target. Nothing here
    depends on the order of either, which is the point: the previous implementation
    assigned ``lr_schedules[i]`` to ``param_groups[i]``, and because ``get_optimizer``
    orders the groups ``[latent, model...]`` the two schedules were applied swapped for
    the whole of every affected run. See ``docs/KNOWN_ISSUES.md``.

    Several groups may share a target -- every decoder and the classification heads all
    take the model schedule.

    Raises
    ------
    KeyError
        If a param group declares no known target, which in practice means the optimizer
        state came from a checkpoint saved before Aug 2026 (the train loop rejects those
        at load time, so this is a backstop).
    """
    if verbose is True:
        print("optimizer param groups: ", optimizer.param_groups)
        print("lr_schedules: ", lr_schedules)

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


def save_model_params(config, list_mesh_paths):

    if not os.path.exists(config["experiment_directory"]):
        os.makedirs(config["experiment_directory"], exist_ok=True)

    path_save = os.path.join(config["experiment_directory"], "model_params_config.json")

    if os.path.exists(path_save):
        return

    dict_save = {
        "list_mesh_paths": list_mesh_paths,
    }
    dict_save.update(config)

    dict_save = filter_non_jsonable(dict_save)

    with open(path_save, "w") as f:
        json.dump(dict_save, f, indent=4)


def get_checkpoints(config):
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
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def filter_non_jsonable(dict_obj):
    return {k: v for k, v in dict_obj.items() if is_jsonable(v)}


def print_gpu_memory():
    # assert cuda is available
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        print("CUDA, GPU Usage:")
        print(f"\tAllocated memory: {allocated / 1024**3:.2f} GB")
        print(f"\tCached memory: {cached / 1024**3:.2f} GB")
    else:
        print("CUDA not available - GPU stats not available")


def clear_gpu_cache(device):
    if "cuda" in device:
        torch.cuda.empty_cache()
    elif "mps" in device:
        torch.mps.empty_cache()
    else:
        warnings.warn("Not clearing cache because not cuda or mps (apple metal)")
