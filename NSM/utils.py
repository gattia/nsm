import torch
import os
import math
import json

try:
    import schedulefree
except ImportError:
    print("schedulefree not found, skipping import")
    schedulefree = None
import warnings


# Semantic names for optimizer param groups. adjust_learning_rate() maps schedules to
# groups by these names, so group ORDER is never load-bearing.
LATENT_GROUP_NAME = "latent"
MODEL_GROUP_PREFIX = "model_"
CLASSIFICATION_HEADS_GROUP_NAME = "classification_heads"


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

# --- MIGRATION SCAFFOLDING (added Aug 2026) ---------------------------------------
# Everything down to the END MIGRATION SCAFFOLDING marker exists only to explain the
# Aug 2026 Target change to someone holding a config written before it. None of it is
# permanent API. Delete the whole block once no config still in use predates the Target
# key; resolve_schedule_targets() then just raises a one-line error.
#
# Which entry drove which group before Aug 2026, keyed by optimizer family. Adam/AdamW
# went through adjust_learning_rate(), which mapped positionally against get_optimizer()'s
# [latent, model...] group order -- so entry 0 drove the latents. schedule_free_* skipped
# adjust_learning_rate() entirely and kept get_optimizer()'s own assignment, where entry 0
# drove the model. The two families therefore migrate to OPPOSITE annotations.
_HISTORICAL_TARGETS_ADAM = (LR_TARGET_LATENT, LR_TARGET_MODEL)
_HISTORICAL_TARGETS_SCHEDULE_FREE = (LR_TARGET_MODEL, LR_TARGET_LATENT)

_LR_TARGET_MIGRATION_MESSAGE = """
Every 'LearningRateSchedule' entry must declare '{key}' ("{model}" or "{latent}").

{problem}

WHY THIS IS REQUIRED

Entry order used to decide which schedule drove which parameter group, and from May 2023
to Aug 2026 a mapping bug applied the two entries swapped on every Adam/AdamW run: the
latent codes trained under entry 0 and the model under entry 1. A config written before
the fix and one written after are byte-identical while meaning opposite things, so the
intent cannot be recovered from the file. It has to be stated.

TO REPRODUCE THIS RUN AS IT ORIGINALLY TRAINED

This config's optimizer is '{optimizer}', for which the historical mapping was
entry 0 -> {hist_0}, entry 1 -> {hist_1}. Annotating the entries that way reproduces the
original run exactly:

{annotated}
{caution}
TO CONFIGURE A NEW RUN

Set '{key}' on each entry to the group you intend it to drive. Order is ignored, so list
them in whichever order reads best.

See docs/KNOWN_ISSUES_HISTORY.md section 1.
""".strip()


_SCHEDULE_FREE_CAUTION = """
CAUTION -- READ BEFORE REPRODUCING THIS ONE

'schedule_free_*' never called adjust_learning_rate(), so it kept get_optimizer()'s
assignment: the OPPOSITE of what an Adam/AdamW run of the same file did. The same config
therefore meant two different things depending on which optimizer you picked.

If these values were copied or tuned from an Adam/AdamW config -- which is how most of
them were written -- then this run applied the latent's rate to the model and the model's
rate to the latent, held CONSTANT for the whole run, since nothing ever decayed them. That
is a plausible reason for a schedule_free run to have trained badly.

So reproducing this run faithfully may not be what you want. Compare these values against
an Adam/AdamW config for the same experiment before you choose the annotation.
"""


def _migration_error(schedule_specs, optimizer, problem):
    """
    Build the error shown when a config predates the ``Target`` key.

    Includes a paste-ready copy of the caller's own entries, annotated with the targets
    that reproduce their historical run -- which differ by optimizer family.
    """
    schedule_free = "schedule_free" in str(optimizer)
    hist_0, hist_1 = (
        _HISTORICAL_TARGETS_SCHEDULE_FREE if schedule_free else _HISTORICAL_TARGETS_ADAM
    )

    annotated = [
        {LR_TARGET_KEY: target, **{k: v for k, v in spec.items() if k != LR_TARGET_KEY}}
        for spec, target in zip(schedule_specs, (hist_0, hist_1))
    ]
    body = json.dumps({"LearningRateSchedule": annotated}, indent=4)

    return ValueError(
        _LR_TARGET_MIGRATION_MESSAGE.format(
            key=LR_TARGET_KEY,
            model=LR_TARGET_MODEL,
            latent=LR_TARGET_LATENT,
            problem=problem,
            optimizer=optimizer,
            hist_0=hist_0,
            hist_1=hist_1,
            annotated="\n".join("    " + line for line in body.splitlines()),
            caution=_SCHEDULE_FREE_CAUTION if schedule_free else "",
        )
    )


# --- END MIGRATION SCAFFOLDING ------------------------------------------------------
# Deleting the block above leaves resolve_schedule_targets() needing a plain one-line
# ValueError in place of its _migration_error() call.


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
        raise _migration_error(
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
    Build learning-rate schedule objects from ``config``, in canonical order.

    Entries are matched to groups by their declared ``Target``, never by position. The
    returned list is in canonical order ``[model, latent]`` so every downstream consumer
    (:func:`get_optimizer`, :func:`adjust_learning_rate`) can index it unconditionally --
    that ordering is an internal calling convention, not something the config controls.

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

    by_target = dict(zip(targets, schedules))

    return [by_target[LR_TARGET_MODEL], by_target[LR_TARGET_LATENT]]


def adjust_learning_rate(lr_schedules, optimizer, epoch, verbose=False):
    """
    Set each optimizer param group's learning rate for ``epoch``, mapping by group NAME.

    ``lr_schedules`` is in canonical order: index 0 = model/decoder, index 1 = latent
    codes. :func:`get_learning_rate_schedules` guarantees this by reading each entry's
    declared ``Target``, so the config's own entry order is irrelevant here.

    Mapping onto groups is by name rather than position. The previous implementation
    assigned ``lr_schedules[i]`` to ``param_groups[i]``, but ``get_optimizer`` orders the
    groups ``[latent, model...]`` -- so the two schedules were applied swapped for the
    whole of every affected run. See ``docs/KNOWN_ISSUES_HISTORY.md``.

    Raises
    ------
    KeyError
        If a param group has no recognized name, which in practice means the optimizer
        state came from a checkpoint saved before Aug 2026 (the train loop rejects those
        at load time, so this is a backstop).
    """
    if verbose is True:
        print("optimizer param groups: ", optimizer.param_groups)
        print("lr_schedules: ", lr_schedules)

    if len(lr_schedules) < 2:
        raise ValueError(
            f"Expected at least 2 lr_schedules (index 0 = model, index 1 = latent codes); "
            f"got {len(lr_schedules)}."
        )

    for param_group in optimizer.param_groups:
        name = param_group.get("name")
        if name == LATENT_GROUP_NAME:
            param_group["lr"] = lr_schedules[1].get_learning_rate(epoch)
        elif name == CLASSIFICATION_HEADS_GROUP_NAME or (
            name is not None and name.startswith(MODEL_GROUP_PREFIX)
        ):
            param_group["lr"] = lr_schedules[0].get_learning_rate(epoch)
        else:
            raise KeyError(
                f"optimizer param_group has no recognized 'name' (got {name!r}; expected "
                f"'{LATENT_GROUP_NAME}', '{MODEL_GROUP_PREFIX}*', or "
                f"'{CLASSIFICATION_HEADS_GROUP_NAME}'). Build the optimizer with "
                f"get_optimizer(); if its state came from a checkpoint saved before "
                f"Aug 2026, that checkpoint cannot be resumed."
            )


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

    Param-group names need no special handling: ``optimizer.state_dict()`` retains the
    ``name`` key, and ``load_state_dict()`` restores it. The names are still validated
    here, so a checkpoint can never be written from an optimizer whose groups have lost
    their identity.
    """
    if type(decoder) not in (list, tuple):
        decoder = [decoder]

    filename = f"{epoch}.pth"

    optimizer_state = "None"
    if optimizer is not None:
        if any(group.get("name") is None for group in optimizer.param_groups):
            raise ValueError(
                "All optimizer param groups must have a 'name' before saving. Build the "
                "optimizer with get_optimizer(); if its state came from a checkpoint "
                "saved before Aug 2026, that checkpoint cannot be resumed."
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
    Build the optimizer with NAMED parameter groups.

    ``lr_schedules`` is in canonical order: index 0 = model/decoder, index 1 = latent
    codes.

    Groups are emitted in the order ``[latent, model_0, model_1, ...]`` for checkpoint
    compatibility with existing runs, but each carries a ``name`` so
    :func:`adjust_learning_rate` can map schedules by name rather than by position.
    """
    if type(model) not in (list, tuple):
        model = [model]

    if len(lr_schedules) < 2:
        raise ValueError(
            f"Expected at least 2 lr_schedules (index 0 = model, index 1 = latent codes); "
            f"got {len(lr_schedules)}."
        )

    list_params = [
        {
            "name": LATENT_GROUP_NAME,
            "params": latent_vecs.parameters(),
            "lr": lr_schedules[1].get_learning_rate(0),
        }
    ]
    for idx, model_ in enumerate(model):
        list_params.append(
            {
                "name": f"{MODEL_GROUP_PREFIX}{idx}",
                "params": model_.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            }
        )

    if optimizer == "Adam":
        optimizer = torch.optim.Adam(list_params)
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


def symmetric_chammfer(p1, p2, n_pts):
    """ """
    pass


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
