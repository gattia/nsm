import numpy as np
import torch
from torch.profiler import profile, tensorboard_trace_handler

from NSM.utils import LR_TARGET_LATENT, LR_TARGET_MODEL, resolve_schedule_targets


def calc_weight(epoch, n_epochs, schedule, cooldown=None):
    """Curriculum weight ramping 0 -> 1 over training (1.0 inside any cooldown tail).

    Its two consumers apply it in OPPOSITE directions, deliberately:
    ``sample_difficulty_weight`` uses the value directly (emphasis grows over
    training), while ``surface_accuracy_e`` uses ``1 - calc_weight(...)`` so the
    error tolerance SHRINKS over training (Curriculum-DeepSDF eq. 5). That also
    means ``schedule="constant"`` (always 1.0) keeps sample-difficulty weighting
    fully on but turns the surface-accuracy tolerance OFF entirely (1 - 1 = 0).
    """
    if cooldown is not None:
        if epoch > (n_epochs - cooldown):
            return 1.0
        else:
            n_epochs = n_epochs - cooldown
    if schedule == "linear":
        return epoch / n_epochs
    elif schedule == "exponential":
        return epoch**2 / n_epochs**2
    elif schedule == "exponential_plateau":
        return 1 - (epoch - n_epochs) ** 2 / n_epochs**2
    elif schedule == "constant":
        return 1.0
    else:
        raise ValueError("Unknown schedule: {}".format(schedule))


def cyclic_anneal_linear(
    epoch,
    n_epochs,
    min_=0,
    max_=1,
    ratio=0.5,  # ratio of the cycle to be increasing; 1-ratio is plateaued @ max_
    n_cycles=5,
):
    """
    https://github.com/haofuml/cyclical_annealing
    """
    # A run shorter than n_cycles would make this 0, and `epoch % 0` is NaN — which
    # silently NaN'd the entire training loss while the run completed and exited 0.
    # Degenerate runs get one-epoch cycles, pinning the weight at min_; any run with
    # n_epochs >= n_cycles is unchanged.
    cycle_length = max(int(np.floor(n_epochs / n_cycles)), 1)
    cycle_progress = epoch % cycle_length

    weight = (cycle_progress / cycle_length) * (1 / ratio)
    weight = np.min([weight, 1])

    return min_ + (max_ - min_) * weight


def get_kld(array, samples_dim=0):
    """Scalar KLD between the BATCH's empirical diagonal Gaussian and N(0, I).

    Not the standard per-sample VAE estimator (which sums a per-row
    ``-0.5 * (1 + log_var - mu**2 - exp(log_var))`` from encoder outputs): this takes
    the empirical mean and variance of ``array`` across ``samples_dim`` and plugs
    those moments into the closed form, summing over latent dimensions into one
    scalar. ``torch.var`` applies Bessel's correction. Because the moments are
    estimated from whatever batch is passed, the value depends on batch size —
    do not compare its magnitude across runs with different batch sizes.

    Reachable only via ``code_regularization_type_prior: "kld_diagonal"``, which is
    not the shipped default.
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    """
    mean = torch.mean(array, dim=samples_dim)
    var = torch.var(array, dim=samples_dim)
    kld = -0.5 * torch.sum(1 + torch.log(var) - mean**2 - var)

    return kld


def add_plain_lr_to_config(config, idx_model=None, idx_latent=None):
    """
    Flatten the two LearningRateSchedule entries into scalar config keys for logging.

    Which entry is the model and which is the latent comes from each entry's declared
    ``Target``, not from its position, so logged ``model_lr_*`` / ``latent_lr_*`` values
    always carry the correct labels. Explicit indices override the lookup.

    Mutates the caller's ``config`` in place and returns that same object (the return
    value is a convenience, not a copy).
    """
    if idx_model is None or idx_latent is None:
        targets = resolve_schedule_targets(
            config["LearningRateSchedule"], optimizer=config.get("optimizer", "Adam")
        )
        if idx_model is None:
            idx_model = targets.index(LR_TARGET_MODEL)
        if idx_latent is None:
            idx_latent = targets.index(LR_TARGET_LATENT)

    schedules = {
        "model": idx_model,
        "latent": idx_latent,
    }

    schedule_specs = config["LearningRateSchedule"]

    for key, idx in schedules.items():
        schedule_ = schedule_specs[idx]
        config[f"{key}_lr_type"] = schedule_["Type"]
        # Constant entries carry "Value" where every other type carries "Initial".
        initial = schedule_.get("Initial", schedule_.get("Value"))
        if initial is not None:
            config[f"{key}_lr_initial"] = initial
        if "Interval" in schedule_.keys():
            config[f"{key}_lr_update_interval"] = schedule_["Interval"]
        if "Factor" in schedule_.keys():
            config[f"{key}_lr_update_factor"] = schedule_["Factor"]
        if "Final" in schedule_.keys():
            config[f"{key}_lr_final"] = schedule_["Final"]
    return config


class NoOpProfiler:
    """
    A profiler that does nothing.
    """

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def step(self):
        pass


def get_profiler(config):
    if config["profiler"]:
        return torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=0, warmup=2, active=6),
            on_trace_ready=tensorboard_trace_handler("./log"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
    else:
        return NoOpProfiler()
