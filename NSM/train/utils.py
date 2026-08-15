import numpy as np
import torch

import torch
from torch.profiler import profile, tensorboard_trace_handler

from NSM.utils import LR_TARGET_LATENT, LR_TARGET_MODEL, resolve_schedule_targets


def calc_weight(epoch, n_epochs, schedule, cooldown=None):

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
    cycle_length = np.floor(n_epochs / n_cycles).astype(int)
    cycle = epoch // cycle_length
    cycle_progress = epoch % cycle_length

    weight = (cycle_progress / cycle_length) * (1 / ratio)
    weight = np.min([weight, 1])

    return min_ + (max_ - min_) * weight


def get_kld(array, samples_dim=0):
    """
    kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    Above is the KLD between a diagonal multivariate normal distribution and a standard normal distribution.
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
        config[f"{key}_lr_initial"] = schedule_["Initial"]
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
