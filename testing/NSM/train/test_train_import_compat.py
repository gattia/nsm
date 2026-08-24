"""
Every public name importable from ``NSM.train.train_deep_sdf`` before the §8.0.D split
stays importable afterwards.

The split adds module-private helpers only — the module keeps every name it has today,
so this list must never change as part of it. The list includes the *leaked* bindings
alongside the deliberate API — ``itertools``, ``np``, ``os``, ``time``, ``torch``,
``wandb``, ``warnings``, and the re-imports from ``NSM.reconstruct``, ``NSM.train.utils``
and ``NSM.utils``. They are frozen because they are importable today, not because they
are endorsed; unleaking any of them is a deliberate, changelogged decision, not a side
effect of moving code (same rule as ``test_reconstruct_import_compat``).
"""

import importlib

import pytest

#: Public (non-underscore) names bound in ``NSM.train.train_deep_sdf`` before the split.
TRAINER_MODULE_NAMES = [
    "DICT_VALIDATION_FUNCS",
    "EIKONAL_UNSUPPORTED",
    "NoOpProfiler",
    "add_plain_lr_to_config",
    "adjust_learning_rate",
    "calc_weight",
    "clear_gpu_cache",
    "compare_cart_thickness",
    "compare_cart_thickness_femur",
    "compare_cart_thickness_patella",
    "compare_cart_thickness_tibia",
    "compare_cart_thickness_whole_joint",
    "cyclic_anneal_linear",
    "eikonal_loss",
    "get_checkpoints",
    "get_kld",
    "get_latent_vecs",
    "get_learning_rate_schedules",
    "get_mean_errors",
    "get_optimizer",
    "get_profiler",
    "itertools",
    "loss_l1",
    "np",
    "os",
    "save_latent_vectors",
    "save_model",
    "save_model_params",
    "time",
    "torch",
    "train_deep_sdf",
    "train_epoch",
    "wandb",
    "warnings",
]

#: Modules ``NSM.train``'s ``__init__`` exposes; frozen so an ``__init__`` edit cannot
#: narrow the package namespace unnoticed either.
PACKAGE_MODULES = ["train_deep_sdf", "train_deep_sdf_multi_head", "utils"]


@pytest.mark.parametrize("name", TRAINER_MODULE_NAMES)
def test_the_trainer_module_keeps_every_public_name(name):
    module = importlib.import_module("NSM.train.train_deep_sdf")
    assert hasattr(module, name), f"NSM.train.train_deep_sdf lost {name}"


@pytest.mark.parametrize("name", PACKAGE_MODULES)
def test_the_package_keeps_its_modules(name):
    package = importlib.import_module("NSM.train")
    assert hasattr(package, name), f"NSM.train lost {name}"
