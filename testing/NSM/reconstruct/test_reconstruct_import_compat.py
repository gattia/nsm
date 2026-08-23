"""
Every name importable from ``NSM.reconstruct`` or ``NSM.reconstruct.main`` before the
§8.0.C split stays importable from the same path afterwards.

Both are live import paths — the trainers and kneepipeline use the package path,
``test_predictive_validation.py`` monkeypatches ``NSM.reconstruct.main``, and downstream
forks are assumed to use both — so the re-import block the split leaves in ``main.py``
is public API, not scaffolding. The lists below are frozen deliberately: dropping a name
must turn this red, not silently narrow the API.

The lists include the *leaked* bindings alongside the deliberate API —
``adjust_learning_rate`` (the ``reconstruct/utils.py`` step-decay one, a documented trap:
``docs/ARCHITECTURE.md`` §6), ``Regress``, ``combine_meshes``, the two readers,
``create_mesh_adaptive``, ``eikonal_loss``, ``compute_recon_loss``, ``fnmatch`` and
``logger``. They are frozen because they are importable today, not because they are
endorsed; unleaking any of them is a deliberate, changelogged decision, not a side
effect of moving code.
"""

import importlib

import pytest

#: Bound in ``main.py`` and visible on both paths (the star-import skips underscores,
#: so ``_process_meshes_for_wandb`` is listed separately).
FROM_MAIN = [
    "EIKONAL_UNSUPPORTED",
    "Regress",
    "adjust_learning_rate",
    "combine_meshes",
    "compute_correlation_coefficient",
    "compute_recon_loss",
    "create_mesh_adaptive",
    "eikonal_loss",
    "fnmatch",
    "get_mean_errors",
    "latent_norm_penalty",
    "logger",
    "prepare_results_for_wandb",
    "project_latent",
    "read_mesh_get_sampled_pts",
    "read_meshes_get_sampled_pts",
    "reconstruct_latent",
    "reconstruct_latent_decoders_type_check",
    "reconstruct_latent_get_lr_update_freq",
    "reconstruct_latent_preprocess_sdf_gt",
    "reconstruct_latent_pts_surface_type_check",
    "reconstruct_latent_sdf_gt_type_check",
    "reconstruct_mesh",
    "tune_reconstruction",
]

MAIN_ONLY = ["_process_meshes_for_wandb"]

#: Reaches ``NSM.reconstruct`` from its sibling modules via ``__init__``; frozen so an
#: ``__init__`` edit cannot narrow the package namespace unnoticed either.
PACKAGE_ONLY = [
    "compare_cart_thickness",
    "compare_cart_thickness_femur",
    "compare_cart_thickness_patella",
    "compare_cart_thickness_tibia",
    "compare_cart_thickness_whole_joint",
    "reconstruct_latent_S3",
]


@pytest.mark.parametrize("module_path", ["NSM.reconstruct", "NSM.reconstruct.main"])
@pytest.mark.parametrize("name", FROM_MAIN)
def test_every_main_name_is_importable_from_both_paths(module_path, name):
    module = importlib.import_module(module_path)
    assert hasattr(module, name), f"{module_path} lost {name}"


@pytest.mark.parametrize("name", MAIN_ONLY)
def test_main_module_keeps_its_underscore_names(name):
    module = importlib.import_module("NSM.reconstruct.main")
    assert hasattr(module, name), f"NSM.reconstruct.main lost {name}"


@pytest.mark.parametrize("name", PACKAGE_ONLY)
def test_package_keeps_its_sibling_module_names(name):
    module = importlib.import_module("NSM.reconstruct")
    assert hasattr(module, name), f"NSM.reconstruct lost {name}"
