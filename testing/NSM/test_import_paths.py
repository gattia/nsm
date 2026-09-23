"""
Every public name stays importable from every path it has been importable from.

``NSM.datasets.sdf_dataset``, ``NSM.reconstruct.main`` and ``NSM.train.train_deep_sdf``
re-import names whose definitions moved to other modules. Those re-import blocks are public
API. Removing a name from this table is a deliberate, changelogged decision.

Some listed names are not endorsed: ``reconstruct.main``'s ``adjust_learning_rate`` is the
step-decay one, a trap (``docs/ARCHITECTURE.md`` §6). They are listed because callers can
import them today. Imported modules (``np``, ``torch``, ``os``) and ``logger`` are not
listed: they are not API.
"""

import importlib

DATASETS = [
    "MultiSurfaceSDFSamples",
    "SDFSamples",
    "check_probabilities",
    "check_probabilities_sum",
    "combine_meshes",
    "derive_seed",
    "get_buffered_cube_mins_maxs",
    "get_cube_mins_maxs",
    "get_pts_center_and_scale",
    "get_rand_uniform_pts",
    "is_zipfile",
    "mesh_content_key",
    "meshfix",
    "read_mesh_get_sampled_pts",
    "read_meshes_get_sampled_pts",
    "today_date",
    "unpack_numpy_data",
    "unpack_pts",
]

RECONSTRUCT = [
    "EIKONAL_UNSUPPORTED",
    "NoZeroLevelSetError",
    "Regress",
    "adjust_learning_rate",
    "combine_meshes",
    "compute_recon_loss",
    "create_mesh_adaptive",
    "eikonal_loss",
    "fnmatch",
    "get_mean_errors",
    "latent_norm_penalty",
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
]

TRAINER = [
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
    "loss_l1",
    "save_latent_vectors",
    "save_model",
    "save_model_params",
    "train_deep_sdf",
    "train_epoch",
]

UTILS = [
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
    "is_jsonable",
    "resolve_schedule_targets",
    "save_latent_vectors",
    "save_model",
    "save_model_params",
]

PATHS = {
    "NSM.datasets": DATASETS,
    "NSM.datasets.sdf_dataset": DATASETS,
    "NSM.mesh.refine_mesh": ["get_faces"],
    "NSM.models": [
        "Decoder",
        "ImplicitDecoder",
        "TriplanarDecoder",
        "get_model_config_template",
        "list_supported_models",
        "load_model",
    ],
    "NSM.reconstruct": RECONSTRUCT
    + [
        "compare_cart_thickness",
        "compare_cart_thickness_femur",
        "compare_cart_thickness_patella",
        "compare_cart_thickness_tibia",
        "compare_cart_thickness_whole_joint",
        "reconstruct_latent_S3",
    ],
    "NSM.reconstruct.main": RECONSTRUCT + ["_process_meshes_for_wandb"],
    "NSM.train": ["train_deep_sdf", "utils"],
    "NSM.train.train_deep_sdf": TRAINER,
    "NSM.utils": UTILS,
}


def test_every_public_name_is_importable_from_every_path():
    """
    Fails if a module in ``PATHS`` stops binding a name listed for it, such as a re-import
    block in ``reconstruct.main``, ``sdf_dataset`` or ``train_deep_sdf`` losing a name.
    """
    missing = [
        f"{path}.{name}"
        for path, names in PATHS.items()
        for name in names
        if not hasattr(importlib.import_module(path), name)
    ]
    assert missing == []
