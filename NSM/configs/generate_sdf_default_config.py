import json
import os

# The values below are a sanitized snapshot of the ShapeMedKnee `647_nsm_femur_v0.0.1`
# model_params_config.json -- the configuration that actually trained a shipped model --
# taken 2026-08-22 (#48). Sanitized out: run identity (EXP_NAME, run_name), machine paths
# (experiment_directory, list_mesh_paths, path_list_meshes, val_paths), wandb org fields,
# the knee-specific validation hook (recon_val_func_name) and descriptive metadata
# (bone/cartilage/tissues), plus keys the trainer derives or writes back itself
# (checkpoints, the plain *_lr_* logging keys), the removed `emd` option (#53), and
# `batch_size_latent_recon` -- dead since batching was removed from reconstruction; the
# trainers stopped reading it in the #16-class sweep.
# `mesh_names` is added -- 647 predates the requirement that multi-surface configs name
# their outputs.
config = {
    # wandb / run identity (set per run)
    "project_name": "nsm",
    "entity": None,
    "entity_name": None,
    "run_name": None,
    "tags": ["nsm"],
    # model (triplanar, as shipped)
    "model_type": "triplanar",
    "decoder_type": "single_head",
    "objects_per_decoder": 2,
    "mesh_names": ["bone", "cart"],
    "latent_size": 512,
    "conv_hidden_dims": [512, 512, 512, 512, 512],
    "conv_deep_image_size": 2,
    "conv_norm": True,
    "conv_norm_type": "layer",
    "conv_pred_sdf": False,
    "conv_start_with_mlp": True,
    # The 647 run this file snapshots predates the activation being appendable.
    "conv_activation": None,
    "sum_conv_output_features": True,
    "sdf_latent_size": 128,
    "sdf_hidden_dims": [512, 512, 512],
    "sdf_skip_connection": [4],
    "concat_latent_input": True,
    # Absent from 647's file -- which is the silent hazard KNOWN_ISSUES "padding is not
    # in the checkpoint" describes (#26). 0.1 is the constructor default every shipped
    # model effectively runs at; recorded here so new runs carry it explicitly.
    "padding": 0.1,
    "modulated": False,
    "progressive_add_depth": False,
    "layer_split": False,
    # MLP-decoder keys, carried by the shipped config for loader fallbacks
    "layer_dimensions": [512] * 8,
    "layers_with_dropout": list(range(8)),
    "dropout_prob": 0,
    "layer_latent_in": [4],
    "xyz_in_all": False,
    "latent_dropout": False,
    "weight_norm": True,
    "activation": "relu",
    "final_activation": "tanh",
    # initialization
    "seed": 52122,
    # dataset (per-surface lists: [bone, cartilage])
    "list_mesh_paths": None,
    "val_paths": None,
    "n_pts_per_object": [500000, 500000],
    "percent_near_surface": [0.45, 0.45],
    "percent_further_from_surface": [0.45, 0.45],
    "sigma_near": [0.7431352501, 0.7431352501],
    "sigma_far": [2.35, 2.35],
    "random_function": "normal",
    "equal_pos_neg": True,
    "center_all_meshes": False,
    "center_pts": False,
    "normalize_pts": False,
    "scale_all_meshes": True,
    "scale_jointly": True,
    "scale_method": "max_rad",
    "mesh_to_scale": 0,
    "reference_mesh": 0,
    "dataset_uniform_pts_buffer": 0.2,
    "multiprocessing": True,
    "n_processes": 16,
    "cache": True,
    "load_cache": True,
    "store_data_in_memory": False,
    "fix_mesh": False,
    "n_samples": None,
    "n_val": None,
    # data loading during training
    "objects_per_batch": 64,
    "batch_split": 1,
    "samples_per_object_per_batch": 17000,
    "num_data_loader_threads": 16,
    "prefetch_factor": 4,
    # training loop
    "n_epochs": 2001,
    "resume_epoch": 0,
    "checkpoint_epochs": 500,
    "additional_checkpoints": [200],
    "save_frequency": 50,
    "device": "cuda:0",
    "profiler": False,
    "verbose": True,
    "log_latent": None,
    # loss
    "enforce_minmax": True,
    "clamp_dist": 1,
    "surface_weighting": [1, 1],
    "eikonal_weight": 0.0,
    # curriculum weighting
    "sample_difficulty_weight": 0.2,
    "sample_difficulty_weight_schedule": "exponential",
    "sample_difficulty_cooldown": 200,
    "surface_accuracy_e": None,
    "surface_accuracy_schedule": "linear",
    "surface_accuracy_cooldown": None,
    "sample_difficulty_lx": None,
    "sample_difficulty_lx_schedule": "linear",
    "sample_difficulty_lx_cooldown": None,
    "sample_difficulty_lx_epsilon": 1e-4,
    # code regularization
    "code_regularization": True,
    "code_regularization_weight": 1e-4,
    "code_regularization_type_prior": "identity",
    "code_regularization_warmup": 100,
    "code_cyclic_anneal": True,
    # optimizer
    "optimizer": "AdamW",
    "weight_decay": 0.0001,
    "grad_clip": None,
    # Each entry MUST declare "Target": "model" or "latent"; entry order is ignored.
    # This annotation reproduces what 647 historically trained under (AdamW ran through
    # the positional adjust_learning_rate, so its entry 0 drove the LATENTS -- see
    # docs/KNOWN_ISSUES.md section History 1, worked example). The shipped models'
    # hyperparameters were tuned under exactly this mapping.
    "LearningRateSchedule": [
        {
            "Target": "latent",
            "Type": "Step",
            "Initial": 0.005,
            "Interval": 16.666666666666668,
            "Factor": 0.9523809523809523,
        },
        {"Target": "model", "Type": "Step", "Initial": 0.0001, "Interval": 1000, "Factor": 0.1},
    ],
    # latent codes
    "latent_bound": 10,
    "latent_init_std": 0.01,
    "latent_init_normal": True,
    "variational": False,
    # reconstruction / validation
    "num_iterations_recon": 2000,
    "lr_recon": 0.005,
    "l2reg_recon": False,
    "clamp_dist_recon": 0.1,
    "n_lr_updates_recon": 100,
    "lr_update_factor_recon": 1.1,
    "chamfer": True,
    "assd": True,
    "convergence_type_recon": "recon_loss",
    "convergence_patience_recon": 50,
    "get_rand_pts_recon": False,
    "n_pts_random_recon": 100000,
    "sigma_rand_pts_recon": 0.01,
    "n_samples_latent_recon": 20000,
    "max_n_samples_latent_recon": None,
    "n_steps_sample_ramp_latent_recon": None,
    "fix_mesh_recon": False,
    # saving results
    "experiment_directory": None,
}

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "default_config.json"
)


def write_default_config(path=DEFAULT_CONFIG_PATH):
    """Write ``config`` to ``path``. Regenerates the shipped ``default_config.json``."""
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    return path


if __name__ == "__main__":
    # Guarded, and writing next to this module rather than to the caller's cwd. Both used
    # to be otherwise: the write ran at IMPORT time and targeted "./", so merely importing
    # this module dropped a default_config.json into whatever directory you were in, while
    # the shipped copy it was meant to regenerate went untouched unless you happened to be
    # cd'd into NSM/configs.
    print(f"wrote {write_default_config()}")
