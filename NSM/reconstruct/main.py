import copy
import logging
import os
import sys
import time
from fnmatch import fnmatch

import numpy as np
import pymskt as mskt
import torch

# Optional (#5): every wandb use is behind an explicit request that raises when absent.
try:
    import wandb
except ImportError:
    wandb = None

from NSM.datasets import read_mesh_get_sampled_pts, read_meshes_get_sampled_pts
from NSM.datasets.sdf_dataset import combine_meshes
from NSM.losses import EIKONAL_UNSUPPORTED, eikonal_loss
from NSM.mesh import create_mesh_adaptive

# The .latent_fit, .wandb_logging and .recon_evaluation imports re-serve definitions
# moved out in the §8.0.C and §8.0.E splits. Public API, not scaffolding:
# ``NSM.reconstruct`` (star-import of this module) and ``NSM.reconstruct.main`` are both
# live import paths, so every name that lived here stays importable from both. Frozen by
# testing/NSM/reconstruct/test_reconstruct_import_compat.py.
from .latent_fit import (  # noqa: F401
    latent_norm_penalty,
    project_latent,
    reconstruct_latent,
    reconstruct_latent_decoders_type_check,
    reconstruct_latent_get_lr_update_freq,
    reconstruct_latent_preprocess_sdf_gt,
    reconstruct_latent_pts_surface_type_check,
    reconstruct_latent_sdf_gt_type_check,
)
from .predictive_validation_class import Regress
from .recon_evaluation import compute_recon_loss, get_mean_errors  # noqa: F401
from .utils import adjust_learning_rate
from .wandb_logging import _process_meshes_for_wandb, prepare_results_for_wandb  # noqa: F401

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NoZeroLevelSetError(RuntimeError):
    """
    The decoder's mean shape (zero latent) has no surface, so registration to the mean
    -- and everything after it -- cannot run. This is the state of every model before it
    has learned a sign change, not an exotic error path.

    Until Aug 2026 this state returned a plausible-looking result instead of raising:
    ``mesh`` of Nones, NaN metrics, and the *untouched zero* ``mean_latent`` under
    ``"latent"``, with every other requested key dropped (#29;
    ``docs/KNOWN_ISSUES.md`` § History 10). ``get_mean_errors`` catches this error and
    scores the subject NaN, so a validation epoch survives an under-trained model.
    """


def reconstruct_mesh(
    path,
    decoders,
    latent_size,
    num_iterations=1000,
    lr=5e-4,
    batch_size=32**3,
    loss_weight=1.0,
    loss_type="l1",
    l2reg=False,
    latent_init_std=0.01,
    latent_init_mean=0.0,
    clamp_dist=None,
    latent_reg_weight=1e-4,
    n_lr_updates=2,
    lr_update_factor=10,
    calc_symmetric_chamfer=False,
    calc_assd=False,
    n_pts_per_axis=256,
    log_wandb=False,
    return_latent=False,
    convergence="num_iterations",
    convergence_patience=50,
    scale_jointly=False,
    register_similarity=False,
    n_pts_per_axis_mean_mesh=128,
    scale_all_meshes=True,  # whether when scaling a model it should be on all points in all meshes or not
    mesh_to_scale=0,  # int index, or list of indices to combine for joint registration
    decoder_to_scale=0,  # PRETTY MUCH ASSUME ALWAYS SCALING FIRST DECODER
    scale_method="max_rad",
    verbose=False,
    objects_per_decoder=1,
    latent_optimizer_name="adam",
    get_rand_pts=False,
    n_pts_random=100000,
    sigma_rand_pts=0.001,
    seed=None,
    n_samples_chamfer=None,
    n_samples_latent_recon=10000,
    max_n_samples_latent_recon=None,  # 100000,
    n_steps_sample_ramp_latent_recon=None,  # 200,
    difficulty_weight_recon=None,
    chamfer_norm=2,
    func=None,
    fix_mesh=True,
    return_registration_params=False,
    return_timing=False,
    device="cuda",
    recon_grid_origin=1.0,
    latent_norm=None,
    # Hybrid optimizer parameters
    hybrid_optimizer=False,  # Whether to use Adam + LBFGS hybrid approach
    adam_iterations=None,  # Number of Adam iterations (if None, uses num_iterations)
    lbfgs_iterations=None,  # Number of LBFGS iterations (if None, no LBFGS phase)
    lbfgs_lr=1.0,  # Learning rate for LBFGS phase
    lbfgs_max_iter=20,  # Max iterations per LBFGS step
    lbfgs_history_size=100,  # LBFGS history size
    # Soft norm constraint parameters (alternative to hard projection)
    use_soft_norm_constraint=True,  # Use soft penalty instead of hard projection
    norm_penalty_weight=1e-3,  # Weight for norm penalty term
    norm_penalty_type="quadratic",  # "quadratic", "huber", or "barrier"
    **kwargs,
):
    """
    Reconstructs mesh at path using decoders.

    `seed` seeds the point sampling when `get_rand_pts` is True; None leaves it unseeded.
    `n_pts_random` is the draw size per surface on that path (honoured since the #16
    fix — before it the samplers' 200,000-point default ran regardless; § History 9).

    NOTES:
    Assumes that length of path = sum(objects_per_decoder)
    That is,
        path0_mesh = decoder0_mesh0
        path1_mesh = decoder0_mesh1 OR decoder1_mesh0
        etc.

    Returns:
        The return TYPE depends on the flags — a deliberate convenience switch:
        - A dict whenever any of calc_symmetric_chamfer, calc_assd, return_latent,
          func, return_registration_params or return_timing is set. Key "mesh" holds
          the ordered mesh list (order = the surface-identity contract above); the
          other keys appear per flag. Every first-party caller takes this branch.
        - Otherwise, the bare list of meshes.

    Raises:
        NoZeroLevelSetError: with `register_similarity` or `scale_jointly` set, when
            the decoder's mean shape has no surface (see the exception's docstring).
    """

    if log_wandb and wandb is None:
        raise ImportError("log_wandb=True requires wandb, which is not installed")

    # warning batch_size_latent_recon is deprecated
    if "batch_size_latent_recon" in kwargs:
        print(
            "Warning: batch_size_latent_recon is deprecated and will be removed in future versions. "
            "Batch processing has been simplified and now processes all data at once for better performance."
        )

    # Check if path is a single mesh or a list of meshes & set multi_object flag
    if isinstance(path, str):
        multi_object = False
    elif isinstance(path, (list, tuple)):
        multi_object = True
        # appropriately set the number of random points for multi-object reconstructions
        if isinstance(n_pts_random, (int, float)):
            n_pts_random = [
                n_pts_random,
            ] * len(path)
        if isinstance(sigma_rand_pts, (int, float)):
            sigma_rand_pts = [
                sigma_rand_pts,
            ] * len(path)
    else:
        raise ValueError("path must be a string or a list/tuple of strings")

    # make decoders a list so that it can be iterated over (make agnostic to number of decoders)
    if not isinstance(decoders, (list, tuple)):
        decoders = [
            decoders,
        ]

    # make objects_per_decoder a list so that it can be iterated over
    if isinstance(objects_per_decoder, (list, tuple)):
        assert len(objects_per_decoder) == len(
            decoders
        ), "If objects_per_decoder is a list, it must be the same length as decoders"
    elif isinstance(objects_per_decoder, int):
        # if single int, assume that all decoders have the same number of objects
        objects_per_decoder = [
            objects_per_decoder,
        ] * len(decoders)

    tic = time.time()

    if (scale_jointly) or (register_similarity is True):
        # if register first, then register new mesh to the mean of the decoder (zero latent vector)
        # create mean mesh of only mesh, or "mesh_to_scale" if more than one.
        mean_latent = torch.zeros(1, latent_size)
        # create mean mesh, assume that using decoder_0 & mesh_0, but
        # technically this can be specified.
        mean_mesh = create_mesh_adaptive(
            decoder=decoders[decoder_to_scale].to(device),
            latent_vector=mean_latent.to(device),
            n_pts_per_axis=n_pts_per_axis_mean_mesh,
            search_bounds=(-recon_grid_origin, recon_grid_origin),
            objects=objects_per_decoder[decoder_to_scale],
            batch_size=batch_size,
            verbose=verbose,
            device=device,
        )

        if objects_per_decoder[decoder_to_scale] > 1:
            if verbose is True:
                print(f"Mean mesh is idx: {mesh_to_scale}")
            # Support multi-surface mean mesh creation
            if isinstance(mesh_to_scale, (list, tuple)):
                if verbose is True:
                    print(f"Combining mean meshes for multi-surface registration: {mesh_to_scale}")
                # Combine multiple mean meshes for registration
                mean_mesh = combine_meshes(mean_mesh, mesh_to_scale)
            else:
                # Single mesh selection (original behavior)
                mean_mesh = mean_mesh[mesh_to_scale]

        if mean_mesh is None:
            raise NoZeroLevelSetError(
                f"The decoder's mean shape has no zero level set: the zero-latent SDF "
                f"never changes sign on the {n_pts_per_axis_mean_mesh}^3 grid. Either "
                f"the model has not learned a sign change yet -- the state of every "
                f"model early in training -- or the grid is too coarse for its surface."
            )
    else:
        mean_mesh = None

    toc = time.time()
    time_load_mean = toc - tic
    tic = time.time()
    if verbose is True:
        print(f"Loaded mean mesh in {time_load_mean:.2f} seconds")

    # read in mesh(es) and get sampled points for fitting decoder too
    # handle single or multiple meshes appropriately.
    if multi_object is False:
        result_ = read_mesh_get_sampled_pts(
            path,
            sigma=sigma_rand_pts,
            center_pts=not scale_jointly,
            norm_pts=not scale_jointly,
            scale_method=scale_method,
            get_random=get_rand_pts,
            register_to_mean_first=True if register_similarity else False,
            mean_mesh=mean_mesh if register_similarity else None,
            n_pts=n_pts_random,
            include_surf_in_pts=get_rand_pts,
            fix_mesh=fix_mesh,
            seed=seed,
        )
    elif multi_object is True:
        result_ = read_meshes_get_sampled_pts(
            paths=path,
            sigma=sigma_rand_pts,
            center_pts=not scale_jointly,
            norm_pts=not scale_jointly,
            scale_all_meshes=scale_all_meshes,
            mesh_to_scale=mesh_to_scale,
            scale_method=scale_method,
            get_random=get_rand_pts,
            register_to_mean_first=True if register_similarity else False,
            mean_mesh=mean_mesh,
            n_pts=n_pts_random,
            include_surf_in_pts=get_rand_pts,
            fix_mesh=fix_mesh,
            seed=seed,
        )
    else:
        raise ValueError("multi_object must be True or False")

    xyz = result_["pts"]
    sdf_gt = result_["sdf"]
    pts_surface = result_["pts_surface"]

    # ensure all data are torch tensors and have the correct shape
    if not isinstance(xyz, torch.Tensor):
        xyz = torch.from_numpy(xyz).float()
    if multi_object is True:
        for sdf_idx, sdf_gt_ in enumerate(sdf_gt):
            if sdf_gt_ is None:
                if verbose is True:
                    print(f"sdf_gt[{sdf_idx}] is None, skipping surface {sdf_idx}")
                continue
            if not isinstance(sdf_gt_, torch.Tensor):
                sdf_gt[sdf_idx] = torch.from_numpy(sdf_gt_).float()

            if len(sdf_gt[sdf_idx].shape) == 1:
                sdf_gt[sdf_idx] = sdf_gt[sdf_idx].unsqueeze(1)
    elif multi_object is False:
        if not isinstance(sdf_gt, torch.Tensor):
            sdf_gt = torch.from_numpy(sdf_gt).float()

        if len(sdf_gt.shape) == 1:
            sdf_gt = sdf_gt.unsqueeze(1)

    toc = time.time()
    time_load_mesh = toc - tic
    if verbose is True:
        print(f"Loaded mesh in {time_load_mesh:.2f} seconds")

    tic = time.time()

    # FIT THE LATENT CODE TO THE MESH
    # specify general reconstruction parameters that apply to
    # all recon methods.
    reconstruct_inputs = {
        "decoders": decoders,
        "num_iterations": num_iterations,
        "latent_size": latent_size,
        "sdf_gt": sdf_gt,
        "xyz": xyz,
        "lr": lr,
        "loss_weight": loss_weight,
        "loss_type": loss_type,
        "l2reg": l2reg,
        "latent_init_std": latent_init_std,
        "latent_init_mean": latent_init_mean,
        "clamp_dist": clamp_dist,
        "latent_reg_weight": latent_reg_weight,
        "n_lr_updates": n_lr_updates,
        "lr_update_factor": lr_update_factor,
        "log_wandb": log_wandb,
        "convergence": convergence,
        "convergence_patience": convergence_patience,
        "verbose": verbose,
        # "max_batch_size" parameter removed - now handled automatically
        "optimizer_name": latent_optimizer_name,
        "n_samples": n_samples_latent_recon,
        "difficulty_weight": difficulty_weight_recon,
        "pts_surface": pts_surface,
        "max_n_samples": max_n_samples_latent_recon,
        "n_steps_sample_ramp": n_steps_sample_ramp_latent_recon,
        "device": device,
        "latent_norm": latent_norm,
        # Hybrid optimizer parameters
        "hybrid_optimizer": hybrid_optimizer,
        "adam_iterations": adam_iterations,
        "lbfgs_iterations": lbfgs_iterations,
        "lbfgs_lr": lbfgs_lr,
        "lbfgs_max_iter": lbfgs_max_iter,
        "lbfgs_history_size": lbfgs_history_size,
        # Soft norm constraint parameters
        "use_soft_norm_constraint": use_soft_norm_constraint,
        "norm_penalty_weight": norm_penalty_weight,
        "norm_penalty_type": norm_penalty_type,
    }

    loss, latent = reconstruct_latent(**reconstruct_inputs)

    toc = time.time()
    time_recon_latent = toc - tic
    if verbose is True:
        print(f"Reconstructed latent in {time_recon_latent:.2f} seconds")
    tic = time.time()

    if verbose is True:
        print(result_["icp_transform"])

    # create mesh(es) from latent
    meshes = []
    for decoder_idx, decoder in enumerate(decoders):
        # pass alignment parameters to return mesh to original position
        # pass number of objects in case decoder is a multi-object decoder
        mesh = create_mesh_adaptive(
            decoder=decoder.to(device),
            latent_vector=latent.to(device),
            n_pts_per_axis=n_pts_per_axis,
            search_bounds=(-recon_grid_origin, recon_grid_origin),
            voxel_origin=(-recon_grid_origin, -recon_grid_origin, -recon_grid_origin),
            voxel_size=recon_grid_origin * 2 / (n_pts_per_axis - 1),
            path_original_mesh=None,
            offset=result_["center"],
            scale=result_["scale"],
            icp_transform=result_["icp_transform"],
            objects=objects_per_decoder[decoder_idx],
            verbose=verbose,
            device=device,
            batch_size=batch_size,
        )
        if objects_per_decoder[decoder_idx] > 1:
            # append sequentially so they match the order of meshes at "path"
            for mesh_ in mesh:
                meshes.append(mesh_)
        else:
            meshes.append(mesh)

    toc = time.time()
    time_create_mesh = toc - tic
    if verbose is True:
        print(f"Created mesh in {time_create_mesh:.2f} seconds")
    tic = time.time()

    if func is not None:
        func_results = func(result_["orig_mesh"], meshes)  # original result, then reconstruction.

    toc = time.time()
    time_calc_recon_funcs = toc - tic
    if verbose is True:
        print(f"metrics in {time_calc_recon_funcs:.2f} seconds")
    tic = time.time()

    if (
        calc_symmetric_chamfer
        or calc_assd
        or return_latent
        or (func is not None)
        or return_registration_params
        or return_timing
    ):
        result = {"mesh": meshes}
        result["orig_mesh"] = result_["orig_mesh"]

        if calc_symmetric_chamfer or calc_assd:
            print("length of meshes: ", len(meshes))
            print("length of orig_mesh: ", len(result_["orig_mesh"]))
            result_recon_metrics = compute_recon_loss(
                meshes=meshes,
                orig_meshes=result_["orig_mesh"],
                # orig_pts=result_['orig_pts'],
                n_samples_chamfer=n_samples_chamfer,
                chamfer_norm=chamfer_norm,
                calc_symmetric_chamfer=calc_symmetric_chamfer,
                calc_assd=calc_assd,
            )
            print("finished computing recon loss")
            toc = time.time()
            time_calc_recon_loss = toc - tic
            if verbose is True:
                print(f"metrics in {time_calc_recon_loss:.2f} seconds")

            result.update(result_recon_metrics)

        if return_latent:
            result["latent"] = latent

        if func is not None:
            result.update(func_results)

        if return_timing:
            result["time_load_mean"] = time_load_mean
            result["time_load_mesh"] = time_load_mesh
            result["time_recon_latent"] = time_recon_latent
            result["time_create_mesh"] = time_create_mesh
            result["time_calc_recon_funcs"] = time_calc_recon_funcs

        if log_wandb is True:
            # Prepare and log results to wandb with 3D point cloud visualization
            result_wandb = prepare_results_for_wandb(result, verbose=verbose)
            wandb.log(result_wandb)
            print("done wandb stuff")

        if return_registration_params:
            result["icp_transform"] = result_["icp_transform"]
            result["center"] = result_["center"]
            result["scale"] = result_["scale"]

        return result
    else:
        return meshes
