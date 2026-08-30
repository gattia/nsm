"""
Evaluation over reconstructions: per-subject losses plus the aggregate validation entry
point.

``get_mean_errors`` moved here verbatim from ``main.py`` in the §8.0.E split (the same
pass deleted ``tune_reconstruction`` and ``compute_correlation_coefficient`` — SCOPE
§2's dead ruling). ``main.py`` re-imports it, so ``NSM.reconstruct`` and
``NSM.reconstruct.main`` both still serve it — that re-import block is public API,
pinned by ``test_reconstruct_import_compat``.
"""

import logging
from fnmatch import fnmatch

import numpy as np

# Optional (#5): every wandb use is behind an explicit request that raises when absent.
try:
    import wandb
except ImportError:
    wandb = None

from .._verbose_deprecation import honour_verbose
from .predictive_validation_class import Regress
from .utils import compute_chamfer  # , compute_assd

# Set up logger for this module
logger = logging.getLogger(__name__)


def compute_recon_loss(
    meshes,
    # orig_pts,
    orig_meshes,
    n_samples_chamfer=None,
    chamfer_norm=1,
    calc_symmetric_chamfer=False,
    calc_assd=False,
):
    """
    Computes the reconstruction loss between the predicted meshes and the ground truth meshes.

    Args:
        meshes (list): A list of predicted meshes.
        orig_meshes (list): A list of ground truth meshes. (Replaced ``orig_pts``, which is
            commented out of the signature.)
        n_samples_chamfer (int, optional): The number of samples to use for the chamfer distance calculation. Defaults to None.
        chamfer_norm (int, optional): The power to which the chamfer distance is raised. Defaults to 1.
        calc_symmetric_chamfer (bool, optional): Whether to calculate the symmetric chamfer distance. Defaults to False.
        calc_assd (bool, optional): Whether to calculate the average symmetric surface distance. Defaults to False.

    Returns:
        dict: A dictionary containing the reconstruction loss for each mesh.

    Neither list is modified, and neither are the meshes in them (#55). The ASSD branch
    used to downcast both sides to ``float32`` in place, under a comment reading "make
    sure the points for the meshes are the same types"; pymskt's ``pcu_sdf`` casts the
    query points and the mesh vertices to ``float64`` itself, so the types were never
    what reached the computation.
    """
    logger.info("Starting reconstruction loss computation")
    logger.debug("Computing loss for %s meshes", len(meshes) if isinstance(meshes, list) else 1)
    logger.debug(
        "Loss calculation settings: chamfer=%s, assd=%s", calc_symmetric_chamfer, calc_assd
    )

    result = {}

    if not isinstance(meshes, list):
        meshes = [meshes]
    if not isinstance(orig_meshes, list):
        orig_meshes = [orig_meshes]

    assert len(meshes) == len(
        orig_meshes
    ), "Number of meshes and number of original points must be equal"

    logger.debug("Processing %s mesh pairs", len(meshes))

    for mesh_idx, mesh in enumerate(meshes):
        logger.debug("Processing mesh %s/%s", mesh_idx + 1, len(meshes))

        if mesh is not None:
            pts_recon_ = mesh.point_coords
            logger.debug("Mesh %s: %s reconstructed points", mesh_idx, len(pts_recon_))
        else:
            pts_recon_ = None
            logger.warning("Mesh %s: No reconstructed mesh provided (None)", mesh_idx)

        # A subject may be missing a structure outright (SCOPE 2.5b): the latent is fitted
        # from the surfaces it has and every surface is still decoded, so the
        # reconstruction exists with nothing to score it against. Same treatment as a
        # reconstruction that did not decode.
        if orig_meshes[mesh_idx] is not None:
            xyz_orig_ = orig_meshes[mesh_idx].point_coords
            logger.debug("Mesh %s: %s original points", mesh_idx, len(xyz_orig_))
        else:
            xyz_orig_ = None
            logger.warning("Mesh %s: No original mesh provided (None)", mesh_idx)

        missing = ", ".join(
            name
            for name, points in (("reconstructed", pts_recon_), ("original", xyz_orig_))
            if points is None
        )

        if calc_symmetric_chamfer:
            logger.debug("Computing Chamfer distance for mesh %s", mesh_idx)
            # if __chamfer__ is True:
            if missing:
                chamfer_loss_ = np.nan
                logger.warning(
                    "Mesh %s: Chamfer distance set to NaN (no %s mesh)", mesh_idx, missing
                )
            else:
                chamfer_loss_ = compute_chamfer(
                    xyz_orig_, pts_recon_, num_samples=n_samples_chamfer, power=chamfer_norm
                )
                logger.debug("Mesh %s: Chamfer distance = %.6f", mesh_idx, chamfer_loss_)
            result[f"chamfer_{mesh_idx}"] = chamfer_loss_
            # elif __chamfer__ is False:
            #     raise ImportError('Cannot calculate symmetric chamfer distance without chamfer_pytorch module')

        if calc_assd:
            logger.debug("Computing ASSD for mesh %s", mesh_idx)
            if missing:
                assd_loss_ = np.nan
                logger.warning("Mesh %s: ASSD set to NaN (no %s mesh)", mesh_idx, missing)
            else:
                assd_loss_ = mesh.get_assd_mesh(orig_meshes[mesh_idx])
                logger.debug("Mesh %s: ASSD = %.6f", mesh_idx, assd_loss_)
            result[f"assd_{mesh_idx}"] = assd_loss_

    logger.info("Reconstruction loss computation completed. Computed %s loss values.", len(result))
    logger.debug("Result keys: %s", list(result.keys()))
    return result


@honour_verbose
def get_mean_errors(
    mesh_paths,
    decoders,
    latent_size,
    calc_symmetric_chamfer=False,
    calc_assd=False,
    log_wandb=False,
    num_iterations=1000,
    n_pts_per_axis=256,
    lr=5e-4,
    loss_weight=1.0,
    loss_type="l1",
    l2reg=False,
    latent_init_std=0.01,
    latent_init_mean=0.0,
    clamp_dist=None,
    latent_reg_weight=1e-4,
    n_lr_updates=2,
    lr_update_factor=10,
    convergence="num_iterations",
    convergence_patience=50,
    config=None,
    register_similarity=False,
    scale_all_meshes=True,
    model_type="deepsdf",
    verbose=False,
    objects_per_decoder=1,
    latent_optimizer_name="adam",
    get_rand_pts=False,
    n_pts_random=100000,
    sigma_rand_pts=0.01,
    n_samples_latent_recon=10000,
    max_n_samples_latent_recon=None,  # 100000,
    n_steps_sample_ramp_latent_recon=None,  # 200,
    difficulty_weight_recon=None,
    chamfer_norm=2,
    recon_func=None,
    predict_val_variables=None,
    scale_jointly=False,
    fix_mesh=True,
    device="cuda",
):
    """
    Reconstruct meshes & compute errors.

    A decoder whose mean shape has no zero level set (``NoZeroLevelSetError`` from
    ``reconstruct_mesh``) scores NaN — per-surface metrics and ``val_prediction_*``
    alike — instead of aborting, so a training run survives its own early validation
    epochs.
    """
    # Call-time import, not module-level: main.py imports this module at top for
    # compute_recon_loss, so importing .main at module scope here would be a cycle.
    # Call-time lookup also preserves the monkeypatch seam —
    # test_predictive_validation patches NSM.reconstruct.main.reconstruct_mesh.
    from .main import NoZeroLevelSetError, reconstruct_mesh

    if log_wandb and wandb is None:
        raise ImportError("log_wandb=True requires wandb, which is not installed")

    loss = {}

    reconstruct_inputs = {
        "latent_size": latent_size,
        "calc_symmetric_chamfer": calc_symmetric_chamfer,
        "calc_assd": calc_assd,
        "register_similarity": register_similarity,
        "scale_jointly": scale_jointly,
        "scale_all_meshes": scale_all_meshes,
        "return_latent": True,
        "device": device,
    }

    if model_type == "deepsdf":
        reconstruct_inputs_ = {
            "decoders": decoders,
            "log_wandb": log_wandb,
            "num_iterations": num_iterations,
            "n_pts_per_axis": n_pts_per_axis,
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
            "convergence": convergence,
            "convergence_patience": convergence_patience,
            "objects_per_decoder": objects_per_decoder,
            "verbose": verbose,
            "latent_optimizer_name": latent_optimizer_name,
            "get_rand_pts": get_rand_pts,
            "n_pts_random": n_pts_random,
            "sigma_rand_pts": sigma_rand_pts,
            "n_samples_latent_recon": n_samples_latent_recon,
            "max_n_samples_latent_recon": max_n_samples_latent_recon,
            "n_steps_sample_ramp_latent_recon": n_steps_sample_ramp_latent_recon,
            "difficulty_weight_recon": difficulty_weight_recon,
            "chamfer_norm": chamfer_norm,
            "func": recon_func,
            "fix_mesh": fix_mesh,
        }

        recon_fx = reconstruct_mesh
    else:
        raise ValueError(f'model_type must be "deepsdf", received {model_type}')

    reconstruct_inputs.update(reconstruct_inputs_)

    if predict_val_variables is not None:
        reg = Regress(list_factors=predict_val_variables, list_paths=mesh_paths)

    n_degenerate = 0
    for idx, mesh_path in enumerate(mesh_paths):
        if log_wandb is True:
            config_ = config.copy()
            config_["mesh_path"] = mesh_path
            config_["mesh_idx"] = idx
            wandb.init(
                # Set the project where this run will be logged
                project=config["project_name"],  # "diffusion-net-predict-sex",
                entity=config["entity_name"],  # "bone-modeling",
                # Track hyperparameters and run metadata
                config=config_,
                name=config["run_name"],
                tags=config["tags"],
            )
        reconstruct_inputs["path"] = mesh_path
        try:
            result_ = recon_fx(**reconstruct_inputs)
        except NoZeroLevelSetError as error:
            # The state of every model before it learns a sign change: score the
            # subject NaN and keep the validation epoch alive rather than killing the
            # training run. No latent exists for this subject, so the predictive
            # validation below reports NaN instead of regressing on fabrications.
            logger.warning("%s: %s", mesh_path, error)
            n_degenerate += 1
            n_decoders = len(decoders) if isinstance(decoders, (list, tuple)) else 1
            n_surfaces = (
                sum(objects_per_decoder)
                if isinstance(objects_per_decoder, (list, tuple))
                else objects_per_decoder * n_decoders
            )
            result_ = {"mesh": [None] * n_surfaces}
            for mesh_idx in range(n_surfaces):
                if calc_symmetric_chamfer:
                    result_[f"chamfer_{mesh_idx}"] = np.nan
                if calc_assd:
                    result_[f"assd_{mesh_idx}"] = np.nan
        if verbose is True:
            logger.debug("result_ %s", result_)

        if predict_val_variables is not None and "latent" in result_:
            reg.add_latent(result_["latent"].detach().cpu().numpy().ravel())

        for mesh_idx in range(len(result_["mesh"])):
            if calc_symmetric_chamfer:
                if idx == 0:
                    loss[f"chamfer_{mesh_idx}"] = []
                loss[f"chamfer_{mesh_idx}"].append(result_[f"chamfer_{mesh_idx}"])
            if calc_assd:
                if idx == 0:
                    loss[f"assd_{mesh_idx}"] = []
                loss[f"assd_{mesh_idx}"].append(result_[f"assd_{mesh_idx}"])

        # if a function was given - append its results.
        # setdefault, not `if idx == 0`: a degenerate subject 0 contributes no func_ keys
        # at all -- the except branch above builds its result dict by hand -- so keying
        # the list's creation on the first subject made the epoch depend on the order of
        # the validation set, and raised KeyError on the second subject when it lost.
        if recon_func is not None:
            for key, val in result_.items():
                if "func_" == key[:5]:
                    loss.setdefault(key[5:], []).append(val)

        if log_wandb is True:
            wandb.finish()

    if verbose is True:
        logger.debug("loss %s", loss)
    loss_ = {}

    if predict_val_variables is not None:
        if n_degenerate:
            # Degeneracy is a property of the decoder, so every subject failed together
            # and the regressor holds no latents. NaN is the honest score; until Aug
            # 2026 the r^2 here was computed against zero vectors (History §10).
            predictive_results = {
                f"val_prediction_{factor}": np.nan for factor in predict_val_variables
            }
        else:
            predictive_results = reg.calc_r2()
        loss_.update(predictive_results)

    for key, item in loss.items():
        logger.debug("%s %s", key, item)
        mean = np.mean(item)
        std = np.std(item)
        median = np.median(item)
        # Skip, not raise, when wandb is absent: training validation reaches here with
        # no wandb request (the metric scalars below are unaffected).
        try:
            hist = wandb.Histogram(item) if wandb is not None else None
        except ValueError:
            hist = None
        loss_[key] = mean
        loss_[f"{key}_std"] = std
        loss_[f"{key}_mean"] = mean
        loss_[f"{key}_median"] = median
        loss_[f"{key}_hist"] = hist

        if fnmatch(key, "cart_thick*_orig_mean"):
            cart_region = key.split("_")[2]
            loss_[f"cart_thick_{cart_region}_corr"] = np.corrcoef(
                loss[f"cart_thick_{cart_region}_orig_mean"],
                loss[f"cart_thick_{cart_region}_recon_mean"],
            )[0, 1]
        if fnmatch(key, "cart_thick*_mean_thick_diff"):
            cart_region = key.split("_")[2]
            loss_[f"cart_thick_{cart_region}_RMSE"] = np.sqrt(
                np.mean(np.square(loss[f"cart_thick_{cart_region}_mean_thick_diff"]))
            )

    return loss_
