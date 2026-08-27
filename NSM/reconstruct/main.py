import logging
import time
from contextlib import contextmanager

import torch

# Optional (#5): every wandb use is behind an explicit request that raises when absent.
try:
    import wandb
except ImportError:
    wandb = None

from NSM.datasets import read_mesh_get_sampled_pts, read_meshes_get_sampled_pts
from NSM.datasets.sdf_dataset import combine_meshes
from NSM.mesh import create_mesh_adaptive

from .._verbose_deprecation import honour_verbose

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
from .recon_evaluation import compute_recon_loss, get_mean_errors  # noqa: F401
from .wandb_logging import _process_meshes_for_wandb, prepare_results_for_wandb  # noqa: F401

# Unused here and re-exported deliberately: the *leaked* half of the import contract
# test_reconstruct_import_compat.py freezes, so unleaking one is changelogged rather than
# tidied. Marked because `.flake8` project-ignores F401 and a reader cannot otherwise tell
# these from the five dead imports deleted alongside them (`KNOWN_ISSUES` § Open).
from fnmatch import fnmatch  # noqa: F401  isort:skip
from NSM.losses import EIKONAL_UNSUPPORTED, eikonal_loss  # noqa: F401  isort:skip
from .predictive_validation_class import Regress  # noqa: F401  isort:skip
from .utils import adjust_learning_rate  # noqa: F401  isort:skip

logger = logging.getLogger(__name__)


class _StageTimings(dict):
    """Wall-clock per stage of a reconstruction, keyed as ``return_timing`` returns it.

    A ``dict``, not a wrapper around one, so ``return_timing`` is
    ``result.update(timings)``: a stage that times itself is a stage that is returned.
    Under the ``tic``/``toc`` pairs this replaces those were two unrelated edits 90 lines
    apart, and ``time_calc_recon_loss`` never got the second one.
    """

    @contextmanager
    def stage(self, name, description):
        """Time the block, record it as ``time_<name>``, and log it at ``debug``.

        Records however the block is left, so a stage that raises still reports its time.
        """
        start = time.time()
        try:
            yield
        finally:
            self[f"time_{name}"] = time.time() - start
            logger.debug("%s in %.2f seconds", description, self[f"time_{name}"])


#: The only keyword ``reconstruct_mesh`` takes without naming it. kneepipeline passes it
#: on every fit, so a refusal must not refuse this one; it is warned about where it is read.
_DEPRECATED_KWARGS = frozenset({"batch_size_latent_recon"})


def _refuse_unknown_kwargs(kwargs):
    """Raise on any keyword ``reconstruct_mesh`` neither names nor deprecates.

    ``**kwargs`` used to swallow them, so a misspelling among 58 near-synonymous parameter
    names ran with the intended parameter's default and said nothing at all.
    """
    unknown = sorted(set(kwargs) - _DEPRECATED_KWARGS)
    if unknown:
        raise TypeError(
            "reconstruct_mesh() got unexpected keyword arguments: "
            + ", ".join(repr(name) for name in unknown)
        )


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


def _build_reference_mesh(
    *,
    decoders,
    decoder_to_scale,
    mesh_to_scale,
    objects_per_decoder,
    latent_size,
    n_pts_per_axis,
    recon_grid_origin,
    batch_size,
    device,
    verbose,
):
    """The decoder's mean shape -- its zero-latent surface -- for a subject to register to.

    ``mesh_to_scale`` selects one surface, or a list of indices combines several, when the
    decoder outputs more than one. Call it only when the subject will actually be
    registered: ``mean_mesh`` has exactly one reader in either sampler, under
    ``register_to_mean_first``.

    Raises:
        NoZeroLevelSetError: the zero-latent SDF never changes sign on the grid.
    """
    mean_latent = torch.zeros(1, latent_size)
    mean_mesh = create_mesh_adaptive(
        decoder=decoders[decoder_to_scale].to(device),
        latent_vector=mean_latent.to(device),
        n_pts_per_axis=n_pts_per_axis,
        search_bounds=(-recon_grid_origin, recon_grid_origin),
        objects=objects_per_decoder[decoder_to_scale],
        batch_size=batch_size,
        verbose=verbose,
        device=device,
    )

    if objects_per_decoder[decoder_to_scale] > 1:
        logger.info("Mean mesh is idx: %s", mesh_to_scale)
        if isinstance(mesh_to_scale, (list, tuple)):
            logger.info("Combining mean meshes for multi-surface registration: %s", mesh_to_scale)
            mean_mesh = combine_meshes(mean_mesh, mesh_to_scale)
        else:
            mean_mesh = mean_mesh[mesh_to_scale]

    if mean_mesh is None:
        raise NoZeroLevelSetError(
            f"The decoder's mean shape has no zero level set: the zero-latent SDF "
            f"never changes sign on the {n_pts_per_axis}^3 grid. Either "
            f"the model has not learned a sign change yet -- the state of every "
            f"model early in training -- or the grid is too coarse for its surface."
        )
    return mean_mesh


def _sample_subject(
    *,
    path,
    multi_object,
    mean_mesh,
    register_to_mean,
    scale_jointly,
    scale_all_meshes,
    mesh_to_scale,
    scale_method,
    get_rand_pts,
    n_pts_random,
    sigma_rand_pts,
    fix_mesh,
    seed,
):
    """Read the subject's surfaces, register and normalize them, sample SDF points.

    Returns the reader's own result dict with ``"pts"`` and ``"sdf"`` coerced to the float
    tensors of the rank ``reconstruct_latent`` requires; every other key is the reader's,
    untouched. A ``None`` entry in a multi-surface ``"sdf"`` is a surface this subject does
    not have -- left ``None`` and warned about, since the decoder still has an output for
    it and the caller needs to know which one went unfitted.
    """
    # Written once: the two readers take eleven of these thirteen identically, and the
    # two lists drifting apart is how the `mean_mesh` asymmetry fixed in this slice
    # arrived -- one call passed `mean_mesh if register_similarity else None` and the
    # other passed `mean_mesh`.
    shared = dict(
        sigma=sigma_rand_pts,
        center_pts=not scale_jointly,
        norm_pts=not scale_jointly,
        scale_method=scale_method,
        get_random=get_rand_pts,
        register_to_mean_first=register_to_mean,
        mean_mesh=mean_mesh,
        n_pts=n_pts_random,
        include_surf_in_pts=get_rand_pts,
        fix_mesh=fix_mesh,
        seed=seed,
    )
    if multi_object is False:
        result = read_mesh_get_sampled_pts(path, **shared)
    else:
        result = read_meshes_get_sampled_pts(
            paths=path, scale_all_meshes=scale_all_meshes, mesh_to_scale=mesh_to_scale, **shared
        )

    xyz = result["pts"]
    sdf_gt = result["sdf"]

    if not isinstance(xyz, torch.Tensor):
        xyz = torch.from_numpy(xyz).float()
    if multi_object is True:
        for sdf_idx, sdf_gt_ in enumerate(sdf_gt):
            if sdf_gt_ is None:
                logger.warning("sdf_gt[%s] is None, skipping surface %s", sdf_idx, sdf_idx)
                continue
            if not isinstance(sdf_gt_, torch.Tensor):
                sdf_gt[sdf_idx] = torch.from_numpy(sdf_gt_).float()

            if len(sdf_gt[sdf_idx].shape) == 1:
                sdf_gt[sdf_idx] = sdf_gt[sdf_idx].unsqueeze(1)
    else:
        if not isinstance(sdf_gt, torch.Tensor):
            sdf_gt = torch.from_numpy(sdf_gt).float()

        if len(sdf_gt.shape) == 1:
            sdf_gt = sdf_gt.unsqueeze(1)

    result["pts"] = xyz
    result["sdf"] = sdf_gt
    return result


def _assemble_result(
    *,
    meshes,
    sampled,
    latent,
    func,
    func_results,
    timings,
    calc_symmetric_chamfer,
    calc_assd,
    n_samples_chamfer,
    chamfer_norm,
    return_latent,
    return_registration_params,
    return_timing,
    log_wandb,
    verbose,
):
    """``reconstruct_mesh``'s return value, in whichever of its two forms the flags ask for.

    The bare mesh list when no flag is set, otherwise a dict where ``"mesh"`` and
    ``"orig_mesh"`` always appear and every other key answers one flag -- the contract
    ``reconstruct_mesh``'s docstring states. The *type* switch lives here with the keys
    rather than at the call site deliberately: two places listing the same six flags is
    how ``time_calc_recon_loss`` came to be measured for years and returned by nothing.
    """
    if not (
        calc_symmetric_chamfer
        or calc_assd
        or return_latent
        or (func is not None)
        or return_registration_params
        or return_timing
    ):
        return meshes

    result = {"mesh": meshes, "orig_mesh": sampled["orig_mesh"]}

    if calc_symmetric_chamfer or calc_assd:
        logger.debug("length of meshes:  %s", len(meshes))
        logger.debug("length of orig_mesh:  %s", len(sampled["orig_mesh"]))
        with timings.stage("calc_recon_loss", "Computed the recon loss"):
            result_recon_metrics = compute_recon_loss(
                meshes=meshes,
                orig_meshes=sampled["orig_mesh"],
                n_samples_chamfer=n_samples_chamfer,
                chamfer_norm=chamfer_norm,
                calc_symmetric_chamfer=calc_symmetric_chamfer,
                calc_assd=calc_assd,
            )
            logger.debug("finished computing recon loss")

        result.update(result_recon_metrics)

    if return_latent:
        result["latent"] = latent

    if func is not None:
        result.update(func_results)

    if return_timing:
        # Every stage that timed itself, including the optional ones. A hand-kept list
        # here is what dropped time_calc_recon_loss.
        result.update(timings)

    if log_wandb is True:
        # Prepare and log results to wandb with 3D point cloud visualization
        result_wandb = prepare_results_for_wandb(result, verbose=verbose)
        wandb.log(result_wandb)
        logger.debug("done wandb stuff")

    if return_registration_params:
        result["icp_transform"] = sampled["icp_transform"]
        result["center"] = sampled["center"]
        result["scale"] = sampled["scale"]

    return result


@honour_verbose
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
        NoZeroLevelSetError: with `register_similarity` set, when the decoder's mean
            shape has no surface (see the exception's docstring). `scale_jointly` does
            not reach it -- it did until Aug 2026, over a mean mesh it never consulted.
    """

    _refuse_unknown_kwargs(kwargs)

    if log_wandb and wandb is None:
        raise ImportError("log_wandb=True requires wandb, which is not installed")

    # warning batch_size_latent_recon is deprecated
    if "batch_size_latent_recon" in kwargs:
        logger.warning(
            "batch_size_latent_recon is deprecated and will be removed in future versions. Batch processing has been simplified and now processes all data at once for better performance."
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

    # Read once. The build below used to test `is True` while the forward to the samplers
    # tested truthiness, so `register_similarity=1` skipped the build and then asked the
    # sampler to register to the mesh that skipping had not made.
    register_to_mean = bool(register_similarity)

    timings = _StageTimings()

    with timings.stage("load_mean", "Loaded mean mesh"):
        mean_mesh = (
            _build_reference_mesh(
                decoders=decoders,
                decoder_to_scale=decoder_to_scale,
                mesh_to_scale=mesh_to_scale,
                objects_per_decoder=objects_per_decoder,
                latent_size=latent_size,
                n_pts_per_axis=n_pts_per_axis_mean_mesh,
                recon_grid_origin=recon_grid_origin,
                batch_size=batch_size,
                device=device,
                verbose=verbose,
            )
            if register_to_mean
            else None
        )

    with timings.stage("load_mesh", "Loaded mesh"):
        sampled = _sample_subject(
            path=path,
            multi_object=multi_object,
            mean_mesh=mean_mesh,
            register_to_mean=register_to_mean,
            scale_jointly=scale_jointly,
            scale_all_meshes=scale_all_meshes,
            mesh_to_scale=mesh_to_scale,
            scale_method=scale_method,
            get_rand_pts=get_rand_pts,
            n_pts_random=n_pts_random,
            sigma_rand_pts=sigma_rand_pts,
            fix_mesh=fix_mesh,
            seed=seed,
        )

    with timings.stage("recon_latent", "Reconstructed latent"):
        # FIT THE LATENT CODE TO THE MESH
        # specify general reconstruction parameters that apply to
        # all recon methods.
        reconstruct_inputs = {
            "decoders": decoders,
            "num_iterations": num_iterations,
            "latent_size": latent_size,
            "sdf_gt": sampled["sdf"],
            "xyz": sampled["pts"],
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
            "pts_surface": sampled["pts_surface"],
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

    logger.debug("icp_transform: %s", sampled["icp_transform"])

    with timings.stage("create_mesh", "Created mesh"):
        # One mesh per surface, back in the subject's frame: the sampler's offset, scale
        # and icp_transform applied in reverse. The order is the surface-identity contract
        # above -- decoder by decoder, and within a multi-surface decoder in its own output
        # order, so the list lines up with `path`.
        meshes = []
        for decoder_idx, decoder in enumerate(decoders):
            mesh = create_mesh_adaptive(
                decoder=decoder.to(device),
                latent_vector=latent.to(device),
                n_pts_per_axis=n_pts_per_axis,
                search_bounds=(-recon_grid_origin, recon_grid_origin),
                voxel_origin=(-recon_grid_origin, -recon_grid_origin, -recon_grid_origin),
                voxel_size=recon_grid_origin * 2 / (n_pts_per_axis - 1),
                path_original_mesh=None,
                offset=sampled["center"],
                scale=sampled["scale"],
                icp_transform=sampled["icp_transform"],
                objects=objects_per_decoder[decoder_idx],
                verbose=verbose,
                device=device,
                batch_size=batch_size,
            )
            if objects_per_decoder[decoder_idx] > 1:
                meshes.extend(mesh)
            else:
                meshes.append(mesh)

    func_results = None
    with timings.stage("calc_recon_funcs", "Ran the recon functions"):
        if func is not None:
            # original result, then reconstruction.
            func_results = func(sampled["orig_mesh"], meshes)

    return _assemble_result(
        meshes=meshes,
        sampled=sampled,
        latent=latent,
        func=func,
        func_results=func_results,
        timings=timings,
        calc_symmetric_chamfer=calc_symmetric_chamfer,
        calc_assd=calc_assd,
        n_samples_chamfer=n_samples_chamfer,
        chamfer_norm=chamfer_norm,
        return_latent=return_latent,
        return_registration_params=return_registration_params,
        return_timing=return_timing,
        log_wandb=log_wandb,
        verbose=verbose,
    )
