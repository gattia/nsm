"""
Latent-code optimization: fit a latent so the decoder(s) reproduce observed SDF samples.

Moved verbatim from ``main.py`` in the §8.0.C split. ``main.py`` re-imports every name
here, so ``NSM.reconstruct`` and ``NSM.reconstruct.main`` both still serve them — that
re-import block is public API, pinned by ``test_reconstruct_import_compat``.
"""

import logging

import numpy as np
import torch

# Optional (#5): every wandb use is behind an explicit request that raises when absent.
try:
    import wandb
except ImportError:
    wandb = None

from NSM.losses import EIKONAL_UNSUPPORTED, eikonal_loss

from .._verbose_deprecation import honour_verbose
from .utils import adjust_learning_rate, refuse_unknown_kwargs

logger = logging.getLogger(__name__)


@honour_verbose
def reconstruct_latent_sdf_gt_type_check(sdf_gt, verbose=False):
    if type(sdf_gt) in (torch.Tensor, np.ndarray):
        sdf_gt = [sdf_gt]
    elif type(sdf_gt) in (list, tuple):
        pass
    elif type(sdf_gt) in (str,):
        raise Exception(
            "Must provided xyz/sdf from mesh - resconstruct latent will not load mesh"
            + "from file. Try reconstruct_mesh instead."
        )
    else:
        raise Exception("Invalid sdf_gt type")

    logger.debug("\tsdf_gt len: %s", len(sdf_gt))
    for sdf in sdf_gt:
        if sdf is not None:
            logger.debug("\tsdf shape: %s", sdf.shape)
            logger.debug("\tsdf type: %s", type(sdf))
        else:
            logger.debug("\tsdf is None")

    return sdf_gt


@honour_verbose
def reconstruct_latent_pts_surface_type_check(pts_surface, verbose=False, device="cuda"):
    if isinstance(pts_surface, (list, tuple)):
        pts_surface = torch.tensor(pts_surface).to(device)
    elif isinstance(pts_surface, np.ndarray):
        pts_surface = torch.from_numpy(pts_surface).to(device)
    elif isinstance(pts_surface, torch.Tensor):
        pass
    else:
        raise ValueError("pts_surface must be list, tuple, np.ndarray, or torch.Tensor")

    logger.debug("\tpts_surface shape: %s", pts_surface.shape)
    logger.debug("\tpts_surface type: %s", type(pts_surface))
    return pts_surface


def reconstruct_latent_decoders_type_check(decoders):
    if isinstance(decoders, torch.nn.Module):
        decoders = [
            decoders,
        ]
    elif isinstance(decoders, (list, tuple)):
        for decoder in decoders:
            if not isinstance(decoder, torch.nn.Module):
                raise ValueError("decoders must be a list of torch.nn.Module")
    else:
        raise ValueError("decoders must be a torch.nn.Module or a list of torch.nn.Module")
    return decoders


def reconstruct_latent_get_lr_update_freq(n_lr_updates, num_iterations):
    # Setup n LR updates
    if (n_lr_updates == 0) or (n_lr_updates is None):
        adjust_lr_every = num_iterations + 1
    else:
        adjust_lr_every = max(1, num_iterations // n_lr_updates)  # Ensure it's never 0

    return adjust_lr_every


@honour_verbose
def reconstruct_latent_preprocess_sdf_gt(sdf_gt, clamp_dist, device="cuda", verbose=False):
    # Set a clamp (maximum) distance to "model"
    for sdf_idx, sdf in enumerate(sdf_gt):
        if sdf is None:
            logger.warning("sdf_gt[%s] is None, skipping surface %s", sdf_idx, sdf_idx)
            continue
        if clamp_dist is not None:
            sdf = torch.clamp(sdf, -clamp_dist, clamp_dist)
        # Move to GPU
        sdf_gt[sdf_idx] = sdf.to(device)
    return sdf_gt


def project_latent(latent, latent_norm):
    """Clamp the latent's L2 norm into [min, max] by rescaling it IN PLACE; returns None.

    Not legacy: this is the live path whenever ``latent_norm`` is set and
    ``use_soft_norm_constraint=False``, under any optimizer. With
    ``use_soft_norm_constraint=True`` (the default) ``latent_norm_penalty`` is used
    instead. Production never sets ``latent_norm``, so neither branch runs today.
    """
    if isinstance(latent_norm, (list, tuple)):
        if len(latent_norm) != 2:
            raise ValueError("latent_norm must be a single value or a tuple/list of two values")
        min_, max_ = latent_norm
    elif isinstance(latent_norm, (int, float)):
        min_ = max_ = latent_norm
    else:
        raise ValueError("latent_norm must be a single value or a tuple/list of two values")

    with torch.no_grad():
        norm = latent.norm(p=2)
        norm_clipped = norm.clamp(min=min_, max=max_)
        latent.data.mul_(norm_clipped / (norm + 1e-8))


def latent_norm_penalty(latent, target_norm, penalty_weight=1.0, penalty_type="quadratic"):
    """
    Compute a soft penalty term to encourage latent norm to be near target_norm.
    This is smoother than explicit projection and works better with gradient-based optimizers.

    Args:
        latent: The latent vector
        target_norm: Target norm value or (min_norm, max_norm) tuple
        penalty_weight: Weight for the penalty term
        penalty_type: "quadratic", "huber", or "barrier". "barrier" is a log
            barrier, defined only while the norm is strictly inside a
            (min_norm, max_norm) range: outside it this raises rather than
            returning NaN, and with a single target it silently computes the
            quadratic penalty instead

    Returns:
        penalty: Scalar penalty term to add to loss
    """
    current_norm = latent.norm(p=2)

    if isinstance(target_norm, (list, tuple)):
        if len(target_norm) != 2:
            raise ValueError("target_norm must be a single value or a tuple/list of two values")
        min_norm, max_norm = target_norm

        if penalty_type == "quadratic":
            # Quadratic penalty outside the range [min_norm, max_norm]
            if current_norm < min_norm:
                penalty = (current_norm - min_norm) ** 2
            elif current_norm > max_norm:
                penalty = (current_norm - max_norm) ** 2
            else:
                penalty = 0.0
        elif penalty_type == "huber":
            # Huber loss outside the range - smoother than quadratic
            delta = (max_norm - min_norm) * 0.1  # 10% of range as threshold
            if current_norm < min_norm:
                diff = min_norm - current_norm
                penalty = torch.where(diff <= delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
            elif current_norm > max_norm:
                diff = current_norm - max_norm
                penalty = torch.where(diff <= delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
            else:
                penalty = 0.0
        elif penalty_type == "barrier":
            # Log barrier penalty that becomes infinite at boundaries
            if not min_norm < current_norm < max_norm:
                raise ValueError(
                    f"norm_penalty_type='barrier' is undefined outside the target range: "
                    f"latent norm is {current_norm.item():.4g}, range is "
                    f"({min_norm}, {max_norm}). A barrier can only hold the latent inside "
                    f"a range it starts in (initialization norm is roughly "
                    f"latent_init_std * sqrt(latent_size)); use 'quadratic' or 'huber' "
                    f"for a penalty defined everywhere."
                )
            eps = 1e-6
            penalty = -torch.log(current_norm - min_norm + eps) - torch.log(
                max_norm - current_norm + eps
            )
        else:
            raise ValueError(f"Unknown penalty_type: {penalty_type}")

    else:
        # Single target norm
        if penalty_type == "quadratic":
            penalty = (current_norm - target_norm) ** 2
        elif penalty_type == "huber":
            diff = torch.abs(current_norm - target_norm)
            delta = target_norm * 0.1  # 10% of target as threshold
            penalty = torch.where(diff <= delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
        elif penalty_type == "barrier":
            # For single target, use quadratic penalty (barrier doesn't make sense)
            penalty = (current_norm - target_norm) ** 2
        else:
            raise ValueError(f"Unknown penalty_type: {penalty_type}")

    return penalty_weight * penalty


def _select_samples(
    *,
    xyz,
    sdf_gt,
    pts_surface,
    n_samples,
    n_samples_init,
    max_n_samples,
    n_steps_sample_ramp,
    step,
    device,
):
    """Draw this step's points, roughly evenly across the surfaces that have ground truth.

    Called once per optimization step. It used to live inside ``compute_loss``, which
    meant a fresh draw on every *call* of it: Adam calls it once per step so the two
    coincided, but LBFGS calls it once per line-search evaluation and once more to record
    the loss, so a single step was measured on seven different point sets and its line
    search moved the objective under itself.

    Returns ``(xyz_input, sdf_gt_)``, the points and the per-surface ground truth aligned
    to them; ``sdf_gt_`` keeps ``None`` for a surface that has none.
    """
    if n_samples_init is not None:
        n_samples_ = n_samples_init + int(
            (max_n_samples - n_samples_init) * min(1.0, (step / n_steps_sample_ramp))
        )
        logger.debug("ramping up samples...  %s", n_samples_)
    else:
        n_samples_ = n_samples

    # make sure not trying to sample more points than available for a surface
    n_samples_per_surface = []
    n_samples_per_surface_ = n_samples_ // len(sdf_gt)
    for surface_idx in range(len(sdf_gt)):
        pts_surface_ = (pts_surface == surface_idx).nonzero(as_tuple=True)[0]
        n_samples_per_surface.append(min(n_samples_per_surface_, pts_surface_.shape[0]))

    n_samples_ = sum(n_samples_per_surface)

    if n_samples_ != xyz.shape[0]:
        if len(sdf_gt) > 1:
            # get roughly equal number of samples from each surface
            # the list pts_surface is a list that indicates
            # which surface each point in xyz belongs to
            # pre allocate array to store random samples

            rand_samp = torch.empty(n_samples_, dtype=torch.int64, device=torch.device(device))
            current_filled = 0

            for idx, n_samples_per_surface_ in enumerate(n_samples_per_surface):
                # get the locations of the points that belong to the current surface
                pts_ = (pts_surface == idx).nonzero(as_tuple=True)[0]
                logger.debug(
                    "Surface %s has %s points, sampling %s points",
                    idx,
                    pts_.shape[0],
                    n_samples_per_surface_,
                )

                perm = torch.randperm(pts_.shape[0])
                pts_ = pts_[perm[:n_samples_per_surface_]]

                start_idx = current_filled
                end_idx = start_idx + n_samples_per_surface_
                rand_samp[start_idx:end_idx] = pts_
                current_filled = end_idx
            if current_filled < n_samples_:
                remaining = n_samples_ - current_filled
                perm = torch.randperm(xyz.shape[0])[:remaining]
                rand_samp[current_filled:] = perm
        else:
            rand_samp = torch.randperm(xyz.shape[0])[:n_samples_]

        # Use rand_samp indices to get xyz and sdf_gt
        xyz_input = xyz[rand_samp, ...]
        sdf_gt_ = [x[rand_samp, ...] if x is not None else None for x in sdf_gt]
    else:
        xyz_input = xyz
        sdf_gt_ = sdf_gt

    return xyz_input, sdf_gt_


def _recon_loss(
    *,
    decoders,
    latent,
    xyz_input,
    sdf_gt_,
    loss_fn,
    loss_weight,
    clamp_dist,
    difficulty_weight,
):
    """Decode this step's points and score them against their ground truth.

    ``sdf_gt_`` is one flat list across all decoders and ``surface_offset`` maps each
    decoder's local output columns onto it -- the positional surface contract
    ``reconstruct_latent``'s docstring states, and the reason this list is never reordered.

    Returns the summed per-decoder mean loss. A decoder that emits more surfaces than there
    is ground truth for stops early, and a surface whose ground truth is ``None`` is
    skipped; both say so at ``warning``, because both drop a surface from the objective.
    """
    recon_loss = 0

    # Iterate over the decoders (if there are multiple). ``sdf_gt_`` is one flat
    # list across all decoders; ``surface_offset`` maps each decoder's local
    # output columns onto it.
    surface_offset = 0
    for decoder_idx, decoder in enumerate(decoders):
        # Fast inference: pass latent and xyz separately
        pred_sdf = decoder(latent=latent.squeeze(0), xyz=xyz_input)

        # initialize loss as zeros with same device (will be averaged later)
        _loss_ = 0

        # Apply clamping distance - to ignore points that are too far away
        if clamp_dist is not None:
            pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        # Compute loss
        if pred_sdf.shape[1] == 1:
            # if only one surface - then just loss_fn (l1/l2) between pred_sdf and sdf_gt
            if difficulty_weight is not None:
                raise NotImplementedError
            _loss_ += (
                loss_fn(
                    pred_sdf.squeeze(),
                    sdf_gt_[surface_offset].squeeze(),
                )
                * loss_weight
            )

        else:
            # if multiple surfaces - then compute loss for each surface and weight them
            for sdf_idx in range(pred_sdf.shape[1]):
                gt_idx = surface_offset + sdf_idx
                if gt_idx >= len(sdf_gt_):
                    # might only have 1 surface (e.g., bone) and trying to reconstruct both
                    # (e.g., bone and cartilage) - in this case, break
                    # TODO: this is a bit of a hack, should be handled better
                    # right now it assumes the first surface is the bone / only of interest
                    # but we might want to reconstruct bone from cartilage (maybe?) or maybe we put
                    # cartilage first? Or maybe we have multiple bones & cartilage?
                    logger.warning(
                        "gt_idx (%s) >= len(sdf_gt_) (%s)... exiting",
                        gt_idx,
                        len(sdf_gt_),
                    )
                    break

                # if sdf_gt_[gt_idx] is None, then skip this surface
                # in fitting latent
                if sdf_gt_[gt_idx] is None:
                    logger.warning("sdf_gt_[gt_idx] is None, skipping surface %s", gt_idx)
                    continue

                if difficulty_weight is not None:
                    error_sign = torch.sign(
                        sdf_gt_[gt_idx].squeeze() - pred_sdf[:, sdf_idx].squeeze()
                    )
                    sdf_gt_sign = torch.sign(sdf_gt_[gt_idx].squeeze())
                    sample_weights = 1 + difficulty_weight * sdf_gt_sign * error_sign
                else:
                    sample_weights = torch.ones_like(pred_sdf[:, sdf_idx].squeeze())
                _loss_ += (
                    loss_fn(
                        pred_sdf[:, sdf_idx].squeeze(),
                        sdf_gt_[gt_idx].squeeze(),
                    )
                    * loss_weight
                    * sample_weights
                )

                logger.debug("loss_%s shape:  %s", sdf_idx, _loss_.shape)
                logger.debug("loss_%s mean:  %s", sdf_idx, _loss_.mean())
                logger.debug("loss_%s std:  %s", sdf_idx, _loss_.std())

        _loss_ = torch.mean(_loss_)
        # update the local loss
        recon_loss += _loss_
        # add the number of surfaces we just iterated over to the surface_offset
        # for the next decoder
        surface_offset += pred_sdf.shape[1]

    return recon_loss


def _regularization_losses(
    *,
    latent,
    l2reg,
    latent_reg_weight,
    latent_norm,
    use_soft_norm_constraint,
    norm_penalty_weight,
    norm_penalty_type,
):
    """The two terms that read only the latent: L2 regularization and the norm penalty.

    The norm penalty is the soft alternative to ``project_latent``'s hard rescaling, and
    the two are mutually exclusive by ``use_soft_norm_constraint``. Both are ``0`` when
    their parameter is unset, so the caller can always add them.

    Returns ``(latent_loss, norm_penalty_loss)``.
    """
    # Constrain new predictions to be close to zero (the mean), penalizing "abnormal" shapes
    latent_loss = latent_reg_weight * torch.mean(latent**2) if l2reg is True else 0

    norm_penalty_loss = 0
    if latent_norm is not None and use_soft_norm_constraint:
        norm_penalty_loss = latent_norm_penalty(
            latent, latent_norm, norm_penalty_weight, norm_penalty_type
        )

    return latent_loss, norm_penalty_loss


#: The values ``optimizer_name`` and ``loss_type`` are built from, named here so the
#: refusal and the branch that consumes them cannot drift apart.
_OPTIMIZER_NAMES = frozenset({"adam", "lbfgs"})
_LOSS_TYPES = frozenset({"l1", "l1_log", "l2"})

#: The only keyword ``reconstruct_latent`` takes without naming it, left over from the
#: chunked forward removed in 4583246; it is warned about where it is read. Issue #75 is
#: the capability that went with it, and its replacement is a named parameter.
_DEPRECATED_KWARGS = frozenset({"max_batch_size"})


@honour_verbose
def reconstruct_latent(
    decoders,
    num_iterations,
    latent_size,
    xyz,  # Nx3
    sdf_gt,  # Nx1 or list of Nx1
    loss_type="l1",
    lr=5e-4,
    loss_weight=1.0,
    l2reg=False,
    latent_init_std=0.01,
    latent_init_mean=0.0,
    clamp_dist=None,
    latent_reg_weight=1e-4,
    n_lr_updates=2,
    lr_update_factor=10,
    convergence="num_iterations",
    convergence_patience=50,
    log_wandb=False,
    log_wandb_step=10,
    verbose=False,
    optimizer_name="adam",
    n_samples=None,
    max_n_samples=None,  # 100000,
    n_steps_sample_ramp=None,  # 200,
    difficulty_weight=None,
    pts_surface=None,
    latent_norm=None,
    device="cuda",
    eikonal_weight=0.0,  # Weight for eikonal loss (0 to disable)
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
    """Optimize a latent code so the decoder(s) reproduce the observed SDF samples.

    Positional surface contract (nothing validates this; violating it silently
    changes the fit): surfaces are numbered 0..N-1 in the flat order the decoders
    emit them — decoder 0's outputs first, then decoder 1's, and so on.
    ``pts_surface[k]`` gives the surface number that sample ``xyz[k]`` /
    ``sdf_gt`` entry belongs to, and ``sdf_gt``, when a list, must be ordered by
    that same numbering. Swapping two ``sdf_gt`` entries relative to the
    ``pts_surface`` labels raises nothing and fits every point against the wrong
    surface. (This is the same positional-identity contract as
    ``reconstruct_mesh``'s result ``mesh`` list — see ``docs/SCOPE.md`` §3.1.)

    Returns:
        (loss, latent): the final loss value and the fitted latent tensor.
    """
    refuse_unknown_kwargs(kwargs, function_name="reconstruct_latent", deprecated=_DEPRECATED_KWARGS)

    if log_wandb and wandb is None:
        raise ImportError("log_wandb=True requires wandb, which is not installed")

    # Check for deprecated parameters
    if "max_batch_size" in kwargs:
        logger.warning(
            "max_batch_size is deprecated and will be removed in future versions. Batch processing has been simplified and now processes all data at once for better performance."
        )

    if eikonal_weight > 0:
        raise NotImplementedError(EIKONAL_UNSUPPORTED)

    sdf_gt = reconstruct_latent_sdf_gt_type_check(sdf_gt, verbose=verbose)
    pts_surface = reconstruct_latent_pts_surface_type_check(
        pts_surface, verbose=verbose, device=device
    )
    decoders = reconstruct_latent_decoders_type_check(decoders)

    # print info about xyz
    logger.debug("\txyz shape: %s", xyz.shape)
    logger.debug("\txyz type: %s", type(xyz))

    # Setup n_samples, if not specified.
    if n_samples is None:
        n_samples = xyz.shape[0]

    if (max_n_samples is not None) and (n_steps_sample_ramp is not None):
        logger.debug("Ramping up number of samples")
        n_samples_init = n_samples
    else:
        n_samples_init = None

    sdf_gt = reconstruct_latent_preprocess_sdf_gt(
        sdf_gt, clamp_dist, device=device, verbose=verbose
    )

    # Initialize random latent vector directly on GPU
    latent = torch.ones(1, latent_size, device=device).normal_(
        mean=latent_init_mean, std=latent_init_std
    )
    latent.requires_grad = True

    # Both of these used to be `if`/`elif` chains with no `else`, so an unrecognised value
    # left `optimizer` or `loss_fn` unassigned and surfaced 100 lines later as an
    # UnboundLocalError naming a local the caller has never seen. Refused here, next to the
    # parameter, rather than normalised: "Adam" is a typo, and accepting it is how a
    # parameter starts having two spellings.
    if optimizer_name not in _OPTIMIZER_NAMES:
        raise ValueError(
            f"optimizer_name must be one of {sorted(_OPTIMIZER_NAMES)}, got {optimizer_name!r}"
        )
    if loss_type not in _LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {sorted(_LOSS_TYPES)}, got {loss_type!r}")
    if hybrid_optimizer and optimizer_name != "adam":
        raise ValueError(
            "hybrid_optimizer=True runs Adam and then LBFGS, so optimizer_name is not "
            f"consulted; it was {optimizer_name!r}. Drop one of the two."
        )

    # Initialize optimizer(s)
    if hybrid_optimizer:
        # Set default values if not specified
        if adam_iterations is None:
            adam_iterations = num_iterations
        if lbfgs_iterations is None:
            lbfgs_iterations = 0

        # Update total iterations to match the sum
        total_iterations = adam_iterations + lbfgs_iterations
        n_adam_iterations = adam_iterations

        # Initialize both optimizers
        adam_optimizer = torch.optim.Adam([latent], lr=lr)
        lbfgs_optimizer = torch.optim.LBFGS(
            [latent], lr=lbfgs_lr, max_iter=lbfgs_max_iter, history_size=lbfgs_history_size
        )

        logger.info(
            "Hybrid optimizer: %s Adam iterations + %s LBFGS iterations",
            adam_iterations,
            lbfgs_iterations,
        )
        logger.info("Total iterations: %s", total_iterations)
    else:
        # Single optimizer mode
        total_iterations = num_iterations
        n_adam_iterations = num_iterations
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam([latent], lr=lr)
        elif optimizer_name == "lbfgs":
            optimizer = torch.optim.LBFGS(
                [latent],
                lr=lr,  # LBFGS typically uses lr=1.0
                max_iter=10,  # More internal iterations per step
                history_size=100,
            )  # Larger history for better Hessian approx

    # The LR schedule spans the phase it steps. This used to be derived from
    # `num_iterations` in both modes, and hybrid mode does not run `num_iterations` steps:
    # with num_iterations=10 and adam_iterations=100 it applied 11 decays for a caller who
    # asked for 2, ending at exactly 0.0, so the Adam phase stopped moving the latent.
    adjust_lr_every = reconstruct_latent_get_lr_update_freq(n_lr_updates, n_adam_iterations)

    # Initialize loss
    if loss_type == "l1":
        loss_fn = torch.nn.L1Loss(reduction="none")
    elif loss_type == "l1_log":
        eps = 1e-8

        def loss_fn(x, y):
            return torch.log(torch.abs(x - y) + eps)

    elif loss_type == "l2":
        loss_fn = torch.nn.MSELoss(reduction="none")

    # Initialize convergence tracking. These are compared against, so the sentinel has to
    # be worse than any loss; `100` was not, and a fit whose losses never dropped below it
    # recorded no step at all and lost the whole run to an UnboundLocalError on `latent_`.
    # `latent_` is bound here for the same reason: every exit from this function returns
    # something, including `num_iterations=0` and a loss that is NaN from the first step.
    patience = 0
    loss = float("inf")
    recon_loss = float("inf")
    latent_ = torch.clone(latent)

    # MOVE DECODERS TO GPU
    # SET DECODERS TO EVAL SO NO BATCH NORM ETC.
    for decoder in decoders:
        decoder.to(device)
        decoder.eval()

    # PASS XYZ TO GPU
    xyz = xyz.to(device)

    # Track whether we've switched to LBFGS in hybrid mode
    switched_to_lbfgs = False

    for step in range(total_iterations):
        # Determine current optimizer and phase
        if hybrid_optimizer:
            current_optimizer_name = "adam" if step < adam_iterations else "lbfgs"
            current_optimizer = adam_optimizer if step < adam_iterations else lbfgs_optimizer

            # Handle transition from Adam to LBFGS
            if step == adam_iterations and not switched_to_lbfgs and lbfgs_iterations > 0:
                switched_to_lbfgs = True
                logger.info(
                    "Switching from Adam to LBFGS at step %s (latent_norm: %.6f)",
                    step,
                    latent.norm().item(),
                )
        else:
            current_optimizer_name = optimizer_name
            current_optimizer = optimizer

        # update LR (only for Adam)
        if current_optimizer_name == "adam":
            if hybrid_optimizer:
                adjust_learning_rate(
                    initial_lr=lr,
                    optimizer=current_optimizer,
                    iteration=step,
                    decreased_by=lr_update_factor,
                    adjust_lr_every=adjust_lr_every,
                )
            else:
                adjust_learning_rate(
                    initial_lr=lr,
                    optimizer=optimizer,
                    iteration=step,
                    decreased_by=lr_update_factor,
                    adjust_lr_every=adjust_lr_every,
                )

        xyz_input, sdf_gt_ = _select_samples(
            xyz=xyz,
            sdf_gt=sdf_gt,
            pts_surface=pts_surface,
            n_samples=n_samples,
            n_samples_init=n_samples_init,
            max_n_samples=max_n_samples,
            n_steps_sample_ramp=n_steps_sample_ramp,
            step=step,
            device=device,
        )

        def compute_loss():
            """The step's objective: reconstruction, regularization, and their sum."""
            recon_loss = _recon_loss(
                decoders=decoders,
                latent=latent,
                xyz_input=xyz_input,
                sdf_gt_=sdf_gt_,
                loss_fn=loss_fn,
                loss_weight=loss_weight,
                clamp_dist=clamp_dist,
                difficulty_weight=difficulty_weight,
            )

            # Unreachable: `eikonal_weight > 0` raises at this function's entry. Kept
            # rather than deleted because §8.2 of the code-health plan owns its repair --
            # it needs a second derivative triplanar models do not have.
            # Eikonal loss computation
            # Compute eikonal loss - enforces ||∇f|| = 1 constraint for valid SDFs
            eikonal_loss_value = 0
            if eikonal_weight > 0:
                # Need to recompute with gradients enabled for eikonal loss
                xyz_input_grad = xyz_input.detach().requires_grad_(True)

                for decoder_idx, decoder in enumerate(decoders):
                    # Fast inference mode for eikonal loss
                    pred_sdf_grad = decoder(latent=latent.squeeze(0), xyz=xyz_input_grad)
                    eik_loss = eikonal_loss(pred_sdf_grad, xyz_input_grad, reduction="mean")
                    eikonal_loss_value += eik_loss

                # Average over decoders if multiple
                if len(decoders) > 1:
                    eikonal_loss_value = eikonal_loss_value / len(decoders)

            latent_loss, norm_penalty_loss = _regularization_losses(
                latent=latent,
                l2reg=l2reg,
                latent_reg_weight=latent_reg_weight,
                latent_norm=latent_norm,
                use_soft_norm_constraint=use_soft_norm_constraint,
                norm_penalty_weight=norm_penalty_weight,
                norm_penalty_type=norm_penalty_type,
            )

            total_loss = (
                recon_loss + latent_loss + eikonal_weight * eikonal_loss_value + norm_penalty_loss
            )

            return total_loss, recon_loss, latent_loss, eikonal_loss_value, norm_penalty_loss

        def step_closure():
            """LBFGS closure - computes loss and gradients, with optional latent projection"""
            current_optimizer.zero_grad()
            total_loss, _, _, _, _ = compute_loss()
            # retain_graph=True because LBFGS will call this closure multiple times
            total_loss.backward(retain_graph=True)

            # Only use hard projection if soft constraint is disabled
            if (
                current_optimizer_name == "lbfgs"
                and latent_norm is not None
                and not use_soft_norm_constraint
            ):
                with torch.no_grad():
                    project_latent(latent, latent_norm)

            return total_loss

        # Run the appropriate optimizer step
        if current_optimizer_name == "adam":
            current_optimizer.zero_grad()
            loss_, recon_loss_, latent_loss_, eikonal_loss_, norm_penalty_loss_ = compute_loss()
            loss_.backward()
            current_optimizer.step()
        elif current_optimizer_name == "lbfgs":
            # L-BFGS optimization step
            loss_ = current_optimizer.step(step_closure)
            # Compute final losses for tracking (without gradients)
            with torch.no_grad():
                _, recon_loss_, latent_loss_, eikonal_loss_, norm_penalty_loss_ = compute_loss()

        # Log progress at reasonable intervals
        if step % 50 == 0 or (step < 10):
            logger.info(
                "Step %s: Loss=%.6f, Recon=%.6f, Latent_norm=%.3f",
                step,
                loss_.item(),
                recon_loss_.item(),
                latent.norm().item(),
            )

        # check if want to project onto hypersphere (skip for LBFGS since it's done in closure)
        # Only use hard projection if soft constraint is disabled
        if (
            latent_norm is not None
            and current_optimizer_name != "lbfgs"
            and not use_soft_norm_constraint
        ):
            logger.info("Projecting latent onto hypersphere of norm in range: %s", latent_norm)
            project_latent(latent, latent_norm)

        # Print progress/loss as appropriate
        if step % 50 == 0:
            optimizer_info = f" ({current_optimizer_name})" if hybrid_optimizer else ""
            logger.debug("Step: %s%s, Loss: %s", step, optimizer_info, loss_.item())
            logger.debug("\tRecon loss:  %s", recon_loss_.item())
            if eikonal_weight > 0:
                eikonal_val = (
                    eikonal_loss_.item() if hasattr(eikonal_loss_, "item") else float(eikonal_loss_)
                )
                logger.debug("\tEikonal loss: %.6f", eikonal_val)
            if latent_norm is not None and use_soft_norm_constraint:
                norm_penalty_val = (
                    norm_penalty_loss_.item()
                    if hasattr(norm_penalty_loss_, "item")
                    else float(norm_penalty_loss_)
                )
                logger.debug("\tNorm penalty loss: %.6f", norm_penalty_val)
            logger.debug("\tLatent norm:  %s", latent.norm().item())

        # Log to wandb as appropriate
        if (log_wandb is True) and (step % log_wandb_step == 0):
            log_dict = {
                "total_loss": loss_.item(),
                "l1_loss": loss_.item(),
                "recon_loss": recon_loss_.item(),
                "latent_loss": latent_loss_.item() if l2reg is True else np.nan,
                "latent_norm": latent.norm().item(),
            }
            if eikonal_weight > 0:
                log_dict["eikonal_loss"] = (
                    eikonal_loss_.item() if hasattr(eikonal_loss_, "item") else float(eikonal_loss_)
                )
            if latent_norm is not None and use_soft_norm_constraint:
                log_dict["norm_penalty_loss"] = (
                    norm_penalty_loss_.item()
                    if hasattr(norm_penalty_loss_, "item")
                    else float(norm_penalty_loss_)
                )
            wandb.log(log_dict)

        # Handle end of loop accounting of loss/latent based on convergence criteria
        if convergence == "overall_loss":
            if loss_ < loss:
                loss = loss_
                latent_ = torch.clone(latent)
                patience = 0
            else:
                patience += 1

            if patience > convergence_patience:
                logger.info(
                    "Converged (overall_loss) after %s steps! Final loss: %.6f", step, loss_.item()
                )
                break
        elif convergence == "recon_loss":
            if recon_loss_ < recon_loss:
                recon_loss = recon_loss_
                # `loss` is what this function returns, so it is recorded with the latent
                # it belongs to. Without this line only `recon_loss` moved, and the
                # returned loss was the initial sentinel -- on the mode
                # `default_config.json` ships (`convergence_type_recon`).
                loss = loss_
                latent_ = torch.clone(latent)
                patience = 0
            else:
                patience += 1

            if patience > convergence_patience:
                logger.info(
                    "Converged (recon_loss) after %s steps! Final recon loss: %.6f",
                    step,
                    recon_loss_.item(),
                )
                break
        else:
            loss = loss_
            latent_ = torch.clone(latent)

    return loss, latent_
