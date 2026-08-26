import itertools
import logging
import os
import time
import warnings

import numpy as np
import torch

# Optional (#5): every wandb use is behind an explicit request that raises when absent.
try:
    import wandb
except ImportError:
    wandb = None

from NSM.losses import EIKONAL_UNSUPPORTED, eikonal_loss
from NSM.reconstruct import (
    compare_cart_thickness,
    compare_cart_thickness_femur,
    compare_cart_thickness_patella,
    compare_cart_thickness_tibia,
    compare_cart_thickness_whole_joint,
    get_mean_errors,
)
from NSM.train.utils import (
    NoOpProfiler,
    add_plain_lr_to_config,
    calc_weight,
    cyclic_anneal_linear,
    get_kld,
    get_profiler,
)
from NSM.utils import (
    adjust_learning_rate,
    clear_gpu_cache,
    get_checkpoints,
    get_latent_vecs,
    get_learning_rate_schedules,
    get_optimizer,
    save_latent_vectors,
    save_model,
    save_model_params,
)

logger = logging.getLogger(__name__)

DICT_VALIDATION_FUNCS = {
    "compare_cart_thickness": compare_cart_thickness,
    "compare_cart_thickness_tibia": compare_cart_thickness_tibia,
    "compare_cart_thickness_patella": compare_cart_thickness_patella,
    "compare_cart_thickness_femur": compare_cart_thickness_femur,
    "compare_cart_thickness_whole_joint": compare_cart_thickness_whole_joint,
    None: None,
}

loss_l1 = torch.nn.L1Loss(reduction="none")


def train_deep_sdf(config, model, sdf_dataset, use_wandb=False):
    """
    Train ``model`` against ``sdf_dataset`` and return the per-epoch history.

    Returns one dict per trained epoch (``n_epochs - resume_epoch`` entries): the
    epoch's wandb payload — ``train_epoch``'s ``log_dict``, plus the validation metrics
    on validation epochs — with four keys the payload does not carry: ``epoch``,
    ``lrs`` and ``targets`` (per param group, keyed by group ``name``, read after the
    epoch so they are the rates it actually ran with), and ``latent_norms`` (one norm
    per training subject). The wandb payload itself is unchanged by the extras.

    The trained weights land in the caller's ``model`` (mutated in place) and in the
    checkpoints under ``config["experiment_directory"]``; the latent embedding exists
    only in the checkpoints and the history — it is constructed here and not returned.
    """
    if use_wandb and wandb is None:
        raise ImportError("use_wandb=True requires wandb, which is not installed")

    # add default params for backwards compatibility between
    # train_deep_sdf and train_deep_sdf_multi_surface.
    config.setdefault("objects_per_decoder", 1)
    config.setdefault("mesh_names", None)
    config.setdefault("resume_epoch", 0)
    config.setdefault("scale_jointly", False)
    config.setdefault("fix_mesh_recon", False)
    config.setdefault("log_latent", None)

    if config.get("eikonal_weight", 0) > 0:
        raise NotImplementedError(EIKONAL_UNSUPPORTED)

    # Surface identity is defined by the order of each subject's mesh-path list, so a
    # dataset that carries mesh_names is the authority (#52): adopt them when the config
    # has none, refuse a disagreeing config. Checked here, at entry, so a wrong
    # declaration cannot survive to save_model_params.
    dataset_mesh_names = getattr(sdf_dataset, "mesh_names", None)
    if dataset_mesh_names is not None:
        if config["mesh_names"] is None:
            config["mesh_names"] = list(dataset_mesh_names)
        elif list(config["mesh_names"]) != list(dataset_mesh_names):
            raise ValueError(
                f"config mesh_names {config['mesh_names']} disagree with the dataset's "
                f"{list(dataset_mesh_names)}. The dataset's names follow each subject's "
                f"mesh-path list order; fix whichever declaration is wrong."
            )

    # Validate mesh_names length matches objects_per_decoder if provided
    if config["mesh_names"] is not None:
        if len(config["mesh_names"]) != config["objects_per_decoder"]:
            raise ValueError(
                f"mesh_names has {len(config['mesh_names'])} entries but "
                f"objects_per_decoder is {config['objects_per_decoder']}. "
                f"These must match."
            )
    elif config["objects_per_decoder"] > 1:
        warnings.warn(
            "No 'mesh_names' provided in config. Downstream consumers will need to "
            "infer mesh identity from the decoder output count, which is fragile. "
            "Consider adding e.g. 'mesh_names': ['bone', 'cart'] to your config.",
            UserWarning,
            stacklevel=2,
        )

    config = add_plain_lr_to_config(config)
    config["checkpoints"] = get_checkpoints(config)
    config["lr_schedules"] = get_learning_rate_schedules(config)

    model = model.to(config["device"])

    if use_wandb is True:
        wandb.login(key=os.environ["WANDB_KEY"])
        wandb.init(
            # Set the project where this run will be logged
            project=config["project_name"],  # "diffusion-net-predict-sex",
            entity=config["entity_name"],  # "bone-modeling",
            # Track hyperparameters and run metadata
            config=config,
            name=config["run_name"],
            tags=config["tags"],
        )
        wandb.watch(model, log="all")

    data_loader = torch.utils.data.DataLoader(
        sdf_dataset,
        batch_size=config["objects_per_batch"],
        shuffle=True,
        num_workers=config["num_data_loader_threads"],
        drop_last=False,
        prefetch_factor=config["prefetch_factor"],
        pin_memory=True,
    )

    latent_vecs = get_latent_vecs(len(data_loader.dataset), config).to(config["device"])
    optimizer = get_optimizer(
        model,
        latent_vecs,
        lr_schedules=config["lr_schedules"],
        optimizer=config["optimizer"],
        weight_decay=config["weight_decay"],
    )

    _resume_from_checkpoint(config, model, latent_vecs, optimizer)

    history = []

    # profiler that runs if config['profiler'] is True, else a dummy profiler is used and should have no effect
    with get_profiler(config) as profiler:

        for epoch in range(config["resume_epoch"] + 1, config["n_epochs"] + 1):
            # not passing latent_vecs because presumably they are being tracked by the
            # and updated in memory?
            log_dict = train_epoch(
                model,
                data_loader,
                latent_vecs,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                n_surfaces=config["objects_per_decoder"],
            )
            val_epoch = (
                (epoch in config["checkpoints"])
                and ("val_paths" in config)
                and (config["val_paths"] is not None)
            )
            checkpoint_epoch = (
                epoch in config["checkpoints"] or epoch % config["save_frequency"] == 0
            )

            if val_epoch or checkpoint_epoch:
                _schedule_free_eval_warmup(
                    model, latent_vecs, data_loader, optimizer, config, epoch
                )

            if checkpoint_epoch:
                _save_checkpoint(config, epoch, model, latent_vecs, optimizer, sdf_dataset)

            if val_epoch:
                log_dict.update(_run_validation(config, model))

            if use_wandb is True:
                wandb.log(log_dict, step=epoch - 1)

            history.append(
                {
                    **log_dict,
                    "epoch": epoch,
                    # Read AFTER train_epoch: adjust_learning_rate() runs at its top, so
                    # these are the rates the epoch actually ran with.
                    "lrs": {group["name"]: group["lr"] for group in optimizer.param_groups},
                    "targets": {group["name"]: group["target"] for group in optimizer.param_groups},
                    "latent_norms": torch.norm(latent_vecs.weight.data, dim=1).tolist(),
                }
            )

            profiler.step()

            clear_gpu_cache(config["device"])

    return history


def _resume_from_checkpoint(config, model, latent_vecs, optimizer):
    """
    Load the ``resume_epoch`` checkpoint into ``model``, ``latent_vecs`` and ``optimizer``.

    ``resume_epoch`` names the last COMPLETED epoch: its checkpoint is loaded and the
    epoch loop continues at ``resume_epoch + 1``. 0 means a fresh run — a no-op here.
    This guard and the loop boundary must share that convention: a ``> 1`` guard here
    once let ``resume_epoch=1`` skip epoch 1 while loading nothing (KNOWN_ISSUES
    section History 11).

    Raises ``ValueError`` for a checkpoint saved without optimizer state, or one from
    before Aug 2026 whose param groups carry no ``target`` (KNOWN_ISSUES section 1).
    """
    if config["resume_epoch"] < 1:
        return
    logger.info("Loading model, optimizer, and latent states from epoch %s", config["resume_epoch"])
    # load each checkpoint once rather than re-reading it per state
    model_checkpoint = torch.load(
        os.path.join(config["experiment_directory"], "model", f'{config["resume_epoch"]}.pth')
    )
    latent_checkpoint = torch.load(
        os.path.join(
            config["experiment_directory"], "latent_codes", f'{config["resume_epoch"]}.pth'
        )
    )

    model.load_state_dict(model_checkpoint["model"])

    # load the optimizer states. Checkpoints saved without an optimizer hold None (or
    # the string "None", written before Sep 2026); load_state_dict would raise an
    # opaque TypeError on either.
    if model_checkpoint["optimizer"] in (None, "None"):
        raise ValueError(
            f"Checkpoint at epoch {config['resume_epoch']} was saved without optimizer "
            f"state, so training cannot resume from it."
        )
    optimizer.load_state_dict(model_checkpoint["optimizer"])
    # state_dict() retains custom param-group keys and load_state_dict() restores
    # them, but it adopts the checkpoint's metadata wholesale -- so a checkpoint saved
    # before Aug 2026 leaves the groups with no 'target' and schedules cannot be
    # mapped. Fail here rather than downstream: adjust_learning_rate() would catch it
    # at epoch 1, but it is skipped for schedule_free_*, which would then run to the
    # first checkpoint save before failing.
    if any(group.get("target") is None for group in optimizer.param_groups):
        raise ValueError(
            f"Checkpoint at epoch {config['resume_epoch']} carries no optimizer "
            f"param-group targets, so it predates Aug 2026 and its learning-rate "
            f"schedules cannot be mapped. Resuming it is not supported; start a "
            f"fresh run. See docs/KNOWN_ISSUES.md section 1."
        )

    # load the latent vectors
    latent_vecs.load_state_dict(latent_checkpoint["latent_codes"])


def _schedule_free_eval_warmup(model, latent_vecs, data_loader, optimizer, config, epoch):
    """
    Put a schedule_free optimizer into eval mode, then recalibrate normalization-layer
    statistics at the averaged weights by running real forward passes:
    https://github.com/facebookresearch/schedule_free/issues/44

    A no-op for every other optimizer family. The batch is unpacked exactly the way
    ``train_epoch`` unpacks it — latent lookup, variational sampling, ``batch_split``
    chunking — which is what #42 was about: this warm-up used to forward the raw
    dataloader item, so every schedule_free run died at its first checkpoint or
    validation epoch.
    """
    if "schedule_free" not in config["optimizer"]:
        return
    optimizer.eval()
    with torch.no_grad():
        for sdf_data, indices in itertools.islice(data_loader, 50):
            xyz = sdf_data["xyz"].to(config["device"]).reshape(-1, 3)
            indices = (
                indices.to(config["device"])
                .unsqueeze(-1)
                .repeat(1, config["samples_per_object_per_batch"])
                .view(-1)
            )
            xyz = torch.chunk(xyz, config["batch_split"])
            indices = torch.chunk(indices, config["batch_split"])
            for split_idx in range(config["batch_split"]):
                batch_vecs = latent_vecs(indices[split_idx])
                if "variational" in config and config["variational"] is True:
                    mu = batch_vecs[:, : config["latent_size"]]
                    logvar = batch_vecs[:, config["latent_size"] :]
                    std = torch.exp(0.5 * logvar)
                    batch_vecs = std * torch.randn_like(std) + mu
                model(torch.cat([batch_vecs, xyz[split_idx]], dim=1), epoch=epoch)


def _save_checkpoint(config, epoch, model, latent_vecs, optimizer, sdf_dataset):
    """
    Persist the epoch: ``model_params_config.json`` (first write wins), the latent
    embedding, and the model+optimizer checkpoint.
    """
    save_model_params(config=config, list_mesh_paths=sdf_dataset.list_mesh_paths)
    save_latent_vectors(
        config=config,
        epoch=epoch,
        latent_vec=latent_vecs,
    )
    save_model(config=config, epoch=epoch, decoder=model, optimizer=optimizer)


def _run_validation(config, model):
    """
    Reconstruct the ``val_paths`` subjects and return ``get_mean_errors``' metric dict.

    The kwarg block is the config→``get_mean_errors`` mapping; the commented-out lines
    name parameters deliberately left at their defaults.
    """
    clear_gpu_cache(config["device"])

    # TODO: Change this to just accept the config?
    # or... update all parameters to be the same in the config and the function call?
    # this will just allow unpacking of the config dict.
    return get_mean_errors(
        mesh_paths=config["val_paths"],
        decoders=model,
        num_iterations=config["num_iterations_recon"],
        register_similarity=True,
        latent_size=config["latent_size"],
        lr=config["lr_recon"],
        # loss_weight
        # loss_type
        l2reg=config["l2reg_recon"],
        # latent_init_std
        # latent_init_mean
        clamp_dist=config["clamp_dist_recon"],
        # latent_reg_weight
        n_lr_updates=config["n_lr_updates_recon"],
        lr_update_factor=config["lr_update_factor_recon"],
        calc_symmetric_chamfer=config["chamfer"],
        calc_assd=config["assd"],
        convergence=config["convergence_type_recon"],
        convergence_patience=config["convergence_patience_recon"],
        # log_wandb
        verbose=config["verbose"],
        objects_per_decoder=config["objects_per_decoder"],
        get_rand_pts=config["get_rand_pts_recon"],
        n_pts_random=config["n_pts_random_recon"],
        sigma_rand_pts=config["sigma_rand_pts_recon"],
        n_samples_latent_recon=config["n_samples_latent_recon"],
        # difficulty_weight_recon
        # chamfer_norm
        scale_all_meshes=True,
        recon_func=(
            None
            if (("recon_val_func_name" not in config))
            else DICT_VALIDATION_FUNCS[config["recon_val_func_name"]]
        ),
        predict_val_variables=(
            None if ("predict_val_variables" not in config) else config["predict_val_variables"]
        ),
        scale_jointly=config["scale_jointly"],
        fix_mesh=config["fix_mesh_recon"],
        device=config["device"],
    )


def train_epoch(
    model,
    data_loader,
    latent_vecs,
    optimizer,
    config,
    epoch,
    n_surfaces=2,
):
    """
    Run one optimization epoch and return its log dict.

    ``adjust_learning_rate`` runs at the TOP for Adam/AdamW — each param group's lr is
    set from its ``target``'s schedule before the first step, so rates read after this
    returns are the rates the epoch actually ran with. schedule_free optimizers skip it
    and get ``optimizer.train()`` instead.

    Returns a flat dict: ``loss``, ``epoch_time_s``, ``l1_loss``,
    ``latent_code_regularization_loss``, ``mean_vec_length``/``std_vec_length`` (epoch
    means over batches), per-surface ``l1_loss_{i}``, the four load-timing keys only
    when the dataset actually timed a disk load (#22), ``eikonal_loss`` only when the
    (gated) eikonal weight is on, and ``latent_{i}`` wandb histograms with their
    mean/std when ``config["log_latent"]`` is set.

    ``n_surfaces`` must match the decoder's ``objects_per_decoder`` and the dataset's
    per-subject surface count: ``gt_sdf`` columns are read positionally by surface.
    """
    # n_surfaces = len(models)
    start = time.time()
    # for model in models:
    model.train()

    if not ("schedule_free" in config["optimizer"]):
        adjust_learning_rate(config["lr_schedules"], optimizer, epoch)
    else:
        optimizer.train()

    step_losses = 0
    step_l1_loss = 0
    step_code_reg_loss = 0
    step_eikonal_loss = 0
    step_l1_losses = [0.0 for _ in range(n_surfaces)]
    step_mean_vec_length = 0
    step_std_vec_length = 0

    # if config['code_regularization_type_prior'] == 'kld_diagonal':
    #     kld_loss = get_kld(latent_vecs)

    step_mean_size = 0
    step_mean_load_time = 0
    step_mean_load_rate = 0
    step_whole_load_time = 0
    timing_batches = 0

    for sdf_data, indices in data_loader:
        if config["verbose"] is True:
            logger.debug("sdf index size: %s", indices.size())
            logger.debug("xyz data size: %s", sdf_data["xyz"].size())
            logger.debug("sdf gt size: %s", sdf_data["gt_sdf"].size())

        xyz = sdf_data["xyz"].to(config["device"])
        xyz = xyz.reshape(-1, 3)

        num_sdf_samples = xyz.shape[0]
        xyz.requires_grad = False

        indices = indices.to(config["device"])

        sdf_gt = []
        if n_surfaces == 1:
            # Handle the case where there is only one surface
            sdf_gt_ = sdf_data["gt_sdf"].reshape(-1, 1)
            if config["enforce_minmax"] is True:
                sdf_gt_ = torch.clamp(sdf_gt_, -config["clamp_dist"], config["clamp_dist"])
            sdf_gt_.requires_grad = False
            sdf_gt.append(sdf_gt_)
        else:
            for surf_idx in range(n_surfaces):
                sdf_gt_ = sdf_data["gt_sdf"][:, :, surf_idx].reshape(-1, 1)
                if config["enforce_minmax"] is True:
                    sdf_gt_ = torch.clamp(sdf_gt_, -config["clamp_dist"], config["clamp_dist"])
                sdf_gt_.requires_grad = False
                sdf_gt.append(sdf_gt_)

        if config["verbose"] is True:
            logger.debug("sdf gt size: %s", [x_.size() for x_ in sdf_gt])

        xyz = torch.chunk(xyz, config["batch_split"])
        indices = torch.chunk(
            indices.unsqueeze(-1)
            .repeat(1, config["samples_per_object_per_batch"])
            .view(-1),  # repeat the index for every sample
            config[
                "batch_split"
            ],  # split the data into the appropriate number of batches - so can fit in ram.
        )

        for surf_idx in range(n_surfaces):
            sdf_gt[surf_idx] = torch.chunk(sdf_gt[surf_idx], config["batch_split"])

        if config["verbose"] is True:
            logger.debug("len sdf_gt %s", len(sdf_gt))
            logger.debug("len sdf_gt chunks: %s", [len(x_) for x_ in sdf_gt])
            logger.debug("len xyz chunks %s", len(xyz))

        batch_loss = 0.0
        batch_l1_loss = 0.0
        batch_l1_losses = [0.0 for _ in range(n_surfaces)]
        batch_code_reg_loss = 0.0
        batch_eikonal_loss = 0.0

        optimizer.zero_grad()

        for split_idx in range(config["batch_split"]):
            if config["verbose"] is True:
                logger.debug("Split idx:  %s", split_idx)

            batch_vecs = latent_vecs(indices[split_idx])
            if "variational" in config and config["variational"] is True:
                mu = batch_vecs[:, : config["latent_size"]]
                logvar = batch_vecs[:, config["latent_size"] :]
                std = torch.exp(0.5 * logvar)
                err = torch.randn_like(std)
                batch_vecs = std * err + mu

            inputs = torch.cat([batch_vecs, xyz[split_idx]], dim=1)
            # inputs = inputs.to(config['device'])

            if config["verbose"] is True:
                logger.debug("model dtype %s", next(model.parameters()).dtype)
                logger.debug("inputs dtype %s", inputs.dtype)
            # pred_sdfs = []
            # for model in models:
            pred_sdf = model(inputs, epoch=epoch)

            if n_surfaces == 1:
                # Ensure pred_sdf is 2D even for single surface
                if pred_sdf.dim() == 2 and pred_sdf.shape[1] == 1:
                    pass  # Already correct shape
                else:
                    pred_sdf = pred_sdf.unsqueeze(1)  # Add surface dimension if needed

            # KNOWN DEFECT, docs/KNOWN_ISSUES.md Open: this clamps the PREDICTION, not just the
            # target, and torch.clamp passes no gradient outside its bounds -- so every
            # sample predicted beyond +/-clamp_dist contributes exactly zero gradient
            # however wrong it is. 44.6% of a freshly built triplanar decoder's
            # predictions are already outside +/-0.1, and the shipped default_config.json
            # uses clamp_dist 0.1 while both ShapeMedKnee configs use 1.0. clamp_dist is
            # a training-dynamics knob, not the target transform its name suggests.
            if config["enforce_minmax"] is True:
                pred_sdf = torch.clamp(pred_sdf, -config["clamp_dist"], config["clamp_dist"])
            # elif config['hard_sample_difficulty_weight'] is not None:
            #     pred_sdf = torch.clamp(pred_sdf, -1, 1)

            if config["verbose"] is True:
                logger.debug("len pred_sdf %s", pred_sdf.shape)
                logger.debug("split idx %s", split_idx)
            l1_losses = []
            for surf_idx in range(n_surfaces):
                if config["verbose"] is True:
                    logger.debug("surf idx %s", surf_idx)
                    logger.debug("%s", len(sdf_gt))
                    logger.debug("%s", len(sdf_gt[surf_idx]))
                    logger.debug("pred_sdf shape %s", pred_sdf.shape)
                    logger.debug("unsqueezed pred_sdf shape %s", pred_sdf[:, surf_idx].shape)
                    logger.debug("sdf_gt shape %s", sdf_gt[surf_idx][split_idx].shape)
                l1_losses.append(
                    loss_l1(
                        pred_sdf[:, surf_idx],
                        sdf_gt[surf_idx][split_idx].squeeze(1).to(config["device"]),
                    )
                )

            if config.get("multi_object_overlap", False) is True:
                raise Exception("Not implemented yet")
                # Should add some weighted penalty to the l1 loss
                # this is similar to surface_accuracy_e below
                # The idea being - the signs of the objects should never have 2 objects as (-) becuase it
                # means one object is inside the other.
                # However, we dont want there to be any gaps between them - so we need to be intelligent about
                # how to penalize this so that it doesnt end up with weird artefacts.

            # curriculum SDF equation 5
            # progressively fine-tune the regions of surface cared about by the network.
            if config["surface_accuracy_e"] is not None:
                weight_schedule = 1 - calc_weight(
                    epoch,
                    config["n_epochs"],
                    config["surface_accuracy_schedule"],
                    config["surface_accuracy_cooldown"],
                )
                for l1_idx, l1_loss in enumerate(l1_losses):
                    l1_losses[l1_idx] = torch.maximum(
                        l1_loss - (weight_schedule * config["surface_accuracy_e"]),
                        torch.zeros_like(l1_loss),
                    )

            # curriculum SDF equation 6
            # progressively fine-tune the regions of surface cared about by the network.
            # weighting gives higher preference to regions closer to surface / with opposite sign.
            if config["sample_difficulty_weight"] is not None:
                weight_schedule = calc_weight(
                    epoch,
                    config["n_epochs"],
                    config["sample_difficulty_weight_schedule"],
                    config["sample_difficulty_cooldown"],
                )
                difficulty_weight = weight_schedule * config["sample_difficulty_weight"]
                for surf_idx, surf_gt_ in enumerate(sdf_gt):
                    # Weights points independently
                    # so, if hard for one surface - then we weight it heavily, but if
                    # easy for another surface - then we weight it less.
                    error_sign = torch.sign(
                        surf_gt_[split_idx].squeeze(1).to(config["device"]) - pred_sdf[:, surf_idx]
                    )
                    sdf_gt_sign = torch.sign(surf_gt_[split_idx].squeeze(1).to(config["device"]))
                    sample_weights = 1 + difficulty_weight * sdf_gt_sign * error_sign
                    l1_losses[surf_idx] = l1_losses[surf_idx] * sample_weights

            # Weight each surface loss by the number of samples it has
            # so that the sum of them all is the same as the mean loss.
            for idx, l1_loss_ in enumerate(l1_losses):
                l1_losses[idx] = l1_loss_ / num_sdf_samples

            # Comput total loss for each mesh (which is the same as the
            # mean).
            l1_loss = 0

            # Create weights for each surface
            if isinstance(config.get("surface_weighting", None), (list, tuple)):
                assert len(config["surface_weighting"]) == n_surfaces
                weights_total = n_surfaces
                weights_sum = sum(config["surface_weighting"])
                weights = []
                for weight in config["surface_weighting"]:
                    weights.append(weight / weights_sum * weights_total)
            else:
                weights = [
                    1,
                ] * n_surfaces

            for l1_idx, l1_loss_ in enumerate(l1_losses):
                l1_loss += l1_loss_.sum() * weights[l1_idx]

            # Normalize by number of surfaces
            l1_loss = l1_loss / len(l1_losses)

            if config["verbose"] is True:
                logger.debug("l1 losses: %s", [l1_loss_.sum().item() for l1_loss_ in l1_losses])
                logger.debug("l1 loss: %s", l1_loss.item())

            batch_l1_loss += l1_loss.item()
            for l1_idx, l1_loss_ in enumerate(l1_losses):
                batch_l1_losses[l1_idx] += l1_loss_.sum().item()
            chunk_loss = l1_loss

            # Add eikonal loss if enabled
            eikonal_loss_value = 0
            if config.get("eikonal_weight", 0) > 0:
                # Recompute SDF with gradients for eikonal loss
                xyz_grad = xyz[split_idx].detach().requires_grad_(True)
                inputs_grad = torch.cat([batch_vecs, xyz_grad], dim=1)
                pred_sdf_grad = model(inputs_grad, epoch=epoch)
                eik_loss = eikonal_loss(pred_sdf_grad, xyz_grad, reduction="mean")
                eikonal_loss_value = eik_loss.item()
                batch_eikonal_loss += eikonal_loss_value
                chunk_loss = chunk_loss + config["eikonal_weight"] * eik_loss

            if config["code_regularization"] is True:
                if "variational" in config and config["variational"] is True:
                    kld = torch.mean(
                        -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
                    )
                    reg_loss = kld
                    code_reg_norm = 1
                else:
                    if config["code_regularization_type_prior"] == "spherical":
                        # spherical prior
                        # all latent vectors should have the same unit length
                        # therefore, the latent dimensions will be correlated
                        # with one another - this is as opposed to PCA (and below).
                        reg_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    elif config["code_regularization_type_prior"] == "identity":
                        # independently penalize each dimension/value of latent code
                        # therefore latent code ends up having identity covariance matrix
                        reg_loss = torch.sum(torch.square(batch_vecs))
                    elif config["code_regularization_type_prior"] == "kld_diagonal":
                        reg_loss = get_kld(batch_vecs)
                    else:
                        raise ValueError(
                            f'Unknown code regularization type prior: {config["code_regularization_type_prior"]}'
                        )
                    code_reg_norm = num_sdf_samples

                reg_loss = (
                    config["code_regularization_weight"]
                    * min(1, epoch / config["code_regularization_warmup"])
                    * reg_loss
                ) / code_reg_norm

                if config["code_cyclic_anneal"] is True:
                    anneal_weight = cyclic_anneal_linear(epoch=epoch, n_epochs=config["n_epochs"])
                    reg_loss = reg_loss * anneal_weight

                chunk_loss = chunk_loss + reg_loss.to(config["device"])
                batch_code_reg_loss += reg_loss.item()

            mean_vec_length = torch.mean(torch.norm(batch_vecs, dim=1))
            std_vec_length = torch.std(torch.norm(batch_vecs, dim=1))

            chunk_loss.backward()

            batch_loss += chunk_loss.item()

        step_losses += batch_loss
        step_l1_loss += batch_l1_loss
        step_code_reg_loss += batch_code_reg_loss
        step_eikonal_loss += batch_eikonal_loss
        for l1_idx, l1_loss_ in enumerate(batch_l1_losses):
            step_l1_losses[l1_idx] += l1_loss_  # l1_loss_

        step_mean_vec_length += mean_vec_length.item()
        step_std_vec_length += std_vec_length.item()

        if config["grad_clip"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        # Load-timing diagnostics are optional: the dataset emits them only when it
        # actually loaded from disk (test_load_times=True and store_data_in_memory=False),
        # so an in-memory dataset trains without them (#22).
        if "size" in sdf_data:
            step_mean_size += torch.mean(sdf_data["size"]).item()
            step_mean_load_time += torch.mean(sdf_data["time"]).item()
            step_mean_load_rate += torch.mean(sdf_data["mb_per_sec"]).item()
            step_whole_load_time += torch.mean(sdf_data["whole_load_time"]).item()
            timing_batches += 1

        optimizer.step()
    end = time.time()

    seconds_elapsed = end - start

    save_loss = step_losses / len(data_loader)
    save_l1_loss = step_l1_loss / len(data_loader)
    save_code_reg_loss = step_code_reg_loss / len(data_loader)
    save_eikonal_loss = step_eikonal_loss / len(data_loader)
    save_l1_losses = [l1_loss_ / len(data_loader) for l1_loss_ in step_l1_losses]
    save_mean_vec_length = step_mean_vec_length / len(data_loader)
    save_std_vec_length = step_std_vec_length / len(data_loader)

    if timing_batches > 0:
        save_mean_size = step_mean_size / timing_batches
        save_mean_load_time = step_mean_load_time / timing_batches
        save_mean_load_rate = step_mean_load_rate / timing_batches
        save_whole_load_time = step_whole_load_time / timing_batches

    logger.info("save loss:  %s", save_loss)
    logger.info("\t save l1 loss:  %s", save_l1_loss)
    logger.info("\t save code loss:  %s", save_code_reg_loss)
    if config.get("eikonal_weight", 0) > 0:
        logger.info("\t save eikonal loss: %.6f", save_eikonal_loss)
    logger.info("\t save l1 losses:  %s", save_l1_losses)

    log_dict = {
        "loss": save_loss,
        "epoch_time_s": seconds_elapsed,
        "l1_loss": save_l1_loss,
        "latent_code_regularization_loss": save_code_reg_loss,
        "mean_vec_length": save_mean_vec_length,
        "std_vec_length": save_std_vec_length,
    }
    if timing_batches > 0:
        log_dict["mean_size"] = save_mean_size
        log_dict["mean_load_time"] = save_mean_load_time
        log_dict["mean_load_rate"] = save_mean_load_rate
        log_dict["whole_load_time"] = save_whole_load_time
    if config.get("eikonal_weight", 0) > 0:
        log_dict["eikonal_loss"] = save_eikonal_loss
    for l1_idx, l1_loss_ in enumerate(save_l1_losses):
        log_dict["l1_loss_{}".format(l1_idx)] = l1_loss_

    if config["log_latent"] is not None:
        if wandb is None:
            raise ImportError("config['log_latent'] requires wandb, which is not installed")
        vecs = latent_vecs.weight.data.cpu().numpy()
        for latent_idx in range(config["log_latent"]):
            log_dict[f"latent_{latent_idx}"] = wandb.Histogram(vecs[:, latent_idx])
            log_dict[f"latent_{latent_idx}_mean"] = vecs[:, latent_idx].mean()
            log_dict[f"latent_{latent_idx}_std"] = vecs[:, latent_idx].std()

    return log_dict
