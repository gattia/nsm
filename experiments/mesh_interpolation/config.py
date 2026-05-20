"""Shared configuration for the mesh-interpolation Phase 0 experiment.

Paths, the model spec, the experiment matrix, and small shared helpers. See
``README.md`` and the plan ``NSM_MESH_INTERPOLATION_IMPROVEMENTS.md``.
"""

import os

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Data / model paths (see plan section 2.1)
# ---------------------------------------------------------------------------

DEMOGRAPHICS_CSV = (
    "/dataNAS/people/aagatti/projects/OAI_DESS/aging_trajectories/data/"
    "demographics/0_demographics_baseline.csv"
)
MESH_ROOT = "/dataNAS/people/aagatti/projects/OAI_DESS/meshes/00m"

_MODEL_DIR = (
    "/dataNAS/people/aagatti/projects/comak_gait_simulation/"
    "COMAK_SIMULATION_REQUIREMENTS/nsm_models/568_nsm_femur_bone_cart_men_v0.0.1"
)
MODEL_STATE_PATH = os.path.join(_MODEL_DIR, "model", "2000.pth")
MODEL_CONFIG_PATH = os.path.join(_MODEL_DIR, "model_params_config.json")

# The femur model is a joint 4-surface decoder. surface_idx -> name.
MESH_NAMES = ["bone", "cart", "med_men", "lat_men"]
# Mesh-file suffixes per surface, under MESH_ROOT/{id}/{id}_{SIDE}_<suffix>.vtk
MESH_SUFFIXES = {
    "bone": "femur",
    "cart": "femur_cart",
    "med_men": "med_men",
    "lat_men": "lat_men",
}

# Output cache / report locations (relative to this file).
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")
REPORT_DIR = os.path.join(HERE, "report")

# ---------------------------------------------------------------------------
# Subject selection (plan section 2.1): 10 knees, KL-stratified, KL3/4 excluded.
# ---------------------------------------------------------------------------

KL_QUOTA = {0: 4, 1: 3, 2: 3}  # KL grade -> number of knees
SELECTION_SEED = 0

# ---------------------------------------------------------------------------
# Production NSM-fitting settings (plan section 2.1; verified against nsosim).
# ---------------------------------------------------------------------------

FIT_KWARGS = dict(
    n_samples_latent_recon=20_000,
    num_iter=None,  # -> model config's num_iterations_recon (2000)
    convergence_patience=10,  # nsosim production override
    use_hybrid_optimizer=False,
    seed=0,
)

# Marching-cubes grid resolution for cached reconstructions.
MARCHING_CUBES_N_PTS = 256

# ---------------------------------------------------------------------------
# Experiment matrix (plan section 3.7)
# ---------------------------------------------------------------------------

# Each config maps to keyword arguments for `interpolate_points`.
#
# `fix1_fix2` is the known-good base (corrector + Newton magnitude). Fix 4
# (tangent Laplacian) and Fix 5 (adaptive steps) are each composed on that base
# *in isolation* -- the plan's cumulative ladder routed them through Fix 3, but
# the Phase 0 run showed Fix 3 (latent predictor) scrambles the mesh
# (fold-over ~0.5), so it is dropped and the remaining fixes are tested cleanly.
# The `*_fix3*` / `all` configs are retained for the record but not submitted.
EXPERIMENT_CONFIGS = {
    "baseline": {},
    "fix1": {"n_corrector_iters": 5},
    "fix2": {"step_magnitude": "newton"},
    "fix1_fix2": {"n_corrector_iters": 5, "step_magnitude": "newton"},
    "fix6": {"n_corrector_iters": 5, "step_magnitude": "line_search"},
    "fix1_fix2_fix4": {
        # Original Fix 4 -- no boundary pinning. Kept matching its already-
        # cached shards (Phase 0 showed it costs ASSD via rim contraction).
        "n_corrector_iters": 5,
        "step_magnitude": "newton",
        "tangent_laplacian": True,
        "tangent_laplacian_pin_boundary": False,
    },
    "fix1_fix2_fix4b": {
        # Boundary-aware Fix 4: rim vertices pinned so Laplacian smoothing
        # cannot contract the mesh boundary.
        "n_corrector_iters": 5,
        "step_magnitude": "newton",
        "tangent_laplacian": True,
        "tangent_laplacian_pin_boundary": True,
    },
    "fix1_fix2_fix7": {
        # Fix 7 -- smoothed-normal projection (smooths the gradient field, not
        # positions; no boundary collapse). Should reduce fold-over without
        # the cart / lat_men ASSD penalty.
        "n_corrector_iters": 5,
        "step_magnitude": "newton",
        "smooth_normals": True,
    },
    "fix1_fix2_fix4b_fix7": {
        # Both smoothing strategies stacked.
        "n_corrector_iters": 5,
        "step_magnitude": "newton",
        "tangent_laplacian": True,
        "tangent_laplacian_pin_boundary": True,
        "smooth_normals": True,
    },
    # ---- Fix 4c / 7c: dihedral-aware seam detection -----------------------
    # Phase 0 showed the OAI cart/menisci meshes are topologically closed,
    # so Fix 4b's topological-boundary pin was a no-op. These configs
    # detect the *geometric* seam via dihedral-angle thresholding (any edge
    # whose two incident face normals differ by > 60 deg) and pin those
    # vertices. Fix 7c also gains a hard step-magnitude clamp to bound the
    # (g.d)-denominator pathology, and holds feature vertices' normals
    # fixed during the Laplacian smoothing of the normal field.
    "fix1_fix2_fix4c": {
        "n_corrector_iters": 5,
        "step_magnitude": "newton",
        "tangent_laplacian": True,
        "tangent_laplacian_pin_boundary": True,
        "tangent_laplacian_feature_angle": 60.0,
    },
    "fix1_fix2_fix7c": {
        "n_corrector_iters": 5,
        "step_magnitude": "newton",
        "smooth_normals": True,
        "smooth_normals_max_step": 0.05,
        "tangent_laplacian_feature_angle": 60.0,
    },
    "fix1_fix2_fix4c_fix7c": {
        "n_corrector_iters": 5,
        "step_magnitude": "newton",
        "tangent_laplacian": True,
        "tangent_laplacian_pin_boundary": True,
        "smooth_normals": True,
        "smooth_normals_max_step": 0.05,
        "tangent_laplacian_feature_angle": 60.0,
    },
    "fix1_fix2_fix5": {
        "n_corrector_iters": 5,
        "step_magnitude": "newton",
        "adaptive_steps": True,
    },
    # --- retained for the record; Fix 3 rejected (see note above) ---
    "fix1_fix2_fix3": {
        "n_corrector_iters": 5,
        "step_magnitude": "newton",
        "latent_predictor": True,
    },
    "fix1_fix2_fix3_fix4": {
        "n_corrector_iters": 5,
        "step_magnitude": "newton",
        "latent_predictor": True,
        "tangent_laplacian": True,
    },
    "all": {
        "n_corrector_iters": 5,
        "step_magnitude": "newton",
        "latent_predictor": True,
        "tangent_laplacian": True,
        "adaptive_steps": True,
    },
}

# NFE sensitivity grid (plan section 2.4).
NFE_GRID = [10, 25, 50, 100, 200]

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_nsm_model():
    """Load the 4-surface femur NSM decoder onto the available device.

    nsosim's ``load_model`` hard-codes ``.cuda()``; this loader mirrors it but
    honours :data:`DEVICE` so the experiment can also be smoke-tested on CPU.

    Returns:
        torch.nn.Module: the loaded decoder in eval mode.
    """
    import json

    from NSM.models import TriplanarDecoder

    with open(MODEL_CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    params = {
        "latent_dim": cfg["latent_size"],
        "n_objects": cfg["objects_per_decoder"],
        "conv_hidden_dims": cfg["conv_hidden_dims"],
        "conv_deep_image_size": cfg["conv_deep_image_size"],
        "conv_norm": cfg["conv_norm"],
        "conv_norm_type": cfg["conv_norm_type"],
        "conv_start_with_mlp": cfg["conv_start_with_mlp"],
        "sdf_latent_size": cfg["sdf_latent_size"],
        "sdf_hidden_dims": cfg["sdf_hidden_dims"],
        "sdf_weight_norm": cfg["weight_norm"],
        "sdf_final_activation": cfg["final_activation"],
        "sdf_activation": cfg["activation"],
        "sdf_dropout_prob": cfg["dropout_prob"],
        "sum_sdf_features": cfg["sum_conv_output_features"],
        "conv_pred_sdf": cfg["conv_pred_sdf"],
    }
    model = TriplanarDecoder(**params)
    state = torch.load(MODEL_STATE_PATH, map_location=DEVICE)
    model.load_state_dict(state["model"])
    model = model.to(DEVICE)
    model.eval()
    return model


def evaluate_sdf(model, points, latent, surface_idx):
    """Forward-only SDF evaluation of one surface at a set of points.

    Args:
        model: the NSM decoder.
        points: (N, 3) array/tensor of query points.
        latent: (D,) latent vector.
        surface_idx: which decoder output to read.

    Returns:
        np.ndarray: (N,) signed-distance values.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    pts = torch.as_tensor(np.asarray(points), device=device, dtype=dtype)
    lat = torch.as_tensor(np.asarray(latent), device=device, dtype=dtype).reshape(1, -1)
    lat = lat.expand(pts.shape[0], -1)
    with torch.no_grad():
        sdf = model(torch.cat([lat, pts], dim=1))
    return sdf[:, surface_idx].detach().cpu().numpy()
