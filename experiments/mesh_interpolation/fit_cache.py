"""Phase 0 step 2 -- fit a latent per knee and cache its reconstructions.

For each knee in the manifest:
  1. Fit an NSM latent the production way (``nsosim.utils.fit_nsm``), using all
     four femur-model meshes (bone, cart, med_men, lat_men).
  2. Decode the latent with marching cubes (``NSM.mesh.create_mesh``) to obtain
     the four canonical-space reconstructed meshes -- the interpolation targets.
  3. Cache the latent (``<key>_latent.npy``) and the four meshes
     (``<key>_<surface>.vtk``) under ``cache/``.

Surface extraction is the slow part; this step is resumable -- knees already
cached are skipped. It requires a CUDA GPU (``nsosim.load_model`` moves the
model to CUDA).

Usage:
    python -m experiments.mesh_interpolation.fit_cache
"""

import json
import os

import numpy as np
import torch

from NSM.mesh import create_mesh

from .config import (
    CACHE_DIR,
    DEVICE,
    FIT_KWARGS,
    MANIFEST_PATH,
    MARCHING_CUBES_N_PTS,
    MESH_NAMES,
    MODEL_CONFIG_PATH,
    MODEL_STATE_PATH,
)


def _latent_path(key):
    return os.path.join(CACHE_DIR, f"{key}_latent.npy")


def _mesh_path(key, surface):
    return os.path.join(CACHE_DIR, f"{key}_{surface}.vtk")


def is_cached(key):
    """True iff the latent and all four reconstructed meshes exist for ``key``."""
    if not os.path.isfile(_latent_path(key)):
        return False
    return all(os.path.isfile(_mesh_path(key, s)) for s in MESH_NAMES)


def fit_one(record):
    """Fit and cache a single knee. Returns the latent (D,) numpy array."""
    from nsosim.utils import fit_nsm

    key = record["key"]
    mesh_paths = [record["mesh_paths"][name] for name in MESH_NAMES]

    print(f"[{key}] fitting NSM latent ...")
    recon = fit_nsm(MODEL_STATE_PATH, MODEL_CONFIG_PATH, mesh_paths, **FIT_KWARGS)

    latent = np.asarray(recon["latent"], dtype=np.float64).reshape(-1)
    model = recon["mesh_result"]["model"]

    print(f"[{key}] decoding {len(MESH_NAMES)} surfaces (marching cubes) ...")
    latent_t = torch.as_tensor(latent, dtype=torch.float, device=DEVICE)
    meshes = create_mesh(
        model,
        latent_t,
        n_pts_per_axis=MARCHING_CUBES_N_PTS,
        objects=len(MESH_NAMES),
        device=DEVICE,
    )
    if not isinstance(meshes, list):
        meshes = [meshes]

    np.save(_latent_path(key), latent)
    for name, mesh in zip(MESH_NAMES, meshes):
        mesh.save(_mesh_path(key, name))
    print(f"[{key}] cached latent + {len(meshes)} meshes")
    return latent


def main():
    if DEVICE != "cuda":
        print(
            "WARNING: no CUDA device found. nsosim.load_model moves the model "
            "to CUDA, so fitting will fail on CPU. Run this step on a GPU node."
        )
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(MANIFEST_PATH) as f:
        records = json.load(f)

    for record in records:
        key = record["key"]
        if is_cached(key):
            print(f"[{key}] already cached -- skipping")
            continue
        fit_one(record)

    print(f"\nDone. Cache at {CACHE_DIR}")


if __name__ == "__main__":
    main()
