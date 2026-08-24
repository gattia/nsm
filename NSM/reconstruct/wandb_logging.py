"""
Preparation of reconstruction results for wandb logging: 3D point-cloud objects plus
JSON-serializable filtering.

Moved verbatim from ``main.py`` in the §8.0.C split; its own module rather than part of
an evaluation module because ``reconstruct_mesh`` calls it — parking it beside
``get_mean_errors`` would make ``main`` and evaluation import each other. ``main.py``
re-imports both names (public API, pinned by ``test_reconstruct_import_compat``).
"""

import copy

import numpy as np
import torch

# Optional (#5): every wandb use is behind an explicit request that raises when absent.
try:
    import wandb
except ImportError:
    wandb = None


def _process_meshes_for_wandb(meshes, mesh_prefix, max_points_3d, log_faces, verbose):
    """
    Helper function to process a list of meshes for wandb logging.

    Args:
        meshes (list): List of mesh objects to process
        mesh_prefix (str): Prefix for wandb keys (e.g., "recon_mesh", "orig_mesh")
        max_points_3d (int): Maximum number of points to log (subsampled if exceeded)
        log_faces (bool): Whether to include mesh faces in 3D visualization if available
        verbose (bool): Whether to print processing details

    Returns:
        dict: Dictionary with wandb-ready mesh data
    """
    mesh_data = {}

    for i, mesh in enumerate(meshes):
        if mesh is not None and hasattr(mesh, "point_coords"):
            points = mesh.point_coords

            # Subsample if too many points
            if len(points) > max_points_3d:
                if verbose:
                    print(
                        f"Subsampling {mesh_prefix}_{i} from {len(points)} to {max_points_3d} points"
                    )
                indices = np.random.choice(len(points), max_points_3d, replace=False)
                points = points[indices]

            # Create 3D object with or without faces
            if log_faces and hasattr(mesh, "faces") and mesh.faces is not None:
                try:
                    mesh_data[f"{mesh_prefix}_{i}"] = wandb.Object3D(
                        {"type": "lidar/beta", "points": points, "faces": mesh.faces}
                    )
                except Exception as e:
                    if verbose:
                        print(
                            f"Failed to log faces for {mesh_prefix}_{i}, logging points only: {e}"
                        )
                    mesh_data[f"{mesh_prefix}_{i}"] = wandb.Object3D(points)
            else:
                mesh_data[f"{mesh_prefix}_{i}"] = wandb.Object3D(points)

            # Log mesh statistics
            mesh_data[f"{mesh_prefix}_{i}_n_points"] = len(mesh.point_coords)
            if hasattr(mesh, "faces") and mesh.faces is not None:
                mesh_data[f"{mesh_prefix}_{i}_n_faces"] = len(mesh.faces)

    return mesh_data


def prepare_results_for_wandb(result, max_points_3d=10000, log_faces=True, verbose=False):
    """
    Prepare reconstruction results for wandb logging with 3D point cloud visualization and robust JSON serialization.

    Args:
        result (dict): Dictionary containing reconstruction results
        max_points_3d (int): Maximum number of points to log for 3D visualization (subsampled if exceeded)
        log_faces (bool): Whether to include mesh faces in 3D visualization if available
        verbose (bool): Whether to print preparation details

    Returns:
        dict: Dictionary ready for wandb logging (JSON serializable + 3D objects)
    """
    if wandb is None:
        raise ImportError("prepare_results_for_wandb requires wandb, which is not installed")
    if verbose:
        print("Preparing results for wandb logging...")

    # Create a copy to avoid modifying the original
    result_wandb = copy.copy(result)

    # Process reconstructed meshes
    if "mesh" in result_wandb and result_wandb["mesh"] is not None:
        recon_mesh_data = _process_meshes_for_wandb(
            result_wandb["mesh"], "recon_mesh", max_points_3d, log_faces, verbose
        )
        result_wandb.update(recon_mesh_data)

    # Process original meshes
    if "orig_mesh" in result_wandb and result_wandb["orig_mesh"] is not None:
        orig_mesh_data = _process_meshes_for_wandb(
            result_wandb["orig_mesh"], "orig_mesh", max_points_3d, log_faces, verbose
        )
        result_wandb.update(orig_mesh_data)

    # Robust JSON serialization filtering
    keys_to_delete = []

    for key, value in result_wandb.items():
        if value is None:
            continue  # None is JSON serializable
        elif isinstance(value, (int, float, str, bool, list, dict, tuple)):
            continue  # Basic JSON types + tuple
        elif isinstance(value, (np.integer, np.floating)):
            result_wandb[key] = float(value)  # Convert numpy scalars
            continue
        elif isinstance(value, np.ndarray):
            if value.size <= 10:  # Only log small arrays
                result_wandb[key] = value.tolist()
                continue
            else:
                if verbose:
                    print(f"Removing large numpy array '{key}' with size {value.size}")
                keys_to_delete.append(key)
        elif isinstance(value, torch.Tensor):
            if value.numel() <= 10:  # Only log small tensors
                result_wandb[key] = value.detach().cpu().numpy().tolist()
                continue
            else:
                if verbose:
                    print(f"Removing large tensor '{key}' with {value.numel()} elements")
                keys_to_delete.append(key)
        elif hasattr(value, "__class__") and "wandb" in str(type(value)):
            continue  # Keep wandb objects (like Object3D)
        else:
            if verbose:
                print(f"Removing non-serializable object '{key}' of type {type(value)}")
            keys_to_delete.append(key)

    # Delete non-serializable items
    for key in keys_to_delete:
        del result_wandb[key]

    # Delete original mesh objects (but keep the 3D point clouds we created)
    if "mesh" in result_wandb:
        del result_wandb["mesh"]
    if "orig_mesh" in result_wandb:
        del result_wandb["orig_mesh"]

    if verbose:
        print(f"Prepared {len(result_wandb)} items for wandb logging")

    return result_wandb
