"""
Leaf helpers for the SDF dataset pipeline: seeding, sampling-cube arithmetic,
normalization, mesh utilities, and the ``.npz`` cache layout.

Moved verbatim out of ``sdf_dataset.py`` (plan §8.0, slice A). Everything here is
importable from ``NSM.datasets`` and ``NSM.datasets.sdf_dataset`` as before; this
module holds the definitions. Nothing in this module may import from other
``NSM.datasets`` modules -- it is the bottom of the package's import graph.
"""

import hashlib
import logging
import os
import zipfile

import numpy as np
import torch
from pymskt.mesh import Mesh

logger = logging.getLogger(__name__)


def derive_seed(seed, *key):
    """
    Derive an independent seed for one point set from the run-level seed plus a key.

    Every point set drawn needs its own seed. One seed shared between the near- and
    far-surface passes hands them the same base surface points; shared across surfaces,
    bone and cartilage get the same offset vectors; shared across subjects, the subjects
    correlate. `key` must identify the draw by what it is sampling, never by a position
    in a list, so that reordering `list_mesh_paths` cannot change a subject's data --
    see `mesh_content_key`.

    Returns None when `seed` is None, i.e. seeding stays off.
    """
    if seed is None:
        return None
    return int(np.random.default_rng([seed, *key]).integers(2**31))


def mesh_content_key(paths):
    """
    A `derive_seed` key identifying a subject by the bytes of its meshes.

    The seed decides which points get drawn, so it should be a function of the meshes
    themselves. Keying it on the mesh path -- as the cache hash is, since the path is
    hashed into it -- means moving your data to another directory silently redraws every
    training sample under the same `random_seed`. The accepted trade-off is the converse:
    re-exporting a mesh with identical geometry but different bytes (a different VTK
    header, different float formatting) does change the seed.

    `paths` is one path or an ordered list of them, and order is significant. Anything
    with no bytes to contribute -- a None surface, which `MultiSurfaceSDFSamples` accepts
    for a subject missing that structure, or a path that does not exist -- contributes a
    marker byte instead, so [a, None] and [None, a] still differ. A missing path is not
    an error here because the samplers skip that whole subject a moment later; raising
    would make a seeded run fail where an unseeded one skips.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    digest = hashlib.md5()
    for path in paths:
        if path is None or not os.path.exists(path):
            digest.update(b"\0")
        else:
            with open(path, "rb") as file:
                digest.update(file.read())
    return int(digest.hexdigest(), 16)


def get_rand_uniform_pts(n_pts, mins=(-1, -1, -1), maxs=(1, 1, 1), seed=None):
    """
    Sample ``n_pts`` points uniformly from the axis-aligned box ``[mins, maxs]``.

    Args:
        n_pts (int): Number of points to sample
        mins (tuple, optional): Per-axis lower bounds. Defaults to (-1, -1, -1).
        maxs (tuple, optional): Per-axis upper bounds. Defaults to (1, 1, 1).
        seed (int, optional): Seed for this draw. Defaults to None (unseeded).

    Returns:
        np.ndarray: (n_pts, 3) array of points
    """
    # seed=None stays on the legacy global stream, so an unseeded call draws exactly the
    # numbers it always did; default_rng is a different stream entirely.
    rand_gen = np.random.uniform if seed is None else np.random.default_rng(seed).uniform

    pts = np.zeros((n_pts, len(mins)))
    mins = np.tile(mins, [n_pts, 1])
    pts[:, :] = rand_gen(mins, maxs)

    return pts


def get_pts_center_and_scale(pts, scale_method="max_rad", return_pts=False, pts_center=None):
    """
    Given a set of points, return the center and scale that normalize them.

    Centering and scaling are **unconditional**. This took ``center`` and ``scale``
    booleans until Aug 2026, but both were shadowed by the values computed from them
    before they were ever read, so neither had any effect at any value. They are removed
    rather than honoured: every caller passes ``scale=norm_pts``, which defaults to
    ``False`` everywhere and is unset in the shipped configs, so making the argument
    authoritative would stop scaling on a default run and change the coordinate frame of
    every dataset, checkpoint and reconstruction ever produced. See #20.

    ``pts`` is not modified; it is copied first. It was mutated in place until Aug 2026,
    which every in-repo caller worked around with a defensive ``np.copy()``. See #21.

    Args:
        pts (np.ndarray): (n_pts, 3) array of points
        scale_method (str, optional): Method to scale the points. Defaults to 'max_rad'.
        return_pts (bool, optional): Whether to also return the normalized points. Defaults to False.
        pts_center (np.ndarray, optional): (n_pts, 3) array to take the center from instead
            of ``pts``. Used to center on the bone alone while scaling by bone + cartilage.

    Returns:
        tuple: ``(center, scale)``, or ``(center, scale, pts)`` if ``return_pts``

    Raises:
        NotImplementedError: If scale_method is not implemented
    """

    pts = np.copy(pts)

    if pts_center is None:
        center = np.mean(pts, axis=0)
    else:
        center = np.mean(pts_center, axis=0)
    pts -= center

    if scale_method == "max_rad":
        scale = np.max(np.linalg.norm(pts, axis=-1), axis=-1)
        pts /= scale
    else:
        raise NotImplementedError(f"Scale Method ** {scale_method} ** Not Implemented")

    if return_pts is True:
        return center, scale, pts

    return center, scale


def is_zipfile(filename):
    """``zipfile.is_zipfile`` that returns False on unreadable paths instead of raising."""
    try:
        return zipfile.is_zipfile(filename)
    except (IOError, zipfile.BadZipfile):
        return False


def meshfix(mesh, assert_=False, assert_error=0.01):
    """
    Fix a mesh in place with pymskt's meshfix wrapper, printing the point-count change.

    Degenerate meshes break SDF fitting, which is why the datasets run this on every
    mesh they read unless ``fix_mesh=False``.

    Args:
        mesh (mskt.mesh.Mesh): Mesh to fix. Modified in place.
        assert_ (bool, optional): If True, raise ``AssertionError`` when fixing dropped
            ``assert_error`` or more of the original points. Defaults to False.
        assert_error (float, optional): Tolerated fraction of dropped points.
            Defaults to 0.01.
    """
    n_pts_orig = mesh.point_coords.shape[0]
    mesh.fix_mesh()
    n_pts_fixed = mesh.point_coords.shape[0]
    # Asserting that no more than 1% of the mesh points were removed
    logger.info(
        "Fixed mesh, %s -> %s (%.2f%%)",
        n_pts_orig,
        n_pts_fixed,
        (n_pts_fixed - n_pts_orig) / n_pts_orig * 100,
    )
    if assert_ is True:
        assert (n_pts_orig - n_pts_fixed) < (
            assert_error * n_pts_orig
        ), f"Mesh dropped too many points, {n_pts_orig} -> {n_pts_fixed}"


def get_cube_mins_maxs(pts):
    """
    The cube the uniform sampler draws from: centred on the centroid of ``pts``, with
    half-width equal to the largest centroid-to-point distance.

    That circumscribes the points' bounding *sphere*, so it is deliberately larger than
    their axis-aligned bounding box -- it matches the ``max_rad`` normalization, which
    scales by the same radius.

    Args:
        pts (np.ndarray): (n_pts, 3) array of points

    Returns:
        tuple: (mins, maxs) of the cube, each an np array of shape (3,)

    Raises:
        ValueError: If input is empty or has wrong dimensions
    """
    if pts.size == 0:
        raise ValueError("Input array is empty")
    if len(pts.shape) != 2 or pts.shape[1] != 3:
        raise ValueError("Input must be a 2D array with shape (n_pts, 3)")

    mean = np.mean(pts, axis=0)
    norm_pts = pts - mean
    radial_max = np.max(np.linalg.norm(norm_pts, axis=-1))
    mins = mean - radial_max
    maxs = mean + radial_max

    return mins, maxs


def get_buffered_cube_mins_maxs(pts, buffer):
    """
    The uniform-sampling cube around ``pts``, expanded symmetrically by ``buffer``.

    Each side moves out by ``buffer / 2`` of the cube's span, so the span grows by a
    factor of ``1 + buffer`` and the centre stays put. The buffer exists so the cube
    still covers the model's full [-1, 1] domain when normalization leaves the object
    smaller than it -- e.g. ``scale_jointly`` with a ``joint_scale_buffer`` (48c5f60).

    One helper for both samplers on purpose: the two carried private copies of this
    arithmetic until Aug 2026 and they diverged -- ``mins`` was defined first, and then
    used when defining ``maxs``, so a nonzero buffer grew the cube more above than
    below, and only the single-mesh copy clipped the result. See ``docs/KNOWN_ISSUES.md``
    § History (#40).

    Args:
        pts (np.ndarray): (n_pts, 3) array of points
        buffer (float): Fraction of the span added across each axis (half per side).

    Returns:
        tuple: (mins, maxs) of the expanded cube, each an np array of shape (3,)
    """
    mins, maxs = get_cube_mins_maxs(pts)
    span = maxs - mins
    return mins - buffer / 2 * span, maxs + buffer / 2 * span


def unpack_pts(data, pts_name="orig_pts"):
    """
    Rebuild one key group from a loaded ``.npz`` cache into a per-surface list.

    ``save_data_to_cache`` flattens list-valued entries to indexed keys
    (``new_pts_0``, ``new_pts_1``, ...); this reads ``{pts_name}_0..N`` back. Keys are
    matched by *substring*, so a ``pts_name`` that is a prefix of another stored group
    would miscount its members -- none of the group names this is called with collide
    that way.

    Args:
        data (np.lib.npyio.NpzFile): Loaded ``.npz`` cache file.
        pts_name (str, optional): Name of the key group to unpack. Defaults to 'orig_pts'.

    Returns:
        list: Torch tensors; index position = surface position. Empty when the group
        is absent.
    """
    # get original points...
    pts = []
    pts_arrays = [x for x in data.files if f"{pts_name}_" in x]
    if len(pts_arrays) > 0:
        for pts_idx in range(len(pts_arrays)):
            pts.append(torch.from_numpy(data[f"{pts_name}_{pts_idx}"]))

    return pts


def unpack_numpy_data(
    data_,
    point_cloud=False,
    list_additional_keys=["orig_pts", "new_pts", "pos_idx", "neg_idx", "surf_idx"],
):
    """
    Normalize a cached sample dict to the in-memory layout the datasets use.

    Accepts the key spellings the cache has used over time -- ``pts``/``xyz`` for
    coordinates, ``sdfs``/``gt_sdf``/``sdf`` for signed distances -- and returns them
    as float32 tensors under ``xyz`` / ``gt_sdf``, plus each requested key group
    unpacked into a per-surface list (see ``unpack_pts``).

    Args:
        data_ (np.lib.npyio.NpzFile): Raw cached data. A plain dict works only with
            ``list_additional_keys=[]`` — unpacking a key group reads ``data_.files``,
            which only an NpzFile has.
        point_cloud (bool, optional): Also convert ``point_cloud``. Defaults to False.
        list_additional_keys (list, optional): Key groups to unpack into per-surface
            lists; absent groups come back as empty lists.

    Returns:
        dict: ``xyz``, ``gt_sdf``, and one list per requested key group.

    Raises:
        ValueError: If no coordinate or no SDF key is present under any known name.
    """
    data = {}

    # Get points / xyz coords
    if "pts" in data_:
        data["xyz"] = torch.from_numpy(data_["pts"]).float()
    elif "xyz" in data_:
        data["xyz"] = torch.from_numpy(data_["xyz"]).float()
    else:
        raise ValueError("No pts or xyz in cached file")

    # Get SDFs of the original points
    if "sdfs" in data_:
        data["gt_sdf"] = torch.from_numpy(data_["sdfs"]).float()
    elif "gt_sdf" in data_:
        data["gt_sdf"] = torch.from_numpy(data_["gt_sdf"]).float()
    elif "sdf" in data_:
        data["gt_sdf"] = torch.from_numpy(data_["sdf"]).float()
    else:
        raise ValueError("No sdfs or gt_sdf or sdf in cached file")

    # Get random point cloud surface points... this was used for Diffusion SDFs model
    if point_cloud is True:
        data["point_cloud"] = torch.from_numpy(data_["point_cloud"]).float()

    for key in list_additional_keys:
        key_data = unpack_pts(data_, pts_name=key)
        data[key] = key_data

    return data


def check_probabilities(p_):
    """Raise ValueError unless ``p_`` is in [0, 1]."""
    if (p_ < 0) or (p_ > 1):
        raise ValueError("Probabilities must be between 0 and 1")


def check_probabilities_sum(p_near_, p_far_):
    """Raise ValueError if the near + far shares exceed 1 (the rest samples uniformly)."""
    if p_near_ + p_far_ > 1:
        raise ValueError("sum of p_near_ & p_far_ must be <=1")


def combine_meshes(meshes, mesh_indices):
    """
    Combine the selected meshes into a single pymskt Mesh.

    Args:
        meshes (list): List of Mesh objects
        mesh_indices (list or int): Indices of meshes to combine

    Returns:
        Mesh: The single selected mesh unchanged, or -- when two or more are combined --
        a new Mesh wrapping their union. pymskt's ``+`` operator returns a pyvista
        ``PolyData``, which has no ``save_mesh``, so the combined result is rewrapped
        before returning (#61).
    """
    if isinstance(mesh_indices, int):
        return meshes[mesh_indices]

    if len(mesh_indices) == 1:
        return meshes[mesh_indices[0]]

    # Start with the first mesh and add subsequent meshes
    combined_mesh = meshes[mesh_indices[0]]

    for idx in mesh_indices[1:]:
        if meshes[idx] is not None:
            combined_mesh = combined_mesh + meshes[idx]

    return Mesh(combined_mesh)
