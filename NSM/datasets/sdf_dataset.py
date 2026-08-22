import gc
import hashlib
import multiprocessing
import os
import time
import warnings
import zipfile
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import point_cloud_utils as pcu
import pymskt as mskt
import torch
import vtk
from pymskt.mesh import Mesh
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

try:
    from pympler import muppy, tracker  # asizeof, summary, muppy, tracker
except ModuleNotFoundError:
    print(
        "Pympler not installed, cannot use asizeof - if trying to debug memory usage, install pympler"
    )


today_date = datetime.now().strftime("%b_%d_%Y")


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
    print(
        f"Fixed mesh, {n_pts_orig} -> {n_pts_fixed} ({(n_pts_fixed - n_pts_orig) / n_pts_orig * 100:.2f}%)"
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


def read_mesh_get_sampled_pts(
    path,
    sigma=1,
    n_pts=200000,
    rand_function="normal",
    center_pts=True,
    norm_pts=False,
    scale_method="max_rad",
    get_random=True,
    register_to_mean_first=False,
    mean_mesh=None,
    fix_mesh=True,
    include_surf_in_pts=False,
    uniform_pts_buffer=0.0,
    seed=None,
    # Single mesh specific
    return_point_cloud=False,
    **kwargs,
):
    """
    Read one mesh; optionally register and normalize it; sample SDF points from it.

    Args:
        path (str): Path to mesh
        sigma (float or None, optional): Width of the perturbation applied to surface
            points when drawing random samples. None instead draws uniformly from the
            buffered cube around the mesh. Defaults to 1.
        n_pts (int, optional): Number of points to sample. Defaults to 200000.
        rand_function (str, optional): Distribution to sample from. Defaults to 'normal'. Also supports 'laplace'.
        center_pts (bool, optional): Defaults to True.
        norm_pts (bool, optional): Defaults to False. The two flags decide *whether*
            normalization runs, not which half of it: if either is True the mesh is both
            centered and scaled by ``get_pts_center_and_scale``; only False/False leaves
            coordinates untouched. See ``docs/KNOWN_ISSUES.md`` § Open,
            "``center_pts`` and ``norm_pts`` do not select which normalization happens".
        scale_method (str, optional): Method to scale the points. Defaults to 'max_rad'.
        get_random (bool, optional): Sample random points around the surface (True), or
            return the surface vertices themselves with SDF 0 (False). Defaults to True.
        register_to_mean_first (bool, optional): Similarity-register the mesh to
            ``mean_mesh`` before any normalization -- rigid + uniform scale, so the
            mesh comes out at ``mean_mesh``'s size. Defaults to False.
        mean_mesh (vtkPolyData or mskt.mesh.Mesh, optional): Mean mesh to register to. Defaults to None.
        return_point_cloud (bool, optional): Also store the normalized surface points
            under ``"point_cloud"``. Defaults to False.
        fix_mesh (bool, optional): Whether to fix the mesh (using meshfix). Defaults to True.
        include_surf_in_pts (bool, optional): Append the surface vertices to the random
            points, so ``"xyz"`` holds ``n_pts`` + n_vertices rows. Defaults to False.
        uniform_pts_buffer (float, optional): Expansion of the uniform sampling cube, as a
            fraction of its span; see get_buffered_cube_mins_maxs. Only used when sigma is
            None. Defaults to 0.0.
        seed (int, optional): Seed for the random draw. Defaults to None (unseeded).

    Returns:
        dict or None: None when ``path`` does not exist -- the datasets skip that
        subject. Otherwise:

        - ``"orig_pts"``, ``"new_pts"``, ``"orig_mesh"``, ``"new_mesh"``: one-element
          lists (points and mesh, before / after transformation), matching
          ``read_meshes_get_sampled_pts``'s per-surface layout.
        - ``"scale"``, ``"center"``: the normalization applied (1 and zeros when none).
        - ``"icp_transform"``: the registration transform, or None.
        - With ``get_random=True``: sample coordinates under ``"xyz"`` (n, 3), signed
          distances under ``"sdf"`` (n,), and ``"pts_surface"`` (n,) of zeros.
        - With ``get_random=False``: the surface vertices under ``"pts"`` -- a
          *different key* than the random path's ``"xyz"`` -- with ``"sdf"`` all zeros.

    Notes:
        Unknown keyword arguments are swallowed silently, except the historical
        ``return_*`` flags and ``mean``, which print a deprecation line.
    """
    # Accepted for backwards compatibility and ignored: each is now unconditionally on,
    # so passing False did not do what it said. They were documented as live parameters
    # defaulting to False until Aug 2026, which is the opposite of what this does.
    # `mean` is different: no code path ever read it, so it was removed outright (Aug
    # 2026) and lands in kwargs for old callers.
    list_deprecated = [
        "return_scale",
        "return_center",
        "return_orig_pts",
        "return_orig_mesh",
        "return_new_mesh",
    ]
    for kwarg in kwargs:
        if kwarg in list_deprecated:
            print(f"{kwarg} is deprecated and not used in this function - always True")
        elif kwarg == "mean":
            print("mean is deprecated and not used in this function - it never had an effect")

    results = {}

    # if mesh path does not exist, return None (skipping)
    if os.path.exists(path) is False:
        print(f"Skipping ... path does not exist, {path}")
        return None

    # read in mesh & "fix" using meshfix if requested
    orig_mesh = Mesh(path)
    if fix_mesh is True:
        meshfix(orig_mesh)

    new_mesh = orig_mesh.copy()

    # return orig_pts expanded dims for compatibility when storing
    # multiple meshes in the same dictionary
    results["orig_pts"] = [orig_mesh.point_coords]

    if register_to_mean_first is True:
        print("Registering mesh to mean mesh")
        # Rigid + scaling alginment of the original mesh to the mean mesh
        # of the model. This allows all downstream scaling to occur as expected
        # it also aligns the new bone with the mean/expected bone of the shape model
        # to maximize fidelity of the reconstruction.

        if mean_mesh is None:
            raise Exception("Must provide mean mesh to register to")
        icp_transform = orig_mesh.rigidly_register(
            other_mesh=mean_mesh,
            as_source=True,
            apply_transform_to_mesh=False,
            return_transformed_mesh=False,
            max_n_iter=100,
            n_landmarks=1000,
            reg_mode="similarity",
            return_transform=True,
        )
        results["icp_transform"] = icp_transform
        new_mesh.apply_transform_to_mesh(icp_transform)
    else:
        print("No registration")
        results["icp_transform"] = None

    if (center_pts is True) or (norm_pts is True):
        print("Scaling and centering mesh")
        center, scale, new_pts = get_pts_center_and_scale(
            new_mesh.point_coords,
            scale_method=scale_method,
            return_pts=True,
        )
        new_mesh.point_coords = new_pts
    else:
        print("Not scaling or centering mesh")
        scale = 1
        center = np.zeros(3)
        new_pts = new_mesh.point_coords

    results["new_pts"] = [new_pts]

    if get_random is True:
        if sigma is not None:
            rand_pts = new_mesh.rand_pts_around_surface(
                n_pts=n_pts,
                surface_method="random",
                distribution=rand_function,
                sigma=sigma,
                seed=seed,
            )
        else:
            mins, maxs = get_buffered_cube_mins_maxs(new_pts, uniform_pts_buffer)
            rand_pts = get_rand_uniform_pts(n_pts, mins=mins, maxs=maxs, seed=seed)

        if include_surf_in_pts is True:
            rand_pts = np.concatenate([rand_pts, new_pts], axis=0)

        rand_sdf = new_mesh.get_sdf_pts(pts=rand_pts, method="pcu")

        results["xyz"] = rand_pts
        results["sdf"] = rand_sdf
        results["pts_surface"] = np.zeros(rand_pts.shape[0], dtype=np.int64)
    else:
        results["pts"] = new_pts
        results["sdf"] = np.zeros(new_pts.shape[0])
        results["pts_surface"] = np.zeros(new_pts.shape[0], dtype=np.int64)

    if return_point_cloud is True:
        results["point_cloud"] = new_pts

    # Theres no reason we shouldnt include these... they are always there and take up effectively no space.
    results["scale"] = scale
    results["center"] = center

    results["orig_mesh"] = [orig_mesh]
    results["new_mesh"] = [new_mesh]

    return results


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


def read_meshes_get_sampled_pts(
    paths,
    sigma=[1, 1],
    n_pts=[200000, 200000],
    rand_function="normal",
    center_pts=True,
    norm_pts=False,
    scale_method="max_rad",
    get_random=True,
    register_to_mean_first=False,
    mean_mesh=None,
    fix_mesh=True,
    include_surf_in_pts=False,
    # Multiple mesh specific
    scale_all_meshes=True,
    center_all_meshes=False,
    mesh_to_scale=0,
    verbose=False,
    icp_transform=None,
    uniform_pts_buffer=0.0,
    seed=None,
    **kwargs,
):
    """
    Read one subject's surfaces; register and normalize them jointly; sample SDF points.

    Args:
        paths (list): Per-surface mesh paths for one subject, in a fixed surface order.
            A None entry marks a missing surface: it is carried through as None
            placeholders. A path that does not *exist* is different -- the whole subject
            returns None.
        sigma (list, optional): Per-surface perturbation widths for draws around each
            surface. A None entry draws that surface's points uniformly from one cube
            around all surfaces jointly. Defaults to [1, 1].
        n_pts (list, optional): Number of points to sample per mesh; 0 contributes no
            points for that surface. Defaults to [200000, 200000].
        rand_function (str, optional): Distribution to sample from. Defaults to 'normal'. Also supports 'laplace'.
        center_pts (bool, optional): Defaults to True.
        norm_pts (bool, optional): Defaults to False. As in
            ``read_mesh_get_sampled_pts``: the two flags decide *whether* normalization
            runs, not which half of it -- either being True both centers and scales; see
            ``docs/KNOWN_ISSUES.md`` § Open.
        scale_method (str, optional): Method to scale the points. Defaults to 'max_rad'.
        get_random (bool, optional): Sample random points around the surfaces (True), or
            use the surface vertices themselves as the points (False). Defaults to True.
        register_to_mean_first (bool, optional): Similarity-register to ``mean_mesh``
            before any normalization -- rigid + uniform scale, so the surfaces come out
            at ``mean_mesh``'s size. Defaults to False.
        mean_mesh (vtkPolyData or mskt.mesh.Mesh, optional): Mean mesh to register to. Defaults to None.
        fix_mesh (bool, optional): Whether to fix meshes (using meshfix). Defaults to True.
        include_surf_in_pts (bool, optional): Append each surface's vertices to its
            random points. Defaults to False.
        scale_all_meshes (bool, optional): Scale using every surface's points (True) or
            only ``mesh_to_scale``'s (False). Defaults to True.
        center_all_meshes (bool, optional): Center on every surface's points (True) or
            only ``mesh_to_scale``'s (False). Under the defaults the frame is centered
            on ``mesh_to_scale`` and scaled so every surface fits the domain -- e.g.
            centered on the bone, scaled by bone + cartilage. Defaults to False.
        mesh_to_scale (int or list, optional): Index(es) of mesh(es) to use for registration and scaling.
            If int, uses single mesh. If list, combines multiple meshes for registration. Defaults to 0.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        icp_transform (vtk.vtkTransform, optional): Pre-computed transform to apply
            instead of registering. The dataset's sampling passes reuse the first pass's
            transform this way, so all of a subject's points share one registration.
            Defaults to None.
        uniform_pts_buffer (float, optional): Expansion of the uniform sampling cube, as
            a fraction of its span; see get_buffered_cube_mins_maxs. Only used for
            surfaces whose sigma is None. Defaults to 0.0.
        seed (int, optional): Seed for the random draws; each surface gets its own seed
            derived from it. Defaults to None (unseeded).

    Returns:
        dict or None: None when any path does not exist. Otherwise:

        - ``"pts"`` (n, 3): sample coordinates, all surfaces concatenated. (This
          function's random path uses ``"pts"`` where the single-mesh one says
          ``"xyz"``; callers probe both.)
        - ``"sdf"``: list with one entry per surface -- each surface's signed distance
          to *all* n points, or None for a missing surface. With ``get_random=False``
          the entries are 0 where the points came from that same surface.
        - ``"pts_surface"`` (n,): which surface each point was drawn around, numbered
          by position in ``paths`` -- a missing surface leaves a gap in the numbering
          rather than renumbering those after it.
        - ``"orig_pts"``, ``"new_pts"``, ``"orig_mesh"``, ``"new_mesh"``: per-surface
          lists, with None placeholders for missing surfaces.
        - ``"scale"``, ``"center"``, ``"icp_transform"``: as in
          ``read_mesh_get_sampled_pts``.

    Notes:
        - When mesh_to_scale is a list (e.g., [0, 1]), multiple surfaces are combined
          with the pymskt Mesh `+` operator (see combine_meshes) for joint registration
          (e.g., medial + lateral menisci)
        - The same ICP transform is applied to all meshes regardless of registration method
        - Unknown keyword arguments are swallowed silently, except the historical
          ``return_*`` flags and ``mean``, which print a deprecation line.
    """
    tic = time.time()
    # Same contract as read_mesh_get_sampled_pts: the return_* flags are unconditionally
    # on, and `mean` was removed outright (Aug 2026) because no code path ever read it.
    list_deprecated = [
        "return_scale",
        "return_center",
        "return_orig_pts",
        "return_orig_mesh",
        "return_new_mesh",
    ]
    for kwarg in kwargs:
        if kwarg in list_deprecated:
            print(f"{kwarg} is deprecated and not used in this function - always True")
        elif kwarg == "mean":
            print("mean is deprecated and not used in this function - it never had an effect")

    # preallocate results dictionary
    results = {}

    # Read all meshes and store in list
    orig_meshes = []
    orig_pts = []
    for path in paths:
        if path is None:
            print(f"Mesh is None... returning None for meshes and pts, {path}")
            orig_meshes.append(None)
            orig_pts.append(None)
            continue
        if os.path.exists(path) is False:
            print(f"Skipping ... path does not exist, {path}")
            return None
        mesh = Mesh(path)
        # fixing meshes ensures they are not degenerate
        # degenerate meshes will cause issues fitting SDFs.
        if fix_mesh is True:
            meshfix(mesh)
        orig_meshes.append(mesh)
        orig_pts.append(mesh.point_coords)

    # return orig_pts
    results["orig_pts"] = orig_pts

    toc = time.time()
    print(f"Finished reading meshes in {toc - tic:.3f}s")
    tic = time.time()

    # Copy all meshes & points to new lists
    new_meshes = []
    new_pts = []
    for mesh_idx, orig_mesh in enumerate(orig_meshes):
        if orig_mesh is None:
            print(f"Mesh is None... returning None for meshes and pts, {paths[mesh_idx]}")
            new_meshes.append(None)
            new_pts.append(None)
            continue
        new_mesh_ = orig_mesh.copy()
        new_pts.append(new_mesh_.point_coords)
        new_meshes.append(new_mesh_)

    if register_to_mean_first is True:
        # Rigid + scaling (similarity) alginment of the original mesh to the mean mesh
        # of the model. This allows all downstream scaling to occur as expected
        # it also aligns the new bone with the mean/expected bone of the shape model
        # to maximize fidelity of the reconstruction.
        print("Registering meshes to mean mesh")
        if mean_mesh is None:
            raise ValueError("Must provide mean mesh to register to")

        if icp_transform is None:
            # Support multiple surface registration
            if isinstance(mesh_to_scale, (list, tuple)):
                print(f"Registering to multiple surfaces: {mesh_to_scale}")
                # Combine multiple meshes for registration
                registration_mesh = combine_meshes(orig_meshes, mesh_to_scale)
            else:
                # Single mesh registration (original behavior)
                registration_mesh = orig_meshes[mesh_to_scale]

            icp_transform = registration_mesh.rigidly_register(
                other_mesh=mean_mesh,
                as_source=True,
                apply_transform_to_mesh=False,
                return_transformed_mesh=False,
                max_n_iter=100,
                n_landmarks=1000,
                reg_mode="similarity",
                return_transform=True,
            )
        results["icp_transform"] = icp_transform

        # apply transform to all meshes
        for idx, new_mesh in enumerate(new_meshes):
            if new_mesh is None:
                print(f"Mesh is None... returning None for meshes and pts, {paths[idx]}")
                new_pts[idx] = None
                continue
            new_mesh.apply_transform_to_mesh(icp_transform)
            new_pts[idx] = new_mesh.point_coords

    else:
        print("No registration")
        results["icp_transform"] = None

    toc = time.time()
    print(f"Finished registering meshes in {toc - tic:.3f}s")
    tic = time.time()

    if (center_pts is True) or (norm_pts is True):
        print("Scaling and centering meshes")
        if scale_all_meshes is True:
            if any(item is None for item in new_pts):
                new_pts_ = [x for x in new_pts if x is not None]
                pts_ = np.concatenate(new_pts_, axis=0)
            else:
                pts_ = np.concatenate(new_pts, axis=0)
            if center_all_meshes is True:
                # Set as None - becuasse scaling and centering on same data
                pts_center = None
            else:
                # set specific points to center becuase they are not the same
                # for centering as they are for scaling (pts_)
                if isinstance(mesh_to_scale, (list, tuple)):
                    # Combine points from multiple meshes for centering
                    pts_center_list = [
                        new_pts[idx] for idx in mesh_to_scale if new_pts[idx] is not None
                    ]
                    pts_center = (
                        np.concatenate(pts_center_list, axis=0) if pts_center_list else None
                    )
                else:
                    pts_center = new_pts[mesh_to_scale]
        else:
            if isinstance(mesh_to_scale, (list, tuple)):
                # Combine points from multiple meshes for scaling
                pts_list = [new_pts[idx] for idx in mesh_to_scale if new_pts[idx] is not None]
                pts_ = np.concatenate(pts_list, axis=0) if pts_list else new_pts[0]
            else:
                pts_ = new_pts[mesh_to_scale]

            if center_all_meshes is True:
                # set specific points to center because scale/center are not on
                # the same data
                if any(item is None for item in new_pts):
                    new_pts_ = [x for x in new_pts if x is not None]
                    pts_center = np.concatenate(new_pts_, axis=0)
                else:
                    pts_center = np.concatenate(new_pts, axis=0)
            else:
                # Set as None - becuasse scaling and centering on same data
                pts_center = None

        center, scale = get_pts_center_and_scale(
            pts_,
            scale_method=scale_method,
            return_pts=False,
            pts_center=pts_center,
        )

        for pts_idx, new_pts_ in enumerate(new_pts):
            if new_pts_ is None:
                continue
            new_pts[pts_idx] = (new_pts_ - center) / scale
    else:
        # Do nothing to the points because they are left the same.
        scale = 1
        center = np.zeros(3)

    toc = time.time()
    print(f"Finished centering and scaling meshes in {toc - tic:.3f}s")
    tic = time.time()

    for mesh_idx, new_mesh in enumerate(new_meshes):
        if new_mesh is None:
            continue
        new_mesh.point_coords = new_pts[mesh_idx]

    results["new_pts"] = new_pts

    if get_random is True:
        rand_pts = []
        rand_sdf = []
        pts_surface = []

        if None in sigma:
            new_pts_ = [x for x in new_pts if x is not None]
            pts_cube = np.concatenate(new_pts_, axis=0)
            mins, maxs = get_buffered_cube_mins_maxs(pts_cube, uniform_pts_buffer)

        for new_pts_idx, new_mesh_ in enumerate(new_meshes):
            if new_mesh_ is None:
                continue
            if n_pts[new_pts_idx] > 0:
                seed_ = derive_seed(seed, new_pts_idx)
                if sigma[new_pts_idx] is not None:
                    rand_pts_ = new_mesh_.rand_pts_around_surface(
                        n_pts=n_pts[new_pts_idx],
                        surface_method="random",
                        distribution=rand_function,
                        sigma=sigma[new_pts_idx],
                        seed=seed_,
                    )
                else:
                    rand_pts_ = get_rand_uniform_pts(
                        n_pts[new_pts_idx], mins=mins, maxs=maxs, seed=seed_
                    )

                if include_surf_in_pts is True:
                    rand_pts_ = np.concatenate([rand_pts_, new_pts_], axis=0)

                rand_pts.append(rand_pts_)
                pts_surface.append(np.full(rand_pts_.shape[0], new_pts_idx, dtype=np.int64))
            else:
                rand_pts.append(np.zeros((0, 3)))
                # 1-D like its siblings: a (0, 3) entry here makes the concatenate below
                # raise as soon as any other surface contributed points.
                pts_surface.append(np.zeros(0, dtype=np.int64))

        rand_pts = np.concatenate(rand_pts, axis=0)
        # NOTE: pts_surface indices correspond to original mesh positions in the input list,
        # not contiguous indices. If meshes [mesh0, None, mesh2] are passed, points will be
        # labeled as surface 0 and surface 2 (no surface 1 points exist).
        # This works correctly with reconstruction code which handles missing surfaces.
        pts_surface = np.concatenate(pts_surface, axis=0)

        for new_mesh in new_meshes:
            if new_mesh is None:
                rand_sdf.append(None)
                continue
            tic_ = time.time()
            rand_sdf.append(new_mesh.get_sdf_pts(pts=rand_pts, method="pcu"))
            toc_ = time.time()
            print(f"Finished calculating SDFs in {toc_ - tic_:.3f}s")

        results["pts"] = rand_pts
        results["sdf"] = rand_sdf
        results["pts_surface"] = pts_surface
    else:
        sdfs = []
        # Need to set SDFs for the same mesh to be 0
        # but need to actually calculate the SDFs for the other
        # meshes.
        for mesh_idx, new_mesh in enumerate(new_meshes):
            if new_mesh is None:
                sdfs.append(None)
                continue
            sdfs_ = []

            for pts_idx, new_pts_ in enumerate(new_pts):
                if new_pts_ is None:
                    continue
                if verbose is True:
                    print(
                        "mesh_idx, new_mesh point_coords shape",
                        mesh_idx,
                        new_mesh.point_coords.shape,
                    )
                if pts_idx == mesh_idx:
                    if verbose is True:
                        print("adding zeros new_pts_ shape (zero)", new_pts_.shape)
                    # same mesh, set SDFs to 0
                    sdfs_.append(np.zeros(new_pts_.shape[0]))
                else:
                    # different mesh, calculate SDFs
                    _sdfs_ = new_mesh.get_sdf_pts(pts=new_pts_, method="pcu")
                    if verbose is True:
                        print("caculating SDFs for new_pts_ ", _sdfs_.shape)
                    sdfs_.append(_sdfs_)

            sdfs.append(np.concatenate(sdfs_, axis=0))

        pts_surface = []
        # NOTE: pts_surface indices correspond to original mesh positions in the input list,
        # not contiguous indices. If meshes [mesh0, None, mesh2] are passed, points will be
        # labeled as surface 0 and surface 2 (no surface 1 points exist).
        # This works correctly with reconstruction code which handles missing surfaces.
        for pts_idx, new_pts_ in enumerate(new_pts):
            if new_pts_ is None:
                continue
            pts_surface.append(np.full(new_pts_.shape[0], pts_idx, dtype=np.int64))
        pts_surface = np.concatenate(pts_surface, axis=0)

        new_pts_filtered = [x for x in new_pts if x is not None]
        results["pts"] = np.concatenate(new_pts_filtered, axis=0)

        results["sdf"] = sdfs
        results["pts_surface"] = pts_surface

    toc = time.time()
    print(f"Finished getting random points and SDFs in {toc - tic:.3f}s")

    results["new_pts"] = new_pts

    results["scale"] = scale
    results["center"] = center

    results["orig_mesh"] = orig_meshes
    results["new_mesh"] = new_meshes

    return results


def check_probabilities(p_):
    """Raise ValueError unless ``p_`` is in [0, 1]."""
    if (p_ < 0) or (p_ > 1):
        raise ValueError("Probabilities must be between 0 and 1")


def check_probabilities_sum(p_near_, p_far_):
    """Raise ValueError if the near + far shares exceed 1 (the rest samples uniformly)."""
    if p_near_ + p_far_ > 1:
        raise ValueError("sum of p_near_ & p_far_ must be <=1")


class SDFSamples(torch.utils.data.Dataset):
    """
    Dataset class for sampling SDFs from meshes.

    Args:
        list_mesh_paths (list): List of paths to meshes
        subsample (int): Number of points each __getitem__ returns. Required, and must be
            a positive int -- there is no working default (#43).
        n_pts (int, optional): Number of points to sample. Defaults to 500000.
        p_near_surface (float, optional): Proportion of points to sample near the surface. Defaults to 0.4.
        p_further_from_surface (float, optional): Proportion of points to sample further from the surface. Defaults to 0.4.
        sigma_near (float, optional): Standard deviation/scale of the distribution for points near the surface. Defaults to 0.01.
        sigma_far (float, optional): Standard deviation/scale of the distribution for points further from the surface. Defaults to 0.1.
        rand_function (str, optional): Distribution to sample from. Defaults to 'normal'. Also supports 'laplace'.
        center_pts (bool, optional): Defaults to True.
        norm_pts (bool, optional): Defaults to False. Together they decide *whether*
            per-subject normalization runs, not which half of it: if either is True
            each subject is both centered and scaled (see ``docs/KNOWN_ISSUES.md``
            § Open on ``center_pts``/``norm_pts``). Only False/False leaves coordinates
            alone, which is what ``scale_jointly`` requires.
        scale_method (str, optional): Method to scale the points. Defaults to 'max_rad'.
        scale_jointly (bool, optional): Whether to center and scale all subjects together
            after loading (norm_and_scale_all_meshes) instead of per subject; requires
            center_pts=False and norm_pts=False. Currently also requires the default
            store_data_in_memory=False -- see norm_and_scale_all_meshes. Defaults to False.
        joint_scale_buffer (float, optional): Margin added to the joint max radius when
            scale_jointly is True, so unseen subjects slightly larger than the training
            set still fit inside the model's domain. Defaults to 0.1.
        loc_save (str, optional): Directory for the cached files. Defaults to the
            LOC_SDF_CACHE environment variable, read when the dataset is constructed
            (an empty value counts as unset), else ~/.cache/nsm_sdf_cache.
        save_cache (bool, optional): Whether to save the cached files. Defaults to True.
        load_cache (bool, optional): Whether to load the cached files. Defaults to True.
        random_seed (int, optional): Seeds the sampling, and is part of the cache key.
            Every subject/surface/sigma draw gets its own seed derived from it, keyed on
            the subject's mesh *contents* rather than on list position or path, so neither
            reordering list_mesh_paths nor moving the meshes changes any subject's data.
            Defaults to None, which leaves sampling unseeded -- the historical behaviour.
            Reproducible sampling requires mskt>=0.1.21.
        reference_mesh (Mesh, str, int or list, optional): What every subject is
            similarity-registered to before sampling; None skips registration. Accepts a
            loaded Mesh, a path, an index into list_mesh_paths, or a list of paths --
            see load_reference_mesh for how each resolves. Similarity = rigid + uniform
            scale, so each subject comes out at the reference's size: between-subject
            size does not survive registration, under scale_jointly or otherwise.
            Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        equal_pos_neg (bool, optional): Draw half of every batch from positive-SDF
            samples and half from negative, instead of uniformly. Defaults to True.
        fix_mesh (bool, optional): Whether to fix the meshes (using meshfix). Defaults to True.
        print_filename (bool, optional): Whether to print the filename when loading. Defaults to False.
        multiprocessing (bool, optional): Build/load subjects in a Pool(n_processes).
            Also makes a reference_mesh spill to disk so workers can share it
            (see load_reference_mesh). Defaults to True.
        n_processes (int, optional): Pool size when multiprocessing. Defaults to 2.
        store_data_in_memory (bool, optional): Keep every subject's sample dict in
            memory (True), or keep only its cache path and reload the .npz on every
            __getitem__ (False). False requires save_cache=True. Defaults to False.
        debug_memory (bool, optional): Print a pympler memory-summary diff every 100th
            subject load (requires pympler installed). Defaults to False.
        test_load_times (bool, optional): Include time/size/mb_per_sec/whole_load_time
            in each disk-backed __getitem__ batch. Optional diagnostics, not batch
            contract: in-memory items never carry them (#22). Defaults to True.
        uniform_pts_buffer (float, optional): Expansion of the uniform sampling cube;
            see get_buffered_cube_mins_maxs. Not part of the cache key (#19). Defaults
            to 0.0.

    Notes:
        ``__getitem__`` returns ``(batch, idx)``: ``batch["xyz"]`` is (subsample, 3)
        and ``batch["gt_sdf"]`` (subsample,), float32, plus the load-time diagnostics
        when enabled.

        Caches are one ``.npz`` per subject under ``loc_save/<Mon_DD_YYYY>/``, the date
        fixed at import time; lookups search all of ``loc_save`` recursively, so hits
        cross dates. The cache key does not cover every parameter that changes the data
        -- see get_hash_params (#19).
    """

    def __init__(
        self,
        list_mesh_paths,
        subsample,
        n_pts=500000,
        p_near_surface=0.4,
        p_further_from_surface=0.4,
        sigma_near=0.01,
        sigma_far=0.1,
        rand_function="normal",
        center_pts=True,
        norm_pts=False,
        scale_method="max_rad",
        scale_jointly=False,
        joint_scale_buffer=0.1,
        loc_save=None,
        save_cache=True,
        load_cache=True,
        random_seed=None,
        reference_mesh=None,
        verbose=False,
        equal_pos_neg=True,
        fix_mesh=True,
        print_filename=False,
        multiprocessing=True,
        n_processes=2,
        store_data_in_memory=False,
        debug_memory=False,
        test_load_times=True,
        uniform_pts_buffer=0.0,
    ):

        # subsample has no working default: every build path divides by it or
        # multiplies with it, so None used to crash downstream in
        # get_samples_per_sign / sdf_pos_neg_idx instead of here (#43).
        if not isinstance(subsample, (int, np.integer)) or subsample <= 0:
            raise ValueError(
                f"subsample must be a positive int -- the number of points each "
                f"__getitem__ returns -- got {subsample!r}."
            )

        # Resolved at call time so setting LOC_SDF_CACHE before construction works; it
        # was frozen into the signature at import time until Aug 2026 (#24). An empty
        # value counts as unset, so a caller blanking the variable gets the home-cache
        # default rather than a cache rooted at the working directory.
        if loc_save is None:
            loc_save = os.environ.get("LOC_SDF_CACHE") or os.path.join(
                os.path.expanduser("~"), ".cache", "nsm_sdf_cache"
            )

        # p_near_surface & p_further_from_surface must be >=0, <=1
        # sum of p_near_surface & p_further_from_surface must be <=1
        if isinstance(p_near_surface, (list, tuple)) & isinstance(
            p_further_from_surface, (list, tuple)
        ):
            for p_near, p_far in zip(p_near_surface, p_further_from_surface):
                check_probabilities(p_near)
                check_probabilities(p_far)
                check_probabilities_sum(p_near, p_far)
        elif isinstance(p_near_surface, float) & isinstance(p_further_from_surface, float):
            check_probabilities(p_near_surface)
            check_probabilities(p_further_from_surface)
            check_probabilities_sum(p_near_surface, p_further_from_surface)
        else:
            raise ValueError(
                "p_near_surface & p_further_from_surface must be floats or lists/tuples of floats"
            )

        self.list_mesh_paths = list_mesh_paths
        self.subsample = subsample
        self.n_pts = n_pts
        self.p_near_surface = p_near_surface
        self.p_further_from_surface = p_further_from_surface
        self.sigma_near = sigma_near
        self.sigma_far = sigma_far
        self.rand_function = rand_function
        self.center_pts = center_pts
        self.norm_pts = norm_pts
        self.scale_method = scale_method
        self.scale_jointly = scale_jointly
        self.joint_scale_buffer = joint_scale_buffer
        self.loc_save = loc_save
        self.random_seed = random_seed
        self.reference_mesh = reference_mesh
        self.verbose = verbose
        self.equal_pos_neg = equal_pos_neg
        self.fix_mesh = fix_mesh
        self.load_cache = load_cache
        self.save_cache = save_cache
        self.print_filename = print_filename
        self.multiprocessing = multiprocessing
        self.n_processes = n_processes
        self.store_data_in_memory = store_data_in_memory
        self.debug_memory = debug_memory
        self._memory_tracker = None
        self._memory_counter = 0
        self.test_load_times = test_load_times
        self.uniform_pts_buffer = uniform_pts_buffer

        # if store_data_in_memory is False & save_cache is False, then raise error
        if (self.store_data_in_memory is False) and (self.save_cache is False):
            raise ValueError(
                "If store_data_in_memory is False, then save_cache must be True."
                "when data not stored in memory, it is loaded from disk - but data is"
                "not saved to disk when save_cache is False."
            )

        # set defaults so can use same 'norm_and_scale_all_meshes' function
        # for single and multiple meshes. The hasattr guards are an initialization-order
        # contract: a subclass that wants its own values (MultiSurfaceSDFSamples does)
        # must set these attributes BEFORE calling super().__init__.
        if not hasattr(self, "reference_object"):
            self.reference_object = 0
        if not hasattr(self, "n_meshes"):
            self.n_meshes = 1

        self.max_radius = None
        self.center = None

        # preprocess inputs before proceeding
        self.preprocess_inputs()

        self.list_hash_params = self.get_hash_params()

        if save_cache is True:
            self.cache_folder = os.path.join(self.loc_save, today_date)
            os.makedirs(self.cache_folder, exist_ok=True)

        # get the combinations of points and sigmas to sample
        self.pt_sample_combos = self.get_pt_sample_combos()

        # preallocate reference mesh path to None
        self.reference_mesh_path = None

        if self.reference_mesh is not None:
            self.load_reference_mesh()

        # function to allow calling additional internal functions from subclasses.
        self.run_before_loading_data()

        self.data = []
        # Wrap this loading loop in a multiprocessing pool
        if self.verbose is True:
            try:
                print(f"CPU affinity:{os.sched_getaffinity(0)}")
            except AttributeError:
                # sched_getaffinity is not available on all platforms (eg., mac/windows)
                print("CPU affinity not available on this platform")
        if self.multiprocessing is True:
            list_inputs = [(loc_mesh, self.verbose) for loc_mesh in self.list_mesh_paths]
            with Pool(processes=self.n_processes) as pool:
                self.data = pool.starmap(self.load_mesh_step, list_inputs)
        else:
            self.data = [
                self.load_mesh_step(loc_mesh, self.verbose) for loc_mesh in self.list_mesh_paths
            ]

        # remove mesh paths that failed to load
        self.list_mesh_paths = [
            x for idx, x in enumerate(self.list_mesh_paths) if self.data[idx] is not None
        ]
        # remove data that failed to load
        self.data = [x for x in self.data if x is not None]

        if self.scale_jointly is True:
            self.norm_and_scale_all_meshes()

    def print_memory_summary(self):
        """Print a pympler summary diff every 100th call (``debug_memory=True`` only)."""
        if self._memory_tracker is None:
            self._memory_tracker = tracker.SummaryTracker()

        # every 100th iteration, print the memory summary
        if self._memory_counter % 100 == 0:
            self._memory_tracker.print_diff()

            # all_objects = muppy.get_objects()
            # numpy_arrays = [obj for obj in all_objects if isinstance(obj, np.ndarray)]
            # refs = gc.get_referrers(numpy_arrays[0])
            # print('REFERENCES TO NUMPY ARRAY')
            # print(refs)
        # size_info = asizeof.asized(self, detail=1)
        # print(size_info)
        # all_objects = muppy.get_objects()
        # memory_summary = summary.summarize(all_objects)
        # if self._memory_summary is not None:
        # self._memory_summary = memory_summary

        self._memory_counter += 1

    def run_before_loading_data(self):
        """Subclass hook, called after setup but before any subject loads."""
        pass

    def load_mesh_step(self, loc_mesh, verbose):
        """
        Per-subject worker: build or load one subject via ``get_sample_data_dict``.

        Returns its result unchanged -- a sample dict, a cache path, or None for a
        failed subject, which ``__init__`` then drops from ``list_mesh_paths`` and
        ``data``.
        """
        if verbose is True:
            print("Loading mesh:", loc_mesh)

        if self.debug_memory is True:
            self.print_memory_summary()

        if self.multiprocessing is True:
            try:
                os.sched_setaffinity(0, range(multiprocessing.cpu_count()))
            except AttributeError:
                # sched_setaffinity is not available on all platforms (eg., mac/windows).
                # Forking a Pool worker resets CPU affinity on Linux; elsewhere there is
                # nothing to reset, so skipping it is the correct no-op rather than a
                # degraded path. Matches the guard on sched_getaffinity above.
                pass
        data = self.get_sample_data_dict(loc_mesh)

        if data is None:
            print("Skipping mesh:", loc_mesh)
            print("Error in loading")

        if verbose is True:
            print("Data type:", type(data))
            print("Finished loading mesh:", loc_mesh)

        gc.collect()

        return data

    def norm_and_scale_all_meshes(self):
        """
        Center and scale every subject into one shared frame (``scale_jointly=True``).

        The shared center is the across-subject mean of each subject's
        ``reference_object`` surface centroid -- the other surfaces follow the reference,
        they do not pull on it. The shared scale is the largest radius any surface of any
        subject reaches from that center, grown by ``joint_scale_buffer`` so unseen
        subjects slightly larger than the training set still land inside the model's
        domain. One frame for everyone removes per-subject position/size as a source of
        variation.

        Nothing is rescaled here on the disk-backed path: the result is stored as
        ``self.center`` / ``self.max_radius`` and applied per batch in ``__getitem__``,
        so the cached ``.npz`` files stay in the unscaled frame.

        KNOWN DEFECT: the ``store_data_in_memory=True`` branch has never worked. It
        reads the flattened ``new_pts_0``-style keys that exist only in the ``.npz``
        layout -- in-memory dicts hold ``new_pts`` as a list -- so it raises
        ``KeyError``; it also omits ``joint_scale_buffer``. Verified 2026-08-22 on both
        dataset classes.
        """
        print("Computing centering and scaling...")
        # if not stored in memory, then get the centers and max radii from the data in memory
        if self.store_data_in_memory is False:
            print("Data not stored in memory... loading from disk")
            tic = time.time()
            centers = []
            for data in self.data:
                # load in the npz dict
                data_ = np.load(data)
                centers.append(np.mean(data_[f"new_pts_{self.reference_object}"], axis=0))
            # new center:
            center = np.mean(centers, axis=0)

            print("Done computing centers")

            max_radii = []
            # for each data, comput the max radius (from the new/global center)
            for data in self.data:
                data_ = np.load(data)
                max_radius = 0
                for mesh_idx in range(self.n_meshes):
                    xyz = data_[f"new_pts_{mesh_idx}"]
                    centered_xyz = xyz - center
                    radii = np.linalg.norm(centered_xyz, axis=-1)
                    max_radius_ = np.max(radii)
                    if max_radius_ > max_radius:
                        max_radius = max_radius_
                max_radii.append(max_radius)
            max_radius = np.max(max_radii)
            # make the biggest radius a bit bigger than observed to enable model to
            # generalize to unseen data that is slightly larger than the observed data.
            max_radius = max_radius * (1 + self.joint_scale_buffer)
            print("Done computing max radii")

            self.max_radius = max_radius.astype(np.float32)
            self.center = center.astype(np.float32)
            toc = time.time()
            print(f"Finished computing centering and scaling in {toc - tic:.3f}s")
            print(f"\tMax radius: {self.max_radius}")
            print(f"\tCenter: {self.center}")

        else:
            # get the center of all of the meshes
            centers = []
            for data in self.data:
                # center around the reference object
                xyz = data[f"new_pts_{self.reference_object}"]
                center = np.mean(xyz, axis=0)
                centers.append(center)
            centers = np.stack(centers, axis=0)
            center = np.mean(centers, axis=0)

            # subtract the center from all of the meshes
            for idx, data in enumerate(self.data):
                self.data[idx]["xyz"] -= center
                # iterate over all of the meshes and subtract the center
                for mesh_idx in range(self.n_meshes):
                    self.data[idx][f"new_pts_{mesh_idx}"] -= center

            # get the max radius of all of the meshes
            max_radii = 0
            for data in self.data:
                for mesh_idx in range(self.n_meshes):
                    xyz = data[f"new_pts_{mesh_idx}"]
                    max_radius = np.max(np.linalg.norm(xyz, axis=-1))
                    if max_radius > max_radii:
                        max_radii = max_radius

            # divide all of the meshes by the max radius
            for idx, data in enumerate(self.data):
                self.data[idx]["xyz"] /= max_radii
                # do the same for the sdf of each point
                self.data[idx]["gt_sdf"] /= max_radii
                # do the same for the original points
                for mesh_idx in range(self.n_meshes):
                    self.data[idx][f"new_pts_{mesh_idx}"] /= max_radii

    def preprocess_inputs(self):
        """
        Validate/normalize constructor inputs before any data loads. Subclasses extend.

        Raises:
            ValueError: If ``scale_jointly`` is combined with ``center_pts`` or
                ``norm_pts`` -- joint scaling requires untouched per-subject coordinates.
        """

        if self.scale_jointly is True:
            if self.center_pts is True:
                raise ValueError(
                    "Scale jointly assumes centering at end... so center should be False"
                )
            if self.norm_pts is True:
                raise ValueError(
                    "Scale jointly assumes normalizing at end... so norm should be False"
                )

    def get_dict_pts(self, data, pts_name):
        """Flatten ``data[pts_name]`` to ``{pts_name}_{i}`` keys for ``np.savez``."""
        dict_pts = {}
        if isinstance(data[pts_name], list):
            for idx_, orig_pts_ in enumerate(data[pts_name]):
                dict_pts[f"{pts_name}_{idx_}"] = orig_pts_
        else:
            dict_pts[f"{pts_name}_0"] = data[pts_name]
        return dict_pts

    def save_data_to_cache(self, data, file_hash, filepath=None):
        """
        Write one subject's sample dict to a ``.npz`` cache file.

        The on-disk spelling differs from the in-memory one: ``xyz`` is stored as
        ``pts``, ``gt_sdf`` as ``sdfs``, and list-valued entries are flattened to
        indexed keys (``new_pts_0``, ...). ``unpack_numpy_data`` reverses all of it.

        Args:
            data (dict): Dictionary of data to save
            file_hash (str): Cache key; names the file ``{file_hash}.npz``
            filepath (str, optional): Write here instead (used to upgrade an existing
                cache file in place). Defaults to None.
        """
        # if want to cache, and new... then save.
        if filepath is None:
            filepath = os.path.join(self.cache_folder, f"{file_hash}.npz")
        dict_pts = {}
        dict_pts.update(self.get_dict_pts(data, "orig_pts"))
        dict_pts.update(self.get_dict_pts(data, "new_pts"))

        additional_keys = [
            "pos_idx",
            "neg_idx",
            "surf_idx",
            "center",
            "max_radius",
            "max_radius_xyz",
        ]
        for key in additional_keys:
            if key in data:
                dict_pts.update(self.get_dict_pts(data, key))
                # dict_pts[key] = data[key]

        # add pos/negative point indices

        np.savez(filepath, pts=data["xyz"], sdfs=data["gt_sdf"], **dict_pts)

    def get_sample_data_dict(self, loc_mesh):
        """
        Build or load one subject's samples; return them, or the path they are cached at.

        On a cache hit (``load_cache=True``): unreadable ``.npz`` files are deleted and
        the next candidate tried; caches from before the ``pos_idx`` layout are upgraded
        in place (indices computed, file resaved). On a miss: each ``pt_sample_combos``
        entry is sampled via ``read_mesh_get_sampled_pts``, with a per-combo seed
        derived from ``random_seed`` and keyed on the mesh contents.

        Args:
            loc_mesh (str): Path to mesh

        Returns:
            dict, str or None: The sample dict (``store_data_in_memory=True``), the
            path of its cached ``.npz`` (False, the default), or None when the mesh
            failed to load -- ``__init__`` then drops the subject.
        """

        # Create hash and filename
        file_hash = self.create_hash(loc_mesh)
        cached_file = self.find_hash(filename=f"{file_hash}.npz")

        file_loaded = False

        if (len(cached_file) > 0) and (self.load_cache is True):
            for cached_file_ in cached_file:
                if not is_zipfile(cached_file_):
                    print("DELETING BAD ZIP FILE:", cached_file_)
                    os.remove(cached_file_)
                    continue

                # if hashed file exists, load it.
                try:
                    data_ = np.load(cached_file_)
                    data = unpack_numpy_data(data_)
                except zipfile.BadZipFile:
                    print("DELETING BAD ZIP FILE:", cached_file_)
                    os.remove(cached_file_)
                    continue

                if ("pos_idx" not in data) or ("neg_idx" not in data) or ("surf_idx" not in data):
                    pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
                    data["pos_idx"] = pos_idx
                    data["neg_idx"] = neg_idx
                    data["surf_idx"] = surf_idx
                    self.save_data_to_cache(data, file_hash, filepath=cached_file_)

                file_loaded = True
                cache_path = cached_file_
                break

        if file_loaded is False:
            # otherwise, load the mesh and create SDF samples.
            print("Creating SDF Samples")
            if self.print_filename is True:
                print(loc_mesh)
            data = {
                "xyz": torch.zeros((self.n_pts, 3)),
                "gt_sdf": torch.zeros((self.n_pts)),
            }
            pts_idx = 0

            if self.multiprocessing is True:
                if self.reference_mesh_path is not None:
                    reference_mesh = Mesh(self.reference_mesh_path)
                else:
                    reference_mesh = None
            else:
                reference_mesh = self.reference_mesh

            if self.verbose is True:
                print("type of reference mesh:", type(reference_mesh))
                print("ref mesh path:", self.reference_mesh_path)

            # Keyed on the mesh contents, not on the subject's index and not on the cache
            # hash: an index would resample every subject when the list is reordered, and
            # the cache hash contains the mesh path, so it would resample everyone when
            # the data is moved. Read once here rather than per combo.
            content_key = mesh_content_key(loc_mesh) if self.random_seed is not None else None

            for idx_, (n_pts_, sigma_) in enumerate(self.pt_sample_combos):
                # A zero-count combo (p_near_surface=0, p_further_from_surface=0, or the
                # two summing to 1) samples nothing; passing it through would crash in
                # point_cloud_utils on an empty point cloud (#23). The seed key stays
                # idx_, so skipping one combo does not re-seed the others.
                if n_pts_ == 0:
                    continue
                result_ = read_mesh_get_sampled_pts(
                    loc_mesh,
                    sigma=sigma_,
                    n_pts=n_pts_,
                    rand_function=self.rand_function,
                    center_pts=self.center_pts,
                    norm_pts=self.norm_pts,
                    scale_method=self.scale_method,
                    get_random=True,
                    fix_mesh=self.fix_mesh,
                    register_to_mean_first=False if reference_mesh is None else True,
                    mean_mesh=reference_mesh,
                    uniform_pts_buffer=self.uniform_pts_buffer,
                    seed=derive_seed(self.random_seed, content_key, idx_),
                )

                if result_ is None:
                    return None

                xyz_ = result_["pts"] if "pts" in result_ else result_["xyz"]
                sdfs_ = result_["sdf"] if "sdf" in result_ else result_["gt_sdf"]

                data["xyz"][pts_idx : pts_idx + n_pts_, :] = torch.from_numpy(xyz_).float()
                data["gt_sdf"][pts_idx : pts_idx + n_pts_] = torch.from_numpy(sdfs_).float()
                pts_idx += n_pts_

                if "orig_pts" not in data:
                    # First combo that actually ran -- not necessarily combo 0, which a
                    # zero count skips. Convert list of arrays to tensors.
                    data["orig_pts"] = [
                        torch.from_numpy(pts).float() for pts in result_["orig_pts"]
                    ]
                    data["new_pts"] = [torch.from_numpy(pts).float() for pts in result_["new_pts"]]

            pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
            data["pos_idx"] = pos_idx
            data["neg_idx"] = neg_idx
            data["surf_idx"] = surf_idx

            if self.save_cache is True:
                self.save_data_to_cache(data, file_hash)
                cache_path = os.path.join(self.cache_folder, f"{file_hash}.npz")

        if self.store_data_in_memory is False:
            if self.verbose is True:
                print("updating data to be cache path")
            # change the data to be the path to the saved cache file
            data = cache_path

        return data

    def get_pt_sample_combos(self):
        """
        The three sampling passes: near-surface, far-surface, and uniform.

        Counts follow ``p_near_surface`` / ``p_further_from_surface``; whatever the two
        (truncated) shares leave of ``n_pts`` is drawn uniformly from the buffered cube,
        marked by sigma None.

        Returns:
            list: List of [n_pts, sigma] pairs, one per pass
        """

        n_p_near_surface = int(self.n_pts * self.p_near_surface)
        n_p_further_from_surface = int(self.n_pts * self.p_further_from_surface)
        n_p_random = self.n_pts - n_p_near_surface - n_p_further_from_surface

        pt_sample_combos = [
            [n_p_near_surface, self.sigma_near],
            [n_p_further_from_surface, self.sigma_far],
            [n_p_random, None],
        ]

        return pt_sample_combos

    def sdf_pos_neg_idx(self, data):
        """
        Index the samples by SDF sign, padded for equal-share batch draws.

        ``pos_idx`` and ``neg_idx`` are tiled (``repeat``) until each holds at least
        ``subsample / 2`` entries, so a scarce sign is drawn with repetition rather
        than exhausted. ``surf_idx`` (exact zeros) is returned unpadded.

        Args:
            data (dict): Dictionary of sampled points and SDFs

        Returns:
            tuple: (pos_idx, neg_idx, surf_idx) index tensors into ``data["xyz"]``

        Raises:
            ValueError: If every sample has the same sign -- equal batches cannot be
                drawn, and a mesh with no interior or no exterior samples is degenerate
                or unclosed (#41).
        """

        pos_idx = (data["gt_sdf"] > 0).nonzero(as_tuple=True)[0]
        neg_idx = (data["gt_sdf"] < 0).nonzero(as_tuple=True)[0]
        surf_idx = (data["gt_sdf"] == 0).nonzero(as_tuple=True)[0]

        for sign, idx_ in (("positive", pos_idx), ("negative", neg_idx)):
            if idx_.numel() == 0:
                # The repeat below would divide by zero (#41), and a mesh whose samples
                # are all one sign has no interior/exterior to learn from.
                raise ValueError(
                    f"The mesh yielded no {sign} SDF samples, so equal positive/negative "
                    f"batches cannot be drawn from it. Is the mesh degenerate or unclosed?"
                )

        # Repeat +/- indices if either of them does not have enough for a full batch.
        samples_per_sign = int(self.subsample / 2)
        pos_idx = pos_idx.repeat(samples_per_sign // pos_idx.size(0) + 1)
        neg_idx = neg_idx.repeat(samples_per_sign // neg_idx.size(0) + 1)

        return pos_idx, neg_idx, surf_idx

    def find_hash(self, filename="hashed_filename.npz"):
        """
        Search the cache tree for ``filename``, stopping at the first match.

        Walks all of ``loc_save`` -- every date folder, not just today's -- so a cache
        written on an earlier day still hits.

        Args:
            filename (str, optional): Hashed filename. Defaults to 'hashed_filename.npz'.

        Returns:
            list: Zero or one path(s); the first match wins.
        """

        files = []
        for p, d, f in os.walk(self.loc_save):
            for filename_ in f:
                if filename_ == filename:
                    files.append(os.path.join(p, filename_))
                    print("File found in cache:", files[-1])
                    return files

        return files

    def load_reference_mesh(self):
        """
        Resolve ``reference_mesh`` into a loaded ``Mesh`` -- or a path workers reload.

        Accepted forms: a ``Mesh``, used as-is; a path string; an int, indexing
        ``list_mesh_paths`` -- a multi-surface subject resolves to its registration
        surface(s), ``mesh_to_scale``, combined into one mesh when that is a list
        (#61); or a list of paths, indexed by ``reference_object``.

        With ``multiprocessing=True`` the mesh is then written to a timestamped ``.vtk``
        in the cache folder and ``self.reference_mesh`` set back to None: pool workers
        reload it from ``self.reference_mesh_path`` rather than receiving the object
        itself, and the timestamp keeps concurrent runs' reference meshes apart.

        Raises:
            TypeError: If reference mesh is not a string, int, list of strings, or
                mesh.Mesh object
        """

        if self.verbose is True:
            print("Loading reference mesh: ", self.reference_mesh)

        if issubclass(type(self.reference_mesh), Mesh):
            pass
        elif isinstance(self.reference_mesh, int):
            if isinstance(self.list_mesh_paths[0], (str, Mesh)):
                self.reference_mesh = Mesh(self.list_mesh_paths[self.reference_mesh])
            elif isinstance(self.list_mesh_paths[0], (list, tuple)):
                # Multi-surface subject: the reference is the surface(s) that drive
                # registration -- combined into one mesh when mesh_to_scale is a list.
                subject = self.list_mesh_paths[self.reference_mesh]
                if isinstance(self.mesh_to_scale, (list, tuple)):
                    meshes = [Mesh(subject[idx]) for idx in self.mesh_to_scale]
                    self.reference_mesh = combine_meshes(meshes, list(range(len(meshes))))
                else:
                    self.reference_mesh = Mesh(subject[self.mesh_to_scale])
            else:
                raise TypeError("provided list_meshes wrong type")
        elif isinstance(self.reference_mesh, str):
            self.reference_mesh = Mesh(self.reference_mesh)
        elif isinstance(self.reference_mesh, list):
            # below will throw error in SDFSamples, but will work in MultiSurfaceSDFSamples
            # where self.mesh_to_scale is defined & a list/tuple type likely
            # (reference_object vs mesh_to_scale: see the MultiSurfaceSDFSamples docstring)
            self.reference_mesh = Mesh(self.reference_mesh[self.reference_object])
        else:
            raise TypeError(
                "Reference mesh must be a string, list of strings, or mesh.Mesh object, not",
                type(self.reference_mesh),
            )

        if self.verbose is True:
            print("type of reference mesh:", type(self.reference_mesh))

        if self.multiprocessing is True:
            # update reference mesh path to be a has on the current time - so as to not end up with
            # multiple training runs of different tissues using the same reference mesh.
            # this happens because the random seed is set - so all models get the same random number.
            hashed_time = hashlib.md5(str(int(time.time())).encode()).hexdigest()
            self.reference_mesh_path = os.path.join(
                self.cache_folder, f"REFERENCE_MESH_{hashed_time}.vtk"
            )
            self.reference_mesh.save_mesh(self.reference_mesh_path)
            self.reference_mesh = None

    def get_hash_params(self):
        """
        Get the parameters to hash for saving/loading the cache.

        KNOWN DEFECTS, #19 (a) and (c): this list is incomplete, and two runs differing
        only in an omitted parameter share a cache key -- so with load_cache=True the
        second silently trains on the first's data. Missing here: `subsample` and
        `uniform_pts_buffer`. Also, a `reference_mesh` passed as a Mesh object is hashed
        via str(), which contains its memory address, so that key is per-object.

        Returns:
            list: List of parameters to hash
        """

        list_hash_params = [
            self.n_pts,
            self.p_near_surface,
            self.sigma_near,
            self.p_further_from_surface,
            self.sigma_far,
            self.center_pts,
            self.norm_pts,
            self.scale_method,
            self.rand_function,
            self.reference_mesh,
            self.fix_mesh,
            self.scale_jointly,
        ]

        return list_hash_params

    def create_hash(self, loc_mesh):
        """
        The cache key for one subject: md5 over its mesh path(s) plus the hash params.

        The path(s) are prepended to ``list_hash_params`` (a list lands in reverse
        order) and ``random_seed`` is appended when set; everything is stringified,
        joined and hashed. The mesh *contents* are not part of the key, so overwriting
        a mesh file in place reuses the stale cache.

        Args:
            loc_mesh (str or list): Path(s) to the subject's mesh(es)

        Returns:
            str: Hashed string
        """

        list_hash_params = self.list_hash_params.copy()
        if isinstance(loc_mesh, str):
            list_hash_params.insert(0, loc_mesh)
        elif isinstance(loc_mesh, (list, tuple)):
            for path in loc_mesh:
                if self.verbose is True:
                    print(loc_mesh)
                list_hash_params.insert(0, path)

        if self.random_seed is not None:
            list_hash_params.append(self.random_seed)  # random seed state
        if self.verbose is True:
            print("List Params", list_hash_params)
        list_hash_params = [str(x) for x in list_hash_params]
        file_params_string = "_".join(list_hash_params)
        hash_str = hashlib.md5(file_params_string.encode()).hexdigest()
        return hash_str

    def __len__(self):
        """Number of subjects that loaded successfully (failures are dropped)."""

        return len(self.data)

    def __getitem__(self, idx):
        """
        One training batch for subject ``idx``.

        Disk-backed datasets (``store_data_in_memory=False``) reload the subject's
        ``.npz`` on every call. With ``equal_pos_neg``, ``subsample / 2`` rows are drawn
        from each sign's index list, topped up with unconstrained draws when the halves
        round short. Under joint scaling the shared center/scale is applied here, to the
        batch, not to the cache.

        Args:
            idx (int): Subject index

        Returns:
            tuple: ``(batch, idx)``. ``batch["xyz"]`` is (subsample, 3) and
            ``batch["gt_sdf"]`` (subsample,), float32. The timing keys (``time``,
            ``size``, ``mb_per_sec``, ``whole_load_time``) appear only when
            ``test_load_times=True`` and the item came from disk (#22).
        """

        tic_whole_load = time.time()

        if self.store_data_in_memory is False:
            # if not storing in memory, then load from cache
            tic = time.time()
            data_ = np.load(self.data[idx])
            toc = time.time()
            time_ = toc - tic

            # get size of the numpy file in mb
            size = os.path.getsize(self.data[idx]) / 1e6

            if self.equal_pos_neg is True:
                list_keys_unpack = ["pos_idx", "neg_idx"]
            else:
                list_keys_unpack = []
            data_ = unpack_numpy_data(data_, list_additional_keys=list_keys_unpack)
        elif self.store_data_in_memory is True:
            # if storing in memory, then just get the data
            data_ = self.data[idx]
        else:
            raise ValueError("store_data_in_memory must be True or False")

        if self.subsample is not None:
            if self.equal_pos_neg is True:
                tic_rand_sample = time.time()
                samples_per_sign = int(self.subsample / 2)

                # idx_pos = data_['pos_idx'].repeat(data_['pos_idx'].size(0)//samples_per_sign + 1)
                # perm_pos = torch.randperm(idx_pos.size(0))
                if isinstance(data_["pos_idx"], list):
                    perm_pos = torch.randperm(data_["pos_idx"][0].size(0))[:samples_per_sign]
                    idx_pos = data_["pos_idx"][0][perm_pos]
                elif isinstance(data_["pos_idx"], torch.Tensor):
                    perm_pos = torch.randperm(data_["pos_idx"].size(0))[:samples_per_sign]
                    idx_pos = data_["pos_idx"][perm_pos]
                else:
                    raise ValueError("pos_idx must be a list or tensor")

                # idx_neg = data_['neg_idx'].repeat(data_['neg_idx'].size(0)//samples_per_sign + 1)
                # perm_neg = torch.randperm(idx_neg.size(0))
                # idx_neg = perm_neg[:samples_per_sign]
                if isinstance(data_["neg_idx"], list):
                    perm_neg = torch.randperm(data_["neg_idx"][0].size(0))[:samples_per_sign]
                    idx_neg = data_["neg_idx"][0][perm_neg]
                elif isinstance(data_["neg_idx"], torch.Tensor):
                    perm_neg = torch.randperm(data_["neg_idx"].size(0))[:samples_per_sign]
                    idx_neg = data_["neg_idx"][perm_neg]
                else:
                    raise ValueError("neg_idx must be a list or tensor")
                toc_rand_sample = time.time()
                if self.verbose is True:
                    print(f"rand sample time: {toc_rand_sample - tic_rand_sample}s")

                tic_cat = time.time()
                idx_ = torch.cat((idx_pos, idx_neg), dim=0)
                toc_cat = time.time()
                if self.verbose is True:
                    print(f"concat time: {toc_cat - tic_cat}s")

                if len(idx_) < self.subsample:
                    # if we don't have enough points, then just take random points
                    tic_rand = time.time()
                    perm = torch.randperm(data_["xyz"].size(0))
                    _idx_ = perm[: self.subsample - len(idx_)]
                    idx_ = torch.cat([idx_, _idx_], dim=0)
                    toc_rand = time.time()
                    if self.verbose is True:
                        print(f"rand additional sub sample time: {toc_rand - tic_rand}s")

            else:
                perm = torch.randperm(data_["xyz"].size(0))
                idx_ = perm[: self.subsample]

            if self.verbose is True:
                print("idx_ size:", idx_.size(), "idx_ min:", idx_.min(), "idx_ max:", idx_.max())
                print("equal neg pos", self.equal_pos_neg)

            # unpack the data
            xyz = data_["xyz"][idx_, :]
            sdf = data_["gt_sdf"][idx_]

            if (self.max_radius is not None) and (self.center is not None):
                # if normalizing at the group level, then normalize here.
                tic_norm = time.time()
                xyz = (xyz - self.center) / self.max_radius
                sdf = sdf / self.max_radius
                toc_norm = time.time()
                if self.verbose is True:
                    print(f"norm time: {toc_norm - tic_norm}s")

            data_ = {
                "xyz": xyz,
                "gt_sdf": sdf,
            }

            toc_whole_load = time.time()
            time_whole_load = toc_whole_load - tic_whole_load

            if (self.test_load_times is True) and (self.store_data_in_memory is False):
                data_["time"] = time_
                data_["size"] = size
                data_["mb_per_sec"] = size / time_
                data_["whole_load_time"] = time_whole_load

        return data_, idx


class MultiSurfaceSDFSamples(SDFSamples):
    """
    Dataset class for sampling SDFs from multiple mesh surfaces with support for
    multi-surface rigid registration.

    Extends SDFSamples to handle multiple anatomical surfaces simultaneously,
    such as bone + cartilage or medial + lateral menisci.

    Args:
        list_mesh_paths (list): One entry per subject, each a list of per-surface mesh
            paths in a fixed surface order, e.g. ``[[bone, cart], ...]``. A None entry
            marks a subject's missing surface -- accepted here, but the build path for
            it has never worked end to end (#67).

        mesh_to_scale (int or list): Index(es) of mesh(es) to use for registration and scaling.
            - If int: Uses single mesh for registration (original behavior)
            - If list: Combines multiple meshes for joint registration
            Example: mesh_to_scale=[0, 1] for medial + lateral menisci registration

        reference_object (int): Index of the surface whose sampled points anchor
            centering, and which element of a list-valued reference_mesh is used.
            A separate knob from mesh_to_scale (which surface(s) drive registration
            and scaling); the two are not kept in sync, and why they are separate is
            an open question inherited from the original implementation.

        scale_all_meshes (bool): Scale using every surface's points (True, default) or
            only ``mesh_to_scale``'s (False). See read_meshes_get_sampled_pts.
        center_all_meshes (bool): Center on every surface's points (True) or only
            ``mesh_to_scale``'s (False, default).

        n_pts (int or list): Per-surface sample counts; a scalar or one-element list is
            broadcast to every surface. The per-surface floats (p_near_surface,
            p_further_from_surface, sigma_near, sigma_far) broadcast the same way.

        Other args: Same as SDFSamples parent class

    Notes:
        - When mesh_to_scale is a list, meshes are combined with the pymskt Mesh `+`
          operator (see combine_meshes)
        - Joint registration improves alignment when multiple related surfaces should
          be considered together rather than individually
        - ``__getitem__`` returns ``batch["gt_sdf"]`` with shape (subsample,
          n_surfaces): every surface's signed distance to every sampled point, not just
          the surface the point was drawn around.
    """

    def __init__(
        self,
        list_mesh_paths,
        subsample,
        n_pts=500000,
        p_near_surface=0.4,
        p_further_from_surface=0.4,
        sigma_near=0.01,
        sigma_far=0.1,
        rand_function="normal",
        center_pts=True,
        norm_pts=False,
        scale_method="max_rad",
        scale_jointly=False,
        joint_scale_buffer=0.1,
        loc_save=None,
        save_cache=True,
        load_cache=True,
        random_seed=None,
        reference_mesh=None,
        verbose=False,
        equal_pos_neg=True,
        fix_mesh=True,
        print_filename=False,
        test_load_times=True,
        uniform_pts_buffer=0.0,
        # Multi surface specific
        scale_all_meshes=True,
        center_all_meshes=False,
        mesh_to_scale=0,
        reference_object=0,
        store_data_in_memory=False,
        multiprocessing=True,
        debug_memory=False,
        n_processes=2,
    ):
        # if n_pts is not a list, then make it a list that is
        # the same length as the number of meshes.
        if not isinstance(n_pts, (list, tuple)):
            n_pts = [n_pts] * len(list_mesh_paths[0])
        if len(n_pts) == 1 and len(list_mesh_paths[0]) > 1:
            n_pts = n_pts * len(list_mesh_paths[0])

        self.times = []
        self.data_size = []
        self.mb_per_sec = []
        self.test_load_times = test_load_times
        # Multi surface specific
        self.mesh_to_scale = mesh_to_scale
        self.total_n_pts = sum(n_pts)
        self.scale_all_meshes = scale_all_meshes
        self.center_all_meshes = center_all_meshes
        self.n_meshes = len(list_mesh_paths[0])
        self.reference_object = reference_object

        super().__init__(
            list_mesh_paths=list_mesh_paths,
            subsample=subsample,
            n_pts=n_pts,
            p_near_surface=p_near_surface,
            p_further_from_surface=p_further_from_surface,
            sigma_near=sigma_near,
            sigma_far=sigma_far,
            rand_function=rand_function,
            center_pts=center_pts,
            norm_pts=norm_pts,
            scale_method=scale_method,
            scale_jointly=scale_jointly,
            joint_scale_buffer=joint_scale_buffer,
            loc_save=loc_save,
            save_cache=save_cache,
            load_cache=load_cache,
            random_seed=random_seed,
            reference_mesh=reference_mesh,
            verbose=verbose,
            equal_pos_neg=equal_pos_neg,
            fix_mesh=fix_mesh,
            print_filename=print_filename,
            store_data_in_memory=store_data_in_memory,
            multiprocessing=multiprocessing,
            n_processes=n_processes,
            debug_memory=debug_memory,
            test_load_times=test_load_times,
            uniform_pts_buffer=uniform_pts_buffer,
        )

    def preprocess_inputs(self):
        """As the parent's, plus: count the surfaces and broadcast scalar per-surface
        parameters (p_near_surface, sigma_near, ...) into per-surface lists."""
        super().preprocess_inputs()

        if isinstance(self.list_mesh_paths[0], (list, tuple)):
            self.n_meshes = len(self.list_mesh_paths[0])
        elif isinstance(self.list_mesh_paths[0], (str, Mesh)):
            self.n_meshes = len(self.list_mesh_paths)

        if not isinstance(self.p_near_surface, (list, int)):
            self.p_near_surface = [self.p_near_surface] * self.n_meshes
        if not isinstance(self.p_further_from_surface, (list, int)):
            self.p_further_from_surface = [self.p_further_from_surface] * self.n_meshes
        if not isinstance(self.sigma_near, (list, int)):
            self.sigma_near = [self.sigma_near] * self.n_meshes
        if not isinstance(self.sigma_far, (list, int)):
            self.sigma_far = [self.sigma_far] * self.n_meshes
        if not isinstance(self.n_pts, (list, int)):
            self.n_pts = [self.n_pts] * self.n_meshes

    def run_before_loading_data(self):
        """Precompute each surface's per-sign batch share before any subject loads."""
        self.get_samples_per_sign()

    def test_if_idx_in_range(self, data):
        """
        Whether every pos/neg index actually points into ``data["xyz"]``.

        Guards against stale caches: ``remove_overlapping_points`` shrinks the point
        set, so index lists computed before an overlap pass can exceed it. A False
        return makes ``get_sample_data_dict`` delete the cache file and rebuild.
        """
        n_pts = data["xyz"].shape[0]

        for name in ["pos_idx", "neg_idx"]:
            indices = data[name]
            max_idx = 0
            for tensor in indices:
                if tensor.numel() == 0:
                    # A missing (None) surface has empty index lists; torch.max
                    # raises on an empty tensor, and empty is trivially in range.
                    continue
                max_idx = torch.max(tensor)
                if max_idx >= n_pts:
                    return False

        return True

    def get_sample_data_dict(self, loc_meshes):
        """
        Build or load one subject's samples; return them, or the path they are cached at.

        The multi-surface differences from ``SDFSamples.get_sample_data_dict``:
        ``gt_sdf`` is built (sum(n_pts), n_surfaces), with a missing (None) surface's
        column all-NaN; ``remove_overlapping_points`` runs on fresh builds *and* on
        cache hits, so pre-overlap-pass caches are upgraded and resaved; cached index
        lists that fail ``test_if_idx_in_range`` delete the cache and force a rebuild;
        and each subject started is appended to ``list_meshes_started_loading.log`` in
        ``loc_save``, so a crash mid-build names its subject.

        Args:
            loc_meshes (list or str): The subject's per-surface mesh path(s).

        Returns:
            dict, str or None: As the parent -- sample dict, cache path, or None for a
            failed subject.
        """
        if type(loc_meshes) not in (tuple, list):
            loc_meshes = [loc_meshes]

        with open(os.path.join(self.loc_save, "list_meshes_started_loading.log"), "a") as f:
            f.write(str(loc_meshes) + "\n")

        # get the number of points to sample per mesh / sign(in/out or pos/neg)
        self.get_samples_per_sign()

        # Create hash and filename
        file_hash = self.create_hash(loc_meshes)
        cached_file = self.find_hash(filename=f"{file_hash}.npz")

        file_loaded = False

        if (len(cached_file) > 0) and (self.load_cache is True):
            if self.verbose is True:
                print("Loading cached file")
            for cache_path in cached_file:
                if not is_zipfile(cache_path):
                    print("DELETEING BAD ZIP FILE:", cache_path)
                    os.remove(cache_path)
                    continue

                try:
                    data = np.load(cache_path)
                    data = unpack_numpy_data(data)
                except zipfile.BadZipFile:
                    print("DELETEING BAD ZIP FILE:", cache_path)
                    os.remove(cache_path)
                    continue

                # if previous pre-processing not yet done, do it now
                # and update/resave the data to cache.
                resave_data = False

                data, in_in = self.remove_overlapping_points(data)

                if in_in > 0:
                    resave_data = True

                if (
                    ("pos_idx" not in data)
                    or (len(data["pos_idx"]) != self.n_meshes)
                    or ("neg_idx" not in data)
                    or (len(data["neg_idx"]) != self.n_meshes)
                    or ("surf_idx" not in data)
                    or (len(data["surf_idx"]) != self.n_meshes)
                ):
                    print("getting pos/neg")
                    pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
                    data["pos_idx"] = pos_idx
                    data["neg_idx"] = neg_idx
                    data["surf_idx"] = surf_idx

                    resave_data = True

                if self.test_if_idx_in_range(data) is False:
                    print("Indices out of range!", cache_path)
                    print("\tDeleting file...")
                    os.remove(cache_path)
                    break

                if resave_data is True:
                    self.save_data_to_cache(
                        data, file_hash, filepath=cache_path
                    )  # resave data to cache - overwriting original.

                file_loaded = True
                break

        if file_loaded is False:
            # otherwise, load the mesh and create SDF samples.
            print("Creating SDF Samples")
            if self.print_filename is True:
                print(loc_meshes)

            data = {
                "xyz": torch.zeros((sum(self.n_pts), 3)),
                "gt_sdf": torch.zeros((sum(self.n_pts), len(loc_meshes))),
            }
            pts_idx = 0
            icp_transform = None

            if self.multiprocessing is True:
                if self.reference_mesh_path is not None:
                    reference_mesh = Mesh(self.reference_mesh_path)
                else:
                    reference_mesh = None
            else:
                reference_mesh = self.reference_mesh

            if self.verbose is True:
                print("type of reference mesh:", type(reference_mesh))
                print("ref mesh path:", self.reference_mesh_path)

            content_key = mesh_content_key(loc_meshes) if self.random_seed is not None else None

            for idx_, (n_pts_, sigma_) in enumerate(self.pt_sample_combos):
                # A combo asked to sample nothing anywhere would crash in
                # point_cloud_utils on an empty point cloud (#23). The seed key stays
                # idx_, so skipping one combo does not re-seed the others.
                if sum(n_pts_) == 0:
                    continue
                tic = time.time()
                result_ = read_meshes_get_sampled_pts(
                    loc_meshes,
                    sigma=sigma_,
                    n_pts=n_pts_,
                    rand_function=self.rand_function,
                    center_pts=self.center_pts,
                    norm_pts=self.norm_pts,
                    scale_method=self.scale_method,
                    get_random=True,
                    fix_mesh=self.fix_mesh,
                    register_to_mean_first=False if reference_mesh is None else True,  #
                    mean_mesh=reference_mesh,  #
                    uniform_pts_buffer=self.uniform_pts_buffer,
                    # Multi surface specific
                    mesh_to_scale=self.mesh_to_scale,
                    scale_all_meshes=self.scale_all_meshes,
                    center_all_meshes=self.center_all_meshes,
                    icp_transform=icp_transform,
                    seed=derive_seed(self.random_seed, content_key, idx_),
                )

                if result_ is None:
                    return None

                icp_transform = result_["icp_transform"]

                toc = time.time()
                print(f"{idx_} - {sigma_}: {toc - tic}s")

                if "orig_pts" not in data:
                    # First combo that actually ran -- not necessarily combo 0,
                    # which a zero count skips.
                    data["orig_pts"] = result_["orig_pts"]
                    data["new_pts"] = result_["new_pts"]

                xyz_ = result_["pts"] if "pts" in result_ else result_["xyz"]
                sdfs_ = result_["sdf"] if "sdf" in result_ else result_["gt_sdf"]

                data["xyz"][pts_idx : pts_idx + sum(n_pts_), :] = torch.from_numpy(xyz_).float()

                for mesh_idx, _sdfs_ in enumerate(sdfs_):
                    if _sdfs_ is None:
                        # If mesh was None, fill with NaN to indicate missing data
                        # don't need this now.. but can handle training on incomplete data in the future.
                        data["gt_sdf"][pts_idx : pts_idx + sum(n_pts_), mesh_idx] = float("nan")
                    else:
                        data["gt_sdf"][pts_idx : pts_idx + sum(n_pts_), mesh_idx] = (
                            torch.from_numpy(_sdfs_).float()
                        )
                pts_idx += sum(n_pts_)

            # Drop points that have are labeled as being inside
            # 2 objects - clearly this is an error.
            data, in_in = self.remove_overlapping_points(data)

            print("getting pos/neg")
            pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
            data["pos_idx"] = pos_idx
            data["neg_idx"] = neg_idx
            data["surf_idx"] = surf_idx

            if (data is not None) and (self.save_cache is True):
                self.save_data_to_cache(data, file_hash)
                cache_path = os.path.join(self.cache_folder, f"{file_hash}.npz")

        if self.store_data_in_memory is False:
            if self.verbose is True:
                print("updating data to be cache path")
            # change the data to be the path to the saved cache file
            data = cache_path

        return data

    def get_samples_per_sign(self):
        """
        Each surface's per-sign share of a batch: ``subsample`` split across surfaces
        in proportion to their ``n_pts``, halved per sign, truncated. Truncation means
        the shares can sum below ``subsample``; ``__getitem__`` tops the batch up with
        unconstrained draws. Stored as ``self.samples_per_sign_``.
        """
        samples_per_mesh = [
            int((n_pts_ / self.total_n_pts) * self.subsample) for n_pts_ in self.n_pts
        ]

        # setup samples per sign
        self.samples_per_sign_ = []
        for subsample_ in samples_per_mesh:
            samples_per_sign = int(subsample_ / 2)
            if self.verbose is True:
                print(samples_per_sign)
            self.samples_per_sign_.append(samples_per_sign)

    def remove_overlapping_points(self, data):
        """
        Drop points labeled inside two or more surfaces -- anatomically impossible,
        so such a point is a segmentation/meshing error and would teach the model a
        false interior.

        All-NaN columns (missing surfaces) are excluded from the count. Removal
        shrinks ``xyz``/``gt_sdf``, which is why index lists must be recomputed after
        this runs (see ``test_if_idx_in_range``).

        Args:
            data (dict): Sample dict with ``gt_sdf`` of shape (n, n_surfaces)

        Returns:
            tuple: ``(data, n_removed)``; a nonzero count on a cache hit triggers a
            resave.
        """
        sdf_ = data["gt_sdf"].clone()

        # Check if we have None values (represented as NaN)
        non_none_mask = ~torch.isnan(sdf_).all(dim=0)

        if non_none_mask.sum() < 2:
            return data, 0  # Can't have overlaps with fewer than 2 surfaces

        # Only process non-None columns for overlap detection
        sdf_filtered = sdf_[:, non_none_mask]

        # "Overlapping" means inside two or more surfaces, where inside is a strictly
        # negative SDF. Count per point — a sign-sum test is equivalent to this count
        # only at exactly two surfaces.
        inside_count = torch.sum(sdf_filtered < 0, dim=1)

        out_all = torch.sum(inside_count == 0)
        in_one = torch.sum(inside_count == 1)
        in_in = torch.sum(inside_count >= 2)

        # Create mask for points to keep (not overlapping)
        keep_mask = inside_count < 2

        # Apply the mask to remove overlapping points from the full dataset
        # This preserves the None columns while removing problematic points
        data["gt_sdf"] = data["gt_sdf"][keep_mask, :]
        data["xyz"] = data["xyz"][keep_mask, :]

        if self.verbose is True:
            print("inside_count shape", inside_count.shape)
            print("inside_count", inside_count)
            print("outside all surfaces", out_all)
            print("inside exactly one", in_one)
            print("inside two or more", in_in)
            print(f"Removed {in_in} overlapping points")

        return data, in_in

    def get_pt_sample_combos(self):
        """
        As the parent's, with per-surface counts: each pass pairs a count list and a
        sigma list, and the uniform pass carries one None per surface.

        Returns:
            list: List of [n_pts_list, sigma_list] pairs, one per pass
        """
        n_p_near_surface = [
            int(n_pts_ * p_near) for n_pts_, p_near in zip(self.n_pts, self.p_near_surface)
        ]
        n_p_further_from_surface = [
            int(n_pts_ * p_far) for n_pts_, p_far in zip(self.n_pts, self.p_further_from_surface)
        ]
        n_p_random = [
            n_pts_ - n_p_near - n_p_far
            for n_pts_, n_p_near, n_p_far in zip(
                self.n_pts, n_p_near_surface, n_p_further_from_surface
            )
        ]

        pt_sample_combos = [
            [n_p_near_surface, self.sigma_near],
            [n_p_further_from_surface, self.sigma_far],
            [
                n_p_random,
                [
                    None,
                ]
                * self.n_meshes,
            ],
        ]

        return pt_sample_combos

    def get_hash_params(self):
        # KNOWN DEFECTS, #19 (a) and (c): incomplete. `mesh_to_scale`,
        # `uniform_pts_buffer` and `subsample` all change what is written to the cache and
        # none are here, so runs differing only in one of them collide on the same key.
        # `mesh_to_scale` is the worst -- it decides which surface drives centering and
        # normalization, so the two runs are in different coordinate frames entirely.
        # A `reference_mesh` given as a Mesh object also hashes by memory address.
        list_hash_params = [
            self.center_pts,
            self.norm_pts,
            self.scale_method,
            self.rand_function,
            self.scale_all_meshes,
            self.center_all_meshes,
            self.reference_mesh,
            self.reference_object,
            # Unexplained literal: present since this list was written, origin unknown
            # (git shows no removed parameter it could stand in for). Its fate is #19's,
            # which rewrites this list; removing it now would invalidate every cache key.
            False,
            self.fix_mesh,
            self.scale_jointly,
        ]

        for n_pts_ in self.n_pts:
            list_hash_params.append(n_pts_)
        for p_near in self.p_near_surface:
            list_hash_params.append(p_near)
        for p_far in self.p_further_from_surface:
            list_hash_params.append(p_far)
        for sigma_near in self.sigma_near:
            list_hash_params.append(sigma_near)
        for sigma_far in self.sigma_far:
            list_hash_params.append(sigma_far)

        return list_hash_params

    def sdf_pos_neg_idx(self, data):
        """
        Per-surface sign indices, padded for each surface's batch share.

        As the parent's, per surface: each surface's ``pos_idx``/``neg_idx`` are tiled
        up to its ``samples_per_sign_`` share. A surface nothing is drawn from -- a
        zero share, or the all-NaN column of a missing surface -- keeps empty index
        lists instead of raising.

        Args:
            data (dict): Sample dict with ``gt_sdf`` of shape (n, n_surfaces)

        Returns:
            tuple: (pos_idx, neg_idx, surf_idx) -- each a list with one index tensor
            per surface, indexing into ``data["xyz"]``

        Raises:
            ValueError: If a surface that *is* drawn from has every sample on one side
                (#41) -- e.g. a surface nested inside another loses every interior
                point to remove_overlapping_points.
        """

        pos_idx = []
        neg_idx = []
        surf_idx = []
        if self.verbose is True:
            print("data", data["xyz"].shape, data["gt_sdf"].shape)

        for mesh_idx in range(self.n_meshes):

            samples_per_sign = self.samples_per_sign_[mesh_idx]

            mesh_sdfs = data["gt_sdf"][:, mesh_idx].clone()
            pos_idx_ = (mesh_sdfs > 0).nonzero(as_tuple=True)[0]
            neg_idx_ = (mesh_sdfs < 0).nonzero(as_tuple=True)[0]
            surf_idx_ = (mesh_sdfs == 0).nonzero(as_tuple=True)[0]

            # A surface nothing is drawn from may be empty: an all-NaN column is a
            # missing (None) surface, and a zero subsample share means __getitem__ never
            # samples it. Its empty index lists are handled -- randperm(0) draws nothing.
            surface_is_drawn_from = samples_per_sign > 0 and not torch.isnan(mesh_sdfs).all()

            if surface_is_drawn_from:
                for sign, idx_ in (("positive", pos_idx_), ("negative", neg_idx_)):
                    if idx_.numel() == 0:
                        # The repeat below would divide by zero (#41), and a surface
                        # with no interior samples trains to garbage.
                        raise ValueError(
                            f"Surface {mesh_idx} has no {sign} SDF samples, so its "
                            f"equal positive/negative batch share cannot be drawn. A "
                            f"surface nested inside another loses every interior point "
                            f"to remove_overlapping_points."
                        )
                # Repeat +/- indices if either does not have enough for a full batch.
                pos_idx_ = pos_idx_.repeat(samples_per_sign // pos_idx_.size(0) + 1)
                neg_idx_ = neg_idx_.repeat(samples_per_sign // neg_idx_.size(0) + 1)

            pos_idx.append(pos_idx_)
            neg_idx.append(neg_idx_)
            surf_idx.append(surf_idx_)

        return pos_idx, neg_idx, surf_idx

    def __getitem__(self, idx):
        """
        One training batch for subject ``idx``.

        As the parent's, with the equal-sign draw done per surface: each surface
        contributes its ``samples_per_sign_`` share from each of its own index lists
        (a zero-share surface contributes nothing), and the batch is topped up with
        unconstrained draws when the truncated shares sum below ``subsample``.

        Args:
            idx (int): Subject index

        Returns:
            tuple: ``(batch, idx)``. ``batch["xyz"]`` is (subsample, 3) and
            ``batch["gt_sdf"]`` (subsample, n_surfaces), float32. Timing keys as in
            the parent (#22).
        """
        tic_whole_load = time.time()
        if self.store_data_in_memory is False:
            # if not storing in memory, then load from cache

            # if self.test_load_times is True:
            tic = time.time()
            data_ = np.load(self.data[idx])
            toc = time.time()
            time_ = toc - tic
            # self.times.append(time_)

            # get size of the numpy file in mb
            size = os.path.getsize(self.data[idx]) / 1e6
            # self.sizes.append(size)

            # self.mb_per_sec.append(size / time_)

            if self.verbose is True:
                print(f"size: {size}mb, time: {time_}s, mb/s: {size / time_}mb/s")

            if self.equal_pos_neg is True:
                list_keys_unpack = ["pos_idx", "neg_idx"]
            else:
                list_keys_unpack = []
            tic_unpack = time.time()
            data_ = unpack_numpy_data(data_, list_additional_keys=list_keys_unpack)
            toc_unpack = time.time()
            if self.verbose is True:
                print(f"unpack time: {toc_unpack - tic_unpack}s")

        elif self.store_data_in_memory is True:
            # if storing in memory, then just get the data
            data_ = self.data[idx]
        else:
            raise ValueError("store_data_in_memory must be True or False")

        if self.subsample is not None:
            if self.equal_pos_neg is True:
                # get number of points for each mesh
                # this is weighted by the number of points in the mesh
                # relative to the total number of points in the dataset
                # samples_per_mesh = [int((n_pts_/self.total_n_pts) * self.subsample) for n_pts_ in self.n_pts]
                idx_ = []
                for mesh_idx, samples_per_sign in enumerate(self.samples_per_sign_):
                    tic_mesh = time.time()
                    # get number of positive and negative points for this mesh
                    # samples_per_sign = int(subsample_/2)
                    if self.verbose is True:
                        print("samples_per_sign", samples_per_sign)
                        print("mesh idx", mesh_idx)
                        print("data_ pos", data_["pos_idx"])

                    if samples_per_sign == 0:
                        continue

                    # get random indices for positive and negative points
                    # idx_pos = data_['pos_idx'][mesh_idx].repeat(data_['pos_idx'][mesh_idx].size(0)//samples_per_sign + 1)
                    perm_pos = torch.randperm(data_["pos_idx"][mesh_idx].size(0))[:samples_per_sign]
                    idx_pos = data_["pos_idx"][mesh_idx][perm_pos]

                    # idx_neg = data_['neg_idx'][mesh_idx].repeat(data_['neg_idx'][mesh_idx].size(0)//samples_per_sign + 1)
                    perm_neg = torch.randperm(data_["neg_idx"][mesh_idx].size(0))[:samples_per_sign]
                    idx_neg = data_["neg_idx"][mesh_idx][perm_neg]

                    # combine positive and negative indices
                    idx_ += [idx_pos, idx_neg]
                    toc_mesh = time.time()
                    if self.verbose is True:
                        print(f"mesh {mesh_idx} time: {toc_mesh - tic_mesh}s")

                tic_cat = time.time()
                # combine indices for all meshes
                idx_ = torch.cat(idx_, dim=0)
                toc_cat = time.time()
                if self.verbose is True:
                    print(f"cat time: {toc_cat - tic_cat}s")

                if len(idx_) < self.subsample:
                    # if we don't have enough points, then just take random points
                    tic_rand = time.time()
                    perm = torch.randperm(data_["xyz"].size(0))
                    _idx_ = perm[: self.subsample - len(idx_)]
                    idx_ = torch.cat([idx_, _idx_], dim=0)
                    toc_rand = time.time()
                    if self.verbose is True:
                        print(f"rand additional sub sample time: {toc_rand - tic_rand}s")

            else:
                perm = torch.randperm(data_["xyz"].size(0))
                idx_ = perm[: self.subsample]

            if self.verbose is True:
                print("idx_ size:", idx_.size(), "idx_ min:", idx_.min(), "idx_ max:", idx_.max())
                print("equal neg pos", self.equal_pos_neg)

            xyz = data_["xyz"][idx_, :]
            sdf = data_["gt_sdf"][idx_, :]

            if (self.max_radius is not None) and (self.center is not None):
                tic_scaling = time.time()
                xyz = (xyz - self.center) / self.max_radius
                sdf = sdf / self.max_radius
                toc_scaling = time.time()
                if self.verbose is True:
                    print(f"scaling time: {toc_scaling - tic_scaling}s")

            data_ = {
                "xyz": xyz,
                "gt_sdf": sdf,
            }

            toc_whole_load = time.time()

            # Same guard as SDFSamples.__getitem__: in-memory items have no disk load to
            # time, so the timing keys are only emitted when one was measured (#22).
            if (self.test_load_times is True) and (self.store_data_in_memory is False):
                data_["time"] = time_
                data_["size"] = size
                data_["mb_per_sec"] = size / time_
                data_["whole_load_time"] = toc_whole_load - tic_whole_load

        return data_, idx


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
