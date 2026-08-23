"""
The two subject-level reader pipelines: read one subject's mesh(es), optionally
register and normalize them, and sample SDF points.

Moved verbatim out of ``sdf_dataset.py`` (plan §8.0, slice A); the multi reader's
internals were then split into the private helpers below (slice B, §8.0.B):
registration, shared-frame computation, per-surface draws. Both readers remain
importable from ``NSM.datasets`` and ``NSM.datasets.sdf_dataset`` as before; this
module holds the definitions. It imports only from ``.utils``.
"""

import os
import time

import numpy as np
from pymskt.mesh import Mesh

from .utils import (
    combine_meshes,
    derive_seed,
    get_buffered_cube_mins_maxs,
    get_pts_center_and_scale,
    get_rand_uniform_pts,
    meshfix,
)


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
            points, so ``"pts"`` holds ``n_pts`` + n_vertices rows. Defaults to False.
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
        - With ``get_random=True``: sample coordinates under ``"pts"`` (n, 3), signed
          distances under ``"sdf"`` (n,), and ``"pts_surface"`` (n,) of zeros. ``"xyz"``
          is a legacy alias of the same array (#15); do not write new readers of it.
        - With ``get_random=False``: the surface vertices under ``"pts"``, with
          ``"sdf"`` all zeros. No ``"xyz"`` alias -- this path never had the key.

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

        results["pts"] = rand_pts
        # Legacy alias, same array (#15). Transitional: delete when the Phase-0b fork
        # survey confirms no external ["xyz"] readers, or at v0.3.0, whichever first.
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


def _register_to_mean(
    orig_meshes, new_meshes, new_pts, paths, mesh_to_scale, mean_mesh, icp_transform
):
    """
    Similarity-register one subject to ``mean_mesh`` (rigid + uniform scale).

    Registration is driven by ``mesh_to_scale``'s surface(s) of ``orig_meshes`` -- a
    list combines them via ``combine_meshes`` -- unless the caller supplies
    ``icp_transform``, in which case registering is skipped and the supplied transform
    is used (the dataset's later sampling passes reuse the first pass's transform this
    way, so all of a subject's points share one registration). The transform is applied
    to every surface: ``new_meshes`` / ``new_pts`` are updated in place. Returns the
    transform used.
    """
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

    # apply transform to all meshes
    for idx, new_mesh in enumerate(new_meshes):
        if new_mesh is None:
            print(f"Mesh is None... returning None for meshes and pts, {paths[idx]}")
            new_pts[idx] = None
            continue
        new_mesh.apply_transform_to_mesh(icp_transform)
        new_pts[idx] = new_mesh.point_coords

    return icp_transform


def _compute_shared_frame(
    new_pts, mesh_to_scale, scale_all_meshes, center_all_meshes, scale_method
):
    """
    ``(center, scale)`` of one subject's shared frame.

    The scaling points and the centering points are chosen independently:
    ``scale_all_meshes`` uses every surface's points for the scale or only
    ``mesh_to_scale``'s, and ``center_all_meshes`` makes the same choice for the
    center. When the two selections coincide, ``pts_center`` stays None and
    ``get_pts_center_and_scale`` centers on the scaling points themselves. A None
    surface contributes to neither.
    """
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
                pts_center = np.concatenate(pts_center_list, axis=0) if pts_center_list else None
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
    return center, scale


def _draw_surface_samples(
    new_meshes, new_pts, sigma, n_pts, rand_function, include_surf_in_pts, uniform_pts_buffer, seed
):
    """
    Draw each surface's random sample points, in the frame the meshes are in now.

    Sigma is interpreted in that frame: normalized coordinates when centering ran
    before this, the meshes' original units otherwise -- which of the two a training
    run gets depends on ``scale_jointly`` (#3 tracks standardizing that). A surface
    with numeric sigma draws around its own surface; a None sigma draws uniformly
    from one buffered cube around all surfaces jointly. Each surface's draw is
    seeded by ``derive_seed(seed, surface index)``; ``include_surf_in_pts`` appends
    the surface's own vertices to its draw.

    Returns ``(rand_pts, pts_surface)``: coordinates (n, 3), and per point the index
    in the input list of the surface it was drawn around. A missing (None) surface
    leaves a gap in that numbering rather than renumbering those after it,
    deliberately: reconstruction matches surfaces by position.
    """
    rand_pts = []
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
                rand_pts_ = np.concatenate([rand_pts_, new_pts[new_pts_idx]], axis=0)

            rand_pts.append(rand_pts_)
            pts_surface.append(np.full(rand_pts_.shape[0], new_pts_idx, dtype=np.int64))
        else:
            rand_pts.append(np.zeros((0, 3)))
            # 1-D like its siblings: a (0, 3) entry here makes the concatenate below
            # raise as soon as any other surface contributed points.
            pts_surface.append(np.zeros(0, dtype=np.int64))

    return np.concatenate(rand_pts, axis=0), np.concatenate(pts_surface, axis=0)


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

        - ``"pts"`` (n, 3): sample coordinates, all surfaces concatenated. The same key
          as the single-mesh reader on every path (#15 unified them; the single reader
          additionally aliases its random draw as ``"xyz"``, this one never did).
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
        results["icp_transform"] = _register_to_mean(
            orig_meshes, new_meshes, new_pts, paths, mesh_to_scale, mean_mesh, icp_transform
        )
    else:
        print("No registration")
        results["icp_transform"] = None

    toc = time.time()
    print(f"Finished registering meshes in {toc - tic:.3f}s")
    tic = time.time()

    if (center_pts is True) or (norm_pts is True):
        print("Scaling and centering meshes")
        center, scale = _compute_shared_frame(
            new_pts, mesh_to_scale, scale_all_meshes, center_all_meshes, scale_method
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
        rand_pts, pts_surface = _draw_surface_samples(
            new_meshes,
            new_pts,
            sigma,
            n_pts,
            rand_function,
            include_surf_in_pts,
            uniform_pts_buffer,
            seed,
        )

        rand_sdf = []
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
