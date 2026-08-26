"""
https://github.com/facebookresearch/DeepSDF/blob/main/deep_sdf/mesh.py

Array Layout Conventions
------------------------
Grid sample order: create_grid_samples generates points with Z varying fastest, then Y, then X.
When reshaped with default C-order, the 3D array becomes (X, Y, Z).

All downstream code (VTK and skimage) is fed (X, Y, Z)-layout arrays, and VTK gets Fortran ravel
to match its point-order expectation.

The coarse bounding helper expects (Z, Y, X) internally for vectorized corner slicing — we
transpose to ZYX when calling it, to keep the helper simple and fast.
"""

import inspect
import logging
import os

import numpy as np
import pymskt as mskt
import pyvista as pv
import torch
import vtk
from skimage.measure import marching_cubes

from .._verbose_deprecation import honour_verbose

logger = logging.getLogger(__name__)


def _dilate6(mask: np.ndarray) -> np.ndarray:
    """
    One-iteration 6-neighborhood dilation (faces only).

    Args:
        mask: Boolean array of shape (Z, Y, X)

    Returns:
        Dilated boolean array of same shape
    """
    z, y, x = mask.shape
    out = mask.copy()
    out |= np.pad(mask[1:, :, :], ((0, 1), (0, 0), (0, 0)), constant_values=False)  # -Z
    out |= np.pad(mask[:-1, :, :], ((1, 0), (0, 0), (0, 0)), constant_values=False)  # +Z
    out |= np.pad(mask[:, 1:, :], ((0, 0), (0, 1), (0, 0)), constant_values=False)  # -Y
    out |= np.pad(mask[:, :-1, :], ((0, 0), (1, 0), (0, 0)), constant_values=False)  # +Y
    out |= np.pad(mask[:, :, 1:], ((0, 0), (0, 0), (0, 1)), constant_values=False)  # -X
    out |= np.pad(mask[:, :, :-1], ((0, 0), (0, 0), (1, 0)), constant_values=False)  # +X
    return out


def coarse_bounds_from_sign_change(
    sdf_coarse_zyx,  # np.ndarray or torch.Tensor, shape (Z, Y, X)
    origin,  # (ox, oy, oz) world origin of coarse grid
    spacing_c: float,  # coarse voxel size
    tau_voxels: float = 1.0,  # near-zero band width, in coarse voxels
    dilate_cells: int = 2,  # dilation iterations in coarse-cell units
    limit_near0_to_band: int = 1,  # restrict near0 to band around sign_change
):
    """
    Compute coarse AABB that tightly encloses the zero level set using:
    - sign change across 2x2x2 cell corners (precise),
    - near-zero corners |SDF| <= tau (recall),
    - small index-space dilation (safety).

    Input must be (Z, Y, X) shape for vectorized corner slicing.

    Args:
        sdf_coarse_zyx: SDF values in (Z, Y, X) layout
        origin: (ox, oy, oz) world origin of coarse grid
        spacing_c: Coarse voxel size
        tau_voxels: Near-zero band width, in coarse voxels
        dilate_cells: Dilation iterations in coarse-cell units
        limit_near0_to_band: Restrict near0 to band around sign_change

    Returns:
        tuple: (bounds_min, bounds_max) where each is (x, y, z). None for either of two
        reasons: the coarse grid is degenerate (fewer than 2 samples on some axis, so
        there are no cells to test), or no cell contains the surface.
    """
    sdf = (
        sdf_coarse_zyx.detach().cpu().numpy()
        if isinstance(sdf_coarse_zyx, torch.Tensor)
        else sdf_coarse_zyx
    )
    Z, Y, X = sdf.shape
    if min(Z, Y, X) < 2:
        return None

    tau = tau_voxels * spacing_c

    # corners stacked: (8, Z-1, Y-1, X-1)
    c = []
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                c.append(sdf[dz : Z - 1 + dz, dy : Y - 1 + dy, dx : X - 1 + dx])
    corners = np.stack(c, axis=0)

    cell_min = corners.min(axis=0)
    cell_max = corners.max(axis=0)
    sign_change = (cell_min <= 0.0) & (cell_max >= 0.0)  # (Z-1, Y-1, X-1)

    near0 = np.any(np.abs(corners) <= tau, axis=0)  # (Z-1, Y-1, X-1)

    # Limit near0 growth: expand sign_change by limit_near0_to_band and only allow near0 inside that band
    band = sign_change.copy()
    for _ in range(max(0, int(limit_near0_to_band))):
        band = _dilate6(band)
    cand = sign_change | (near0 & band)

    # Dilation for breathing room
    for _ in range(max(0, int(dilate_cells))):
        cand = _dilate6(cand)

    idx = np.argwhere(cand)
    if idx.size == 0:
        return None

    z0, y0, x0 = idx.min(axis=0)
    z1, y1, x1 = idx.max(axis=0) + 1  # +1 to include far faces

    # map coarse cell indices → world bounds (X,Y,Z), remembering input was Z,Y,X
    bounds_min = np.array(
        [origin[0] + x0 * spacing_c, origin[1] + y0 * spacing_c, origin[2] + z0 * spacing_c],
        dtype=float,
    )
    bounds_max = np.array(
        [origin[0] + x1 * spacing_c, origin[1] + y1 * spacing_c, origin[2] + z1 * spacing_c],
        dtype=float,
    )
    return bounds_min, bounds_max


@honour_verbose
def scale_mesh_(mesh, scale=1.0, offset=(0.0, 0.0, 0.0), icp_transform=None, verbose=False):
    """
    Scale, offset, and (optionally) inverse-ICP-transform a mesh — sometimes in place.

    The trailing underscore only half holds: a mskt Mesh input is mutated in place and
    also returned; any other input (vtk, pyvista, file path) is first wrapped in a new
    mskt Mesh, so the caller's object is untouched. The only in-repo caller is
    scale_mesh below.

    Args:
        mesh: mskt Mesh (mutated in place) or anything mskt.mesh.Mesh() accepts.
        scale (float): multiplier applied to the point coordinates first.
        offset: translation added after scaling.
        icp_transform (vtkTransform, optional): if given, its INVERSE is applied after
            scale+offset (undoing a registration).
        verbose (bool): print progress.

    Returns:
        mskt Mesh: the input object (Mesh input) or the new wrapper (other inputs).
    """
    if not issubclass(type(mesh), mskt.mesh.Mesh):
        mesh = mskt.mesh.Mesh(mesh)

    if verbose is True:
        logger.debug("scale_mesh_. scale: %s", scale)

    pts = mesh.point_coords * scale
    pts += offset

    mesh.point_coords = pts

    if icp_transform is not None:
        transform = vtk.vtkTransform()
        transform.SetMatrix(icp_transform.GetMatrix())
        transform.Inverse()
        if verbose is True:
            logger.debug("%s", icp_transform)
            logger.debug("INVERSE")
            logger.debug("%s", transform)
        mesh.apply_transform_to_mesh(transform)

    return mesh


@honour_verbose
def scale_mesh(
    new_mesh,
    old_mesh=None,
    scale=1.0,
    offset=(0.0, 0.0, 0.0),
    scale_method="max_rad",
    icp_transform=None,
    verbose=False,
):
    """
    Scale/offset new_mesh, deriving the transform from old_mesh when one is given.

    With old_mesh: the passed scale/offset are ignored and recomputed from old_mesh
    (offset = its centroid; scale = its max point radius after centering, the "max_rad"
    method — the only one implemented). Without old_mesh: the passed scale/offset are
    used as-is. Either way the work is done by scale_mesh_ (see its docstring for the
    in-place caveat and the inverse-ICP behaviour).

    Returns:
        mskt Mesh: the transformed mesh.
    """
    if old_mesh is not None:
        old_mesh = mskt.mesh.Mesh(old_mesh)  # should handle vtk, pyvista, or string path to file
        old_pts = old_mesh.point_coords

        if not issubclass(type(new_mesh), mskt.mesh.Mesh):
            new_mesh = mskt.mesh.Mesh(
                new_mesh
            )  # should handle vtk, pyvista, or string path to file

        offset = np.mean(old_pts, axis=0)
        old_pts -= offset

        if scale_method == "max_rad":
            scale = np.max(np.linalg.norm(old_pts, axis=-1), axis=-1)
        else:
            raise NotImplementedError

    mesh = scale_mesh_(
        new_mesh, scale=scale, offset=offset, icp_transform=icp_transform, verbose=verbose
    )
    return mesh


@honour_verbose
def create_mesh(
    decoder,
    latent_vector,
    n_pts_per_axis=256,
    voxel_origin=(-1, -1, -1),
    voxel_size=None,
    batch_size=32**3,
    scale=1.0,
    offset=(0.0, 0.0, 0.0),
    path_save=None,
    filename="mesh_{mesh_idx}.vtk",
    path_original_mesh=None,
    scale_to_original_mesh=True,
    icp_transform=None,
    objects=1,
    verbose=False,
    device="cuda",
    use_vtk=True,
):
    """
    Reconstruct surface mesh(es) from a decoder + latent by dense marching cubes.

    Evaluates the decoder on a full n_pts_per_axis^3 grid (voxel_size defaults to
    2/(n_pts_per_axis - 1), spanning [-1, 1]^3 from voxel_origin), extracts the zero
    level set per object (VTK FlyingEdges when use_vtk, else skimage marching cubes),
    then optionally rescales each mesh back to the original coordinate frame
    (scale_to_original_mesh via scale_mesh) and saves it (path_save/filename).

    Returns:
        A single mskt Mesh when objects == 1, else a list of length `objects`.
        An object whose SDF never crosses zero yields None in its slot (with a
        warning when verbose).
    """
    if voxel_size is None:
        voxel_size = 2.0 / (n_pts_per_axis - 1)

    decoder.eval()

    samples = create_grid_samples(n_pts_per_axis, voxel_origin, voxel_size)
    sdf_values_ = get_sdfs(
        decoder, samples, latent_vector, batch_size, objects=objects, device=device
    )

    # Reshape SDFs into grid: C-order reshape (default) makes last index vary fastest.
    # Since samples have z-fastest order, this gives array[x, y, z] → (X,Y,Z) layout
    sdf_values = torch.zeros((n_pts_per_axis, n_pts_per_axis, n_pts_per_axis, objects))
    for i in range(objects):
        sdf_values[..., i] = sdf_values_[..., i].reshape(
            n_pts_per_axis, n_pts_per_axis, n_pts_per_axis
        )
    # sdf_values = sdf_values.reshape(n_pts_per_axis, n_pts_per_axis, n_pts_per_axis)

    # create mesh from gridded SDFs
    meshes = []
    for mesh_idx in range(objects):
        # iterate over all the meshes
        sdf_values_ = sdf_values[..., mesh_idx]

        # check if there is a surface
        if 0 < sdf_values_.min() or 0 > sdf_values_.max():
            if verbose is True:
                logger.warning("SDF values do not span 0 - there is no surface")
                logger.warning("\tSDF min:  %s", sdf_values_.min())
                logger.warning("\tSDF max:  %s", sdf_values_.max())
                logger.warning("\tSDF mean:  %s", sdf_values_.mean())
            meshes.append(None)
        else:
            # if there is a surface, then extract it & post-process
            # for mesh_idx in range(objects):
            if use_vtk:
                mesh = sdf_grid_to_mesh_vtk(sdf_values_, voxel_origin, voxel_size)
            else:
                mesh = sdf_grid_to_mesh(sdf_values_, voxel_origin, voxel_size)
            meshes.append(mesh)

            if scale_to_original_mesh:
                if verbose is True:
                    logger.debug("Scaling mesh to original mesh... ")
                    logger.debug("%s", icp_transform)
                # for mesh_idx, mesh in enumerate(meshes):
                mesh = scale_mesh(
                    meshes[mesh_idx],
                    old_mesh=path_original_mesh,
                    scale=scale,
                    offset=offset,
                    icp_transform=icp_transform,
                    verbose=verbose,
                )
                meshes[mesh_idx] = mesh

            # save the mesh (if desired)
            if path_save is not None:
                # for mesh_idx, mesh in enumerate(meshes):
                meshes[mesh_idx].save_mesh(
                    os.path.join(path_save, filename.format(mesh_idx=mesh_idx))
                )
    return meshes[0] if objects == 1 else meshes


@honour_verbose
def sdf_grid_to_mesh(
    sdf_values,
    voxel_origin,
    voxel_size,
    verbose=False,
    narrow_band=True,
    band_width=3.0,
    pad_voxels=2,
):
    """
    Extract the zero level set of a gridded SDF with skimage marching cubes.

    The torch-tensor twin of sdf_grid_to_mesh_vtk (this one requires a torch tensor —
    the first line calls .cpu(); the VTK twin accepts numpy too). With narrow_band,
    the volume is first cropped by crop_sdf_to_narrow_band.

    Args:
        sdf_values (torch.Tensor): SDF grid in array[x, y, z] layout.
        voxel_origin: (x, y, z) world position of grid index (0, 0, 0).
        voxel_size (float): isotropic voxel edge length.
        narrow_band / band_width / pad_voxels: see crop_sdf_to_narrow_band.

    Returns:
        mskt Mesh: the extracted surface.
    """
    if verbose is True:
        logger.debug("Starting marching cubes... ")

    sub_sdf, crop_origin = _volume_and_origin(
        sdf_values, voxel_origin, voxel_size, narrow_band, band_width, pad_voxels, verbose
    )

    verts, faces, normals, values = marching_cubes(
        sub_sdf, level=0, spacing=(voxel_size, voxel_size, voxel_size)
    )

    if verbose is True:
        logger.debug("Starting vert/face conversion...")

    verts += crop_origin

    faces_ = []
    for face_idx in range(faces.shape[0]):
        face = np.insert(faces[face_idx, :], 0, faces.shape[1])
        faces_.append(face)

    faces = np.hstack(faces_)

    if verbose is True:
        logger.debug("Creating mesh... ")

    mesh = mskt.mesh.Mesh(mesh=pv.PolyData(verts, faces))

    return mesh


@honour_verbose
def crop_sdf_to_narrow_band(
    sdf_values, voxel_origin, voxel_size, band_width=3.0, pad_voxels=2, verbose=False
):
    """
    Crop SDF volume to a narrow band around the surface for faster processing.

    Args:
        sdf_values: numpy array containing SDF values
        voxel_origin: Origin point of the voxel grid (x, y, z)
        voxel_size: Size of each voxel
        band_width: Width of narrow band, as a multiplier of voxel_size
        pad_voxels: Number of voxels to pad around cropped region
        verbose: Whether to print progress messages

    Returns:
        tuple: (cropped_sdf, new_origin) or (original_sdf, original_origin) if no cropping needed
    """
    orig_nx, orig_ny, orig_nz = sdf_values.shape

    if verbose:
        logger.debug(
            "Applying narrow band optimization (band_width=%s * voxel_size)...", band_width
        )

    # Find voxels within the narrow band around the surface.
    # The volume is in array[x, y, z] layout, so np.where's axes are (X, Y, Z).
    band = band_width * voxel_size
    mask = np.abs(sdf_values) <= band
    ix, iy, iz = np.where(mask)

    if len(ix) == 0:
        if verbose:
            logger.warning("No voxels found within narrow band - using full volume")
        return sdf_values, voxel_origin

    # Calculate cropping bounds with padding
    x0 = max(ix.min() - pad_voxels, 0)
    x1 = min(ix.max() + pad_voxels + 1, orig_nx)
    y0 = max(iy.min() - pad_voxels, 0)
    y1 = min(iy.max() + pad_voxels + 1, orig_ny)
    z0 = max(iz.min() - pad_voxels, 0)
    z1 = min(iz.max() + pad_voxels + 1, orig_nz)

    # Extract subvolume
    sub_sdf = sdf_values[x0:x1, y0:y1, z0:z1]

    crop_origin = (
        voxel_origin[0] + x0 * voxel_size,
        voxel_origin[1] + y0 * voxel_size,
        voxel_origin[2] + z0 * voxel_size,
    )

    if verbose:
        logger.debug("Cropped volume from %sx%sx%s to %s", orig_nx, orig_ny, orig_nz, sub_sdf.shape)
        logger.debug("Original origin: %s, Cropped origin: %s", voxel_origin, crop_origin)

    return sub_sdf, crop_origin


def _volume_and_origin(
    sdf_values, voxel_origin, voxel_size, narrow_band, band_width, pad_voxels, verbose
):
    """The input handling both extraction twins share, in one place so it cannot drift.

    It drifted once (#60): the VTK twin guarded the tensor conversion with ``hasattr``
    and defaulted ``narrow_band`` to True, the skimage twin called ``.cpu()``
    unconditionally and defaulted it to False -- so ``use_vtk``, which is meant to pick
    an extraction backend, also picked an accepted input type and a cropping policy.

    Returns:
        tuple: (numpy volume in array[x, y, z] layout, world origin of its index 0).
    """
    if hasattr(sdf_values, "cpu"):
        sdf_values = sdf_values.cpu().numpy()

    if not narrow_band:
        return sdf_values, voxel_origin

    return crop_sdf_to_narrow_band(
        sdf_values, voxel_origin, voxel_size, band_width, pad_voxels, verbose
    )


@honour_verbose
def sdf_grid_to_mesh_vtk(
    sdf_values,
    voxel_origin,
    voxel_size,
    verbose=False,
    narrow_band=True,
    band_width=3.0,
    pad_voxels=2,
):
    """
    Create mesh from SDF values using VTK Flying Edges algorithm instead of marching cubes.

    Args:
        sdf_values: torch tensor or numpy array containing SDF values
        voxel_origin: Origin point of the voxel grid (x, y, z)
        voxel_size: Size of each voxel
        verbose: Whether to print progress messages
        narrow_band: Whether to crop volume to narrow band around surface for speed
        band_width: Width of narrow band, as a multiplier of voxel_size
        pad_voxels: Number of voxels to pad around cropped region

    Returns:
        mskt.mesh.Mesh object
    """
    if verbose:
        logger.debug("Starting VTK Flying Edges mesh extraction...")

    sub_sdf, crop_origin = _volume_and_origin(
        sdf_values, voxel_origin, voxel_size, narrow_band, band_width, pad_voxels, verbose
    )

    # Get grid dimensions (cropped or original)
    nx, ny, nz = sub_sdf.shape

    # Create PyVista ImageData: dimensions are (X,Y,Z) from array.shape = (nx, ny, nz)
    # Flatten with order="F" (Fortran) so X varies fastest, matching VTK's expectation
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)  # VTK expects (X, Y, Z) counts
    grid.spacing = (voxel_size, voxel_size, voxel_size)
    grid.origin = crop_origin  # World coordinates (X, Y, Z)
    grid["sdf"] = sub_sdf.ravel(order="F")  # Fortran-order: X varies fastest

    # Apply Flying Edges 3D algorithm
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(grid)
    fe.SetValue(0, 0.0)  # SDF iso-level
    fe.ComputeNormalsOff()  # compute later from SDF grads if desired
    fe.Update()

    # Wrap the output as PyVista mesh and create mskt mesh directly
    mesh = mskt.mesh.Mesh(mesh=fe.GetOutput())
    if verbose:
        logger.debug(
            "Extracted mesh with %s vertices and %s faces", mesh.n_points, mesh.n_faces_strict
        )
        logger.debug("Creating final mesh object...")

    return mesh


def create_grid_samples_in_bounds(
    bounds_min,
    bounds_max,
    original_spacing,
    padding=0.1,
    min_dim=64,
    min_pad_voxels_fine=3,
):
    """
    Create dense grid samples within discovered bounds using consistent spacing.

    Args:
        bounds_min: (x, y, z) minimum bounds, as a numpy array (element-wise
            arithmetic is performed on it; a plain tuple raises TypeError)
        bounds_max: (x, y, z) maximum bounds, as a numpy array
        original_spacing: Voxel spacing from original full grid
        padding: Extra space around bounds (world units)
        min_dim: Minimum dimension per axis (for VTK stability)
        min_pad_voxels_fine: Minimum padding in fine voxels

    Returns:
        tuple: (samples, grid_dims, voxel_origin) where:
            - samples: Grid samples of shape (N, 3)
            - grid_dims: (nx, ny, nz) dimensions
            - voxel_origin: (ox, oy, oz) world origin
    """
    # World padding: keep the larger of explicit padding and min-pad in fine voxels
    pad_world = max(padding, min_pad_voxels_fine * original_spacing)

    padded_min = bounds_min - pad_world
    padded_max = bounds_max + pad_world

    # Calculate grid dimensions to maintain original spacing, enforcing minimum
    nx = max(int((padded_max[0] - padded_min[0]) / original_spacing) + 1, min_dim)
    ny = max(int((padded_max[1] - padded_min[1]) / original_spacing) + 1, min_dim)
    nz = max(int((padded_max[2] - padded_min[2]) / original_spacing) + 1, min_dim)

    n_pts_total = nx * ny * nz

    indices = torch.arange(0, n_pts_total, out=torch.LongTensor())
    samples = torch.zeros(n_pts_total, 3)

    # Generate samples with Z varying fastest (same pattern as create_grid_samples)
    samples[:, 2] = indices % nz
    samples[:, 1] = (indices // nz) % ny
    samples[:, 0] = ((indices // nz) // ny) % nx

    # Scale to actual coordinates within bounds
    samples[:, 0] = samples[:, 0] * original_spacing + padded_min[0]
    samples[:, 1] = samples[:, 1] * original_spacing + padded_min[1]
    samples[:, 2] = samples[:, 2] * original_spacing + padded_min[2]

    return samples, (nx, ny, nz), (padded_min[0], padded_min[1], padded_min[2])


@honour_verbose
def create_mesh_adaptive(
    decoder,
    latent_vector,
    n_pts_per_axis=256,
    voxel_origin=None,
    voxel_size=None,
    n_pts_coarse=64,
    search_bounds=(-1.0, 1.0),
    bounds_padding=0.05,
    tau_voxels=1.0,
    dilate_cells=2,
    limit_near0_to_band=1,
    min_dim=64,
    min_pad_voxels_fine=3,
    batch_size=300_000,
    scale=1.0,
    offset=(0.0, 0.0, 0.0),
    path_save=None,
    filename="mesh_{mesh_idx}.vtk",
    path_original_mesh=None,
    scale_to_original_mesh=True,
    icp_transform=None,
    objects=1,
    verbose=False,
    device="cuda",
    use_vtk=True,
    fallback_to_original=True,
):
    """
    Create mesh using adaptive two-pass sampling with deterministic coarse grid bounds detection.

    Adaptive meshing overview
    -------------------------
    Pass 1 (coarse): Evaluate a coarse SDF grid over [search_bounds]^3.
    We detect candidate cells via:
      - sign_change across 2x2x2 corners (precise),
      - near-zero corners |SDF| <= tau_voxels * coarse_spacing (recall),
      - small index-space dilation (safety).
    This yields a coarse AABB in world coords.

    Pass 2 (dense): Build a dense grid at the original fine voxel size only
    within that AABB, padded by >= min_pad_voxels_fine fine voxels per side.
    Extract with VTK FlyingEdges (or skimage) preserving array layout:
      - samples are generated Z-fastest, reshape with C-order -> (X,Y,Z),
      - VTK gets Fortran-ravel of (X,Y,Z).

    This method is deterministic, robust to thin structures, and avoids
    the randomness/clumping of point sampling.

    Args:
        decoder: SDF decoder network
        latent_vector: Latent code for the shape
        n_pts_per_axis: Sets the fine voxel size whenever voxel_size is None
            (extent / (n_pts_per_axis - 1)) — the mean-mesh caller's case — and
            the full-grid resolution on the fallback path
        voxel_origin: Origin of the *fallback* full grid, and of nothing else -- the
            two-pass path derives its own origin from the detected bounds and
            overwrites this. None (the default) takes it from search_bounds, which
            is the only value consistent with the voxel_size derived from the same
            place; see #60
        voxel_size: Voxel size for dense grid (computed if None)
        n_pts_coarse: Coarse grid resolution per axis
        search_bounds: (min, max) bounds for coarse grid
        bounds_padding: Extra world-space padding (see also min_pad_voxels_fine)
        tau_voxels: Near-zero band width in coarse voxels
        dilate_cells: Dilation iterations on coarse grid
        limit_near0_to_band: Restrict near-zero expansion to band around sign-change
        min_dim: Minimum dimension per axis for dense grid (VTK stability)
        min_pad_voxels_fine: Minimum padding in fine voxels
        batch_size: Batch size for SDF evaluation
        scale: Scale factor for final mesh
        offset: Offset for final mesh
        path_save: Directory to save meshes
        filename: Filename pattern for saving
        path_original_mesh: Path to original mesh for scaling
        scale_to_original_mesh: Whether to scale to original mesh
        icp_transform: ICP transform to apply
        objects: Number of objects to extract
        verbose: Print progress messages
        device: Device for computation
        use_vtk: Use VTK Flying Edges (vs marching cubes)
        fallback_to_original: Fall back to full grid if bounds detection fails

    Returns:
        Single mesh (if objects==1) or list of meshes

    Falls back to original create_mesh if bounds detection fails.
    """
    if verbose:
        logger.debug("Starting adaptive mesh creation...")

    # Calculate voxel size if not provided
    if voxel_size is None:
        original_extent = search_bounds[1] - search_bounds[0]
        voxel_size = original_extent / (n_pts_per_axis - 1)

    # The fallback grid's origin has to come from the same place its spacing does. It
    # used to default to (-1, -1, -1) independently of search_bounds, so a caller who
    # moved the search region got a fallback grid that did not cover it (#60): with
    # search_bounds=(0, 4) and n_pts_per_axis=17 the grid spanned [-1, 3].
    if voxel_origin is None:
        voxel_origin = (search_bounds[0],) * 3

    # Use voxel_size as the original spacing
    original_spacing = voxel_size

    decoder.eval()

    # Pass 1: Coarse grid to find object bounds
    if verbose:
        logger.debug("Pass 1: Evaluating coarse %s^3 grid...", n_pts_coarse)

    coarse_origin = (search_bounds[0], search_bounds[0], search_bounds[0])
    coarse_extent = search_bounds[1] - search_bounds[0]
    coarse_spacing = coarse_extent / (n_pts_coarse - 1)

    # Generate coarse grid samples
    coarse_samples = create_grid_samples(
        n_pts_per_axis=n_pts_coarse, voxel_origin=coarse_origin, voxel_size=coarse_spacing
    )

    # Evaluate SDF on coarse grid
    coarse_sdf_values_flat = get_sdfs(
        decoder, coarse_samples, latent_vector, batch_size, objects=objects, device=device
    )

    # Union across objects (inside if any object is inside): min over objects
    coarse_sdf_flat = torch.min(coarse_sdf_values_flat, dim=1)[0]

    # Reshape to (X,Y,Z), then transpose to (Z,Y,X) for the helper
    sdf_coarse_xyz = coarse_sdf_flat.reshape(n_pts_coarse, n_pts_coarse, n_pts_coarse)
    sdf_coarse_zyx = np.transpose(sdf_coarse_xyz.cpu().numpy(), (2, 1, 0))

    # Detect bounds using sign-change + near-zero + dilation
    bounds_result = coarse_bounds_from_sign_change(
        sdf_coarse_zyx,
        origin=coarse_origin,
        spacing_c=coarse_spacing,
        tau_voxels=tau_voxels,
        dilate_cells=dilate_cells,
        limit_near0_to_band=limit_near0_to_band,
    )

    if bounds_result is None:
        if verbose:
            logger.warning("Coarse pass found no surface. Falling back.")
        if fallback_to_original:
            if verbose:
                logger.warning("Falling back to original create_mesh...")
            # Keywords, not the 17 positionals this used to be: create_mesh's
            # signature is public and any insertion into it would have shifted them
            # silently. ARCHITECTURE.md section 7 lists this call as its example of
            # the LR bug's shape.
            return create_mesh(
                decoder,
                latent_vector,
                n_pts_per_axis=n_pts_per_axis,
                voxel_origin=voxel_origin,
                voxel_size=voxel_size,
                batch_size=batch_size,
                scale=scale,
                offset=offset,
                path_save=path_save,
                filename=filename,
                path_original_mesh=path_original_mesh,
                scale_to_original_mesh=scale_to_original_mesh,
                icp_transform=icp_transform,
                objects=objects,
                verbose=verbose,
                device=device,
                use_vtk=use_vtk,
            )
        else:
            return [None] * objects if objects > 1 else None

    bounds_min, bounds_max = bounds_result

    if verbose:
        logger.debug("Coarse spacing: %.6f, tau: %.6f", coarse_spacing, tau_voxels * coarse_spacing)
        logger.debug("Coarse bounds: min=%s, max=%s", bounds_min, bounds_max)
        extent = bounds_max - bounds_min
        logger.debug("Object extent: %s", extent)

    # Pass 2: Dense sampling in bounded region
    if verbose:
        logger.debug("Pass 2: Creating dense grid in bounded region...")

    samples, grid_dims, voxel_origin = create_grid_samples_in_bounds(
        bounds_min,
        bounds_max,
        original_spacing,
        bounds_padding,
        min_dim=min_dim,
        min_pad_voxels_fine=min_pad_voxels_fine,
    )

    if verbose:
        logger.debug("Dense dims: %s, voxel_size: %.6f", grid_dims, original_spacing)
        logger.debug("Dense grid: %s points (vs %s original)", samples.shape[0], n_pts_per_axis**3)
        logger.debug("Speedup: %.1fx fewer points", n_pts_per_axis**3 / samples.shape[0])

    # Get SDF values for dense grid
    sdf_values_ = get_sdfs(
        decoder, samples, latent_vector, batch_size, objects=objects, device=device
    )

    # Reshape SDF values: C-order makes array[x, y, z] correspond to world (X,Y,Z)
    nx, ny, nz = grid_dims
    sdf_values = torch.zeros((nx, ny, nz, objects))
    for i in range(objects):
        sdf_values[..., i] = sdf_values_[..., i].reshape(nx, ny, nz)

    # Calculate voxel size for the bounded grid
    voxel_size = original_spacing

    # Create meshes from gridded SDFs (same as original pipeline)
    meshes = []
    for mesh_idx in range(objects):
        sdf_values_ = sdf_values[..., mesh_idx]

        # Check if there is a surface
        if 0 < sdf_values_.min() or 0 > sdf_values_.max():
            if verbose is True:
                logger.warning("SDF values do not span 0 - there is no surface")
                logger.warning("\tSDF min:  %s", sdf_values_.min())
                logger.warning("\tSDF max:  %s", sdf_values_.max())
                logger.warning("\tSDF mean:  %s", sdf_values_.mean())
            meshes.append(None)
        else:
            # Extract surface using VTK or marching cubes
            if use_vtk:
                mesh = sdf_grid_to_mesh_vtk(sdf_values_, voxel_origin, voxel_size, verbose)
            else:
                mesh = sdf_grid_to_mesh(sdf_values_, voxel_origin, voxel_size, verbose)
            meshes.append(mesh)

            if scale_to_original_mesh:
                if verbose is True:
                    logger.debug("Scaling mesh to original mesh... ")
                    logger.debug("%s", icp_transform)
                mesh = scale_mesh(
                    meshes[mesh_idx],
                    old_mesh=path_original_mesh,
                    scale=scale,
                    offset=offset,
                    icp_transform=icp_transform,
                    verbose=verbose,
                )
                meshes[mesh_idx] = mesh

            # Save the mesh (if desired)
            if path_save is not None:
                meshes[mesh_idx].save_mesh(
                    os.path.join(path_save, filename.format(mesh_idx=mesh_idx))
                )

    return meshes[0] if objects == 1 else meshes


def create_grid_samples(
    n_pts_per_axis=256,
    voxel_origin=(-1, -1, -1),
    voxel_size=None,
):
    """
    Build the flat (N^3, 3) coordinate list for a regular grid.

    Sample order is Z-fastest (x=0,y=0,z=0 then x=0,y=0,z=1, ...), so a C-order
    reshape of per-sample values gives array[x, y, z] layout — the convention every
    grid consumer in this module relies on.

    Args:
        n_pts_per_axis (int): samples per axis (N).
        voxel_origin: (x, y, z) world position of grid index (0, 0, 0).
        voxel_size (float): grid spacing. REQUIRED despite the None default
            (None raises TypeError in the scaling arithmetic).

    Returns:
        torch.Tensor: (n_pts_per_axis**3, 3) world coordinates.
    """
    n_pts_total = n_pts_per_axis**3

    indices = torch.arange(0, n_pts_total, out=torch.LongTensor())
    samples = torch.zeros(n_pts_total, 3)

    # Generate samples with Z varying fastest, then Y, then X
    # samples[0] = (x=0, y=0, z=0), samples[1] = (x=0, y=0, z=1), ...
    # When reshaped with C-order (default), this produces array[x, y, z] indexing
    samples[:, 2] = indices % n_pts_per_axis
    samples[:, 1] = (indices // n_pts_per_axis) % n_pts_per_axis
    samples[:, 0] = ((indices // n_pts_per_axis) // n_pts_per_axis) % n_pts_per_axis

    # scale & transform the grid as appropriate
    samples[:, :3] = samples[:, :3] * voxel_size
    for axis in range(3):
        samples[:, axis] = samples[:, axis] + voxel_origin[axis]

    return samples


def get_sdfs(decoder, samples, latent_vector, batch_size=32**3, objects=1, device="cuda"):
    """
    Get SDF values for samples.

    Args:
        decoder: The decoder model
        samples: Sample points to evaluate
        latent_vector: Latent code for the shape
        batch_size: Batch size for processing points
        objects: Number of objects
        device: Device to run on
    """
    n_pts_total = samples.shape[0]
    current_idx = 0
    sdf_values = torch.zeros(samples.shape[0], objects)

    if batch_size > n_pts_total:
        logger.warning(
            "batch_size is greater than the number of samples, setting batch_size to the number of samples"
        )
        batch_size = n_pts_total

    batch_num = 0
    while current_idx < n_pts_total:
        current_batch_size = min(batch_size, n_pts_total - current_idx)
        sampled_pts = samples[current_idx : current_idx + current_batch_size, :3].to(device)

        sdf_values[current_idx : current_idx + current_batch_size, :] = (
            decode_sdf(decoder, latent_vector, sampled_pts).detach().cpu()
        )

        current_idx += current_batch_size
        logger.debug(
            "Processed %s / %s points (batch %s: CNN+MLP, size=%s)",
            current_idx,
            n_pts_total,
            batch_num + 1,
            current_batch_size,
        )
        batch_num += 1

    return sdf_values


def decode_sdf(decoder, latent_vector, queries):
    """
    Decode SDF values for query points.

    Args:
        decoder: The decoder model
        latent_vector: Latent code for the shape
        queries: Query points (N, 3)
    """
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
        return decoder(inputs)
    else:
        # Check if decoder supports fast inference interface (latent + xyz)
        if hasattr(decoder, "forward"):
            sig = inspect.signature(decoder.forward)
            if "latent" in sig.parameters and "xyz" in sig.parameters:
                # Use fast inference interface
                return decoder(latent=latent_vector.squeeze(), xyz=queries)

        # Fall back to legacy concatenated interface
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], dim=1)
        return decoder(inputs)
