"""SDF-stepping latent interpolation / shape correspondence.

This module flows the vertices of one shape onto the surface of another while
the NSM latent is interpolated between the two shapes. The endpoint map
``x(0) in surface_A -> x(1) in surface_B`` is the point correspondence.

Each latent increment does a single Newton projection onto the level set
(``x <- x - SDF * grad / ||grad||^2``). When ``tangent_laplacian=True`` and
``faces`` is provided, a tangent-projected Laplacian smoothing pass is applied
after the projection -- vertices on dihedral-angle seams (geometric features)
are pinned in place. This is the production-recommended configuration from
the mesh-interpolation experiments; without ``tangent_laplacian`` it reduces
to a single Newton step per latent increment.

The full set of fixes that were explored (six kwarg-gated variants plus two
added during the work) is recorded in the completed plan
``NSM_MESH_INTERPOLATION_IMPROVEMENTS_COMPLETED.md``. (It also cites an
archive branch/tag — ``mesh-interpolation-improvements``,
``archive/mesh-interp-full-exploration`` — which was never pushed to origin;
as of 2026-08-22 it exists, if anywhere, only in a local clone.)
"""

import numpy as np
import pyvista as pv
import scipy
import torch
from vtk.util.numpy_support import numpy_to_vtk

from .._verbose_deprecation import honour_verbose

EPS = 1e-8


def assert_finite(tensor, name):
    """Helper function to check for NaN/Inf values"""
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN/Inf")


def add_cell_idx(mesh):
    if "cell_idx" not in mesh.scalar_names:
        n_cells = mesh.mesh.GetNumberOfCells()
        cells = np.arange(n_cells)
        cells_ = numpy_to_vtk(cells)
        cells_.SetName("cell_idx")
        mesh.mesh.GetCellData().AddArray(cells_)


def slerp_latent(latent1, latent2, step):
    """
    Spherical linear interpolation of two latent vectors

    Args:
    - latent1 (np.ndarray): The first latent vector
    - latent2 (np.ndarray): The second latent vector
    - step (float): The interpolation step

    Returns:
    - new_latent (np.ndarray): The new latent vector
    """
    assert (step > 0) and (step <= 1)

    latent1_mag = np.linalg.norm(latent1)
    latent2_mag = np.linalg.norm(latent2)

    if latent1_mag < EPS or latent2_mag < EPS:
        return linear_interp_latent(latent1, latent2, step)

    latent1_norm = latent1 / latent1_mag
    latent2_norm = latent2 / latent2_mag

    latent_norm = scipy.spatial.geometric_slerp(latent1_norm, latent2_norm, step)
    latent_mag = (1 - step) * latent1_mag + step * latent2_mag

    new_latent = latent_norm * latent_mag

    return new_latent


def linear_interp_latent(latent1, latent2, step):
    """
    Linear interpolation of two latent vectors

    Args:
    - latent1 (np.ndarray): The first latent vector
    - latent2 (np.ndarray): The second latent vector
    - step (float): The interpolation step

    Returns:
    - new_latent (np.ndarray): The new latent vector
    """
    assert (step > 0) and (step <= 1)

    new_latent = ((1 - step) * latent1) + (step * latent2)

    return new_latent


# ---------------------------------------------------------------------------
# Low-level SDF stepping primitives (all GPU-resident)
# ---------------------------------------------------------------------------


def _to_model_tensor(x, model):
    """Move ``x`` onto the model's device/dtype as a tensor."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if not torch.is_tensor(x):
        x = torch.as_tensor(np.asarray(x))
    return x.to(device=device, dtype=dtype)


def _sdf_step_eval(model, points, latent, surface_idx):
    """Single-surface SDF + spatial-gradient evaluation, GPU-resident.

    Returns:
    - grad_pos (torch.Tensor): d SDF / d x, shape (B, 3).
    - sdf (torch.Tensor): signed distance for the surface, shape (B,).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    points = points.to(device=device, dtype=dtype)
    latent = latent.to(device=device, dtype=dtype)
    if latent.ndim == 1:
        latent = latent.unsqueeze(0)
    B = points.shape[0]
    if latent.shape[0] == 1:
        latent = latent.expand(B, -1)

    pos = points.detach().requires_grad_(True)
    p = torch.cat([latent.detach(), pos], dim=1)

    was_training = model.training
    model.eval()
    sdf_values = model(p)
    assert_finite(sdf_values, "SDF values")
    y = sdf_values[:, surface_idx]
    (grad_pos,) = torch.autograd.grad(y.sum(), pos, create_graph=False, retain_graph=False)
    model.train(was_training)

    grad_pos = grad_pos.detach()
    assert_finite(grad_pos, "Spatial SDF gradient")
    return grad_pos, y.detach()


def _sdf_only(model, points, latent, surface_idx):
    """Forward-only SDF evaluation for one surface (no autograd)."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    points = points.to(device=device, dtype=dtype)
    latent = latent.to(device=device, dtype=dtype)
    if latent.ndim == 1:
        latent = latent.unsqueeze(0)
    if latent.shape[0] == 1:
        latent = latent.expand(points.shape[0], -1)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        sdf_values = model(torch.cat([latent, points], dim=1))
    model.train(was_training)
    return sdf_values[:, surface_idx]


def _unit_gradient(grad_pos):
    """Normalise spatial gradients; return (unit_grad, raw_norm, flat_mask)."""
    grad_norm = torch.norm(grad_pos, dim=1, keepdim=True)
    flat_mask = (grad_norm < EPS).squeeze(-1)
    safe_norm = grad_norm.clamp_min(EPS)
    unit = grad_pos / safe_norm
    unit[flat_mask] = 0.0
    return unit, grad_norm.squeeze(-1), flat_mask


def _project_once(model, latent, points, surface_idx):
    """One Newton projection step onto the level set, GPU-resident.

    ``x <- x - SDF * grad / ||grad||^2``. Equals the unit-normal step when
    ``||grad|| == 1`` (Eikonal) but is strictly more accurate when the
    decoder's gradient norm departs from 1.

    Returns ``(new_points, sdf)`` where ``sdf`` is the *signed* SDF at the
    pre-step points (its absolute value is the off-surface residual).
    """
    grad_pos, sdf = _sdf_step_eval(model, points, latent, surface_idx)
    unit, grad_norm, flat_mask = _unit_gradient(grad_pos)

    scale = sdf / grad_norm.clamp_min(EPS)
    step = unit * scale.unsqueeze(1)
    step[flat_mask] = 0.0
    assert_finite(step, "Projection step")
    new_points = points - step
    assert_finite(new_points, "Projected points")
    return new_points, sdf


# ---------------------------------------------------------------------------
# Mesh structure helpers (used when tangent_laplacian is enabled)
# ---------------------------------------------------------------------------


def build_mesh_laplacian(faces, n_points, device, dtype=torch.float32):
    """Build a row-normalised (umbrella) adjacency operator as a torch sparse tensor.

    Not itself a Laplacian: ``operator @ x`` gives, per vertex, the mean of its
    neighbours' positions; the discrete Laplacian displacement is
    ``(operator @ x) - x``. Built once from the source-mesh connectivity and
    reused every step.

    Args:
    - faces (np.ndarray): triangle connectivity, shape (M, 3).
    - n_points (int): number of mesh vertices.
    - device: torch device for the sparse tensor.
    - dtype: torch dtype for the values.

    Returns:
    - torch.sparse_coo_tensor: row-normalised adjacency, shape (n_points, n_points).
    """
    faces = np.asarray(faces).reshape(-1, 3).astype(np.int64)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    edges = np.unique(edges, axis=0)
    i, j = edges[:, 0], edges[:, 1]
    deg = np.bincount(i, minlength=n_points).astype(np.float64)
    deg[deg == 0] = 1.0
    vals = (1.0 / deg[i]).astype(np.float32)
    idx = torch.as_tensor(np.stack([i, j]), dtype=torch.long, device=device)
    adjacency = torch.sparse_coo_tensor(
        idx, torch.as_tensor(vals, device=device, dtype=dtype), (n_points, n_points)
    ).coalesce()
    return adjacency


def compute_feature_mask(faces, points, dihedral_threshold_deg=45.0):
    """Boolean mask of vertices on geometric features OR topological boundaries.

    For each edge, computes the angle between its two incident face normals;
    edges with dihedral angle above ``dihedral_threshold_deg`` are *feature
    edges*. Topological boundary edges (incident to only one face) and
    non-manifold edges (> 2 faces) are also flagged. Vertices on any such
    edge are returned as True.

    On marching-cubes surfaces the topological boundary detector (one face per
    edge) is a no-op because the meshes are closed; the geometric seam where
    the surface folds back sharply -- e.g. the meniscus thin shell -- is what
    actually needs to be pinned during tangent Laplacian smoothing, and that
    is what the dihedral threshold catches.

    Args:
    - faces (np.ndarray): triangle connectivity, shape (M, 3).
    - points (np.ndarray): vertex positions, shape (N, 3); the face normals
      are taken at these positions (source-mesh positions, computed once).
    - dihedral_threshold_deg (float): edges where the angle between incident
      face normals exceeds this value are feature edges.

    Returns:
    - np.ndarray[bool]: shape (n_points,), True for feature / boundary vertices.
    """
    faces = np.asarray(faces).reshape(-1, 3).astype(np.int64)
    pts = np.asarray(points, dtype=np.float64)
    n_points = max(int(faces.max()) + 1, len(pts))
    n_tri = len(faces)

    e1 = pts[faces[:, 1]] - pts[faces[:, 0]]
    e2 = pts[faces[:, 2]] - pts[faces[:, 0]]
    fn = np.cross(e1, e2)
    fn = fn / np.clip(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20, None)

    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    face_idx = np.tile(np.arange(n_tri), 3)

    unique_e, inverse = np.unique(edges, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    inv_sorted = inverse[order]
    face_sorted = face_idx[order]
    starts = np.concatenate(([0], np.where(np.diff(inv_sorted) != 0)[0] + 1, [len(inv_sorted)]))

    cos_thr = float(np.cos(np.deg2rad(dihedral_threshold_deg)))
    mask = np.zeros(n_points, dtype=bool)
    for k in range(len(unique_e)):
        s, e = starts[k], starts[k + 1]
        n_inc = e - s
        if n_inc != 2:
            mask[unique_e[k, 0]] = True
            mask[unique_e[k, 1]] = True
        else:
            d = float(fn[face_sorted[s]] @ fn[face_sorted[s + 1]])
            if d < cos_thr:
                mask[unique_e[k, 0]] = True
                mask[unique_e[k, 1]] = True
    return mask


def _tangent_laplacian_step(
    model, latent, points, surface_idx, laplacian, pin_mask, alpha, n_iters
):
    """Tangent-projected Laplacian smoothing with feature-vertex pinning.

    Redistributes points *along the surface* without the off-surface pull that
    full-3D smoothing causes: the Laplacian displacement is projected onto the
    local tangent plane (its normal component removed) before being applied,
    and the result is re-projected onto the level set afterwards. Vertices in
    ``pin_mask`` (geometric seams from :func:`compute_feature_mask`) are held
    in place so smoothing cannot blur across a sharp dihedral.
    """
    for _ in range(n_iters):
        grad_pos, _ = _sdf_step_eval(model, points, latent, surface_idx)
        unit, _, _ = _unit_gradient(grad_pos)
        lap = torch.sparse.mm(laplacian, points) - points
        normal_comp = (lap * unit).sum(dim=1, keepdim=True) * unit
        lap_tan = lap - normal_comp
        if pin_mask is not None:
            lap_tan[pin_mask] = 0.0
        points = points + alpha * lap_tan
        points, _ = _project_once(model, latent, points, surface_idx)
    return points


# ---------------------------------------------------------------------------
# Public single-step API (kept backward compatible)
# ---------------------------------------------------------------------------


@honour_verbose
def update_positions(model, new_latent, current_points, surface_idx=0, verbose=True):
    """Single Newton projection of ``current_points`` onto the level set.

    Kept with its original signature and CPU-tensor return value for backward
    compatibility; the interpolation loop uses the GPU-resident primitives
    (:func:`_project_once`) directly.

    Args:
    - model (nn.Module): the SDF decoder.
    - new_latent (np.ndarray or torch.Tensor): the target latent vector.
    - current_points (np.ndarray or torch.Tensor): points to project.
    - surface_idx (int): which decoder output / surface to use.
    - verbose (bool): unused; retained for signature compatibility.

    Returns:
    - new_points (torch.Tensor): the projected points, on CPU.
    """
    latent = _to_model_tensor(new_latent, model)
    points = _to_model_tensor(current_points, model)
    new_points, _ = _project_once(model, latent, points, surface_idx)
    return new_points.detach().cpu()


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def _latent_at(latent1, latent2, t, spherical):
    """Return the interpolated latent at fraction ``t`` in [0, 1]."""
    if t <= 0.0:
        return np.asarray(latent1)
    if t >= 1.0:
        return np.asarray(latent2)
    if spherical:
        return slerp_latent(latent1, latent2, t)
    return linear_interp_latent(latent1, latent2, t)


def _advance(
    model,
    points,
    z_end,
    surface_idx,
    laplacian,
    pin_mask,
    tangent_laplacian,
    tangent_alpha,
    tangent_iters,
):
    """Apply one latent increment: Newton projection (+ optional smoothing)."""
    z_end_t = _to_model_tensor(z_end, model)
    points, _ = _project_once(model, z_end_t, points, surface_idx)
    if tangent_laplacian:
        points = _tangent_laplacian_step(
            model,
            z_end_t,
            points,
            surface_idx,
            laplacian,
            pin_mask,
            tangent_alpha,
            tangent_iters,
        )
    return points


@honour_verbose
def interpolate_common(
    model,
    latent1,
    latent2,
    n_steps=100,
    data=None,
    surface_idx=0,
    verbose=False,
    spherical=True,
    is_mesh=False,
    max_edge_len=0.04,
    adaptive=False,
    smooth=True,
    smooth_type="laplacian",
    faces=None,
    tangent_laplacian=False,
    tangent_laplacian_alpha=0.5,
    tangent_laplacian_iters=1,
    tangent_laplacian_feature_angle=45.0,
):
    """Shared engine behind interpolate_points and interpolate_mesh.

    Steps the latent from ``latent1`` to ``latent2`` in ``n_steps`` increments
    (spherical or linear interpolation), carrying ``data`` along the level set:
    each increment does a single Newton projection onto the surface of the
    current latent (see the module docstring).

    Args:
    - model (nn.Module): the SDF decoder.
    - latent1, latent2 (np.ndarray): source and target latent vectors.
    - n_steps (int): number of latent increments.
    - data: REQUIRED despite the ``None`` default. With ``is_mesh=False``, an
      (N, 3) np.ndarray or torch tensor of point positions on shape A. With
      ``is_mesh=True``, a mskt Mesh whose ``point_coords`` are advanced
      **in place** (the returned object is the caller's own mesh, mutated).
    - surface_idx (int): which decoder output surface to project onto.
    - spherical (bool): slerp (True) or linear (False) latent interpolation.
    - is_mesh (bool): selects the mesh path (per-step VTK subdivide/smooth via
      ``adaptive``/``smooth``/``smooth_type``/``max_edge_len``) or the points
      path (optional tangent-Laplacian smoothing via the ``tangent_*`` args,
      which needs ``faces``; see interpolate_points).

    Returns:
    - With ``is_mesh=False``: (N, 3) np.ndarray of final point positions.
    - With ``is_mesh=True``: the (mutated) input mesh object.
    """
    if data is None:
        raise TypeError(
            "interpolate_common() requires `data` (points array/tensor, or a mesh "
            "with is_mesh=True); it only defaults to None for signature compatibility."
        )

    if is_mesh:
        if not isinstance(data.mesh, pv.PolyData):
            data.mesh = pv.PolyData(data.mesh)
        add_cell_idx(data)

    device = next(model.parameters()).device

    if is_mesh:
        # The mesh path keeps its existing per-step VTK subdivide/smooth
        # behaviour. Per-step projection is a single Newton step; the
        # tangent-Laplacian smoothing (points-path only) is not exposed here
        # because the VTK smoothing already redistributes the mesh.
        for idx, step in enumerate(np.linspace(1 / n_steps, 1, n_steps)):
            if verbose:
                print(f"{idx + 1}/{n_steps}")
            new_latent = _to_model_tensor(
                (
                    slerp_latent(latent1, latent2, step)
                    if spherical
                    else linear_interp_latent(latent1, latent2, step)
                ),
                model,
            )
            points = _to_model_tensor(data.point_coords.copy(), model)
            points, _ = _project_once(model, new_latent, points, surface_idx)
            data.point_coords = points.detach().cpu().numpy()
            if adaptive:
                data.mesh.subdivide_adaptive(
                    max_edge_len=max_edge_len,
                    max_tri_area=None,
                    max_n_tris=None,
                    max_n_passes=3,
                    inplace=True,
                    progress_bar=False,
                )
            if smooth:
                if smooth_type == "laplacian":
                    data.mesh.smooth(inplace=True, relaxation_factor=0.01, n_iter=2)
                elif smooth_type == "taubin":
                    data.mesh.smooth_taubin(inplace=True, n_iter=2, pass_band=0.1)
                else:
                    raise Exception(f"Unknown smoothing type: {smooth_type}")
        return data

    if isinstance(data, np.ndarray):
        points = torch.as_tensor(data, dtype=torch.float).to(device)
    elif torch.is_tensor(data):
        points = data.to(device)
    else:
        raise Exception(f"Unknown data type: {type(data)}")

    if tangent_laplacian and faces is None:
        raise ValueError(
            "tangent_laplacian=True requires the source-mesh `faces` "
            "connectivity; pass `faces=` to interpolate_points."
        )

    laplacian = None
    pin_mask = None
    if tangent_laplacian:
        laplacian = build_mesh_laplacian(faces, points.shape[0], device=device, dtype=points.dtype)
        # Pin geometric seams (high dihedral) plus any topological boundaries
        # so smoothing cannot blur across sharp folds or contract open rims.
        pin_np = compute_feature_mask(
            faces,
            points.detach().cpu().numpy(),
            dihedral_threshold_deg=tangent_laplacian_feature_angle,
        )
        pin_mask = torch.as_tensor(pin_np, device=device)

    for idx, t in enumerate(np.linspace(1 / n_steps, 1, n_steps)):
        if verbose:
            print(f"{idx + 1}/{n_steps}")
        z_end = _latent_at(latent1, latent2, float(t), spherical)
        points = _advance(
            model,
            points,
            z_end,
            surface_idx,
            laplacian,
            pin_mask,
            tangent_laplacian,
            tangent_laplacian_alpha,
            tangent_laplacian_iters,
        )

    return points.detach().cpu().numpy()


@honour_verbose
def interpolate_points(
    model,
    latent1,
    latent2,
    n_steps=100,
    points1=None,
    surface_idx=0,
    verbose=False,
    spherical=True,
    *,
    faces=None,
    tangent_laplacian=False,
    tangent_laplacian_alpha=0.5,
    tangent_laplacian_iters=1,
    tangent_laplacian_feature_angle=45.0,
):
    """Flow ``points1`` (vertices on shape A) onto the surface of shape B.

    Each latent increment performs a single Newton projection
    (``x <- x - SDF * grad / ||grad||^2``) onto the level set of the current
    latent. With ``tangent_laplacian=True`` and ``faces`` provided, an
    additional tangent-projected Laplacian smoothing pass follows each
    projection; vertices on geometric seams (edges whose incident face normals
    differ by more than ``tangent_laplacian_feature_angle`` degrees) are
    pinned. This is the production-recommended configuration on the four
    OAI knee surfaces -- see ``NSM_MESH_INTERPOLATION_IMPROVEMENTS_COMPLETED``.

    Args:
    - model (nn.Module): the SDF decoder.
    - latent1, latent2 (np.ndarray): the source and target latent vectors.
    - n_steps (int): number of latent increments (the NFE knob).
    - points1 (np.ndarray): source vertices, on shape A's surface, shape (N, 3).
    - surface_idx (int): which decoder output / surface to interpolate.
    - verbose (bool): print per-step progress.
    - spherical (bool): slerp the latent (vs linear interpolation).
    - faces (np.ndarray): source-mesh triangle connectivity (M, 3); required
      when ``tangent_laplacian=True``.
    - tangent_laplacian (bool): enable the tangent-projected smoothing pass.
    - tangent_laplacian_alpha (float): smoothing step size (default 0.5).
    - tangent_laplacian_iters (int): smoothing iterations per latent step.
    - tangent_laplacian_feature_angle (float): dihedral threshold (degrees)
      above which an edge is treated as a geometric feature and its vertices
      are pinned during smoothing (default 45 -- the value identified by the
      mesh-interpolation experiments as the best single-config setting).

    Returns:
    - np.ndarray: the warped points, shape (N, 3).
    """
    return interpolate_common(
        model,
        latent1,
        latent2,
        n_steps,
        points1,
        surface_idx,
        verbose,
        spherical,
        is_mesh=False,
        faces=faces,
        tangent_laplacian=tangent_laplacian,
        tangent_laplacian_alpha=tangent_laplacian_alpha,
        tangent_laplacian_iters=tangent_laplacian_iters,
        tangent_laplacian_feature_angle=tangent_laplacian_feature_angle,
    )


@honour_verbose
def interpolate_mesh(
    model,
    latent1,
    latent2,
    n_steps=100,
    mesh=None,
    surface_idx=0,
    verbose=False,
    spherical=True,
    max_edge_len=0.04,
    adaptive=False,
    smooth=True,
    smooth_type="laplacian",
):
    """Flow a full mesh from shape A onto shape B (the VTK-smoothed path).

    This is the ``is_mesh`` path: each latent increment does a single Newton
    projection followed by the existing per-step VTK subdivide / smooth
    behaviour. For correspondence-quality stepping use :func:`interpolate_points`
    with ``tangent_laplacian=True`` instead.
    """
    return interpolate_common(
        model,
        latent1,
        latent2,
        n_steps,
        mesh,
        surface_idx,
        verbose,
        spherical,
        is_mesh=True,
        max_edge_len=max_edge_len,
        adaptive=adaptive,
        smooth=smooth,
        smooth_type=smooth_type,
    )
