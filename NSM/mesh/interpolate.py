"""SDF-stepping latent interpolation / shape correspondence.

This module flows the vertices of one shape onto the surface of another while
the NSM latent is linearly interpolated between the two shapes. The endpoint
map ``x(0) in surface_A -> x(1) in surface_B`` is the point correspondence.

The baseline method is a first-order closest-point projection done once per
latent increment. Six independently kwarg-gated *fixes* improve the numerical
quality of that stepping (see ``NSM_MESH_INTERPOLATION_IMPROVEMENTS`` plan):

- Fix 1 -- per-step convergence loop (``n_corrector_iters`` > 1).
- Fix 2 -- true Newton magnitude (``step_magnitude="newton"``).
- Fix 3 -- latent-advection predictor (``latent_predictor=True``).
- Fix 4b -- tangent-projected Laplacian smoothing (``tangent_laplacian=True``).
- Fix 5 -- adaptive latent step-sizing (``adaptive_steps=True``).
- Fix 6 -- batched line-search magnitude (``step_magnitude="line_search"``).

Every fix defaults to OFF; the default call reproduces the original behaviour
exactly (one ``"normal"`` projection per latent increment).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pyvista as pv
import scipy
import torch
from vtk.util.numpy_support import numpy_to_vtk

from NSM.utils import print_gpu_memory

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


def sdf_gradients(sdf_model, points, latent, surface_idx=None, verbose=False):
    """
    Computes gradients of SDF with respect to 3D positions (not latent).
    If surface_idx is provided, computes only that surface's gradient (fastest).
    Otherwise returns gradients for all surfaces.

    If the points are on the surface of the specific latent, then the gradients
    are equivalent to the normal vectors of the surface. If they are not on the
    surface, then they are the gradient of the SDF at that point and indicate
    the direction of the steepest ascent.

    Note: this function only computes the *spatial* gradient ``d SDF / d x``.
    The latent-advection term ``d SDF / d z`` used by Fix 3 is computed by the
    private helper :func:`_sdf_step_eval` (a directional derivative along the
    latent increment, which is cheaper than the full latent Jacobian).

    Args:
    - sdf_model (nn.Module): The model that computes the SDF
    - points (np.ndarray or torch.tensor): The points for which to compute gradients (B, 3)
    - latent (np.ndarray or torch.tensor): The latent vector for the specific shape
    - surface_idx (int, optional): If provided, only compute gradients for this surface (0-based)
    - verbose (bool): If True, print the GPU memory usage after gradient computation

    Returns:
    - gradients (torch.Tensor):
        - If surface_idx provided: gradients for that surface only (B, latent_dim + 3)
        - If surface_idx is None: list of gradients for each surface
    - sdf_values (torch.Tensor): The SDF values for each point (B, num_surfaces)
    """
    # Convert to tensors
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)
    if isinstance(latent, np.ndarray):
        latent = torch.from_numpy(latent)

    # Get device and dtype from model
    device = next(sdf_model.parameters()).device
    dtype = next(sdf_model.parameters()).dtype
    points = points.to(device=device, dtype=dtype)
    latent = latent.to(device=device, dtype=dtype)

    B = points.shape[0]
    D_lat = latent.shape[-1]
    assert points.shape[-1] == 3, "points must be (B, 3)"

    # Handle latent vector shape
    if latent.ndim == 1:
        latent = latent.unsqueeze(0)  # (1, D_lat)
    if latent.shape[0] == 1:
        latent = latent.expand(B, -1)  # (B, D_lat)

    # Only positions need gradients (more efficient than full input)
    pos = points.detach().requires_grad_(True)  # (B, 3)
    vecs = latent.detach()  # (B, D_lat) no grad needed

    # Concatenate for model input
    p = torch.cat([vecs, pos], dim=1)  # (B, D_lat + 3)

    # Set model to eval mode for stability during gradient computation
    was_training = sdf_model.training
    sdf_model.eval()

    # Forward pass
    sdf_values = sdf_model(p)  # (B, Ns)
    assert_finite(sdf_values, "SDF values")

    def _finish(g):
        if verbose:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            print_gpu_memory()
        return g.detach().cpu(), sdf_values.detach().cpu()

    # Fast path: single surface only
    if surface_idx is not None:
        y = sdf_values[:, surface_idx]  # (B,)
        # Use sum() trick - equivalent to one-hot grad_outputs but more efficient
        (grad_pos,) = torch.autograd.grad(
            y.sum(), pos, create_graph=False, retain_graph=False, allow_unused=False
        )
        sdf_model.train(was_training)
        assert_finite(grad_pos, f"Gradients for surface {surface_idx}")

        # Reconstruct full gradient (latent + position) for backward compatibility
        grad_latent_zeros = torch.zeros(B, D_lat, device=device, dtype=dtype)
        full_grad = torch.cat([grad_latent_zeros, grad_pos], dim=1)
        return _finish(full_grad)  # (B, D_lat + 3), (B, Ns)

    # All surfaces (for backward compatibility)
    Ns = sdf_values.shape[1]
    gradients = []

    for i in range(Ns):
        y = sdf_values[:, i]
        (grad_pos,) = torch.autograd.grad(
            y.sum(), pos, create_graph=False, retain_graph=(i < Ns - 1)
        )
        assert_finite(grad_pos, f"Gradients for surface {i}")

        # Reconstruct full gradient for backward compatibility
        grad_latent_zeros = torch.zeros(B, D_lat, device=device, dtype=dtype)
        full_grad = torch.cat([grad_latent_zeros, grad_pos], dim=1)
        gradients.append(full_grad.detach().cpu())

    sdf_model.train(was_training)
    return gradients, sdf_values.detach().cpu()


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

    # Protect against zero magnitude latents
    if latent1_mag < EPS or latent2_mag < EPS:
        # Fall back to linear interpolation if either vector has near-zero magnitude
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
# Step configuration
# ---------------------------------------------------------------------------


@dataclass
class StepConfig:
    """Bundles every fix toggle so it can be threaded through the stepping code.

    Every field defaults to the *baseline* behaviour; an all-default config
    reproduces the original single-projection method exactly.
    """

    # Fix 1 -- per-step corrector loop.
    n_corrector_iters: int = 1
    corrector_tol: float = 1e-4
    # Fix 2 / Fix 6 -- magnitude of the projection step.
    #   "normal"      -> x - SDF * n_hat              (baseline)
    #   "newton"      -> x - SDF * grad / ||grad||^2  (Fix 2)
    #   "line_search" -> batched line-search / quadratic fit (Fix 6)
    step_magnitude: str = "normal"
    line_search_scales: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    # Fix 3 -- latent-advection predictor.
    latent_predictor: bool = False
    # Cap on the per-point predictor displacement (normalised units). Where
    # ||grad_x SDF|| is small the 1/||grad||^2 ODE factor explodes; without this
    # cap a single predictor step can fling a point off-surface and diverge.
    predictor_max_step: float = 0.1
    # Fix 4b -- tangent-projected Laplacian smoothing (needs source faces).
    tangent_laplacian: bool = False
    tangent_laplacian_alpha: float = 0.5
    tangent_laplacian_iters: int = 1
    # Fix 5 -- adaptive latent step-sizing.
    adaptive_steps: bool = False
    adaptive_tol: Optional[float] = None
    adaptive_estimator: str = "richardson"  # "richardson" | "residual"
    adaptive_max_depth: int = 4

    def __post_init__(self):
        if self.step_magnitude not in ("normal", "newton", "line_search"):
            raise ValueError(
                f"step_magnitude must be 'normal', 'newton' or 'line_search', "
                f"got {self.step_magnitude!r}"
            )
        if self.adaptive_estimator not in ("richardson", "residual"):
            raise ValueError(
                f"adaptive_estimator must be 'richardson' or 'residual', "
                f"got {self.adaptive_estimator!r}"
            )
        if self.n_corrector_iters < 1:
            raise ValueError("n_corrector_iters must be >= 1")


@dataclass
class _Diagnostics:
    """Lightweight record of how the integrator behaved (see Fix 5)."""

    n_advance_calls: int = 0
    n_decoder_evals: int = 0
    final_residual_max: float = float("nan")
    # (t0, t1, error) for steps accepted at the subdivision floor without
    # meeting tolerance -- candidate high-curvature / topology-mismatch sites.
    struggled_intervals: list = field(default_factory=list)


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


def _sdf_step_eval(model, points, latent, surface_idx, dz=None):
    """Single-surface SDF evaluation returning everything stepping needs.

    Stays entirely on the model's device (no CPU round-trip), and -- when
    ``dz`` is given (Fix 3) -- also returns the directional latent derivative
    ``dSDF/dz . dz`` per point. That directional derivative is obtained with
    one extra scalar leaf (``alpha``) per point: ``latent(alpha) = z + alpha*dz``
    and ``d SDF / d alpha`` at ``alpha = 0`` equals ``dSDF/dz . dz``. This is far
    cheaper than the full ``(B, D_lat)`` latent Jacobian and is computed in the
    *same* backward pass as the spatial gradient.

    Args:
    - model (nn.Module): the SDF decoder.
    - points (torch.Tensor): query points (B, 3) on the model device.
    - latent (torch.Tensor): latent vector (D_lat,) or (1, D_lat) / (B, D_lat).
    - surface_idx (int): which decoder output to use.
    - dz (torch.Tensor, optional): latent increment (D_lat,); enables Fix 3.

    Returns:
    - grad_pos (torch.Tensor): spatial gradient d SDF / d x, shape (B, 3).
    - sdf (torch.Tensor): signed distance for the surface, shape (B,).
    - latent_dir_deriv (torch.Tensor or None): dSDF/dz . dz per point, (B,).
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
    grad_inputs = [pos]

    if dz is not None:
        dz_t = dz.to(device=device, dtype=dtype).reshape(-1)
        alpha = torch.zeros(B, 1, device=device, dtype=dtype, requires_grad=True)
        # latent(alpha) = z + alpha * dz   (broadcast (B,1) * (D,) -> (B,D))
        latent_in = latent.detach() + alpha * dz_t
        grad_inputs.append(alpha)
    else:
        latent_in = latent.detach()

    p = torch.cat([latent_in, pos], dim=1)

    was_training = model.training
    model.eval()
    sdf_values = model(p)  # (B, Ns)
    assert_finite(sdf_values, "SDF values")
    y = sdf_values[:, surface_idx]  # (B,)
    grads = torch.autograd.grad(y.sum(), grad_inputs, create_graph=False, retain_graph=False)
    model.train(was_training)

    grad_pos = grads[0].detach()
    assert_finite(grad_pos, "Spatial SDF gradient")
    sdf = y.detach()
    latent_dir_deriv = None
    if dz is not None:
        latent_dir_deriv = grads[1].detach().reshape(-1)
        assert_finite(latent_dir_deriv, "Latent directional derivative")
    return grad_pos, sdf, latent_dir_deriv


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
    grad_norm = torch.norm(grad_pos, dim=1, keepdim=True)  # (B, 1)
    flat_mask = (grad_norm < EPS).squeeze(-1)  # (B,)
    safe_norm = grad_norm.clamp_min(EPS)
    unit = grad_pos / safe_norm
    unit[flat_mask] = 0.0
    return unit, grad_norm.squeeze(-1), flat_mask


def _line_search_scale(model, points, latent, surface_idx, unit, sdf0, scales):
    """Fix 6 -- batched line-search magnitude.

    Evaluates the SDF at several candidate step scalings ``x - alpha*SDF*n_hat``
    in a single forward-only pass, fits the signed ``SDF(alpha)`` with a
    least-squares parabola, and returns the per-point ``alpha`` of the root
    nearest ``alpha = 1`` (clamped to ``[0, 3]``). The candidate ``alpha = 0`` is
    free -- its SDF is ``sdf0``, already known.

    Returns the per-point step *scale* ``alpha`` (shape (B,)); the caller forms
    the step as ``unit * (alpha * sdf0)``.
    """
    scales = tuple(float(s) for s in scales)
    device = points.device
    dtype = points.dtype

    # Gather SDF samples at each candidate scale.
    samples = []
    eval_scales = []
    for s in scales:
        if abs(s) < EPS:
            samples.append(sdf0)
        else:
            cand = points - (s * sdf0).unsqueeze(1) * unit
            samples.append(_sdf_only(model, cand, latent, surface_idx))
        eval_scales.append(s)
    S = torch.stack(samples, dim=1)  # (B, n_scales)

    # Least-squares quadratic fit q(a) = c2 a^2 + c1 a + c0 over the scales.
    a = torch.tensor(eval_scales, device=device, dtype=dtype)  # (n,)
    V = torch.stack([a * a, a, torch.ones_like(a)], dim=1)  # (n, 3)
    # Pseudo-inverse of the (shared) design matrix -> (3, n).
    pinv = torch.linalg.pinv(V)
    coeffs = (pinv @ S.T).T  # (B, 3): [c2, c1, c0]
    c2, c1, c0 = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    # Linear root (used where the quadratic term is negligible / degenerate).
    c1_ok = c1.abs() > EPS
    safe_c1 = torch.where(c1_ok, c1, torch.ones_like(c1))
    lin_root = torch.where(c1_ok, -c0 / safe_c1, torch.ones_like(c0))

    disc = c1 * c1 - 4.0 * c2 * c0
    has_real = (disc >= 0) & (c2.abs() > EPS)
    sqrt_disc = torch.sqrt(disc.clamp_min(0.0))
    denom = torch.where(has_real, 2.0 * c2, torch.ones_like(c2))
    root_a = (-c1 + sqrt_disc) / denom
    root_b = (-c1 - sqrt_disc) / denom
    # Pick the quadratic root nearest alpha = 1.
    pick_a = (root_a - 1.0).abs() <= (root_b - 1.0).abs()
    quad_root = torch.where(pick_a, root_a, root_b)

    alpha = torch.where(has_real, quad_root, lin_root)
    alpha = torch.nan_to_num(alpha, nan=1.0, posinf=3.0, neginf=0.0)
    alpha = alpha.clamp(0.0, 3.0)
    return alpha


def _project_once(model, latent, points, surface_idx, config, diag=None):
    """One projection (corrector) step. GPU-resident.

    Returns ``(new_points, sdf)`` where ``sdf`` is the *signed* SDF at the
    pre-step points (its absolute value is the off-surface residual).
    """
    grad_pos, sdf, _ = _sdf_step_eval(model, points, latent, surface_idx)
    if diag is not None:
        diag.n_decoder_evals += 1
    unit, grad_norm, flat_mask = _unit_gradient(grad_pos)

    if config.step_magnitude == "normal":
        scale = sdf  # x <- x - SDF * n_hat
    elif config.step_magnitude == "newton":
        # True first-order Newton root step: x <- x - SDF * grad / ||grad||^2.
        # Equals the normal step only when ||grad|| == 1 (Eikonal).
        scale = sdf / grad_norm.clamp_min(EPS)
    else:  # "line_search"
        alpha = _line_search_scale(
            model, points, latent, surface_idx, unit, sdf, config.line_search_scales
        )
        if diag is not None:
            diag.n_decoder_evals += sum(abs(s) >= EPS for s in config.line_search_scales)
        scale = alpha * sdf

    step = unit * scale.unsqueeze(1)
    step[flat_mask] = 0.0  # leave flat points unchanged
    assert_finite(step, "Projection step")
    new_points = points - step
    assert_finite(new_points, "Projected points")
    return new_points, sdf


def _corrector_loop(model, latent, points, surface_idx, config, diag=None):
    """Fix 1 -- iterate :func:`_project_once` until converged or capped.

    With ``n_corrector_iters == 1`` this is a single projection (baseline).
    Returns ``(points, residual_max)`` where ``residual_max`` is the largest
    ``|SDF|`` seen on the final evaluation.
    """
    residual_max = float("nan")
    for it in range(config.n_corrector_iters):
        new_points, sdf = _project_once(model, latent, points, surface_idx, config, diag)
        residual_max = sdf.abs().max().item()
        points = new_points
        if residual_max < config.corrector_tol:
            break
    return points, residual_max


def _latent_predictor_step(model, latent, points, surface_idx, dz, config, diag=None):
    """Fix 3 -- implicit-function ODE predictor.

    Moves points along ``dx = -(dSDF/dz . dz) / ||grad_x SDF||^2 * grad_x SDF``,
    the least-norm motion that keeps a point on the moving level set as the
    latent advances by ``dz``. It is applied *before* the corrector.

    The ``1/||grad_x SDF||^2`` factor is unbounded: where the (non-Eikonal)
    decoder's spatial gradient is small it produces an enormous displacement
    that flings the point off-surface and diverges. Since the predictor only
    needs to be approximately right -- the corrector re-projects afterwards --
    the per-point displacement is capped at ``config.predictor_max_step``.
    """
    grad_pos, _, latent_dir_deriv = _sdf_step_eval(model, points, latent, surface_idx, dz=dz)
    if diag is not None:
        diag.n_decoder_evals += 1
    grad_norm_sq = (grad_pos * grad_pos).sum(dim=1)  # ||grad||^2
    flat_mask = grad_norm_sq < (EPS * EPS)
    coeff = latent_dir_deriv / grad_norm_sq.clamp_min(EPS * EPS)  # (B,)
    dx = -coeff.unsqueeze(1) * grad_pos
    dx[flat_mask] = 0.0
    # Cap the per-point displacement so a small-gradient point cannot diverge.
    dx_norm = dx.norm(dim=1, keepdim=True)
    clip = (config.predictor_max_step / dx_norm.clamp_min(EPS)).clamp_max(1.0)
    dx = dx * clip
    assert_finite(dx, "Latent-advection predictor step")
    return points + dx


def build_mesh_laplacian(faces, n_points, device, dtype=torch.float32):
    """Build a row-normalised (umbrella) graph Laplacian as a torch sparse tensor.

    ``Laplacian @ x`` gives, per vertex, the mean of its neighbours' positions;
    the discrete Laplacian displacement is then ``(Laplacian @ x) - x``. Built
    once from the source-mesh connectivity and reused every step (Fix 4b).

    Args:
    - faces (np.ndarray): triangle connectivity, shape (M, 3).
    - n_points (int): number of mesh vertices.
    - device: torch device for the sparse tensor.
    - dtype: torch dtype for the values.

    Returns:
    - torch.sparse_coo_tensor: row-normalised adjacency, shape (n_points, n_points).
    """
    faces = np.asarray(faces).reshape(-1, 3).astype(np.int64)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)  # symmetric
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


def _tangent_laplacian_step(model, latent, points, surface_idx, laplacian, config, diag=None):
    """Fix 4b -- tangent-projected Laplacian smoothing.

    Redistributes points *along the surface* without the off-surface pull that
    full-3D VTK smoothing causes: the Laplacian displacement is projected onto
    the local tangent plane (its normal component removed) before being applied,
    and the result is re-projected onto the level set afterwards.
    """
    for _ in range(config.tangent_laplacian_iters):
        grad_pos, _, _ = _sdf_step_eval(model, points, latent, surface_idx)
        if diag is not None:
            diag.n_decoder_evals += 1
        unit, _, _ = _unit_gradient(grad_pos)  # surface normal at each point
        lap = torch.sparse.mm(laplacian, points) - points  # umbrella displacement
        normal_comp = (lap * unit).sum(dim=1, keepdim=True) * unit
        lap_tan = lap - normal_comp  # tangent-only displacement
        points = points + config.tangent_laplacian_alpha * lap_tan
        # Re-project onto the level set to undo any residual off-surface drift.
        points, _ = _corrector_loop(model, latent, points, surface_idx, config, diag)
    return points


# ---------------------------------------------------------------------------
# Public single-step API (kept backward compatible)
# ---------------------------------------------------------------------------


def update_positions(model, new_latent, current_points, surface_idx=0, verbose=True):
    """Single first-order closest-point projection of ``current_points``.

    This is the original baseline step ``x <- x - SDF * n_hat``. It is kept with
    its original signature and CPU-tensor return value for backward
    compatibility; the interpolation loop now uses the GPU-resident primitives
    (:func:`_project_once`, :func:`_corrector_loop`) directly.

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
    config = StepConfig()  # baseline: single "normal" projection
    new_points, _ = _project_once(model, latent, points, surface_idx, config)
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


def _advance(model, points, z_start, z_end, surface_idx, config, laplacian, diag):
    """Apply one latent increment ``z_start -> z_end`` to ``points``.

    predictor (Fix 3) -> corrector loop (Fix 1, with Fix 2/6 magnitude) ->
    tangent Laplacian (Fix 4b). Returns the advanced points (GPU tensor).
    """
    diag.n_advance_calls += 1
    z_end_t = _to_model_tensor(z_end, model)

    if config.latent_predictor:
        z_start_t = _to_model_tensor(z_start, model)
        dz = z_end_t - z_start_t
        points = _latent_predictor_step(
            model, z_start_t, points, surface_idx, dz, config, diag
        )

    points, _ = _corrector_loop(model, z_end_t, points, surface_idx, config, diag)

    if config.tangent_laplacian:
        points = _tangent_laplacian_step(
            model, z_end_t, points, surface_idx, laplacian, config, diag
        )
    return points


def _advance_adaptive(
    model, points, latent1, latent2, t0, t1, surface_idx, config, laplacian, diag, spherical, depth
):
    """Fix 5 -- adaptively subdivide the latent interval ``[t0, t1]``.

    Estimates the local error of a full ``[t0, t1]`` step; if it exceeds the
    tolerance and the subdivision depth is below ``adaptive_max_depth``, the
    interval is halved and each half advanced in turn. The depth cap is the hard
    guarantee against an infinite split loop -- a step accepted at the cap
    without meeting tolerance is recorded in ``diag.struggled_intervals``.
    """
    z0 = _latent_at(latent1, latent2, t0, spherical)
    z1 = _latent_at(latent1, latent2, t1, spherical)

    if config.adaptive_estimator == "residual":
        advanced = _advance(model, points, z0, z1, surface_idx, config, laplacian, diag)
        z1_t = _to_model_tensor(z1, model)
        residual = _sdf_only(model, advanced, z1_t, surface_idx).abs().max().item()
        diag.n_decoder_evals += 1
        if residual <= config.adaptive_tol or depth >= config.adaptive_max_depth:
            if residual > config.adaptive_tol:
                diag.struggled_intervals.append((float(t0), float(t1), float(residual)))
            return advanced
    else:  # "richardson": one full step vs two half steps
        big = _advance(model, points, z0, z1, surface_idx, config, laplacian, diag)
        tm = 0.5 * (t0 + t1)
        zm = _latent_at(latent1, latent2, tm, spherical)
        half1 = _advance(model, points, z0, zm, surface_idx, config, laplacian, diag)
        small = _advance(model, half1, zm, z1, surface_idx, config, laplacian, diag)
        error = (big - small).norm(dim=1).max().item()
        if error <= config.adaptive_tol or depth >= config.adaptive_max_depth:
            if error > config.adaptive_tol:
                diag.struggled_intervals.append((float(t0), float(t1), float(error)))
            return small  # the two-half estimate is the more accurate one

    # Subdivide and recurse.
    tm = 0.5 * (t0 + t1)
    points = _advance_adaptive(
        model, points, latent1, latent2, t0, tm, surface_idx, config, laplacian, diag,
        spherical, depth + 1,
    )
    points = _advance_adaptive(
        model, points, latent1, latent2, tm, t1, surface_idx, config, laplacian, diag,
        spherical, depth + 1,
    )
    return points


def _resolve_adaptive_tol(config, points, faces):
    """Pick a scale-relative tolerance for Fix 5 when one is not supplied."""
    if config.adaptive_tol is not None:
        return config.adaptive_tol
    if config.adaptive_estimator == "residual":
        # A small multiple of the corrector convergence tolerance.
        return 5.0 * config.corrector_tol
    # Richardson: a small fraction of the source mesh's median edge length, or
    # of the bounding-box diagonal when no connectivity is available.
    pts = points.detach().cpu().numpy()
    if faces is not None and len(faces) > 0:
        faces = np.asarray(faces).reshape(-1, 3)
        e0 = np.linalg.norm(pts[faces[:, 0]] - pts[faces[:, 1]], axis=1)
        e1 = np.linalg.norm(pts[faces[:, 1]] - pts[faces[:, 2]], axis=1)
        e2 = np.linalg.norm(pts[faces[:, 2]] - pts[faces[:, 0]], axis=1)
        scale = float(np.median(np.concatenate([e0, e1, e2])))
    else:
        scale = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    return 0.05 * scale


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
    config=None,
    faces=None,
    return_diagnostics=False,
):
    if data is None:
        raise Exception("Not implemented")
        # create function that gets the surface points for latent1 as a starting point.

    if config is None:
        config = StepConfig()

    if is_mesh:
        if not isinstance(data.mesh, pv.PolyData):
            data.mesh = pv.PolyData(data.mesh)
        add_cell_idx(data)

    device = next(model.parameters()).device
    diag = _Diagnostics()

    if is_mesh:
        # The mesh path keeps its existing VTK subdivide/smooth behaviour
        # (Fix 4a). Per-step projection is routed through the corrector loop so
        # Fix 1 / Fix 2 / Fix 6 still apply. Predictor (Fix 3), tangent Laplacian
        # (Fix 4b) and adaptive step-sizing (Fix 5) are points-path features and
        # are not used here.
        mesh_config = StepConfig(
            n_corrector_iters=config.n_corrector_iters,
            corrector_tol=config.corrector_tol,
            step_magnitude=config.step_magnitude,
            line_search_scales=config.line_search_scales,
        )
        for idx, step in enumerate(np.linspace(1 / n_steps, 1, n_steps)):
            if verbose:
                print(f"{idx + 1}/{n_steps}")
            new_latent = _to_model_tensor(
                slerp_latent(latent1, latent2, step)
                if spherical
                else linear_interp_latent(latent1, latent2, step),
                model,
            )
            points = _to_model_tensor(data.point_coords.copy(), model)
            points, residual = _corrector_loop(
                model, new_latent, points, surface_idx, mesh_config, diag
            )
            diag.final_residual_max = residual
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
        result = data
    else:
        if isinstance(data, np.ndarray):
            points = torch.as_tensor(data, dtype=torch.float).to(device)
        elif torch.is_tensor(data):
            points = data.to(device)
        else:
            raise Exception(f"Unknown data type: {type(data)}")

        if config.tangent_laplacian and faces is None:
            raise ValueError(
                "tangent_laplacian=True (Fix 4b) requires the source-mesh "
                "`faces` connectivity; pass `faces=` to interpolate_points."
            )

        laplacian = None
        if config.tangent_laplacian:
            laplacian = build_mesh_laplacian(
                faces, points.shape[0], device=device, dtype=points.dtype
            )

        if config.adaptive_steps:
            config.adaptive_tol = _resolve_adaptive_tol(config, points, faces)
            t_prev = 0.0
            for t in np.linspace(1 / n_steps, 1, n_steps):
                if verbose:
                    print(f"adaptive interval ({t_prev:.4f}, {t:.4f}]")
                points = _advance_adaptive(
                    model, points, latent1, latent2, t_prev, float(t), surface_idx,
                    config, laplacian, diag, spherical, depth=0,
                )
                t_prev = float(t)
        else:
            t_prev = 0.0
            for idx, t in enumerate(np.linspace(1 / n_steps, 1, n_steps)):
                if verbose:
                    print(f"{idx + 1}/{n_steps}")
                z_start = _latent_at(latent1, latent2, t_prev, spherical)
                z_end = _latent_at(latent1, latent2, float(t), spherical)
                points = _advance(
                    model, points, z_start, z_end, surface_idx, config, laplacian, diag
                )
                t_prev = float(t)

        # Final off-surface residual at the destination latent.
        z_b = _to_model_tensor(latent2, model)
        diag.final_residual_max = _sdf_only(model, points, z_b, surface_idx).abs().max().item()
        diag.n_decoder_evals += 1
        result = points.detach().cpu().numpy()

    if return_diagnostics:
        return result, diag
    return result


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
    n_corrector_iters=1,
    corrector_tol=1e-4,
    step_magnitude="normal",
    line_search_scales=(0.0, 0.5, 1.0, 1.5, 2.0),
    latent_predictor=False,
    predictor_max_step=0.1,
    faces=None,
    tangent_laplacian=False,
    tangent_laplacian_alpha=0.5,
    tangent_laplacian_iters=1,
    adaptive_steps=False,
    adaptive_tol=None,
    adaptive_estimator="richardson",
    adaptive_max_depth=4,
    return_diagnostics=False,
):
    """Flow ``points1`` (vertices on shape A) onto the surface of shape B.

    The default call reproduces the original single-projection method exactly.
    The keyword-only arguments toggle the numerical fixes from the
    ``NSM_MESH_INTERPOLATION_IMPROVEMENTS`` plan; each is independently gated so
    a config can be tested alone and then composed.

    Args:
    - model (nn.Module): the SDF decoder.
    - latent1, latent2 (np.ndarray): the source and target latent vectors.
    - n_steps (int): number of latent increments (the NFE knob).
    - points1 (np.ndarray): source vertices, on shape A's surface, shape (N, 3).
    - surface_idx (int): which decoder output / surface to interpolate.
    - verbose (bool): print per-step progress.
    - spherical (bool): slerp the latent (vs linear interpolation).
    - n_corrector_iters (int): Fix 1 -- projections per latent step (1 = baseline).
    - corrector_tol (float): Fix 1 -- ``max|SDF|`` early-exit tolerance.
    - step_magnitude (str): ``"normal"`` (baseline), ``"newton"`` (Fix 2) or
      ``"line_search"`` (Fix 6).
    - line_search_scales (tuple): Fix 6 -- candidate step scalings.
    - latent_predictor (bool): Fix 3 -- enable the latent-advection predictor.
    - predictor_max_step (float): Fix 3 -- cap on the per-point predictor
      displacement (normalised units); guards against small-gradient blow-up.
    - faces (np.ndarray): source-mesh triangle connectivity (M, 3); required by
      Fix 4b and used to scale the Fix 5 Richardson tolerance.
    - tangent_laplacian (bool): Fix 4b -- enable tangent-projected smoothing.
    - tangent_laplacian_alpha (float): Fix 4b -- smoothing step size.
    - tangent_laplacian_iters (int): Fix 4b -- smoothing iterations per step.
    - adaptive_steps (bool): Fix 5 -- adaptively subdivide latent steps.
    - adaptive_tol (float): Fix 5 -- error tolerance (auto, scale-relative, if None).
    - adaptive_estimator (str): Fix 5 -- ``"richardson"`` or ``"residual"``.
    - adaptive_max_depth (int): Fix 5 -- maximum subdivision depth.
    - return_diagnostics (bool): also return a diagnostics object.

    Returns:
    - np.ndarray: the warped points (N, 3). If ``return_diagnostics`` is True,
      returns ``(points, diagnostics)``.
    """
    config = StepConfig(
        n_corrector_iters=n_corrector_iters,
        corrector_tol=corrector_tol,
        step_magnitude=step_magnitude,
        line_search_scales=tuple(line_search_scales),
        latent_predictor=latent_predictor,
        predictor_max_step=predictor_max_step,
        tangent_laplacian=tangent_laplacian,
        tangent_laplacian_alpha=tangent_laplacian_alpha,
        tangent_laplacian_iters=tangent_laplacian_iters,
        adaptive_steps=adaptive_steps,
        adaptive_tol=adaptive_tol,
        adaptive_estimator=adaptive_estimator,
        adaptive_max_depth=adaptive_max_depth,
    )
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
        config=config,
        faces=faces,
        return_diagnostics=return_diagnostics,
    )


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
    *,
    n_corrector_iters=1,
    corrector_tol=1e-4,
    step_magnitude="normal",
    line_search_scales=(0.0, 0.5, 1.0, 1.5, 2.0),
    return_diagnostics=False,
):
    """Flow a full mesh from shape A onto shape B (the VTK-smoothed path).

    This is the ``is_mesh`` path: it keeps the existing per-step VTK
    subdivide/smooth behaviour (Fix 4a) and additionally routes projection
    through the corrector loop, so Fix 1 / Fix 2 / Fix 6 are available. The
    points-path-only fixes (3, 4b, 5) are not exposed here.

    Args mirror :func:`interpolate_points` plus the existing mesh-path knobs
    (``max_edge_len``, ``adaptive``, ``smooth``, ``smooth_type``).
    """
    config = StepConfig(
        n_corrector_iters=n_corrector_iters,
        corrector_tol=corrector_tol,
        step_magnitude=step_magnitude,
        line_search_scales=tuple(line_search_scales),
    )
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
        config=config,
        return_diagnostics=return_diagnostics,
    )
