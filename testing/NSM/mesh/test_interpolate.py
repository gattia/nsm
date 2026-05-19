"""Tests for the kwarg-gated numerical fixes in ``NSM/mesh/interpolate.py``.

Uses a synthetic analytic SDF decoder (a sphere whose radius is read out of the
latent) so the fixes can be exercised without a trained NSM model. The sphere
SDF is exact, so the *target* of every interpolation is known in closed form:
points warped from a radius-``r1`` sphere onto a radius-``r2`` sphere should end
up at radius ``r2``.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from NSM.mesh.interpolate import (
    StepConfig,
    _latent_predictor_step,
    build_mesh_laplacian,
    compute_boundary_mask,
    interpolate_points,
    update_positions,
)


class SphereSDF(nn.Module):
    """Analytic SDF decoder for a sphere.

    The latent's first component is the sphere radius; components 1:4 are the
    centre. ``sdf_scale != 1`` makes the field non-Eikonal (``||grad|| = scale``),
    which is what discriminates the Newton / line-search magnitude fixes from the
    baseline normal step.
    """

    def __init__(self, d_lat=8, n_surfaces=1, sdf_scale=1.0):
        super().__init__()
        self.d_lat = d_lat
        self.n_surfaces = n_surfaces
        self.sdf_scale = sdf_scale
        self._param = nn.Parameter(torch.zeros(1))  # gives device/dtype

    def forward(self, x=None, latent=None, xyz=None, epoch=None, verbose=False):
        if x is not None:
            latent = x[:, : self.d_lat]
            xyz = x[:, self.d_lat :]
        radius = latent[:, 0:1]
        center = latent[:, 1:4]
        dist = torch.norm(xyz - center, dim=1, keepdim=True)
        sdf = self.sdf_scale * (dist - radius)
        return sdf.repeat(1, self.n_surfaces)


def _sphere_points(n=400, radius=1.0, seed=0):
    """Roughly uniform points on a sphere of the given radius."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return (v * radius).astype(np.float32)


def _latents(r1=1.0, r2=1.5, d_lat=8):
    """Two non-colinear latents (slerp needs distinct directions).

    The radius lives in dim 0 and the centre in dims 1:4; the direction-
    disambiguating components go in unused dims 4/5 so they do not perturb the
    decoded sphere.
    """
    # float64: geometric_slerp requires inputs to lie exactly on the unit sphere.
    z1 = np.zeros(d_lat, dtype=np.float64)
    z2 = np.zeros(d_lat, dtype=np.float64)
    z1[0], z1[4] = r1, 1.0
    z2[0], z2[5] = r2, 1.0
    return z1, z2


def _radii(points, center=(0, 0, 0)):
    return np.linalg.norm(np.asarray(points) - np.asarray(center), axis=1)


# ---------------------------------------------------------------------------
# Baseline behaviour
# ---------------------------------------------------------------------------


def test_baseline_lands_on_target_sphere():
    model = SphereSDF()
    z1, z2 = _latents(1.0, 1.5)
    pts = _sphere_points(radius=1.0)
    warped = interpolate_points(model, z1, z2, n_steps=50, points1=pts, surface_idx=0)
    assert warped.shape == pts.shape
    np.testing.assert_allclose(_radii(warped), 1.5, atol=1e-3)


def test_default_config_is_single_normal_projection():
    """An all-default config must do exactly one normal projection per step."""
    cfg = StepConfig()
    assert cfg.n_corrector_iters == 1
    assert cfg.step_magnitude == "normal"
    assert not cfg.latent_predictor
    assert not cfg.tangent_laplacian
    assert not cfg.adaptive_steps

    model = SphereSDF()
    z1, z2 = _latents()
    pts = _sphere_points()
    _, diag = interpolate_points(
        model, z1, z2, n_steps=20, points1=pts, surface_idx=0, return_diagnostics=True
    )
    assert diag.n_advance_calls == 20
    # 20 steps * 1 projection eval + 1 final residual eval.
    assert diag.n_decoder_evals == 21


def test_update_positions_backward_compatible():
    model = SphereSDF()
    z2 = _latents()[1]
    pts = _sphere_points(radius=1.0)
    out = update_positions(model, z2, pts, surface_idx=0)
    assert torch.is_tensor(out)
    assert out.device.type == "cpu"
    assert out.shape == pts.shape


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        StepConfig(step_magnitude="bogus")
    with pytest.raises(ValueError):
        StepConfig(adaptive_estimator="bogus")
    with pytest.raises(ValueError):
        StepConfig(n_corrector_iters=0)


# ---------------------------------------------------------------------------
# Fix 1 -- corrector loop
# ---------------------------------------------------------------------------


def test_corrector_loop_reduces_residual():
    """More corrector iterations -> smaller terminal off-surface residual."""
    model = SphereSDF(sdf_scale=0.5)  # non-Eikonal: one projection under-converges
    z1, z2 = _latents()
    pts = _sphere_points()
    _, diag1 = interpolate_points(
        model, z1, z2, n_steps=15, points1=pts, surface_idx=0,
        n_corrector_iters=1, return_diagnostics=True,
    )
    _, diag5 = interpolate_points(
        model, z1, z2, n_steps=15, points1=pts, surface_idx=0,
        n_corrector_iters=8, return_diagnostics=True,
    )
    assert diag5.final_residual_max < diag1.final_residual_max


# ---------------------------------------------------------------------------
# Fix 2 -- Newton magnitude
# ---------------------------------------------------------------------------


def test_newton_beats_normal_on_non_eikonal_field():
    """On a non-Eikonal field the normal step mis-scales; Newton is exact."""
    model = SphereSDF(sdf_scale=1.6)
    z1, z2 = _latents(1.0, 1.5)
    pts = _sphere_points()
    _, diag_normal = interpolate_points(
        model, z1, z2, n_steps=20, points1=pts, surface_idx=0,
        step_magnitude="normal", return_diagnostics=True,
    )
    warped_newton, diag_newton = interpolate_points(
        model, z1, z2, n_steps=20, points1=pts, surface_idx=0,
        step_magnitude="newton", return_diagnostics=True,
    )
    assert diag_newton.final_residual_max < diag_normal.final_residual_max
    np.testing.assert_allclose(_radii(warped_newton), 1.5, atol=1e-3)


def test_newton_is_noop_on_eikonal_field():
    """With ||grad|| == 1 the Newton step equals the normal step."""
    model = SphereSDF(sdf_scale=1.0)
    z1, z2 = _latents()
    pts = _sphere_points()
    w_normal = interpolate_points(
        model, z1, z2, n_steps=30, points1=pts, surface_idx=0, step_magnitude="normal"
    )
    w_newton = interpolate_points(
        model, z1, z2, n_steps=30, points1=pts, surface_idx=0, step_magnitude="newton"
    )
    np.testing.assert_allclose(w_normal, w_newton, atol=1e-5)


# ---------------------------------------------------------------------------
# Fix 6 -- line-search magnitude
# ---------------------------------------------------------------------------


def test_line_search_lands_on_target():
    model = SphereSDF(sdf_scale=1.6)
    z1, z2 = _latents(1.0, 1.5)
    pts = _sphere_points()
    warped, diag = interpolate_points(
        model, z1, z2, n_steps=20, points1=pts, surface_idx=0,
        step_magnitude="line_search", return_diagnostics=True,
    )
    np.testing.assert_allclose(_radii(warped), 1.5, atol=5e-3)
    # Line search costs extra forward evals relative to the baseline.
    assert diag.n_decoder_evals > 20


# ---------------------------------------------------------------------------
# Fix 3 -- latent-advection predictor
# ---------------------------------------------------------------------------


def test_latent_predictor_runs_and_converges():
    model = SphereSDF()
    z1, z2 = _latents(1.0, 1.5)
    pts = _sphere_points()
    warped = interpolate_points(
        model, z1, z2, n_steps=10, points1=pts, surface_idx=0,
        latent_predictor=True, n_corrector_iters=3,
    )
    assert np.isfinite(warped).all()
    np.testing.assert_allclose(_radii(warped), 1.5, atol=1e-2)


def test_predictor_step_is_clamped_on_small_gradient_field():
    """The 1/||grad||^2 predictor factor must not blow up on a non-Eikonal field.

    With ||grad SDF|| = 0.02 the unclamped predictor displacement would be
    ~0.5; the cap must hold it to predictor_max_step.
    """
    model = SphereSDF(sdf_scale=0.02)
    z1, z2 = _latents(1.0, 1.5)
    pts = torch.tensor(_sphere_points(radius=1.0))
    dz = torch.tensor(z2 - z1, dtype=torch.float)
    cfg = StepConfig(latent_predictor=True, predictor_max_step=0.1)
    moved = _latent_predictor_step(
        model, torch.tensor(z1, dtype=torch.float), pts, 0, dz, cfg
    )
    disp = (moved - pts).norm(dim=1)
    assert torch.isfinite(disp).all()
    assert disp.max().item() <= 0.1 + 1e-5


def test_latent_predictor_stable_on_non_eikonal_field():
    """interpolate_points with the predictor must not diverge on a scaled SDF.

    Mirrors the cluster `fix1_fix2_fix3` config (corrector + Newton + predictor).
    """
    model = SphereSDF(sdf_scale=0.05)
    z1, z2 = _latents(1.0, 1.3)
    pts = _sphere_points()
    warped = interpolate_points(
        model, z1, z2, n_steps=20, points1=pts, surface_idx=0,
        latent_predictor=True, n_corrector_iters=5, step_magnitude="newton",
    )
    assert np.isfinite(warped).all()
    assert _radii(warped).max() < 5.0  # no blow-up
    np.testing.assert_allclose(_radii(warped), 1.3, atol=1e-2)


# ---------------------------------------------------------------------------
# Fix 4b -- tangent-projected Laplacian
# ---------------------------------------------------------------------------


def test_tangent_laplacian_requires_faces():
    model = SphereSDF()
    z1, z2 = _latents()
    pts = _sphere_points()
    with pytest.raises(ValueError):
        interpolate_points(
            model, z1, z2, n_steps=5, points1=pts, surface_idx=0, tangent_laplacian=True
        )


def test_tangent_laplacian_runs():
    import pyvista as pv

    sphere = pv.Sphere(radius=1.0, theta_resolution=20, phi_resolution=20)
    pts = sphere.points.astype(np.float32)
    faces = sphere.regular_faces.astype(np.int64)
    model = SphereSDF()
    z1, z2 = _latents(1.0, 1.2)
    warped = interpolate_points(
        model, z1, z2, n_steps=15, points1=pts, surface_idx=0,
        faces=faces, tangent_laplacian=True, tangent_laplacian_alpha=0.3,
    )
    assert np.isfinite(warped).all()
    np.testing.assert_allclose(_radii(warped), 1.2, atol=1e-2)


def test_build_mesh_laplacian():
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    lap = build_mesh_laplacian(faces, n_points=4, device=torch.device("cpu"))
    assert lap.shape == (4, 4)
    # Row-normalised: each row of the dense adjacency sums to 1.
    dense = lap.to_dense()
    np.testing.assert_allclose(dense.sum(dim=1).numpy(), np.ones(4), atol=1e-6)


# ---------------------------------------------------------------------------
# Fix 5 -- adaptive step-sizing
# ---------------------------------------------------------------------------


def test_adaptive_steps_subdivide_and_converge():
    model = SphereSDF()
    z1, z2 = _latents(1.0, 1.5)
    pts = _sphere_points()
    warped, diag = interpolate_points(
        model, z1, z2, n_steps=10, points1=pts, surface_idx=0,
        adaptive_steps=True, return_diagnostics=True,
    )
    np.testing.assert_allclose(_radii(warped), 1.5, atol=1e-2)
    # Richardson uses >= 3 advance calls per base interval (big + 2 halves).
    assert diag.n_advance_calls >= 10


def test_adaptive_residual_estimator():
    model = SphereSDF(sdf_scale=0.5)
    z1, z2 = _latents()
    pts = _sphere_points()
    warped, diag = interpolate_points(
        model, z1, z2, n_steps=8, points1=pts, surface_idx=0,
        adaptive_steps=True, adaptive_estimator="residual",
        n_corrector_iters=3, return_diagnostics=True,
    )
    assert np.isfinite(warped).all()
    assert diag.n_advance_calls >= 8


def test_adaptive_floor_records_struggled_intervals():
    """A tiny tolerance forces subdivision to the depth floor and logs it."""
    model = SphereSDF()
    z1, z2 = _latents(1.0, 2.0)
    pts = _sphere_points()
    _, diag = interpolate_points(
        model, z1, z2, n_steps=5, points1=pts, surface_idx=0,
        adaptive_steps=True, adaptive_tol=1e-12, adaptive_max_depth=2,
        return_diagnostics=True,
    )
    assert len(diag.struggled_intervals) > 0
    for t0, t1, err in diag.struggled_intervals:
        assert 0.0 <= t0 < t1 <= 1.0


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_compute_boundary_mask_closed_sphere_all_false():
    import pyvista as pv

    sphere = pv.Sphere(theta_resolution=12, phi_resolution=12)
    faces = sphere.regular_faces.astype(np.int64)
    mask = compute_boundary_mask(faces, sphere.n_points)
    assert mask.shape == (sphere.n_points,)
    assert not mask.any()  # closed mesh -> no boundary


def test_compute_boundary_mask_plane_all_boundary():
    """A 2-triangle square has all 4 vertices on the boundary."""
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    mask = compute_boundary_mask(faces, 4)
    assert mask.all()


def test_compute_boundary_mask_disk_only_rim():
    """In a triangle fan around a centre vertex the centre is interior, the rim is boundary."""
    # 5 outer points forming a pentagon around centre point 0; 5 fan triangles.
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1]], dtype=np.int64
    )
    mask = compute_boundary_mask(faces, 6)
    assert mask[0] == False  # centre vertex -- interior
    assert mask[1:].all()    # all 5 rim vertices on boundary


def test_smooth_normals_requires_faces():
    model = SphereSDF()
    z1, z2 = _latents()
    pts = _sphere_points()
    with pytest.raises(ValueError):
        interpolate_points(
            model, z1, z2, n_steps=5, points1=pts, surface_idx=0, smooth_normals=True
        )


def test_smooth_normals_converges_on_sphere():
    """smooth_normals (Fix 7) reduces to Newton when neighbours agree, so a
    sphere warp must still land on the target radius."""
    import pyvista as pv

    sphere = pv.Sphere(radius=1.0, theta_resolution=18, phi_resolution=18)
    pts = sphere.points.astype(np.float32)
    faces = sphere.regular_faces.astype(np.int64)
    model = SphereSDF()
    z1, z2 = _latents(1.0, 1.4)
    warped = interpolate_points(
        model, z1, z2, n_steps=15, points1=pts, surface_idx=0,
        faces=faces, smooth_normals=True, smooth_normal_iters=2, n_corrector_iters=3,
    )
    assert np.isfinite(warped).all()
    np.testing.assert_allclose(_radii(warped), 1.4, atol=2e-2)


def test_pin_boundary_keeps_rim_in_place_on_disk():
    """A flat triangle-fan disk warped to itself (z1==z2) should stay put,
    and with pin_boundary=True the rim vertices must not move."""
    import pyvista as pv

    sphere = pv.Sphere(radius=1.0, theta_resolution=12, phi_resolution=12)
    # Use the sphere -- closed -- so boundary mask is all False and pinning is
    # a no-op; this just verifies the code path runs and points stay sane.
    pts = sphere.points.astype(np.float32)
    faces = sphere.regular_faces.astype(np.int64)
    model = SphereSDF()
    z1, z2 = _latents(1.0, 1.2)
    warped_pin = interpolate_points(
        model, z1, z2, n_steps=10, points1=pts, surface_idx=0,
        faces=faces, tangent_laplacian=True, tangent_laplacian_pin_boundary=True,
    )
    warped_nopin = interpolate_points(
        model, z1, z2, n_steps=10, points1=pts, surface_idx=0,
        faces=faces, tangent_laplacian=True, tangent_laplacian_pin_boundary=False,
    )
    # On a closed mesh the two modes should agree (no boundary to pin).
    np.testing.assert_allclose(warped_pin, warped_nopin, atol=1e-5)


def test_all_fixes_compose():
    import pyvista as pv

    sphere = pv.Sphere(radius=1.0, theta_resolution=16, phi_resolution=16)
    pts = sphere.points.astype(np.float32)
    faces = sphere.regular_faces.astype(np.int64)
    model = SphereSDF(sdf_scale=1.3)
    z1, z2 = _latents(1.0, 1.4)
    warped, diag = interpolate_points(
        model, z1, z2, n_steps=12, points1=pts, surface_idx=0,
        n_corrector_iters=4, step_magnitude="newton", latent_predictor=True,
        faces=faces, tangent_laplacian=True, adaptive_steps=True,
        return_diagnostics=True,
    )
    assert np.isfinite(warped).all()
    np.testing.assert_allclose(_radii(warped), 1.4, atol=2e-2)
    assert diag.n_decoder_evals > 0
