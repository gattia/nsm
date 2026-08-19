"""Tests for ``NSM/mesh/interpolate.py``.

Uses a synthetic analytic SDF decoder (a sphere whose radius is read out of
the latent) so the interpolation can be exercised without a trained NSM model.
The sphere SDF is exact, so the *target* of every interpolation is known in
closed form: points warped from a radius-``r1`` sphere onto a radius-``r2``
sphere should end up at radius ``r2``.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from NSM.mesh.interpolate import (
    build_mesh_laplacian,
    compute_feature_mask,
    interpolate_points,
    update_positions,
)


class SphereSDF(nn.Module):
    """Analytic SDF decoder for a sphere.

    The latent's first component is the sphere radius; components 1:4 are the
    centre. ``sdf_scale != 1`` makes the field non-Eikonal (``||grad|| = scale``),
    which is what distinguishes the Newton magnitude from a plain unit-normal
    step.
    """

    def __init__(self, d_lat=8, n_surfaces=1, sdf_scale=1.0):
        super().__init__()
        self.d_lat = d_lat
        self.n_surfaces = n_surfaces
        self.sdf_scale = sdf_scale
        self._param = nn.Parameter(torch.zeros(1))

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
    """Two non-colinear latents (slerp needs distinct directions)."""
    z1 = np.zeros(d_lat, dtype=np.float64)
    z2 = np.zeros(d_lat, dtype=np.float64)
    z1[0], z1[4] = r1, 1.0
    z2[0], z2[5] = r2, 1.0
    return z1, z2


def _radii(points, center=(0, 0, 0)):
    return np.linalg.norm(np.asarray(points) - np.asarray(center), axis=1)


# ---------------------------------------------------------------------------
# Baseline behaviour (Newton magnitude is unconditional)
# ---------------------------------------------------------------------------


def test_baseline_lands_on_target_sphere():
    """Default config lands on the target sphere within Newton's convergence."""
    model = SphereSDF()
    z1, z2 = _latents(1.0, 1.5)
    pts = _sphere_points(radius=1.0)
    warped = interpolate_points(model, z1, z2, n_steps=50, points1=pts, surface_idx=0)
    assert warped.shape == pts.shape
    np.testing.assert_allclose(_radii(warped), 1.5, atol=1e-3)


def test_baseline_lands_on_target_non_eikonal_field():
    """Newton magnitude is exact even when the SDF is not unit-gradient."""
    model = SphereSDF(sdf_scale=1.6)
    z1, z2 = _latents(1.0, 1.5)
    pts = _sphere_points()
    warped = interpolate_points(model, z1, z2, n_steps=20, points1=pts, surface_idx=0)
    np.testing.assert_allclose(_radii(warped), 1.5, atol=1e-3)


def test_update_positions_backward_compatible():
    model = SphereSDF()
    z2 = _latents()[1]
    pts = _sphere_points(radius=1.0)
    out = update_positions(model, z2, pts, surface_idx=0)
    assert torch.is_tensor(out)
    assert out.device.type == "cpu"
    assert out.shape == pts.shape


# ---------------------------------------------------------------------------
# Tangent Laplacian smoothing (opt-in via tangent_laplacian=True)
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
        model,
        z1,
        z2,
        n_steps=15,
        points1=pts,
        surface_idx=0,
        faces=faces,
        tangent_laplacian=True,
        tangent_laplacian_alpha=0.3,
    )
    assert np.isfinite(warped).all()
    np.testing.assert_allclose(_radii(warped), 1.2, atol=1e-2)


def test_build_mesh_laplacian():
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    lap = build_mesh_laplacian(faces, n_points=4, device=torch.device("cpu"))
    assert lap.shape == (4, 4)
    dense = lap.to_dense()
    np.testing.assert_allclose(dense.sum(dim=1).numpy(), np.ones(4), atol=1e-6)


# ---------------------------------------------------------------------------
# Feature-mask detection (the pin used by tangent Laplacian)
# ---------------------------------------------------------------------------


def test_compute_feature_mask_sphere_smooth_no_features():
    """A finely tessellated sphere has only gently-varying dihedral angles."""
    import pyvista as pv

    sphere = pv.Sphere(theta_resolution=32, phi_resolution=32)
    pts = np.asarray(sphere.points, dtype=np.float64)
    faces = sphere.regular_faces.astype(np.int64)
    mask = compute_feature_mask(faces, pts, dihedral_threshold_deg=60.0)
    assert not mask.any()


def test_compute_feature_mask_tent_detects_ridge():
    """Two triangles sharing an edge but folded 90 degrees: every vertex on
    the shared ridge is a feature vertex."""
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 1.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
    mask = compute_feature_mask(faces, points, dihedral_threshold_deg=60.0)
    # Ridge vertices (0, 1) flagged; isolated tips (2, 3) flagged via the
    # boundary edges they sit on -- all four end up on at least one feature/
    # boundary edge.
    assert mask.all()


def test_compute_feature_mask_finds_seam_on_thin_disk():
    """A thin closed disk (two-sided) has a high-dihedral seam at the rim."""
    import pyvista as pv

    cyl = pv.Cylinder(radius=1.0, height=0.05, resolution=24).triangulate()
    pts = np.asarray(cyl.points, dtype=np.float64)
    faces = cyl.regular_faces.astype(np.int64)
    mask = compute_feature_mask(faces, pts, dihedral_threshold_deg=60.0)
    assert mask.any(), "thin disk should have a high-dihedral seam"
