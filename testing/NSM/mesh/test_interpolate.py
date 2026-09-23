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

    def forward(self, x=None, latent=None, xyz=None, epoch=None):
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


@pytest.mark.parametrize("sdf_scale", [1.0, 1.6], ids=["eikonal", "non-eikonal"])
def test_points_land_on_the_target_sphere(sdf_scale):
    """
    Fails if ``interpolate_points`` steps by the unit normal instead of the Newton step
    ``x - sdf * grad / |grad|^2``, or ``update_positions`` stops returning a CPU tensor of
    the input's shape.

    Only the non-Eikonal case (``|grad| = 1.6``) separates the two steps. Measured, a
    unit-normal step misses the target radius by 2.3e-3 there, against ``atol=1e-3``, and by
    1.7e-7 in the Eikonal case.
    """
    z1, z2 = _latents(1.0, 1.5)
    pts = _sphere_points(radius=1.0)
    warped = interpolate_points(
        SphereSDF(sdf_scale=sdf_scale), z1, z2, n_steps=20, points1=pts, surface_idx=0
    )
    assert warped.shape == pts.shape
    np.testing.assert_allclose(_radii(warped), 1.5, atol=1e-3)

    out = update_positions(SphereSDF(), z2, pts, surface_idx=0)
    assert torch.is_tensor(out) and out.device.type == "cpu" and out.shape == pts.shape


def test_tangent_laplacian_smoothing_needs_faces_and_keeps_the_target():
    """
    Fails if ``interpolate_points(tangent_laplacian=True)`` runs without ``faces``, its
    smoothing pulls points off the target sphere, or ``build_mesh_laplacian``'s rows stop
    summing to 1.

    The row-sum assert is the only check on the normalisation. Measured, an unnormalised
    adjacency still lands every point within 4.4e-7 of the target.
    """
    import pyvista as pv

    z1, z2 = _latents(1.0, 1.2)
    with pytest.raises(ValueError):
        interpolate_points(
            SphereSDF(), z1, z2, n_steps=5, points1=_sphere_points(), tangent_laplacian=True
        )

    sphere = pv.Sphere(radius=1.0, theta_resolution=20, phi_resolution=20)
    warped = interpolate_points(
        SphereSDF(),
        z1,
        z2,
        n_steps=15,
        points1=sphere.points.astype(np.float32),
        surface_idx=0,
        faces=sphere.regular_faces.astype(np.int64),
        tangent_laplacian=True,
        tangent_laplacian_alpha=0.3,
    )
    np.testing.assert_allclose(_radii(warped), 1.2, atol=1e-2)

    lap = build_mesh_laplacian(np.array([[0, 1, 2], [1, 2, 3]]), n_points=4, device="cpu")
    np.testing.assert_allclose(lap.to_dense().sum(dim=1).numpy(), np.ones(4), atol=1e-6)


def test_the_feature_mask_flags_sharp_edges_only():
    """
    Fails if ``compute_feature_mask`` flags a vertex of a smooth closed sphere, or misses the
    vertices on an open boundary edge.

    The tent and the disk are flagged by their boundary edges alone: the tent is two open
    triangles, and ``pv.Cylinder``'s caps do not share its rim vertices (96 boundary edges).
    Measured, turning the dihedral-angle test off still passes, so it is not pinned here.
    """
    import pyvista as pv

    sphere = pv.Sphere(theta_resolution=32, phi_resolution=32)
    assert not compute_feature_mask(sphere.regular_faces, np.asarray(sphere.points), 60.0).any()

    tent = np.array([[0.0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1]])
    assert compute_feature_mask(np.array([[0, 1, 2], [0, 1, 3]]), tent, 60.0).all()

    disk = pv.Cylinder(radius=1.0, height=0.05, resolution=24).triangulate()
    assert compute_feature_mask(disk.regular_faces, np.asarray(disk.points), 60.0).any()
