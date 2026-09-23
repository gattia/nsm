"""
The pieces of adaptive meshing in ``NSM/mesh/main.py``: a coarse pass finds where the
surface is, and a fine grid is built only there.
"""

import numpy as np
import pytest
import torch

from NSM.mesh.main import (
    _dilate6,
    coarse_bounds_from_sign_change,
    create_grid_samples,
    create_grid_samples_in_bounds,
    create_mesh_adaptive,
    crop_sdf_to_narrow_band,
)


def test_dilate6_grows_a_voxel_to_its_face_neighbours():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True
    out = _dilate6(mask)
    assert out.sum() == 7 and out[1, 2, 2] and out[2, 3, 2] and out[2, 2, 1]
    assert mask.sum() == 1, "the input was mutated"

    corner = np.zeros((3, 3, 3), dtype=bool)
    corner[0, 0, 0] = True
    assert _dilate6(corner).sum() == 4


def _sphere_sdf_zyx(n, radius=0.3, center_zyx=(0.5, 0.5, 0.5)):
    """A sphere's SDF on a [0, 1]^3 grid, in the (Z, Y, X) layout the function takes."""
    lin = np.linspace(0, 1, n)
    z, y, x = np.meshgrid(lin, lin, lin, indexing="ij")
    cz, cy, cx = center_zyx
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - radius


class TestCoarseBounds:
    def test_the_bounds_enclose_the_surface_in_xyz_order(self):
        """An off-centre sphere catches a swapped ZYX-to-XYZ mapping."""
        n = 32
        spacing = 1.0 / (n - 1)
        sdf = _sphere_sdf_zyx(n, radius=0.15, center_zyx=(0.2, 0.5, 0.8))
        lo, hi = coarse_bounds_from_sign_change(sdf, (0, 0, 0), spacing, tau_voxels=0.5)
        np.testing.assert_allclose((lo + hi) / 2, [0.8, 0.5, 0.2], atol=0.1)

        centred = _sphere_sdf_zyx(n)
        lo, hi = coarse_bounds_from_sign_change(centred, (0, 0, 0), spacing, dilate_cells=1)
        assert np.all(lo < 0.25) and np.all(hi > 0.75)
        shifted = coarse_bounds_from_sign_change(centred, (10, 20, 30), spacing, dilate_cells=1)
        np.testing.assert_allclose(shifted[0] - lo, [10, 20, 30], atol=1e-10)

        tight = coarse_bounds_from_sign_change(centred, (0, 0, 0), spacing, 0.0, dilate_cells=0)
        wide = coarse_bounds_from_sign_change(centred, (0, 0, 0), spacing, 1.0, dilate_cells=3)
        assert np.all(wide[0] <= tight[0]) and np.all(wide[1] >= tight[1])

    def test_no_sign_change_is_none(self):
        for value in (5.0, -5.0):
            assert coarse_bounds_from_sign_change(np.full((10,) * 3, value), (0, 0, 0), 0.1) is None


def test_grids_are_z_fastest_padded_and_at_least_min_dim():
    n = 4
    samples = create_grid_samples(n, (0, 0, 0), 1.0 / (n - 1))
    assert samples.shape == (n**3, 3)
    np.testing.assert_allclose(samples[:n, 2], np.arange(n) / (n - 1))
    assert torch.all(samples[:n, :2] == 0)

    samples, (nx, ny, nz), origin = create_grid_samples_in_bounds(
        np.zeros(3), np.ones(3), 0.5, padding=0.0, min_dim=1, min_pad_voxels_fine=0
    )
    np.testing.assert_allclose(samples[:nz, 2], origin[2] + 0.5 * np.arange(nz))

    lo, hi = np.full(3, -5.0), np.full(3, 5.0)
    samples, _, _ = create_grid_samples_in_bounds(lo, hi, 0.05, padding=0.5)
    low, high = samples.min(0).values.numpy(), samples.max(0).values.numpy()
    assert np.all(low >= lo - 0.55) and np.all(hi <= high) and np.all(high <= hi + 0.55)

    _, dims, _ = create_grid_samples_in_bounds(np.zeros(3), np.full(3, 0.001), 0.01, min_dim=64)
    assert min(dims) >= 64
    _, _, origin = create_grid_samples_in_bounds(
        np.zeros(3), np.ones(3), 0.1, padding=0.01, min_pad_voxels_fine=5
    )
    assert np.all(np.asarray(origin) < -0.4), "min_pad_voxels_fine should override padding"


def test_cropping_to_the_narrow_band():
    lin = np.linspace(-1, 1, 64)
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
    sdf = np.sqrt(x**2 + y**2 + z**2) - 0.3
    sub, _ = crop_sdf_to_narrow_band(sdf, (-1, -1, -1), 2.0 / 63, band_width=3.0, pad_voxels=2)
    assert sub.size < sdf.size and sub.min() <= 0 <= sub.max()

    far = np.full((10,) * 3, 100.0)
    sub, origin = crop_sdf_to_narrow_band(far, (0, 0, 0), 0.1, band_width=3.0)
    assert sub is far or np.array_equal(sub, far)
    assert origin == (0, 0, 0)

    corner = np.full((32,) * 3, 10.0)
    corner[20:25, 20:25, 20:25] = 0.0
    _, origin = crop_sdf_to_narrow_band(corner, (0, 0, 0), 0.1, band_width=1.0, pad_voxels=1)
    assert any(c > 0 for c in origin)


class _Spheres(torch.nn.Module):
    """One SDF column per sphere; only the last three input columns are read."""

    def __init__(self, radii, centers):
        super().__init__()
        self.radii, self.centers = radii, torch.tensor(centers, dtype=torch.float32)
        self.points_evaluated = 0

    def forward(self, pts):
        self.points_evaluated += pts.shape[0]
        xyz = pts[:, -3:]
        return torch.stack(
            [torch.linalg.norm(xyz - c, dim=1) - r for r, c in zip(self.radii, self.centers)],
            dim=1,
        )


def test_create_mesh_adaptive_meshes_every_object_for_less_than_a_full_grid():
    common = dict(search_bounds=(-1.0, 1.0), scale_to_original_mesh=False, device="cpu")
    single = _Spheres([0.2], [[0.0, 0.0, 0.0]])
    mesh = create_mesh_adaptive(
        single, torch.zeros(1, 64), 128, n_pts_coarse=16, objects=1, **common
    )
    assert mesh.n_points > 0
    assert single.points_evaluated < 128**3

    pair = _Spheres([0.2, 0.15], [[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]])
    meshes = create_mesh_adaptive(
        pair, torch.zeros(1, 64), 64, n_pts_coarse=16, objects=2, **common
    )
    assert len(meshes) == 2 and all(m.n_points > 0 for m in meshes)


@pytest.mark.parametrize("fallback", [True, False])
def test_no_surface_is_none_with_or_without_the_dense_fallback(fallback):
    no_surface = _Spheres([-1.0], [[0.0, 0.0, 0.0]])
    assert (
        create_mesh_adaptive(
            no_surface,
            torch.zeros(1, 64),
            16,
            n_pts_coarse=8,
            objects=1,
            scale_to_original_mesh=False,
            device="cpu",
            fallback_to_original=fallback,
        )
        is None
    )
