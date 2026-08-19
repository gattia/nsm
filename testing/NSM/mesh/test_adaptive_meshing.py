"""
Tests for adaptive meshing functions in NSM/mesh/main.py.

Covers: _dilate6, coarse_bounds_from_sign_change, create_grid_samples_in_bounds,
        crop_sdf_to_narrow_band, and create_mesh_adaptive (integration with mocked decoder).
"""

from unittest.mock import MagicMock

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


# ---------------------------------------------------------------------------
# _dilate6
# ---------------------------------------------------------------------------
class TestDilate6:
    def test_single_voxel_center(self):
        """A single True voxel in the center should expand to its 6 face-neighbours."""
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True
        out = _dilate6(mask)
        assert out[2, 2, 2]  # original
        # 6 face neighbours
        assert out[1, 2, 2] and out[3, 2, 2]
        assert out[2, 1, 2] and out[2, 3, 2]
        assert out[2, 2, 1] and out[2, 2, 3]
        # total should be 7 (center + 6 neighbours)
        assert out.sum() == 7

    def test_corner_voxel(self):
        """A corner voxel should only dilate to the 3 available face-neighbours."""
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[0, 0, 0] = True
        out = _dilate6(mask)
        # 1 original + 3 neighbours (no negative-index neighbours)
        assert out.sum() == 4
        assert out[0, 0, 0] and out[1, 0, 0] and out[0, 1, 0] and out[0, 0, 1]

    def test_does_not_mutate_input(self):
        mask = np.zeros((3, 3, 3), dtype=bool)
        mask[1, 1, 1] = True
        original = mask.copy()
        _dilate6(mask)
        np.testing.assert_array_equal(mask, original)


# ---------------------------------------------------------------------------
# coarse_bounds_from_sign_change
# ---------------------------------------------------------------------------
class TestCoarseBoundsFromSignChange:
    def _make_sphere_sdf(self, n, radius=0.3, center=(0.5, 0.5, 0.5)):
        """Create SDF of a sphere on a [0,1]^3 grid, returned in (Z,Y,X) layout."""
        lin = np.linspace(0, 1, n)
        # meshgrid with indexing="ij" gives (Z, Y, X) when we pass z, y, x
        Z, Y, X = np.meshgrid(lin, lin, lin, indexing="ij")
        sdf = np.sqrt((X - center[2]) ** 2 + (Y - center[1]) ** 2 + (Z - center[0]) ** 2) - radius
        return sdf  # (Z, Y, X)

    def test_sphere_bounds_enclose_surface(self):
        """Bounds should enclose the sphere surface."""
        n = 32
        sdf_zyx = self._make_sphere_sdf(n, radius=0.3, center=(0.5, 0.5, 0.5))
        spacing = 1.0 / (n - 1)
        origin = (0.0, 0.0, 0.0)  # world origin (ox, oy, oz)

        result = coarse_bounds_from_sign_change(
            sdf_zyx, origin, spacing, tau_voxels=1.0, dilate_cells=1
        )
        assert result is not None
        bounds_min, bounds_max = result

        # The sphere surface is at radius 0.3 from center (0.5, 0.5, 0.5)
        # So surface spans ~[0.2, 0.8] in each axis
        # Bounds should contain that (with some dilation margin)
        for i in range(3):
            assert bounds_min[i] < 0.25, f"bounds_min[{i}]={bounds_min[i]} should be < 0.25"
            assert bounds_max[i] > 0.75, f"bounds_max[{i}]={bounds_max[i]} should be > 0.75"

    def test_asymmetric_center_axis_ordering(self):
        """Sphere at asymmetric center — verifies ZYX→XYZ mapping is correct.

        If axes are swapped, bounds_min/max will reflect the wrong center coordinates.
        Center (z=0.2, y=0.5, x=0.8) should produce bounds centered near x=0.8, y=0.5, z=0.2.
        """
        n = 32
        # center arg is (z, y, x) because _make_sphere_sdf uses that convention
        sdf_zyx = self._make_sphere_sdf(n, radius=0.15, center=(0.2, 0.5, 0.8))
        spacing = 1.0 / (n - 1)

        result = coarse_bounds_from_sign_change(
            sdf_zyx, (0, 0, 0), spacing, tau_voxels=0.5, dilate_cells=1
        )
        assert result is not None
        bounds_min, bounds_max = result

        # Returned bounds are (x, y, z) in world coords
        mid_x = (bounds_min[0] + bounds_max[0]) / 2
        mid_y = (bounds_min[1] + bounds_max[1]) / 2
        mid_z = (bounds_min[2] + bounds_max[2]) / 2

        # Should be near the sphere center: x≈0.8, y≈0.5, z≈0.2
        assert abs(mid_x - 0.8) < 0.1, f"mid_x={mid_x}, expected ~0.8"
        assert abs(mid_y - 0.5) < 0.1, f"mid_y={mid_y}, expected ~0.5"
        assert abs(mid_z - 0.2) < 0.1, f"mid_z={mid_z}, expected ~0.2"

    def test_all_positive_returns_none(self):
        """If SDF is entirely positive (no surface), should return None."""
        sdf = np.ones((10, 10, 10)) * 5.0
        result = coarse_bounds_from_sign_change(sdf, (0, 0, 0), 0.1, tau_voxels=0.5)
        assert result is None

    def test_all_negative_returns_none(self):
        """If SDF is entirely negative (no surface crossing), should return None."""
        sdf = np.ones((10, 10, 10)) * -5.0
        result = coarse_bounds_from_sign_change(sdf, (0, 0, 0), 0.1, tau_voxels=0.5)
        assert result is None

    def test_torch_tensor_matches_numpy(self):
        """Torch tensor input should produce identical bounds to numpy array."""
        n = 16
        sdf_zyx = self._make_sphere_sdf(n)
        spacing = 1.0 / (n - 1)

        result_np = coarse_bounds_from_sign_change(sdf_zyx, (0, 0, 0), spacing)
        sdf_tensor = torch.tensor(sdf_zyx, dtype=torch.float32)
        result_torch = coarse_bounds_from_sign_change(sdf_tensor, (0, 0, 0), spacing)

        assert result_np is not None and result_torch is not None
        np.testing.assert_allclose(result_torch[0], result_np[0], atol=1e-10)
        np.testing.assert_allclose(result_torch[1], result_np[1], atol=1e-10)

    def test_no_dilation_tighter_bounds(self):
        """With dilate_cells=0 bounds should be tighter than with dilation."""
        n = 32
        sdf_zyx = self._make_sphere_sdf(n, radius=0.2)
        spacing = 1.0 / (n - 1)
        origin = (0.0, 0.0, 0.0)

        result_tight = coarse_bounds_from_sign_change(
            sdf_zyx, origin, spacing, dilate_cells=0, tau_voxels=0.0
        )
        result_wide = coarse_bounds_from_sign_change(
            sdf_zyx, origin, spacing, dilate_cells=3, tau_voxels=1.0
        )
        assert result_tight is not None and result_wide is not None
        min_t, max_t = result_tight
        min_w, max_w = result_wide
        # Wide bounds should be >= tight bounds in every axis
        for i in range(3):
            assert min_w[i] <= min_t[i]
            assert max_w[i] >= max_t[i]

    def test_origin_offset_shifts_bounds(self):
        """Non-zero origin should shift bounds by the origin amount."""
        n = 16
        sdf_zyx = self._make_sphere_sdf(n, radius=0.3, center=(0.5, 0.5, 0.5))
        spacing = 1.0 / (n - 1)

        result_zero = coarse_bounds_from_sign_change(sdf_zyx, (0, 0, 0), spacing)
        result_shifted = coarse_bounds_from_sign_change(sdf_zyx, (10, 20, 30), spacing)

        assert result_zero is not None and result_shifted is not None
        min0, max0 = result_zero
        min_s, max_s = result_shifted
        np.testing.assert_allclose(min_s - min0, [10, 20, 30], atol=1e-10)
        np.testing.assert_allclose(max_s - max0, [10, 20, 30], atol=1e-10)


# ---------------------------------------------------------------------------
# create_grid_samples_in_bounds
# ---------------------------------------------------------------------------
class TestCreateGridSamplesInBounds:
    def test_min_dim_enforced(self):
        """Even for tiny bounds, grid dims should be >= min_dim."""
        bounds_min = np.array([0.0, 0.0, 0.0])
        bounds_max = np.array([0.001, 0.001, 0.001])
        spacing = 0.01
        _, dims, _ = create_grid_samples_in_bounds(bounds_min, bounds_max, spacing, min_dim=64)
        for d in dims:
            assert d >= 64

    def test_padding_expands_beyond_bounds(self):
        """Origin should be less than bounds_min when padding is applied."""
        bounds_min = np.array([0.5, 0.5, 0.5])
        bounds_max = np.array([1.5, 1.5, 1.5])
        spacing = 0.01
        _, _, origin = create_grid_samples_in_bounds(bounds_min, bounds_max, spacing, padding=0.2)
        for i in range(3):
            assert origin[i] < bounds_min[i]

    def test_samples_cover_and_bound_padded_region(self):
        """Grid should cover bounds_max and not extend wildly beyond padded region.

        Use a large extent relative to spacing so min_dim doesn't inflate the grid.
        """
        bounds_min = np.array([-5.0, -5.0, -5.0])
        bounds_max = np.array([5.0, 5.0, 5.0])
        spacing = 0.05
        padding = 0.5
        samples, dims, origin = create_grid_samples_in_bounds(
            bounds_min, bounds_max, spacing, padding=padding
        )
        for i in range(3):
            # Grid should not start before padded_min
            assert samples[:, i].min() >= bounds_min[i] - padding - spacing
            # Grid should not extend far beyond padded_max
            assert samples[:, i].max() <= bounds_max[i] + padding + spacing
            # Grid must actually reach past bounds_max (coverage)
            assert (
                samples[:, i].max() >= bounds_max[i]
            ), f"axis {i}: grid max {samples[:, i].max()} doesn't cover bounds_max {bounds_max[i]}"

    def test_min_pad_voxels_fine(self):
        """min_pad_voxels_fine should override padding when it's larger."""
        bounds_min = np.array([0.0, 0.0, 0.0])
        bounds_max = np.array([1.0, 1.0, 1.0])
        spacing = 0.1
        # padding=0.01 is tiny, but min_pad_voxels_fine=5 means 0.5 world padding
        _, _, origin_small_pad = create_grid_samples_in_bounds(
            bounds_min, bounds_max, spacing, padding=0.01, min_pad_voxels_fine=5
        )
        # With min_pad_voxels_fine=5 and spacing=0.1, effective pad = 0.5
        # Origin should be at -0.5 approximately
        for i in range(3):
            assert origin_small_pad[i] < bounds_min[i] - 0.4

    def test_z_varies_fastest(self):
        """Samples should follow Z-fastest ordering, matching create_grid_samples."""
        bounds_min = np.array([0.0, 0.0, 0.0])
        bounds_max = np.array([1.0, 1.0, 1.0])
        spacing = 0.5
        samples, (nx, ny, nz), origin = create_grid_samples_in_bounds(
            bounds_min, bounds_max, spacing, padding=0.0, min_dim=1, min_pad_voxels_fine=0
        )
        # First nz samples should have constant x and y, incrementing z
        for i in range(min(nz, samples.shape[0])):
            assert samples[i, 0].item() == pytest.approx(origin[0])
            assert samples[i, 1].item() == pytest.approx(origin[1])
            assert samples[i, 2].item() == pytest.approx(origin[2] + i * spacing)


# ---------------------------------------------------------------------------
# crop_sdf_to_narrow_band
# ---------------------------------------------------------------------------
class TestCropSdfToNarrowBand:
    def test_crop_reduces_volume(self):
        """Cropping a sphere SDF should produce a smaller volume."""
        n = 64
        lin = np.linspace(-1, 1, n)
        X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
        sdf = np.sqrt(X**2 + Y**2 + Z**2) - 0.3  # sphere radius 0.3
        voxel_size = 2.0 / (n - 1)
        origin = (-1, -1, -1)

        sub_sdf, crop_origin = crop_sdf_to_narrow_band(
            sdf, origin, voxel_size, band_width=3.0, pad_voxels=2
        )
        # Should be smaller than original
        assert sub_sdf.size < sdf.size
        # Should still contain the surface (min <= 0, max >= 0)
        assert sub_sdf.min() <= 0
        assert sub_sdf.max() >= 0

    def test_all_far_returns_original(self):
        """If no voxels are near the surface, return original."""
        sdf = np.ones((10, 10, 10)) * 100.0
        origin = (0, 0, 0)
        sub_sdf, crop_origin = crop_sdf_to_narrow_band(sdf, origin, 0.1, band_width=3.0)
        np.testing.assert_array_equal(sub_sdf, sdf)
        assert crop_origin == origin

    def test_crop_origin_shifts(self):
        """Crop origin should shift when the band doesn't start at index 0."""
        n = 32
        sdf = np.ones((n, n, n)) * 10.0
        # Put the surface in one corner only
        sdf[20:25, 20:25, 20:25] = 0.0
        origin = (0, 0, 0)
        voxel_size = 0.1

        sub_sdf, crop_origin = crop_sdf_to_narrow_band(
            sdf, origin, voxel_size, band_width=1.0, pad_voxels=1
        )
        # Origin should have shifted (not at 0,0,0)
        assert any(c > 0 for c in crop_origin)


# ---------------------------------------------------------------------------
# create_grid_samples (existing function — verify Z-fastest ordering)
# ---------------------------------------------------------------------------
class TestCreateGridSamples:
    def test_z_varies_fastest(self):
        """First few samples should have same x,y but incrementing z."""
        n = 4
        voxel_size = 1.0 / (n - 1)
        samples = create_grid_samples(n, (0, 0, 0), voxel_size)
        # samples[0] = (0,0,0), samples[1] = (0,0,dz), samples[2] = (0,0,2*dz), ...
        for i in range(n):
            assert samples[i, 0].item() == pytest.approx(0.0)
            assert samples[i, 1].item() == pytest.approx(0.0)
            assert samples[i, 2].item() == pytest.approx(i * voxel_size)

    def test_total_samples(self):
        n = 8
        samples = create_grid_samples(n, (-1, -1, -1), 2.0 / (n - 1))
        assert samples.shape == (n**3, 3)


# ---------------------------------------------------------------------------
# create_mesh_adaptive (integration with mocked decoder)
# ---------------------------------------------------------------------------
class TestCreateMeshAdaptive:
    def _make_sphere_decoder(self, radius=0.3):
        """Return a mock decoder whose forward pass computes SDF of a sphere at origin."""
        decoder = MagicMock()
        decoder.eval = MagicMock()

        def forward(pts):
            # pts: (N, 3) — compute sphere SDF
            dist = torch.sqrt((pts**2).sum(dim=-1, keepdim=True))
            return dist - radius

        decoder.side_effect = forward
        decoder.__call__ = forward
        return decoder

    def _make_constant_decoder(self, value=5.0):
        """Return a mock decoder that always returns a constant SDF value."""
        decoder = MagicMock()
        decoder.eval = MagicMock()

        def forward(pts):
            return torch.ones(pts.shape[0], 1) * value

        decoder.side_effect = forward
        decoder.__call__ = forward
        return decoder

    def _make_multi_sphere_decoder(self, radii, centers):
        """Return a mock decoder that outputs multiple SDF channels (one per object)."""
        decoder = MagicMock()
        decoder.eval = MagicMock()
        n_objects = len(radii)

        def forward(pts):
            # pts: (N, 3) or (N, 3+latent_dim) — we only use last 3 cols
            xyz = pts[:, -3:]
            out = torch.zeros(xyz.shape[0], n_objects)
            for i, (r, c) in enumerate(zip(radii, centers)):
                c_t = torch.tensor(c, dtype=xyz.dtype)
                out[:, i] = torch.sqrt(((xyz - c_t) ** 2).sum(dim=-1)) - r
            return out

        decoder.side_effect = forward
        decoder.__call__ = forward
        return decoder

    def test_adaptive_produces_mesh(self):
        """Adaptive meshing with a sphere decoder should produce a valid mesh."""
        decoder = self._make_sphere_decoder(radius=0.3)
        latent = torch.zeros(1, 64)

        mesh = create_mesh_adaptive(
            decoder,
            latent,
            n_pts_per_axis=64,
            n_pts_coarse=16,
            search_bounds=(-1.0, 1.0),
            batch_size=50_000,
            scale_to_original_mesh=False,
            objects=1,
            device="cpu",
            use_vtk=True,
            verbose=False,
        )
        assert mesh is not None
        assert mesh.n_points > 0

    def test_fallback_when_no_surface(self):
        """When SDF is all-positive, should fall back to create_mesh (and also return None there)."""
        decoder = self._make_constant_decoder(5.0)
        latent = torch.zeros(1, 64)

        # With fallback, it calls create_mesh which also won't find a surface → returns None
        mesh = create_mesh_adaptive(
            decoder,
            latent,
            n_pts_per_axis=16,
            n_pts_coarse=8,
            search_bounds=(-1.0, 1.0),
            batch_size=50_000,
            scale_to_original_mesh=False,
            objects=1,
            device="cpu",
            verbose=False,
            fallback_to_original=True,
        )
        assert mesh is None

    def test_no_fallback_returns_none(self):
        """With fallback disabled and no surface, should return None."""
        decoder = self._make_constant_decoder(5.0)
        latent = torch.zeros(1, 64)

        result = create_mesh_adaptive(
            decoder,
            latent,
            n_pts_per_axis=16,
            n_pts_coarse=8,
            search_bounds=(-1.0, 1.0),
            batch_size=50_000,
            scale_to_original_mesh=False,
            objects=1,
            device="cpu",
            verbose=False,
            fallback_to_original=False,
        )
        assert result is None

    def test_adaptive_fewer_points_than_full(self):
        """Adaptive should evaluate fewer dense points than a full grid for a small sphere."""
        decoder = self._make_sphere_decoder(radius=0.2)
        latent = torch.zeros(1, 64)

        n_full = 128
        # Track how many points are evaluated in pass 2 by counting calls
        call_counts = {"total_pts": 0}
        original_call = decoder.__call__

        def counting_call(pts):
            call_counts["total_pts"] += pts.shape[0]
            return original_call(pts)

        decoder.__call__ = counting_call

        mesh = create_mesh_adaptive(
            decoder,
            latent,
            n_pts_per_axis=n_full,
            n_pts_coarse=16,
            search_bounds=(-1.0, 1.0),
            batch_size=100_000,
            scale_to_original_mesh=False,
            objects=1,
            device="cpu",
            verbose=False,
        )
        assert mesh is not None
        # Total points evaluated (coarse + dense) should be less than full grid
        assert call_counts["total_pts"] < n_full**3

    def test_multi_object_produces_list(self):
        """Multi-object decoder should produce a list of meshes, one per object."""
        decoder = self._make_multi_sphere_decoder(
            radii=[0.2, 0.15],
            centers=[[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]],
        )
        latent = torch.zeros(1, 64)

        meshes = create_mesh_adaptive(
            decoder,
            latent,
            n_pts_per_axis=64,
            n_pts_coarse=16,
            search_bounds=(-1.0, 1.0),
            batch_size=100_000,
            scale_to_original_mesh=False,
            objects=2,
            device="cpu",
            verbose=False,
        )
        assert isinstance(meshes, list)
        assert len(meshes) == 2
        for m in meshes:
            assert m is not None
            assert m.n_points > 0
