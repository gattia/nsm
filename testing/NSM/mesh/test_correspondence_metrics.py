"""
Tests for NSM/mesh/correspondence_metrics.py.

All tests use synthetic geometry (pyvista primitives or hand-crafted meshes)
and are deterministic — no random seeds or external data required.
"""

import numpy as np
import pytest
import pyvista as pv

from NSM.mesh.correspondence_metrics import (
    assd,
    directed_distance_percentiles,
    foldover_count,
    forward_backward_disagreement,
    off_surface_error,
    roundtrip_distance,
    score_correspondence,
    self_intersection_count,
    triangle_health,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sphere(
    radius: float = 1.0, theta_resolution: int = 20, phi_resolution: int = 20
) -> pv.PolyData:
    """Return a triangulated pyvista sphere."""
    return pv.Sphere(
        radius=radius, theta_resolution=theta_resolution, phi_resolution=phi_resolution
    )


def _make_plane_mesh() -> pv.PolyData:
    """Return a simple 2-triangle flat plane mesh."""
    # 4 vertices forming a unit square in the XY-plane
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([3, 0, 1, 2, 3, 0, 2, 3])  # two triangles
    return pv.PolyData(points, faces)


def _translate_mesh(mesh: pv.PolyData, offset: np.ndarray) -> pv.PolyData:
    """Return a copy of mesh shifted by offset."""
    new_pts = mesh.points + offset
    return pv.PolyData(new_pts, mesh.faces)


def _make_flipped_mesh(source_mesh: pv.PolyData, flip_indices) -> np.ndarray:
    """Return warped_points where specified triangles are flipped.

    For each triangle index in flip_indices the winding order is reversed by
    swapping two vertices in the warped point array.  This is achieved by
    moving those vertices to degenerate positions that produce a negative
    normal dot product.
    """
    warped = source_mesh.points.copy()
    faces = source_mesh.faces.reshape(-1, 4)[:, 1:]

    for tri_idx in flip_indices:
        i0, i1, i2 = faces[tri_idx]
        # Swap v1 and v2 in the warped array — reverses winding
        warped[[i1, i2]] = warped[[i2, i1]]

    return warped


# ---------------------------------------------------------------------------
# assd
# ---------------------------------------------------------------------------


class TestAssd:
    def test_identical_meshes_zero(self):
        """ASSD of a mesh against itself must be 0."""
        sphere = _make_sphere()
        result = assd(sphere, sphere)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_translated_mesh_reflects_offset(self):
        """ASSD with a translated copy should grow with translation distance.

        For a unit sphere translated by 5 along x the vertex-based ASSD will
        be somewhat less than 5 (nearest-vertex distances range from ~3 to 5
        depending on which pole is closest).  We just verify the result is in
        a sensible range and is clearly non-zero.
        """
        sphere = _make_sphere(radius=1.0, theta_resolution=30, phi_resolution=30)
        offset = np.array([5.0, 0.0, 0.0])
        shifted = _translate_mesh(sphere, offset)
        result = assd(sphere, shifted)
        # Mean nearest-vertex distance should be significantly above 0 and below 6
        assert result > 2.0
        assert result < 6.0

    def test_scaled_mesh_has_larger_assd(self):
        """A scaled-up copy should give a larger ASSD than an unscaled copy."""
        sphere = _make_sphere(radius=1.0)
        big_sphere = _make_sphere(radius=2.0)
        result = assd(sphere, big_sphere)
        assert result > 0.5

    def test_returns_float(self):
        sphere = _make_sphere()
        result = assd(sphere, sphere)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# directed_distance_percentiles
# ---------------------------------------------------------------------------


class TestDirectedDistancePercentiles:
    def _make_cloud(self, n=50):
        return np.zeros((n, 3))

    def test_identical_clouds_zero(self):
        """Distances from a cloud to itself should all be zero."""
        pts = np.random.default_rng(0).standard_normal((100, 3))
        result = directed_distance_percentiles(pts, pts)
        assert result["min"] == pytest.approx(0.0, abs=1e-10)
        assert result["max"] == pytest.approx(0.0, abs=1e-10)

    def test_offset_cloud_correct_distance(self):
        """All distances from cloud A to cloud B (translated by 3) should be ~3."""
        pts_a = np.zeros((50, 3))
        pts_b = np.zeros((50, 3))
        pts_b[:, 0] = 3.0
        result = directed_distance_percentiles(pts_a, pts_b)
        assert result["min"] == pytest.approx(3.0, abs=1e-8)
        assert result["max"] == pytest.approx(3.0, abs=1e-8)
        assert result["mean"] == pytest.approx(3.0, abs=1e-8)

    def test_all_keys_present(self):
        pts = np.zeros((10, 3))
        result = directed_distance_percentiles(pts, pts)
        for key in ("min", "p25", "p50", "mean", "p75", "p95", "max"):
            assert key in result


# ---------------------------------------------------------------------------
# off_surface_error
# ---------------------------------------------------------------------------


class TestOffSurfaceError:
    def test_all_zeros_returns_zero(self):
        sdf_vals = np.zeros(100)
        result = off_surface_error(sdf_vals)
        assert result["min"] == pytest.approx(0.0)
        assert result["max"] == pytest.approx(0.0)
        assert result["mean"] == pytest.approx(0.0)
        assert result["rms"] == pytest.approx(0.0)

    def test_known_values(self):
        """Test with known SDF values: abs are [1, 2, 3, 4]."""
        sdf_vals = np.array([-1.0, 2.0, -3.0, 4.0])
        result = off_surface_error(sdf_vals)
        assert result["min"] == pytest.approx(1.0)
        assert result["max"] == pytest.approx(4.0)
        assert result["mean"] == pytest.approx(2.5)
        expected_rms = float(np.sqrt(np.mean([1.0, 4.0, 9.0, 16.0])))
        assert result["rms"] == pytest.approx(expected_rms)

    def test_rms_key_present(self):
        result = off_surface_error(np.array([1.0, 2.0, 3.0]))
        assert "rms" in result

    def test_all_keys_present(self):
        result = off_surface_error(np.ones(10))
        for key in ("min", "p25", "p50", "mean", "p75", "p95", "max", "rms"):
            assert key in result


# ---------------------------------------------------------------------------
# triangle_health
# ---------------------------------------------------------------------------


class TestTriangleHealth:
    def test_regular_mesh_low_edge_ratio(self):
        """A well-formed sphere mesh should have a low edge aspect ratio."""
        sphere = _make_sphere(theta_resolution=30, phi_resolution=30)
        result = triangle_health(sphere)
        # Well-formed mesh: edge ratio mean should be modest (not pathological)
        assert result["edge_ratio_mean"] < 10.0
        assert result["degenerate_count"] == 0

    def test_plane_mesh_zero_degenerate(self):
        """A clean 2-triangle plane mesh should have zero degenerate triangles."""
        plane = _make_plane_mesh()
        result = triangle_health(plane)
        assert result["degenerate_count"] == 0

    def test_all_keys_present(self):
        sphere = _make_sphere()
        result = triangle_health(sphere)
        expected_keys = [
            "edge_length_mean",
            "edge_length_std",
            "edge_length_min",
            "edge_length_max",
            "area_mean",
            "area_std",
            "area_min",
            "area_max",
            "edge_ratio_mean",
            "edge_ratio_p95",
            "edge_ratio_max",
            "degenerate_count",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_degenerate_triangle_detected(self):
        """Mesh with a collapsed triangle should report at least one degenerate.

        Two vertices coincident within the same triangle: v0 and v1 both at
        the origin, so triangle (0, 1, 2) has a zero-length edge.
        """
        points = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=float,
        )
        faces = np.array([3, 0, 1, 2, 3, 0, 2, 3])
        mesh_collapsed = pv.PolyData(points, faces)
        result = triangle_health(mesh_collapsed)
        assert result["degenerate_count"] >= 1

    def test_edge_lengths_positive(self):
        sphere = _make_sphere()
        result = triangle_health(sphere)
        assert result["edge_length_min"] > 0.0


# ---------------------------------------------------------------------------
# self_intersection_count
# ---------------------------------------------------------------------------


class TestSelfIntersectionCount:
    def test_clean_sphere_no_intersections(self):
        """A clean sphere mesh should have zero self-intersections."""
        sphere = _make_sphere(theta_resolution=15, phi_resolution=15)
        count = self_intersection_count(sphere)
        assert count == 0

    def test_plane_no_intersections(self):
        """A 2-triangle plane should have zero self-intersections."""
        plane = _make_plane_mesh()
        count = self_intersection_count(plane)
        assert count == 0

    def test_exceeds_max_triangles_returns_none(self):
        """Meshes exceeding max_triangles should return None with a warning."""
        sphere = _make_sphere(theta_resolution=30, phi_resolution=30)
        with pytest.warns(RuntimeWarning, match="max_triangles"):
            result = self_intersection_count(sphere, max_triangles=10)
        assert result is None

    def test_self_intersecting_mesh(self):
        """A figure-8 / bowtie mesh should have at least one intersecting pair."""
        # Two triangles that cross each other in 3-D (non-adjacent)
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
                [0.5, 0.5, -1.0],
                [1.5, 0.5, -1.0],
                [1.0, 0.5, 1.0],
            ],
            dtype=float,
        )
        # Triangle 0: (0,1,2) in XY-plane, Triangle 1: (3,4,5) cutting through it
        faces = np.array([3, 0, 1, 2, 3, 3, 4, 5])
        mesh = pv.PolyData(points, faces)
        count = self_intersection_count(mesh, broadphase=True)
        assert count >= 1

    def test_returns_int(self):
        plane = _make_plane_mesh()
        result = self_intersection_count(plane)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# foldover_count
# ---------------------------------------------------------------------------


class TestFoldoverCount:
    def test_no_foldover_identity(self):
        """Warped points identical to source should have zero flips."""
        sphere = _make_sphere(theta_resolution=20, phi_resolution=20)
        warped = sphere.points.copy()
        result = foldover_count(sphere, warped)
        assert result["flipped_count"] == 0
        assert result["flipped_fraction"] == pytest.approx(0.0)

    def test_global_flip_detected(self):
        """Explicitly reversing the winding of all triangles is detected.

        Source triangles: (p0, p1, p2) with p2 at y=+1 → CCW winding, +z normal.
        Warped triangles: p2 moved to y=-1 → normal becomes -z (dot with +z < 0).
        Each triangle is independent (no shared vertices) so the result is exact.
        """
        # 3 independent triangles in the XY plane, each with a +z normal.
        # Connectivity: tri0=(0,1,2), tri1=(3,4,5), tri2=(6,7,8)
        # Normal for tri (p0,p1,p2)=((0,0,0),(1,0,0),(0.5,+1,0)):
        #   e1=(1,0,0), e2=(0.5,+1,0) → n=(0,0,+1)
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],  # tri 0
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.5, 1.0, 0.0],  # tri 1
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [4.5, 1.0, 0.0],  # tri 2
            ],
            dtype=float,
        )
        faces = np.array([3, 0, 1, 2, 3, 3, 4, 5, 3, 6, 7, 8])
        mesh = pv.PolyData(points, faces)

        # Flip: move each third vertex (indices 2, 5, 8) from y=+1 to y=-1.
        # New normal: e1=(1,0,0), e2=(0.5,-1,0) → n=(0,0,-1) → dot with (0,0,+1) = -1 < 0.
        warped = points.copy()
        warped[2, 1] = -1.0
        warped[5, 1] = -1.0
        warped[8, 1] = -1.0

        result = foldover_count(mesh, warped)
        assert result["flipped_count"] == mesh.n_cells

    def test_partial_flip_detected(self):
        """Flipping winding of one triangle in a 2-triangle mesh is detected."""
        plane = _make_plane_mesh()
        warped = _make_flipped_mesh(plane, flip_indices=[0])
        result = foldover_count(plane, warped)
        assert result["flipped_count"] >= 1

    def test_flipped_fraction_range(self):
        sphere = _make_sphere()
        warped = sphere.points.copy()
        result = foldover_count(sphere, warped)
        assert 0.0 <= result["flipped_fraction"] <= 1.0

    def test_near_degenerate_key_present(self):
        sphere = _make_sphere()
        result = foldover_count(sphere, sphere.points.copy())
        assert "near_degenerate" in result


# ---------------------------------------------------------------------------
# roundtrip_distance
# ---------------------------------------------------------------------------


class TestRoundtripDistance:
    def test_identical_roundtrip_zero(self):
        """Round-trip that returns to origin should give zero distances."""
        pts = np.zeros((50, 3))
        result = roundtrip_distance(original_points=pts, roundtrip_points=pts)
        assert result["min"] == pytest.approx(0.0)
        assert result["max"] == pytest.approx(0.0)
        assert "per_vertex" in result
        np.testing.assert_allclose(result["per_vertex"], 0.0, atol=1e-10)

    def test_known_displacement(self):
        """All vertices displaced by 2 in x → per-vertex distance all 2."""
        orig = np.zeros((20, 3))
        rt = orig.copy()
        rt[:, 0] = 2.0
        result = roundtrip_distance(original_points=orig, roundtrip_points=rt)
        assert result["mean"] == pytest.approx(2.0)
        assert result["min"] == pytest.approx(2.0)
        assert result["max"] == pytest.approx(2.0)
        assert result["per_vertex"].shape == (20,)

    def test_all_keys_present(self):
        pts = np.ones((10, 3))
        result = roundtrip_distance(original_points=pts, roundtrip_points=pts)
        for key in ("min", "p25", "p50", "mean", "p75", "p95", "max", "per_vertex"):
            assert key in result


class TestTheReversedPairHidesASwap:
    """
    #56, part 4 — "adjacent metrics take their arguments in opposite order".

    ``roundtrip_distance(original_points, roundtrip_points)`` and
    ``forward_backward_disagreement(roundtrip_points, original_points)`` take **the same
    two arrays in opposite order**, forty lines apart in one module, and neither swap was
    visible in the numbers a caller reads:

    * ``roundtrip_distance`` is ``norm(rt - orig)`` — symmetric, so a swap is a no-op;
    * ``forward_backward_disagreement`` sign-flips ``field`` and leaves
      ``magnitude_percentiles`` untouched, so a swap is invisible in the summary and
      shows only in the raw displacement field.

    The plan's dispositions table sent ``roundtrip_distance`` / ``directed_distance_
    percentiles`` to keyword-only. That was the wrong sibling, and the third test says
    why: ``directed_distance_percentiles`` is asymmetric, documented as directional, and
    takes a point array against a *mesh-or-array* — a swap there changes the number
    rather than hiding, so it keeps its positional signature.

    The two that hide a swap are keyword-only as of §8.0.N, so a swap is a ``TypeError``.
    The three tests above call by keyword and still pass, which is what says the *values*
    did not move when the signatures did.
    """

    def _pair(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(50, 3))
        return a, a + rng.normal(scale=0.1, size=(50, 3))

    def test_roundtrip_distance_is_symmetric_so_a_swap_is_invisible(self):
        a, b = self._pair()
        np.testing.assert_array_equal(
            roundtrip_distance(original_points=a, roundtrip_points=b)["per_vertex"],
            roundtrip_distance(original_points=b, roundtrip_points=a)["per_vertex"],
        )

    def test_the_disagreement_field_flips_sign_while_its_summary_does_not(self):
        a, b = self._pair()
        forward = forward_backward_disagreement(roundtrip_points=a, original_points=b)
        backward = forward_backward_disagreement(roundtrip_points=b, original_points=a)
        np.testing.assert_allclose(forward["field"], -backward["field"])
        assert forward["magnitude_percentiles"]["mean"] == backward["magnitude_percentiles"]["mean"]

    def test_the_directed_metric_is_asymmetric_so_a_swap_shows(self):
        """Which is why it is not in the pair the keyword-only change covers."""
        a, b = self._pair()
        assert directed_distance_percentiles(a, b)["mean"] != pytest.approx(
            directed_distance_percentiles(b, a)["mean"]
        )

    @pytest.mark.parametrize(
        "func", [roundtrip_distance, forward_backward_disagreement], ids=["roundtrip", "disagree"]
    )
    def test_a_positional_call_is_refused(self, func):
        a, b = self._pair()
        with pytest.raises(TypeError):
            func(a, b)


# ---------------------------------------------------------------------------
# forward_backward_disagreement
# ---------------------------------------------------------------------------


class TestForwardBackwardDisagreement:
    def test_identical_gives_zero_field(self):
        """Zero disagreement when roundtrip equals original."""
        pts = np.zeros((30, 3))
        result = forward_backward_disagreement(roundtrip_points=pts, original_points=pts)
        np.testing.assert_allclose(result["field"], 0.0, atol=1e-10)
        assert result["magnitude_percentiles"]["max"] == pytest.approx(0.0)

    def test_field_shape(self):
        orig = np.zeros((25, 3))
        rt = np.ones((25, 3))
        result = forward_backward_disagreement(roundtrip_points=rt, original_points=orig)
        assert result["field"].shape == (25, 3)

    def test_known_field(self):
        """All points displaced by (1, 0, 0) → field all (1,0,0), magnitudes all 1."""
        orig = np.zeros((10, 3))
        rt = np.zeros((10, 3))
        rt[:, 0] = 1.0
        result = forward_backward_disagreement(roundtrip_points=rt, original_points=orig)
        np.testing.assert_allclose(result["field"][:, 0], 1.0)
        np.testing.assert_allclose(result["field"][:, 1:], 0.0)
        assert result["magnitude_percentiles"]["mean"] == pytest.approx(1.0)

    def test_magnitude_percentiles_keys(self):
        pts = np.zeros((5, 3))
        result = forward_backward_disagreement(roundtrip_points=pts, original_points=pts)
        for key in ("min", "p25", "p50", "mean", "p75", "p95", "max"):
            assert key in result["magnitude_percentiles"]


# ---------------------------------------------------------------------------
# score_correspondence — top-level integration tests
# ---------------------------------------------------------------------------


class TestScoreCorrespondence:
    def test_mesh_against_itself_all_metrics(self):
        """Mesh scored against itself should give near-zero ASSD and zero foldover."""
        sphere = _make_sphere(theta_resolution=15, phi_resolution=15)
        sdf_vals = np.zeros(sphere.n_points)
        rt_pts = sphere.points.copy()

        result = score_correspondence(
            warped_mesh=sphere,
            target_mesh=sphere,
            source_mesh=sphere,
            sdf_values=sdf_vals,
            roundtrip_points=rt_pts,
            compute_self_intersection=True,
        )

        assert result["assd"] == pytest.approx(0.0, abs=1e-10)
        assert result["foldover_count"]["flipped_count"] == 0
        assert result["off_surface_error"]["max"] == pytest.approx(0.0)
        assert result["roundtrip_distance"]["max"] == pytest.approx(0.0)

    def test_without_optional_args_skips_gracefully(self):
        """Missing optional inputs should produce skipped entries, not crashes."""
        sphere = _make_sphere(theta_resolution=10, phi_resolution=10)
        result = score_correspondence(
            warped_mesh=sphere,
            target_mesh=sphere,
        )

        # Compulsory metrics should be present
        assert "assd" in result
        assert "triangle_health" in result
        assert "directed_distance_warped_to_target" in result

        # Optional metrics should be skipped
        assert result["foldover_count"]["skipped"] is True
        assert result["off_surface_error"]["skipped"] is True
        assert result["roundtrip_distance"]["skipped"] is True
        assert result["forward_backward_disagreement"]["skipped"] is True

    def test_skip_self_intersection(self):
        sphere = _make_sphere(theta_resolution=10, phi_resolution=10)
        result = score_correspondence(
            warped_mesh=sphere,
            target_mesh=sphere,
            compute_self_intersection=False,
        )
        assert result["self_intersection_count"]["skipped"] is True

    def test_translated_mesh_nonzero_assd(self):
        """Translated mesh should give non-trivial ASSD."""
        sphere = _make_sphere(radius=1.0, theta_resolution=20, phi_resolution=20)
        shifted = _translate_mesh(sphere, np.array([10.0, 0.0, 0.0]))
        result = score_correspondence(
            warped_mesh=shifted,
            target_mesh=sphere,
            compute_self_intersection=False,
        )
        assert result["assd"] > 5.0

    def test_result_is_dict(self):
        sphere = _make_sphere()
        result = score_correspondence(sphere, sphere, compute_self_intersection=False)
        assert isinstance(result, dict)

    def test_skip_reason_strings(self):
        """Skipped entries should include a non-empty 'reason' string."""
        sphere = _make_sphere()
        result = score_correspondence(sphere, sphere)
        skipped_keys = [k for k, v in result.items() if isinstance(v, dict) and v.get("skipped")]
        for key in skipped_keys:
            assert isinstance(result[key].get("reason"), str)
            assert len(result[key]["reason"]) > 0
