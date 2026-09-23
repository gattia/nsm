"""``NSM/mesh/correspondence_metrics.py`` on synthetic geometry with known answers."""

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

PERCENTILES = {"min", "p25", "p50", "mean", "p75", "p95", "max"}


def _sphere(radius=1.0, resolution=20):
    return pv.Sphere(radius=radius, theta_resolution=resolution, phi_resolution=resolution)


def _plane():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    return pv.PolyData(points, np.array([3, 0, 1, 2, 3, 0, 2, 3]))


def test_assd_is_zero_on_itself_and_grows_with_the_offset():
    """
    Fails if ``assd`` is nonzero for a mesh against itself, returns a non-float, or sums
    signed point-to-surface distances instead of their absolute values.

    Only the radius-2 case catches a signed sum: the two spheres' signed distances to each
    other have opposite signs.
    """
    sphere = _sphere(resolution=30)
    assert assd(sphere, sphere) == pytest.approx(0.0, abs=1e-10)
    assert isinstance(assd(sphere, sphere), float)
    assert 2.0 < assd(sphere, pv.PolyData(sphere.points + [5.0, 0, 0], sphere.faces)) < 6.0
    assert assd(sphere, _sphere(radius=2.0)) > 0.5


def test_the_distance_summaries_on_known_inputs():
    """
    Fails if ``directed_distance_percentiles``, ``off_surface_error`` or
    ``roundtrip_distance`` change their keys, or miscompute a constant 3-unit shift, the
    absolute SDF values, or their RMS.
    """
    origin = np.zeros((20, 3))
    shifted = origin + [3.0, 0, 0]

    directed = directed_distance_percentiles(origin, shifted)
    assert set(directed) == PERCENTILES
    assert directed["min"] == directed["max"] == pytest.approx(3.0)
    assert directed_distance_percentiles(origin, origin)["max"] == pytest.approx(0.0)

    errors = off_surface_error(np.array([-1.0, 2.0, -3.0, 4.0]))
    assert set(errors) == PERCENTILES | {"rms"}
    assert (errors["min"], errors["max"], errors["mean"]) == pytest.approx((1.0, 4.0, 2.5))
    assert errors["rms"] == pytest.approx(np.sqrt(7.5))

    roundtrip = roundtrip_distance(original_points=origin, roundtrip_points=shifted)
    assert set(roundtrip) == PERCENTILES | {"per_vertex"}
    np.testing.assert_allclose(roundtrip["per_vertex"], 3.0)


def test_triangle_health_on_clean_and_collapsed_meshes():
    """
    Fails if ``triangle_health`` changes its 12 keys, reports degenerate triangles on a clean
    sphere or plane, or misses a triangle with a zero-length edge.
    """
    health = triangle_health(_sphere(resolution=30))
    assert set(health) == {
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
    }
    assert health["edge_ratio_mean"] < 10.0 and health["edge_length_min"] > 0.0
    assert health["degenerate_count"] == triangle_health(_plane())["degenerate_count"] == 0

    points = np.array([[0.0, 0, 0], [0.0, 0, 0], [0.5, 1, 0], [2.0, 0, 0]])
    collapsed = pv.PolyData(points, np.array([3, 0, 1, 2, 3, 0, 2, 3]))
    assert triangle_health(collapsed)["degenerate_count"] >= 1


def test_self_intersections_are_counted_and_large_meshes_are_skipped():
    """
    Fails if ``self_intersection_count`` counts crossings on a clean sphere or plane, misses
    two crossing triangles, or does not return None with a ``RuntimeWarning`` above
    ``max_triangles``.

    The crossing case runs only with ``broadphase=True``.
    """
    assert self_intersection_count(_sphere(resolution=15)) == 0
    assert self_intersection_count(_plane()) == 0
    assert isinstance(self_intersection_count(_plane()), int)

    crossing = pv.PolyData(
        np.array([[0, 0, 0], [2, 0, 0], [1, 2, 0], [0.5, 0.5, -1], [1.5, 0.5, -1], [1, 0.5, 1]]),
        np.array([3, 0, 1, 2, 3, 3, 4, 5]),
    )
    assert self_intersection_count(crossing, broadphase=True) >= 1

    with pytest.warns(RuntimeWarning, match="max_triangles"):
        assert self_intersection_count(_sphere(resolution=30), max_triangles=10) is None


def test_foldovers_are_counted():
    """
    Fails if ``foldover_count`` counts flips on an unwarped mesh, drops ``near_degenerate``,
    or misses a triangle whose orientation reversed.
    """
    sphere = _sphere()
    unchanged = foldover_count(sphere, sphere.points.copy())
    assert unchanged["flipped_count"] == 0 and unchanged["flipped_fraction"] == 0.0
    assert "near_degenerate" in unchanged

    # Three separate triangles with +z normals; moving each apex to y = -1 flips all three.
    points = np.array(
        [[x + dx, dy, 0.0] for x in (0, 2, 4) for dx, dy in ((0, 0), (1, 0), (0.5, 1))]
    )
    triangles = pv.PolyData(points, np.array([3, 0, 1, 2, 3, 3, 4, 5, 3, 6, 7, 8]))
    warped = points.copy()
    warped[[2, 5, 8], 1] = -1.0
    assert foldover_count(triangles, warped)["flipped_count"] == 3

    plane = _plane()
    swapped = plane.points.copy()
    swapped[[1, 2]] = swapped[[2, 1]]
    assert foldover_count(plane, swapped)["flipped_count"] >= 1


class TestTheReversedPairHidesASwap:
    """
    ``roundtrip_distance(original_points, roundtrip_points)`` and
    ``forward_backward_disagreement(roundtrip_points, original_points)`` take the same two
    arrays in opposite order, and a swap is invisible in both: the first is symmetric, and
    the second flips the sign of ``field`` but not its summary. So both are keyword-only
    (#56). ``directed_distance_percentiles`` is asymmetric, so a swap shows, and it stays
    positional.
    """

    def test_a_swap_is_invisible_so_a_positional_call_is_refused(self):
        """
        Fails if ``roundtrip_distance`` or ``forward_backward_disagreement`` accepts
        positional arguments (#56).

        The asserts before the loop check the premise in the class docstring. Only the loop
        guards the fix.
        """
        rng = np.random.default_rng(0)
        a = rng.normal(size=(50, 3))
        b = a + rng.normal(scale=0.1, size=(50, 3))

        np.testing.assert_array_equal(
            roundtrip_distance(original_points=a, roundtrip_points=b)["per_vertex"],
            roundtrip_distance(original_points=b, roundtrip_points=a)["per_vertex"],
        )
        forward = forward_backward_disagreement(roundtrip_points=a, original_points=b)
        backward = forward_backward_disagreement(roundtrip_points=b, original_points=a)
        np.testing.assert_allclose(forward["field"], -backward["field"])
        assert forward["magnitude_percentiles"] == backward["magnitude_percentiles"]
        assert directed_distance_percentiles(a, b)["mean"] != pytest.approx(
            directed_distance_percentiles(b, a)["mean"]
        )
        for func in (roundtrip_distance, forward_backward_disagreement):
            with pytest.raises(TypeError):
                func(a, b)

    def test_the_disagreement_field_on_a_known_displacement(self):
        """
        Fails if ``forward_backward_disagreement`` returns ``field`` as original minus
        roundtrip, or miscomputes the magnitude summary of a unit shift.
        """
        original = np.zeros((10, 3))
        moved = original + [1.0, 0, 0]
        result = forward_backward_disagreement(roundtrip_points=moved, original_points=original)
        np.testing.assert_allclose(result["field"], moved)
        assert set(result["magnitude_percentiles"]) == PERCENTILES
        assert result["magnitude_percentiles"]["mean"] == pytest.approx(1.0)


class TestScoreCorrespondence:
    def test_a_mesh_scored_against_itself(self):
        """
        Fails if ``score_correspondence``, given every input, skips or swallows an error in
        ``assd``, ``foldover_count``, ``self_intersection_count``, ``off_surface_error`` or
        ``roundtrip_distance``, or scores a mesh against itself as nonzero.

        One mesh plays every role, so a swap between source, warped and target cannot show.
        """
        sphere = _sphere(resolution=15)
        result = score_correspondence(
            warped_mesh=sphere,
            target_mesh=sphere,
            source_mesh=sphere,
            sdf_values=np.zeros(sphere.n_points),
            roundtrip_points=sphere.points.copy(),
        )
        assert result["assd"] == pytest.approx(0.0, abs=1e-10)
        assert result["foldover_count"]["flipped_count"] == 0
        assert result["self_intersection_count"] == 0
        assert result["off_surface_error"]["max"] == 0.0
        assert result["roundtrip_distance"]["max"] == 0.0

    def test_missing_inputs_skip_with_a_reason(self):
        """
        Fails if ``score_correspondence`` computes or errors on a metric whose input is
        absent instead of skipping it with a reason, or skips one that needs only the two
        meshes.
        """
        sphere = _sphere(resolution=10)
        result = score_correspondence(
            pv.PolyData(sphere.points + [10.0, 0, 0], sphere.faces),
            sphere,
            compute_self_intersection=False,
        )
        assert result["assd"] > 5.0
        assert {"triangle_health", "directed_distance_warped_to_target"} <= set(result)
        skipped = {k for k, v in result.items() if isinstance(v, dict) and v.get("skipped")}
        assert skipped == {
            "foldover_count",
            "off_surface_error",
            "roundtrip_distance",
            "forward_backward_disagreement",
            "self_intersection_count",
        }
        assert all(result[k]["reason"] for k in skipped)
