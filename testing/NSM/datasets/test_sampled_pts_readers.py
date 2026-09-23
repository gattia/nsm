"""
The two subject-level readers, ``read_mesh_get_sampled_pts`` and
``read_meshes_get_sampled_pts``.

The meshes are analytic spheres with known centres and radii, so the frame arithmetic
under each ``mesh_to_scale`` / ``scale_all_meshes`` / ``center_all_meshes`` combination is
asserted against arithmetic, not baselines.
"""

import logging

import numpy as np
import pytest
import pyvista as pv
from pymskt.mesh import Mesh

from NSM.datasets.sdf_dataset import read_mesh_get_sampled_pts, read_meshes_get_sampled_pts
from NSM.datasets.utils import get_buffered_cube_mins_maxs

BONE_CENTER = np.array([2.0, 0.0, 0.0])
BONE_RADIUS = 1.0
CART_CENTER = np.array([5.0, 0.0, 0.0])
CART_RADIUS = 0.5

#: Farthest vertex of either sphere from the bone's centre.
JOINT_RADIUS_FROM_BONE = 3.5


@pytest.fixture(scope="module")
def sphere_paths(tmp_path_factory):
    """``(bone_path, cart_path)``: analytic spheres, no meshfix, no randomness."""
    directory = tmp_path_factory.mktemp("reader_meshes")
    paths = []
    for name, radius, center, resolution in (
        ("bone", BONE_RADIUS, BONE_CENTER, 24),
        ("cart", CART_RADIUS, CART_CENTER, 18),
    ):
        path = str(directory / f"{name}.vtk")
        pv.Sphere(
            radius=radius, center=center, theta_resolution=resolution, phi_resolution=resolution
        ).triangulate().save(path)
        paths.append(path)
    return tuple(paths)


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_both_readers_refuse_registration_without_a_mean_mesh(sphere_paths):
    """
    Fails if ``read_mesh_get_sampled_pts`` or ``read_meshes_get_sampled_pts`` proceeds with
    ``register_to_mean_first=True`` and no ``mean_mesh``, or changes its exception type.

    The single reader raises a bare ``Exception`` and the multi reader ``ValueError``.
    """
    with pytest.raises(Exception, match="Must provide mean mesh") as excinfo:
        read_mesh_get_sampled_pts(
            sphere_paths[0], n_pts=10, register_to_mean_first=True, fix_mesh=False
        )
    assert type(excinfo.value) is Exception
    with pytest.raises(ValueError, match="Must provide mean mesh"):
        read_meshes_get_sampled_pts(
            list(sphere_paths),
            sigma=[0.1, 0.1],
            n_pts=[5, 5],
            register_to_mean_first=True,
            fix_mesh=False,
        )


def test_deprecated_keywords_warn_and_unknown_ones_are_ignored(sphere_paths, caplog, capsys):
    """
    Fails if the readers reword their ``return_*`` or ``mean`` deprecation warnings, honour a
    ``return_*=False``, reject or log an unknown keyword, or print to stdout.

    The deprecated flags are always on and ``mean`` has no effect. Both readers take
    ``**kwargs``, so an unknown keyword is swallowed without a word.
    """
    with caplog.at_level(logging.DEBUG, logger="NSM"):
        single = read_mesh_get_sampled_pts(
            sphere_paths[0],
            n_pts=10,
            sigma=0.1,
            fix_mesh=False,
            return_scale=False,
            mean=3,
            not_a_parameter=True,
        )
        multi = read_meshes_get_sampled_pts(
            list(sphere_paths),
            sigma=[0.1, 0.1],
            n_pts=[10, 10],
            fix_mesh=False,
            return_orig_mesh=False,
            mean=1,
        )
    messages = _warnings(caplog)
    assert "return_scale is deprecated and not used in this function - always True" in messages
    assert (
        messages.count("mean is deprecated and not used in this function - it never had an effect")
        == 2
    )
    assert "return_orig_mesh is deprecated and not used in this function - always True" in messages
    assert "scale" in single and "orig_mesh" in multi
    assert "not_a_parameter" not in " ".join(r.getMessage() for r in caplog.records)
    assert capsys.readouterr().out == ""


class TestSingleMeshReader:
    def test_its_outputs_on_each_path(self, sphere_paths):
        """
        Fails if ``read_mesh_get_sampled_pts`` returns anything but None for a missing path,
        nonzero SDFs or non-vertex points under ``get_random=False``, any normalization with
        ``center_pts`` and ``norm_pts`` both off, or an ``xyz`` key (#15).
        """
        assert read_mesh_get_sampled_pts("/nonexistent.vtk", n_pts=10, fix_mesh=False) is None

        surface = read_mesh_get_sampled_pts(sphere_paths[0], get_random=False, fix_mesh=False)
        assert "xyz" not in surface
        assert np.all(surface["sdf"] == 0) and np.all(surface["pts_surface"] == 0)
        np.testing.assert_array_equal(surface["pts"], surface["new_pts"][0])

        raw = read_mesh_get_sampled_pts(
            sphere_paths[0],
            n_pts=20,
            sigma=0.1,
            center_pts=False,
            norm_pts=False,
            return_point_cloud=True,
            fix_mesh=False,
        )
        assert "xyz" not in raw and raw["pts"].shape == (20, 3)
        assert raw["scale"] == 1 and np.all(raw["center"] == 0)
        np.testing.assert_array_equal(raw["new_pts"][0], raw["orig_pts"][0])
        np.testing.assert_array_equal(raw["point_cloud"], raw["new_pts"][0])

    def test_include_surf_in_pts_appends_the_vertices(self, sphere_paths):
        """
        Fails if ``read_mesh_get_sampled_pts(include_surf_in_pts=True)`` does not append the
        mesh's normalized vertices after the ``n_pts`` random points, each with a
        ``pts_surface`` entry.
        """
        result = read_mesh_get_sampled_pts(
            sphere_paths[0], n_pts=15, sigma=0.1, include_surf_in_pts=True, fix_mesh=False
        )
        vertices = result["new_pts"][0]
        np.testing.assert_array_equal(result["pts"][15:], vertices)
        assert result["pts_surface"].shape[0] == 15 + vertices.shape[0]

    def test_registration_takes_the_references_position_and_size(self, sphere_paths):
        """
        Fails if ``read_mesh_get_sampled_pts(register_to_mean_first=True)`` leaves the subject
        at its own centre or size instead of similarity-registering it to ``mean_mesh``, or
        returns no ``icp_transform``.
        """
        bone_path, cart_path = sphere_paths
        result = read_mesh_get_sampled_pts(
            cart_path,
            n_pts=10,
            sigma=0.1,
            register_to_mean_first=True,
            mean_mesh=Mesh(bone_path),
            center_pts=False,
            norm_pts=False,
            fix_mesh=False,
        )
        registered = result["new_pts"][0]
        center = registered.mean(axis=0)
        assert result["icp_transform"] is not None
        assert np.linalg.norm(center - BONE_CENTER) < 0.1
        assert abs(np.linalg.norm(registered - center, axis=1).mean() - BONE_RADIUS) < 0.1

    def test_fix_mesh_runs_meshfix(self, sphere_paths, caplog):
        """
        Fails if ``read_mesh_get_sampled_pts(fix_mesh=True)`` does not run ``meshfix``.

        Detected by ``meshfix``'s ``Fixed mesh,`` log line, so rewording that line fails it.
        """
        with caplog.at_level(logging.DEBUG, logger="NSM"):
            result = read_mesh_get_sampled_pts(sphere_paths[0], n_pts=10, sigma=0.1, fix_mesh=True)
        assert result is not None
        assert any(r.getMessage().startswith("Fixed mesh,") for r in caplog.records)


class TestMultiMeshReader:
    def test_a_missing_path_fails_the_subject_and_a_none_entry_does_not(self, sphere_paths):
        """
        Fails if ``read_meshes_get_sampled_pts`` returns a subject with a nonexistent path,
        or, for a ``None`` surface, raises, drops its ``None`` placeholders or draws for it.
        """
        kwargs = dict(sigma=[0.1, 0.1], n_pts=[10, 10], fix_mesh=False)
        assert read_meshes_get_sampled_pts([sphere_paths[0], "/nonexistent.vtk"], **kwargs) is None

        result = read_meshes_get_sampled_pts([sphere_paths[0], None], **kwargs)
        assert result["orig_pts"][1] is None and result["new_pts"][1] is None
        assert result["sdf"][1] is None
        assert set(np.unique(result["pts_surface"])) == {0}
        assert result["pts"].shape[0] == 10

    def test_get_random_false_zeroes_each_surfaces_own_vertices(self, sphere_paths):
        """
        Fails if ``read_meshes_get_sampled_pts(get_random=False)`` does not give each surface
        SDF 0 on its own vertices and a computed SDF on the other's, or mislabels
        ``pts_surface``.
        """
        result = read_meshes_get_sampled_pts(list(sphere_paths), get_random=False, fix_mesh=False)
        n0, n1 = (pts.shape[0] for pts in result["new_pts"])
        assert result["pts"].shape[0] == n0 + n1
        assert np.all(result["sdf"][0][:n0] == 0) and np.any(result["sdf"][0][n0:] != 0)
        assert np.all(result["sdf"][1][n0:] == 0)
        np.testing.assert_array_equal(result["pts_surface"], np.repeat([0, 1], [n0, n1]))

    def test_the_frame_follows_the_scaling_options(self, sphere_paths):
        """
        Fails if ``read_meshes_get_sampled_pts`` centres or scales on the wrong surfaces for
        ``scale_all_meshes``, ``center_all_meshes`` or a list ``mesh_to_scale``.

        Default: centred on surface 0, scaled so every surface fits. ``scale_all_meshes=False``
        scales by surface 0 alone. ``center_all_meshes`` centres on the union. A
        ``mesh_to_scale`` list uses the union for both, with and without ``scale_all_meshes``.
        """

        def frame(**options):
            result = read_meshes_get_sampled_pts(
                list(sphere_paths), sigma=[0.1, 0.1], n_pts=[20, 20], fix_mesh=False, **options
            )
            union = np.vstack(result["orig_pts"])
            return result, union, union.mean(axis=0)

        result, _, _ = frame()
        np.testing.assert_allclose(result["center"], BONE_CENTER, atol=0.02)
        np.testing.assert_allclose(result["scale"], JOINT_RADIUS_FROM_BONE, atol=0.02)
        assert np.linalg.norm(result["new_pts"][0].mean(axis=0)) < 1e-5
        np.testing.assert_allclose(
            np.linalg.norm(np.vstack(result["new_pts"]), axis=1).max(), 1.0, atol=1e-6
        )

        result, _, _ = frame(scale_all_meshes=False)
        np.testing.assert_allclose(result["scale"], BONE_RADIUS, atol=0.02)

        result, _, center = frame(scale_all_meshes=False, center_all_meshes=True)
        np.testing.assert_allclose(result["center"], center, atol=1e-5)
        scale = np.linalg.norm(result["orig_pts"][0] - center, axis=1).max()
        np.testing.assert_allclose(result["scale"], scale, atol=1e-5)

        for options in (dict(scale_all_meshes=False), {}):
            result, union, center = frame(mesh_to_scale=[0, 1], **options)
            np.testing.assert_allclose(result["center"], center, atol=1e-5)
            scale = np.linalg.norm(union - center, axis=1).max()
            np.testing.assert_allclose(result["scale"], scale, atol=1e-5)

    def test_a_zero_count_surface_draws_nothing_and_still_gets_an_sdf(self, sphere_paths):
        """
        Fails if ``read_meshes_get_sampled_pts`` crashes on or draws points for a surface with
        ``n_pts`` 0, or omits that surface's SDF to the other surface's points.
        """
        result = read_meshes_get_sampled_pts(
            list(sphere_paths), sigma=[0.1, 0.1], n_pts=[30, 0], fix_mesh=False
        )
        assert result["pts"].shape[0] == 30
        assert set(np.unique(result["pts_surface"])) == {0}
        assert result["sdf"][1].shape[0] == 30

    def test_a_supplied_icp_transform_is_used_instead_of_registering(self, sphere_paths):
        """
        Fails if ``read_meshes_get_sampled_pts`` registers again instead of returning and
        applying the ``icp_transform`` it was given.

        The dataset registers on its first sampling pass and passes that transform to every
        later pass, so all of a subject's points share one registration. Registration is
        deterministic here (checked 2026-09-23), so only the ``is`` assert catches a second
        registration.
        """
        bone_path, cart_path = sphere_paths
        kwargs = dict(
            sigma=[0.1, 0.1],
            n_pts=[10, 10],
            register_to_mean_first=True,
            mean_mesh=Mesh(bone_path),
            center_pts=False,
            norm_pts=False,
            fix_mesh=False,
        )
        first = read_meshes_get_sampled_pts([cart_path, bone_path], **kwargs)
        second = read_meshes_get_sampled_pts(
            [cart_path, bone_path], icp_transform=first["icp_transform"], **kwargs
        )
        assert first["icp_transform"] is not None
        assert second["icp_transform"] is first["icp_transform"]
        for surf_idx in range(2):
            np.testing.assert_array_equal(second["new_pts"][surf_idx], first["new_pts"][surf_idx])

    def test_a_none_sigma_draws_from_one_cube_around_all_surfaces(self, sphere_paths):
        """
        Fails if a ``None``-sigma surface in ``read_meshes_get_sampled_pts`` draws from a cube
        around its own surface instead of one cube around all surfaces.

        A cube around the bone alone ends at x = 3, so a bone-labelled sample beyond 3.5 can
        only come from the joint cube.
        """
        result = read_meshes_get_sampled_pts(
            list(sphere_paths),
            sigma=[None, None],
            n_pts=[200, 200],
            center_pts=False,
            norm_pts=False,
            fix_mesh=False,
        )
        mins, maxs = get_buffered_cube_mins_maxs(np.vstack(result["new_pts"]), 0.0)
        assert result["pts"].shape[0] == 400
        assert np.all(result["pts"] >= mins) and np.all(result["pts"] <= maxs)
        assert result["pts"][result["pts_surface"] == 0][:, 0].max() > 3.5

    @pytest.mark.parametrize("sigma", [0.1, None], ids=["near-surface", "uniform-cube"])
    def test_include_surf_in_pts_appends_each_surfaces_own_vertices(self, sphere_paths, sigma):
        """
        Fails if ``read_meshes_get_sampled_pts(include_surf_in_pts=True)`` appends anything but
        each surface's own vertices after that surface's draw, or raises on the uniform-cube
        path (#17).
        """
        result = read_meshes_get_sampled_pts(
            list(sphere_paths),
            sigma=[sigma, sigma],
            n_pts=[10, 10],
            include_surf_in_pts=True,
            center_pts=False,
            norm_pts=False,
            fix_mesh=False,
        )
        bone_verts, cart_verts = result["new_pts"]
        end_of_bone = 10 + bone_verts.shape[0]
        np.testing.assert_array_equal(result["pts"][10:end_of_bone], bone_verts)
        np.testing.assert_array_equal(result["pts"][end_of_bone + 10 :], cart_verts)
