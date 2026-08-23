"""
Characterization tests for the two subject-level reader pipelines,
``read_mesh_get_sampled_pts`` and ``read_meshes_get_sampled_pts``, written immediately
before their move to ``NSM/datasets/mesh_sampling.py`` (plan §8.0, slice A).

Everything asserts behaviour as it stands today, including the warts the docstrings
already record: the bare-``Exception``-vs-``ValueError`` raise asymmetry between the two
readers, and the always-on deprecated flags. The ``include_surf_in_pts`` tests on the
multi reader were ``xfail(strict=True)`` pins of #17 until its fix landed, and the
``"pts"``-vs-``"xyz"`` key asymmetry was pinned here until #15 unified the readers on
``"pts"``; those now assert the fixed behaviour.

Meshes are analytic spheres with known centers and radii, so the frame math
(``center`` / ``scale`` under each ``mesh_to_scale`` / ``scale_all_meshes`` /
``center_all_meshes`` combination) is assertable against arithmetic, not baselines.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from pymskt.mesh import Mesh  # noqa: E402

from NSM.datasets.sdf_dataset import (  # noqa: E402
    read_mesh_get_sampled_pts,
    read_meshes_get_sampled_pts,
)
from NSM.datasets.utils import get_buffered_cube_mins_maxs  # noqa: E402

BONE_CENTER = np.array([2.0, 0.0, 0.0])
BONE_RADIUS = 1.0
CART_CENTER = np.array([5.0, 0.0, 0.0])
CART_RADIUS = 0.5

#: Farthest any vertex of either sphere lies from the bone's center:
#: |CART_CENTER - BONE_CENTER| + CART_RADIUS.
JOINT_RADIUS_FROM_BONE = 3.5


@pytest.fixture(scope="module")
def sphere_paths(tmp_path_factory):
    """(bone_path, cart_path): analytic spheres — no meshfix, no randomness."""
    directory = tmp_path_factory.mktemp("reader_meshes")
    bone = pv.Sphere(
        radius=BONE_RADIUS, center=BONE_CENTER, theta_resolution=24, phi_resolution=24
    ).triangulate()
    cart = pv.Sphere(
        radius=CART_RADIUS, center=CART_CENTER, theta_resolution=18, phi_resolution=18
    ).triangulate()
    bone_path = str(directory / "bone.vtk")
    cart_path = str(directory / "cart.vtk")
    bone.save(bone_path)
    cart.save(cart_path)
    return bone_path, cart_path


class TestSingleMeshReader:
    def test_a_missing_path_returns_none(self):
        result = read_mesh_get_sampled_pts("/nonexistent/mesh.vtk", n_pts=10, fix_mesh=False)
        assert result is None

    def test_get_random_false_uses_the_pts_key_with_zero_sdf(self, sphere_paths):
        """``pts`` on this path too; ``xyz`` was only ever the random path's key."""
        result = read_mesh_get_sampled_pts(sphere_paths[0], get_random=False, fix_mesh=False)
        assert "pts" in result and "xyz" not in result
        assert np.all(result["sdf"] == 0)
        assert np.all(result["pts_surface"] == 0)
        np.testing.assert_array_equal(result["pts"], result["new_pts"][0])

    def test_the_unnormalized_branch_returns_the_identity_frame(self, sphere_paths):
        result = read_mesh_get_sampled_pts(
            sphere_paths[0], n_pts=20, sigma=0.1, center_pts=False, norm_pts=False, fix_mesh=False
        )
        assert result["scale"] == 1
        assert np.all(result["center"] == np.zeros(3))
        np.testing.assert_array_equal(result["new_pts"][0], result["orig_pts"][0])

    def test_include_surf_in_pts_appends_the_surface_vertices(self, sphere_paths):
        n_random = 15
        result = read_mesh_get_sampled_pts(
            sphere_paths[0], n_pts=n_random, sigma=0.1, include_surf_in_pts=True, fix_mesh=False
        )
        vertices = result["new_pts"][0]
        assert result["pts"].shape[0] == n_random + vertices.shape[0]
        np.testing.assert_array_equal(result["pts"][n_random:], vertices)
        assert result["pts_surface"].shape[0] == n_random + vertices.shape[0]

    def test_the_random_path_returns_its_draw_under_pts(self, sphere_paths):
        """
        Was the #15 strict xfail: ``pts`` is the one key. The legacy ``xyz`` briefly
        survived as an alias, deleted before it ever shipped in a release (maintainer,
        2026-08-23): an external reader gets a loud KeyError, not a silent second name.
        """
        result = read_mesh_get_sampled_pts(sphere_paths[0], n_pts=10, sigma=0.1, fix_mesh=False)
        assert "pts" in result
        assert "xyz" not in result

    def test_registering_without_a_mean_mesh_raises_a_bare_exception(self, sphere_paths):
        """The multi reader raises ValueError for the same mistake; pinned as-is."""
        with pytest.raises(Exception, match="Must provide mean mesh") as excinfo:
            read_mesh_get_sampled_pts(
                sphere_paths[0], n_pts=10, register_to_mean_first=True, fix_mesh=False
            )
        assert type(excinfo.value) is Exception

    def test_registration_moves_the_subject_to_the_references_frame_and_size(self, sphere_paths):
        """
        Similarity = rigid + uniform scale: the cartilage sphere (r=0.5 at x=5) comes out
        at the bone's position AND size (r=1 at x=2). Subject size does not survive
        registration — the documented scale-erasure fact.
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
        assert result["icp_transform"] is not None
        registered = result["new_pts"][0]
        center = registered.mean(axis=0)
        assert np.linalg.norm(center - BONE_CENTER) < 0.1
        radii = np.linalg.norm(registered - center, axis=1)
        assert abs(radii.mean() - BONE_RADIUS) < 0.1

    def test_deprecated_kwargs_print_and_change_nothing(self, sphere_paths, capsys):
        result = read_mesh_get_sampled_pts(
            sphere_paths[0], n_pts=10, sigma=0.1, fix_mesh=False, return_scale=False, mean=3
        )
        out = capsys.readouterr().out
        assert "return_scale is deprecated" in out
        assert "mean is deprecated" in out
        assert "scale" in result  # returned despite return_scale=False: always-on

    def test_unknown_kwargs_are_swallowed_silently(self, sphere_paths, capsys):
        result = read_mesh_get_sampled_pts(
            sphere_paths[0], n_pts=10, sigma=0.1, fix_mesh=False, not_a_parameter=True
        )
        assert result is not None
        assert "not_a_parameter" not in capsys.readouterr().out

    def test_return_point_cloud_stores_the_normalized_surface(self, sphere_paths):
        result = read_mesh_get_sampled_pts(
            sphere_paths[0], n_pts=10, sigma=0.1, return_point_cloud=True, fix_mesh=False
        )
        np.testing.assert_array_equal(result["point_cloud"], result["new_pts"][0])

    def test_fix_mesh_runs_the_meshfix_wrapper(self, sphere_paths, capsys):
        result = read_mesh_get_sampled_pts(sphere_paths[0], n_pts=10, sigma=0.1, fix_mesh=True)
        assert result is not None
        assert "Fixed mesh," in capsys.readouterr().out


class TestMultiMeshReader:
    def test_any_missing_path_fails_the_whole_subject(self, sphere_paths):
        """A missing *path* is an error for the subject; a None *entry* is not (below)."""
        result = read_meshes_get_sampled_pts(
            [sphere_paths[0], "/nonexistent/mesh.vtk"],
            sigma=[0.1, 0.1],
            n_pts=[10, 10],
            fix_mesh=False,
        )
        assert result is None

    def test_a_none_surface_is_carried_as_none_placeholders(self, sphere_paths):
        result = read_meshes_get_sampled_pts(
            [sphere_paths[0], None], sigma=[0.1, 0.1], n_pts=[10, 10], fix_mesh=False
        )
        assert result["orig_pts"][1] is None
        assert result["new_pts"][1] is None
        assert result["sdf"][1] is None
        assert set(np.unique(result["pts_surface"])) == {0}
        assert result["pts"].shape[0] == 10

    def test_get_random_false_zeroes_each_surfaces_own_vertices(self, sphere_paths):
        result = read_meshes_get_sampled_pts(list(sphere_paths), get_random=False, fix_mesh=False)
        n0 = result["new_pts"][0].shape[0]
        n1 = result["new_pts"][1].shape[0]
        assert result["pts"].shape[0] == n0 + n1
        assert np.all(result["sdf"][0][:n0] == 0)
        assert np.any(result["sdf"][0][n0:] != 0)
        assert np.all(result["sdf"][1][n0:] == 0)
        np.testing.assert_array_equal(
            result["pts_surface"], np.concatenate([np.zeros(n0), np.ones(n1)]).astype(np.int64)
        )

    def test_registering_without_a_mean_mesh_raises_valueerror(self, sphere_paths):
        """The single reader raises a bare Exception for this; pinned as-is."""
        with pytest.raises(ValueError, match="Must provide mean mesh"):
            read_meshes_get_sampled_pts(
                list(sphere_paths),
                sigma=[0.1, 0.1],
                n_pts=[5, 5],
                register_to_mean_first=True,
                fix_mesh=False,
            )

    def test_the_default_frame_centers_on_mesh_to_scale_and_scales_by_all(self, sphere_paths):
        """Defaults: centered on surface 0, scaled so every surface fits the domain."""
        result = read_meshes_get_sampled_pts(
            list(sphere_paths), sigma=[0.1, 0.1], n_pts=[20, 20], fix_mesh=False
        )
        np.testing.assert_allclose(result["center"], BONE_CENTER, atol=0.02)
        np.testing.assert_allclose(result["scale"], JOINT_RADIUS_FROM_BONE, atol=0.02)
        bone_pts, cart_pts = result["new_pts"]
        assert np.linalg.norm(bone_pts.mean(axis=0)) < 1e-5
        all_radii = np.linalg.norm(np.vstack([bone_pts, cart_pts]), axis=1)
        np.testing.assert_allclose(all_radii.max(), 1.0, atol=1e-6)

    def test_scale_all_meshes_false_scales_by_the_reference_surface_only(self, sphere_paths):
        result = read_meshes_get_sampled_pts(
            list(sphere_paths),
            sigma=[0.1, 0.1],
            n_pts=[20, 20],
            scale_all_meshes=False,
            fix_mesh=False,
        )
        np.testing.assert_allclose(result["center"], BONE_CENTER, atol=0.02)
        np.testing.assert_allclose(result["scale"], BONE_RADIUS, atol=0.02)

    def test_center_all_meshes_true_centers_on_every_surface(self, sphere_paths):
        result = read_meshes_get_sampled_pts(
            list(sphere_paths),
            sigma=[0.1, 0.1],
            n_pts=[20, 20],
            scale_all_meshes=False,
            center_all_meshes=True,
            fix_mesh=False,
        )
        union = np.vstack([result["orig_pts"][0], result["orig_pts"][1]])
        expected_center = union.mean(axis=0)
        expected_scale = np.linalg.norm(result["orig_pts"][0] - expected_center, axis=1).max()
        np.testing.assert_allclose(result["center"], expected_center, atol=1e-5)
        np.testing.assert_allclose(result["scale"], expected_scale, atol=1e-5)

    def test_a_mesh_to_scale_list_scales_by_the_union(self, sphere_paths):
        result = read_meshes_get_sampled_pts(
            list(sphere_paths),
            sigma=[0.1, 0.1],
            n_pts=[20, 20],
            scale_all_meshes=False,
            mesh_to_scale=[0, 1],
            fix_mesh=False,
        )
        union = np.vstack([result["orig_pts"][0], result["orig_pts"][1]])
        expected_center = union.mean(axis=0)
        expected_scale = np.linalg.norm(union - expected_center, axis=1).max()
        np.testing.assert_allclose(result["center"], expected_center, atol=1e-5)
        np.testing.assert_allclose(result["scale"], expected_scale, atol=1e-5)

    def test_a_mesh_to_scale_list_also_drives_centering(self, sphere_paths):
        """scale_all_meshes=True + mesh_to_scale=[0, 1]: the centering points come from
        the listed surfaces (with two surfaces total, the same union as above — but a
        different branch of the frame code)."""
        result = read_meshes_get_sampled_pts(
            list(sphere_paths),
            sigma=[0.1, 0.1],
            n_pts=[20, 20],
            mesh_to_scale=[0, 1],
            fix_mesh=False,
        )
        union = np.vstack([result["orig_pts"][0], result["orig_pts"][1]])
        expected_center = union.mean(axis=0)
        expected_scale = np.linalg.norm(union - expected_center, axis=1).max()
        np.testing.assert_allclose(result["center"], expected_center, atol=1e-5)
        np.testing.assert_allclose(result["scale"], expected_scale, atol=1e-5)

    def test_a_zero_count_surface_contributes_no_points(self, sphere_paths):
        """n_pts=0 for a surface: no points drawn around it, but its SDF column is
        still computed for every point the other surfaces drew."""
        result = read_meshes_get_sampled_pts(
            list(sphere_paths), sigma=[0.1, 0.1], n_pts=[30, 0], fix_mesh=False
        )
        assert result["pts"].shape[0] == 30
        assert set(np.unique(result["pts_surface"])) == {0}
        assert result["sdf"][1] is not None
        assert result["sdf"][1].shape[0] == 30

    def test_a_supplied_icp_transform_is_used_instead_of_registering(self, sphere_paths):
        """
        The dataset's cross-combo contract (``MultiSurfaceSDFSamples.get_sample_data_dict``):
        the first sampling pass registers, every later pass passes that pass's transform
        back in, so all of a subject's points share one registration. Supplying a
        transform must skip registration -- the same object comes back out -- and land
        the surfaces in the identical frame.
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
        transform = first["icp_transform"]
        assert transform is not None

        second = read_meshes_get_sampled_pts(
            [cart_path, bone_path], icp_transform=transform, **kwargs
        )
        assert second["icp_transform"] is transform
        for surf_idx in range(2):
            np.testing.assert_array_equal(second["new_pts"][surf_idx], first["new_pts"][surf_idx])

    def test_a_none_sigma_draws_from_one_cube_around_all_surfaces(self, sphere_paths):
        """
        ``None`` sigmas draw uniformly from a single cube around every surface jointly
        (``get_buffered_cube_mins_maxs`` over the concatenated surfaces), not one cube
        per surface -- so a bone-surface draw can land far away, inside the cartilage's
        corner of the cube.
        """
        result = read_meshes_get_sampled_pts(
            list(sphere_paths),
            sigma=[None, None],
            n_pts=[200, 200],
            center_pts=False,
            norm_pts=False,
            fix_mesh=False,
        )
        union = np.vstack([result["new_pts"][0], result["new_pts"][1]])
        mins, maxs = get_buffered_cube_mins_maxs(union, 0.0)
        assert result["pts"].shape[0] == 400
        assert np.all(result["pts"] >= mins) and np.all(result["pts"] <= maxs)
        # Both surfaces' draws span the joint cube, not their own sphere's extent:
        # bone vertices stay below x=3, but bone-labeled samples reach the cartilage's
        # half of the cube.
        bone_draws = result["pts"][result["pts_surface"] == 0]
        assert bone_draws[:, 0].max() > 3.5

    def test_the_same_seed_reproduces_the_draws_and_a_different_seed_changes_them(
        self, sphere_paths
    ):
        kwargs = dict(sigma=[0.1, 0.1], n_pts=[25, 25], fix_mesh=False)
        first = read_meshes_get_sampled_pts(list(sphere_paths), seed=7, **kwargs)
        again = read_meshes_get_sampled_pts(list(sphere_paths), seed=7, **kwargs)
        other = read_meshes_get_sampled_pts(list(sphere_paths), seed=8, **kwargs)
        np.testing.assert_array_equal(again["pts"], first["pts"])
        for surf_idx in range(2):
            np.testing.assert_array_equal(again["sdf"][surf_idx], first["sdf"][surf_idx])
        assert not np.array_equal(other["pts"], first["pts"])

    def test_include_surf_in_pts_appends_the_vertices_on_the_uniform_cube_path_too(
        self, sphere_paths
    ):
        """
        Before the #17 fix this configuration could not even run: the leaked
        ``new_pts_`` was a *list* on the uniform-cube path, so the append raised
        ``ValueError`` -- the second of #17's three behaviours (executed
        determination, plan §8.0.B).
        """
        n_random = 10
        result = read_meshes_get_sampled_pts(
            list(sphere_paths),
            sigma=[None, None],
            n_pts=[n_random, n_random],
            include_surf_in_pts=True,
            center_pts=False,
            norm_pts=False,
            fix_mesh=False,
        )
        bone_verts, cart_verts = result["new_pts"]
        end_of_surface_0 = n_random + bone_verts.shape[0]
        np.testing.assert_array_equal(result["pts"][n_random:end_of_surface_0], bone_verts)
        np.testing.assert_array_equal(result["pts"][end_of_surface_0 + n_random :], cart_verts)

    def test_include_surf_in_pts_appends_each_surfaces_own_vertices(self, sphere_paths):
        n_random = 10
        result = read_meshes_get_sampled_pts(
            list(sphere_paths),
            sigma=[0.1, 0.1],
            n_pts=[n_random, n_random],
            include_surf_in_pts=True,
            fix_mesh=False,
        )
        bone_verts, cart_verts = result["new_pts"]
        end_of_surface_0 = n_random + bone_verts.shape[0]
        np.testing.assert_array_equal(result["pts"][n_random:end_of_surface_0], bone_verts)
        np.testing.assert_array_equal(result["pts"][end_of_surface_0 + n_random :], cart_verts)

    def test_deprecated_kwargs_print_and_change_nothing(self, sphere_paths, capsys):
        result = read_meshes_get_sampled_pts(
            list(sphere_paths),
            sigma=[0.1, 0.1],
            n_pts=[10, 10],
            fix_mesh=False,
            return_orig_mesh=False,
            mean=1,
        )
        out = capsys.readouterr().out
        assert "return_orig_mesh is deprecated" in out
        assert "mean is deprecated" in out
        assert "orig_mesh" in result  # returned despite return_orig_mesh=False: always-on
