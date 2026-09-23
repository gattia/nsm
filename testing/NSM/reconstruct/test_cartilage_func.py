"""
``reconstruct/cartilage_func.py``: the five ``DICT_VALIDATION_FUNCS`` and the two frames
above them that consume what they return. ``SCOPE.md`` §2.5 rules the module production.

The geometry is two concentric spheres, so every reconstructed thickness is the sphere's
diameter. No assertion depends on that value except the pinned numbers in
``TestTheCoercion``.
"""

import inspect

import numpy as np
import pytest
import pyvista as pv
from pymskt.mesh import BoneMesh, Mesh

import NSM.reconstruct.main as recon_main
from NSM.reconstruct.cartilage_func import (
    CART_REGIONS,
    CART_REGIONS_DICT,
    compare_cart_thickness,
    compare_cart_thickness_femur,
    compare_cart_thickness_patella,
    compare_cart_thickness_tibia,
    compare_cart_thickness_whole_joint,
)
from NSM.reconstruct.recon_evaluation import compute_recon_loss

#: The four keys ``compare_cart_thickness`` records for one region. Written out rather
#: than built, so a change to the naming scheme is visible here as a diff.
REGION_11_KEYS = {
    "func_cart_thick_11_orig_mean",
    "func_cart_thick_11_recon_mean",
    "func_cart_thick_11_mean_thick_diff",
    "func_cart_thick_11_std_thick_diff",
}


def _sphere(radius, resolution=8):
    return pv.Sphere(
        radius=radius, theta_resolution=resolution, phi_resolution=resolution
    ).triangulate()


def _plain(radius=1.0):
    """A bare ``mskt.mesh.Mesh`` — what ``create_mesh`` and ``read_meshes`` both return."""
    return Mesh(_sphere(radius))


def _original_bone(label=11, thickness=1.5, label_name="labels", with_thickness=True):
    """
    An original bone as this module requires one: carrying its region labels **and** its
    thickness already. The original's thickness is read, never computed here.
    """
    mesh = Mesh(_sphere(1.0))
    n_points = mesh.GetNumberOfPoints()
    mesh.point_data[label_name] = np.full(n_points, label, dtype=np.int64)
    if with_thickness:
        mesh.point_data["thickness (mm)"] = np.full(n_points, thickness, dtype=float)
    return mesh


def _pair(label=11):
    """``(orig_meshes, recon_meshes)`` for one bone/cartilage pair, both scoreable."""
    return [_original_bone(label), _plain(1.1)], [_plain(1.0), _plain(1.1)]


def test_an_absent_surface_on_either_side_is_scored_nan():
    """
    A decoder early in training leaves a surface ``None``, and validation runs then. Before
    the guard a ``None`` reconstructed bone raised from pymskt, a ``None`` cartilage killed
    the interpreter (``CartilageMesh(None)``, exit 139, SIGSEGV), and a ``None`` original
    bone raised ``KeyError: 'labels'``.
    """
    for side, index in (("recon", 0), ("recon", 1), ("orig", 0)):
        orig_meshes, recon_meshes = _pair()
        (recon_meshes if side == "recon" else orig_meshes)[index] = None
        result = compare_cart_thickness(orig_meshes, recon_meshes, cart_regions=(11,))
        assert set(result) == REGION_11_KEYS
        assert all(np.isnan(value) for value in result.values()), (side, index)

    # compute_recon_loss, called on the same subject, scores a missing surface NaN too.
    result = compute_recon_loss(
        meshes=[_plain(1.0), None],
        orig_meshes=[_plain(1.0), _plain(1.1)],
        calc_symmetric_chamfer=True,
        n_samples_chamfer=64,
    )
    assert not np.isnan(result["chamfer_0"]) and np.isnan(result["chamfer_1"])


def test_func_keys_are_collected_whichever_subject_is_degenerate(monkeypatch):
    """
    ``get_mean_errors`` created each ``func_`` list at subject 0. A degenerate subject 0
    has no ``func_`` keys, so the healthy subject 1 raised ``KeyError``. The fake returns
    all four keys because ``get_mean_errors`` parses the names.
    """
    from NSM.reconstruct.main import NoZeroLevelSetError

    for degenerate in ("subj0-mesh.vtk", "subj1-mesh.vtk"):

        def fake(path=None, **kwargs):
            if path == degenerate:
                raise NoZeroLevelSetError("no zero level set")
            return {
                "mesh": [None],
                "func_cart_thick_11_orig_mean": 1.5,
                "func_cart_thick_11_recon_mean": 1.2,
                "func_cart_thick_11_mean_thick_diff": 0.3,
                "func_cart_thick_11_std_thick_diff": 0.0,
            }

        monkeypatch.setattr(recon_main, "reconstruct_mesh", fake)
        result = recon_main.get_mean_errors(
            mesh_paths=["subj0-mesh.vtk", "subj1-mesh.vtk"],
            decoders=None,
            latent_size=4,
            recon_func=compare_cart_thickness,
        )
        assert result["cart_thick_11_orig_mean"] == pytest.approx(1.5)


def test_the_region_array_must_be_called_labels():
    """
    pymskt reads ``labels``. The ``regions_label`` parameter, removed in v0.4.0, never
    worked with any other value. The ``KeyError`` names the array to rename.
    """
    orig_meshes, recon_meshes = _pair()
    result = compare_cart_thickness(orig_meshes, recon_meshes, cart_regions=(11,))
    assert set(result) == REGION_11_KEYS
    assert not any(np.isnan(value) for value in result.values())

    for function in (
        compare_cart_thickness,
        compare_cart_thickness_tibia,
        compare_cart_thickness_patella,
        compare_cart_thickness_femur,
        compare_cart_thickness_whole_joint,
    ):
        assert "regions_label" not in inspect.signature(function).parameters

    renamed = [_original_bone(label_name="cart_regions"), _plain(1.1)]
    with pytest.raises(KeyError, match="labels"):
        compare_cart_thickness(renamed, [_plain(1.0), _plain(1.1)], cart_regions=(11,))


class TestTheOriginalCartilageIsNeverRead:
    """
    ``orig_cart`` is unpacked and never used: the original's thickness is read off the
    original bone. The slot stays because both lists share one layout (``SCOPE.md`` §2.5).
    """

    def test_anything_may_stand_in_for_it_but_the_bone_needs_its_thickness(self):
        expected = compare_cart_thickness(*_pair(), cart_regions=(11,))
        for substitute in (None, "not a mesh at all", 7):
            orig_meshes, recon_meshes = _pair()
            orig_meshes[1] = substitute
            assert compare_cart_thickness(orig_meshes, recon_meshes, cart_regions=(11,)) == (
                expected
            )

        orig_meshes, recon_meshes = _pair()
        orig_meshes[0] = _original_bone(with_thickness=False)
        with pytest.raises(KeyError, match="thickness"):
            compare_cart_thickness(orig_meshes, recon_meshes, cart_regions=(11,))


def test_each_function_scores_its_own_joints_regions():
    """
    The bare ``compare_cart_thickness`` defaults to the femur's regions, so a tibial pair
    scores 20 NaNs with only pymskt's warning to show for it. The wrappers exist to ask
    for the right set.
    """
    assert tuple(CART_REGIONS) == tuple(CART_REGIONS_DICT["femur"])
    result = compare_cart_thickness(*_pair(label=2))
    assert len(result) == 20 and all(np.isnan(value) for value in result.values())

    for joint, wrapper in (
        ("femur", compare_cart_thickness_femur),
        ("tibia", compare_cart_thickness_tibia),
        ("patella", compare_cart_thickness_patella),
    ):
        regions = CART_REGIONS_DICT[joint]
        result = wrapper(*_pair(label=regions[0]))
        assert len(result) == 4 * len(regions)
        assert not np.isnan(result[f"func_cart_thick_{regions[0]}_orig_mean"])


class TestTheMeshListLength:
    """
    Every wrapper sliced ``[:2]`` and nothing checked the length. The functions are
    fixed-layout by ruling (``SCOPE.md`` §2.5): a multi-surface layout gets its own
    ``DICT_VALIDATION_FUNCS`` entry when it needs one.
    """

    @staticmethod
    def _whole_joint(n_pairs):
        orig_meshes, recon_meshes = [], []
        for label in (11, 2, 4)[:n_pairs]:
            orig_meshes += [_original_bone(label), _plain(1.1)]
            recon_meshes += [_plain(1.0), _plain(1.1)]
        return orig_meshes, recon_meshes

    def test_six_meshes_score_all_three_joints(self):
        result = compare_cart_thickness_whole_joint(*self._whole_joint(3))
        assert len(result) == 4 * (5 + 2 + 1)
        for region in (11, 2, 4):
            assert not np.isnan(result[f"func_cart_thick_{region}_orig_mean"])

    def test_a_list_of_the_wrong_length_is_refused_by_name(self):
        """
        A whole-joint list into the tibia wrapper returned eight NaNs and exit 0: ``[:2]``
        took the femur's pair. The four-surface femur layout (bone, cart, menisci) gave
        ``KeyError: 'labels'`` in the whole-joint function. Into the femur wrapper it
        sliced the right pair and scored correctly, and it is refused anyway, by ruling.
        """
        with pytest.raises(ValueError, match="6"):
            compare_cart_thickness_tibia(*self._whole_joint(3))
        for n_meshes in (1, 3):
            meshes = [_original_bone()] + [_plain(1.1)] * (n_meshes - 1)
            with pytest.raises(ValueError, match="compare_cart_thickness"):
                compare_cart_thickness(meshes, list(meshes), cart_regions=(11,))
        four_orig = [_original_bone(11), _plain(1.1), _plain(0.3), _plain(0.3)]
        four_recon = [_plain(1.0), _plain(1.1), _plain(0.3), _plain(0.3)]
        with pytest.raises(ValueError, match="6"):
            compare_cart_thickness_whole_joint(four_orig, four_recon)
        with pytest.raises(ValueError, match="got 4"):
            compare_cart_thickness_femur(four_orig, four_recon)


class TestTheCoercion:
    def test_the_scored_values_are_what_the_geometry_gives(self):
        """
        Deleting the redundant ``.mesh`` coercion branch changed nothing but three printed
        lines per subject. These are the values before and after. The reconstruction is a
        sphere, so every ray measures its diameter, 2 x 1.1.
        """
        orig_bone = Mesh(_sphere(1.0))
        n_points = orig_bone.GetNumberOfPoints()
        labels = np.full(n_points, 11, dtype=np.int64)
        labels[n_points // 2 :] = 12
        orig_bone.point_data["labels"] = labels
        orig_bone.point_data["thickness (mm)"] = np.linspace(1.0, 2.0, n_points)

        result = compare_cart_thickness(
            [orig_bone, _plain(1.1)], [_plain(1.0), _plain(1.1)], cart_regions=(11, 12)
        )
        assert result["func_cart_thick_11_orig_mean"] == pytest.approx(1.2448979591836733)
        assert result["func_cart_thick_11_recon_mean"] == pytest.approx(2.1999999960322536)
        assert result["func_cart_thick_11_mean_thick_diff"] == pytest.approx(-0.9551020368485803)
        assert result["func_cart_thick_11_std_thick_diff"] == pytest.approx(0.1471653092003338)
        assert result["func_cart_thick_12_orig_mean"] == pytest.approx(1.7551020408163265)
        assert result["func_cart_thick_12_recon_mean"] == pytest.approx(2.1999999902320386)

    def test_a_bonemesh_scores_the_same_and_is_the_only_argument_mutated(self):
        """
        A ``BoneMesh`` keeps its identity and comes back carrying ``labels`` and
        ``thickness (mm)``. A plain ``Mesh``, what ``create_mesh`` returns, is copied and
        comes back untouched. No in-repo caller passes a ``BoneMesh``.
        """
        orig_meshes, recon_meshes = _pair()
        plain = recon_meshes[0]
        from_plain = compare_cart_thickness(orig_meshes, recon_meshes, cart_regions=(11,))
        assert "thickness (mm)" not in plain.point_data

        orig_meshes, recon_meshes = _pair()
        bone = BoneMesh(_sphere(1.0))
        recon_meshes[0] = bone
        assert compare_cart_thickness(orig_meshes, recon_meshes, cart_regions=(11,)) == from_plain
        assert {"labels", "thickness (mm)"} <= set(bone.point_data.keys())
