"""
``reconstruct/cartilage_func.py`` — the five functions of ``DICT_VALIDATION_FUNCS``, and
the two frames above them that consume what those functions return.

Plan §8.0.N′. The module was at **19% coverage — 54 statements, 44 uncovered** — and the
uncovered set was every line of all five public functions: the suite imported it and had
never called it. ``SCOPE`` §2.5 rules it production.

``compute_recon_loss`` and ``get_mean_errors`` are tested here rather than beside
``recon_evaluation``'s other tests because both defects they carry are only reachable
through this module: a missing surface reaching the metric loop, and the ``func_`` key set
these functions produce being aggregated across subjects.

The synthetic geometry is two concentric spheres, so every reconstructed thickness comes
out at the sphere's diameter — the ray cast passes clean through. The *value* is an
artifact; only its constancy is used, and no assertion here depends on the number.
"""

import json
import os

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


class TestAnAbsentReconstructedSurface:
    """
    ``_finish_meshes`` leaves a surface's slot ``None`` when its SDF does not cross zero,
    and ``create_mesh_adaptive`` returns ``[None] * objects`` when the coarse grid finds
    no bounds. That is the ordinary state of a decoder early in training, which is when
    validation runs — so ``None`` is an input this function must survive, not a
    programming error.
    """

    @pytest.mark.parametrize("missing_index", [0, 1])
    def test_a_none_reconstructed_surface_is_scored_nan(self, missing_index):
        """
        Index 0 is the bone, index 1 the cartilage, and before the guard they failed
        differently: the bone raised ``AttributeError: 'NoneType' object has no attribute
        'GetNumberOfPoints'`` from inside pymskt, and **the cartilage killed the
        interpreter**. ``CartilageMesh(None)`` builds a 0-point mesh, ``vtkOBBTree``
        reports "Can't build OBB tree - no data available!", and the process dies —
        measured in its own child process on this checkout, **exit 139, `SIGSEGV`**.
        That is why the characterization for this one ran in a subprocess and why it is
        gone: no ``except`` catches a segfault, so there is nothing to assert in-process
        except the contract that replaces it.
        """
        orig_meshes, recon_meshes = _pair()
        recon_meshes[missing_index] = None
        result = compare_cart_thickness(orig_meshes, recon_meshes, cart_regions=(11,))
        assert set(result) == REGION_11_KEYS
        assert all(np.isnan(value) for value in result.values())

    def test_a_none_original_bone_is_scored_nan(self):
        """Today: ``KeyError: 'labels'``, raised while copying scalars off ``None``."""
        orig_meshes, recon_meshes = _pair()
        orig_meshes[0] = None
        result = compare_cart_thickness(orig_meshes, recon_meshes, cart_regions=(11,))
        assert set(result) == REGION_11_KEYS
        assert all(np.isnan(value) for value in result.values())

    def test_the_sibling_metric_path_scores_the_same_input_nan(self):
        """
        The measurement that makes the defect above a defect rather than a limitation.
        ``compute_recon_loss`` is reached from the same ``reconstruct_mesh`` call on the
        same subject, forty lines from the recon-func call, and it handles exactly this
        input: ``if mesh is not None`` at ``recon_evaluation.py:78``.
        """
        result = compute_recon_loss(
            meshes=[_plain(1.0), None],
            orig_meshes=[_plain(1.0), _plain(1.1)],
            calc_symmetric_chamfer=True,
            n_samples_chamfer=64,
        )
        assert not np.isnan(result["chamfer_0"])
        assert np.isnan(result["chamfer_1"])


class TestAMissingOriginalSurface:
    """
    ``SCOPE`` §2.5b ruled ``None`` surfaces **supported at reconstruction** on 2026-08-29,
    from ``latent_fit.py``. The fit is fine; ``compute_recon_loss`` guards the
    reconstructed mesh and reads the original unguarded one line below.
    """

    @pytest.mark.parametrize("flag", ["calc_symmetric_chamfer", "calc_assd"])
    def test_a_none_original_mesh_is_scored_nan(self, flag):
        """Before the guard: ``AttributeError: 'NoneType' object has no attribute
        'point_coords'``, one line below the guard on the reconstructed mesh."""
        result = compute_recon_loss(
            meshes=[_plain(1.0), _plain(1.1)],
            orig_meshes=[_plain(1.0), None],
            n_samples_chamfer=64,
            **{flag: True},
        )
        suffix = "chamfer" if flag == "calc_symmetric_chamfer" else "assd"
        assert not np.isnan(result[f"{suffix}_0"])
        assert np.isnan(result[f"{suffix}_1"])

    def test_the_shipped_config_asks_for_both_metrics(self):
        """
        Which is what makes the guard above production behaviour rather than an option:
        ``get_mean_errors`` is the only production caller and passes
        ``calc_symmetric_chamfer=config["chamfer"]`` and ``calc_assd=config["assd"]``.
        """
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(recon_main.__file__))),
            "configs",
            "default_config.json",
        )
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
        assert config["chamfer"] is True
        assert config["assd"] is True


class TestFuncKeysAcrossSubjects:
    """
    ``get_mean_errors`` collects the ``func_`` keys with ``if idx == 0: loss[key] = []``
    and then appends. A degenerate subject 0 contributes no ``func_`` keys at all — its
    result dict is built by hand in the ``except NoZeroLevelSetError`` branch — so the
    list is never created and subject 1 raises. It depends on the order of the validation
    set, which is the property that lets it sit unnoticed for years.

    The fake returns the whole four-key quartet because ``get_mean_errors`` parses these
    names: ``fnmatch(key, "cart_thick*_orig_mean")`` reaches across to
    ``cart_thick_{region}_recon_mean`` for a correlation, and the ``*_mean_thick_diff``
    key for an RMSE. That is the cross-module contract on the naming scheme, and the
    reason the NaN path this slice adds must return the full key set rather than fewer.
    """

    @staticmethod
    def _mean_errors(monkeypatch, degenerate_first):
        from NSM.reconstruct.main import NoZeroLevelSetError

        paths = ["subj0-mesh.vtk", "subj1-mesh.vtk"]
        degenerate = paths[0] if degenerate_first else paths[1]

        def fake_reconstruct_mesh(path=None, **kwargs):
            if path == degenerate:
                raise NoZeroLevelSetError("no zero level set")
            return {
                "mesh": [None],
                "func_cart_thick_11_orig_mean": 1.5,
                "func_cart_thick_11_recon_mean": 1.2,
                "func_cart_thick_11_mean_thick_diff": 0.3,
                "func_cart_thick_11_std_thick_diff": 0.0,
            }

        monkeypatch.setattr(recon_main, "reconstruct_mesh", fake_reconstruct_mesh)
        return recon_main.get_mean_errors(
            mesh_paths=paths,
            decoders=None,
            latent_size=4,
            recon_func=compare_cart_thickness,
        )

    def test_a_healthy_first_subject_collects_the_keys(self, monkeypatch):
        result = self._mean_errors(monkeypatch, degenerate_first=False)
        assert result["cart_thick_11_orig_mean"] == pytest.approx(1.5)

    def test_a_degenerate_first_subject_collects_them_too(self, monkeypatch):
        """Before the fix: ``KeyError: 'cart_thick_11_orig_mean'``, from the second
        subject — the one that had results."""
        result = self._mean_errors(monkeypatch, degenerate_first=True)
        assert result["cart_thick_11_orig_mean"] == pytest.approx(1.5)


class TestRegionsLabel:
    """
    The transfer honours ``regions_label``; the read ignores it. pymskt's
    ``get_cart_thickness_mean``/``_std`` open ``self.get_scalar("labels")`` with the name
    hardcoded, so no other value can work — and no caller passes one:
    ``get_mean_errors`` invokes these functions with two positional arguments.
    """

    def test_the_default_scores_normally(self):
        orig_meshes, recon_meshes = _pair()
        result = compare_cart_thickness(orig_meshes, recon_meshes, cart_regions=(11,))
        assert set(result) == REGION_11_KEYS
        assert not any(np.isnan(value) for value in result.values())

    @pytest.mark.parametrize("also_label_it_labels", [False, True])
    def test_a_non_default_name_is_refused_by_name(self, also_label_it_labels):
        """
        Both arrangements raised ``KeyError: 'labels'`` before, from opposite sides. With
        the original carrying only the alternative name it is the original's read that
        fails; with the original carrying **both**, the read of the original succeeds and
        the copy lands on the reconstruction under the caller's name, so it is the
        reconstruction's read that fails. There is no arrangement that works.
        """
        orig_bone = _original_bone(label_name="cart_regions")
        if also_label_it_labels:
            orig_bone.point_data["labels"] = orig_bone.point_data["cart_regions"]
        with pytest.raises(ValueError, match="regions_label"):
            compare_cart_thickness(
                [orig_bone, _plain(1.1)],
                [_plain(1.0), _plain(1.1)],
                cart_regions=(11,),
                regions_label="cart_regions",
            )


class TestTheOriginalCartilageIsNeverRead:
    """
    ``orig_bone, orig_cart = orig_meshes`` and ``orig_cart`` is never referenced again.
    The original's thickness is read off the array it arrived with; only the
    reconstruction's is computed at this call. ``CLAUDE.md`` and #20 both say the fix for
    an unread argument is to delete it, not to honour it — honouring this one, by
    computing the original's thickness here, would move every ``orig_mean`` the function
    has ever reported.
    """

    @pytest.mark.parametrize("substitute", [None, "not a mesh at all", 7])
    def test_anything_may_stand_in_for_it(self, substitute):
        orig_meshes, recon_meshes = _pair()
        expected = compare_cart_thickness(orig_meshes, recon_meshes, cart_regions=(11,))

        orig_meshes, recon_meshes = _pair()
        orig_meshes[1] = substitute
        assert compare_cart_thickness(orig_meshes, recon_meshes, cart_regions=(11,)) == expected

    def test_an_original_without_thickness_cannot_be_scored(self):
        """
        The consequence of reading rather than computing, and the one failure mode this
        slice leaves as it is: the message names the array, which is the thing to go and
        put on the mesh.
        """
        orig_meshes, recon_meshes = _pair()
        orig_meshes[0] = _original_bone(with_thickness=False)
        with pytest.raises(KeyError, match="thickness"):
            compare_cart_thickness(orig_meshes, recon_meshes, cart_regions=(11,))


class TestTheDefaultRegionSet:
    def test_it_is_the_femur_entry(self):
        """
        ``CART_REGIONS`` is a second copy of ``CART_REGIONS_DICT["femur"]``, with the
        tibial and patellar indices commented out above it. Asserted equal so that
        making it *be* that entry (commit 8) is visibly inert.
        """
        assert tuple(CART_REGIONS) == tuple(CART_REGIONS_DICT["femur"])

    def test_a_tibial_pair_scores_every_femoral_region_nan(self):
        """
        ``DICT_VALIDATION_FUNCS`` exposes the bare ``compare_cart_thickness``, so a config
        naming it for a tibia or patella model takes this default and scores 20 NaNs —
        with pymskt's ``UserWarning`` and a ``print`` as the only signal, and
        ``get_mean_errors`` averaging them into the logged metric.
        """
        orig_meshes, recon_meshes = _pair(label=2)
        result = compare_cart_thickness(orig_meshes, recon_meshes)
        assert len(result) == 20
        assert all(np.isnan(value) for value in result.values())


class TestTheJointWrappers:
    """
    The three single-joint functions are the same three lines and differ only in the
    region set they ask for — which is the whole of what they are for, since the bare
    ``compare_cart_thickness`` defaults to the femur's.
    """

    @pytest.mark.parametrize("joint", ["femur", "tibia", "patella"])
    def test_each_scores_its_own_joint(self, joint):
        wrapper = {
            "femur": compare_cart_thickness_femur,
            "tibia": compare_cart_thickness_tibia,
            "patella": compare_cart_thickness_patella,
        }[joint]
        regions = CART_REGIONS_DICT[joint]
        orig_meshes, recon_meshes = _pair(label=regions[0])
        result = wrapper(orig_meshes, recon_meshes)

        assert len(result) == 4 * len(regions)
        assert not np.isnan(result[f"func_cart_thick_{regions[0]}_orig_mean"])


class TestTheMeshListLength:
    """
    Every wrapper slices ``[:2]`` and nothing checks what it sliced from. The three
    single-joint wrappers are the same three lines and differ only in ``cart_regions``.
    """

    @staticmethod
    def _whole_joint_lists(n_pairs):
        orig_meshes, recon_meshes = [], []
        for label in (11, 2, 4)[:n_pairs]:
            orig_meshes += [_original_bone(label), _plain(1.1)]
            recon_meshes += [_plain(1.0), _plain(1.1)]
        return orig_meshes, recon_meshes

    def test_a_whole_joint_list_into_the_tibia_wrapper_is_refused(self):
        """
        Before the check it returned eight NaNs and exited 0: ``[:2]`` took the
        **femur's** pair and scored it against the tibial region indices. The silent
        case, and the reason the check is worth its lines.
        """
        orig_meshes, recon_meshes = self._whole_joint_lists(3)
        with pytest.raises(ValueError, match="6"):
            compare_cart_thickness_tibia(orig_meshes, recon_meshes)

    @pytest.mark.parametrize("n_meshes", [1, 3])
    def test_a_pair_of_the_wrong_length_is_refused_by_name(self, n_meshes):
        """Before the check: ``ValueError: not enough values to unpack``, or ``too many``,
        naming neither the function nor the count it wanted."""
        meshes = [_original_bone()] + [_plain(1.1)] * (n_meshes - 1)
        with pytest.raises(ValueError, match="compare_cart_thickness"):
            compare_cart_thickness(meshes, list(meshes), cart_regions=(11,))

    def test_the_four_surface_layout_is_refused_by_name(self):
        """
        ``["bone", "cart", "med_men", "lat_men"]`` is the four-surface femur layout
        ``CLAUDE.md`` documents. Into the whole-joint function it gave
        ``KeyError: 'labels'``, from treating the medial meniscus as the tibia's bone — a
        message that named neither the count nor the function.
        """
        orig_meshes = [_original_bone(11), _plain(1.1), _plain(1.1), _plain(1.1)]
        recon_meshes = [_plain(1.0), _plain(1.1), _plain(1.1), _plain(1.1)]
        with pytest.raises(ValueError, match="6"):
            compare_cart_thickness_whole_joint(orig_meshes, recon_meshes)

    def test_six_meshes_score_all_three_joints(self):
        orig_meshes, recon_meshes = self._whole_joint_lists(3)
        result = compare_cart_thickness_whole_joint(orig_meshes, recon_meshes)
        assert len(result) == 4 * (5 + 2 + 1)
        assert not np.isnan(result["func_cart_thick_11_orig_mean"])
        assert not np.isnan(result["func_cart_thick_2_orig_mean"])
        assert not np.isnan(result["func_cart_thick_4_orig_mean"])


class TestTheCoercion:
    def test_the_mesh_property_returns_the_object_itself(self):
        """
        Which is what makes ``BoneMesh(m.mesh)`` and ``BoneMesh(m)`` the same call, and
        the ``elif isinstance(m, pymskt.mesh.Mesh)`` branch a no-op that ``print``s
        pymskt's "this property is redundant" notice — three times per subject, on
        stdout, in the branch production takes every time.
        """
        mesh = _plain(1.0)
        assert mesh.mesh is mesh

    def test_the_scored_values_are_what_the_geometry_gives(self):
        """
        The pin that made deleting the ``.mesh`` branch provably inert. A
        production-shaped call — a plain ``mskt.mesh.Mesh`` on both sides, which is what
        ``create_mesh`` and ``read_meshes`` produce — was run before and after the
        deletion and the two result dicts differed in nothing but the three lines of
        stdout the branch printed. These are those numbers, kept so a later change to the
        coercion cannot move them quietly.

        Both regions read the same reconstructed thickness because the reconstruction is
        a sphere: the ray passes clean through and measures the diameter, 2 × 1.1.
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

    def test_a_bonemesh_argument_and_a_plain_mesh_argument_score_the_same(self):
        orig_meshes, recon_meshes = _pair()
        from_plain = compare_cart_thickness(orig_meshes, recon_meshes, cart_regions=(11,))

        orig_meshes, recon_meshes = _pair()
        recon_meshes[0] = BoneMesh(_sphere(1.0))
        from_bone = compare_cart_thickness(orig_meshes, recon_meshes, cart_regions=(11,))
        assert from_bone == from_plain

    def test_a_bonemesh_argument_is_mutated_and_a_plain_mesh_is_not(self):
        """
        The one contract this slice documents rather than changes. A ``BoneMesh``
        argument keeps its identity through the coercion and comes back carrying
        ``labels`` and ``thickness (mm)``; a plain ``Mesh`` is copied and comes back
        untouched. No in-repo caller passes a ``BoneMesh`` — ``create_mesh`` and
        ``read_meshes`` both produce the plain kind — so making it uniform would be a
        behaviour change with no reported instance.
        """
        orig_meshes, recon_meshes = _pair()
        plain = recon_meshes[0]
        compare_cart_thickness(orig_meshes, recon_meshes, cart_regions=(11,))
        assert "thickness (mm)" not in plain.point_data

        orig_meshes, recon_meshes = _pair()
        bone = BoneMesh(_sphere(1.0))
        recon_meshes[0] = bone
        compare_cart_thickness(orig_meshes, recon_meshes, cart_regions=(11,))
        assert {"labels", "thickness (mm)"} <= set(bone.point_data.keys())
