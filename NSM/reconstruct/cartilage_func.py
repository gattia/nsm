"""
Cartilage-thickness agreement between an original knee and its NSM reconstruction.

These five functions are the whole of ``train_deep_sdf.DICT_VALIDATION_FUNCS``, chosen per
run by the config key ``recon_val_func_name`` and called once per validation subject by
``reconstruct_mesh`` as ``func(orig_meshes, recon_meshes)``. ``docs/SCOPE.md`` §2.5 rules
the module production: it was critical to the ShapeMedKnee paper's validation, and it owns
the only region-index maps in the repository.

**The two sides are not measured the same way, and that is the contract.** The
reconstruction's thickness is computed here, by pymskt's ray cast from the bone surface
into the cartilage. The original's is *read* off the ``thickness (mm)`` array the original
bone already carries -- nothing here computes it, and the original cartilage mesh these
functions are handed is never looked at. Computing it here instead would move every
``orig_mean`` the library has ever reported, so what an original bone arrives carrying is
what it is scored on. An original without that array raises ``KeyError: 'thickness (mm)'``,
which names the thing to go and put on the mesh.
"""

import logging

import numpy as np
from pymskt.mesh import BoneMesh, CartilageMesh

logger = logging.getLogger(__name__)

CART_REGIONS_DICT = {
    "tibia": (2, 3),  # medial tibial, lateral tibial
    "patella": (4,),
    # trochlea, medial and lateral weight-bearing, medial and lateral posterior
    "femur": (11, 12, 13, 14, 15),
}

#: What ``compare_cart_thickness`` scores when the caller names no regions -- the femur's
#: subregions, not a joint-neutral set. A config naming the bare ``compare_cart_thickness``
#: in ``recon_val_func_name`` for a tibia or patella model gets NaN for every region.
CART_REGIONS = CART_REGIONS_DICT["femur"]


def _require_meshes(caller, n_expected, layout, orig_meshes, recon_meshes):
    """
    Refuse a mesh list of the wrong length, at the function that slices it.

    Every wrapper took ``[:2]`` and nothing checked what it sliced from, so a list of the
    wrong length was scored silently or died unnamed depending on which one it was --
    ``testing/NSM/reconstruct/test_cartilage_func.py::TestTheMeshListLength`` has the
    measurements.
    """
    for name, meshes in (("orig_meshes", orig_meshes), ("recon_meshes", recon_meshes)):
        if len(meshes) != n_expected:
            raise ValueError(
                f"{caller} needs exactly {n_expected} meshes in {name}, in the order "
                f"({layout}); got {len(meshes)}. These validators are fixed-layout by "
                "design -- a model with another surface layout needs its own entry in "
                "DICT_VALIDATION_FUNCS (docs/SCOPE.md section 2.5 has the ruling)."
            )


def compare_cart_thickness_tibia(orig_meshes, recon_meshes, regions_label="labels"):
    """``compare_cart_thickness`` over the two tibial plateaus, for a tibia model."""
    _require_meshes(
        "compare_cart_thickness_tibia",
        2,
        "tibia bone, tibia cartilage",
        orig_meshes,
        recon_meshes,
    )
    return compare_cart_thickness(
        orig_meshes,
        recon_meshes,
        cart_regions=CART_REGIONS_DICT["tibia"],
        regions_label=regions_label,
    )


def compare_cart_thickness_patella(orig_meshes, recon_meshes, regions_label="labels"):
    """``compare_cart_thickness`` over the patellar cartilage, for a patella model."""
    _require_meshes(
        "compare_cart_thickness_patella",
        2,
        "patella bone, patella cartilage",
        orig_meshes,
        recon_meshes,
    )
    return compare_cart_thickness(
        orig_meshes,
        recon_meshes,
        cart_regions=CART_REGIONS_DICT["patella"],
        regions_label=regions_label,
    )


def compare_cart_thickness_femur(orig_meshes, recon_meshes, regions_label="labels"):
    """
    ``compare_cart_thickness`` over the five femoral subregions, for a femur model.

    The same regions as the bare ``compare_cart_thickness``, which defaults to them --
    this one says so in its name.
    """
    _require_meshes(
        "compare_cart_thickness_femur",
        2,
        "femur bone, femur cartilage",
        orig_meshes,
        recon_meshes,
    )
    return compare_cart_thickness(
        orig_meshes,
        recon_meshes,
        cart_regions=CART_REGIONS_DICT["femur"],
        regions_label=regions_label,
    )


def compare_cart_thickness_whole_joint(orig_meshes, recon_meshes, regions_label="labels"):
    """
    All three joints of a six-surface knee model, scored as three independent pairs.

    The six meshes are ``(femur bone, femur cartilage, tibia bone, tibia cartilage,
    patella bone, patella cartilage)``. That order is this function's alone -- it is
    declared nowhere else in the repository, ``mesh_names`` being the only other place a
    model's surface order is written down, and the two are not checked against each other.
    The returned keys cannot collide: the three joints' region indices are disjoint.
    """
    _require_meshes(
        "compare_cart_thickness_whole_joint",
        6,
        "femur bone, femur cartilage, tibia bone, tibia cartilage, "
        "patella bone, patella cartilage",
        orig_meshes,
        recon_meshes,
    )

    dict_results = {}

    fem_results = compare_cart_thickness(
        orig_meshes[:2],
        recon_meshes[:2],
        cart_regions=CART_REGIONS_DICT["femur"],
        regions_label=regions_label,
    )

    tib_results = compare_cart_thickness(
        orig_meshes[2:4],
        recon_meshes[2:4],
        cart_regions=CART_REGIONS_DICT["tibia"],
        regions_label=regions_label,
    )

    pat_results = compare_cart_thickness(
        orig_meshes[4:6],
        recon_meshes[4:6],
        cart_regions=CART_REGIONS_DICT["patella"],
        regions_label=regions_label,
    )

    dict_results.update(fem_results)
    dict_results.update(tib_results)
    dict_results.update(pat_results)

    return dict_results


def _as_mesh(mesh_class, mesh):
    """
    Coerce to ``mesh_class``, keeping the object when it already is one.

    Two branches, not three: the third read ``elif isinstance(mesh, pymskt.mesh.Mesh):
    mesh_class(mesh.mesh)``, and ``Mesh.mesh`` **returns self** while printing "this
    property is redundant" -- the same call, plus stdout, in the branch production takes
    every time.

    A mesh that already is ``mesh_class`` keeps its identity and is therefore **mutated**:
    it gains ``labels``, ``thickness (mm)`` and a ``list_cartilage_meshes``. Anything else
    is copied. Nothing in NSM passes the former.
    """
    if isinstance(mesh, mesh_class):
        return mesh
    return mesh_class(mesh)


def _region_keys(cart_region):
    """
    The four keys recorded for one cartilage region, in one place.

    ``get_mean_errors`` parses these names: it matches ``cart_thick*_orig_mean`` and then
    reaches for ``..._recon_mean`` and ``..._mean_thick_diff`` by name, to build a
    correlation and an RMSE. A result dict missing any of the four raises there, two
    frames from here, so the scored path and the NaN path build their keys from this
    function rather than each writing the set out.
    """
    return (
        f"func_cart_thick_{cart_region}_orig_mean",
        f"func_cart_thick_{cart_region}_recon_mean",
        f"func_cart_thick_{cart_region}_mean_thick_diff",
        f"func_cart_thick_{cart_region}_std_thick_diff",
    )


def compare_cart_thickness(
    orig_meshes,
    recon_meshes,
    cart_regions=CART_REGIONS,
    regions_label="labels",
):
    """
    Score one bone/cartilage pair: mean and standard-deviation thickness per region.

    Returns four keys per region of ``cart_regions`` -- the original's mean, the
    reconstruction's mean, and the two differences ``original - reconstruction``. See the
    module docstring for which side is computed and which is read. ``cart_regions``
    defaults to the **femur's** subregions, so a tibia or patella model reaching this
    function by name scores NaN for every region; the three joint-named wrappers above
    exist to pick the right set.

    A surface the decoder did not produce -- ``None`` in either reconstructed slot, or in
    the original bone's -- scores the whole key set NaN, the same answer
    ``compute_recon_loss`` gives the same subject.

    ``regions_label`` is refused unless it is ``"labels"``: pymskt's readers hardcode that
    name, so no other value has ever worked. It is kept only until the v0.4.0 signature
    change (plan §8.0.S) removes it.

    A ``BoneMesh`` or ``CartilageMesh`` argument is **mutated** -- see ``_as_mesh``.
    """
    if regions_label != "labels":
        raise ValueError(
            f"regions_label={regions_label!r} cannot work. The transfer honours it, but "
            "pymskt's BoneMesh.get_cart_thickness_mean and .get_cart_thickness_std both "
            'read get_scalar("labels") with the name hardcoded, so the region array must '
            'be called "labels" on the original mesh and on the reconstruction. Rename '
            "the array on the original mesh instead."
        )

    _require_meshes("compare_cart_thickness", 2, "bone, cartilage", orig_meshes, recon_meshes)

    orig_bone, orig_cart = orig_meshes
    recon_bone, recon_cart = recon_meshes

    # `mesh.main._finish_meshes` leaves the slot None when a surface's SDF does not cross
    # zero. Handing that on builds a 0-point mesh and vtkOBBTree takes the interpreter
    # down with SIGSEGV, which nothing upstream can catch.
    absent = [
        name
        for name, mesh in (
            ("the original bone", orig_bone),
            ("the reconstructed bone", recon_bone),
            ("the reconstructed cartilage", recon_cart),
        )
        if mesh is None
    ]
    if absent:
        logger.warning(
            "Scoring every cartilage region NaN: %s absent from this subject.",
            ", ".join(absent),
        )
        return {key: np.nan for region in cart_regions for key in _region_keys(region)}

    recon_bone = _as_mesh(BoneMesh, recon_bone)
    recon_cart = _as_mesh(CartilageMesh, recon_cart)
    orig_bone = _as_mesh(BoneMesh, orig_bone)

    recon_bone.copy_scalars_from_other_mesh_to_current(orig_bone, orig_scalars_name=regions_label)

    recon_bone.list_cartilage_meshes = recon_cart
    recon_bone.calc_cartilage_thickness()

    dict_results = {}

    for cart_region in cart_regions:
        orig_mean = orig_bone.get_cart_thickness_mean(cart_region)
        recon_mean = recon_bone.get_cart_thickness_mean(cart_region)
        orig_std = orig_bone.get_cart_thickness_std(cart_region)
        recon_std = recon_bone.get_cart_thickness_std(cart_region)

        dict_results.update(
            zip(
                _region_keys(cart_region),
                (orig_mean, recon_mean, orig_mean - recon_mean, orig_std - recon_std),
            )
        )

    return dict_results
