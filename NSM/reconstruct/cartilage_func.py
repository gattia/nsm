import logging

import numpy as np
from pymskt.mesh import BoneMesh, CartilageMesh
from scipy.stats import entropy

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

    Every wrapper took ``[:2]`` and nothing checked what it sliced from. Six meshes into a
    single-joint wrapper scored the **femur's** pair against that joint's region indices
    and returned NaN for every one, exit 0; four meshes -- the ``["bone", "cart",
    "med_men", "lat_men"]`` layout -- into the whole-joint function treated the medial
    meniscus as the tibia's bone and gave ``KeyError: 'labels'``. Neither named the count,
    the argument or the function.
    """
    for name, meshes in (("orig_meshes", orig_meshes), ("recon_meshes", recon_meshes)):
        if len(meshes) != n_expected:
            raise ValueError(
                f"{caller} needs exactly {n_expected} meshes in {name}, in the order "
                f"({layout}); got {len(meshes)}."
            )


def compare_cart_thickness_tibia(orig_meshes, recon_meshes, regions_label="labels"):
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

    Two branches, not three. The third used to read ``elif isinstance(mesh,
    pymskt.mesh.Mesh): mesh_class(mesh.mesh)``, and ``Mesh.mesh`` **returns self** while
    printing "this property is redundant" -- so it was the fallback call plus a line of
    stdout, in the branch production takes every time: ``create_mesh`` and ``read_meshes``
    both produce the plain ``Mesh``, three of them per subject per validation epoch.

    A mesh that already is ``mesh_class`` keeps its identity and is therefore **mutated**
    by the caller, gaining ``labels``, ``thickness (mm)`` and a ``list_cartilage_meshes``;
    anything else is copied and the caller's object is untouched. Nothing in NSM passes
    the former.
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

    # A surface whose SDF never crosses zero comes back as None (`mesh.main._finish_meshes`),
    # which is the ordinary state of a decoder early in training -- when validation runs.
    # Handing that None on builds a 0-point mesh and vtkOBBTree takes the interpreter down
    # with SIGSEGV, which no `except` upstream can catch. compute_recon_loss scores the
    # same subject NaN; this does the same.
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

    # transfer region scalars to recon_bone
    # should add 'labels' to the reconned bone (these are cartialge regions)
    recon_bone.copy_scalars_from_other_mesh_to_current(orig_bone, orig_scalars_name=regions_label)

    # compute cart thickness for bone
    # this should add a new caritalge thickness array to bone - test to make sure doesnt cause issues.
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

    # # Compute KL divergence between two distributions
    # thickness_kld = entropy(orig_array, qk=recon_array)

    # dict_results['func_thickness_kld'] = thickness_kld

    return dict_results
