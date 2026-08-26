"""
What the ``mesh/`` package promises about its inputs, and what it does with the ones it
was never meant to take.

Plan §8.0.I. Four independent contracts, one file, because they share the mesh fixtures
and because splitting them would hide that they are one shape: a function that accepts an
input it cannot handle and returns a plausible number instead of refusing.

1. **Face arrays** (#57). Five sites reshape a VTK-style face array with nothing checking
   the cell type. Whether that raises or silently fabricates triangles depends on the cell
   count mod 3 or mod 4 -- see ``test_reshape_success_is_a_modular_coincidence``, which
   pins the measurement the plan statement rests on.
2. **The ``use_vtk`` twins** (#60). ``sdf_grid_to_mesh`` and ``sdf_grid_to_mesh_vtk`` are
   swapped by one boolean and take different inputs under different defaults.
3. **The adaptive fallback grid** (#60). It is built from two parameters that disagree.
4. **Refusal versus invention** (#54). ``get_target_cells`` raises ``UnboundLocalError``
   on its own defaults; ``score_correspondence`` substitutes the warped mesh for a missing
   source and reports the resulting number as a measurement.

Strict xfails mark the ones NSM does not honour yet. Each is retired by the commit that
fixes it.
"""

import numpy as np
import pytest
import pyvista as pv
import torch

import NSM.mesh.main as mesh_main
from NSM.mesh.correspondence_metrics import (
    foldover_count,
    score_correspondence,
    self_intersection_count,
)
from NSM.mesh.interpolate import build_mesh_laplacian, compute_feature_mask
from NSM.mesh.main import create_mesh_adaptive, sdf_grid_to_mesh, sdf_grid_to_mesh_vtk
from NSM.mesh.refine_mesh import get_faces, get_target_cells, subdivide_large_triangles

ISSUE_57 = "https://github.com/gattia/nsm/issues/57"
ISSUE_60 = "https://github.com/gattia/nsm/issues/60"
ISSUE_54 = "https://github.com/gattia/nsm/issues/54"


def broken(issue, reason):
    return pytest.mark.xfail(strict=True, reason=f"{reason} ({issue})")


# ---------------------------------------------------------------------------
# Fixtures: the four cell layouts
# ---------------------------------------------------------------------------

#: A strip of vertices shared by every quad fixture, so cell counts are the only variable.
STRIP_POINTS = np.array([[i // 2, i % 2, 0.0] for i in range(16)], dtype=float)


def quad_strip(n_quads):
    """``n_quads`` axis-aligned quads sharing edges, as VTK-style face data."""
    faces = []
    for k in range(n_quads):
        faces += [4, 2 * k, 2 * k + 2, 2 * k + 3, 2 * k + 1]
    return pv.PolyData(STRIP_POINTS, np.array(faces))


def mixed_strip(n_tris=4, n_quads=4):
    """Triangles and quads in one mesh -- ``regular_faces`` is undefined for it."""
    faces = []
    for k in range(n_tris):
        faces += [3, 2 * k, 2 * k + 2, 2 * k + 1]
    for k in range(n_quads):
        faces += [4, 2 * k, 2 * k + 2, 2 * k + 3, 2 * k + 1]
    return pv.PolyData(STRIP_POINTS, np.array(faces))


def triangle_sphere():
    """The control: an all-triangle mesh every site is supposed to accept."""
    return pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()


#: ``n_quads`` chosen so the two reshape widths disagree about which one raises.
#: 3 quads -> flat length 15 (15 % 4 == 3, 15 % 3 == 0); 4 quads -> 20 (20 % 4 == 0).
NON_TRIANGLE_MESHES = {
    "quads_3": lambda: quad_strip(3),
    "quads_4": lambda: quad_strip(4),
    "mixed": lambda: mixed_strip(),
}


def test_reshape_success_is_a_modular_coincidence():
    """The measurement §8.0.I's statement rests on, kept so the claim cannot go stale.

    A flat VTK face array reshapes into (-1, 4) or (-1, 3) exactly when its length
    happens to divide -- which is a fact about the cell count, not about the mesh being
    triangular. Both columns must contain a silent success, or the defect this file
    characterises has stopped being reachable and the fixtures need rechecking.
    """
    table = {}
    for n in range(1, 7):
        flat = quad_strip(n).faces
        table[n] = (flat.size % 4 == 0, flat.size % 3 == 0)
    assert table[3] == (False, True), table
    assert table[4] == (True, False), table
    # A triangle mesh's own VTK-style array is the (-1, 3) sites' real-world case.
    assert triangle_sphere().faces.size % 3 == 0


# ---------------------------------------------------------------------------
# 1. Face arrays (#57)
# ---------------------------------------------------------------------------

#: The five sites, each reduced to "call me with a mesh".
MESH_SITES = {
    "self_intersection_count": lambda m: self_intersection_count(m),
    "foldover_count": lambda m: foldover_count(m, np.asarray(m.points)),
    "refine_mesh.get_faces": lambda m: get_faces(m),
}

#: The two that take a face *array* rather than a mesh.
ARRAY_SITES = {
    "build_mesh_laplacian": lambda f, m: build_mesh_laplacian(f, m.n_points, "cpu"),
    "compute_feature_mask": lambda f, m: compute_feature_mask(f, np.asarray(m.points)),
}


@pytest.mark.parametrize("site", sorted(MESH_SITES))
@pytest.mark.parametrize("layout", sorted(NON_TRIANGLE_MESHES))
@broken(ISSUE_57, "#57: a non-triangular mesh is reshaped, not refused")
def test_mesh_site_refuses_non_triangles(site, layout):
    """Every mesh-taking site names the problem instead of reshaping past it.

    Today two things happen and neither is this: ``quads_3`` raises a bare
    ``ValueError: cannot reshape array of size 15 into shape (4)``, and ``quads_4``
    succeeds -- returning five fabricated triangles for four real quads.
    """
    with pytest.raises(ValueError) as exc:
        MESH_SITES[site](NON_TRIANGLE_MESHES[layout]())
    assert "triangle" in str(exc.value).lower()


@pytest.mark.parametrize("site", sorted(ARRAY_SITES))
@broken(ISSUE_57, "#57: a VTK-style flat face array is reshaped, not refused")
def test_array_site_refuses_vtk_style_faces(site):
    """``mesh.faces`` is the array a caller reaches for, and it is the wrong one.

    ``pv.Sphere(8, 8).faces`` has 384 entries and 384 % 3 == 0, so the reshape succeeds
    and builds 128 rows of interleaved counts and indices for 96 real triangles.
    """
    sphere = triangle_sphere()
    with pytest.raises(ValueError) as exc:
        ARRAY_SITES[site](np.asarray(sphere.faces), sphere)
    assert "triangle" in str(exc.value).lower()


@pytest.mark.parametrize("site", sorted(ARRAY_SITES))
@broken(ISSUE_57, "#57: a (M, 4) quad array is reshaped, not refused")
def test_array_site_refuses_quad_arrays(site):
    quads = quad_strip(4)
    with pytest.raises(ValueError) as exc:
        ARRAY_SITES[site](np.asarray(quads.regular_faces), quads)
    assert "triangle" in str(exc.value).lower()


def test_vtk_style_faces_would_build_a_different_operator():
    """Why #57 is a wrong-answer defect and not a hygiene one.

    Measured on ``pv.Sphere(8, 8)``: the smoothing operator built from the VTK-style
    array has 373 non-zeros against the correct 288, and the feature mask flags 50
    vertices against 8. The numbers are asserted rather than quoted so that a pyvista
    change which made them coincide would go red here rather than quietly weaken
    ``test_array_site_refuses_vtk_style_faces``.
    """
    sphere = triangle_sphere()
    correct = build_mesh_laplacian(np.asarray(sphere.regular_faces), sphere.n_points, "cpu")
    wrong = build_mesh_laplacian(np.asarray(sphere.faces).reshape(-1, 3), sphere.n_points, "cpu")
    assert wrong._nnz() != correct._nnz()
    assert not torch.allclose(wrong.to_dense(), correct.to_dense())


def test_regular_faces_matches_the_reshape_for_triangles():
    """The accessor's compatibility claim: no triangle mesh moves.

    ``regular_faces`` is what the fix reads; ``faces.reshape(-1, 4)[:, 1:]`` is what
    every site read before it. For an all-triangle mesh they must be the same array, or
    the fix is a behaviour change rather than a refactor.
    """
    sphere = triangle_sphere()
    assert np.array_equal(
        np.asarray(sphere.regular_faces), np.asarray(sphere.faces).reshape(-1, 4)[:, 1:]
    )


@pytest.mark.parametrize("site", sorted(MESH_SITES))
def test_mesh_site_accepts_triangles(site):
    """The control. Whatever the refusal does, an all-triangle mesh still works."""
    assert MESH_SITES[site](triangle_sphere()) is not None


@pytest.mark.parametrize("site", sorted(ARRAY_SITES))
def test_array_site_accepts_an_m_by_3_array(site):
    sphere = triangle_sphere()
    assert ARRAY_SITES[site](np.asarray(sphere.regular_faces), sphere) is not None


# ---------------------------------------------------------------------------
# 2. The use_vtk twins (#60)
# ---------------------------------------------------------------------------

GRID_N = 32


def sphere_sdf_grid(radius=0.5, n=GRID_N):
    """An analytic SDF on the (X, Y, Z) grid layout this module documents."""
    lin = np.linspace(-1, 1, n)
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
    return (np.sqrt(x**2 + y**2 + z**2) - radius).astype(np.float32)


VOXEL_SIZE = 2.0 / (GRID_N - 1)
ORIGIN = (-1.0, -1.0, -1.0)

TWINS = {"skimage": sdf_grid_to_mesh, "vtk": sdf_grid_to_mesh_vtk}


@pytest.mark.parametrize(
    "twin",
    [
        pytest.param(
            "skimage",
            marks=broken(ISSUE_60, "#60: sdf_grid_to_mesh calls .cpu() unguarded"),
        ),
        "vtk",
    ],
)
def test_both_twins_accept_numpy(twin):
    """``use_vtk`` selects an extraction backend; it must not also select an input type.

    Only the skimage twin is marked: the VTK one already guards with ``hasattr``, and
    the whole defect is that the two differ.
    """
    mesh = TWINS[twin](sphere_sdf_grid(), ORIGIN, VOXEL_SIZE)
    assert mesh.point_coords.shape[0] > 0


@pytest.mark.parametrize("twin", sorted(TWINS))
def test_both_twins_accept_torch(twin):
    mesh = TWINS[twin](torch.from_numpy(sphere_sdf_grid()), ORIGIN, VOXEL_SIZE)
    assert mesh.point_coords.shape[0] > 0


@broken(ISSUE_60, "#60: the twins ship different narrow_band defaults")
def test_twins_share_their_narrow_band_default():
    import inspect

    def default(fn):
        target = getattr(fn, "__wrapped__", fn)
        return inspect.signature(target).parameters["narrow_band"].default

    assert default(sdf_grid_to_mesh) == default(sdf_grid_to_mesh_vtk)


@pytest.mark.parametrize("twin", sorted(TWINS))
def test_narrow_band_does_not_move_the_surface(twin):
    """What makes aligning the defaults safe, pinned before they are aligned.

    Cropping to the band and extracting the full volume are the same extraction on a
    surface the band fully contains: measured max vertex displacement 6.2e-08 (skimage)
    and 7.5e-08 (VTK) on this 32^3 grid, against float32's ~1e-07 at unit magnitude. The
    1e-06 tolerance leaves roughly an order of magnitude of headroom over that; a
    regression that actually re-tessellated would move vertices by a voxel (0.065),
    four orders of magnitude above the tolerance.
    """
    grid = torch.from_numpy(sphere_sdf_grid())
    full = TWINS[twin](grid, ORIGIN, VOXEL_SIZE, narrow_band=False).point_coords
    band = TWINS[twin](grid, ORIGIN, VOXEL_SIZE, narrow_band=True).point_coords
    assert full.shape == band.shape
    assert np.abs(np.sort(full, axis=0) - np.sort(band, axis=0)).max() < 1e-6


# ---------------------------------------------------------------------------
# 3. The adaptive fallback grid (#60)
# ---------------------------------------------------------------------------


class _OffsetSphereDecoder(torch.nn.Module):
    """A surface small enough that a coarse pass over ``search_bounds`` misses it."""

    def forward(self, x):
        return torch.linalg.norm(x[:, -3:] - 2.0, dim=1, keepdim=True) - 0.05


@broken(ISSUE_60, "#60: the fallback grid origin ignores search_bounds")
def test_fallback_grid_covers_search_bounds(monkeypatch):
    """The fallback searches where the caller asked, not where the default points.

    Measured today with ``search_bounds=(0.0, 4.0)`` and ``n_pts_per_axis=17``: the
    fallback grid spans [-1, 3] on every axis, because ``voxel_origin`` arrives as its
    own ``(-1, -1, -1)`` default while ``voxel_size`` was derived from ``search_bounds``.
    """
    seen = {}

    def spy(
        decoder,
        latent_vector,
        n_pts_per_axis=256,
        voxel_origin=(-1, -1, -1),
        voxel_size=None,
        *a,
        **k,
    ):
        seen.update(n=n_pts_per_axis, origin=voxel_origin, size=voxel_size)
        return None

    monkeypatch.setattr(mesh_main, "create_mesh", spy)
    create_mesh_adaptive(
        _OffsetSphereDecoder(),
        None,
        n_pts_per_axis=17,
        n_pts_coarse=4,
        search_bounds=(0.0, 4.0),
        device="cpu",
    )
    assert seen, "the fallback branch did not run -- the fixture no longer misses the surface"
    span = [
        (seen["origin"][i], seen["origin"][i] + seen["size"] * (seen["n"] - 1)) for i in range(3)
    ]
    assert span == [(0.0, 4.0)] * 3, span


# ---------------------------------------------------------------------------
# 4. Refusal versus invention (#54)
# ---------------------------------------------------------------------------


@broken(ISSUE_54, "#54: get_target_cells raises UnboundLocalError on its own defaults")
def test_get_target_cells_runs_on_its_own_defaults():
    """SCOPE §2.3 condition 1. Documenting a module that raises describes nothing."""
    assert len(get_target_cells(triangle_sphere())) == 0


@broken(ISSUE_54, "#54: subdivide_large_triangles inherits the UnboundLocalError")
def test_subdivide_large_triangles_runs_on_its_own_defaults():
    assert subdivide_large_triangles(triangle_sphere()) is not None


@broken(ISSUE_54, "#54: score_correspondence substitutes the warped mesh for the source")
def test_roundtrip_metrics_skip_without_a_source_mesh():
    """Every sibling in the same dict skips; these two invent.

    Measured today: mean roundtrip distance 0.2500 where the true displacement against
    the source is 0.0017 -- a factor of 144, reported as a measurement, in the same
    return value where ``foldover_count`` correctly says it was skipped.
    """
    source = triangle_sphere()
    warped = source.copy()
    warped.points = np.asarray(source.points) * 1.5
    roundtrip = np.asarray(source.points) + 0.001

    result = score_correspondence(
        warped,
        source,
        source_mesh=None,
        roundtrip_points=roundtrip,
        compute_self_intersection=False,
    )
    assert result["roundtrip_distance"] == {"skipped": True, "reason": "source_mesh not provided"}
    assert result["forward_backward_disagreement"]["skipped"] is True


def test_roundtrip_metrics_are_unchanged_when_the_source_is_given():
    """The other half of the fix: the working path keeps the number it always had."""
    source = triangle_sphere()
    warped = source.copy()
    warped.points = np.asarray(source.points) * 1.5
    roundtrip = np.asarray(source.points) + 0.001

    result = score_correspondence(
        warped,
        source,
        source_mesh=source,
        roundtrip_points=roundtrip,
        compute_self_intersection=False,
    )
    expected = float(np.linalg.norm(roundtrip - np.asarray(source.points), axis=1).mean())
    assert result["roundtrip_distance"]["mean"] == pytest.approx(expected)
