"""
What the ``mesh/`` package does with inputs it was not meant to take: it refuses them
rather than returning a plausible number.

1. **Face arrays (#57).** Sites that take a face array refuse anything that is not
   all-triangle.
2. **The ``use_vtk`` twins (#60)** take the same inputs under the same defaults.
3. **The adaptive fallback grid (#60)** covers the ``search_bounds`` it was given.
4. **Refusal versus invention (#54).**
"""

import inspect
import warnings

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
from NSM.mesh.main import create_mesh, create_mesh_adaptive, sdf_grid_to_mesh, sdf_grid_to_mesh_vtk
from NSM.mesh.refine_mesh import (
    get_faces,
    get_target_cells,
    subdivide_large_triangles,
    subdivide_triangles_on_base_mesh,
)

#: A strip of vertices shared by every quad fixture, so the cell count is the only variable.
STRIP_POINTS = np.array([[i // 2, i % 2, 0.0] for i in range(16)], dtype=float)


def strip(n_tris=0, n_quads=0):
    """Triangles then quads along one strip, as VTK-style face data."""
    faces = []
    for k in range(n_tris):
        faces += [3, 2 * k, 2 * k + 2, 2 * k + 1]
    for k in range(n_quads):
        faces += [4, 2 * k, 2 * k + 2, 2 * k + 3, 2 * k + 1]
    return pv.PolyData(STRIP_POINTS, np.array(faces))


def triangle_sphere():
    return pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()


def test_face_array_sites_accept_triangles_and_refuse_everything_else():
    """
    Fails if ``self_intersection_count``, ``foldover_count``, ``get_faces``,
    ``build_mesh_laplacian`` or ``compute_feature_mask`` takes a quad or mixed mesh, or a flat
    VTK face array, without raising ``ValueError`` (KNOWN_ISSUES History 17).

    The fixtures are the cases a bare reshape passes silently. 3 quads flatten to 15 entries
    (15 % 3 == 0) and 4 quads to 20 (20 % 4 == 0): 4 quads reshape into five fabricated
    triangles. A mesh's own ``.faces`` is the wrong array for the two sites that take a face
    array, and its 384 entries divide by 3. The first assert keeps the fixtures that way.
    """
    assert (strip(n_quads=3).faces.size % 3, strip(n_quads=4).faces.size % 4) == (0, 0)

    mesh_sites = [
        self_intersection_count,
        lambda m: foldover_count(m, np.asarray(m.points)),
        get_faces,
    ]
    array_sites = [
        lambda f, m: build_mesh_laplacian(f, m.n_points, "cpu"),
        lambda f, m: compute_feature_mask(f, np.asarray(m.points)),
    ]
    sphere = triangle_sphere()
    for site in mesh_sites:
        assert site(sphere) is not None
        for bad in (strip(n_quads=3), strip(n_quads=4), strip(n_tris=4, n_quads=4)):
            with pytest.raises(ValueError, match="(?i)triangle"):
                site(bad)
    quads = strip(n_quads=4)
    for site in array_sites:
        assert site(np.asarray(sphere.regular_faces), sphere) is not None
        for faces, mesh in ((sphere.faces, sphere), (quads.regular_faces, quads)):
            with pytest.raises(ValueError, match="(?i)triangle"):
                site(np.asarray(faces), mesh)


# ---------------------------------------------------------------------------
# The use_vtk twins
# ---------------------------------------------------------------------------

GRID_N = 32
VOXEL_SIZE = 2.0 / (GRID_N - 1)
ORIGIN = (-1.0, -1.0, -1.0)


def sphere_sdf_grid(radius=0.5):
    lin = np.linspace(-1, 1, GRID_N)
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
    return (np.sqrt(x**2 + y**2 + z**2) - radius).astype(np.float32)


def test_both_twins_take_numpy_or_torch_with_one_narrow_band_default():
    """
    Fails if ``sdf_grid_to_mesh`` and ``sdf_grid_to_mesh_vtk`` default ``narrow_band``
    differently, or either refuses a numpy array or a torch tensor (#60).
    """

    def default(fn):
        return inspect.signature(fn).parameters["narrow_band"].default

    assert default(sdf_grid_to_mesh) == default(sdf_grid_to_mesh_vtk)
    for twin in (sdf_grid_to_mesh, sdf_grid_to_mesh_vtk):
        for grid in (sphere_sdf_grid(), torch.from_numpy(sphere_sdf_grid())):
            assert twin(grid, ORIGIN, VOXEL_SIZE).point_coords.shape[0] > 0


@pytest.mark.parametrize("twin", [sdf_grid_to_mesh, sdf_grid_to_mesh_vtk])
def test_the_narrow_band_does_not_move_the_surface(twin):
    """
    Fails if narrow-band cropping in ``sdf_grid_to_mesh`` or ``sdf_grid_to_mesh_vtk`` moves
    a vertex or changes the vertex count, as a wrong crop origin would.

    Measured max vertex displacement 6.2e-08 (skimage) and 7.5e-08 (VTK), about float32's
    resolution. A re-tessellation would move a vertex by a voxel, 0.065.
    """
    grid = torch.from_numpy(sphere_sdf_grid())
    full = twin(grid, ORIGIN, VOXEL_SIZE, narrow_band=False).point_coords
    band = twin(grid, ORIGIN, VOXEL_SIZE, narrow_band=True).point_coords
    assert full.shape == band.shape
    assert np.abs(np.sort(full, axis=0) - np.sort(band, axis=0)).max() < 1e-6


# ---------------------------------------------------------------------------
# create_mesh and create_mesh_adaptive
# ---------------------------------------------------------------------------


class _TwoSpheres(torch.nn.Module):
    def forward(self, x):
        q = x[:, -3:]
        return torch.cat(
            [
                torch.linalg.norm(q, dim=1, keepdim=True) - 0.5,
                torch.linalg.norm(q - 0.2, dim=1, keepdim=True) - 0.3,
            ],
            dim=1,
        )


class _SmallOffsetSphere(torch.nn.Module):
    """Small enough that a coarse pass over ``search_bounds`` misses it."""

    def forward(self, x):
        return torch.linalg.norm(x[:, -3:] - 2.0, dim=1, keepdim=True) - 0.05


def test_the_fallback_grid_spans_the_search_bounds(monkeypatch):
    """
    Fails if ``create_mesh_adaptive``'s dense fallback takes its grid origin from a default
    rather than ``search_bounds``, or overrides an explicit ``voxel_origin``
    (KNOWN_ISSUES History 19).

    ``create_mesh`` is replaced to capture the grid the fallback asks for. At the default
    bounds the derived origin is ``(-1, -1, -1)``, the origin every existing run used.
    """
    seen = []
    monkeypatch.setattr(mesh_main, "create_mesh", lambda *a, **k: seen.append(k))

    def fallback(**kwargs):
        create_mesh_adaptive(
            _SmallOffsetSphere(), None, n_pts_per_axis=17, n_pts_coarse=4, device="cpu", **kwargs
        )
        return seen.pop()

    grid = fallback(search_bounds=(0.0, 4.0))
    span = [grid["voxel_origin"][i] + grid["voxel_size"] * 16 for i in range(3)]
    assert (tuple(grid["voxel_origin"]), span) == ((0.0, 0.0, 0.0), [4.0] * 3)
    assert tuple(fallback()["voxel_origin"]) == (-1.0, -1.0, -1.0)
    assert fallback(search_bounds=(0.0, 4.0), voxel_origin=(7.0, 7.0, 7.0))["voxel_origin"] == (
        7.0,
        7.0,
        7.0,
    )


@pytest.mark.parametrize("use_vtk", [True, False])
def test_dense_and_adaptive_extract_the_same_surfaces(use_vtk):
    """
    Fails if ``create_mesh_adaptive`` samples its fine grid off ``create_mesh``'s lattice,
    so the two extract different vertices from the same decoder.

    Measured max displacement 7.0e-07 against a 0.087 voxel. The coarse pass finds the whole
    search box here and ``min_dim=64`` pads the fine grid past it, so no crop is tested.
    Both shapes are symmetric under an axis swap, so an X-fastest fine grid passes too.
    """
    common = dict(objects=2, device="cpu", scale_to_original_mesh=False, use_vtk=use_vtk)
    dense = create_mesh(_TwoSpheres(), None, n_pts_per_axis=24, **common)
    adaptive = create_mesh_adaptive(
        _TwoSpheres(), None, n_pts_per_axis=24, n_pts_coarse=12, **common
    )
    assert len(dense) == len(adaptive) == 2
    for d, a in zip(dense, adaptive):
        assert np.abs(d.point_coords - a.point_coords).max() < 1e-5


def test_scale_and_offset_reach_the_finished_mesh():
    """
    Fails if ``create_mesh(scale_to_original_mesh=True)`` does not map each vertex to
    ``vertex * scale + offset``, for example by adding the offset before scaling.
    """
    common = dict(objects=2, device="cpu", n_pts_per_axis=24)
    plain = create_mesh(_TwoSpheres(), None, scale_to_original_mesh=False, **common)
    scaled = create_mesh(
        _TwoSpheres(),
        None,
        scale=2.5,
        offset=(0.1, -0.2, 0.3),
        scale_to_original_mesh=True,
        **common,
    )
    expected = plain[0].point_coords * 2.5 + np.array([0.1, -0.2, 0.3])
    assert np.allclose(scaled[0].point_coords, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# refine_mesh and score_correspondence
# ---------------------------------------------------------------------------


def test_refine_mesh_runs_on_its_defaults_and_each_threshold_alone():
    """
    Fails if ``get_target_cells`` raises on its defaults, selects cells with no threshold
    set, or ignores ``area_threshold``, ``length_threshold`` or ``max_length_threshold``
    passed alone (#54).

    ``refine_mesh`` is research code (``SCOPE`` §2.3).
    """
    sphere = triangle_sphere()
    assert len(get_target_cells(sphere)) == 0
    assert subdivide_large_triangles(sphere) is not None
    for threshold, value in (
        ("area_threshold", -1.0),
        ("length_threshold", 0.0),
        ("max_length_threshold", 0.0),
    ):
        assert len(get_target_cells(sphere, **{threshold: value})) == sphere.n_cells


def _warp(mesh):
    """Stands in for ``interpolate_points``: every vertex moves, no face changes."""
    warped = mesh.copy()
    pts = np.asarray(mesh.points)
    warped.points = pts * (1.0 + 0.35 * np.sin(3 * pts[:, [2]]))
    return warped


def test_subdividing_warns_when_the_two_meshes_are_different_triangles():
    """
    Fails if ``subdivide_triangles_on_base_mesh`` warns when ``mesh`` is ``base_mesh`` warped
    or copied, or stays silent when ``base_mesh`` has a different cell count.

    A cell index selected on one tessellation means nothing in another, and the result is
    a wrong mesh rather than an error. The check compares faces, not points, so the
    documented use (``mesh`` is ``base_mesh`` warped) stays quiet. Reusing a mesh after
    subdividing its base (``refined``) is the iterative caller's mistake: measured, it takes
    624 cells to 1110 with no error. Both mismatched cases differ in cell count, so the
    same-count, different-faces branch is not exercised.
    """
    base = pv.Sphere(theta_resolution=12, phi_resolution=12).triangulate()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        refined = subdivide_triangles_on_base_mesh(base, _warp(base), max_length_threshold=0.25)
        subdivide_triangles_on_base_mesh(base.copy(), base, max_length_threshold=0.25)

    other = pv.Sphere(theta_resolution=16, phi_resolution=16).triangulate()
    for mismatched in (other, refined):
        with pytest.warns(UserWarning, match="do not refer to the same triangles"):
            subdivide_triangles_on_base_mesh(mismatched, _warp(base), max_length_threshold=0.25)


def test_roundtrip_metrics_skip_without_a_source_mesh():
    """
    Fails if ``score_correspondence`` measures ``roundtrip_distance`` or
    ``forward_backward_disagreement`` against the warped mesh instead of skipping when
    ``source_mesh`` is None, or against anything but ``source_mesh`` when it is given
    (KNOWN_ISSUES History 18).

    The warped mesh is the source scaled 1.5x, so measuring against it gives a mean
    roundtrip distance of 0.2500 against a true 0.0017.
    """
    source = triangle_sphere()
    warped = source.copy()
    warped.points = np.asarray(source.points) * 1.5
    roundtrip = np.asarray(source.points) + 0.001
    kwargs = dict(roundtrip_points=roundtrip, compute_self_intersection=False)

    skipped = score_correspondence(warped, source, source_mesh=None, **kwargs)
    assert skipped["roundtrip_distance"] == {"skipped": True, "reason": "source_mesh not provided"}
    assert skipped["forward_backward_disagreement"]["skipped"] is True

    given = score_correspondence(warped, source, source_mesh=source, **kwargs)
    expected = np.linalg.norm(roundtrip - np.asarray(source.points), axis=1).mean()
    assert given["roundtrip_distance"]["mean"] == pytest.approx(expected)
