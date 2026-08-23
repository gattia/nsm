"""
The SDF convention for open (clipped) meshes, measured and pinned.

The knee pipeline clips the femur top with a planar ``pyvista`` ``.clip`` — no cap —
and both the shipped training config and the shipped recon config run
``fix_mesh: False``, so pcu computes signed distances against the *open* mesh on both
paths. That is safe, and this file is the evidence: ``pcu.signed_distance_to_mesh``
signs by the closest-point pseudonormal, not by global containment, so a planar cut
reads as if it were capped — every point beyond the cut is *outside*, at its distance
to the cut, coherently (measured 2026-08-23: zero above-cut points labeled inside on
a 9,870-point slice; fewer sign transitions than the closed mesh).

Because training and reconstruction share the clip, the ``fix_mesh`` setting, and the
pcu method, the learned field and any sampled-reconstruction supervision agree on
this convention. What would break the agreement is *mixing* conventions — e.g.
supervising against the un-clipped bone.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from pymskt.mesh import Mesh  # noqa: E402

from NSM.datasets.utils import meshfix  # noqa: E402

#: On-axis probe heights. The sphere has radius 1 and is cut at z = 0.6, so
#: 0.65–0.99 lie inside the *closed* sphere but beyond the cut, 1.1+ outside both.
PROBE_Z = np.array([0.65, 0.8, 0.99, 1.1, 1.7])

#: Max measured |open - capped| over the probes is 0.010 (at z = 2.5, where meshfix's
#: remeshing moves the far field slightly); 0.02 gives ~2x headroom.
OPEN_VS_CAPPED_ATOL = 0.02


@pytest.fixture(scope="module")
def clipped_sphere():
    sphere = pv.Sphere(radius=1.0, theta_resolution=48, phi_resolution=48).triangulate()
    clipped = sphere.clip("z", value=0.6, invert=True)
    boundary = clipped.extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False
    )
    assert boundary.n_cells > 0, "premise: the clip must leave an open rim"
    return clipped


def _axis_points():
    zeros = np.zeros_like(PROBE_Z)
    return np.c_[zeros, zeros, PROBE_Z]


def test_beyond_a_planar_cut_reads_as_outside(clipped_sphere):
    """No phantom interior: pcu never labels the missing-cap volume as inside."""
    sdf = Mesh(pv.PolyData(clipped_sphere)).get_sdf_pts(_axis_points(), method="pcu")
    assert np.all(sdf > 0), sdf


def test_the_open_mesh_matches_its_capped_counterpart(clipped_sphere):
    """The open mesh's field is the capped mesh's field, not a third convention."""
    open_sdf = Mesh(pv.PolyData(clipped_sphere)).get_sdf_pts(_axis_points(), method="pcu")
    capped = Mesh(pv.PolyData(clipped_sphere.copy()))
    meshfix(capped)
    capped_boundary = pv.PolyData(capped.mesh).extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False
    )
    assert capped_boundary.n_cells == 0, "premise: meshfix must close the rim"
    capped_sdf = capped.get_sdf_pts(_axis_points(), method="pcu")
    np.testing.assert_allclose(open_sdf, capped_sdf, atol=OPEN_VS_CAPPED_ATOL)
