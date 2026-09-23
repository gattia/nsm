"""
The SDF convention for open (clipped) meshes.

The knee pipeline clips the femur top with no cap, and both shipped configs run
``fix_mesh: False``, so pcu computes signed distances against the open mesh. That is safe:
pcu signs by the closest-point pseudonormal, so the cut reads as capped. Measured
2026-08-23: no point above the cut labelled inside, on a 9,870-point slice. Training and
reconstruction share the convention; mixing it, e.g. supervising against the unclipped
bone, would break the agreement.
"""

import numpy as np
import pytest
import pyvista as pv
from pymskt.mesh import Mesh

from NSM.datasets.utils import meshfix

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


def test_an_open_mesh_reads_as_its_capped_counterpart(clipped_sphere):
    """
    Fails if pymskt's ``get_sdf_pts(method="pcu")`` labels a probe beyond an open cut as
    inside, or reads the open mesh more than ``OPEN_VS_CAPPED_ATOL`` away from its
    ``meshfix``-capped counterpart.

    It calls pymskt directly, so it guards the dependency, not the readers' choice of pcu.
    """
    open_sdf = Mesh(pv.PolyData(clipped_sphere)).get_sdf_pts(_axis_points(), method="pcu")
    assert np.all(open_sdf > 0), open_sdf

    capped = Mesh(pv.PolyData(clipped_sphere.copy()))
    meshfix(capped)
    capped_boundary = pv.PolyData(capped.mesh).extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False
    )
    assert capped_boundary.n_cells == 0, "premise: meshfix must close the rim"
    capped_sdf = capped.get_sdf_pts(_axis_points(), method="pcu")
    np.testing.assert_allclose(open_sdf, capped_sdf, atol=OPEN_VS_CAPPED_ATOL)
