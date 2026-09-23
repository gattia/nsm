"""
Issue #55: functions that mutated a caller's object and also returned it.

The fix differs by site, as #55 says it may. The ``sdf_gt`` preprocess and
``compute_recon_loss`` had no reason to mutate, so they stopped. ``interpolate_mesh``
mutates by design, since it carries a mesh along the level set, and says so in its
docstring.
"""

import numpy as np
import pyvista as pv
import torch
from pymskt.mesh import Mesh

from NSM.mesh.interpolate import interpolate_mesh, interpolate_points
from NSM.reconstruct.latent_fit import (
    reconstruct_latent_preprocess_sdf_gt,
    reconstruct_latent_sdf_gt_type_check,
)
from NSM.reconstruct.recon_evaluation import compute_recon_loss


def _plain(radius):
    """A mesh with float64 ``point_coords``, as a caller would hold it."""
    mesh = Mesh(pv.Sphere(radius=radius, theta_resolution=12, phi_resolution=12))
    mesh.point_coords = mesh.point_coords.astype(np.float64)
    return mesh


def test_the_sdf_gt_preprocess_leaves_the_callers_list_alone():
    """
    It clamped the caller's list in place and returned it. The tensors were always safe;
    the list's slots were rebound. A tuple, which the type check names as supported,
    raised ``TypeError`` on the in-place assignment and now works.
    """
    caller = [torch.tensor([[-5.0], [0.5], [5.0]]), torch.tensor([[-9.0], [9.0]])]
    checked = reconstruct_latent_sdf_gt_type_check(caller)
    result = reconstruct_latent_preprocess_sdf_gt(checked, 1.0, device="cpu")

    assert checked is not caller and result is not caller
    assert caller[0].flatten().tolist() == [-5.0, 0.5, 5.0]
    assert [r.flatten().tolist() for r in result] == [[-1.0, 0.5, 1.0], [-1.0, 1.0]]

    as_tuple = reconstruct_latent_sdf_gt_type_check((torch.tensor([[-5.0]]),))
    assert reconstruct_latent_preprocess_sdf_gt(as_tuple, 1.0, device="cpu")[0].item() == -1.0


class TestSite2TheAssdDowncast:
    """
    ``compute_recon_loss(calc_assd=True)`` used to cast both of the caller's meshes to
    float32, and only on the ASSD path. mskt 0.1.21's ``pcu_sdf`` casts to float64 itself,
    so the cast could only lose precision (7.2e-09 on float64 input, § History 28).
    """

    def test_neither_metric_changes_the_callers_dtypes_or_the_value(self):
        """
        The emulated old cast is lossless on these meshes, whose points originate as float32
        in VTK, so the ASSD must match exactly. The bound is a sanity check. A transcribed
        ASSD went red on macOS: it encoded Linux's sphere tessellation.
        """
        recon, orig = _plain(1.0), _plain(1.1)
        compute_recon_loss([recon], [orig], calc_symmetric_chamfer=True, n_samples_chamfer=64)
        as_is = compute_recon_loss([recon], [orig], calc_assd=True)["assd_0"]
        assert recon.point_coords.dtype == orig.point_coords.dtype == np.float64

        recon32, orig32 = _plain(1.0), _plain(1.1)
        recon32.point_coords = recon32.point_coords.astype(np.float32)
        orig32.point_coords = orig32.point_coords.astype(np.float32)
        assert compute_recon_loss([recon32], [orig32], calc_assd=True)["assd_0"] == as_is
        assert 0.05 < as_is < 0.15


class TestSite2TheMixedDtypePair:
    """
    The cast had a reason after all. mskt 0.1.19, which production ran, passes both sides
    straight to ``point_cloud_utils``, which refuses a mixed pair: ``ValueError: Invalid
    type (double, Row Major) for argument 'v'``. Production always has one: float64 meshes
    on disk against a float32 reconstruction. Deleting the cast crashed every fit.

    The fix aligns a mixed pair on copies. This suite's mskt would forgive a mixed pair, so
    what is pinned is the pair that reaches ``get_assd_mesh``.
    """

    def test_the_pair_is_aligned_on_copies_and_scores_the_matched_value(self, monkeypatch):
        seen = []
        unpatched = Mesh.get_assd_mesh

        def spy(self, other_mesh):
            seen.append((self.point_coords.dtype, other_mesh.point_coords.dtype))
            return unpatched(self, other_mesh)

        monkeypatch.setattr(Mesh, "get_assd_mesh", spy)
        recon, orig = _plain(1.0), _plain(1.1)
        recon.point_coords = recon.point_coords.astype(np.float32)
        mixed = compute_recon_loss([recon], [orig], calc_assd=True)["assd_0"]

        assert seen == [(np.float64, np.float64)]
        assert recon.point_coords.dtype == np.float32
        assert mixed == compute_recon_loss([_plain(1.0)], [_plain(1.1)], calc_assd=True)["assd_0"]


class TestSite3TheInterpolationMesh:
    """
    ``interpolate_mesh`` moves the caller's mesh and returns it, by design.
    ``interpolate_points`` leaves its input alone. Each docstring says which.
    """

    class _SphereSDF(torch.nn.Module):
        """SDF of a sphere of radius ``1 + latent[0]``."""

        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            latent, xyz = x[:, :1], x[:, 1:]
            return (xyz.norm(dim=1, keepdim=True) - (1.0 + latent)) + 0.0 * self.p

    def test_the_mesh_moves_in_place(self):
        mesh = Mesh(pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10))
        before = mesh.point_coords.copy()
        out = interpolate_mesh(
            self._SphereSDF(), np.array([0.0]), np.array([0.5]), n_steps=3, mesh=mesh
        )
        assert out is mesh
        assert np.abs(before - mesh.point_coords).max() > 0.1
        assert "in place" in interpolate_mesh.__doc__

    def test_the_points_are_not_modified(self):
        model = self._SphereSDF()
        array = np.random.default_rng(0).normal(size=(20, 3))
        before = array.copy()
        out = interpolate_points(model, np.array([0.0]), np.array([0.5]), n_steps=2, points1=array)
        assert out is not array and np.array_equal(array, before)

        tensor = torch.tensor(before, dtype=torch.float)
        interpolate_points(model, np.array([0.0]), np.array([0.5]), n_steps=2, points1=tensor)
        assert torch.equal(tensor, torch.tensor(before, dtype=torch.float))
        assert "not modified" in interpolate_points.__doc__
