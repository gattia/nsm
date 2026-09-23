"""
Functions that take a caller's object and return one: which of them mutate it (#55).

The ``sdf_gt`` preprocess and ``compute_recon_loss`` leave the caller's objects alone.
``interpolate_mesh`` mutates by design, since it carries a mesh along the level set, and
its docstring says so.
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
    Fails if ``reconstruct_latent_sdf_gt_type_check`` or
    ``reconstruct_latent_preprocess_sdf_gt`` returns the caller's list or rebinds its slots
    to the clamped tensors, or a tuple ``sdf_gt`` raises (#55).

    ``torch.clamp`` returns a new tensor, so the risk is to the list's slots, not the
    tensors.
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
    ``compute_recon_loss(calc_assd=True)`` must not cast the caller's meshes to float32
    (``KNOWN_ISSUES.md`` § History 28). mskt 0.1.21's ``pcu_sdf`` casts to float64 itself,
    so a float32 cast can only lose precision: 7.2e-09 on float64 input.
    """

    def test_neither_metric_changes_the_callers_dtypes_or_the_value(self):
        """
        Fails if ``compute_recon_loss`` changes the dtype of the caller's float64 meshes on
        the chamfer or ASSD path, or scores their float32 copies differently
        (KNOWN_ISSUES History 28).

        The points originate as float32 in VTK, so the float32 copies are exact and the two
        ASSDs must match. The 0.05-0.15 bound is a sanity check, not a transcribed value:
        the exact ASSD differs between Linux's and macOS's sphere tessellations.
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
    mskt 0.1.19 passes both sides straight to ``point_cloud_utils``, which refuses a mixed
    pair: ``ValueError: Invalid type (double, Row Major) for argument 'v'``. Production
    always has one: float64 meshes on disk against a float32 reconstruction.
    ``compute_recon_loss`` aligns a mixed pair on float64 copies.

    This suite's mskt accepts a mixed pair, so what is pinned is the pair that reaches
    ``get_assd_mesh``.
    """

    def test_the_pair_is_aligned_on_copies_and_scores_the_matched_value(self, monkeypatch):
        """
        Fails if ``compute_recon_loss(calc_assd=True)`` hands ``Mesh.get_assd_mesh`` a mixed
        float32/float64 pair, casts the caller's float32 mesh in place, or scores the mixed
        pair differently from an all-float64 pair.

        The spy is on pymskt's ``Mesh.get_assd_mesh``, so this breaks if ASSD moves to
        another pymskt call.
        """
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
        """
        Fails if ``interpolate_mesh`` returns a new mesh or leaves the caller's mesh unmoved,
        or its docstring stops saying it works "in place".
        """
        mesh = Mesh(pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10))
        before = mesh.point_coords.copy()
        out = interpolate_mesh(
            self._SphereSDF(), np.array([0.0]), np.array([0.5]), n_steps=3, mesh=mesh
        )
        assert out is mesh
        assert np.abs(before - mesh.point_coords).max() > 0.1
        assert "in place" in interpolate_mesh.__doc__

    def test_the_points_are_not_modified(self):
        """
        Fails if ``interpolate_points`` modifies the caller's ndarray or tensor, returns the
        caller's array, or its docstring stops saying the input is "not modified".
        """
        model = self._SphereSDF()
        array = np.random.default_rng(0).normal(size=(20, 3))
        before = array.copy()
        out = interpolate_points(model, np.array([0.0]), np.array([0.5]), n_steps=2, points1=array)
        assert out is not array and np.array_equal(array, before)

        tensor = torch.tensor(before, dtype=torch.float)
        interpolate_points(model, np.array([0.0]), np.array([0.5]), n_steps=2, points1=tensor)
        assert torch.equal(tensor, torch.tensor(before, dtype=torch.float))
        assert "not modified" in interpolate_points.__doc__
