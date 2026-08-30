"""
Issue #55 — functions that mutate a caller's object and also return it.

One issue, three modules, so one file rather than three: the defect is a *property* of
the calling convention, not of any one function, and PR #38 fixed a fourth instance of it
in ``get_pts_center_and_scale`` before these three were found. Filed 2026-08-22 and
unverified since; three slices have rewritten these call sites, so plan §8.0.N re-ran all
three before touching them. All three still reproduce, and re-running changed the shape of
the third.

**The fix is not the same at all three sites, and #55 says so** — "each site copies, or
documents the mutation and stops returning the object; docstrings say which":

* ``compute_recon_loss`` has no reason to mutate — the downcast is for one comparison —
  so it copies.
* the ``sdf_gt`` preprocess has no reason either; the clamp is the function's product.
* ``interpolate_mesh`` mutates **by design** — carrying a mesh along the level set is
  what it is for, and copying an 80k-vertex mesh per call is not free. That one is
  documented, and the documentation has to be where the caller reads it.
"""

import numpy as np
import pytest
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
    """A mesh whose ``point_coords`` are float64, which is what a caller would hand over."""
    mesh = Mesh(pv.Sphere(radius=radius, theta_resolution=12, phi_resolution=12))
    mesh.point_coords = mesh.point_coords.astype(np.float64)
    return mesh


class TestSite1TheSdfGtPreprocess:
    """
    ``reconstruct_latent_preprocess_sdf_gt`` clamps and device-moves the caller's list.

    Measured on ``main`` at ``09c3834``: a caller holding ``[-5, 0.5, 5]`` and ``[-9, 9]``
    gets back ``[-1, 0.5, 1]`` and ``[-1, 1]`` at ``clamp_dist=1.0``, in its own list
    object, with each element rebound. ``reconstruct_latent`` is the only caller and
    passes the list it built, so nothing in the repo notices — but the list is the
    caller's ``sdf_gt`` argument, one frame up.
    """

    def _caller_list(self):
        return [torch.tensor([[-5.0], [0.5], [5.0]]), torch.tensor([[-9.0], [9.0]])]

    @pytest.mark.xfail(strict=True, reason="#55 site 1 — fixed in this slice's commit 4")
    def test_the_callers_list_object_is_not_returned(self):
        sdf_gt = self._caller_list()
        assert reconstruct_latent_preprocess_sdf_gt(sdf_gt, 1.0, device="cpu") is not sdf_gt

    @pytest.mark.xfail(strict=True, reason="#55 site 1 — fixed in this slice's commit 4")
    def test_the_callers_values_are_not_clamped(self):
        sdf_gt = self._caller_list()
        reconstruct_latent_preprocess_sdf_gt(sdf_gt, 1.0, device="cpu")
        assert sdf_gt[0].flatten().tolist() == [-5.0, 0.5, 5.0]
        assert sdf_gt[1].flatten().tolist() == [-9.0, 9.0]

    def test_the_returned_values_are_clamped_either_way(self):
        """The fix must not move a number. This is the assertion that says it did not."""
        result = reconstruct_latent_preprocess_sdf_gt(self._caller_list(), 1.0, device="cpu")
        assert result[0].flatten().tolist() == [-1.0, 0.5, 1.0]
        assert result[1].flatten().tolist() == [-1.0, 1.0]

    @pytest.mark.xfail(strict=True, reason="#55 site 1 — fixed in this slice's commit 4")
    def test_the_type_check_does_not_hand_back_the_callers_list(self):
        """Same class, one frame earlier: a list in is the same object out."""
        sdf_gt = [torch.tensor([[1.0]])]
        assert reconstruct_latent_sdf_gt_type_check(sdf_gt) is not sdf_gt


class TestSite2TheAssdDowncast:
    """
    ``compute_recon_loss(calc_assd=True)`` moves the caller's meshes to ``float32``.

    Both of them — ``mesh.point_coords`` (the reconstruction) and
    ``orig_meshes[mesh_idx].point_coords`` (the caller's ground truth), at
    ``recon_evaluation.py:125-128``, under a comment that says "make sure the points for
    the meshes are the same types". The comment is right about why; the target is the
    caller's object rather than a copy.

    **The mutation is conditional on a flag**, which is what makes it hard to notice: the
    chamfer path leaves both at ``float64``. A caller that scores chamfer only, then adds
    ASSD later, silently loses precision on meshes it still holds.
    """

    @pytest.mark.xfail(strict=True, reason="#55 site 2 — fixed in this slice's commit 3")
    @pytest.mark.parametrize("which", ["reconstruction", "original"])
    def test_assd_does_not_downcast_the_callers_mesh(self, which):
        recon, orig = _plain(1.0), _plain(1.1)
        compute_recon_loss([recon], [orig], calc_assd=True)
        subject = recon if which == "reconstruction" else orig
        assert subject.point_coords.dtype == np.float64

    def test_chamfer_alone_downcasts_neither(self):
        """
        The pin that says the fix removed the mutation rather than widening it: this one
        passes today and must keep passing.
        """
        recon, orig = _plain(1.0), _plain(1.1)
        compute_recon_loss([recon], [orig], calc_symmetric_chamfer=True, n_samples_chamfer=64)
        assert recon.point_coords.dtype == np.float64
        assert orig.point_coords.dtype == np.float64

    def test_the_assd_value_does_not_move(self):
        """
        ASSD is still computed at ``float32``, on copies. The number is the one measured on
        ``main`` at ``09c3834`` for this fixed geometry; ``rel=1e-6`` is float32's own
        resolution, so a change of *regime* — computing at float64 instead — would break
        this while a change of precision would not.
        """
        result = compute_recon_loss([_plain(1.0)], [_plain(1.1)], calc_assd=True)
        assert result["assd_0"] == pytest.approx(0.09831770188973551, rel=1e-6)


class TestSite3TheInterpolationMesh:
    """
    ``interpolate_mesh`` advances the caller's mesh in place and returns it — **by
    design**, and that is the ruling rather than a defect to fix.

    What #55 asks for here is the documentation, and the documentation exists at the
    wrong frame. ``interpolate_common``, the private engine, says "the returned object is
    the caller's own mesh, mutated". ``interpolate_mesh`` and ``interpolate_points``, the
    two public entry points, say nothing — so the contract is stated in the file the
    author had open and not in the one a caller reads.
    """

    class _SphereSDF(torch.nn.Module):
        """SDF of a sphere of radius ``1 + latent[0]``; one parameter, for device/dtype."""

        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            latent, xyz = x[:, :1], x[:, 1:]
            return (xyz.norm(dim=1, keepdim=True) - (1.0 + latent)) + 0.0 * self.p

    def test_the_callers_mesh_is_the_returned_object_and_its_points_move(self):
        """Measured: 82 points, moved by up to 0.498 over three steps toward radius 1.5."""
        mesh = Mesh(pv.Sphere(radius=1.0, theta_resolution=10, phi_resolution=10))
        before = mesh.point_coords.copy()
        out = interpolate_mesh(
            self._SphereSDF(), np.array([0.0]), np.array([0.5]), n_steps=3, mesh=mesh
        )
        assert out is mesh
        assert np.abs(before - mesh.point_coords).max() > 0.1

    @pytest.mark.xfail(strict=True, reason="#55 site 3 — documented in this slice's commit 5")
    @pytest.mark.parametrize("func", [interpolate_mesh, interpolate_points])
    def test_the_public_entry_points_say_they_mutate(self, func):
        """
        A docs-only fix needs a falsifiable pin or it is not a fix. ``in place`` is the
        phrase ``interpolate_common`` already uses, so this asserts the contract reached
        the frame above it rather than asserting any particular wording.
        """
        assert "in place" in func.__doc__
