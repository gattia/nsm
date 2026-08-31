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

* ``compute_recon_loss`` mutated for a reason that turned out not to hold, so the code
  went with it (``CLAUDE.md``: never inherit a rationale along with the code).
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
    got back ``[-1, 0.5, 1]`` and ``[-1, 1]`` at ``clamp_dist=1.0``, in its own list
    object, with each slot rebound. ``reconstruct_latent`` is the only caller and passes
    the list it built, so nothing in the repo noticed — but the list is the caller's
    ``sdf_gt`` argument, one frame up.

    Note what was *not* mutated: ``torch.clamp`` returns a new tensor, so the caller's
    tensors were always safe. What moved was the list's slots.
    """

    def _caller_list(self):
        return [torch.tensor([[-5.0], [0.5], [5.0]]), torch.tensor([[-9.0], [9.0]])]

    def test_the_callers_list_object_is_not_returned(self):
        sdf_gt = self._caller_list()
        assert reconstruct_latent_preprocess_sdf_gt(sdf_gt, 1.0, device="cpu") is not sdf_gt

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

    def test_the_type_check_does_not_hand_back_the_callers_list(self):
        """Same class, one frame earlier: a list in is the same object out."""
        sdf_gt = [torch.tensor([[1.0]])]
        assert reconstruct_latent_sdf_gt_type_check(sdf_gt) is not sdf_gt


class TestSite2TheAssdDowncast:
    """
    ``compute_recon_loss(calc_assd=True)`` moved the caller's meshes to ``float32``.

    Both of them — the reconstruction and the caller's ground truth — under a comment
    reading "make sure the points for the meshes are the same types". **In this suite's
    pymskt the comment's reason does not hold**: mskt 0.1.21's ``pcu_sdf`` casts the
    query points *and* the mesh vertices to ``float64`` itself
    (``points_dtype=np.float64``), so the caller's dtype never reaches the computation.
    Measured, every mixed and matched combination returns the identical value. The
    downcast could only lose precision, never supply it, so it was deleted rather than
    performed on copies — and *that* scoping error is ``TestSite2TheMixedDtypePair``'s
    story: on mskt 0.1.19 the reason does hold, and the deletion crashed production.

    **The mutation was conditional on a flag**, which is what made it hard to notice: the
    chamfer path left both at ``float64``. A caller that scored chamfer only, then added
    ASSD later, silently lost precision on meshes it still held.
    """

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
        Removing the downcast changed no reported number, asserted by running both paths
        on identical meshes **on the same platform**: one pair scored as-is, one scored
        after the ``astype(np.float32)`` the deleted code performed. Equal exactly, and
        by construction on any platform: ``_plain``'s ``point_coords`` *originate* as
        float32 (VTK stores points at single precision) before the fixture upcasts them,
        so the emulated round-trip is lossless, and ``pcu_sdf`` casts both sides to
        float64 either way.

        The first form of this pin transcribed the ASSD itself (0.09831770188973551) and
        went red on macOS CI — the constant encoded *linux's* sphere tessellation, off by
        8.8e-07 against macOS float32 geometry, ~9x the tolerance. The claim was never
        about the digits; a platform constant cannot pin a before/after invariant.

        For a caller holding genuine float64 meshes the old cast was **not** a no-op —
        measured, a sphere pair perturbed below float32 resolution moves ASSD by
        **7.2e-09** through the round-trip. That is § History 28's population.
        """
        recon_a, orig_a = _plain(1.0), _plain(1.1)
        recon_b, orig_b = _plain(1.0), _plain(1.1)
        # What the deleted lines in compute_recon_loss did to the caller's meshes.
        recon_b.point_coords = recon_b.point_coords.astype(np.float32)
        orig_b.point_coords = orig_b.point_coords.astype(np.float32)

        as_is = compute_recon_loss([recon_a], [orig_a], calc_assd=True)["assd_0"]
        downcast = compute_recon_loss([recon_b], [orig_b], calc_assd=True)["assd_0"]

        assert as_is == downcast
        # A sanity bound, not a transcription: red if the metric returns nonsense,
        # indifferent to the platform's last digits.
        assert 0.05 < as_is < 0.15


class TestSite2TheMixedDtypePair:
    """
    The downcast's comment — "make sure the points for the meshes are the same types" —
    had a real reason after all, and deleting the cast outright broke production.

    Whether ``get_assd_mesh`` tolerates a mixed-dtype pair depends on the pymskt
    version: mskt 0.1.21 (this suite's environment) casts both sides to float64 inside
    ``pcu_sdf``, but mskt 0.1.19 (the knee-pipeline production environment) hands both
    straight to ``point_cloud_utils``, which raises ``ValueError: Invalid type (double,
    Row Major) for argument 'v'`` — measured in that environment, 2026-08-30, the day
    after the deletion merged. And the production path *always* produces a mixed pair:
    the knee pipeline's input meshes carry float64 points on disk (measured on an
    archived job, all of them) while the reconstruction arrives float32 from marching
    cubes, so every ``fit_nsm`` call in the knee pipeline crashed at the ASSD step.

    The fix aligns a mixed pair on copies, upcasting the float32 side. This class pins
    the NSM-side half of the contract — the pair that reaches ``get_assd_mesh`` shares
    one dtype — because this suite's pymskt would forgive a mixed pair and hide the
    regression. The 0.1.19 behaviour itself can only be exercised in an environment
    that has it.
    """

    def _mixed(self):
        """The production shape: float32 reconstruction, float64 original."""
        recon, orig = _plain(1.0), _plain(1.1)
        recon.point_coords = recon.point_coords.astype(np.float32)
        return recon, orig

    def test_the_pair_reaching_get_assd_mesh_shares_one_dtype(self, monkeypatch):
        seen = []
        unpatched = Mesh.get_assd_mesh

        def spy(self, other_mesh):
            seen.append((self.point_coords.dtype, other_mesh.point_coords.dtype))
            return unpatched(self, other_mesh)

        monkeypatch.setattr(Mesh, "get_assd_mesh", spy)
        recon, orig = self._mixed()
        compute_recon_loss([recon], [orig], calc_assd=True)
        assert seen == [(np.float64, np.float64)]

    def test_a_mixed_pair_scores_and_the_caller_keeps_both_dtypes(self):
        recon, orig = self._mixed()
        result = compute_recon_loss([recon], [orig], calc_assd=True)
        assert 0.05 < result["assd_0"] < 0.15
        assert recon.point_coords.dtype == np.float32
        assert orig.point_coords.dtype == np.float64

    def test_the_aligned_value_is_the_matched_float64_value(self):
        """The upcast is exact, so aligning must not move the number: a mixed pair and
        an all-float64 pair of the same geometry score identically. (The fixture's
        points originate as float32 from VTK, so the float64 "original" is the same
        geometry bit-for-bit.)"""
        recon32, orig64 = self._mixed()
        mixed = compute_recon_loss([recon32], [orig64], calc_assd=True)
        matched = compute_recon_loss([_plain(1.0)], [_plain(1.1)], calc_assd=True)
        assert mixed["assd_0"] == matched["assd_0"]


class TestSite3TheInterpolationMesh:
    """
    ``interpolate_mesh`` advances the caller's mesh in place and returns it — **by
    design**, and that is the ruling rather than a defect to fix.

    What #55 asked for here was the documentation, and the documentation existed at the
    wrong frame. ``interpolate_common``, the private engine, said "the returned object is
    the caller's own mesh, mutated"; ``interpolate_mesh`` and ``interpolate_points``, the
    two public entry points, said nothing — the contract stated in the file the author had
    open and not in the one a caller reads.

    Writing the pin found that the two entry points are **opposite**, which this file's
    first draft had wrong: it asserted both would document a mutation.
    ``interpolate_points`` does not mutate, measured for both input types, so what each
    needed was its own half.
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

    def test_interpolate_points_does_not_mutate_its_input(self):
        """
        The measurement that corrected this file's first draft, which asserted both entry
        points document a mutation. ``interpolate_points`` converts to a tensor and
        returns a fresh ``ndarray``; the caller's points are untouched for an ndarray
        *and* for a torch tensor already on the device. So the pair is **opposite**, not
        alike, and that is the contract worth writing down.
        """
        model = self._SphereSDF()
        array = np.random.default_rng(0).normal(size=(20, 3))
        before = array.copy()
        out = interpolate_points(model, np.array([0.0]), np.array([0.5]), n_steps=2, points1=array)
        assert out is not array and np.array_equal(array, before)

        tensor = torch.tensor(before, dtype=torch.float)
        interpolate_points(model, np.array([0.0]), np.array([0.5]), n_steps=2, points1=tensor)
        assert torch.equal(tensor, torch.tensor(before, dtype=torch.float))

    @pytest.mark.parametrize(
        "func,phrase",
        [(interpolate_mesh, "in place"), (interpolate_points, "not modified")],
        ids=["mesh-mutates", "points-does-not"],
    )
    def test_each_public_entry_point_states_its_own_half(self, func, phrase):
        """
        A docs-only fix needs a falsifiable pin or it is not a fix. ``in place`` is the
        phrase ``interpolate_common`` already used, so the first half asserts the contract
        reached the frame above it; the second asserts the sibling says the opposite
        rather than saying nothing, which is how a caller reading one of them learns there
        is a choice to make.
        """
        assert phrase in func.__doc__


def test_a_tuple_of_sdf_samples_now_survives_the_preprocess():
    """
    Fixed for free by the #55 repair, and worth a pin because it is the
    accepted-and-broken shape: ``reconstruct_latent_sdf_gt_type_check`` names ``tuple``
    as a supported type, and the preprocess below it assigned by index — so every tuple
    raised ``TypeError: 'tuple' object does not support item assignment``, at every call,
    since the function was written. Building a new list honours the declared type instead
    of the code needing to grow a branch for it.
    """
    sdf_gt = reconstruct_latent_sdf_gt_type_check((torch.tensor([[-5.0]]), torch.tensor([[5.0]])))
    result = reconstruct_latent_preprocess_sdf_gt(sdf_gt, 1.0, device="cpu")
    assert [tensor.item() for tensor in result] == [-1.0, 1.0]
