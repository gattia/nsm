"""
Pins on what ``reconstruct_mesh`` hands its collaborators, via recorder stubs — no
meshes are read and no fitting runs, so those cost nothing — plus the one end-to-end
run of the single-object sampled branch, which had never executed before #15/#16
(its sampler key crashed it, and its draw size was ignored).

The recorder tests pin #16 and its class (plan §8.0.C), both fixed and their strict
xfails unmarked: ``n_pts_random`` was forwarded under a name neither reader has, and
``get_mean_errors`` forwarded the deprecated ``batch_size_latent_recon`` into
``reconstruct_mesh``'s ``**kwargs`` — so every validation pass printed the deprecation
warning at itself. Both readers accept ``**kwargs``, which is what made each of these
silent.
"""

import logging

import numpy as np
import pytest
import torch

import NSM.reconstruct.main as recon_main


class _Abort(Exception):
    """Raised by recorders so ``reconstruct_mesh`` stops at the recorded call."""


class TestNPtsRandomReachesTheReaders:
    """Were the #16 strict xfails: the request reaches the readers as their ``n_pts``."""

    def test_the_multi_object_request_is_honoured(self, monkeypatch):
        captured = {}

        def recorder(**kwargs):
            captured.update(kwargs)
            raise _Abort

        monkeypatch.setattr(recon_main, "read_meshes_get_sampled_pts", recorder)
        with pytest.raises(_Abort):
            recon_main.reconstruct_mesh(
                path=["bone.vtk", "cart.vtk"],
                decoders=torch.nn.Linear(1, 1),
                latent_size=8,
                get_rand_pts=True,
                n_pts_random=200,
            )
        assert captured.get("n_pts") == [200, 200]
        assert "n_pts_random" not in captured

    def test_the_single_object_request_is_honoured(self, monkeypatch):
        captured = {}

        def recorder(path, **kwargs):
            captured.update(kwargs)
            raise _Abort

        monkeypatch.setattr(recon_main, "read_mesh_get_sampled_pts", recorder)
        with pytest.raises(_Abort):
            recon_main.reconstruct_mesh(
                path="bone.vtk",
                decoders=torch.nn.Linear(1, 1),
                latent_size=8,
                get_rand_pts=True,
                n_pts_random=200,
            )
        assert captured.get("n_pts") == 200
        assert "n_pts_random" not in captured


class TestDeprecatedBatchSizeLatentRecon:
    def test_the_shim_warns_and_stays(self, caplog, capsys):
        """
        The ``batch_size_latent_recon`` check in ``reconstruct_mesh`` is the migration
        surface for external callers and outlives the in-repo plumbing removal. It fires
        before any work: an invalid ``path`` aborts immediately after it.

        It is a ``logger.warning`` since §8.0.G, not a ``print`` -- so it reaches the
        host's handlers, and stdout stays the consumer's.
        """
        with caplog.at_level(logging.WARNING, logger="NSM"):
            with pytest.raises(ValueError, match="path must be a string"):
                recon_main.reconstruct_mesh(
                    path=42, decoders=None, latent_size=8, batch_size_latent_recon=1
                )
        assert any(
            "batch_size_latent_recon is deprecated" in record.getMessage()
            and record.levelno == logging.WARNING
            for record in caplog.records
        )
        assert capsys.readouterr().out == ""

    def test_get_mean_errors_does_not_forward_the_deprecated_kwarg(self, monkeypatch):
        """Was the #16-class strict xfail: the parameter and its plumbing are deleted."""
        captured = {}

        def fake_reconstruct_mesh(path=None, **kwargs):
            captured.update(kwargs)
            return {"mesh": [], "latent": torch.zeros(1, 4)}

        monkeypatch.setattr(recon_main, "reconstruct_mesh", fake_reconstruct_mesh)
        recon_main.get_mean_errors(mesh_paths=["a.vtk"], decoders=None, latent_size=4)
        assert "batch_size_latent_recon" not in captured


class TestGetMeanErrorsSurvivesADegenerateModel:
    """
    #29's aggregate seam. ``reconstruct_mesh`` raises ``NoZeroLevelSetError`` when the
    mean shape has no surface; ``get_mean_errors`` must catch it and score NaN -- a
    training run has to survive its own early validation epochs. Until Aug 2026 the
    failure was invisible instead: NaN metrics but a *zero* latent recorded as if
    fitted, so ``val_prediction_*`` was regressed against fabrications (History §10).
    """

    def test_nan_scores_and_no_crash(self, monkeypatch):
        def degenerate_reconstruct_mesh(path=None, **kwargs):
            raise recon_main.NoZeroLevelSetError("no zero level set (stub)")

        monkeypatch.setattr(recon_main, "reconstruct_mesh", degenerate_reconstruct_mesh)
        ages = (45, 60)
        paths = [f"subj{i}_age_{a}-mesh.vtk" for i, a in enumerate(ages)]
        results = recon_main.get_mean_errors(
            mesh_paths=paths,
            decoders=None,
            latent_size=4,
            calc_symmetric_chamfer=True,
            calc_assd=True,
            predict_val_variables=["age"],
        )
        assert np.isnan(results["chamfer_0"])
        assert np.isnan(results["assd_0"])
        assert np.isnan(results["val_prediction_age"])


class SphereDecoder(torch.nn.Module):
    """
    An analytic sphere SDF (radius 0.5), one output column. ``+ 0.0 * latent.sum()``
    keeps the output on the latent's graph so ``reconstruct_latent``'s backward has a
    leaf to reach. Matches the calling convention ``mesh/main.decode_sdf`` inspects for
    — keyword ``latent``/``xyz`` — like the harness's ``NoZeroLevelSetDecoder``.
    """

    def forward(self, x=None, latent=None, xyz=None, epoch=None, verbose=False):
        pts = xyz if xyz is not None else x[:, -3:]
        sdf = torch.norm(pts, dim=1, keepdim=True) - 0.5
        if latent is not None:
            sdf = sdf + 0.0 * latent.sum()
        return sdf


class TestSingleObjectSampledBranch:
    """
    The one end-to-end run of ``reconstruct_mesh`` with a single path and
    ``get_rand_pts=True``. This branch had never executed: the sampler-key mismatch
    (#15) crashed it at the handoff, and even reaching it would have drawn 200,000
    points because ``n_pts_random`` was swallowed (#16). Cheap now precisely because
    the request is honoured.
    """

    def test_it_completes_and_returns_the_dict_contract(self, tmp_path):
        pv = pytest.importorskip("pyvista")
        sphere = pv.Sphere(radius=0.5, theta_resolution=18, phi_resolution=18).triangulate()
        path = str(tmp_path / "sphere.vtk")
        sphere.save(path)

        result = recon_main.reconstruct_mesh(
            path=path,
            decoders=SphereDecoder(),
            latent_size=8,
            num_iterations=2,
            get_rand_pts=True,
            n_pts_random=50,
            sigma_rand_pts=0.05,
            seed=3,
            n_pts_per_axis=32,
            return_latent=True,
            fix_mesh=False,
            device="cpu",
        )

        assert result["latent"].shape == (1, 8)
        assert len(result["mesh"]) == 1
        assert result["mesh"][0] is not None
        assert np.asarray(result["mesh"][0].point_coords).shape[1] == 3
