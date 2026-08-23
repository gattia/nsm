"""
Pins on what ``reconstruct_mesh`` hands its collaborators, via recorder stubs — no
meshes are read and no fitting runs, so these cost nothing.

The two strict xfails are #16 and its class (plan §8.0.C): ``n_pts_random`` is forwarded
under a name neither reader has, and ``get_mean_errors`` forwards the deprecated
``batch_size_latent_recon`` into ``reconstruct_mesh``'s ``**kwargs`` — so every
validation pass prints the deprecation warning at itself. Both readers accept
``**kwargs``, which is what makes each of these silent.
"""

import pytest
import torch

import NSM.reconstruct.main as recon_main


class _Abort(Exception):
    """Raised by recorders so ``reconstruct_mesh`` stops at the recorded call."""


class TestNPtsRandomReachesTheReaders:
    @pytest.mark.xfail(strict=True, reason="#16: n_pts_random lands in the reader's **kwargs")
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

    @pytest.mark.xfail(strict=True, reason="#16: n_pts_random lands in the reader's **kwargs")
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
    def test_the_shim_warns_and_stays(self, capsys):
        """
        The ``batch_size_latent_recon`` check in ``reconstruct_mesh`` is the migration
        surface for external callers and outlives the in-repo plumbing removal. It fires
        before any work: an invalid ``path`` aborts immediately after it.
        """
        with pytest.raises(ValueError, match="path must be a string"):
            recon_main.reconstruct_mesh(
                path=42, decoders=None, latent_size=8, batch_size_latent_recon=1
            )
        assert "batch_size_latent_recon is deprecated" in capsys.readouterr().out

    @pytest.mark.xfail(
        strict=True,
        reason="#16's class: get_mean_errors forwards the deprecated kwarg it never uses",
    )
    def test_get_mean_errors_does_not_forward_the_deprecated_kwarg(self, monkeypatch):
        captured = {}

        def fake_reconstruct_mesh(path=None, **kwargs):
            captured.update(kwargs)
            return {"mesh": [], "latent": torch.zeros(1, 4)}

        monkeypatch.setattr(recon_main, "reconstruct_mesh", fake_reconstruct_mesh)
        recon_main.get_mean_errors(mesh_paths=["a.vtk"], decoders=None, latent_size=4)
        assert "batch_size_latent_recon" not in captured
