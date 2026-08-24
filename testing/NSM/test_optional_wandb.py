"""
#5: wandb must be optional — needed when a caller explicitly asks for wandb logging,
never at import time.

Until the #5 fix (plan §8.0.E), ``import wandb`` sat at module top across the
reconstruct and train packages, so with wandb absent, ``import NSM.reconstruct`` (the
consumer path — kneepipeline's ``reconstruct_mesh``) and ``import NSM.train`` both died
with ``ModuleNotFoundError`` — and wandb appears nowhere in ``pyproject.toml``. Each
module now guards the import behind a ``wandb = None`` sentinel; every explicit request
(``log_wandb`` / ``use_wandb`` / ``config["log_latent"]``) raises ``ImportError`` by
name, at entry, when wandb is absent.

The probe runs in a subprocess: wandb is installed in the dev environment and already
imported by the suite, so absence can only be simulated in a fresh interpreter, via a
meta-path blocker. One subprocess covers both packages, because each interpreter start
pays the torch + pymskt import cost.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import wandb

import NSM.reconstruct.main as recon_main

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Blocks wandb the way an uninstalled environment would (raising in ``find_spec``
#: makes every ``import wandb`` see ``ModuleNotFoundError``), then imports both
#: packages and checks each guarded module's sentinel.
_PROBE = """
import sys


class _BlockWandb:
    def find_spec(self, name, path=None, target=None):
        if name == "wandb" or name.startswith("wandb."):
            raise ModuleNotFoundError(f"No module named '{name}' (blocked by test)")


sys.meta_path.insert(0, _BlockWandb())

import NSM.reconstruct
import NSM.train

assert NSM.reconstruct.main.wandb is None, "main's wandb sentinel"
assert NSM.reconstruct.latent_fit.wandb is None, "latent_fit's wandb sentinel"
assert NSM.train.train_deep_sdf.wandb is None, "train_deep_sdf's wandb sentinel"
print("OK")
"""


class TestImportWithoutWandb:
    def test_both_packages_import_without_wandb(self):
        result = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=300,
        )
        assert result.returncode == 0, result.stderr


class TestHistogramTailWithWandbPresent:
    """
    ``get_mean_errors``' metric tail builds ``wandb.Histogram`` per key whenever wandb
    is importable — it is the file's one wandb use with no ``log_wandb`` gate. Pinned
    so the #5 fix provably changes nothing for environments that have wandb: this test
    must stay green, untouched, across the fix.
    """

    def test_metric_hists_are_wandb_histograms(self, monkeypatch):
        def fake_reconstruct_mesh(path=None, **kwargs):
            return {"mesh": [None], "chamfer_0": 0.5}

        monkeypatch.setattr(recon_main, "reconstruct_mesh", fake_reconstruct_mesh)
        results = recon_main.get_mean_errors(
            mesh_paths=["a.vtk", "b.vtk"],
            decoders=None,
            latent_size=4,
            calc_symmetric_chamfer=True,
        )
        assert results["chamfer_0"] == 0.5
        assert isinstance(results["chamfer_0_hist"], wandb.Histogram)


class TestExplicitRequestsRaiseWithoutWandb:
    """
    Each explicit wandb request fails loudly, at entry, when wandb is absent. Absence
    is simulated by patching the sentinel to ``None`` in the module hosting the
    function — looked up via ``__module__`` where §8.0.E's split will move the
    function, so these tests survive the move without edits.
    """

    @staticmethod
    def _absent(monkeypatch, func):
        monkeypatch.setattr(sys.modules[func.__module__], "wandb", None)

    def test_reconstruct_mesh_log_wandb(self, monkeypatch):
        self._absent(monkeypatch, recon_main.reconstruct_mesh)
        with pytest.raises(ImportError, match="wandb"):
            recon_main.reconstruct_mesh(path=None, decoders=None, latent_size=4, log_wandb=True)

    def test_get_mean_errors_log_wandb(self, monkeypatch):
        self._absent(monkeypatch, recon_main.get_mean_errors)
        with pytest.raises(ImportError, match="wandb"):
            recon_main.get_mean_errors(
                mesh_paths=["a.vtk"], decoders=None, latent_size=4, log_wandb=True
            )

    def test_reconstruct_latent_log_wandb(self, monkeypatch):
        self._absent(monkeypatch, recon_main.reconstruct_latent)
        with pytest.raises(ImportError, match="wandb"):
            recon_main.reconstruct_latent(
                decoders=None,
                num_iterations=1,
                latent_size=2,
                xyz=None,
                sdf_gt=None,
                log_wandb=True,
            )

    def test_prepare_results_for_wandb(self, monkeypatch):
        self._absent(monkeypatch, recon_main.prepare_results_for_wandb)
        with pytest.raises(ImportError, match="wandb"):
            recon_main.prepare_results_for_wandb({})

    def test_reconstruct_latent_S3_log_wandb(self, monkeypatch):
        from NSM.reconstruct.reconstruct_latent_S3 import reconstruct_latent_S3

        self._absent(monkeypatch, reconstruct_latent_S3)
        with pytest.raises(ImportError, match="wandb"):
            reconstruct_latent_S3(
                decoder=None, num_iterations=1, latent_size=2, new_sdf=None, log_wandb=True
            )

    def test_train_deep_sdf_use_wandb(self, monkeypatch):
        import NSM.train.train_deep_sdf as trainer

        monkeypatch.setattr(trainer, "wandb", None)
        with pytest.raises(ImportError, match="wandb"):
            trainer.train_deep_sdf(config={}, model=None, sdf_dataset=None, use_wandb=True)

    def test_train_deep_sdf_multi_head_use_wandb(self, monkeypatch):
        import NSM.train.train_deep_sdf_multi_head as multi_head

        monkeypatch.setattr(multi_head, "wandb", None)
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ImportError, match="wandb"):
                multi_head.train_deep_sdf(config={}, models=(), sdf_dataset=None, use_wandb=True)


class TestValidationSurvivesWithoutWandb:
    """
    The one wandb use with *no* explicit request — ``get_mean_errors``' histogram
    tail — skips instead of raising when wandb is absent: a training run's validation
    epochs (which never ask for wandb) must complete in a wandb-less environment. The
    metric scalars are unaffected; the ``*_hist`` values become ``None``.
    """

    def test_hist_none_metrics_intact(self, monkeypatch):
        monkeypatch.setattr(sys.modules[recon_main.get_mean_errors.__module__], "wandb", None)

        def fake_reconstruct_mesh(path=None, **kwargs):
            return {"mesh": [None], "chamfer_0": 0.5}

        monkeypatch.setattr(recon_main, "reconstruct_mesh", fake_reconstruct_mesh)
        results = recon_main.get_mean_errors(
            mesh_paths=["a.vtk", "b.vtk"],
            decoders=None,
            latent_size=4,
            calc_symmetric_chamfer=True,
        )
        assert results["chamfer_0"] == 0.5
        assert results["chamfer_0_hist"] is None
