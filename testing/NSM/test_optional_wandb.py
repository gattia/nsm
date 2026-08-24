"""
#5: wandb must be optional — needed when a caller explicitly asks for wandb logging,
never at import time.

``import wandb`` sits at module top across the reconstruct and train packages, so with
wandb absent, ``import NSM.reconstruct`` (the consumer path — kneepipeline's
``reconstruct_mesh``) and ``import NSM.train`` both die with ``ModuleNotFoundError``,
and wandb appears nowhere in ``pyproject.toml``. Plan §8.0.E; the import probe is
strict-xfail until the fix commit.

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
    @pytest.mark.xfail(
        strict=True,
        reason="#5: top-level `import wandb` in reconstruct/ and train/ modules",
    )
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
