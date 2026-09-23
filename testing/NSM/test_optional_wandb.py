"""
wandb is optional (#5): needed when a caller asks for wandb logging, never at import.

Each module guards ``import wandb`` behind a ``wandb = None`` sentinel. Every explicit
request (``log_wandb``, ``use_wandb``) raises ``ImportError`` naming wandb, at entry, when it
is absent.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import wandb

import NSM.reconstruct.main as recon_main
import NSM.train.train_deep_sdf as trainer
from NSM.reconstruct.reconstruct_latent_S3 import reconstruct_latent_S3

REPO_ROOT = Path(__file__).resolve().parents[2]

#: wandb is installed here, so absence is simulated in a fresh interpreter by a
#: meta-path blocker that makes every ``import wandb`` raise ``ModuleNotFoundError``.
_PROBE = """
import sys


class _BlockWandb:
    def find_spec(self, name, path=None, target=None):
        if name == "wandb" or name.startswith("wandb."):
            raise ModuleNotFoundError(f"No module named '{name}' (blocked by test)")


sys.meta_path.insert(0, _BlockWandb())

import NSM.reconstruct
import NSM.train

assert NSM.reconstruct.main.wandb is None
assert NSM.reconstruct.latent_fit.wandb is None
assert NSM.train.train_deep_sdf.wandb is None
"""


def test_both_packages_import_without_wandb():
    """
    Fails if importing ``NSM.reconstruct`` or ``NSM.train`` needs wandb, or
    ``reconstruct.main``, ``latent_fit`` or ``train_deep_sdf`` does not bind ``wandb = None``
    when it is absent (#5).
    """
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=300,
    )
    assert result.returncode == 0, result.stderr


def test_every_explicit_request_raises_without_wandb(monkeypatch):
    """
    Fails if ``reconstruct_mesh``, ``get_mean_errors``, ``reconstruct_latent``,
    ``prepare_results_for_wandb``, ``reconstruct_latent_S3`` or ``train_deep_sdf``, asked for
    wandb logging with wandb absent, does not raise ``ImportError`` at entry (#5).

    The other arguments are ``None``, so a check placed later fails with a different error.
    ``train_epoch``'s ``log_latent`` is not covered: it raises only after the epoch has run.
    """
    requests = {
        recon_main.reconstruct_mesh: dict(path=None, decoders=None, latent_size=4, log_wandb=True),
        recon_main.get_mean_errors: dict(
            mesh_paths=["a.vtk"], decoders=None, latent_size=4, log_wandb=True
        ),
        recon_main.reconstruct_latent: dict(
            decoders=None,
            num_iterations=1,
            latent_size=2,
            xyz=None,
            sdf_gt=None,
            pts_surface=[0],
            log_wandb=True,
        ),
        recon_main.prepare_results_for_wandb: dict(result={}),
        reconstruct_latent_S3: dict(
            decoder=None, num_iterations=1, latent_size=2, new_sdf=None, log_wandb=True
        ),
        trainer.train_deep_sdf: dict(config={}, model=None, sdf_dataset=None, use_wandb=True),
    }
    for function, kwargs in requests.items():
        monkeypatch.setattr(sys.modules[function.__module__], "wandb", None)
        with pytest.raises(ImportError, match="wandb"):
            function(**kwargs)


def test_metric_histograms_follow_wandb(monkeypatch):
    """
    Fails if ``get_mean_errors`` stops adding ``<metric>_hist`` (a ``wandb.Histogram`` with
    wandb, ``None`` without), or loses the metric value when wandb is absent (#5).

    The histogram has no ``log_wandb`` gate: training validation reaches it with no wandb
    request, so an absent wandb must skip it, not raise. ``reconstruct_mesh`` is stubbed.
    """
    monkeypatch.setattr(
        recon_main,
        "reconstruct_mesh",
        lambda path=None, **kwargs: {"mesh": [None], "chamfer_0": 0.5},
    )

    def run():
        return recon_main.get_mean_errors(
            mesh_paths=["a.vtk", "b.vtk"],
            decoders=None,
            latent_size=4,
            calc_symmetric_chamfer=True,
        )

    results = run()
    assert results["chamfer_0"] == 0.5
    assert isinstance(results["chamfer_0_hist"], wandb.Histogram)

    monkeypatch.setattr(sys.modules[recon_main.get_mean_errors.__module__], "wandb", None)
    results = run()
    assert results["chamfer_0"] == 0.5
    assert results["chamfer_0_hist"] is None
