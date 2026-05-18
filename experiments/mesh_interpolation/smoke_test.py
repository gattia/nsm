"""Smoke test for the experiment harness -- no GPU, no real data.

Exercises the full ``run_matrix`` cell path (``interpolate_points`` with every
fix config -> ``score_correspondence`` -> ``write_report``) against an analytic
sphere-SDF decoder, so the plumbing can be verified before committing a GPU
node to the real Phase 0 run.

Usage:
    python -m experiments.mesh_interpolation.smoke_test
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pyvista as pv
import torch
import torch.nn as nn

from .config import EXPERIMENT_CONFIGS
from .run_matrix import run_cell, write_report


class SphereSDF(nn.Module):
    """Analytic SDF decoder: the latent's first component is the sphere radius."""

    def __init__(self, d_lat=8):
        super().__init__()
        self.d_lat = d_lat
        self._param = nn.Parameter(torch.zeros(1))

    def forward(self, x=None, latent=None, xyz=None, epoch=None, verbose=False):
        if x is not None:
            latent, xyz = x[:, : self.d_lat], x[:, self.d_lat :]
        radius = latent[:, 0:1]
        return torch.norm(xyz, dim=1, keepdim=True) - radius


def _latent(radius, disambig_dim, d_lat=8):
    """A latent encoding the radius, with a unit in a unique dim for slerp."""
    z = np.zeros(d_lat, dtype=np.float64)
    z[0] = radius
    z[disambig_dim] = 1.0
    return z


def main():
    model = SphereSDF()

    # Three synthetic "knees": spheres of increasing radius.
    radii = {"knee_a": 1.0, "knee_b": 1.3, "knee_c": 1.6}
    disambig = {"knee_a": 4, "knee_b": 5, "knee_c": 6}
    latents = {k: _latent(r, disambig[k]) for k, r in radii.items()}
    meshes = {
        k: [pv.Sphere(radius=r, theta_resolution=16, phi_resolution=16)]
        for k, r in radii.items()
    }

    pairs = [("knee_a", "knee_b"), ("knee_b", "knee_c"), ("knee_c", "knee_a")]
    nfe_grid = [10, 50]

    rows = []
    for cfg_name, cfg_kwargs in EXPERIMENT_CONFIGS.items():
        for nfe in nfe_grid:
            for key_a, key_b in pairs:
                flat = run_cell(
                    model, latents, meshes, key_a, key_b, 0,
                    cfg_kwargs, nfe, roundtrip=True, self_intersect=True,
                )
                flat.update(
                    config=cfg_name, nfe=nfe, pair=f"{key_a}->{key_b}",
                    source=key_a, target=key_b, surface=0, surface_name="bone",
                    error="",
                )
                rows.append(flat)

    df = pd.DataFrame(rows)

    # --- assertions: the harness produced sane numbers --------------------
    assert len(df) == len(EXPERIMENT_CONFIGS) * len(nfe_grid) * len(pairs)
    assert df["assd"].notna().all(), "ASSD should be finite for every cell"
    # An analytic sphere warp should land close to the target surface.
    base = df[(df["config"] == "baseline") & (df["nfe"] == 50)]
    assert base["assd"].max() < 0.05, f"baseline ASSD too high: {base['assd'].max()}"
    assert base["off_surface_p95"].max() < 0.05, "baseline off-surface too high"
    # More corrector iterations should not make the terminal residual worse.
    for nfe in nfe_grid:
        b = df[(df["config"] == "baseline") & (df["nfe"] == nfe)]["off_surface_p95"].mean()
        f1 = df[(df["config"] == "fix1") & (df["nfe"] == nfe)]["off_surface_p95"].mean()
        assert f1 <= b + 1e-6, f"fix1 worse than baseline at NFE={nfe}: {f1} vs {b}"

    with tempfile.TemporaryDirectory() as tmp:
        report_path = os.path.join(tmp, "report.md")
        write_report(df, report_path)
        assert os.path.isfile(report_path)
        assert os.path.getsize(report_path) > 0

    print(f"SMOKE TEST PASSED: {len(df)} cells scored across "
          f"{len(EXPERIMENT_CONFIGS)} configs.")
    print(df.groupby("config")["assd"].mean().round(5).to_string())


if __name__ == "__main__":
    main()
