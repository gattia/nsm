"""
GPU-only checks. Skipped without a CUDA device, which is every CI run.

**The CPU baselines do not bound GPU divergence.** Measured on torch 2.8.0+cu128: the same
reconstruction on CUDA moves the fitted latent by ~2.9e-2 (``FITTED_LATENT_ATOL`` is 5e-4),
changes the vertex count, and moves the surface centroid by ~7.2e-4 (``GEOMETRY_ATOL`` is
3e-4). Two runs on the same GPU agree only to about 1e-6. A GPU baseline would need its own
file, generated on pinned hardware, with a tolerance above that floor.

Also measured on that stack: seeding before or after ``model.cuda()`` gives the same CUDA
random stream, so kneepipeline's "seed after ``.cuda()``" rule is not load-bearing there.
"""

import pytest
import torch
from _harness import ARCHITECTURE, build_model, run_reconstruction

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def test_a_gpu_reconstruction_has_the_same_shape_and_is_broadly_correct(
    synthetic_meshes, reconstruction_model, reconstruction
):
    """
    Fails if ``reconstruct_mesh(device="cuda")`` returns different keys, mesh count or latent
    shape from the CPU run, a ``None`` mesh, or an ASSD more than 20% off the CPU value.

    20% on the surface metrics is far above the measured divergence (0.1% and 0.9%) and far
    below anything broken. The model is copied first: ``reconstruct_mesh`` moves a module to
    the device in place, which would move the session fixture for every later CPU test.
    """
    copy = build_model(dict(ARCHITECTURE))
    copy.load_state_dict(reconstruction_model.state_dict())
    gpu = run_reconstruction(synthetic_meshes[0], copy.eval().cuda(), device="cuda")

    assert set(gpu) == set(reconstruction)
    assert len(gpu["mesh"]) == len(reconstruction["mesh"])
    assert all(mesh is not None for mesh in gpu["mesh"])
    assert gpu["latent"].shape == reconstruction["latent"].shape
    for key in ("assd_0", "assd_1"):
        assert gpu[key] == pytest.approx(reconstruction[key], rel=0.2)
