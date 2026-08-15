"""
GPU-only checks. Skipped entirely when no CUDA device is present, which is every CI run.

**The CPU baselines in this harness do not bound GPU divergence.** That is not a caveat,
it is measured here: the same reconstruction, same seed, same weights, run on CUDA instead
of CPU moves the fitted latent by ~4e-2 against a CPU tolerance of 1e-4, changes the
reconstructed vertex count, and shifts the surface centroid by ~3e-3 against a CPU
tolerance of 1e-4. A GPU run is a different numerical experiment. If GPU results ever need
regression cover, they need their own baselines, generated on pinned hardware.

The second thing here is the seed-ordering constraint the downstream consumer builds on.
``kneepipeline/steps/run_nsm.py:172`` seeds *after* ``model.cuda()`` and documents why:
"model.cuda() consumes CUDA random state". On this stack that is not observable -- see
``TestSeedOrderingAroundCudaTransfer``, which pins the measurement rather than the belief.
"""

import numpy as np
import pytest
import torch
from _harness import ARCHITECTURE, build_model, run_reconstruction

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

#: Tolerances the CPU modules use, repeated here so the comparison is explicit.
CPU_LATENT_ATOL = 1e-4
CPU_GEOMETRY_ATOL = 1e-4


def on_cuda(model):
    """
    A CUDA copy that leaves ``model`` alone.

    ``reconstruct_mesh`` calls ``decoders[i].to(device)``, which moves a module in place --
    so handing it a session-scoped fixture would migrate that fixture to the GPU and
    change every CPU test that ran afterwards.
    """
    copy = build_model(dict(ARCHITECTURE))
    copy.load_state_dict(model.state_dict())
    copy.eval()
    return copy.cuda()


class TestSeedOrderingAroundCudaTransfer:
    """
    The consumer's rule is "seed AFTER ``.cuda()``, never before". Whether that matters is
    a property of the torch build, so it is measured, not assumed. On torch 2.8.0+cu128
    the two orderings give the same CUDA random stream.

    These tests pin the measurement. If a torch upgrade makes ``.cuda()`` consume RNG state
    again, they go red -- which is exactly when the consumer's ordering rule becomes
    load-bearing and someone needs to know.
    """

    @staticmethod
    def _draw(seed_before_transfer):
        model = build_model(dict(ARCHITECTURE), seed=0)
        if seed_before_transfer:
            torch.manual_seed(42)
            model.cuda()
        else:
            model.cuda()
            torch.manual_seed(42)
        return torch.randn(8, device="cuda").cpu()

    def test_both_orderings_give_the_same_cuda_random_stream(self):
        assert torch.equal(self._draw(True), self._draw(False)), (
            "seeding before .cuda() now differs from seeding after -- the consumer's "
            "ordering requirement in kneepipeline/steps/run_nsm.py is load-bearing again"
        )

    def test_a_cuda_transfer_between_seed_and_draw_changes_nothing(self):
        # Both models are built before either seed call: build_model() seeds too, and
        # doing it between the seed and the draw would be measuring the wrong thing.
        model = build_model(dict(ARCHITECTURE), seed=0)

        torch.manual_seed(7)
        without = torch.randn(8, device="cuda").cpu()

        torch.manual_seed(7)
        model.cuda()
        with_transfer = torch.randn(8, device="cuda").cpu()

        assert torch.equal(without, with_transfer)

    def test_the_seed_is_what_decides_the_stream(self):
        """Guard: if seeding did nothing, the tests above would pass vacuously."""
        torch.manual_seed(1)
        one = torch.randn(8, device="cuda").cpu()
        torch.manual_seed(2)
        two = torch.randn(8, device="cuda").cpu()
        assert not torch.equal(one, two)


class TestGpuDivergesFromTheCpuBaseline:
    """
    Runs the harness's reconstruction on CUDA and compares it to the CPU result the
    baselines were generated from.
    """

    @pytest.fixture(scope="class")
    def gpu_reconstruction(self, synthetic_meshes, reconstruction_model):
        return run_reconstruction(synthetic_meshes[0], on_cuda(reconstruction_model), device="cuda")

    def test_the_gpu_latent_is_outside_the_cpu_tolerance(self, gpu_reconstruction, reconstruction):
        """
        The headline: CPU baselines applied to a GPU run fail, by a wide margin. Anyone
        tempted to run this harness on a GPU box and trust a green result should see this
        test first.
        """
        cpu = reconstruction["latent"].detach().cpu().numpy().ravel()
        gpu = gpu_reconstruction["latent"].detach().cpu().numpy().ravel()
        divergence = float(np.abs(cpu - gpu).max())
        assert divergence > CPU_LATENT_ATOL, (
            f"GPU and CPU latents now agree to {divergence:.3e}, inside the CPU tolerance "
            f"of {CPU_LATENT_ATOL}. If that holds generally, the CPU baselines could cover "
            f"GPU runs too -- worth confirming before relying on it."
        )

    def test_the_gpu_geometry_is_outside_the_cpu_tolerance(
        self, gpu_reconstruction, reconstruction
    ):
        cpu = np.asarray(reconstruction["mesh"][0].point_coords).mean(axis=0)
        gpu = np.asarray(gpu_reconstruction["mesh"][0].point_coords).mean(axis=0)
        assert float(np.abs(cpu - gpu).max()) > CPU_GEOMETRY_ATOL

    def test_the_gpu_result_is_still_the_same_shape(self, gpu_reconstruction, reconstruction):
        """
        Divergence is expected; a structurally different result is not. This is the part
        of the GPU path a CPU-only suite genuinely cannot cover.
        """
        assert len(gpu_reconstruction["mesh"]) == len(reconstruction["mesh"])
        assert all(mesh is not None for mesh in gpu_reconstruction["mesh"])
        assert gpu_reconstruction["latent"].shape == reconstruction["latent"].shape
        assert set(gpu_reconstruction) == set(reconstruction)

    def test_the_gpu_result_is_still_broadly_correct(self, gpu_reconstruction, reconstruction):
        """
        A loose bound, so a GPU-only breakage is still caught even though the tight CPU
        baselines cannot be used. 20% on the surface metrics is far above the ~2% observed
        divergence and far below anything that would count as broken.
        """
        for key in ("assd_0", "assd_1"):
            assert gpu_reconstruction[key] == pytest.approx(reconstruction[key], rel=0.2)

    def test_the_gpu_result_is_reproducible_only_to_about_1e_6(
        self, synthetic_meshes, reconstruction_model, gpu_reconstruction
    ):
        """
        Divergence from CPU is one thing; run-to-run stability on one device is another,
        and it bounds how tight a GPU baseline could ever be.

        The same reconstruction run twice on the same GPU is **not** bitwise identical --
        it agrees to roughly a float32 ulp. The CPU path, by contrast, is exactly
        reproducible (``test_reconstruction_regression`` asserts ``rtol=atol=0``). Any
        future GPU baseline needs a tolerance, and this measures the floor for it.
        """
        again = run_reconstruction(
            synthetic_meshes[0], on_cuda(reconstruction_model), device="cuda"
        )
        first = gpu_reconstruction["latent"].detach().cpu().numpy()
        second = again["latent"].detach().cpu().numpy()

        assert np.allclose(first, second, rtol=0, atol=1e-6), np.abs(first - second).max()
        if np.array_equal(first, second):
            pytest.skip("this GPU run was bitwise identical; the bound above still holds")
