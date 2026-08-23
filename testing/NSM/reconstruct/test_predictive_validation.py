"""#48: ``get_mean_errors`` handed ``Regress.add_latent`` the whole result dict, so a
run with ``predict_val_variables`` enabled died with ``TypeError`` in ``calc_r2`` at
the end of its first validation pass — after all its reconstructions had run. The
seam now hands over the fitted latent as a flat float vector.
"""

import numpy as np
import pytest
import torch

import NSM.reconstruct.main as recon_main
from NSM.reconstruct.predictive_validation_class import Regress


class TestRegress:
    def test_r2_on_a_perfectly_encoded_factor(self):
        paths = [f"subj{i}_age_{a}-mesh.vtk" for i, a in enumerate((45, 60, 50))]
        reg = Regress(list_factors=["age"], list_paths=paths)
        rng = np.random.default_rng(0)
        for age in (45, 60, 50):
            reg.add_latent(np.r_[age / 100.0, rng.normal(size=7)])
        results = reg.calc_r2()
        assert results["val_prediction_age"] == pytest.approx(1.0)


class TestGetMeanErrorsPredictiveValidation:
    def test_latents_reach_the_regressor(self, monkeypatch):
        """The stub returns a (1, L) grad-tracking tensor under ``"latent"`` — the
        shape and autograd state ``reconstruct_mesh`` actually returns — encoding the
        factor, so ``r2 == 1`` proves the vector survived the handoff intact.
        """
        ages = (45, 60, 50, 55)
        paths = [f"subj{i}_age_{a}-mesh.vtk" for i, a in enumerate(ages)]

        def fake_reconstruct_mesh(path=None, **kwargs):
            age = float(path.split("age_")[1].split("-")[0])
            return {"mesh": [], "latent": torch.full((1, 4), age / 100.0, requires_grad=True)}

        monkeypatch.setattr(recon_main, "reconstruct_mesh", fake_reconstruct_mesh)
        results = recon_main.get_mean_errors(
            mesh_paths=paths,
            decoders=None,
            latent_size=4,
            predict_val_variables=["age"],
        )
        assert results["val_prediction_age"] == pytest.approx(1.0)
