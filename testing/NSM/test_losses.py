"""
The eikonal loss is gated off (plan §8.2), and each of its three entry points refuses it.

Delete this file as part of making the loss work. ``train_epoch`` carries its own gate
because it is public: a caller reaches it without going through ``train_deep_sdf``.
"""

from unittest.mock import MagicMock

import pytest
import torch

from NSM.reconstruct import reconstruct_latent
from NSM.train.train_deep_sdf import train_deep_sdf, train_epoch
from NSM.utils import get_latent_vecs, get_learning_rate_schedules, get_optimizer


class _MLP(torch.nn.Module):
    def __init__(self, latent_size=8):
        super().__init__()
        self.linear = torch.nn.Linear(latent_size + 3, 1)

    def forward(self, x=None, latent=None, xyz=None, epoch=None):
        if x is None:
            x = torch.cat([latent.expand(xyz.shape[0], -1), xyz], dim=1)
        return self.linear(x)


def test_reconstruct_latent_refuses_a_positive_weight():
    """
    Fails if ``reconstruct_latent`` accepts ``eikonal_weight > 0`` instead of raising
    ``NotImplementedError``, or stops fitting at weight 0.
    """

    def fit(weight):
        return reconstruct_latent(
            decoders=[_MLP()],
            num_iterations=2,
            latent_size=8,
            xyz=torch.rand(16, 3),
            sdf_gt=torch.rand(16, 1),
            pts_surface=[0] * 16,
            device="cpu",
            eikonal_weight=weight,
        )

    with pytest.raises(NotImplementedError, match="eikonal"):
        fit(0.1)
    assert isinstance(fit(0)[1], torch.Tensor)


def test_train_deep_sdf_refuses_before_anything_is_built():
    """
    Fails if ``train_deep_sdf`` accepts ``eikonal_weight > 0``, or refuses it only after
    reading the rest of the config.

    The rest of the config is empty, so a gate placed after the first config lookup fails
    with a different error.
    """
    with pytest.raises(NotImplementedError, match="eikonal"):
        train_deep_sdf({"eikonal_weight": 1e-9}, model=MagicMock(), sdf_dataset=MagicMock())


def test_train_epoch_refuses_a_positive_weight():
    """
    Fails if ``train_epoch``, called directly, accepts ``eikonal_weight > 0`` instead of
    raising ``NotImplementedError``, or reports an ``eikonal_loss`` at weight 0.
    """

    def epoch(weight):
        config = {
            "optimizer": "Adam",
            "device": "cpu",
            "batch_split": 1,
            "samples_per_object_per_batch": 4,
            "enforce_minmax": False,
            "clamp_dist": 1.0,
            "surface_accuracy_e": None,
            "sample_difficulty_weight": None,
            "code_regularization": False,
            "n_epochs": 1,
            "grad_clip": None,
            "log_latent": None,
            "latent_size": 8,
            "latent_bound": 10,
            "latent_init_std": 0.01,
            "latent_init_normal": True,
            "variational": False,
            "eikonal_weight": weight,
            "LearningRateSchedule": [
                {"Target": "model", "Type": "Constant", "Value": 1e-3},
                {"Target": "latent", "Type": "Constant", "Value": 1e-3},
            ],
        }
        config["lr_schedules"] = get_learning_rate_schedules(config)
        torch.manual_seed(0)
        model = _MLP()
        latent_vecs = get_latent_vecs(2, config)
        data = [({"xyz": torch.rand(4, 3), "gt_sdf": torch.rand(4, 1)}, i) for i in range(2)]
        optimizer = get_optimizer(model, latent_vecs, lr_schedules=config["lr_schedules"])
        return train_epoch(
            model,
            torch.utils.data.DataLoader(data, batch_size=2),
            latent_vecs,
            optimizer,
            config,
            epoch=1,
            n_surfaces=1,
        )

    with pytest.raises(NotImplementedError, match="eikonal"):
        epoch(0.1)
    assert "eikonal_loss" not in epoch(0.0)
