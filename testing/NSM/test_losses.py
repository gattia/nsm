"""The eikonal loss is gated off; these tests assert it stays refused.

They fail the moment someone makes it work, which is the point -- delete this file as
part of fixing it. Why it is gated:
.claude/plans/NSM_CODE_HEALTH_REFACTOR.md section 8.2.

**Three entry points, not two.** ``train_deep_sdf`` and ``reconstruct_latent`` are the two
the plan and CLAUDE.md name, and both are orchestrators. ``train_epoch`` is the code they
guard, it is a frozen public name (``test_train_import_compat``), and
``train_deep_sdf_multi_head`` calls it with no gate of its own -- so it is reachable two
ways. Counting the doors is this file's job (§8.0.L).
"""

from unittest.mock import MagicMock

import pytest
import torch

from NSM.reconstruct import reconstruct_latent
from NSM.train.train_deep_sdf import train_deep_sdf, train_epoch


class _MLP(torch.nn.Module):
    """Minimal decoder with the keyword interface reconstruct_latent requires."""

    def __init__(self, latent_size=8):
        super().__init__()
        self.linear = torch.nn.Linear(latent_size + 3, 1)

    def forward(self, x=None, latent=None, xyz=None, **kwargs):
        return self.linear(torch.cat([latent.expand(xyz.shape[0], -1), xyz], dim=1))


def _recon(**kwargs):
    return reconstruct_latent(
        decoders=[_MLP()],
        num_iterations=2,
        latent_size=8,
        xyz=torch.rand(16, 3),
        sdf_gt=torch.rand(16, 1),
        pts_surface=[0] * 16,
        device="cpu",
        **kwargs,
    )


@pytest.mark.parametrize("weight", [0.1, 1.0])
def test_reconstruct_refuses_eikonal_weight(weight):
    with pytest.raises(NotImplementedError, match="eikonal"):
        _recon(eikonal_weight=weight)


@pytest.mark.parametrize("weight", [0, 0.0])
def test_reconstruct_allows_eikonal_weight_off(weight):
    loss, latent = _recon(eikonal_weight=weight)
    assert isinstance(latent, torch.Tensor)


def test_train_refuses_eikonal_weight():
    with pytest.raises(NotImplementedError, match="eikonal"):
        train_deep_sdf({"eikonal_weight": 0.1}, model=MagicMock(), sdf_dataset=MagicMock())


def test_the_guard_fires_before_anything_expensive():
    """A config that is otherwise empty must still raise -- the point is failing in
    seconds, not after the dataset and optimizer are built."""
    with pytest.raises(NotImplementedError, match="eikonal"):
        train_deep_sdf({"eikonal_weight": 1e-9}, model=MagicMock(), sdf_dataset=MagicMock())


def _one_epoch(**config_overrides):
    """One CPU epoch through ``train_epoch``, on the smallest inputs it accepts."""
    from NSM.utils import get_latent_vecs, get_learning_rate_schedules, get_optimizer

    class _Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8 + 3, 1)

        def forward(self, x, epoch=None):
            return self.linear(x)

    class _Data(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {"xyz": torch.rand(4, 3), "gt_sdf": torch.rand(4, 1)}, index

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
        "code_regularization_warmup": 1,
        "code_cyclic_anneal": False,
        "n_epochs": 1,
        "grad_clip": None,
        "verbose": False,
        "log_latent": None,
        "latent_size": 8,
        "latent_bound": 10,
        "latent_init_std": 0.01,
        "latent_init_normal": True,
        "variational": False,
        "LearningRateSchedule": [
            {"Target": "model", "Type": "Constant", "Value": 1e-3},
            {"Target": "latent", "Type": "Constant", "Value": 1e-3},
        ],
    }
    config.update(config_overrides)
    config["lr_schedules"] = get_learning_rate_schedules(config)

    torch.manual_seed(0)
    model = _Linear()
    dataset = _Data()
    latent_vecs = get_latent_vecs(len(dataset), config)
    optimizer = get_optimizer(
        model, latent_vecs, lr_schedules=config["lr_schedules"], optimizer="Adam"
    )
    return train_epoch(
        model,
        torch.utils.data.DataLoader(dataset, batch_size=2),
        latent_vecs,
        optimizer,
        config,
        epoch=1,
        n_surfaces=1,
    )


def test_train_epoch_allows_eikonal_weight_off():
    """The other half of the gate: an epoch with the weight at zero still runs."""
    assert "eikonal_loss" not in _one_epoch(eikonal_weight=0.0)


def test_train_epoch_refuses_eikonal_weight():
    """
    Was a strict xfail. Measured before the fix: an epoch with ``eikonal_weight=0.1``
    completed and returned an ``eikonal_loss`` key. It did not even hit the
    double-backward crash §8.2 describes -- that needs a decoder whose second derivative
    torch declines, and a linear one has a perfectly good one. A gate is what stops this,
    not the arithmetic.
    """
    with pytest.raises(NotImplementedError, match="eikonal"):
        _one_epoch(eikonal_weight=0.1)
