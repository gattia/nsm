"""The eikonal loss is gated off; these tests assert it stays refused.

They fail the moment someone makes it work, which is the point -- delete this file as
part of fixing it. Why it is gated:
.claude/plans/NSM_CODE_HEALTH_REFACTOR.md section 8.2.
"""

from unittest.mock import MagicMock

import pytest
import torch

from NSM.reconstruct import reconstruct_latent
from NSM.train.train_deep_sdf import train_deep_sdf


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
