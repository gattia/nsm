"""
``mesh_names`` validation in ``train_deep_sdf``.

Surface identity is the order of each subject's mesh-path list, so the names are best
declared on the dataset (#52). The trainer adopts the dataset's names, or refuses a config
that disagrees, before anything reaches ``model_params_config.json``.
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest

from NSM.datasets import MultiSurfaceSDFSamples
from NSM.train.train_deep_sdf import train_deep_sdf

#: Stops ``train_deep_sdf`` right after validation.
STOP_AFTER_VALIDATION = "NSM.train.train_deep_sdf.add_plain_lr_to_config"


def validate(config, dataset_names=None):
    """Run ``train_deep_sdf`` up to its first step after validation; return its warnings."""
    with patch(STOP_AFTER_VALIDATION, side_effect=StopIteration):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(StopIteration):
                train_deep_sdf(
                    config, model=MagicMock(), sdf_dataset=MagicMock(mesh_names=dataset_names)
                )
    return [w for w in caught if "mesh_names" in str(w.message)]


def test_the_config_names_must_match_the_surface_count():
    """
    Fails if ``train_deep_sdf`` accepts ``mesh_names`` of a length other than
    ``objects_per_decoder``, warns on a named or single-surface config, or stops warning on
    an unnamed multi-surface one.
    """
    with pytest.raises(ValueError, match="mesh_names has 1 entries"):
        train_deep_sdf(
            {"objects_per_decoder": 2, "mesh_names": ["bone"]},
            model=MagicMock(),
            sdf_dataset=MagicMock(mesh_names=None),
        )
    assert validate({"objects_per_decoder": 2, "mesh_names": ["bone", "cart"]}) == []
    assert validate({"objects_per_decoder": 1, "mesh_names": None}) == []
    assert len(validate({"objects_per_decoder": 3, "mesh_names": None})) == 1


def test_the_dataset_declares_the_names_and_the_trainer_adopts_them():
    """
    Fails if ``MultiSurfaceSDFSamples`` accepts ``mesh_names`` of the wrong length, or
    ``train_deep_sdf`` does not copy the dataset's names into a config that has none, or
    accepts a config whose names are in a different order from the dataset's (#52).
    """
    with pytest.raises(ValueError, match="mesh_names"):
        MultiSurfaceSDFSamples(
            list_mesh_paths=[["a.vtk", "b.vtk"]], subsample=4, mesh_names=["bone"]
        )

    config = {"objects_per_decoder": 2, "mesh_names": None}
    validate(config, dataset_names=["bone", "cart"])
    assert config["mesh_names"] == ["bone", "cart"]

    with pytest.raises(ValueError, match="mesh_names"):
        train_deep_sdf(
            {"objects_per_decoder": 2, "mesh_names": ["bone", "cart"]},
            model=MagicMock(),
            sdf_dataset=MagicMock(mesh_names=["cart", "bone"]),
        )
