"""
What ``train_deep_sdf`` checks before its first epoch: ``mesh_names``, and the config that
validation reads.

Surface identity is the order of each subject's mesh-path list, so the names are best
declared on the dataset (#52). The trainer adopts the dataset's names, or refuses a config
that disagrees, before anything reaches ``model_params_config.json``.
"""

import json
import warnings
from unittest.mock import MagicMock, patch

import pytest

from NSM.configs.generate_sdf_default_config import DEFAULT_CONFIG_PATH
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


def test_a_config_validation_would_fail_on_is_refused_before_training(tmp_path):
    """
    Fails if ``train_deep_sdf`` with ``val_paths`` set starts training on a config that
    validation would fail on, or refuses a valid one. Validation first runs at a checkpoint
    epoch, which can be hours in.

    Each case is one mistake: a missing key, a non-bool ``l2reg_recon``, an unknown
    ``convergence_type_recon``, a validator for another surface count, a subject with the
    wrong number of meshes or a missing file, and ``predict_val_variables`` the path names
    do not carry.
    """
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        default = json.load(f)
    bone, cart = tmp_path / "bone_age_30-.vtk", tmp_path / "cart_age_30-.vtk"
    bone.touch()
    cart.touch()
    subject = [str(bone), str(cart)]
    valid = dict(default, val_paths=[subject])
    missing_key = dict(valid)
    del missing_key["chamfer"]

    cases = [
        (missing_key, KeyError, "chamfer"),
        (dict(valid, l2reg_recon=1), TypeError, "l2reg_recon"),
        (dict(valid, convergence_type_recon="recon-loss"), ValueError, "convergence_type_recon"),
        (dict(valid, recon_val_func_name="compare_cart_thickness_whole_joint"), ValueError, "6"),
        (dict(valid, val_paths=[subject[:1]]), ValueError, "objects_per_decoder"),
        (
            dict(valid, val_paths=[[str(bone), str(tmp_path / "gone.vtk")]]),
            FileNotFoundError,
            "gone",
        ),
        (dict(valid, predict_val_variables=["weight"]), ValueError, "predict_val_variables"),
    ]
    single = dict(
        valid,
        objects_per_decoder=1,
        mesh_names=["bone"],
        val_paths=[str(bone)],
        predict_val_variables=["age"],
    )
    with patch(STOP_AFTER_VALIDATION, side_effect=StopIteration):
        for config, error, match in cases:
            with pytest.raises(error, match=match):
                train_deep_sdf(config, model=MagicMock(), sdf_dataset=MagicMock(mesh_names=None))
        for config in (valid, single):
            with pytest.raises(StopIteration):
                train_deep_sdf(config, model=MagicMock(), sdf_dataset=MagicMock(mesh_names=None))
