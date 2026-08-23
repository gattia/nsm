"""
Tests for mesh_names config validation in train_deep_sdf and train_deep_sdf_multi_head.
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest

from NSM.datasets import MultiSurfaceSDFSamples
from NSM.train.train_deep_sdf import train_deep_sdf as train_single
from NSM.train.train_deep_sdf_multi_head import train_deep_sdf as train_multi_head

# Patch add_plain_lr_to_config to stop execution after validation
PATCH_SINGLE = "NSM.train.train_deep_sdf.add_plain_lr_to_config"
PATCH_MULTI = "NSM.train.train_deep_sdf_multi_head.add_plain_lr_to_config"


def _make_single_config(**overrides):
    """Minimal config dict for train_deep_sdf (single-head)."""
    cfg = {}
    cfg.update(overrides)
    return cfg


def _make_multi_config(**overrides):
    """Minimal config dict for train_deep_sdf_multi_head."""
    cfg = {}
    cfg.update(overrides)
    return cfg


class TestSingleHeadMeshNamesValidation:
    """Tests for mesh_names validation in train_deep_sdf (single-head)."""

    def test_mesh_names_length_mismatch_raises(self):
        config = _make_single_config(
            objects_per_decoder=2,
            mesh_names=["bone"],
        )
        with pytest.raises(ValueError, match="mesh_names has 1 entries"):
            train_single(config, model=MagicMock(), sdf_dataset=MagicMock())

    def test_mesh_names_matching_length_no_error(self):
        config = _make_single_config(
            objects_per_decoder=2,
            mesh_names=["bone", "cart"],
        )
        with patch(PATCH_SINGLE, side_effect=StopIteration):
            with pytest.raises(StopIteration):
                train_single(config, model=MagicMock(), sdf_dataset=MagicMock())
        # If we got here, no ValueError was raised — validation passed

    def test_no_warning_when_single_surface(self):
        config = _make_single_config(objects_per_decoder=1, mesh_names=None)
        with patch(PATCH_SINGLE, side_effect=StopIteration):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with pytest.raises(StopIteration):
                    train_single(config, model=MagicMock(), sdf_dataset=MagicMock())
                mesh_warnings = [x for x in w if "mesh_names" in str(x.message)]
                assert len(mesh_warnings) == 0

    def test_warning_when_multi_surface_no_mesh_names(self):
        config = _make_single_config(objects_per_decoder=3, mesh_names=None)
        with patch(PATCH_SINGLE, side_effect=StopIteration):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with pytest.raises(StopIteration):
                    train_single(config, model=MagicMock(), sdf_dataset=MagicMock())
                mesh_warnings = [x for x in w if "mesh_names" in str(x.message)]
                assert len(mesh_warnings) == 1


class TestMultiHeadMeshNamesValidation:
    """Tests for mesh_names validation in train_deep_sdf_multi_head."""

    def test_mesh_names_length_mismatch_raises(self):
        models = (MagicMock(), MagicMock())  # 2 decoders
        config = _make_multi_config(
            objects_per_decoder=2,
            mesh_names=["bone", "cart"],  # 2 names but 2 decoders * 2 opd = 4 needed
        )
        with pytest.raises(ValueError, match="mesh_names has 2 entries"):
            train_multi_head(config, models=models, sdf_dataset=MagicMock())

    def test_mesh_names_matching_length_no_error(self):
        models = (MagicMock(), MagicMock())  # 2 decoders
        config = _make_multi_config(
            objects_per_decoder=2,
            mesh_names=["bone", "cart", "med_men", "lat_men"],  # 2 * 2 = 4
        )
        with patch(PATCH_MULTI, side_effect=StopIteration):
            with pytest.raises(StopIteration):
                train_multi_head(config, models=models, sdf_dataset=MagicMock())

    def test_warning_when_no_mesh_names(self):
        models = (MagicMock(), MagicMock())
        config = _make_multi_config(mesh_names=None)
        with patch(PATCH_MULTI, side_effect=StopIteration):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with pytest.raises(StopIteration):
                    train_multi_head(config, models=models, sdf_dataset=MagicMock())
                mesh_warnings = [x for x in w if "mesh_names" in str(x.message)]
                assert len(mesh_warnings) == 1


class TestDatasetCarriedMeshNames:
    """
    #52: ``mesh_names`` is a free-floating config key persisted to
    ``model_params_config.json`` as ground truth, while the per-surface ordering it
    claims to describe is defined somewhere else entirely — the order of each subject's
    mesh-path list in the dataset. Nothing ties the two together, so the persisted names
    can be silently wrong. The fix moves the declaration next to the ordering:
    ``MultiSurfaceSDFSamples`` accepts ``mesh_names``, and the trainer adopts or
    cross-checks it at entry. Strict xfails until that lands.
    """

    @pytest.mark.xfail(
        strict=True, reason="#52: MultiSurfaceSDFSamples does not yet accept mesh_names"
    )
    def test_dataset_validates_names_against_its_own_surface_count(self):
        """One name for a two-surface subject list must be refused at construction."""
        with pytest.raises(ValueError, match="mesh_names"):
            MultiSurfaceSDFSamples(
                list_mesh_paths=[["a.vtk", "b.vtk"]], subsample=4, mesh_names=["bone"]
            )

    @pytest.mark.xfail(
        strict=True, reason="#52: the trainer does not yet adopt dataset-carried mesh_names"
    )
    def test_trainer_adopts_the_datasets_names_when_config_has_none(self):
        config = _make_single_config(objects_per_decoder=2, mesh_names=None)
        dataset = MagicMock()
        dataset.mesh_names = ["bone", "cart"]
        with patch(PATCH_SINGLE, side_effect=StopIteration):
            with pytest.raises(StopIteration):
                train_single(config, model=MagicMock(), sdf_dataset=dataset)
        assert config["mesh_names"] == ["bone", "cart"]

    @pytest.mark.xfail(
        strict=True, reason="#52: config-vs-dataset mesh_names disagreement is not detected"
    )
    def test_disagreeing_declarations_raise_at_entry(self):
        config = _make_single_config(objects_per_decoder=2, mesh_names=["bone", "cart"])
        dataset = MagicMock()
        dataset.mesh_names = ["cart", "bone"]
        with pytest.raises(ValueError, match="mesh_names"):
            train_single(config, model=MagicMock(), sdf_dataset=dataset)
