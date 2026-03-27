"""
Tests for mesh_names config validation in train_deep_sdf and train_deep_sdf_multi_head.
"""

import warnings
from unittest.mock import patch, MagicMock

import pytest

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
