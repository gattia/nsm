"""
Characterization of ``prepare_results_for_wandb`` / ``_process_meshes_for_wandb``,
written immediately before their move to ``wandb_logging.py`` (plan §8.0.C).

``wandb.Object3D`` construction is local — no run, no network — so the filtering
contract is assertable offline.
"""

import numpy as np
import pytest
import torch
import wandb

from NSM.reconstruct.main import _process_meshes_for_wandb, prepare_results_for_wandb


class FakeMesh:
    def __init__(self, n_points=10):
        self.point_coords = np.random.default_rng(0).normal(size=(n_points, 3))
        self.faces = None


class TestSerializationFiltering:
    @pytest.fixture(scope="class")
    def prepared(self):
        original = {
            "kept_int": 1,
            "kept_none": None,
            "kept_tuple": (1, 2),
            "np_scalar": np.float64(2.5),
            "small_array": np.arange(5),
            "large_array": np.zeros(100),
            "small_tensor": torch.ones(3),
            "large_tensor": torch.zeros(50),
            "unserializable": object(),
        }
        return original, prepare_results_for_wandb(original)

    def test_basic_types_and_none_are_kept(self, prepared):
        _, result = prepared
        assert result["kept_int"] == 1
        assert result["kept_none"] is None
        assert result["kept_tuple"] == (1, 2)

    def test_numpy_scalars_become_floats(self, prepared):
        _, result = prepared
        assert isinstance(result["np_scalar"], float)

    def test_arrays_and_tensors_up_to_ten_elements_become_lists(self, prepared):
        _, result = prepared
        assert result["small_array"] == [0, 1, 2, 3, 4]
        assert result["small_tensor"] == [1.0, 1.0, 1.0]

    def test_larger_arrays_tensors_and_unserializable_objects_are_dropped(self, prepared):
        _, result = prepared
        assert "large_array" not in result
        assert "large_tensor" not in result
        assert "unserializable" not in result

    def test_the_original_dict_is_not_mutated(self, prepared):
        original, _ = prepared
        assert set(original) == {
            "kept_int",
            "kept_none",
            "kept_tuple",
            "np_scalar",
            "small_array",
            "large_array",
            "small_tensor",
            "large_tensor",
            "unserializable",
        }
        assert isinstance(original["large_array"], np.ndarray)


class TestMeshProcessing:
    def test_meshes_become_object3d_entries_and_the_mesh_keys_are_deleted(self):
        result = prepare_results_for_wandb({"mesh": [FakeMesh()], "orig_mesh": [FakeMesh()]})
        assert isinstance(result["recon_mesh_0"], wandb.Object3D)
        assert isinstance(result["orig_mesh_0"], wandb.Object3D)
        assert "mesh" not in result and "orig_mesh" not in result

    def test_none_meshes_produce_no_entries(self):
        mesh_data = _process_meshes_for_wandb([None], "recon_mesh", 10_000, True, False)
        assert mesh_data == {}

    def test_n_points_reports_the_full_count_even_when_subsampled(self):
        """The 3D object is subsampled to ``max_points_3d``; the count key is not."""
        mesh_data = _process_meshes_for_wandb([FakeMesh(10)], "recon_mesh", 4, True, False)
        assert mesh_data["recon_mesh_0_n_points"] == 10
