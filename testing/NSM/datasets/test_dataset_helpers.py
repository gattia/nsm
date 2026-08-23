"""
Characterization tests for ``sdf_dataset.py``'s leaf helpers, written immediately
before their move to ``NSM/datasets/utils.py`` (plan §8.0, slice A).

Everything here pins behaviour as it stands today. The cube helpers' and
``get_pts_center_and_scale``'s *math* is already pinned by the regression suite
(``testing/NSM/regression/test_dataset_cache.py``); this file covers the branches
nothing executed before the move: the key-spelling fallbacks, the named raises, and
the unreadable-path fallbacks.
"""

import numpy as np
import pytest
import torch

from NSM.datasets.sdf_dataset import (
    check_probabilities,
    check_probabilities_sum,
    get_cube_mins_maxs,
    get_pts_center_and_scale,
    is_zipfile,
    unpack_numpy_data,
    unpack_pts,
)


def _npz(directory, coord_key="pts", sdf_key="sdfs", **extra):
    """A minimal cache-shaped ``.npz``, loaded the way the datasets load it."""
    arrays = {
        coord_key: np.arange(12, dtype=np.float64).reshape(4, 3),
        sdf_key: np.array([-1.0, -0.5, 0.5, 1.0]),
    }
    arrays.update(extra)
    path = directory / "sample.npz"
    np.savez(path, **arrays)
    return np.load(path)


class TestUnpackNumpyData:
    """The cache's key spellings have varied over time; unpacking accepts them all."""

    def test_pts_and_xyz_spellings_are_equivalent(self, tmp_path):
        via_pts = unpack_numpy_data(_npz(tmp_path, coord_key="pts"))
        via_xyz = unpack_numpy_data(_npz(tmp_path, coord_key="xyz"))
        assert torch.equal(via_pts["xyz"], via_xyz["xyz"])

    def test_pts_wins_when_both_coordinate_spellings_are_present(self, tmp_path):
        data = _npz(tmp_path, coord_key="pts", xyz=np.ones((4, 3)))
        unpacked = unpack_numpy_data(data)
        assert torch.equal(unpacked["xyz"], torch.from_numpy(data["pts"]).float())

    @pytest.mark.parametrize("sdf_key", ["sdfs", "gt_sdf", "sdf"])
    def test_each_sdf_spelling_is_accepted(self, tmp_path, sdf_key):
        unpacked = unpack_numpy_data(_npz(tmp_path, sdf_key=sdf_key))
        assert torch.equal(unpacked["gt_sdf"], torch.tensor([-1.0, -0.5, 0.5, 1.0]))

    def test_sdf_spelling_precedence_is_sdfs_then_gt_sdf_then_sdf(self, tmp_path):
        data = _npz(tmp_path, sdf_key="sdfs", gt_sdf=np.full(4, 2.0), sdf=np.full(4, 3.0))
        assert torch.equal(unpack_numpy_data(data)["gt_sdf"], torch.tensor([-1.0, -0.5, 0.5, 1.0]))
        data = _npz(tmp_path, sdf_key="gt_sdf", sdf=np.full(4, 3.0))
        assert torch.equal(unpack_numpy_data(data)["gt_sdf"], torch.tensor([-1.0, -0.5, 0.5, 1.0]))

    def test_missing_coordinates_raise_by_name(self, tmp_path):
        path = tmp_path / "no_coords.npz"
        np.savez(path, sdfs=np.zeros(4))
        with pytest.raises(ValueError, match="No pts or xyz"):
            unpack_numpy_data(np.load(path))

    def test_missing_sdfs_raise_by_name(self, tmp_path):
        path = tmp_path / "no_sdf.npz"
        np.savez(path, pts=np.zeros((4, 3)))
        with pytest.raises(ValueError, match="No sdfs or gt_sdf or sdf"):
            unpack_numpy_data(np.load(path))

    def test_absent_key_groups_come_back_as_empty_lists(self, tmp_path):
        unpacked = unpack_numpy_data(_npz(tmp_path))
        for key in ["orig_pts", "new_pts", "pos_idx", "neg_idx", "surf_idx"]:
            assert unpacked[key] == []

    def test_point_cloud_is_only_converted_on_request(self, tmp_path):
        data = _npz(tmp_path, point_cloud=np.ones((4, 3)))
        assert "point_cloud" not in unpack_numpy_data(data)
        unpacked = unpack_numpy_data(data, point_cloud=True)
        assert unpacked["point_cloud"].dtype == torch.float32

    def test_outputs_are_float32_regardless_of_input_dtype(self, tmp_path):
        unpacked = unpack_numpy_data(_npz(tmp_path))
        assert unpacked["xyz"].dtype == torch.float32
        assert unpacked["gt_sdf"].dtype == torch.float32

    def test_a_plain_dict_only_works_without_additional_key_groups(self):
        """
        The docstring said "NpzFile or dict" (PR #70), but any call with a non-empty
        ``list_additional_keys`` — the default — reads ``data.files``, which only an
        NpzFile has. No in-repo caller passes a dict; pinned so the constraint is
        explicit when the function moves.
        """
        data = {"pts": np.zeros((4, 3)), "sdfs": np.zeros(4)}
        unpacked = unpack_numpy_data(data, list_additional_keys=[])
        assert unpacked["xyz"].shape == (4, 3)
        with pytest.raises(AttributeError):
            unpack_numpy_data(data)


class TestUnpackPts:
    def test_indexed_keys_rebuild_in_order(self, tmp_path):
        data = _npz(tmp_path, new_pts_0=np.zeros((2, 3)), new_pts_1=np.ones((3, 3)))
        pts = unpack_pts(data, pts_name="new_pts")
        assert len(pts) == 2
        assert pts[0].shape == (2, 3) and pts[1].shape == (3, 3)
        assert all(isinstance(p, torch.Tensor) for p in pts)

    def test_an_absent_group_is_an_empty_list(self, tmp_path):
        assert unpack_pts(_npz(tmp_path), pts_name="new_pts") == []


class TestIsZipfile:
    """``zipfile.is_zipfile`` raises on unreadable paths; the wrapper returns False."""

    def test_a_nonexistent_path_is_false(self, tmp_path):
        assert is_zipfile(str(tmp_path / "never_written.npz")) is False

    def test_a_non_zip_file_is_false(self, tmp_path):
        path = tmp_path / "plain.txt"
        path.write_text("not a zip", encoding="utf-8")
        assert is_zipfile(str(path)) is False

    def test_a_real_npz_is_true(self, tmp_path):
        path = tmp_path / "cache.npz"
        np.savez(path, pts=np.zeros((2, 3)))
        assert is_zipfile(str(path)) is True


class TestProbabilityValidators:
    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_out_of_range_probabilities_raise(self, bad):
        with pytest.raises(ValueError, match="between 0 and 1"):
            check_probabilities(bad)

    @pytest.mark.parametrize("ok", [0.0, 0.5, 1.0])
    def test_the_bounds_are_inclusive(self, ok):
        check_probabilities(ok)

    def test_shares_summing_past_one_raise(self):
        with pytest.raises(ValueError, match="must be <=1"):
            check_probabilities_sum(0.6, 0.5)

    def test_shares_summing_to_exactly_one_pass(self):
        check_probabilities_sum(0.6, 0.4)


class TestCubeBoundsValidation:
    """The cube math itself is pinned in the regression suite; these are the raises."""

    def test_an_empty_array_raises(self):
        with pytest.raises(ValueError, match="empty"):
            get_cube_mins_maxs(np.zeros((0, 3)))

    def test_a_wrong_shape_raises(self):
        with pytest.raises(ValueError, match=r"\(n_pts, 3\)"):
            get_cube_mins_maxs(np.zeros((4, 2)))


class TestGetPtsCenterAndScaleValidation:
    def test_an_unknown_scale_method_raises_naming_it(self):
        with pytest.raises(NotImplementedError, match="not_a_method"):
            get_pts_center_and_scale(np.zeros((4, 3)), scale_method="not_a_method")
