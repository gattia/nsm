"""
The leaf helpers in ``NSM/datasets/utils.py``: cache unpacking, validators, and
``combine_meshes``. The cube and centring arithmetic is covered in
``regression/test_dataset_cache.py``.
"""

import numpy as np
import pytest
import torch
from pymskt.mesh import Mesh

from NSM.datasets.sdf_dataset import (
    check_probabilities,
    check_probabilities_sum,
    combine_meshes,
    get_cube_mins_maxs,
    get_pts_center_and_scale,
    is_zipfile,
    unpack_numpy_data,
    unpack_pts,
)

SDF = [-1.0, -0.5, 0.5, 1.0]


def _npz(directory, coord_key="pts", sdf_key="sdfs", **extra):
    """A minimal cache-shaped ``.npz``, loaded the way the datasets load it."""
    arrays = {coord_key: np.arange(12, dtype=np.float64).reshape(4, 3), sdf_key: np.array(SDF)}
    arrays.update(extra)
    path = directory / "sample.npz"
    np.savez(path, **arrays)
    return np.load(path)


class TestUnpackNumpyData:
    """The cache's key spellings changed over time, and unpacking accepts all of them."""

    def test_every_spelling_unpacks_and_the_precedence_is_fixed(self, tmp_path):
        """``pts`` over ``xyz``; ``sdfs`` over ``gt_sdf`` over ``sdf``."""
        via_pts = unpack_numpy_data(_npz(tmp_path, coord_key="pts"))
        assert torch.equal(via_pts["xyz"], unpack_numpy_data(_npz(tmp_path, "xyz"))["xyz"])
        for sdf_key in ("sdfs", "gt_sdf", "sdf"):
            assert unpack_numpy_data(_npz(tmp_path, sdf_key=sdf_key))["gt_sdf"].tolist() == SDF

        both = _npz(tmp_path, xyz=np.ones((4, 3)), gt_sdf=np.full(4, 2.0), sdf=np.full(4, 3.0))
        unpacked = unpack_numpy_data(both)
        assert torch.equal(unpacked["xyz"], torch.from_numpy(both["pts"]).float())
        assert unpacked["gt_sdf"].tolist() == SDF
        gt_sdf_over_sdf = _npz(tmp_path, sdf_key="gt_sdf", sdf=np.ones(4))
        assert unpack_numpy_data(gt_sdf_over_sdf)["gt_sdf"].tolist() == SDF

    def test_a_missing_group_raises_by_name(self, tmp_path):
        np.savez(tmp_path / "no_coords.npz", sdfs=np.zeros(4))
        with pytest.raises(ValueError, match="No pts or xyz"):
            unpack_numpy_data(np.load(tmp_path / "no_coords.npz"))
        np.savez(tmp_path / "no_sdf.npz", pts=np.zeros((4, 3)))
        with pytest.raises(ValueError, match="No sdfs or gt_sdf or sdf"):
            unpack_numpy_data(np.load(tmp_path / "no_sdf.npz"))

    def test_the_output_shape(self, tmp_path):
        """
        float32 whatever came in; absent index groups as empty lists; ``point_cloud`` only
        on request. A plain dict works only with no additional key groups, because the
        default reads ``data.files``: no in-repo caller passes a dict.
        """
        data = _npz(tmp_path, point_cloud=np.ones((4, 3)))
        unpacked = unpack_numpy_data(data)
        assert unpacked["xyz"].dtype == unpacked["gt_sdf"].dtype == torch.float32
        assert all(unpacked[k] == [] for k in ("orig_pts", "new_pts", "pos_idx", "neg_idx"))
        assert "point_cloud" not in unpacked
        assert unpack_numpy_data(data, point_cloud=True)["point_cloud"].dtype == torch.float32

        plain = {"pts": np.zeros((4, 3)), "sdfs": np.zeros(4)}
        assert unpack_numpy_data(plain, list_additional_keys=[])["xyz"].shape == (4, 3)
        with pytest.raises(AttributeError):
            unpack_numpy_data(plain)


def test_unpack_pts_rebuilds_indexed_keys_in_order(tmp_path):
    data = _npz(tmp_path, new_pts_0=np.zeros((2, 3)), new_pts_1=np.ones((3, 3)))
    pts = unpack_pts(data, pts_name="new_pts")
    assert [p.shape for p in pts] == [(2, 3), (3, 3)]
    assert all(isinstance(p, torch.Tensor) for p in pts)
    assert unpack_pts(_npz(tmp_path), pts_name="new_pts") == []


def test_is_zipfile_answers_false_for_anything_unreadable(tmp_path):
    """``zipfile.is_zipfile`` raises on a missing path; the wrapper answers False."""
    (tmp_path / "plain.txt").write_text("not a zip", encoding="utf-8")
    np.savez(tmp_path / "cache.npz", pts=np.zeros((2, 3)))
    assert is_zipfile(str(tmp_path / "never_written.npz")) is False
    assert is_zipfile(str(tmp_path / "plain.txt")) is False
    assert is_zipfile(str(tmp_path / "cache.npz")) is True


def test_the_validators_refuse_by_name():
    for ok in (0.0, 0.5, 1.0):
        check_probabilities(ok)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="between 0 and 1"):
            check_probabilities(bad)
    check_probabilities_sum(0.6, 0.4)
    with pytest.raises(ValueError, match="must be <=1"):
        check_probabilities_sum(0.6, 0.5)

    with pytest.raises(ValueError, match="empty"):
        get_cube_mins_maxs(np.zeros((0, 3)))
    with pytest.raises(ValueError, match=r"\(n_pts, 3\)"):
        get_cube_mins_maxs(np.zeros((4, 2)))
    with pytest.raises(NotImplementedError, match="not_a_method"):
        get_pts_center_and_scale(np.zeros((4, 3)), scale_method="not_a_method")


def test_combine_meshes_returns_one_mesh_or_their_union():
    """
    One index, bare or in a list, returns that mesh. Several return a pymskt ``Mesh``:
    pymskt's ``+`` gives a pyvista ``PolyData``, and ``load_reference_mesh`` calls
    ``save_mesh`` on the result (#61).
    """
    meshes = [
        Mesh(np.array([[x, 0, 0], [x + 1, 0, 0], [x, 1, 0]]), np.array([[0, 1, 2]]))
        for x in (0, 2, 4)
    ]
    assert combine_meshes(meshes, 1) is meshes[1]
    assert combine_meshes(meshes, [0]) is meshes[0]
    for indices in ([0, 1], [0, 1, 2]):
        combined = combine_meshes(meshes, indices)
        assert isinstance(combined, Mesh)
        assert combined.point_coords.shape[0] == 3 * len(indices)
