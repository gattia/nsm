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
    """``unpack_numpy_data`` accepts every key spelling the cache has used."""

    def test_every_spelling_unpacks_and_the_precedence_is_fixed(self, tmp_path):
        """
        Fails if ``unpack_numpy_data`` stops reading ``xyz``, ``gt_sdf`` or ``sdf``, or
        changes the precedence ``pts`` over ``xyz`` and ``sdfs`` over ``gt_sdf`` over ``sdf``.
        """
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
        """Fails if ``unpack_numpy_data`` accepts data missing its coordinate or SDF key."""
        np.savez(tmp_path / "no_coords.npz", sdfs=np.zeros(4))
        with pytest.raises(ValueError, match="No pts or xyz"):
            unpack_numpy_data(np.load(tmp_path / "no_coords.npz"))
        np.savez(tmp_path / "no_sdf.npz", pts=np.zeros((4, 3)))
        with pytest.raises(ValueError, match="No sdfs or gt_sdf or sdf"):
            unpack_numpy_data(np.load(tmp_path / "no_sdf.npz"))

    def test_the_output_shape(self, tmp_path):
        """
        Fails if ``unpack_numpy_data`` returns non-float32 tensors, anything but ``[]`` for an
        absent key group, or ``point_cloud`` unasked.

        A plain dict works only with ``list_additional_keys=[]``, because unpacking a key
        group reads ``data.files``. The ``AttributeError`` check pins that limit. No in-repo
        caller passes a dict.
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
    """
    Fails if ``unpack_pts`` returns a group's ``{name}_0..N`` arrays out of index order or
    not as tensors, or anything but ``[]`` for an absent group.
    """
    data = _npz(tmp_path, new_pts_0=np.zeros((2, 3)), new_pts_1=np.ones((3, 3)))
    pts = unpack_pts(data, pts_name="new_pts")
    assert [p.shape for p in pts] == [(2, 3), (3, 3)]
    assert all(isinstance(p, torch.Tensor) for p in pts)
    assert unpack_pts(_npz(tmp_path), pts_name="new_pts") == []


def test_the_validators_refuse_by_name():
    """
    Fails if ``check_probabilities`` or ``check_probabilities_sum`` accepts a value outside
    [0, 1] or a near + far sum above 1, or ``get_cube_mins_maxs`` / ``get_pts_center_and_scale``
    accepts an empty or non-(n, 3) array or an unknown ``scale_method``.
    """
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
    Fails if ``combine_meshes`` returns a pyvista ``PolyData`` or drops a mesh for two or more
    indices, or anything but the mesh itself for one index (#61).

    pymskt's ``+`` gives a ``PolyData``, and ``load_reference_mesh`` calls ``save_mesh`` on
    the result, which ``PolyData`` lacks.
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
