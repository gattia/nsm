"""
Every name that lived in ``sdf_dataset.py`` before the §8.0 slice-A split stays
importable from both of its historical paths.

``NSM.datasets`` and ``NSM.datasets.sdf_dataset`` are both live import paths —
``reconstruct/main.py`` uses both in adjacent lines, and downstream forks are assumed
to as well — so the re-import block in ``sdf_dataset.py`` is public API, not
scaffolding of the move. The list below is frozen deliberately: deleting a name from
the re-import block must turn this red, not silently narrow the API.
"""

import importlib

import pytest

MOVED_OR_KEPT = [
    "MultiSurfaceSDFSamples",
    "SDFSamples",
    "check_probabilities",
    "check_probabilities_sum",
    "combine_meshes",
    "derive_seed",
    "get_buffered_cube_mins_maxs",
    "get_cube_mins_maxs",
    "get_pts_center_and_scale",
    "get_rand_uniform_pts",
    "is_zipfile",
    "mesh_content_key",
    "meshfix",
    "read_mesh_get_sampled_pts",
    "read_meshes_get_sampled_pts",
    "today_date",
    "unpack_numpy_data",
    "unpack_pts",
]


@pytest.mark.parametrize("module_path", ["NSM.datasets", "NSM.datasets.sdf_dataset"])
@pytest.mark.parametrize("name", MOVED_OR_KEPT)
def test_every_historical_name_is_importable(module_path, name):
    module = importlib.import_module(module_path)
    assert hasattr(module, name), f"{module_path} lost {name}"


def test_the_definitions_live_in_the_new_modules():
    """The re-imports point at the moved definitions, not at copies."""
    from NSM.datasets import mesh_sampling, sdf_dataset, utils

    assert sdf_dataset.read_mesh_get_sampled_pts is mesh_sampling.read_mesh_get_sampled_pts
    assert sdf_dataset.combine_meshes is utils.combine_meshes
    assert sdf_dataset.unpack_numpy_data is utils.unpack_numpy_data
