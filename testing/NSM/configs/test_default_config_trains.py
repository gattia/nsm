"""
The shipped ``default_config.json`` drives ``train_deep_sdf`` (#48). Overrides may change a
shipped value but never add a key, so a key missing from the file still raises here.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "regression"))
from _harness import (  # noqa: E402
    ARCHITECTURE,
    SUBSAMPLE,
    build_dataset,
    build_model,
    run_training,
    write_synthetic_meshes,
)

from NSM.configs.generate_sdf_default_config import DEFAULT_CONFIG_PATH  # noqa: E402


def test_the_shipped_default_config_drives_train_deep_sdf(tmp_path):
    """
    Fails if ``train_deep_sdf`` reads a key the shipped ``default_config.json`` lacks, or two
    CPU epochs from it give a non-finite loss or write no ``.pth`` checkpoint (#48).
    """
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)

    overrides = {
        **ARCHITECTURE,
        "device": "cpu",
        "n_epochs": 2,
        "checkpoint_epochs": 2,
        "additional_checkpoints": [],
        "save_frequency": 2,
        "objects_per_batch": 2,
        "samples_per_object_per_batch": SUBSAMPLE,
        "num_data_loader_threads": 0,
        "prefetch_factor": None,
        "experiment_directory": str(tmp_path / "experiment"),
    }
    unknown = set(overrides) - set(config)
    assert not unknown, f"overrides would mask missing shipped keys: {sorted(unknown)}"
    config.update(overrides)

    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    mesh_paths = write_synthetic_meshes(mesh_dir)
    dataset = build_dataset(mesh_paths, tmp_path / "cache")
    model = build_model(config)

    records, _ = run_training(config, model, dataset)

    assert len(records) == config["n_epochs"]
    assert all(torch.isfinite(torch.tensor(record["loss"])) for record in records)
    checkpoints = list((tmp_path / "experiment").rglob("*.pth"))
    assert checkpoints, "the checkpoint epoch wrote nothing under experiment_directory"
