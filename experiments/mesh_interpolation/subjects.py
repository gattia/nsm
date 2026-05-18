"""Phase 0 step 1 -- select the pilot knees and write a manifest.

Reads the OAI baseline demographics CSV, picks a KL-stratified set of knees
(KL3/4 excluded -- plan section 2.1), verifies that all four femur-model mesh
files exist for each candidate, and writes ``cache/manifest.json``.

This step needs no GPU and no NSM model -- run it first.

Usage:
    python -m experiments.mesh_interpolation.subjects
"""

import json
import os

import numpy as np
import pandas as pd

from .config import (
    DEMOGRAPHICS_CSV,
    KL_QUOTA,
    MANIFEST_PATH,
    MESH_ROOT,
    MESH_SUFFIXES,
    SELECTION_SEED,
)

OSTEOPHYTE_COLS = [
    "osteophytes_fem_ant_lat_score",
    "osteophytes_fem_ant_med_score",
    "osteophytes_fem_cent_lat_score",
    "osteophytes_fem_cent_med_score",
    "osteophytes_fem_post_lat_score",
    "osteophytes_fem_post_med_score",
]


def mesh_paths_for(knee_id, side):
    """Return the {surface: path} dict of the four femur-model meshes."""
    folder = os.path.join(MESH_ROOT, str(knee_id))
    return {
        name: os.path.join(folder, f"{knee_id}_{side}_{suffix}.vtk")
        for name, suffix in MESH_SUFFIXES.items()
    }


def all_meshes_exist(knee_id, side):
    """True iff every femur-model mesh file is present for this knee."""
    return all(os.path.isfile(p) for p in mesh_paths_for(knee_id, side).values())


def select_subjects(verbose=True):
    """Select the KL-stratified pilot knees and return manifest records.

    Returns:
        list[dict]: one record per selected knee, with id, side, kl, the four
        mesh paths, and the per-region osteophyte scores.
    """
    df = pd.read_csv(DEMOGRAPHICS_CSV)
    df = df.dropna(subset=["id", "side", "kl"]).copy()
    df["id"] = df["id"].astype(int)
    df["kl"] = df["kl"].astype(int)

    rng = np.random.default_rng(SELECTION_SEED)
    records = []

    for kl_grade, quota in sorted(KL_QUOTA.items()):
        candidates = df[df["kl"] == kl_grade]
        # Keep only knees whose four mesh files are all on disk.
        valid = [
            row
            for _, row in candidates.iterrows()
            if all_meshes_exist(row["id"], row["side"])
        ]
        if len(valid) < quota:
            raise RuntimeError(
                f"KL{kl_grade}: only {len(valid)} knees with all meshes present, "
                f"need {quota}."
            )
        idx = rng.choice(len(valid), size=quota, replace=False)
        for i in sorted(idx):
            row = valid[i]
            knee_id, side = int(row["id"]), str(row["side"])
            records.append(
                {
                    "id": knee_id,
                    "side": side,
                    "kl": kl_grade,
                    "key": f"{knee_id}_{side}",
                    "mesh_paths": mesh_paths_for(knee_id, side),
                    "osteophyte_scores": {
                        c: (None if pd.isna(row.get(c)) else float(row[c]))
                        for c in OSTEOPHYTE_COLS
                        if c in row
                    },
                }
            )
        if verbose:
            print(f"KL{kl_grade}: selected {quota} of {len(valid)} eligible knees")

    return records


def main():
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    records = select_subjects()
    with open(MANIFEST_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote {len(records)} knees to {MANIFEST_PATH}")
    for r in records:
        print(f"  {r['key']}  KL{r['kl']}")


if __name__ == "__main__":
    main()
