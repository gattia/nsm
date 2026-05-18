"""Phase 0 step 3 -- run the experiment matrix and write the report.

Loads the cached latents and reconstructed meshes, then for every
(config x NFE x ordered-pair x surface) cell:

  * warps the source surface onto the target via ``interpolate_points``,
  * (optionally) warps back for the round-trip / bijectivity signal,
  * scores the result with ``NSM.mesh.correspondence_metrics``.

Results are written long-format to ``report/results.csv`` (one row per cell)
and ``report/results.json``; an aggregated markdown summary goes to
``report/report.md``.

The matrix is large (configs x NFE x pairs x 4 surfaces x 2 warps). Use the
CLI flags to subset it -- start small, then scale up on a GPU node.

Usage:
    python -m experiments.mesh_interpolation.run_matrix [options]

Options:
    --configs a,b,c     subset of EXPERIMENT_CONFIGS keys (default: all)
    --nfe 50,100        subset of the NFE grid (default: all)
    --surfaces 0,2,3    surface indices to score (default: 0,1,2,3)
    --max-pairs N       cap the number of ordered pairs (default: all)
    --no-roundtrip      skip the backward warp (no bijectivity metrics)
    --no-self-intersect skip self-intersection counting (slow on big meshes)
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import pyvista as pv

from NSM.mesh.correspondence_metrics import score_correspondence
from NSM.mesh.interpolate import interpolate_points

from .config import (
    EXPERIMENT_CONFIGS,
    MANIFEST_PATH,
    MESH_NAMES,
    NFE_GRID,
    REPORT_DIR,
    evaluate_sdf,
    load_nsm_model,
)
from .fit_cache import _latent_path, _mesh_path, is_cached


def load_cache(keys):
    """Load cached latents and reconstructed meshes for the given knee keys.

    Returns:
        (latents, meshes): ``latents[key]`` is a (D,) array; ``meshes[key][s]``
        is the pyvista PolyData for surface index ``s``.
    """
    latents, meshes = {}, {}
    for key in keys:
        if not is_cached(key):
            raise RuntimeError(f"{key} not cached -- run fit_cache.py first.")
        latents[key] = np.load(_latent_path(key))
        meshes[key] = [pv.read(_mesh_path(key, s)) for s in MESH_NAMES]
    return latents, meshes


def _flatten_score(score):
    """Flatten a score_correspondence result into a flat dict of scalars."""
    flat = {}
    assd = score.get("assd")
    flat["assd"] = assd if isinstance(assd, (int, float)) else np.nan

    for direction in ("warped_to_target", "target_to_warped"):
        d = score.get(f"directed_distance_{direction}", {})
        if isinstance(d, dict):
            for k in ("p50", "mean", "p95", "max"):
                flat[f"dist_{direction}_{k}"] = d.get(k, np.nan)

    ose = score.get("off_surface_error", {})
    if isinstance(ose, dict):
        flat["off_surface_mean"] = ose.get("mean", np.nan)
        flat["off_surface_p95"] = ose.get("p95", np.nan)
        flat["off_surface_rms"] = ose.get("rms", np.nan)

    th = score.get("triangle_health", {})
    if isinstance(th, dict):
        flat["edge_ratio_p95"] = th.get("edge_ratio_p95", np.nan)
        flat["degenerate_count"] = th.get("degenerate_count", np.nan)
        flat["edge_length_min"] = th.get("edge_length_min", np.nan)

    si = score.get("self_intersection_count")
    flat["self_intersections"] = si if isinstance(si, (int, float)) else np.nan

    fo = score.get("foldover_count", {})
    if isinstance(fo, dict):
        flat["foldover_fraction"] = fo.get("flipped_fraction", np.nan)
        flat["foldover_count"] = fo.get("flipped_count", np.nan)

    rt = score.get("roundtrip_distance", {})
    if isinstance(rt, dict):
        flat["roundtrip_mean"] = rt.get("mean", np.nan)
        flat["roundtrip_p95"] = rt.get("p95", np.nan)
    return flat


def run_cell(model, latents, meshes, key_a, key_b, surface_idx, cfg_kwargs,
             nfe, roundtrip, self_intersect):
    """Warp + score a single matrix cell. Returns a flat result dict."""
    source = meshes[key_a][surface_idx]
    target = meshes[key_b][surface_idx]
    z_a, z_b = latents[key_a], latents[key_b]
    faces = source.regular_faces.astype(np.int64)
    src_pts = np.asarray(source.points, dtype=np.float64)

    warped = interpolate_points(
        model, z_a, z_b, n_steps=nfe, points1=src_pts,
        surface_idx=surface_idx, faces=faces, **cfg_kwargs,
    )
    warped_mesh = pv.PolyData(np.asarray(warped), source.faces)
    sdf_vals = evaluate_sdf(model, warped, z_b, surface_idx)

    roundtrip_pts = None
    if roundtrip:
        roundtrip_pts = interpolate_points(
            model, z_b, z_a, n_steps=nfe, points1=np.asarray(warped),
            surface_idx=surface_idx, faces=faces, **cfg_kwargs,
        )

    score = score_correspondence(
        warped_mesh, target, source_mesh=source, sdf_values=sdf_vals,
        roundtrip_points=roundtrip_pts, compute_self_intersection=self_intersect,
    )
    return _flatten_score(score)


def _df_to_md(df):
    """Render a DataFrame as a markdown table (no tabulate dependency)."""
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(out)


def write_report(df, path):
    """Aggregate the long-format results into a markdown summary."""
    lines = ["# Mesh-interpolation Phase 0 report", ""]
    lines.append(f"Cells scored: {len(df)}")
    lines.append("")

    metrics = ["assd", "off_surface_p95", "foldover_fraction", "roundtrip_mean"]
    ref_nfe = 100 if 100 in df["nfe"].unique() else sorted(df["nfe"].unique())[-1]
    lines.append(f"## Per-config means at NFE={ref_nfe} (lower is better)")
    lines.append("")
    sub = df[df["nfe"] == ref_nfe]
    for surf in sorted(sub["surface"].unique()):
        lines.append(f"### Surface: {MESH_NAMES[surf]} (idx {surf})")
        lines.append("")
        tbl = sub[sub["surface"] == surf].groupby("config")[metrics].mean()
        lines.append(_df_to_md(tbl.round(5)))
        lines.append("")

    lines.append("## NFE sensitivity (ASSD mean, all surfaces)")
    lines.append("")
    pivot = df.pivot_table(index="config", columns="nfe", values="assd", aggfunc="mean")
    lines.append(_df_to_md(pivot.round(5)))
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def merge_results():
    """Concatenate every results_*.csv shard into results.csv + report.md."""
    import glob

    shards = sorted(glob.glob(os.path.join(REPORT_DIR, "results_*.csv")))
    if not shards:
        raise SystemExit(f"no results_*.csv shards found in {REPORT_DIR}")
    df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
    df.to_csv(os.path.join(REPORT_DIR, "results.csv"), index=False)
    write_report(df, os.path.join(REPORT_DIR, "report.md"))
    print(f"Merged {len(shards)} shards ({len(df)} cells) -> "
          f"{REPORT_DIR}/results.csv + report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=None)
    parser.add_argument("--nfe", default=None)
    parser.add_argument("--surfaces", default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--max-knees", type=int, default=None,
        help="use only the first N manifest knees; ordered pairs are then "
        "N*(N-1). The manifest is KL-ordered so a small N still spans grades.",
    )
    parser.add_argument("--no-roundtrip", action="store_true")
    parser.add_argument("--no-self-intersect", action="store_true")
    parser.add_argument(
        "--out-tag", default=None,
        help="suffix for the per-shard output file (results_<tag>.csv). Lets "
        "parallel SLURM jobs write without clobbering; merge with --merge.",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="do not run the matrix; concatenate every report/results_*.csv "
        "shard into results.csv and write report.md.",
    )
    args = parser.parse_args()

    if args.merge:
        merge_results()
        return

    configs = list(EXPERIMENT_CONFIGS)
    if args.configs:
        configs = [c.strip() for c in args.configs.split(",")]
    nfe_grid = NFE_GRID
    if args.nfe:
        nfe_grid = [int(n) for n in args.nfe.split(",")]
    surfaces = list(range(len(MESH_NAMES)))
    if args.surfaces:
        surfaces = [int(s) for s in args.surfaces.split(",")]

    # KL-interleave the manifest knees so a --max-knees prefix still spans
    # grades (the manifest is grouped KL0, KL1, KL2).
    with open(MANIFEST_PATH) as f:
        records = json.load(f)
    by_kl = {}
    for r in records:
        by_kl.setdefault(r["kl"], []).append(r["key"])
    keys = []
    while any(by_kl.values()):
        for kl in sorted(by_kl):
            if by_kl[kl]:
                keys.append(by_kl[kl].pop(0))
    if args.max_knees is not None:
        keys = keys[: args.max_knees]
    latents, meshes = load_cache(keys)

    # All ordered pairs A != B (both warp directions -- plan section 2.2).
    pairs = [(a, b) for a, b in itertools.permutations(keys, 2)]
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]

    model = load_nsm_model()
    os.makedirs(REPORT_DIR, exist_ok=True)

    suffix = f"_{args.out_tag}" if args.out_tag else ""
    shard_path = os.path.join(REPORT_DIR, f"results{suffix}.csv")

    # Resume support: reload an existing shard and skip cells already scored.
    # Each matrix job checkpoints after every pair, so a SLURM timeout loses at
    # most one pair's worth of work and a resubmit picks up where it left off.
    rows = []
    done_cells = set()
    if os.path.isfile(shard_path):
        prev = pd.read_csv(shard_path)
        rows = prev.to_dict("records")
        done_cells = {
            (r["config"], int(r["nfe"]), r["pair"], int(r["surface"]))
            for r in rows
        }
        print(f"Resuming -- {len(done_cells)} cells already in {shard_path}")

    total = len(configs) * len(nfe_grid) * len(pairs) * len(surfaces)
    print(f"Scoring {total} cells "
          f"({len(configs)} configs x {len(nfe_grid)} NFE x {len(pairs)} pairs "
          f"x {len(surfaces)} surfaces)")

    n_new = 0
    t0 = time.time()
    for cfg_name in configs:
        cfg_kwargs = EXPERIMENT_CONFIGS[cfg_name]
        for nfe in nfe_grid:
            for (key_a, key_b) in pairs:
                pair_str = f"{key_a}->{key_b}"
                pair_had_new = False
                for surf in surfaces:
                    if (cfg_name, nfe, pair_str, surf) in done_cells:
                        continue
                    try:
                        flat = run_cell(
                            model, latents, meshes, key_a, key_b, surf,
                            cfg_kwargs, nfe, not args.no_roundtrip,
                            not args.no_self_intersect,
                        )
                        flat["error"] = ""
                    except Exception as exc:  # keep the sweep alive
                        flat = {"error": f"{type(exc).__name__}: {exc}"}
                    flat.update(
                        config=cfg_name, nfe=nfe, pair=pair_str,
                        source=key_a, target=key_b, surface=surf,
                        surface_name=MESH_NAMES[surf],
                    )
                    rows.append(flat)
                    n_new += 1
                    pair_had_new = True
                if pair_had_new:  # checkpoint after every pair
                    pd.DataFrame(rows).to_csv(shard_path, index=False)
            elapsed = time.time() - t0
            print(f"  {cfg_name} NFE={nfe}: +{n_new} new cells "
                  f"({elapsed:.0f}s elapsed)")

    df = pd.DataFrame(rows)
    df.to_csv(shard_path, index=False)
    with open(os.path.join(REPORT_DIR, f"results{suffix}.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    if args.out_tag:
        # A shard: leave the aggregated report to the --merge step.
        print(f"\nWrote shard results{suffix}.csv to {REPORT_DIR}")
    else:
        write_report(df, os.path.join(REPORT_DIR, "report.md"))
        print(f"\nWrote results + report to {REPORT_DIR}")


if __name__ == "__main__":
    main()
