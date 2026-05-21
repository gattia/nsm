"""Test the iterative source-refinement wrapper (Fix 8) on the worst-fold-over
pairs and render a comparison panel.

For each thin-shell surface (cart, med_men, lat_men), picks the same
worst-fold-over pair used by `dump_visuals.py` and warps it under:

  * baseline (no fixes)
  * fix1_fix2_fix4c (current best from the Phase 0 sweep)
  * fix1_fix2_fix4c + iterative refinement, correspondence_mode='vertex'
  * fix1_fix2_fix4c + iterative refinement, correspondence_mode='smoothed'

Saves the ORIGINAL-N correspondence as a `.vtk` (so each method's reported
correspondence can be compared head-to-head), and renders a 3-row PNG
(fold-over | target_distance | warp_travel).
"""

import json
import os

import numpy as np
import pandas as pd
import pyvista as pv

from NSM.mesh.correspondence_metrics import _point_to_surface_distances
from NSM.mesh.interpolate import (
    compute_feature_mask,
    interpolate_points,
    interpolate_points_refined,
)

from .config import EXPERIMENT_CONFIGS, MANIFEST_PATH, MESH_NAMES, load_nsm_model
from .dump_visuals import foldover_per_triangle, render_panel
from .run_matrix import load_cache

OUT_DIR = os.path.join(os.path.dirname(__file__), "report", "visuals_refined")

COLUMN_LABELS = {
    "baseline": "baseline",
    "fix4c": "fix1_fix2_fix4c\n(current best)",
    "refined_vertex": "fix4c + refine\n(no pre-split)",
    "refined_preseam_vertex": "fix4c + refine\n+ pre-split seam (1 hop)",
}

# Fix 4c kwargs (the current best non-refined config).
FIX4C_KWARGS = dict(EXPERIMENT_CONFIGS["fix1_fix2_fix4c"])


def _make_position_rgb(points, percentile_clip=2):
    """Map (N, 3) positions to (N, 3) RGB in [0, 1] via per-axis percentile
    clipping. Percentile clip avoids outlier vertices flattening the gradient."""
    p = np.asarray(points, dtype=np.float64)
    lo = np.percentile(p, percentile_clip, axis=0)
    hi = np.percentile(p, 100 - percentile_clip, axis=0)
    rng = np.clip(hi - lo, 1e-12, None)
    rgb = np.clip((p - lo) / rng, 0.0, 1.0)
    return rgb.astype(np.float32)


def render_matched_rgb_panel(source, warped_meshes, surface_name, out_png):
    """Side-by-side: source + each warped mesh, all colour-coded by the same
    per-vertex RGB derived from the source-mesh position. Coherent colour
    bands across the warps = correspondence preserved; scrambled = bad."""
    pv.set_plot_theme("document")
    n_cols = len(warped_meshes) + 1
    plotter = pv.Plotter(
        off_screen=True, shape=(1, n_cols), window_size=(360 * n_cols, 500)
    )

    rgb = _make_position_rgb(source.points)
    src_col = source.copy()
    src_col.point_data["rgb"] = rgb
    plotter.subplot(0, 0)
    plotter.add_mesh(src_col, scalars="rgb", rgb=True, show_edges=False, lighting=True)
    plotter.add_text(
        "source\ncolour = source position",
        font_size=9, position="upper_left",
    )

    for col, (label, mesh) in enumerate(warped_meshes.items()):
        plotter.subplot(0, col + 1)
        # mesh has source's face connectivity and the warped positions; the
        # per-vertex RGB transfers by vertex index directly.
        m2 = mesh.copy()
        m2.point_data["rgb"] = rgb
        plotter.add_mesh(m2, scalars="rgb", rgb=True, show_edges=False, lighting=True)
        plotter.add_text(label, font_size=9, position="upper_left")

    plotter.link_views()
    if surface_name in ("cart", "med_men", "lat_men"):
        plotter.view_xy()
        plotter.camera.zoom(1.2)
    else:
        plotter.view_isometric()
    plotter.screenshot(out_png)
    plotter.close()


def warp_baseline(model, latents, source, target, sidx):
    """Plain baseline warp -- no fixes."""
    src_pts = np.asarray(source.points, dtype=np.float64)
    faces = source.regular_faces.astype(np.int64)
    w = interpolate_points(
        model, latents[0], latents[1], n_steps=100, points1=src_pts,
        surface_idx=sidx, faces=faces,
    )
    return np.asarray(w)


def warp_fix4c(model, latents, source, target, sidx):
    src_pts = np.asarray(source.points, dtype=np.float64)
    faces = source.regular_faces.astype(np.int64)
    w = interpolate_points(
        model, latents[0], latents[1], n_steps=100, points1=src_pts,
        surface_idx=sidx, faces=faces, **FIX4C_KWARGS,
    )
    return np.asarray(w)


def warp_refined(model, latents, source, target, sidx, mode, pre_split_hops=0):
    """Fix 4c with iterative source-mesh refinement.

    ``pre_split_hops``: 0 = no pre-split (reactive only); >0 = subdivide
    triangles around the dihedral seam first, then iterate.
    """
    corr, refined_src, refined_warped = interpolate_points_refined(
        model, latents[0], latents[1], source_mesh=source,
        surface_idx=sidx, n_steps=100,
        max_refine_passes=2, area_growth_threshold=2.0, refine_flipped=True,
        pre_split_seam_hops=pre_split_hops, pre_split_seam_angle_deg=60.0,
        correspondence_mode=mode, correspondence_alpha=0.5,
        correspondence_reproject=True,
        return_refined=True, verbose=True,
        **FIX4C_KWARGS,
    )
    return np.asarray(corr), refined_src, refined_warped


def build_corr_mesh(source, warped_pts, target):
    """Attach the three scalars (flipped/target_distance/warp_travel) to a
    PolyData built from source.faces + warped points."""
    mesh = pv.PolyData(warped_pts, source.faces)
    mesh.cell_data["flipped"] = foldover_per_triangle(source, warped_pts)
    mesh.point_data["target_distance"] = _point_to_surface_distances(warped_pts, target)
    mesh.point_data["warp_travel"] = np.linalg.norm(
        warped_pts - np.asarray(source.points), axis=1
    )
    return mesh


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        pv.start_xvfb()
    except Exception as exc:  # pragma: no cover
        print(f"[warn] start_xvfb failed: {exc}")
    pv.OFF_SCREEN = True

    print("loading model + cache ...")
    model = load_nsm_model()
    with open(MANIFEST_PATH) as f:
        keys = [r["key"] for r in json.load(f)]
    latents, meshes = load_cache(keys)

    base = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "report", "results_baseline__100.csv")
    )

    # Skip bone (no seam; refinement not needed) and skip cart (the sweep
    # already showed cart's ASSD penalty disappears at theta=30 without
    # needing refinement). Focus the experiment on the menisci, where
    # fold-over is still 40-50% even with the best non-refined config.
    for sidx, sname in enumerate(MESH_NAMES):
        if sname in ("bone", "cart"):
            continue
        sub = base[base.surface_name == sname]
        worst = sub.sort_values("foldover_fraction", ascending=False).iloc[0]
        key_a, key_b = worst["source"], worst["target"]
        print(
            f"\n=== {sname}: {key_a} -> {key_b}  "
            f"(baseline fold-over {worst['foldover_fraction']:.4f}) ==="
        )

        source = meshes[key_a][sidx]
        target = meshes[key_b][sidx]
        feature_mask = compute_feature_mask(
            source.regular_faces.astype(np.int64),
            np.asarray(source.points, dtype=np.float64),
            dihedral_threshold_deg=60.0,
        )
        lats = (latents[key_a], latents[key_b])

        warped_panels = {}

        print("  baseline ...")
        w_base = warp_baseline(model, lats, source, target, sidx)
        warped_panels[COLUMN_LABELS["baseline"]] = build_corr_mesh(source, w_base, target)

        print("  fix1_fix2_fix4c ...")
        w_4c = warp_fix4c(model, lats, source, target, sidx)
        warped_panels[COLUMN_LABELS["fix4c"]] = build_corr_mesh(source, w_4c, target)

        # Two refined variants: reactive only (no pre-split) vs. pre-split seam.
        refined_variants = [
            ("refined_vertex", "vertex", 0),
            ("refined_preseam_vertex", "vertex", 1),
        ]
        for col_key, mode, pre_hops in refined_variants:
            tag = f"pre{pre_hops}_{mode}"
            print(f"  fix4c + refine ({tag}) ...")
            w_ref, refined_src, refined_warped = warp_refined(
                model, lats, source, target, sidx, mode, pre_split_hops=pre_hops,
            )
            warped_panels[COLUMN_LABELS[col_key]] = build_corr_mesh(
                source, w_ref, target
            )
            refined_src.save(
                os.path.join(
                    OUT_DIR,
                    f"{sname}_{key_a}_to_{key_b}_refined_source_{tag}.vtk",
                )
            )
            refined_warped.save(
                os.path.join(
                    OUT_DIR,
                    f"{sname}_{key_a}_to_{key_b}_refined_warped_{tag}.vtk",
                )
            )

        # Save the N-original correspondence .vtks too.
        for label, mesh in warped_panels.items():
            slug = label.split("\n")[0].replace(" ", "_").replace("+", "_")
            mesh.save(os.path.join(OUT_DIR, f"{sname}_{key_a}_to_{key_b}_{slug}.vtk"))

        png_path = os.path.join(OUT_DIR, f"{sname}_{key_a}_to_{key_b}_refined.png")
        render_panel(source, target, feature_mask, warped_panels, sname, png_path)
        print(f"  -> {png_path}")

        rgb_path = os.path.join(OUT_DIR, f"{sname}_{key_a}_to_{key_b}_rgb.png")
        render_matched_rgb_panel(source, warped_panels, sname, rgb_path)
        print(f"  -> {rgb_path}")

    print(f"\nDone. Visuals + .vtks under {OUT_DIR}")


if __name__ == "__main__":
    main()
