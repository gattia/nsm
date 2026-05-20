"""Render before/after warps for representative cells.

For each cart / menisci surface, finds the worst-fold-over pair in the
cached baseline results, re-runs that pair under five configs (baseline,
fix1_fix2, fix1_fix2_fix4, fix1_fix2_fix4c, fix1_fix2_fix4c_fix7c), saves
the warped ``.vtk`` files for inspection in ParaView, and renders a
comparison PNG with:

  * source mesh (with the dihedral seam highlighted in green),
  * target mesh,
  * each warped mesh with flipped triangles in red and an inset summary
    showing the fold-over fraction.

Requires a GPU for the interpolation. Off-screen rendering via xvfb.
"""

import os

import numpy as np
import pandas as pd
import pyvista as pv

from NSM.mesh.interpolate import compute_feature_mask, interpolate_points

from .config import EXPERIMENT_CONFIGS, MESH_NAMES, MANIFEST_PATH, load_nsm_model
from .run_matrix import load_cache

# Output directory for VTKs and PNGs.
OUT_DIR = os.path.join(os.path.dirname(__file__), "report", "visuals")

# Configs to show side-by-side, in this column order.
COLUMN_CONFIGS = [
    "baseline",
    "fix1_fix2",
    "fix1_fix2_fix4",
    "fix1_fix2_fix4c",
    "fix1_fix2_fix4c_fix7c",
]
# Concise display labels (the panel titles).
COLUMN_LABELS = {
    "baseline": "baseline",
    "fix1_fix2": "fix1+2\n(corrector+Newton)",
    "fix1_fix2_fix4": "+fix4\n(tangent Lap)",
    "fix1_fix2_fix4c": "+fix4c\n(dihedral pin)",
    "fix1_fix2_fix4c_fix7c": "+fix4c+fix7c\n(stacked)",
}


def foldover_per_triangle(source, warped_pts):
    """Per-triangle flag: 1 if orientation flipped vs source, else 0."""
    f = np.asarray(source.regular_faces)
    s = np.asarray(source.points)
    n_src = np.cross(s[f[:, 1]] - s[f[:, 0]], s[f[:, 2]] - s[f[:, 0]])
    n_wrp = np.cross(
        warped_pts[f[:, 1]] - warped_pts[f[:, 0]],
        warped_pts[f[:, 2]] - warped_pts[f[:, 0]],
    )
    return ((n_src * n_wrp).sum(axis=1) < 0).astype(np.uint8)


def render_panel(source, target, feature_mask, warped_meshes, surface_name, out_png):
    """One row of subplots: source(+seam), target, then each warped config."""
    pv.set_plot_theme("document")
    n = len(warped_meshes) + 2
    width = 360 * n
    plotter = pv.Plotter(off_screen=True, shape=(1, n), window_size=(width, 500))

    # Source with the dihedral seam highlighted.
    plotter.subplot(0, 0)
    plotter.add_mesh(source, color="lightsteelblue", show_edges=False, lighting=True)
    if feature_mask.any():
        seam = source.points[feature_mask]
        plotter.add_points(
            seam, color="green", point_size=5, render_points_as_spheres=True
        )
    plotter.add_text(
        f"source\nseam {int(feature_mask.sum())} pts",
        font_size=9,
        position="upper_left",
    )

    plotter.subplot(0, 1)
    plotter.add_mesh(target, color="peachpuff", show_edges=False, lighting=True)
    plotter.add_text("target", font_size=9, position="upper_left")

    for col, (label, warped) in enumerate(warped_meshes.items()):
        plotter.subplot(0, col + 2)
        flipped = warped.cell_data["flipped"]
        frac = 100.0 * float(flipped.sum()) / max(warped.n_cells, 1)
        plotter.add_mesh(
            warped,
            scalars="flipped",
            cmap=["lightgray", "red"],
            clim=[0, 1],
            show_scalar_bar=False,
            show_edges=False,
            lighting=True,
        )
        plotter.add_text(
            f"{label}\nfold {frac:.2f}%", font_size=9, position="upper_left"
        )

    plotter.link_views()
    if surface_name in ("cart", "med_men", "lat_men"):
        # Thin shells -- looking down the z-axis shows the geometry best.
        plotter.view_xy()
        plotter.camera.zoom(1.2)
    else:
        plotter.view_isometric()
    plotter.screenshot(out_png)
    plotter.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        pv.start_xvfb()
    except Exception as exc:  # pragma: no cover - render env best-effort
        print(f"[warn] start_xvfb failed: {exc}")
    pv.OFF_SCREEN = True

    print("loading model + cache ...")
    model = load_nsm_model()
    import json
    with open(MANIFEST_PATH) as f:
        keys = [r["key"] for r in json.load(f)]
    latents, meshes = load_cache(keys)

    # For each surface, find the pair with the highest baseline fold-over.
    base = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "report", "results_baseline__100.csv")
    )

    for sidx, sname in enumerate(MESH_NAMES):
        if sname == "bone":
            continue  # bone is the easy case; focus on the thin-shell pain
        sub = base[base.surface_name == sname]
        worst = sub.sort_values("foldover_fraction", ascending=False).iloc[0]
        key_a, key_b = worst["source"], worst["target"]
        print(f"\n=== {sname}: {key_a} -> {key_b}  "
              f"(baseline fold-over {worst['foldover_fraction']:.4f}) ===")

        source = meshes[key_a][sidx]
        target = meshes[key_b][sidx]
        faces = source.regular_faces.astype(np.int64)
        src_pts = np.asarray(source.points, dtype=np.float64)
        feature_mask = compute_feature_mask(
            faces, src_pts, dihedral_threshold_deg=60.0
        )

        warped_panels = {}
        for cfg_name in COLUMN_CONFIGS:
            cfg = EXPERIMENT_CONFIGS[cfg_name]
            print(f"  warping under {cfg_name} ...")
            warped_pts = np.asarray(
                interpolate_points(
                    model, latents[key_a], latents[key_b],
                    n_steps=100, points1=src_pts, surface_idx=sidx,
                    faces=faces, **cfg,
                )
            )
            warped = pv.PolyData(warped_pts, source.faces)
            warped.cell_data["flipped"] = foldover_per_triangle(source, warped_pts)
            vtk_path = os.path.join(
                OUT_DIR, f"{sname}_{key_a}_to_{key_b}_{cfg_name}.vtk"
            )
            warped.save(vtk_path)
            warped_panels[COLUMN_LABELS[cfg_name]] = warped

        png_path = os.path.join(OUT_DIR, f"{sname}_{key_a}_to_{key_b}.png")
        render_panel(source, target, feature_mask, warped_panels, sname, png_path)
        print(f"  -> {png_path}")

    print(f"\nDone. Visuals + .vtk dumps under {OUT_DIR}")


if __name__ == "__main__":
    main()
