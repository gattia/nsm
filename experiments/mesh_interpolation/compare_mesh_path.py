"""Compare interpolate_points (points path) against interpolate_mesh
(mesh path with pyvista's per-step adaptive subdivision + VTK smoothing).

For each thin-shell surface (med_men, lat_men) on the worst-fold-over
pair, run:

  * baseline (interpolate_points, no fixes)
  * fix4c    (interpolate_points + Newton magnitude + corrector + tangent
              Laplacian with dihedral seam pin)
  * mesh_adaptive_only  (interpolate_mesh adaptive=True, smooth=False)
  * mesh_adaptive_smooth (interpolate_mesh adaptive=True, smooth=True
                          Taubin -- pulls less off-surface than Laplacian)

The mesh path subdivides triangles per latent step via
``pyvista.PolyData.subdivide_adaptive`` and (optionally) Laplacian or
Taubin smooths the mesh. Crucially, ``subdivide_adaptive`` preserves
the original point IDs in [0..N_orig), so we can extract the
correspondence as ``warped.points[:N_orig]`` directly.

Outputs both PNGs (3-row metric panel + matched-RGB panel) and saved
.vtks for every variant.
"""

import json
import os

import numpy as np
import pandas as pd
import pyvista as pv
from pymskt.mesh import Mesh

from NSM.mesh.correspondence_metrics import _point_to_surface_distances
from NSM.mesh.interpolate import (
    compute_feature_mask,
    interpolate_mesh,
    interpolate_points,
)

from .config import EXPERIMENT_CONFIGS, MANIFEST_PATH, MESH_NAMES, load_nsm_model
from .compare_refined import (
    _make_position_rgb,
    foldover_per_triangle,
    render_matched_rgb_panel,
)
from .dump_visuals import render_panel
from .run_matrix import load_cache

OUT_DIR = os.path.join(os.path.dirname(__file__), "report", "visuals_mesh")

# Fix 4c kwargs (points path).
FIX4C_KWARGS = dict(EXPERIMENT_CONFIGS["fix1_fix2_fix4c"])

# Mesh-path kwargs: the mesh path doesn't support tangent_laplacian /
# feature pinning, but we still get Newton magnitude + corrector loop.
MESH_FIX_KWARGS = dict(n_corrector_iters=5, step_magnitude="newton")

COLUMN_LABELS = {
    "baseline":              "baseline",
    "fix4c":                 "fix4c\n(current best)",
    "mesh_adaptive_only":    "mesh_adaptive\n(no smooth)",
    "mesh_adaptive_smooth":  "mesh_adaptive\n+ Taubin smooth",
}
COLUMN_SLUGS = {
    "baseline":              "baseline",
    "fix4c":                 "fix4c",
    "mesh_adaptive_only":    "mesh_adaptive_only",
    "mesh_adaptive_smooth":  "mesh_adaptive_smooth",
}


def warp_points(model, latents, source, sidx, kw=None):
    kw = kw or {}
    src_pts = np.asarray(source.points, dtype=np.float64)
    faces = source.regular_faces.astype(np.int64)
    return np.asarray(
        interpolate_points(
            model, latents[0], latents[1], n_steps=100, points1=src_pts,
            surface_idx=sidx, faces=faces, **kw,
        )
    )


def warp_mesh(model, latents, source, sidx, *, adaptive, smooth,
              max_edge_len=0.04, smooth_type="taubin"):
    """interpolate_mesh modifies the wrapper in place; return both the
    over-resolved warped mesh AND the N-original correspondence (first
    n_orig points -- subdivide_adaptive preserves original IDs)."""
    n_orig = source.n_points
    wrapper = Mesh(pv.PolyData(source))  # pymskt wraps PolyData
    interpolate_mesh(
        model, latents[0], latents[1], n_steps=100, mesh=wrapper,
        surface_idx=sidx, verbose=True, spherical=True,
        max_edge_len=max_edge_len, adaptive=adaptive,
        smooth=smooth, smooth_type=smooth_type,
        **MESH_FIX_KWARGS,
    )
    refined_warped = wrapper.mesh if isinstance(wrapper.mesh, pv.PolyData) \
        else pv.PolyData(wrapper.mesh)
    corr = np.asarray(refined_warped.points[:n_orig])
    return corr, refined_warped


def build_corr_mesh(source, warped_pts, target):
    mesh = pv.PolyData(warped_pts, source.faces)
    mesh.cell_data["flipped"] = foldover_per_triangle(source, warped_pts)
    mesh.point_data["target_distance"] = _point_to_surface_distances(warped_pts, target)
    mesh.point_data["warp_travel"] = np.linalg.norm(
        warped_pts - np.asarray(source.points), axis=1
    )
    return mesh


def attach_metrics_to_refined(refined, source, target):
    """Tag a refined warped mesh with its own fold-over + target distance.

    For fold-over the 'parent' (source-side) connectivity is what the
    cells encode; since subdivide_adaptive preserves cell_idx, we can
    look up each cell's source-triangle, compute the source-side normal
    from that triangle's source coordinates, and the warped normal from
    the cell itself.
    """
    src_pts = np.asarray(source.points)
    src_faces = source.regular_faces
    # cell_idx -- which source triangle did this refined cell come from?
    if "cell_idx" not in refined.cell_data:
        # No parent info -- compute warped-only fold-over against
        # itself using a sign-convention from triangle centroids vs outward.
        refined.cell_data["flipped"] = np.zeros(refined.n_cells, dtype=np.uint8)
    else:
        parents = np.asarray(refined.cell_data["cell_idx"], dtype=np.int64)
        wf = refined.regular_faces.astype(np.int64)
        wp = np.asarray(refined.points)
        n_wrp = np.cross(wp[wf[:, 1]] - wp[wf[:, 0]], wp[wf[:, 2]] - wp[wf[:, 0]])
        n_src_parent = np.cross(
            src_pts[src_faces[parents, 1]] - src_pts[src_faces[parents, 0]],
            src_pts[src_faces[parents, 2]] - src_pts[src_faces[parents, 0]],
        )
        refined.cell_data["flipped"] = (
            (n_src_parent * n_wrp).sum(axis=1) < 0
        ).astype(np.uint8)
    refined.point_data["target_distance"] = _point_to_surface_distances(
        np.asarray(refined.points), target
    )
    return refined


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        pv.start_xvfb()
    except Exception as exc:
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

    for sidx, sname in enumerate(MESH_NAMES):
        if sname not in ("med_men", "lat_men"):
            continue
        sub = base[base.surface_name == sname]
        worst = sub.sort_values("foldover_fraction", ascending=False).iloc[0]
        key_a, key_b = worst["source"], worst["target"]
        print(f"\n=== {sname}: {key_a} -> {key_b}  "
              f"(baseline fold {worst['foldover_fraction']:.4f}) ===")
        source = meshes[key_a][sidx]
        target = meshes[key_b][sidx]
        lats = (latents[key_a], latents[key_b])
        feature_mask = compute_feature_mask(
            source.regular_faces.astype(np.int64),
            np.asarray(source.points, dtype=np.float64),
            dihedral_threshold_deg=60.0,
        )
        warped_panels = {}

        print("  baseline ...")
        wp = warp_points(model, lats, source, sidx, kw={})
        warped_panels[COLUMN_LABELS["baseline"]] = build_corr_mesh(source, wp, target)

        print("  fix4c ...")
        wp = warp_points(model, lats, source, sidx, kw=FIX4C_KWARGS)
        warped_panels[COLUMN_LABELS["fix4c"]] = build_corr_mesh(source, wp, target)

        print("  mesh_adaptive_only ...")
        corr, refined = warp_mesh(model, lats, source, sidx,
                                  adaptive=True, smooth=False)
        warped_panels[COLUMN_LABELS["mesh_adaptive_only"]] = \
            build_corr_mesh(source, corr, target)
        attach_metrics_to_refined(refined, source, target)
        refined.save(os.path.join(
            OUT_DIR, f"{sname}_{key_a}_to_{key_b}_refined_mesh_adaptive_only.vtk"))

        print("  mesh_adaptive_smooth (Taubin) ...")
        corr, refined = warp_mesh(model, lats, source, sidx,
                                  adaptive=True, smooth=True, smooth_type="taubin")
        warped_panels[COLUMN_LABELS["mesh_adaptive_smooth"]] = \
            build_corr_mesh(source, corr, target)
        attach_metrics_to_refined(refined, source, target)
        refined.save(os.path.join(
            OUT_DIR, f"{sname}_{key_a}_to_{key_b}_refined_mesh_adaptive_smooth.vtk"))

        # Save N-correspondence .vtks.
        for col, slug in COLUMN_SLUGS.items():
            warped_panels[COLUMN_LABELS[col]].save(
                os.path.join(OUT_DIR, f"{sname}_{key_a}_to_{key_b}_{slug}.vtk")
            )

        png_path = os.path.join(OUT_DIR, f"{sname}_{key_a}_to_{key_b}_mesh.png")
        render_panel(source, target, feature_mask, warped_panels, sname, png_path)
        print(f"  -> {png_path}")

        rgb_path = os.path.join(OUT_DIR, f"{sname}_{key_a}_to_{key_b}_rgb.png")
        render_matched_rgb_panel(source, warped_panels, sname, rgb_path)
        print(f"  -> {rgb_path}")

    print(f"\nDone. Visuals + .vtks under {OUT_DIR}")


if __name__ == "__main__":
    main()
