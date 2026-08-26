"""
Correspondence quality metrics for warped meshes / point sets.

Scores a *warped* mesh or point set against a *target* mesh. All functions are
project-agnostic: no hard-coded paths or anatomy-specific logic.

Meshes are accepted as ``pyvista.PolyData`` (triangular). A *warped point set*
is an ``(N, 3)`` numpy array that shares the source mesh's connectivity (faces).

Families
--------
Family 1 — surface fit and mesh health:
    assd, directed_distance_percentiles, off_surface_error,
    triangle_health, self_intersection_count

Family 2 — correspondence-specific:
    foldover_count, roundtrip_distance, forward_backward_disagreement

Top-level scorer:
    score_correspondence
"""

import warnings
from typing import Dict, Optional, Union

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from NSM.mesh.triangle_metrics import (
    TriangleProperties,
    calculate_triangle_areas,
    get_edge_lengths,
    triangle_faces,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PERCENTILE_KEYS = ("min", "p25", "p50", "mean", "p75", "p95", "max")


def _percentile_dict(values: np.ndarray) -> Dict[str, float]:
    """Return a dict of summary statistics for a 1-D array.

    Args:
        values: 1-D array of non-negative scalar values.

    Returns:
        Dict with keys ``min``, ``p25``, ``p50``, ``mean``, ``p75``, ``p95``, ``max``.
    """
    v = np.asarray(values, dtype=float).ravel()
    return {
        "min": float(np.min(v)),
        "p25": float(np.percentile(v, 25)),
        "p50": float(np.percentile(v, 50)),
        "mean": float(np.mean(v)),
        "p75": float(np.percentile(v, 75)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
    }


def _mesh_points(mesh: pv.PolyData) -> np.ndarray:
    """Return the (N, 3) float64 vertex array from a pyvista PolyData."""
    return np.asarray(mesh.points, dtype=float)


# Exact point-to-surface (point-to-triangle) distance, via pymskt's wrapper
# around point_cloud_utils. Imported lazily-safely so the module still imports
# if pymskt/pcu is unavailable (the nearest-vertex fallback is then used).
try:  # pragma: no cover - import guard
    from pymskt.mesh.meshes import pcu_sdf as _pcu_sdf
except Exception:  # pragma: no cover
    _pcu_sdf = None


def _point_to_surface_distances(points: np.ndarray, mesh: pv.PolyData) -> np.ndarray:
    """Unsigned distance from each point to the *surface* of ``mesh``.

    Uses point_cloud_utils' exact point-to-triangle distance (through pymskt's
    ``pcu_sdf``). This is the accurate metric: nearest-*vertex* (point-to-point)
    distance systematically over-estimates -- it floors out at roughly the
    target mesh's edge length even when a point lies exactly on the surface.

    Falls back to nearest-vertex distance only if point_cloud_utils is
    unavailable.

    Args:
        points: (N, 3) array of query points.
        mesh: pyvista PolyData triangular mesh to measure distance to.

    Returns:
        (N,) array of non-negative point-to-surface distances.
    """
    pts = np.asarray(points, dtype=float)
    if _pcu_sdf is not None:
        # point_cloud_utils requires the query points and the mesh vertices to
        # share a dtype; pyvista meshes are typically float32, so match it.
        try:
            mesh_dtype = np.asarray(mesh.points).dtype
        except AttributeError:  # pragma: no cover - vtkPolyData without .points
            mesh_dtype = np.float64
        return np.abs(np.asarray(_pcu_sdf(pts.astype(mesh_dtype), mesh), dtype=float))
    # Fallback: nearest-vertex distance (over-estimates; see note above).
    dist, _ = cKDTree(_mesh_points(mesh)).query(pts, k=1)
    return dist


# ---------------------------------------------------------------------------
# Family 1 — surface fit and mesh health
# ---------------------------------------------------------------------------


def assd(mesh_a: pv.PolyData, mesh_b: pv.PolyData) -> float:
    """Average symmetric surface distance between two meshes (point-to-surface).

    Each vertex of ``mesh_a`` is measured to the nearest point on the *surface*
    of ``mesh_b`` (exact point-to-triangle distance), and vice versa; the ASSD
    is the mean of both sets of distances. This matches
    :meth:`pymskt.mesh.Mesh.get_assd_mesh`.

    .. note::
        Point-to-*surface* distance is used deliberately. Nearest-*vertex*
        (point-to-point) distance over-estimates: it cannot fall below roughly
        the target mesh's edge length even for a point lying exactly on the
        surface, so it conflates correspondence error with mesh resolution.

    Args:
        mesh_a: First pyvista PolyData mesh.
        mesh_b: Second pyvista PolyData mesh.

    Returns:
        Average symmetric surface distance as a float.
    """
    dist_a_to_b = _point_to_surface_distances(_mesh_points(mesh_a), mesh_b)
    dist_b_to_a = _point_to_surface_distances(_mesh_points(mesh_b), mesh_a)

    total = float(dist_a_to_b.sum() + dist_b_to_a.sum())
    return total / (len(dist_a_to_b) + len(dist_b_to_a))


def directed_distance_percentiles(
    source_points: np.ndarray, target: Union[pv.PolyData, np.ndarray]
) -> Dict[str, float]:
    """Directed distance distribution from source points to a target.

    For each point in ``source_points`` the distance to ``target`` is computed.
    When ``target`` is a mesh the *point-to-surface* distance is used (exact,
    accurate); when ``target`` is an ``(M, 3)`` point array the nearest-vertex
    (point-to-point) distance is used -- a fast proxy that over-estimates (it
    floors out near the target's vertex spacing). Prefer passing a mesh.

    Call this function twice -- once in each direction -- to characterise both
    over-reach (warped -> target) and coverage gaps (target -> warped).

    Args:
        source_points: (N, 3) array of query points.
        target: target geometry, either a pyvista PolyData mesh
            (point-to-surface) or an (M, 3) point array (point-to-point).

    Returns:
        Dict with keys ``min``, ``p25``, ``p50``, ``mean``, ``p75``, ``p95``,
        ``max``.
    """
    src = np.asarray(source_points, dtype=float)

    if isinstance(target, np.ndarray):
        distances, _ = cKDTree(target).query(src, k=1)
    else:
        distances = _point_to_surface_distances(src, target)

    return _percentile_dict(distances)


def off_surface_error(sdf_values: np.ndarray) -> Dict[str, float]:
    """Distribution of absolute SDF residuals at warped points.

    Quantifies how far the warped points are from the zero-level-set of the
    target SDF decoder.  Caller is responsible for evaluating the decoder at
    the warped points and passing the resulting signed-distance values here.

    Args:
        sdf_values: 1-D array of signed-distance values at the warped points
            (as returned by the decoder).  Positive ↔ outside, negative ↔
            inside.

    Returns:
        Dict with keys ``min``, ``p25``, ``p50``, ``mean``, ``p75``, ``p95``,
        ``max``, ``rms``.
    """
    abs_vals = np.abs(np.asarray(sdf_values, dtype=float).ravel())
    result = _percentile_dict(abs_vals)
    result["rms"] = float(np.sqrt(np.mean(abs_vals**2)))
    return result


def triangle_health(mesh: pv.PolyData) -> Dict[str, float]:
    """Mesh quality statistics derived from edge lengths and triangle areas.

    Uses :class:`~NSM.mesh.triangle_metrics.TriangleProperties` to compute
    per-triangle edge lengths and areas, then summarises them.

    Args:
        mesh: pyvista PolyData triangular mesh.

    Returns:
        Flat dict with keys:

        * ``edge_length_mean``, ``edge_length_std``, ``edge_length_min``,
          ``edge_length_max``
        * ``area_mean``, ``area_std``, ``area_min``, ``area_max``
        * ``edge_ratio_mean``, ``edge_ratio_p95``, ``edge_ratio_max``
        * ``degenerate_count`` — triangles whose shortest edge is shorter than
          ``1e-6 * median(all edge lengths)``
    """
    vtk_polydata = mesh.GetOutput() if hasattr(mesh, "GetOutput") else mesh
    # pyvista.PolyData *is* a vtkPolyData subclass; pass it directly.
    props = TriangleProperties(vtk_polydata)
    props.compute_edge_lengths()  # shape (n_cells, 3)

    edge_lengths = props.edge_lengths  # (n_cells, 3)
    all_edges = edge_lengths.ravel()

    areas = np.array(calculate_triangle_areas(vtk_polydata), dtype=float)

    # Edge ratio (aspect) — handle degenerate triangles gracefully
    min_edges = np.min(edge_lengths, axis=1)
    max_edges = np.max(edge_lengths, axis=1)
    median_edge = float(np.median(all_edges))
    degenerate_threshold = 1e-6 * max(median_edge, 1e-12)

    degenerate_mask = min_edges < degenerate_threshold
    degenerate_count = int(np.sum(degenerate_mask))

    # Compute edge ratios only for non-degenerate triangles to avoid division by zero
    valid_mask = ~degenerate_mask
    if np.any(valid_mask):
        ratios = max_edges[valid_mask] / np.maximum(min_edges[valid_mask], degenerate_threshold)
    else:
        ratios = np.array([np.nan])

    return {
        "edge_length_mean": float(np.mean(all_edges)),
        "edge_length_std": float(np.std(all_edges)),
        "edge_length_min": float(np.min(all_edges)),
        "edge_length_max": float(np.max(all_edges)),
        "area_mean": float(np.mean(areas)),
        "area_std": float(np.std(areas)),
        "area_min": float(np.min(areas)),
        "area_max": float(np.max(areas)),
        "edge_ratio_mean": float(np.nanmean(ratios)),
        "edge_ratio_p95": float(np.nanpercentile(ratios, 95)),
        "edge_ratio_max": float(np.nanmax(ratios)),
        "degenerate_count": degenerate_count,
    }


def self_intersection_count(
    mesh: pv.PolyData, broadphase: bool = True, max_triangles: int = 50_000
) -> Optional[int]:
    """Count self-intersecting triangle pairs in the mesh.

    Uses a bounding-box broadphase to quickly discard non-overlapping triangle
    pairs, then a per-pair Möller–Trumbore triangle-triangle intersection test
    for the candidates that share no vertex.

    For meshes larger than ``max_triangles`` the function returns ``None`` and
    emits a warning rather than hanging.

    Args:
        mesh: pyvista PolyData triangular mesh.
        broadphase: If ``True`` (default), apply an AABB broadphase to reduce
            the number of narrow-phase tests.  Set to ``False`` to skip
            broadphase (useful only for very small meshes).
        max_triangles: Upper limit on triangle count.  If the mesh has more
            triangles than this, return ``None`` and warn.  Default is 50 000.

    Returns:
        Number of self-intersecting triangle pairs (int), or ``None`` if the
        mesh exceeds ``max_triangles``.
    """
    n_tris = mesh.n_cells
    if n_tris > max_triangles:
        warnings.warn(
            f"self_intersection_count: mesh has {n_tris} triangles, which exceeds "
            f"max_triangles={max_triangles}. Returning None to avoid excessive runtime.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    pts = np.asarray(mesh.points, dtype=float)
    faces = triangle_faces(mesh)  # shape (n_tris, 3)

    # Precompute triangle vertices as arrays for vectorised ops
    v0 = pts[faces[:, 0]]  # (n_tris, 3)
    v1 = pts[faces[:, 1]]
    v2 = pts[faces[:, 2]]

    # Bounding boxes: (n_tris, 3) min/max
    tri_min = np.minimum(np.minimum(v0, v1), v2)  # (n_tris, 3)
    tri_max = np.maximum(np.maximum(v0, v1), v2)

    # Build a sorted list of (axis_value, event_type, tri_idx) for a sweep-and-prune
    # along the x-axis to find AABB overlaps quickly.
    if broadphase:
        candidates = _aabb_broadphase(tri_min, tri_max, n_tris)
    else:
        # Brute-force: all non-adjacent pairs
        i_idx, j_idx = np.triu_indices(n_tris, k=1)
        candidates = list(zip(i_idx.tolist(), j_idx.tolist()))

    count = 0
    for i, j in candidates:
        # Skip pairs that share a vertex (adjacent triangles)
        if _share_vertex(faces[i], faces[j]):
            continue
        if _tri_tri_intersect(v0[i], v1[i], v2[i], v0[j], v1[j], v2[j]):
            count += 1

    return count


def _aabb_broadphase(tri_min: np.ndarray, tri_max: np.ndarray, n_tris: int):
    """Sweep-and-prune broadphase on the x-axis.

    Returns a list of (i, j) index pairs whose AABBs overlap in all 3 axes.
    """
    # Sort triangles by x-min
    order = np.argsort(tri_min[:, 0])
    sorted_min_x = tri_min[order, 0]
    sorted_max_x = tri_max[order, 0]

    candidates = []
    for rank_i in range(n_tris):
        i = int(order[rank_i])
        max_x_i = sorted_max_x[rank_i]
        # Advance j through triangles whose x-min <= max_x_i
        for rank_j in range(rank_i + 1, n_tris):
            if sorted_min_x[rank_j] > max_x_i:
                break  # no further overlap possible in x
            j = int(order[rank_j])
            # Check overlap in y and z
            if tri_max[i, 1] < tri_min[j, 1] or tri_max[j, 1] < tri_min[i, 1]:
                continue
            if tri_max[i, 2] < tri_min[j, 2] or tri_max[j, 2] < tri_min[i, 2]:
                continue
            # Canonical ordering: smaller index first
            a, b = (i, j) if i < j else (j, i)
            candidates.append((a, b))

    return candidates


def _share_vertex(face_i: np.ndarray, face_j: np.ndarray) -> bool:
    """Return True if two triangles (vertex index triples) share any vertex."""
    return bool(np.any(np.isin(face_i, face_j)))


def _tri_tri_intersect(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    eps: float = 1e-7,
) -> bool:
    """Möller (1997) triangle-triangle intersection test.

    Returns True if triangle (p0, p1, p2) intersects triangle (q0, q1, q2)
    in their *interiors* (i.e. they cross through each other).  Triangles that
    merely touch at a single point or share an edge are **not** counted as
    intersecting, which avoids false positives on manifold meshes where
    adjacent faces share boundary geometry.

    Coplanar triangles are conservatively treated as non-intersecting.

    Args:
        p0, p1, p2: Vertices of the first triangle.
        q0, q1, q2: Vertices of the second triangle.
        eps: Numerical tolerance for plane-distance comparisons.  Vertices
            with ``|d| < eps`` are treated as lying on the plane.

    Returns:
        True if the triangles properly intersect, False otherwise.
    """
    # Plane of triangle Q: nq·x + dq = 0
    nq = np.cross(q1 - q0, q2 - q0)
    dq_offset = -np.dot(nq, q0)

    # Signed distances of P's vertices to Q's plane
    dp = np.array(
        [np.dot(nq, p0) + dq_offset, np.dot(nq, p1) + dq_offset, np.dot(nq, p2) + dq_offset]
    )

    # Clamp near-zero distances to zero to avoid sign-flip noise
    dp = np.where(np.abs(dp) < eps, 0.0, dp)

    if np.all(dp >= 0) or np.all(dp <= 0):
        # All on one side (or touching a plane) — no proper crossing
        return False

    # Plane of triangle P: np_·x + dp2 = 0
    np_ = np.cross(p1 - p0, p2 - p0)
    dp2_offset = -np.dot(np_, p0)

    # Signed distances of Q's vertices to P's plane
    dq_vals = np.array(
        [np.dot(np_, q0) + dp2_offset, np.dot(np_, q1) + dp2_offset, np.dot(np_, q2) + dp2_offset]
    )
    dq_vals = np.where(np.abs(dq_vals) < eps, 0.0, dq_vals)

    if np.all(dq_vals >= 0) or np.all(dq_vals <= 0):
        return False

    # Coplanar case: skip (conservative — avoids false positives)
    line_dir = np.cross(nq, np_)
    line_norm = np.linalg.norm(line_dir)
    if line_norm < eps:
        return False
    line_dir = line_dir / line_norm

    def _interval(v0_, v1_, v2_, d_):
        """Compute the 1-D interval of a triangle's intersection with the line."""
        # Identify the isolated vertex (the one on the opposite side from the other two)
        signs = np.sign(d_)  # -1, 0, or +1
        # Among the three vertices, find the one whose sign differs from the majority
        if signs[0] == signs[1] or (signs[0] != 0 and signs[1] == 0 and signs[0] == signs[2]):
            iso, other1, other2 = v2_, v0_, v1_
            d_iso, d_o1, d_o2 = d_[2], d_[0], d_[1]
        elif signs[0] == signs[2] or (signs[0] != 0 and signs[2] == 0 and signs[0] == signs[1]):
            iso, other1, other2 = v1_, v0_, v2_
            d_iso, d_o1, d_o2 = d_[1], d_[0], d_[2]
        else:
            iso, other1, other2 = v0_, v1_, v2_
            d_iso, d_o1, d_o2 = d_[0], d_[1], d_[2]

        proj_iso = np.dot(line_dir, iso)
        proj_o1 = np.dot(line_dir, other1)
        proj_o2 = np.dot(line_dir, other2)

        denom1 = d_iso - d_o1
        denom2 = d_iso - d_o2
        t1 = proj_o1 + (proj_iso - proj_o1) * (d_o1 / denom1) if abs(denom1) > eps else proj_o1
        t2 = proj_o2 + (proj_iso - proj_o2) * (d_o2 / denom2) if abs(denom2) > eps else proj_o2
        return min(t1, t2), max(t1, t2)

    t_p = _interval(p0, p1, p2, dp)
    t_q = _interval(q0, q1, q2, dq_vals)

    # Intervals must *properly* overlap (not just touch at a single point)
    return t_p[0] < t_q[1] - eps and t_q[0] < t_p[1] - eps


# ---------------------------------------------------------------------------
# Family 2 — correspondence-specific
# ---------------------------------------------------------------------------


def foldover_count(
    source_mesh: pv.PolyData, warped_points: np.ndarray
) -> Dict[str, Union[int, float]]:
    """Count triangles whose orientation flipped after warping.

    For each triangle the un-normalised normal is computed on the source mesh
    and on the warped point set.  If ``dot(n_source, n_warped) < 0`` the
    triangle has flipped.

    Args:
        source_mesh: Original (un-warped) pyvista PolyData mesh.
        warped_points: (N, 3) array of warped vertex positions; must share
            the connectivity (faces) of ``source_mesh``.

    Returns:
        Dict with keys:

        * ``flipped_count`` — number of triangles whose orientation reversed.
        * ``flipped_fraction`` — ``flipped_count / n_triangles``.
        * ``near_degenerate`` — count of warped triangles whose normal magnitude
          is < 1e-10 (collapsed triangles).
    """
    src_pts = _mesh_points(source_mesh)
    warped = np.asarray(warped_points, dtype=float)

    faces = triangle_faces(source_mesh)  # (n_tris, 3)
    n_tris = len(faces)

    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]

    # Source normals
    e1_src = src_pts[i1] - src_pts[i0]
    e2_src = src_pts[i2] - src_pts[i0]
    n_src = np.cross(e1_src, e2_src)  # (n_tris, 3)

    # Warped normals
    e1_wrp = warped[i1] - warped[i0]
    e2_wrp = warped[i2] - warped[i0]
    n_wrp = np.cross(e1_wrp, e2_wrp)  # (n_tris, 3)

    dot_products = np.einsum("ij,ij->i", n_src, n_wrp)  # (n_tris,)

    flipped = int(np.sum(dot_products < 0))
    near_deg = int(np.sum(np.linalg.norm(n_wrp, axis=1) < 1e-10))

    return {
        "flipped_count": flipped,
        "flipped_fraction": float(flipped / n_tris) if n_tris > 0 else 0.0,
        "near_degenerate": near_deg,
    }


def roundtrip_distance(
    original_points: np.ndarray, roundtrip_points: np.ndarray
) -> Dict[str, object]:
    """Per-vertex distance after a forward-then-backward warp (A → B → A).

    Measures how close the round-trip lands to the starting position.  A
    perfect bijective correspondence would give zero displacement everywhere.

    Args:
        original_points: (N, 3) array of starting vertex positions (source A).
        roundtrip_points: (N, 3) array of positions after A → B → A warp.

    Returns:
        Dict containing the percentile statistics (keys: ``min``, ``p25``,
        ``p50``, ``mean``, ``p75``, ``p95``, ``max``) plus ``per_vertex``
        — the raw (N,) distance array.
    """
    orig = np.asarray(original_points, dtype=float)
    rt = np.asarray(roundtrip_points, dtype=float)

    per_vertex = np.linalg.norm(rt - orig, axis=1)
    result = _percentile_dict(per_vertex)
    result["per_vertex"] = per_vertex
    return result


def forward_backward_disagreement(
    roundtrip_points: np.ndarray, original_points: np.ndarray
) -> Dict[str, object]:
    """Displacement field between round-trip positions and originals.

    Returns both the raw (N, 3) displacement field and the magnitude
    statistics.  High values indicate regions where the forward and backward
    maps disagree — a signal of topology mismatch or poor correspondence
    quality.

    Args:
        roundtrip_points: (N, 3) positions after A → B → A warp.
        original_points: (N, 3) original vertex positions (source A).

    Returns:
        Dict with keys:

        * ``field`` — (N, 3) displacement array (``roundtrip - original``).
        * ``magnitude_percentiles`` — percentile dict of per-vertex displacement
          magnitudes (keys: ``min``, ``p25``, ``p50``, ``mean``, ``p75``,
          ``p95``, ``max``).
    """
    rt = np.asarray(roundtrip_points, dtype=float)
    orig = np.asarray(original_points, dtype=float)

    field = rt - orig  # (N, 3)
    magnitudes = np.linalg.norm(field, axis=1)

    return {
        "field": field,
        "magnitude_percentiles": _percentile_dict(magnitudes),
    }


# ---------------------------------------------------------------------------
# Top-level scorer
# ---------------------------------------------------------------------------


def score_correspondence(
    warped_mesh: pv.PolyData,
    target_mesh: pv.PolyData,
    source_mesh: Optional[pv.PolyData] = None,
    sdf_values: Optional[np.ndarray] = None,
    roundtrip_points: Optional[np.ndarray] = None,
    compute_self_intersection: bool = True,
) -> Dict[str, object]:
    """Run all applicable correspondence quality metrics and return a nested dict.

    Metrics requiring optional inputs are skipped gracefully: when an input is
    absent the corresponding key maps to ``{"skipped": True, "reason": "..."}``.

    Args:
        warped_mesh: Warped pyvista PolyData (the correspondence being evaluated).
        target_mesh: Ground-truth target pyvista PolyData.
        source_mesh: Original (un-warped) pyvista PolyData.  Required for
            ``foldover_count``.  If ``None``, that metric is skipped.
        sdf_values: 1-D array of signed-distance values at the warped vertices
            (caller evaluates the decoder).  Required for ``off_surface_error``.
            If ``None``, that metric is skipped.
        roundtrip_points: (N, 3) positions after a forward+backward warp
            (A → B → A).  Required for ``roundtrip_distance`` and
            ``forward_backward_disagreement``, **together with** ``source_mesh``,
            which supplies the positions they are measured against.  If either is
            ``None``, those two metrics are skipped.
        compute_self_intersection: If ``False``, skip ``self_intersection_count``
            entirely (useful for large meshes where even the broadphase is slow).

    Returns:
        Nested dict keyed by metric name.  Each value is one of three shapes:
        the metric result dict / scalar; ``{"skipped": True, "reason": "<str>"}``
        when a required input was absent; or ``{"error": "<str>"}`` when the
        metric raised (the exception is swallowed so one bad metric cannot sink
        the rest).
    """
    results: Dict[str, object] = {}

    # ---- assd ---------------------------------------------------------------
    try:
        results["assd"] = assd(warped_mesh, target_mesh)
    except Exception as exc:  # pragma: no cover
        results["assd"] = {"error": str(exc)}

    # ---- directed_distance_percentiles --------------------------------------
    # Pass meshes (not point arrays) so distances are point-to-surface.
    try:
        warped_pts = _mesh_points(warped_mesh)
        target_pts = _mesh_points(target_mesh)
        results["directed_distance_warped_to_target"] = directed_distance_percentiles(
            warped_pts, target_mesh
        )
        results["directed_distance_target_to_warped"] = directed_distance_percentiles(
            target_pts, warped_mesh
        )
    except Exception as exc:  # pragma: no cover
        results["directed_distance_warped_to_target"] = {"error": str(exc)}
        results["directed_distance_target_to_warped"] = {"error": str(exc)}

    # ---- off_surface_error --------------------------------------------------
    if sdf_values is not None:
        try:
            results["off_surface_error"] = off_surface_error(sdf_values)
        except Exception as exc:  # pragma: no cover
            results["off_surface_error"] = {"error": str(exc)}
    else:
        results["off_surface_error"] = {
            "skipped": True,
            "reason": "sdf_values not provided",
        }

    # ---- triangle_health (warped mesh) --------------------------------------
    try:
        results["triangle_health"] = triangle_health(warped_mesh)
    except Exception as exc:  # pragma: no cover
        results["triangle_health"] = {"error": str(exc)}

    # ---- self_intersection_count --------------------------------------------
    if compute_self_intersection:
        try:
            results["self_intersection_count"] = self_intersection_count(warped_mesh)
        except Exception as exc:  # pragma: no cover
            results["self_intersection_count"] = {"error": str(exc)}
    else:
        results["self_intersection_count"] = {
            "skipped": True,
            "reason": "compute_self_intersection=False",
        }

    # ---- foldover_count -----------------------------------------------------
    if source_mesh is not None:
        try:
            results["foldover_count"] = foldover_count(source_mesh, _mesh_points(warped_mesh))
        except Exception as exc:  # pragma: no cover
            results["foldover_count"] = {"error": str(exc)}
    else:
        results["foldover_count"] = {
            "skipped": True,
            "reason": "source_mesh not provided",
        }

    # ---- roundtrip_distance / forward_backward_disagreement -----------------
    # Both measure how far a forward+backward warp lands from where it started, so the
    # starting positions are the *source* mesh's. Substituting the warped mesh (which is
    # what this did until Aug 2026) measures the warp itself and reports it as a
    # round-trip error: on a 1.5x scaling, 0.2500 where the true answer was 0.0017.
    if roundtrip_points is None or source_mesh is None:
        reason = (
            "roundtrip_points not provided"
            if roundtrip_points is None
            else "source_mesh not provided"
        )
        results["roundtrip_distance"] = {"skipped": True, "reason": reason}
        results["forward_backward_disagreement"] = {"skipped": True, "reason": reason}
    else:
        original_pts = _mesh_points(source_mesh)
        try:
            results["roundtrip_distance"] = roundtrip_distance(original_pts, roundtrip_points)
        except Exception as exc:  # pragma: no cover
            results["roundtrip_distance"] = {"error": str(exc)}
        try:
            results["forward_backward_disagreement"] = forward_backward_disagreement(
                roundtrip_points, original_pts
            )
        except Exception as exc:  # pragma: no cover
            results["forward_backward_disagreement"] = {"error": str(exc)}

    return results
