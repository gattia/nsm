"""Per-triangle quality metrics on VTK meshes (areas, edge lengths, aspect ratios).

Importers: ``correspondence_metrics`` and ``refine_mesh``. Two conventions here are
shared with those modules and are easy to violate silently:

- **Edge ordering** is (p0-p1, p1-p2, p2-p0) — the same cyclic order
  ``refine_mesh.new_vertices_faces`` iterates edges in.
- **Failure policy on degenerate triangles diverges deliberately**:
  ``TriangleProperties.edge_ratio`` raises on a zero-length edge, while
  ``correspondence_metrics.triangle_health`` computes the same statistic and degrades
  gracefully (reporting a ``degenerate_count`` instead). See ``docs/SCOPE.md`` §2.6.
"""

import numpy as np
import vtk


def get_triangle_area(cell):
    """Area of one VTK cell; raises on any non-triangle cell type."""
    if cell.GetCellType() == vtk.VTK_TRIANGLE:
        p0 = cell.GetPoints().GetPoint(0)
        p1 = cell.GetPoints().GetPoint(1)
        p2 = cell.GetPoints().GetPoint(2)

        # Compute the area using vtkTriangle
        area = vtk.vtkTriangle.TriangleArea(p0, p1, p2)
    else:
        raise Exception("only support triangle meshes")

    return area


def calculate_triangle_areas(polyData):
    """List of per-cell triangle areas for a vtkPolyData, in cell order."""
    areas = []
    for i in range(polyData.GetNumberOfCells()):
        cell = polyData.GetCell(i)
        area = get_triangle_area(cell)
        areas.append(area)
    return areas


def length(x1, x2):
    """Euclidean distance between two points."""
    return np.sqrt(sum((x1 - x2) ** 2))


def get_edge_lengths(cell):
    """The three edge lengths of a triangle cell, ordered (p0-p1, p1-p2, p2-p0).

    The order is a contract: refine_mesh iterates edges in the same cyclic order,
    so index i here refers to the same edge there.
    """
    p0 = np.asarray(cell.GetPoints().GetPoint(0))
    p1 = np.asarray(cell.GetPoints().GetPoint(1))
    p2 = np.asarray(cell.GetPoints().GetPoint(2))

    edge_lengths = []
    edge_lengths.append(length(p0, p1))
    edge_lengths.append(length(p1, p2))
    edge_lengths.append(length(p2, p0))

    return edge_lengths


class TriangleProperties:
    """Lazily-computed per-triangle metrics for one VTK mesh (results cached)."""

    def __init__(self, mesh):
        self._mesh = mesh
        self.edge_lengths = None
        self._areas = None

    def areas(self, norm=True):
        """Per-triangle areas — but NOT areas at the default.

        With ``norm=True`` (the default) this returns the dimensionless *relative
        deviation from the mean area*, ``(area - mean) / mean`` — negative for
        below-average triangles, zero mean by construction. Only ``norm=False``
        returns actual areas. This is the trap behind refine_mesh's
        ``area_threshold``, whose docstrings call it "the maximum area of a
        triangle": the value is compared against this relative deviation.
        """
        if self._areas is None:
            self._areas = calculate_triangle_areas(self._mesh)

        if norm is True:
            ref_area = np.mean(self._areas)
            areas = (self._areas - ref_area) / ref_area
        else:
            areas = self._areas.copy()

        return np.asarray(areas)

    def compute_edge_lengths(self):
        """Fill self.edge_lengths: (n_cells, 3) array in the module's edge order."""
        self.edge_lengths = []
        for i in range(self._mesh.GetNumberOfCells()):
            cell = self._mesh.GetCell(i)
            lengths = get_edge_lengths(cell)
            self.edge_lengths.append(lengths)

        self.edge_lengths = np.array(self.edge_lengths)

    def edge_ratio(self):
        """Per-triangle aspect ratio (longest edge / shortest edge).

        Raises on any zero-length edge — deliberately stricter than
        ``correspondence_metrics.triangle_health``, which degrades instead
        (see the module docstring).
        """
        if self.edge_lengths is None:
            self.compute_edge_lengths()

        min_ = np.min(self.edge_lengths, axis=1)
        max_ = np.max(self.edge_lengths, axis=1)

        if sum(min_ == 0) > 0:
            zero_area = np.where(min_ == 0)
            raise Exception(f"edge length zero! triangle with zero length edge: {zero_area}")

        lengths_ratio = max_ / min_

        return lengths_ratio

    def edge_sd(self):
        """Per-triangle standard deviation of the three edge lengths."""
        if self.edge_lengths is None:
            self.compute_edge_lengths()

        return np.std(self.edge_lengths, axis=1)

    def edge_length_max(self):
        """Per-triangle longest edge length."""
        if self.edge_lengths is None:
            self.compute_edge_lengths()

        return np.max(self.edge_lengths, axis=1)


def triangle_faces(mesh_or_faces):
    """(M, 3) triangle connectivity, refusing anything that is not triangles.

    The single accessor for #57. Five sites across ``correspondence_metrics``,
    ``interpolate`` and ``refine_mesh`` used to reshape a face array directly, which
    validates nothing: a VTK-style array reshapes into (-1, 4) or (-1, 3) exactly when
    its flat length happens to divide, so a quad mesh raises or silently yields
    fabricated triangles depending on its cell count mod 4.

    ``pyvista.PolyData.regular_faces`` is not the guard on its own -- it returns an
    (M, 4) array for an all-quad mesh rather than refusing -- so the cell type is
    checked first.

    Args:
    - mesh_or_faces: a pyvista mesh, or a face array already in (M, 3) form.

    Returns:
    - np.ndarray: (M, 3) vertex indices, one row per triangle. A view into the input
      where the input allows one, as the reshape it replaces was.

    Raises:
    - ValueError: for a non-triangular mesh, a VTK-style flat array (which is where a
      caller reaching for ``mesh.faces`` lands), or an array of any other width.
    """
    is_all_triangles = getattr(mesh_or_faces, "is_all_triangles", None)
    if is_all_triangles is not None:
        if not is_all_triangles:
            raise ValueError(
                f"expected a triangle mesh; got one with {mesh_or_faces.n_cells} cells "
                "of mixed or non-triangular type. Triangulate it first "
                "(pyvista: mesh.triangulate())."
            )
        return np.asarray(mesh_or_faces.regular_faces)

    faces = np.asarray(mesh_or_faces)
    if faces.ndim == 1:
        raise ValueError(
            f"expected (M, 3) triangle connectivity; got a flat array of length "
            f"{faces.size}, which is the VTK-style [n, i0, ..., in, n, ...] layout. "
            "Pass mesh.regular_faces (or the mesh itself), not mesh.faces."
        )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"expected (M, 3) triangle connectivity; got shape {faces.shape}. "
            "Triangulate the mesh first (pyvista: mesh.triangulate())."
        )
    return faces
