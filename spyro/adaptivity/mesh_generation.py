"""Triangular meshes of a padded rectangle from a mesh-size function.

The meshes follow spyro's convention: the first coordinate is the depth ``z``
(negative downwards, ``z = 0`` at the surface) and the second is ``x``.
Boundary markers match :func:`spyro.meshing.rectangle_mesh`:

``1``
    top, ``z = z_max``;
``2``
    bottom, ``z = z_min``;
``3``
    ``x = x_min``;
``4``
    ``x = x_max``.

Horizontal interfaces (for instance the sea floor) can be imposed as mesh
edges, so that the velocity jump is not smeared across a layer of elements.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

try:
    import gmsh
except ImportError:  # pragma: no cover - gmsh is optional for spyro
    gmsh = None

#: Boundary markers, in spyro's convention.
TOP, BOTTOM, X_MIN, X_MAX = 1, 2, 3, 4


@dataclass(frozen=True)
class RectangularDomain:
    """Padded rectangular domain in spyro's ``(z, x)`` coordinates.

    The physical region of interest is
    ``[z_min + pad, z_max] x [x_min + pad, x_max - pad]``: the pad is the
    absorbing layer at the bottom and on both sides. The top is not padded.

    Parameters
    ----------
    z_min : float
        Depth of the bottom of the padded domain (negative, km).
    x_min : float
        Left end of the padded domain (km).
    x_max : float
        Right end of the padded domain (km).
    z_max : float, optional
        Top of the domain (km). Default is ``0``.
    pad : float, optional
        Width of the absorbing pad (km). Default is ``0``.
    """

    z_min: float
    x_min: float
    x_max: float
    z_max: float = 0.0
    pad: float = 0.0

    def __post_init__(self) -> None:
        """Validate the extents.

        Raises
        ------
        ValueError
            If the domain is empty or the pad leaves no region of interest.
        """
        if not (self.z_min < self.z_max and self.x_min < self.x_max):
            raise ValueError("The domain must have a positive extent.")
        if self.pad < 0 or 2 * self.pad >= self.x_max - self.x_min \
                or self.pad >= self.z_max - self.z_min:
            raise ValueError("The pad must be non-negative and leave an interior.")

    def in_interior(self, z: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Tell which points lie in the region of interest (outside the pad).

        Parameters
        ----------
        z : numpy.ndarray
            Depths of the points.
        x : numpy.ndarray
            Horizontal positions of the points.

        Returns
        -------
        numpy.ndarray
            Boolean array, ``True`` inside the region of interest.
        """
        z = np.asarray(z)
        x = np.asarray(x)
        return ((z >= self.z_min + self.pad) & (x >= self.x_min + self.pad)
                & (x <= self.x_max - self.pad))

    def grid(self, spacing: float) -> tuple[np.ndarray, np.ndarray]:
        """Regular grid covering the padded domain.

        Parameters
        ----------
        spacing : float
            Grid spacing (km).

        Returns
        -------
        tuple of numpy.ndarray
            One-dimensional ``z`` and ``x`` coordinates, both increasing.
        """
        z = np.arange(self.z_min, self.z_max + spacing / 2, spacing)
        x = np.arange(self.x_min, self.x_max + spacing / 2, spacing)
        return z, x


def generate_mesh(
    path: str | Path,
    domain: RectangularDomain,
    size: Callable[[float, float], float],
    conforming_depths: Sequence[float] = (),
    seed: int = 0,
    algorithm: int = 6,
) -> Path:
    """Generate a triangular mesh with gmsh and write it to a file.

    This runs gmsh in the calling process only; under MPI call it on a single
    rank and share the file name.

    Parameters
    ----------
    path : str or pathlib.Path
        Output ``.msh`` file (gmsh format 2.2).
    domain : RectangularDomain
        Domain to mesh.
    size : callable
        Target edge length ``size(z, x)`` in km.
    conforming_depths : sequence of float, optional
        Depths of horizontal interfaces imposed as mesh edges.
    seed : int, optional
        Seed of gmsh's random perturbations, for replicas of the same design.
    algorithm : int, optional
        gmsh 2D algorithm. Default ``6`` (Frontal-Delaunay).

    Returns
    -------
    pathlib.Path
        The written mesh file.

    Raises
    ------
    ImportError
        If gmsh is not installed.
    ValueError
        If an interface depth is outside the domain.
    """
    if gmsh is None:
        raise ImportError("gmsh is required to generate adaptive meshes.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for depth in conforming_depths:
        if not domain.z_min < depth < domain.z_max:
            raise ValueError(f"Interface depth {depth} is outside the domain.")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(path.stem)
        occ = gmsh.model.occ
        # gmsh's first coordinate is spyro's z, its second is x.
        surface = occ.addRectangle(domain.z_min, domain.x_min, 0.0,
                                   domain.z_max - domain.z_min,
                                   domain.x_max - domain.x_min)
        cuts = []
        for depth in conforming_depths:
            p1 = occ.addPoint(depth, domain.x_min, 0.0)
            p2 = occ.addPoint(depth, domain.x_max, 0.0)
            cuts.append((1, occ.addLine(p1, p2)))
        if cuts:
            occ.fragment([(2, surface)], cuts)
        occ.synchronize()
        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]

        markers = {TOP: [], BOTTOM: [], X_MIN: [], X_MAX: []}
        boundary = gmsh.model.getBoundary([(2, s) for s in surfaces],
                                          oriented=False, combined=True)
        for dim, tag in boundary:
            z, x, _ = occ.getCenterOfMass(dim, abs(tag))
            if abs(z - domain.z_max) < 1e-9:
                markers[TOP].append(abs(tag))
            elif abs(z - domain.z_min) < 1e-9:
                markers[BOTTOM].append(abs(tag))
            elif abs(x - domain.x_min) < 1e-9:
                markers[X_MIN].append(abs(tag))
            else:
                markers[X_MAX].append(abs(tag))
        for marker, curves in markers.items():
            gmsh.model.addPhysicalGroup(1, curves, marker)
        gmsh.model.addPhysicalGroup(2, surfaces, 1)

        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Algorithm", algorithm)
        gmsh.option.setNumber("Mesh.Smoothing", 10)
        gmsh.option.setNumber("Mesh.RandomSeed", seed + 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.mesh.setSizeCallback(
            lambda dim, tag, z, x, y, lc: float(size(z, x)))
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def read_triangulation(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the vertices and triangles of a mesh file.

    Parameters
    ----------
    path : str or pathlib.Path
        A gmsh mesh file.

    Returns
    -------
    points : numpy.ndarray
        Vertex coordinates ``(z, x)``, shape ``(n_vertices, 2)``.
    triangles : numpy.ndarray
        Zero-based vertex indices, shape ``(n_triangles, 3)``.

    Raises
    ------
    ImportError
        If gmsh is not installed.
    """
    if gmsh is None:
        raise ImportError("gmsh is required to read meshes.")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        _, node_tags = gmsh.model.mesh.getElementsByType(2)
    finally:
        gmsh.finalize()
    index = np.zeros(int(tags.max()) + 1, dtype=np.int64)
    index[tags.astype(np.int64)] = np.arange(len(tags))
    points = coordinates.reshape(-1, 3)[:, :2]
    triangles = index[node_tags.astype(np.int64)].reshape(-1, 3)
    return points, triangles


def mean_edge_lengths(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Mean edge length of each triangle.

    This is the convention of the target size handed to gmsh, so a mesh
    built from ``size`` has ``mean_edge_lengths`` close to ``size``.

    Parameters
    ----------
    points : numpy.ndarray
        Vertex coordinates, shape ``(n_vertices, 2)``.
    triangles : numpy.ndarray
        Vertex indices, shape ``(n_triangles, 3)``.

    Returns
    -------
    numpy.ndarray
        One length per triangle.
    """
    q = points[triangles]
    return np.linalg.norm(q[:, [1, 2, 0]] - q, axis=2).mean(axis=1)


def local_mesh_size(points: np.ndarray, triangles: np.ndarray,
                    z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sample the local element size of a mesh on a regular grid.

    Each grid point takes the mean edge length of the triangle that contains
    it; points that fall on no triangle take the value of the nearest point
    that does.

    Parameters
    ----------
    points : numpy.ndarray
        Vertex coordinates ``(z, x)``.
    triangles : numpy.ndarray
        Vertex indices.
    z : numpy.ndarray
        Increasing grid depths, shape ``(nz,)``.
    x : numpy.ndarray
        Increasing grid positions, shape ``(nx,)``.

    Returns
    -------
    numpy.ndarray
        Element size on the grid, shape ``(nz, nx)``.
    """
    import matplotlib.tri as mtri
    from scipy.ndimage import distance_transform_edt

    sizes = mean_edge_lengths(points, triangles)
    Z, X = np.meshgrid(z, x, indexing="ij")
    finder = mtri.Triangulation(points[:, 0], points[:, 1],
                                triangles).get_trifinder()
    cell = finder(Z, X)
    h = np.where(cell >= 0, sizes[np.maximum(cell, 0)], np.nan)
    missing = np.isnan(h)
    if missing.any():
        nearest = distance_transform_edt(missing, return_distances=False,
                                         return_indices=True)
        h = h[tuple(nearest)]
    return h


@cache
def _kmv_entity_dofs(degree: int) -> tuple[int, int, int]:
    """Number of KMV degrees of freedom per vertex, edge and triangle.

    Parameters
    ----------
    degree : int
        Polynomial degree of the KMV element.

    Returns
    -------
    tuple of int
        Degrees of freedom on each vertex, each edge and each cell interior.
    """
    import FIAT

    element = FIAT.KongMulderVeldhuizen(FIAT.ufc_simplex(2), degree)
    entity_dofs = element.entity_dofs()
    return tuple(len(entity_dofs[dim][0]) for dim in range(3))


def count_kmv_dofs(triangles: np.ndarray, degree: int) -> int:
    """Count the KMV degrees of freedom of a triangulation.

    Parameters
    ----------
    triangles : numpy.ndarray
        Vertex indices, shape ``(n_triangles, 3)``.
    degree : int
        Polynomial degree of the KMV element.

    Returns
    -------
    int
        Dimension of the scalar KMV space on the mesh.
    """
    per_vertex, per_edge, per_cell = _kmv_entity_dofs(degree)
    edges = np.sort(triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    n_edges = len(np.unique(edges, axis=0))
    n_vertices = len(np.unique(triangles))
    return int(per_vertex * n_vertices + per_edge * n_edges
               + per_cell * len(triangles))
