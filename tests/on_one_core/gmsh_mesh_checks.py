"""Shared helpers for the Gmsh meshing tests.

The checks work on the Firedrake mesh that `AutomaticMesh` returns and on the
physical groups stored in the `.msh` file. Coordinates follow the spyro
convention: `(z, x)` in 2D and `(z, x, y)` in 3D, with `z <= 0`.
"""

from pathlib import Path

import firedrake as fire
import gmsh
import numpy as np

from spyro.abc.hyp_lay import HyperLayer
from spyro.meshing.meshing_functions import AutomaticMesh
from spyro.meshing.meshing_parameters import MeshingParameters


VELOCITY_MODELS_DIR = Path(__file__).resolve().parents[2] / "velocity_models"


def create_gmsh_mesh(parameters: dict) -> fire.MeshGeometry:
    """Build a mesh through `MeshingParameters` and `AutomaticMesh`.

    Parameters
    ----------
    parameters : dict
        Input mesh dictionary for `MeshingParameters`.

    Returns
    -------
    firedrake.MeshGeometry
        The generated mesh.
    """
    mesh_parameters = MeshingParameters(input_mesh_dictionary=parameters)
    return AutomaticMesh(mesh_parameters=mesh_parameters).create_mesh()


def physical_groups(msh_file: str | Path) -> dict[int, dict[str, int]]:
    """Read the physical groups stored in a `.msh` file.

    Parameters
    ----------
    msh_file : str or pathlib.Path
        Mesh file written by Gmsh.

    Returns
    -------
    dict of int to dict of str to int
        For each dimension, a map from physical-group name to tag.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(str(msh_file))
        groups = {}
        for dim, tag in gmsh.model.getPhysicalGroups():
            groups.setdefault(dim, {})[gmsh.model.getPhysicalName(dim, tag)] = tag
        return groups
    finally:
        gmsh.finalize()


def assert_cell_count(
    mesh: fire.MeshGeometry, expected: int, rtol: float = 0.0
) -> None:
    """Assert that the number of cells matches a baseline.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Mesh to check.
    expected : int
        Baseline number of cells.
    rtol : float, optional
        Allowed relative deviation. `0.0` requires an exact match.

    Raises
    ------
    AssertionError
        If the cell count is outside the allowed interval.
    """
    actual = mesh.num_cells()
    allowed = int(np.ceil(rtol * expected))
    assert abs(actual - expected) <= allowed, (
        f"Got {actual} cells, expected {expected} +/- {allowed} "
        f"(rtol={rtol})."
    )


def assert_cell_type(mesh: fire.MeshGeometry, cellname: str) -> None:
    """Assert the UFL cell type of the mesh.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Mesh to check.
    cellname : str
        Expected UFL cell name, for example `"triangle"` or `"hexahedron"`.

    Raises
    ------
    AssertionError
        If the mesh has a different cell type.
    """
    actual = mesh.ufl_cell().cellname
    if callable(actual):
        actual = actual()
    assert actual == cellname


def assert_bounds(
    mesh: fire.MeshGeometry,
    lower: tuple[float, ...],
    upper: tuple[float, ...],
    rtol: float = 1e-6,
) -> None:
    """Assert the bounding box of the mesh coordinates.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Mesh to check.
    lower : tuple of float
        Expected minimum coordinate along each axis, in spyro order.
    upper : tuple of float
        Expected maximum coordinate along each axis, in spyro order.
    rtol : float, optional
        Tolerance relative to the largest extent of the expected box.

    Raises
    ------
    AssertionError
        If the bounding box differs from the expected one.
    """
    coordinates = mesh.coordinates.dat.data_ro
    atol = rtol * float(np.max(np.subtract(upper, lower)))
    np.testing.assert_allclose(coordinates.min(axis=0), lower, rtol=0, atol=atol)
    np.testing.assert_allclose(coordinates.max(axis=0), upper, rtol=0, atol=atol)


def measure(mesh: fire.MeshGeometry, subdomain_id: int | str = "everywhere") -> float:
    """Integrate one over the mesh or over one tagged cell region.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Mesh to integrate over.
    subdomain_id : int or str, optional
        Cell physical-group tag, or `"everywhere"` for the whole mesh.

    Returns
    -------
    float
        Area in 2D or volume in 3D.
    """
    return fire.assemble(fire.Constant(1.0) * fire.dx(subdomain_id, domain=mesh))


def assert_measure(
    mesh: fire.MeshGeometry, expected: float, rtol: float = 1e-9
) -> None:
    """Assert the total area or volume of the mesh.

    Firedrake integrates the absolute Jacobian determinant, so folded or
    overlapping cells make the total larger than the exact domain measure.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Mesh to check.
    expected : float
        Expected area (2D) or volume (3D).
    rtol : float, optional
        Allowed relative deviation.

    Raises
    ------
    AssertionError
        If the measure differs from `expected`.
    """
    actual = measure(mesh)
    assert np.isclose(actual, expected, rtol=rtol, atol=0.0), (
        f"Mesh measure is {actual}, expected {expected} (rtol={rtol})."
    )


def assert_no_degenerate_cells(mesh: fire.MeshGeometry) -> None:
    """Assert that no cell has a vanishing Jacobian at its centre.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Mesh to check.

    Raises
    ------
    AssertionError
        If any cell has a Jacobian determinant that is effectively zero.
    """
    V = fire.FunctionSpace(mesh, "DG", 0)
    det_j = np.abs(
        fire.Function(V).interpolate(fire.JacobianDeterminant(mesh)).dat.data_ro
    )
    assert det_j.min() > 1e-8 * det_j.max(), (
        f"Degenerate cell: |det J| ranges from {det_j.min()} to {det_j.max()}."
    )


def assert_cell_regions(
    mesh: fire.MeshGeometry,
    cell_groups: dict[str, int],
    expected_measures: dict[tuple[str, ...], float],
    rtol: float = 1e-9,
) -> None:
    """Assert the measure of the cell physical groups.

    The groups named in `expected_measures` must be exactly the groups that
    Firedrake sees as cell markers, and together they must cover the mesh
    exactly once.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Mesh to check.
    cell_groups : dict of str to int
        Map from cell physical-group name to tag.
    expected_measures : dict of tuple of str to float
        Expected combined area or volume of each tuple of group names, for
        example `{("Water", "Subsurface"): box, ("Padding",): padding}`.
    rtol : float, optional
        Allowed relative deviation for each combined measure.

    Raises
    ------
    AssertionError
        If the groups differ, have the wrong measure or do not cover the mesh.
    """
    names = {name for group in expected_measures for name in group}
    assert names <= set(cell_groups), (
        f"Missing cell groups {sorted(names - set(cell_groups))}."
    )
    total = 0.0
    for group, expected in expected_measures.items():
        actual = sum(measure(mesh, cell_groups[name]) for name in group)
        assert np.isclose(actual, expected, rtol=rtol, atol=0.0), (
            f"Regions {group} measure {actual}, expected {expected}."
        )
        total += actual
    assert np.isclose(total, measure(mesh), rtol=1e-12, atol=0.0), (
        "The cell physical groups do not cover the mesh exactly once."
    )


def assert_region_above(
    mesh: fire.MeshGeometry, upper_tag: int, lower_tag: int
) -> None:
    """Assert that one cell region lies above another on average.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Mesh to check.
    upper_tag : int
        Cell tag of the region expected to be shallower.
    lower_tag : int
        Cell tag of the region expected to be deeper.

    Raises
    ------
    AssertionError
        If the mean depth of `upper_tag` is not above that of `lower_tag`.
    """
    z = fire.SpatialCoordinate(mesh)[0]

    def mean_z(tag: int) -> float:
        return fire.assemble(z * fire.dx(tag, domain=mesh)) / measure(mesh, tag)

    assert mean_z(upper_tag) > mean_z(lower_tag)


def hyperellipse_area(
    length_z: float, length_x: float, padding_z: float, padding_x: float, n: float
) -> float:
    """Area of the 2D hyperelliptical domain cut at `z = 0`.

    The hyperellipse is centred on the model box and its semi-axes reach
    `padding_x` beyond each side and `padding_z` below the bottom.

    Parameters
    ----------
    length_z : float
        Model depth.
    length_x : float
        Model width.
    padding_z : float
        Padding below the model.
    padding_x : float
        Padding on each side of the model.
    n : float
        Hyperellipse exponent.

    Returns
    -------
    float
        Area of the part of the hyperellipse with `z <= 0`.
    """
    semi_z = length_z / 2.0 + padding_z
    semi_x = length_x / 2.0 + padding_x
    # Lower half plus the slab between the centre and the cut at z = 0.
    return float(
        HyperLayer.half_hyp_area(semi_x, semi_z, n)
        + HyperLayer.trunc_half_hyp_area(semi_x, semi_z, n, length_z / 2.0)
    )
