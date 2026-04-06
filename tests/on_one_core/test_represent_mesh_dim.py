import pytest
import firedrake as fire
import numpy as np
from spyro.meshing.meshing_functions import AutomaticMesh


class WrapperMeshParameters:
    """Wrapper for mesh parameters."""

    def __init__(self, dimension, quadrilateral):
        self.length_z = 1.
        self.length_x = 1.
        self.length_y = 1.
        self.mesh_type = "firedrake_mesh"
        self.dimension = dimension
        self.quadrilateral = quadrilateral
        self.comm = None
        self.edge_length = 0.1
        self.abc_pad_length = 0.0
        self.periodic = False
        self.cells_per_wavelength = None
        self.source_frequency = None
        self.minimum_velocity = None
        self.velocity_model = None
        self.output_filename = None


def define_function_space(quadrilateral, mesh, ele_family, ele_degree):
    """Create a function space for the mesh."""

    if quadrilateral:  # Q_Elements
        element = fire.FiniteElement("CG", mesh.ufl_cell(),
                                     degree=ele_degree,
                                     variant="spectral")
        V = fire.FunctionSpace(mesh, element)

    else:  # T_Elements
        V = fire.FunctionSpace(mesh, "KMV", ele_degree)

    return V


@pytest.mark.parametrize('quadrilateral, dimension', [
    (False, 2),   # Triangular 2D
    (True, 2),    # Quadrilateral 2D
    (False, 3),   # Tetrahedral 3D
    (True, 3)])   # Hexahedral 3D
def test_representative_mesh_dimensions_2d(quadrilateral, dimension):
    """Test representative_mesh_dimensions for 2D and 3D meshes."""

    mesh_params = WrapperMeshParameters(dimension, quadrilateral)
    automatic_mesh = AutomaticMesh(mesh_parameters=mesh_params)
    n_div = 10
    # Create appropriate mesh and funcion space for dimension
    q = {"overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)}
    if dimension == 2:
        automatic_mesh.mesh = fire.RectangleMesh(
            n_div, n_div, automatic_mesh.length_z, automatic_mesh.length_x,
            quadrilateral=automatic_mesh.quadrilateral,
            distribution_parameters=q)

        automatic_mesh.function_space = define_function_space(
            quadrilateral, automatic_mesh.mesh, "KMV", 4)
    else:
        automatic_mesh.mesh = fire.BoxMesh(
            n_div, n_div, n_div, automatic_mesh.length_z,
            automatic_mesh.length_x, automatic_mesh.length_y,
            distribution_parameters=q)

        automatic_mesh.function_space = define_function_space(
            quadrilateral, automatic_mesh.mesh, "CG", 3)

    # Call the method
    automatic_mesh.representative_mesh_dimensions()

    # Print mesh info and computed values
    print(f"\nMesh Information:")
    print(f"  - Mesh type: {'Quadrilateral' if quadrilateral else 'Triangular'}")
    print(f"  - Number of cells: {automatic_mesh.mesh.num_cells()}")
    print(f"  - Number of vertices: {automatic_mesh.mesh.num_vertices()}")
    print(f"\nComputed representative mesh dimensions:")
    print(f"  - diam_mesh: {automatic_mesh.diam_mesh}")
    print(f"  - lmin (minimum cell diameter): {automatic_mesh.lmin:.8f}")
    print(f"  - lmax (maximum cell diameter): {automatic_mesh.lmax:.8f}")
    print(f"  - alpha (lmax/lmin ratio): {automatic_mesh.alpha:.6f}")
    print(f"  - tol (tolerance): {automatic_mesh.tol:.2e}")

    # Verify attributes are set
    assert hasattr(automatic_mesh, 'diam_mesh'), "diam_mesh attribute not set"
    assert hasattr(automatic_mesh, 'lmin'), "lmin attribute not set"
    assert hasattr(automatic_mesh, 'lmax'), "lmax attribute not set"
    assert hasattr(automatic_mesh, 'alpha'), "alpha attribute not set"
    assert hasattr(automatic_mesh, 'tol'), "tol attribute not set"

    # Verify values are reasonable
    assert automatic_mesh.lmin > 0, f"lmin should be positive, got {automatic_mesh.lmin}"
    assert automatic_mesh.lmax > 0, f"lmax should be positive, got {automatic_mesh.lmax}"
    assert automatic_mesh.lmin <= automatic_mesh.lmax, \
        f"lmin ({automatic_mesh.lmin}) should be <= lmax ({automatic_mesh.lmax})"
    assert automatic_mesh.alpha >= 1., f"alpha should be >= 1, got {automatic_mesh.alpha}"
    assert automatic_mesh.tol > 0, f"tol should be positive, got {automatic_mesh.tol}"
    assert automatic_mesh.tol <= 1e-6, f"tol should be small, got {automatic_mesh.tol}"
    expected_lmin = expected_lmax = 0.1
    assert np.isclose(automatic_mesh.lmin, expected_lmin, rtol=1e-6)
    assert np.isclose(automatic_mesh.lmax, expected_lmax, rtol=1e-6)
