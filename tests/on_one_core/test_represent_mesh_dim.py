import pytest
import warnings
import firedrake as fire
from numpy import isclose
from spyro.habc.habc import HABC_Wave
import spyro.meshing.meshing_operations as mshops
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}
warnings.filterwarnings("ignore", category=RuntimeWarning)


def wave_dict(element_type, dimension):
    '''
    Create a dictionary with parameters for the model

    Parameters
    ----------
    element_type : `str`
        Type of finite element. 'T' for triangles or 'Q' for quadrilaterals
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D

    Returns
    -------
    dictionary : `dict`
        Dictionary containing the parameters for the model
    '''

    dictionary = {}
    dictionary["options"] = {
        # Simplexes: triangles or tetrahedra (T) or quadrilaterals (Q)
        "cell_type": element_type,
        "variant": "lumped",  # Options: lumped, equispaced or DG.
        # Default is lumped "method":"MLT"
        # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral)
        # You can either specify a cell_type+variant or a method
        # accepted_variants = ["lumped", "equispaced", "DG"]
        "degree": 4 if dimension == 2 else 3,  # p<=4 for 2D and p<=3 for 3D
        "dimension": dimension,  # dimension
    }

    # Number of cores for the shot. For simplicity, we keep things serial.
    # spyro however supports both spatial parallelism and "shot" parallelism.
    # Options: automatic (same number of cores for evey processor) or spatial
    dictionary["parallelism"] = {
        "type": "automatic",
    }

    # Define the domain size without the PML or AL.
    if dimension == 2:
        Lz, Lx, Ly = [1., 1., 0.]
    elif dimension == 3:
        Lz, Lx, Ly = [1., 1., 1.]  # in km
    dictionary["mesh"] = {
        "length_z": Lz,  # depth in km - always positive
        "length_x": Lx,  # width in km - always positive
        "length_y": Ly,  # thickness in km - always positive
        "mesh_type": "firedrake_mesh",
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at a specified
    # point of the mesh. We also specify to record the solution at the corners
    # of the domain to verify the efficiency of the absorbing layer.
    fn = 1. if negative_z else -1.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": ([(-fn * 0.5, 0.25)] if dimension == 2
                             else [(-fn * 0.5, 0.25, 0.5)]),
        "frequency": 5.,  # in Hz
        "delay": 1.5,
        "receiver_locations": ([(-fn * Lz, 0.), (-fn * Lz, Lx), (0., 0.), (0., Lx)]
                               if dimension == 2
                               else [(-fn * Lz, 0., 0.), (-fn * Lz, Lx, 0.),
                                     (0., 0., 0), (0., Lx, 0.),
                                     (-fn * Lz, 0., Ly), (-fn * Lz, Lx, Ly),
                                     (0., 0., Ly), (0., Lx, Ly)])
    }

    # Simulate for 1. seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.,  # Initial time for event
        "final_time": 2.,    # Final time for event
        "dt": 0.001,  # timestep size in seconds
        "amplitude": 1.,  # The Ricker has an amplitude of 1.
        "output_frequency": 100,  # How frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # How frequently to save to RAM
    }

    # Define Parameters for absorbing boundary conditions
    dictionary["absorving_boundary_conditions"] = {
        "status": False,  # Activate ABCs
        "damping_type": "hybrid",  # Activate HABC
    }

    # Define parameters for visualization
    dictionary["visualization"] = {}

    return dictionary


@pytest.mark.parametrize('element_type, dimension', [
    ("T", 2),   # Triangular 2D
    ("Q", 2),    # Quadrilateral 2D
    ("T", 3),   # Tetrahedral 3D
    ("Q", 3)])   # Hexahedral 3D
def test_representative_mesh_dimensions(element_type, dimension):
    """Test representative_mesh_dimensions for 2D and 3D meshes."""

    # Create dictionary with parameters for the model
    dictionary = wave_dict(element_type, dimension)

    # Create the acoustic wave object
    Wave_obj = HABC_Wave(dictionary=dictionary)

    # Mesh the domain with the specified edge length
    edge_length = 0.1
    Wave_obj.set_mesh(input_mesh_parameters={"edge_length": edge_length})

    # Call the method and assign the outputs to the mesh_parameters object
    mesh_derived_parameters = \
        Wave_obj.mesh_ops.representative_mesh_dimensions(Wave_obj.mesh,
                                                         Wave_obj.function_space)
    Wave_obj.mesh_parameters.diam_mesh = mesh_derived_parameters[0]
    Wave_obj.mesh_parameters.lmin = mesh_derived_parameters[1]
    Wave_obj.mesh_parameters.lmax = mesh_derived_parameters[2]
    Wave_obj.mesh_parameters.alpha = mesh_derived_parameters[3]
    Wave_obj.mesh_parameters.tol = mesh_derived_parameters[4]

    # Print mesh info and computed values
    print("\nMesh Information:")
    print(f"  - Mesh type: {element_type}")
    print(f"  - Number of cells: {Wave_obj.mesh.num_cells()}")
    print(f"  - Number of vertices: {Wave_obj.mesh.num_vertices()}")
    print("\nComputed representative mesh dimensions:")
    print(f"  - lmin (minimum mesh size): {Wave_obj.mesh_parameters.lmin:.6f}")
    print(f"  - lmax (maximum mesh size): {Wave_obj.mesh_parameters.lmax:.6f}")
    print(f"  - alpha (lmax/lmin ratio): {Wave_obj.mesh_parameters.alpha:.6f}")
    print(f"  - tol (tolerance): {Wave_obj.mesh_parameters.tol:.2e}")

    # Verify attributes are set
    assert hasattr(Wave_obj.mesh_parameters, 'diam_mesh'), "'diam_mesh' not set"
    assert hasattr(Wave_obj.mesh_parameters, 'lmin'), "'lmin' not set"
    assert hasattr(Wave_obj.mesh_parameters, 'lmax'), "'lmax' not set"
    assert hasattr(Wave_obj.mesh_parameters, 'alpha'), "'alpha' not set"
    assert hasattr(Wave_obj.mesh_parameters, 'tol'), "'tol' not set"

    # Verify values are reasonable
    assert Wave_obj.mesh_parameters.lmin > 0, \
        f"lmin should be positive, got {Wave_obj.mesh_parameters.lmin}"
    assert Wave_obj.mesh_parameters.lmax > 0, \
        f"lmax should be positive, got {Wave_obj.mesh_parameters.lmax}"
    assert Wave_obj.mesh_parameters.lmin <= Wave_obj.mesh_parameters.lmax, \
        f"lmin ({Wave_obj.mesh_parameters.lmin}) should " + \
        f"be <= lmax ({Wave_obj.mesh_parameters.lmax})"
    assert Wave_obj.mesh_parameters.alpha >= 1., \
        f"alpha should be >= 1, got {Wave_obj.mesh_parameters.alpha}"
    assert Wave_obj.mesh_parameters.tol > 0, \
        f"tol should be positive, got {Wave_obj.mesh_parameters.tol}"
    assert Wave_obj.mesh_parameters.tol <= 1e-6, \
        f"tol should be small, got {Wave_obj.mesh_parameters.tol}"
    expected_lmin = expected_lmax = 0.1
    assert isclose(Wave_obj.mesh_parameters.lmin, expected_lmin, rtol=1e-6)
    assert isclose(Wave_obj.mesh_parameters.lmax, expected_lmax, rtol=1e-6)


@pytest.mark.parametrize('element_type, dimension', [
    ("T", 2),   # Triangular 2D
    ("Q", 2),    # Quadrilateral 2D
    ("T", 3),   # Tetrahedral 3D
    ("Q", 3)])   # Hexahedral 3D
def test_boundary_ids(element_type, dimension):
    """Test representative_mesh_dimensions for 2D and 3D meshes."""

    # Crete object for mesh operations
    domain_dim = (1., 1.) if dimension == 2 else (1., 1., 1.)
    quadrilateral = True if element_type == "Q" else False
    mesh_ops = mshops.MeshOps(domain_dim, dimension=dimension,
                              quadrilateral=quadrilateral,
                              func_space_type='scalar')

    # Assuming Spyro's system coordinates: (z, x, y)
    # Boundaries2D = [top (zmax, 2), bottom(zmin, 1), right(xmax, 4), left(xmin, 3)]
    boundaries = [False, True, False, True]
    expected_map = [True, False, True, False]
    expected_ids = [1, 2, 3, 4]
    if dimension == 3:
        # Boundaries3D = Boundaries2D + [front (ymin, 5), back(ymax, 6)]
        boundaries.extend([True, False])
        expected_map.extend([True, False])
        expected_ids.extend([5, 6]) if element_type == "T" \
            else expected_ids.extend(['bottom', 'top'])

    # Mesh parameters
    degree_ele = 4 if dimension == 2 else 3
    family = "CG" if element_type == "Q" else "KMV"

    # Create mesh
    if quadrilateral:
        mesh = fire.UnitSquareMesh(10, 10, quadrilateral=quadrilateral)
        if dimension == 3:
            quad_mesh = mesh
            layer_height = 1. / 10
            mesh = fire.ExtrudedMesh(quad_mesh, 10, layer_height=layer_height)
        element = fire.FiniteElement(family, mesh.ufl_cell(),
                                     degree=degree_ele, variant="spectral")
        V = fire.FunctionSpace(mesh, element)
    else:
        mesh = fire.UnitSquareMesh(10, 10) if dimension == 2 \
            else fire.UnitCubeMesh(10, 10, 10)
        V = fire.FunctionSpace(mesh, family, degree_ele)

    # Build the boundary ID mapping
    boundary_idx_map = mesh_ops.mapping_boundary_ids(mesh, V,
                                                     boundaries,
                                                     box_domain=True)

    # Checking the mapping
    assert len(boundary_idx_map) == (4 if dimension == 2 else 6)
    for key, expected_value in zip(expected_ids, expected_map):
        assert key in boundary_idx_map, f"✗ Boundary ID {key} not found in the map"
        assert boundary_idx_map[key] == expected_value, \
            f"✗ Boundary ID {key} expected to be {expected_value} " + \
            f"but got {boundary_idx_map[key]}"
    print(f"✓ Boundary ID mapping for {element_type} {dimension}D mesh is correct.",
          flush=True)
