import pytest
import warnings
import firedrake as fire
from numpy import isclose
from spyro.solvers.acoustic_wave import AcousticWave
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
    degree : `int`
        Degree of the finite element. p<=4 for 2D and p<=3 for 3D

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
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": ([(-0.5, 0.25)] if dimension == 2
                             else [(-0.5, 0.25, 0.5)]),
        "frequency": 5.,  # in Hz
        "delay": 1.5,
        "receiver_locations": ([(-Lz, 0.), (-Lz, Lx), (0., 0.), (0., Lx)]
                               if dimension == 2
                               else [(-Lz, 0., 0.), (-Lz, Lx, 0.),
                                     (0., 0., 0), (0., Lx, 0.),
                                     (-Lz, 0., Ly), (-Lz, Lx, Ly),
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
    }

    # Define parameters for visualization
    dictionary["visualization"] = {}

    return dictionary


@pytest.mark.parametrize('element_type, dimension', [
    ("T", 2),   # Triangular 2D
    ("Q", 2),    # Quadrilateral 2D
    ("T", 3),   # Tetrahedral 3D
    ("Q", 3)])   # Hexahedral 3D
def test_representative_mesh_dimensions_2d(element_type, dimension):
    """Test representative_mesh_dimensions for 2D and 3D meshes."""

    # Create dictionary with parameters for the model
    dictionary = wave_dict(element_type, dimension)

    # Create the acoustic wave object
    Wave_obj = AcousticWave(dictionary=dictionary)

    # Mesh the domain with the specified edge length
    edge_length = 0.1
    Wave_obj.set_mesh(input_mesh_parameters={"edge_length": edge_length})

    # Call the method
    Wave_obj.representative_mesh_dimensions()

    # Print mesh info and computed values
    print(f"\nMesh Information:")
    print(f"  - Mesh type: {element_type}")
    print(f"  - Number of cells: {Wave_obj.mesh.num_cells()}")
    print(f"  - Number of vertices: {Wave_obj.mesh.num_vertices()}")
    print(f"\nComputed representative mesh dimensions:")
    print(f"  - diam_mesh: {Wave_obj.diam_mesh}")
    print(f"  - lmin (minimum cell diameter): {Wave_obj.lmin:.8f}")
    print(f"  - lmax (maximum cell diameter): {Wave_obj.lmax:.8f}")
    print(f"  - alpha (lmax/lmin ratio): {Wave_obj.alpha:.6f}")
    print(f"  - tol (tolerance): {Wave_obj.tol:.2e}")

    # Verify attributes are set
    assert hasattr(Wave_obj, 'diam_mesh'), "diam_mesh attribute not set"
    assert hasattr(Wave_obj, 'lmin'), "lmin attribute not set"
    assert hasattr(Wave_obj, 'lmax'), "lmax attribute not set"
    assert hasattr(Wave_obj, 'alpha'), "alpha attribute not set"
    assert hasattr(Wave_obj, 'tol'), "tol attribute not set"

    # Verify values are reasonable
    assert Wave_obj.lmin > 0, f"lmin should be positive, got {Wave_obj.lmin}"
    assert Wave_obj.lmax > 0, f"lmax should be positive, got {Wave_obj.lmax}"
    assert Wave_obj.lmin <= Wave_obj.lmax, \
        f"lmin ({Wave_obj.lmin}) should be <= lmax ({Wave_obj.lmax})"
    assert Wave_obj.alpha >= 1., f"alpha should be >= 1, got {Wave_obj.alpha}"
    assert Wave_obj.tol > 0, f"tol should be positive, got {Wave_obj.tol}"
    assert Wave_obj.tol <= 1e-6, f"tol should be small, got {Wave_obj.tol}"
    expected_lmin = expected_lmax = 0.1
    assert isclose(Wave_obj.lmin, expected_lmin, rtol=1e-6)
    assert isclose(Wave_obj.lmax, expected_lmax, rtol=1e-6)
