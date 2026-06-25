import firedrake as fire
import math
from importlib.metadata import version
import spyro


def make_minas_cheese_conditional(mesh_z, mesh_x):
    """Create a conditional velocity field with nested geometric regions.

    Constructs a three-region velocity model with a circular region and a
    square region embedded in a background. Used to model complex synthetic
    velocity structures for mesh adaptation testing.

    Parameters
    ----------
    mesh_z : firedrake.Expression
        The z-coordinate (depth) component of the mesh spatial coordinates.
    mesh_x : firedrake.Expression
        The x-coordinate (horizontal) component of the mesh spatial coordinates.

    Returns
    -------
    firedrake.conditional
        A conditional expression defining velocity values based on spatial
        location. Velocity is 3.0 inside the square, 2.0 inside the circle,
        and 1.5 outside.

    Notes
    -----
    Regions are defined as:
    - Circle: center at (-1.0, 1.0), radius 0.5, velocity 2.0
    - Square: centered at (z=-1.0, x=1.0), dimensions 0.2 x 0.2, velocity 3.0
    - Outside: velocity 1.5

    This creates a "minas cheese" structure - a square embedded within a
    circular region, representing a meia-cura minas cheese with gold hidden
    inside, like Brasilian settlers used to do to smuggle gold during Portugal's
    imperial reign.
    """
    outside_vp = 1.5
    circle_vp = 2.0
    square_vp = 3.0
    r_c = 0.5
    center_z = -1.0
    center_x = 1.0
    square_top_z = -0.9
    square_bot_z = -1.1
    square_left_x = 0.9
    square_right_x = 1.1
    cond = fire.conditional((mesh_z-center_z)**2 + (mesh_x-center_x)**2 < r_c**2, circle_vp, outside_vp)
    cond = fire.conditional(
        fire.And(
            fire.And(mesh_z < square_top_z, mesh_z > square_bot_z),
            fire.And(mesh_x > square_left_x, mesh_x < square_right_x)
        ),
        square_vp,
        cond,
    )
    return cond


def create_grid_based_velocity_field(grid_spacing, length_z, length_x):
    """Create a grid-based velocity field from a conditional expression.

    Generates a regular grid of velocity values by evaluating a conditional
    velocity model over a Firedrake mesh and converting to a structured grid.

    Parameters
    ----------
    grid_spacing : float
        The regular spacing between grid points in both dimensions.
    length_z : float
        The depth extent of the domain in the z-direction.
    length_x : float
        The width extent of the domain in the x-direction.

    Returns
    -------
    tuple
        A tuple containing:
        - grid_based_vp : array-like
            Regular grid of velocity values (structured grid format).
        - cond : firedrake.Expression
            The conditional expression defining the velocity structure.

    Notes
    -----
    Grid spacing is hardcoded to 0.02 regardless of input parameter.
    Creates a 2D domain with no y-extent (2D acoustic simulation).
    """
    grid_spacing = 0.02

    dictionary = {
        "length_z": length_z,
        "length_x": length_x,
        "length_y": 0.0,
        "mesh_type": "firedrake_mesh",
        "edge_length": grid_spacing,
        "dimension": 2,
    }

    mesh_params = spyro.meshing.MeshingParameters(input_mesh_dictionary=dictionary)
    mesh_generator = spyro.meshing.AutomaticMesh(mesh_parameters=mesh_params)
    mesh = mesh_generator.create_mesh()

    mesh_z, mesh_x = fire.SpatialCoordinate(mesh)
    cond = make_minas_cheese_conditional(mesh_z, mesh_x)

    return spyro.utils.scalar_conditional_to_grid(
        conditional=cond,
        domain_dimensions=(length_z, length_x),
        grid_spacing=grid_spacing,
    ), cond


def test_gmsh_adaptation():
    import gmsh  # noqa: F401
    """Test Gmsh mesh adaptation with conditional velocity field.

    Validates that Gmsh mesh adaptation correctly adjusts element sizes
    based on local velocity variations in a synthetic velocity model.
    Performs element size verification at three locations: within a
    high-velocity square, within a medium-velocity circle, and in a
    low-velocity background region.

    Returns
    -------
    None
        Implicitly returns None. Passes if all assertions succeed.

    Raises
    ------
    AssertionError
        If element diameters deviate from expected values based on
        wavelength and cells-per-wavelength requirements.

    Notes
    -----
    Test configuration:
    - 2D domain: 2.0 km (depth) x 2.5 km (width)
    - Cells per wavelength: 2.6
    - Frequency: 20 Hz
    - Time step: 0.001 s
    - Polynomial degree: 4
    - Winslow smoothing enabled with 3000 iterations

    Expected wavelengths (lambda = vp/frequency):
    - Center (square, vp=3.0): lambda = 0.15 km → h = 0.0577 km
    - Circle (vp=2.0): lambda = 0.1 km → h = 0.0385 km
    - Outside (vp=1.5): lambda = 0.075 km → h = 0.0288 km

    The test uses 10% relative tolerance for central and circle regions,
    due to potentional grading and smoothing effects and 1% for outside
    region.
    """
    quadrilateral = False
    mesh_type = "gmsh_mesh"
    cells_per_wavelength = 2.6
    dt = 0.001
    length_z = 2.0
    length_x = 2.5
    vp_grid, cond = create_grid_based_velocity_field(0.02, length_z, length_x)

    if not quadrilateral:
        cell_type = "T"
    else:
        cell_type = "Q"

    dictionary = {}
    dictionary["options"] = {
        "cell_type": cell_type,  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "variant": "lumped",  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
        "degree": 4,  # p order
        "dimension": 2,  # dimension
    }
    dictionary["parallelism"] = {
        "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
    }
    # Define the domain size without the layer.
    dictionary["mesh"] = {
        "length_z": length_z,  # depth in km - always positive
        "length_x": length_x,  # width in km - always positive
        "length_y": 0.0,  # thickness in km - always positive
        "mesh_type": mesh_type,
        "velocity_model": vp_grid,
        "cells_per_wavelength": cells_per_wavelength,
        "padding_type": None,  # Padding types "rectangular" "hyperelliptical" None
        "hmin_segy": 0.0,  # Minimum Element size for segy, will apply if higher than function minimum
        "grade": None,  # function grading for smooth element transition, None = no smooth, 0.9 = high smooth, 0.1 = low smooth

        # Water Interface
        "water_interface": False,  # If True detect and implement water interface

        # Structured Mesh & Winslow Smoothing
        "structured_mesh": False,  # True if structured quad mesh, False if triangular unstructured mesh
        "min_element_size": 0.01,  # Element size for structured mesh
        "apply_winslow": True,  # If True apply winslow smoothing
        "winslow_implementation": "fast",  # Winslow version to use, default, fast and numba are options
        "winslow_iterations": 3000,  # Number of iterations for Winslow Smoothing
        "winslow_omega": 0.5,  # Winslow Smoothing node movement factor
        "extend_segy": False,  # Extend the segy function into the padding ( for unstructured mesh )
        "h_padding": 0.5,  # If extend_segy = False, use this value of constant padding size
    }
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.2, length_x/2.0)],
        "frequency": 20.0,
        "delay": 0.3,
        "receiver_locations": [(-0.5, length_x/2.0)],
        "delay_type": "time",
        "use_vertex_only_mesh": True,
    }
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": 1.5,  # Final time for event
        "dt": dt,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM
    }
    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
    }

    wave_obj = spyro.AcousticWave(dictionary=dictionary)
    wave_obj.set_initial_velocity_model(conditional=cond)
    spyro.plots.debug_pvd(wave_obj.initial_velocity_model)
    # wave_obj.forward_solve()

    DG0 = fire.FunctionSpace(wave_obj.mesh, "DG", 0)
    f = fire.assemble(fire.interpolate(fire.CellSize(wave_obj.mesh), DG0))

    # at centre of minas cheese (gold square):
    lbda_centre = 3.0/20.0  # vp/freq
    h_centre = lbda_centre/cells_per_wavelength
    cell_diam_centre = float(f.at(-1.0, 1.0))
    print(f'Using Firedrake version {version("firedrake")}')
    print(f'Using gmsh version {version("gmsh")}')
    print(f"Cell diameter should be {h_centre}, and is measured at {cell_diam_centre}")
    assert math.isclose(h_centre, cell_diam_centre, rel_tol=1e-15)

    # at circle of minas cheese (meia cura cheesy goodness):
    lbda_circle = 2.0/20.0  # vp/freq
    h_circle = lbda_circle/cells_per_wavelength
    cell_diam_circle = float(f.at(-1.0, 1.3))
    assert math.isclose(h_circle, cell_diam_circle, rel_tol=1e-15)

    # outside the cheese
    lbda_outside = 1.5/20.0
    h_outside = lbda_outside/cells_per_wavelength
    cell_diam_outside = float(f.at(-1.0, 1.7))
    assert math.isclose(h_outside, cell_diam_outside, rel_tol=1e-25)

    print("END")


if __name__ == "__main__":
    test_gmsh_adaptation()
