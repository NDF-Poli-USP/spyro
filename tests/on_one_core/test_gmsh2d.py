import math
from spyro.meshing.meshing_parameters import MeshingParameters
from spyro.meshing.meshing_functions import AutomaticMesh


def test_gmsh2d():
    avenir_params = {
        "mesh_type": "gmsh_mesh",  # Type of automatic mesh
        "dimension": 2,  # Dimension of the mesh

        # Dimensions
        "length_z": 7760.0,
        "length_x": 32040.0,
        "output_filename": "avenir.msh",

        # SEGY / Seismic constraints
        "velocity_model": "tests/inputfiles/velocity_models/avenir.segy",  # Velocity model segy file
        "cells_per_wavelength": 2.0,
        "source_frequency": 3.0,

        # Padding Parameters
        "padding_type": "rectangular",  # Padding types "rectangular" "hyperelliptical" None
        "padding_x": 3000.0,  # Padding size in x direction
        "padding_z": 3000.0,  # Padding size in z direction
        "hyper_n": 3.0,  # Hyperellipse exponent
        "hmin_segy": 0.0,  # Minimum Element size for segy, will apply if higher than function minimum
        "grade": 0.75,  # function grading for smooth element transition, None = no smooth, 0.9 = small smooth, 0.1 = high smooth

        # Water Interface
        "water_interface": True,  # If True detect and implement water interface
        "water_search_value": 0.0,  # If enabled water interface, search for this water value to make the interface
        "vp_water": 500.0,  # Substitute Water speed for this value if vs = 0.0

        # Structured Mesh & Winslow Smoothing
        "structured_mesh": True,  # True if structured quad mesh, False if triangular unstructured mesh
        "min_element_size": 150.0,  # Element size for structured mesh
        "apply_winslow": True,  # If True apply winslow smoothing
        "winslow_implementation": "fast",  # Winslow version to use, default, fast and numba are options
        "winslow_iterations": 100,  # Number of iterations for Winslow Smoothing
        "winslow_omega": 0.5,  # Winslow Smoothing node movement factor
        "extend_segy": False,  # Extend the segy function into the padding ( for unstructured mesh )
        "h_padding": 500.0  # If extend_segy = False, use this value of constant padding size
    }

    mesh_params = MeshingParameters(input_mesh_dictionary=avenir_params, velocity_model=avenir_params["velocity_model"])

    mesh_generator = AutomaticMesh(mesh_parameters=mesh_params)

    print("Starting Gmsh mesh generation...")
    firedrake_mesh = mesh_generator.create_mesh()

    if not hasattr(firedrake_mesh.topology, "_entity_classes") and hasattr(firedrake_mesh, "init"):
        firedrake_mesh.init()

    print(firedrake_mesh.cell_set.core_size, firedrake_mesh.topology.num_cells())

    assert math.isclose(firedrake_mesh.cell_set.core_size, 19275)
