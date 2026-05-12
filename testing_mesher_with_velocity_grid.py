import spyro
import matplotlib.pyplot as plt
from spyro.meshing.meshing_parameters import MeshingParameters
from spyro.meshing.meshing_functions import AutomaticMesh
import firedrake as fire


def make_minas_cheese_conditional(mesh_z, mesh_x):
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


def create_grid_based_velocity_field(grid_spacing):
    grid_spacing = 0.02

    dictionary = {
        "length_z": 2.5,
        "length_x": 2.0,
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
        domain_dimensions=(2.0, 2.0),
        grid_spacing=grid_spacing,
    )

vp_grid = create_grid_based_velocity_field(0.02)

mesh_dictionary = {
    "mesh_type": "gmsh_mesh",  # Type of automatic mesh
    "dimension": 2,  # Dimension of the mesh

    # Dimensions
    "length_z": 2.5,
    "length_x": 2.0,
    "output_filename": "minas_cheese_best.msh",

    # Seismic constraints
    "velocity_model": vp_grid,
    "cells_per_wavelength": 2.0,
    "source_frequency": 15.0,

    # Padding Parameters
    "padding_type": "rectangular",  # Padding types "rectangular" "hyperelliptical" None
    "padding_x": 0.2,  # Padding size in x direction
    "padding_z": 0.2,  # Padding size in z direction
    "hmin_segy": 0.0,  # Minimum Element size for segy, will apply if higher than function minimum
    "grade": 0.1,  # function grading for smooth element transition, None = no smooth, 0.1 = small smooth, 0.9 = high smooth

    # Water Interface
    "water_interface": False,  # If True detect and implement water interface

    # Structured Mesh & Winslow Smoothing
    "structured_mesh": False,  # True if structured quad mesh, False if triangular unstructured mesh
    "min_element_size": 0.05,  # Element size for structured mesh
    "apply_winslow": True,  # If True apply winslow smoothing
    "winslow_implementation": "fast",  # Winslow version to use, default, fast and numba are options
    "winslow_iterations": 300,  # Number of iterations for Winslow Smoothing
    "winslow_omega": 0.5,  # Winslow Smoothing node movement factor
    "extend_segy": True,  # Extend the segy function into the padding ( for unstructured mesh )
    "h_padding": 0.5  # If extend_segy = False, use this value of constant padding size
}

mesh_params = MeshingParameters(input_mesh_dictionary=mesh_dictionary)

mesh_generator = AutomaticMesh(mesh_parameters=mesh_params)

print("Starting Gmsh mesh generation...")
firedrake_mesh = mesh_generator.create_mesh()

if firedrake_mesh is not None:
    print(f"Mesh successfully generated and loaded from {mesh_params.output_filename}!")

    fig, axes = plt.subplots(figsize=(60, 60))
    fire.triplot(firedrake_mesh, axes=axes)
    axes.set_aspect('equal')
    axes.set_title("Marmousi Mesh")
    axes.set_xlabel("Distance X (m)")
    axes.set_ylabel("Depth Z (m)")
    plt.show()


print("END")
