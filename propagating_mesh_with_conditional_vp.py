import firedrake as fire
import spyro


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


def create_grid_based_velocity_field(grid_spacing, length_z, length_x):
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
        domain_dimensions=(2.0, 2.0),
        grid_spacing=grid_spacing,
    ), cond


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
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.2, length_x/2.0)],
    "frequency": 5.0,
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
wave_obj.set_initial_velocity_model(conditional=make_minas_cheese_conditional(wave_obj.mesh_z, wave_obj.mesh_x))
wave_obj.forward_solve()

print("END")
