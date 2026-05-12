import spyro
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



# Let us first create a velocity grid based on the minas cheese model
# First, create a simple mesh to generate a velocity grid
grid_spacing = 0.02
dictionary = {
    "length_z": 2.0,  # Depth in km (always positive)
    "length_x": 2.0,  # Width in km (always positive)
    "length_y": 0.0,  # Thickness in km (0 for 2D)
    "mesh_type": "firedrake_mesh",
    "edge_length": grid_spacing,
    "dimension": 2,
}

# Creating mesh that is a regular grid as well
mesh_params = spyro.meshing.MeshingParameters(input_mesh_dictionary=dictionary)
mesh_generator = spyro.meshing.AutomaticMesh(mesh_parameters=mesh_params)
mesh = mesh_generator.create_mesh()

# Creating Minas Cheese velocity model
# Getting spatialCoordinates 
mesh_z, mesh_x = fire.SpatialCoordinate(mesh)
cond = make_minas_cheese_conditional(mesh_z, mesh_x)
V = fire.FunctionSpace(mesh, "CG", 1)
vp = fire.Function(V).interpolate(cond)

# Creading a grid velocity data dictionary, this can be used to
# create a wave-adapted mesh later on and is similar to the data
# read from a segy file
z = spyro.io.write_function_to_grid(vp, V, grid_spacing, buffer=False)
grid_velocity_data = {
    "vp_values": z,
    "grid_spacing": grid_spacing,
}

print("END")
