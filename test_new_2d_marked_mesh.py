import spyro
import numpy as np
import firedrake as fire
import PacMeshIgnore.PacMesh as pm
from RectanglePhysicalGroupNoBorder import build_big_rect_with_inner_element_group


input_mesh_parameters = {
        "length_z": 2.0,
        "length_x": 2.0,
        "dimension": 2,
        "mesh_type": "firedrake_mesh",
        "output_filename": "trial.msh",
        "edge_length": 0.05,
    }

mesh_parameters = spyro.meshing.MeshingParameters(input_mesh_dictionary=input_mesh_parameters)
meshing_obj = spyro.meshing.AutomaticMesh(mesh_parameters)
mesh_ho = meshing_obj.create_mesh()
mesh_z, mesh_x = fire.SpatialCoordinate(mesh_ho)

V_ho = fire.FunctionSpace(mesh_ho, "KMV", 4)
r_c = 0.5
center_z = -1.0
center_x = 1.0
square_top_z   = -0.9
square_bot_z   = -1.1
square_left_x  = 0.9
square_right_x = 1.1
cond = fire.conditional((mesh_z-center_z)**2 + (mesh_x-center_x)**2 < r_c**2, 2.0, 1.5)
cond =  fire.conditional(
    fire.And(
        fire.And(mesh_z < square_top_z, mesh_z > square_bot_z),
        fire.And(mesh_x > square_left_x, mesh_x < square_right_x)
    ),
    3.5,
    cond,
)
u_ho = fire.Function(V_ho)
u_ho.interpolate(cond)
output_file = fire.VTKFile("debug_ho.pvd")
output_file.write(u_ho)

grid_spacing = 0.01
input_mesh_parameters = {
        "length_z": 2.0,
        "length_x": 2.0,
        "dimension": 2,
        "mesh_type": "firedrake_mesh",
        "output_filename": "trial.vtk",
        "edge_length": grid_spacing,
    }

mesh_parameters = spyro.meshing.MeshingParameters(input_mesh_dictionary=input_mesh_parameters)
meshing_obj = spyro.meshing.AutomaticMesh(mesh_parameters)
mesh = meshing_obj.create_mesh()

V = fire.FunctionSpace(mesh, "CG", 1)
u = fire.Function(V).interpolate(u_ho, allow_missing_dofs=True)
output_file = fire.VTKFile("debug.pvd")
output_file.write(u)

z = spyro.io.write_function_to_grid(u, V, grid_spacing, buffer=True)
grid_velocity_data = {
    "vp_values": z,
    "grid_spacing": grid_spacing,
}
mesh_parameters.grid_velocity_data = grid_velocity_data

mask_boundaries = {
        "z_min": -1.3,
        "z_max": -0.7,
        "x_min": 0.7,
        "x_max": 1.3,
    }
mesh_parameters.gradient_mask = mask_boundaries
mesh_parameters.cells_per_wavelength = 2.7
mesh_parameters.source_frequency = 5.0
mesh_parameters.mesh_type = "spyro_mesh"
meshing_obj = spyro.meshing.AutomaticMesh(mesh_parameters)
mesh = meshing_obj.create_mesh()

print("END")
