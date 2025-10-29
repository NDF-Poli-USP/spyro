from copy import deepcopy
import firedrake as fire
from ..meshing import MeshingParameters, AutomaticMesh
from ..io import write_function_to_grid


def velocity_to_grid(wave_obj, grid_spacing, output=False):

    mesh_parameters_original = wave_obj.mesh_parameters
    u, V = change_vp_resolution(wave_obj, grid_spacing)
    z = write_function_to_grid(u, V, grid_spacing, buffer=True)
    if output:
        output_file = fire.VTKFile("debug_velocity_to_grid.pvd")
        output_file.write(u)

    if mesh_parameters_original.abc_pad_length is None:
        pad_length = 0.0
    else:
        pad_length = deepcopy(mesh_parameters_original.abc_pad_length)

    grid_velocity_data = {
        "vp_values": z,
        "grid_spacing": grid_spacing,
        "length_z": deepcopy(mesh_parameters_original.length_z),
        "length_x": deepcopy(mesh_parameters_original.length_x),
        "length_y": deepcopy(mesh_parameters_original.length_y),
        "abc_pad_length": pad_length,
    }
    return grid_velocity_data


def change_vp_resolution(wave_obj, grid_spacing):
    mesh_parameters_original = wave_obj.mesh_parameters
    input_mesh_parameters_cg1 = {
        "dimension": mesh_parameters_original.dimension,
        "length_z": mesh_parameters_original.length_z,
        "length_x": mesh_parameters_original.length_x,
        "length_y": mesh_parameters_original.length_y,
        "mesh_type": "firedrake_mesh",
        "edge_length": grid_spacing,
        "abc_pad_length": mesh_parameters_original.abc_pad_length
    }
    meshing_parameters_cg1 = MeshingParameters(
        input_mesh_dictionary=input_mesh_parameters_cg1,
        comm=mesh_parameters_original.comm
    )
    meshing_obj = AutomaticMesh(meshing_parameters_cg1)
    mesh = meshing_obj.create_mesh()
    V = fire.FunctionSpace(mesh, "CG", 1)
    u = fire.Function(V).interpolate(wave_obj.initial_velocity_model, allow_missing_dofs=True)

    return (u, V)
