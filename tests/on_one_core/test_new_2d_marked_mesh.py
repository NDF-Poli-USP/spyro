import spyro
import numpy as np
import firedrake as fire


def test_2d_gradient_marked_mesh():
    outside_vp = 1.5
    circle_vp = 2.0
    square_vp = 3.0
    frequency = 5.0
    cells_per_wavelength = 2.7
    input_mesh_parameters = {
            "length_z": 2.0,
            "length_x": 2.0,
            "dimension": 2,
            "mesh_type": "firedrake_mesh",
            "output_filename": "trial01.msh",
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
    cond = fire.conditional((mesh_z-center_z)**2 + (mesh_x-center_x)**2 < r_c**2, circle_vp, outside_vp)
    cond =  fire.conditional(
        fire.And(
            fire.And(mesh_z < square_top_z, mesh_z > square_bot_z),
            fire.And(mesh_x > square_left_x, mesh_x < square_right_x)
        ),
        square_vp,
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
            "output_filename": "trial.msh",
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
    mesh_parameters.cells_per_wavelength = cells_per_wavelength
    mesh_parameters.source_frequency = frequency
    mesh_parameters.mesh_type = "spyro_mesh"
    meshing_obj = spyro.meshing.AutomaticMesh(mesh_parameters)
    mesh = meshing_obj.create_mesh()

    # Getting mesh cell diameters
    V = fire.FunctionSpace(mesh, "CG", 1)
    cd_ufl = fire.CellSize(mesh)
    cd_function = fire.Function(V)
    cd_function.interpolate(cd_ufl)
    cell_diameters = cd_function.dat.data[:]

    # calculating expected cells diameters
    expected_smallest_cd = outside_vp / (frequency * cells_per_wavelength)
    expected_circle_cd = circle_vp / (frequency * cells_per_wavelength)
    expected_square_cd = square_vp / (frequency * cells_per_wavelength)

    # Sorting and removing smallest cell diameters (because of worse elements)
    sorted_cd = np.sort(cell_diameters)
    reduced_cd = sorted_cd[10:]

    # Checking smallest value
    assert np.isclose(reduced_cd[0], expected_smallest_cd, rtol=1e-1)

    # Checking largest value
    assert (expected_circle_cd < reduced_cd[-1]) and (reduced_cd[-1] < expected_square_cd)

    # Checking mask
    dx = fire.dx
    V = fire.FunctionSpace(mesh, "DG", 0)
    u = fire.Function(V)
    u.interpolate(1.0)

    # Calculating area outside mask
    area_out = 2**2 - 0.6**2
    form_out = u*dx(1)
    area_out_fire = fire.assemble(form_out)

    assert np.isclose(area_out, area_out_fire, rtol=1e-2)

    # Calculating area inside mask
    area_in = 0.6**2
    form_in = u*dx(2)
    area_in_fire = fire.assemble(form_in)

    assert np.isclose(area_in, area_in_fire, rtol=1e-1)

    print("END")


if __name__ == "__main__":
    test_2d_gradient_marked_mesh()
