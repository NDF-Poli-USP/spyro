import spyro
import numpy as np
import firedrake as fire


def test_2d_wave_adapted_marked_mesh():
    # Making a firedrake type mesh to store the initial
    # velocity model. This isnt the velocity grid we will use
    # in the interpolation. Rather it is a velocity model
    # that has to be converted into the velocity grid for interpolation
    outside_vp = 1.5
    circle_vp = 2.0
    square_vp = 3.0
    frequency = 5.0
    input_mesh_parameters = {
        "length_z": 2.0,
        "length_x": 2.0,
        "dimension": 2,
        "mesh_type": "firedrake_mesh",
        "output_filename": "trial01.msh",
        "edge_length": 0.05,
    }

    mesh_parameters = spyro.meshing.MeshingParameters(
        input_mesh_dictionary=input_mesh_parameters
    )
    meshing_obj = spyro.meshing.AutomaticMesh(mesh_parameters)
    mesh_ho = meshing_obj.create_mesh()
    mesh_z, mesh_x = fire.SpatialCoordinate(mesh_ho)

    V_ho = fire.FunctionSpace(mesh_ho, "KMV", 4)
    r_c = 0.5
    center_z = -1.0
    center_x = 1.0
    square_top_z = -0.9
    square_bot_z = -1.1
    square_left_x = 0.9
    square_right_x = 1.1
    cond = fire.conditional(
        (mesh_z - center_z) ** 2 + (mesh_x - center_x) ** 2 < r_c**2,
        circle_vp,
        outside_vp,
    )
    cond = fire.conditional(
        fire.And(
            fire.And(mesh_z < square_top_z, mesh_z > square_bot_z),
            fire.And(mesh_x > square_left_x, mesh_x < square_right_x),
        ),
        square_vp,
        cond,
    )

    # This is the velocity model used
    # it was generated with a coarses mesh on a MLT4tri element
    # similar to the control variable calculated internally in
    # a multiscale FWI process
    u_ho = fire.Function(V_ho)
    u_ho.interpolate(cond)
    output_file = fire.VTKFile("debug_ho.pvd")
    output_file.write(u_ho)

    # Making a grid like the one we can use to save a .segy file
    # this is usually 0.01 or 0.02 km on a structured mesh
    grid_spacing = 0.01
    input_mesh_parameters = {
        "length_z": 2.0,
        "length_x": 2.0,
        "dimension": 2,
        "mesh_type": "firedrake_mesh",
        "output_filename": "trial.msh",
        "edge_length": grid_spacing,
    }

    mesh_parameters = spyro.meshing.MeshingParameters(
        input_mesh_dictionary=input_mesh_parameters
    )
    meshing_obj = spyro.meshing.AutomaticMesh(mesh_parameters)
    mesh = meshing_obj.create_mesh()

    V = fire.FunctionSpace(mesh, "CG", 1)
    u = fire.Function(V).interpolate(u_ho, allow_missing_dofs=True)
    output_file = fire.VTKFile("debug.pvd")
    output_file.write(u)

    # Creading a grid velocity data dictionary, this can be used to7
    # create a wave-adapted mesh later on and is similar to the data
    # read from a segy file
    z = spyro.io.write_function_to_grid(u, V, grid_spacing, buffer=True)
    grid_velocity_data = {
        "vp_values": z,
        "grid_spacing": grid_spacing,
    }
    # This below is the attribute that the mesh_parameters object uses for
    # any mesh adaption currently programmed in spyro
    mesh_parameters.grid_velocity_data = grid_velocity_data

    # Let us add our mask boundaries
    mask_boundaries = {
        "z_min": -1.3,
        "z_max": -0.7,
        "x_min": 0.7,
        "x_max": 1.3,
    }
    cells_per_wavelength = 2.7
    mesh_parameters.gradient_mask = mask_boundaries
    mesh_parameters.cells_per_wavelength = cells_per_wavelength
    mesh_parameters.source_frequency = frequency
    mesh_parameters.mesh_type = "spyro_mesh"
    meshing_obj = spyro.meshing.AutomaticMesh(mesh_parameters)
    mesh = meshing_obj.create_mesh()  # This is our mesh adapted to the velocity model

    # LEt us check if our wave adapted mesh is wave adapted
    # ---------------------------------------------------------

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
    assert (expected_circle_cd < reduced_cd[-1]) and (
        reduced_cd[-1] < expected_square_cd
    )

    # Let us check mask if the mask was applied
    # ------------------------------------------------

    dx = fire.dx
    V = fire.FunctionSpace(mesh, "DG", 0)
    u = fire.Function(V)
    u.interpolate(1.0)

    # Calculating area outside mask
    area_out = 2**2 - 0.6**2
    form_out = u * dx(1)
    area_out_fire = fire.assemble(form_out)

    assert np.isclose(area_out, area_out_fire, rtol=1e-2)

    # Calculating area inside mask
    area_in = 0.6**2
    form_in = u * dx(2)
    area_in_fire = fire.assemble(form_in)

    assert np.isclose(area_in, area_in_fire, rtol=1e-1)

    print("END")


if __name__ == "__main__":
    test_2d_wave_adapted_marked_mesh()
