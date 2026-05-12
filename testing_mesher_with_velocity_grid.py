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


def create_grid_based_velocity_field(grid_spacing):
    grid_spacing = 0.02

    dictionary = {
        "length_z": 2.0,
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


print("END")
