import numpy as np
from scipy.interpolate import RegularGridInterpolator

import firedrake as fire

from spyro.meshing import AutomaticMesh, MeshingParameters
from spyro.utils.velocity_to_grid import scalar_conditional_to_grid


def make_cheese_conditional(mesh_z, mesh_x):
    outside_vp = 1.0
    circle_vp = 2.0
    r_c = 0.5
    center_z = -1.0
    center_x = 1.0
    cond = fire.conditional(
        (mesh_z - center_z) ** 2 + (mesh_x - center_x) ** 2 < r_c**2,
        circle_vp,
        outside_vp,
    )
    return cond


def test_scalar_conditional_to_grid():
    np.random.seed(1)
    grid_spacing = 0.02

    dictionary = {
        "length_z": 2.0,
        "length_x": 2.0,
        "length_y": 0.0,
        "mesh_type": "firedrake_mesh",
        "edge_length": grid_spacing,
        "dimension": 2,
    }

    mesh_params = MeshingParameters(input_mesh_dictionary=dictionary)
    mesh_generator = AutomaticMesh(mesh_parameters=mesh_params)
    mesh = mesh_generator.create_mesh()

    mesh_z, mesh_x = fire.SpatialCoordinate(mesh)
    cond = make_cheese_conditional(mesh_z, mesh_x)

    grid_velocity_data = scalar_conditional_to_grid(
        conditional=cond,
        domain_dimensions=(2.0, 2.0),
        grid_spacing=grid_spacing,
    )

    vp = grid_velocity_data["vp_values"]
    assert np.isclose(grid_velocity_data["grid_spacing"], grid_spacing)
    assert np.isclose(grid_velocity_data["length_z"], 2.0)
    assert np.isclose(grid_velocity_data["length_x"], 2.0)
    assert np.isclose(grid_velocity_data["length_y"], 0.0)
    assert np.isclose(grid_velocity_data["abc_pad_length"], 0.0)

    nz, nx = vp.shape
    z_grid = np.linspace(-grid_velocity_data["length_z"], 0.0, nz, dtype=np.float32)
    x_grid = np.linspace(0.0, grid_velocity_data["length_x"], nx, dtype=np.float32)
    interpolator = RegularGridInterpolator(
        (z_grid, x_grid), vp, bounds_error=False
    )

    points_inside = []
    points_outside = []
    zc = -1.0
    xc = 1.0
    rc = 0.5
    tol = 0.2  # High value to avoid erros close to heterogenuity
    while len(points_inside) < 5:
        z_rand = np.random.uniform(zc - rc, zc + rc)
        x_rand = np.random.uniform(xc - rc, xc + rc)
        if (z_rand - zc) ** 2 + (x_rand - xc) ** 2 + tol < rc**2:
            points_inside.append((z_rand, x_rand))

    while len(points_outside) < 5:
        z_rand = np.random.uniform(-2.0, 0.0)
        x_rand = np.random.uniform(0.0, 2.0)
        if (z_rand - zc) ** 2 + (x_rand - xc) ** 2 > rc**2 + tol:
            points_outside.append((z_rand, x_rand))

    for point in points_inside:
        velocity = float(interpolator(point))
        assert np.isclose(velocity, 2.0)

    for point in points_outside:
        velocity = float(interpolator(point))
        assert np.isclose(velocity, 1.0)


if __name__ == "__main__":
    test_scalar_conditional_to_grid()
