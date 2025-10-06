import numpy as np
from scipy.interpolate import RegularGridInterpolator
import spyro
import firedrake as fire
import pytest


def test_velocity_to_grid():
    final_time = 1.0
    dx = 0.006546536707079771

    dictionary = {}
    dictionary["options"] = {
        "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "variant": "lumped",  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
        "degree": 4,  # p order
        "dimension": 2,  # dimension
    }

    # Number of cores for the shot. For simplicity, we keep things serial.
    # spyro however supports both spatial parallelism and "shot" parallelism.
    dictionary["parallelism"] = {
        "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
    }

    # Define the domain size without the PML. Here we'll assume a 1.00 x 1.00 km
    # domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
    # outgoing waves on three sides (eg., -z, +-x sides) of the domain.
    dictionary["mesh"] = {
        "length_z": 3.0,  # depth in km - always positive
        "length_x": 3.0,  # width in km - always positive
        "length_y": 0.0,  # thickness in km - always positive
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",  # options: firedrake_mesh or user_mesh
    }

    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at the center of the mesh.
    # We also specify to record the solution at a microphone near the top of the domain.
    # This transect of receivers is created with the helper function `create_transect`.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-1.5 - dx, 1.5 + dx)],
        "frequency": 5.0,
        "delay": 0.3,
        "receiver_locations": [(-1.5 - dx, 2.0 + dx)],
        "delay_type": "time",
    }

    # Simulate for 2.0 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": final_time,  # Final time for event
        "dt": 0.001,  # timestep size
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
    wave_obj.set_mesh(input_mesh_parameters={"edge_length": 0.02})
    z = wave_obj.mesh_z
    x = wave_obj.mesh_x
    zc = -1.5
    xc = 1.5
    rc = 0.7
    cond = fire.conditional(
                (z - zc) ** 2 + (x - xc) ** 2 < rc**2, 2.0, 1.5
            )
    wave_obj.set_initial_velocity_model(conditional=cond)
    grid_velocity_data = spyro.utils.velocity_to_grid(wave_obj, 0.02)

    vp = grid_velocity_data["vp_values"]
    nz, nx = vp.shape
    z_grid = np.linspace(-grid_velocity_data["length_z"], 0.0, nz, dtype=np.float32)
    x_grid = np.linspace(0.0, grid_velocity_data["length_x"], nx, dtype=np.float32)
    interpolator = RegularGridInterpolator(
        (z_grid, x_grid), vp, bounds_error=False
    )

    # Generate random points inside and outside the circle
    # Generate 5 random points inside the circle
    points_inside = []
    tol = 1e-5
    while len(points_inside) < 5:
        # Generate random point within a square around the circle
        z_rand = np.random.uniform(zc - rc, zc + rc)
        x_rand = np.random.uniform(xc - rc, xc + rc)
        
        # Check if point is inside circle (excluding boundary)
        distance_squared = (z_rand - zc)**2 + (x_rand - xc)**2
        if distance_squared + tol < rc**2:  # Strictly inside (no boundary)
            points_inside.append((z_rand, x_rand))

    # Generate 5 random points outside the circle
    points_outside = []
    domain_z_min, domain_z_max = -3.0, 0.0
    domain_x_min, domain_x_max = 0.0, 3.0

    while len(points_outside) < 5:
        # Generate random point within the domain
        z_rand = np.random.uniform(domain_z_min, domain_z_max)
        x_rand = np.random.uniform(domain_x_min, domain_x_max)
        
        # Check if point is outside circle (excluding boundary)
        distance_squared = (z_rand - zc)**2 + (x_rand - xc)**2
        if distance_squared > rc**2 + tol:  # Strictly outside (no boundary)
            points_outside.append((z_rand, x_rand))

    for point in points_inside:
        velocity = float(interpolator(point))
        assert np.isclose(velocity, 2.0)

    for point in points_outside:
        velocity = float(interpolator(point))
        assert np.isclose(velocity, 1.5)

    print("END")


if __name__ == "__main__":
    test_velocity_to_grid()
