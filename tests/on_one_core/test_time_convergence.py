import spyro
import numpy as np
import math
import pytest


def error_calc(p_numerical, p_analytical, nt):
    norm = np.linalg.norm(p_numerical, 2) / np.sqrt(nt)
    error_time = np.linalg.norm(p_analytical - p_numerical, 2) / np.sqrt(nt)
    div_error_time = error_time / norm
    return div_error_time


def run_forward(dt):
    # dt = float(sys.argv[1])

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
        "Lz": 3.0,  # depth in km - always positive
        "Lx": 3.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
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

    Wave_obj = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj.set_mesh(input_mesh_parameters={"edge_length": 0.02, "periodic": True})

    Wave_obj.set_initial_velocity_model(constant=1.5)
    Wave_obj.forward_solve()

    rec_out = Wave_obj.receivers_output

    return rec_out


@pytest.mark.slow
def test_second_order_time_convergence():
    """Test that the second order time convergence
    of the central difference method is achieved"""

    dts = [
        0.0005,
        0.0001,
    ]

    analytical_files = [
        "tests/inputfiles/analytical_solution_dt_0.0005.npy",
        "tests/inputfiles/analytical_solution_dt_0.0001.npy",
    ]

    numerical_results = []
    errors = []

    for i in range(len(dts)):
        dt = dts[i]
        rec_out = run_forward(dt)
        rec_anal = np.load(analytical_files[i])
        time = np.linspace(0.0, 1.0, int(1.0 / dts[i]) + 1)
        nt = len(time)
        numerical_results.append(rec_out.flatten())
        errors.append(error_calc(rec_out.flatten(), rec_anal, nt))

    theory = [t**2 for t in dts]
    theory = [errors[0] * th / theory[0] for th in theory]

    assert math.isclose(np.log(theory[-1]), np.log(errors[-1]), rel_tol=1e-2)


if __name__ == "__main__":
    test_second_order_time_convergence()
