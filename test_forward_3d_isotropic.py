"""Test second order time convergence on the 3D isotropic elastic wave."""

import numpy as np
import pytest

import spyro
from spyro.tools.error_measure import MeasureError


def run_elastic_forward(dt):
    """Run a forward 3d isotropic elastic propagation."""
    source_z = -1.0
    receiver_z = -1.2
    edge_length = 0.05
    source_locations = [(source_z, -source_z, -source_z)]
    receiver_locations = [(receiver_z, -receiver_z, -receiver_z)]
    final_time = 0.8
    time_delay = 0.2
    frequency = 5.0
    amplitude = np.array([0.0, 1.0, 0.0])

    rho = 0.1
    vp = 1.5
    vs = 1.0

    """Run forward isotropic elastic case."""
    dictionary = {
        "options": {
            "cell_type": "Q",
            "variant": "lumped",
            "degree": 4,
            "dimension": 3,
        },
        "parallelism": {
            "type": "automatic",
        },
        "mesh": {
            "length_z": 2.0,
            "length_x": 2.0,
            "length_y": 2.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": source_locations,
            "frequency": frequency,
            "delay": time_delay,
            "delay_type": "time",
            "receiver_locations": receiver_locations,
            "amplitude": amplitude,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": final_time,
            "dt": dt,
            "output_frequency": 100,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": True,
            "fwi_velocity_model_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
    }

    dictionary["synthetic_data"] = {
        "type": "object",
        "density": rho,
        "p_wave_velocity": vp,
        "s_wave_velocity": vs,
        "real_velocity_file": None,
    }

    wave = spyro.IsotropicWave(dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": edge_length, "periodic": False})

    analitical_solution = spyro.utils.analytical_solution_elastic(
        "force_source",
        np.array(source_locations[0]) - np.array(receiver_locations[0]),
        p_wave_velocity=vp,
        s_wave_velocity=vs,
        density=rho,
        amplitude=1.0,
        force_direction=1,
        frequency=frequency,
        time_delay=time_delay,
        final_time=final_time,
        dt=dt,
        dimension=3,
    )

    wave.forward_solve()

    np.save("forward_elastic.npy", wave.forward_solution_receivers)
    l2_error = MeasureError.calculate_normalized_L2_error(
        wave.forward_solution_receivers[:, 0, 0], analitical_solution[0]
    )

    return l2_error, wave.comm


@pytest.mark.slow
def test_second_order_time_convergence():
    """Test that the second order time convergence is achieved."""
    dts = [
        3.2e-3,
        0.002898550724637681,
        2.5e-3,
        2e-3,
    ]

    errors = []

    for i in range(len(dts)):
        dt = dts[i]
        error, comm = run_elastic_forward(dt)
        print(f"For dt of {dt}, error is {error}")
        errors.append(error)

    errors = comm.ensemble_comm.bcast(errors, root=0)
    log_dts = np.log(dts)
    log_errors = np.log(errors)
    p, _ = np.polyfit(log_dts, log_errors, 1)

    assert np.isclose(p, 2.0, rtol=1e-1)


if __name__ == "__main__":
    test_second_order_time_convergence()
