import numpy as np
from matplotlib import use
use("agg")

import spyro
from spyro.tools.error_measure import calculate_normalized_L2_error

def run_elastic_forward(dt):
    source_z = -2.0
    receiver_z = -3.0
    edge_length = 0.1
    source_locations = [(source_z, -source_z)]
    receiver_locations = [(receiver_z, -receiver_z)]
    final_time = 1.5
    time_delay = 0.2
    frequency = 5.0
    amplitude = np.array([0.0, 1.0])

    rho = 0.1
    vp = 1.5
    vs = 1.0

    dictionary = {
        "options": {
            "cell_type": "Q",
            "variant": "lumped",
            "degree": 4,
            "dimension": 2,
        },
        "parallelism": {
            "type": "automatic",
        },
        "mesh": {
            "length_z": 6.0,
            "length_x": 6.0,
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
            "forward_output": False,
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
    dictionary["acquisition"]["amplitude"] = np.array([0.0, 1.0])
    wave = spyro.IsotropicWave(dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": edge_length, "periodic": False})

    anal_sol = spyro.utils.analytical_solution_elastic(
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
        dimension=2,
    )

    wave.forward_solve()

    nt = int(final_time/dt) + 1
    time_vector = np.linspace(0.0, final_time, nt)
    import matplotlib.pyplot as plt

    fig = spyro.plots.plot_displacement_components(
        time_vector, wave.forward_solution_receivers[:, 0], show=False, hold=True,
    )

    axes = fig.get_axes()
    axes[0].plot(time_vector, anal_sol[0], label="analitical")
    axes[0].legend()
    axes[1].plot(time_vector, anal_sol[1], label="analitical")
    axes[1].legend()
    plt.savefig("test.png")
    l2_error = calculate_normalized_L2_error(wave.forward_solution_receivers[:, 0, 0], anal_sol[0])
    print(f"Normalized L2 error of {l2_error}")
    peak_fraction = np.max(wave.forward_solution_receivers[:, 0, 0]) / np.max(anal_sol[0])
    print(f"Peak fraction of {peak_fraction}")
    print("END")
    plt.close(fig)
    return l2_error

def test_second_order_time_convergence():
    """Test that the second order time convergence
    of the central difference method is achieved"""

    dts = [
        5e-3,
        1e-3,
    ]

    errors = []

    for i in range(len(dts)):
        dt = dts[i]
        error = run_elastic_forward(dt)
        errors.append(error)

    theory = [t**2 for t in dts]
    theory = [errors[0] * th / theory[0] for th in theory]

    assert np.isclose(np.log(theory[-1]), np.log(errors[-1]), rtol=3e-2)


if __name__ == "__main__":
    test_second_order_time_convergence()
