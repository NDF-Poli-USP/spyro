import matplotlib.pyplot as plt
import numpy as np
import spyro


def test_analytical_solution():
    frequency = 5.0
    offset = 0.5
    c_value = 1.5
    dictionary = {}
    dictionary["absorving_boundary_conditions"] = {
        "status": False,
    }
    dictionary["mesh"] = {
        "Lz": 3.0,  # depth in km - always positive
        "Lx": 3.0,  # width in km - always positive
    }
    dictionary["acquisition"] = {
        "delay_type": "time",
        "frequency": frequency,
        "delay": c_value/frequency,
        "source_locations": [(-1.5, 1.5)],
        "receiver_locations": [(-1.5 - offset, 1.5)],
    }
    Wave_obj = spyro.examples.Rectangle_acoustic(dictionary=dictionary)
    Wave_obj.set_initial_velocity_model(constant=c_value)
    analytical_p = spyro.utils.nodal_homogeneous_analytical(Wave_obj, offset, c_value)

    time_vector = np.linspace(0.0, 1.0, int(1.0/Wave_obj.dt)+1)
    plt.plot(time_vector, analytical_p, label="Analytical", color="black", linestyle="--")
    # Wave_obj.forward_solve()
    # numerical_p = Wave_obj.receivers_output
    # numerical_p.flatten()

    # plt.plot(time_vector, numerical_p, label="Numerical", color="red")
    plt.legend()
    # plt.plot(time, -(p_analytical - p_numerical))
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (Pa)")
    plt.show()


if __name__ == "__main__":
    test_analytical_solution()
