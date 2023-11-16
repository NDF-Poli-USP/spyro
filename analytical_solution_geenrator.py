import numpy as np
import spyro


def generate_analytical_solution(dt):
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
        "delay": c_value / frequency,
        "source_locations": [(-1.5, 1.5)],
        "receiver_locations": [(-1.5 - offset, 1.5)],
    }
    dictionary["time_axis"] = {
        "dt": dt,
    }
    Wave_obj = spyro.examples.Rectangle_acoustic(
        dictionary=dictionary, periodic=True
    )
    Wave_obj.set_initial_velocity_model(constant=c_value)
    analytical_p = spyro.utils.nodal_homogeneous_analytical(
        Wave_obj, offset, c_value
    )

    np.save("analytical_solution_dt"+str(dt)+".npy", analytical_p)


if __name__ == "__main__":
    dts = [
        5e-4,
        1e-4,
        5e-5,
    ]
    for dt in dts:
        print(f"Generating analytical solution for timestep of {dt} s")
        generate_analytical_solution(dt)
