import spyro
from spyro import create_transect
import math


def test_estimate_timestep_mlt():
    rectangle_dictionary = {}
    rectangle_dictionary["options"] = {
        # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "cell_type": "T",
        "variant": "lumped",
    }
    rectangle_dictionary["mesh"] = {
        "Lz": 0.75,  # depth in km - always positive
        "Lx": 1.5,
        "h": 0.05,
    }
    rectangle_dictionary["acquisition"] = {
        "source_locations": [(-0.1, 0.75)],
        "receiver_locations": create_transect((-0.10, 0.1), (-0.10, 1.4), 50),
        "frequency": 8.0,
    }
    rectangle_dictionary["time_axis"] = {
        "final_time": 1.0,  # Final time for event
    }
    Wave_obj = spyro.examples.Rectangle_acoustic(dictionary=rectangle_dictionary)
    layer_values = [1.5, 3.0]
    z_switches = [-0.5]
    Wave_obj.multiple_layer_velocity_model(z_switches, layer_values)

    # Tests value and if stable for 0.7 times estimated timestep
    maxdt = Wave_obj.get_and_set_maximum_dt(fraction=0.7)
    print(maxdt)
    test1 = math.isclose(maxdt, 0.00085928546, rel_tol=1e-3)

    test2 = False
    try:
        Wave_obj.forward_solve()
        test2 = True
    except AssertionError:
        test2 = False

    # Tests value and if unstable for 1.1 times estimated timestep
    Wave_obj.current_time = 0.0
    maxdt = Wave_obj.get_and_set_maximum_dt(fraction=1.1)
    test3 = math.isclose(maxdt, 0.001350305724782782, rel_tol=1e-3)

    test4 = False
    try:
        Wave_obj.forward_solve()
        test4 = False
    except AssertionError:
        test4 = True

    print("Test 1: ", test1)
    print("Test 2: ", test2)
    print("Test 3: ", test3)
    print("Test 4: ", test4)

    assert all([test1, test2, test3, test4])


if __name__ == "__main__":
    test_estimate_timestep_mlt()
