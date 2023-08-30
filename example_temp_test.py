import spyro
import math


def test_camembert_forward():
    dictionary = {}
    dictionary["absorving_boundary_conditions"] = {
        "status": True,
        "damping_type": "PML",
        "exponent": 2,
        "cmax": 4.5,
        "R": 1e-6,
        "pad_length": 0.25,
    }
    Wave_obj = spyro.examples.Camembert_acoustic(dictionary=dictionary)

    # Check if velocity model is correct
    c_center = 4.6
    c_outside_center = 1.6
    c_wave = Wave_obj.initial_velocity_model
    test1 = math.isclose(c_center, c_wave.at(-0.5, 0.5))
    test2 = math.isclose(c_outside_center, c_wave.at(-0.1, 0.5))

    # Check if forward solve runs
    Wave_obj.forward_solve()
    test3 = True

    assert all([test1, test2, test3])


def test_rectangle_forward():
    dictionary = {}
    dictionary["absorving_boundary_conditions"] = {
        "status": True,
        "damping_type": "PML",
        "exponent": 2,
        "cmax": 4.5,
        "R": 1e-6,
        "pad_length": 0.25,
    }
    Wave_obj = spyro.examples.Rectangle_acoustic(dictionary=dictionary)

    # Check if velocity model is correct
    c_center = 4.6
    c_outside_center = 1.6
    c_wave = Wave_obj.initial_velocity_model
    test1 = math.isclose(c_center, c_wave.at(-0.5, 0.5))
    test2 = math.isclose(c_outside_center, c_wave.at(-0.1, 0.5))

    # Check if forward solve runs
    Wave_obj.forward_solve()
    test3 = True

    assert all([test1, test2, test3])


if __name__ == "__main__":
    test_camembert_forward()
    test_rectangle_forward()
