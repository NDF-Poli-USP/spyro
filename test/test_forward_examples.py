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
    Wave_obj = spyro.examples.Rectangle_acoustic()

    # Check if velocity model is correct
    layer_values = [1.5, 2.0, 2.5, 3.0]
    z_switches = [-0.25, -0.5, -0.75]
    Wave_obj.multiple_layer_velocity_model(z_switches, layer_values)
    c_wave = Wave_obj.initial_velocity_model

    c0 = layer_values[0]
    test1 = math.isclose(c0, c_wave.at(-0.2, 0.5))

    c2 = layer_values[2]
    test2 = math.isclose(c2, c_wave.at(-0.6, 0.5))

    # Check if forward solve runs
    Wave_obj.forward_solve()
    test3 = True

    assert all([test1, test2, test3])


def test_camembert_elastic():
    from spyro.examples.camembert_elastic import wave
    wave.forward_solve()


def test_elastic_cube_3D():
    from spyro.examples.elastic_cube_3D import wave
    wave.forward_solve()


if __name__ == "__main__":
    test_camembert_forward()
    test_rectangle_forward()
