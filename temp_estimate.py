import spyro
from spyro import create_transect


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
        "final_time": 2.0,  # Final time for event
    }
    Wave_obj = spyro.examples.Rectangle_acoustic(dictionary=rectangle_dictionary)
    layer_values = [1.5, 3.0]
    z_switches = [-0.5]
    Wave_obj.multiple_layer_velocity_model(z_switches, layer_values)
    maxdt = Wave_obj.get_and_set_maximum_dt()
    print(maxdt)
    Wave_obj.forward_solve()


if __name__ == "__main__":
    test_estimate_timestep_mlt()
