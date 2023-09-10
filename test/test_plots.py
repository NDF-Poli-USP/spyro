import spyro
from spyro import create_transect


def test_plot():
    rectangle_dictionary = {}
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
    Wave_obj = spyro.examples.Rectangle_acoustic(
        dictionary=rectangle_dictionary
    )
    layer_values = [1.5, 3.0]
    z_switches = [-0.5]
    Wave_obj.multiple_layer_velocity_model(z_switches, layer_values)
    Wave_obj.forward_solve()
    spyro.plots.plot_shots(Wave_obj)


if __name__ == "__main__":
    test_plot()
