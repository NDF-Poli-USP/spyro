import spyro
import numpy as np


def test_polygon_vp():
    dictionary = {}
    dictionary["polygon_options"] = {
        "water_layer_is_present": True,
        "upper_layer": 2.0,
        "middle_layer": 2.5,
        "lower_layer": 3.0,
        "polygon_layer_perturbation": 0.3,
    }
    dictionary["absorving_boundary_conditions"] = {
        "pad_length": 1.0,  # True or false
    }
    wave = spyro.examples.Polygon_acoustic(dictionary=dictionary, periodic=True)

    # Check water layer velocity
    test_locations = [
        (-0.1, 0.5),  # Water layer p1
        (-0.05, 0.7),  # Water layer p2
        (-0.22, 0.1),  # Upper layer p1
        (-0.25, 0.6), # Upper layer p2
        (-0.50, 0.1), # Middle layer p1
        (-0.55, 0.1), # Bottom layer p1
        (-0.57, 0.2), # Bottom layer p2
        (-0.3, 0.5),  # polygon p1
        (-0.4, 0.6),  # polygon p2
        (-1.3, 0.5),  # pad before change
        (-1.6, 0.5),  # pad after change
    ]
    expected_values = [
        1.5,
        1.5,
        dictionary["polygon_options"]["upper_layer"],
        dictionary["polygon_options"]["upper_layer"],
        dictionary["polygon_options"]["middle_layer"],
        dictionary["polygon_options"]["lower_layer"],
        dictionary["polygon_options"]["lower_layer"],
        dictionary["polygon_options"]["middle_layer"]*(1+dictionary["polygon_options"]["polygon_layer_perturbation"]),
        dictionary["polygon_options"]["middle_layer"]*(1+dictionary["polygon_options"]["polygon_layer_perturbation"]),
        dictionary["polygon_options"]["lower_layer"],
        1.5,
    ]

    # Check upper layer
    test_array = np.isclose(wave.initial_velocity_model.at(test_locations), expected_values)
    test = test_array.all()

    print(f"All points arrive at expected values: {test}", flush=True)
    assert test


def test_real_shot_record_generation_for_polygon_and_save_and_load():
    dictionary = {}
    dictionary["absorving_boundary_conditions"] = {
        "pad_length": 2.0,  # True or false
    }
    dictionary["mesh"] = {
        "h": 0.05,  # mesh size in km
    }
    dictionary["polygon_options"] = {
        "water_layer_is_present": True,
        "upper_layer": 2.0,
        "middle_layer": 2.5,
        "lower_layer": 3.0,
        "polygon_layer_perturbation": 0.3,
    }
    dictionary["acquisition"] = {
        "source_locations": spyro.create_transect((-0.1, 0.1), (-0.1, 0.9), 1),
        "frequency": 5.0,
        "receiver_locations": spyro.create_transect((-0.16, 0.1), (-0.16, 0.9), 100),
    }
    dictionary["visualization"] = {
        "debug_output": True,
    }
    dictionary["time_axis"] = {
        "final_time": 1.0,  # Final time for event
        "dt": 0.0005,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 500,  # how frequently to output solution to pvds
        # how frequently to save solution to RAM
        "gradient_sampling_frequency": 1,
    }
    fwi = spyro.examples.Polygon_acoustic_FWI(dictionary=dictionary, periodic=True)
    fwi.generate_real_shot_record(plot_model=True, save_shot_record=True)

    dictionary["inversion"] = {
        "real_shot_record_files": "shots/shot_record_",
    }
    fwi2 = spyro.examples.Polygon_acoustic_FWI(dictionary=dictionary, periodic=True)

    test1 = np.isclose(np.max(fwi2.real_shot_record[:, 0]), 0.18, atol=1e-2)
    test2 = np.isclose(np.max(fwi2.real_shot_record[:, -1]), 0.0243, atol=1e-3)

    test = all([test1, test2])

    print(f"Correctly loaded shots: {test}")

    assert test


if __name__ == "__main__":
    test_polygon_vp()
    test_real_shot_record_generation_for_polygon_and_save_and_load()
