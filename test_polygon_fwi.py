# from mpi4py.MPI import COMM_WORLD
# import debugpy
# debugpy.listen(3000 + COMM_WORLD.rank)
# debugpy.wait_for_client()
import spyro
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



def test_real_shot_record_generation_parallel():
    dictionary = {}
    dictionary["absorving_boundary_conditions"] = {
        "pad_length": 1.0,  # True or false
    }
    dictionary["mesh"] = {
        "h": 0.03,  # mesh size in km
    }
    dictionary["polygon_options"] = {
        "water_layer_is_present": True,
        "water_layer_depth": 0.2,
        "upper_layer": 2.0,
        "middle_layer": 2.5,
        "lower_layer": 3.0,
        "polygon_layer_perturbation": 0.3,
    }
    dictionary["acquisition"] = {
        "source_locations": spyro.create_transect((-0.1, 1.0), (-0.1, 2.0), 7),
        "frequency": 5.0,
        "receiver_locations": spyro.create_transect((-0.15, 1.0), (-0.15, 2.0), 100),
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

    # dictionary["inversion"] = {
    #     "real_shot_record_files": "shots/shot_record_",
    # }
    # fwi2 = spyro.examples.Polygon_acoustic_FWI(dictionary=dictionary, periodic=True)

    # max_value = np.max(fwi2.real_shot_record[:, fwi2.comm.ensemble_comm.rank*33])
    # test_core = np.isclose(max_value, 0.184, atol=1e-2)

    # test_core_all = fwi2.comm.ensemble_comm.allgather(test_core)
    # test = all(test_core_all)

    # print(f"Correctly loaded shots: {test}", flush=True)

    # assert test


def test_velocity_smoother_in_fwi():
    dictionary = {}
    dictionary["absorving_boundary_conditions"] = {
        "pad_length": 1.0,  # True or false
    }
    dictionary["mesh"] = {
        "h": 0.03,  # mesh size in km
    }
    dictionary["polygon_options"] = {
        "water_layer_is_present": True,
        "water_layer_depth": 0.2,
        "upper_layer": 2.0,
        "middle_layer": 2.5,
        "lower_layer": 3.0,
        "polygon_layer_perturbation": 0.3,
    }
    dictionary["acquisition"] = {
        "source_locations": spyro.create_transect((-0.1, 1.0), (-0.1, 2.0), 1),
        "frequency": 5.0,
        "receiver_locations": spyro.create_transect((-0.15, 1.0), (-0.15, 2.0), 100),
    }
    fwi = spyro.examples.Polygon_acoustic_FWI(dictionary=dictionary, periodic=True)
    spyro.io.create_segy(
        fwi.initial_velocity_model,
        fwi.function_space,
        10.0/1000.0,
        "velocity_models/true_case1.segy",
    )
    spyro.tools.velocity_smoother.smooth_velocity_field_file("velocity_models/true_case1.segy", "velocity_models/case1_sigma10.segy", 10, show=True, write_hdf5=True)
    plt.savefig("velocity_models/case1_sigma10.png")
    plt.close()


def test_realistic_fwi():
    dictionary = {}
    dictionary["absorving_boundary_conditions"] = {
        "pad_length": 1.0,  # True or false
    }
    dictionary["mesh"] = {
        "h": 0.05,  # mesh size in km
    }
    dictionary["polygon_options"] = {
        "water_layer_is_present": True,
        "water_layer_depth": 0.2,
        "upper_layer": 2.0,
        "middle_layer": 2.5,
        "lower_layer": 3.0,
        "polygon_layer_perturbation": 0.3,
    }
    dictionary["acquisition"] = {
        "source_locations": spyro.create_transect((-0.1, 1.0), (-0.1, 2.0), 7),
        "frequency": 5.0,
        "receiver_locations": spyro.create_transect((-0.15, 1.0), (-0.15, 2.0), 100),
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
    dictionary["inversion"] = {
        "real_shot_record_files": "shots/shot_record_",
    }
    fwi = spyro.examples.Polygon_acoustic_FWI(dictionary=dictionary, periodic=True)
    fwi.set_guess_velocity_model(new_file="velocity_models/case1_sigma10.hdf5")
    mask_boundaries = {
        "z_min": -2.0,
        "z_max": -0.,
        "x_min": 0.0,
        "x_max": 3.0,
    }
    fwi.set_gradient_mask(boundaries=mask_boundaries)
    fwi.run_fwi(vmin=1.5, vmax=3.25, maxiter=5)


if __name__ == "__main__":
    # test_real_shot_record_generation_parallel()
    # test_velocity_smoother_in_fwi()
    test_realistic_fwi()
