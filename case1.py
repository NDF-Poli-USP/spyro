# from mpi4py.MPI import COMM_WORLD
# import debugpy
# debugpy.listen(3000 + COMM_WORLD.rank)
# debugpy.wait_for_client()
import spyro
import matplotlib.pyplot as plt
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
    "source_type": "ricker",
    "source_locations": spyro.create_transect((-0.1, 0.1), (-0.1, 0.9), 1),
    "frequency": 5.0,
    "delay": 1.0,
    "receiver_locations": spyro.create_transect((-0.16, 0.1), (-0.16, 0.9), 100),
}
dictionary["visualization"] = {
    "debug_output": True,
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": 1.0,  # Final time for event
    "dt": 0.0001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 500,  # how frequently to output solution to pvds
    # how frequently to save solution to RAM
    "gradient_sampling_frequency": 1,
}
dictionary["inversion"] = {
    "initial_guess_model_file": "velocity_models/case1_sigma10.hdf5",
}
fwi = spyro.examples.Polygon_acoustic_FWI(dictionary=dictionary, periodic=True)
fwi.generate_real_shot_record(plot_model=True)
fwi.initial_velocity_model = None
fwi.set_guess_velocity_model(new_file="velocity_models/case1_sigma10.hdf5")
# spyro.io.save_shots(fwi)
# spyro.io.load_shots(fwi)
# fwi.real_shot_record = fwi.forward_solution_receivers
fwi.run_fwi(vmin=1.5, vmax=3.25, maxiter=5)
# spyro.io.create_segy(
#     fwi.initial_velocity_model,
#     fwi.function_space,
#     10.0/1000.0,
#     "velocity_models/true_case1.segy",
#     )

# plt.close()
# spyro.tools.velocity_smoother.smooth_velocity_field_file("velocity_models/true_case1.segy", "velocity_models/case1_sigma10.segy", 10, show=True)
# plt.savefig("velocity_models/case1_sigma10.png")
print("END")
