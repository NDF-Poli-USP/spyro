import spyro
import time


frequency = 5.0
# pad = 4.5/frequency

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",
    "variant": "lumped",
    "degree": 4,
    "dimension": 2,
}
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}
dictionary["mesh"] = {
    "Lz": 3.5,
    "Lx": 17.0,
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": "meshes/marmousi_mlt4_5hz.msh",
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.01, 1.0)],
    "frequency": frequency,
    "receiver_locations": spyro.create_transect((-0.10, 0.1), (-0.10, 17.0), 500),
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": 5.0,  # Final time for event
    "dt": 0.0025,  # from spectral radius: 0.00254194
    "output_frequency": 400,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM    - Perguntar Daiane 'gradient_sampling_frequency'
}
dictionary["synthetic_data"] = {
    "real_velocity_file": "velocity_models/vp_marmousi-ii.hdf5",
}
dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": "results/Gradient.pvd",
    "adjoint_output": False,
    "adjoint_filename": None,
    "debug_output": False,
}
# dictionary["absorving_boundary_conditions"] = {
#     "status": True,
#     "damping_type": "PML",
#     "exponent": 2,
#     "cmax": 4.5,
#     "R": 1e-6,
#     "pad_length": pad,
# }
print("Generating wave object", flush=True)
wave = spyro.AcousticWave(dictionary=dictionary)
print("First plot", flush=True)
spyro.plots.plot_mesh_sizes(mesh_filename="meshes/marmousi_mlt4_5hz.msh", title_str="Marmousi wave adapted mesh", output_filename="mesh.png")
t0 = time.time()
wave._initialize_model_parameters()
wave.get_and_set_maximum_dt(fraction=1.0)
wave.forward_solve()
t1 = time.time()
print(f"Total time: {t1-t0}s")
