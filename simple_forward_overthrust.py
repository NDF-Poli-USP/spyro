import spyro
import numpy as np

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}
dictionary["mesh"] = {
    "Lz": 2.8,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "Lx": 6.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": "meshes/cut_overthrust.msh",
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.01, 3.0)],
    "frequency": 5.0,
    "receiver_locations": spyro.create_transect((-0.37, 0.2), (-0.37, 5.8), 300),
}
dictionary["absorving_boundary_conditions"] = {
    "status": True,
    "damping_type": "PML",
    "exponent": 2,
    "cmax": 4.5,
    "R": 1e-6,
    "pad_length": 0.75,
}
dictionary["synthetic_data"] = {
    "real_velocity_file": "velocity_models/cut_overthrust.hdf5",
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": 5.00,  # Final time for event
    "dt": 0.0005,  # timestep size
    "output_frequency": 200,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM    - Perguntar Daiane 'gradient_sampling_frequency'
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
Wave_obj = spyro.AcousticWave(dictionary=dictionary)
Wave_obj.forward_solve()
spyro.plots.plot_model(Wave_obj, filename="model_overthrust.png", show=True)
shot_record = Wave_obj.receivers_output
vmax = 0.1*np.max(shot_record)
spyro.plots.plot_shots(Wave_obj, contour_lines=100, vmin=-vmax, vmax=vmax, show=True)