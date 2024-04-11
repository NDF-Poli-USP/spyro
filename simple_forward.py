import spyro
import matplotlib.pyplot as plt
import numpy as np


dictionary = {}
dictionary["options"] = {
    "cell_type": "Q",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}
dictionary["mesh"] = {
    "Lz": 3.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "Lx": 3.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-1.1, 1.5)],
    "frequency": 5.0,
    "delay": 0.2,
    "delay_type": "time",
    "receiver_locations": spyro.create_transect((-1.9, 1.2), (-1.9, 1.8), 300),
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": 1.0,  # Final time for event
    "dt": 0.001,  # timestep size
    "output_frequency": 100,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
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
Wave_obj.set_mesh(mesh_parameters={"dx": 0.02, "periodic": True})

Wave_obj.set_initial_velocity_model(constant=1.5)

spyro.plots.plot_model(Wave_obj, filename="model.png", flip_axis=False, show=True)

Wave_obj.forward_solve()

result_function = Wave_obj.u_n
spyro.plots.plot_function(result_function)

result = result_function.dat.data[:]

spyro.plots.plot_shots(Wave_obj, contour_lines=100, vmin=-np.max(result), vmax=np.max(result), show=True)

print("END")
