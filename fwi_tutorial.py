import spyro
import numpy as np

v_min = 1.5
v_max = 2.5
input_dictionary = {}
input_dictionary["mesh"] = {
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "h": 0.05,  # mesh size in km
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",  # options: firedrake_mesh or user_mesh
}
input_dictionary["parallelism"] = {
    "type": "spatial",
}
input_dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": spyro.create_transect((-0.10, 0.1), (-0.10, 0.9), 10),
    "frequency": 5.0,
    # "delay": 1.0,
    "receiver_locations": spyro.create_transect((-0.90, 0.1), (-0.90, 0.9), 30),
}
input_dictionary["time_axis"] = {
    "dt": 0.0005,  # timestep size
}
input_dictionary["camembert_options"] = {
    "radius": 0.2,
    "circle_center": (-0.5, 0.5),
    "outside_velocity": v_min,
    "inside_circle_velocity": v_max,
}
input_dictionary["absorving_boundary_conditions"] = {
    "status": True,
    "damping_type": "local",
    "absorb_top": True,
    "absorb_bottom": True,
    "absorb_right": True,
    "absorb_left": True,
}
optimization_parameters = {
    "General": {
        "Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 10}
    },
    "Step": {
        "Type": "Augmented Lagrangian",
        "Augmented Lagrangian": {
            "Subproblem Step Type": "Line Search",
            "Subproblem Iteration Limit": 5.0,
        },
        "Line Search": {"Descent Method": {"Type": "Quasi-Newton Step"}},
    },
    "Status Test": {
        "Gradient Tolerance": 1e-16,
        "Iteration Limit": None,
        "Step Tolerance": 1.0e-16,
    },
}
input_dictionary["inversion"] = {
    "perform_fwi": True,  # switch to true to make a FWI
    "initial_guess_model_file": None,
    "shot_record_file": None,
    "optimization_parameters": optimization_parameters,
}

fwiobj = spyro.examples.Camembert_acoustic_FWI(dictionary=input_dictionary)

spyro.plots.plot_model(fwiobj, filename="true_experiment.png")
spyro.plots.plot_model_in_p1(fwiobj, dx=0.005, filename="true_experiment_in_p1.png")
spyro.plots.plot_mesh_sizes(firedrake_mesh=fwiobj.real_mesh, output_filename="true_mesh.png", show_size_contour=False)

fwiobj.generate_real_shot_record(save_shot_record=True)
# input_dictionary["inversion"]["real_shot_record_files"] = "shots/shot_record_"
# fwiobj = spyro.examples.Camembert_acoustic_FWI(dictionary=input_dictionary)

# fwiobj.set_guess_mesh()
# fwiobj.set_guess_velocity_model(constant=1.6)
# spyro.plots.plot_mesh_sizes(firedrake_mesh=fwiobj.guess_mesh, output_filename="guess_mesh.png", show_size_contour=False)
# fwiobj.run_fwi(vmin=v_min, vmax=v_max, maxiter=5)

print("END")
