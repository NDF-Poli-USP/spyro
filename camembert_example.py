import spyro
import numpy as np

camembert_dictionary = {}
camembert_dictionary["mesh"] = {
    "Lz": 1.5,  # depth in km - always positive
    "Lx": 1.5,  # width in km - always positive
    "h": 0.05,  # mesh size in km
}
camembert_dictionary["camembert_options"] = {
    "radius": 0.2,
    "outside_velocity": 2.0,
    "inside_circle_velocity": 2.5,
}
camembert_dictionary["acquisition"] = {
    "source_locations": spyro.create_transect((-0.1, 0.1), (-0.1, 0.9), 10),  # This is just a list of tuples
    "frequency": 6.0,
    "receiver_locations": spyro.create_transect((-0.9, 0.1), (-0.9, 0.9), 30),
}
camembert_dictionary["visualization"] = {
    "debug_output": True,
}

fwi_obj = spyro.examples.Camembert_acoustic_FWI(dictionary=camembert_dictionary)

# If we are not loading external
run_true_synthectic_data = True
if run_true_synthectic_data:
    fwi_obj.set_real_mesh(input_mesh_parameters={"edge_length": 0.05})

    fwi_obj.generate_real_shot_record(plot_model=True, model_filename="True_experiment.png")
    np.save("real_shot_record", fwi_obj.real_shot_record)
else:
    fwi_obj.load_real_shot_record(filename="real_shot_record")

fwi_obj.set_guess_mesh(input_mesh_parameters={"edge_length": 0.1})
fwi_obj.set_guess_velocity_model(constant=2.0)

# Setting gradient mask
mask_boundaries = {
    "z_min": -0.8,
    "z_max": -0.2,
    "x_min": 0.2,
    "x_max": 1.3,
}
fwi_obj.set_gradient_mask(boundaries=mask_boundaries)

fwi_obj.run_fwi(vmin=2.0, vmax=2.5, maxiter=10)
