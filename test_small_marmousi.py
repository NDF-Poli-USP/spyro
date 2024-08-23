import spyro
import sys


degree = int(sys.argv[1])
frequency = float(sys.argv[2])
final_time = float(sys.argv[3])


dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": degree,  # p order
    "dimension": 2,  # dimension
}
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}
dictionary["mesh"] = {
    "Lz": 3.5,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "Lx": 17.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": spyro.create_transect((-0.01, 4.0), (-0.01, 12.0), 20),
    # "source_locations": [(-0.01, 4.0)],
    "frequency": frequency,
    # "frequency_filter": frequency_filter,
    "delay": 0.2,
    "delay_type": "time",
    "receiver_locations": spyro.create_transect((-1.45, 4.0), (-1.45, 12.0), 100),
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM
}
dictionary["visualization"] = {
    "forward_output": False,
    "forward_output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": "results/Gradient.pvd",
    "adjoint_output": False,
    "adjoint_filename": None,
    "debug_output": False,
}
dictionary["inversion"] = {
    "perform_fwi": True,  # switch to true to make a FWI
    "initial_guess_model_file": None,
    "shot_record_file": None,
}

def test_real_shot_record_generation_parallel():
    dictionary["mesh"]["mesh_file"] = "meshes/real5hz.msh"
    
    fwi = spyro.FullWaveformInversion(dictionary=dictionary)
    fwi.set_real_velocity_model(new_file="velocity_models/vp_marmousi-ii.hdf5")
    fwi.generate_real_shot_record(plot_model=True, save_shot_record=True)


def test_realistic_fwi():
    dictionary["inversion"] = {
        "perform_fwi": True,
        "real_shot_record_files": f"shots/shot_record_",
        "initial_guess_model_file": "velocity_models/initial_guess_15Hz.hdf5",
    }
    fwi = spyro.FullWaveformInversion(dictionary=dictionary)
    fwi.set_guess_velocity_model(new_file="velocity_models/initial_guess_15Hz.hdf5")
    mask_boundaries = {
        "z_min": -1.3,
        "z_max": -0.7,
        "x_min": 0.7,
        "x_max": 1.3,
    }
    fwi.set_gradient_mask(boundaries=mask_boundaries)
    fwi.run_fwi(vmin=2.5, vmax=3.5, maxiter=60)


if __name__ == "__main__":
    test_real_shot_record_generation_parallel()
    # test_realistic_fwi()
