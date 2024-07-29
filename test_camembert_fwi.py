# from mpi4py.MPI import COMM_WORLD
# import debugpy
# debugpy.listen(3000 + COMM_WORLD.rank)
# debugpy.wait_for_client()
import spyro
import firedrake as fire
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


final_time = 0.9

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
    "Lz": 2.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "Lx": 2.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": spyro.create_transect((-0.55, 0.7), (-0.55, 1.3), 6),
    # "source_locations": [(-1.1, 1.5)],
    "frequency": 5.0,
    "delay": 0.2,
    "delay_type": "time",
    "receiver_locations": spyro.create_transect((-1.45, 0.7), (-1.45, 1.3), 200),
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM    - Perguntar Daiane 'gradient_sampling_frequency'
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
    
    fwi = spyro.FullWaveformInversion(dictionary=dictionary)

    fwi.set_real_mesh(mesh_parameters={"dx": 0.1})
    center_z = -1.0
    center_x = 1.0
    mesh_z = fwi.mesh_z
    mesh_x = fwi.mesh_x
    cond = fire.conditional((mesh_z-center_z)**2 + (mesh_x-center_x)**2 < .2**2, 3.0, 2.5)

    fwi.set_real_velocity_model(conditional=cond, output=True, dg_velocity_model=False)
    fwi.generate_real_shot_record(plot_model=True, save_shot_record=True)


def test_realistic_fwi():
    dictionary["inversion"] = {
        "real_shot_record_files": "shots/shot_record_",
    }
    fwi = spyro.FullWaveformInversion(dictionary=dictionary)
    fwi.set_guess_mesh(mesh_parameters={"dx": 0.1})
    fwi.set_guess_velocity_model(constant=2.5)
    mask_boundaries = {
        "z_min": -1.3,
        "z_max": -0.7,
        "x_min": 0.7,
        "x_max": 1.3,
    }
    fwi.set_gradient_mask(boundaries=mask_boundaries)
    fwi.run_fwi(vmin=2.5, vmax=3.0, maxiter=5)


if __name__ == "__main__":
    test_real_shot_record_generation_parallel()
    # test_realistic_fwi()
