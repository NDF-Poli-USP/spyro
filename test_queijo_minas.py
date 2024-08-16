import spyro
import firedrake as fire
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
warnings.filterwarnings("ignore")


degree = 4  # int(sys.argv[2])
frequency = 15.0  # float(sys.argv[1])

def cells_per_wavelength(degree):
    cell_per_wavelength_dictionary = {
        'kmv2tri': 7.20,
        'kmv3tri': 3.97,
        'kmv4tri': 2.67,
        'kmv5tri': 2.03,
        'kmv6tri': 1.5,
        'kmv2tet': 6.12,
        'kmv3tet': 3.72,
    }

    cell_type = 'tri'

    key = 'kmv'+str(degree)+cell_type

    return cell_per_wavelength_dictionary.get(key)


cpw = cells_per_wavelength(degree)
final_time = 0.9
# dx = 2.5/(frequency*cpw)
# # dx = 0.01
# print(f"DX:{dx}", flush=True)

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
    "Lz": 2.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "Lx": 2.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_type": "SeismicMesh"
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": spyro.create_transect((-0.55, 0.7), (-0.55, 1.3), 1),
    # "source_locations": [(-1.1, 1.5)],
    "frequency": frequency,
    # "frequency_filter": frequency_filter,
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

    fwi.set_real_mesh(mesh_parameters={"cells_per_wavelength": cpw, "minimum_velocity": 2.5})
    center_z = -1.0
    center_x = 1.0
    mesh_z = fwi.mesh_z
    mesh_x = fwi.mesh_x
    square_top_z   = -0.9
    square_bot_z   = -1.1
    square_left_x  = 0.9
    square_right_x = 1.1
    cond = fire.conditional((mesh_z-center_z)**2 + (mesh_x-center_x)**2 < .2**2, 3.0, 2.5)
    cond =  fire.conditional(
        fire.And(
            fire.And(mesh_z < square_top_z, mesh_z > square_bot_z),
            fire.And(mesh_x > square_left_x, mesh_x < square_right_x)
        ),
        3.5,
        cond,
    )

    fwi.set_real_velocity_model(conditional=cond, output=True, dg_velocity_model=False)
    fwi.generate_real_shot_record(plot_model=True, save_shot_record=True)


def test_realistic_fwi():
    dictionary["inversion"] = {
        "perform_fwi": True,
        "real_shot_record_files": f"shots/shot_record_{frequency}_",
    }
    fwi = spyro.FullWaveformInversion(dictionary=dictionary)
    fwi.set_guess_mesh(mesh_parameters={"cells_per_wavelength": cpw, "minimum_velocity": 2.5})
    fwi.set_guess_velocity_model(constant=2.5)
    mask_boundaries = {
        "z_min": -1.3,
        "z_max": -0.7,
        "x_min": 0.7,
        "x_max": 1.3,
    }
    fwi.set_gradient_mask(boundaries=mask_boundaries)
    fwi.run_fwi(vmin=2.5, vmax=3.5, maxiter=30)


if __name__ == "__main__":
    test_real_shot_record_generation_parallel()
    # test_realistic_fwi()
