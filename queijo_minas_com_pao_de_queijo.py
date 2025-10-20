import spyro
import firedrake as fire
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import sys
# import debugpy
# from mpi4py.MPI import COMM_WORLD
# debugpy.listen(3000 + COMM_WORLD.rank)
# debugpy.wait_for_client()
warnings.filterwarnings("ignore")


# degree = int(sys.argv[2])
# frequency = float(sys.argv[1])
degree = 4
frequency = 5.0

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
final_time = 1.3

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
    "length_z": 2.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "length_x": 2.0,  # width in km - always positive
    "length_y": 0.0,  # thickness in km - always positive
    # "mesh_file": "meshes/guess7Hz.msh"
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": spyro.create_transect((-0.35, 0.5), (-0.35, 1.5), 8),
    "frequency": frequency,
    # "frequency_filter": frequency_filter,
    "delay": 1.0/frequency,
    "delay_type": "time",
    "receiver_locations": spyro.create_transect((-1.65, 0.5), (-1.65, 1.5), 200),
}
dictionary["absorving_boundary_conditions"] = {
    "status": True,
    "damping_type": "local",
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": 0.0001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequency'
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
dictionary["inversion"] = {
    "perform_fwi": True,  # switch to true to make a FWI
    "initial_guess_model_file": None,
    "shot_record_file": None,
}

def test_real_shot_record_generation_parallel():

    fwi = spyro.FullWaveformInversion(dictionary=dictionary)

    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.05, "mesh_type": "firedrake_mesh"})
    center_z = -1.0
    center_x = 1.0
    radius = 0.2
    mesh_z = fwi.mesh_z
    mesh_x = fwi.mesh_x
    square_top_z   = -0.9
    square_bot_z   = -1.1
    square_left_x  = 0.9
    square_right_x = 1.1
    cond = fire.conditional((mesh_z-center_z)**2 + (mesh_x-center_x)**2 < radius**2, 3.0, 2.5)
    cond =  fire.conditional(
        fire.And(
            fire.And(mesh_z < square_top_z, mesh_z > square_bot_z),
            fire.And(mesh_x > square_left_x, mesh_x < square_right_x)
        ),
        3.5,
        cond,
    )

    # Create 3 irregular cheese bread-like shapes
    # First cheese bread (main one, slightly elliptical)
    cheese_bread0 = ((mesh_z - (-1.3))/0.18)**2 + ((mesh_x - 0.7)/0.12)**2 < 1.0

    # Second cheese bread (tilted ellipse)
    rotated_z = (mesh_z - (-0.7)) * 0.8 + (mesh_x - 0.6) * 0.6
    rotated_x = -(mesh_z - (-0.7)) * 0.6 + (mesh_x - 0.6) * 0.8
    cheese_bread1 = (rotated_z/0.1)**2 + (rotated_x/0.08)**2 < 1.0

    # Third cheese bread (horizontal ellipse)
    cheese_bread2 = ((mesh_z - (-1.3))/0.08)**2 + ((mesh_x - 1.4)/0.14)**2 < 1.0

    # Combine all 3 cheese bread areas
    cheese_bread_areas = fire.Or(cheese_bread0, fire.Or(cheese_bread1, cheese_bread2))

    # Set conditional
    cond = fire.conditional(cheese_bread_areas, 3.0, cond)


    fwi.set_real_velocity_model(conditional=cond, output=True, dg_velocity_model=False)
    fwi.generate_real_shot_record(plot_model=True, save_shot_record=True, shot_filename=f"shots/shot_record_f{frequency}_")


def test_realistic_fwi():
    dictionary["inversion"] = {
        "perform_fwi": True,
        "real_shot_record_file": f"shots/shot_record_f{frequency}_",
    }
    fwi = spyro.FullWaveformInversion(dictionary=dictionary)

    # Since I'm using a constant velocity model isntead of loadgin one. I'm going to first create 
    # a simple Firedrake mesh to project it into a velocity grid
    fwi.set_guess_mesh(input_mesh_parameters={"mesh_type": "firedrake_mesh", "edge_length": 0.05})
    fwi.set_guess_velocity_model(constant=2.5)
    grid_data = spyro.utils.velocity_to_grid(fwi, 0.01, output=True)
    mask_boundaries = {
        "z_min": -1.55,
        "z_max": -0.45,
        "x_min": 0.45,
        "x_max": 1.55,
    }

    fwi.set_guess_mesh(input_mesh_parameters={
        "mesh_type": "spyro_mesh",
        "cells_per_wavelength": 2.7,
        "grid_velocity_data": grid_data,
        "gradient_mask": mask_boundaries,
        # "output_filename": "test.vtk"
    })
    fwi.set_guess_velocity_model(constant=2.5)

    fwi.run_fwi(vmin=2.5, vmax=3.5, maxiter=10)


if __name__ == "__main__":
    t0 = time.time()
    # test_real_shot_record_generation_parallel()
    test_realistic_fwi()
    t1 = time.time()
    print(f"Total runtime{t1-t0}", flush=True)
