import numpy as np
import math
import spyro
import time


def test_marmousi_fig21():
    cpw = 3.0

    dictionary = {}
    dictionary["options"] = {
        "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "variant": "lumped",  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
        "degree": 4,  # p order
        "dimension": 2,  # dimension
    }

    # Number of cores for the shot. For simplicity, we keep things serial.
    # spyro however supports both spatial parallelism and "shot" parallelism.
    dictionary["parallelism"] = {
        "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
    }
    # Define the domain size without the PML. Here we'll assume a 1.00 x 1.00 km
    # domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
    # outgoing waves on three sides (eg., -z, +-x sides) of the domain.
    dictionary["mesh"] = {
        "Lz": 3.0,  # depth in km - always positive
        "Lx": 9.2,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "mesh_file": None,
        "mesh_type": "SeismicMesh",  # options: firedrake_mesh or user_mesh
        "cells_per_wavelength": cpw,
    }
    dictionary["absorving_boundary_conditions"] = {
        "status": False,
    }
    # Create a source injection operator. Here we use a single source with a
    # Ricker wavelet that has a peak frequency of 5 Hz injected at the center of the mesh.
    # We also specify to record the solution at a microphone near the top of the domain.
    # This transect of receivers is created with the helper function `create_transect`.
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.025, 5.65)],
        "frequency": 5.0,
        "receiver_locations": [(-0.1, 1.0), (-0.1, 3.0)],
    }
    # Simulate for 2.0 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": 2.0,  # Final time for event
        "dt": 0.001,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 1000,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM
    }
    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/temp_cut_marmousipropagation.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
        "debug_output": True,
    }
    dictionary["synthetic_data"] = {
        "real_velocity_file": "/home/olender/Documents/reuniao_noneikonal/spyro-1/cut_marmousi_ruben.segy",
        # "real_velocity_file": "/home/olender/common_files/velocity_models/vp_marmousi-ii.segy",
    }

    Wave_obj = spyro.AcousticWave(dictionary=dictionary)
    spyro.io.saving_source_and_receiver_location_in_csv(dictionary)
    Wave_obj.set_mesh(mesh_parameters={"cells_per_wavelength": cpw})
    t0 = time.time()
    Wave_obj.forward_solve()
    t1 = time.time()
    print(f"Time for forward problem: {t1-t0} s")

    min_value = Wave_obj.noneikonal_minimum
    paper_min = 1.0391

    # Testing minimum values
    test_min = math.isclose(min_value, paper_min, rel_tol=0.1)
    print("min_value: ", min_value)
    print("paper_min: ", paper_min)
    print(f"Passed the minimum value test: {test_min}")

    # Testing minimum location
    z_min, x_min = Wave_obj.noneikonal_minimum_point
    paper_z_min = -3.0
    paper_x_min = 5.6625

    test_z_min = math.isclose(z_min, paper_z_min, rel_tol=0.1)
    test_x_min = math.isclose(x_min, paper_x_min, rel_tol=0.1)

    test_min_point = all([test_z_min, test_x_min])
    print("x_min: ", x_min)
    print("paper_x_min: ", paper_x_min)
    print(f"Passed the minimum point location test: {test_min_point}")

    assert all([test_min, test_min_point])


# Verificar valores das distancias como lref e velocidades
if __name__ == "__main__":
    test_marmousi_fig21()

# 0.2227, 0.3067, 0.3403, 0.4748 fref=3.37 lmin=12.5m
# xloc[m] #yloc[m] #c[km/s] #eik[ms]
# Marmousi hmin = 12.5m
# 9.200000000000000000e+03	1.687500000000000000e+03	3.500000000000000444e+00	1.122692634216587066e+03
# 5.662500000000000000e+03	0.000000000000000000e+00	3.859560000000000102e+00	1.039170607108715558e+03


