from mpi4py.MPI import COMM_WORLD
from mpi4py import MPI
# import debugpy
# debugpy.listen(3000 + COMM_WORLD.rank)
# debugpy.wait_for_client()
import spyro
import numpy as np
import math


def error_calc(p_numerical, p_analytical, nt):
    norm = np.linalg.norm(p_numerical, 2) / np.sqrt(nt)
    error_time = np.linalg.norm(p_analytical - p_numerical, 2) / np.sqrt(nt)
    div_error_time = error_time / norm
    return div_error_time


def test_forward_supershot():
    dt = 0.0005

    final_time = 1.0

    dictionary = {}
    dictionary["options"] = {
        "cell_type": "Q",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "variant": "lumped",  # lumped, equispaced or DG, default is lumped "method":"MLT", # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
        "degree": 4,  # p order
        "dimension": 2,  # dimension
    }

    # Number of cores for the shot. For simplicity, we keep things serial.
    # spyro however supports both spatial parallelism and "shot" parallelism.
    dictionary["parallelism"] = {
        "type": "custom",  # options: automatic (same number of cores for evey processor) or spatial
        "shot_ids_per_propagation": [[0, 1, 2]],
    }

    # Define the domain size without the PML. Here we'll assume a 1.00 x 1.00 km
    # domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
    # outgoing waves on three sides (eg., -z, +-x sides) of the domain.
    dictionary["mesh"] = {
        "Lz": 2.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
        "Lx": 2.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    }
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": spyro.create_transect((-0.55, 0.7), (-0.55, 1.3), 2),
        "frequency": 5.0,
        "delay": 0.2,
        "delay_type": "time",
        "receiver_locations": spyro.create_transect((-0.55, 0.5), (-0.55, 1.5), 200),
    }

    # Simulate for 2.0 seconds.
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": final_time,  # Final time for event
        "dt": dt,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds
        "gradient_sampling_frequency": 100,  # how frequently to save solution to RAM
    }

    dictionary["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
    }

    Wave_obj = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj.set_mesh(mesh_parameters={"dx": 0.02, "periodic": True})

    Wave_obj.set_initial_velocity_model(constant=1.5)
    Wave_obj.forward_solve()
    comm = Wave_obj.comm

    rec_out = Wave_obj.receivers_output
    if comm.comm.rank == 0:
        analytical_p = spyro.utils.nodal_homogeneous_analytical(Wave_obj, 0.2, 1.5, n_extra=100)
    else:
        analytical_p = None

    analytical_p = comm.comm.bcast(analytical_p, root=0)

    arr0 = rec_out[:, 0]
    arr0 = arr0.flatten()
    arr199 = rec_out[:, 199]
    arr199 = arr199.flatten()

    error0 = error_calc(arr0[:430], analytical_p[:430], 430)
    error199 = error_calc(arr199[:430], analytical_p[:430], 430)
    error = error0 + error199
    error_all = COMM_WORLD.allreduce(error, op=MPI.SUM)
    error_all /= 2
    comm.comm.barrier()

    if comm.comm.rank == 0:
        print(f"Combined error for shots {Wave_obj.current_sources} is {error_all} and test has passed equals {np.abs(error_all) < 0.01}", flush=True)

    return rec_out


if __name__ == "__main__":
    test_forward_supershot()
