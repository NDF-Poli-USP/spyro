from mpi4py.MPI import COMM_WORLD
from mpi4py import MPI
import numpy as np
import firedrake as fire
import spyro


def error_calc(p_numerical, p_analytical, nt):
    norm = np.linalg.norm(p_numerical, 2) / np.sqrt(nt)
    error_time = np.linalg.norm(p_analytical - p_numerical, 2) / np.sqrt(nt)
    div_error_time = error_time / norm
    return div_error_time


def test_forward_3_shots():
    final_time = 1.0

    dictionary = {}
    dictionary["options"] = {
        "cell_type": "Q",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "variant": "lumped",  # lumped, equispaced or DG, default is lumped
        "degree": 4,  # p order
        "dimension": 2,  # dimension
    }
    dictionary["parallelism"] = {
        "type": "spatial",  # options: automatic (same number of cores for evey processor) or spatial
        "shot_ids_per_propagation": [[0], [1], [2]],
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
        "source_locations": spyro.create_transect((-0.55, 0.7), (-0.55, 1.3), 2),
        "frequency": 5.0,
        "delay": 0.2,
        "delay_type": "time",
        "receiver_locations": spyro.create_transect((-0.75, 0.7), (-0.75, 1.3), 200),
    }
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": final_time,  # Final time for event
        "dt": 0.0005,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
        "gradient_sampling_frequency": 1,
    }
    dictionary["visualization"] = {
        "forward_output": False,
        "forward_output_filename": "results/forward_output.pvd",
        "fwi_velocity_model_output": False,
        "velocity_model_filename": None,
        "gradient_output": False,
        "gradient_filename": None,
    }

    Wave_obj = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj.set_mesh(mesh_parameters={"dx": 0.1})

    mesh_z = Wave_obj.mesh_z
    cond = fire.conditional(mesh_z < -1.5, 3.5, 1.5)
    Wave_obj.set_initial_velocity_model(conditional=cond, output=True)

    Wave_obj.forward_solve()

    comm = Wave_obj.comm

    arr = Wave_obj.receivers_output

    if comm.comm.rank == 0:
        analytical_p = spyro.utils.nodal_homogeneous_analytical(
            Wave_obj, 0.2, 1.5, n_extra=100
        )
    else:
        analytical_p = None
    analytical_p = comm.comm.bcast(analytical_p, root=0)

    # Checking if error before reflection matches
    if comm.ensemble_comm.rank == 0:
        rec_id = 0
    elif comm.ensemble_comm.rank == 1:
        rec_id = 150
    elif comm.ensemble_comm.rank == 2:
        rec_id = 300

    arr0 = arr[:, rec_id]
    arr0 = arr0.flatten()

    error = error_calc(arr0[:430], analytical_p[:430], 430)
    if comm.comm.rank == 0:
        print(f"Error for shot {Wave_obj.current_sources} is {error} and test has passed equals {np.abs(error) < 0.01}", flush=True)
    error_all = COMM_WORLD.allreduce(error, op=MPI.SUM)
    error_all /= 3

    test = np.abs(error_all) < 0.01

    assert test


if __name__ == "__main__":
    test_forward_3_shots()
