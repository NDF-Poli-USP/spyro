from mpi4py.MPI import COMM_WORLD
from mpi4py import MPI
import numpy as np
import firedrake as fire
import spyro
import matplotlib.pyplot as plt


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
        "shot_ids_per_propagation": [[0], [1]],
    }
    dictionary["mesh"] = {
        "length_z": 2.0,  # depth in km - always positive
        "length_x": 2.0,  # width in km - always positive
        "length_y": 0.0,  # thickness in km - always positive
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    }
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": spyro.create_transect((-0.55, 0.7), (-0.55, 1.3), 3),
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
        "output_frequency": 100,  # how frequently to output solution to pvds
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
    Wave_obj.set_mesh(input_mesh_parameters={"edge_length": 0.1})

    mesh_z = Wave_obj.mesh_z
    cond = fire.conditional(mesh_z < -1.5, 3.5, 1.5)
    Wave_obj.set_initial_velocity_model(conditional=cond, output=True)

    Wave_obj.forward_solve()

    comm = Wave_obj.comm

    if comm.comm.rank == 0:
        analytical_p = spyro.utils.nodal_homogeneous_analytical(
            Wave_obj, 0.2, 1.5, n_extra=100
        )
    else:
        analytical_p = None
    analytical_p = comm.comm.bcast(analytical_p, root=0)

    time_vector = np.linspace(0.0, 1.0, 2001)
    cutoff = 830
    errors = []

    for i in range(Wave_obj.number_of_sources):
        plt.close()
        plt.plot(time_vector[:cutoff], analytical_p[:cutoff], "--", label="analyt")
        spyro.io.switch_serial_shot(Wave_obj, i)
        rec_out = Wave_obj.forward_solution_receivers
        if i == 0:
            rec0 = rec_out[:, 0].flatten()
        elif i == 1:
            rec0 = rec_out[:, 99].flatten()
        elif i == 2:
            rec0 = rec_out[:, 199].flatten()
        plt.plot(time_vector[:cutoff], rec0[:cutoff], label="numerical")
        plt.title(f"Source {i}")
        plt.legend()
        plt.savefig(f"test{i}.png")
        error_core = error_calc(rec0[:cutoff], analytical_p[:cutoff], cutoff)
        error = COMM_WORLD.allreduce(error_core, op=MPI.SUM)
        error /= comm.comm.size
        errors.append(error)
        print(f"Shot {i} produced error of {error}", flush=True)

    error_all = (errors[0] + errors[1] + errors[2]) / 3
    comm.comm.barrier()

    if comm.comm.rank == 0:
        print(f"Combined error for all shots is {error_all} and test has passed equals {np.abs(error_all) < 0.01}", flush=True)

    test = np.abs(error_all) < 0.01

    assert test


if __name__ == "__main__":
    test_forward_3_shots()
