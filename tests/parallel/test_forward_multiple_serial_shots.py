from mpi4py.MPI import COMM_WORLD
from mpi4py import MPI
from firedrake import conditional
from numpy import linspace
import matplotlib.pyplot as plt
import spyro
from spyro.io.basicio import parallel_print as pprint
from spyro.tools.error_measure import MeasureError


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

    wave = spyro.AcousticWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    wave.store_forward_time_steps = True

    mesh_z = wave.mesh_z
    cond = conditional(mesh_z < -1.5, 3.5, 1.5)
    wave.set_initial_velocity_model(conditional=cond, output=True)

    wave.forward_solve()

    comm = wave.comm

    if comm.comm.rank == 0:
        analytical_p = spyro.utils.nodal_homogeneous_analytical(
            wave, 0.2, 1.5, n_extra=100
        )
    else:
        analytical_p = None
    analytical_p = comm.comm.bcast(analytical_p, root=0)

    time_vector = linspace(0.0, 1.0, 2001)
    cutoff = 830
    eNRMS = []
    errIt = []
    errPk = []

    for i in range(wave.number_of_sources):
        plt.close()
        plt.plot(time_vector[:cutoff], analytical_p[:cutoff], "--", label="analyt")
        spyro.io.switch_serial_shot(wave, i)
        rec_out = wave.forward_solution_receivers
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

        # Computing errors
        eNRMS_core = MeasureError().normalized_root_mean_square_error(
            rec0[:cutoff], analytical_p[:cutoff])
        errIt_core = MeasureError().integral_error(
            rec0[:cutoff], analytical_p[:cutoff], wave.dt)
        errPk_core = MeasureError().peak_error(
            rec0[:cutoff], analytical_p[:cutoff])[0]

        eNRMS_shot = COMM_WORLD.allreduce(eNRMS_core, op=MPI.SUM) / comm.comm.size
        errIt_shot = COMM_WORLD.allreduce(errIt_core, op=MPI.SUM) / comm.comm.size
        errPk_shot = COMM_WORLD.allreduce(errPk_core, op=MPI.SUM) / comm.comm.size

        eNRMS.append(eNRMS_shot)
        errIt.append(errIt_shot)
        errPk.append(errPk_shot)

        pprint(f"Shot {i} produced NRMS error of {eNRMS_shot:.4e}", comm=comm)
        pprint(f"Shot {i} produced Integral error of {errIt_shot:.4e}", comm=comm)
        pprint(f"Shot {i} produced Peak error of {errPk_shot:.4e}", comm=comm)

    eNRMS = sum(eNRMS) / 3
    errIt = sum(errIt) / 3
    errPk = sum(errPk) / 3

    comm.comm.barrier()

    assert abs(eNRMS) < 0.01 and abs(errIt) < 0.01 and abs(errPk) < 0.01, \
        f"Error is too high for forward test with multiple shots."

    pprint(f"Combined NRMS error for all shots is {eNRMS:.4e} and test "
           f"has passed equals {abs(eNRMS) < 0.01}", comm=comm)
    pprint(f"Combined Integral error for all shots is {errIt:.4e} and test "
           f"has passed equals {abs(errIt) < 0.01}", comm=comm)
    pprint(f"Combined Peak error for all shots is {errPk:.4e} and test "
           f"has passed equals {abs(errPk) < 0.01}", comm=comm)


if __name__ == "__main__":
    test_forward_3_shots()
