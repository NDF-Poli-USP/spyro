from pytest import mark
from mpi4py.MPI import COMM_WORLD
from mpi4py import MPI
from firedrake import conditional
import spyro
from spyro.io.basicio import parallel_print as pprint
from spyro.tools.error_measure import MeasureError


@mark.parallel(6)
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
        "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
    }
    dictionary["mesh"] = {
        "length_z": 3.0,  # depth in km - always positive
        "length_x": 3.0,  # width in km - always positive
        "length_y": 0.0,  # thickness in km - always positive
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    }
    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-1.1, 1.2), (-1.1, 1.5), (-1.1, 1.8)],
        "frequency": 5.0,
        "delay": 0.2,
        "delay_type": "time",
        "receiver_locations": spyro.create_transect((-1.3, 1.2), (-1.3, 1.8), 301),
    }
    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": final_time,  # Final time for event
        "dt": 0.001,  # timestep size
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

    mesh_z = wave.mesh_z
    cond = conditional(mesh_z < -1.5, 3.5, 1.5)
    wave.set_initial_velocity_model(conditional=cond, output=True)

    wave.forward_solve()

    comm = wave.comm

    arr = wave.forward_solution_receivers

    if comm.ensemble_comm.rank == 0:
        analytical_p = spyro.utils.nodal_homogeneous_analytical(
            wave, 0.2, 1.5, n_extra=100
        )
    else:
        analytical_p = None
    analytical_p = comm.ensemble_comm.bcast(analytical_p, root=0)

    # Checking if error before reflection matches
    if comm.ensemble_comm.rank == 0:
        rec_id = 0
    elif comm.ensemble_comm.rank == 1:
        rec_id = 150
    elif comm.ensemble_comm.rank == 2:
        rec_id = 300

    arr0 = arr[:, rec_id]
    arr0 = arr0.flatten()

    # Computing errors
    measure_error = MeasureError()
    errPk = measure_error.peak_error(arr0[:430], analytical_p[:430])[0]
    errIt = measure_error.integral_error(arr0[:430], analytical_p[:430], wave.dt)
    eNRMS = measure_error.normalized_root_mean_square_error(arr0[:430], analytical_p[:430])
    pprint(f"NRMS Error for shot {wave.current_sources} is {eNRMS:.4e} and test "
           f"has passed equals {abs(eNRMS) < 0.01}", comm=comm)
    pprint(f"Integral Error for shot {wave.current_sources} is {errIt:.4e} and test "
           f"has passed equals {abs(errIt) < 0.01}", comm=comm)
    pprint(f"Peak Error for shot {wave.current_sources} is {errPk:.4e} and test "
           f"has passed equals {abs(errPk) < 0.01}", comm=comm)

    error_rm = COMM_WORLD.allreduce(eNRMS, op=MPI.SUM) / 3.
    error_it = COMM_WORLD.allreduce(errIt, op=MPI.SUM) / 3.
    error_pk = COMM_WORLD.allreduce(errPk, op=MPI.SUM) / 3.

    assert abs(error_rm) < 0.01 and abs(error_it) < 0.01 and abs(error_pk) < 0.01, \
        "Error is too high for forward test."


if __name__ == "__main__":
    test_forward_3_shots()
