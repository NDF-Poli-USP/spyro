from mpi4py.MPI import COMM_WORLD
from mpi4py import MPI
import spyro
from spyro.io.basicio import parallel_print as pprint
from spyro.tools.error_measure import MeasureError


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
        "shot_ids_per_propagation": [[0, 1]],
    }

    # Define the domain size without the PML. Here we'll assume a 1.00 x 1.00 km
    # domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
    # outgoing waves on three sides (eg., -z, +-x sides) of the domain.
    dictionary["mesh"] = {
        "length_z": 2.0,  # depth in km - always positive
        "length_x": 2.0,  # width in km - always positive
        "length_y": 0.0,  # thickness in km - always positive
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

    wave = spyro.AcousticWave(dictionary=dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.02, "periodic": True})

    wave.set_initial_velocity_model(constant=1.5)
    wave.forward_solve()
    comm = wave.comm

    rec_out = wave.forward_solution_receivers
    if comm.comm.rank == 0:
        analytical_p = spyro.utils.nodal_homogeneous_analytical(wave, 0.2, 1.5, n_extra=100)
    else:
        analytical_p = None

    analytical_p = comm.comm.bcast(analytical_p, root=0)

    arr0 = rec_out[:, 0]
    arr0 = arr0.flatten()
    arr199 = rec_out[:, 199]
    arr199 = arr199.flatten()

    # Computing errors
    measure_error = MeasureError()
    error0_nrms = measure_error.normalized_root_mean_square_error(arr0[:430],
                                                                  analytical_p[:430])
    error199_nrms = measure_error.normalized_root_mean_square_error(arr199[:430],
                                                                    analytical_p[:430])
    error0_it = measure_error.integral_error(arr0[:430], analytical_p[:430], wave.dt)
    error199_it = measure_error.integral_error(arr199[:430], analytical_p[:430], wave.dt)
    error0_pk = measure_error.peak_error(arr0[:430], analytical_p[:430])[0]
    error199_pk = measure_error.peak_error(arr199[:430], analytical_p[:430])[0]

    eNRMS = error0_nrms + error199_nrms
    error_rm = COMM_WORLD.allreduce(eNRMS, op=MPI.SUM) / 2
    errIt = error0_it + error199_it
    error_it = COMM_WORLD.allreduce(errIt, op=MPI.SUM) / 2
    errPk = error0_pk + error199_pk
    error_pk = COMM_WORLD.allreduce(errPk, op=MPI.SUM) / 2

    comm.comm.barrier()

    assert abs(error_rm) < 0.01 and abs(error_it) < 0.01 and abs(error_pk) < 0.01, \
        "Error is too high for forward test with supershot."

    pprint(f"Combined NRMS error for shots {wave.current_sources} is {error_rm} "
           f"and test has passed equals {abs(error_rm) < 0.01}", comm=comm)
    pprint(f"Combined Integral error for shots {wave.current_sources} is {error_it} "
           f"and test has passed equals {abs(error_it) < 0.01}", comm=comm)
    pprint(f"Combined Peak error for shots {wave.current_sources} is {error_pk} "
           f"and test has passed equals {abs(error_pk) < 0.01}", comm=comm)


if __name__ == "__main__":
    test_forward_supershot()
