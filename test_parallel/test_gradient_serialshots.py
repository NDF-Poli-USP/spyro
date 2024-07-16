from mpi4py.MPI import COMM_WORLD
import numpy as np
import firedrake as fire
import random
import spyro
import warnings


warnings.filterwarnings("ignore")


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
    "Lz": 3.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "Lx": 3.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}

dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-1.1, 1.3), (-1.1, 1.7)],
    "frequency": 5.0,
    "delay": 1.5,
    "delay_type": "multiples_of_minimun",
    "receiver_locations": spyro.create_transect((-1.8, 1.2), (-1.8, 1.8), 10),
}

dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": 0.0005,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM    - Perguntar Daiane 'gradient_sampling_frequency'
}

dictionary["visualization"] = {
    "forward_output": True,
    "forward_output_filename": "results/forward_true.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": "results/Gradient.pvd",
    "adjoint_output": False,
    "adjoint_filename": None,
    "debug_output": False,
}


def get_gradient(parallelism_type, points):

    dictionary["parallelism"]["type"] = parallelism_type
    print(f"Calculating exact", flush=True)
    Wave_obj_exact = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_exact.set_mesh(mesh_parameters={"dx": 0.1})

    cond = fire.conditional(Wave_obj_exact.mesh_z > -1.5, 1.5, 3.5)
    Wave_obj_exact.set_initial_velocity_model(
        conditional=cond,
    )

    Wave_obj_exact.forward_solve()

    print(f"Calculating guess", flush=True)
    Wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_guess.set_mesh(mesh_parameters={"dx": 0.1})
    Wave_obj_guess.set_initial_velocity_model(constant=2.0)
    Wave_obj_guess.forward_solve()

    if parallelism_type == "automatic":
        misfit = Wave_obj_exact.forward_solution_receivers - Wave_obj_guess.forward_solution_receivers
    elif parallelism_type == "spatial":
        misfit_list = []
        for source_id in range(len(dictionary["acquisition"]["source_locations"])):
            spyro.io.switch_serial_shot(Wave_obj_exact, source_id)
            spyro.io.switch_serial_shot(Wave_obj_guess, source_id)
            misfit_list.append(Wave_obj_exact.forward_solution_receivers - Wave_obj_guess.forward_solution_receivers)
        misfit= misfit_list

    gradient = Wave_obj_guess.gradient_solve(misfit=misfit)
    Wave_obj_guess.comm.comm.barrier()
    spyro.io.delete_tmp_files(Wave_obj_guess)
    spyro.io.delete_tmp_files(Wave_obj_exact)

    gradient_point_values = []
    for point in points:
        gradient_point_values.append(gradient.at(point))

    return gradient_point_values


def test_gradient_serialshots():
    comm = COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        points = [(random.uniform(-3, 0), random.uniform(0, 3)) for _ in range(20)]
    else:
        points = None
    points = comm.bcast(points, root=0)
    gradient_ensemble_parallelism = get_gradient("automatic", points)
    gradient_serial_shot = get_gradient("spatial", points)

    # Check if the gradients are equal within a tolerance
    tolerance = 1e-8
    test = all(np.isclose(a, b, atol=tolerance) for a, b in zip(gradient_ensemble_parallelism, gradient_serial_shot))

    print(f"Gradient is equal: {test}", flush=True)


if __name__ == "__main__":
    test_gradient_serialshots()
