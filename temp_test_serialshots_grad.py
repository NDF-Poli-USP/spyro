# from mpi4py.MPI import COMM_WORLD
# import debugpy
# debugpy.listen(3000 + COMM_WORLD.rank)
# debugpy.wait_for_client()
import numpy as np
import math
import matplotlib.pyplot as plt
from copy import deepcopy
from firedrake import File
import firedrake as fire
import spyro
import warnings


warnings.filterwarnings("ignore")

def check_gradient(Wave_obj_guess, dJ, rec_out_exact_list, Jm_list, plot=False):
    steps = [1e-3, 1e-4, 1e-5]  # step length

    errors = []
    V_c = Wave_obj_guess.function_space
    dm = fire.Function(V_c)
    size, = np.shape(dm.dat.data[:])
    dm_data = np.random.rand(size)
    # np.save(f"dmdata{COMM_WORLD.rank}", dm_data)
    # dm_data = np.load(f"dmdata{COMM_WORLD.rank}.npy")
    dm.dat.data[:] = dm_data

    for step in steps:

        grad_fd = 0.0
        for snum in range(Wave_obj_guess.number_of_sources):
            Wave_obj_guess.reset_pressure()
            c_guess = fire.Constant(2.0) + step*dm
            Wave_obj_guess.initial_velocity_model = c_guess
            Wave_obj_guess.forward_solve()
            misfit_plusdm = rec_out_exact_list[snum] - Wave_obj_guess.receivers_output
            J_plusdm = spyro.utils.compute_functional(Wave_obj_guess, misfit_plusdm)

            grad_fd += (J_plusdm - Jm_list[snum]) / (step)
        projnorm = fire.assemble(dJ * dm * fire.dx(scheme=Wave_obj_guess.quadrature_rule))

        error = 100 * ((grad_fd - projnorm) / projnorm)

        errors.append(error)

    errors = np.array(errors)

    # Checking if error is first order in step
    theory = [t for t in steps]
    theory = [errors[0] * th / theory[0] for th in theory]
    if plot:
        plt.close()
        plt.plot(steps, errors, label="Error")
        plt.plot(steps, theory, "--", label="first order")
        plt.legend()
        plt.title(" Adjoint gradient versus finite difference gradient")
        plt.xlabel("Step")
        plt.ylabel("Error %")
        plt.savefig("gradient_error_verification.png")
        plt.close()

    # Checking if every error is less than 1 percent

    test1 = abs(errors[-1]) < 1
    print(f"Last gradient error less than 1 percent: {test1}")

    # Checking if error follows expected finite difference error convergence
    test2 = math.isclose(np.log(abs(theory[-1])), np.log(abs(errors[-1])), rel_tol=1e-1)

    print(f"Gradient error behaved as expected: {test2}")

    assert all([test1, test2])


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


def get_forward_model():

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

    rec_exact_list = []
    rec_guess_list = []
    print(f"Sending shot records and guess object", flush=True)
    for propagation_id in range(Wave_obj_exact.number_of_sources):
        spyro.io.switch_serial_shot(Wave_obj_exact, propagation_id)
        rec_exact_list.append(Wave_obj_exact.receivers_output)
        rec_guess_list.append(Wave_obj_guess.receivers_output)

    return rec_exact_list, rec_guess_list, Wave_obj_guess


def test_gradient_serialshots():
    print(f"Starting", flush=True)
    rec_exact_list, rec_guess_list, Wave_obj_guess = get_forward_model()

    Jm_list = []
    print(f"Saving cost functionals", flush=True)
    for propagation_id in range(Wave_obj_guess.number_of_sources):
        misfit = rec_exact_list[propagation_id] - rec_guess_list[propagation_id]
        Jm = spyro.utils.compute_functional(Wave_obj_guess, misfit)
        print(f"Cost functional : {Jm}", flush=True)
        Jm_list.append(Jm)

    # compute the gradient of the control (to be verified)
    print(f"Gradient calculation", flush=True)
    dJ = Wave_obj_guess.gradient_solve()
    File("gradient.pvd").write(dJ)

    check_gradient(Wave_obj_guess, dJ, rec_exact_list, Jm_list, plot=True)


if __name__ == "__main__":
    test_gradient_serialshots()