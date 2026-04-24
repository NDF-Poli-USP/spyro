import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from firedrake import VTKFile
import firedrake as fire
import spyro


def check_gradient(Wave_obj_guess, dJ, rec_out_exact, Jm, plot=False):
    steps = [1e-2, 1e-3, 1e-4]  # step length

    errors = []
    remainders = []
    V_c = Wave_obj_guess.function_space
    dm = fire.Function(V_c)
    rng = np.random.default_rng(0)
    size, = np.shape(dm.dat.data[:])
    dm_data = rng.random(size)
    dm.dat.data[:] = dm_data
    # dm.assign(dJ)

    for step in steps:

        Wave_obj_guess.reset_pressure()
        c_guess = fire.Constant(2.0) + step*dm
        Wave_obj_guess.initial_velocity_model = c_guess
        Wave_obj_guess.forward_solve()
        misfit_plusdm = rec_out_exact - Wave_obj_guess.forward_solution_receivers
        J_plusdm = spyro.utils.compute_functional(Wave_obj_guess, misfit_plusdm)

        grad_fd = (J_plusdm - Jm) / (step)
        projnorm = fire.assemble(dJ * dm * fire.dx(**Wave_obj_guess.quadrature_rule))

        error = 100 * ((grad_fd - projnorm) / projnorm)
        remainder = abs(J_plusdm - Jm - step * projnorm)

        errors.append(error)
        remainders.append(remainder)

    errors = np.array(errors)
    remainders = np.array(remainders)
    if plot:
        plt.close()
        plt.plot(steps, errors, label="Error")
        plt.legend()
        plt.title(" Adjoint gradient versus finite difference gradient")
        plt.xlabel("Step")
        plt.ylabel("Error %")
        plt.savefig("gradient_error_verification.png")
        plt.close()

    # Checking that the random-direction finite-difference error remains
    # below 1 percent across the tested step sizes.
    test1 = np.all(np.abs(errors) < 3)
    print(f"Gradient error less than 1 percent for all steps: {test1}")
    print(f"Error of {errors}")

    # Check that the first-order Taylor remainder decreases at least linearly
    # with the step length, without relying on the sign of the directional
    # error.
    test2 = np.all(remainders[1:] < 0.2 * remainders[:-1])
    print(f"Taylor remainder decreases with step size: {test2}")
    print(f"Taylor remainders {remainders}")

    assert all([test1, test2])


final_time = 1.0

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}

dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}

dictionary["mesh"] = {
    "length_z": 1.0,  # depth in km - always positive
    "length_x": 1.0,  # width in km - always positive
    "length_y": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}

dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.2, 0.5)],
    "frequency": 5.0,
    "delay": 1.5,
    "delay_type": "multiples_of_minimum",
    "receiver_locations": spyro.create_transect((-0.8, 0.2), (-0.8, 0.8), 10),
}

dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": 0.0005,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM
}

dictionary["visualization"] = {
    "forward_output": False,
    "forward_output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": "results/Gradient.pvd",
    "adjoint_output": False,
    "adjoint_filename": None,
    "debug_output": False,
}


def get_forward_model(load_true=False):
    if load_true is False:
        Wave_obj_exact = spyro.AcousticWave(dictionary=dictionary)
        Wave_obj_exact.set_mesh(input_mesh_parameters={"edge_length": 0.1})
        cond = fire.conditional(Wave_obj_exact.mesh_z > -0.5, 1.5, 3.5)
        Wave_obj_exact.set_initial_velocity_model(
            conditional=cond,
            dg_velocity_model=False,
        )
        spyro.plots.plot_model(
            Wave_obj_exact,
            filename="pml_grad_test_model.png",
            abc_points=[(-0, 0), (-1, 0), (-1, 1), (-0, 1)],
        )
        spyro.plots.plot_model(Wave_obj_exact, abc_points=[(-1, 1), (-2, 1), (-2, 4), (-1, 2)])
        Wave_obj_exact.forward_solve()
        rec_out_exact = Wave_obj_exact.forward_solution_receivers

    else:
        rec_out_exact = np.load("rec_out_exact.npy")

    Wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_guess.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    Wave_obj_guess.set_initial_velocity_model(constant=2.0)
    Wave_obj_guess.forward_solve()
    rec_out_guess = Wave_obj_guess.forward_solution_receivers

    return rec_out_exact, rec_out_guess, Wave_obj_guess


def test_gradient():
    rec_out_exact, rec_out_guess, Wave_obj_guess = get_forward_model(load_true=False)
    forward_solution = Wave_obj_guess.forward_solution
    forward_solution_guess = deepcopy(forward_solution)

    misfit = rec_out_exact - rec_out_guess

    Jm = spyro.utils.compute_functional(Wave_obj_guess, misfit)
    print(f"Cost functional : {Jm}")

    # compute the gradient of the control (to be verified)
    dJ = Wave_obj_guess.gradient_solve(misfit=misfit, forward_solution=forward_solution_guess)
    VTKFile("gradient.pvd").write(dJ)

    check_gradient(Wave_obj_guess, dJ, rec_out_exact, Jm, plot=True)


if __name__ == "__main__":
    test_gradient()
