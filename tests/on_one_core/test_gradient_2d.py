import math
from copy import deepcopy
from pyadjoint import annotate_tape

import firedrake as fire
import matplotlib.pyplot as plt
import numpy as np
import pytest
import spyro


def check_gradient(Wave_obj_guess, dJ, plot=False):
    steps = [1e-3, 1e-4, 1e-5]  # step length

    errors = []
    V_c = Wave_obj_guess.function_space
    Jm = Wave_obj_guess.functional_value
    dm = fire.Function(V_c)
    size, = np.shape(dm.dat.data[:])
    dm_data = np.random.rand(size)
    dm.dat.data[:] = dm_data
    # dm.assign(dJ)

    for step in steps:

        Wave_obj_guess.reset_pressure()
        c_guess = fire.Constant(2.0) + step*dm
        Wave_obj_guess.initial_velocity_model = c_guess
        Wave_obj_guess.forward_solve()
        J_plusdm = Wave_obj_guess.functional_value

        grad_fd = (J_plusdm - Jm) / (step)
        projnorm = fire.assemble(dJ * dm * fire.dx(**Wave_obj_guess.quadrature_rule))

        error = np.abs(100 * ((grad_fd - projnorm) / projnorm))

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

    test1 = np.all(errors < 1)
    print(f"Gradient error stays below 1 percent: {test1}")

    assert test1


final_time = 1.0

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": 1,  # p order
    "dimension": 2,  # dimension
}

dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}

dictionary["mesh"] = {
    "length_z": 3.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "length_x": 3.0,  # width in km - always positive
    "length_y": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}

dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.5, 0.5)],
    "frequency": 7.0,
    # "delay": 1.2227264394269568,
    # "delay_type": "time",
    "delay": 1.5,
    "delay_type": "multiples_of_minimum",
    "receiver_locations": spyro.create_transect((-0.6, 0.2), (-0.6, 0.8), 10),
}

dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": 0.002,  # timestep size
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


def get_forward_model(automated_adjoint, load_true=False):

    if load_true is False:
        Wave_obj_exact = spyro.AcousticWave(dictionary=dictionary)
        Wave_obj_exact.set_mesh(input_mesh_parameters={"edge_length": 0.1})
        # Wave_obj_exact.set_initial_velocity_model(constant=3.0)
        cond = fire.conditional(Wave_obj_exact.mesh_z > -1.5, 1.5, 3.5)
        Wave_obj_exact.set_initial_velocity_model(
            conditional=cond,
            # output=True
        )
        spyro.plots.plot_model(Wave_obj_exact, abc_points=[
            (-1, 1), (-2, 1), (-2, 4), (-1, 2)]
            )
        Wave_obj_exact.forward_solve()
        rec_out_exact = Wave_obj_exact.forward_solution_receivers
    else:
        rec_out_exact = np.load("rec_out_exact.npy")

    Wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_guess.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    Wave_obj_guess.set_initial_velocity_model(constant=2.0)
    Wave_obj_guess.real_shot_record = rec_out_exact
    if automated_adjoint:
        Wave_obj_guess.enable_automated_adjoint()
        assert Wave_obj_guess._store_forward_time_steps is False
    else:
        Wave_obj_guess.enable_store_misfit()
        assert Wave_obj_guess._store_misfit is True
        # Store forward solution time steps for gradient calculation
        # is `True` by default in spyro.
        Wave_obj_guess.store_forward_time_steps = True
        Wave_obj_guess.forward_solve()
    return Wave_obj_guess


@pytest.mark.parametrize("automated_adjoint", [False, True])
def test_gradient(automated_adjoint):
    Wave_obj_guess = get_forward_model(automated_adjoint, load_true=False)
    forward_solution_guess = None
    if automated_adjoint:
        assert annotate_tape() is False
    if not automated_adjoint:
        assert isinstance(Wave_obj_guess.misfit, list)
        forward_solution_guess = deepcopy(Wave_obj_guess.forward_solution)
    if automated_adjoint:
        dJ = Wave_obj_guess.gradient_solve()
        assert isinstance(dJ, fire.Function)
        direction = fire.Function(Wave_obj_guess.function_space)
        np.random.seed(1)
        direction.dat.data[:] = np.random.rand(direction.dat.data.size)
        rate = Wave_obj_guess.automated_adjoint.verify_gradient(
            Wave_obj_guess.c,
            direction=direction,
        )
        assert math.isclose(rate, 2.0, rel_tol=1e-2)
    else:
        assert isinstance(forward_solution_guess, list)
        dJ = Wave_obj_guess.gradient_solve(
            forward_solution=forward_solution_guess)
        check_gradient(Wave_obj_guess, dJ, plot=True) 


if __name__ == "__main__":
    test_gradient(automated_adjoint=True)
