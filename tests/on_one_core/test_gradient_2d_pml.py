import numpy as np
import matplotlib.pyplot as plt
import firedrake as fire
import spyro
import pytest

def check_gradient(Wave_obj_guess, dJ, rec_out_exact, Jm, plot=False):
    steps = [1e-3]  # step length

    errors = []
    V_c = Wave_obj_guess.function_space
    dm = fire.Function(V_c)
    dm.assign(dJ)

    for step in steps:

        Wave_obj_guess.reset_pressure()
        c_guess = fire.Constant(2.0) + step*dm
        Wave_obj_guess.initial_velocity_model = c_guess
        Wave_obj_guess.forward_solve(compute_functional=False)
        misfit_plusdm = rec_out_exact - Wave_obj_guess.receivers_output
        J_plusdm = spyro.utils.compute_functional(Wave_obj_guess, misfit_plusdm)

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

    # Checking if every error is less than 5 percent

    test1 = (abs(errors[-1]) < 5)
    print(f"Gradient error less than 5 percent: {test1}")
    print(f"Error of {errors}")

    # Checking if error follows expected finite difference error convergence
    # this is not done in PML yet. A samll percentage error is present here and in old spyro
    # test2 = math.isclose(np.log(theory[-1]), np.log(errors[-1]), rel_tol=1e-1)

    # print(f"Gradient error behaved as expected: {test2}")

    assert all([test1])


def set_dictionary(PML=False):
    final_time = 1.0

    dictionary = {}
    dictionary["options"] = {
        "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
        "variant": "lumped",  # lumped, equispaced or DG, default is lumped
        "degree": 4,  # p order
        "dimension": 2,  # dimension
        "automatic_adjoint": False,
    }

    dictionary["parallelism"] = {
        "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
    }

    dictionary["mesh"] = {
        "Lz": 1.0,  # depth in km - always positive
        "Lx": 1.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    }

    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [(-0.1, 0.5)],
        "frequency": 5.0,
        "delay": 1.5,
        "delay_type": "multiples_of_minimum",
        "receiver_locations": spyro.create_transect((-0.8, 0.1), (-0.8, 0.9), 10),
        "use_vertex_only_mesh": False,
    }

    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": final_time,  # Final time for event
        "dt": 0.0002,  # timestep size
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
    if PML:
        dictionary["absorving_boundary_conditions"] = {
            "status": True,
            "damping_type": "PML",
            "exponent": 2,
            "cmax": 4.5,
            "R": 1e-6,
            "pad_length": 0.25,
        }
    return dictionary


def get_forward_model(dictionary=None, automatic_adjoint=False):
    dictionary["options"]["automatic_adjoint"] = automatic_adjoint
    dictionary["acquisition"]["use_vertex_only_mesh"] = automatic_adjoint

    # Exact model
    Wave_obj_exact = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_exact.set_mesh(input_mesh_parameters={"edge_length": 0.03})
    cond = fire.conditional(Wave_obj_exact.mesh_z > -0.5, 1.5, 3.5)
    Wave_obj_exact.set_initial_velocity_model(
        conditional=cond,
        dg_velocity_model=False,
    )
    Wave_obj_exact.forward_solve()
    rec_out_exact = Wave_obj_exact.receivers_output

    # Guess model
    Wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_guess.set_mesh(input_mesh_parameters={"edge_length": 0.03})
    Wave_obj_guess.set_initial_velocity_model(constant=2.0)
    Wave_obj_guess.true_receivers = rec_out_exact
    Wave_obj_guess.compute_functional = automatic_adjoint

    return rec_out_exact, Wave_obj_guess


@pytest.mark.slow
@pytest.mark.parametrize("automatic_adjoint", [True, False])
def test_gradient_pml(automatic_adjoint):
    dictionary = set_dictionary(PML=True)
    rec_out_exact, Wave_obj_guess = get_forward_model(
        dictionary=dictionary,
        automatic_adjoint=automatic_adjoint,
    )

    assert hasattr(Wave_obj_guess, "compute_functional")
    assert hasattr(Wave_obj_guess, "true_receivers")
    assert Wave_obj_guess.compute_functional is automatic_adjoint
    assert Wave_obj_guess.true_receivers is rec_out_exact

    if automatic_adjoint:
        with pytest.raises(
            NotImplementedError,
            match="SpyroReducedFunctional is not supported for PML",
        ):
            Wave_obj_guess.compute_gradient()
        return

    dJ = Wave_obj_guess.compute_gradient()

    assert Wave_obj_guess.forward_solution_receivers is not None
    assert Wave_obj_guess.receivers_output is not None
    assert len(Wave_obj_guess.forward_solution) > 0

    Jm = spyro.utils.compute_functional(
        Wave_obj_guess,
        rec_out_exact - Wave_obj_guess.forward_solution_receivers,
    )

    mask = spyro.utils.Gradient_mask_for_pml(Wave_obj_guess)
    dJ = mask.apply_mask(dJ)

    check_gradient(Wave_obj_guess, dJ, rec_out_exact, Jm, plot=False)


if __name__ == "__main__":
    test_gradient_pml(automatic_adjoint=True)
