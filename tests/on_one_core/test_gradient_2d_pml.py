import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from firedrake import VTKFile
import firedrake as fire
import spyro
import pytest


class Gradient_mask_for_pml():
    def __init__(self, Wave_obj=None):
        if Wave_obj.abc_active is False:
            pass

        # Gatting necessary data from wave object
        pad = Wave_obj.mesh_parameters.abc_pad_length  # noqa: F841
        z = Wave_obj.mesh_z
        x = Wave_obj.mesh_x
        V = Wave_obj.function_space

        # building firedrake function for mask
        z_min = -(Wave_obj.mesh_parameters.length_z)
        x_min = 0.0
        x_max = Wave_obj.mesh_parameters.length_x
        mask = fire.Function(V)
        cond = fire.conditional(z < z_min, 1, 0)
        cond = fire.conditional(x < x_min, 1, cond)
        cond = fire.conditional(x > x_max, 1, cond)
        mask.interpolate(cond)

        # saving mask dofs
        self.mask_dofs = np.where(mask.dat.data[:] > 0.95)
        print("DEBUG")

    def apply_mask(self, dJ):
        dJ.dat.data[self.mask_dofs] = 0.0
        return dJ


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
        Wave_obj_guess.forward_solve()
        misfit_plusdm = rec_out_exact - Wave_obj_guess.receivers_output
        J_plusdm = spyro.utils.compute_functional(Wave_obj_guess, misfit_plusdm)

        grad_fd = (J_plusdm - Jm) / (step)
        projnorm = fire.assemble(dJ * dm * fire.dx(scheme=Wave_obj_guess.quadrature_rule))

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
    }

    dictionary["parallelism"] = {
        "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
    }

    dictionary["mesh"] = {
        "Lz": 1.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
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
        "delay_type": "multiples_of_minimun",
        "receiver_locations": spyro.create_transect((-0.8, 0.1), (-0.8, 0.9), 10),
    }

    dictionary["time_axis"] = {
        "initial_time": 0.0,  # Initial time for event
        "final_time": final_time,  # Final time for event
        "dt": 0.0002,  # timestep size
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "output_frequency": 100,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
        "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM    - Perguntar Daiane 'gradient_sampling_frequency'
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


def get_forward_model(dictionary=None):

    # Exact model
    Wave_obj_exact = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_exact.set_mesh(input_mesh_parameters={"edge_length": 0.03})
    cond = fire.conditional(Wave_obj_exact.mesh_z > -0.5, 1.5, 3.5)
    Wave_obj_exact.set_initial_velocity_model(
        conditional=cond,
        dg_velocity_model=False,
    )
    spyro.plots.plot_model(Wave_obj_exact, filename="pml_grad_test_model.png", abc_points=[(-0, 0), (-1, 0), (-1, 1), (-0, 1)])
    Wave_obj_exact.forward_solve()
    rec_out_exact = Wave_obj_exact.receivers_output

    # Guess model
    Wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_guess.set_mesh(input_mesh_parameters={"edge_length": 0.03})
    Wave_obj_guess.set_initial_velocity_model(constant=2.0)
    Wave_obj_guess.forward_solve()
    rec_out_guess = Wave_obj_guess.receivers_output

    return rec_out_exact, rec_out_guess, Wave_obj_guess


@pytest.mark.slow
def test_gradient(PML=False):
    dictionary = set_dictionary(PML=PML)
    rec_out_exact, rec_out_guess, Wave_obj_guess = get_forward_model(dictionary=dictionary)
    forward_solution = Wave_obj_guess.forward_solution
    forward_solution_guess = deepcopy(forward_solution)

    misfit = rec_out_exact - rec_out_guess

    Jm = spyro.utils.compute_functional(Wave_obj_guess, misfit)
    print(f"Cost functional : {Jm}")

    # compute the gradient of the control (to be verified)
    dJ = Wave_obj_guess.gradient_solve(misfit=misfit, forward_solution=forward_solution_guess)
    VTKFile("gradient_premask.pvd").write(dJ)
    Mask_data = Gradient_mask_for_pml(Wave_obj=Wave_obj_guess)
    dJ = Mask_data.apply_mask(dJ)
    VTKFile("gradient.pvd").write(dJ)

    check_gradient(Wave_obj_guess, dJ, rec_out_exact, Jm, plot=True)


@pytest.mark.slow
def test_gradient_pml():
    return test_gradient(PML=True)


if __name__ == "__main__":
    test_gradient_pml()
