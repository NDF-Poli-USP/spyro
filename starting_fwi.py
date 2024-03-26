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


def check_gradient(Wave_obj_guess, dJ, rec_out_exact, Jm, plot=False):
    """
    Function to check the accuracy of the gradient computation using finite difference method.

    Parameters:
    ------------
    Wave_obj_guess:
        The wave object representing the initial guess.
    dJ:
        The derivative of the objective functional with respect to the model parameters.
    rec_out_exact:
        The exact output of the receivers.
    Jm:
        The value of the objective functional for the initial guess.
    plot:
        A boolean indicating whether to plot the error.

    Returns:
    --------
    None
    """
    steps = [1e-3, 1e-4, 1e-5]  # step length

    errors = []
    V_c = Wave_obj_guess.function_space
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
        misfit_plusdm = rec_out_exact - Wave_obj_guess.receivers_output
        J_plusdm = spyro.utils.compute_functional(Wave_obj_guess, misfit_plusdm)

        grad_fd = (J_plusdm - Jm) / (step)
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

    test1 = all(abs(error) < 1 for error in errors)
    print(f"Gradient error less than 1 percent: {test1}")

    # Checking if error follows expected finite difference error convergence
    test2 = math.isclose(np.log(theory[-1]), np.log(errors[-1]), rel_tol=1e-1)

    print(f"Gradient error behaved as expected: {test2}")

    assert all([test1, test2])


def mask_gradient(Wave_obj, dJ, zlim, xlim):
    mesh_z = Wave_obj.mesh_z
    mesh_x = Wave_obj.mesh_x
    zlim_lower, zlim_upper = zlim
    xlim_lower, xlim_upper = xlim
    mask = fire.Function(Wave_obj.function_space)
    cond = fire.conditional(mesh_z < zlim_lower, 1, mask)
    mask.interpolate(cond)
    cond = fire.conditional(mesh_z > zlim_upper, 1, mask)
    mask.interpolate(cond)
    cond = fire.conditional(mesh_x < xlim_lower, 1, mask)
    mask.interpolate(cond)
    cond = fire.conditional(mesh_x > xlim_upper, 1, mask)
    mask.interpolate(cond)
    mask_dofs = np.where(mask.dat.data[:] > 0.5)
    return mask_dofs


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
    "Lz": 3.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "Lx": 3.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": spyro.create_transect((-1.1, 1.2), (-1.1, 1.8), 8),
    # "source_locations": [(-1.1, 1.5)],
    # "source_locations": [(-1.1, 1.5)],
    "frequency": 5.0,
    "delay": 0.2,
    "delay_type": "time",
    "receiver_locations": spyro.create_transect((-1.9, 1.2), (-1.9, 1.8), 300),
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
dictionary["inversion"] = {
    "perform_fwi": True,  # switch to true to make a FWI
    "initial_guess_model_file": None,
    "shot_record_file": None,
}


def test_fwi(load_real_shot=False):
    """
    Run the Full Waveform Inversion (FWI) test.

    Parameters
    ----------
        load_real_shot (bool, optional): Whether to load a real shot record or not. Defaults to False.
    """

    # Setting up to run synthetic real problem
    if load_real_shot is False:
        FWI_obj = spyro.FullWaveformInversion(dictionary=dictionary)
        comm = FWI_obj.comm

        FWI_obj.set_real_mesh(mesh_parameters={"dx": 0.05})
        center_z = -1.5
        center_x = 1.5
        mesh_z = FWI_obj.mesh_z
        mesh_x = FWI_obj.mesh_x
        cond = fire.conditional((mesh_z-center_z)**2 + (mesh_x-center_x)**2 < .2**2, 3.0, 2.5)

        FWI_obj.set_real_velocity_model(conditional=cond, output=True)
        FWI_obj.generate_real_shot_record()
        np.save("real_shot_record", FWI_obj.real_shot_record)

        spyro_shots = FWI_obj.real_shot_record

    else:
        dictionary["inversion"]["shot_record_file"] = "real_shot_record.npy"
        FWI_obj = spyro.FullWaveformInversion(dictionary=dictionary)

    # Setting up initial guess problem
    FWI_obj.set_guess_mesh(mesh_parameters={"dx": 0.05})
    FWI_obj.set_guess_velocity_model(constant=2.5)

    # Getting functional
    # Jm = FWI_obj.get_functional()
    # print(f"Functional :{Jm}")

    # Calculating gradient
    FWI_obj.get_gradient(save=True)
    dJ = FWI_obj.gradient
    mask = mask_gradient(FWI_obj, dJ, (-1.9, -1.1), (1.1, 1.9))
    if FWI_obj.comm.comm.rank == 0:
        np.save("gradient.npy", dJ.dat.data[:])
    dJ.dat.data[mask] = 0.0
    gradfile = fire.File("Gradient.pvd")
    gradfile.write(dJ)
    # check_gradient(
    #     FWI_obj,
    #     FWI_obj.gradient,
    #     FWI_obj.real_shot_record,
    #     FWI_obj.functional,
    #     plot=True,
    # )

    # Running the optimization

    print("END", flush=True)


if __name__ == "__main__":
    test_fwi(load_real_shot=False)
