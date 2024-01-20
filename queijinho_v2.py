import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from firedrake import File
import firedrake as fire
import spyro


def compute_functional(Wave_object, residual):
    """Compute the functional to be optimized.
    Accepts the velocity optionally and uses
    it if regularization is enabled
    """
    num_receivers = Wave_object.number_of_receivers
    dt = Wave_object.dt
    tf = Wave_object.final_time
    # nt = int(tf / dt) + 1  # number of timesteps

    # J = np.zeros((num_receivers))
    # for ti in range(nt):
    #     for rn in range(num_receivers):
    #         first_integral[ti] += residual[ti][rn] ** 2
    J = 0
    for rn in range(num_receivers):
        J += np.trapz(residual[:, rn] ** 2, dx=dt)

    J *= 0.5
    return J


final_time = 1.0

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "method": "MLT",  # (MLT/spectral_quadrilateral/DG_triangle/DG_quadrilateral) You can either specify a cell_type+variant or a method
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}

# Number of cores for the shot. For simplicity, we keep things serial.
# spyro however supports both spatial parallelism and "shot" parallelism.
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the PML. Here we'll assume a 0.75 x 1.50 km
# domain and reserve the remaining 250 m for the Perfectly Matched Layer (PML) to absorb
# outgoing waves on three sides (eg., -z, +-x sides) of the domain.
dictionary["mesh"] = {
    "Lz": 3.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "Lx": 3.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}
# Create a source injection operator. Here we use a single source with a
# Ricker wavelet that has a peak frequency of 8 Hz injected at the center of the mesh.
# We also specify to record the solution at 101 microphones near the top of the domain.
# This transect of receivers is created with the helper function `create_transect`.
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-1.5, 1.02)],
    "frequency": 10.0,
    "delay": 0.1,
    "delay_type": "time",
    # "delay": 1.5,
    # "delay_type": "multiples_of_minimun",
    # "receiver_locations": spyro.create_transect((-2.0, 0.5), (-2.0, 2.5), 100),
    "receiver_locations": spyro.create_transect((-1.0, 1.98), (-2.0, 1.98), 101),
}

# Simulate for 2.0 seconds.
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
    "forward_output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": True,
    "gradient_filename": "results/Gradient.pvd",
    "adjoint_output": True,
    "adjoint_filename": "results/adjoint.pvd",
    "debug_output": True,
}


def test_gradient():
    # beginning of debugging variables
    num_recvs = 100
    dt = 0.0005
    tf = final_time
    show = True
    vabs = 1e-2
    timevector = np.linspace(0.0, tf, 2001)

    # devito_shots = np.load("true_data_camembert.npy")
    # devito_nt, _ = np.shape(devito_shots)
    # devito_timevector = np.linspace(0.0, tf, devito_nt)

    # end of debugging variables

    Wave_obj_exact = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_exact.set_mesh(mesh_parameters={"dx": 0.05})

    center_z = -1.5
    center_x = 1.5
    mesh_z = Wave_obj_exact.mesh_z
    mesh_x = Wave_obj_exact.mesh_x
    cond = fire.conditional((mesh_z-center_z)**2 + (mesh_x-center_x)**2 < .15**2, 3.0, 2.5)
    Wave_obj_exact.set_initial_velocity_model(
        conditional=cond,
        output=True
    )
    spyro.plots.plot_model(
        Wave_obj_exact,
        abc_points = [(-1, 1), (-2, 1), (-2, 2), (-1, 2)]
    )
    Wave_obj_exact.forward_solve()
    forward_solution_exact = Wave_obj_exact.forward_solution
    rec_out_exact = Wave_obj_exact.receivers_output

    # # Saving figures
    # checked_receivers = [0, 25, 50, 75, 100]
    # for i in checked_receivers:
    #     title = "Receiver "+str(i)
    #     figname = "devito_camembert_test_r"+str(i)+".png"
    #     plt.plot(timevector, 100*rec_out_exact[:, i], label="spyro")
    #     plt.plot(devito_timevector, devito_shots[:, i], label="devito")
    #     plt.legend()
    #     plt.title(title)
    #     plt.savefig(figname)
    #     plt.close()

    Wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_guess.set_mesh(mesh_parameters={"dx": 0.05})
    Wave_obj_guess.set_initial_velocity_model(constant=2.5)
    Wave_obj_guess.forward_solve()
    forward_solution = Wave_obj_guess.forward_solution
    forward_solution_guess = deepcopy(forward_solution)
    rec_out_guess = Wave_obj_guess.receivers_output

    misfit = rec_out_guess - rec_out_exact
    # misfit_devito = np.load("misfit_camembert.npy")

    # # Saving figures
    # checked_receivers = [0, 25, 50, 75, 100]
    # for i in checked_receivers:
    #     title = "Misfit "+str(i)
    #     figname = "devito_camembert_misfit_test_r"+str(i)+".png"
    #     plt.plot(timevector, 100*misfit[:, i], label="spyro")
    #     plt.plot(devito_timevector, misfit_devito[:, i], label="devito")
    #     plt.legend()
    #     plt.title(title)
    #     plt.savefig(figname)
    #     plt.close()

    Jm = compute_functional(Wave_obj_guess, misfit)
    print(f"Cost functional : {Jm}")
    # devito_objective = 5880.085343733546
    # Fixing devito objective
    # obj*dt/((dx**2)**2)
    # devito_objective = devito_objective*devito_timevector[1]/(10**4)
    # objective_error = 100*np.abs(Jm-devito_objective)/np.abs(devito_objective)
    # print(f" Error with devito cost functional: {objective_error}\%")

    dJ = Wave_obj_guess.gradient_solve(misfit=misfit, forward_solution=forward_solution_guess)

    File("gradient.pvd").write(dJ)
    gradient = dJ.dat.data[:]

    # steps = [1e-3, 1e-4, 1e-5]  # step length

    # errors = []
    # V_c = Wave_obj_guess.function_space
    # dm = fire.Function(V_c)
    # dm.assign(dJ)

    # for step in steps:

    #     Wave_obj_guess.reset_pressure()
    #     c_guess = fire.Constant(3.0) + step*dm
    #     Wave_obj_guess.initial_velocity_model = c_guess
    #     Wave_obj_guess.forward_solve()
    #     misfit_plusdm = rec_out_exact - Wave_obj_guess.receivers_output
    #     J_plusdm = compute_functional(Wave_obj_guess, misfit_plusdm)

    #     grad_fd = (J_plusdm - Jm) / (step)
    #     projnorm = fire.assemble(dJ * dm * fire.dx(scheme=Wave_obj_guess.quadrature_rule))

    #     error = 100 * ((grad_fd - projnorm) / projnorm)

    #     errors.append(error)
    #     print(f"Error : {error}")
    #     # step /= 2

    # # all errors less than 1 %
    # errors = np.array(errors)
    # assert (np.abs(errors) < 5.0).all()
    print("END")


if __name__ == "__main__":
    test_gradient()
