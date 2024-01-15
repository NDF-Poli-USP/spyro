import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from firedrake import File
import firedrake as fire
import spyro


def calc_error_distribution(
        p_numerical_rec, p_analytical_rec,
        savefig=True,
        fname="error_distribution.png"
    ):
    rec_len, num_rec = np.shape(p_numerical_rec)
    errors = []
    for i in range(num_rec):
        error = error_calc_other(
            p_numerical_rec[:, i],
            p_analytical_rec[:, i],
            rec_len
        )
        errors.append(error)
    plt.close()
    plt.plot(range(num_rec), errors)
    if savefig is True:
        plt.savefig(fname)


def error_calc_other(p_numerical, p_analytical, nt):
    norm = np.linalg.norm(p_numerical, 2) / np.sqrt(nt)
    error_time = np.linalg.norm(p_analytical - p_numerical, 2) / np.sqrt(nt)
    div_error_time = error_time / norm
    return div_error_time


def error_calc(receivers, analytical, dt):
    """
    Calculates the error between the numerical and analytical solutions.

    Parameters
    ----------
    receivers : np.ndarray
        the numerical solution to be evaluated
    analytical : np.ndarray
        the analytical or reference solution
    dt : float
        the time-step used in the numerical solution

    Returns
    -------
    error : float
        the error between the numerical and analytical solutions
    """
    rec_len, num_rec = np.shape(receivers)

    # Interpolate analytical solution into numerical dts
    final_time = dt * (rec_len - 1)
    time_vector_rec = np.linspace(0.0, final_time, rec_len)
    time_vector_ana = np.linspace(0.0, final_time, len(analytical[:, 0]))
    ana = np.zeros(np.shape(receivers))
    for i in range(num_rec):
        ana[:, i] = np.interp(
            time_vector_rec, time_vector_ana, analytical[:, i]
        )

    total_numerator = 0.0
    total_denumenator = 0.0
    for i in range(num_rec):
        diff = receivers[:, i] - ana[:, i]
        diff_squared = np.power(diff, 2)
        numerator = np.trapz(diff_squared, dx=dt)
        ref_squared = np.power(ana[:, i], 2)
        denominator = np.trapz(ref_squared, dx=dt)
        total_numerator += numerator
        total_denumenator += denominator

    squared_error = total_numerator / total_denumenator

    error = np.sqrt(squared_error)
    return error


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


def compute_functional_partial_derivative(Wave_object, residual):
    """Compute the functional to be optimized.
    Accepts the velocity optionally and uses
    it if regularization is enabled
    """
    num_receivers = Wave_object.number_of_receivers
    dt = Wave_object.dt
    tf = Wave_object.final_time
    nt = int(tf / dt) + 1  # number of timesteps

    # first_integral = np.zeros((nt))
    # for ti in range(nt):
    #     for rn in range(num_receivers):
    #         first_integral[ti] += residual[ti][rn]

    # J = np.trapz(first_integral, dx=dt)
    J = 0
    for rn in range(num_receivers):
        J += np.trapz(residual[:, rn], dx=dt)

    return J


def plot_shots(
    num_recvs,
    dt,
    tf,
    shots,
    show=False,
    file_name="1",
    vmin=-1e-5,
    vmax=1e-5,
    file_format="pdf",
    start_index=0,
    end_index=0,
):
    """Plot a shot record and save the image to disk. Note that
    this automatically will rename shots when ensmeble paralleism is
    activated.
    Parameters
    ----------
    model: `dictionary`
        Contains model parameters and options.
    comm:A Firedrake commmunicator
        The communicator you get from calling spyro.utils.mpi_init()
    arr: array-like
        An array in which rows are intervals in time and columns are receivers
    show: `boolean`, optional
        Should the images appear on screen?
    file_name: string, optional
        The name of the saved image
    vmin: float, optional
        The minimum value to plot on the colorscale
    vmax: float, optional
        The maximum value to plot on the colorscale
    file_format: string, optional
        File format, pdf or png
    start_index: integer, optional
        The index of the first receiver to plot
    end_index: integer, optional
        The index of the last receiver to plot
    Returns
    -------
    None
    """

    arr = shots

    nt = int(tf / dt) + 1  # number of timesteps

    if end_index == 0:
        end_index = num_recvs

    x_rec = np.linspace(start_index, end_index, num_recvs)
    t_rec = np.linspace(0.0, tf, nt)
    X, Y = np.meshgrid(x_rec, t_rec)

    cmap = plt.get_cmap("gray")
    plt.contourf(X, Y, arr, 700, cmap=cmap, vmin=vmin, vmax=vmax)
    # savemat("test.mat", {"mydata": arr})
    plt.xlabel("receiver number", fontsize=18)
    plt.ylabel("time (s)", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(start_index, end_index)
    plt.ylim(tf, 0)
    plt.subplots_adjust(left=0.18, right=0.95, bottom=0.14, top=0.95)
    plt.savefig(file_name + "." + file_format, format=file_format)
    # plt.axis("image")
    if show:
        plt.show()
    plt.close()
    return None


final_time = 2.0

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
    "source_locations": [(-0.5, 1.5)],
    "frequency": 5.0,
    "delay": 0.553002223869533-0.3565,
    "delay_type": "time",
    # "receiver_locations": spyro.create_transect((-2.0, 0.5), (-2.0, 2.5), 100),
    "receiver_locations": [(-1.0, 1.5) , (-1.3, 1.5), (-2.0, 1.5), (-2.5, 1.5), (-0.5, 1.0), (-0.5, 1.5)]
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
    "adjoint_output": False,
    "adjoint_filename": None,
    "debug_output": True,
}


def test_gradient():
    # beginning of debugging variables
    num_recvs = 100
    dt = 0.0005
    tf = final_time
    show = True
    vabs = 1e-2
    timevector = np.linspace(0.0, tf, 4001)

    devito_r0 = np.load("devito_r0.npy")
    devito_r1 = np.load("devito_r1.npy")
    devito_r2 = np.load("devito_r2.npy")
    dev_nt, = np.shape(devito_r0)
    devito_timevector = np.linspace(0.0, tf, dev_nt)
    # end of debugging variables

    Wave_obj_exact = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_exact.set_mesh(mesh_parameters={"dx": 0.05})
    # Wave_obj_exact.set_initial_velocity_model(
    #     expression="4.0 + 1.0 * tanh(10.0 * (0.5 - sqrt((x - 1.5) ** 2 + (z + 1.5) ** 2)))",
    #     output=True
    # )
    cond = fire.conditional(Wave_obj_exact.mesh_z > -1.5, 1.5, 3.5)
    Wave_obj_exact.set_initial_velocity_model(
        conditional=cond,
        output=True
    )
    Wave_obj_exact.forward_solve()
    forward_solution_exact = Wave_obj_exact.forward_solution
    rec_out_exact = Wave_obj_exact.receivers_output

    # old_rec_out_exact = np.load("old_real_model_receivers.npy")
    # diff = error_calc(old_rec_out_exact, rec_out_exact, dt)
    # print(f"Difference with old propagation Real = {diff}")
    # calc_error_distribution(rec_out_exact, old_rec_out_exact, fname="real_error_dist.png")

    Wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_guess.set_mesh(mesh_parameters={"dx": 0.05})
    Wave_obj_guess.set_initial_velocity_model(constant=3.0)
    Wave_obj_guess.forward_solve()
    forward_solution = Wave_obj_guess.forward_solution
    forward_solution_guess = deepcopy(forward_solution)
    rec_out_guess = Wave_obj_guess.receivers_output

    old_rec_out_guess = np.load("old_guess_model_receivers.npy")
    diff = error_calc(old_rec_out_guess, rec_out_guess, dt)
    print(f"Difference with old propagation Guess= {diff}")
    calc_error_distribution(rec_out_guess, old_rec_out_guess, fname="guess_error_dist.png")


    misfit = rec_out_exact - rec_out_guess
    # misfit_old = np.load("misfit_old.npy")

    Jm = compute_functional(Wave_obj_guess, misfit)
    print(f"Cost functional : {Jm}")

    # compute the gradient of the control (to be verified)
    # plot_shots(num_recvs, dt, tf, misfit, show=show, vmin=-vabs, vmax=vabs, file_name="misfit_new")
    # plot_shots(num_recvs, dt, tf, misfit_old, show=show, vmin=-vabs, vmax=vabs, file_name="misfit_old")
    # plot_shots(num_recvs, dt, tf, misfit-misfit_old, show=show, vmin=-vabs, vmax=vabs, file_name="diff_misfit")
    dJ = Wave_obj_guess.gradient_solve(misfit=misfit, forward_solution=forward_solution_guess)
    # dJ.dat.data[:] = dJ.dat.data[:] * mask.dat.data[:]
    File("gradient.pvd").write(dJ)
    gradient = dJ.dat.data[:]

    steps = [1e-3, 1e-4, 1e-5]  # step length

    errors = []
    V_c = Wave_obj_guess.function_space
    dm = fire.Function(V_c)
    dm.assign(dJ)

    for step in steps:

        Wave_obj_guess.reset_pressure()
        c_guess = fire.Constant(3.0) + step*dm
        Wave_obj_guess.initial_velocity_model = c_guess
        Wave_obj_guess.forward_solve()
        misfit_plusdm = rec_out_exact - Wave_obj_guess.receivers_output
        J_plusdm = compute_functional(Wave_obj_guess, misfit_plusdm)

        grad_fd = (J_plusdm - Jm) / (step)
        projnorm = fire.assemble(dJ * dm * fire.dx(scheme=Wave_obj_guess.quadrature_rule))

        error = 100 * ((grad_fd - projnorm) / projnorm)

        errors.append(error)
        print(f"Error : {error}")
        # step /= 2

    # all errors less than 1 %
    errors = np.array(errors)
    assert (np.abs(errors) < 5.0).all()
    print("END")


if __name__ == "__main__":
    test_gradient()
