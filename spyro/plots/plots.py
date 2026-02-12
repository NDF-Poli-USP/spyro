# from scipy.io import savemat
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import firedrake
import copy
from ..io import ensemble_save
from spyro.utils.stats_tools import coeff_of_determination
plt.rcParams.update({"font.family": "serif"})
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \usepackage{amsmath}'
__all__ = ["plot_shots", "plot_hist_receivers"]


@ensemble_save
def plot_shots(
    Wave_object,
    show=False,
    file_name="plot_of_shot",
    shot_ids=[0],
    vmin=-1e-5,
    vmax=1e-5,
    contour_lines=700,
    file_format="pdf",
    start_index=0,
    end_index=0,
    out_index=None,
):
    """
    Plot a shot record and save the image to disk. Note that
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
    file_name = file_name + str(shot_ids) + "." + file_format
    num_recvs = Wave_object.number_of_receivers

    dt = Wave_object.dt
    tf = Wave_object.final_time

    if out_index is None:
        arr = Wave_object.receivers_output
    else:
        arr = Wave_object.receivers_output[:, :, out_index]

    nt = int(tf / dt) + 1  # number of timesteps

    if end_index == 0:
        end_index = num_recvs

    x_rec = np.linspace(start_index, end_index, num_recvs)
    t_rec = np.linspace(0.0, tf, nt)
    X, Y = np.meshgrid(x_rec, t_rec)

    cmap = plt.get_cmap("gray")
    plt.contourf(X, Y, arr, contour_lines, cmap=cmap, vmin=vmin, vmax=vmax)
    # savemat("test.mat", {"mydata": arr})
    plt.xlabel("receiver number", fontsize=18)
    plt.ylabel("time (s)", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(start_index, end_index)
    plt.ylim(tf, 0)
    plt.subplots_adjust(left=0.18, right=0.95, bottom=0.14, top=0.95)
    plt.savefig(file_name, format=file_format)
    # plt.axis("image")
    if show:
        plt.show()
    # plt.close()
    return None


def plot_mesh_sizes(
    mesh_filename=None,
    firedrake_mesh=None,
    title_str=None,
    output_filename=None,
    show=False,
    show_size_contour=True,
):
    # plt.rcParams['font.family'] = "Times New Roman"
    plt.rcParams['font.size'] = 12

    if mesh_filename is not None:
        mesh = firedrake.Mesh(mesh_filename)
    elif firedrake_mesh is not None:
        mesh = firedrake_mesh
    else:
        raise ValueError("Please specify mesh")

    coordinates = copy.deepcopy(mesh.coordinates.dat.data)

    mesh.coordinates.dat.data[:, 0] = coordinates[:, 1]
    mesh.coordinates.dat.data[:, 1] = coordinates[:, 0]

    DG0 = firedrake.FunctionSpace(mesh, "DG", 0)
    f = firedrake.interpolate(firedrake.CellSize(mesh), DG0)

    fig, axes = plt.subplots()
    if show_size_contour:
        im = firedrake.tricontourf(f, axes=axes)
    else:
        im = firedrake.triplot(mesh, axes=axes)

    axes.set_aspect("equal", "box")
    plt.xlabel("X (km)")
    plt.ylabel("Z (km)")
    plt.title(title_str)

    if show_size_contour:
        cbar = fig.colorbar(im, orientation="horizontal")
        cbar.ax.set_xlabel("circumcircle radius (km)")
    fig.set_size_inches(13, 10)
    if show:
        plt.show()
    if output_filename is not None:
        plt.savefig(output_filename)

    # Flip back mesh coordinates so it does not change outside of method
    coordinates = copy.deepcopy(mesh.coordinates.dat.data)

    mesh.coordinates.dat.data[:, 0] = coordinates[:, 1]
    mesh.coordinates.dat.data[:, 1] = coordinates[:, 0]


def plot_model(Wave_object, filename="model.png",
               abc_points=None, show=False, flip_axis=True):
    """
    Plot the model with source and receiver locations.

    Parameters
    -----------
    Wave_object:
        The Wave object containing the model and locations.
    filename (optional):
        The filename to save the plot (default: "model.png").
    abc_points (optional):
        List of points to plot an ABC line (default: None).
    """
    plt.close()
    fig = plt.figure(figsize=(9, 9))
    axes = fig.add_subplot(111)
    fig.set_figwidth = 9.0
    fig.set_figheight = 9.0
    vp_object = Wave_object.initial_velocity_model
    vp_image = firedrake.tripcolor(vp_object, axes=axes)
    for source in Wave_object.source_locations:
        z, x = source
        plt.scatter(z, x, c="green")
    for receiver in Wave_object.receiver_locations:
        z, x = receiver
        plt.scatter(z, x, c="red")

    if flip_axis:
        axes.invert_yaxis()

    axes.set_xlabel("Z (km)")

    if flip_axis:
        axes.set_ylabel("X (km)", rotation=-90, labelpad=20)
        plt.setp(axes.get_xticklabels(), rotation=-90, va="top", ha="center")
        plt.setp(axes.get_yticklabels(), rotation=-90, va="center", ha="left")
    else:
        axes.set_ylabel("X (km)")

    cbar = plt.colorbar(vp_image, orientation="horizontal")
    cbar.set_label("Velocity (km/s)")
    if flip_axis:
        cbar.ax.tick_params(rotation=-90)
    axes.tick_params(axis='y', pad=20)
    axes.axis('equal')

    if abc_points is not None:
        zs = []
        xs = []

        first = True
        for point in abc_points:
            z, x = point
            zs.append(z)
            xs.append(x)
            if first:
                z_first = z
                x_first = x
            first = False
        zs.append(z_first)
        xs.append(x_first)
        plt.plot(zs, xs, "--")
    print(f"File name {filename}", flush=True)
    plt.savefig(filename)

    if flip_axis:
        img = Image.open(filename)
        img_rotated = img.rotate(90)

        # Save the rotated image
        img_rotated.save(filename)
    if show:
        plt.show()
    else:
        plt.close()


def plot_function(function):
    plt.close()
    fig = plt.figure(figsize=(9, 9))
    axes = fig.add_subplot(111)
    fig.set_figwidth = 9.0
    fig.set_figheight = 9.0
    firedrake.tricontourf(function, axes=axes)
    axes.axis('equal')


def debug_plot(function, filename="debug.png"):
    plot_function(function)
    plt.savefig(filename)


def debug_pvd(function, filename="debug.pvd"):
    out = firedrake.VTKFile(filename)
    out.write(function)


def plot_hist_receivers(Wave_object, show=False):
    '''
    Plot the comparison of the time-domain response at the
    receivers between the reference model and the HABC scheme.
    The plots are saved in PDF and PNG formats.

    Parameters
    ----------
    Wave_object: `wave`
        The Wave object containing the simulation results.
    show: `bool`, optional
        Whether to show the plot. Default is False.

    Returns
    -------
    None
    '''

    print("\nPlotting Time Comparison")

    # Time data
    dt = Wave_object.dt
    tf = Wave_object.final_time
    nt = int(tf / dt) + 1  # number of timesteps
    t_rec = np.linspace(0.0, tf, nt)

    # Setting fonts
    plt.rcParams['font.size'] = 7

    # Setting subplots
    num_recvs = Wave_object.number_of_receivers
    plt.rcParams['axes.grid'] = True
    fig, axes = plt.subplots(nrows=num_recvs, ncols=1)
    fig.subplots_adjust(hspace=0.6)

    # Setting colormap
    cl_rc = (0., 1., 0., 1.)  # RGB-alpha (Green)
    cl_rf = (1., 0., 0., 1.)  # RGB-alpha (Red)

    for rec in range(num_recvs):

        # Plot the receiver data
        rc_dat = Wave_object.receivers_output[:, rec]
        rf_dat = Wave_object.receivers_reference[:, rec]
        axes[rec].plot(t_rec, rc_dat, color=cl_rc, linestyle='-', linewidth=2)
        axes[rec].plot(t_rec, rf_dat, color=cl_rf, linestyle='--', linewidth=2)

        # Adding the receiver number label
        axes[rec].text(0.995, 0.9, "R" + str(rec + 1), fontsize=8.5,
                       transform=axes[rec].transAxes, fontweight='bold',
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(facecolor='none', edgecolor='none'))

        # Centered title
        if rec == num_recvs // 2:
            axes[rec].set_ylabel(r'$sol \; recs$')

        # Hide all the xticks for receiver different of the last one
        hide_xticks = False if rec < num_recvs - 1 else True
        plt.setp(axes[rec].get_xticklabels(), visible=hide_xticks)

        # Axis format
        axes[rec].set_xlim(0, tf)
        axes[rec].ticklabel_format(
            axis='y', style='scientific', scilimits=(-2, 2))
        if rec == num_recvs - 1:
            axes[rec].set_xlabel(r'$t \; (s)$')

    # Saving the plot
    time_str = Wave_object.path_save + Wave_object.case_habc + "/time"
    plt.savefig(time_str + ".png", bbox_inches='tight')
    plt.savefig(time_str + ".pdf", bbox_inches='tight')
    plt.show() if show else None
    plt.close()


def plot_rfft_receivers(Wave_object, fxlim=4., show=False):
    '''
    Plot the comparison of the frequency-domain response at the
    receivers between the reference model and the HABC scheme.
    The plots are saved in PDF and PNG formats.

    Parameters
    ----------
    Wave_object: `wave`
        Wave object containing the simulation results.
    fxlim: `float`, optional
        Factor to set the x-axis limits in the plots realtive to
        the source frequency. Default is 4 and the minimum is 2.
    show: `bool`, optional
        Whether to show the plot. Default is False.

    Returns
    -------
    None
    '''

    print("\nPlotting Frequency Comparison")

    # Frequency data
    f_Nyq = Wave_object.f_Nyq
    f_sou = Wave_object.frequency
    pfft = Wave_object.receivers_out_fft.shape[0] - 1
    df = f_Nyq / pfft
    limf = round(min(max(fxlim, 2.) * f_sou, f_Nyq), 1)
    idx_lim = int(limf / df) + 1
    f_rec = np.linspace(0, df * idx_lim, idx_lim)

    # Setting fonts
    plt.rcParams['font.size'] = 7

    # Setting subplots
    num_recvs = Wave_object.number_of_receivers
    plt.rcParams['axes.grid'] = True
    fig, axes = plt.subplots(nrows=num_recvs, ncols=1)
    fig.subplots_adjust(hspace=0.6)

    # Setting colormap
    cl_rc = (0., 1., 0., 1.)  # RGB-alpha (Green)
    cl_rf = (1., 0., 0., 1.)  # RGB-alpha (Red)

    for rec in range(num_recvs):

        # Plot the receiver data
        rc_dat = Wave_object.receivers_out_fft[:idx_lim, rec]
        rf_dat = Wave_object.receivers_ref_fft[:idx_lim, rec]
        axes[rec].plot(f_rec, rc_dat, color=cl_rc, linestyle='-', linewidth=2)
        axes[rec].plot(f_rec, rf_dat, color=cl_rf, linestyle='--', linewidth=2)

        # Add a vertical line at f_ref and f_sou
        if f_sou == Wave_object.freq_ref:
            f_ref = f_sou
            f_str = r'$f_{ref} = f_{sou}$'
        else:
            f_ref = Wave_object.freq_ref
            f_str = r'$f_{ref}$'
            axes[rec].axvline(
                x=f_sou, color='black', linestyle='-', linewidth=1.25)

        axes[rec].axvline(
            x=f_ref, color='black', linestyle='-', linewidth=1.25)

        # Adding the receiver number label
        axes[rec].text(0.995, 0.9, "R" + str(rec + 1), fontsize=8.5,
                       transform=axes[rec].transAxes, fontweight='bold',
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(facecolor='none', edgecolor='none'))

        # Centered title
        if rec == num_recvs // 2:
            axes[rec].set_ylabel(r'$FFT \; recs_{norm}$')

        # Hide all the xticks for receiver different of the last one
        hide_xticks = False if rec < num_recvs - 1 else True
        plt.setp(axes[rec].get_xticklabels(), visible=hide_xticks)

        # Axis format
        axes[rec].set_xlim(0, limf)
        axes[rec].ticklabel_format(
            axis='y', style='scientific', scilimits=(-2, 2))
        if rec == num_recvs - 1:
            axes[rec].set_xlabel(r'$f \; (Hz)$')

            # Adding the frequency labels
            axes[rec].text(f_ref - limf / 500., axes[rec].get_ylim()[0] * 1.05,
                           f_str, color='black', fontsize=8, fontweight='bold',
                           ha='right', va='bottom')
            axes[rec].text(f_sou + limf / 500., axes[rec].get_ylim()[0] * 1.05,
                           r'$f_{sou}$', color='black', fontsize=8,
                           fontweight='bold', ha='left', va='bottom') \
                if f_sou != Wave_object.freq_ref else None

    # Saving the plot
    time_str = Wave_object.path_save + Wave_object.case_habc + "/freq"
    plt.savefig(time_str + ".png", bbox_inches='tight')
    plt.savefig(time_str + ".pdf", bbox_inches='tight')
    plt.show() if show else None
    plt.close()


def plot_xCR_opt(Wave_object, data_regr_xCR, show=False):
    '''
    Plot the regression curve for the optimal xCR value.

    Parameters
    ----------
    Wave_object: `wave`
        The Wave object containing the simulation results
    data_regr_xCR: `list`
        Data for the regression of the parameter xCR.
        Structure: [xCR, max_errIt, max_errPK, crit_opt]
        - xCR: Values of xCR used in the regression.
          The last value IS the optimal xCR
        - max_errIt: Values of the maximum integral error.
          The last value corresponds to the optimal xCR
        - max_errPK: Values of the maximum peak error.
          The last value corresponds to the optimal xCR
        - crit_opt : Criterion for the optimal heuristic factor.
          * 'error_difference' : Difference between integral and peak errors
          * 'error_integral' : Minimum integral error
    show: `bool`, optional
        Whether to show the plot. Default is False.

    Returns
    -------
    None
    '''

    # Data for regression
    xCR, max_errIt, max_errPk, crit_opt = data_regr_xCR
    xCR_opt = xCR[-1]
    err_opt = max_errIt[-1]
    eq_eI = np.polyfit(xCR[:-1], max_errIt[:-1], 2)
    eq_eP = np.polyfit(xCR[:-1], max_errPk[:-1], 2)

    # Compute R^2 values
    y_eI_true = max_errIt[:-1]
    y_eI_pred = np.polyval(eq_eI, xCR[:-1])
    y_eP_true = max_errPk[:-1]
    y_eP_pred = np.polyval(eq_eP, xCR[:-1])
    p = 2  # Quadratic model (Predictors: x and x^2)
    r2_eI = coeff_of_determination(y_eI_true, y_eI_pred, p)
    r2_eP = coeff_of_determination(y_eP_true, y_eP_pred, p)

    # Format equations
    qua_reg = r'${:.3e} x^{{2}} + {:.3e} x + {:.3e}, R^{{2}} = {:.3f}$'
    eq_str_eI = (
        r'$e_I = $' + qua_reg).format(*eq_eI, r2_eI).replace("+ -", "- ")
    eq_str_eP = (
        r'$e_P = $' + qua_reg).format(*eq_eP, r2_eP).replace("+ -", "- ")

    # Regression points
    plt.plot(xCR[:-1], 100 * np.asarray(max_errIt[:-1]), 'ro',
             label=r'Integral Error: ' + eq_str_eI)
    plt.plot(xCR[:-1], 100 * np.asarray(max_errPk[:-1]), 'bo',
             label=r'Peak Error: ' + eq_str_eP)

    # xCR limits
    xCR_inf, xCR_sup = Wave_object.xCR_bounds[0]

    # Regression curves
    xgraf = np.linspace(xCR_inf, xCR_sup, int((xCR_sup - xCR_inf) / 0.1))
    y_eI = np.polyval(eq_eI, xgraf)
    y_eP = np.polyval(eq_eP, xgraf)
    plt.plot(xgraf, 100 * y_eI, color='r', linestyle='--')
    plt.plot(xgraf, 100 * y_eP, color='b', linestyle='--')

    # Locating the optimal value
    plt.plot([xCR_opt, xCR_opt], [0., 100 * err_opt], 'k-')
    xopt_str = r'Optimized Heuristic Factor: $X^{{*}}_{{C_{{R}}}} = {:.3f}$'
    if round(100 * np.polyval(eq_eI, xCR_opt), 2) == round(
            100 * np.polyval(eq_eP, xCR_opt), 2):
        xopt_str += r' | $e_{{I}} = e_{{P}} = {:.2f}\%$'
        label = xopt_str.format(xCR_opt, 100 * err_opt)
    else:
        xopt_str += r' | $e_{{I}} = {:.2f}\%$ | $e_{{P}} = {:.2f}\%$'
        label = xopt_str.format(xCR_opt, 100 * err_opt, 100 * max_errPk[-1])
    plt.plot(xCR_opt, 100 * err_opt, marker=r'$\ast$', color='k',
             markersize=10, label=label)
    plt.legend(loc="best", fontsize=8.5)

    # Formatting the plot
    max_err = max(max(max_errIt[:-1]), max(max_errPk[:-1]))
    plt.xlim(0, round(xCR_sup, 1) + 0.1)
    plt.ylim(0, round(100 * max_err, 1) + 0.1)
    if crit_opt == 'error_difference':
        str_crt = r' (Criterion: Min $(e_I - e_P)$)'
    elif crit_opt == 'error_integral':
        str_crt = r' (Criterion: Min $e_I$)'

    plt.xlabel(r'$X_{C_{R}}$' + str_crt)
    plt.tight_layout(pad=2)
    plt.ylabel(r'$e_I \; | \; e_P \; (\%)$')

    # Saving the plot
    xcr_str = Wave_object.path_save + Wave_object.case_habc + "/xCR"
    plt.savefig(xcr_str + '.png', bbox_inches='tight')
    plt.savefig(xcr_str + '.pdf', bbox_inches='tight')
    plt.show() if show else None
    plt.close()


def plot_model_in_p1(Wave_object, dx=0.01, filename="model.png", abc_points=None, show=False, flip_axis=True):
    """
    Plots the velocity model of a given wave_object projected into a P1 (linear) finite element discretization.
    This function creates a deep copy of the input Wave_object's configuration, modifies it to use
    a P1 (degree 1) continuous Galerkin, and then generates a plot of the resulting velocity model.
    The plot can be saved to a file and optionally displayed.

    Parameters
    -----------
    Wave_object:
        An instance of a wave simulation object containing the velocity model and configuration.
    dx (float, optional):
        The mesh spacing to use for the new model. Defaults to 0.01.
    filename (str, optional):
        The filename to save the plot image. Defaults to "model.png".
    abc_points (list or None, optional):
        Points for absorbing boundary conditions to be marked on the plot. Defaults to None.
    show (bool, optional):
        Whether to display the plot interactively. Defaults to False.
    flip_axis (bool, optional):
        Whether to flip the plot axes for visualization. Defaults to True.

    Returns
    -------
    The result of the plot_model function.
    """

    # Local import to avoid circular import
    from ..solvers import AcousticWave
    p1_obj_dict = copy.deepcopy(Wave_object.input_dictionary)
    p1_obj_dict["options"]["method"] = "CG"
    p1_obj_dict["options"]["variant"] = "equispaced"
    p1_obj_dict["options"]["degree"] = 1

    new_wave_obj = AcousticWave(dictionary=p1_obj_dict)
    new_wave_obj.set_mesh(input_mesh_parameters={"edge_length": dx})
    new_wave_obj.set_initial_velocity_model(conditional=Wave_object.initial_velocity_model)

    return plot_model(new_wave_obj, filename=filename, abc_points=abc_points, show=show, flip_axis=flip_axis)

def plot_validation_acoustic(time_points, analytical_solution, numerical_solution):
    plt.figure(figsize=(14, 7))
    plt.plot(time_points, analytical_solution, 'r-', label='Analytical', linewidth=2.5)
    plt.plot(time_points, numerical_solution, 'b--', label='Numerical', linewidth=2.0)
    plt.title("Validation of the numerical solution - Acoustic Wave", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"validation_forward_acoustic.png")
