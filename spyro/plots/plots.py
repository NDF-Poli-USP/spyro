# from scipy.io import savemat
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import firedrake
import copy
from ..io import ensemble_save
from ..utils import change_scalar_field_resolution
from spyro.utils.stats_tools import coeff_of_determination

plt.rcParams.update({"font.family": "serif"})
plt.rcParams["text.latex.preamble"] = r"\usepackage{bm} \usepackage{amsmath}"
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
    """Plot shot records and save to disk.

    Creates a contour plot of seismic shot records showing receiver responses
    over time. The plot is automatically saved with a filename that includes
    the shot IDs, and the @ensemble_save decorator handles naming when using
    ensemble parallelism.

    Parameters
    ----------
    Wave_object : Wave
        Wave simulation object containing the shot record data in the
        receivers_output attribute, along with timing and receiver information.
    show : bool, optional
        If True, display the plot interactively. Default is False.
    file_name : str, optional
        Base name for the saved image file (without extension).
        Default is "plot_of_shot".
    shot_ids : list of int, optional
        List of shot IDs to include in the filename. Default is [0].
    vmin : float, optional
        Minimum value for the colorscale. Default is -1e-5.
    vmax : float, optional
        Maximum value for the colorscale. Default is 1e-5.
    contour_lines : int, optional
        Number of contour lines to plot. Default is 700.
    file_format : str, optional
        Output file format, either "pdf" or "png". Default is "pdf".
    start_index : int, optional
        Index of the first receiver to plot. Default is 0.
    end_index : int, optional
        Index of the last receiver to plot. If 0, uses all receivers.
        Default is 0.
    out_index : int, optional
        Index for selecting a specific output dimension from receivers_output.
        If None, uses the entire array. Default is None.

    Returns
    -------
    None
        The function saves the plot to disk and returns None.

    Notes
    -----
    The plot uses a grayscale colormap with time on the y-axis (inverted,
    with 0 at top) and receiver number on the x-axis. The @ensemble_save
    decorator automatically modifies the filename when running with ensemble
    parallelism.

    Examples
    --------
    >>> plot_shots(wave_obj, show=True, file_name="my_shot", shot_ids=[0, 1])
    >>> plot_shots(wave_obj, vmin=-1e-3, vmax=1e-3, file_format="png")
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
    """Plot mesh cell sizes with optional contour visualization.

    Visualizes the mesh structure by plotting cell sizes (circumcircle radii)
    either as a filled contour plot or as a triangular mesh plot. Coordinates
    are swapped (z, x) for proper visualization.

    Parameters
    ----------
    mesh_filename : str, optional
        Path to the mesh file to load. If None, firedrake_mesh must be provided.
    firedrake_mesh : firedrake.Mesh, optional
        A Firedrake mesh object. If None, mesh_filename must be provided.
    title_str : str, optional
        Title for the plot. Default is None.
    output_filename : str, optional
        Path to save the plot. If None, plot is not saved.
    show : bool, optional
        Whether to display the plot. Default is False.
    show_size_contour : bool, optional
        If True, show filled contour of cell sizes. If False, show triangular
        mesh plot. Default is True.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If neither mesh_filename nor firedrake_mesh is specified.

    Notes
    -----
    The function temporarily swaps mesh coordinates for visualization and
    restores them afterwards to avoid side effects.
    """
    # plt.rcParams['font.family'] = "Times New Roman"
    plt.rcParams["font.size"] = 12

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


def plot_model(
    Wave_object,
    filename="model.png",
    abc_points=None,
    show=False,
    flip_axis=True,
    high_resolution=False,
    high_resolution_grid_value=0.01,
):
    """Plot the velocity model with source and receiver locations.

    Creates a visualization of the velocity model using tripcolor plotting,
    overlaying source locations (green) and receiver locations (red). Optionally
    plots absorbing boundary condition (ABC) lines and supports high-resolution
    rendering.

    Parameters
    ----------
    Wave_object : Wave
        The Wave object containing the velocity model, source locations,
        and receiver locations.
    filename : str, optional
        The filename to save the plot. Default is "model.png".
    abc_points : list of tuple, optional
        List of (z, x) coordinate tuples defining the ABC boundary line.
        If provided, a dashed line connecting these points is plotted.
        Default is None.
    show : bool, optional
        Whether to display the plot interactively. Default is False.
    flip_axis : bool, optional
        If True, inverts the y-axis and rotates the saved image by 90 degrees
        for conventional seismic visualization. Default is True.
    high_resolution : bool, optional
        If True, interpolates the velocity model to a finer resolution (0.01 km)
        before plotting. Default is False.
    high_resolution_grid_value: float, optional
        High resolution visualization value. Default is 0.01 km.

    Returns
    -------
    None

    Notes
    -----
    The plot includes:
    - Velocity model as a filled contour
    - Green markers for source locations
    - Red markers for receiver locations
    - Dashed line for ABC boundary (if abc_points provided)
    - Colorbar indicating velocity in km/s
    """
    plt.close()
    fig = plt.figure(figsize=(9, 9))
    axes = fig.add_subplot(111)
    fig.set_figwidth = 9.0
    fig.set_figheight = 9.0
    if high_resolution:
        vp_object, _ = change_scalar_field_resolution(
            Wave_object, high_resolution_grid_value
        )

    else:
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
    axes.tick_params(axis="y", pad=20)
    axes.axis("equal")

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
    """Plot a Firedrake function using filled contour visualization.

    Creates a filled contour plot of a Firedrake function with equal aspect ratio.

    Parameters
    ----------
    function : firedrake.Function
        The Firedrake function to visualize.

    Returns
    -------
    None

    Notes
    -----
    The plot is created but not saved or displayed. Use plt.savefig() or
    plt.show() after calling this function to save or display the result.
    """
    plt.close()
    fig = plt.figure(figsize=(9, 9))
    axes = fig.add_subplot(111)
    fig.set_figwidth = 9.0
    fig.set_figheight = 9.0
    firedrake.tricontourf(function, axes=axes)
    axes.axis("equal")


def debug_plot(function, filename="debug.png"):
    """Quick debug plot of a Firedrake function saved to a file.

    Convenience function that plots a Firedrake function and immediately
    saves it to a PNG file for debugging purposes.

    Parameters
    ----------
    function : firedrake.Function
        The Firedrake function to visualize.
    filename : str, optional
        The filename to save the debug plot. Default is "debug.png".

    Returns
    -------
    None

    See Also
    --------
    plot_function : The underlying plotting function.
    debug_pvd : Alternative debug output using VTK format.
    """
    plot_function(function)
    plt.savefig(filename)


def debug_pvd(function, filename="debug.pvd"):
    """Save a Firedrake function to a VTK file for visualization.

    Exports a Firedrake function in ParaView VTK format (.pvd) for
    detailed visualization and analysis in external tools like ParaView.

    Parameters
    ----------
    function : firedrake.Function
        The Firedrake function to export.
    filename : str, optional
        The filename for the VTK output. Default is "debug.pvd".

    Returns
    -------
    None

    See Also
    --------
    debug_plot : Alternative debug output as PNG image.

    Notes
    -----
    The .pvd format can be opened directly in ParaView for 3D visualization
    and advanced post-processing.
    """
    out = firedrake.VTKFile(filename)
    out.write(function)


def plot_hist_receivers(Wave_object, show=False):
    """Plot time-domain receiver response comparison.

    Creates a multi-panel plot comparing the time-domain response at each
    receiver between the computed solution and a reference solution. Each
    receiver is plotted in its own subplot with the computed solution in
    green and the reference solution in red dashed line.

    Parameters
    ----------
    Wave_object : Wave
        The Wave object containing the simulation results. Must have the
        following attributes:
        - receivers_output: Computed receiver data
        - receivers_reference: Reference receiver data
        - dt: Time step
        - final_time: Final simulation time
        - number_of_receivers: Number of receivers
        - path_save: Directory path for saving plots
        - case_habc: Case name for file naming

    show : bool, optional
        Whether to display the plot interactively. Default is False.

    Returns
    -------
    None

    Notes
    -----
    The function saves two files:
    - {path_save}/{case_habc}/time.png
    - {path_save}/{case_habc}/time.pdf

    The green solid line represents the computed solution (HABC scheme),
    while the red dashed line represents the reference solution.
    """
    print("\nPlotting Time Comparison")

    # Time data
    dt = Wave_object.dt
    tf = Wave_object.final_time
    nt = int(tf / dt) + 1  # number of timesteps
    t_rec = np.linspace(0.0, tf, nt)

    # Setting fonts
    plt.rcParams["font.size"] = 7

    # Setting subplots
    num_recvs = Wave_object.number_of_receivers
    plt.rcParams["axes.grid"] = True
    fig, axes = plt.subplots(nrows=num_recvs, ncols=1)
    fig.subplots_adjust(hspace=0.6)

    # Setting colormap
    cl_rc = (0.0, 1.0, 0.0, 1.0)  # RGB-alpha (Green)
    cl_rf = (1.0, 0.0, 0.0, 1.0)  # RGB-alpha (Red)

    for rec in range(num_recvs):

        # Plot the receiver data
        rc_dat = Wave_object.receivers_output[:, rec]
        rf_dat = Wave_object.receivers_reference[:, rec]
        axes[rec].plot(t_rec, rc_dat, color=cl_rc, linestyle="-", linewidth=2)
        axes[rec].plot(t_rec, rf_dat, color=cl_rf, linestyle="--", linewidth=2)

        # Adding the receiver number label
        axes[rec].text(
            0.995,
            0.9,
            "R" + str(rec + 1),
            fontsize=8.5,
            transform=axes[rec].transAxes,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="none", edgecolor="none"),
        )

        # Centered title
        if rec == num_recvs // 2:
            axes[rec].set_ylabel(r"$sol \; recs$")

        # Hide all the xticks for receiver different of the last one
        hide_xticks = False if rec < num_recvs - 1 else True
        plt.setp(axes[rec].get_xticklabels(), visible=hide_xticks)

        # Axis format
        axes[rec].set_xlim(0, tf)
        axes[rec].ticklabel_format(
            axis="y", style="scientific", scilimits=(-2, 2)
        )
        if rec == num_recvs - 1:
            axes[rec].set_xlabel(r"$t \; (s)$")

    # Saving the plot
    time_str = Wave_object.path_save + Wave_object.case_habc + "/time"
    plt.savefig(time_str + ".png", bbox_inches="tight")
    plt.savefig(time_str + ".pdf", bbox_inches="tight")
    plt.show() if show else None
    plt.close()


def plot_rfft_receivers(Wave_object, fxlim=4.0, show=False):
    """Plot frequency-domain receiver response comparison.

    Creates a multi-panel plot comparing the normalized frequency-domain
    (FFT) response at each receiver between the computed solution and a
    reference solution. Vertical lines indicate the source and reference
    frequencies.

    Parameters
    ----------
    Wave_object : Wave
        Wave object containing the simulation results. Must have the
        following attributes:
        - receivers_out_fft: FFT of computed receiver data
        - receivers_ref_fft: FFT of reference receiver data
        - f_Nyq: Nyquist frequency
        - frequency: Source frequency
        - freq_ref: Reference frequency
        - number_of_receivers: Number of receivers
        - path_save: Directory path for saving plots
        - case_habc: Case name for file naming

    fxlim : float, optional
        Factor to set the x-axis limits relative to the source frequency.
        The plot will show frequencies up to fxlim * source_frequency,
        capped at the Nyquist frequency. Minimum value is 2.
        Default is 4.

    show : bool, optional
        Whether to display the plot interactively. Default is False.

    Returns
    -------
    None

    Notes
    -----
    The function saves two files:
    - {path_save}/{case_habc}/freq.png
    - {path_save}/{case_habc}/freq.pdf

    The green solid line represents the FFT of the computed solution,
    while the red dashed line represents the FFT of the reference solution.
    Black vertical lines mark the source and reference frequencies.
    """
    print("\nPlotting Frequency Comparison")

    # Frequency data
    f_Nyq = Wave_object.f_Nyq
    f_sou = Wave_object.frequency
    pfft = Wave_object.receivers_out_fft.shape[0] - 1
    df = f_Nyq / pfft
    limf = round(min(max(fxlim, 2.0) * f_sou, f_Nyq), 1)
    idx_lim = int(limf / df) + 1
    f_rec = np.linspace(0, df * idx_lim, idx_lim)

    # Setting fonts
    plt.rcParams["font.size"] = 7

    # Setting subplots
    num_recvs = Wave_object.number_of_receivers
    plt.rcParams["axes.grid"] = True
    fig, axes = plt.subplots(nrows=num_recvs, ncols=1)
    fig.subplots_adjust(hspace=0.6)

    # Setting colormap
    cl_rc = (0.0, 1.0, 0.0, 1.0)  # RGB-alpha (Green)
    cl_rf = (1.0, 0.0, 0.0, 1.0)  # RGB-alpha (Red)

    for rec in range(num_recvs):

        # Plot the receiver data
        rc_dat = Wave_object.receivers_out_fft[:idx_lim, rec]
        rf_dat = Wave_object.receivers_ref_fft[:idx_lim, rec]
        axes[rec].plot(f_rec, rc_dat, color=cl_rc, linestyle="-", linewidth=2)
        axes[rec].plot(f_rec, rf_dat, color=cl_rf, linestyle="--", linewidth=2)

        # Add a vertical line at f_ref and f_sou
        if f_sou == Wave_object.freq_ref:
            f_ref = f_sou
            f_str = r"$f_{ref} = f_{sou}$"
        else:
            f_ref = Wave_object.freq_ref
            f_str = r"$f_{ref}$"
            axes[rec].axvline(
                x=f_sou, color="black", linestyle="-", linewidth=1.25
            )

        axes[rec].axvline(x=f_ref, color="black", linestyle="-", linewidth=1.25)

        # Adding the receiver number label
        axes[rec].text(
            0.995,
            0.9,
            "R" + str(rec + 1),
            fontsize=8.5,
            transform=axes[rec].transAxes,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="none", edgecolor="none"),
        )

        # Centered title
        if rec == num_recvs // 2:
            axes[rec].set_ylabel(r"$FFT \; recs_{norm}$")

        # Hide all the xticks for receiver different of the last one
        hide_xticks = False if rec < num_recvs - 1 else True
        plt.setp(axes[rec].get_xticklabels(), visible=hide_xticks)

        # Axis format
        axes[rec].set_xlim(0, limf)
        axes[rec].ticklabel_format(
            axis="y", style="scientific", scilimits=(-2, 2)
        )
        if rec == num_recvs - 1:
            axes[rec].set_xlabel(r"$f \; (Hz)$")

            # Adding the frequency labels
            axes[rec].text(
                f_ref - limf / 500.0,
                axes[rec].get_ylim()[0] * 1.05,
                f_str,
                color="black",
                fontsize=8,
                fontweight="bold",
                ha="right",
                va="bottom",
            )
            (
                axes[rec].text(
                    f_sou + limf / 500.0,
                    axes[rec].get_ylim()[0] * 1.05,
                    r"$f_{sou}$",
                    color="black",
                    fontsize=8,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                )
                if f_sou != Wave_object.freq_ref
                else None
            )

    # Saving the plot
    time_str = Wave_object.path_save + Wave_object.case_habc + "/freq"
    plt.savefig(time_str + ".png", bbox_inches="tight")
    plt.savefig(time_str + ".pdf", bbox_inches="tight")
    plt.show() if show else None
    plt.close()


def plot_xCR_opt(Wave_object, data_regr_xCR, show=False):
    """Plot quadratic regression analysis for optimal xCR parameter.

    Creates a plot showing the quadratic regression of integral and peak
    errors as a function of the heuristic factor xCR, highlighting the
    optimal value based on a specified criterion. The plot includes
    regression equations, R² values, and the optimal xCR marker.

    Parameters
    ----------
    Wave_object : Wave
        The Wave object containing the simulation results and configuration.
        Must have attributes:
        - xCR_bounds: Bounds for the xCR parameter
        - path_save: Directory path for saving plots
        - case_habc: Case name for file naming

    data_regr_xCR : list
        Data for the regression of the parameter xCR with structure:
        [xCR, max_errIt, max_errPk, crit_opt] where:

        - xCR : array-like
            Values of xCR used in the regression. The last value is the
            optimal xCR.
        - max_errIt : array-like
            Values of the maximum integral error at each xCR.
            The last value corresponds to the optimal xCR.
        - max_errPk : array-like
            Values of the maximum peak error at each xCR.
            The last value corresponds to the optimal xCR.
        - crit_opt : str
            Criterion used to determine the optimal xCR.
            Options:

            - 'error_difference': Minimizes difference between integral
              and peak errors
            - 'error_integral': Minimizes integral error

    show : bool, optional
        Whether to display the plot interactively. Default is False.

    Returns
    -------
    None

    Notes
    -----
    The function saves two files:
    - {path_save}/{case_habc}/xCR.png
    - {path_save}/{case_habc}/xCR.pdf

    The plot displays:
    - Red circles and dashed curve for integral error (eI)
    - Blue circles and dashed curve for peak error (eP)
    - Black star marker for the optimal xCR value
    - Quadratic regression equations with R² values
    - Vertical line from x-axis to optimal point
    """
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
    qua_reg = r"${:.3e} x^{{2}} + {:.3e} x + {:.3e}, R^{{2}} = {:.3f}$"
    eq_str_eI = (
        (r"$e_I = $" + qua_reg).format(*eq_eI, r2_eI).replace("+ -", "- ")
    )
    eq_str_eP = (
        (r"$e_P = $" + qua_reg).format(*eq_eP, r2_eP).replace("+ -", "- ")
    )

    # Regression points
    plt.plot(
        xCR[:-1],
        100 * np.asarray(max_errIt[:-1]),
        "ro",
        label=r"Integral Error: " + eq_str_eI,
    )
    plt.plot(
        xCR[:-1],
        100 * np.asarray(max_errPk[:-1]),
        "bo",
        label=r"Peak Error: " + eq_str_eP,
    )

    # xCR limits
    xCR_inf, xCR_sup = Wave_object.xCR_bounds[0]

    # Regression curves
    xgraf = np.linspace(xCR_inf, xCR_sup, int((xCR_sup - xCR_inf) / 0.1))
    y_eI = np.polyval(eq_eI, xgraf)
    y_eP = np.polyval(eq_eP, xgraf)
    plt.plot(xgraf, 100 * y_eI, color="r", linestyle="--")
    plt.plot(xgraf, 100 * y_eP, color="b", linestyle="--")

    # Locating the optimal value
    plt.plot([xCR_opt, xCR_opt], [0.0, 100 * err_opt], "k-")
    xopt_str = r"Optimized Heuristic Factor: $X^{{*}}_{{C_{{R}}}} = {:.3f}$"
    if round(100 * np.polyval(eq_eI, xCR_opt), 2) == round(
        100 * np.polyval(eq_eP, xCR_opt), 2
    ):
        xopt_str += r" | $e_{{I}} = e_{{P}} = {:.2f}\%$"
        label = xopt_str.format(xCR_opt, 100 * err_opt)
    else:
        xopt_str += r" | $e_{{I}} = {:.2f}\%$ | $e_{{P}} = {:.2f}\%$"
        label = xopt_str.format(xCR_opt, 100 * err_opt, 100 * max_errPk[-1])
    plt.plot(
        xCR_opt,
        100 * err_opt,
        marker=r"$\ast$",
        color="k",
        markersize=10,
        label=label,
    )
    plt.legend(loc="best", fontsize=8.5)

    # Formatting the plot
    max_err = max(max(max_errIt[:-1]), max(max_errPk[:-1]))
    plt.xlim(0, round(xCR_sup, 1) + 0.1)
    plt.ylim(0, round(100 * max_err, 1) + 0.1)
    if crit_opt == "error_difference":
        str_crt = r" (Criterion: Min $(e_I - e_P)$)"
    elif crit_opt == "error_integral":
        str_crt = r" (Criterion: Min $e_I$)"

    plt.xlabel(r"$X_{C_{R}}$" + str_crt)
    plt.tight_layout(pad=2)
    plt.ylabel(r"$e_I \; | \; e_P \; (\%)$")

    # Saving the plot
    xcr_str = Wave_object.path_save + Wave_object.case_habc + "/xCR"
    plt.savefig(xcr_str + ".png", bbox_inches="tight")
    plt.savefig(xcr_str + ".pdf", bbox_inches="tight")
    plt.show() if show else None
    plt.close()


def plot_model_in_p1(
    Wave_object,
    dx=0.01,
    filename="model.png",
    abc_points=None,
    show=False,
    flip_axis=True,
):
    """Plot velocity model with P1 finite element projection.

    Creates a visualization of the velocity model by first projecting it onto
    a P1 (piecewise linear) continuous Galerkin finite element space. This is
    useful for visualizing higher-order velocity models in a simpler, linear
    representation.

    Parameters
    ----------
    Wave_object : Wave
        An instance of a wave simulation object containing the velocity model
        and configuration dictionary.
    dx : float, optional
        The mesh spacing (edge length) to use for the P1 discretization.
        Default is 0.01.
    filename : str, optional
        The filename to save the plot image. Default is "model.png".
    abc_points : list of tuple, optional
        List of (z, x) coordinate tuples for absorbing boundary condition
        markers to be plotted. Default is None.
    show : bool, optional
        Whether to display the plot interactively. Default is False.
    flip_axis : bool, optional
        Whether to flip the plot axes for conventional seismic visualization.
        Default is True.

    Returns
    -------
    result
        The return value from the plot_model function.

    Notes
    -----
    This function:
    1. Deep copies the Wave_object's input dictionary
    2. Modifies it to use CG (Continuous Galerkin) method with degree 1
    3. Creates a new AcousticWave object with the modified configuration
    4. Sets up a new mesh with the specified edge length
    5. Projects the original velocity model onto the new P1 space
    6. Calls plot_model to generate the visualization

    See Also
    --------
    plot_model : The underlying plotting function.
    """
    # Local import to avoid circular import
    from ..solvers import AcousticWave

    p1_obj_dict = copy.deepcopy(Wave_object.input_dictionary)
    p1_obj_dict["options"]["method"] = "CG"
    p1_obj_dict["options"]["variant"] = "equispaced"
    p1_obj_dict["options"]["degree"] = 1

    new_wave_obj = AcousticWave(dictionary=p1_obj_dict)
    new_wave_obj.set_mesh(input_mesh_parameters={"edge_length": dx})
    new_wave_obj.set_initial_velocity_model(
        conditional=Wave_object.initial_velocity_model
    )

    return plot_model(
        new_wave_obj,
        filename=filename,
        abc_points=abc_points,
        show=show,
        flip_axis=flip_axis,
    )
