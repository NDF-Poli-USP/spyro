"""General plotting routines for simulation data and diagnostic outputs."""

import copy
from typing import TYPE_CHECKING, List, Optional, Tuple

from firedrake import tripcolor, tricontourf, Function
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ..io import ensemble_save
from ..utils import change_scalar_field_resolution
from .plot_helpers import _finalize_figure

if TYPE_CHECKING:  # Avoinding circular imports lazily
    from ..solvers.wave import Wave


def plot_model(
    wave: "Wave",
    filename: str = "model.png",
    abc_points: Optional[List[Tuple[float, float]]] = None,
    show: bool = False,
    flip_axis: bool = True,
    high_resolution: bool = False,
    high_resolution_grid_value: float = 0.01,
) -> None:
    """
    Plot the velocity model with source and receiver locations.

    Creates a visualization of the velocity model using tripcolor plotting,
    overlaying source locations (green) and receiver locations (red). Optionally
    plots absorbing boundary condition (ABC) lines and supports high-resolution
    rendering.

    Parameters
    ----------
    wave : Wave
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
    high_resolution_grid_value : float, optional
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
    if high_resolution:
        vp_object, _ = change_scalar_field_resolution(wave, high_resolution_grid_value)

    else:
        vp_object = wave.initial_velocity_model
    vp_image = tripcolor(vp_object, axes=axes)
    for source in wave.source_locations:
        z, x = source
        plt.scatter(z, x, c="green")
    for receiver in wave.receiver_locations:
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

    _finalize_figure(fig, filename=filename, show=show)

    if flip_axis:
        img = Image.open(filename)
        img_rotated = img.rotate(90)

        # Save the rotated image
        img_rotated.save(filename)


def plot_model_in_p1(
    wave: "Wave",
    dx: float = 0.01,
    filename: str = "model.png",
    abc_points: Optional[List[Tuple[float, float]]] = None,
    show: bool = False,
    flip_axis: bool = True,
) -> None:
    """
    Plot velocity model with P1 finite element projection.

    Creates a visualization of the velocity model by first projecting it onto
    a P1 (piecewise linear) continuous Galerkin finite element space. This is
    useful for visualizing higher-order velocity models in a simpler, linear
    representation.

    Parameters
    ----------
    wave : Wave
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

    See Also
    --------
    plot_model : The underlying plotting function.

    Notes
    -----
    This function:
    1. Deep copies the wave's input dictionary
    2. Modifies it to use CG (Continuous Galerkin) method with degree 1
    3. Creates a new AcousticWave object with the modified configuration
    4. Sets up a new mesh with the specified edge length
    5. Projects the original velocity model onto the new P1 space
    6. Calls plot_model to generate the visualization
    """
    # Local import to avoid circular import
    from ..solvers import AcousticWave

    p1_obj_dict = copy.deepcopy(wave.input_dictionary)
    p1_obj_dict["options"]["method"] = "CG"
    p1_obj_dict["options"]["variant"] = "equispaced"
    p1_obj_dict["options"]["degree"] = 1

    new_wave_obj = AcousticWave(dictionary=p1_obj_dict)
    new_wave_obj.set_mesh(input_mesh_parameters={"edge_length": dx})
    new_wave_obj.set_initial_velocity_model(conditional=wave.initial_velocity_model)

    return plot_model(
        new_wave_obj,
        filename=filename,
        abc_points=abc_points,
        show=show,
        flip_axis=flip_axis,
    )


@ensemble_save
def plot_shots(
    wave: "Wave",
    show: bool = False,
    filename: str = "plot_of_shot",
    shot_ids: List[int] = [0],
    vmin: float = -1e-5,
    vmax: float = 1e-5,
    contour_lines: int = 700,
    file_format: str = "pdf",
    start_index: int = 0,
    end_index: int = 0,
    out_index: Optional[int] = None,
) -> None:
    """
    Plot shot records and save to disk.

    Creates a contour plot of seismic shot records showing receiver responses
    over time. The plot is automatically saved with a filename that includes
    the shot IDs, and the @ensemble_save decorator handles naming when using
    ensemble parallelism.

    Parameters
    ----------
    wave : Wave
        Wave simulation object containing the shot record data in the
        forward_solution_receivers attribute, along with timing and receiver information.
    show : bool, optional
        If True, display the plot interactively. Default is False.
    filename : str, optional
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
        Index for selecting a specific output dimension from forward_solution_receivers.
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
    >>> plot_shots(wave, show=True, file_name="my_shot", shot_ids=[0, 1])
    >>> plot_shots(wave, vmin=-1e-3, vmax=1e-3, file_format="png")
    """
    filename = filename + str(shot_ids) + "." + file_format
    num_recvs = wave.number_of_receivers

    dt = wave.dt
    tf = wave.final_time

    if out_index is None:
        arr = wave.forward_solution_receivers
    else:
        arr = wave.forward_solution_receivers[:, :, out_index]

    nt = int(tf / dt) + 1  # number of timesteps

    if end_index == 0:
        end_index = num_recvs

    x_rec = np.linspace(start_index, end_index, num_recvs)
    t_rec = np.linspace(0.0, tf, nt)
    X, Y = np.meshgrid(x_rec, t_rec)

    cmap = plt.get_cmap("gray")
    plt.contourf(X, Y, arr, contour_lines, cmap=cmap, vmin=vmin, vmax=vmax)
    fig = plt.gcf()
    plt.xlabel("receiver number", fontsize=18)
    plt.ylabel("time (s)", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(start_index, end_index)
    plt.ylim(tf, 0)
    plt.subplots_adjust(left=0.18, right=0.95, bottom=0.14, top=0.95)
    _finalize_figure(fig, filename=filename, show=show)
    return None


def plot_function(function: Function, **kwargs) -> None:
    """
    Plot a Firedrake function using filled contour visualization.

    Creates a filled contour plot of a Firedrake function with equal aspect ratio.

    Parameters
    ----------
    function : firedrake.Function
        The Firedrake function to visualize.
    kwargs : Same as for matplotlib.tricontourf

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
    contours = tricontourf(function, axes=axes, **kwargs)
    plt.colorbar(contours)
    axes.axis("equal")
