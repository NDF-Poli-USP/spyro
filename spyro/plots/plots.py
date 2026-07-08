# from scipy.io import savemat
import os
from matplotlib import use
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import firedrake
import copy
from ..io import ensemble_save
from ..domains.space import create_function_space
from ..utils import change_scalar_field_resolution
from ..tools.version_control import is_firedrake_new
plt.rcParams.update({"font.family": "serif"})
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \usepackage{amsmath}'
__all__ = ["plot_shots"]


# Use non-interactive backend in headless/container environments.
# Some containers may have DISPLAY set to ':0' without a real X server,
# so treat an empty DISPLAY or ':0' as headless for safety.
display_val = os.environ.get("DISPLAY", "")
if display_val == "" or display_val == ":0":
    use("Agg")
    print("A", flush=True)
else:
    print(f"display_config = {os.environ.get("DISPLAY", "")}", flush=True)


if is_firedrake_new() is False:
    from firedrake.__future__ import interpolate
    firedrake.interpolate = interpolate


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
    Plot shot records and save to disk.

    Creates a contour plot of seismic shot records showing receiver responses
    over time. The plot is automatically saved with a filename that includes
    the shot IDs, and the @ensemble_save decorator handles naming when using
    ensemble parallelism.

    Parameters
    ----------
    Wave_object : Wave
        Wave simulation object containing the shot record data in the
        forward_solution_receivers attribute, along with timing and receiver information.
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
    >>> plot_shots(wave_obj, show=True, file_name="my_shot", shot_ids=[0, 1])
    >>> plot_shots(wave_obj, vmin=-1e-3, vmax=1e-3, file_format="png")
    """
    file_name = file_name + str(shot_ids) + "." + file_format
    num_recvs = Wave_object.number_of_receivers

    dt = Wave_object.dt
    tf = Wave_object.final_time

    if out_index is None:
        arr = Wave_object.forward_solution_receivers
    else:
        arr = Wave_object.forward_solution_receivers[:, :, out_index]

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
    """
    Plot mesh cell sizes with optional contour visualization.

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

    DG0 = create_function_space(mesh, "DG0", 0)
    f = firedrake.assemble(interpolate(firedrake.CellSize(mesh), DG0))

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
    """
    Plot the velocity model with source and receiver locations.

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
        vp_object, _ = change_scalar_field_resolution(Wave_object, high_resolution_grid_value)

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
    """
    Plot a Firedrake function using filled contour visualization.

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
    axes.axis('equal')


def debug_plot(function, filename="debug.png"):
    """
    Quick debug plot of a Firedrake function saved to a file.

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
    """
    Save a Firedrake function to a VTK file for visualization.

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


def plot_model_in_p1(Wave_object, dx=0.01, filename="model.png", abc_points=None, show=False, flip_axis=True):
    """
    Plot velocity model with P1 finite element projection.

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
    new_wave_obj.set_initial_velocity_model(conditional=Wave_object.initial_velocity_model)

    return plot_model(new_wave_obj, filename=filename, abc_points=abc_points, show=show, flip_axis=flip_axis)
