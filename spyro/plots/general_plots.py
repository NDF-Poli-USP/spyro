"""General plotting routines for simulation data and diagnostic outputs."""

import copy
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

from firedrake import tripcolor, tricontourf, Function
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
from PIL import Image
from ..io import ensemble_save
from ..utils import change_scalar_field_resolution
from ..utils.physical_parameters import as_list
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
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
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
        forward_solution_receivers attribute, along with timing and receiver info.
    show : bool, optional
        If True, display the plot interactively. Default is False.
    filename : str, optional
        Base name for the saved image file (without extension).
        Default is "plot_of_shot".
    shot_ids : list of int, optional
        List of shot IDs to include in the filename. Default is [0].
    vmin : float, optional
        Minimum value for the colorscale. When neither limit is given, the
        scale is symmetric about zero and spans the largest absolute value
        of the record, so that the plot shows whatever amplitude the data
        have.
    vmax : float, optional
        Maximum value for the colorscale. See ``vmin``.
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

    if vmin is None and vmax is None:
        scale = float(np.max(np.abs(arr)))
        vmin, vmax = -scale, scale

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


def _domain_grid(mesh, spacing: float) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """Build a regular grid of points covering a mesh.

    Parameters
    ----------
    mesh : firedrake.mesh.MeshGeometry
        A two-dimensional mesh, with the depth as its first coordinate.
    spacing : float
        Distance between grid points, in the mesh's units.

    Returns
    -------
    numpy.ndarray
        The grid points, of shape ``(rows * columns, 2)``, ordered row by
        row from the top of the domain (largest depth coordinate) down, and
        from left to right within a row.
    tuple of float
        ``(rows, columns, x_min, x_max, z_min, z_max)``: the shape of the grid
        and the bounding box of the mesh, gathered over its communicator.

    Notes
    -----
    Collective over the mesh communicator.
    """
    coordinates = mesh.coordinates.dat.data_ro
    # A rank may own no vertices; it then contributes nothing to the box.
    local_min = coordinates.min(axis=0) if coordinates.size else np.full(2, np.inf)
    local_max = coordinates.max(axis=0) if coordinates.size else np.full(2, -np.inf)
    z_min, x_min = (mesh.comm.allreduce(float(v), op=MPI.MIN) for v in local_min)
    z_max, x_max = (mesh.comm.allreduce(float(v), op=MPI.MAX) for v in local_max)
    depths = np.arange(z_max, z_min - spacing / 2, -spacing)
    positions = np.arange(x_min, x_max + spacing / 2, spacing)
    Z, X = np.meshgrid(depths, positions, indexing="ij")
    points = np.column_stack([Z.ravel(), X.ravel()])
    return points, (len(depths), len(positions), x_min, x_max, z_min, z_max)


def plot_scalar_field(
    fields: Union[Function, Sequence[Function]],
    filename: Union[str, Path, None] = None,
    *,
    titles: Optional[Sequence[str]] = None,
    vmin: Union[float, Sequence[Optional[float]], None] = None,
    vmax: Union[float, Sequence[Optional[float]], None] = None,
    cmap: str = "viridis",
    colorbar_label: Optional[str] = None,
    sources: Optional[Sequence[Tuple[float, float]]] = None,
    receivers: Optional[Sequence[Tuple[float, float]]] = None,
    spacing: float = 0.01,
    columns: Optional[int] = None,
    show: bool = False,
) -> plt.Figure:
    """Draw scalar fields as sections: depth downwards, position across.

    Each field is sampled on a regular grid covering its mesh and drawn with
    ``imshow``, one panel per field, side by side or on a grid of
    ``columns`` panels per row. The first mesh coordinate
    is taken as the depth and put on the vertical axis pointing down, the
    second as the horizontal position, which is how a velocity model, or a
    gradient with respect to one, is looked at. Sampling on a grid rather
    than plotting the finite element function directly keeps a high-order
    field smooth in the picture, with no dependence on the element type.

    Parameters
    ----------
    fields : firedrake.Function or sequence of firedrake.Function
        Scalar fields to draw, one panel each, in order.
    filename : str or pathlib.Path, optional
        Where to save the figure. Saved by the first rank of the mesh
        communicator only.
    titles : sequence of str, optional
        One title per panel.
    vmin : float or sequence of float, optional
        Lower limit of the colour scale, for every panel or one per panel.
        A missing limit is taken from the samples of that panel.
    vmax : float or sequence of float, optional
        Upper limit of the colour scale, likewise.
    cmap : str, optional
        Matplotlib colour map. Default is ``"viridis"``.
    colorbar_label : str, optional
        Label of the colour bars, such as the unit of the fields.
    sources : sequence of tuple of float, optional
        Source positions, ``(z, x)``, marked with red stars.
    receivers : sequence of tuple of float, optional
        Receiver positions, ``(z, x)``, marked with white triangles.
    spacing : float, optional
        Distance between the sampling points, in the mesh's units. Default
        is 0.01.
    columns : int, optional
        Panels per row; the panels fill the rows in order. Default is all
        of them on one row.
    show : bool, optional
        Whether to display the figure interactively. Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, closed after saving unless ``show`` is set.

    Raises
    ------
    ValueError
        If a field is not scalar-valued, or the number of titles or limits
        does not match the number of fields.

    Notes
    -----
    Sampling a field is collective over its mesh communicator, so under
    spatial parallelism every rank of that communicator has to call this;
    the figure is saved by its first rank alone.

    Examples
    --------
    >>> plot_scalar_field(
    ...     [wave.c, wave.c_s], "velocities.png",
    ...     titles=["$c_p$", "$c_s$"], colorbar_label="km/s",
    ...     sources=wave.source_locations, receivers=wave.receiver_locations,
    ... )
    """
    # Imported here: point evaluation is recent in Firedrake, and this module
    # is loaded by ``import spyro`` on older versions too.
    from firedrake import PointEvaluator

    fields = as_list(fields)
    count = len(fields)
    if count == 0:
        raise ValueError("At least one field is required.")
    for field in fields:
        if field.ufl_shape != ():
            raise ValueError(
                f"plot_scalar_field draws scalar fields; got shape {field.ufl_shape}."
            )

    def per_panel(value, name):
        """Expand one value, or one per panel, into a list of ``count``."""
        values = list(value) if isinstance(value, (list, tuple)) else [value] * count
        if len(values) != count:
            raise ValueError(
                f"{name} takes one entry per field: {count} fields, "
                f"{len(values)} entries."
            )
        return values

    titles = per_panel(None, "titles") if titles is None else per_panel(titles, "titles")
    lower = per_panel(vmin, "vmin")
    upper = per_panel(vmax, "vmax")

    columns = count if columns is None else columns
    rows = -(-count // columns)   # ceiling division
    figure, axes = plt.subplots(
        rows, columns, figsize=(5.5 * columns, 4.6 * rows), squeeze=False,
    )
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    evaluators = {}
    for axis, field, title, low, high in zip(axes.flat, fields, titles, lower, upper):
        mesh = field.function_space().mesh()
        if id(mesh) not in evaluators:
            points, layout = _domain_grid(mesh, spacing)
            # No tolerance is passed on purpose: Firedrake would store it on
            # the mesh, and every later point location on that mesh -- the
            # sources and receivers of a forward solve included -- would use
            # it. A plot must not change what a solver computes.
            evaluators[id(mesh)] = (PointEvaluator(mesh, points), layout)
        evaluator, (rows, columns, x_min, x_max, z_min, z_max) = evaluators[id(mesh)]
        samples = np.asarray(evaluator.evaluate(field)).reshape(rows, columns)

        image = axis.imshow(
            samples, extent=[x_min, x_max, z_min, z_max],
            vmin=low, vmax=high, cmap=cmap,
        )
        figure.colorbar(image, ax=axis, label=colorbar_label)
        # ``is not None`` rather than truthiness: the positions usually come
        # from ``create_transect`` as a NumPy array, which has no truth value.
        for z_s, x_s in (sources if sources is not None else ()):
            axis.plot(x_s, z_s, "*", color="red", markersize=10)
        for z_r, x_r in (receivers if receivers is not None else ()):
            axis.plot(x_r, z_r, "v", color="white", markersize=3)
        if title is not None:
            axis.set_title(title)
        axis.set_xlabel("x (km)")
        axis.set_ylabel("z (km)")
    figure.tight_layout()

    rank = fields[0].function_space().mesh().comm.rank
    return _finalize_figure(figure, filename=filename if rank == 0 else None, show=show)
