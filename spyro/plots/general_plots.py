"""General plotting routines for simulation data and diagnostic outputs."""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

from firedrake import tricontourf, Function
from firedrake.pyplot import FunctionPlotter
import matplotlib.pyplot as plt
import numpy as np
from ..io import ensemble_save
from ..utils import change_scalar_field_resolution
from ..utils.physical_parameters import as_list
from .plot_helpers import _finalize_figure

if TYPE_CHECKING:  # Avoiding circular imports lazily
    from ..solvers.wave import Wave


def plot_model(
    wave: "Wave",
    filename: Union[str, Path, None] = "model.png",
    abc_points: Optional[List[Tuple[float, float]]] = None,
    show: bool = False,
    flip_axis: bool = True,
    high_resolution: bool = False,
    high_resolution_grid_value: float = 0.01,
    *,
    fields: Union[Function, Sequence[Function], None] = None,
    titles: Optional[Sequence[str]] = None,
    vmin: Union[float, Sequence[Optional[float]], None] = None,
    vmax: Union[float, Sequence[Optional[float]], None] = None,
    cmap: str = "viridis",
    colorbar_label: str = "Velocity (km/s)",
    columns: Optional[int] = None,
    show_acquisition: bool = True,
) -> Optional[plt.Figure]:
    """Plot one or more material fields with optional finer CG1 interpolation.

    Parameters
    ----------
    wave : Wave
        Supplies mesh metadata and source and receiver locations.
    filename : str or pathlib.Path, optional
        Output filename. ``None`` creates the figure without saving it.
    abc_points : list of tuple, optional
        Boundary vertices in ``(z, x)`` order, drawn as a closed dashed line.
    show : bool, optional
        Display the figure. Default is False.
    flip_axis : bool, optional
        Plot x horizontally and z vertically when True (the default).
        False uses the mesh coordinate order: z horizontally, x vertically.
    high_resolution : bool, optional
        Interpolate each field onto a finer CG1 mesh before plotting.
        Default is False.
    high_resolution_grid_value : float, optional
        Requested edge length of the visualisation mesh, in the mesh's
        units. Default is 0.01. Used only when ``high_resolution`` is True.
    fields : firedrake.Function or sequence of firedrake.Function, optional
        Fields to plot. Defaults to the solver's independent material
        parameters: the velocity of an acoustic medium; the density and
        either the wave speeds or the Lamé parameters of an isotropic
        elastic one, built from its model if a forward solve has not yet.
        For high resolution, their domains must match ``wave.mesh_parameters``.
    titles : sequence of str, optional
        One title per field. Defaults to the parameter names when the
        fields are the solver's own, to nothing otherwise.
    vmin : float or sequence of float, optional
        Lower colour limit, shared or one per field.
    vmax : float or sequence of float, optional
        Upper colour limit, shared or one per field.
    cmap : str, optional
        Matplotlib colour map. Default is ``"viridis"``.
    colorbar_label : str, optional
        Colour bar label. Default is ``"Velocity (km/s)"``.
    columns : int, optional
        Number of panels per row. Defaults to one row containing all fields.
    show_acquisition : bool, optional
        Overlay sources in green and receivers in red. Default is True.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure on spatial rank zero, closed unless ``show`` is True.
        Other ranks return None.

    Raises
    ------
    ValueError
        If fields are empty, not scalar or not two-dimensional, panel options
        have incompatible lengths, columns is not a positive integer, or no
        fields are given and the solver has no material model to draw.

    Notes
    -----
    Collective over the fields' mesh communicator. Call on every spatial
    rank of one ensemble member; only its rank zero draws and saves.
    Fields must share that communicator. The simulation mesh is not modified.

    See Also
    --------
    spyro.utils.change_scalar_field_resolution : Shared CG1 regridding tool.
    """
    if fields is None:
        fields, names = _material_fields(wave)
        titles = names if titles is None else titles
    fields = as_list(fields)
    count = len(fields)
    if not count:
        raise ValueError("At least one field is required.")
    for field in fields:
        if field.ufl_shape != () or field.function_space().mesh().geometric_dimension != 2:
            raise ValueError("plot_model requires scalar fields on two-dimensional meshes.")

    def per_panel(value: object, name: str) -> list:
        """Expand a scalar or validate a sequence of panel settings.

        Parameters
        ----------
        value : object
            Shared value or sequence with one entry per field.
        name : str
            Setting name for error messages.

        Returns
        -------
        list
            One value per panel.

        Raises
        ------
        ValueError
            If the sequence length differs from the number of fields.
        """
        values = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [value] * count
        if len(values) != count:
            raise ValueError(f"{name} takes one entry per field: expected {count}, got {len(values)}.")
        return values

    titles = per_panel(titles, "titles")
    lower, upper = per_panel(vmin, "vmin"), per_panel(vmax, "vmax")
    columns = count if columns is None else columns
    if isinstance(columns, bool) or not isinstance(columns, (int, np.integer)) or columns < 1:
        raise ValueError("columns must be a positive integer.")
    rows = -(-count // columns)
    comm = fields[0].function_space().mesh().comm
    figure = None
    if comm.rank == 0:
        figure, axes = plt.subplots(
            rows, columns, figsize=(5.5 * columns, 4.6 * rows), squeeze=False,
        )
        for axis in axes.flat[count:]:
            axis.set_visible(False)

    # The regridding mesh depends on the domain and the spacing alone, so
    # one serves every field; likewise one sampler serves every field on a
    # mesh.
    fine_space = None
    samplers = {}
    for index, field in enumerate(fields):
        if high_resolution:
            field, fine_space = change_scalar_field_resolution(
                field, wave.mesh_parameters, high_resolution_grid_value,
                function_space=fine_space,
            )
        mesh = field.function_space().mesh()
        if id(mesh) not in samplers:
            samplers[id(mesh)] = _owned_sampler(mesh, 1 if high_resolution else 10)
        plotter, points, triangles = samplers[id(mesh)]
        tri = plotter.triangulation
        pieces = comm.gather(
            (tri.x[:points], tri.y[:points], tri.triangles[:triangles], plotter(field)[:points]),
            root=0,
        )
        if comm.rank != 0:
            continue
        z = np.concatenate([piece[0] for piece in pieces])
        x = np.concatenate([piece[1] for piece in pieces])
        offsets = np.cumsum([0] + [len(piece[0]) for piece in pieces[:-1]])
        triangles = np.concatenate([piece[2] + offset for piece, offset in zip(pieces, offsets)])
        values = np.concatenate([piece[3] for piece in pieces])
        axis = axes.flat[index]
        horizontal, vertical = (x, z) if flip_axis else (z, x)
        image = axis.tripcolor(
            horizontal, vertical, triangles, values, shading="gouraud",
            vmin=lower[index], vmax=upper[index], cmap=cmap,
        )
        figure.colorbar(image, ax=axis, label=colorbar_label)
        if show_acquisition:
            for locations, colour in ((wave.source_locations, "green"), (wave.receiver_locations, "red")):
                points = np.asarray(locations).reshape(-1, 2)
                if flip_axis:
                    points = points[:, ::-1]
                axis.scatter(points[:, 0], points[:, 1], c=colour)
        if abc_points is not None and len(abc_points):
            boundary = np.asarray([*abc_points, abc_points[0]])
            if flip_axis:
                boundary = boundary[:, ::-1]
            axis.plot(boundary[:, 0], boundary[:, 1], "--")
        if titles[index] is not None:
            axis.set_title(titles[index])
        axis.set_xlabel("x (km)" if flip_axis else "z (km)")
        axis.set_ylabel("z (km)" if flip_axis else "x (km)")
        axis.set_aspect("equal")

    if figure is not None:
        figure.tight_layout()
        return _finalize_figure(figure, filename=filename, show=show)
    return None


def _material_fields(wave: "Wave") -> Tuple[list, list]:
    """Return the fields a solver's model is made of, and their names.

    Parameters
    ----------
    wave : Wave
        The solver. Its material parameters are built from the model it
        holds if a forward solve has not done so yet.

    Returns
    -------
    list of firedrake.Function
        The independent material parameters, in the solver's order.
    list of str
        Their names.

    Raises
    ------
    ValueError
        If the solver carries no model to build them from.
    """
    try:
        parameters = wave.physical_parameters
    except ValueError:
        parameters = wave.initialize_physical_parameters()
    except AttributeError:
        # Not a spyro solver: whatever velocity model it carries.
        velocity = getattr(wave, "initial_velocity_model", None)
        if velocity is None:
            raise ValueError(
                "The solver has no material model to draw; pass fields=.",
            ) from None
        return [velocity], [None]
    fields = parameters.select()
    return list(fields.values()), [name.value for name in fields]


def _owned_sampler(mesh, num_sample_points: int) -> Tuple[FunctionPlotter, int, int]:
    """Return a sampler of a mesh and how much of it the rank owns.

    The sampler is the one behind Firedrake's ``tripcolor``. It lays its
    points and triangles out cell by cell, the owned cells first, so the
    counts returned cut the halo cells off: they are owned, and drawn, by a
    neighbouring rank.

    Parameters
    ----------
    mesh : firedrake.mesh.MeshGeometry
        Mesh to sample.
    num_sample_points : int
        Sample points per cell, as ``FunctionPlotter`` takes them.

    Returns
    -------
    firedrake.pyplot.FunctionPlotter
        The sampler.
    int
        Number of its points that lie in owned cells.
    int
        Number of its triangles that lie in owned cells.
    """
    plotter = FunctionPlotter(mesh, num_sample_points)
    tri = plotter.triangulation
    num_cells = mesh.coordinates.function_space().cell_node_list.shape[0]
    points = mesh.cell_set.size * (len(tri.x) // num_cells)
    triangles = mesh.cell_set.size * (len(tri.triangles) // num_cells)
    return plotter, points, triangles


def plot_model_in_p1(
    wave: "Wave",
    dx: float = 0.01,
    filename: Union[str, Path, None] = "model.png",
    abc_points: Optional[List[Tuple[float, float]]] = None,
    show: bool = False,
    flip_axis: bool = True,
) -> Optional[plt.Figure]:
    """Plot the material model with a P1 finite element projection.

    The model is interpolated onto a CG1 space on a structured mesh of the
    same domain with edge length ``dx``, by
    :func:`spyro.utils.change_scalar_field_resolution`, and drawn from
    there: :func:`plot_model` with ``high_resolution=True``.

    Parameters
    ----------
    wave : Wave
        Wave object containing the material model and mesh metadata.
    dx : float, optional
        Edge length of the visualisation mesh, in the mesh's units. Default
        is 0.01.
    filename : str or pathlib.Path, optional
        Output filename. Default is "model.png"; ``None`` creates the figure
        without saving it.
    abc_points : list of tuple, optional
        Boundary vertices in ``(z, x)`` order.
    show : bool, optional
        Display the figure. Default is False.
    flip_axis : bool, optional
        Put x horizontally and z vertically. Default is True.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure on spatial rank zero; None on other ranks.

    See Also
    --------
    plot_model : Plots single fields or multi-panel comparisons.
    """
    return plot_model(
        wave, filename=filename, abc_points=abc_points, show=show,
        flip_axis=flip_axis, high_resolution=True, high_resolution_grid_value=dx,
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
    contours = tricontourf(function, axes=axes, **kwargs)
    plt.colorbar(contours)
    axes.axis("equal")
