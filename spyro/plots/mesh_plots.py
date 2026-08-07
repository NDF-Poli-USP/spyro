import copy
from firedrake import (
    assemble,
    CellSize,
    interpolate,
    Mesh,
    tricontourf,
    triplot,
)
import matplotlib.pyplot as plt
from ..domains.space import create_function_space
from ..tools.version_control import is_firedrake_new


if is_firedrake_new() is False:
    from firedrake.__future__ import interpolate

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
    plt.rcParams["font.size"] = 12

    if mesh_filename is not None:
        mesh = Mesh(mesh_filename)
    elif firedrake_mesh is not None:
        mesh = firedrake_mesh
    else:
        raise ValueError("Please specify mesh")

    coordinates = copy.deepcopy(mesh.coordinates.dat.data)

    mesh.coordinates.dat.data[:, 0] = coordinates[:, 1]
    mesh.coordinates.dat.data[:, 1] = coordinates[:, 0]

    DG0 = create_function_space(mesh, "DG0", 0)
    f = assemble(interpolate(CellSize(mesh), DG0))

    fig, axes = plt.subplots()
    if show_size_contour:
        im = tricontourf(f, axes=axes)
    else:
        im = triplot(mesh, axes=axes)

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