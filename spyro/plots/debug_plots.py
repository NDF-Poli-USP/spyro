"""Plots for debugging purposes."""

from firedrake import VTKFile, Function
import matplotlib.pyplot as plt
from pathlib import Path
from .general_plots import plot_function


def debug_plot(
    function: Function,
    filename: str | Path = "debug.png",
):
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


def debug_pvd(
    function: Function,
    filename: str | Path = "debug.pvd",
):
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
    out = VTKFile(filename)
    out.write(function)
