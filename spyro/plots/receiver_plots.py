"""Receiver visualization plotting routines."""

from pathlib import Path
from random import choice

import matplotlib.pyplot as plt
import numpy as np

from .plot_helpers import _finalize_figure


def plot_receiver_response(
    receiver_data: np.ndarray | list[float] | tuple[float, ...],
    final_time: float,
    show: bool = False,
    filename: str | Path | None = None,
    receiver_id_for_title: int | str | None = None,
    hold: bool = False,
    color: str | None = None,
    name: str | None = None,
    **plot_kwargs,
):
    """Plot the time-series response for a single receiver.

    Parameters
    ----------
    receiver_data : array_like
        Receiver values sampled in time.
    final_time : float
        Final simulation time used to build the time axis.
    show : bool, optional
        Whether to display the plot interactively. Default is False.
    filename : str or pathlib.Path, optional
        If provided, save the plot to this file.
    receiver_id_for_title : int or str, optional
        Identifier included in the title when provided.
    hold : bool, optional
        If True, plot on the current axes so multiple receivers can be
        overlaid in sequence. Default is False.
    color : str, optional
        Line color to use when plotting. If None and hold is True, a random
        color is selected.
    name : str, optional
        Label to use in the legend. When provided, the plot is added to the
        legend so multiple held traces can be identified.
    **plot_kwargs : dict
        Any additional keyword arguments accepted by matplotlib.axes.Axes.plot.

    Returns
    -------
    None
        The function creates the plot and optionally saves or displays it.
    """
    receiver_data = np.asarray(receiver_data)
    if receiver_data.ndim != 1:
        raise ValueError("receiver_data must be one-dimensional")

    num_times = receiver_data.size
    if num_times == 0:
        raise ValueError("receiver_data must contain at least one sample")

    time_vector = np.linspace(0.0, final_time, num_times)

    if hold is False:
        plt.close()
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = plt.gcf()
        ax = plt.gca()
        if not fig.get_axes():
            fig, ax = plt.subplots(figsize=(10, 4))

    if color is None and hold:
        color_choices = plt.rcParams["axes.prop_cycle"].by_key().get("color", [None])
        color = choice(color_choices)

    line_kwargs = dict(plot_kwargs)
    line_kwargs.setdefault("linewidth", 2)
    line_kwargs.setdefault("color", color)
    if name is not None:
        line_kwargs.setdefault("label", name)

    ax.plot(time_vector, receiver_data, **line_kwargs)
    ax.set_xlabel("time (s)", fontsize=18)
    ax.set_ylabel("receiver response", fontsize=18)
    if receiver_id_for_title is not None:
        ax.set_title(f"Receiver ID {receiver_id_for_title} data.", fontsize=22)
    else:
        ax.set_title("Receiver data.", fontsize=22)
    if line_kwargs.get("label") is not None:
        ax.legend()
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _finalize_figure(fig, filename=filename, show=show)


def plot_displacement_components(
    time_vector: np.ndarray | list[float] | tuple[float, ...],
    displacement_tuple: tuple[
        np.ndarray | list[float] | tuple[float, ...],
        np.ndarray | list[float] | tuple[float, ...],
        np.ndarray | list[float] | tuple[float, ...],
    ],
    source_type: str = "Unknown",
    save_plots: bool = False,
    output_dir: str | Path | None = None,
):
    """Plot displacement components over time.

    Parameters
    ----------
    time_vector : array_like
        Time samples.
    displacement_tuple : tuple of array_like
        Displacement components ``(ux, uy, uz)``.
    source_type : str, optional
        Type of source used in plot titles and output filenames.
    save_plots : bool, optional
        Whether to save plots to files.
    output_dir : str or pathlib.Path, optional
        Directory to save plots if ``save_plots`` is True. Defaults to the
        current directory.

    Returns
    -------
    None
        Creates matplotlib figures.

    Raises
    ------
    ValueError
        Raised when the inputs are not one-dimensional or have mismatched
        lengths.
    """
    time_vector = np.asarray(time_vector)
    ux, uy, uz = (np.asarray(component) for component in displacement_tuple)

    if time_vector.ndim != 1:
        raise ValueError("time_vector must be one-dimensional")
    if any(component.ndim != 1 for component in (ux, uy, uz)):
        raise ValueError("all displacement components must be one-dimensional")
    if not (len(time_vector) == len(ux) == len(uy) == len(uz)):
        raise ValueError(
            "time_vector and displacement components must have the same length"
        )

    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)

    separated_fig, separated_axes = plt.subplots(
        3,
        1,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
    )

    component_data = (
        (ux, "b", "Ux (displacement in x)", "Displacement Component Ux"),
        (uy, "r", "Uy (displacement in y)", "Displacement Component Uy"),
        (uz, "g", "Uz (displacement in z)", "Displacement Component Uz"),
    )

    for axis, (component, color, label, title) in zip(separated_axes, component_data):
        axis.plot(time_vector, component, color=color, linewidth=2, label=label)
        axis.set_ylabel("Amplitude")
        axis.set_title(f"{title} - {source_type}")
        axis.grid(True, alpha=0.3)
        axis.legend()

    separated_axes[-1].set_xlabel("Time (s)")

    combined_fig, combined_ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    combined_ax.plot(time_vector, ux, color="b", linewidth=2, label="Ux")
    combined_ax.plot(time_vector, uy, color="r", linewidth=2, label="Uy")
    combined_ax.plot(time_vector, uz, color="g", linewidth=2, label="Uz")
    combined_ax.set_xlabel("Time (s)")
    combined_ax.set_ylabel("Amplitude")
    combined_ax.set_title(f"All Displacement Components - {source_type}")
    combined_ax.legend()
    combined_ax.grid(True, alpha=0.3)

    if save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)
        basename = source_type.replace(" ", "_")

        separated_fig.savefig(
            output_dir / f"analytical_{basename}_displacement_components_separated.png",
            dpi=300,
            bbox_inches="tight",
        )
        combined_fig.savefig(
            output_dir / f"analytical_{basename}_displacement_components_combined.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
    plt.close(separated_fig)
    plt.close(combined_fig)
