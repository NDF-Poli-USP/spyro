"""Receiver visualization plotting routines."""

from pathlib import Path
from random import choice
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np

from .plot_helpers import _finalize_figure

if TYPE_CHECKING:  # Avoinding circular imports lazily
    from ..solvers.wave import Wave

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

    if hold:
        if show:
            plt.show()
    else:
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
    show: bool = False,
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
    show : bool, optional
        Whether to display the plot interactively. Default is False.

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

    if show:
        plt.show()
    plt.close(separated_fig)
    plt.close(combined_fig)


def plot_comparison_of_receivers_to_reference(
        wave: "Wave",
        reference_array: np.array,
        show: bool = False,
        filename: str | Path | None = None,
    ):
    """Plot receiver time-domain comparisons from a Wave object.

    This is a convenience wrapper around
    :func:`plot_compare_receivers_array` that extracts the receiver data
    and time vector from a ``Wave_object`` before generating the plot.

    Parameters
    ----------
    wave : Wave
        Wave object containing the receiver data and simulation metadata.
        The following attributes are used:

        - ``dt``: simulation time step.
        - ``final_time``: final simulation time.
        - ``forward_solution_receivers``: receiver data from the simulation.
        - ``path_case_habc``: output directory for the generated figures.
    reference_array: reference receiver data.
    show : bool, optional
        Whether to display the figure interactively. Defaults to ``False``.

    Returns
    -------
    None

    See Also
    --------
    plot_compare_receivers_array
        Generic plotting function that compares two receiver arrays.
    """
    dt = wave.dt
    final_time = wave.final_time
    num_timesteps = int(round(final_time / dt)) + 1

    time_values = np.linspace(0.0, final_time, num_timesteps)

    if filename is None and hasattr(wave, "path_case_abc"):
        filename = Path(wave.path_case_abc) / "time_comparison"

    plot_compare_receivers_array(
        receiver_data_first=wave.forward_solution_receivers,
        receiver_data_second=reference_array,
        time_values=time_values,
        first_label="Simulation",
        second_label="Reference",
        output_path=filename,
        show=show,
    )


def plot_compare_receivers_array(
    receiver_data_first: np.ndarray,
    receiver_data_second: np.ndarray,
    time_values: np.ndarray,
    *,
    first_label: str = "Solution",
    second_label: str = "Reference",
    output_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot receiver time-domain comparisons.

    Parameters
    ----------
    receiver_data_first
        Receiver data with shape ``(n_timesteps, n_receivers)``.
    receiver_data_second
        Receiver data with shape ``(n_timesteps, n_receivers)``.
    time_values
        Time corresponding to each sample.
    first_label
        Legend label for the first receiver array.
    second_label
        Legend label for the second receiver array.
    output_path
        Path (without extension) where the figure should be saved.
        If ``None``, the figure is not saved.
    show
        Whether to display the figure.

    Returns
    -------
    None
    """
    print("\nPlotting receiver comparison", flush=True)

    if receiver_data_first.shape != receiver_data_second.shape:
        raise ValueError("Receiver arrays must have the same shape.")

    num_receivers = receiver_data_first.shape[1]

    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.grid"] = True

    figure, receiver_axes = plt.subplots(
        nrows=num_receivers,
        ncols=1,
        sharex=True,
    )

    if num_receivers == 1:
        receiver_axes = [receiver_axes]

    figure.subplots_adjust(hspace=0.6)

    first_color = (0.0, 1.0, 0.0, 1.0)
    second_color = (1.0, 0.0, 0.0, 1.0)

    final_time = time_values[-1]

    for receiver_index in range(num_receivers):

        first_receiver_trace = receiver_data_first[:, receiver_index]
        second_receiver_trace = receiver_data_second[:, receiver_index]

        receiver_axes[receiver_index].plot(
            time_values,
            first_receiver_trace,
            color=first_color,
            linewidth=2,
            label=first_label,
        )

        receiver_axes[receiver_index].plot(
            time_values,
            second_receiver_trace,
            color=second_color,
            linestyle="--",
            linewidth=2,
            label=second_label,
        )

        receiver_axes[receiver_index].text(
            0.995,
            0.9,
            f"R{receiver_index + 1}",
            fontsize=8.5,
            transform=receiver_axes[receiver_index].transAxes,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="right",
        )

        if receiver_index == num_receivers // 2:
            receiver_axes[receiver_index].set_ylabel(r"$sol \; recs$")

        receiver_axes[receiver_index].set_xlim(0.0, final_time)
        receiver_axes[receiver_index].ticklabel_format(
            axis="y",
            style="scientific",
            scilimits=(-2, 2),
        )

    receiver_axes[-1].set_xlabel(r"$t \; (s)$")
    receiver_axes[0].legend()

    _finalize_figure(
        plt.gcf(), output_path, formats=("png", "pdf"), show=show, bbox_inches="tight"
    )
