import matplotlib.pyplot as plt
import numpy as np
import os
from .plot_helpers import _finalize_figure

def plot_receiver_response(
    receiver_data,
    final_time,
    show=False,
    filename=None,
    receiver_id_for_title=None,
    hold=False,
    color=None,
    name=None,
    **plot_kwargs,
):
    """Plot the time-series response for a single receiver.

    Parameters
    ----------
    receiver_data : np.array
        Receiver data to plot.
    show : bool, optional
        Whether to display the plot interactively. Default is False.
    filename : str, optional
        If provided, save the plot to this file.
    hold : bool, optional
        If True, plot on the current axes so multiple receivers can be
        overlaid in sequence. Default is False.
    color : str, optional
        Line color to use when plotting. If None and hold is True, a random
        color is selected.
    name : str, optional
        Label to use in the legend. When provided, the plot is added to the
        legend so multiple held traces can be identified.
    **plot_kwargs
        Any additional keyword arguments accepted by matplotlib.axes.Axes.plot.

    Returns
    -------
    None
        The function creates the plot and displays it.
    """
    num_times = len(receiver_data)

    time_vector = np.linspace(0.0, final_time, num_times)

    if hold is False:
        plt.close()
        fig, axes = plt.subplots(figsize=(10, 4))
    else:
        fig = plt.gcf()
        axes = plt.gca()
        if fig.get_axes() == []:
            fig, axes = plt.subplots(figsize=(10, 4))

    if color is None and hold:
        color_choices = plt.rcParams["axes.prop_cycle"].by_key().get("color", [None])
        color = np.random.choice(color_choices)

    line_kwargs = dict(plot_kwargs)
    line_kwargs.setdefault("linewidth", 2)
    line_kwargs.setdefault("color", color)
    if name is not None:
        line_kwargs.setdefault("label", name)

    axes.plot(time_vector, receiver_data, **line_kwargs)
    axes.set_xlabel("time (s)", fontsize=18)
    axes.set_ylabel("receiver response", fontsize=18)
    if receiver_id_for_title is not None:
        axes.set_title(f"Receiver ID{receiver_id_for_title} data.", fontsize=22)
    else:
        axes.set_title("Receiver data.", fontsize=22)
    if line_kwargs.get("label") is not None:
        axes.legend()
    axes.tick_params(axis="both", labelsize=18)
    axes.grid(True, alpha=0.3)
    fig.tight_layout()

    _finalize_figure(fig, filename=filename, show=show)


def plot_displacement_components(
    time_vector,
    displacement_tuple,
    source_type="Unknown",
    save_plots=False,
    output_dir=None,
):
    """
    Plot displacement components (ux, uy, uz) over time.

    Parameters
    ----------
    time_vector : numpy array
        Time vector
    displacement_tuple : tuple of numpy arrays
        (ux, uy, uz) displacement components
    source_type : str, optional
        Type of source ("force_source", "explosive_source", etc.)
    save_plots : bool, optional
        Whether to save plots to files
    output_dir : str, optional
        Directory to save plots if save_plots is True

    Returns
    -------
    None
        Creates matplotlib figures
    """
    ux, uy, uz = displacement_tuple
    if output_dir is None:
        output_dir = "."

    # Create the plot with separated subplots
    plt.figure(figsize=(12, 8))

    # Plot all three components
    plt.subplot(3, 1, 1)
    plt.plot(time_vector, ux, "b-", linewidth=2, label="Ux (displacement in x)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Displacement Component Ux - {source_type}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_vector, uy, "r-", linewidth=2, label="Uy (displacement in y)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Displacement Component Uy - {source_type}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_vector, uz, "g-", linewidth=2, label="Uz (displacement in z)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Displacement Component Uz - {source_type}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Also create a combined plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_vector, ux, "b-", linewidth=2, label="Ux")
    plt.plot(time_vector, uy, "r-", linewidth=2, label="Uy")
    plt.plot(time_vector, uz, "g-", linewidth=2, label="Uz")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"All Displacement Components - {source_type}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_plots:

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        basename = source_type.replace(" ", "_")

        # Save plots
        plt.figure(1)  # Select the first figure (subplots)
        plt.savefig(
            os.path.join(
                output_dir,
                f"analytical_{basename}_displacement_components_separated.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.figure(2)  # Select the second figure (combined)
        plt.savefig(
            os.path.join(
                output_dir,
                f"analytical_{basename}_displacement_components_combined.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Plots saved to {output_dir}")

    plt.show()
