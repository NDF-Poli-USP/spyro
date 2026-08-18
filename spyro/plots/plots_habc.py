
"""HABC-related plotting routines."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..abc.lay_len import f_layer, loop_roots
from .plot_helpers import _finalize_figure

plt.rcParams.update({"font.family": "serif"})
plt.rcParams["text.latex.preamble"] = r"\usepackage{bm} \usepackage{amsmath}"


def create_folder(folder: str | Path) -> None:
    """Ensure a directory exists, creating it if needed.

    Parameters
    ----------
    folder : str or pathlib.Path
        Path to the directory to create.

    Returns
    -------
    None
    """
    # TODO: replace with the validation method to come
    folder_path = Path(folder)
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)


def plot_function_layer_size(
        layer_parameters: Sequence[float],
        frequency_parameters: Sequence[float],
        geometry_parameters: Sequence[float],
        reference_frequency_layer_sizes: Sequence[float],
        output_folder: str | Path = "output/",
        show: bool = False,) -> None:
    """Plot the layer-size criterion for the HABC scheme.

    Parameters
    ----------
    layer_parameters : sequence of float
        Parameters of the absorbing layer:
        - ``a`` : float
            Dimensionless propagation speed parameter (``a = z / f``).
        - ``z_par`` : float
            Inverse of the minimum Eikonal parameter.
    frequency_parameters : sequence of float
        Parameters of the frequency:
        - ``reference_frequency`` : float
            Reference frequency of the wave.
        - ``source_frequency`` : float
            Source frequency.
    geometry_parameters : sequence of float
        Parameters of the domain geometry:
        - ``lmin`` : float
            Minimal dimension of the finite element in the mesh.
        - ``lref`` : float
            Reference length for the absorbing layer.
    reference_frequency_layer_sizes : sequence of float
        Layer-size values associated with the reference frequency.
    output_folder : str or pathlib.Path, optional
        Folder used to save the output plots. Default is ``"output/"``.
    show : bool, optional
        Whether to show the plot interactively. Default is ``False``.

    Returns
    -------
    None
    """
    create_folder(output_folder)

    a, z = layer_parameters
    reference_frequency, source_frequency = frequency_parameters
    minimum_element_size, reference_length = geometry_parameters

    layer_size_values = [a]
    layer_size_samples = [reference_frequency_layer_sizes]
    frequency_labels = [f"{reference_frequency:.2f}"]
    colors = ["C0"]
    frequency_label_tokens = [r"f_{\mathrm{sou}}"]

    if source_frequency == reference_frequency:
        source_frequency_layer_sizes = []
    else:
        source_frequency_parameter = z / source_frequency
        source_frequency_layer_sizes = loop_roots(
            source_frequency_parameter,
            minimum_element_size,
            reference_length,
            len(reference_frequency_layer_sizes),
            show_ig=False,
        )
        layer_size_values.append(source_frequency_parameter)
        layer_size_samples.append(source_frequency_layer_sizes)
        frequency_labels.append(f"{source_frequency:.2f}")
        colors.append("C1")
        frequency_label_tokens = [r"f_{\mathrm{bnd}}", r"f_{\mathrm{sou}}"]

    maximum_layer_size = (
        max(list(reference_frequency_layer_sizes) + source_frequency_layer_sizes) + 0.4
    )
    layer_size_limit = np.ceil(maximum_layer_size * 10) / 10
    layer_size_axis = np.linspace(0.001, layer_size_limit, int(layer_size_limit * 1e3))

    plt.figure(figsize=(12, 6))
    axes = plt.gca()
    minimum_criterion_value = np.inf

    for (
        layer_size_value,
        size_samples,
        frequency_label,
        color,
        frequency_label_token,
    ) in zip(
        layer_size_values,
        layer_size_samples,
        frequency_labels,
        colors,
        frequency_label_tokens,
    ):
        criterion = f_layer(layer_size_axis, layer_size_value)
        minimum_criterion_value = min(minimum_criterion_value, criterion.min())
        plt.plot(
            layer_size_axis,
            criterion,
            color=color,
            zorder=2,
            label=rf"$\Psi_{{F_L}}({frequency_label_token}={frequency_label}\text{{Hz}})$",
        )
        plt.scatter(size_samples, np.zeros(len(size_samples)), color=color, zorder=3)

    delta_x = layer_size_limit / 40.0
    delta_y = abs(minimum_criterion_value) / 2.0
    offset_x = 0.5 * delta_x
    offset_y = 0.85 * delta_y

    for layer_index, (size_samples, color) in enumerate(
        zip(layer_size_samples, colors)
    ):
        base_y = -1.3 * delta_y if layer_index == 0 else 0.8 * delta_y
        used_positions = []

        for sample_index, layer_size_value in enumerate(size_samples):
            x_layer_size = (
                layer_size_value + delta_x
                if sample_index % 2 == 0
                else layer_size_value - delta_x
            )
            y_layer_size = base_y

            for previous_x, previous_y in used_positions:
                if (
                    abs(x_layer_size - previous_x) < 2.6 * delta_x
                    and abs(y_layer_size - previous_y) < 0.9 * offset_y
                ):
                    x_layer_size += -offset_x if sample_index % 2 == 0 else offset_x
                    y_layer_size += -offset_y if layer_index == 0 else offset_y
            used_positions.append((x_layer_size, y_layer_size))

            axes.annotate(
                f"{layer_size_value:.4f}",
                xy=(layer_size_value, 0),
                xytext=(x_layer_size, y_layer_size),
                ha="center",
                va="bottom",
                zorder=4,
                bbox=dict(facecolor=color, alpha=0.9),
                arrowprops=dict(
                    arrowstyle="-",
                    color="black",
                    linewidth=0.8,
                    alpha=0.9,
                    connectionstyle="arc3,rad=0.",
                ),
            )

    layer_size_label = r"$F_L \; (L_{{\xi}} \; = \; L_{{ref}} \, F_L \;$"
    reference_length_label = r"$\therefore \; L_{{ref}} \; = \; {:.4f}\text{{km}})$"
    plt.xlabel((layer_size_label + reference_length_label).format(reference_length))
    plt.ylabel(r"$\Psi_{{F_L}} \; = \; |C_{Rmin}| \; - \; R$")
    plt.xticks(
        np.arange(0, layer_size_limit + 0.01, 0.5 if layer_size_limit > 1 else 0.2)
    )
    plt.xlim((0, layer_size_limit))
    plt.ylim((minimum_criterion_value - 0.01, 1.01))
    plt.grid(zorder=1)
    plt.legend()

    output_path = Path(output_folder) / "layer_opts"
    _finalize_figure(
        plt.gcf(),
        output_path,
        formats=("png", "pdf"),
        show=show,
        bbox_inches="tight",
    )
