"""HABC related plotting routines."""

from matplotlib.pyplot import (
    figure,
    gca,
    grid,
    legend,
    plot,
    rcParams,
    scatter,
    xlabel,
    xlim,
    xticks,
    ylabel,
    ylim,
)

# from numpy import arange, asarray, ceil, clip, linspace, inf, polyfit, polyval, zeros
from numpy import arange, ceil, linspace, inf, zeros
from os import makedirs, path
from ..abc.lay_len import f_layer, loop_roots
from .plot_helpers import _finalize_figure

# from ..utils.stats_tools import coeff_of_determination
rcParams.update({"font.family": "serif"})
rcParams["text.latex.preamble"] = r"\usepackage{bm} \usepackage{amsmath}"


def create_folder(folder):
    """Verify if a folder exists, if not, it creates the folder.

    Parameters
    ----------
    folder: `str`
        Path to the folder to be created

    Returns
    -------
    None
    """
    # Create the folder if it does not exist
    if not path.isdir(folder):
        makedirs(folder)


def plot_function_layer_size(
    lay_par, freq_par, geom_par, FLpos, output_folder="output/", show=False
):
    """Plot the function of the layer size criterion for the HABC scheme.

    Parameters
    ----------
    lay_par: `list`
        Parameters of the layer:
        - a : `float`
            Adimensional propagation speed parameter (a = z / f, z = c / l).
        - z_par : `float`
            Inverse of min. Eikonal (1 / phi_min, equivalent to c_bound/lref).
    freq_par: `list`
        Parameters of the frequency:
        - reference_frequency : `float`
            Reference frequency of the wave.
        - source_frequency : `float`
            Source frequency.
    geom_par: `list`
        Parameters of the domain geometry:
        - lmin : `float`
            Minimal dimension of finite element in mesh.
        - lref : `float`
            Reference length for the size of the absorbing layer.
    FLpos: `list`
        List of size parameters for the reference frequency.
    output_folder: `str`, optional
        Folder to save the output plots. Default is "output/".
    show: `bool`, optional
        Whether to show the plot. Default is `False`.

    Returns
    -------
    None
    """
    # Create the output folder if it does not exist
    create_folder(output_folder)

    # Unpack the parameters
    a, z = lay_par
    reference_frequency, source_frequency = freq_par
    lmin, lref = geom_par

    # Prepare the data for the plot
    a_lst = [a]
    F_lst = [FLpos]
    l_lst = ["{:.2f}".format(reference_frequency)]
    c_lst = ["C0"]

    if source_frequency == reference_frequency:

        # Layer size based on source frequency
        FLsou = []
        w_lst = ["f_{{sou}}"]

    else:

        # Calculate the size parameter for the source frequency
        a_sou = z / source_frequency  # Adimensional parameter
        FLsou = loop_roots(a_sou, lmin, lref, len(FLpos), show_ig=False)
        a_lst.append(a_sou)
        F_lst.append(FLsou)
        l_lst.append("{:.2f}".format(source_frequency))
        c_lst.append("C1")
        w_lst = ["f_{{bnd}}", "f_{{sou}}"]

    # Calculate the maximum layer size for the plot
    FL_max = max(FLpos + FLsou) + 0.4
    FL_lim = ceil(FL_max * 10) / 10
    F_L = linspace(0.001, FL_lim, int(FL_lim * 1e3))

    # Plot the size criterion
    figure(figsize=(12, 6))  # Set figure size
    ax = gca()
    lim_crit = inf
    for a_pr, FL_rt, lab, col, w_str in zip(a_lst, F_lst, l_lst, c_lst, w_lst):
        crit = f_layer(F_L, a_pr)
        lim_crit = min(lim_crit, crit.min())
        plot(
            F_L,
            crit,
            color=col,
            zorder=2,
            label=r"$\Psi_{{F_L}}({}={}\text{{Hz}})$".format(w_str, lab),
        )
        scatter(FL_rt, zeros(len(FL_rt)), color=col, zorder=3)

    # Identify the roots of the criterion function
    delta_x = FL_lim / 40.0
    delta_y = abs(lim_crit) / 2.0
    off_x = 0.5 * delta_x
    off_y = 0.85 * delta_y
    for lay, (FL_rt, col) in enumerate(zip(F_lst, c_lst)):
        base_y = -1.3 * delta_y if lay == 0 else 0.8 * delta_y
        used_positions = []

        for rt, FL_par in enumerate(FL_rt):
            xFL = FL_par + delta_x if rt % 2 == 0 else FL_par - delta_x
            y_FL = base_y

            # Check for overlap and adjust if needed
            for prev_x, prev_y in used_positions:
                if (
                    abs(xFL - prev_x) < 2.6 * delta_x
                    and abs(y_FL - prev_y) < 0.9 * off_y
                ):
                    xFL += -off_x if rt % 2 == 0 else off_x
                    y_FL += -off_y if lay == 0 else off_y
            used_positions.append((xFL, y_FL))

            ax.annotate(
                f"{FL_par:.4f}",  # Text
                xy=(FL_par, 0),  # Point to connect to
                xytext=(xFL, y_FL),  # Text position
                ha="center",
                va="bottom",
                zorder=4,
                bbox=dict(facecolor=col, alpha=0.9),
                arrowprops=dict(
                    arrowstyle="-",
                    color="black",
                    linewidth=0.8,
                    alpha=0.9,
                    connectionstyle="arc3,rad=0.",
                ),
            )

    # Formatting the plot
    FL_str = r"$F_L \; (L_{{\xi}} \; = \; L_{{ref}} \, F_L \;$"
    lref_str = r"$\therefore \; L_{{ref}} \; = \; {:.4f}\text{{km}})$"
    xlabel((FL_str + lref_str).format(lref))
    ylabel(r"$\Psi_{{F_L}} \; = \; |C_{Rmin}| \; - \; R$")
    xticks(arange(0, FL_lim + 0.01, 0.5 if FL_lim > 1 else 0.2))
    xlim((0, FL_lim))
    ylim((lim_crit - 0.01, 1.01))
    grid(zorder=1)
    legend()

    # Saving the plot
    layer_str = output_folder + "layer_opts"

    _finalize_figure(
        plot.gcf(),
        layer_str,
        formats=("png", "pdf"),
        show=show,
        bbox_inches="tight",
    )
