# from scipy.io import savemat
import matplotlib.pyplot as plt

import numpy as np

__all__ = ["plot_shotrecords", "plot_shotrecords_difference"]


def plot_shotrecords(
    model,
    arr,
    appear=False,
    name="MODEL",
    vmin=-1e-5,
    vmax=1e-5,
    ft="PDF",
    start_index=0,
    end_index=0,
):
    """Plot a shot record and save it to disk.

    Parameters
    ----------
    model: `dictionary`
        Contains model parameters and options.
    arr: array-like
        An array in which rows are intervals in time and columns and receivers
    appear: `boolean`, optional
        Should the images appear on screen?
    name: string, optional
        The name of the saved PDF
    vmin: float, optional
        The minimum value to plot on the colorscale
    vmax: float, optional
        The maximum value to plot on the colorscale
    ft: string, optional
        File format, PDF or png
    start_index: integer, optional
        The index of the first receiver to plot
    end_index: integer, optional
        The index of the last receiver to plot

    Returns
    -------
    None

    """

    num_recvs = model["acquisition"]["num_receivers"]

    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]

    nt = int(tf / dt)  # number of timesteps

    if end_index == 0:
        end_index = num_recvs

    x_rec = np.linspace(start_index, end_index, num_recvs)
    t_rec = np.linspace(0.0, tf, nt)
    X, Y = np.meshgrid(x_rec, t_rec)

    cmap = plt.get_cmap("gray")
    plt.contourf(X, Y, arr, cmap=cmap, vmin=vmin, vmax=vmax)
    # savemat("test.mat", {"mydata": arr})
    plt.xlabel("receiver number", fontsize=18)
    plt.ylabel("time (s)", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(start_index, end_index)
    plt.ylim(tf, 0)
    plt.subplots_adjust(left=0.18, right=0.95, bottom=0.14, top=0.95)
    plt.savefig("shot_number_" + name + "." + ft, format=ft)
    # plt.axis("image")
    if appear:
        plt.show()
    plt.close()
    return None


def plot_shotrecords_difference(model, arr1, arr2, appear=False, name="DIFF", ft="PDF"):
    """Plot the difference (e.g., arr1-arr2) between two shot records

    Parameters
    ----------
    model: `dictionary`
        Contains model parameters and options.
    arr1: array-like
        An array in which rows are intervals in time and columns and receivers
    arr2: array-like
        An array in which rows are intervals in time and columns and receivers
    appear: `boolean`, optional
        Should the images appear on screen?
    name: string, optional
        The name of the saved PDF
    ft: string, optional
        File format, PDF or png

    Returns
    -------
    None

    Returns
    -------
    None

    """

    num_recvs = model["acquisition"]["num_receivers"]
    start_x = model["acquisition"]["start_recv_x"]
    end_x = model["acquisition"]["end_recv_x"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]

    nt = int(tf / dt)  # number of timesteps

    x_rec = np.linspace(start_x, end_x, num_recvs)
    t_rec = np.linspace(0.0, tf, nt)
    X, Y = np.meshgrid(x_rec, t_rec)

    # Difference between the fields
    field_dif = arr1 - arr2

    cmap = plt.get_cmap("gray")
    cnt = plt.contourf(X, Y, field_dif, 700, cmap=cmap, vmax=8e-2, vmin=-8e-2)
    for c in cnt.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000000000001)
    plt.ylabel("time (s)", fontsize=18)
    plt.xlabel("$x$ position (km)", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(start_x, end_x)
    plt.ylim(tf, 0)
    plt.subplots_adjust(left=0.18, right=0.95, bottom=0.14, top=0.95)
    plt.savefig("shot_number_difference." + ft, format=ft)
    if appear:
        plt.show()
    plt.close()

    return None


def plot_pml_error(model, error, appear=False):
    """Plot the error from a PML aproximation captured by the receivers from
    two different velocitiy models (experimental and reference solutions)

    Parameters
    ----------
    model: `dictionary`
        Contains model parameters and options.
    error: array-like
        An array with the error associated to the pml (mistfit in the receivers
        with respect to the referece solution)
    appear: `boolean`, optional
        Should the images appear on screen?

    Returns
    -------
    None

    """

    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]

    nt = int(tf / dt)  # number of timesteps

    t_rec = np.linspace(0.0, tf, nt)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.tick_params(reset=True, direction="in", which="both")
    plt.rc("legend", **{"fontsize": 18})
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("Error", fontsize=18)
    plt.xlabel("time (s)", fontsize=18)
    plt.xlim(0.0, tf)
    plt.plot(
        t_rec, error, "r", color="k", linewidth=4, label=r"PML error"
    )  # $\sigma_{1}$
    ax.yaxis.get_offset_text().set_fontsize(18)
    plt.subplots_adjust(left=0.18, right=0.95, bottom=0.14, top=0.95)
    plt.legend(loc="best")
    plt.savefig("pml_error.pdf", format="PDF")
    if appear:
        plt.show()
    plt.close()
    return None
