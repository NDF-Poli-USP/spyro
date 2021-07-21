# from scipy.io import savemat
import matplotlib.pyplot as plt

import numpy as np
import spyro

from ..io import ensemble_plot
__all__ = ["plot_shots", "plot_receiver_difference"]



@ensemble_plot
def plot_shots(
    model,
    comm,
    arr,
    show=False,
    file_name="1",
    vmin=-1e-5,
    vmax=1e-5,
    file_format="pdf",
    start_index=0,
    end_index=0,
):
    """Plot a shot record and save the image to disk. Note that
    this automatically will rename shots when ensmeble paralleism is
    activated.

    Parameters
    ----------
    model: `dictionary`
        Contains model parameters and options.
    comm:A Firedrake commmunicator
        The communicator you get from calling spyro.utils.mpi_init()
    arr: array-like
        An array in which rows are intervals in time and columns are receivers
    show: `boolean`, optional
        Should the images appear on screen?
    file_name: string, optional
        The name of the saved image
    vmin: float, optional
        The minimum value to plot on the colorscale
    vmax: float, optional
        The maximum value to plot on the colorscale
    file_format: string, optional
        File format, pdf or png
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
    plt.savefig("shot_number_" + file_name + "." + file_format, format=file_format)
    # plt.axis("image")
    if show:
        plt.show()
    plt.close()
    return None

def plot_receiver_difference(model, p_receiver0, p_receiver1, id, appear = False, name = 'Receivers', ft = 'PDF'):
    """Plots two receivers in time

    Parameters
    ----------
    model: `dictionary`
        Contains model parameters and options.
    p_receiver0: 
        
    p_receiver1: 
        
    id:


    appear: `boolean`, optional
        Should the images appear on screen?
    name: string, optional
        The name of the saved PDF
    vmax: float, optional
        The maximum value to plot on the colorscale
    ft: string, optional
        File format, PDF or png

    Returns
    -------
    None

    """

    final_time = model["timeaxis"]["tf"]
    # Check if shapes are matching
    times0, receivers0 = p_receiver0.shape
    times1, receivers1 = p_receiver1.shape

    dt0 = final_time/times0
    dt1 = final_time/times1
    
    nt0 = round(final_time / dt0)  # number of timesteps
    nt1 = round(final_time / dt1)  # number of timesteps

    time_vector0 = np.linspace(0.0, final_time, nt0)
    time_vector1 = np.linspace(0.0, final_time, nt1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.tick_params(reset=True, direction="in", which="both")
    plt.rc("legend", **{"fontsize": 18})
    plt.plot(time_vector0, p_receiver0[:, id], 'bo', time_vector1, p_receiver1[:,id], 'go' ) 
    ax.yaxis.get_offset_text().set_fontsize(18)
    plt.subplots_adjust(left=0.18, right=0.95, bottom=0.14, top=0.95)
    plt.legend(loc="best")
    plt.xlabel("time (s)", fontsize=18)
    plt.ylabel("value ", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(model['acquisition']['delay']*0.8, final_time)
    #plt.ylim(tf, 0)
    plt.savefig("Receivers." + ft, format=ft)
    # plt.axis("image")
    if appear:
        plt.show()
    plt.close()
    return None