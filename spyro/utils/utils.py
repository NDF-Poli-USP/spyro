import copy

import numpy as np
from firedrake import *

from mpi4py import MPI
from scipy.signal import butter, filtfilt


def create_weighting_function(V, const=100.0, M=5, width=0.1, show=False):
    """Create a weighting function g, which is large near the
    boundary of the domain and a constant smaller value in the interior

    Inputs
    ------
       V: Firedrake.FunctionSpace
    const: the weight function is equal to this constant value, except close to the boundary
    M:   maximum value on the boundary will be M**2
    width:  the decimal fraction of the domain where the weight is > constant
    show: Visualize the weighting function

    Outputs
    -------
    wei: a Firedrake.Function containing the weights

    """

    # get coordinates of DoFs
    m = V.ufl_domain()
    W2 = VectorFunctionSpace(m, V.ufl_element())
    coords = interpolate(m.coordinates, W2)
    Z, X = coords.dat.data_ro_with_halos[:, 0], coords.dat.data_ro_with_halos[:, 1]

    a0 = np.amin(X)
    a1 = np.amax(X)
    b0 = np.amin(Z)
    b1 = np.amax(Z)

    cx = a1 - a0  # x-coordinate of center of rectangle
    cz = b1 - b0  # z-coordinate of center of rectangle

    def h(t, d):
        L = width * d  # fraction of the domain where the weight is > constant
        return (np.maximum(0.0, M / L * t + M)) ** 2

    w = const * (
        1.0
        + np.maximum(
            h(X - a1, a1 - a0) + h(a0 - X, a1 - a0),
            h(b0 - Z, b1 - b0) + h(Z - b1, b1 - b0),
        )
    )
    if show:
        import matplotlib.pyplot as plt

        plt.scatter(Z, X, 5, c=w)
        plt.colorbar()
        plt.show()

    wei = Function(V, w, name="weighting_function")
    return wei


def butter_lowpass_filter(shot, cutoff, fs, order=2):
    """Low-pass filter the shot record with sampling-rate fs Hz
    and cutoff freq. Hz
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    nr, nc = np.shape(shot)
    filtered_shot = np.zeros((nr, nc))
    for rec, ts in enumerate(shot.T):
        filtered_shot[:, rec] = filtfilt(b, a, ts)
    return filtered_shot


def compute_functional(model, comm, residual):
    """ Compute the functional to be optimized """
    num_receivers = model["acquisition"]["num_receivers"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]
    nt = int(tf / dt)  # number of timesteps

    Jtemp = 0.0
    J = 0.0
    Jlist = []
    for ti in range(nt):
        for rn in range(num_receivers):
            Jtemp += 0.5 * (residual[ti][rn] ** 2)
        Jlist.append(Jtemp)
    # Integrate in time (trapezoidal rule)
    for i in range(1, nt - 1):
        J += 0.5 * (Jlist[i - 1] + Jlist[i]) * float(dt)
    J = 0.5 * float(J)
    return J


def evaluate_misfit(model, my_ensemble, guess, exact):
    """Compute the difference between the guess and exact
    at the receiver locations"""

    if "skip" in model["timeaxis"]:
        skip = model["timeaxis"]["skip"]
    else:
        skip = 1
    l = int(exact.shape[0] / skip)
    ds_exact = exact[::skip]
    return ds_exact[:l] - guess


def myrank(COMM=COMM_SELF):
    return COMM.Get_rank()


def mysize(COMM=COMM_SELF):
    return COMM.Get_size()


def mpi_init(model):
    """ Initialize computing environment """
    rank = myrank()
    size = mysize()
    available_cores = COMM_WORLD.size

    if model["parallelism"]["type"] == "automatic":
        num_cores_per_shot = available_cores / model["acquisition"]["num_sources"]
        if available_cores % model["acquisition"]["num_sources"] != 0:
            raise ValueError(
                "Available cores cannot be divided between sources equally."
            )

    elif model["parallelism"]["type"] == "off":
        num_cores_per_shot = available_cores
    elif model["parallelism"]["type"] == "custom":
        raise ValueError("Custom parallelism not yet implemented")

    comm_ens = Ensemble(COMM_WORLD, num_cores_per_shot)
    return comm_ens


def communicate(array, my_ensemble):
    """Communicate shot record to all processors

    Parameters
    ----------
    array: array-like
        Array of data to all-reduce across both ensemble
        and spatial communicators.
    comm: Firedrake.comm
        A Firedrake ensemble communicator

    Returns
    -------
    array_reduced: array-like
        Array of data max all-reduced
        amongst the ensemble communicator

    """
    array_reduced = copy.copy(array)
    if my_ensemble.comm.size > 1:
        if my_ensemble.comm.rank == 0 and my_ensemble.ensemble_comm.rank == 0:
            print("Spatial parallelism, reducing to comm 0", flush=True)
        my_ensemble.comm.Allreduce(array, array_reduced, op=MPI.MAX)
    return array_reduced


def analytical_solution_for_pressure_based_on_MMS(model, mesh, time):
    """to do docstring"""
    degree = model["opts"]["degree"]
    V = FunctionSpace(mesh, "CG", degree)
    z, x = SpatialCoordinate(mesh)
    p = Function(V).interpolate((time ** 2) * sin(pi * z) * sin(pi * x))
    return p
