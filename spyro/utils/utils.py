import copy
from firedrake import *
import numpy as np
from mpi4py import MPI
from scipy.signal import butter, filtfilt


def butter_lowpass_filter(shot, cutoff, fs, order=2):
    """Low-pass filter the shot record with sampling-rate fs Hz
    and cutoff freq. Hz

    Parameters
    ----------
    shot : numpy array
        Shot record
    cutoff : float
        Cutoff frequency in Hertz
    fs : float
        Sampling rate in Hertz
    order : int
        Order of the filter

    Returns
    -------
    filtered_shot : numpy array
        Filtered shot record
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


def compute_functional(model, residual, velocity=None):
    """Compute the functional to be optimized.
    Accepts the velocity optionally and uses
    it if regularization is enabled
    """
    num_receivers = len(model["acquisition"]["receiver_locations"])
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]
    nt = int(tf / dt)  # number of timesteps
    if "regularization" in model["opts"]:
        regularize = model["opts"]["regularization"]
    else:
        regularize = False

    if regularize:
        gamma = model["opt"]["gamma"]
        Ns = model["acquisition"]["num_sources"]
        gamma /= Ns

    J = 0.0
    for ti in range(nt):
        for rn in range(num_receivers):
            J += residual[ti][rn] ** 2
    J *= 0.5

    # if regularize:
    #     Jreg = assemble(0.5 * gamma * dot(grad(vp), grad(vp)) * dx)
    #     J += Jreg
    return J


def evaluate_misfit(model, guess, exact):
    """Compute the difference between the guess and exact
    at the receiver locations given downsampling"""

    if "skip" in model["timeaxis"]:
        skip = model["timeaxis"]["skip"]
    else:
        skip = 1

    ll = int(exact.shape[0] / skip)
    ds_exact = exact[::skip]
    return ds_exact[:ll] - guess


def myrank(COMM=COMM_SELF):
    return COMM.Get_rank()


def mysize(COMM=COMM_SELF):
    return COMM.Get_size()


def mpi_init(model, spatial_core_parallelism=None):
    """Initialize computing environment"""
    # rank = myrank()
    # size = mysize()
    if (
        model["parallelism"]["type"] == "shots_parallelism"
        or model["parallelism"]["type"] == "automatic"
    ):
        
        num_cores_per_shot = COMM_WORLD.size / len(model["acquisition"]["source_pos"])
        if COMM_WORLD.size % len(model["acquisition"]["source_pos"]) != 0:
            raise ValueError(
                "Available cores cannot be divided between sources equally."
            )
    elif model["parallelism"]["type"] == "spatial":
        # Parallellism is only over spatial domain. No shots parallelism.
        num_cores_per_shot = COMM_WORLD.size
    elif model["parallelism"]["type"] == "custom":
        raise ValueError("Custom parallelism not yet implemented")
    elif model["parallelism"]["type"] == "none":
        num_cores_per_shot = 1

    if model["parallelism"]["type"] == "shots_parallelism":
        # Parrallellism is over shots. No spatial parallelism.
        spatial_comm = COMM_WORLD.Split(
            COMM_WORLD.rank % len(model["acquisition"]["source_pos"])
            )
    else:
        spatial_comm = None
    comm_ens = Ensemble(COMM_WORLD, num_cores_per_shot)
    return comm_ens, spatial_comm


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
    # print(array_reduced,array)
    return array_reduced


def communicate_receiver_vom(array, comm):
    """Gather all coordinates from all ranks.

    Parameters
    ----------
    array: numpy array
        Array of data to gather across all ranks.
    comm: Firedrake.comm
        Firedrake ensemble communicator.

    Returns
    -------
    array: numpy array
        Array of data gathered across all ranks.
    """
    array = array.copy()
    array = comm.allgather(array)
    array = np.concatenate(array)
    return array

def analytical_solution_for_pressure_based_on_MMS(model, mesh, time):
    degree = model["opts"]["degree"]
    V = FunctionSpace(mesh, "CG", degree)
    z, x = SpatialCoordinate(mesh)
    p = Function(V).interpolate((time**2) * sin(pi * z) * sin(pi * x))
    return p
