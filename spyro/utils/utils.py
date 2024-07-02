import copy
from firedrake import *  # noqa: F403
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


def compute_functional(Wave_object, residual):
    """Compute the functional to be optimized.
    Accepts the velocity optionally and uses
    it if regularization is enabled
    """
    num_receivers = Wave_object.number_of_receivers
    dt = Wave_object.dt
    comm = Wave_object.comm

    J = 0
    for rn in range(num_receivers):
        J += np.trapz(residual[:, rn] ** 2, dx=dt)

    J *= 0.5

    J_total = np.zeros((1))
    J_total[0] += J
    J_total = COMM_WORLD.allreduce(J_total, op=MPI.SUM)
    J_total[0] /= comm.comm.size
    return J_total[0]


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


def myrank(COMM=COMM_SELF):  # noqa: F405
    return COMM.Get_rank()


def mysize(COMM=COMM_SELF):  # noqa: F405
    return COMM.Get_size()


def mpi_init(model):
    """Initialize computing environment"""
    # rank = myrank()
    # size = mysize()
    available_cores = COMM_WORLD.size  # noqa: F405
    print(f"Parallelism type: {model.parallelism_type}", flush=True)
    if model.parallelism_type == "automatic":
        num_cores_per_shot = available_cores / model.number_of_sources
        if available_cores % model.number_of_sources != 0:
            raise ValueError(
                f"Available cores cannot be divided between sources equally {available_cores}/{model.number_of_sources}."
            )
    elif model.parallelism_type == "spatial":
        num_cores_per_shot = available_cores
    elif model.parallelism_type == "custom":
        raise ValueError("Custom parallelism not yet implemented")

    comm_ens = Ensemble(COMM_WORLD, num_cores_per_shot)  # noqa: F405
    return comm_ens


def mpi_init_simple(number_of_sources):
    """Initialize computing environment"""
    rank = myrank()  # noqa: F841
    size = mysize()  # noqa: F841
    available_cores = COMM_WORLD.size  # noqa: F405

    num_cores_per_shot = available_cores / number_of_sources
    if available_cores % number_of_sources != 0:
        raise ValueError(
            "Available cores cannot be divided between sources equally."
        )

    comm_ens = Ensemble(COMM_WORLD, num_cores_per_shot)  # noqa: F405
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
    # print(array_reduced,array)
    return array_reduced


# def analytical_solution_for_pressure_based_on_MMS(model, mesh, time):
#     degree = model["opts"]["degree"]
#     V = FunctionSpace(mesh, "CG", degree)  # noqa: F405
#     z, x = SpatialCoordinate(mesh)  # noqa: F405
#     p = Function(V).interpolate(  # noqa: F405
#         (time**2) * sin(pi * z) * sin(pi * x)  # noqa: F405
#     )
#     return p
