import copy
from firedrake import *
import numpy as np
from mpi4py import MPI
from scipy.signal import butter, filtfilt

def weiner_filter_shot(shot, desired_frequency, dt, final_time):
    nr, nc = np.shape(shot)
    filtered_shot = np.zeros((nr, nc))
    target = spyro.full_ricker_wavelet(dt,final_time,filter_frequency)
    target_f = fft(target)
    for rec, ts in enumerate(shot.T):
        source = copy.deepcopy(shot[:, rec])
        
        e = 0.001
        source_f = fft(source)
        f = target_f * np.conjugate(source_f) / ( np.abs(source_f)**2  +e**2 )
        new_source_f = f*source_f
        new_source = ifft(new_source_f)
        filtered_shot[:, rec] = new_source

    return filtered_shot

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

    if regularize:
        Jreg = assemble(0.5 * gamma * dot(grad(vp), grad(vp)) * dx)
        J += Jreg
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


def mpi_init(model):
    """Initialize computing environment"""
    rank = myrank()
    size = mysize()
    available_cores = COMM_WORLD.size
    if model["parallelism"]["type"] == "automatic":
        num_cores_per_shot = available_cores / len(model["acquisition"]["source_pos"])
        if available_cores % len(model["acquisition"]["source_pos"]) != 0:
            raise ValueError(
                "Available cores cannot be divided between sources equally."
            )
    elif model["parallelism"]["type"] == "spatial":
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
    # print(array_reduced,array)
    return array_reduced


def analytical_solution_for_pressure_based_on_MMS(model, mesh, time):
    degree = model["opts"]["degree"]
    V = FunctionSpace(mesh, "CG", degree)
    z, x = SpatialCoordinate(mesh)
    p = Function(V).interpolate((time ** 2) * sin(pi * z) * sin(pi * x))
    return p
