import copy
from firedrake import *
import numpy as np
from mpi4py import MPI
from scipy.signal import butter, filtfilt


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


def compute_functional_ad(p_rec, p_true_rec, P, comm, local_rec_index):
    """Compute the functional to be optimized.
    This function computes the misfit at the time t,
    and must be used when automatic differention 
    is employed

    Parameters
    ----------
    p_rec: Firedrake.interpolator 
        Solution stored at the receivers at the time t
        and interpolated onto the VertexOnlyMesh
    p_true_rec:  numpy data
        Receiver data related to the True velocity model
    P: Firedrake.FunctionSpace
        DG0 Function Space
    comm: Firedrake.ensemble_communicator
        The MPI communicator for parallelism
    local_rec_index: python list
        List of the local receiver index 

    Returns
    -------
    J: float
        Cost function
    misfit_data: numpy data
        Misfit function
    """
    true_rec = scatter_data_function(
                        p_true_rec, P, comm,
                        local_rec_index, name="true_rec"
                        )
    J = assemble(0.5*inner(p_rec-true_rec, p_rec-true_rec) * dx)   
    misfit_data = p_rec.dat.data[:] - true_rec.dat.data[:]
    return J, misfit_data
    
    
def compute_functional(model, residual, vp=None):
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
        gamma = model["opts"]["gamma"]
        Ns = model["acquisition"]["num_sources"]
        gamma /= Ns

    J = 0.0
    for ti in range(nt):
        for rn in range(num_receivers):
            J += residual[ti][rn] ** 2
    J *= 0.5

    if regularize:
        Jreg = assemble(0.5 * gamma * dot(grad(vp), grad(vp)) * dx)
        print(">>>>>> VALUE OF GRAD REGULATIZATION = ", Jreg)
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
    # print(array_reduced, array)
    return array_reduced


def analytical_solution_for_pressure_based_on_MMS(model, mesh, time):
    degree = model["opts"]["degree"]
    V = FunctionSpace(mesh, "CG", degree)
    z, x = SpatialCoordinate(mesh)
    p = Function(V).interpolate((time ** 2) * sin(pi * z) * sin(pi * x))
    return p


def scatter_data_function(arr, space_f, comm, local_index, name=None):

    # if comm.ensemble_comm.rank == 0:
    f = Function(space_f, name=name)
    n = len(f.dat.data[:])

    if len(local_index) > 0:
        index_0 = local_index[0]
        
        f.dat.data[:] = arr[index_0:(index_0+n)]

    return f

