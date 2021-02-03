from firedrake import *
from firedrake.petsc import PETSc

import h5py
import copy
from mpi4py import MPI
import numpy as np
import math
from scipy.signal import butter, filtfilt

from ..domains import quadrature

def helmholtz_filter(u, r_min):
    """Smooth scalar field"""

    V = u.function_space()
    qr_x, _, _ = quadrature.quadrature_rules(V)

    s = Function(V)
    u_ = TrialFunction(V)
    v = TestFunction(V)
    a = r_min**2*inner(grad(u_), grad(v))*dx(rule=qr_x) + u_*v*dx(rule=qr_x)
    L = u*v*dx(rule=qr_x) 
    parameters = {'kse_type': 'preonly', 'pctype': 'lu'}
    solve(a == L, s, solver_parameters=parameters)

    return s


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


def pml_error(model, p_pml, p_ref):
    """ Erro with PML for a each shot (source) ..."""

    num_sources = model["acquisition"]["num_sources"]
    num_receivers = model["acquisition"]["num_receivers"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]

    nt = int(tf / dt)  # number of timesteps
    error = []

    for sn in range(num_sources):
        error.append([])
        for ti in range(nt):
            soma = 0
            for rn in range(num_receivers):
                soma += (p_pml[sn][rn][ti] - p_ref[sn][rn][ti]) * (
                    p_pml[sn][rn][ti] - p_ref[sn][rn][ti]
                )
            error[sn].append(math.sqrt(soma / num_receivers))

    return error


def compute_functional(model, comm, residual):
    """ Compute the functional to be optimized """
    num_receivers = model["acquisition"]["num_receivers"]
    dt = model["timeaxis"]["dt"]
    tf = model["timeaxis"]["tf"]
    nt = int(tf / dt)  # number of timesteps

    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
        print("Computing the functional...", flush=True)

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

    if my_ensemble.comm.rank == 0 and my_ensemble.ensemble_comm.rank == 0:
        print("Computing the misfit...", flush=True)

    return guess - exact


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
        num_cores_per_shot = available_cores/model["acquisition"]["num_sources"]
        if available_cores % model["acquisition"]["num_sources"] != 0:
            raise ValueError("Available cores cannot be divided between sources equally.")

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
    degree = model["opts"]["degree"]
    V = FunctionSpace(mesh, "CG", degree)
    z, x = SpatialCoordinate(mesh)
    p = Function(V).interpolate((time ** 2) * sin(pi * z) * sin(pi * x))
    return p


def normalize_vp(model, vp):

    control = firedrake.Function(vp)

    if "material" in model:
        if model["material"]["type"] is "simp":
            vp_min = model["material"]["vp_min"]
            vp_max = model["material"]["vp_max"]
            penal = model["material"]["penal"]
            control.dat.data[:] -= vp_min
            control.dat.data[:] /= (vp_max - vp_min)
            control.dat.data[:] = control.dat.data[:] ** (1 / penal)

    return control


def discretize_field(c, bins=None, n=4):

    c_ = firedrake.Function(c)
    vp = c.dat.data
    # Get histogram
    if bins is None:
        counts, bins = np.histogram(c.dat.data, bins=n)
    else:
        counts, _ = np.histogram(c.dat.data, bins=n)

    for i, count in enumerate(counts):
        c_.dat.data[(bins[i] <= vp)&(vp <= bins[i+1])] = (bins[i]+bins[i+1])/2

    return c_, bins


def control_to_vp(model, control):

    vp = firedrake.Function(control)

    if "material" in model:
        if model["material"]["type"] is "simp":
            vp_min = Constant(model["material"]["vp_min"])
            vp_max = Constant(model["material"]["vp_max"])
            penal = Constant(model["material"]["penal"])

            vp.assign(vp_min + (vp_max - vp_min) * control ** penal)

    return vp


def save_velocity_model(comm, vp_model, dest_file):
    """Save Firedrake.Function representative of a seismic velocity model.
    Stores both nodal values and coordinates into a HDF5 file.

    Parameters
    comm: Firedrake.ensemble_communicator
        The MPI communicator for parallelism
    vp_model: Firedrake.Function
        P-wave seismic velocity interpolated onto the nodes of the
        finite elements.
    dest_file: str
        path to hdf5 file to be written.

    """
    # Sanitize units
    _check_units(vp_model)
    # # Get coordinates
    V = vp_model.function_space()
    W = firedrake.VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
    coords = firedrake.interpolate(V.ufl_domain().coordinates, W)

    # Gather vectors on the master rank
    vp_global = vp_model.vector().gather()
    coords_global = coords.vector().gather()

    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
        print("Writing velocity model: " + dest_file, flush=True)
        with h5py.File(dest_file, "w") as f:
            f.create_dataset("velocity_model", data=vp_global, dtype="f")
            f.create_dataset("coordinates", data=coords_global, dtype="f")
            f.attrs["geometric dimension"] = coords.dat.data.shape[1]
            f.attrs["units"] = "km/s"


def load_velocity_model(params, V, source_file=None):
    """Load Firedrake.Function representative of a seismic velocity model
    from a HDF5 file.

    Prameters
    ---------
    V: Firedrake.FunctionSpace object
        The space of the finite elements.
    dsource_file: str
        path to hdf5 file to be loaded.

    Returns
    -------
    vp_model: Firedrake.Function
        P-wave seismic velocity interpolated onto the nodes of the
        finite elements.

    """

    if not source_file:
        source_file = params['input']['model']

    # Get interpolant
    with h5py.File(source_file, "r") as f:
        vp = np.asarray(f.get("velocity_model")[()])
        gd = f.attrs['geometric dimension']
        coords = np.asarray(f.get("coordinates")[()])
        coords = coords.reshape((-1, gd))

    interpolant = NearestNDInterpolator(coords, vp)

    # Get current coordinates
    W = firedrake.VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
    coordinates = firedrake.interpolate(V.ufl_domain().coordinates, W)

    # Get velocity model
    vp_model = firedrake.Function(V)
    vp_model.dat.data[:] = interpolant(coordinates.dat.data)

    return _check_units(vp_model)


def _check_units(c):
    if min(c.dat.data[:]) > 1000.0:
        # data is in m/s but must be in km/s
        if firedrake.COMM_WORLD.rank == 0:
            print("INFO: converting from m/s to km/s", flush=True)
        c.assign(c / 1000.0)  # meters to kilometers
    return c

def spatial_scatter(comm, xi, u):
    """Scatter xi through processes"""

    # Spatial communicator rank and size
    rank = comm.comm.rank
    size = comm.comm.size

    # Update control xi from rank 0
    xi = COMM_WORLD.bcast(xi, root=0)

    # Update Function u
    n = len(u.dat.data[:])
    N = [comm.comm.bcast(n, r) for r in range(size)]
    indices = np.insert(np.cumsum(N), 0, 0)
    u.dat.data[:] = xi[indices[rank] : indices[rank + 1]]
