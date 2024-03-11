from firedrake import *
from firedrake.adjoint import *
from checkpoint_schedules import Revolve, SingleMemoryStorageSchedule
import spyro
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from mpi4py import MPI
# clear cache constantly to measure memory usage
import gc
import warnings
warnings.filterwarnings("ignore")

# --- Basid setup to run a forward simulation with AD --- #
model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadrature": "KMV",  # Equi or KMV
    "degree": 1,  # p order
    "dimension": 2,  # dimension
    "regularization": False,  # regularization is on?
    "gamma": 1e-5,  # regularization parameter
}

model["parallelism"] = {
    # options:
    # `"shots_parallelism"` (same number of cores for every processor. Apply only
    # shots parallelism, i.e., the spatial domain is not parallelised.)
    # `"automatic"` (same number of cores for every processor. Apply shots and
    # spatial parallelism.)
    # `"spatial"` (Only spatial parallelisation).
    # `None` (No parallelisation).
    "type": "None",
}

# Define the domain size without the ABL.
model["mesh"] = {
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

# Specify a 250-m Absorbing Boundary Layer (ABL) on the three sides of the domain to damp outgoing waves.
model["BCs"] = {
    "status": False,  # True or False, used to turn on any type of BC
    "outer_bc": "non-reflective",  # none or non-reflective (outer boundary condition)
    "abl_bc": "none",  # none, gaussian-taper, or alid
    "lz": 0.0,  # thickness of the ABL in the z-direction (km) - always positive
    "lx": 0.0,  # thickness of the ABL in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "source_pos": spyro.create_transect((0.3, 0.15), (0.7, 0.15), 5),
    "frequency": 7.0,
    "delay": 1.0,
    "receiver_locations": spyro.create_transect((0.2, 0.8), (0.8, 0.8), 10),
}
model["aut_dif"] = {
    "status": True,
}

model["timeaxis"] = {
    "t0": 0.0,  # Initial time for event
    "tf": 1.0,  # Final time for event (for test 7)
    "dt": 0.001,  # timestep size (divided by 2 in the test 4. dt for test 3 is 0.00050)
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 1,  # how frequently to save solution to RAM
}


def make_vp_circle(vp_guess=False, plot_vp=False):
    """Acoustic velocity model"""
    x, z = SpatialCoordinate(mesh)
    if vp_guess:
        vp = Function(V).interpolate(1.5)
    else:
        vp = Function(V).interpolate(
            2.5
            + 1 * tanh(100 * (0.125 - sqrt((x - 0.5) ** 2 + (z - 0.5) ** 2)))
        )
    if plot_vp:
        outfile = File("acoustic_cp.pvd")
        outfile.write(vp)
    return vp
true_receiver_data = []
iterations = 0
nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])  # number of timesteps
def J(mesh, comm, vp_exact, wavelet, vom, source_number, vp_guess,
                plot_receiver_data=False):
    global true_receiver_data
    guess_receiver_data, functional = spyro.solvers.forward_AD(
        model, mesh, comm, vp_guess, wavelet, vom, source_number=source_number,
        true_receiver_data=true_receiver_data[source_number], fwi=True
    )
    if plot_receiver_data:
        data = [rec.dat.data_ro[:] for rec in guess_receiver_data]
        spyro.plots.plot_shots(model, comm, data, vmax=1e-08, vmin=-1e-08)
    return functional

save_J = []
def run_fwi(vp_guess_data):
    global iterations
    if iterations == 0:
        true_data = []
        for sn in range(len(model["acquisition"]["source_pos"])):
            true_receiver_data.append(spyro.solvers.forward_AD(model, mesh, comm, vp_exact,
                                                          wavelet, vom, debug=True,
                                                          source_number=sn))
    J_total = 0.0
    dJ_total = Function(V)
    vp_guess = Function(V)
    vp_guess.dat.data[:] = vp_guess_data
    File("vp_end" + str(iterations) + ".pvd").write(vp_guess)
    if size == 1:
        for sn in range(len(model["acquisition"]["source_pos"])):
            continue_annotation()
            tape = get_working_tape()
            tape.progress_bar = ProgressBar
            get_working_tape().enable_checkpointing(Revolve(nt, nt//4))
            J_total += J(mesh, comm, vp_exact, wavelet, vom, sn, vp_guess)
            dJ_total += compute_gradient(J_total, Control(vp_guess))
            get_working_tape().clear_tape()
    elif size == len(model["acquisition"]["source_pos"]):
        J_local = J(mesh, comm, vp_exact, wavelet, vom, rank, vp_guess)
        dJ_local = compute_gradient(J_local, Control(vp_guess))
        J_total = COMM_WORLD.allreduce(J_local, op=MPI.SUM)
        dJ_total = comm.allreduce(dJ_local, dJ_total, op=MPI.SUM)
    else:
        raise NotImplementedError("`size` must be 1 or equal to `num_sources`."
                                  "Different values are not supported yet.")
    iterations += 1
    return J_total, dJ_total.dat.data[:]


comm, spatial_comm = spyro.utils.mpi_init(model)
if model["parallelism"]["type"] == "shots_parallelism":
    # Only shots parallelism.
    mesh = UnitSquareMesh(100, 100, comm=spatial_comm)
else:
    mesh = UnitSquareMesh(100, 100)

element = spyro.domains.space.FE_method(
    mesh, model["opts"]["method"], model["opts"]["degree"]
)

V = FunctionSpace(mesh, element)
# Receiver mesh.
vom = VertexOnlyMesh(mesh, model["acquisition"]["receiver_locations"])

wavelet = spyro.full_ricker_wavelet(
    dt=model["timeaxis"]["dt"],
    tf=model["timeaxis"]["tf"],
    freq=model["acquisition"]["frequency"],
)
# True acoustic velocity model
vp_exact = make_vp_circle(plot_vp=True)
vp_guess = make_vp_circle(plot_vp=True, vp_guess=True)
# Processor number.
rank = comm.ensemble_comm.rank
# Number of processors used in the simulation.
size = comm.ensemble_comm.size
vmax = 3.5
vmin = 1.5
vp_0 = vp_guess.vector().gather()
bounds = [(vmin, vmax) for _ in range(len(vp_0))]
result_data = scipy_minimize(run_fwi, vp_0, method='L-BFGS-B',
                             jac=True, tol=1e-15, bounds=bounds,
                             options={"disp": True, "eps": 1e-15,
                                      "gtol": 1e-15, "maxiter": 20})
vp_end = Function(V)
vp_end.dat.data[:] = result_data.x
File("vp_end.pvd").write(vp_end)