import firedrake as fire
import firedrake.adjoint as fire_ad
import spyro
from numpy.random import rand
from checkpoint_schedules import Revolve
import pytest


# --- Basid setup to run a forward simulation with AD --- #
model = {}

model["options"] = {
    "cell_type": "T",  # T or Q
    "variant": "lumped",
    "degree": 1,  # p order
    "dimension": 2,  # dimension
}

model["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the ABL.
model["mesh"] = {
    "Lz": 1.0,  # depth in km - always positive
    "Lx": 1.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
}

model["absorving_boundary_conditions"] = {
    "status": False,
    "pad_length": 0.,
}

model["acquisition"] = {
    "source_type": "ricker",
    "source_locations": spyro.create_transect((0.2, 0.15), (0.8, 0.15), 1),
    "frequency": 7.0,
    "delay": 1.0,
    "receiver_locations": spyro.create_transect((0.2, 0.2), (0.8, 0.2), 10),
}
model["aut_dif"] = {
    "status": True,
    "checkpointing": False,
}

model["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": 0.4,  # Final time for event (for test 7)
    "dt": 0.004,  # timestep size (divided by 2 in the test 4. dt for test 3 is 0.00050)
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM
}


def make_c_camembert(V, mesh, c_guess=False, plot_c=False):
    """Acoustic velocity model"""
    x, z = fire.SpatialCoordinate(mesh)
    if c_guess:
        c = fire.Function(V).interpolate(1.5 + 0.0 * x)
    else:
        c = fire.Function(V).interpolate(
            2.5
            + 1 * fire.tanh(100 * (0.125 - fire.sqrt((x - 0.5) ** 2 + (z - 0.5) ** 2)))
        )
    if plot_c:
        outfile = fire.VTKFile("acoustic_cp.pvd")
        outfile.write(c)
    return c


def forward(
        c, fwd_solver, wavelet, ensemble,
        compute_functional=False, true_data_receivers=None, annotate=False
):
    if annotate:
        fire_ad.continue_annotation()
        if model["aut_dif"]["checkpointing"]:
            total_steps = int(model["time_axis"]["final_time"] / model["time_axis"]["dt"]) + 1
            steps_store = int(total_steps / 10)  # Store 10% of the steps.
            tape = fire_ad.get_working_tape()
            tape.progress_bar = fire.ProgressBar
            tape.enable_checkpointing(Revolve(total_steps, steps_store))
    # source_number based on the ensemble.ensemble_comm.rank
    source_number = ensemble.ensemble_comm.rank
    receiver_data, J = fwd_solver.execute_acoustic(
        c, source_number, wavelet,
        compute_functional=compute_functional,
        true_data_receivers=true_data_receivers
    )
    return receiver_data, J


def test_taylor():
    # Test only serial for now.
    M = 1
    my_ensemble = fire.Ensemble(fire.COMM_WORLD, M)
    mesh = fire.UnitSquareMesh(20, 20, comm=my_ensemble.comm)
    ufl_cell_obj = mesh.ufl_cell()
    if ufl_cell_obj._cellname == "triangle":
        method = "KMV"
    else:
        method = "CG"
    print(f"Using {method} method")
    element = fire.FiniteElement(
        method, mesh.ufl_cell(), degree=model["options"]["degree"]
    )
    V = fire.FunctionSpace(mesh, element)

    fwd_solver = spyro.solvers.forward_ad.ForwardSolver(model, mesh, V)
    # Ricker wavelet
    wavelet = spyro.full_ricker_wavelet(
        model["time_axis"]["dt"], model["time_axis"]["final_time"],
        model["acquisition"]["frequency"],
    )
    c_true = make_c_camembert(V, mesh)
    true_rec, _ = forward(c_true, fwd_solver, wavelet, my_ensemble)

    # --- Gradient with AD --- #
    c_guess = make_c_camembert(V, mesh, c_guess=True)
    _, J = forward(
        c_guess, fwd_solver, wavelet, my_ensemble,
        compute_functional=True,
        true_data_receivers=true_rec, annotate=True
    )

    # :class:`~.EnsembleReducedFunctional` is employed to recompute in
    # parallel the functional and its gradient associated.
    J_hat = fire_ad.EnsembleReducedFunctional(
        J, fire_ad.Control(c_guess), my_ensemble)
    h = fire.Function(V)
    h.dat.data[:] = rand(V.dim())
    assert fire_ad.taylor_test(J_hat, c_guess, h) > 1.9
    fire_ad.get_working_tape().clear_tape()
    fire_ad.pause_annotation()


@pytest.mark.skip(reason="Breaking everywhere, even in main if retested")
def test_taylor_checkpointing():
    model["aut_dif"]["checkpointing"] = True
    test_taylor()
