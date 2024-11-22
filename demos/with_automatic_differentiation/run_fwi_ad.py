import firedrake as fire
import firedrake.adjoint as fire_ad
from checkpoint_schedules import Revolve
import spyro
from demos.with_automatic_differentiation import utils
import os
os.environ["OMP_NUM_THREADS"] = "1"

# --- Basid setup to run a FWI --- #
model = utils.model_settings()


def forward(
        c, compute_functional=False, true_data_receivers=None, annotate=False
):
    """Time-stepping acoustic forward solver.

    The time integration is done using a central difference scheme.

    Parameters
    ----------
    c : firedrake.Function
        Velocity field.
    compute_functional : bool, optional
        Whether to compute the functional. If True, the true receiver
        data must be provided.
    true_data_receivers : list, optional
        True receiver data. This is used to compute the functional.
    annotate : bool, optional
        If True, the forward model is annotated for automatic differentiation.

    Returns
    -------
    (receiver_data : list, J_val : float)
        Receiver data and functional value.
    """
    if annotate:
        fire_ad.continue_annotation()
        if model["aut_dif"]["checkpointing"]:
            total_steps = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
            steps_store = int(total_steps / 10)  # Store 10% of the steps.
            tape = fire_ad.get_working_tape()
            tape.progress_bar = fire.ProgressBar
            tape.enable_checkpointing(Revolve(total_steps, steps_store))

    if model["parallelism"]["type"] is None:
        outfile = fire.VTKFile("solution.pvd")
        receiver_data = []
        J = 0.0
        for sn in range(len(model["acquisition"]["source_pos"])):
            rec_data, J_val = forward_solver.execute_acoustic(c, sn, wavelet)
            receiver_data.append(rec_data)
            J += J_val
            sol = forward_solver.solution
            outfile.write(sol)

    else:
        # source_number based on the ensemble.ensemble_comm.rank
        source_number = my_ensemble.ensemble_comm.rank
        receiver_data, J = forward_solver.execute_acoustic(
            c, source_number, wavelet,
            compute_functional=compute_functional,
            true_data_receivers=true_data_receivers
        )
        sol = forward_solver.solution
        fire.VTKFile(
            "solution_" + str(source_number) + ".pvd", comm=my_ensemble.comm
        ).write(sol)

    return receiver_data, J


# Use emsemble parallelism.
M = model["parallelism"]["num_spacial_cores"]
my_ensemble = fire.Ensemble(fire.COMM_WORLD, M)
mesh = fire.UnitSquareMesh(50, 50, comm=my_ensemble.comm)
element = fire.FiniteElement(
    model["opts"]["method"], mesh.ufl_cell(), degree=model["opts"]["degree"],
    variant=model["opts"]["quadrature"]
)
V = fire.FunctionSpace(mesh, element)


forward_solver = spyro.solvers.forward_ad.ForwardSolver(model, mesh, V)
# Camembert model.
c_true = utils.make_c_camembert(mesh, V)
# Ricker wavelet
wavelet = spyro.full_ricker_wavelet(
    model["timeaxis"]["dt"], model["timeaxis"]["tf"],
    model["acquisition"]["frequency"],
)

true_rec, _ = forward(c_true)

# --- FWI with AD --- #
c_guess = utils.make_c_camembert(mesh, V, c_guess=True)
guess_rec, J = forward(
    c_guess, compute_functional=True, true_data_receivers=true_rec,
    annotate=True
)

# :class:`~.EnsembleReducedFunctional` is employed to recompute in
# parallel the functional and its gradient associated with the multiple sources
# (3 in this case).
J_hat = fire_ad.EnsembleReducedFunctional(
    J, fire_ad.Control(c_guess), my_ensemble)
c_optimised = fire_ad.minimize(J_hat, method="L-BFGS-B",
                               options={"disp": True, "maxiter": 10},
                               bounds=(1.5, 3.5),
                               derivative_options={"riesz_representation": 'l2'})

fire.VTKFile("c_optimised.pvd").write(c_optimised)
