import firedrake as fire
import firedrake.adjoint as fire_ad
from pyadjoint import set_working_tape, Tape
from checkpoint_schedules import Revolve
from spyro.solvers import DifferentiableWaveEquation, AutomatedGradientOptimisation
from demos.with_automatic_differentiation.utils import \
    model_settings, make_c_camembert
import os
os.environ["OMP_NUM_THREADS"] = "1"

# --- Basid setup to run a FWI --- #
model = model_settings()
fire_ad.continue_annotation()

# Use emsemble parallelism.
M = model["parallelism"]["num_spacial_cores"]
my_ensemble = fire.Ensemble(fire.COMM_WORLD, M)
mesh = fire.UnitSquareMesh(80, 80, comm=my_ensemble.comm)
wave_equation = DifferentiableWaveEquation(model, mesh)


def run_forward(
        c, to_generate_true_data=False, compute_functional=False,
        true_data_receivers=None, annotate=False):
    if annotate:
        fire_ad.continue_annotation()
    # source_number based on the ensemble.ensemble_comm.rank
    source_number = my_ensemble.ensemble_comm.rank
    wave_equation.acoustic_solver(
        c, source_number, compute_functional=compute_functional,
        true_data_receivers=true_data_receivers)
    if to_generate_true_data:
        return wave_equation.receiver_data
    if compute_functional and annotate:
        return fire_ad.EnsembleReducedFunctional(
            wave_equation.functional_value, fire_ad.Control(c),
            my_ensemble)


c_true = make_c_camembert(mesh, wave_equation.function_space)

# Get the true data.
true_data_rec = run_forward(c_true, to_generate_true_data=True)

# --- FWI with AD --- #
c_guess = make_c_camembert(mesh, wave_equation.function_space, c_guess=True)
J_hat = run_forward(
    c_guess, compute_functional=True, true_data_receivers=true_data_rec,
    annotate=True
)

# Optimisation
c_optimised = AutomatedGradientOptimisation(J_hat).minimise_scipy(
    bounds=(1.5, 2.5), max_iter=20)

fire.VTKFile("c_optimised.pvd").write(c_optimised)
