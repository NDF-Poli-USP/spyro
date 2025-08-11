import firedrake as fire
import firedrake.adjoint as fire_ad
from spyro.solvers import DifferentiableWaveEquation
import spyro
from numpy.random import rand
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

M = model["parallelism"]["num_spacial_cores"]
my_ensemble = fire.Ensemble(fire.COMM_WORLD, M)
mesh = fire.UnitSquareMesh(20, 20, comm=my_ensemble.comm)
wave_equation = DifferentiableWaveEquation(model, mesh)


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
    return c


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


def test_taylor():
    # Test only serial for now.
    c_true = make_c_camembert(wave_equation.function_space, mesh)
    true_data_rec = run_forward(c_true, to_generate_true_data=True)

    # --- Gradient with AD --- #
    c_guess = make_c_camembert(wave_equation.function_space, mesh, c_guess=True)
    J_hat = run_forward(
        c_guess, compute_functional=True, true_data_receivers=true_data_rec,
        annotate=True
    )
    h = fire.Function(wave_equation.function_space)
    h.dat.data[:] = rand(wave_equation.function_space.dim())
    assert fire_ad.taylor_test(J_hat, c_guess, h) > 1.9
    fire_ad.get_working_tape().clear_tape()
    fire_ad.pause_annotation()
