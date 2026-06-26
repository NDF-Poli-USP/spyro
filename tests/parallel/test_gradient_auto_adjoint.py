"""Parallel automated-adjoint gradient verification.

This test exercises spyro's *automated adjoint* (algorithmic differentiation via
pyadjoint) under **ensemble (shot) parallelism**. It is meant to be run with two
MPI ranks::

    mpiexec -n 2 pytest tests/parallel/test_gradient_auto_adjoint.py

With ``parallelism = "automatic"`` and two sources, spyro sets
``num_cores_per_propagation = available_cores / number_of_sources = 2 / 2 = 1``
and builds an ``Ensemble(COMM_WORLD, 1)``. This yields **two ensemble members**
(``ensemble_comm`` of size 2), one per source, each integrating its shot on a
single spatial core (``comm`` of size 1).

Each ensemble member records the forward solve for its own source on its own
pyadjoint tape and accumulates that shot's functional ``J_i``. The reduced
functional created by :meth:`AutomatedAdjoint.create_reduced_functional` is a
:class:`firedrake.adjoint.EnsembleReducedFunctional` whose ``ensemble`` argument
is ``wave.comm``; differentiating it ``allreduce``-sums
``dJ/dm = sum_i dJ_i/dm`` over the ensemble communicator. The gradient is then
validated with a Taylor test.

Note
----
The automated adjoint evaluates the per-timestep misfit on a vertex-only mesh of
the receivers. That path parallelises over *shots* (one core per shot), exactly
like spyro's ensemble FWI tests; it does not split the receivers across spatial
cores. This test therefore uses one spatial core per shot. The perturbation
direction is built from a deterministic function of the mesh coordinates so that
it is identical on every ensemble member -- a requirement for the ensemble Taylor
test to converge at second order.
"""
import firedrake as fire
import firedrake.adjoint as fire_ad
import spyro
import pytest


final_time = 0.6

dictionary = {}
dictionary["options"] = {
    "cell_type": "T",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}

# "automatic" ensemble parallelism: with two sources and two MPI ranks each
# ensemble member integrates one shot (source parallelism) on a single core.
dictionary["parallelism"] = {
    "type": "automatic",
}

dictionary["mesh"] = {
    "length_z": 1.0,  # depth in km - always positive
    "length_x": 1.0,  # width in km - always positive
    "length_y": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}

# Two sources => two propagations distributed across the ensemble.
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-0.1, 0.3), (-0.1, 0.7)],
    "frequency": 5.0,
    "delay": 1.5,
    "delay_type": "multiples_of_minimum",
    "receiver_locations": spyro.create_transect((-0.8, 0.2), (-0.8, 0.8), 10),
}

dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": 0.0005,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds
    "gradient_sampling_frequency": 1,  # how frequently to save solution to RAM
}

dictionary["visualization"] = {
    "forward_output": False,
    "forward_output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": "results/Gradient.pvd",
    "adjoint_output": False,
    "adjoint_filename": None,
    "debug_output": False,
}


def build_direction(Wave_obj):
    """Build a deterministic perturbation direction shared across the ensemble.

    The direction is interpolated from a smooth function of the mesh
    coordinates, so it represents the *same* field on every ensemble member.
    This consistency across the ensemble is required for the ensemble Taylor
    test to converge at second order: the ``EnsembleReducedFunctional`` evaluates
    each ``J_i`` at the control value it is handed, so a direction that differed
    between members would corrupt the summed directional derivative.

    Parameters
    ----------
    Wave_obj : spyro.AcousticWave
        Wave object whose control function space and mesh coordinates are used.

    Returns
    -------
    firedrake.Function
        The perturbation direction in the control function space.
    """
    z = Wave_obj.mesh_z
    x = Wave_obj.mesh_x
    direction = fire.Function(Wave_obj.c.function_space(), name="direction")
    # Smooth O(1) field, deterministic and identical on every ensemble member.
    direction.interpolate(1.0 + 0.25 * fire.sin(3.0 * x) * fire.cos(3.0 * z))
    return direction


def get_forward_model():
    """Build exact and guess models and record the automated-adjoint tape.

    The exact model uses a two-layer velocity contrast; the guess model is a
    constant background. Under ensemble parallelism each member only solves and
    records the shot it owns, so ``rec_out_exact`` already corresponds to that
    member's source.

    Returns
    -------
    Wave_obj_guess : spyro.AcousticWave
        Guess model whose forward solve has been taped with the automated
        adjoint enabled and recording stopped.
    """
    # Exact model (observed data).
    Wave_obj_exact = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_exact.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    cond = fire.conditional(Wave_obj_exact.mesh_z > -0.5, 1.5, 3.5)
    Wave_obj_exact.set_initial_velocity_model(
        conditional=cond,
        dg_velocity_model=False,
    )
    Wave_obj_exact.forward_solve()
    rec_out_exact = Wave_obj_exact.forward_solution_receivers

    # Guess model (control to be differentiated).
    Wave_obj_guess = spyro.AcousticWave(dictionary=dictionary)
    Wave_obj_guess.real_shot_record = rec_out_exact
    Wave_obj_guess.set_mesh(input_mesh_parameters={"edge_length": 0.1})
    Wave_obj_guess.set_initial_velocity_model(constant=2.0)

    # The control must be a Function for pyadjoint to differentiate it.
    assert isinstance(Wave_obj_guess.c, fire.Function)
    Wave_obj_guess.enable_automated_adjoint()

    # The ensemble passed to the EnsembleReducedFunctional is wave.comm.
    assert Wave_obj_guess.automated_adjoint.ensemble is Wave_obj_guess.comm

    Wave_obj_guess.forward_solve()

    # The forward solve must have produced a tape and an AdjFloat functional.
    assert Wave_obj_guess.automated_adjoint._tape is not None
    Wave_obj_guess.automated_adjoint.stop_recording()

    return Wave_obj_guess


@pytest.mark.newer_firedrake
@pytest.mark.parallel(2)
def test_gradient_auto_adjoint_parallel():
    """Taylor-test the ensemble automated-adjoint gradient.

    Runs on two cores: two sources (ensemble parallelism), one core per shot.
    """
    Wave_obj_guess = get_forward_model()

    # Sanity check the ensemble (shot) parallelism is active, one core per shot.
    comm = Wave_obj_guess.comm
    assert comm.ensemble_comm.size == 2, "Expected 2 ensemble members (sources)."
    assert comm.comm.size == 1, "Expected 1 spatial core per shot."

    # Build the reduced functional. With wave.comm as the ensemble this is an
    # EnsembleReducedFunctional summing the per-shot functionals/gradients.
    reduced_functional = Wave_obj_guess.automated_adjoint.create_reduced_functional(
        Wave_obj_guess.functional_value
    )
    assert isinstance(
        reduced_functional, fire_ad.EnsembleReducedFunctional
    ), "Reduced functional must be an EnsembleReducedFunctional."

    # The ensemble-summed gradient is a Function in the control space.
    dJ = Wave_obj_guess.automated_adjoint.compute_gradient()
    assert isinstance(dJ, fire.Function)
    assert dJ.dat.data.shape == Wave_obj_guess.c.dat.data.shape

    # Deterministic perturbation direction, identical on both ensemble members.
    direction = build_direction(Wave_obj_guess)

    # Let taylor_test compute the directional derivative from the
    # EnsembleReducedFunctional itself so the ensemble reduction stays
    # consistent (do not pass dJdm here).
    rate = Wave_obj_guess.automated_adjoint.verify_gradient(
        Wave_obj_guess.c, direction=direction
    )
    print(f"Automated-adjoint Taylor convergence rate: {rate}", flush=True)
    assert rate > 1.9, (
        "Automated adjoint gradient verification failed: Taylor convergence "
        f"rate {rate} is below the expected second-order rate."
    )

    # Clean up the tape so the test leaves no global annotation state behind.
    Wave_obj_guess.automated_adjoint.clear_tape()
    assert Wave_obj_guess.automated_adjoint._tape is None


if __name__ == "__main__":
    test_gradient_auto_adjoint_parallel()
