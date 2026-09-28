"""FWI with the automated adjoint under ensemble (shot) parallelism.

Meant to be run with two MPI ranks::

    mpiexec -n 2 pytest tests/parallel/test_fwi_auto_adjoint.py

With ``parallelism = "automatic"`` and two sources, spyro builds an ensemble of
two members, one shot each on a single spatial core. Every member records its
own shot on its own tape, and the ``EnsembleReducedFunctional`` sums the
per-shot functionals and gradients across the ensemble communicator.

What that leaves for the optimizer is the point of these tests. The controls
are *replicated* on every ensemble member, so TAO has to run on the **spatial**
communicator: on ``COMM_WORLD`` it would treat each member's copy of the
material parameters as separate degrees of freedom. Being handed the summed
functional and gradient, each member's TAO then takes the same step, and the
models must still agree across the ensemble when the run ends.

Both media are covered, because they reach the optimizer differently: an
acoustic inversion moves one control, its velocity model, and an isotropic
elastic one moves the three parameters its equation is written in, each on its
own scale and so with its own bounds.
"""
import firedrake as fire
import firedrake.adjoint as fire_ad
import numpy as np
import pytest

import spyro
from spyro.utils.typing import AdjointType


Parameter = spyro.ElasticMaterialParameter

ELASTIC_GUESS = {
    "density": 2.0,
    "p_wave_velocity": 2.5,
    "s_wave_velocity": 1.2,
}
ELASTIC_REAL = {
    "density": 2.2,
    "p_wave_velocity": 3.0,
    "s_wave_velocity": 1.5,
}


def build_dictionary():
    """Return a two-source acoustic FWI configuration.

    Returns
    -------
    dict
        Model dictionary sized so the wave reaches the receivers while
        staying cheap enough for a handful of optimizer iterations.
    """
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 1,
            "dimension": 2,
        },
        # Two sources over two ranks: one shot per ensemble member.
        "parallelism": {"type": "automatic"},
        "mesh": {
            "length_z": 1.0,
            "length_x": 1.0,
            "length_y": 0.0,
            "mesh_file": None,
            "mesh_type": "firedrake_mesh",
        },
        "acquisition": {
            "source_type": "ricker",
            "source_locations": [(-0.1, 0.3), (-0.1, 0.7)],
            "frequency": 4.0,
            "delay": 0.0,
            "delay_type": "time",
            "receiver_locations": [(-0.2, 0.25), (-0.2, 0.75)],
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.4,
            "dt": 0.002,
            "amplitude": 1.0,
            "output_frequency": 100,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
    }


def build_elastic_dictionary(material):
    """Return the same configuration for an isotropic elastic medium.

    Parameters
    ----------
    material : dict
        Material the solver is built from, which is also the set the equation
        ends up written in terms of.

    Returns
    -------
    dict
        Model dictionary.
    """
    dictionary = build_dictionary()
    # A vector source, and the materials the elastic equation reads.
    dictionary["acquisition"]["amplitude"] = np.array([0.0, 1.0])
    dictionary["time_axis"].pop("amplitude")
    dictionary["synthetic_data"] = {
        "type": "object",
        **material,
        "real_velocity_file": None,
    }
    return dictionary


def assert_ensemble_agrees(fwi, values):
    """Assert the ensemble members converged on the same model.

    Every member is handed the same summed functional and gradient, so they
    have to follow the same path. A TAO built on the wrong communicator is
    what this catches.

    Parameters
    ----------
    fwi : spyro.FullWaveformInversion
        The inversion that has just run.
    values : numpy.ndarray
        The inverted material values on this member.
    """
    ensemble_comm = fwi.wave.comm.ensemble_comm

    from_root = ensemble_comm.bcast(np.array(values), root=0)
    assert np.allclose(values, from_root), (
        "Ensemble members disagree on the inverted model."
    )

    history_from_root = ensemble_comm.bcast(fwi.functional_history, root=0)
    assert np.allclose(fwi.functional_history, history_from_root), (
        "Ensemble members disagree on the functional they minimized."
    )
    assert fwi.functional_history[-1] < fwi.functional_history[0]


@pytest.mark.newer_firedrake
@pytest.mark.parallel(2)
def test_fwi_auto_adjoint_parallel():
    """Run an acoustic FWI over two shots, inverting the velocity model."""
    vmin, vmax = 2.0, 3.5
    guess = 2.5
    fwi = spyro.FullWaveformInversion(dictionary=build_dictionary())
    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_model(3.0)
    fwi.generate_real_shot_record(save_shot_record=False)

    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_velocity_model(constant=guess)

    # One shot per ensemble member, on one spatial core each.
    comm = fwi.wave.comm
    assert comm.ensemble_comm.size == 2, "Expected 2 ensemble members (sources)."
    assert comm.comm.size == 1, "Expected 1 spatial core per shot."

    result = fwi.run_fwi(
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        vmin=vmin, vmax=vmax, maxiter=3,
    )

    assert fwi.wave.adjoint_type == AdjointType.AUTOMATED_ADJOINT
    assert isinstance(
        fwi.wave.automated_adjoint.reduced_functional,
        fire_ad.EnsembleReducedFunctional,
    ), "The per-shot functionals have to be summed over the ensemble."

    # One control, presented as the single field it is.
    assert isinstance(result, fire.Function)
    values = result.dat.data_ro
    assert values.min() >= vmin - 1e-10
    assert values.max() <= vmax + 1e-10
    assert not np.allclose(values, guess), "The optimizer did not move the model."

    assert_ensemble_agrees(fwi, values)


@pytest.mark.newer_firedrake
@pytest.mark.parallel(2)
def test_fwi_elastic_auto_adjoint_parallel():
    """Run an elastic FWI over two shots, inverting three parameters at once.

    Density and the two wave speeds are on different scales, so each takes its
    own bounds, and each comes back as a control of its own.
    """
    vmin = [1.0, 1.5, 0.5]
    vmax = [3.0, 4.0, 2.5]
    fwi = spyro.FullWaveformInversion(
        dictionary=build_elastic_dictionary(ELASTIC_GUESS),
        wave_class=spyro.IsotropicWave,
    )
    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_model({
        Parameter(name): value for name, value in ELASTIC_REAL.items()
    })
    fwi.generate_real_shot_record(save_shot_record=False)

    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    comm = fwi.wave.comm
    assert comm.ensemble_comm.size == 2, "Expected 2 ensemble members (sources)."
    assert comm.comm.size == 1, "Expected 1 spatial core per shot."

    result = fwi.run_fwi(
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        vmin=vmin, vmax=vmax, maxiter=3,
    )

    assert isinstance(
        fwi.wave.automated_adjoint.reduced_functional,
        fire_ad.EnsembleReducedFunctional,
    ), "The per-shot functionals have to be summed over the ensemble."

    # One control per parameter the equation is written in, in that order.
    assert [control.name() for control in result] == list(ELASTIC_GUESS)
    for control, low, high, start in zip(
        result, vmin, vmax, ELASTIC_GUESS.values(),
    ):
        values = control.dat.data_ro
        assert values.min() >= low - 1e-10
        assert values.max() <= high + 1e-10
        assert not np.allclose(values, start), (
            f"{control.name()} was left where it started."
        )

    assert_ensemble_agrees(
        fwi, np.concatenate([control.dat.data_ro for control in result]),
    )
