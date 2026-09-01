"""FWI with the automated adjoint under ensemble (shot) parallelism.

Meant to be run with two MPI ranks::

    mpiexec -n 2 pytest tests/parallel/test_fwi_auto_adjoint.py

With ``parallelism = "automatic"`` and two sources, spyro builds an ensemble of
two members, one shot each on a single spatial core. Every member records its
own shot on its own tape, and the ``EnsembleReducedFunctional`` sums the
per-shot functionals and gradients across the ensemble communicator.

What that leaves for the optimizer is the point of this test. The control is
*replicated* on every ensemble member, so TAO has to run on the **spatial**
communicator: on ``COMM_WORLD`` it would treat each member's copy of the
velocity model as separate degrees of freedom. Being handed the summed
functional and gradient, each member's TAO then takes the same step, and the
models must still agree across the ensemble when the run ends.
"""
import firedrake as fire
import firedrake.adjoint as fire_ad
import numpy as np
import pytest

import spyro
from spyro.utils.typing import AdjointType


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


@pytest.mark.newer_firedrake
@pytest.mark.parallel(2)
def test_fwi_auto_adjoint_parallel():
    """Run FWI with the automated adjoint over two shots."""
    vmin, vmax = 2.0, 3.5
    fwi = spyro.FullWaveformInversion(dictionary=build_dictionary())
    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_velocity_model(constant=3.0)
    fwi.generate_real_shot_record(save_shot_record=False)

    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_velocity_model(constant=2.5)

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

    assert isinstance(result, fire.Function)
    values = result.dat.data_ro
    assert values.min() >= vmin - 1e-10
    assert values.max() <= vmax + 1e-10
    assert not np.allclose(values, 2.5), "The optimizer did not move the model."

    # Both members were handed the same summed functional and gradient, so
    # they must have followed the same path to the same model. A TAO built on
    # the wrong communicator is what this would catch.
    from_root = comm.ensemble_comm.bcast(np.array(values), root=0)
    assert np.allclose(values, from_root), (
        "Ensemble members disagree on the inverted model."
    )

    functional_from_root = comm.ensemble_comm.bcast(fwi.functional_history, root=0)
    assert np.allclose(fwi.functional_history, functional_from_root)
    assert fwi.functional_history[-1] < fwi.functional_history[0]


if __name__ == "__main__":
    test_fwi_auto_adjoint_parallel()
