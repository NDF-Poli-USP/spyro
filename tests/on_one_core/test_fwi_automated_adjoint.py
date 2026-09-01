"""Full-waveform inversion driven by the automated adjoint.

The implemented adjoint and the automated one reach the optimizer by
different routes, and these tests pin the automated one down:

* the forward solve is recorded on a pyadjoint tape *once*, and the reduced
  functional replays it for every new control, so the driver does not re-run
  the forward solve per iterate;
* the optimization is done by PETSc TAO rather than by
  :func:`scipy.optimize.minimize`, which changes what the bounds look like
  (one pair per control, not one per degree of freedom) and what ``run_fwi``
  hands back (the control itself);
* enabling it must not disturb the implemented-adjoint path, which stays the
  default.

The model is deliberately small -- a coarse mesh, two receivers, a constant
velocity contrast -- because what is under test is the plumbing between the
driver, the tape and TAO, not the quality of the reconstruction. It still has
to run long enough for the wave to reach the receivers, or the residual would
be zero and there would be nothing for the optimizer to do.
"""
import firedrake as fire
import firedrake.adjoint as fire_ad
import numpy as np
import pytest

import spyro
from spyro.utils.typing import AcousticMaterialParameter, AdjointType


def build_dictionary():
    """Return a minimal acoustic FWI configuration.

    Returns
    -------
    dict
        Model dictionary with a coarse mesh and two receivers.
    """
    return {
        "options": {
            "cell_type": "T",
            "variant": "lumped",
            "degree": 1,
            "dimension": 2,
        },
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
            "source_locations": [(-0.1, 0.5)],
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


def build_inversion(tmp_path, monkeypatch, real_velocity=3.0, guess_velocity=2.5):
    """Build an inversion with synthetic observed data.

    Every artefact ``FullWaveformInversion`` writes (control snapshots, the
    functional history, the result vector) lands in ``tmp_path`` instead of the
    repository.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory the inversion runs in.
    monkeypatch : _pytest.monkeypatch.MonkeyPatch
        Used to change the working directory.
    real_velocity : float, optional
        Velocity of the model generating the observed data.
    guess_velocity : float, optional
        Velocity the inversion starts from.

    Returns
    -------
    spyro.FullWaveformInversion
        Inversion with the observed data already generated and the guess model
        configured.
    """
    monkeypatch.chdir(tmp_path)
    fwi = spyro.FullWaveformInversion(dictionary=build_dictionary())

    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_velocity_model(constant=real_velocity)
    fwi.generate_real_shot_record(save_shot_record=False)

    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_velocity_model(constant=guess_velocity)
    return fwi


def build_automated_inversion(tmp_path, monkeypatch):
    """Build an inversion set to differentiate with the automated adjoint.

    ``run_fwi`` takes the choice as an argument and sets this attribute; the
    tests that drive ``get_functional`` and ``get_gradient`` on their own set
    it themselves.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory the inversion runs in.
    monkeypatch : _pytest.monkeypatch.MonkeyPatch
        Used to change the working directory.

    Returns
    -------
    spyro.FullWaveformInversion
        Inversion ready to solve. The solver itself is configured by the
        first forward solve, not here.
    """
    fwi = build_inversion(tmp_path, monkeypatch)
    fwi.adjoint_type = AdjointType.AUTOMATED_ADJOINT
    return fwi


@pytest.mark.newer_firedrake
def test_the_control_is_the_solvers_own_field(tmp_path, monkeypatch):
    """The tape records the velocity model the wave equation reads.

    The control has to be the solver's own ``Function``, not a copy of it, or
    the optimizer would be moving something the forward solve never reads.
    """
    fwi = build_automated_inversion(tmp_path, monkeypatch)

    fwi.get_functional()

    assert fwi.wave.adjoint_type == AdjointType.AUTOMATED_ADJOINT
    automated_adjoint = fwi.wave.automated_adjoint
    assert automated_adjoint.controls == [fwi.wave.c]
    assert automated_adjoint.control_parameter_names == [
        AcousticMaterialParameter.P_WAVE_VELOCITY,
    ]
    # The ensemble the reduced functional will sum over is the solver's.
    assert automated_adjoint.ensemble is fwi.wave.comm


@pytest.mark.newer_firedrake
def test_run_fwi_automated_adjoint(
    tmp_path, monkeypatch,
):
    """``run_fwi`` drives TAO from the recorded tape and returns the control.

    A fixed iteration budget is how FWI is normally run, and TAO reports that
    as a failure to converge; the driver has to hand back the last iterate
    rather than let the exception through.
    """
    vmin, vmax = 2.0, 3.5
    fwi = build_inversion(tmp_path, monkeypatch)

    result = fwi.run_fwi(
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        vmin=vmin, vmax=vmax, maxiter=3,
    )

    # TAO optimizes the control itself, so that is what comes back.
    assert isinstance(result, fire.Function)
    assert result.function_space() == fwi.wave.c.function_space()

    reduced_functional = fwi.wave.automated_adjoint.reduced_functional
    assert isinstance(reduced_functional, fire_ad.EnsembleReducedFunctional)

    # The bounds are respected, and the control actually moved.
    values = result.dat.data_ro
    assert values.min() >= vmin - 1e-10
    assert values.max() <= vmax + 1e-10
    assert not np.allclose(values, 2.5)

    # The monitor logged the iterates on top of the recorded starting point.
    assert len(fwi.functional_history) > 1
    assert fwi.functional_history[-1] < fwi.functional_history[0]
    assert fwi.functional == fwi.functional_history[-1]

    # The optimum is written back into the driver and the solver alike.
    assert np.allclose(
        fwi.control_parameter_result.dat.data_ro, values,
    )
    assert np.allclose(fwi.wave.c.dat.data_ro, values)
    assert (tmp_path / "result.npy").exists()
