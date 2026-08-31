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
from pyadjoint import AdjFloat

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


def build_inversion(
    tmp_path, monkeypatch, real_velocity=3.0, guess_velocity=2.5,
    **constructor_options,
):
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
    **constructor_options
        Passed to :class:`spyro.FullWaveformInversion`, so a test can choose
        the adjoint there rather than switching it on afterwards.

    Returns
    -------
    spyro.FullWaveformInversion
        Inversion with the observed data already generated and the guess model
        configured.
    """
    monkeypatch.chdir(tmp_path)
    fwi = spyro.FullWaveformInversion(
        dictionary=build_dictionary(), **constructor_options,
    )

    fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_real_velocity_model(constant=real_velocity)
    fwi.generate_real_shot_record(save_shot_record=False)

    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_velocity_model(constant=guess_velocity)
    return fwi


@pytest.mark.newer_firedrake
def test_enable_automated_adjoint_differentiates_the_wave_control(
    tmp_path, monkeypatch,
):
    """The driver switch reaches the solver and picks up the velocity model.

    The control the tape records has to be the solver's own ``Function``,
    the one the wave equation is written in terms of, not a copy of it, or
    the optimizer would be moving something the forward solve never reads.
    """
    fwi = build_inversion(tmp_path, monkeypatch)
    fwi.enable_automated_adjoint()

    assert fwi.wave.adjoint_type == AdjointType.AUTOMATED_ADJOINT
    automated_adjoint = fwi.wave.automated_adjoint
    assert automated_adjoint is not None
    assert automated_adjoint.controls == [fwi.wave.c]
    assert automated_adjoint.control_parameter_names == [
        AcousticMaterialParameter.P_WAVE_VELOCITY,
    ]
    # The ensemble the reduced functional will sum over is the solver's.
    assert automated_adjoint.ensemble is fwi.wave.comm


@pytest.mark.newer_firedrake
def test_enable_automated_adjoint_forwards_the_checkpointing_settings(
    tmp_path, monkeypatch,
):
    """Checkpointing asked for on the driver has to survive to the tape."""
    fwi = build_inversion(tmp_path, monkeypatch)
    fwi.enable_automated_adjoint(checkpointing=True, snapshots=3)

    automated_adjoint = fwi.wave.automated_adjoint
    assert automated_adjoint._checkpointing is True
    assert automated_adjoint._snapshots == 3


@pytest.mark.newer_firedrake
def test_the_adjoint_can_be_chosen_at_construction(tmp_path, monkeypatch):
    """Choosing the adjoint up front leaves the run looking like any other.

    The automated adjoint cannot be switched on at construction -- its
    controls are fields, and there is no mesh yet -- so the choice is recorded
    and the solver is configured from it by the first forward solve. Nothing
    in between has to know.
    """
    fwi = build_inversion(
        tmp_path, monkeypatch,
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        adjoint_options={"checkpointing": True, "snapshots": 3},
    )

    # Recorded, but nothing built: the solver has no mesh of its own yet.
    assert fwi.adjoint_type == AdjointType.AUTOMATED_ADJOINT
    assert fwi.wave.automated_adjoint is None

    fwi.get_functional()

    assert fwi.wave.adjoint_type == AdjointType.AUTOMATED_ADJOINT
    assert fwi.wave.automated_adjoint._tape is not None
    # The settings travelled from the constructor to the tape.
    assert fwi.wave.automated_adjoint._checkpointing is True
    assert fwi.wave.automated_adjoint._snapshots == 3


def test_adjoint_options_are_checked_at_construction(tmp_path, monkeypatch):
    """A misspelt setting fails where it was written, not at the first solve.

    They are applied a long way from here, and silently ignoring one would
    show up only as memory that never came down.
    """
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="not settings of the automated"):
        spyro.FullWaveformInversion(
            dictionary=build_dictionary(),
            adjoint_type=AdjointType.AUTOMATED_ADJOINT,
            adjoint_options={"snapshot": 3},
        )

    with pytest.raises(ValueError, match="needs adjoint_type"):
        spyro.FullWaveformInversion(
            dictionary=build_dictionary(),
            adjoint_options={"snapshots": 3},
        )


@pytest.mark.newer_firedrake
def test_the_functional_comes_off_the_tape(tmp_path, monkeypatch):
    """The forward solve records a tape and leaves the functional on it.

    The residual is accumulated step by step while taping, so it never exists
    as an array, and the functional is read off the tape rather than
    recomputed from one.
    """
    fwi = build_inversion(tmp_path, monkeypatch)
    fwi.enable_automated_adjoint()

    functional = fwi.get_functional()

    assert fwi.wave.automated_adjoint._tape is not None
    assert isinstance(fwi.wave.functional_value, AdjFloat)
    assert functional == pytest.approx(float(fwi.wave.functional_value))
    assert fwi.functional_history == [functional]


@pytest.mark.newer_firedrake
def test_calculate_misfit_is_refused_under_the_automated_adjoint(
    tmp_path, monkeypatch,
):
    """There is no residual array to hand back, so asking for one is an error.

    Returning ``None`` from a method named for what it computes would leave
    the caller to discover the difference on their own.
    """
    fwi = build_inversion(tmp_path, monkeypatch)
    fwi.enable_automated_adjoint()

    with pytest.raises(ValueError, match="never exists as an array"):
        fwi.calculate_misfit()


@pytest.mark.newer_firedrake
def test_get_gradient_differentiates_the_recorded_tape(tmp_path, monkeypatch):
    """``get_gradient`` reads the tape instead of the backward propagator.

    The residual and the forward wavefield the implemented adjoint is handed
    explicitly are both already on the tape, so neither is passed; what has to
    come back is a gradient in the control's own space.
    """
    fwi = build_inversion(tmp_path, monkeypatch)
    fwi.enable_automated_adjoint()

    fwi.get_gradient(save=False)

    assert isinstance(fwi.gradient, fire.Function)
    assert fwi.gradient.function_space() == fwi.wave.c.function_space()
    # A guess slower than the truth leaves a residual to descend.
    assert np.linalg.norm(fwi.gradient.dat.data_ro) > 0.0


def test_tao_bounds_are_one_pair_per_control(tmp_path, monkeypatch):
    """TAO takes bounds per control, L-BFGS-B one per degree of freedom.

    A scalar stays a scalar for TAO to broadcast, while a bound that varies
    over the mesh becomes a ``Function`` in the control's own space.
    """
    fwi = build_inversion(tmp_path, monkeypatch)
    control = fwi.control_parameters

    assert fwi._tao_bound(2.5, control) == 2.5

    varying = np.linspace(2.0, 3.0, control.dat.data_ro.size)
    bound = fwi._tao_bound(varying, control)
    assert isinstance(bound, fire.Function)
    assert bound.function_space() == control.function_space()
    assert np.allclose(bound.dat.data_ro, varying)


@pytest.mark.newer_firedrake
def test_run_fwi_optimizes_with_tao_under_the_automated_adjoint(
    tmp_path, monkeypatch,
):
    """``run_fwi`` drives TAO from the recorded tape and returns the control.

    A fixed iteration budget is how FWI is normally run, and TAO reports that
    as a failure to converge; the driver has to hand back the last iterate
    rather than let the exception through.
    """
    vmin, vmax = 2.0, 3.5
    fwi = build_inversion(tmp_path, monkeypatch)
    fwi.enable_automated_adjoint()

    result = fwi.run_fwi(vmin=vmin, vmax=vmax, maxiter=3)

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


def test_run_fwi_still_uses_scipy_without_the_automated_adjoint(
    tmp_path, monkeypatch,
):
    """The implemented adjoint stays the default, with the scipy result.

    Guards the dispatch in ``run_fwi``: a driver that was never told which
    adjoint to use has ``AdjointType.NONE`` on its solver, and that must land
    on the L-BFGS-B path rather than fall between the two branches.
    """
    fwi = build_inversion(tmp_path, monkeypatch)
    assert fwi.wave.adjoint_type == AdjointType.NONE

    result = fwi.run_fwi(vmin=2.0, vmax=3.5, maxiter=1)

    assert hasattr(result, "x")
    assert fwi.wave.adjoint_type == AdjointType.IMPLEMENTED_ADJOINT
    assert fwi.wave.automated_adjoint is None
    # The backward propagator ran and the driver logged its iterates.
    assert isinstance(fwi.gradient, fire.Function)
    assert fwi.functional_history
    assert isinstance(fwi.control_parameter_result, fire.Function)
