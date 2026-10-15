"""The implemented adjoint behind pyadjoint's reduced functional interface.

``FullWaveformInversion.reduced_functional`` is the single interface both
adjoints are differentiated through. What is under test is that the
implemented adjoint, wrapped in it, gives the same values and gradients the
driver computes directly, that its derivative and its Riesz map agree, and
that an optimizer written against the interface -- TAO here -- runs with it.
"""
import firedrake as fire
import firedrake.adjoint as fire_ad
import numpy as np
import pytest

import spyro
from spyro.reduced_functionals import ImplementedAdjointReducedFunctional
from spyro.utils.typing import AdjointType


GUESS = 2.5
REAL = 3.0


def build_dictionary() -> dict:
    """Return a one-source acoustic FWI configuration.

    Returns
    -------
    dict
        Model dictionary sized so the wave reaches the receivers -- otherwise
        the residual and the gradient would be zero -- while staying cheap
        enough for a handful of evaluations.
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


@pytest.fixture
def fwi(tmp_path, monkeypatch) -> spyro.FullWaveformInversion:
    """Return an acoustic inversion with observed data and a guess model.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory the driver writes its models and logs into.
    monkeypatch : pytest.MonkeyPatch
        Moves the working directory there.

    Returns
    -------
    spyro.FullWaveformInversion
        Driver holding synthetic data from a constant true model and a
        constant guess that differs from it.
    """
    monkeypatch.chdir(tmp_path)
    driver = spyro.FullWaveformInversion(dictionary=build_dictionary())
    driver.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
    driver.set_real_model(REAL)
    driver.generate_real_shot_record(save_shot_record=False)
    driver.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    driver.set_guess_velocity_model(constant=GUESS)
    return driver


def constant_model(fwi: spyro.FullWaveformInversion, value: float) -> fire.Function:
    """Return a constant velocity model in the control space.

    Parameters
    ----------
    fwi : spyro.FullWaveformInversion
        Driver whose control space the model lives in.
    value : float
        The velocity everywhere.

    Returns
    -------
    firedrake.Function
        The model.
    """
    space = fwi.control_parameters.function_space()
    return fire.Function(space).assign(value)


def test_driver_hands_out_the_implemented_adjoint(fwi):
    """The default adjoint is the implemented one, behind the interface.

    Parameters
    ----------
    fwi : spyro.FullWaveformInversion
        Acoustic inversion with observed data and a guess model.
    """
    reduced_functional = fwi.reduced_functional()

    assert isinstance(reduced_functional, ImplementedAdjointReducedFunctional)
    (control,) = reduced_functional.controls
    assert control.function_space() == fwi.wave.c.function_space()


def test_value_matches_the_driver(fwi):
    """Evaluating moves the control and gives the driver's own functional.

    Parameters
    ----------
    fwi : spyro.FullWaveformInversion
        Acoustic inversion with observed data and a guess model.
    """
    reduced_functional = fwi.reduced_functional()
    model = constant_model(fwi, 2.8)

    value = reduced_functional(model)

    assert np.allclose(reduced_functional.controls[0].tape_value().dat.data_ro, 2.8)
    assert np.allclose(fwi.control_parameters.dat.data_ro, 2.8)
    fwi.set_guess_control(model)
    assert float(value) == pytest.approx(fwi.get_functional())
    assert float(value) > 0.0


def test_gradient_matches_the_driver(fwi):
    """The gradient is the one ``get_gradient`` computes at the same model.

    Parameters
    ----------
    fwi : spyro.FullWaveformInversion
        Acoustic inversion with observed data and a guess model.
    """
    reduced_functional = fwi.reduced_functional()
    model = constant_model(fwi, 2.8)
    reduced_functional(model)

    gradient = reduced_functional.derivative(apply_riesz=True)

    fwi.set_guess_control(model)
    expected = fwi.get_gradient(save=False)
    assert isinstance(gradient, fire.Function)
    assert np.linalg.norm(expected.dat.data_ro) > 0.0
    assert np.allclose(gradient.dat.data_ro, expected.dat.data_ro)


def test_derivative_and_riesz_map_agree(fwi):
    """The control's Riesz map takes the derivative back to the gradient.

    That is how pyadjoint turns a derivative into a gradient, so an optimizer
    that asks for one or the other has to get the same thing.

    Parameters
    ----------
    fwi : spyro.FullWaveformInversion
        Acoustic inversion with observed data and a guess model.
    """
    reduced_functional = fwi.reduced_functional()
    reduced_functional(constant_model(fwi, 2.8))
    (control,) = reduced_functional.controls

    derivative = reduced_functional.derivative()
    gradient = reduced_functional.derivative(apply_riesz=True)
    doubled = reduced_functional.derivative(adj_input=2.0, apply_riesz=True)

    assert isinstance(derivative, fire.Cofunction)
    converted = control.control._ad_convert_riesz(
        derivative, riesz_map=control.riesz_map,
    )
    scale = np.abs(gradient.dat.data_ro).max()
    assert np.allclose(converted.dat.data_ro, gradient.dat.data_ro, atol=1e-8 * scale)
    assert np.allclose(doubled.dat.data_ro, 2.0 * gradient.dat.data_ro)


def test_derivative_before_any_evaluation(fwi):
    """Without an evaluation, the derivative is taken at the guess model.

    Parameters
    ----------
    fwi : spyro.FullWaveformInversion
        Acoustic inversion with observed data and a guess model.
    """
    reduced_functional = fwi.reduced_functional()

    gradient = reduced_functional.derivative(apply_riesz=True)

    expected = fwi.get_gradient(save=False)
    assert np.allclose(gradient.dat.data_ro, expected.dat.data_ro)


@pytest.mark.newer_firedrake
@pytest.mark.filterwarnings("ignore:TAOSolver failed to converge")
def test_tao_runs_on_the_implemented_adjoint(fwi):
    """TAO, so far driven only by the automated adjoint, runs with this one.

    Parameters
    ----------
    fwi : spyro.FullWaveformInversion
        Acoustic inversion with observed data and a guess model.
    """
    from spyro.tools.optimization import minimize_with_tao

    vmin, vmax = 2.0, 3.5
    reduced_functional = fwi.reduced_functional()
    start = float(reduced_functional(constant_model(fwi, GUESS)))

    (result,) = minimize_with_tao(
        reduced_functional,
        bounds=[(vmin, vmax)],
        comm=fwi.wave.comm.comm,
        options={"tao_max_it": 3},
    )

    values = result.dat.data_ro
    assert values.min() >= vmin - 1e-10
    assert values.max() <= vmax + 1e-10
    assert not np.allclose(values, GUESS)
    assert float(reduced_functional(result)) < start


def test_automated_adjoint_is_refused(fwi):
    """The implemented adjoint is not built for an automated inversion.

    Parameters
    ----------
    fwi : spyro.FullWaveformInversion
        Acoustic inversion with observed data and a guess model.
    """
    fwi.adjoint_type = AdjointType.AUTOMATED_ADJOINT

    with pytest.raises(ValueError, match="automated adjoint"):
        ImplementedAdjointReducedFunctional(fwi)


def test_missing_control_is_refused(tmp_path, monkeypatch):
    """A driver with no guess model has nothing to differentiate.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory the driver may write into.
    monkeypatch : pytest.MonkeyPatch
        Moves the working directory there.
    """
    monkeypatch.chdir(tmp_path)
    driver = spyro.FullWaveformInversion(dictionary=build_dictionary())
    driver.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    with pytest.raises(ValueError, match="No guess control"):
        driver.reduced_functional()


@pytest.mark.newer_firedrake
def test_driver_hands_out_the_recorded_functional(fwi):
    """Under the automated adjoint, the functional is the recorded one.

    Parameters
    ----------
    fwi : spyro.FullWaveformInversion
        Acoustic inversion with observed data and a guess model.
    """
    fwi.adjoint_type = AdjointType.AUTOMATED_ADJOINT

    reduced_functional = fwi.reduced_functional()

    assert isinstance(reduced_functional, fire_ad.EnsembleReducedFunctional)
    # Asking again reuses the recording instead of making a new one.
    assert fwi.reduced_functional() is reduced_functional
