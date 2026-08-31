"""Isotropic-elastic FWI driven by the automated adjoint.

An acoustic medium is inverted for one control, its velocity model. An
isotropic elastic one is inverted for several at once -- density and the two
wave speeds, or density and the Lame parameters, whichever set the equation is
written in -- and that is what these tests pin down:

* the controls are the parameters the automated adjoint differentiates, in the
  same order, so an iterate is never written back into the wrong control;
* selecting a subset of them narrows the inversion to it, and which set they
  are drawn from is whichever one the equation is written in -- the two wave
  speeds or the two Lame parameters -- so density can be held fixed while the
  Lame parameters alone are fitted;
* bounds take one entry per control, since density and a wave speed are not
  bounded by the same numbers;
* the derivatives come back one per control, keyed by the parameter each
  belongs to;
* without the automated adjoint there is no elastic gradient at all, and the
  driver has to say so rather than fall through to L-BFGS-B.

The model is deliberately small; what is under test is the plumbing between the
driver, the tape and TAO, not the quality of the reconstruction.
"""
import firedrake as fire
import numpy as np
import pytest

import spyro
from spyro.utils.typing import AdjointType


Parameter = spyro.ElasticMaterialParameter

GUESS_MATERIAL = {
    "density": 2.0,
    "p_wave_velocity": 2.5,
    "s_wave_velocity": 1.2,
}
REAL_MATERIAL = {
    "density": 2.2,
    "p_wave_velocity": 3.0,
    "s_wave_velocity": 1.5,
}
# The same medium written in Lame parameters instead. Density is the same in
# both models, so an inversion that leaves it out of the controls is chasing a
# target it can reach.
LAME_GUESS_MATERIAL = {"density": 2.0, "lambda": 6.0, "mu": 3.0}
LAME_REAL_MATERIAL = {"density": 2.0, "lambda": 9.0, "mu": 4.5}

# The order the equation carries its independent parameters in, which is the
# order the controls come back in.
VELOCITY_PARAMETERS = (
    Parameter.DENSITY,
    Parameter.P_WAVE_VELOCITY,
    Parameter.S_WAVE_VELOCITY,
)
LAME_PARAMETERS = (
    Parameter.DENSITY,
    Parameter.LAMBDA,
    Parameter.MU,
)


def build_dictionary(material):
    """Return a minimal isotropic-elastic FWI configuration.

    Parameters
    ----------
    material : dict
        Material parameterization the solver is built from.

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
            "amplitude": np.array([0.0, 1.0]),
            "receiver_locations": [(-0.2, 0.25), (-0.2, 0.75)],
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.4,
            "dt": 0.002,
            "output_frequency": 100,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
        "synthetic_data": {
            "type": "object",
            **material,
            "real_velocity_file": None,
        },
    }


def build_inversion(
    tmp_path, monkeypatch, observed_data=True,
    guess_material=None, real_material=None,
):
    """Build an elastic inversion, with observed data if asked for.

    Which material set is given here is what the equation ends up written in,
    and so which parameters can be controls.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory the inversion runs in, so its artefacts stay out of the
        repository.
    monkeypatch : _pytest.monkeypatch.MonkeyPatch
        Used to change the working directory.
    observed_data : bool, optional
        Whether to propagate the true model and keep its shot record. Skipped
        by the tests that never run a forward solve.
    guess_material : dict, optional
        Material the inversion starts from. Defaults to the velocity set.
    real_material : dict, optional
        Material generating the observed data, keyed the same way.

    Returns
    -------
    spyro.FullWaveformInversion
        Inversion with the guess model configured.
    """
    guess_material = GUESS_MATERIAL if guess_material is None else guess_material
    real_material = REAL_MATERIAL if real_material is None else real_material

    monkeypatch.chdir(tmp_path)
    fwi = spyro.FullWaveformInversion(
        dictionary=build_dictionary(guess_material),
        wave_class=spyro.IsotropicWave,
    )
    if observed_data:
        fwi.set_real_mesh(input_mesh_parameters={"edge_length": 0.25})
        fwi.set_real_model({
            Parameter(name): value for name, value in real_material.items()
        })
        fwi.generate_real_shot_record(save_shot_record=False)

    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    return fwi


@pytest.mark.newer_firedrake
def test_controls_are_the_parameters_the_equation_is_written_in(
    tmp_path, monkeypatch,
):
    """Enabling the adjoint makes its controls the inversion's controls.

    The elastic guess model comes from the input dictionary rather than from a
    velocity setter, so this is also what gives the inversion its controls in
    the first place. They have to be the solver's own parameters: the
    optimizer moves what the forward solve reads.
    """
    fwi = build_inversion(tmp_path, monkeypatch, observed_data=False)
    fwi.enable_automated_adjoint()

    assert fwi.wave.adjoint_type == AdjointType.AUTOMATED_ADJOINT
    assert tuple(
        fwi.wave.automated_adjoint.control_parameter_names
    ) == VELOCITY_PARAMETERS

    controls = fwi.control_parameters
    assert isinstance(controls, list)
    assert [control.name() for control in controls] == [
        parameter.value for parameter in VELOCITY_PARAMETERS
    ]
    # The driver holds detached copies, but of the very controls being taped.
    for control, taped in zip(controls, fwi.wave.automated_adjoint.controls):
        assert control.function_space() == taped.function_space()
        assert np.allclose(control.dat.data_ro, taped.dat.data_ro)


@pytest.mark.newer_firedrake
def test_selecting_a_subset_narrows_the_inversion(tmp_path, monkeypatch):
    """Inverting for one parameter leaves the inversion with one control."""
    fwi = build_inversion(tmp_path, monkeypatch, observed_data=False)
    fwi.enable_automated_adjoint(
        control_parameters={Parameter.S_WAVE_VELOCITY},
    )

    assert fwi.wave.automated_adjoint.controls[0] is fwi.wave.c_s
    # One control is presented the way an acoustic one always has been.
    control = fwi.control_parameters
    assert isinstance(control, fire.Function)
    assert control.name() == Parameter.S_WAVE_VELOCITY.value


@pytest.mark.newer_firedrake
def test_controls_follow_the_set_the_equation_is_written_in(
    tmp_path, monkeypatch,
):
    """A model given in Lame parameters is inverted in Lame parameters.

    Which parameters carry their own values, and which are computed from
    those, is what the equation is written in. Only the ones carrying values
    can be controls.
    """
    fwi = build_inversion(
        tmp_path, monkeypatch, observed_data=False,
        guess_material=LAME_GUESS_MATERIAL,
    )
    fwi.enable_automated_adjoint()

    assert tuple(
        fwi.wave.automated_adjoint.control_parameter_names
    ) == LAME_PARAMETERS
    assert [control.name() for control in fwi.control_parameters] == [
        parameter.value for parameter in LAME_PARAMETERS
    ]
    # The wave speeds are computed from these, carrying no values of their
    # own, which is exactly why they cannot be controls here.
    assert not isinstance(fwi.wave.c, fire.Function)
    assert not isinstance(fwi.wave.c_s, fire.Function)


@pytest.mark.newer_firedrake
def test_a_parameter_computed_from_the_others_cannot_be_a_control(
    tmp_path, monkeypatch,
):
    """The two sets are swapped between, never mixed."""
    fwi = build_inversion(
        tmp_path, monkeypatch, observed_data=False,
        guess_material=LAME_GUESS_MATERIAL,
    )

    with pytest.raises(TypeError, match="computed from the other physical"):
        fwi.enable_automated_adjoint(
            control_parameters={Parameter.P_WAVE_VELOCITY},
        )


@pytest.mark.newer_firedrake
def test_controls_flatten_into_one_vector(tmp_path, monkeypatch):
    """Several controls concatenate into the vector each iterate is saved as.

    Nothing splits that vector apart again -- TAO works on the controls
    themselves -- so this is the whole of what the driver asks of it.
    """
    fwi = build_inversion(tmp_path, monkeypatch, observed_data=False)
    fwi.enable_automated_adjoint()
    controls = fwi.control_parameters

    flat = fwi._flatten_control(controls)

    assert flat.size == sum(control.dat.data_ro.size for control in controls)
    offset = 0
    for control in controls:
        values = control.dat.data_ro
        assert np.allclose(flat[offset:offset + values.size], values)
        offset += values.size


@pytest.mark.newer_firedrake
def test_bounds_take_one_entry_per_control(tmp_path, monkeypatch):
    """Each parameter is bounded on its own scale, or all of them alike."""
    fwi = build_inversion(tmp_path, monkeypatch, observed_data=False)
    fwi.enable_automated_adjoint()
    controls = fwi.control_parameters

    # One scalar bounds every control the same way.
    assert fwi._tao_bounds(1.5, controls) == [1.5, 1.5, 1.5]
    # One entry per control is how they are bounded apart.
    assert fwi._tao_bounds([1.0, 1.5, 0.5], controls) == [1.0, 1.5, 0.5]

    with pytest.raises(ValueError, match="controls 3 parameters"):
        fwi._tao_bounds([1.0, 1.5], controls)


@pytest.mark.newer_firedrake
def test_get_gradient_returns_one_derivative_per_control(
    tmp_path, monkeypatch,
):
    """The derivatives come back keyed by the parameter each belongs to."""
    fwi = build_inversion(tmp_path, monkeypatch)
    fwi.enable_automated_adjoint()

    fwi.get_gradient(save=False)

    assert tuple(fwi.gradient) == VELOCITY_PARAMETERS
    for derivative in fwi.gradient.values():
        assert isinstance(derivative, fire.Function)
    # A guess wrong in every parameter leaves something to descend in each.
    assert all(
        np.linalg.norm(derivative.dat.data_ro) > 0.0
        for derivative in fwi.gradient.values()
    )


@pytest.mark.newer_firedrake
def test_run_fwi_inverts_every_control(tmp_path, monkeypatch):
    """TAO moves all three parameters, each within its own bounds."""
    vmin = [1.0, 1.5, 0.5]
    vmax = [3.0, 4.0, 2.5]
    fwi = build_inversion(tmp_path, monkeypatch)
    fwi.enable_automated_adjoint()

    result = fwi.run_fwi(vmin=vmin, vmax=vmax, maxiter=3)

    assert isinstance(result, list)
    assert [control.name() for control in result] == [
        parameter.value for parameter in VELOCITY_PARAMETERS
    ]

    for control, low, high, start in zip(
        result, vmin, vmax, GUESS_MATERIAL.values(),
    ):
        values = control.dat.data_ro
        assert values.min() >= low - 1e-10
        assert values.max() <= high + 1e-10
        assert not np.allclose(values, start), (
            f"{control.name()} was left where it started."
        )

    assert len(fwi.functional_history) > 1
    assert fwi.functional_history[-1] < fwi.functional_history[0]

    # The optimum is written back into the solver's own material parameters.
    for control, parameter in zip(
        result, (fwi.wave.rho, fwi.wave.c, fwi.wave.c_s),
    ):
        assert np.allclose(control.dat.data_ro, parameter.dat.data_ro)


@pytest.mark.newer_firedrake
def test_run_fwi_inverts_the_lame_parameters_with_density_held_fixed(
    tmp_path, monkeypatch,
):
    """A subset of the set in use is an inversion of its own.

    Density is left out of the controls, so it has to come out of the run
    untouched while the two Lame parameters are fitted.
    """
    vmin = [1.0, 0.5]
    vmax = [20.0, 10.0]
    fwi = build_inversion(
        tmp_path, monkeypatch,
        guess_material=LAME_GUESS_MATERIAL,
        real_material=LAME_REAL_MATERIAL,
    )
    fwi.enable_automated_adjoint(
        control_parameters={Parameter.LAMBDA, Parameter.MU},
    )

    assert [control.name() for control in fwi.control_parameters] == [
        Parameter.LAMBDA.value,
        Parameter.MU.value,
    ]
    density_before = np.array(fwi.wave.rho.dat.data_ro)

    result = fwi.run_fwi(vmin=vmin, vmax=vmax, maxiter=3)

    assert len(result) == 2
    for control, low, high, start in zip(
        result, vmin, vmax,
        (LAME_GUESS_MATERIAL["lambda"], LAME_GUESS_MATERIAL["mu"]),
    ):
        values = control.dat.data_ro
        assert values.min() >= low - 1e-10
        assert values.max() <= high + 1e-10
        assert not np.allclose(values, start), (
            f"{control.name()} was left where it started."
        )

    assert fwi.functional_history[-1] < fwi.functional_history[0]
    # Not a control, so nothing in the run had any business moving it.
    assert np.allclose(fwi.wave.rho.dat.data_ro, density_before)


def test_run_fwi_without_the_automated_adjoint_is_refused(
    tmp_path, monkeypatch,
):
    """There is no hand-written elastic adjoint to fall back on.

    Falling through to L-BFGS-B would run a forward solve per iterate and only
    then fail, deep inside ``gradient_solve``.
    """
    fwi = build_inversion(tmp_path, monkeypatch, observed_data=False)

    with pytest.raises(NotImplementedError, match="automated adjoint"):
        fwi.run_fwi(maxiter=1)
