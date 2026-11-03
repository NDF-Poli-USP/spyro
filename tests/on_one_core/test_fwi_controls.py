from copy import deepcopy

import numpy as np
import firedrake as fire
import pytest
from scipy.optimize import OptimizeResult

import spyro
from spyro.utils.typing import (AcousticMaterialParameter, AdjointType,
                                ElasticMaterialParameter)


def build_acoustic_dictionary():
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
            "final_time": 0.02,
            "dt": 0.002,
            "amplitude": 1.0,
            "output_frequency": 10,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
    }


def build_elastic_dictionary(material_parameters=None):
    if material_parameters is None:
        material_parameters = {
            "density": 1.0,
            "p_wave_velocity": 2.5,
            "s_wave_velocity": 1.0,
        }
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
        "synthetic_data": {
            "type": "object",
            **material_parameters,
            "real_velocity_file": None,
        },
        "time_axis": {
            "initial_time": 0.0,
            "final_time": 0.02,
            "dt": 0.002,
            "output_frequency": 10,
            "gradient_sampling_frequency": 1,
        },
        "visualization": {
            "forward_output": False,
            "gradient_output": False,
            "adjoint_output": False,
            "debug_output": False,
        },
    }


VELOCITY = AcousticMaterialParameter.P_WAVE_VELOCITY


def build_elastic_wave():
    wave = spyro.IsotropicWave(dictionary=build_elastic_dictionary())
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.25})
    wave.initialize_physical_parameters()
    return wave


def test_full_waveform_inversion_uses_composition():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())

    assert not isinstance(fwi, spyro.Wave)
    assert isinstance(fwi.wave, spyro.AcousticWave)
    assert fwi.wave_type is spyro.WaveType.ISOTROPIC_ACOUSTIC


def test_full_waveform_inversion_rejects_non_wave_instance():
    with pytest.raises(TypeError, match="wave must be an instance of Wave"):
        spyro.FullWaveformInversion(wave=object())


def test_full_waveform_inversion_rejects_non_wave_class():
    with pytest.raises(TypeError, match="wave_class must be a Wave subclass"):
        spyro.FullWaveformInversion(
            dictionary=build_acoustic_dictionary(),
            wave_class=object,
        )


# Physical parameters belong to the wave equation.


def test_acoustic_wave_declares_its_physical_parameters():
    wave = spyro.AcousticWave(dictionary=build_acoustic_dictionary())
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.25})
    wave.set_initial_velocity_model(constant=2.0)

    parameters = wave.initialize_physical_parameters()

    assert set(parameters) == {VELOCITY}
    assert parameters[VELOCITY] is wave.c


def test_isotropic_wave_declares_its_physical_parameters():
    wave = build_elastic_wave()

    assert set(wave.physical_parameters) == set(ElasticMaterialParameter)


def test_physical_parameters_compare_as_a_set_of_names():
    wave = build_elastic_wave()

    assert {
        ElasticMaterialParameter.DENSITY,
        ElasticMaterialParameter.MU,
    } <= wave.physical_parameters
    assert wave.physical_parameters.issuperset(
        {ElasticMaterialParameter.S_WAVE_VELOCITY},
    )
    # An acoustic medium names its wave speed with its own enum, so the
    # elastic solver does not model that parameter.
    assert VELOCITY not in wave.physical_parameters


def test_physical_parameters_raise_before_initialization():
    wave = spyro.AcousticWave(dictionary=build_acoustic_dictionary())

    with pytest.raises(ValueError, match="Physical parameters have not been set"):
        wave.physical_parameters


def test_updating_a_physical_parameter_writes_into_the_field():
    wave = build_elastic_wave()
    density = wave.physical_parameters[ElasticMaterialParameter.DENSITY]

    wave.physical_parameters.update(
        ElasticMaterialParameter.DENSITY, fire.Constant(2.0),
    )

    # Writing into the field is what keeps the assembled forms, and the
    # parameters computed from this one, valid without being rebuilt.
    assert wave.physical_parameters[ElasticMaterialParameter.DENSITY] is density
    assert wave.rho is density
    assert np.allclose(density.dat.data_ro, 2.0)


def test_unknown_physical_parameter_is_rejected():
    wave = build_elastic_wave()

    with pytest.raises(KeyError, match="p_wave_velocity"):
        wave.physical_parameters.update(VELOCITY, 1.0)


def test_physical_parameters_are_keyed_by_material_parameter_enums():
    wave = build_elastic_wave()

    assert (
        wave.physical_parameters[spyro.ElasticMaterialParameter.DENSITY]
        is wave.rho
    )


def test_elastic_automated_adjoint_defaults_to_current_parameterization():
    """Unasked, the controls are the set in use, and survive a rebuild."""
    wave = build_elastic_wave()

    wave.enable_automated_adjoint()

    assert wave.automated_adjoint.control_parameter_names == [
        ElasticMaterialParameter.DENSITY,
        ElasticMaterialParameter.P_WAVE_VELOCITY,
        ElasticMaterialParameter.S_WAVE_VELOCITY,
    ]
    assert all(
        control is field
        for control, field in zip(
            wave.automated_adjoint.controls,
            (wave.rho, wave.c, wave.c_s),
        )
    )

    controls = list(wave.automated_adjoint.controls)
    wave.initialize_physical_parameters()
    assert all(
        control is field
        for control, field in zip(controls, (wave.rho, wave.c, wave.c_s))
    )


def test_elastic_automated_adjoint_accepts_one_control():
    """A subset of the set in use is a valid selection."""
    wave = build_elastic_wave()

    wave.enable_automated_adjoint(
        control_parameters={ElasticMaterialParameter.S_WAVE_VELOCITY},
    )

    assert wave.automated_adjoint.control_parameter_names == [
        ElasticMaterialParameter.S_WAVE_VELOCITY,
    ]
    assert wave.automated_adjoint.controls[0] is wave.c_s


def test_elastic_controls_follow_the_equation_parameterization():
    """Changing the set is the equation's decision, made before selecting."""
    wave = build_elastic_wave()
    assert set(wave.physical_parameters.select()) == {
        ElasticMaterialParameter.DENSITY,
        ElasticMaterialParameter.P_WAVE_VELOCITY,
        ElasticMaterialParameter.S_WAVE_VELOCITY,
    }

    wave.set_physical_parameterization(
        spyro.ElasticMaterialParameterization.LAME,
    )
    wave.enable_automated_adjoint(
        control_parameters={ElasticMaterialParameter.LAMBDA},
    )
    control = wave.automated_adjoint.controls[0]

    assert isinstance(wave.lmbda, fire.Function)
    assert isinstance(wave.mu, fire.Function)
    assert control is wave.lmbda

    # The change of variables is on the solver, not an edit of the model the
    # user handed in.
    assert set(wave.input_dictionary["synthetic_data"]) >= {
        "density", "p_wave_velocity", "s_wave_velocity",
    }
    assert "lambda" not in wave.input_dictionary["synthetic_data"]
    assert "mu" not in wave.input_dictionary["synthetic_data"]

    # The forward solve initializes the material parameters again. It must
    # retain the exact field registered as a control.
    wave.initialize_physical_parameters()
    assert wave.lmbda is control


def test_elastic_controls_reject_the_other_set():
    """Selecting a parameter the equation is not written in is an error."""
    wave = build_elastic_wave()

    with pytest.raises(TypeError, match="computed from the other physical"):
        wave.enable_automated_adjoint(
            control_parameters={ElasticMaterialParameter.LAMBDA},
        )


# Control parameters belong to the inversion.


def test_control_parameters_are_a_subset_of_the_physical_ones():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_control(fire.Constant(2.0))

    # Which parameters are inverted for is recorded by the keys of the control
    # set, and nowhere else. An acoustic medium is described by its velocity
    # model alone, so that is the whole selection.
    assert set(fwi._control_parameters) == {VELOCITY}
    assert set(fwi._control_parameters) <= fwi.wave.physical_parameters


def test_acoustic_constant_control_is_converted_to_function():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    fwi.set_guess_control(fire.Constant(2.0))

    control = fwi.control_parameters
    assert isinstance(control, fire.Function)
    assert np.allclose(control.dat.data_ro, 2.0)


def test_control_parameters_update_the_wave_physical_parameter():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_control(fire.Constant(2.0))
    velocity = fwi.wave.physical_parameters[VELOCITY]

    fwi.set_guess_control(fire.Constant(3.0))

    assert fwi.wave.physical_parameters[VELOCITY] is velocity
    assert np.allclose(velocity.dat.data_ro, 3.0)


def test_control_parameters_are_independent_of_the_wave_field():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_control(fire.Constant(2.0))

    fwi.wave.physical_parameters[VELOCITY].assign(9.0)

    assert np.allclose(fwi.control_parameters.dat.data_ro, 2.0)


def test_control_parameters_accept_a_mapping_keyed_by_parameter_name():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    fwi.set_guess_control({VELOCITY: fire.Constant(2.5)})

    assert np.allclose(
        fwi.wave.physical_parameters[VELOCITY].dat.data_ro,
        2.5,
    )


def test_control_parameters_reject_parameters_the_wave_does_not_model():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    with pytest.raises(KeyError, match="not a physical parameter"):
        fwi.set_guess_control(
            {ElasticMaterialParameter.DENSITY: fire.Constant(1.0)},
        )


def test_velocity_model_setters_capture_the_control():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    fwi.set_guess_velocity_model(constant=2.0)

    assert isinstance(fwi.control_parameters, fire.Function)
    assert np.allclose(fwi.control_parameters.dat.data_ro, 2.0)


@pytest.mark.parametrize("real_first", [True, False])
def test_velocity_model_setters_capture_each_model_separately(real_first):
    # Both models are configured on the same solver, one after the other, which
    # is how every synthetic inversion is set up. Whichever runs second must
    # not read back the values the first one left registered.
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    if real_first:
        fwi.set_real_velocity_model(constant=3.0)
        fwi.set_guess_velocity_model(constant=2.0)
    else:
        fwi.set_guess_velocity_model(constant=2.0)
        fwi.set_real_velocity_model(constant=3.0)

    assert np.allclose(fwi.control_parameters.dat.data_ro, 2.0)
    (real,) = fwi._real_model_parameters.values()
    assert np.allclose(real.dat.data_ro, 3.0)


def test_guess_control_rejects_an_empty_value():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_control(fire.Constant(2.0))

    with pytest.raises(ValueError, match="guess control value is required"):
        fwi.set_guess_control(None)

    # The configured control survives the rejected call.
    assert np.allclose(fwi.control_parameters.dat.data_ro, 2.0)


def test_misfit_without_a_configured_control_raises():
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})

    with pytest.raises(ValueError, match="No guess control parameter"):
        fwi.calculate_misfit()


def test_run_fwi_rejects_unknown_options():
    """A misspelled option fails before the driver is touched.

    ``max_iter`` for ``maxiter`` used to be dropped, and the inversion ran
    the default 20 iterations with no word about it.
    """
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_velocity_model(constant=2.0)

    with pytest.raises(TypeError, match="max_iter"):
        fwi.run_fwi(adjoint_type=AdjointType.AUTOMATED_ADJOINT, max_iter=5)

    # Rejected before any option was read, the valid one included.
    assert fwi.adjoint_type is AdjointType.NONE


@pytest.mark.parametrize(
    "options, expected",
    [
        (
            {"maxiter": 2, "eps": 1e-12, "ftol": 1e-9},
            {"maxiter": 2, "eps": 1e-12, "ftol": 1e-9},
        ),
        (
            {"scipy_options": {"maxiter": 2, "gtol": 1e-6}},
            {"maxiter": 2, "gtol": 1e-6},
        ),
    ],
)
def test_run_fwi_passes_scipy_options_to_scipy(
    options, expected, tmp_path, monkeypatch,
):
    """The options the implemented adjoint path accepts reach L-BFGS-B.

    Parameters
    ----------
    options : dict
        Keyword arguments given to ``run_fwi``.
    expected : dict
        Entries the ``options`` handed to :func:`scipy.optimize.minimize`
        must contain.
    tmp_path : pathlib.Path
        Directory the run writes its final model into.
    monkeypatch : pytest.MonkeyPatch
        Replaces the scipy optimizer, so no wave is propagated.
    """
    monkeypatch.chdir(tmp_path)
    received = {}

    def fake_minimize(fun, x0, **kwargs):
        """Record the call and stop at the starting point.

        Parameters
        ----------
        fun : callable
            Objective, never evaluated.
        x0 : numpy.ndarray
            Starting point, returned as the solution.
        **kwargs : dict
            Remaining arguments of :func:`scipy.optimize.minimize`.

        Returns
        -------
        scipy.optimize.OptimizeResult
            A result whose solution is ``x0``.
        """
        received.update(kwargs)
        return OptimizeResult(x=x0)

    monkeypatch.setattr(
        "spyro.solvers.inversion.scipy_minimize", fake_minimize,
    )
    fwi = spyro.FullWaveformInversion(dictionary=build_acoustic_dictionary())
    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.25})
    fwi.set_guess_velocity_model(constant=2.0)

    fwi.run_fwi(vmin=1.5, vmax=3.0, **options)

    assert received["options"].items() >= expected.items()


def build_elastic_gradient_dictionary():
    dictionary = build_elastic_dictionary()
    dictionary["time_axis"]["final_time"] = 0.5
    dictionary["time_axis"]["dt"] = 0.002
    dictionary["time_axis"]["output_frequency"] = 100
    return dictionary


def solve_elastic_gradient_model(
    model, edge_length=0.5, implemented_adjoint=False,
    real_shot_record=None, use_vertex_only_mesh=False,
    control_parameters=None,
):
    wave = spyro.IsotropicWave(dictionary=deepcopy(model))
    wave.set_mesh(input_mesh_parameters={"edge_length": edge_length})
    if real_shot_record is not None:
        wave.real_shot_record = real_shot_record
    if implemented_adjoint:
        wave.enable_implemented_adjoint(
            adjoint_type=AdjointType.UFL_DERIVED_ADJOINT,
            control_parameters=control_parameters,
        )
    elif use_vertex_only_mesh:
        wave.use_vertex_only_mesh = True
    wave.forward_solve()
    return wave


def normalized_scalar_direction(function_space, seed):
    values = np.random.default_rng(seed).random(function_space.dim()) - 0.5
    direction = fire.Function(function_space, val=values)
    direction.assign(direction / fire.norm(direction))
    return direction


def test_elastic_ufl_derived_adjoint_uses_forward_residual_form():
    """The UFL-derived adjoint returns one gradient per selected control."""
    model = build_elastic_gradient_dictionary()
    exact_model = deepcopy(model)
    exact_model["synthetic_data"]["density"] = 1.2

    exact = solve_elastic_gradient_model(
        exact_model, use_vertex_only_mesh=True,
    )
    guess = solve_elastic_gradient_model(
        model,
        implemented_adjoint=True,
        real_shot_record=exact.forward_solution_receivers,
    )

    functional = guess.functional_value
    gradient = guess.gradient_solve(
        adjoint_type=AdjointType.UFL_DERIVED_ADJOINT,
    )

    assert guess.forward_residual_form is not None
    assert guess.forward_residual_states is not None
    assert guess.get_adjoint_source().function_space() == (
        guess.source_function.function_space()
    )
    assert np.isfinite(functional)
    assert set(gradient) == {
        ElasticMaterialParameter.DENSITY,
        ElasticMaterialParameter.P_WAVE_VELOCITY,
        ElasticMaterialParameter.S_WAVE_VELOCITY,
    }
    assert all(isinstance(value, fire.Function) for value in gradient.values())
    assert all(np.isfinite(fire.norm(value)) for value in gradient.values())


def test_elastic_ufl_derived_adjoint_accepts_velocity_controls():
    """With v_p and v_s selected, density is not a control."""
    model = build_elastic_gradient_dictionary()
    exact_model = deepcopy(model)
    exact_model["synthetic_data"]["s_wave_velocity"] = 1.2

    exact = solve_elastic_gradient_model(
        exact_model, use_vertex_only_mesh=True,
    )
    controls = {
        ElasticMaterialParameter.P_WAVE_VELOCITY,
        ElasticMaterialParameter.S_WAVE_VELOCITY,
    }
    guess = solve_elastic_gradient_model(
        model,
        implemented_adjoint=True,
        real_shot_record=exact.forward_solution_receivers,
        control_parameters=controls,
    )
    gradient = guess.gradient_solve(
        adjoint_type=AdjointType.UFL_DERIVED_ADJOINT,
    )

    assert set(gradient) == controls


@pytest.mark.slow
@pytest.mark.parametrize(
    "parameter",
    [
        ElasticMaterialParameter.DENSITY,
        ElasticMaterialParameter.P_WAVE_VELOCITY,
        ElasticMaterialParameter.S_WAVE_VELOCITY,
    ],
)
def test_elastic_ufl_derived_adjoint_taylor_remainder(parameter):
    """The Taylor remainder of each control converges at second order."""
    model = build_elastic_gradient_dictionary()
    exact_model = deepcopy(model)
    exact_model["synthetic_data"][parameter.value] = (
        1.2 * exact_model["synthetic_data"][parameter.value]
    )

    exact = solve_elastic_gradient_model(
        exact_model, use_vertex_only_mesh=True,
    )
    guess = solve_elastic_gradient_model(
        model,
        implemented_adjoint=True,
        real_shot_record=exact.forward_solution_receivers,
    )
    field = guess.physical_parameters[parameter]
    base = fire.Function(field.function_space()).assign(field)

    base_functional = guess.functional_value
    gradient = guess.gradient_solve(
        adjoint_type=AdjointType.UFL_DERIVED_ADJOINT,
    )

    direction = normalized_scalar_direction(base.function_space(), 5)
    directional_derivative = fire.assemble(
        gradient[parameter] * direction * fire.dx(**guess.quadrature_rule)
    )

    steps = np.array([1e-2, 5e-3, 2.5e-3, 1.25e-3])
    remainders = []
    directional_errors = []
    for step in steps:
        guess.physical_parameters.update(parameter, base + step * direction)
        guess.forward_solve()
        perturbed_functional = guess.functional_value
        finite_difference = (perturbed_functional - base_functional) / step
        remainders.append(abs(
            perturbed_functional
            - base_functional
            - step * directional_derivative
        ))
        directional_errors.append(abs(
            (finite_difference - directional_derivative)
            / directional_derivative
        ))

    remainders = np.array(remainders)
    directional_errors = np.array(directional_errors)
    convergence_rates = np.log(remainders[:-1] / remainders[1:]) / np.log(
        steps[:-1] / steps[1:]
    )

    assert np.all(np.isfinite(remainders))
    assert np.all(convergence_rates > 1.8), convergence_rates
    assert np.all(directional_errors < 0.1)
