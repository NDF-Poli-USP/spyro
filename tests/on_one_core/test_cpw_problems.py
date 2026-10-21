"""Tests for the cells-per-wavelength problems and calculator."""

from types import SimpleNamespace

import numpy as np
import pytest

import spyro
from spyro.io.time_io import interpolate_time_series
from spyro.tools.cells_per_wavelength_calculator import (
    ACOUSTIC,
    CPW_PROBLEMS,
    AcousticHomogeneousProblem,
    CpwParameters,
    CpwProblem,
    WavePhysics,
)
from spyro.tools.cells_per_wavelength_calculator import calculator as calculator_module
from spyro.utils.typing import (
    CpwSearchDirection,
    TimeStepCalculationType,
    VelocityProfileType,
    WaveType,
)

#: Calculator parameters of a homogeneous acoustic search, with no file access.
CALCULATOR_PARAMETERS = {
    "source_frequency": 5.0,
    "mesh_frequency_factor": 2.0,
    "minimum_velocity_in_the_domain": 1.5,
    "FEM_method_to_evaluate": "mass_lumped_triangle",
    "dimension": 2,
    "testing": True,
    "desired_degree": 2,
    "C_initial": 3.0,
    "C_accuracy": 0.1,
    "accepted_error_threshold": 0.05,
}


PROBLEM_CLASSES = [
    AcousticHomogeneousProblem,
]


def make_problem(problem_class: type[CpwProblem], **overrides) -> CpwProblem:
    """Build a problem with typical settings and no file access.

    Parameters
    ----------
    problem_class : type[CpwProblem]
        Problem to build.
    **overrides
        Parameters replacing the defaults.

    Returns
    -------
    CpwProblem
        The problem.
    """
    parameters = {**CALCULATOR_PARAMETERS, **overrides}
    return problem_class(CpwParameters.from_dict(parameters))


def test_interpolate_time_series_keeps_trailing_axes() -> None:
    """Elastic receiver data keeps its receiver and direction axes."""
    final_time = 1.0
    source_time = np.linspace(0.0, final_time, 11)
    values = np.stack(
        [
            np.column_stack([source_time, 2.0 * source_time]),
            np.column_stack([-source_time, 3.0 * source_time]),
        ],
        axis=1,
    )

    interpolated = interpolate_time_series(values, 0.05, final_time=final_time)

    target_time = np.linspace(0.0, final_time, 21)
    assert interpolated.shape == (21, 2, 2)
    assert np.allclose(interpolated[:, 0, 1], 2.0 * target_time)
    assert np.allclose(interpolated[:, 1, 0], -target_time)


@pytest.mark.parametrize(
    "wave_type, physics, profile",
    [
        (WaveType.ISOTROPIC_ACOUSTIC, ACOUSTIC, VelocityProfileType.HOMOGENEOUS),
    ],
)
def test_cpw_problems_registry(
    wave_type: WaveType, physics: WavePhysics, profile: VelocityProfileType,
) -> None:
    """Every problem is keyed by its own wave physics and velocity profile.

    Parameters
    ----------
    wave_type : WaveType
        Wave type key of the problem.
    physics : WavePhysics
        Wave physics the problem must have.
    profile : VelocityProfileType
        Velocity profile key of the problem.
    """
    problem_class = CPW_PROBLEMS[(wave_type, profile)]

    assert problem_class.physics is physics
    assert problem_class.velocity_profile_type is profile


@pytest.mark.parametrize(
    "problem_class, expected",
    [(AcousticHomogeneousProblem, 1.5)],
)
def test_homogeneous_minimum_wavelength_velocity(
    problem_class: type[CpwProblem], expected: float,
) -> None:
    """The wavelength is the acoustic velocity, or the S-wave velocity."""
    assert make_problem(problem_class).minimum_wavelength_velocity == expected


@pytest.mark.parametrize("problem_class", PROBLEM_CLASSES)
def test_problem_rejects_3d(problem_class: type[CpwProblem]) -> None:
    """Only 2D problems are implemented."""
    with pytest.raises(NotImplementedError, match="only supports 2D"):
        make_problem(problem_class, dimension=3)


@pytest.mark.parametrize("problem_class", PROBLEM_CLASSES)
def test_problem_rejects_non_positive_mesh_frequency_factor(
    problem_class: type[CpwProblem],
) -> None:
    """The meshing frequency must be a positive multiple of the peak."""
    with pytest.raises(ValueError, match="mesh_frequency_factor must be positive"):
        make_problem(problem_class, mesh_frequency_factor=-1.0)


def test_calculator_rejects_unregistered_problem(monkeypatch) -> None:
    """A wave and profile type pair with no problem is not implemented."""
    monkeypatch.setattr(calculator_module, "CPW_PROBLEMS", {})

    with pytest.raises(NotImplementedError, match="No cells-per-wavelength problem"):
        spyro.tools.MeshingParameterCalculator(CALCULATOR_PARAMETERS)


@pytest.mark.parametrize(
    "threshold_settings",
    [{}, {"accepted_error_threshold": 0.05, "threshold_discretization": True}],
)
def test_calculator_needs_exactly_one_threshold(threshold_settings: dict) -> None:
    """A threshold is given directly or by a discretization, not both."""
    parameters = {
        key: value
        for key, value in CALCULATOR_PARAMETERS.items()
        if key != "accepted_error_threshold"
    }
    parameters.update(threshold_settings)

    with pytest.raises(ValueError, match="exactly one"):
        spyro.tools.MeshingParameterCalculator(parameters)


@pytest.mark.parametrize(
    "problem_class", [AcousticHomogeneousProblem]
)
def test_problem_builds_dictionary_for_requested_discretization(
    problem_class: type[CpwProblem],
) -> None:
    """The threshold discretization can use a method and degree the search does not."""
    problem = make_problem(problem_class)

    dictionary = problem.model_dictionary("spectral_quadrilateral", 4)

    assert dictionary["options"]["method"] == "spectral_quadrilateral"
    assert dictionary["options"]["degree"] == 4
    assert dictionary["mesh"]["mesh_type"] == "firedrake_mesh"


def test_calculator_rejects_non_positive_mesh_frequency_factor() -> None:
    """The meshing frequency must be a positive multiple of the peak."""
    parameters = {**CALCULATOR_PARAMETERS, "mesh_frequency_factor": 0.0}

    with pytest.raises(ValueError, match="mesh_frequency_factor must be positive"):
        spyro.tools.MeshingParameterCalculator(parameters)


@pytest.mark.parametrize("factor", [1.0, 2.0, 2.5])
@pytest.mark.parametrize("problem_class", PROBLEM_CLASSES)
def test_mesh_frequency_factor_scales_mesh_size(
    factor: float, problem_class: type[CpwProblem],
) -> None:
    """A cpw refers to the wavelength at ``factor`` times the peak frequency."""
    frequency, cpw = 5.0, 3.0
    problem = make_problem(
        problem_class, source_frequency=frequency, mesh_frequency_factor=factor
    )
    velocity = problem.minimum_wavelength_velocity
    requested = {}

    class RecordingWave:
        """Records the mesh request instead of building a mesh."""

        def set_mesh(self, input_mesh_parameters: dict) -> None:
            """Record the mesh parameters.

            Parameters
            ----------
            input_mesh_parameters : dict
                Mesh parameters passed by the problem.
            """
            requested.update(input_mesh_parameters)

        def set_initial_velocity_model(self, constant: float) -> None:
            """Accept and ignore a constant velocity.

            Parameters
            ----------
            constant : float
                Constant velocity set by the problem.
            """

    problem.setup_wave(RecordingWave(), cpw)

    assert np.isclose(
        requested["edge_length"], velocity / (factor * frequency * cpw)
    )


def test_interpolate_time_series_applies_time_shift() -> None:
    """Target sample i is taken at initial_time + time_shift + i * dt."""
    final_time = 1.0
    source_time = np.linspace(0.0, final_time, 101)

    interpolated = interpolate_time_series(
        3.0 * source_time, 0.1, final_time=final_time, time_shift=0.05
    )

    expected = 3.0 * np.minimum(0.05 + 0.1 * np.arange(11), final_time)
    assert np.allclose(interpolated, expected)


@pytest.mark.parametrize(
    "problem_class, expected_steps",
    [
        (AcousticHomogeneousProblem, 1.0),
    ],
)
def test_reference_time_shift(
    problem_class: type[CpwProblem], expected_steps: float,
) -> None:
    """Only analytical acoustic references are shifted one solver step."""
    wave = SimpleNamespace(dt=2e-3)

    shift = make_problem(problem_class).reference_time_shift(wave)

    assert np.isclose(shift, expected_steps * wave.dt)


@pytest.mark.parametrize(
    "problem_class, expected",
    [
        (AcousticHomogeneousProblem, "acoustic_homogeneous_cpw_reference.npy"),
    ],
)
def test_default_reference_file(
    problem_class: type[CpwProblem], expected: str,
) -> None:
    """The default reference file is named after the physics and profile."""
    assert make_problem(problem_class).default_reference_file() == expected


def test_homogeneous_reference_is_analytical(monkeypatch) -> None:
    """A homogeneous reference is the analytical data of the initial wave.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the analytical solution of the problem.
    """
    problem = make_problem(AcousticHomogeneousProblem)
    initial_wave = SimpleNamespace()
    reference = np.ones((3, 2))
    monkeypatch.setattr(
        problem,
        "analytical_solution",
        lambda wave: reference if wave is initial_wave else None,
    )

    assert problem.reference_solution(initial_wave, "mass_lumped_triangle") is reference


def test_reference_solution_saves_and_loads_default_file(
    tmp_path, monkeypatch,
) -> None:
    """A saved reference is found again by a later load with no file name."""
    monkeypatch.chdir(tmp_path)
    reference = np.arange(6.0).reshape(3, 2)
    calculator = SimpleNamespace(
        reference_solution_file=make_problem(
            AcousticHomogeneousProblem
        ).default_reference_file(),
        parameters=SimpleNamespace(
            save_reference=True,
            load_reference=False,
            fem_method="mass_lumped_triangle",
        ),
        initial_wave=None,
        problem=SimpleNamespace(
            reference_solution=lambda initial_wave, method: reference
        ),
    )

    spyro.tools.MeshingParameterCalculator.get_reference_solution(calculator)
    assert (tmp_path / "acoustic_homogeneous_cpw_reference.npy").is_file()

    calculator.parameters.save_reference = False
    calculator.parameters.load_reference = True
    calculator.problem = None
    loaded = spyro.tools.MeshingParameterCalculator.get_reference_solution(calculator)
    assert np.array_equal(loaded, reference)


def test_parameters_from_dict_maps_legacy_keys() -> None:
    """Dictionary keys map to fields, and enum strings are converted."""
    parameters = CpwParameters.from_dict(
        {
            **CALCULATOR_PARAMETERS,
            "C_max": 4.0,
            "search_direction": "forward",
            "timestep_calculation": "float",
            "timestep": 1e-3,
        }
    )

    assert parameters.fem_method == "mass_lumped_triangle"
    assert parameters.minimum_velocity == 1.5
    assert parameters.cpw_initial == 3.0
    assert parameters.cpw_accuracy == 0.1
    assert parameters.cpw_max == 4.0
    assert parameters.wave_type is WaveType.ISOTROPIC_ACOUSTIC
    assert parameters.velocity_profile_type is VelocityProfileType.HOMOGENEOUS
    assert parameters.search_direction is CpwSearchDirection.FORWARD
    assert parameters.timestep_calculation is TimeStepCalculationType.FIXED
    assert parameters.fixed_timestep == 1e-3


def test_parameters_from_dict_warns_on_unknown_keys() -> None:
    """Unknown keys, such as typos, are reported and ignored."""
    with pytest.warns(UserWarning, match="C_inital"):
        CpwParameters.from_dict({**CALCULATOR_PARAMETERS, "C_inital": 2.0})


def test_parameters_from_dict_names_missing_keys() -> None:
    """Missing required keys are named as dictionary keys."""
    parameters = {
        key: value
        for key, value in CALCULATOR_PARAMETERS.items()
        if key != "FEM_method_to_evaluate"
    }

    with pytest.raises(KeyError, match="FEM_method_to_evaluate"):
        CpwParameters.from_dict(parameters)


def test_parameters_require_mesh_frequency_factor() -> None:
    """The mesh frequency factor has no default and must be given."""
    parameters = {
        key: value
        for key, value in CALCULATOR_PARAMETERS.items()
        if key != "mesh_frequency_factor"
    }

    with pytest.raises(KeyError, match="mesh_frequency_factor"):
        CpwParameters.from_dict(parameters)


@pytest.mark.parametrize(
    "timestep_settings",
    [
        {"timestep_calculation": "float"},
        {"timestep_calculation": "estimate", "timestep": 1e-3},
        {"timestep_calculation": "exact", "timestep": 1e-3},
    ],
)
def test_parameters_pair_fixed_timestep_with_calculation(
    timestep_settings: dict,
) -> None:
    """A time step is given exactly when the time step is fixed."""
    with pytest.raises(ValueError, match="timestep"):
        CpwParameters.from_dict({**CALCULATOR_PARAMETERS, **timestep_settings})


@pytest.mark.parametrize(
    "timestep_settings, expected",
    [
        ({}, TimeStepCalculationType.ESTIMATE),
        ({"timestep": None}, TimeStepCalculationType.ESTIMATE),
        ({"timestep": 1e-3}, TimeStepCalculationType.FIXED),
        ({"timestep_calculation": "estimate"}, TimeStepCalculationType.ESTIMATE),
    ],
)
def test_parameters_infer_timestep_calculation(
    timestep_settings: dict, expected: TimeStepCalculationType,
) -> None:
    """A given time step is fixed; otherwise the time step is estimated."""
    parameters = CpwParameters.from_dict(
        {**CALCULATOR_PARAMETERS, **timestep_settings}
    )

    assert parameters.timestep_calculation is expected


def test_problem_requires_its_material() -> None:
    """A missing material parameter is named by its dictionary key."""
    with pytest.raises(ValueError, match="minimum_velocity_in_the_domain"):
        make_problem(
            AcousticHomogeneousProblem, minimum_velocity_in_the_domain=None
        )


@pytest.mark.parametrize("save_shot_records", [False, True])
def test_calculate_error_saves_shot_records_on_request(
    monkeypatch, save_shot_records: bool,
) -> None:
    """Shot records are written only when save_shot_records is set."""
    saved = []
    monkeypatch.setattr(
        spyro.io, "save_shots", lambda wave, file_name: saved.append(file_name)
    )
    evaluation = calculator_module.CpwEvaluation(2.0, 1e-3, 0.5, 1.0)
    calculator = SimpleNamespace(
        parameters=SimpleNamespace(save_shot_records=save_shot_records),
        build_wave=lambda cpw: None,
        _solve_and_measure=lambda wave, cpw: evaluation,
        search_history=[],
    )

    error = spyro.tools.MeshingParameterCalculator.calculate_error(calculator, 2.0)

    assert error == 0.5
    assert calculator.search_history == [evaluation]
    assert saved == (["test_shot_record2.0"] if save_shot_records else [])
