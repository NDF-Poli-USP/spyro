"""Tests for the cells-per-wavelength problems and calculator."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

import spyro
from spyro.io.time_io import interpolate_time_series
from spyro.receivers.Receivers import RECEIVER_STEP_OFFSET
from spyro.tools.cells_per_wavelength_calculator import (
    ACOUSTIC,
    CPW_PROBLEMS,
    AcousticHeterogeneousProblem,
    AcousticHomogeneousProblem,
    CpwParameters,
    CpwProblem,
    HeterogeneousCpwProblem,
    MaterialFileCase,
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

#: Heterogeneous case with a given minimum velocity, so no file is read.
VELOCITY_FILE_CASE = MaterialFileCase(
    name="velocity",
    length_z=3.0,
    length_x=17.0,
    material_files={"p_wave_velocity": "velocity.segy"},
    minimum_velocity=1.5,
)

PROBLEM_CLASSES = [
    AcousticHomogeneousProblem,
    AcousticHeterogeneousProblem,
]


def make_problem(problem_class: type[CpwProblem], **overrides) -> CpwProblem:
    """Build a problem with typical settings and no file access.

    A heterogeneous problem gets :data:`VELOCITY_FILE_CASE`, so no
    material file is read.

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
    parameters = {
        **CALCULATOR_PARAMETERS,
        "reference_degree": 5,
        "C_reference": 6.0,
    }
    if issubclass(problem_class, HeterogeneousCpwProblem):
        parameters["heterogeneous_case"] = VELOCITY_FILE_CASE
    parameters.update(overrides)
    return problem_class(CpwParameters(**parameters))


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
        (WaveType.ISOTROPIC_ACOUSTIC, ACOUSTIC, VelocityProfileType.HETEROGENEOUS),
    ],
)
def test_cpw_problems_registry(
    wave_type: WaveType,
    physics: WavePhysics,
    profile: VelocityProfileType,
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
    problem_class: type[CpwProblem],
    expected: float,
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
    with pytest.raises(ValueError, match="mesh_frequency_factor"):
        make_problem(problem_class, mesh_frequency_factor=-1.0)


@pytest.mark.parametrize(
    "problem_class, case",
    [
        (AcousticHomogeneousProblem, VELOCITY_FILE_CASE),
        (AcousticHeterogeneousProblem, None),
    ],
)
def test_problem_matches_case_to_profile(
    problem_class: type[CpwProblem], case: MaterialFileCase | None,
) -> None:
    """Only heterogeneous problems have a heterogeneous case.

    Parameters
    ----------
    problem_class : type[CpwProblem]
        Problem to build.
    case : MaterialFileCase or None
        Case given to the problem, of the wrong kind for its profile.
    """
    with pytest.raises(ValueError, match="heterogeneous_case should be given"):
        make_problem(problem_class, heterogeneous_case=case)


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


@pytest.mark.parametrize("problem_class", [AcousticHomogeneousProblem])
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

    with pytest.raises(ValueError, match="mesh_frequency_factor"):
        spyro.tools.MeshingParameterCalculator(parameters)


@pytest.mark.parametrize("factor", [1.0, 2.0, 2.5])
@pytest.mark.parametrize("problem_class", PROBLEM_CLASSES)
def test_mesh_frequency_factor_scales_mesh_size(
    factor: float,
    problem_class: type[CpwProblem],
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

    assert np.isclose(requested["edge_length"], velocity / (factor * frequency * cpw))


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
        (AcousticHomogeneousProblem, RECEIVER_STEP_OFFSET),
        (AcousticHeterogeneousProblem, 0.0),
    ],
)
def test_reference_time_shift(
    problem_class: type[CpwProblem],
    expected_steps: float,
) -> None:
    """Analytical references are resampled with the receiver offset."""
    wave = SimpleNamespace(dt=2e-3)

    shift = make_problem(problem_class).reference_time_shift(wave)

    assert np.isclose(shift, expected_steps * wave.dt)


@pytest.mark.parametrize(
    "problem_class, expected",
    [
        (AcousticHomogeneousProblem, "acoustic_homogeneous_cpw_reference.npz"),
        (AcousticHeterogeneousProblem, "acoustic_heterogeneous_cpw_reference.npz"),
    ],
)
def test_default_reference_file(
    problem_class: type[CpwProblem],
    expected: str,
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


def test_heterogeneous_reference_uses_reference_discretization(monkeypatch) -> None:
    """A heterogeneous reference is solved at the reference degree and cpw.

    It also saves its pressure field under the case name.
    """
    problem = make_problem(AcousticHeterogeneousProblem)
    reference = np.ones((3, 2))
    built = {}

    def build_wave(dictionary: dict, cpw: float) -> SimpleNamespace:
        """Record the reference discretization instead of building a wave.

        Parameters
        ----------
        dictionary : dict
            Model dictionary of the reference.
        cpw : float
            Cells-per-wavelength of the reference.

        Returns
        -------
        SimpleNamespace
            Stand-in wave whose solve returns ``reference``.
        """
        built.update(dictionary, cpw=cpw)
        return SimpleNamespace(
            forward_solve=lambda: None, forward_solution_receivers=reference
        )

    monkeypatch.setattr(
        problem,
        "model_dictionary",
        lambda method, degree: {
            "method": method,
            "degree": degree,
            "visualization": {"forward_output": False},
        },
    )
    monkeypatch.setattr(problem, "build_wave", build_wave)

    result = problem.reference_solution(None, "mass_lumped_triangle")

    assert result is reference
    assert built == {
        "method": "mass_lumped_triangle",
        "degree": 5,
        "visualization": {
            "forward_output": True,
            "forward_output_filename": "results/velocity_reference_forward_output.pvd",
        },
        "cpw": 6.0,
    }


def test_acoustic_analytical_reference_has_no_receiver_offset(monkeypatch) -> None:
    """The stored reference is the field at its time labels, for any solver dt.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Records the calls to the analytical solution.
    """
    time_shifts = []

    def record_time_shift(wave, offset, velocity, **kwargs) -> np.ndarray:
        """Record the requested time shift and return zero data.

        Parameters
        ----------
        wave : SimpleNamespace
            Stand-in wave holding the time axis.
        offset : float
            Source-receiver distance, unused.
        velocity : float
            Medium velocity, unused.
        **kwargs
            Keyword arguments, holding `time_shift`.

        Returns
        -------
        np.ndarray
            Zeros, one per time sample.
        """
        time_shifts.append(kwargs.get("time_shift"))
        return np.zeros(round(wave.final_time / wave.dt) + 1)

    monkeypatch.setattr(spyro.utils, "nodal_homogeneous_analytical", record_time_shift)
    wave = SimpleNamespace(
        dt=0.1,
        final_time=1.0,
        number_of_receivers=2,
        source_locations=[(0.0, 0.0)],
        receiver_locations=[(-1.0, 0.0), (-1.0, 1.0)],
    )

    make_problem(AcousticHomogeneousProblem).analytical_solution(wave)

    assert time_shifts == [0.0, 0.0]


def fake_analytical_solution(problem: CpwProblem, wave) -> np.ndarray:
    """Return distinct receiver data shaped like the analytical solution.

    Parameters
    ----------
    problem : CpwProblem
        Problem whose analytical solution is replaced, unused.
    wave : Wave
        Wave object holding the acquisition and time axis.

    Returns
    -------
    np.ndarray
        Data shaped `(round(final_time / dt) + 1, number_of_receivers)`.
    """
    num_t = round(wave.final_time / wave.dt) + 1
    return np.arange(num_t * wave.number_of_receivers, dtype=float).reshape(
        num_t, wave.number_of_receivers
    )


def save_reference(**overrides) -> spyro.tools.MeshingParameterCalculator:
    """Build a calculator that saves a fake reference to the default file.

    Parameters
    ----------
    **overrides
        Parameters replacing :data:`CALCULATOR_PARAMETERS`.

    Returns
    -------
    spyro.tools.MeshingParameterCalculator
        The calculator, whose reference is in the working directory.
    """
    return spyro.tools.MeshingParameterCalculator(
        {**CALCULATOR_PARAMETERS, "save_reference": True, **overrides}
    )


def test_reference_solution_saves_and_loads_default_file(
    tmp_path,
    monkeypatch,
) -> None:
    """A saved reference is found again by a later load with no file name.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Working directory of the saved reference.
    monkeypatch : pytest.MonkeyPatch
        Replaces the analytical solution and the working directory.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        AcousticHomogeneousProblem, "analytical_solution", fake_analytical_solution
    )
    saved = save_reference()
    assert (tmp_path / "acoustic_homogeneous_cpw_reference.npz").is_file()

    loaded = spyro.tools.MeshingParameterCalculator(
        {**CALCULATOR_PARAMETERS, "load_reference": True}
    )
    assert np.array_equal(loaded.reference_solution, saved.reference_solution)


@pytest.mark.parametrize(
    "overrides, mismatched",
    [
        ({"source_frequency": 6.0}, "source_frequency"),
        ({"minimum_velocity_in_the_domain": 2.0}, "velocity"),
        ({"testing": False}, "receiver_locations"),
    ],
)
def test_loaded_reference_must_match_settings(
    tmp_path,
    monkeypatch,
    overrides: dict,
    mismatched: str,
) -> None:
    """A reference saved for other settings is rejected, naming them.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Working directory of the saved reference.
    monkeypatch : pytest.MonkeyPatch
        Replaces the analytical solution and the working directory.
    overrides : dict
        Parameters of the load that differ from the save.
    mismatched : str
        Metadata key the error must name.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        AcousticHomogeneousProblem, "analytical_solution", fake_analytical_solution
    )
    save_reference()

    with pytest.raises(ValueError, match=f"different .*{mismatched}"):
        spyro.tools.MeshingParameterCalculator(
            {**CALCULATOR_PARAMETERS, "load_reference": True, **overrides}
        )


def test_loaded_reference_needs_metadata(tmp_path) -> None:
    """A bare array, as the calculator used to save, is rejected.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory of the reference file.
    """
    file = tmp_path / "reference.npz"
    with open(file, "wb") as stream:
        np.save(stream, np.zeros((3, 4)))

    with pytest.raises(ValueError, match="holds no metadata"):
        spyro.tools.MeshingParameterCalculator(
            {
                **CALCULATOR_PARAMETERS,
                "load_reference": True,
                "reference_solution_file": str(file),
            }
        )


def test_loaded_reference_must_match_time_axis(tmp_path, monkeypatch) -> None:
    """Data with matching metadata but the wrong number of samples is rejected.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Working directory of the saved reference.
    monkeypatch : pytest.MonkeyPatch
        Replaces the analytical solution and the working directory.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        AcousticHomogeneousProblem,
        "analytical_solution",
        lambda problem, wave: fake_analytical_solution(problem, wave)[:-1],
    )
    save_reference()

    with pytest.raises(ValueError, match="has shape"):
        spyro.tools.MeshingParameterCalculator(
            {**CALCULATOR_PARAMETERS, "load_reference": True}
        )


def test_parameters_reject_reference_file_without_npz_suffix() -> None:
    """Other suffixes are rejected, since `np.savez` would append `.npz`."""
    with pytest.raises(ValidationError, match=r"\.npz"):
        CpwParameters(**CALCULATOR_PARAMETERS, reference_solution_file="reference.npy")


def test_stored_acoustic_reference_matches_its_settings() -> None:
    """The stored reference loads for the full acquisition it was made for."""
    stored_file = (
        Path(__file__).parents[1]
        / "inputfiles"
        / "acoustic_homogeneous_cpw_reference.npz"
    )
    calculator = spyro.tools.MeshingParameterCalculator(
        {
            **CALCULATOR_PARAMETERS,
            "testing": False,
            "load_reference": True,
            "reference_solution_file": str(stored_file),
        }
    )

    assert calculator.reference_solution.shape[1] == 36


def test_search_history_files_are_named_by_method(tmp_path, monkeypatch) -> None:
    """Searches of two methods at the same degree keep separate histories.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Working directory of the saved histories.
    monkeypatch : pytest.MonkeyPatch
        Replaces the analytical solution and the working directory.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        AcousticHomogeneousProblem, "analytical_solution", fake_analytical_solution
    )
    error_scales = {"mass_lumped_triangle": 1.0, "spectral_quadrilateral": 0.5}
    for method, error_scale in error_scales.items():
        calculator = spyro.tools.MeshingParameterCalculator(
            {**CALCULATOR_PARAMETERS, "FEM_method_to_evaluate": method}
        )

        def calculate_error(cpw: float, calculator=calculator, scale=error_scale):
            """Record a solve whose error falls with cpw, without solving.

            Parameters
            ----------
            cpw : float
                Cells-per-wavelength of the solve.
            calculator : spyro.tools.MeshingParameterCalculator
                Calculator whose history the solve is appended to.
            scale : float
                Error at 1 cell-per-wavelength.

            Returns
            -------
            float
                The receiver error.
            """
            evaluation = calculator_module.CpwEvaluation(cpw, 1e-3, scale / cpw**2, 0.0)
            calculator.search_history.append(evaluation)
            return evaluation.error

        calculator.calculate_error = calculate_error
        calculator.find_minimum(savetxt=True)

    # Each file holds its own method's errors, so neither was overwritten
    for method, error_scale in error_scales.items():
        history = np.loadtxt(tmp_path / f"{method}_p2_cpw_results.txt", ndmin=2)
        cpw, error = history[:, 0], history[:, 2]
        assert np.allclose(error, error_scale / cpw**2)


def test_parameters_from_dict_maps_legacy_keys() -> None:
    """Dictionary keys map to fields, and enum strings are converted."""
    parameters = CpwParameters(
        **{
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


def test_parameters_from_dict_rejects_unknown_keys() -> None:
    """Unknown keys, such as typos, are rejected."""
    with pytest.raises(ValidationError, match="C_inital"):
        CpwParameters(**{**CALCULATOR_PARAMETERS, "C_inital": 2.0})


def test_parameters_infer_heterogeneous_profile() -> None:
    """A heterogeneous case makes the profile heterogeneous by default."""
    parameters = CpwParameters(
        **{**CALCULATOR_PARAMETERS, "heterogeneous_case": VELOCITY_FILE_CASE}
    )

    assert parameters.velocity_profile_type is VelocityProfileType.HETEROGENEOUS


def test_parameters_from_dict_names_missing_keys() -> None:
    """Missing required keys are named as dictionary keys."""
    parameters = {
        key: value
        for key, value in CALCULATOR_PARAMETERS.items()
        if key != "FEM_method_to_evaluate"
    }

    with pytest.raises(ValidationError, match="FEM_method_to_evaluate"):
        CpwParameters(**parameters)


def test_parameters_require_mesh_frequency_factor() -> None:
    """The mesh frequency factor has no default and must be given."""
    parameters = {
        key: value
        for key, value in CALCULATOR_PARAMETERS.items()
        if key != "mesh_frequency_factor"
    }

    with pytest.raises(ValidationError, match="mesh_frequency_factor"):
        CpwParameters(**parameters)


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
        CpwParameters(**{**CALCULATOR_PARAMETERS, **timestep_settings})


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
    timestep_settings: dict,
    expected: TimeStepCalculationType,
) -> None:
    """A given time step is fixed; otherwise the time step is estimated."""
    parameters = CpwParameters(**{**CALCULATOR_PARAMETERS, **timestep_settings})

    assert parameters.timestep_calculation is expected


def test_problem_requires_its_material() -> None:
    """A missing material parameter is named by its dictionary key."""
    with pytest.raises(ValueError, match="minimum_velocity_in_the_domain"):
        make_problem(AcousticHomogeneousProblem, minimum_velocity_in_the_domain=None)


@pytest.mark.parametrize("save_shot_records", [False, True])
def test_calculate_error_saves_shot_records_on_request(
    monkeypatch,
    save_shot_records: bool,
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
