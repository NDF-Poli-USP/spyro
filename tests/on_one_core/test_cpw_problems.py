"""Tests for the cells-per-wavelength problems and calculator."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import firedrake as fire
import numpy as np
import pytest
from pydantic import ValidationError

import spyro
from spyro.io.time_io import interpolate_time_series
from spyro.receivers.Receivers import RECEIVER_STEP_OFFSET
from spyro.tools.cells_per_wavelength_calculator import (
    ACOUSTIC,
    CPW_PROBLEMS,
    ISOTROPIC_ELASTIC,
    AcousticHeterogeneousProblem,
    AcousticHomogeneousProblem,
    CpwParameters,
    CpwProblem,
    ElasticHeterogeneousProblem,
    ElasticHomogeneousProblem,
    HeterogeneousCpwProblem,
    MaterialFileCase,
    WavePhysics,
)
from spyro.tools.cells_per_wavelength_calculator import calculator as calculator_module
from spyro.tools.cells_per_wavelength_calculator.elastic_2d_material_model import (
    elastic_2d_material,
)
from spyro.tools.cells_per_wavelength_calculator.heterogeneous_cases import (
    ElasticSyntheticCase,
)
from spyro.tools.cells_per_wavelength_calculator.input_models import (
    create_elastic_model_for_meshing_parameter_2D_homogeneous,
)
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
    ElasticHomogeneousProblem,
    ElasticHeterogeneousProblem,
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
        "p_wave_velocity": 5.6,
        "s_wave_velocity": 2.75,
        "density": 2.0,
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
        (
            WaveType.ISOTROPIC_ELASTIC,
            ISOTROPIC_ELASTIC,
            VelocityProfileType.HOMOGENEOUS,
        ),
        (
            WaveType.ISOTROPIC_ELASTIC,
            ISOTROPIC_ELASTIC,
            VelocityProfileType.HETEROGENEOUS,
        ),
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
    [(AcousticHomogeneousProblem, 1.5), (ElasticHomogeneousProblem, 2.75)],
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
        (ElasticHomogeneousProblem, VELOCITY_FILE_CASE),
        (ElasticHeterogeneousProblem, None),
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


def test_elastic_model_geometry_avoids_periodic_images() -> None:
    """No periodic-image P-wave reaches a receiver before the final time."""
    vp, vs, frequency = 5.6, 2.75, 10.0
    dictionary = create_elastic_model_for_meshing_parameter_2D_homogeneous(
        method="mass_lumped_triangle",
        degree=2,
        frequency=frequency,
        p_wave_velocity=vp,
        s_wave_velocity=vs,
        density=2.0,
        dt=1e-4,
    )

    length = dictionary["mesh"]["length_z"]
    assert dictionary["mesh"]["length_x"] == length
    assert dictionary["mesh"]["periodic"]
    final_time = dictionary["time_axis"]["final_time"]
    source = np.array(dictionary["acquisition"]["source_locations"][0])
    receivers = np.array(dictionary["acquisition"]["receiver_locations"])
    assert len(receivers) == 11

    # All receivers lie inside the box
    assert np.all((receivers[:, 0] < 0.0) & (receivers[:, 0] > -length))
    assert np.all((receivers[:, 1] > 0.0) & (receivers[:, 1] < length))

    # The S-wave reaches the farthest receiver before the final time
    distances = np.linalg.norm(receivers - source, axis=1)
    assert distances.max() / vs < final_time

    # Nearest periodic image of the source, over the 8 neighbouring copies
    shifts = (
        np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)])
        * length
    )
    images = source + shifts
    image_distance = np.linalg.norm(
        receivers[:, None, :] - images[None, :, :],
        axis=2,
    ).min()
    assert image_distance / vp > final_time


def test_elastic_maximum_dt_uses_elastic_operator() -> None:
    """The elastic stable dt sits between the acoustic ones at vp and vs."""
    vp, vs = 2.0, 1.0
    dictionary = create_elastic_model_for_meshing_parameter_2D_homogeneous(
        method="mass_lumped_triangle",
        degree=2,
        frequency=5.0,
        p_wave_velocity=vp,
        s_wave_velocity=vs,
        density=1.0,
        dt=1e-3,
        reduced=True,
    )
    dictionary["mesh"]["length_z"] = 1.0
    dictionary["mesh"]["length_x"] = 1.0
    dictionary["acquisition"]["source_locations"] = [(-0.5, 0.5)]
    dictionary["acquisition"]["receiver_locations"] = [(-0.6, 0.5)]
    dictionary["time_axis"]["final_time"] = 0.1

    elastic = spyro.IsotropicWave(dictionary)
    elastic.set_mesh(input_mesh_parameters={"edge_length": 0.1, "periodic": True})
    elastic.initialize_physical_parameters()
    elastic_dt = elastic.get_and_set_maximum_dt(fraction=1.0)

    acoustic_dictionary = {
        key: value for key, value in dictionary.items() if key != "synthetic_data"
    }
    acoustic_dictionary["acquisition"] = {**dictionary["acquisition"], "amplitude": 1.0}
    acoustic_dts = []
    for velocity in (vp, vs):
        acoustic = spyro.AcousticWave(acoustic_dictionary)
        acoustic.set_mesh(input_mesh_parameters={"edge_length": 0.1, "periodic": True})
        acoustic.set_initial_velocity_model(constant=velocity)
        acoustic.initialize_physical_parameters()
        acoustic_dts.append(acoustic.get_and_set_maximum_dt(fraction=1.0))

    assert acoustic_dts[0] < elastic_dt * 1.05
    assert elastic_dt < acoustic_dts[1]

    # For a constant velocity, the solver-consistent acoustic forms (velocity
    # in the mass) give the same dt as the modal solver's default forms
    # (velocity in the stiffness).
    dt_solver = spyro.solvers.modal.modal_sol.Modal_Solver(
        2,
        method="ARNOLDI",
        calc_max_dt=True,
    )
    default_forms_dt = dt_solver.estimate_timestep(
        acoustic.c,
        acoustic.function_space,
        acoustic.final_time,
        quad_rule=acoustic.quadrature_rule,
        fraction=1.0,
    )
    assert np.isclose(acoustic_dts[1], default_forms_dt)


@pytest.mark.slow
def test_elastic_cpw_search_meets_threshold() -> None:
    """The elastic search result passes, and one step coarser fails."""
    parameters = {
        "wave_type": WaveType.ISOTROPIC_ELASTIC,
        "source_frequency": 10.0,
        "mesh_frequency_factor": 2.0,
        "p_wave_velocity": 5.6,
        "s_wave_velocity": 2.75,
        "density": 2.0,
        "velocity_profile_type": "homogeneous",
        "FEM_method_to_evaluate": "mass_lumped_triangle",
        "dimension": 2,
        "testing": True,
        "timestep_calculation": "exact",
        "desired_degree": 4,
        "C_initial": 0.5,
        "C_max": 3.0,
        "search_direction": "forward",
        "accepted_error_threshold": 0.05,
        "C_accuracy": 0.5,
    }
    calculator = spyro.tools.MeshingParameterCalculator(parameters)
    assert calculator.reference_solution.shape[1:] == (3, 2)

    cpw = calculator.find_minimum()

    errors = {e.cpw: e.error for e in calculator.search_history}
    assert errors[cpw] <= 0.05
    assert errors[cpw - 0.5] > 0.05


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
    "problem_class", [AcousticHomogeneousProblem, ElasticHomogeneousProblem]
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

        def set_initial_velocity_model(self, **velocity: object) -> None:
            """Accept and ignore the velocity, constant or from a file.

            Parameters
            ----------
            **velocity
                Keyword arguments of `Wave.set_initial_velocity_model`, e.g.
                `constant` or `new_file` and `fast_interpolate`.
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
    "problem_class, expected_shift",
    [
        # Analytical: the whole receiver offset of the solve
        (AcousticHomogeneousProblem, RECEIVER_STEP_OFFSET * 2e-3),
        (ElasticHomogeneousProblem, RECEIVER_STEP_OFFSET * 2e-3),
        # Numerical: the reference already holds its own offset
        (AcousticHeterogeneousProblem, RECEIVER_STEP_OFFSET * (2e-3 - 1e-4)),
        (ElasticHeterogeneousProblem, RECEIVER_STEP_OFFSET * (2e-3 - 1e-4)),
    ],
)
def test_reference_time_shift(
    problem_class: type[CpwProblem],
    expected_shift: float,
) -> None:
    """References are shifted by the receiver offset they do not hold."""
    wave = SimpleNamespace(dt=2e-3)

    shift = make_problem(problem_class).reference_time_shift(wave, 1e-4)

    assert np.isclose(shift, expected_shift)


@pytest.mark.parametrize("solver_dt", [5e-4, 1e-3, 2e-3])
def test_numerical_reference_resamples_onto_solve_times(solver_dt: float) -> None:
    """A reference solved at another dt matches the solve after resampling.

    Both records hold a Ricker wavelet at `(i + RECEIVER_STEP_OFFSET) * dt`,
    as receiver data does. Each solver step is a whole number of reference
    steps, so the shifted samples land on reference samples and match
    exactly.

    Parameters
    ----------
    solver_dt : float
        Time step of the forward solve.
    """
    final_time, reference_dt, frequency = 1.0, 1e-4, 5.0

    def recorded(dt: float) -> np.ndarray:
        """Return a Ricker trace sampled as receiver data at a time step.

        Parameters
        ----------
        dt : float
            Time step of the record.

        Returns
        -------
        np.ndarray
            The trace, shaped `(n_time, 1)`.
        """
        steps = np.arange(round(final_time / dt) + 1) + RECEIVER_STEP_OFFSET
        argument = (np.pi * frequency * (steps * dt - 1.0 / frequency)) ** 2
        return ((1.0 - 2.0 * argument) * np.exp(-argument))[:, np.newaxis]

    problem = make_problem(AcousticHeterogeneousProblem)
    shift = problem.reference_time_shift(SimpleNamespace(dt=solver_dt), reference_dt)
    resampled = interpolate_time_series(
        recorded(reference_dt), solver_dt, final_time=final_time, time_shift=shift
    )

    assert np.allclose(resampled, recorded(solver_dt), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "problem_class, expected",
    [
        (AcousticHomogeneousProblem, "acoustic_homogeneous_cpw_reference.npz"),
        (AcousticHeterogeneousProblem, "acoustic_heterogeneous_cpw_reference.npz"),
        (ElasticHomogeneousProblem, "elastic_homogeneous_cpw_reference.npz"),
        (ElasticHeterogeneousProblem, "elastic_heterogeneous_cpw_reference.npz"),
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


@pytest.mark.parametrize(
    "problem_class, analytical_name, components",
    [
        (AcousticHomogeneousProblem, "nodal_homogeneous_analytical", None),
        (ElasticHomogeneousProblem, "analytical_solution_elastic", 2),
    ],
)
def test_analytical_reference_has_no_receiver_offset(
    monkeypatch,
    problem_class: type[CpwProblem],
    analytical_name: str,
    components: int | None,
) -> None:
    """The stored reference is the field at its time labels, for any solver dt.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Records the calls to the analytical solution.
    problem_class : type[CpwProblem]
        Homogeneous problem under test.
    analytical_name : str
        Name, in `spyro.utils`, of the analytical solution it calls.
    components : int or None
        Number of displacement components the analytical solution returns,
        or None for a single trace.
    """
    time_shifts = []
    num_t = 11

    def record_time_shift(*args, **kwargs) -> np.ndarray | tuple[np.ndarray, ...]:
        """Record the requested time shift and return zero data.

        Parameters
        ----------
        *args
            Positional arguments, unused.
        **kwargs
            Keyword arguments, holding `time_shift`.

        Returns
        -------
        np.ndarray or tuple of np.ndarray
            Zeros, one per time sample, once per component.
        """
        time_shifts.append(kwargs.get("time_shift"))
        trace = np.zeros(num_t)
        return trace if components is None else (trace,) * components

    monkeypatch.setattr(spyro.utils, analytical_name, record_time_shift)
    wave = SimpleNamespace(
        dt=0.1,
        final_time=0.1 * (num_t - 1),
        number_of_receivers=2,
        source_locations=[(0.0, 0.0)],
        receiver_locations=[(-1.0, 0.0), (-1.0, 1.0)],
        amplitude=np.array([1.0, 0.0]),
        dimension=2,
        frequency=5.0,
        delay=0.2,
    )

    make_problem(problem_class).analytical_solution(wave)

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


#: Heterogeneous case with a short time axis and two receivers, so a fake
#: reference stays small.
SMALL_VELOCITY_FILE_CASE = replace(
    VELOCITY_FILE_CASE,
    receiver_locations=[(-1.0, 5.0), (-1.0, 6.0)],
    final_time=0.1,
    dt=1e-2,
)

#: Calculator parameters of a heterogeneous acoustic search, with no file access.
HETEROGENEOUS_CALCULATOR_PARAMETERS = {
    **CALCULATOR_PARAMETERS,
    "heterogeneous_case": SMALL_VELOCITY_FILE_CASE,
    "reference_degree": 3,
    "C_reference": 4.0,
}


def fake_numerical_reference(
    problem: CpwProblem, initial_wave, method: str
) -> np.ndarray:
    """Return distinct receiver data shaped like a numerical reference.

    Parameters
    ----------
    problem : CpwProblem
        Problem whose reference is replaced.
    initial_wave : Wave
        Wave object holding the acquisition and time axis.
    method : str
        Finite element method of the reference, unused.

    Returns
    -------
    np.ndarray
        Data from :func:`fake_analytical_solution`.
    """
    return fake_analytical_solution(problem, initial_wave)


def test_heterogeneous_reference_saves_and_loads(tmp_path, monkeypatch) -> None:
    """A heterogeneous reference, with text and bool metadata, loads again.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Working directory of the saved reference.
    monkeypatch : pytest.MonkeyPatch
        Replaces the reference solve and the working directory.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        AcousticHeterogeneousProblem, "reference_solution", fake_numerical_reference
    )
    saved = spyro.tools.MeshingParameterCalculator(
        {**HETEROGENEOUS_CALCULATOR_PARAMETERS, "save_reference": True}
    )

    loaded = spyro.tools.MeshingParameterCalculator(
        {**HETEROGENEOUS_CALCULATOR_PARAMETERS, "load_reference": True}
    )

    assert np.array_equal(loaded.reference_solution, saved.reference_solution)


@pytest.mark.parametrize(
    "overrides, mismatched",
    [
        ({"FEM_method_to_evaluate": "spectral_quadrilateral"}, "method"),
        ({"reference_degree": 4}, "reference_degree"),
        ({"C_reference": 5.0}, "cpw_reference"),
        ({"mesh_frequency_factor": 3.0}, "mesh_frequency"),
        (
            {"heterogeneous_case": replace(SMALL_VELOCITY_FILE_CASE, name="other")},
            "case_name",
        ),
        (
            {"heterogeneous_case": replace(SMALL_VELOCITY_FILE_CASE, length_x=16.0)},
            "length_x",
        ),
        (
            {
                "heterogeneous_case": replace(
                    SMALL_VELOCITY_FILE_CASE, minimum_velocity=2.0
                )
            },
            "minimum_wavelength_velocity",
        ),
        ({"velocity_adapted_mesh": True}, "velocity_adapted_mesh"),
    ],
)
def test_loaded_heterogeneous_reference_must_match_settings(
    tmp_path,
    monkeypatch,
    overrides: dict,
    mismatched: str,
) -> None:
    """A heterogeneous reference saved for another discretization is rejected.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Working directory of the saved reference.
    monkeypatch : pytest.MonkeyPatch
        Replaces the reference solve and the working directory.
    overrides : dict
        Parameters of the load that differ from the save.
    mismatched : str
        Metadata key the error must name.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        AcousticHeterogeneousProblem, "reference_solution", fake_numerical_reference
    )
    spyro.tools.MeshingParameterCalculator(
        {**HETEROGENEOUS_CALCULATOR_PARAMETERS, "save_reference": True}
    )

    with pytest.raises(ValueError, match=f"different .*{mismatched}"):
        spyro.tools.MeshingParameterCalculator(
            {
                **HETEROGENEOUS_CALCULATOR_PARAMETERS,
                "load_reference": True,
                **overrides,
            }
        )


def test_parameters_reject_reference_file_without_npz_suffix() -> None:
    """Other suffixes are rejected, since `np.savez` would append `.npz`."""
    with pytest.raises(ValidationError, match=r"\.npz"):
        CpwParameters(**CALCULATOR_PARAMETERS, reference_solution_file="reference.npy")


@pytest.mark.parametrize(
    "parameters, stored_name, receiver_shape",
    [
        ({}, "acoustic_homogeneous_cpw_reference.npz", (36,)),
        (
            {
                "wave_type": WaveType.ISOTROPIC_ELASTIC,
                "source_frequency": 10.0,
                "p_wave_velocity": 5.6,
                "s_wave_velocity": 2.75,
                "density": 2.0,
            },
            "elastic_homogeneous_cpw_reference.npz",
            (11, 2),
        ),
    ],
    ids=["acoustic", "elastic"],
)
def test_stored_homogeneous_reference_matches_its_settings(
    parameters: dict, stored_name: str, receiver_shape: tuple[int, ...]
) -> None:
    """The stored reference loads for the full acquisition it was made for.

    Parameters
    ----------
    parameters : dict
        Physics-specific parameters replacing :data:`CALCULATOR_PARAMETERS`.
    stored_name : str
        Name of the stored reference in `tests/inputfiles`.
    receiver_shape : tuple of int
        Shape of the reference data after its time axis.
    """
    stored_file = Path(__file__).parents[1] / "inputfiles" / stored_name
    calculator = spyro.tools.MeshingParameterCalculator(
        {
            **CALCULATOR_PARAMETERS,
            **parameters,
            "testing": False,
            "load_reference": True,
            "reference_solution_file": str(stored_file),
        }
    )

    assert calculator.reference_solution.shape[1:] == receiver_shape


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


@pytest.mark.parametrize("method", ["mass_lumped_triangle", "spectral_quadrilateral"])
def test_elastic_heterogeneous_setup_meshes_and_reads_material(
    method: str, tmp_path,
) -> None:
    """The mesh is sized by the minimum S-wavelength; the material is read."""
    frequency, factor, cpw = 10.0, 2.0, 0.1
    minimum_s_wave_velocity = 1.2
    problem = make_problem(
        ElasticHeterogeneousProblem,
        heterogeneous_case=ElasticSyntheticCase(name=str(tmp_path / "salt")),
        source_frequency=frequency,
        mesh_frequency_factor=factor,
    )

    wave = problem.build_wave(problem.model_dictionary(method, 2), cpw)
    wave.initialize_physical_parameters()

    edge_length = minimum_s_wave_velocity / (factor * frequency * cpw)
    cells_per_square = 1 if method == "spectral_quadrilateral" else 2
    expected_cells = (
        cells_per_square * round(5.0 / edge_length) * round(10.0 / edge_length)
    )
    assert wave.mesh.num_cells() == expected_cells

    # The file values match the material model at the nodes
    space = wave.c_s.function_space()
    coordinates = fire.Function(
        fire.VectorFunctionSpace(wave.mesh, space.ufl_element())
    ).interpolate(wave.mesh.coordinates).dat.data_ro
    expected = elastic_2d_material(coordinates[:, 0], coordinates[:, 1], 10.0)
    for parameter, values in [
        ("p_wave_velocity", wave.c),
        ("s_wave_velocity", wave.c_s),
        ("density", wave.rho),
    ]:
        # Interface nodes may pick either side of the sampled grid
        mismatch = ~np.isclose(values.dat.data_ro, expected[parameter], atol=1e-3)
        assert mismatch.mean() < 0.05, parameter


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
