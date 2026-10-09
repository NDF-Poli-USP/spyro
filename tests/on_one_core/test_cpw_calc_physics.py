"""Tests for the physics-generic cells-per-wavelength calculator."""

import math
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

import spyro
from spyro.io.time_io import interpolate_time_series
from spyro.tools.cells_per_wavelength_calculator import (
    CPW_PHYSICS,
    AcousticCpwPhysics,
    IsotropicElasticCpwPhysics,
    MAX_CPW,
    ThresholdCase,
    _parse_threshold_case,
    backward_line_search,
    forward_line_search,
)
from spyro.tools.input_models import (
    create_elastic_model_for_meshing_parameter_2D_homogeneous,
)
from spyro.utils.typing import VelocityProfileType, WaveType


def make_error_function(
    critical_cpw: float,
) -> tuple[Callable[[float], float], list[float]]:
    """Build a monotonically decreasing error that crosses 1 at ``critical_cpw``.

    Parameters
    ----------
    critical_cpw : float
        Cells-per-wavelength where the error equals the threshold of 1.

    Returns
    -------
    calculate_error : Callable[[float], float]
        Error as a function of the cells-per-wavelength.
    calls : list of float
        Every value ``calculate_error`` was called with, in order.
    """
    calls = []

    def calculate_error(cpw: float) -> float:
        calls.append(cpw)
        return (critical_cpw / cpw) ** 2

    return calculate_error, calls


@pytest.mark.parametrize(
    "starting_cpw, critical_cpw, expected",
    [
        (6.0, 3.14, 3.2),
        (6.0, 3.2, 3.2),
        (3.25, 2.0, 2.0),
        (2.35, 2.31, 2.35),
        (2.35, 2.25, 2.3),
    ],
)
def test_backward_line_search_finds_smallest_passing_cpw(
    starting_cpw: float, critical_cpw: float, expected: float,
) -> None:
    """The search returns the smallest grid value at or above the crossing."""
    calculate_error, calls = make_error_function(critical_cpw)

    cpw = backward_line_search(
        calculate_error, starting_cpw, 1.0, 0.1, max_cpw=10.0
    )

    assert np.isclose(cpw, expected)
    assert len(calls) == len(set(calls)), "A cpw value was evaluated twice"


def test_backward_line_search_coarse_phase_saves_evaluations() -> None:
    """Coarse steps use far fewer solves than stepping by the accuracy."""
    calculate_error, calls = make_error_function(3.0)

    backward_line_search(calculate_error, 10.0, 1.0, 0.1, max_cpw=10.0)

    assert len(calls) < 25


def test_backward_line_search_rejects_failing_start() -> None:
    """A starting value above the threshold is an input error."""
    calculate_error, _ = make_error_function(3.0)

    with pytest.raises(ValueError, match="Start from a larger value"):
        backward_line_search(calculate_error, 2.0, 1.0, 0.1)


@pytest.mark.parametrize(
    "starting_cpw, critical_cpw, expected",
    [
        (2.0, 3.14, 3.2),
        (2.0, 3.2, 3.2),
        (1.0, 2.0, 2.0),
        (2.25, 2.31, 2.4),
        (2.25, 2.29, 2.3),
        (1.0, 4.0, 4.0),
        (4.0, 4.97, 5.0),
    ],
)
def test_forward_line_search_finds_smallest_passing_cpw(
    starting_cpw: float, critical_cpw: float, expected: float,
) -> None:
    """The search returns the smallest grid value at or above the crossing."""
    calculate_error, calls = make_error_function(critical_cpw)

    cpw = forward_line_search(calculate_error, starting_cpw, 1.0, 0.1)

    assert np.isclose(cpw, expected)
    assert len(calls) == len(set(calls)), "A cpw value was evaluated twice"


@pytest.mark.parametrize("starting_cpw", [1.0, 2.0, 2.55])
def test_forward_and_backward_line_searches_agree(starting_cpw: float) -> None:
    """Both directions find the same minimum cells-per-wavelength."""
    critical_cpw = 2.57
    calculate_error, _ = make_error_function(critical_cpw)

    forward = forward_line_search(calculate_error, starting_cpw, 1.0, 0.1)
    backward = backward_line_search(
        calculate_error, 10.0, 1.0, 0.1, max_cpw=10.0
    )

    assert np.isclose(forward, backward)


def test_forward_line_search_coarse_phase_saves_evaluations() -> None:
    """Coarse steps use far fewer solves than stepping by the accuracy."""
    calculate_error, calls = make_error_function(4.5)

    forward_line_search(calculate_error, 1.0, 1.0, 0.1)

    assert len(calls) < 25


def test_forward_line_search_rejects_passing_start() -> None:
    """A starting value within the threshold is an input error."""
    calculate_error, _ = make_error_function(3.0)

    with pytest.raises(ValueError, match="Start from a smaller value"):
        forward_line_search(calculate_error, 4.0, 1.0, 0.1)


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


def test_cpw_physics_registry() -> None:
    """Every registered physics is keyed by the WaveType of its solver."""
    assert CPW_PHYSICS[WaveType.ISOTROPIC_ACOUSTIC] is AcousticCpwPhysics
    assert CPW_PHYSICS[WaveType.ISOTROPIC_ELASTIC] is IsotropicElasticCpwPhysics
    assert IsotropicElasticCpwPhysics({"s_wave_velocity": 2.75}).wavelength_velocity == 2.75


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
    shifts = np.array(
        [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]
    ) * length
    images = source + shifts
    image_distance = np.linalg.norm(
        receivers[:, None, :] - images[None, :, :], axis=2,
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
        2, method="ARNOLDI", calc_max_dt=True,
    )
    default_forms_dt = dt_solver.estimate_timestep(
        acoustic.c, acoustic.function_space, acoustic.final_time,
        quad_rule=acoustic.quadrature_rule, fraction=1.0,
    )
    assert np.isclose(acoustic_dts[1], default_forms_dt)


@pytest.mark.slow
def test_elastic_cpw_search_meets_threshold() -> None:
    """The elastic search result passes, and one step coarser fails."""
    parameters = {
        "wave_type": WaveType.ISOTROPIC_ELASTIC,
        "source_frequency": 10.0,
        "p_wave_velocity": 5.6,
        "s_wave_velocity": 2.75,
        "density": 2.0,
        "velocity_profile_type": "homogeneous",
        "velocity_model_file_name": None,
        "FEM_method_to_evaluate": "mass_lumped_triangle",
        "dimension": 2,
        "receiver_setup": "line",
        "testing": True,
        "time-step_calculation": "exact",
        "reference_degree": None,
        "C_reference": None,
        "desired_degree": 2,
        "C_initial": 8.0,
        "C_max": 8.0,
        "accepted_error_threshold": 0.05,
        "C_accuracy": 0.5,
    }
    calculator = spyro.tools.MeshingParameterCalculator(parameters)
    assert calculator.reference_solution.shape[1:] == (3, 2)

    cpw = calculator.find_minimum()

    errors = {e.cpw: e.error for e in calculator.search_history}
    assert errors[cpw] <= 0.05
    assert errors[cpw - 0.5] > 0.05


def test_forward_line_search_stops_at_max_cpw() -> None:
    """A value above the cap is never evaluated, and failing it raises."""
    calculate_error, calls = make_error_function(6.0)

    with pytest.raises(ValueError, match="Maximum cells-per-wavelength"):
        forward_line_search(calculate_error, 1.0, 1.0, 0.1, max_cpw=5.0)

    assert max(calls) == 5.0


def test_forward_line_search_rejects_start_above_max_cpw() -> None:
    """A starting value at or above the cap is an input error."""
    calculate_error, _ = make_error_function(6.0)

    with pytest.raises(ValueError, match="not below the maximum"):
        forward_line_search(calculate_error, 5.0, 1.0, 0.1, max_cpw=5.0)


@pytest.mark.parametrize("line_search", [backward_line_search, forward_line_search])
@pytest.mark.parametrize(
    "starting_cpw, accuracy, coarse_step_fraction, name",
    [
        (0.0, 0.1, 0.1, "starting_cpw"),
        (2.0, 0.0, 0.1, "accuracy"),
        (2.0, 0.1, -0.1, "coarse_step_fraction"),
    ],
)
def test_line_searches_reject_non_positive_settings(
    line_search: Callable[..., float],
    starting_cpw: float,
    accuracy: float,
    coarse_step_fraction: float,
    name: str,
) -> None:
    """Non-positive settings are rejected before any error is evaluated."""
    calculate_error, calls = make_error_function(3.0)

    with pytest.raises(ValueError, match=f"{name} must be positive"):
        line_search(
            calculate_error, starting_cpw, 1.0, accuracy,
            coarse_step_fraction=coarse_step_fraction,
        )

    assert calls == []


def test_backward_line_search_rejects_start_above_max_cpw() -> None:
    """A starting value above the cap is rejected before any evaluation."""
    calculate_error, calls = make_error_function(3.0)

    with pytest.raises(ValueError, match="above the maximum"):
        backward_line_search(calculate_error, 6.0, 1.0, 0.1, max_cpw=5.0)

    assert calls == []


def test_backward_line_search_accepts_start_at_max_cpw() -> None:
    """The cap itself is a valid starting value."""
    calculate_error, _ = make_error_function(3.0)

    cpw = backward_line_search(calculate_error, 5.0, 1.0, 0.1, max_cpw=5.0)

    assert np.isclose(cpw, 3.0)


@pytest.mark.parametrize("critical_cpw", np.linspace(1.03, 4.97, 41))
def test_line_searches_match_exhaustive_search(critical_cpw: float) -> None:
    """Both searches return the smallest passing value an exhaustive scan finds."""
    starting_cpw = 0.95
    grid = [round(0.1 * i, 10) for i in range(10, 51)]
    calculate_error, _ = make_error_function(critical_cpw)
    expected = min(cpw for cpw in grid if calculate_error(cpw) <= 1.0)

    forward = forward_line_search(calculate_error, starting_cpw, 1.0, 0.1)
    backward = backward_line_search(calculate_error, 5.0, 1.0, 0.1)

    assert np.isclose(forward, expected)
    assert np.isclose(backward, expected)


def test_fine_phase_bisects() -> None:
    """The fine phase needs a logarithmic number of solves in its bracket."""
    calculate_error, calls = make_error_function(1.01)

    cpw = backward_line_search(
        calculate_error, 5.0, 1.0, 0.01, coarse_step_fraction=0.5
    )

    assert np.isclose(cpw, 1.01)
    # Coarse phase: 5.0, 2.5, 1.25, 0.62; then bisecting 62 grid values
    assert len(calls) <= 4 + math.ceil(math.log2(62))


def test_backward_line_search_uses_default_max_cpw() -> None:
    """Without an explicit cap, a start above MAX_CPW is rejected."""
    calculate_error, calls = make_error_function(3.0)

    with pytest.raises(ValueError, match="above the maximum"):
        backward_line_search(calculate_error, MAX_CPW + 1.0, 1.0, 0.1)

    assert calls == []


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (False, None),
        (True, ThresholdCase("spectral_quadrilateral", 4, 2.0)),
        ({"cpw": 3.0}, ThresholdCase("spectral_quadrilateral", 4, 3.0)),
        (
            ThresholdCase("mass_lumped_triangle", 2, 5.0),
            ThresholdCase("mass_lumped_triangle", 2, 5.0),
        ),
    ],
)
def test_parse_threshold_case(
    value: bool | dict | ThresholdCase | None, expected: ThresholdCase | None,
) -> None:
    """The threshold case parameter defaults to P4 quadrilaterals at cpw 2."""
    assert _parse_threshold_case(value) == expected


@pytest.mark.parametrize("value", [{"order": 4}, 4.0])
def test_parse_threshold_case_rejects_bad_values(value: dict | float) -> None:
    """Unknown keys and unsupported types are rejected."""
    with pytest.raises(TypeError):
        _parse_threshold_case(value)


@pytest.mark.parametrize(
    "threshold_settings",
    [{}, {"accepted_error_threshold": 0.05, "threshold_case": True}],
)
def test_calculator_needs_exactly_one_threshold(threshold_settings: dict) -> None:
    """A threshold is given directly or through a threshold case, not both."""
    parameters = {
        "source_frequency": 5.0,
        "minimum_velocity_in_the_domain": 1.5,
        "velocity_profile_type": "homogeneous",
        "velocity_model_file_name": None,
        "FEM_method_to_evaluate": "mass_lumped_triangle",
        "dimension": 2,
        "receiver_setup": "near",
        **threshold_settings,
    }

    with pytest.raises(ValueError, match="exactly one"):
        spyro.tools.MeshingParameterCalculator(parameters)


def make_calculator_stub(
    wave_type: WaveType, method: str, degree: int,
) -> SimpleNamespace:
    """Build the calculator attributes the physics dictionary builders read.

    Parameters
    ----------
    wave_type : WaveType
        Physics whose dictionary is built.
    method : str
        Method under evaluation by the calculator.
    degree : int
        Degree under evaluation by the calculator.

    Returns
    -------
    SimpleNamespace
        Stand-in for a :class:`MeshingParameterCalculator`.
    """
    velocity = 1.5 if wave_type is WaveType.ISOTROPIC_ACOUSTIC else 2.75
    return SimpleNamespace(
        dimension=2,
        velocity_profile_type=VelocityProfileType.HOMOGENEOUS,
        minimum_velocity=velocity,
        source_frequency=5.0,
        cpw_initial=3.0,
        FEM_method_to_evaluate=method,
        desired_degree=degree,
        reduced_obj_for_testing=True,
        fixed_timestep=None,
    )


@pytest.mark.parametrize(
    "wave_type, parameters",
    [
        (WaveType.ISOTROPIC_ACOUSTIC, {"minimum_velocity_in_the_domain": 1.5}),
        (
            WaveType.ISOTROPIC_ELASTIC,
            {"p_wave_velocity": 5.6, "s_wave_velocity": 2.75, "density": 2.0},
        ),
    ],
)
def test_physics_builds_dictionary_for_requested_discretization(
    wave_type: WaveType, parameters: dict,
) -> None:
    """The threshold case can use a method and degree the search does not."""
    physics = CPW_PHYSICS[wave_type](parameters)
    calculator = make_calculator_stub(wave_type, "mass_lumped_triangle", 2)

    dictionary = physics.build_dictionary(calculator, "spectral_quadrilateral", 4)

    assert dictionary["options"]["method"] == "spectral_quadrilateral"
    assert dictionary["options"]["degree"] == 4
    assert dictionary["mesh"]["mesh_type"] == "firedrake_mesh"


def test_calculator_rejects_non_positive_mesh_frequency_factor() -> None:
    """The meshing frequency must be a positive multiple of the peak."""
    parameters = {
        "source_frequency": 5.0,
        "minimum_velocity_in_the_domain": 1.5,
        "velocity_profile_type": "homogeneous",
        "velocity_model_file_name": None,
        "mesh_frequency_factor": 0.0,
    }

    with pytest.raises(ValueError, match="mesh_frequency_factor must be positive"):
        spyro.tools.MeshingParameterCalculator(parameters)


@pytest.mark.parametrize("factor", [1.0, 2.0, 2.5])
@pytest.mark.parametrize(
    "profile", [VelocityProfileType.HOMOGENEOUS, VelocityProfileType.HETEROGENEOUS]
)
def test_mesh_frequency_factor_scales_mesh_size(
    factor: float, profile: VelocityProfileType,
) -> None:
    """A cpw refers to the wavelength at ``factor`` times the peak frequency."""
    velocity, frequency, cpw = 1.5, 5.0, 3.0
    requested = {}

    class RecordingWave:
        """Records the mesh request instead of building a mesh."""

        def __init__(self, dictionary: dict) -> None:
            """Accept and ignore the model dictionary.

            Parameters
            ----------
            dictionary : dict
                Spyro model dictionary.
            """

        def set_mesh(self, input_mesh_parameters: dict) -> None:
            """Record the mesh parameters.

            Parameters
            ----------
            input_mesh_parameters : dict
                Mesh parameters passed by the calculator.
            """
            requested.update(input_mesh_parameters)

    def record_edge_length(wave: RecordingWave, edge_length: float) -> None:
        """Record the edge length of a homogeneous mesh.

        Parameters
        ----------
        wave : RecordingWave
            Wave object being set up.
        edge_length : float
            Mesh edge length requested by the calculator.
        """
        requested["edge_length"] = edge_length

    calculator = SimpleNamespace(
        initial_dictionary={},
        velocity_profile_type=profile,
        minimum_velocity=velocity,
        mesh_frequency_factor=factor,
        mesh_frequency=factor * frequency,
        physics=SimpleNamespace(
            wave_class=RecordingWave, setup_homogeneous_wave=record_edge_length
        ),
    )

    spyro.tools.MeshingParameterCalculator.build_current_object(calculator, cpw)

    # The mesher turns cells_per_wavelength into velocity / (frequency * cpw)
    if profile is VelocityProfileType.HOMOGENEOUS:
        edge_length = requested["edge_length"]
    else:
        edge_length = velocity / (frequency * requested["cells_per_wavelength"])
    assert np.isclose(edge_length, velocity / (factor * frequency * cpw))
