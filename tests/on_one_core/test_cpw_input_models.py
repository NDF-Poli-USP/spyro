"""Tests for the cells-per-wavelength calculator model dictionaries."""

from collections.abc import Callable

import numpy as np
import pytest

from spyro.tools.cells_per_wavelength_calculator import input_models
from spyro.tools.cells_per_wavelength_calculator import (
    AcousticHomogeneousProblem,
    CpwParameters,
)

METHODS = ["mass_lumped_triangle", "spectral_quadrilateral"]


def make_homogeneous_model(
    method: str = "spectral_quadrilateral",
    velocity: float = 1.5,
    frequency: float = 5.0,
    reduced: bool = False,
) -> dict:
    """Build a homogeneous acoustic model with typical settings.

    Parameters
    ----------
    method : str, optional
        Finite element method. Default is ``"spectral_quadrilateral"``.
    velocity : float, optional
        Acoustic velocity. Default is 1.5.
    frequency : float, optional
        Source peak frequency. Default is 5.0.
    reduced : bool, optional
        Use the reduced receiver grid. Default is False.

    Returns
    -------
    dict
        The model dictionary.
    """
    return input_models.create_acoustic_model_for_meshing_parameter_2D_homogeneous(
        method=method,
        degree=4,
        frequency=frequency,
        velocity=velocity,
        dt=1e-3,
        reduced=reduced,
    )


#: Problem parameters shared by the tests that build a problem.
PROBLEM_PARAMETERS = {
    "source_frequency": 5.0,
    "mesh_frequency_factor": 2.0,
    "FEM_method_to_evaluate": "mass_lumped_triangle",
    "dimension": 2,
    "testing": True,
    "desired_degree": 2,
    "C_initial": 2.8,
    "C_accuracy": 0.1,
    "accepted_error_threshold": 0.05,
}


def make_parameters(**overrides) -> CpwParameters:
    """Build problem parameters from :data:`PROBLEM_PARAMETERS`.

    Parameters
    ----------
    **overrides
        Dictionary entries replacing the defaults.

    Returns
    -------
    CpwParameters
        The parameters.
    """
    return CpwParameters.from_dict({**PROBLEM_PARAMETERS, **overrides})


@pytest.mark.parametrize(
    "method, expected_mesh_type",
    [
        ("mass_lumped_triangle", "SeismicMesh"),
        ("spectral_quadrilateral", "firedrake_mesh"),
    ],
)
def test_set_mesh_type(method: str, expected_mesh_type: str) -> None:
    """Each supported method maps to its mesh generator."""
    assert input_models.set_mesh_type(method) == expected_mesh_type


def test_set_mesh_type_raises_for_invalid_method() -> None:
    """An unsupported method is rejected."""
    with pytest.raises(ValueError, match="Method is not"):
        input_models.set_mesh_type("unsupported_method")


@pytest.mark.parametrize("method", METHODS)
def test_homogeneous_model_is_periodic_without_pml(method: str) -> None:
    """Both methods use a periodic Firedrake mesh and no absorbing layer."""
    model = make_homogeneous_model(method=method)

    assert model["options"]["method"] == method
    assert model["options"]["degree"] == 4
    assert model["mesh"]["periodic"] is True
    assert model["mesh"]["mesh_type"] == "firedrake_mesh"
    assert model["absorving_boundary_conditions"]["status"] is False


@pytest.mark.parametrize("reduced, expected_receivers", [(True, 4), (False, 36)])
def test_homogeneous_model_receiver_quantity(
    reduced: bool, expected_receivers: int
) -> None:
    """The reduced flag shrinks the receiver grid for testing."""
    model = make_homogeneous_model(reduced=reduced)

    assert len(model["acquisition"]["receiver_locations"]) == expected_receivers


@pytest.mark.parametrize("velocity, frequency", [(1.5, 5.0), (3.0, 12.0)])
def test_homogeneous_model_geometry_avoids_periodic_images(
    velocity: float, frequency: float
) -> None:
    """The direct wave arrives in time, and no periodic image does."""
    model = make_homogeneous_model(velocity=velocity, frequency=frequency)

    length_z = model["mesh"]["length_z"]
    length_x = model["mesh"]["length_x"]
    final_time = model["time_axis"]["final_time"]
    source = np.array(model["acquisition"]["source_locations"][0])
    receivers = np.array(model["acquisition"]["receiver_locations"])

    # Source and receivers lie inside the box
    points = np.vstack([source, receivers])
    assert np.all((points[:, 0] < 0.0) & (points[:, 0] > -length_z))
    assert np.all((points[:, 1] > 0.0) & (points[:, 1] < length_x))

    # Receivers sit below the source, 7.5 to 12.5 wavelengths deep
    wavelength = velocity / frequency
    depths = (source[0] - receivers[:, 0]) / wavelength
    assert np.isclose(depths.min(), 7.5)
    assert np.isclose(depths.max(), 12.5)

    # The wave reaches the farthest receiver before the final time
    distances = np.linalg.norm(receivers - source, axis=1)
    assert distances.max() / velocity < final_time

    # Nearest periodic image of the source, over the 8 neighbouring copies
    shifts = np.array(
        [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]
    ) * np.array([length_z, length_x])
    images = source + shifts
    image_distance = np.linalg.norm(
        receivers[:, None, :] - images[None, :, :], axis=2
    ).min()
    assert image_distance / velocity > final_time


def test_homogeneous_model_warns_for_large_velocity() -> None:
    """A velocity in m/s instead of km/s triggers a warning."""
    with pytest.warns(UserWarning, match="Velocity in meters per second"):
        make_homogeneous_model(velocity=1500.0)


@pytest.mark.parametrize(
    "time, dt, expected",
    [(1.0, 0.25, 1.0), (1.01, 0.25, 1.25), (3.14951, 0.002, 3.15)],
)
def test_round_up_to_time_step(time: float, dt: float, expected: float) -> None:
    """Times are rounded up onto the grid, and grid times stay in place."""
    assert np.isclose(input_models._round_up_to_time_step(time, dt), expected)


@pytest.mark.parametrize("dt", [5e-4, 2e-3, 7e-4])
@pytest.mark.parametrize(
    "build",
    [
        lambda dt: input_models.create_acoustic_model_for_meshing_parameter_2D_homogeneous(
            method="spectral_quadrilateral",
            degree=4,
            frequency=5.0,
            velocity=1.5,
            dt=dt,
        ),
    ],
    ids=["acoustic"],
)
def test_homogeneous_final_time_is_on_reference_grid(
    dt: float, build: Callable[[float], dict],
) -> None:
    """The last of ``round(T / dt) + 1`` reference samples is at ``T``.

    Parameters
    ----------
    dt : float
        Time step of the reference.
    build : callable
        Builds the homogeneous model dictionary of one physics at ``dt``.
    """
    model = build(dt)

    final_time = model["time_axis"]["final_time"]
    num_steps = round(final_time / dt) + 1
    assert np.isclose((num_steps - 1) * dt, final_time, rtol=0.0, atol=1e-12)


def test_acoustic_problem_builds_homogeneous_model() -> None:
    """The reference time step is fixed, whatever the solver time step."""
    problem = AcousticHomogeneousProblem(
        make_parameters(
            **{
                "minimum_velocity_in_the_domain": 1.5,
                "timestep_calculation": "float",
                "timestep": 2e-3,
            }
        )
    )

    model = problem.model_dictionary("mass_lumped_triangle", 3)

    assert model["mesh"]["periodic"] is True
    assert model["acquisition"]["frequency"] == 5.0
    assert model["time_axis"]["dt"] == input_models.HOMOGENEOUS_REFERENCE_DT
    assert len(model["acquisition"]["receiver_locations"]) == 4
