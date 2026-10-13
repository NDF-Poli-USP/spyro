"""Tests for the cells-per-wavelength calculator model dictionaries."""

import numpy as np
import pytest

from spyro.tools.cells_per_wavelength_calculator import input_models
from spyro.tools.cells_per_wavelength_calculator import (
    AcousticHeterogeneousProblem,
    AcousticHomogeneousProblem,
    CpwParameters,
    ElasticHeterogeneousProblem,
    ElasticSyntheticCase,
    MaterialFileCase,
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
    "reference_degree": 5,
    "C_reference": 6.0,
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
    "method, expected_mesh_type",
    [
        ("mass_lumped_triangle", "SeismicMesh"),
        ("spectral_quadrilateral", "firedrake_mesh"),
    ],
)
def test_heterogeneous_model_sets_core_fields(
    method: str, expected_mesh_type: str
) -> None:
    """The heterogeneous model has a one-wavelength PML and a receiver line."""
    model = input_models.create_acoustic_model_for_meshing_parameter_2D_heterogeneous(
        method=method,
        degree=5,
        frequency=8.0,
        minimum_velocity=1.6,
        case=MaterialFileCase(
            name="my_model",
            length_z=12.0,
            length_x=30.0,
            material_files={"p_wave_velocity": "my_model.segy"},
        ),
    )

    assert model["options"]["method"] == method
    assert model["options"]["degree"] == 5
    assert model["mesh"]["mesh_type"] == expected_mesh_type
    assert model["mesh"]["periodic"] is False
    assert model["mesh"]["length_z"] == 12.0
    assert model["mesh"]["length_x"] == 30.0
    assert "cells_per_wavelength" not in model["mesh"]
    abc = model["absorving_boundary_conditions"]
    assert abc["status"] is True
    assert abc["abc_type"] == "PML"
    assert abc["pad_length"] == pytest.approx(0.2)
    assert model["synthetic_data"]["real_velocity_file"] == "my_model.segy"

    receivers = model["acquisition"]["receiver_locations"]
    assert len(receivers) == 500
    assert tuple(receivers[0]) == pytest.approx((-0.3, 5.0))
    assert tuple(receivers[-1]) == pytest.approx((-0.3, 13.0))
    assert model["acquisition"]["source_locations"] == [(-0.3, 3.0)]
    assert model["time_axis"]["final_time"] == 7.5
    assert model["time_axis"]["dt"] == 1e-4


@pytest.mark.parametrize(
    "build",
    [
        lambda case: input_models.create_acoustic_model_for_meshing_parameter_2D_heterogeneous(
            "spectral_quadrilateral", 4, 5.0, 1.5, case
        ),
        lambda case: input_models.create_elastic_model_for_meshing_parameter_2D_heterogeneous(
            "spectral_quadrilateral", 4, 5.0, case
        ),
    ],
    ids=["acoustic", "elastic"],
)
def test_heterogeneous_case_overrides_acquisition_and_time_axis(build) -> None:
    """Each acquisition or time axis value given by the case is used."""
    case = MaterialFileCase(
        name="custom",
        length_z=2.0,
        length_x=4.0,
        material_files={
            "p_wave_velocity": "custom_vp.segy",
            "s_wave_velocity": "custom_vs.segy",
            "density": "custom_rho.segy",
        },
        source_locations=[(-0.1, 1.0)],
        receiver_locations=[(-1.0, 2.0), (-1.0, 3.0)],
        final_time=0.8,
        dt=2e-4,
    )

    model = build(case)

    assert model["mesh"]["length_z"] == 2.0
    assert model["mesh"]["length_x"] == 4.0
    assert model["acquisition"]["source_locations"] == [(-0.1, 1.0)]
    assert model["acquisition"]["receiver_locations"] == [(-1.0, 2.0), (-1.0, 3.0)]
    assert model["time_axis"]["final_time"] == 0.8
    assert model["time_axis"]["dt"] == 2e-4


def test_periodic_model_rejects_pml() -> None:
    """A periodic box has no boundary for a PML."""
    with pytest.raises(ValueError, match="periodic mesh cannot have a PML"):
        input_models._build_acoustic_model_dictionary(
            method="spectral_quadrilateral",
            degree=2,
            length_z=1.0,
            length_x=1.0,
            source_locations=[(-0.5, 0.5)],
            receiver_locations=[(-0.6, 0.5)],
            frequency=5.0,
            final_time=1.0,
            dt=1e-3,
            pml_pad_length=0.3,
            periodic=True,
        )


@pytest.mark.parametrize(
    "time, dt, expected",
    [(1.0, 0.25, 1.0), (1.01, 0.25, 1.25), (3.14951, 0.002, 3.15)],
)
def test_round_up_to_time_step(time: float, dt: float, expected: float) -> None:
    """Times are rounded up onto the grid, and grid times stay in place."""
    assert np.isclose(input_models._round_up_to_time_step(time, dt), expected)


@pytest.mark.parametrize("dt", [5e-4, 2e-3, 7e-4])
@pytest.mark.parametrize("wave", ["acoustic", "elastic"])
def test_homogeneous_final_time_is_on_reference_grid(dt: float, wave: str) -> None:
    """The last of ``round(T / dt) + 1`` reference samples is at ``T``."""
    if wave == "acoustic":
        model = input_models.create_acoustic_model_for_meshing_parameter_2D_homogeneous(
            method="spectral_quadrilateral",
            degree=4,
            frequency=5.0,
            velocity=1.5,
            dt=dt,
        )
    else:
        model = input_models.create_elastic_model_for_meshing_parameter_2D_homogeneous(
            method="spectral_quadrilateral",
            degree=4,
            frequency=5.0,
            p_wave_velocity=5.6,
            s_wave_velocity=2.75,
            density=2.0,
            dt=dt,
        )

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


def test_acoustic_problem_builds_heterogeneous_model() -> None:
    """The heterogeneous model reads its lengths and file from the inputs."""
    case = MaterialFileCase(
        name="velocity",
        length_z=3.0,
        length_x=17.0,
        material_files={"p_wave_velocity": "velocity.segy"},
        minimum_velocity=1.5,
    )
    problem = AcousticHeterogeneousProblem(make_parameters(heterogeneous_case=case))

    model = problem.model_dictionary("spectral_quadrilateral", 4)

    assert model["mesh"]["length_z"] == 3.0
    assert model["mesh"]["length_x"] == 17.0
    assert model["synthetic_data"]["real_velocity_file"] == "velocity.segy"


@pytest.mark.parametrize("method", METHODS)
def test_elastic_heterogeneous_model_sets_core_fields(method: str, tmp_path) -> None:
    """The model has Stacey boundaries, no layer, and a deep receiver line."""
    model = input_models.create_elastic_model_for_meshing_parameter_2D_heterogeneous(
        method=method,
        degree=4,
        frequency=10.0,
        case=ElasticSyntheticCase(name=str(tmp_path / "salt")),
    )

    assert model["options"]["method"] == method
    assert model["options"]["degree"] == 4
    assert model["mesh"]["mesh_type"] == "firedrake_mesh"
    assert model["mesh"]["periodic"] is False
    assert model["mesh"]["length_z"] == 5.0
    assert model["mesh"]["length_x"] == 10.0
    abc = model["absorving_boundary_conditions"]
    assert abc["status"] is True
    assert abc["abc_type"] == "nrbc"
    assert abc["nrbc"] == {"type": "Stacey", "dt_scheme": "backward"}

    acquisition = model["acquisition"]
    assert acquisition["source_locations"] == [(-0.5, 5.0)]
    assert acquisition["delay"] == pytest.approx(0.1)
    assert np.array_equal(acquisition["amplitude"], [0.0, 1.0])
    receivers = np.asarray(acquisition["receiver_locations"])
    assert receivers.shape == (40, 2)
    assert np.allclose(receivers[:, 0], -4.0)
    assert np.allclose(receivers[:, 1], np.arange(3.0, 7.0, 0.1))

    assert model["time_axis"]["final_time"] == 2.0
    assert model["time_axis"]["dt"] == 1e-4


def test_elastic_heterogeneous_model_writes_material_files(tmp_path) -> None:
    """The material SEG-Y files exist once the dictionary is returned."""
    base_name = str(tmp_path / "salt")
    model = input_models.create_elastic_model_for_meshing_parameter_2D_heterogeneous(
        method="spectral_quadrilateral",
        degree=4,
        frequency=10.0,
        case=ElasticSyntheticCase(name=base_name),
    )

    assert model["synthetic_data"] == {
        "type": "file",
        "p_wave_velocity": f"{base_name}_vp.segy",
        "s_wave_velocity": f"{base_name}_vs.segy",
        "density": f"{base_name}_rho.segy",
    }
    for parameter in ("p_wave_velocity", "s_wave_velocity", "density"):
        assert (tmp_path / model["synthetic_data"][parameter]).is_file()


def test_acoustic_heterogeneous_model_writes_only_p_wave_file(tmp_path) -> None:
    """An acoustic model of an elastic material writes its P-wave velocity."""
    base_name = str(tmp_path / "salt")
    model = input_models.create_acoustic_model_for_meshing_parameter_2D_heterogeneous(
        method="spectral_quadrilateral",
        degree=4,
        frequency=10.0,
        minimum_velocity=2.5,
        case=ElasticSyntheticCase(name=base_name),
    )

    assert model["synthetic_data"]["real_velocity_file"] == f"{base_name}_vp.segy"
    assert [path.name for path in tmp_path.iterdir()] == ["salt_vp.segy"]


def test_elastic_heterogeneous_model_rejects_invalid_method(tmp_path) -> None:
    """Only the two supported methods are accepted, before writing files."""
    with pytest.raises(ValueError, match="Method is not"):
        input_models.create_elastic_model_for_meshing_parameter_2D_heterogeneous(
            method="CG",
            degree=4,
            frequency=10.0,
            case=ElasticSyntheticCase(name=str(tmp_path / "salt")),
        )
    assert not any(tmp_path.iterdir())


def test_elastic_problem_builds_heterogeneous_model(tmp_path) -> None:
    """The heterogeneous elastic model reads its domain and files from the case."""
    base_name = str(tmp_path / "salt")
    problem = ElasticHeterogeneousProblem(
        make_parameters(
            source_frequency=10.0,
            heterogeneous_case=ElasticSyntheticCase(name=base_name, length_z=3.0),
        )
    )

    model = problem.model_dictionary("mass_lumped_triangle", 3)

    assert model["options"]["method"] == "mass_lumped_triangle"
    assert model["options"]["degree"] == 3
    assert model["mesh"]["length_z"] == 3.0
    assert model["mesh"]["length_x"] == 10.0
    assert model["acquisition"]["frequency"] == 10.0
    assert model["synthetic_data"]["s_wave_velocity"] == f"{base_name}_vs.segy"
