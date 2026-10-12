"""Tests for the base classes of the cells-per-wavelength heterogeneous cases."""

from dataclasses import dataclass

import h5py
import numpy as np
import pytest

import spyro
from spyro.io.segy_io import create_segy_from_grid, read_segy_velocity_model
from spyro.tools.cells_per_wavelength_calculator import (
    CPW_PROBLEMS,
    CpwParameters,
    HeterogeneousCase,
    HeterogeneousCpwProblem,
    MaterialFileCase,
    MaterialModelCase,
)
from spyro.utils.typing import VelocityProfileType, WaveType


@dataclass(frozen=True, kw_only=True)
class LinearCase(MaterialModelCase):
    """Material increasing with depth and along x, on a coarse 2 by 3 grid."""

    name: str = "linear"
    length_z: float = 2.0
    length_x: float = 3.0
    grid_spacing: float = 0.5

    def material(self, z: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate the linear material.

        Parameters
        ----------
        z : np.ndarray
            Vertical coordinates, negative below the surface.
        x : np.ndarray
            Horizontal coordinates.

        Returns
        -------
        dict
            P-wave velocity, S-wave velocity and density arrays.
        """
        z, x = np.broadcast_arrays(z, x)
        return {
            "p_wave_velocity": 2.0 - 0.5 * z + 0.1 * x,
            "s_wave_velocity": 1.0 - 0.25 * z + 0.05 * x,
            "density": np.full(z.shape, 2.0),
        }


@dataclass(frozen=True, kw_only=True)
class AcousticOnlyCase(LinearCase):
    """Linear case defining only a P-wave velocity."""

    def material(self, z: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate the P-wave velocity of the linear material.

        Parameters
        ----------
        z : np.ndarray
            Vertical coordinates, negative below the surface.
        x : np.ndarray
            Horizontal coordinates.

        Returns
        -------
        dict
            P-wave velocity array.
        """
        return {"p_wave_velocity": super().material(z, x)["p_wave_velocity"]}


@pytest.mark.parametrize("overrides", [{"length_z": 0.0}, {"length_x": -1.0}])
def test_case_rejects_non_positive_lengths(overrides: dict) -> None:
    """The domain lengths must be positive."""
    with pytest.raises(ValueError, match="lengths must be positive"):
        LinearCase(**overrides)


@pytest.mark.parametrize("case_class", [HeterogeneousCase, MaterialModelCase])
def test_case_bases_are_abstract(case_class: type) -> None:
    """Only concrete cases can be created."""
    with pytest.raises(TypeError, match="abstract"):
        case_class(name="base", length_z=1.0, length_x=1.0)


def test_case_fields_are_keyword_only() -> None:
    """Cases are created by field name, so subclasses can add fields freely."""
    with pytest.raises(TypeError):
        MaterialFileCase("vp", 1.0, 1.0, {"p_wave_velocity": "vp.segy"})


def test_case_writes_requested_material_files(tmp_path) -> None:
    """Each requested parameter is sampled on the grid and written."""
    case = LinearCase(name=str(tmp_path / "linear"))

    file_names = case.write_material_files(("p_wave_velocity", "density"))

    assert file_names == {
        "p_wave_velocity": f"{tmp_path}/linear_vp.segy",
        "density": f"{tmp_path}/linear_rho.segy",
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "linear_rho.segy",
        "linear_vp.segy",
    ]
    z = np.linspace(-2.0, 0.0, 5)[:, np.newaxis]
    x = np.linspace(0.0, 3.0, 7)[np.newaxis, :]
    values, nz, nx = read_segy_velocity_model(file_names["p_wave_velocity"])
    assert (nz, nx) == (5, 7)
    assert np.allclose(values, case.material(z, x)["p_wave_velocity"])


def test_case_returns_given_material_files() -> None:
    """Given files are used as they are, without writing anything."""
    files = {"p_wave_velocity": "vp.segy", "density": "rho.hdf5"}
    case = MaterialFileCase(
        name="files", length_z=1.0, length_x=1.0, material_files=files
    )

    assert case.write_material_files(("p_wave_velocity",)) == {
        "p_wave_velocity": "vp.segy"
    }


@pytest.mark.parametrize("material", ["model", "files"])
def test_case_rejects_missing_parameter(material: str, tmp_path) -> None:
    """An elastic material needs more than a P-wave velocity."""
    if material == "model":
        case = AcousticOnlyCase(name=str(tmp_path / "vp_only"))
    else:
        case = MaterialFileCase(
            name="vp_only",
            length_z=1.0,
            length_x=1.0,
            material_files={"p_wave_velocity": "vp.segy"},
        )

    with pytest.raises(ValueError, match="does not define"):
        case.write_material_files(("p_wave_velocity", "s_wave_velocity"))


def test_case_minimum_of_material_model() -> None:
    """The minimum is taken over the sampled grid."""
    case = LinearCase()

    assert case.minimum("p_wave_velocity") == pytest.approx(2.0)
    assert case.minimum("s_wave_velocity") == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [1.0, 1000.0], ids=["km/s", "m/s"])
@pytest.mark.parametrize("extension", [".segy", ".hdf5"])
def test_case_minimum_of_material_file(
    scale: float, extension: str, tmp_path,
) -> None:
    """A file minimum is read in km/s, whatever unit the file is in."""
    velocity = scale * np.array([[1.8, 2.4], [3.0, 3.6]])
    file_name = str(tmp_path / f"vp{extension}")
    if extension == ".segy":
        create_segy_from_grid(velocity, file_name, rotate=True)
    else:
        with h5py.File(file_name, "w") as f:
            f.create_dataset("velocity_model", data=velocity)
    case = MaterialFileCase(
        name="vp",
        length_z=1.0,
        length_x=1.0,
        material_files={"p_wave_velocity": file_name},
    )

    assert case.minimum("p_wave_velocity") == pytest.approx(1.8)


def make_heterogeneous_problem(
    wave_type: WaveType, case: HeterogeneousCase,
) -> HeterogeneousCpwProblem:
    """Build the heterogeneous problem of a wave type on a case.

    Parameters
    ----------
    wave_type : WaveType
        Wave physics of the problem.
    case : HeterogeneousCase
        The heterogeneous case.

    Returns
    -------
    HeterogeneousCpwProblem
        The problem.
    """
    problem_class = CPW_PROBLEMS[(wave_type, VelocityProfileType.HETEROGENEOUS)]
    return problem_class(
        CpwParameters(
            source_frequency=5.0,
            mesh_frequency_factor=2.0,
            fem_method="mass_lumped_triangle",
            dimension=2,
            desired_degree=2,
            cpw_initial=3.0,
            cpw_accuracy=0.1,
            accepted_error_threshold=0.05,
            wave_type=wave_type,
            heterogeneous_case=case,
            reference_degree=5,
            cpw_reference=6.0,
        )
    )


@pytest.mark.parametrize(
    "wave_type, expected",
    [(WaveType.ISOTROPIC_ACOUSTIC, 2.0), (WaveType.ISOTROPIC_ELASTIC, 1.0)],
)
def test_problem_minimum_wavelength_velocity_of_case(
    wave_type: WaveType, expected: float,
) -> None:
    """The wavelength is the minimum P- or S-wave velocity of the case."""
    problem = make_heterogeneous_problem(wave_type, LinearCase())

    assert problem.minimum_wavelength_velocity == pytest.approx(expected)


def test_problem_minimum_wavelength_velocity_override() -> None:
    """A case minimum velocity replaces the material minimum."""
    case = LinearCase(minimum_velocity=0.8)
    problem = make_heterogeneous_problem(WaveType.ISOTROPIC_ELASTIC, case)

    assert problem.minimum_wavelength_velocity == 0.8


@pytest.mark.parametrize(
    "profile, give_case",
    [
        (VelocityProfileType.HETEROGENEOUS, False),
        (VelocityProfileType.HOMOGENEOUS, True),
    ],
)
def test_calculator_matches_case_to_profile(
    profile: VelocityProfileType, give_case: bool,
) -> None:
    """Only heterogeneous models have a heterogeneous case."""
    parameters = {
        "wave_type": WaveType.ISOTROPIC_ELASTIC,
        "source_frequency": 5.0,
        "mesh_frequency_factor": 2.0,
        "velocity_profile_type": profile,
        "FEM_method_to_evaluate": "mass_lumped_triangle",
        "dimension": 2,
        "desired_degree": 2,
        "C_initial": 3.0,
        "C_accuracy": 0.1,
        "accepted_error_threshold": 0.05,
    }
    if give_case:
        parameters["heterogeneous_case"] = LinearCase()

    with pytest.raises(ValueError, match="heterogeneous_case should be given"):
        spyro.tools.MeshingParameterCalculator(parameters)


def test_case_writes_material_files_before_ensemble_exists(
    monkeypatch, tmp_path,
) -> None:
    """The first dictionary is built before any wave creates the ensemble."""
    monkeypatch.setattr(spyro.mpi.spyro_mpi.SpyroEnsemble, "ensemble", None)
    case = LinearCase(name=str(tmp_path / "linear"))

    file_names = case.write_material_files(("s_wave_velocity",))

    assert file_names == {"s_wave_velocity": f"{tmp_path}/linear_vs.segy"}
    assert (tmp_path / "linear_vs.segy").is_file()


def test_case_missing_parameter_writes_nothing(tmp_path) -> None:
    """A missing parameter is raised before any file is written."""
    case = AcousticOnlyCase(name=str(tmp_path / "vp_only"))

    with pytest.raises(ValueError, match="does not define"):
        case.write_material_files(("p_wave_velocity", "density"))
    assert not any(tmp_path.iterdir())


def test_case_samples_material_once(monkeypatch) -> None:
    """Every use of the sampled material shares one evaluation."""
    case = LinearCase()
    calls = []
    material = LinearCase.material

    def counting_material(
        self: LinearCase, z: np.ndarray, x: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Count the evaluations of the linear material.

        Parameters
        ----------
        self : LinearCase
            Case being sampled.
        z : np.ndarray
            Vertical coordinates.
        x : np.ndarray
            Horizontal coordinates.

        Returns
        -------
        dict
            The linear material.
        """
        calls.append(None)
        return material(self, z, x)

    monkeypatch.setattr(LinearCase, "material", counting_material)

    case.minimum("p_wave_velocity")
    case.minimum("s_wave_velocity")

    assert len(calls) == 1


def test_case_writes_each_material_file_once(tmp_path) -> None:
    """Files already written are not written again; new parameters are."""
    case = LinearCase(name=str(tmp_path / "linear"))

    case.write_material_files(("p_wave_velocity",))
    vp_file = tmp_path / "linear_vp.segy"
    vp_file.write_bytes(b"sentinel")

    file_names = case.write_material_files(("p_wave_velocity", "density"))

    assert vp_file.read_bytes() == b"sentinel"
    assert (tmp_path / "linear_rho.segy").is_file()
    assert file_names["p_wave_velocity"] == str(vp_file)
    assert case.written_parameters == {"p_wave_velocity", "density"}


def test_written_parameters_do_not_affect_equality(tmp_path) -> None:
    """Two cases with the same fields are equal, whatever they wrote."""
    case = LinearCase(name=str(tmp_path / "linear"))
    case.write_material_files(("density",))

    assert case == LinearCase(name=str(tmp_path / "linear"))
