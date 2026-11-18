"""Velocity adapted meshes of the acoustic heterogeneous cpw problem."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from spyro.meshing.seismic_mesh_based_wrappers import SeismicMesh
from spyro.tools.cells_per_wavelength_calculator import (
    AcousticHeterogeneousProblem,
    CpwParameters,
    MaterialModelCase,
)
from spyro.utils.typing import VelocityProfileType, WaveType

#: P-wave velocities of the top and bottom layers of :class:`TwoLayerCase`.
TOP_VELOCITY, BOTTOM_VELOCITY = 1.5, 3.0
#: Depth of the interface between the layers of :class:`TwoLayerCase`.
INTERFACE_DEPTH = 0.5


@dataclass(frozen=True, kw_only=True)
class TwoLayerCase(MaterialModelCase):
    """A slow layer over a layer twice as fast, on a 1 by 2 domain.

    A source and a receiver near the surface keep the acquisition inside it.
    """

    name: str = "two_layer"
    length_z: float = 1.0
    length_x: float = 2.0
    grid_spacing: float = 0.01
    source_locations: list[tuple[float, float]] = field(
        default_factory=lambda: [(-0.1, 0.5)]
    )
    receiver_locations: list[tuple[float, float]] = field(
        default_factory=lambda: [(-0.1, 1.5)]
    )

    def material(self, z: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate the two layer P-wave velocity.

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
        z, _ = np.broadcast_arrays(z, x)
        return {
            "p_wave_velocity": np.where(
                z > -INTERFACE_DEPTH, TOP_VELOCITY, BOTTOM_VELOCITY
            )
        }


def make_parameters(**overrides) -> CpwParameters:
    """Build acoustic heterogeneous calculator parameters on the two layers.

    Parameters
    ----------
    **overrides
        Fields replacing the defaults.

    Returns
    -------
    CpwParameters
        The parameters, with a velocity adapted mesh.
    """
    fields = {
        "source_frequency": 5.0,
        "mesh_frequency_factor": 2.0,
        "fem_method": "mass_lumped_triangle",
        "dimension": 2,
        "desired_degree": 2,
        "cpw_initial": 2.0,
        "cpw_accuracy": 0.1,
        "accepted_error_threshold": 0.05,
        "heterogeneous_case": TwoLayerCase(),
        "reference_degree": 3,
        "cpw_reference": 3.0,
        "velocity_adapted_mesh": True,
        **overrides,
    }
    return CpwParameters(**fields)


def layer_edge_lengths(
    problem: AcousticHeterogeneousProblem, method: str, cpw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mesh the two layer case and return the cell sizes in each layer.

    Parameters
    ----------
    problem : AcousticHeterogeneousProblem
        Problem whose mesh is built.
    method : str
        Finite element method of the mesh.
    cpw : float
        Cells-per-wavelength of the mesh.

    Returns
    -------
    tuple of np.ndarray
        Mean edge length of the cells of the top layer and of the bottom
        layer, away from the interface and the absorbing layer.
    """
    wave = problem.build_wave(problem.model_dictionary(method, 2), cpw)
    coordinates = wave.mesh.coordinates
    vertices = coordinates.dat.data_ro[coordinates.cell_node_map().values]
    edges = np.linalg.norm(vertices - np.roll(vertices, 1, axis=1), axis=2)
    h = edges.mean(axis=1)
    z, x = vertices.mean(axis=1).T
    case = problem.heterogeneous_case
    inside = (x > 0.0) & (x < case.length_x) & (z > -case.length_z)
    top = inside & (z > -INTERFACE_DEPTH + 0.15)
    bottom = inside & (z < -INTERFACE_DEPTH - 0.15)
    return h[top], h[bottom]


@pytest.mark.skipif(SeismicMesh is None, reason="SeismicMesh is not installed")
def test_adapted_mesh_follows_local_wavelength(tmp_path, monkeypatch) -> None:
    """Each layer has about cpw cells per local wavelength.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory the velocity and mesh files are written to.
    monkeypatch : pytest.MonkeyPatch
        Changes into ``tmp_path``.
    """
    monkeypatch.chdir(tmp_path)
    problem = AcousticHeterogeneousProblem(make_parameters())
    cpw = 2.0

    top, bottom = layer_edge_lengths(problem, "mass_lumped_triangle", cpw)

    for h, velocity in ((top, TOP_VELOCITY), (bottom, BOTTOM_VELOCITY)):
        target = velocity / (problem.mesh_frequency * cpw)
        assert 0.85 < np.median(h) / target < 1.2
        assert np.max(h) / target < 1.4


@pytest.mark.skipif(SeismicMesh is None, reason="SeismicMesh is not installed")
def test_unadapted_mesh_is_uniform(tmp_path, monkeypatch) -> None:
    """Without adaptation both layers are sized by the slowest velocity.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory the velocity and mesh files are written to.
    monkeypatch : pytest.MonkeyPatch
        Changes into ``tmp_path``.
    """
    monkeypatch.chdir(tmp_path)
    problem = AcousticHeterogeneousProblem(make_parameters(velocity_adapted_mesh=False))

    top, bottom = layer_edge_lengths(problem, "mass_lumped_triangle", 2.0)

    assert np.median(bottom) == pytest.approx(np.median(top), rel=0.1)


@pytest.mark.skipif(SeismicMesh is None, reason="SeismicMesh is not installed")
@pytest.mark.parametrize("velocity_adapted_mesh", [True, False])
def test_built_wave_holds_the_velocity(
    tmp_path, monkeypatch, velocity_adapted_mesh: bool,
) -> None:
    """The wave is built with the case velocity, which a PML forward solve needs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory the velocity and mesh files are written to.
    monkeypatch : pytest.MonkeyPatch
        Changes into ``tmp_path``.
    velocity_adapted_mesh : bool
        Whether the mesh is adapted to the velocity.
    """
    monkeypatch.chdir(tmp_path)
    problem = AcousticHeterogeneousProblem(
        make_parameters(velocity_adapted_mesh=velocity_adapted_mesh)
    )

    wave = problem.build_wave(problem.model_dictionary("mass_lumped_triangle", 2), 2.0)

    assert wave.c is not None
    velocity = wave.c.dat.data_ro
    assert velocity.min() == pytest.approx(TOP_VELOCITY)
    assert velocity.max() == pytest.approx(BOTTOM_VELOCITY)


def test_quadrilateral_mesh_stays_uniform(tmp_path, monkeypatch) -> None:
    """A quadrilateral mesh cannot be adapted and is built uniformly.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory the velocity file is written to.
    monkeypatch : pytest.MonkeyPatch
        Changes into ``tmp_path``.
    """
    monkeypatch.chdir(tmp_path)
    problem = AcousticHeterogeneousProblem(make_parameters())

    wave = problem.build_wave(problem.model_dictionary("spectral_quadrilateral", 2), 2.0)

    assert wave.mesh_parameters.mesh_type == "firedrake_mesh"
    assert wave.mesh_parameters.velocity_model is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"wave_type": WaveType.ISOTROPIC_ELASTIC},
        {
            "velocity_profile_type": VelocityProfileType.HOMOGENEOUS,
            "heterogeneous_case": None,
            "minimum_velocity": 1.5,
        },
    ],
)
def test_adapted_mesh_only_for_acoustic_heterogeneous(overrides: dict) -> None:
    """Other problems reject a velocity adapted mesh.

    Parameters
    ----------
    overrides : dict
        Fields making the problem elastic or homogeneous.
    """
    with pytest.raises(ValueError, match="velocity_adapted_mesh"):
        make_parameters(**overrides)
