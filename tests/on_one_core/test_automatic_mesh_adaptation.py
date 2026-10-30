"""Automatic mesh adaptation: meshes, size law, adjoint pilot and loop.

The pilot test checks the dual-weighted-residual identity of the adjoint
indicator, which is exact for the linear scheme and the quadratic trace
functional: the localized residual of the prolonged coarse trajectory,
weighted by the automated adjoint of the refined scheme, sums to
``-2 J`` of the refined solution.
"""

import numpy as np
import pytest

import firedrake as fire
from spyro.adaptivity import (AdaptiveSizeField, RectangularDomain,
                              TwoLevelPilot, count_kmv_dofs, generate_mesh,
                              gradient_density, local_mesh_size,
                              read_triangulation)
from spyro.adaptivity.automatic_loop import (AutomaticMeshAdaptation,
                                             LoopParameters)
from spyro.adaptivity.size_fields import VelocityGrid

INTERFACE = -0.3
DOMAIN = RectangularDomain(z_min=-1.0, x_min=0.0, x_max=1.2, pad=0.2)
RECEIVERS = [(-0.1, x) for x in np.linspace(0.2, 1.0, 9)]


def two_layer(z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Water over a faster layer, with the jump at ``INTERFACE``.

    Parameters
    ----------
    z : numpy.ndarray
        Depths.
    x : numpy.ndarray
        Horizontal positions.

    Returns
    -------
    numpy.ndarray
        Velocity (km/s).
    """
    return np.where(np.asarray(z) > INTERFACE, 1.5, 2.5) + 0.0 * np.asarray(x)


def velocity_grid(spacing: float = 0.01) -> VelocityGrid:
    """The two-layer model on a grid covering the domain.

    Parameters
    ----------
    spacing : float, optional
        Grid spacing (km).

    Returns
    -------
    VelocityGrid
        The model, sampled at the cell centres of the grid rows so that the
        jump stays between two rows.
    """
    z = np.arange(DOMAIN.z_min, DOMAIN.z_max + spacing / 2, spacing)
    x = np.arange(DOMAIN.x_min, DOMAIN.x_max + spacing / 2, spacing)
    Z, X = np.meshgrid(z, x, indexing="ij")
    return VelocityGrid(two_layer(Z - 1e-9, X), z[0], x[0], spacing, spacing)


def dictionary(n_shots: int = 1, final_time: float = 0.6) -> dict:
    """spyro dictionary of the small test problem.

    Parameters
    ----------
    n_shots : int, optional
        Number of sources.
    final_time : float, optional
        Duration (s).

    Returns
    -------
    dict
        Model dictionary with KMV2 triangles and non-reflecting boundaries.
    """
    sources = [(-0.05, 0.4), (-0.05, 0.8)][:n_shots]
    return {
        "options": {"cell_type": "T", "variant": "lumped", "degree": 2,
                    "dimension": 2},
        "parallelism": {"type": "spatial"},
        "mesh": {"length_z": 1.0, "length_x": 1.2, "length_y": 0.0,
                 "mesh_type": "user_mesh"},
        "acquisition": {"source_type": "ricker", "source_locations": sources,
                        "frequency": 5.0, "delay": 1.5,
                        "delay_type": "multiples_of_minimum",
                        "receiver_locations": RECEIVERS},
        "absorving_boundary_conditions": {"status": True, "abc_type": "nrbc"},
        "time_axis": {"initial_time": 0.0, "final_time": final_time,
                      "dt": 1e-3, "amplitude": 1, "output_frequency": 10**9,
                      "gradient_sampling_frequency": 1},
        "visualization": {"forward_output": False, "forward_output_filename": None,
                          "fwi_velocity_model_output": False,
                          "velocity_model_filename": None, "gradient_output": False,
                          "gradient_filename": None, "adjoint_output": False,
                          "adjoint_filename": None, "debug_output": False},
    }


def late_window(times: np.ndarray, shot: int) -> np.ndarray:
    """Window that drops the first 0.25 s of every trace.

    Parameters
    ----------
    times : numpy.ndarray
        Sample times.
    shot : int
        Source index (unused: the window is the same for every shot).

    Returns
    -------
    numpy.ndarray
        Boolean ``(n_times, n_receivers)``.
    """
    return np.broadcast_to(times[:, None] > 0.25, (len(times), len(RECEIVERS)))


@pytest.fixture(scope="module")
def cpw_mesh(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A coarse CPW mesh conforming to the interface.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Temporary directory factory.

    Returns
    -------
    str
        Path of the mesh file.
    """
    path = tmp_path_factory.mktemp("meshes") / "cpw3.msh"
    generate_mesh(path, DOMAIN, lambda z, x: float(two_layer(z, x)) / (3.0 * 5.0),
                  conforming_depths=[INTERFACE])
    return str(path)


def test_mesh_markers_interface_and_dofs(cpw_mesh: str) -> None:
    """Markers follow spyro's convention, the interface is a mesh line and
    the DOF count matches Firedrake's.

    Parameters
    ----------
    cpw_mesh : str
        Mesh fixture.
    """
    mesh = fire.Mesh(cpw_mesh)
    one = fire.Constant(1.0)
    lengths = [fire.assemble(one * fire.ds(k, domain=mesh)) for k in (1, 2, 3, 4)]
    width = DOMAIN.x_max - DOMAIN.x_min
    height = DOMAIN.z_max - DOMAIN.z_min
    assert np.allclose(lengths, [width, width, height, height])
    z = mesh.coordinates.dat.data_ro[:, 0]
    assert z.min() == pytest.approx(DOMAIN.z_min)
    assert np.sum(np.isclose(z, INTERFACE)) >= 5

    points, triangles = read_triangulation(cpw_mesh)
    for degree in (2, 4):
        V = fire.FunctionSpace(mesh, "KMV", degree)
        assert count_kmv_dofs(triangles, degree) == V.dim()
    edges = points[triangles[:, 1:]] - points[triangles[:, :1]]
    areas = 0.5 * np.abs(edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    assert areas.sum() == pytest.approx(width * height)


def test_size_law_budget_and_bounds(cpw_mesh: str) -> None:
    """The size law meets a budget, stays within its bounds and refines where
    the density is large.

    Parameters
    ----------
    cpw_mesh : str
        Mesh fixture.
    """
    spacing = 0.01
    z, x = DOMAIN.grid(spacing)
    points, triangles = read_triangulation(cpw_mesh)
    current = local_mesh_size(points, triangles, z, x)
    Z, X = np.meshgrid(z, x, indexing="ij")
    density = gradient_density(velocity_grid()(Z, X), spacing, 0.03,
                               DOMAIN.in_interior(Z, X))
    assert np.all(density[~DOMAIN.in_interior(Z, X)] == 0.0)

    field = AdaptiveSizeField(density, current, z, x)
    field.set_floor(velocity_grid()(Z, X), 5.0, equivalent_cpw=6.0)
    base = float((current ** -2).sum() * spacing**2)
    scale = field.scale_for(1.3 * base)
    field.set_scale(scale)
    assert field.relative_cost(scale) == pytest.approx(1.3 * base, rel=1e-3)
    assert np.all(field.h <= current + 1e-12)
    assert np.all(field.h >= field.h_min - 1e-12)
    near = np.abs(Z - INTERFACE) < 0.02
    far = DOMAIN.in_interior(Z, X) & (np.abs(Z - INTERFACE) > 0.3)
    assert np.median(field.h[near] / current[near]) < np.median(field.h[far] / current[far])


@pytest.mark.slow
def test_pilot_adjoint_identity(cpw_mesh: str) -> None:
    """The adjoint indicator closes the dual-weighted-residual identity.

    Parameters
    ----------
    cpw_mesh : str
        Mesh fixture.
    """
    pilot = TwoLevelPilot(dictionary(), two_layer, window=late_window)
    forward_only = pilot.run(cpw_mesh)
    with_adjoint = pilot.run(cpw_mesh, compute_indicator=True)

    assert forward_only.dofs_fine > 3 * forward_only.dofs_coarse
    assert 0.0 < forward_only.trace_errors[0] < 1.0
    assert with_adjoint.trace_errors[0] == pytest.approx(forward_only.trace_errors[0])
    assert with_adjoint.identity_ratios[0] == pytest.approx(-1.0, abs=1e-8)
    if fire.COMM_WORLD.rank == 0:
        n_cells = with_adjoint.centroids.shape[0]
        assert with_adjoint.indicators.shape == (1, n_cells)
        assert np.abs(with_adjoint.indicators).sum() > 0.0


@pytest.mark.slow
@pytest.mark.parametrize("strategy", ["cpw", "gradient", "adjoint"])
def test_automatic_loop(strategy: str, tmp_path: pytest.TempPathFactory) -> None:
    """Each strategy refines from the same initial mesh until it accepts.

    Parameters
    ----------
    strategy : str
        Strategy of the loop.
    tmp_path : pathlib.Path
        Output directory.
    """
    pilot = TwoLevelPilot(dictionary(), two_layer, window=late_window)
    parameters = LoopParameters(tolerance=0.05, initial_cpw=2.5,
                                max_iterations=3, grid_spacing=0.01)
    loop = AutomaticMeshAdaptation(strategy, pilot, velocity_grid(), DOMAIN,
                                   frequency=5.0, degree=2, output_dir=tmp_path,
                                   parameters=parameters,
                                   conforming_depths=[INTERFACE])
    report = loop.run()

    iterations = report["iterations"]
    assert len(iterations) >= 2
    assert iterations[0]["mesh"] == "cpw_C2p5_s0.msh"
    dofs = [it["dofs"] for it in iterations]
    assert all(b > a for a, b in zip(dofs, dofs[1:]))
    assert iterations[-1]["eta_worst"] < iterations[0]["eta_worst"]
    if report["accepted"]:
        assert iterations[-1]["eta_worst"] <= parameters.theta * parameters.tolerance
    if strategy == "adjoint":
        assert iterations[0]["identity_ratios"][0] == pytest.approx(-1.0, abs=1e-8)
    assert (tmp_path / f"report_{strategy}.json").exists()
