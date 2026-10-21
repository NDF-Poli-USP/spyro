"""Isotropic elastic wave with the synthetic cpw material read from files."""

from pathlib import Path

import firedrake as fire
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import spyro
from spyro.domains.space import create_function_space
from spyro.tools.cells_per_wavelength_calculator import ElasticSyntheticCase
from spyro.tools.cells_per_wavelength_calculator.input_models import (
    create_elastic_model_for_meshing_parameter_2D_heterogeneous,
)

dt = 1e-4
number_of_timesteps = 4


def nodal_values(wave: spyro.IsotropicWave, expression) -> np.ndarray:
    """Interpolate a material expression onto the scalar space of the wave.

    Parameters
    ----------
    wave : spyro.IsotropicWave
        Wave whose mesh and discretization define the scalar space.
    expression : ufl.core.expr.Expr
        Material parameter, either a ``Function`` or a UFL expression.

    Returns
    -------
    np.ndarray
        Values of ``expression`` at the nodes of the scalar space.
    """
    space = create_function_space(wave.mesh, wave.method, wave.degree, dim=1)
    return fire.Function(space).interpolate(expression).dat.data_ro.copy()


def expected_material(
    case: ElasticSyntheticCase, z: np.ndarray, x: np.ndarray,
) -> dict[str, np.ndarray]:
    """Interpolate the sampled case material linearly at the given points.

    This is what reading the case files must reproduce: the files hold the
    material sampled on the case grid, and the grid is interpolated
    linearly between samples.

    Parameters
    ----------
    case : ElasticSyntheticCase
        Case whose sampled material is interpolated.
    z : np.ndarray
        Vertical coordinates, negative below the surface.
    x : np.ndarray
        Horizontal coordinates.

    Returns
    -------
    dict
        Density, P-wave velocity and S-wave velocity arrays, keyed by their
        ``synthetic_data`` names.
    """
    grid_z = np.linspace(
        -case.length_z, 0.0, round(case.length_z / case.grid_spacing) + 1,
    )
    grid_x = np.linspace(
        0.0, case.length_x, round(case.length_x / case.grid_spacing) + 1,
    )
    return {
        parameter: RegularGridInterpolator((grid_z, grid_x), values)((z, x))
        for parameter, values in case.sampled_material.items()
    }


def test_isotropic_wave_synthetic_material_from_file(tmp_path: Path) -> None:
    """The cpw synthetic files load into the parameters and the wave runs."""
    # A coarse grid and mesh keep the file writing and the solve fast
    case = ElasticSyntheticCase(
        name=str(tmp_path / "elastic_synthetic"),
        grid_spacing=0.1,
        dt=dt,
        final_time=number_of_timesteps * dt,
    )
    dictionary = create_elastic_model_for_meshing_parameter_2D_heterogeneous(
        method="spectral_quadrilateral",
        degree=2,
        frequency=5.0,
        case=case,
    )
    wave = spyro.IsotropicWave(dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.5})
    wave._initialize_model_parameters()

    coordinates = fire.SpatialCoordinate(wave.mesh)
    expected = expected_material(
        case,
        nodal_values(wave, coordinates[0]),
        nodal_values(wave, coordinates[1]),
    )
    rho = expected["density"]
    vp = expected["p_wave_velocity"]
    vs = expected["s_wave_velocity"]

    # The files declare the velocity set, so these three are fields.
    for parameter in (wave.rho, wave.c, wave.c_s):
        assert isinstance(parameter, fire.Function)

    # SEG-Y stores single precision
    tolerance = {"rtol": 1e-5, "atol": 0.0}
    assert np.allclose(nodal_values(wave, wave.rho), rho, **tolerance)
    assert np.allclose(nodal_values(wave, wave.c), vp, **tolerance)
    assert np.allclose(nodal_values(wave, wave.c_s), vs, **tolerance)
    assert np.allclose(nodal_values(wave, wave.mu), rho * vs**2, **tolerance)
    assert np.allclose(
        nodal_values(wave, wave.lmbda), rho * (vp**2 - 2 * vs**2), **tolerance,
    )

    # Hard coded S-wave velocities, one point per layer, each in an element
    # lying wholly inside its layer
    for point, s_wave_velocity in [
        # Sediments, 1.2 + 0.12 * depth
        ((-0.25, 4.75), 1.23),
        # Salt
        ((-1.75, 5.25), 2.3),
        # Pre-salt carbonate, 2.8 + 0.08 * depth
        ((-3.25, 5.25), 3.06),
        # Basement
        ((-4.25, 5.25), 3.5),
    ]:
        assert np.isclose(wave.c_s.at(point), s_wave_velocity, **tolerance)

    wave.forward_solve()

    receivers = np.asarray(wave.forward_solution_receivers)
    assert receivers.shape[0] == number_of_timesteps + 1
    assert np.all(np.isfinite(receivers))
    assert np.all(np.isfinite(wave.u_n.dat.data_ro))
    assert fire.norm(wave.u_n) > 0.0
