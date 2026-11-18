"""Tests for the heterogeneous elastic material model of the cpw calculator."""

import firedrake as fire
import numpy as np
import pytest

import spyro
from spyro.tools.cells_per_wavelength_calculator import (
    ElasticSyntheticCase,
    elastic_2d_material,
)
from spyro.tools.cells_per_wavelength_calculator.input_models import (
    create_elastic_model_for_meshing_parameter_2D_heterogeneous,
)


@pytest.mark.parametrize(
    "z, x, expected",
    [
        # Sediments at the surface
        (0.0, 0.0, (2.5, 1.2, 2.10)),
        # Middle of the salt body
        (-1.5, 5.0, (4.5, 2.3, 2.20)),
        # Pre-salt carbonate, below the salt and above the basement
        (-3.0, 5.0, (5.3 + 0.45, 2.8 + 0.24, 2.55 + 0.06)),
        # Basement
        (-4.5, 5.0, (6.2, 3.5, 2.85)),
    ],
)
def test_elastic_2d_material_layers(
    z: float, x: float, expected: tuple[float, float, float],
) -> None:
    """Each layer has its own P-wave velocity, S-wave velocity and density."""
    material = elastic_2d_material(z, x, length_x=10.0)

    assert material["p_wave_velocity"] == pytest.approx(expected[0])
    assert material["s_wave_velocity"] == pytest.approx(expected[1])
    assert material["density"] == pytest.approx(expected[2])


def test_elastic_2d_material_broadcasts() -> None:
    """A column of z and a row of x give a (nz, nx) grid."""
    z = np.linspace(-5.0, 0.0, 6)[:, np.newaxis]
    x = np.linspace(0.0, 10.0, 11)[np.newaxis, :]

    material = elastic_2d_material(z, x, length_x=10.0)

    for values in material.values():
        assert values.shape == (6, 11)
    assert material["s_wave_velocity"].min() == pytest.approx(1.2)


@pytest.mark.parametrize("fast_interpolate", [True, False])
def test_isotropic_wave_reads_material_files(
    fast_interpolate: bool, tmp_path,
) -> None:
    """Both interpolation paths load the material written to the SEG-Ys."""
    length_x = 10.0
    # A coarse grid keeps the projection path, which meshes the grid, fast
    case = ElasticSyntheticCase(name=str(tmp_path / "salt"), grid_spacing=0.1)
    dictionary = create_elastic_model_for_meshing_parameter_2D_heterogeneous(
        method="spectral_quadrilateral",
        degree=2,
        frequency=10.0,
        case=case,
    )
    wave = spyro.IsotropicWave(dictionary)
    wave.set_mesh(input_mesh_parameters={"edge_length": 0.5})

    wave.initialize_model_parameters_from_file(
        dictionary["synthetic_data"], fast_interpolate=fast_interpolate
    )

    space = wave.c_s.function_space()
    coordinates = fire.Function(
        fire.VectorFunctionSpace(wave.mesh, space.ufl_element())
    ).interpolate(wave.mesh.coordinates).dat.data_ro
    expected = elastic_2d_material(coordinates[:, 0], coordinates[:, 1], length_x)
    for parameter, values in [
        ("p_wave_velocity", wave.c),
        ("s_wave_velocity", wave.c_s),
        ("density", wave.rho),
    ]:
        # Away from the layer interfaces the grid is linear between samples
        mismatch = ~np.isclose(values.dat.data_ro, expected[parameter], atol=2e-2)
        assert mismatch.mean() < 0.1, parameter
