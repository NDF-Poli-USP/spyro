"""Tests for the heterogeneous cases of the cells-per-wavelength calculator."""

import numpy as np
import pytest

from spyro.tools.cells_per_wavelength_calculator import (
    ElasticSyntheticCase,
    elastic_2d_material,
)


def test_ElasticSyntheticCase() -> None:
    """The case samples the model and takes its minima from it."""
    case = ElasticSyntheticCase(final_time=1.0)

    assert case.name == "elastic_synthetic"
    assert (case.length_z, case.length_x) == (5.0, 10.0)
    assert case.final_time == 1.0
    assert case.minimum("s_wave_velocity") == pytest.approx(1.2)
    assert case.minimum("p_wave_velocity") == pytest.approx(2.5)
    material = case.sampled_material
    assert material["density"].shape == (501, 1001)


def test_elastic_synthetic_case_material_follows_length_x() -> None:
    """The salt interfaces repeat over the case length in x."""
    case = ElasticSyntheticCase(length_x=20.0)
    z, x = np.array([-1.05]), np.array([5.0])

    assert np.allclose(
        case.material(z, x)["p_wave_velocity"],
        elastic_2d_material(z, x, length_x=20.0)["p_wave_velocity"],
    )
