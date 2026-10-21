"""Tests for the heterogeneous cases of the cells-per-wavelength calculator."""

import numpy as np
import pytest

from spyro.io.segy_io import create_segy_from_grid
from spyro.tools.cells_per_wavelength_calculator import (
    MARMOUSI2_P_WAVE_VELOCITY_FILE,
    AcousticMarmousi2Case,
)
from spyro.tools.cells_per_wavelength_calculator.heterogeneous_cases import (
    MARMOUSI2_GROUP_INTERVAL,
    MARMOUSI2_SOURCE_DEPTH,
    MARMOUSI2_SOURCE_X,
    MARMOUSI2_STREAMER_DEPTH,
)
from spyro.tools.cells_per_wavelength_calculator.input_models import (
    create_acoustic_model_for_meshing_parameter_2D_heterogeneous,
)

#: Depth of the Marmousi2 water layer, in km.
MARMOUSI2_WATER_DEPTH = 0.45


def write_layered_marmousi2_stand_in(
    file_name: str, water: float = 1500.0, rock: float = 2500.0,
) -> None:
    """Write a coarse two-layer stand-in of a Marmousi2 material SEG-Y.

    Parameters
    ----------
    file_name : str
        SEG-Y file to write.
    water : float, optional
        Value in the water layer. Default is 1500.0, the water P-wave
        velocity in m/s, as in the Marmousi2 distribution.
    rock : float, optional
        Value below the water layer. Default is 2500.0.
    """
    z = np.linspace(-3.5, 0.0, 36)[:, np.newaxis]
    x = np.linspace(0.0, 17.0, 171)[np.newaxis, :]
    values = np.where(z > -MARMOUSI2_WATER_DEPTH, water, rock) + 0.0 * x
    create_segy_from_grid(values.astype(np.float32), file_name, rotate=True)


def test_AcousticMarmousi2Case() -> None:
    """The case is the Marmousi2 domain, read from its P-wave velocity file."""
    case = AcousticMarmousi2Case()

    assert case.name == "acoustic_marmousi2"
    assert (case.length_z, case.length_x) == (3.5, 17.0)
    assert case.material_files == {
        "p_wave_velocity": MARMOUSI2_P_WAVE_VELOCITY_FILE
    }
    assert case.write_material_files(("p_wave_velocity",)) == {
        "p_wave_velocity": MARMOUSI2_P_WAVE_VELOCITY_FILE
    }


def test_acoustic_marmousi2_case_is_acoustic_only() -> None:
    """Only the P-wave velocity is defined, so elastic models are rejected."""
    case = AcousticMarmousi2Case()

    with pytest.raises(ValueError, match="does not define"):
        case.write_material_files(
            ("p_wave_velocity", "s_wave_velocity", "density")
        )


def test_acoustic_marmousi2_case_minimum_is_water(tmp_path) -> None:
    """The wavelength velocity is the water velocity, read in km/s."""
    file_name = str(tmp_path / "vp.segy")
    write_layered_marmousi2_stand_in(file_name)
    case = AcousticMarmousi2Case(material_files={"p_wave_velocity": file_name})

    assert case.minimum("p_wave_velocity") == pytest.approx(1.5)


def test_acoustic_marmousi2_case_has_streamer_acquisition(tmp_path) -> None:
    """A shallow source records on a trailing streamer at 2 to 10 km offsets."""
    file_name = str(tmp_path / "vp.segy")
    write_layered_marmousi2_stand_in(file_name)
    case = AcousticMarmousi2Case(material_files={"p_wave_velocity": file_name})

    model = create_acoustic_model_for_meshing_parameter_2D_heterogeneous(
        method="mass_lumped_triangle",
        degree=4,
        frequency=5.0,
        minimum_velocity=1.5,
        case=case,
    )

    acquisition = model["acquisition"]
    [(source_z, source_x)] = acquisition["source_locations"]
    receivers = np.asarray(acquisition["receiver_locations"])
    assert source_z == pytest.approx(-MARMOUSI2_SOURCE_DEPTH)
    assert np.allclose(receivers[:, 0], -MARMOUSI2_STREAMER_DEPTH)
    # Both are towed in the water, just below the sea surface
    assert -0.02 < receivers[0, 0] < source_z < 0.0
    offsets = receivers[:, 1] - source_x
    assert offsets.min() == pytest.approx(2.0)
    assert offsets.max() == pytest.approx(10.0)
    assert np.allclose(np.diff(offsets), MARMOUSI2_GROUP_INTERVAL)
    assert len(receivers) == 641
    assert receivers[:, 1].max() < case.length_x
    assert (model["mesh"]["length_z"], model["mesh"]["length_x"]) == (3.5, 17.0)
    assert model["synthetic_data"]["real_velocity_file"] == file_name


def test_acoustic_marmousi2_case_acquisition_can_be_replaced() -> None:
    """Default acquisition lists are per case and can be overridden."""
    case = AcousticMarmousi2Case(source_locations=[(-0.01, 5.0)])

    assert case.source_locations == [(-0.01, 5.0)]
    assert AcousticMarmousi2Case().source_locations == [
        (-MARMOUSI2_SOURCE_DEPTH, MARMOUSI2_SOURCE_X)
    ]
