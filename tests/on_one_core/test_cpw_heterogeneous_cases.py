"""Tests for the heterogeneous cases of the cells-per-wavelength calculator."""

import numpy as np
import pytest

from spyro.io.segy_io import create_segy_from_grid
from spyro.tools.cells_per_wavelength_calculator import (
    MARMOUSI2_DENSITY_FILE,
    MARMOUSI2_P_WAVE_VELOCITY_FILE,
    MARMOUSI2_S_WAVE_VELOCITY_FILE,
    AcousticMarmousi2Case,
    ElasticMarmousi2Case,
    ElasticSyntheticCase,
    elastic_2d_material,
)
from spyro.tools.cells_per_wavelength_calculator.heterogeneous_cases import (
    MARMOUSI2_GROUP_INTERVAL,
    MARMOUSI2_SOURCE_DEPTH,
    MARMOUSI2_SOURCE_X,
    MARMOUSI2_STREAMER_DEPTH,
)
from spyro.io.segy_io import read_segy_velocity_model
from spyro.tools.cells_per_wavelength_calculator.input_models import (
    HETEROGENEOUS_FINAL_TIME,
    create_acoustic_model_for_meshing_parameter_2D_heterogeneous,
    create_elastic_model_for_meshing_parameter_2D_heterogeneous,
)

#: Depth of the Marmousi2 water layer, in km.
MARMOUSI2_WATER_DEPTH = 0.45


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


def make_elastic_marmousi2_stand_in(
    directory, rock_s_wave_velocity: float = 1200.0, **overrides,
) -> ElasticMarmousi2Case:
    """Build the elastic Marmousi2 case on two-layer stand-in SEG-Ys in SI units.

    Parameters
    ----------
    directory : pathlib.Path
        Directory of the stand-in SEG-Ys and of the written material.
    rock_s_wave_velocity : float, optional
        S-wave velocity below the water layer, in m/s. Default is 1200.0.
    **overrides
        Other case fields.

    Returns
    -------
    ElasticMarmousi2Case
        The case, named ``directory / "marmousi2"``.
    """
    layers = {
        "p_wave_velocity": (1500.0, 2500.0),
        "s_wave_velocity": (0.0, rock_s_wave_velocity),
        "density": (1000.0, 2200.0),
    }
    segy_files = {}
    for parameter, (water, rock) in layers.items():
        segy_files[parameter] = str(directory / f"{parameter}.segy")
        write_layered_marmousi2_stand_in(segy_files[parameter], water, rock)
    return ElasticMarmousi2Case(
        name=str(directory / "marmousi2"), segy_files=segy_files, **overrides,
    )


def test_ElasticMarmousi2Case() -> None:
    """The case is the Marmousi2 domain, read from its three material files."""
    case = ElasticMarmousi2Case()

    assert case.name == "elastic_marmousi2"
    assert (case.length_z, case.length_x) == (3.5, 17.0)
    assert case.segy_files == {
        "p_wave_velocity": MARMOUSI2_P_WAVE_VELOCITY_FILE,
        "s_wave_velocity": MARMOUSI2_S_WAVE_VELOCITY_FILE,
        "density": MARMOUSI2_DENSITY_FILE,
    }
    assert case.final_time == HETEROGENEOUS_FINAL_TIME
    assert case.source_locations == AcousticMarmousi2Case().source_locations
    assert case.receiver_locations == AcousticMarmousi2Case().receiver_locations


def test_elastic_marmousi2_material_is_in_km_and_g_units(tmp_path) -> None:
    """Velocities in m/s and density in kg/m^3 are read in km/s and g/cm^3."""
    case = make_elastic_marmousi2_stand_in(tmp_path)

    material = case.material(np.array([[-0.1], [-2.0]]), np.array([[8.0]]))

    assert np.allclose(material["p_wave_velocity"].ravel(), [1.5, 2.5])
    assert np.allclose(material["s_wave_velocity"].ravel(), [0.0, 1.2])
    assert np.allclose(material["density"].ravel(), [1.0, 2.2])


def test_elastic_marmousi2_material_keeps_the_seafloor_sharp(tmp_path) -> None:
    """Resampling makes up no S-wave velocity between water and rock."""
    case = make_elastic_marmousi2_stand_in(tmp_path)

    s_wave_velocity = case.sampled_material["s_wave_velocity"]

    assert s_wave_velocity.shape == (351, 1701)
    assert np.allclose(np.unique(np.round(s_wave_velocity, 6)), [0.0, 1.2])


@pytest.mark.parametrize(
    "rock_s_wave_velocity, expected",
    [(1200.0, 1.2), (1800.0, 1.5)],
    ids=["rock S-wave", "water P-wave"],
)
def test_elastic_marmousi2_minimum_is_slowest_wave(
    rock_s_wave_velocity: float, expected: float, tmp_path,
) -> None:
    """The wavelength velocity ignores the water's zero S-wave velocity."""
    case = make_elastic_marmousi2_stand_in(tmp_path, rock_s_wave_velocity)

    assert case.minimum("s_wave_velocity") == pytest.approx(expected)
    assert case.minimum("p_wave_velocity") == pytest.approx(1.5)


def test_elastic_marmousi2_writes_converted_material(tmp_path) -> None:
    """The written S-wave velocity is in km/s, as the interpolator cannot convert it."""
    case = make_elastic_marmousi2_stand_in(tmp_path)

    model = create_elastic_model_for_meshing_parameter_2D_heterogeneous(
        method="spectral_quadrilateral", degree=4, frequency=5.0, case=case,
    )

    synthetic_data = model["synthetic_data"]
    assert synthetic_data["s_wave_velocity"] == f"{tmp_path}/marmousi2_vs.segy"
    values, _, _ = read_segy_velocity_model(synthetic_data["s_wave_velocity"])
    assert values.min() == pytest.approx(0.0)
    assert values.max() == pytest.approx(1.2)
    acquisition = model["acquisition"]
    assert acquisition["source_locations"] == case.source_locations
    assert len(acquisition["receiver_locations"]) == 641
    assert model["time_axis"]["final_time"] == HETEROGENEOUS_FINAL_TIME
    assert (model["mesh"]["length_z"], model["mesh"]["length_x"]) == (3.5, 17.0)
