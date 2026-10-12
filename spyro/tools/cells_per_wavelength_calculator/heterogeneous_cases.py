"""Heterogeneous cases of the cells-per-wavelength calculator.

Each case is a dataclass whose defaults describe it, given as the
calculator ``"heterogeneous_case"`` parameter. Any field can be replaced
when creating it, e.g. ``ElasticSyntheticCase(final_time=3.0)``. A case
read from files subclasses
:class:`~.heterogeneous_case.MaterialFileCase`, and a case defined by a
function of the coordinates subclasses
:class:`~.heterogeneous_case.MaterialModelCase`.
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ...io.segy_io import read_segy_velocity_model
from .elastic_2d_material_model import elastic_2d_material
from .heterogeneous_case import (
    METERS_PER_SECOND_THRESHOLD,
    MaterialFileCase,
    MaterialModelCase,
)
from .input_models import HETEROGENEOUS_FINAL_TIME

#: P-wave velocity SEG-Y of the Marmousi2 model, with its water layer. It
#: is not distributed with spyro: download the P-wave velocity SEG-Y of
#: Martin et al. (2006) (``MODEL_P-WAVE_VELOCITY_1.25m.segy`` in the
#: University of Houston AGL distribution) and save it under this name.
MARMOUSI2_P_WAVE_VELOCITY_FILE = "velocity_models/vp_marmousi-ii.segy"
#: S-wave velocity SEG-Y of the Marmousi2 model, zero in the water layer
#: (``MODEL_S-WAVE_VELOCITY_1.25m.segy`` in the same distribution).
MARMOUSI2_S_WAVE_VELOCITY_FILE = "velocity_models/vs_marmousi-ii.segy"
#: Density SEG-Y of the Marmousi2 model (``MODEL_DENSITY_1.25m.segy`` in
#: the same distribution).
MARMOUSI2_DENSITY_FILE = "velocity_models/rho_marmousi-ii.segy"

# Single streamer shot of the Marmousi2 case, in km. The vessel sails
# towards -x, towing the source and, behind it, the streamer.
#: Depth of the air-gun source below the sea surface.
MARMOUSI2_SOURCE_DEPTH = 0.006
#: Horizontal position of the source.
MARMOUSI2_SOURCE_X = 3.0
#: Depth of the streamer below the sea surface.
MARMOUSI2_STREAMER_DEPTH = 0.008
#: Horizontal offsets of the first and last receiver groups from the source.
MARMOUSI2_STREAMER_OFFSETS = (2.0, 10.0)
#: Distance between consecutive receiver groups along the streamer.
MARMOUSI2_GROUP_INTERVAL = 0.0125


def marmousi2_streamer_source() -> list[tuple[float, float]]:
    """Return the air-gun source of the Marmousi2 streamer shot.

    Returns
    -------
    list of tuple of float
        The source ``(z, x)`` location, at :data:`MARMOUSI2_SOURCE_DEPTH`
        and :data:`MARMOUSI2_SOURCE_X`.
    """
    return [(-MARMOUSI2_SOURCE_DEPTH, MARMOUSI2_SOURCE_X)]


def marmousi2_streamer_receivers() -> list[tuple[float, float]]:
    """Return the receiver groups of the Marmousi2 streamer.

    Returns
    -------
    list of tuple of float
        Receiver ``(z, x)`` locations at :data:`MARMOUSI2_STREAMER_DEPTH`,
        every :data:`MARMOUSI2_GROUP_INTERVAL` from the near to the far
        offset of :data:`MARMOUSI2_STREAMER_OFFSETS`, behind the source.
    """
    near_offset, far_offset = MARMOUSI2_STREAMER_OFFSETS
    quantity = round((far_offset - near_offset) / MARMOUSI2_GROUP_INTERVAL) + 1
    offsets = np.linspace(near_offset, far_offset, quantity)
    return [
        (-MARMOUSI2_STREAMER_DEPTH, float(MARMOUSI2_SOURCE_X + offset))
        for offset in offsets
    ]


@dataclass(frozen=True, kw_only=True)
class ElasticSyntheticCase(MaterialModelCase):
    """Synthetic pre-salt section, the 2D isotropic elastic case.

    Its material is
    :func:`~.elastic_2d_material_model.elastic_2d_material`.

    Attributes
    ----------
    name : str
        Default is ``"elastic_synthetic"``.
    length_z : float
        Default is 5.0.
    length_x : float
        Default is 10.0.
    """

    name: str = "elastic_synthetic"
    length_z: float = 5.0
    length_x: float = 10.0

    def material(self, z: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate the pre-salt material.

        Parameters
        ----------
        z : np.ndarray
            Vertical coordinates, negative below the surface.
        x : np.ndarray
            Horizontal coordinates, broadcastable against ``z``.

        Returns
        -------
        dict
            P-wave velocity, S-wave velocity and density arrays.
        """
        return elastic_2d_material(z, x, self.length_x)


@dataclass(frozen=True, kw_only=True)
class AcousticMarmousi2Case(MaterialFileCase):
    """Marmousi2 model with its water layer, the 2D acoustic real case.

    The model is the 17 km by 3.5 km Marmousi2 of Martin, Wiley and
    Marfurt (2006), "Marmousi2: An elastic upgrade for Marmousi", The
    Leading Edge 25(2), 156-166, doi: 10.1190/1.2172306, whose top 0.45 km
    is water. Only its P-wave velocity is read, so it is acoustic only.

    The acquisition is a single shot of a towed-streamer survey: an
    air-gun source :data:`MARMOUSI2_SOURCE_DEPTH` below the sea surface,
    and a streamer :data:`MARMOUSI2_STREAMER_DEPTH` deep trailing it, with
    receiver groups every :data:`MARMOUSI2_GROUP_INTERVAL` at offsets from
    2 km to 10 km. The top of the domain is the sea surface. The time axis
    is the default heterogeneous acoustic one.

    Attributes
    ----------
    name : str
        Default is ``"acoustic_marmousi2"``.
    length_z : float
        Default is 3.5.
    length_x : float
        Default is 17.0.
    material_files : dict of str to str
        Default is :data:`MARMOUSI2_P_WAVE_VELOCITY_FILE` as the P-wave
        velocity.
    source_locations : list of tuple of float
        Default is :func:`marmousi2_streamer_source`.
    receiver_locations : list of tuple of float
        Default is :func:`marmousi2_streamer_receivers`.
    """

    name: str = "acoustic_marmousi2"
    length_z: float = 3.5
    length_x: float = 17.0
    material_files: dict[str, str] = field(
        default_factory=lambda: {
            "p_wave_velocity": MARMOUSI2_P_WAVE_VELOCITY_FILE,
        }
    )
    source_locations: list[tuple[float, float]] = field(
        default_factory=marmousi2_streamer_source
    )
    receiver_locations: list[tuple[float, float]] = field(
        default_factory=marmousi2_streamer_receivers
    )


@dataclass(frozen=True, kw_only=True)
class ElasticMarmousi2Case(MaterialModelCase):
    """Marmousi2 model with its water layer, the 2D isotropic elastic real case.

    The model is the one of :class:`AcousticMarmousi2Case`, with its
    S-wave velocity and density as well. The water layer is the elastic
    material with a zero S-wave velocity.

    The SEG-Y files are read once and resampled, nearest neighbour, on the
    case grid, so no S-wave velocity between the water's zero and the
    seafloor's is made up. The resampled material is written in km/s and
    g/cm^3, as the velocity interpolator converts from m/s only fields
    with no value below its threshold, which an S-wave velocity zero in the
    water never is.

    The acquisition is the streamer shot of :class:`AcousticMarmousi2Case`,
    with the point force of the elastic model as the source. The time axis
    is that of :class:`AcousticMarmousi2Case`, long enough for the far
    offsets.

    Attributes
    ----------
    name : str
        Default is ``"elastic_marmousi2"``.
    length_z : float
        Default is 3.5.
    length_x : float
        Default is 17.0.
    segy_files : dict of str to str
        Marmousi2 SEG-Y of each material parameter, keyed like
        :data:`~.heterogeneous_case.MATERIAL_FILE_SUFFIXES`, in m/s or
        km/s and in kg/m^3 or g/cm^3. Default is
        :data:`MARMOUSI2_P_WAVE_VELOCITY_FILE`,
        :data:`MARMOUSI2_S_WAVE_VELOCITY_FILE` and
        :data:`MARMOUSI2_DENSITY_FILE`.
    source_locations : list of tuple of float
        Default is :func:`marmousi2_streamer_source`.
    receiver_locations : list of tuple of float
        Default is :func:`marmousi2_streamer_receivers`.
    final_time : float
        Default is
        :data:`~.input_models.HETEROGENEOUS_FINAL_TIME`.
    """

    name: str = "elastic_marmousi2"
    length_z: float = 3.5
    length_x: float = 17.0
    segy_files: dict[str, str] = field(
        default_factory=lambda: {
            "p_wave_velocity": MARMOUSI2_P_WAVE_VELOCITY_FILE,
            "s_wave_velocity": MARMOUSI2_S_WAVE_VELOCITY_FILE,
            "density": MARMOUSI2_DENSITY_FILE,
        }
    )
    source_locations: list[tuple[float, float]] = field(
        default_factory=marmousi2_streamer_source
    )
    receiver_locations: list[tuple[float, float]] = field(
        default_factory=marmousi2_streamer_receivers
    )
    final_time: float | None = HETEROGENEOUS_FINAL_TIME

    @cached_property
    def segy_material(self) -> dict[str, RegularGridInterpolator]:
        """Nearest-neighbour interpolant of each SEG-Y, read once.

        Returns
        -------
        dict
            Interpolant of ``(z, x)`` points per material parameter, in
            km/s for velocities and g/cm^3 for density. The SEG-Y grid
            spans the case domain.
        """
        interpolants = {}
        for parameter, file_name in self.segy_files.items():
            # Rows go from z = -length_z up to the surface
            values, nz, nx = read_segy_velocity_model(file_name)
            interpolants[parameter] = RegularGridInterpolator(
                (
                    np.linspace(-self.length_z, 0.0, nz),
                    np.linspace(0.0, self.length_x, nx),
                ),
                _from_si_units(values),
                method="nearest",
            )
        return interpolants

    def material(self, z: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate the Marmousi2 material.

        Parameters
        ----------
        z : np.ndarray
            Vertical coordinates, negative below the surface, within the
            case domain.
        x : np.ndarray
            Horizontal coordinates, broadcastable against ``z``, within the
            case domain.

        Returns
        -------
        dict
            P-wave velocity, S-wave velocity and density arrays, in km/s
            and g/cm^3.
        """
        z, x = np.broadcast_arrays(z, x)
        return {
            parameter: interpolant((z, x))
            for parameter, interpolant in self.segy_material.items()
        }

    def minimum(self, parameter: str) -> float:
        """Return the minimum of a material parameter over the case grid.

        The S-wave velocity minimum is that of the slowest wave, which is
        either the slowest S-wave in the rock or the P-wave in the water,
        where there is no S-wave.

        Parameters
        ----------
        parameter : str
            Material parameter, e.g. ``"s_wave_velocity"``.

        Returns
        -------
        float
            Minimum of the sampled parameter, ignoring the water for the
            S-wave velocity.

        Raises
        ------
        ValueError
            If the case does not define ``parameter``.
        """
        if parameter != "s_wave_velocity":
            return super().minimum(parameter)
        self._check_defines(
            ("p_wave_velocity", "s_wave_velocity"), self.sampled_material
        )
        s_wave_velocity = self.sampled_material["s_wave_velocity"]
        fluid = s_wave_velocity <= 0.0
        return float(
            min(
                np.min(s_wave_velocity[~fluid], initial=np.inf),
                np.min(
                    self.sampled_material["p_wave_velocity"][fluid],
                    initial=np.inf,
                ),
            )
        )


def _from_si_units(values: np.ndarray) -> np.ndarray:
    """Convert a material from m/s or kg/m^3 to km/s or g/cm^3, if needed.

    Parameters
    ----------
    values : np.ndarray
        Velocity or density values.

    Returns
    -------
    np.ndarray
        ``values`` divided by 1000 if any of them is above
        :data:`~.heterogeneous_case.METERS_PER_SECOND_THRESHOLD`, else
        ``values``. The maximum is tested, since an S-wave velocity is zero
        in water whatever its unit.
    """
    if np.max(values) > METERS_PER_SECOND_THRESHOLD:
        return values / 1000.0
    return values
