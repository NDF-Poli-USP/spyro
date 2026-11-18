"""Heterogeneous cases of the cells-per-wavelength calculator.

Each case is a dataclass whose defaults describe it, given as the
calculator ``"heterogeneous_case"`` parameter. Any field can be replaced
when creating it, e.g. ``AcousticMarmousi2Case(final_time=3.0)``. A case
read from files subclasses
:class:`~.heterogeneous_case.MaterialFileCase`, and a case defined by a
function of the coordinates subclasses
:class:`~.heterogeneous_case.MaterialModelCase`.
"""

from dataclasses import dataclass, field

import numpy as np

from .heterogeneous_case import MaterialFileCase

#: P-wave velocity SEG-Y of the Marmousi2 model, with its water layer. It
#: is not distributed with spyro: download the P-wave velocity SEG-Y of
#: Martin et al. (2006) (``MODEL_P-WAVE_VELOCITY_1.25m.segy`` in the
#: University of Houston AGL distribution) and save it under this name.
MARMOUSI2_P_WAVE_VELOCITY_FILE = "velocity_models/vp_marmousi-ii.segy"

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
