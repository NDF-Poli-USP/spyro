"""Wave physics the cells-per-wavelength calculator supports."""

from dataclasses import dataclass

import spyro
from ...solvers.wave import Wave


@dataclass(frozen=True)
class WavePhysics:
    """Solver-level facts about a wave physics.

    Attributes
    ----------
    name : str
        Short name, starting the default reference solution file name.
    wave_class : type[Wave]
        Wave solver class built for every forward solve.
    has_displacement : bool
        Whether receiver data has a trailing direction axis.
    wavelength_parameter : str
        Material parameter of the slowest wave, whose minimum sets the
        wavelength.
    """

    name: str
    wave_class: type[Wave]
    has_displacement: bool
    wavelength_parameter: str


#: Isotropic acoustic physics.
ACOUSTIC = WavePhysics(
    name="acoustic",
    wave_class=spyro.AcousticWave,
    has_displacement=False,
    wavelength_parameter="p_wave_velocity",
)

#: Isotropic elastic physics, whose slowest body wave is the S-wave.
ISOTROPIC_ELASTIC = WavePhysics(
    name="elastic",
    wave_class=spyro.IsotropicWave,
    has_displacement=True,
    wavelength_parameter="s_wave_velocity",
)
