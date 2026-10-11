"""Base classes of the heterogeneous cells-per-wavelength cases.

Concrete cases are in :mod:`~.heterogeneous_cases`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import h5py
import numpy as np

from ...io.segy_io import create_segy_from_grid, read_segy_velocity_model
from ...mpi.spyro_mpi import SpyroEnsemble

#: Suffix of the SEG-Y file written for each material parameter.
MATERIAL_FILE_SUFFIXES = {
    "p_wave_velocity": "_vp",
    "s_wave_velocity": "_vs",
    "density": "_rho",
}

#: Spacing of the grid a material model is sampled on, in km.
MATERIAL_GRID_SPACING = 0.01

#: File velocities above this are in m/s, as in the velocity interpolator.
METERS_PER_SECOND_THRESHOLD = 100.0


@dataclass(frozen=True, kw_only=True)
class HeterogeneousCase(ABC):
    """Domain and material of a heterogeneous cells-per-wavelength case.

    The acquisition and time axis default to those of the wave physics,
    and each one given here replaces its default. Subclasses define where
    the material comes from: :class:`MaterialFileCase` or
    :class:`MaterialModelCase`.

    Attributes
    ----------
    name : str
        Case name, the base name of any SEG-Y file written for it.
    length_z : float
        Domain length in the z direction, without absorbing layer.
    length_x : float
        Domain length in the x direction, without absorbing layer.
    minimum_velocity : float or None
        Velocity whose wavelength the cells-per-wavelength refers to, the
        P-wave velocity for acoustic and the S-wave velocity for elastic.
        Default is None, for its minimum over the material.
    source_locations : list of tuple of float or None
        Source ``(z, x)`` locations. Default is None, for the physics
        default.
    receiver_locations : list of tuple of float or None
        Receiver ``(z, x)`` locations. Default is None, for the physics
        default.
    final_time : float or None
        Final time of the simulation. Default is None, for the physics
        default.
    dt : float or None
        Time step of the reference solve. Default is None, for the physics
        default.
    """

    name: str
    length_z: float
    length_x: float
    minimum_velocity: float | None = None
    source_locations: list[tuple[float, float]] | None = None
    receiver_locations: list[tuple[float, float]] | None = None
    final_time: float | None = None
    dt: float | None = None

    def __post_init__(self) -> None:
        """Check the domain.

        Raises
        ------
        ValueError
            If a length is not positive.
        """
        if self.length_z <= 0.0 or self.length_x <= 0.0:
            raise ValueError(
                f"Case {self.name!r} lengths must be positive, got "
                f"length_z={self.length_z} and length_x={self.length_x}."
            )

    @abstractmethod
    def write_material_files(self, parameters: tuple[str, ...]) -> dict[str, str]:
        """Return the file of each material parameter, writing it if needed.

        Parameters
        ----------
        parameters : tuple of str
            Material parameters the wave physics reads.

        Returns
        -------
        dict
            File name keyed by material parameter.

        Raises
        ------
        ValueError
            If the case does not define one of ``parameters``.
        """

    @abstractmethod
    def minimum(self, parameter: str) -> float:
        """Return the minimum of a material parameter, in km/s for velocities.

        Parameters
        ----------
        parameter : str
            Material parameter, e.g. ``"s_wave_velocity"``.

        Returns
        -------
        float
            Minimum of the parameter over the material.

        Raises
        ------
        ValueError
            If the case does not define ``parameter``.
        """

    def _check_defines(self, parameters: tuple[str, ...], material: dict) -> None:
        """Check that a material defines every parameter.

        Parameters
        ----------
        parameters : tuple of str
            Material parameters needed.
        material : dict
            Files or arrays keyed by material parameter.

        Raises
        ------
        ValueError
            If ``material`` lacks one of ``parameters``.
        """
        missing = [parameter for parameter in parameters if parameter not in material]
        if missing:
            raise ValueError(f"Case {self.name!r} does not define {missing}.")


@dataclass(frozen=True, kw_only=True)
class MaterialFileCase(HeterogeneousCase):
    """Heterogeneous case whose material is read from files.

    Attributes
    ----------
    material_files : dict of str to str
        SEG-Y or HDF5 file of each material parameter, keyed like
        :data:`MATERIAL_FILE_SUFFIXES`.
    """

    material_files: dict[str, str]

    def write_material_files(self, parameters: tuple[str, ...]) -> dict[str, str]:
        """Return the given file of each material parameter.

        Parameters
        ----------
        parameters : tuple of str
            Material parameters the wave physics reads.

        Returns
        -------
        dict
            File name keyed by material parameter.

        Raises
        ------
        ValueError
            If the case does not define one of ``parameters``.
        """
        self._check_defines(parameters, self.material_files)
        return {parameter: self.material_files[parameter] for parameter in parameters}

    def minimum(self, parameter: str) -> float:
        """Return the minimum of a material file, in km/s for velocities.

        Parameters
        ----------
        parameter : str
            Material parameter, e.g. ``"p_wave_velocity"``.

        Returns
        -------
        float
            Minimum of the file values. Values above
            :data:`METERS_PER_SECOND_THRESHOLD` are taken as m/s.

        Raises
        ------
        ValueError
            If the case does not define ``parameter``.
        """
        self._check_defines((parameter,), self.material_files)
        file_name = self.material_files[parameter]
        if file_name.endswith(".segy"):
            values, _, _ = read_segy_velocity_model(file_name)
        else:
            with h5py.File(file_name, "r") as f:
                values = np.asarray(f["velocity_model"][()])
        minimum = float(np.min(values))
        if minimum > METERS_PER_SECOND_THRESHOLD:
            # The interpolator converts such files to km/s
            minimum /= 1000.0
        return minimum


@dataclass(frozen=True, kw_only=True)
class MaterialModelCase(HeterogeneousCase):
    """Heterogeneous case whose material is a function of the coordinates.

    The material is sampled on a grid and written to SEG-Y files named
    ``"<name>_vp.segy"``, ``"<name>_vs.segy"`` and ``"<name>_rho.segy"``
    when the model dictionary is built. Subclasses implement
    :meth:`material`.

    Attributes
    ----------
    grid_spacing : float
        Spacing of the sampling grid. Default is
        :data:`MATERIAL_GRID_SPACING`.
    """

    grid_spacing: float = MATERIAL_GRID_SPACING

    @abstractmethod
    def material(self, z: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate the material at the given coordinates.

        Parameters
        ----------
        z : np.ndarray
            Vertical coordinates, negative below the surface.
        x : np.ndarray
            Horizontal coordinates, broadcastable against ``z``.

        Returns
        -------
        dict
            Array per material parameter, keyed like
            :data:`MATERIAL_FILE_SUFFIXES`.
        """

    def sample_material(self) -> dict[str, np.ndarray]:
        """Sample :meth:`material` on the case grid.

        Returns
        -------
        dict
            Array per material parameter, shaped ``(nz, nx)``. Rows go from
            ``z = -length_z`` up to the surface and columns along x.
        """
        z = np.linspace(
            -self.length_z, 0.0, round(self.length_z / self.grid_spacing) + 1
        )
        x = np.linspace(
            0.0, self.length_x, round(self.length_x / self.grid_spacing) + 1
        )
        return self.material(z[:, np.newaxis], x[np.newaxis, :])

    def write_material_files(self, parameters: tuple[str, ...]) -> dict[str, str]:
        """Sample the material and write one SEG-Y per parameter.

        The files are written by global rank 0, and every rank waits for
        them.

        Parameters
        ----------
        parameters : tuple of str
            Material parameters the wave physics reads.

        Returns
        -------
        dict
            File name keyed by material parameter.

        Raises
        ------
        ValueError
            If the case does not define one of ``parameters``.
        """
        file_names = {
            parameter: f"{self.name}{MATERIAL_FILE_SUFFIXES[parameter]}.segy"
            for parameter in parameters
        }
        # Raised on every rank, as the other ranks wait for rank 0's result
        self._check_defines(parameters, _write_material_segys(self, file_names))
        return file_names

    def minimum(self, parameter: str) -> float:
        """Return the minimum of a material parameter over the case grid.

        Parameters
        ----------
        parameter : str
            Material parameter, e.g. ``"s_wave_velocity"``.

        Returns
        -------
        float
            Minimum of the sampled parameter.

        Raises
        ------
        ValueError
            If the case does not define ``parameter``.
        """
        material = self.sample_material()
        self._check_defines((parameter,), material)
        return float(np.min(material[parameter]))


@SpyroEnsemble.run_in_one_core_and_broadcast
def _write_material_segys(
    case: MaterialModelCase, file_names: dict[str, str]
) -> dict[str, None]:
    """Sample a case material and write the SEG-Y of each parameter.

    Runs on global rank 0 only, and broadcasts which parameters the case
    defines, so every rank can raise if one is missing.

    Parameters
    ----------
    case : MaterialModelCase
        Case whose material is written.
    file_names : dict of str to str
        File name keyed by material parameter.

    Returns
    -------
    dict
        Key per parameter the material defines, written if it is in
        ``file_names``. Nothing is written if one of ``file_names`` is
        missing.
    """
    material = case.sample_material()
    if file_names.keys() <= material.keys():
        for parameter, file_name in file_names.items():
            # This layout is the one create_segy_from_grid(rotate=True)
            # round-trips through the velocity interpolator
            create_segy_from_grid(
                np.asarray(material[parameter], dtype=np.float32),
                file_name,
                rotate=True,
            )
    return dict.fromkeys(material)
