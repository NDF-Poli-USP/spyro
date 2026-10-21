"""Wave problems the cells-per-wavelength search solves.

A problem is one wave physics on one kind of velocity profile. It builds
the model dictionary, meshes a wave object for a cells-per-wavelength
value, and computes the reference receiver data the error is measured
against. A new problem is supported by subclassing
:class:`HomogeneousCpwProblem` and registering it in :data:`CPW_PROBLEMS`.
"""

from abc import ABC, abstractmethod

import numpy as np

import spyro
from .input_models import (
    create_acoustic_model_for_meshing_parameter_2D_homogeneous,
)
from .parameters import CpwParameters
from .wave_physics import ACOUSTIC, WavePhysics
from ...solvers.wave import Wave
from ...utils.typing import VelocityProfileType, WaveType


class CpwProblem(ABC):
    """Wave problem solved at every cells-per-wavelength of the search.

    Attributes
    ----------
    physics : WavePhysics
        Wave physics of the problem.
    velocity_profile_type : VelocityProfileType
        Velocity profile of the problem.
    source_frequency : float
        Ricker peak frequency.
    mesh_frequency_factor : float
        Multiple of ``source_frequency`` whose wavelength every
        cells-per-wavelength value refers to.
    mesh_frequency : float
        ``mesh_frequency_factor * source_frequency``.
    dimension : int
        Spatial dimension. Only 2 is supported.
    reduced_obj_for_testing : bool
        Use a reduced acquisition, for faster tests.
    fixed_timestep : float or None
        Solver time step when it is fixed, else None.
    minimum_wavelength_velocity : float
        Minimum velocity of the slowest wave, whose wavelength the
        cells-per-wavelength refers to. Set by each subclass.
    """

    physics: WavePhysics
    velocity_profile_type: VelocityProfileType
    minimum_wavelength_velocity: float

    def __init__(self, parameters: CpwParameters) -> None:
        """Read the settings shared by every problem.

        Parameters
        ----------
        parameters : CpwParameters
            The calculator parameters.

        Raises
        ------
        NotImplementedError
            For 3D models.
        """
        self.source_frequency: float = parameters.source_frequency
        self.mesh_frequency_factor: float = parameters.mesh_frequency_factor
        self.mesh_frequency = self.mesh_frequency_factor * self.source_frequency
        self.dimension: int = parameters.dimension
        if self.dimension != 2:
            raise NotImplementedError(
                f"{self.physics.name.capitalize()} cpw calculation only supports 2D."
            )
        self.reduced_obj_for_testing: bool = parameters.testing
        self.fixed_timestep: float | None = parameters.fixed_timestep

    def default_reference_file(self) -> str:
        """Return the default file name of a saved or loaded reference.

        Returns
        -------
        str
            ``"<physics>_<profile>_cpw_reference.npy"``, e.g.
            ``"acoustic_homogeneous_cpw_reference.npy"``.
        """
        return (
            f"{self.physics.name}_{self.velocity_profile_type.value}"
            "_cpw_reference.npy"
        )

    def build_wave(self, dictionary: dict, cpw: float) -> Wave:
        """Build a wave object and set up its mesh and material.

        Parameters
        ----------
        dictionary : dict
            Model dictionary, from :meth:`model_dictionary`.
        cpw : float
            Cells-per-wavelength of the mesh, at the meshing frequency.

        Returns
        -------
        Wave
            Wave object with its mesh built.
        """
        wave = self.physics.wave_class(dictionary)
        self.setup_wave(wave, cpw)
        return wave

    def setup_wave(self, wave: Wave, cpw: float) -> None:
        """Build a uniform mesh sized by the minimum wavelength.

        Any periodicity comes from the model dictionary. Subclasses add any
        material not read from the dictionary.

        Parameters
        ----------
        wave : Wave
            Wave object to set up.
        cpw : float
            Cells-per-wavelength of the mesh, at the meshing frequency.
        """
        wavelength = self.minimum_wavelength_velocity / self.mesh_frequency
        wave.set_mesh(input_mesh_parameters={"edge_length": wavelength / cpw})

    def reference_time_shift(self, wave: Wave) -> float:
        """Return the time shift applied when resampling the reference.

        Parameters
        ----------
        wave : Wave
            Wave object of the forward solve.

        Returns
        -------
        float
            0.0, no shift.
        """
        return 0.0

    @abstractmethod
    def model_dictionary(self, method: str, degree: int) -> dict:
        """Build the model dictionary of a wave object.

        Parameters
        ----------
        method : str
            Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
        degree : int
            Spatial polynomial degree.

        Returns
        -------
        dict
            Spyro model dictionary for ``physics.wave_class``.
        """

    @abstractmethod
    def reference_solution(self, initial_wave: Wave, method: str) -> np.ndarray:
        """Compute the reference receiver data.

        Parameters
        ----------
        initial_wave : Wave
            Wave object of the discretization under evaluation, holding the
            acquisition and time axis.
        method : str
            Finite element method under evaluation.

        Returns
        -------
        np.ndarray
            The reference receiver data.
        """


class HomogeneousCpwProblem(CpwProblem):
    """Constant material on a periodic box, with an analytical reference."""

    velocity_profile_type = VelocityProfileType.HOMOGENEOUS

    def reference_solution(self, initial_wave: Wave, method: str) -> np.ndarray:
        """Compute the analytical reference.

        Parameters
        ----------
        initial_wave : Wave
            Wave object holding the acquisition and time axis.
        method : str
            Finite element method under evaluation, unused.

        Returns
        -------
        np.ndarray
            The analytical receiver data, at ``initial_wave.dt``.
        """
        return self.analytical_solution(initial_wave)

    @abstractmethod
    def analytical_solution(self, wave: Wave) -> np.ndarray:
        """Compute the analytical receiver data.

        Parameters
        ----------
        wave : Wave
            Wave object holding the acquisition and time axis.

        Returns
        -------
        np.ndarray
            Receiver data sampled at ``wave.dt``, shaped like
            ``wave.forward_solution_receivers``.
        """


class AcousticHomogeneousProblem(HomogeneousCpwProblem):
    """Acoustic wave in a constant velocity.

    The velocity is the ``minimum_velocity`` parameter.
    """

    physics = ACOUSTIC

    def __init__(self, parameters: CpwParameters) -> None:
        """Read the velocity.

        Parameters
        ----------
        parameters : CpwParameters
            The calculator parameters.

        Raises
        ------
        ValueError
            If no ``minimum_velocity`` is given.
        """
        super().__init__(parameters)
        self.minimum_wavelength_velocity = parameters.require("minimum_velocity")

    def model_dictionary(self, method: str, degree: int) -> dict:
        """Build the homogeneous acoustic model dictionary.

        Parameters
        ----------
        method : str
            Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
        degree : int
            Spatial polynomial degree.

        Returns
        -------
        dict
            Spyro model dictionary for :class:`spyro.AcousticWave`.
        """
        return create_acoustic_model_for_meshing_parameter_2D_homogeneous(
            method=method,
            degree=degree,
            frequency=self.source_frequency,
            velocity=self.minimum_wavelength_velocity,
            reduced=self.reduced_obj_for_testing,
        )

    def setup_wave(self, wave: Wave, cpw: float) -> None:
        """Build a periodic mesh and set a constant velocity.

        Parameters
        ----------
        wave : Wave
            Wave object to set up.
        cpw : float
            Cells-per-wavelength of the mesh, at the meshing frequency.
        """
        super().setup_wave(wave, cpw)
        wave.set_initial_velocity_model(constant=self.minimum_wavelength_velocity)

    def reference_time_shift(self, wave: Wave) -> float:
        """Return one solver time step.

        Parameters
        ----------
        wave : Wave
            Wave object of the forward solve.

        Returns
        -------
        float
            ``wave.dt``.
        """
        # TODO: The receivers store the field at time (i + 1) * dt at index
        # i, one solver step ahead of their time label. Until the receiver
        # indexing is fixed in its own PR, the analytical reference is
        # sampled one solver step later to match. Remove this shift then.
        return wave.dt

    def analytical_solution(self, wave: Wave) -> np.ndarray:
        """Compute the acoustic analytical receiver data.

        Parameters
        ----------
        wave : Wave
            Wave object holding the acquisition and time axis.

        Returns
        -------
        np.ndarray
            Pressure at the receivers, shaped ``(n_time, n_receivers)``.
        """
        velocity = self.minimum_wavelength_velocity
        num_t = round(wave.final_time / wave.dt) + 1
        analytical = np.zeros((num_t, wave.number_of_receivers))
        source = np.asarray(wave.source_locations[0])
        for i, receiver in enumerate(wave.receiver_locations):
            offset = np.linalg.norm(np.asarray(receiver) - source)
            analytical[:, i] = spyro.utils.nodal_homogeneous_analytical(
                wave, offset, velocity
            )
        return analytical


#: Cells-per-wavelength problems, selected by the ``"wave_type"`` and
#: ``"velocity_profile_type"`` parameters.
CPW_PROBLEMS: dict[tuple[WaveType, VelocityProfileType], type[CpwProblem]] = {
    (
        WaveType.ISOTROPIC_ACOUSTIC,
        VelocityProfileType.HOMOGENEOUS,
    ): AcousticHomogeneousProblem,
}
