"""Wave problems the cells-per-wavelength search solves.

A problem is one wave physics on one kind of velocity profile. It builds
the model dictionary, meshes a wave object for a cells-per-wavelength
value, and computes the reference receiver data the error is measured
against. A new problem is supported by subclassing
:class:`HomogeneousCpwProblem` or :class:`HeterogeneousCpwProblem` and
registering it in :data:`CPW_PROBLEMS`.
"""

from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np

import spyro
from .heterogeneous_case import HeterogeneousCase
from .input_models import (
    create_acoustic_model_for_meshing_parameter_2D_heterogeneous,
    create_acoustic_model_for_meshing_parameter_2D_homogeneous,
    create_elastic_model_for_meshing_parameter_2D_heterogeneous,
    create_elastic_model_for_meshing_parameter_2D_homogeneous,
)
from .wave_physics import ACOUSTIC, ISOTROPIC_ELASTIC, WavePhysics
from ...solvers.wave import Wave
from ...utils.typing import VelocityProfileType, WaveType

_CASE_MISMATCH = (
    "heterogeneous_case should be given for heterogeneous models and only for them."
)


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
        cells-per-wavelength refers to.
    """

    physics: WavePhysics
    velocity_profile_type: VelocityProfileType
    minimum_wavelength_velocity: float

    def __init__(
        self, parameters_dictionary: dict, fixed_timestep: float | None = None
    ) -> None:
        """Read the settings shared by every problem.

        Parameters
        ----------
        parameters_dictionary : dict
            The calculator parameters dictionary.
        fixed_timestep : float, optional
            Solver time step when it is fixed. Default is None.

        Raises
        ------
        ValueError
            If ``"mesh_frequency_factor"`` is not positive.
        NotImplementedError
            For 3D models.
        """
        self.source_frequency: float = parameters_dictionary["source_frequency"]
        self.mesh_frequency_factor: float = parameters_dictionary.get(
            "mesh_frequency_factor", 1.0
        )
        if self.mesh_frequency_factor <= 0.0:
            raise ValueError(
                "mesh_frequency_factor must be positive, got "
                f"{self.mesh_frequency_factor}."
            )
        self.mesh_frequency = self.mesh_frequency_factor * self.source_frequency
        self.dimension: int = parameters_dictionary["dimension"]
        if self.dimension != 2:
            raise NotImplementedError(
                f"{self.physics.name.capitalize()} cpw calculation only supports 2D."
            )
        self.reduced_obj_for_testing: bool = parameters_dictionary.get(
            "testing", False
        )
        self.fixed_timestep = fixed_timestep

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
    def setup_wave(self, wave: Wave, cpw: float) -> None:
        """Build the mesh, and any material not read from files.

        Parameters
        ----------
        wave : Wave
            Wave object to set up.
        cpw : float
            Cells-per-wavelength of the mesh, at the meshing frequency.
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

    def __init__(
        self, parameters_dictionary: dict, fixed_timestep: float | None = None
    ) -> None:
        """Check that no heterogeneous case is given, then read settings.

        Parameters
        ----------
        parameters_dictionary : dict
            The calculator parameters dictionary.
        fixed_timestep : float, optional
            Solver time step when it is fixed. Default is None.

        Raises
        ------
        ValueError
            If a ``"heterogeneous_case"`` is given.
        """
        if parameters_dictionary.get("heterogeneous_case") is not None:
            raise ValueError(_CASE_MISMATCH)
        super().__init__(parameters_dictionary, fixed_timestep)

    def setup_wave(self, wave: Wave, cpw: float) -> None:
        """Build a periodic mesh sized by the minimum wavelength.

        Parameters
        ----------
        wave : Wave
            Wave object to set up.
        cpw : float
            Cells-per-wavelength of the mesh, at the meshing frequency.
        """
        wavelength = self.minimum_wavelength_velocity / self.mesh_frequency
        wave.set_mesh(
            input_mesh_parameters={"edge_length": wavelength / cpw, "periodic": True}
        )

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


class HeterogeneousCpwProblem(CpwProblem):
    """Material of a heterogeneous case, with a numerical reference.

    Attributes
    ----------
    heterogeneous_case : HeterogeneousCase
        Domain, material and any acquisition or time axis overrides.
    reference_degree : int
        Polynomial degree of the numerical reference.
    cpw_reference : float
        Cells-per-wavelength of the numerical reference.
    """

    velocity_profile_type = VelocityProfileType.HETEROGENEOUS

    def __init__(
        self, parameters_dictionary: dict, fixed_timestep: float | None = None
    ) -> None:
        """Read the heterogeneous case and the reference settings.

        Parameters
        ----------
        parameters_dictionary : dict
            The calculator parameters dictionary.
        fixed_timestep : float, optional
            Solver time step when it is fixed. Default is None.

        Raises
        ------
        ValueError
            If no ``"heterogeneous_case"`` is given.
        """
        case = parameters_dictionary.get("heterogeneous_case")
        if case is None:
            raise ValueError(_CASE_MISMATCH)
        super().__init__(parameters_dictionary, fixed_timestep)
        self.heterogeneous_case: HeterogeneousCase = case
        self.reference_degree: int = parameters_dictionary["reference_degree"]
        self.cpw_reference: float = parameters_dictionary["C_reference"]

    @cached_property
    def minimum_wavelength_velocity(self) -> float:
        """Minimum velocity of the slowest wave in the case.

        Returns
        -------
        float
            The case ``minimum_velocity`` if given, else the minimum of its
            ``physics.wavelength_parameter`` material.
        """
        case = self.heterogeneous_case
        if case.minimum_velocity is not None:
            return case.minimum_velocity
        return case.minimum(self.physics.wavelength_parameter)

    def reference_solution(self, initial_wave: Wave, method: str) -> np.ndarray:
        """Solve at ``reference_degree`` and ``cpw_reference``.

        Parameters
        ----------
        initial_wave : Wave
            Wave object of the discretization under evaluation, unused.
        method : str
            Finite element method of the reference.

        Returns
        -------
        np.ndarray
            The reference receiver data.
        """
        wave = self.build_wave(
            self.model_dictionary(method, self.reference_degree), self.cpw_reference
        )
        wave.forward_solve()
        return wave.forward_solution_receivers


class AcousticHomogeneousProblem(HomogeneousCpwProblem):
    """Acoustic wave in a constant velocity.

    The velocity is the ``"minimum_velocity_in_the_domain"`` parameter.
    """

    physics = ACOUSTIC

    def __init__(
        self, parameters_dictionary: dict, fixed_timestep: float | None = None
    ) -> None:
        """Read the velocity.

        Parameters
        ----------
        parameters_dictionary : dict
            The calculator parameters dictionary.
        fixed_timestep : float, optional
            Solver time step when it is fixed. Default is None.
        """
        super().__init__(parameters_dictionary, fixed_timestep)
        self.minimum_wavelength_velocity = parameters_dictionary[
            "minimum_velocity_in_the_domain"
        ]

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


class ElasticHomogeneousProblem(HomogeneousCpwProblem):
    """Isotropic elastic wave in a constant material.

    Needs the ``"p_wave_velocity"``, ``"s_wave_velocity"`` and
    ``"density"`` parameters. The wavelength is the S-wavelength.

    Attributes
    ----------
    p_wave_velocity : float
        P-wave velocity.
    density : float
        Density.

    Notes
    -----
    The elastic analytical solution is already shifted one reference step,
    so :meth:`reference_time_shift` keeps the default of no shift.
    """

    physics = ISOTROPIC_ELASTIC

    def __init__(
        self, parameters_dictionary: dict, fixed_timestep: float | None = None
    ) -> None:
        """Read the material.

        Parameters
        ----------
        parameters_dictionary : dict
            The calculator parameters dictionary.
        fixed_timestep : float, optional
            Solver time step when it is fixed, also used as the time step of
            the analytical reference. Default is None.
        """
        super().__init__(parameters_dictionary, fixed_timestep)
        self.p_wave_velocity: float = parameters_dictionary["p_wave_velocity"]
        self.minimum_wavelength_velocity = parameters_dictionary["s_wave_velocity"]
        self.density: float = parameters_dictionary["density"]

    def model_dictionary(self, method: str, degree: int) -> dict:
        """Build the homogeneous isotropic elastic model dictionary.

        Parameters
        ----------
        method : str
            Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
        degree : int
            Spatial polynomial degree.

        Returns
        -------
        dict
            Spyro model dictionary for :class:`spyro.IsotropicWave`.
        """
        dt = self.fixed_timestep
        if dt is None:
            # Fine enough that interpolating onto the solver dt is negligible
            dt = 1.0 / (500.0 * self.source_frequency)

        return create_elastic_model_for_meshing_parameter_2D_homogeneous(
            method=method,
            degree=degree,
            frequency=self.source_frequency,
            p_wave_velocity=self.p_wave_velocity,
            s_wave_velocity=self.minimum_wavelength_velocity,
            density=self.density,
            dt=dt,
            reduced=self.reduced_obj_for_testing,
        )

    def analytical_solution(self, wave: Wave) -> np.ndarray:
        """Compute the 2D point-force analytical receiver displacement.

        Parameters
        ----------
        wave : Wave
            Wave object holding the acquisition and time axis.

        Returns
        -------
        np.ndarray
            Displacement at the receivers, shaped
            ``(n_time, n_receivers, 2)``.

        Raises
        ------
        ValueError
            If the source amplitude is not aligned with a single axis.
        """
        amplitude = np.asarray(wave.amplitude, dtype=float)
        force_directions = np.flatnonzero(amplitude)
        if len(force_directions) != 1:
            raise ValueError(
                "The analytical elastic solution needs a source amplitude "
                f"along a single axis, got {amplitude}."
            )
        force_direction = int(force_directions[0])

        num_t = round(wave.final_time / wave.dt) + 1
        analytical = np.zeros((num_t, wave.number_of_receivers, wave.dimension))
        source = np.asarray(wave.source_locations[0])
        for i, receiver in enumerate(wave.receiver_locations):
            components = spyro.utils.analytical_solution_elastic(
                source_type="force_source",
                offsets=source - np.asarray(receiver),
                p_wave_velocity=self.p_wave_velocity,
                s_wave_velocity=self.minimum_wavelength_velocity,
                density=self.density,
                amplitude=amplitude[force_direction],
                frequency=wave.frequency,
                time_delay=wave.delay,
                final_time=wave.final_time,
                dt=wave.dt,
                force_direction=force_direction,
                dimension=wave.dimension,
            )
            analytical[:, i, :] = np.column_stack(components)
        return analytical


class AcousticHeterogeneousProblem(HeterogeneousCpwProblem):
    """Acoustic wave in the P-wave velocity of a heterogeneous case.

    Attributes
    ----------
    cpw_initial : float
        The ``"C_initial"`` parameter, written to the model dictionary.
    """

    physics = ACOUSTIC

    def __init__(
        self, parameters_dictionary: dict, fixed_timestep: float | None = None
    ) -> None:
        """Read the initial cells-per-wavelength.

        Parameters
        ----------
        parameters_dictionary : dict
            The calculator parameters dictionary.
        fixed_timestep : float, optional
            Solver time step when it is fixed. Default is None.
        """
        super().__init__(parameters_dictionary, fixed_timestep)
        self.cpw_initial: float = parameters_dictionary["C_initial"]

    def model_dictionary(self, method: str, degree: int) -> dict:
        """Build the heterogeneous acoustic model dictionary.

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
        return create_acoustic_model_for_meshing_parameter_2D_heterogeneous(
            method=method,
            degree=degree,
            frequency=self.source_frequency,
            minimum_velocity=self.minimum_wavelength_velocity,
            case=self.heterogeneous_case,
            cells_per_wavelength=self.cpw_initial,
        )

    def setup_wave(self, wave: Wave, cpw: float) -> None:
        """Build a velocity-adapted mesh; the velocity comes from the file.

        Parameters
        ----------
        wave : Wave
            Wave object to set up.
        cpw : float
            Cells-per-wavelength of the mesh, at the meshing frequency.
        """
        # The mesher sizes cells from the source frequency, and the edge
        # length depends only on frequency * cpw
        wave.set_mesh(
            input_mesh_parameters={
                "cells_per_wavelength": cpw * self.mesh_frequency_factor
            }
        )


class ElasticHeterogeneousProblem(HeterogeneousCpwProblem):
    """Isotropic elastic wave in the material of a heterogeneous case.

    The case must define the P- and S-wave velocities and the density.
    """

    physics = ISOTROPIC_ELASTIC

    def model_dictionary(self, method: str, degree: int) -> dict:
        """Build the heterogeneous isotropic elastic model dictionary.

        Parameters
        ----------
        method : str
            Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
        degree : int
            Spatial polynomial degree.

        Returns
        -------
        dict
            Spyro model dictionary for :class:`spyro.IsotropicWave`.
        """
        return create_elastic_model_for_meshing_parameter_2D_heterogeneous(
            method=method,
            degree=degree,
            frequency=self.source_frequency,
            case=self.heterogeneous_case,
        )

    def setup_wave(self, wave: Wave, cpw: float) -> None:
        """Build a uniform mesh; the material comes from the model files.

        The mesh is sized by the minimum S-wavelength in the domain.

        Parameters
        ----------
        wave : Wave
            Wave object to set up.
        cpw : float
            Cells-per-wavelength of the mesh, at the meshing frequency.
        """
        wavelength = self.minimum_wavelength_velocity / self.mesh_frequency
        wave.set_mesh(input_mesh_parameters={"edge_length": wavelength / cpw})


#: Cells-per-wavelength problems, selected by the ``"wave_type"`` and
#: ``"velocity_profile_type"`` parameters.
CPW_PROBLEMS: dict[tuple[WaveType, VelocityProfileType], type[CpwProblem]] = {
    (
        WaveType.ISOTROPIC_ACOUSTIC,
        VelocityProfileType.HOMOGENEOUS,
    ): AcousticHomogeneousProblem,
    (
        WaveType.ISOTROPIC_ACOUSTIC,
        VelocityProfileType.HETEROGENEOUS,
    ): AcousticHeterogeneousProblem,
    (
        WaveType.ISOTROPIC_ELASTIC,
        VelocityProfileType.HOMOGENEOUS,
    ): ElasticHomogeneousProblem,
    (
        WaveType.ISOTROPIC_ELASTIC,
        VelocityProfileType.HETEROGENEOUS,
    ): ElasticHeterogeneousProblem,
}
