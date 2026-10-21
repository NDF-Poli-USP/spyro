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
    create_elastic_model_for_meshing_parameter_2D_homogeneous,
)
from .parameters import CpwParameters
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

    def __init__(self, parameters: CpwParameters) -> None:
        """Check that no heterogeneous case is given, then read settings.

        Parameters
        ----------
        parameters : CpwParameters
            The calculator parameters.

        Raises
        ------
        ValueError
            If a ``heterogeneous_case`` is given.
        """
        if parameters.heterogeneous_case is not None:
            raise ValueError(_CASE_MISMATCH)
        super().__init__(parameters)

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

    def __init__(self, parameters: CpwParameters) -> None:
        """Read the heterogeneous case and the reference settings.

        Parameters
        ----------
        parameters : CpwParameters
            The calculator parameters.

        Raises
        ------
        ValueError
            If no ``heterogeneous_case``, ``reference_degree`` or
            ``cpw_reference`` is given.
        """
        if parameters.heterogeneous_case is None:
            raise ValueError(_CASE_MISMATCH)
        super().__init__(parameters)
        self.heterogeneous_case: HeterogeneousCase = parameters.heterogeneous_case
        self.reference_degree: int = parameters.require("reference_degree")
        self.cpw_reference: float = parameters.require("cpw_reference")

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

        Notes
        -----
        The pressure field is saved for visualization in
        ``results/<case name>_reference_forward_outputsn0.pvd``, every
        ``output_frequency`` time steps of the model dictionary.
        """
        dictionary = self.model_dictionary(method, self.reference_degree)
        dictionary["visualization"].update(
            forward_output=True,
            forward_output_filename=(
                f"results/{self.heterogeneous_case.name}_reference_forward_output.pvd"
            ),
        )
        wave = self.build_wave(dictionary, self.cpw_reference)
        from firedrake import VTKFile
        out = VTKFile("debug.pvd")
        out.write(wave.c)
        wave.forward_solve()
        return wave.forward_solution_receivers


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


class ElasticHomogeneousProblem(HomogeneousCpwProblem):
    """Isotropic elastic wave in a constant material.

    Needs the ``p_wave_velocity``, ``s_wave_velocity`` and ``density``
    parameters. The wavelength is the S-wavelength.

    Attributes
    ----------
    p_wave_velocity : float
        P-wave velocity.
    s_wave_velocity : float
        S-wave velocity.
    density : float
        Density.

    Notes
    -----
    The elastic analytical solution is already shifted one reference step,
    so :meth:`reference_time_shift` keeps the default of no shift.
    """

    physics = ISOTROPIC_ELASTIC

    def __init__(self, parameters: CpwParameters) -> None:
        """Read the material.

        A fixed solver time step is also used as the time step of the
        analytical reference.

        Parameters
        ----------
        parameters : CpwParameters
            The calculator parameters.

        Raises
        ------
        ValueError
            If a material parameter is not given.
        """
        super().__init__(parameters)
        self.p_wave_velocity: float = parameters.require("p_wave_velocity")
        self.s_wave_velocity: float = parameters.require("s_wave_velocity")
        self.density: float = parameters.require("density")

    @property
    def minimum_wavelength_velocity(self) -> float:
        """The S-wave velocity, the slowest body wave.

        Returns
        -------
        float
            ``s_wave_velocity``.
        """
        return self.s_wave_velocity

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
        return create_elastic_model_for_meshing_parameter_2D_homogeneous(
            method=method,
            degree=degree,
            frequency=self.source_frequency,
            p_wave_velocity=self.p_wave_velocity,
            s_wave_velocity=self.s_wave_velocity,
            density=self.density,
            dt=self.fixed_timestep,
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
                s_wave_velocity=self.s_wave_velocity,
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

    By default the mesh is uniform, sized by the minimum P-wavelength in the
    domain. With
    ``velocity_adapted_mesh``, SeismicMesh triangle meshes are instead sized
    by the local P-wavelength.

    Attributes
    ----------
    velocity_adapted_mesh : bool
        Size SeismicMesh meshes by the local P-wavelength.
    """

    physics = ACOUSTIC

    def __init__(self, parameters: CpwParameters) -> None:
        """Read the heterogeneous case and the meshing choice.

        Parameters
        ----------
        parameters : CpwParameters
            The calculator parameters.
        """
        super().__init__(parameters)
        self.velocity_adapted_mesh: bool = parameters.velocity_adapted_mesh

    def setup_wave(self, wave: Wave, cpw: float) -> None:
        """Build the mesh, adapted to the P-wave velocity if requested, and load the velocity.

        The adapted mesh has, everywhere, ``cpw`` cells per local
        P-wavelength at the meshing frequency, limited by SeismicMesh's
        size gradation, which only refines. Only SeismicMesh meshes can be
        adapted, so a quadrilateral mesh stays uniform. The forward solve
        does not read the velocity file itself with a PML, so it is
        interpolated onto the mesh nodes here.

        Parameters
        ----------
        wave : Wave
            Wave object to set up.
        cpw : float
            Cells-per-wavelength of the mesh, at the meshing frequency.

        Raises
        ------
        ValueError
            If the case's P-wave velocity file is not a SEG-Y, the only
            format SeismicMesh sizes from here.
        """
        velocity_file = self.heterogeneous_case.write_material_files(
            ("p_wave_velocity",)
        )["p_wave_velocity"]
        if not self.velocity_adapted_mesh or wave.mesh_parameters.mesh_type != "SeismicMesh":
            super().setup_wave(wave, cpw)
        elif not velocity_file.endswith(".segy"):
            raise ValueError(
                f"A velocity adapted mesh needs a SEG-Y velocity, got {velocity_file!r}."
            )
        else:
            wave.set_mesh(
                input_mesh_parameters={
                    "velocity_model": velocity_file,
                    "cells_per_wavelength": cpw,
                    "minimum_velocity": self.minimum_wavelength_velocity,
                    # Meshing only, the source keeps its own frequency
                    "source_frequency": self.mesh_frequency,
                }
            )
        # Fast interpolation reads the velocity at the mesh nodes directly,
        # without building a grid mesh to project from
        wave.set_initial_velocity_model(new_file=velocity_file, fast_interpolate=True)

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
        )


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
}
