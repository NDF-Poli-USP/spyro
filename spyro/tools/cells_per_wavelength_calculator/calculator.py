"""Find the minimum cells-per-wavelength meshing parameter for a wave physics."""

import copy
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

import spyro
from .line_search import LINE_SEARCHES, MAX_CPW
from .threshold_case import _parse_threshold_case
from ..error_measure import MeasureError
from .input_models import (
    create_acoustic_model_for_meshing_parameter_2D_heterogeneous,
    create_acoustic_model_for_meshing_parameter_2D_homogeneous,
    create_elastic_model_for_meshing_parameter_2D_homogeneous,
)
from ...io.time_io import interpolate_time_series
from ...mpi.spyro_mpi import SpyroEnsemble
from ...solvers.wave import Wave
from ...utils.error_management import validate_enum
from ...utils.typing import (
    CpwSearchDirection,
    TimeStepCalculationType,
    VelocityProfileType,
    WaveType,
)


@dataclass(frozen=True)
class CpwEvaluation:
    """One forward solve of the cells-per-wavelength search.

    Attributes
    ----------
    cpw : float
        Cells-per-wavelength used to build the mesh.
    dt : float
        Time step used in the forward solve.
    error : float
        Receiver error against the reference solution.
    runtime : float
        Wall-clock time of the forward solve, in seconds.
    """

    cpw: float
    dt: float
    error: float
    runtime: float


class CpwPhysics(ABC):
    """Wave-physics-specific part of the cells-per-wavelength calculator.

    A new physics is supported by subclassing this and registering it in
    :data:`CPW_PHYSICS` under its :class:`~spyro.utils.typing.WaveType`.

    Attributes
    ----------
    parameters_dictionary : dict
        The calculator parameters dictionary.
    wave_class : type[Wave]
        Wave solver class built for every forward solve.
    has_displacement : bool
        Whether receiver data has a trailing direction axis.
    shift_analytical_reference : bool
        Whether the analytical reference is shifted one solver step when
        resampled, to match the receiver time convention.
    """

    wave_class: type[Wave]
    has_displacement: bool
    shift_analytical_reference: bool

    def __init__(self, parameters_dictionary: dict) -> None:
        """Store the parameters dictionary.

        Parameters
        ----------
        parameters_dictionary : dict
            The calculator parameters dictionary.
        """
        self.parameters_dictionary = parameters_dictionary

    @property
    @abstractmethod
    def wavelength_velocity(self) -> float:
        """Velocity whose wavelength the cells-per-wavelength refers to.

        Returns
        -------
        float
            The minimum propagation velocity of the physics.
        """

    @abstractmethod
    def build_dictionary(
        self, calculator: "MeshingParameterCalculator", method: str, degree: int
    ) -> dict:
        """Build the model dictionary of a wave object.

        Parameters
        ----------
        calculator : MeshingParameterCalculator
            The calculator, providing the shared search settings.
        method : str
            Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
        degree : int
            Spatial polynomial degree.

        Returns
        -------
        dict
            Spyro model dictionary for :attr:`wave_class`.
        """

    @abstractmethod
    def setup_homogeneous_wave(self, wave: Wave, edge_length: float) -> None:
        """Build the mesh and the constant material of a homogeneous model.

        Parameters
        ----------
        wave : Wave
            Wave object to set up.
        edge_length : float
            Mesh edge length.
        """

    @abstractmethod
    def analytical_solution(self, wave: Wave) -> np.ndarray:
        """Compute the analytical receiver data of the homogeneous model.

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


class AcousticCpwPhysics(CpwPhysics):
    """Isotropic acoustic physics for the cells-per-wavelength calculator."""

    wave_class = spyro.AcousticWave
    has_displacement = False
    shift_analytical_reference = True

    @property
    def wavelength_velocity(self) -> float:
        """Minimum acoustic velocity in the domain.

        Returns
        -------
        float
            The ``"minimum_velocity_in_the_domain"`` parameter.
        """
        return self.parameters_dictionary["minimum_velocity_in_the_domain"]

    def build_dictionary(
        self, calculator: "MeshingParameterCalculator", method: str, degree: int
    ) -> dict:
        """Build the acoustic model dictionary.

        Parameters
        ----------
        calculator : MeshingParameterCalculator
            The calculator, providing the shared search settings.
        method : str
            Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
        degree : int
            Spatial polynomial degree.

        Returns
        -------
        dict
            Spyro model dictionary for :class:`spyro.AcousticWave`.

        Raises
        ------
        NotImplementedError
            For 3D models.
        """
        if calculator.dimension != 2:
            raise NotImplementedError("Acoustic cpw calculation only supports 2D.")

        if calculator.velocity_profile_type is VelocityProfileType.HOMOGENEOUS:
            return create_acoustic_model_for_meshing_parameter_2D_homogeneous(
                method=method,
                degree=degree,
                frequency=calculator.source_frequency,
                velocity=self.wavelength_velocity,
                reduced=calculator.reduced_obj_for_testing,
            )
        return create_acoustic_model_for_meshing_parameter_2D_heterogeneous(
            method=method,
            degree=degree,
            frequency=calculator.source_frequency,
            minimum_velocity=self.wavelength_velocity,
            length_z=self.parameters_dictionary["length_z"],
            length_x=self.parameters_dictionary["length_x"],
            velocity_model_file_name=calculator.velocity_model_file_name,
            cells_per_wavelength=calculator.cpw_initial,
        )

    def setup_homogeneous_wave(self, wave: Wave, edge_length: float) -> None:
        """Build a periodic mesh and set a constant velocity.

        Parameters
        ----------
        wave : Wave
            Wave object to set up.
        edge_length : float
            Mesh edge length.
        """
        wave.set_mesh(
            input_mesh_parameters={"edge_length": edge_length, "periodic": True}
        )
        wave.set_initial_velocity_model(constant=self.wavelength_velocity)

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
        velocity = self.wavelength_velocity
        num_t = round(wave.final_time / wave.dt) + 1
        analytical = np.zeros((num_t, wave.number_of_receivers))
        source = np.asarray(wave.source_locations[0])
        for i, receiver in enumerate(wave.receiver_locations):
            offset = np.linalg.norm(np.asarray(receiver) - source)
            analytical[:, i] = spyro.utils.nodal_homogeneous_analytical(
                wave, offset, velocity
            )
        return analytical


class IsotropicElasticCpwPhysics(CpwPhysics):
    """Isotropic elastic physics for the cells-per-wavelength calculator.

    The parameters dictionary must hold ``"p_wave_velocity"``,
    ``"s_wave_velocity"`` and ``"density"``. The wavelength is the
    S-wavelength.
    """

    wave_class = spyro.IsotropicWave
    has_displacement = True
    # The elastic analytical solution is already shifted one reference step
    shift_analytical_reference = False

    @property
    def wavelength_velocity(self) -> float:
        """S-wave velocity, the slowest elastic body wave.

        Returns
        -------
        float
            The ``"s_wave_velocity"`` parameter.
        """
        return self.parameters_dictionary["s_wave_velocity"]

    def build_dictionary(
        self, calculator: "MeshingParameterCalculator", method: str, degree: int
    ) -> dict:
        """Build the isotropic elastic model dictionary.

        Parameters
        ----------
        calculator : MeshingParameterCalculator
            The calculator, providing the shared search settings.
        method : str
            Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
        degree : int
            Spatial polynomial degree.

        Returns
        -------
        dict
            Spyro model dictionary for :class:`spyro.IsotropicWave`.

        Raises
        ------
        NotImplementedError
            For heterogeneous or 3D models.
        """
        if calculator.velocity_profile_type is not VelocityProfileType.HOMOGENEOUS:
            raise NotImplementedError(
                "Elastic cpw calculation only supports homogeneous models."
            )
        if calculator.dimension != 2:
            raise NotImplementedError("Elastic cpw calculation only supports 2D.")

        frequency = calculator.source_frequency
        dt = calculator.fixed_timestep
        if dt is None:
            # Fine enough that interpolating onto the solver dt is negligible
            dt = 1.0 / (500.0 * frequency)

        return create_elastic_model_for_meshing_parameter_2D_homogeneous(
            method=method,
            degree=degree,
            frequency=frequency,
            p_wave_velocity=self.parameters_dictionary["p_wave_velocity"],
            s_wave_velocity=self.wavelength_velocity,
            density=self.parameters_dictionary["density"],
            dt=dt,
            reduced=calculator.reduced_obj_for_testing,
        )

    def setup_homogeneous_wave(self, wave: Wave, edge_length: float) -> None:
        """Build a periodic mesh; the material comes from the dictionary.

        Parameters
        ----------
        wave : Wave
            Wave object to set up.
        edge_length : float
            Mesh edge length.
        """
        wave.set_mesh(
            input_mesh_parameters={"edge_length": edge_length, "periodic": True}
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
                p_wave_velocity=self.parameters_dictionary["p_wave_velocity"],
                s_wave_velocity=self.wavelength_velocity,
                density=self.parameters_dictionary["density"],
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


#: Cells-per-wavelength physics, selected by the ``"wave_type"`` parameter.
CPW_PHYSICS: dict[WaveType, type[CpwPhysics]] = {
    WaveType.ISOTROPIC_ACOUSTIC: AcousticCpwPhysics,
    WaveType.ISOTROPIC_ELASTIC: IsotropicElasticCpwPhysics,
}


class MeshingParameterCalculator:
    """Calculate the minimum meshing parameter C (cells-per-wavelength).

    The physics is chosen by the optional ``"wave_type"`` parameter, a
    :class:`~spyro.utils.typing.WaveType` defaulting to
    ``WaveType.ISOTROPIC_ACOUSTIC``. See :data:`CPW_PHYSICS`.

    Attributes
    ----------
    parameters_dictionary : dict
        All the parameters needed for the calculation.
    wave_type : WaveType
        The wave physics being evaluated.
    physics : CpwPhysics
        The physics-specific part of the calculation.
    source_frequency : float
        Source (Ricker peak) frequency in Hz.
    mesh_frequency_factor : float
        Ratio of the meshing frequency to ``source_frequency``.
    mesh_frequency : float
        Frequency in Hz whose wavelength every cells-per-wavelength value
        refers to, ``mesh_frequency_factor * source_frequency``.
    minimum_velocity : float
        Velocity whose wavelength the cpw refers to, in km/s. The minimum
        velocity for acoustic, the S-wave velocity for elastic.
    velocity_profile_type : VelocityProfileType
        Homogeneous (analytical reference) or heterogeneous (numerical
        reference).
    velocity_model_file_name : str or None
        Velocity model file of a heterogeneous model.
    FEM_method_to_evaluate : str
        Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
    dimension : int
        Spatial dimension, 2 or 3.
    receiver_setup : str
        Receiver setup, ``"near"``, ``"line"`` or ``"far"``.
    accepted_error_threshold : float
        Largest receiver error accepted. Given directly, or the error of
        ``threshold_case``.
    threshold_case : ThresholdCase or None
        Discretization whose error sets ``accepted_error_threshold``, or
        None if the threshold is given directly.
    threshold_case_evaluation : CpwEvaluation or None
        The forward solve of ``threshold_case``.
    desired_degree : int
        Polynomial degree the cpw is calculated for.
    reference_degree : int or None
        Polynomial degree of a heterogeneous numerical reference.
    cpw_reference : float or None
        Cells-per-wavelength of a heterogeneous numerical reference.
    cpw_initial : float
        Starting cells-per-wavelength. Must meet the error threshold for a
        backward search and exceed it for a forward search.
    cpw_accuracy : float
        Resolution of the cells-per-wavelength result.
    cpw_max : float
        Largest cells-per-wavelength evaluated.
    search_direction : CpwSearchDirection
        Direction of the line search run by :meth:`find_minimum`.
    reduced_obj_for_testing : bool
        Use a smaller receiver set, for testing.
    save_reference : bool
        Save the reference solution to ``reference_solution.npy``.
    load_reference : bool
        Load the reference solution from ``"reference_solution_file"``.
    timestep_calculation : TimeStepCalculationType
        How the time step of each forward solve is chosen.
    fixed_timestep : float or None
        Time step when ``timestep_calculation`` is ``FIXED``.
    search_history : list of CpwEvaluation
        Every forward solve of the last :meth:`find_minimum` call.
    initial_dictionary : dict
        Model dictionary of the initial guess object.
    initial_guess_object : Wave
        Wave object built from ``initial_dictionary``.
    comm : SpyroEnsemble
        The MPI communicator.
    reference_solution : np.ndarray
        Receiver data of the reference, sampled at the initial object's dt.
    """

    def __init__(self, parameters_dictionary: dict) -> None:
        """Initialize the calculator and compute its reference solution.

        Parameters
        ----------
        parameters_dictionary : dict
            All the parameters needed for the calculation:

            - ``"wave_type"``: WaveType, optional. Default acoustic.
            - ``"source_frequency"``: float.
            - ``"mesh_frequency_factor"``: float, optional. The wavelength
              of every cells-per-wavelength value is taken at this multiple
              of ``"source_frequency"``, e.g. 2.0 (about 90% of the Ricker
              energy) or 2.5 (close to its maximum frequency). Default is
              1.0, the peak frequency.
            - ``"minimum_velocity_in_the_domain"``: float, acoustic only.
            - ``"p_wave_velocity"``, ``"s_wave_velocity"``, ``"density"``:
              float, elastic only.
            - ``"velocity_profile_type"``: ``"homogeneous"`` or
              ``"heterogeneous"``.
            - ``"velocity_model_file_name"``: str or None.
            - ``"FEM_method_to_evaluate"``: str.
            - ``"dimension"``: int.
            - ``"receiver_setup"``: str.
            - ``"accepted_error_threshold"``: float. Exactly one of this
              and ``"threshold_case"`` must be given.
            - ``"threshold_case"``: True, dict or ThresholdCase, optional.
              Use the error of this discretization as the threshold. True
              is a degree 4 ``"spectral_quadrilateral"`` at cpw 2.0, and a
              dict overrides fields of :class:`~.threshold_case.ThresholdCase`.
            - ``"desired_degree"``: int.
            - ``"reference_degree"``, ``"C_reference"``: heterogeneous
              reference settings.
            - ``"C_initial"``, ``"C_accuracy"``: float, search settings.
            - ``"C_max"``: float, optional. Largest cells-per-wavelength
              evaluated. Default is :data:`MAX_CPW`.
            - ``"search_direction"``: ``"backward"`` (default) or
              ``"forward"``.
            - ``"time-step_calculation"``: ``"exact"`` (default),
              ``"estimate"`` or ``"float"``, with ``"time-step"`` for the
              latter.
            - ``"testing"``, ``"save_reference"``, ``"load_reference"``,
              ``"reference_solution_file"``: optional.

        Raises
        ------
        ValueError
            If ``"mesh_frequency_factor"`` is not positive, or both or
            neither of ``"accepted_error_threshold"`` and
            ``"threshold_case"`` are given.
        """
        self.parameters_dictionary = parameters_dictionary
        self.wave_type = validate_enum(
            "wave_type",
            parameters_dictionary.get("wave_type", WaveType.ISOTROPIC_ACOUSTIC),
            WaveType,
        )
        if self.wave_type not in CPW_PHYSICS:
            raise NotImplementedError(
                f"No cells-per-wavelength physics for {self.wave_type.name}."
            )
        self.physics = CPW_PHYSICS[self.wave_type](parameters_dictionary)

        self.source_frequency = parameters_dictionary["source_frequency"]
        self.mesh_frequency_factor = parameters_dictionary.get(
            "mesh_frequency_factor", 1.0
        )
        if self.mesh_frequency_factor <= 0.0:
            raise ValueError(
                "mesh_frequency_factor must be positive, got "
                f"{self.mesh_frequency_factor}."
            )
        self.mesh_frequency = self.mesh_frequency_factor * self.source_frequency
        self.minimum_velocity = self.physics.wavelength_velocity
        self.velocity_profile_type = validate_enum(
            "velocity_profile_type",
            parameters_dictionary["velocity_profile_type"],
            VelocityProfileType,
        )
        self.velocity_model_file_name = parameters_dictionary[
            "velocity_model_file_name"
        ]
        self._check_velocity_profile_type()
        self.FEM_method_to_evaluate = parameters_dictionary["FEM_method_to_evaluate"]
        self.dimension = parameters_dictionary["dimension"]
        self.receiver_setup = parameters_dictionary["receiver_setup"]
        self.accepted_error_threshold = parameters_dictionary.get(
            "accepted_error_threshold"
        )
        self.threshold_case = _parse_threshold_case(
            parameters_dictionary.get("threshold_case")
        )
        if (self.accepted_error_threshold is None) == (self.threshold_case is None):
            raise ValueError(
                "Give exactly one of accepted_error_threshold and threshold_case."
            )
        self.threshold_case_evaluation: CpwEvaluation | None = None
        self.desired_degree = parameters_dictionary["desired_degree"]

        # Only for use in heterogeneous models
        self.reference_degree = parameters_dictionary["reference_degree"]
        self.cpw_reference = parameters_dictionary["C_reference"]

        # Search parameters
        self.cpw_initial = parameters_dictionary["C_initial"]
        self.cpw_accuracy = parameters_dictionary["C_accuracy"]
        self.cpw_max = parameters_dictionary.get("C_max", MAX_CPW)
        self.search_direction = validate_enum(
            "search_direction",
            parameters_dictionary.get("search_direction", CpwSearchDirection.BACKWARD),
            CpwSearchDirection,
        )
        self.search_history: list[CpwEvaluation] = []

        # Debugging and testing parameters
        self.reduced_obj_for_testing = parameters_dictionary.get("testing", False)
        self.save_reference = parameters_dictionary.get("save_reference", False)
        self.load_reference = parameters_dictionary.get("load_reference", False)

        self.timestep_calculation = validate_enum(
            "time-step_calculation",
            parameters_dictionary.get(
                "time-step_calculation", TimeStepCalculationType.EXACT
            ),
            TimeStepCalculationType,
        )
        self.fixed_timestep = None
        if self.timestep_calculation is TimeStepCalculationType.FIXED:
            self.fixed_timestep = parameters_dictionary["time-step"]

        self.initial_dictionary = self.physics.build_dictionary(
            self, self.FEM_method_to_evaluate, self.desired_degree
        )
        self.initial_guess_object = self.physics.wave_class(
            copy.deepcopy(self.initial_dictionary)
        )
        self.comm = self.initial_guess_object.comm
        self.reference_solution = self.get_reference_solution()
        if self.threshold_case is not None:
            self.accepted_error_threshold = self._calculate_threshold()

    def _check_velocity_profile_type(self) -> None:
        """Check the velocity model inputs against the profile type.

        Raises
        ------
        ValueError
            If a homogeneous model names a velocity file, or a heterogeneous
            model lacks one or its domain lengths.
        """
        if self.velocity_profile_type is VelocityProfileType.HOMOGENEOUS:
            if self.velocity_model_file_name is not None:
                raise ValueError(
                    "Velocity model file name should be None for homogeneous models"
                )
        else:
            self._check_heterogeneous_mesh_lengths()
            if self.velocity_model_file_name is None:
                raise ValueError(
                    "Velocity model file name should be defined for heterogeneous models"
                )

    def _check_heterogeneous_mesh_lengths(self) -> None:
        """Check the domain lengths of a heterogeneous model.

        Raises
        ------
        ValueError
            If a length is missing, or ``length_x`` is negative.
        """
        parameters = self.parameters_dictionary
        if parameters.get("length_z") is None:
            raise ValueError("Length in z direction not defined")
        if parameters.get("length_x") is None:
            raise ValueError("Length in x direction not defined")
        if parameters["length_z"] < 0.0:
            parameters["length_z"] = abs(parameters["length_z"])
        if parameters["length_x"] < 0.0:
            raise ValueError("Length in x direction must be positive")

    def get_reference_solution(self) -> np.ndarray:
        """Load, compute numerically or compute analytically the reference.

        Returns
        -------
        np.ndarray
            The reference receiver data.
        """
        if self.load_reference:
            filename = self.parameters_dictionary.get(
                "reference_solution_file", "reference_solution.npy"
            )
            return np.load(filename)

        if self.velocity_profile_type is VelocityProfileType.HETEROGENEOUS:
            reference = self.calculate_reference_solution()
        else:
            reference = self.calculate_analytical_solution()

        if self.save_reference:
            np.save("reference_solution.npy", reference)
        return reference

    def calculate_reference_solution(self) -> np.ndarray:
        """Compute the numerical reference of a heterogeneous model.

        Uses the ``C_reference`` and ``reference_degree`` parameters.

        Returns
        -------
        np.ndarray
            The reference receiver data.
        """
        wave = self.build_current_object(
            self.cpw_reference, degree=self.reference_degree
        )
        wave.forward_solve()
        return wave.forward_solution_receivers

    def calculate_analytical_solution(self) -> np.ndarray:
        """Compute the analytical reference of a homogeneous model.

        Returns
        -------
        np.ndarray
            The reference receiver data, at the initial object's dt.
        """
        return self.physics.analytical_solution(self.initial_guess_object)

    def build_current_object(
        self, cpw: float, degree: int | None = None, method: str | None = None
    ) -> Wave:
        """Build the wave object for a cells-per-wavelength value.

        Parameters
        ----------
        cpw : float
            Cells-per-wavelength of the mesh.
        degree : int, optional
            Polynomial degree. Default is ``desired_degree``.
        method : str, optional
            Finite element method. Default is ``FEM_method_to_evaluate``.

        Returns
        -------
        Wave
            Wave object with its mesh built.
        """
        if degree is None and method is None:
            dictionary = copy.deepcopy(self.initial_dictionary)
        else:
            dictionary = self.physics.build_dictionary(
                self,
                method if method is not None else self.FEM_method_to_evaluate,
                degree if degree is not None else self.desired_degree,
            )
        wave = self.physics.wave_class(dictionary)
        if self.velocity_profile_type is VelocityProfileType.HOMOGENEOUS:
            wavelength = self.minimum_velocity / self.mesh_frequency
            self.physics.setup_homogeneous_wave(wave, wavelength / cpw)
        else:
            # The mesher sizes cells from the source frequency, and the edge
            # length depends only on frequency * cpw
            wave.set_mesh(
                input_mesh_parameters={
                    "cells_per_wavelength": cpw * self.mesh_frequency_factor
                }
            )
        return wave

    def calculate_error(self, cpw: float) -> float:
        """Solve at a cells-per-wavelength and measure the receiver error.

        The shot record is saved to ``test_shot_record<cpw>`` and the
        result is appended to :attr:`search_history`.

        Parameters
        ----------
        cpw : float
            Cells-per-wavelength of the mesh.

        Returns
        -------
        float
            Receiver error from :meth:`MeasureError.calculate_receiver_error`.
        """
        SpyroEnsemble.print(f"Trying cells-per-wavelength = {cpw}")
        wave = self.build_current_object(cpw)
        evaluation = self._solve_and_measure(wave, cpw)
        spyro.io.save_shots(wave, file_name=f"test_shot_record{cpw}")
        self.search_history.append(evaluation)
        return evaluation.error

    def _calculate_threshold(self) -> float:
        """Measure the receiver error of :attr:`threshold_case`.

        The solve is stored in :attr:`threshold_case_evaluation`.

        Returns
        -------
        float
            The error of the threshold case, used as the threshold.
        """
        case = self.threshold_case
        SpyroEnsemble.print(
            f"Computing the error threshold from {case.method} degree "
            f"{case.degree} at cells-per-wavelength = {case.cpw}"
        )
        wave = self.build_current_object(
            case.cpw, degree=case.degree, method=case.method
        )
        self.threshold_case_evaluation = self._solve_and_measure(wave, case.cpw)
        SpyroEnsemble.print(
            f"Accepted error threshold is {self.threshold_case_evaluation.error}"
        )
        return self.threshold_case_evaluation.error

    def _solve_and_measure(self, wave: Wave, cpw: float) -> CpwEvaluation:
        """Run a forward solve and measure its error against the reference.

        The time step is set per :attr:`timestep_calculation` and the
        reference is interpolated onto it.

        Parameters
        ----------
        wave : Wave
            Wave object with its mesh built.
        cpw : float
            Cells-per-wavelength the mesh was built with.

        Returns
        -------
        CpwEvaluation
            The cpw, time step, receiver error and runtime of the solve.
        """
        wave.initialize_physical_parameters()
        self._set_time_step(wave)
        SpyroEnsemble.print(f"Maximum dt (seconds) is {wave.dt}")

        start = time.perf_counter()
        wave.forward_solve()
        runtime = time.perf_counter() - start

        reference = interpolate_time_series(
            self.reference_solution,
            wave.dt,
            initial_time=wave.initial_time,
            final_time=wave.final_time,
            time_shift=self._reference_time_shift(wave),
        )
        error = MeasureError.calculate_receiver_error(
            wave.forward_solution_receivers,
            reference,
            wave.dt,
            has_displacement=self.physics.has_displacement,
        )
        SpyroEnsemble.print(f"Error is {error}")
        return CpwEvaluation(cpw, wave.dt, error, runtime)

    def _reference_time_shift(self, wave: Wave) -> float:
        """Return the time shift applied when resampling the reference.

        Parameters
        ----------
        wave : Wave
            Wave object of the forward solve.

        Returns
        -------
        float
            One solver time step for an analytical reference of a physics
            with ``shift_analytical_reference``, otherwise 0.0.
        """
        # TODO: The receivers store the field at time (i + 1) * dt at index
        # i, one solver step ahead of their time label. Until the receiver
        # indexing is fixed in its own PR, the analytical reference is
        # sampled one solver step later to match. Remove this shift then.
        if (
            self.physics.shift_analytical_reference
            and self.velocity_profile_type is VelocityProfileType.HOMOGENEOUS
        ):
            return wave.dt
        return 0.0

    def _set_time_step(self, wave: Wave) -> None:
        """Set the forward solve time step.

        Parameters
        ----------
        wave : Wave
            Wave object with its mesh and material set.
        """
        if self.timestep_calculation is TimeStepCalculationType.FIXED:
            wave.dt = self.fixed_timestep
        else:
            wave.get_and_set_maximum_dt(
                fraction=0.5,
                estimate_max_eigenvalue=(
                    self.timestep_calculation is TimeStepCalculationType.ESTIMATE
                ),
            )

    def find_minimum(
        self,
        starting_cpw: float | None = None,
        TOL: float | None = None,
        accuracy: float | None = None,
        savetxt: bool = False,
        search_direction: CpwSearchDirection | str | None = None,
        max_cpw: float | None = None,
    ) -> float:
        """Find the minimum cells-per-wavelength below the error threshold.

        Runs :func:`~.line_search.backward_line_search` or
        :func:`~.line_search.forward_line_search`
        from ``starting_cpw``.

        Parameters
        ----------
        starting_cpw : float, optional
            Starting cells-per-wavelength, which must meet the threshold
            for a backward search and exceed it for a forward search.
            Default is ``cpw_initial``.
        TOL : float, optional
            Largest accepted receiver error. Default is
            ``accepted_error_threshold``.
        accuracy : float, optional
            Resolution of the result. Default is ``cpw_accuracy``.
        savetxt : bool, optional
            Save the search history to ``p<degree>_cpw_results.txt``.
            Default is False.
        search_direction : CpwSearchDirection or str, optional
            Direction of the line search. Default is ``search_direction``.
        max_cpw : float, optional
            Largest cells-per-wavelength evaluated. Default is ``cpw_max``.

        Returns
        -------
        float
            The smallest evaluated cells-per-wavelength meeting ``TOL``.
        """
        if starting_cpw is None:
            starting_cpw = self.cpw_initial
        if TOL is None:
            TOL = self.accepted_error_threshold
        if accuracy is None:
            accuracy = self.cpw_accuracy
        if search_direction is None:
            search_direction = self.search_direction
        search_direction = validate_enum(
            "search_direction", search_direction, CpwSearchDirection
        )
        if max_cpw is None:
            max_cpw = self.cpw_max
        line_search = LINE_SEARCHES[search_direction]

        SpyroEnsemble.print(f"Starting {search_direction} line search")
        self.search_history = []
        cpw = line_search(
            self.calculate_error, starting_cpw, TOL, accuracy, max_cpw=max_cpw
        )

        if savetxt:
            np.savetxt(
                f"p{self.initial_guess_object.degree}_cpw_results.txt",
                [(e.cpw, e.dt, e.error, e.runtime) for e in self.search_history],
                header="cpw dt error runtime",
            )
        return cpw
