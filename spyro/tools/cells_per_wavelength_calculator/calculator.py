"""Find the minimum cells-per-wavelength meshing parameter of a wave problem."""

import copy
import time
from dataclasses import dataclass

import numpy as np

import spyro
from .line_search import LINE_SEARCHES, MAX_CPW
from .problems import CPW_PROBLEMS
from .threshold_discretization import _parse_threshold_discretization
from ..error_measure import MeasureError
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


class MeshingParameterCalculator:
    """Calculate the minimum meshing parameter C (cells-per-wavelength).

    The wave problem is chosen by the optional ``"wave_type"`` parameter, a
    :class:`~spyro.utils.typing.WaveType` defaulting to
    ``WaveType.ISOTROPIC_ACOUSTIC``, and the ``"velocity_profile_type"``
    parameter. See :data:`~.problems.CPW_PROBLEMS`.

    Attributes
    ----------
    parameters_dictionary : dict
        All the parameters needed for the calculation.
    wave_type : WaveType
        The wave physics being evaluated.
    velocity_profile_type : VelocityProfileType
        Homogeneous (analytical reference) or heterogeneous (numerical
        reference).
    timestep_calculation : TimeStepCalculationType
        How the time step of each forward solve is chosen.
    fixed_timestep : float or None
        Time step when ``timestep_calculation`` is ``FIXED``.
    problem : CpwProblem
        The wave problem solved at every cells-per-wavelength.
    FEM_method_to_evaluate : str
        Either ``"mass_lumped_triangle"`` or ``"spectral_quadrilateral"``.
    receiver_setup : str
        Receiver setup, ``"near"``, ``"line"`` or ``"far"``.
    accepted_error_threshold : float
        Largest receiver error accepted. Given directly, or the error of
        ``threshold_discretization``.
    threshold_discretization : ThresholdDiscretization or None
        Discretization whose error sets ``accepted_error_threshold``, or
        None if the threshold is given directly.
    threshold_evaluation : CpwEvaluation or None
        The forward solve of ``threshold_discretization``.
    desired_degree : int
        Polynomial degree the cpw is calculated for.
    cpw_initial : float
        Starting cells-per-wavelength. Must meet the error threshold for a
        backward search and exceed it for a forward search.
    cpw_accuracy : float
        Resolution of the cells-per-wavelength result.
    cpw_max : float
        Largest cells-per-wavelength evaluated.
    search_direction : CpwSearchDirection
        Direction of the line search run by :meth:`find_minimum`.
    search_history : list of CpwEvaluation
        Every forward solve of the last :meth:`find_minimum` call.
    save_reference : bool
        Save the reference solution to ``reference_solution_file``.
    load_reference : bool
        Load the reference solution from ``reference_solution_file``.
    reference_solution_file : str
        File the reference is saved to or loaded from. Defaults to
        :meth:`CpwProblem.default_reference_file`, e.g.
        ``"acoustic_homogeneous_cpw_reference.npy"``.
    initial_dictionary : dict
        Model dictionary of the discretization under evaluation.
    initial_wave : Wave
        Wave object built from ``initial_dictionary``.
    comm : SpyroEnsemble
        The MPI communicator.
    reference_solution : np.ndarray
        Receiver data of the reference, sampled at the initial wave's dt.
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
            - ``"minimum_velocity_in_the_domain"``: float, homogeneous
              acoustic only.
            - ``"p_wave_velocity"``, ``"s_wave_velocity"``, ``"density"``:
              float, homogeneous elastic only.
            - ``"heterogeneous_case"``: HeterogeneousCase, heterogeneous
              only. Its domain, material and any acquisition or time axis
              overrides. See
              :class:`~.heterogeneous_case.HeterogeneousCase`.
            - ``"velocity_profile_type"``: ``"homogeneous"`` or
              ``"heterogeneous"``, optional. Default is heterogeneous when
              ``"heterogeneous_case"`` is given, else homogeneous.
            - ``"FEM_method_to_evaluate"``: str.
            - ``"dimension"``: int.
            - ``"receiver_setup"``: str.
            - ``"accepted_error_threshold"``: float. Exactly one of this
              and ``"threshold_discretization"`` must be given.
            - ``"threshold_discretization"``: True, dict or
              ThresholdDiscretization, optional. Use the error of this
              discretization as the threshold. True is a degree 4
              ``"spectral_quadrilateral"`` at cpw 2.0, and a dict overrides
              fields of
              :class:`~.threshold_discretization.ThresholdDiscretization`.
            - ``"desired_degree"``: int.
            - ``"reference_degree"``, ``"C_reference"``: heterogeneous
              only, numerical reference settings.
            - ``"C_initial"``, ``"C_accuracy"``: float, search settings.
            - ``"C_max"``: float, optional. Largest cells-per-wavelength
              evaluated. Default is :data:`MAX_CPW`.
            - ``"search_direction"``: ``"backward"`` (default) or
              ``"forward"``.
            - ``"time-step_calculation"``: ``"exact"`` (default),
              ``"estimate"`` or ``"float"``, with ``"time-step"`` for the
              latter.
            - ``"testing"``, ``"save_reference"``, ``"load_reference"``,
              ``"reference_solution_file"``: optional. The reference file
              defaults to ``"<physics>_<profile>_cpw_reference.npy"``, e.g.
              ``"acoustic_homogeneous_cpw_reference.npy"``.

        Raises
        ------
        NotImplementedError
            If no problem is registered for the wave and profile types.
        ValueError
            If ``"mesh_frequency_factor"`` is not positive, the
            heterogeneous case does not match the profile type, or both or
            neither of ``"accepted_error_threshold"`` and
            ``"threshold_discretization"`` are given.
        """
        self.parameters_dictionary = parameters_dictionary
        self.wave_type = validate_enum(
            "wave_type",
            parameters_dictionary.get("wave_type", WaveType.ISOTROPIC_ACOUSTIC),
            WaveType,
        )
        self.velocity_profile_type = validate_enum(
            "velocity_profile_type",
            parameters_dictionary.get(
                "velocity_profile_type",
                VelocityProfileType.HOMOGENEOUS
                if parameters_dictionary.get("heterogeneous_case") is None
                else VelocityProfileType.HETEROGENEOUS,
            ),
            VelocityProfileType,
        )
        problem_class = CPW_PROBLEMS.get((self.wave_type, self.velocity_profile_type))
        if problem_class is None:
            raise NotImplementedError(
                f"No cells-per-wavelength problem for {self.wave_type.name} "
                f"with a {self.velocity_profile_type.value} velocity profile."
            )

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

        self.problem = problem_class(parameters_dictionary, self.fixed_timestep)

        self.FEM_method_to_evaluate = parameters_dictionary["FEM_method_to_evaluate"]
        self.receiver_setup = parameters_dictionary["receiver_setup"]
        self.accepted_error_threshold = parameters_dictionary.get(
            "accepted_error_threshold"
        )
        self.threshold_discretization = _parse_threshold_discretization(
            parameters_dictionary.get("threshold_discretization")
        )
        if (self.accepted_error_threshold is None) == (
            self.threshold_discretization is None
        ):
            raise ValueError(
                "Give exactly one of accepted_error_threshold and "
                "threshold_discretization."
            )
        self.threshold_evaluation: CpwEvaluation | None = None
        self.desired_degree = parameters_dictionary["desired_degree"]

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
        self.save_reference = parameters_dictionary.get("save_reference", False)
        self.load_reference = parameters_dictionary.get("load_reference", False)
        self.reference_solution_file = parameters_dictionary.get(
            "reference_solution_file", self.problem.default_reference_file()
        )

        self.initial_dictionary = self.problem.model_dictionary(
            self.FEM_method_to_evaluate, self.desired_degree
        )
        self.initial_wave = self.problem.physics.wave_class(
            copy.deepcopy(self.initial_dictionary)
        )
        self.comm = self.initial_wave.comm
        self.reference_solution = self.get_reference_solution()
        if self.threshold_discretization is not None:
            self.accepted_error_threshold = self._calculate_threshold()

    def get_reference_solution(self) -> np.ndarray:
        """Load or compute the reference receiver data.

        Returns
        -------
        np.ndarray
            The reference receiver data.
        """
        if self.load_reference:
            return np.load(self.reference_solution_file)

        reference = self.problem.reference_solution(
            self.initial_wave, self.FEM_method_to_evaluate
        )

        if self.save_reference:
            np.save(self.reference_solution_file, reference)
        return reference

    def build_wave(
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
            dictionary = self.problem.model_dictionary(
                method if method is not None else self.FEM_method_to_evaluate,
                degree if degree is not None else self.desired_degree,
            )
        return self.problem.build_wave(dictionary, cpw)

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
        wave = self.build_wave(cpw)
        evaluation = self._solve_and_measure(wave, cpw)
        spyro.io.save_shots(wave, file_name=f"test_shot_record{cpw}")
        self.search_history.append(evaluation)
        return evaluation.error

    def _calculate_threshold(self) -> float:
        """Measure the receiver error of :attr:`threshold_discretization`.

        The solve is stored in :attr:`threshold_evaluation`.

        Returns
        -------
        float
            The error of the threshold discretization, used as the threshold.
        """
        discretization = self.threshold_discretization
        SpyroEnsemble.print(
            f"Computing the error threshold from {discretization.method} degree "
            f"{discretization.degree} at cells-per-wavelength = "
            f"{discretization.cpw}"
        )
        wave = self.build_wave(
            discretization.cpw,
            degree=discretization.degree,
            method=discretization.method,
        )
        self.threshold_evaluation = self._solve_and_measure(wave, discretization.cpw)
        SpyroEnsemble.print(
            f"Accepted error threshold is {self.threshold_evaluation.error}"
        )
        return self.threshold_evaluation.error

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
            time_shift=self.problem.reference_time_shift(wave),
        )
        error = MeasureError.calculate_receiver_error(
            wave.forward_solution_receivers,
            reference,
            wave.dt,
            has_displacement=self.problem.physics.has_displacement,
        )
        SpyroEnsemble.print(f"Error is {error}")
        return CpwEvaluation(cpw, wave.dt, error, runtime)

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
                f"p{self.initial_wave.degree}_cpw_results.txt",
                [(e.cpw, e.dt, e.error, e.runtime) for e in self.search_history],
                header="cpw dt error runtime",
            )
        return cpw
