import firedrake as fire
import warnings
from scipy.optimize import minimize as scipy_minimize
from mpi4py import MPI
from pyadjoint import MinimizationProblem, TAOSolver
from pyadjoint.optimization.tao_solver import TAOConvergenceError
import numpy as np
import resource
import glob
import os

from .wave import Wave
from .acoustic_wave import AcousticWave
from ..utils import compute_functional
from ..utils import Gradient_mask_for_pml, Mask
from ..utils.typing import AdjointType, WaveType
from ..utils.physical_parameters import PhysicalParameters
from ..plots import plot_model as spyro_plot_model
from ..io.basicio import parallel_print
from ..io.basicio import load_shots, save_shots
from ..io.parallelism_wrappers import switch_serial_shot
from ..io import create_segy
from ..io.parallelism_wrappers import run_in_one_core


try:
    from ROL.firedrake_vector import FiredrakeVector as FireVector
    import ROL
    RObjective = ROL.Objective
except ImportError:
    ROL = None
    RObjective = object

# ROL = None


def get_peak_memory():
    """
    Get the peak memory usage of the current process.

    Returns
    -------
    float
        Peak memory usage in megabytes (MB).

    Notes
    -----
    This function uses resource.getrusage() to get the peak resident set size
    (ru_maxrss) and converts it from kilobytes to megabytes.
    """
    peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_memory_mb = peak_memory_kb / 1024
    return peak_memory_mb


class L2Inner(object):
    """
    DEPRECATED: L2 inner product operator for optimization.

    This class implements the L2 inner product using a mass matrix assembled
    with the quadrature rule from the wave object. It's used in ROL-based
    optimization algorithms, which are DEPRECATED in spyro

    Parameters
    ----------
    wave : AcousticWave
        Wave object containing the function space and quadrature rule.

    Attributes
    ----------
    A : firedrake.Matrix
        Mass matrix assembled with the quadrature rule.
    Ap : PETSc.Mat
        PETSc backend matrix for efficient matrix-vector operations.

    Methods
    -------
    eval(_u, _v)
        Evaluate the L2 inner product between two functions.
    """
    def __init__(self, wave):
        """
        Initialize the L2 inner product operator.

        Parameters
        ----------
        wave : AcousticWave
            Wave object containing the function space and quadrature rule.
        """
        V = wave.function_space
        dxlump = fire.dx(**wave.quadrature_rule)
        self.A = fire.assemble(
            fire.TrialFunction(V) * fire.TestFunction(V) * dxlump,
            mat_type="matfree"
        )
        self.Ap = fire.as_backend_type(self.A).mat()

    def eval(self, _u, _v):
        """
        Evaluate the L2 inner product between two functions.

        Parameters
        ----------
        _u : firedrake.Function
            First function.
        _v : firedrake.Function
            Second function.

        Returns
        -------
        float
            The L2 inner product <_u, _v>.
        """
        upet = fire.as_backend_type(_u).vec()
        vpet = fire.as_backend_type(_v).vec()
        A_u = self.Ap.createVecLeft()
        self.Ap.mult(upet, A_u)
        return vpet.dot(A_u)


class Objective(RObjective):
    """
    DEPRECATED ROL-compatible objective function for FWI.

    This class wraps the full waveform inversion objective function for use
    with the ROL (Rapid Optimization Library) optimization framework. It
    provides methods to compute the functional value, gradient, and update
    the inversion control during optimization.

    Parameters
    ----------
    inner_product : L2Inner
        Inner product operator for the optimization.
    FWI_obj : FullWaveformInversion
        Full waveform inversion object containing the problem setup.

    Attributes
    ----------
    inner_product : L2Inner
        Inner product operator.
    p_guess : None
        Placeholder for pressure guess (currently unused).
    misfit : float
        Current misfit value.
    real_shot_record : array_like
        Real/observed shot record data.
    inversion_obj : FullWaveformInversion
        Reference to the FWI object.
    comm : MPI.Comm
        MPI communicator for parallel execution.

    Methods
    -------
    value(x, tol)
        Compute the objective functional value.
    gradient(g, x, tol)
        Compute the gradient of the objective functional.
    update(x, flag, iteration)
        Update the inversion control with a new optimization iterate.
    """
    def __init__(self, inner_product, FWI_obj):
        """
        Initialize the objective function.

        Parameters
        ----------
        inner_product : L2Inner
            Inner product operator for the optimization.
        FWI_obj : FullWaveformInversion
            Full waveform inversion object containing the problem setup.

        Raises
        ------
        ImportError
            If the ROL module is not available.
        """
        if ROL is None:
            raise ImportError("The ROL module is not available.")
        ROL.Objective.__init__(self)
        self.inner_product = inner_product
        self.p_guess = None
        self.misfit = 0.0
        self.real_shot_record = FWI_obj.real_shot_record
        self.inversion_obj = FWI_obj
        self.comm = FWI_obj.comm

    def value(self, x, tol):
        """
        Compute the objective functional value.

        Parameters
        ----------
        x : FiredrakeVector
            Current control iterate.
        tol : float
            Tolerance for the computation (unused).

        Returns
        -------
        float
            The objective functional value.
        """
        J_total = np.zeros((1))
        self.inversion_obj.misfit = None
        self.inversion_obj.wave.reset_pressure()
        Jm = self.inversion_obj.get_functional()
        self.misfit = self.inversion_obj.misfit
        J_total[0] += Jm

        return J_total[0]

    def gradient(self, g, x, tol):
        """
        Compute the gradient of the objective functional.

        Parameters
        ----------
        g : FiredrakeVector
            Vector to store the gradient (modified in-place).
        x : FiredrakeVector
            Current control iterate.
        tol : float
            Tolerance for the computation (unused).
        """
        self.inversion_obj.get_gradient(calculate_functional=False)
        dJ = self.inversion_obj.gradient
        g.scale(0)
        g.vec += dJ

    def update(self, x, flag, iteration):
        """
        Update the inversion control with a new optimization iterate.

        Parameters
        ----------
        x : FiredrakeVector
            New control iterate.
        flag : int
            Update flag from ROL.
        iteration : int
            Current iteration number.
        """
        control_reference = self.inversion_obj.control_parameters
        if control_reference is None:
            raise ValueError("No guess control parameter has been configured.")
        updated_control = fire.Function(
            control_reference.function_space(),
            x.vec,
            name=control_reference.name(),
        )
        self.inversion_obj.set_guess_control(updated_control)


class FullWaveformInversion:
    """FWI driver composed around a wave solver.

    By default, the inversion driver uses :class:`AcousticWave`. The driver is
    composed around a wave solver so other wave equations can be integrated in
    the future, but FWI is currently supported only for acoustic waves because
    the adjoint solver is implemented only for :class:`AcousticWave`.

    Notes
    -----
    The inversion driver composes a wave solver instead of inheriting from one.
    Pass ``wave_class`` to construct a compatible acoustic solver, or pass
    ``wave`` to reuse an already initialized acoustic solver instance.

    The inversion can be run using either ``scipy.optimize.minimize`` (L-BFGS-B)
    via ``run_fwi()`` or the deprecated ROL library via ``run_fwi_rol()``.

    Adjoint types
    -------------
    The gradient driving the optimization comes from one of two adjoints, and
    which one is used decides how ``run_fwi()`` is run:

    Implemented adjoint (the default)
        The hand-written adjoint solver. Every optimizer iterate re-runs the
        forward solve and the backward propagator, and the control is handed
        to :func:`scipy.optimize.minimize` as a flat vector of degrees of
        freedom.
    Automated adjoint
        Algorithmic differentiation through :mod:`firedrake.adjoint`, enabled
        with :meth:`enable_automated_adjoint`. The forward solve is recorded on
        a pyadjoint tape *once*; the resulting reduced functional replays it
        for every new control, so the driver does not re-run the forward solve
        itself. The control stays a ``Function`` and the optimization is done
        by PETSc TAO.

    Methods
    -------
    enable_automated_adjoint(control_parameters=None, ...)
        Differentiate with the automated adjoint instead of the implemented
        one.
    calculate_misfit(c=None)
        Calculate the receiver-data residual for the current guess model.
    generate_real_shot_record(plot_model=False, ...)
        Generate synthetic observed data from the configured real model.
    set_real_velocity_model(constant=None, ...)
        Configure the acoustic model used to generate synthetic observations.
    set_guess_velocity_model(constant=None, ...)
        Configure the acoustic model used as the inversion starting point.
    set_real_mesh(user_mesh=None, input_mesh_parameters=None)
        Set the mesh used by the real model.
    set_guess_mesh(user_mesh=None, input_mesh_parameters=None)
        Set the mesh used by the inversion guess model.
    get_functional(c=None)
        Compute the objective functional.
    get_gradient(c=None, save=True, calculate_functional=True)
        Compute the acoustic adjoint gradient.
    return_functional_and_gradient(c)
        Return the functional and flattened gradient for scipy optimizers.
    run_fwi(**kwargs)
        Run full waveform inversion with scipy L-BFGS-B, or with PETSc TAO
        when the automated adjoint is enabled.
    run_fwi_rol(**kwargs)
        Run the deprecated ROL-based inversion path.

    Examples
    --------
    >>> fwi = FullWaveformInversion(dictionary=config_dict, comm=comm)
    >>> fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.1})
    >>> fwi.set_guess_velocity_model(constant=2.0)
    >>> fwi.load_real_shot_record("shots/observed_")
    >>> fwi.run_fwi(maxiter=50, vmin=1.5, vmax=4.5)
    """

    def __init__(
        self, dictionary=None, comm=None, wave_class=AcousticWave, wave=None
    ):
        """Initialize the full waveform inversion driver.

        Parameters
        ----------
        dictionary : dict, optional
            Model and inversion configuration used to construct ``spyro.solvers.Wave``
            class when ``wave`` is not provided.
        comm : object, optional
            Communicator passed when constructing the wave solver.
        wave_class : type, optional
            Wave solver class used when ``wave`` is not provided. The class
            must construct a :class:`spyro.solvers.Wave` with
            :attr:`WaveType.ISOTROPIC_ACOUSTIC` while FWI support is limited
            to acoustic adjoint solves.
        wave : object, optional
            Preconstructed wave solver instance. When provided, the inversion
            driver uses this instance directly and infers ``wave_class`` from
            its type. The instance must be a :class:`Wave` with
            :attr:`WaveType.ISOTROPIC_ACOUSTIC`.
        """
        if wave is not None:
            if not isinstance(wave, Wave):
                raise TypeError(
                    "wave must be an instance of Wave. "
                    f"Received {type(wave).__name__}.",
                )
            self.wave = wave
            self.wave_class = type(wave)
        else:
            self.wave_class = AcousticWave if wave_class is None else wave_class
            if (
                not isinstance(self.wave_class, type)
                or not issubclass(self.wave_class, Wave)
            ):
                raise TypeError(
                    "wave_class must be a Wave subclass. "
                    f"Received {self.wave_class}.",
                )
            self.wave = self.wave_class(dictionary=dictionary, comm=comm)
        self.wave_type = self.wave.wave_type

        self.input_dictionary = self.wave.input_dictionary
        self.comm = self.wave.comm

        default_optimization_parameters = {
            "General": {"Secant": {
                "Type": "Limited-Memory BFGS",
                "Maximum Storage": 10,
            }},
            "Step": {
                "Type": "Augmented Lagrangian",
                "Augmented Lagrangian": {
                    "Subproblem Step Type": "Line Search",
                    "Subproblem Iteration Limit": 5.0,
                },
                "Line Search": {"Descent Method": {"Type": "Quasi-Newton Step"}},
            },
            "Status Test": {
                "Gradient Tolerance": 1e-16,
                "Iteration Limit": None,
                "Step Tolerance": 1.0e-16,
            },
        }
        self.input_dictionary.setdefault("inversion", {})
        inversion_dictionary = self.input_dictionary["inversion"]
        inversion_dictionary.setdefault("initial_guess_model_file", None)
        inversion_dictionary.setdefault(
            "optimization_parameters",
            default_optimization_parameters,
        )
        inversion_dictionary.setdefault("real_shot_record_file", None)
        inversion_dictionary.setdefault("control_output_file", "fwi/control.pvd")
        inversion_dictionary.setdefault("gradient_output_file", "fwi/gradient.pvd")
        inversion_dictionary.setdefault("real_velocity_model_file", None)

        self.real_mesh = None
        self.guess_mesh = None
        self._control_parameters = PhysicalParameters()
        self._real_model_parameters = PhysicalParameters()

        self.control_out = fire.VTKFile(inversion_dictionary["control_output_file"])
        self.gradient_out = fire.VTKFile(inversion_dictionary["gradient_output_file"])
        self.real_velocity_model_file = inversion_dictionary["real_velocity_model_file"]
        self.real_shot_record = None
        self.real_shot_record_files = inversion_dictionary["real_shot_record_file"]

        self.guess_shot_record = None
        self.gradient = None
        self.control_parameter_result = None
        self.current_iteration = 0
        self.mesh_iteration = 0
        self.iteration_limit = 100
        self.inner_product = "L2"
        self.misfit = None
        self.functional = None
        self.guess_forward_solution = None
        self.has_gradient_mask = False
        self.gradient_mask_available = False
        self.functional_history = []

    def _sync_wave_real_shot_record(self):
        """Copy observed data from the FWI driver to the wave solver.

        The FWI object owns ``real_shot_record`` as inversion state, while the
        acoustic forward solver expects the same data on ``wave.real_shot_record``
        when functional evaluation happens during a forward solve. This helper
        keeps both objects synchronized before calling solver routines.

        Examples
        --------
        After ``generate_real_shot_record()``, ``self.real_shot_record`` is set
        on the driver. Calling this method makes the same array available as
        ``self.wave.real_shot_record``.
        """
        if self.real_shot_record is not None:
            self.wave.real_shot_record = self.real_shot_record

    @property
    def control_parameters(self):
        """Return the parameters this inversion is inverting for.

        The controls are held internally as a set of material parameters
        mapping each to its field, so which parameters are inverted for is
        recorded by its members and their current values by the fields. There
        is no separate record of the selection.

        How they are presented depends on the physics: an acoustic medium is
        described by its velocity model alone, so its control is that single
        ``Function``, which is the API acoustic FWI has always had. An elastic
        medium is inverted for several parameters at once and needs a
        different shape, which is not defined yet.

        Returns
        -------
        firedrake.Function or None
            For an acoustic inversion, the velocity model, or ``None`` if none
            is configured.

        Raises
        ------
        NotImplementedError
            If the wave equation is not acoustic.

        Examples
        --------
        >>> set(fwi._control_parameters) <= fwi.wave.physical_parameters
        True
        """
        if self.wave_type is not WaveType.ISOTROPIC_ACOUSTIC:
            raise NotImplementedError(
                "Inversion controls are only defined for acoustic media; "
                f"{self.wave_type.name} controls are not implemented yet.",
            )
        if not self._control_parameters:
            return None
        (field,) = self._control_parameters.values()
        return field

    def _controlled_parameters(self):
        """Return the parameters being inverted for.

        Three sources are tried in turn, each one answering the case where the
        previous is not available yet:

        1. The control set, once ``set_guess_control`` has established it.
           This is the normal case.
        2. The solver's physical parameters, when no controls have been set:
           an inversion that has not been told otherwise inverts for
           everything the wave equation is written in terms of.
        3. The parameters the solver *class* declares, when the solver has not
           built its physical parameters either. Those only exist after the
           first forward solve, but their names are known from the start, and
           they are the same set.

        Cases 2 and 3 are what lets ``set_guess_control`` work on a fresh
        inversion: it is handed a bare ``Function`` that carries no indication
        of which parameter it belongs to, and the answer has to come from
        somewhere.

        Returns
        -------
        tuple of enum.Enum
            Controlled material parameters, ordered by name. A tuple rather
            than a set because callers take the first entry to name the single
            acoustic control.

        Examples
        --------
        On an acoustic inversion, all three cases give
        ``(AcousticMaterialParameter.P_WAVE_VELOCITY,)``, since an acoustic
        medium has that one parameter and nothing else to choose from.
        """
        if self._control_parameters:
            return tuple(self._control_parameters)
        try:
            physical = self.wave.physical_parameters
        except ValueError:
            physical = type(self.wave)._physical_parameter_names
        return tuple(sorted(physical, key=lambda p: p.value))

    def _control_by_parameter(self, control):
        """Return control values keyed by the parameter each one belongs to.

        Callers pass controls in two shapes, and this is where the two meet:

        1. A **mapping**, which already says which parameter each value
           belongs to, and is used as given.
        2. A **bare value** — a ``Function``, a ``Constant``, a number — which
           says nothing about which parameter it is, so it is paired with the
           controlled parameter, of which there has to be exactly one. This is
           the acoustic API, ``set_guess_control(velocity)``, and also how the
           optimizer feeds back each iterate: ``_rebuild_control_from_vector``
           builds a ``Function`` out of a plain array of numbers, which
           carries no parameter name at all.

        ``None`` gives an empty mapping, so callers can pass "no control
        given" without a special case.

        The parameters are not checked here. Applying a control reaches
        ``PhysicalParameters.update``, which already rejects anything the wave
        equation does not model, and names the known parameters while doing
        so.

        Parameters
        ----------
        control : mapping, firedrake.Function, firedrake.Constant, scalar, UFL expression, or None
            Control values to key.

        Returns
        -------
        dict
            Control values keyed by material parameter. Empty if ``control``
            is ``None``.

        Raises
        ------
        ValueError
            If a bare value is given while several parameters are controlled,
            since there is then no way to tell which one it means.

        Examples
        --------
        On an acoustic inversion, both spellings give the same result::

            fwi.set_guess_control(velocity)
            fwi.set_guess_control(
                {AcousticMaterialParameter.P_WAVE_VELOCITY: velocity},
            )
        """
        if control is None:
            return {}
        # A mapping is told apart from a bare value by whether it has items():
        # asking is cheaper and safer than testing against a list of types the
        # caller might legitimately pass.
        try:
            items = control.items()
        except AttributeError:
            items = None
        if items is not None:
            return dict(items)
        # Unpacking, rather than taking the first entry, so that a bare value
        # on a multi-parameter inversion fails instead of being assigned to
        # whichever parameter happens to sort first.
        (parameter,) = self._controlled_parameters()
        return {parameter: control}

    def _write_parameters_into_wave(self, wave, values):
        """Write material values into a solver's physical parameters.

        This is the one place where an inversion value becomes a physical
        parameter. The solver is only asked to set a parameter of its own
        equation; it is never told where the value came from, which is why
        this serves both the controls being optimized and the true model that
        generates the observed data.

        Parameters
        ----------
        wave : Wave
            Solver to update.
        values : mapping
            Values keyed by material parameter.

        Raises
        ------
        ValueError
            If the solver has no fields yet and several parameters are
            controlled. Only the acoustic velocity model can be created here.
        """
        try:
            parameters = wave.physical_parameters
        except ValueError:
            # A value is applied by writing into the field of the parameter
            # it belongs to, so a solver that has never been given a model has
            # nothing to write into. Create that field, empty, through the
            # solver's own public model API; the loop below fills it in, the
            # same way it writes every later iterate.
            (parameter,) = self._controlled_parameters()
            field = fire.Function(
                self._control_function_space(wave), name=parameter.value,
            )
            wave.set_initial_velocity_model(velocity_model_function=field)
            parameters = wave.initialize_physical_parameters()
        for name, value in values.items():
            parameters.update(name, value)

    def _control_function_space(self, wave=None):
        """Return the function space FWI controls live in.

        Returns
        -------
        firedrake.FunctionSpace
            The solver function space, built if it does not exist yet. FWI
            is acoustic-only, so this is the space of the velocity model.
        """
        wave = self.wave if wave is None else wave
        if wave.function_space is None:
            wave.force_rebuild_function_space()
        return wave.function_space

    def _copy_parameters_from_wave(self, wave):
        """Copy the controlled parameters' current values out of a solver.

        The counterpart of :meth:`_write_parameters_into_wave`. Only the
        parameters being inverted for are copied, whether the solver holds a
        guess model or the true one.

        Parameters
        ----------
        wave : Wave
            Solver to read the physical parameters from. They are built
            first if they do not exist yet.

        Returns
        -------
        PhysicalParameters
            Independent copies of the controlled parameter fields.
        """
        try:
            parameters = wave.physical_parameters
        except ValueError:
            parameters = wave.initialize_physical_parameters()
        return parameters.copy(self._controlled_parameters())

    def _flatten_control(self, control):
        """Flatten a control ``Function`` into an optimizer vector.

        Parameters
        ----------
        control : firedrake.Function
            Control parameter to flatten.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of control degrees of freedom.

        Raises
        ------
        ValueError
            If ``control`` is ``None``.
        TypeError
            If ``control`` is not a Firedrake ``Function``.
        """
        if control is None:
            raise ValueError("No control parameter has been configured.")
        if not isinstance(control, fire.Function):
            raise TypeError(
                "FWI control must be a firedrake Function. "
                f"Received {type(control).__name__}.",
            )
        return np.asarray(control.dat.data_ro, dtype=float).reshape(-1)

    def _rebuild_control_from_vector(self, control_reference, flat_vector):
        """Rebuild a control ``Function`` from an optimizer vector.

        Inverse of :meth:`_flatten_control`.

        Parameters
        ----------
        control_reference : firedrake.Function
            Control function used as the reconstruction reference. Its function
            space, name, and data shape define how the optimizer vector is
            converted back into a Firedrake ``Function``.
        flat_vector : array_like
            Optimizer vector.

        Returns
        -------
        firedrake.Function
            Rebuilt control function.

        Raises
        ------
        TypeError
            If ``control_reference`` is not a Firedrake ``Function``.
        ValueError
            If the vector size does not match the control reference.
        """
        if not isinstance(control_reference, fire.Function):
            raise TypeError(
                "FWI control reference must be a firedrake Function. "
                f"Received {type(control_reference).__name__}.",
            )
        flat_vector = np.asarray(flat_vector, dtype=float).reshape(-1)
        reference_shape = np.asarray(control_reference.dat.data_ro).shape
        expected = int(np.prod(reference_shape))
        if flat_vector.size != expected:
            raise ValueError("Control vector size does not match the configured control.")
        return fire.Function(
            control_reference.function_space(), name=control_reference.name(),
            val=flat_vector.reshape(reference_shape))

    def _expand_bound(self, bound, control_reference):
        """Expand one bound specification to match a control component.

        Bounds may be scalar, one-element arrays, or arrays with one entry per
        control degree of freedom. This helper converts any supported form to a
        vector with the same size as ``control_reference``.

        Parameters
        ----------
        bound : scalar or array_like
            Bound value for one control component.
        control_reference : object
            Control component whose flattened size determines the output size.

        Returns
        -------
        numpy.ndarray
            Bound vector matching the flattened control size.

        Raises
        ------
        ValueError
            If an array bound does not match the control size.

        Examples
        --------
        ``vmin=1.5`` for a control with ``n`` degrees of freedom
        becomes an array of length ``n`` filled with ``1.5``.
        """
        size = self._flatten_control(control_reference).size
        if np.isscalar(bound):
            return np.full(size, float(bound))

        bound_array = np.asarray(bound, dtype=float).reshape(-1)
        if bound_array.size == 1:
            return np.full(size, float(bound_array[0]))
        if bound_array.size != size:
            raise ValueError("Control bounds do not match the control size.")
        return bound_array

    def set_guess_control(self, control):
        """Set the initial guess control parameter for inversion.

        Parameters
        ----------
        control : firedrake.Function or firedrake.Constant or list of firedrake.Function or list of firedrake.Constant
            Starting control parameter for the FWI optimization.
            ``Constant`` inputs are converted to a ``Function`` before being
            stored. FWI accepts a single control parameter for acoustic inversion,
            or a list of control parameters for multi-parameter inversion, e. g., for elastic inversion.

        Returns
        -------
        None
            Updates ``control_parameters`` and ``guess_mesh``. The cached misfit is
            reset.

        Raises
        ------
        ValueError
            If ``control`` holds no value.

        Examples
        --------
        ``set_guess_control(control)`` stores a defensive copy of the control
        ``Function``. ``set_guess_control(fire.Constant(2.0))`` creates a
        uniform control ``Function`` filled with ``2.0``.
        """
        control = self._control_by_parameter(control)
        if not control:
            # Storing an empty mapping would drop the selection and leave the
            # inversion looking unconfigured, several calls away from here.
            raise ValueError(
                "A guess control value is required. Received "
                f"{control!r}.",
            )
        self._write_parameters_into_wave(self.wave, control)
        self.guess_mesh = self.wave.get_mesh()
        self._control_parameters = self.wave.physical_parameters.copy(control)
        self.misfit = None

    @property
    def real_velocity_model_file(self):
        """
        Get the real velocity model file path.

        Returns
        -------
        str or None
            Path to the real velocity model file.
        """
        return self._real_velocity_model_file

    @real_velocity_model_file.setter
    def real_velocity_model_file(self, value):
        """
        Set the real velocity model file path.

        Parameters
        ----------
        value : str or None
            Path to the real velocity model file.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """
        if value is not None and not os.path.exists(value):
            raise FileNotFoundError(
                f"Velocity model file '{value}' does not exist",
            )
        self._real_velocity_model_file = value

    @property
    def real_shot_record_files(self):
        """
        Get the real shot record file path or pattern.

        Returns
        -------
        str or None
            Path or prefix pattern for the real shot record files.
        """
        return self._real_shot_record_files

    @real_shot_record_files.setter
    def real_shot_record_files(self, value):
        """
        Set the real shot record file path or pattern.

        This setter loads the real shot record if a valid path is provided.

        Parameters
        ----------
        value : str or None
            Path or prefix pattern for the real shot record files.

        Raises
        ------
        FileNotFoundError
            If the specified file or files matching the pattern do not exist.
        """
        if value is not None:
            if not os.path.exists(value) and not glob.glob(value + "*"):
                raise FileNotFoundError(
                    f"Shot record file '{value}' does not exist",
                )
        self._real_shot_record_files = value
        if value is not None:
            self.load_real_shot_record(file_name=value)

    def enable_automated_adjoint(
        self, control_parameters=None, checkpointing: bool = False,
        snapshots: int | None = None,
        gc_timestep_frequency: int | None = None,
    ):
        """Differentiate the inversion with the automated adjoint.

        Hands the gradient over to algorithmic differentiation through
        :mod:`firedrake.adjoint` instead of the hand-written adjoint solver.
        The choice is recorded on the wave solver, so it holds for every
        method of this driver that needs a functional or a gradient, and it
        is what makes :meth:`run_fwi` optimize with PETSc TAO.

        Call this *after* the guess mesh and the guess model are configured:
        the control has to be an existing field before it can be recorded as
        one.

        Parameters
        ----------
        control_parameters : enum.Enum or iterable of enum.Enum, optional
            Physical parameters to differentiate with respect to. ``None``
            takes every parameter the wave equation offers, which for an
            acoustic medium is the velocity model.
        checkpointing : bool, optional
            Whether to manage the tape with a checkpoint schedule. ``False``
            (the default) keeps every forward step on the tape.
        snapshots : int, optional
            How many checkpoints to keep in RAM, which is also what selects
            the schedule. ``None`` (the default) keeps every time step.
        gc_timestep_frequency : int, optional
            Run a garbage collection every this many time steps, lowering the
            peak memory of a checkpointed tape. ``None`` (the default)
            disables it.

        Returns
        -------
        None
            The wave solver is configured in place.

        See Also
        --------
        spyro.solvers.wave.Wave.enable_automated_adjoint :
            The solver-side switch this forwards to.
        spyro.solvers.automatic_differentiation_solver.AutomatedAdjoint :
            Which schedule each setting selects, and the references behind
            them.

        Examples
        --------
        >>> fwi.set_guess_mesh(input_mesh_parameters={"edge_length": 0.1})
        >>> fwi.set_guess_velocity_model(constant=2.5)
        >>> fwi.enable_automated_adjoint(checkpointing=True, snapshots=10)
        >>> fwi.run_fwi(vmin=2.5, vmax=3.0, maxiter=10)
        """
        # The solver needs a mesh and a model before a control can be a field,
        # and the driver may still be holding both: mirror what a forward
        # solve would do, so the call works wherever it is placed after the
        # guess model is configured.
        if self.wave.mesh is None and self.guess_mesh is not None:
            self.wave.set_mesh(user_mesh=self.guess_mesh, input_mesh_parameters={})
        if self._control_parameters:
            self._write_parameters_into_wave(self.wave, self._control_parameters)

        self.wave.enable_automated_adjoint(
            control_parameters=control_parameters,
            checkpointing=checkpointing,
            snapshots=snapshots,
            gc_timestep_frequency=gc_timestep_frequency,
        )

    def calculate_misfit(self, c=None, save_output=False):
        """
        Calculate the misfit between observed and simulated data.

        Runs the forward model with the current control parameter and computes
        the difference between the simulated shot records and the real/observed
        shot records. The forward solve is executed every time this method is
        called so the misfit always corresponds to the current control.

        Parameters
        ----------
        c : array_like, optional
            Control parameter values to use. If provided, updates the guess
            control parameters before running the forward solve.

        Returns
        -------
        misfit : ndarray or list of ndarray or None
            Misfit between real and simulated shot records. Returns a list
            if using spatial parallelism with multiple sources, otherwise
            returns a single array. ``None`` under the automated adjoint,
            which accumulates the functional on the tape as the forward solve
            runs and never forms the residual as an array.

        Notes
        -----
        This method also saves the current control parameters to disk for
        debugging and checkpoint purposes.
        """
        if self.wave.mesh is None and self.guess_mesh is not None:
            self.wave.set_mesh(user_mesh=self.guess_mesh, input_mesh_parameters={})

        if c is not None:
            updated_control = self._rebuild_control_from_vector(
                self.control_parameters,
                c,
            )
            self.set_guess_control(updated_control)
        elif self._control_parameters:
            self._write_parameters_into_wave(self.wave, self._control_parameters)
        else:
            raise ValueError("No guess control parameter has been configured.")

        self._sync_wave_real_shot_record()
        if self.wave.adjoint_type == AdjointType.IMPLEMENTED_ADJOINT:
            self.wave.enable_implemented_adjoint()
        self.wave.forward_solve()
        current_control = self.control_parameters
        fire.VTKFile(f"control_{self.current_iteration}.pvd").write(current_control)
        np.save(
            f"control{self.comm.ensemble_comm.rank}_{self.comm.comm.rank}",
            self._flatten_control(current_control),
        )

        # The automated adjoint accumulates the functional on the tape, one
        # time step at a time, so the residual never exists as an array to
        # subtract here and the whole tape would be discarded by rebuilding
        # it. Every other adjoint needs the residual computed explicitly.
        if self.wave.adjoint_type != AdjointType.AUTOMATED_ADJOINT:
            self._compute_misfit()
        return self.misfit

    def _compute_misfit(self):
        """Compute the receiver-data residual left by the last forward solve.

        Sets :attr:`guess_shot_record` and :attr:`misfit` from the forward
        solution at the receivers, one entry per source under spatial
        parallelism with more than one source, and a single array otherwise.

        Returns
        -------
        None
            The residual is stored on the driver.
        """
        if self.wave.parallelism_type == "spatial" and self.wave.number_of_sources > 1:
            misfit_list = []
            guess_shot_record_list = []
            for snum in range(self.wave.number_of_sources):
                switch_serial_shot(self.wave, snum)
                guess_shot_record_list.append(self.wave.forward_solution_receivers)
                misfit_list.append(
                    self.real_shot_record[snum] - self.wave.forward_solution_receivers,
                )
            self.guess_shot_record = guess_shot_record_list
            self.misfit = misfit_list
        else:
            self.guess_shot_record = self.wave.forward_solution_receivers
            self.guess_forward_solution = self.wave.forward_solution
            self.misfit = self.real_shot_record - self.guess_shot_record

    def generate_real_shot_record(
        self,
        plot_model=False,
        model_filename="model.png",
        abc_points=None,
        save_shot_record=True,
        shot_filename="shots/shot_record_",
        high_resolution_model=False,
    ):
        """
        Generate synthetic shot records from the configured real control.

        Create a wave solver with the real control, and solve the forward
        problem, and optionally saves the shot records and plots the model.
        This is used only for synthetic test cases.

        Parameters
        ----------
        plot_model : bool, optional
            If True, plot and save the configured acoustic model. Default is
            False.
        model_filename : str, optional
            Filename for the model plot. Default is "model.png".
        abc_points : list of tuple, optional
            Points defining absorbing boundary condition markers for plotting.
            Default is None.
        save_shot_record : bool, optional
            If True, save the shot records to files. Default is True.
        shot_filename : str, optional
            Prefix for shot record file names. Default is "shots/shot_record_".
        high_resolution_model : bool, optional
            If True, use high resolution for model plotting. Default is False.

        Notes
        -----
        This method creates observed data for synthetic inversion tests. The
        generated shot records are stored in self.real_shot_record.
        """
        real_wave = self.wave_class(dictionary=self.input_dictionary, comm=self.comm)
        if self.real_mesh is not None:
            real_wave.set_mesh(user_mesh=self.real_mesh, input_mesh_parameters={})

        if self._real_model_parameters:
            self._write_parameters_into_wave(real_wave, self._real_model_parameters)
        elif self.real_velocity_model_file is not None:
            try:
                real_wave.initial_velocity_model_file
            except AttributeError:
                raise ValueError(
                    "No real control parameter has been configured.",
                ) from None
            real_wave.initial_velocity_model_file = self.real_velocity_model_file
        else:
            raise ValueError("No real control parameter has been configured.")

        if (
            plot_model
            and real_wave.comm.comm.rank == 0
            and real_wave.comm.ensemble_comm.rank == 0
        ):
            spyro_plot_model(
                real_wave,
                filename=model_filename,
                abc_points=abc_points,
                high_resolution=high_resolution_model,
            )

        real_wave.forward_solve()
        if save_shot_record:
            save_shots(real_wave, file_name=shot_filename)

        if real_wave.parallelism_type == "spatial" and real_wave.number_of_sources > 1:
            real_shot_record_list = []
            for snum in range(real_wave.number_of_sources):
                switch_serial_shot(real_wave, snum)
                real_shot_record_list.append(real_wave.forward_solution_receivers)
            self.real_shot_record = real_shot_record_list
        else:
            self.real_shot_record = real_wave.forward_solution_receivers
        self._sync_wave_real_shot_record()

    def set_real_velocity_model(
        self,
        constant=None,
        conditional=None,
        velocity_model_function=None,
        expression=None,
        new_file=None,
        output=False,
        dg_velocity_model=True,
    ):
        """
        Set the true velocity model for synthetic test cases.

        This method sets the real/true velocity model that is used only for
        generating synthetic observed data. It delegates model initialization
        to the internal wave solver.

        Parameters
        ----------
        constant : float, optional
            Constant velocity value for a homogeneous model.
        conditional : firedrake.Conditional, optional
            Firedrake conditional object defining the velocity distribution.
        velocity_model_function : firedrake.Function, optional
            Firedrake function to use as the velocity model. Must be in the
            same function space as the object.
        expression : str, optional
            Mathematical expression string for the velocity model. Can use
            variables: x, y, z, pi, tanh, sqrt. Example: "2.0 + 0.5*tanh((x-2.0)/0.1)".
            Will be interpolated into the function space.
        new_file : str, optional
            Path to file containing the velocity model.
        output : bool, optional
            If True, output the velocity model to a pvd file for visualization.
            Default is False.
        dg_velocity_model : bool, optional
            If True, use DG0 function space. Default is True.

        Notes
        -----
        Only one of the parameters (constant, conditional, velocity_model_function,
        expression, or new_file) should be provided.
        """
        self.wave.set_initial_velocity_model(
            constant=constant,
            conditional=conditional,
            velocity_model_function=velocity_model_function,
            expression=expression,
            new_file=new_file,
            output=output,
            dg_velocity_model=dg_velocity_model,
        )
        self.real_mesh = self.wave.get_mesh()
        self._real_model_parameters = self._copy_parameters_from_wave(self.wave)
        if new_file is not None:
            self.real_velocity_model_file = new_file

    def set_guess_velocity_model(
        self,
        constant=None,
        conditional=None,
        velocity_model_function=None,
        expression=None,
        new_file=None,
        output=False,
        dg_velocity_model=True,
    ):
        """
        Set the initial guess velocity model for inversion.

        This method sets the starting velocity model for the FWI optimization.
        It delegates model initialization to the internal wave solver and
        resets the misfit.

        Parameters
        ----------
        constant : float, optional
            Constant velocity value for a homogeneous initial model.
        conditional : firedrake.Conditional, optional
            Firedrake conditional object defining the velocity distribution.
        velocity_model_function : firedrake.Function, optional
            Firedrake function to use as the velocity model. Must be in the
            same function space as the object.
        expression : str, optional
            Mathematical expression string for the velocity model. Can use
            variables: x, y, z, pi, tanh, sqrt. Example: "2.0 + 0.5*tanh((x-2.0)/0.1)".
            Will be interpolated into the function space.
        new_file : str, optional
            Path to file containing the velocity model.
        output : bool, optional
            If True, output the velocity model to a pvd file for visualization.
            Default is False.
        dg_velocity_model : bool, optional
            If True, use DG0 function space. Default is True.

        Notes
        -----
        Only one of the parameters (constant, conditional, velocity_model_function,
        expression, or new_file) should be provided. Setting a new guess model
        will reset the misfit to None.
        """
        self.wave.set_initial_velocity_model(
            constant=constant,
            conditional=conditional,
            velocity_model_function=velocity_model_function,
            expression=expression,
            new_file=new_file,
            output=output,
            dg_velocity_model=dg_velocity_model,
        )
        self.guess_mesh = self.wave.get_mesh()
        self._control_parameters = self._copy_parameters_from_wave(self.wave)
        self.misfit = None

    def set_real_mesh(self, user_mesh=None, input_mesh_parameters=None):
        """
        Set the mesh for the true/real velocity model.

        This method sets up the mesh used for generating synthetic observed
        data from the true velocity model.

        Parameters
        ----------
        user_mesh : spyro.Mesh, optional
            User-provided mesh object. Default is None.
        input_mesh_parameters : dict, optional
            Dictionary of mesh parameters. Default is None, which will be
            converted to an empty dictionary internally.

        Notes
        -----
        The mesh type defaults to "firedrake_mesh" if not specified in
        input_mesh_parameters.
        """
        if input_mesh_parameters is None:
            input_mesh_parameters = {}
        input_mesh_parameters.setdefault("mesh_type", "firedrake_mesh")
        self.wave.set_mesh(
            user_mesh=user_mesh,
            input_mesh_parameters=input_mesh_parameters,
        )
        self.real_mesh = self.wave.get_mesh()

    def set_guess_mesh(self, user_mesh=None, input_mesh_parameters=None):
        """
        Set the mesh for the guess/inversion model.

        This method sets up the mesh used for the FWI optimization. It also
        checks for gradient mask options in the mesh parameters.

        Parameters
        ----------
        user_mesh : spyro.Mesh, optional
            User-provided mesh object. Default is None.
        input_mesh_parameters : dict, optional
            Dictionary of mesh parameters. Can include "gradient_mask" to
            enable masking functionality. Default is an empty dictionary.

        Notes
        -----
        If "gradient_mask" is present in input_mesh_parameters, sets
        self.gradient_mask_available to True.
        """
        if input_mesh_parameters is None:
            input_mesh_parameters = {}
        if input_mesh_parameters.get("gradient_mask") is not None:
            self.gradient_mask_available = True
            self.wave.gradient_mask_available = True
        self.wave.set_mesh(
            user_mesh=user_mesh,
            input_mesh_parameters=input_mesh_parameters,
        )
        self.guess_mesh = self.wave.get_mesh()

    def get_functional(self, c=None):
        """
        Calculate and return the objective functional value.

        Computes a fresh misfit for the current control, then evaluates the
        objective functional. Also tracks the functional history and peak
        memory usage.

        Parameters
        ----------
        c : array_like, optional
            Control parameter values to use for the calculation. If provided,
            updates the model before computing the functional.

        Returns
        -------
        Jm : float
            The objective functional value (typically L2 norm of misfit).

        Notes
        -----
        This method writes the functional value and memory usage to text files
        for tracking convergence and resource consumption.

        Under the automated adjoint the functional is not recomputed here: the
        forward solve accumulates it on the tape, so the value it left on the
        solver is the one recorded, and recomputing it from a residual would
        both duplicate the work and detach the number from the tape it has to
        be differentiated through.
        """
        self.calculate_misfit(c=c)
        if self.wave.adjoint_type == AdjointType.AUTOMATED_ADJOINT:
            # Each ensemble member taped its own shots, so the recorded value
            # is that member's J_i. Summing over the ensemble is what the
            # reduced functional does when it is differentiated, and what
            # ``ensemble_functional`` does for the implemented adjoint, so the
            # number reported here has to mean the same thing. A plain float
            # besides: the recorded ``AdjFloat`` carries a tape node with it,
            # and the history would keep the whole tape alive.
            Jm = self.comm.ensemble_comm.allreduce(
                float(self.wave.functional_value), op=MPI.SUM,
            )
        else:
            Jm = compute_functional(self.wave, self.misfit)

        self.functional_history.append(Jm)
        self.functional = Jm
        peak_memory_mb = get_peak_memory()
        parallel_print(
            f"Functional: {Jm} at iteration: {self.current_iteration}",
            self.comm,
        )
        if self.comm.ensemble_comm.rank == 0 and self.comm.comm.rank == 0:
            with open("functional_values.txt", "a") as file:
                file.write(
                    f"Iteration: {self.current_iteration}, Functional: {Jm}\n",
                )

            with open("peak_memory.txt", "a") as file:
                file.write(f"Peak memory usage: {peak_memory_mb:.2f} MB \n")

        return Jm

    def get_gradient(self, c=None, save=True, calculate_functional=True):
        """
        Calculate the gradient of the objective functional.

        Computes the gradient with respect to the control parameter using the
        adjoint method. Optionally calculates the functional value first and
        saves the gradient to a VTK file.

        Parameters
        ----------
        c : array_like or firedrake.Function or list of firedrake.Function, optional
            Control parameter values to use. If provided and
            calculate_functional is True, updates the model before computing
            the functional.
        save : bool, optional
            If True, save the gradient to a VTK file for visualization.
            Default is True.
        calculate_functional : bool, optional
            If True, calculate the functional (and misfit) before computing
            the gradient. Default is True.

        Notes
        -----
        This method increments the current_iteration counter and applies any
        gradient mask that has been set. The gradient is computed using the
        adjoint-state method implemented in gradient_solve(), or by
        differentiating the recorded tape when the automated adjoint is
        enabled.
        """
        comm = self.comm
        if calculate_functional:
            self.get_functional(c=c)
        elif c is not None:
            updated_control = self._rebuild_control_from_vector(
                self.control_parameters,
                c,
            )
            self.set_guess_control(updated_control)

        comm.comm.barrier()
        if self.wave.adjoint_type == AdjointType.AUTOMATED_ADJOINT:
            # The tape holds the forward solve and the residual alike, so
            # neither is passed: reverse-mode differentiation reads them both
            # off the recording made by the forward solve above.
            self.gradient = self.wave.gradient_solve(
                adjoint_type=AdjointType.AUTOMATED_ADJOINT,
            )
        else:
            self.gradient = self.wave.gradient_solve(
                misfit=self.misfit,
                forward_solution=self.guess_forward_solution,
            )
        self._apply_gradient_mask()
        if save:
            fire.VTKFile(f"gradient_{self.current_iteration}.pvd").write(self.gradient)
        self.current_iteration += 1
        comm.comm.barrier()

    def return_functional_and_gradient(self, c):
        """
        Compute and return both the functional value and gradient.

        This method is used as the objective function for scipy.optimize.minimize.
        It computes the gradient (which also computes the functional) and returns
        both values.

        Parameters
        ----------
        c : array_like
            Current control parameter values.

        Returns
        -------
        functional : float
            The objective functional value.
        dJ : ndarray
            The gradient of the functional with respect to the control
            parameters.
        """
        self.get_gradient(c=c)
        return self.functional, self._flatten_control(self.gradient)

    def run_fwi(self, **kwargs):
        """
        Run full waveform inversion.

        The optimization minimizes the misfit between observed and simulated
        data by updating the configured control parameter. Which optimizer
        does it follows from the adjoint in use:

        Implemented adjoint (the default)
            :func:`scipy.optimize.minimize` with the L-BFGS-B method, which
            calls back into :meth:`return_functional_and_gradient` and so
            re-runs the forward and adjoint solves for every iterate.
        Automated adjoint
            PETSc TAO, driven by the pyadjoint reduced functional built from a
            single recorded forward solve. Enable it with
            :meth:`enable_automated_adjoint` before calling this method.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for customizing the optimization:

            vmin : float or array_like, optional
                Lower bound for the control parameter. Default is 1.429. A
                bound given per degree of freedom becomes a pair of arrays
                for L-BFGS-B, and a pair of fields for TAO.
            vmax : float or array_like, optional
                Upper bound for the control parameter. Default is 6.0.
            maxiter : int, optional
                Maximum number of iterations. Default is 20.
            scipy_options : dict, optional
                Additional options passed to scipy.optimize.minimize.
                Default includes disp=True, eps=1e-15, ftol=1e-11.
            tao_options : dict, optional
                PETSc options for the TAO solver, merged over the defaults
                ``{"tao_type": "blmvm", "tao_max_it": maxiter}``. Only used
                under the automated adjoint.
            adjoint_type : AdjointType, optional
                Adjoint to run with. Enables it on the wave solver, so
                ``AdjointType.AUTOMATED_ADJOINT`` here is equivalent to
                calling :meth:`enable_automated_adjoint` with its defaults
                beforehand.

        Returns
        -------
        scipy.optimize.OptimizeResult or firedrake.Function
            The scipy result under the implemented adjoint, and the optimal
            control field under the automated one, which is what TAO returns.

        Notes
        -----
        The final control parameter is stored in ``control_parameter_result``
        and saved to ``control_end.pvd``. The raw optimizer vector is also
        saved to ``result.npy``.

        Examples
        --------
        >>> fwi.run_fwi(maxiter=100, vmin=1.5, vmax=5.0)
        """
        maxiter = kwargs.pop("maxiter", 20)
        parameters = {
            "vmin": kwargs.pop("vmin", 1.429),
            "vmax": kwargs.pop("vmax", 6.0),
            "maxiter": maxiter,
            "scipy_options": {
                "disp": True,
                "eps": kwargs.pop("eps", 1e-15),
                "ftol": kwargs.pop("ftol", 1e-11),
                "maxiter": maxiter,
            },
        }
        tao_options = kwargs.pop("tao_options", None)
        adjoint_type = kwargs.pop("adjoint_type", None)
        if adjoint_type is not None:
            self.adjoint_type = adjoint_type
            # Naming an adjoint here has to switch it on, or the run would
            # silently fall back to whichever one the solver already had.
            if adjoint_type == AdjointType.AUTOMATED_ADJOINT:
                if self.wave.automated_adjoint is None:
                    self.enable_automated_adjoint()
            elif adjoint_type == AdjointType.IMPLEMENTED_ADJOINT:
                self.wave.enable_implemented_adjoint()
        parameters.update(kwargs)

        control_reference = self.control_parameters
        if self.wave.adjoint_type == AdjointType.AUTOMATED_ADJOINT:
            result = self._run_fwi_tao(
                parameters, control_reference, tao_options=tao_options,
            )
            result_vector = self._flatten_control(result)
        else:
            lower = self._expand_bound(parameters["vmin"], control_reference)
            upper = self._expand_bound(parameters["vmax"], control_reference)
            bounds = list(zip(lower, upper))
            control_0 = self._flatten_control(control_reference)
            options = parameters["scipy_options"]
            result = scipy_minimize(
                self.return_functional_and_gradient,
                control_0,
                method="L-BFGS-B",
                jac=True,
                tol=1e-15,
                bounds=bounds,
                options=options,
            )
            result_vector = result.x

        self.control_parameter_result = self._rebuild_control_from_vector(
            control_reference,
            result_vector,
        )
        self.set_guess_control(self.control_parameter_result)

        fire.VTKFile("control_end.pvd").write(self.control_parameter_result)

        np.save("result", result_vector)
        return result

    def _run_fwi_tao(self, parameters, control_reference, tao_options=None):
        """Optimize the recorded reduced functional with PETSc TAO.

        The forward solve is recorded once, here, and every functional value
        and gradient the optimizer asks for afterwards is a replay of that
        recording at a new control value. This is what separates the automated
        adjoint from the implemented one, where each iterate re-runs the
        forward and backward propagators from this driver.

        Under ensemble (shot) parallelism the reduced functional is an
        ``EnsembleReducedFunctional``, which sums the per-shot functionals and
        gradients across ensemble members. TAO is therefore given the
        *spatial* communicator: the control is replicated on every member, and
        the default (``COMM_WORLD``) would count each copy as separate degrees
        of freedom.

        Parameters
        ----------
        parameters : dict
            Optimization parameters assembled by :meth:`run_fwi`. ``vmin``,
            ``vmax`` and ``maxiter`` are read from it.
        control_reference : firedrake.Function
            Control whose function space shapes the bounds and the solution.
        tao_options : dict, optional
            PETSc options merged over the defaults.

        Returns
        -------
        firedrake.Function
            The optimal control.

        Warns
        -----
        UserWarning
            If TAO stops without converging, which is what reaching
            ``maxiter`` amounts to. The last iterate is returned rather than
            raising, since a fixed iteration budget is a normal way to run
            FWI.
        """
        # Records the tape, and logs the starting functional the same way the
        # scipy path logs every iterate.
        self.get_functional()

        automated_adjoint = self.wave.automated_adjoint
        reduced_functional = automated_adjoint.reduced_functional
        if reduced_functional is None:
            reduced_functional = automated_adjoint.create_reduced_functional(
                self.wave.functional_value,
            )

        # One bound pair per control, not per degree of freedom: TAO takes the
        # bounds as fields (or scalars broadcast over them), while L-BFGS-B
        # takes a pair for each entry of the flattened control.
        lower = self._tao_bound(parameters["vmin"], control_reference)
        upper = self._tao_bound(parameters["vmax"], control_reference)
        bounds = [(lower, upper) for _ in reduced_functional.controls]
        problem = MinimizationProblem(reduced_functional, bounds=bounds)

        options = {
            "tao_type": "blmvm",
            "tao_max_it": parameters["maxiter"],
        }
        if tao_options:
            options.update(tao_options)

        solver = TAOSolver(problem, options, comm=self.wave.comm.comm)
        solver.tao.setMonitor(self._tao_monitor)
        try:
            solution = solver.solve()
        except TAOConvergenceError as error:
            warnings.warn(
                f"{error} Returning the last iterate; raise the iteration "
                "limit or loosen the tolerances through 'tao_options' if the "
                "inversion is meant to run to convergence.",
            )
            solution = self._control_from_tao(solver, control_reference)
        return solution

    def _tao_bound(self, bound, control_reference):
        """Return one bound in the form TAO takes it.

        Parameters
        ----------
        bound : scalar or array_like
            Bound value for one control.
        control_reference : firedrake.Function
            Control whose function space an array bound is rebuilt in.

        Returns
        -------
        float or firedrake.Function
            The bound as a scalar, broadcast by TAO over the control, or as a
            field when it varies per degree of freedom.
        """
        if np.isscalar(bound):
            return float(bound)
        return self._rebuild_control_from_vector(
            control_reference,
            self._expand_bound(bound, control_reference),
        )

    def _control_from_tao(self, solver, control_reference):
        """Read the current iterate out of a TAO solver.

        Used when TAO stops without converging, where the solver raises before
        handing the solution back.

        Parameters
        ----------
        solver : pyadjoint.TAOSolver
            Solver whose solution vector is read.
        control_reference : firedrake.Function
            Control whose function space the vector is read into.

        Returns
        -------
        firedrake.Function
            The last iterate.

        Raises
        ------
        NotImplementedError
            If the problem has more than one control, where the solution
            vector is a concatenation that this does not split.
        """
        if len(solver.tao_objective.reduced_functional.controls) != 1:
            raise NotImplementedError(
                "Recovering an unconverged iterate is only implemented for a "
                "single control.",
            )
        solution = fire.Function(
            control_reference.function_space(), name=control_reference.name(),
        )
        with solution.dat.vec_wo as vec:
            solver.tao.getSolution().copy(vec)
        return solution

    def _tao_monitor(self, tao):
        """Record one TAO iteration, mirroring what ``get_functional`` logs.

        TAO drives the reduced functional itself, so the per-iterate
        bookkeeping the scipy path does inside :meth:`get_functional` has to
        happen here instead.

        Parameters
        ----------
        tao : petsc4py.PETSc.TAO
            The solver being monitored.

        Returns
        -------
        None
            The functional history and iteration counter are updated in place.
        """
        iteration, functional = tao.getSolutionStatus()[:2]
        if iteration == 0:
            # The starting point is the control the tape was recorded at,
            # already logged by the functional evaluation that recorded it.
            return
        self.current_iteration = iteration
        self.functional = functional
        self.functional_history.append(functional)
        parallel_print(
            f"Functional: {functional} at iteration: {iteration}",
            self.comm,
        )
        if self.comm.ensemble_comm.rank == 0 and self.comm.comm.rank == 0:
            with open("functional_values.txt", "a") as file:
                file.write(
                    f"Iteration: {iteration}, Functional: {functional}\n",
                )

            with open("peak_memory.txt", "a") as file:
                file.write(f"Peak memory usage: {get_peak_memory():.2f} MB \n")

    def run_fwi_rol(self, **kwargs):
        """
        Run full waveform inversion using ROL optimizer (deprecated).

        Performs FWI optimization using the Rapid Optimization Library (ROL).
        This method is deprecated as the pyROL library is no longer supported.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for customizing the optimization:

            vmin : float, optional
                Lower bound for the control parameter. Default is 1.429.
            vmax : float, optional
                Upper bound for the control parameter. Default is 6.0.
            maxiter : int, optional
                Maximum number of iterations. Default is 20.
            ROL_options : dict, optional
                ROL-specific optimization parameters.

        Raises
        ------
        ImportError
            If the ROL module is not available.

        Warnings
        --------
        DeprecationWarning
            This method is deprecated. Use run_fwi() instead.

        Notes
        -----
        The ROL library provided advanced optimization algorithms but is no
        longer maintained. Consider using run_fwi() with scipy instead.
        """
        if ROL is None:
            raise ImportError("The ROL module is not available.")
        control_reference = self.control_parameters
        if not isinstance(control_reference, fire.Function):
            raise NotImplementedError(
                "The deprecated ROL inversion path only supports a single "
                "Firedrake Function control.",
            )

        parameters = {
            "vmin": 1.429,
            "vmax": 6.0,
            "ROL_options": {
                "General": {"Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 10}},
                "Step": {
                    "Type": "Augmented Lagrangian",
                    "Augmented Lagrangian": {
                        "Subproblem Step Type": "Line Search",
                        "Subproblem Iteration Limit": 5.0,
                    },
                    "Line Search": {"Descent Method": {"Type": "Quasi-Newton Step"}},
                },
                "Status Test": {
                    "Gradient Tolerance": 1e-16,
                    "Iteration Limit": kwargs.pop("maxiter", 20),
                    "Step Tolerance": 1.0e-16,
                },
            },
        }
        parameters.update(kwargs)
        vmin = parameters["vmin"]
        vmax = parameters["vmax"]

        warnings.warn(
            "This functionality is deprecated, since the pyROL library is no longer supported.",
        )
        params = ROL.ParameterList(parameters["ROL_options"], "Parameters")

        inner_product = L2Inner(self.wave)

        obj = Objective(inner_product, self)

        u = fire.Function(
            control_reference.function_space(),
            name=control_reference.name(),
        ).assign(control_reference)
        opt = FireVector(u.vector(), inner_product)

        lower_bound = fire.Function(control_reference.function_space())
        lower_bound.interpolate(fire.Constant(vmin))
        x_lo = FireVector(lower_bound.vector(), inner_product)

        upper_bound = fire.Function(control_reference.function_space())
        upper_bound.interpolate(fire.Constant(vmax))
        x_up = FireVector(upper_bound.vector(), inner_product)

        bnd = ROL.Bounds(x_lo, x_up, 1.0)

        algo = ROL.Algorithm("Line Search", params)

        algo.run(opt, obj, bnd)

    def set_gradient_mask(self, boundaries=None):
        """
        DEPRECATED: Set a gradient mask to zero out gradients outside defined boundaries.

        The gradient mask is used to restrict updates to certain regions of
        the model domain, which is useful for excluding absorbing boundary
        layers or other regions where the control parameter should not be
        updated.

        This method is deprecated since we prefer to use mesh based tags for now. In
        the future we will use the new submesh functionality in Firedrake.

        Parameters
        ----------
        boundaries : list of float, optional
            List of boundary values defining the mask region. If not provided
            and abc_active is True, uses PML boundary locations automatically.

        Raises
        ------
        ValueError
            If abc_active is False and boundaries is None.
            If the mask options configuration doesn't make sense.

        Warnings
        --------
        UserWarning
            If abc_active is True and boundaries is provided, the boundaries
            parameter will override the automatic PML boundaries.

        Notes
        -----
        The mask is applied automatically during get_gradient() via the
        _apply_gradient_mask() method.

        Examples
        --------
        >>> fwi.set_gradient_mask(boundaries=[0.0, 0.5, 5.0, 5.5])
        """
        self.has_gradient_mask = True

        if self.wave.abc_active is False and boundaries is None:
            raise ValueError("If no abc boundary please define boundaries for the mask")
        elif self.wave.abc_active and boundaries is None:
            mask_obj = Gradient_mask_for_pml(self.wave)
        elif self.wave.abc_active and boundaries is not None:
            warnings.warn("Boundaries overuling PML boundaries for mask")
            mask_obj = Mask(boundaries, self.wave)
        elif self.wave.abc_active is False and boundaries is not None:
            mask_obj = Mask(boundaries, self.wave)
        else:
            raise ValueError("Mask options do not make sense")

        self.mask_obj = mask_obj

    def _apply_gradient_mask(self):
        """
        DEPRECATED: apply the gradient mask to the computed gradient.

        If a gradient mask has been set via set_gradient_mask(), this method
        applies the mask to zero out gradient values outside the defined region.
        This is called automatically during get_gradient().

        Notes
        -----
        This method is deprecated since we prefer to use mesh based tags for now. In
        the future we will use the new submesh functionality in Firedrake.
        """
        if self.has_gradient_mask:
            self.gradient = self.mask_obj.apply_mask(self.gradient)
        else:
            pass

    def load_real_shot_record(self, file_name="shots/shot_record_"):
        """
        Load real/observed shot records from files.

        This method loads previously saved shot records and assigns them as
        the real shot record data for the inversion.

        Parameters
        ----------
        file_name : str, optional
            File name prefix for the shot record files. Default is "shots/shot_record_".

        Notes
        -----
        After loading, the forward_solution_receivers attribute is cleared to
        save memory.
        """
        load_shots(self.wave, file_name=file_name)
        self.real_shot_record = self.wave.forward_solution_receivers
        self.wave.real_shot_record = self.real_shot_record
        self.wave.forward_solution_receivers = None

    @run_in_one_core
    def save_result_as_segy(self, file_name="final_vp.segy", grid_spacing=0.01):
        """
        Save the final scalar control result as a SEG-Y file.

        This method exports the final scalar inversion control to SEG-Y format,
        which is a standard format for seismic data. The operation is performed
        on a single core.

        Parameters
        ----------
        file_name : str, optional
            Output SEG-Y file name. Default is "final_vp.segy".
        grid_spacing: float, optional
            Segy grid spacing, default is 0.01 km.

        Notes
        -----
        This method uses a fixed spacing of 10 meters for the SEG-Y export.
        The @run_in_one_core decorator ensures this operation runs on a single
        MPI rank to avoid conflicts.
        """
        if self.control_parameter_result is None:
            raise ValueError(
                "SEG-Y export requires a single scalar inversion control result.",
            )
        create_segy(
            self.control_parameter_result,
            self.control_parameter_result.function_space(),
            grid_spacing,
            file_name,
        )


class SyntheticRealAcousticWave(AcousticWave):
    """
    The SyntheticRealAcousticWave class is a subclass of the AcousticWave class.
    It is used to generate synthetic real acoustic wave data.

    Attributes:
    -----------
    dictionary: (dict)
        A dictionary containing parameters for the inversion.
    comm: MPI communicator

    Methods:
    --------
    __init__(self, dictionary=None, comm=None):
        Initializes a new instance of the SyntheticRealAcousticWave class.
    forward_solve():
        Solves the forward problem.
    """

    def __init__(self, dictionary=None, comm=None):
        """
        Initialize a SyntheticRealAcousticWave instance.

        Parameters
        ----------
        dictionary : dict, optional
            A dictionary containing parameters for the wave simulation.
        comm : MPI.Comm, optional
            MPI communicator for parallel execution.
        """
        super().__init__(dictionary=dictionary, comm=comm)

    def forward_solve(self):
        """
        Solve the forward acoustic wave problem.

        This method solves the forward problem for the real/true velocity model
        to generate synthetic observed data. It simply calls the parent class's
        forward_solve method.

        Returns
        -------
        None
        """
        super().forward_solve()
        if self.parallelism_type == "spatial" and self.number_of_sources > 1:
            real_shot_record_list = []
            for snum in range(self.number_of_sources):
                switch_serial_shot(self, snum)
                real_shot_record_list.append(self.forward_solution_receivers)
            self.real_shot_record = real_shot_record_list
        else:
            self.real_shot_record = self.forward_solution_receivers
