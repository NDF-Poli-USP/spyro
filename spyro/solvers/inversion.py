import firedrake as fire
import warnings
from scipy.optimize import minimize as scipy_minimize
from mpi4py import MPI  # noqa: F401
import numpy as np
import copy
import resource
import glob
import os

from .acoustic_wave import AcousticWave
from ..utils import compute_functional
from ..utils import Gradient_mask_for_pml, Mask
from ..plots import plot_model as spyro_plot_model
from ..io.basicio import switch_serial_shot
from ..io.basicio import load_shots, save_shots, create_segy
from ..utils import run_in_one_core


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
    Wave_obj : AcousticWave
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
    def __init__(self, Wave_obj):
        """
        Initialize the L2 inner product operator.

        Parameters
        ----------
        Wave_obj : AcousticWave
            Wave object containing the function space and quadrature rule.
        """
        V = Wave_obj.function_space
        # print(f"Dir {dir(Wave_obj)}", flush=True)
        dxlump = fire.dx(**Wave_obj.quadrature_rule)
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
    the velocity model during optimization.

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
        Update the velocity model with new optimization iterate.
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
            Current velocity model iterate.
        tol : float
            Tolerance for the computation (unused).

        Returns
        -------
        float
            The objective functional value.
        """
        J_total = np.zeros((1))
        self.inversion_obj.misfit = None
        self.inversion_obj.reset_pressure()
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
            Current velocity model iterate.
        tol : float
            Tolerance for the computation (unused).
        """
        self.inversion_obj.get_gradient(calculate_functional=False)
        dJ = self.inversion_obj.gradient
        g.scale(0)
        g.vec += dJ

    def update(self, x, flag, iteration):
        """
        Update the velocity model with new optimization iterate.

        Parameters
        ----------
        x : FiredrakeVector
            New velocity model iterate.
        flag : int
            Update flag from ROL.
        iteration : int
            Current iteration number.
        """
        vp = self.inversion_obj.initial_velocity_model
        vp.assign(fire.Function(
            self.inversion_obj.function_space,
            x.vec,
            name="velocity")
        )


class FullWaveformInversion:
    """FWI driver composed around a wave solver.

    By default, the inversion driver uses :class:`AcousticWave`, but any wave
    solver that implements ``forward_solve()``, ``get_control_parameters()``
    and ``set_control_parameters()`` can be used for data misfit evaluation.
    Gradient-based optimization additionally requires ``gradient_solve()``.
    """

    def __init__(
        self,
        dictionary=None,
        comm=None,
        wave_class=AcousticWave,
        wave=None,
    ):
        if wave is not None:
            self.wave = wave
            self.wave_class = type(wave)
        else:
            self.wave_class = AcousticWave if wave_class is None else wave_class
            self.wave = self.wave_class(dictionary=dictionary, comm=comm)

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
        self.real_control = None
        self.guess_control = None
        self.real_velocity_model = None
        self.guess_velocity_model = None

        self.control_out = fire.VTKFile(inversion_dictionary["control_output_file"])
        self.gradient_out = fire.VTKFile(inversion_dictionary["gradient_output_file"])
        self.real_velocity_model_file = inversion_dictionary["real_velocity_model_file"]
        self.real_shot_record = None
        self.real_shot_record_files = inversion_dictionary["real_shot_record_file"]

        self.guess_shot_record = None
        self.gradient = None
        self.control_result = None
        self.vp_result = None
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

    def __getattr__(self, name):
        wave = self.__dict__.get("wave")
        if wave is not None and hasattr(wave, name):
            return getattr(wave, name)
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    def _create_wave_solver(self):
        return self.wave_class(dictionary=self.input_dictionary, comm=self.comm)

    def _control_items(self, control):
        if isinstance(control, dict):
            return list(control.items())
        if isinstance(control, list):
            return list(enumerate(control))
        if isinstance(control, tuple):
            return list(enumerate(control))
        return [("control", control)]

    def _constant_array(self, value):
        try:
            return np.asarray(value.values(), dtype=float)
        except AttributeError:
            return np.asarray(value, dtype=float)

    def _copy_control_value(self, value):
        if isinstance(value, fire.Function):
            copied = fire.Function(value.function_space(), name=value.name())
            copied.assign(value)
            return copied
        if isinstance(value, fire.Constant):
            constant_data = self._constant_array(value)
            if constant_data.shape == ():
                return fire.Constant(float(constant_data))
            return fire.Constant(constant_data)
        return copy.deepcopy(value)

    def _copy_control_structure(self, control):
        if control is None:
            return None
        if isinstance(control, dict):
            return {
                key: self._copy_control_value(value)
                for key, value in control.items()
            }
        if isinstance(control, list):
            return [self._copy_control_value(value) for value in control]
        if isinstance(control, tuple):
            return tuple(self._copy_control_value(value) for value in control)
        return self._copy_control_value(control)

    def _flatten_control_value(self, value):
        if isinstance(value, fire.Function):
            return np.asarray(value.dat.data_ro, dtype=float).reshape(-1)
        if isinstance(value, fire.Constant):
            return self._constant_array(value).reshape(-1)
        return np.atleast_1d(np.asarray(value, dtype=float)).reshape(-1)

    def _rebuild_control_value(self, template, flat_values):
        flat_values = np.asarray(flat_values, dtype=float).reshape(-1)
        if isinstance(template, fire.Function):
            rebuilt = self._copy_control_value(template)
            template_shape = np.asarray(template.dat.data_ro).shape
            rebuilt.dat.data[:] = flat_values.reshape(template_shape)
            return rebuilt
        if isinstance(template, fire.Constant):
            if flat_values.size == 1:
                return fire.Constant(float(flat_values[0]))
            return fire.Constant(flat_values)
        if np.isscalar(template):
            if flat_values.size != 1:
                raise ValueError("Scalar controls must rebuild from one value.")
            return float(flat_values[0])
        template_array = np.asarray(template)
        return flat_values.reshape(template_array.shape)

    def _flatten_control(self, control):
        if control is None:
            raise ValueError("No control parameter has been configured.")
        flattened = [
            self._flatten_control_value(value)
            for _, value in self._control_items(control)
        ]
        return np.concatenate(flattened) if flattened else np.zeros((0,), dtype=float)

    def _rebuild_control_from_vector(self, template, flat_vector):
        flat_vector = np.asarray(flat_vector, dtype=float).reshape(-1)
        offset = 0
        rebuilt_items = []
        for key, template_value in self._control_items(template):
            size = self._flatten_control_value(template_value).size
            rebuilt_value = self._rebuild_control_value(
                template_value,
                flat_vector[offset:offset + size],
            )
            rebuilt_items.append((key, rebuilt_value))
            offset += size

        if offset != flat_vector.size:
            raise ValueError("Control vector size does not match the configured control.")

        if isinstance(template, dict):
            return {key: value for key, value in rebuilt_items}
        if isinstance(template, list):
            return [value for _, value in rebuilt_items]
        if isinstance(template, tuple):
            return tuple(value for _, value in rebuilt_items)
        return rebuilt_items[0][1]

    def _control_functions(self, control):
        return [
            value
            for _, value in self._control_items(control)
            if isinstance(value, fire.Function)
        ]

    def _write_control_snapshot(self, control, filename):
        if control is None:
            return
        functions = self._control_functions(control)
        if functions:
            fire.VTKFile(filename).write(*functions)

    def _guess_control_template(self):
        template = self.guess_control
        if template is None:
            template = self.wave.get_control_parameters()
        if template is None:
            raise ValueError("No guess control parameter has been configured.")
        return template

    def _expand_bound(self, bound, template_value):
        size = self._flatten_control_value(template_value).size
        if np.isscalar(bound):
            return np.full(size, float(bound))

        bound_array = np.asarray(bound, dtype=float).reshape(-1)
        if bound_array.size == 1:
            return np.full(size, float(bound_array[0]))
        if bound_array.size != size:
            raise ValueError("Control bounds do not match the control size.")
        return bound_array

    def _extract_bound(self, bound, key, template_value):
        if isinstance(bound, dict):
            if key not in bound:
                raise KeyError(f"Missing bound for control '{key}'.")
            return self._expand_bound(bound[key], template_value)
        return self._expand_bound(bound, template_value)

    def _build_bounds(self, vmin, vmax, template):
        lower = []
        upper = []
        for key, value in self._control_items(template):
            lower.append(self._extract_bound(vmin, key, value))
            upper.append(self._extract_bound(vmax, key, value))
        return list(zip(np.concatenate(lower), np.concatenate(upper)))

    def get_control_vector(self, control=None):
        if control is None:
            control = self._guess_control_template()
        return self._flatten_control(control)

    def set_real_control(self, control):
        self.wave.set_control_parameters(self._copy_control_structure(control))
        self.real_mesh = self.wave.get_mesh()
        self.real_control = self._copy_control_structure(
            self.wave.get_control_parameters(),
        )
        self.real_velocity_model = (
            self._copy_control_structure(self.real_control)
            if isinstance(self.real_control, fire.Function)
            else None
        )

    def set_guess_control(self, control):
        self.wave.set_control_parameters(self._copy_control_structure(control))
        self.guess_mesh = self.wave.get_mesh()
        self.guess_control = self._copy_control_structure(
            self.wave.get_control_parameters(),
        )
        self.guess_velocity_model = (
            self._copy_control_structure(self.guess_control)
            if isinstance(self.guess_control, fire.Function)
            else None
        )
        self.misfit = None

    @property
    def real_velocity_model_file(self):
        return self._real_velocity_model_file

    @real_velocity_model_file.setter
    def real_velocity_model_file(self, value):
        if value is not None and not os.path.exists(value):
            raise FileNotFoundError(
                f"Velocity model file '{value}' does not exist",
            )
        self._real_velocity_model_file = value

    @property
    def real_shot_record_files(self):
        return self._real_shot_record_files

    @real_shot_record_files.setter
    def real_shot_record_files(self, value):
        if value is not None:
            if not os.path.exists(value) and not glob.glob(value + "*"):
                raise FileNotFoundError(
                    f"Shot record file '{value}' does not exist",
                )
        self._real_shot_record_files = value
        if value is not None:
            self.load_real_shot_record(file_name=value)

    def calculate_misfit(self, c=None):
        if self.wave.mesh is None and self.guess_mesh is not None:
            self.wave.set_mesh(user_mesh=self.guess_mesh, input_mesh_parameters={})

        if c is not None:
            updated_control = self._rebuild_control_from_vector(
                self._guess_control_template(),
                c,
            )
            self.set_guess_control(updated_control)
        elif self.guess_control is not None:
            self.wave.set_control_parameters(
                self._copy_control_structure(self.guess_control),
            )
        elif self.wave.get_control_parameters() is None:
            raise ValueError("No guess control parameter has been configured.")

        self.wave.forward_solve()
        current_control = self.wave.get_control_parameters()
        self._write_control_snapshot(
            current_control,
            f"control_{self.current_iteration}.pvd",
        )
        np.save(
            f"control{self.comm.ensemble_comm.rank}_{self.comm.comm.rank}",
            self._flatten_control(current_control),
        )

        if self.parallelism_type == "spatial" and self.number_of_sources > 1:
            misfit_list = []
            guess_shot_record_list = []
            for snum in range(self.number_of_sources):
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
        return self.misfit

    def generate_real_shot_record(
        self,
        plot_model=False,
        model_filename="model.png",
        abc_points=None,
        save_shot_record=True,
        shot_filename="shots/shot_record_",
        high_resolution_model=False,
    ):
        real_wave = self._create_wave_solver()
        if self.real_mesh is not None:
            real_wave.set_mesh(user_mesh=self.real_mesh, input_mesh_parameters={})

        if self.real_control is not None:
            real_wave.set_control_parameters(self._copy_control_structure(self.real_control))
        elif (
            hasattr(real_wave, "initial_velocity_model_file")
            and self.real_velocity_model_file is not None
        ):
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
                real_shot_record_list.append(real_wave.receivers_output)
            self.real_shot_record = real_shot_record_list
        else:
            self.real_shot_record = real_wave.receivers_output

    def set_smooth_guess_velocity_model(self, real_velocity_model_file=None):
        if real_velocity_model_file is not None:
            real_velocity_model_file = real_velocity_model_file
        else:
            real_velocity_model_file = self.real_velocity_model_file

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
        self.real_velocity_model = self.wave.initial_velocity_model
        self.real_control = self._copy_control_structure(
            self.wave.get_control_parameters(),
        )
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
        self.guess_velocity_model = self.wave.initial_velocity_model
        self.guess_control = self._copy_control_structure(
            self.wave.get_control_parameters(),
        )
        self.misfit = None

    def set_real_mesh(self, user_mesh=None, input_mesh_parameters=None):
        if input_mesh_parameters is None:
            input_mesh_parameters = {}
        input_mesh_parameters.setdefault("mesh_type", "firedrake_mesh")
        self.wave.set_mesh(
            user_mesh=user_mesh,
            input_mesh_parameters=input_mesh_parameters,
        )
        self.real_mesh = self.wave.get_mesh()

    def set_guess_mesh(self, user_mesh=None, input_mesh_parameters=None):
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
        self.calculate_misfit(c=c)
        Jm = compute_functional(self.wave, self.misfit)

        self.functional_history.append(Jm)
        self.functional = Jm
        peak_memory_mb = get_peak_memory()
        if self.comm.ensemble_comm.rank == 0 and self.comm.comm.rank == 0:
            print(
                f"Functional: {Jm} at iteration: {self.current_iteration}",
                flush=True,
            )
            with open("functional_values.txt", "a") as file:
                file.write(
                    f"Iteration: {self.current_iteration}, Functional: {Jm}\n",
                )

            with open("peak_memory.txt", "a") as file:
                file.write(f"Peak memory usage: {peak_memory_mb:.2f} MB \n")

        return Jm

    def get_gradient(self, c=None, save=True, calculate_functional=True):
        if not hasattr(self.wave, "gradient_solve"):
            raise NotImplementedError(
                f"{type(self.wave).__name__} does not implement gradient_solve().",
            )

        comm = self.comm
        if calculate_functional:
            self.get_functional(c=c)
        elif c is not None:
            updated_control = self._rebuild_control_from_vector(
                self._guess_control_template(),
                c,
            )
            self.set_guess_control(updated_control)

        comm.comm.barrier()
        self.gradient = self.wave.gradient_solve(
            misfit=self.misfit,
            forward_solution=self.guess_forward_solution,
        )
        self._apply_gradient_mask()
        if save:
            self._write_control_snapshot(
                self.gradient,
                f"gradient_{self.current_iteration}.pvd",
            )
        self.current_iteration += 1
        comm.comm.barrier()

    def return_functional_and_gradient(self, c):
        self.get_gradient(c=c)
        return self.functional, self._flatten_control(self.gradient)

    def run_fwi(self, **kwargs):
        parameters = {
            "vmin": kwargs.pop("vmin", 1.429),
            "vmax": kwargs.pop("vmax", 6.0),
            "scipy_options": {
                "disp": True,
                "eps": kwargs.pop("eps", 1e-15),
                "ftol": kwargs.pop("ftol", 1e-11),
                "maxiter": kwargs.pop("maxiter", 20),
            },
        }
        parameters.update(kwargs)

        template = self._guess_control_template()
        bounds = self._build_bounds(parameters["vmin"], parameters["vmax"], template)
        control_0 = self.get_control_vector(template)
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

        self.control_result = self._rebuild_control_from_vector(template, result.x)
        self.set_guess_control(self.control_result)

        if isinstance(self.control_result, fire.Function):
            self.vp_result = self._copy_control_structure(self.control_result)
            fire.VTKFile("vp_end.pvd").write(self.vp_result)
        else:
            self.vp_result = None
            self._write_control_snapshot(self.control_result, "vp_end.pvd")

        np.save("result", result.x)
        return result

    def run_fwi_rol(self, **kwargs):
        """
        Run full waveform inversion using ROL optimizer (deprecated).
        """
        if ROL is None:
            raise ImportError("The ROL module is not available.")
        if not isinstance(self._guess_control_template(), fire.Function):
            raise NotImplementedError(
                "ROL inversion only supports a single Firedrake Function control.",
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

        u = fire.Function(self.function_space, name="velocity").assign(self.guess_velocity_model)
        opt = FireVector(u.vector(), inner_product)

        xlo = fire.Function(self.function_space)
        xlo.interpolate(fire.Constant(vmin))
        x_lo = FireVector(xlo.vector(), inner_product)

        xup = fire.Function(self.function_space)
        xup.interpolate(fire.Constant(vmax))
        x_up = FireVector(xup.vector(), inner_product)

        bnd = ROL.Bounds(x_lo, x_up, 1.0)

        algo = ROL.Algorithm("Line Search", params)

        algo.run(opt, obj, bnd)

    def set_gradient_mask(self, boundaries=None):
        """
        DEPRECATED: Set a gradient mask to zero out gradients outside defined boundaries.

        The gradient mask is used to restrict updates to certain regions of
        the model domain, which is useful for excluding absorbing boundary
        layers or other regions where the velocity model should not be updated.

        This method is deprecated since we prefer tu use mesh based tags for now. In
        the future we will use the new submesh functionality in FIredrake

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

        if self.abc_active is False and boundaries is None:
            raise ValueError("If no abc boundary please define boundaries for the mask")
        elif self.abc_active and boundaries is None:
            mask_obj = Gradient_mask_for_pml(self)
        elif self.abc_active and boundaries is not None:
            warnings.warn("Boundaries overuling PML boundaries for mask")
            mask_obj = Mask(boundaries, self)
        elif self.abc_active is False and boundaries is not None:
            mask_obj = Mask(boundaries, self)
        else:
            raise ValueError("Mask options do not make sense")

        self.mask_obj = mask_obj

    def _apply_gradient_mask(self):
        """
        DEPRECATED Apply the gradient mask to the computed gradient.

        If a gradient mask has been set via set_gradient_mask(), this method
        applies the mask to zero out gradient values outside the defined region.
        This is called automatically during get_gradient().

        Notes
        -----
        This method is deprecated since we prefer tu use mesh based tags for now. In
        the future we will use the new submesh functionality in FIredrake
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
        load_shots(self, file_name=file_name)
        self.real_shot_record = self.forward_solution_receivers
        self.forward_solution_receivers = None

    @run_in_one_core
    def save_result_as_segy(self, file_name="final_vp.segy", grid_spacing=0.01):
        """
        Save the final velocity model result as a SEG-Y file.

        This method exports the final inverted velocity model to SEG-Y format,
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
        if self.vp_result is None:
            raise ValueError(
                "SEG-Y export requires a single scalar inversion control result.",
            )
        create_segy(
            self.vp_result,
            self.vp_result.function_space(),
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
                real_shot_record_list.append(self.receivers_output)
            self.real_shot_record = real_shot_record_list
        else:
            self.real_shot_record = self.receivers_output
