import firedrake as fire
import warnings
from scipy.optimize import minimize as scipy_minimize
from mpi4py import MPI  # noqa: F401
import numpy as np
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


class FullWaveformInversion(AcousticWave):
    """
    Classical Full waveform inversion for acoustic wave data.

    This class implements full waveform inversion (FWI) as a subclass of
    AcousticWave. FWI is an optimization-based method for reconstructing
    subsurface velocity models from seismic data by minimizing the misfit
    between observed and simulated waveforms.

    Parameters
    ----------
    dictionary : dict, optional
        Dictionary containing parameters for the inversion, including mesh,
        timestepping, source/receiver configuration, and inversion options.
    comm : MPI.Comm, optional
        MPI communicator for parallel execution.

    Attributes
    ----------
    real_velocity_model : firedrake.Function or None
        The true velocity model, used only for generating synthetic data.
    real_velocity_model_file : str or None
        Path to file containing the true velocity model for synthetic tests.
    real_shot_record_files : str or None
        Path or prefix pattern for real/observed shot record files.
    guess_shot_record : array_like or None
        Shot records from the current guess velocity model.
    gradient : firedrake.Function or None
        Most recently computed gradient of the objective functional.
    current_iteration : int
        Current FWI iteration number, starts at 0.
    mesh_iteration : int
        Current mesh iteration for multiscale remeshing (default FWI doesn't use this).
    iteration_limit : int
        Maximum number of iterations, default is 100.
    inner_product : str
        Type of inner product for optimization, default is 'L2', only used in ROL.
    misfit : array_like or None
        Misfit between current forward solution and real observed data.
    guess_forward_solution : firedrake.Function or None
        Complete forward solution from the guess velocity model.
    has_gradient_mask : bool
        Whether a gradient mask has been set.
    gradient_mask_available : bool
        Whether gradient mask functionality is available.
    functional_history : list
        History of functional values at each iteration.
    control_out : firedrake.VTKFile
        VTK file object for saving velocity model iterates.
    gradient_out : firedrake.VTKFile
        VTK file object for saving gradient fields.

    Methods
    -------
    calculate_misfit(c=None)
        Calculate misfit between observed and simulated data.
    generate_real_shot_record(plot_model=False, ...)
        Generate synthetic shot records from the true velocity model.
    set_smooth_guess_velocity_model(real_velocity_model_file=None)
        Set a smoothed initial guess based on the true model.
    set_real_velocity_model(constant=None, ...)
        Set the true velocity model for synthetic tests.
    set_guess_velocity_model(constant=None, ...)
        Set the initial guess velocity model.
    set_real_mesh(user_mesh=None, input_mesh_parameters=None)
        Set the mesh for the true model.
    set_guess_mesh(user_mesh=None, input_mesh_parameters=None)
        Set the mesh for the guess/inversion model.
    get_functional(c=None)
        Compute the objective functional value.
    get_gradient(c=None, save=True, calculate_functional=True)
        Compute the gradient of the objective functional.
    return_functional_and_gradient(c)
        Compute and return both functional and gradient.
    run_fwi(**kwargs)
        Run the full waveform inversion using scipy.optimize.
    run_fwi_rol(**kwargs)
        Run the full waveform inversion using ROL (deprecated).
    set_gradient_mask(boundaries=None)
        Set a mask to zero out gradient values in certain regions.
    load_real_shot_record(file_name=\"shots/shot_record_\")
        Load observed shot records from files.
    save_result_as_segy(file_name=\"final_vp.segy\")
        Save the final inverted velocity model as SEG-Y.

    See Also
    --------
    AcousticWave : Parent class for acoustic wave simulation.

    Notes
    -----
    The inversion can be run using either scipy.optimize.minimize (L-BFGS-B)
    via run_fwi() or the deprecated ROL library via run_fwi_rol().

    Examples
    --------
    >>> fwi = FullWaveformInversion(dictionary=config_dict, comm=comm)
    >>> fwi.set_guess_mesh(input_mesh_parameters={'edge_length': 0.1})
    >>> fwi.set_guess_velocity_model(constant=2.0)
    >>> fwi.load_real_shot_record(\"shots/observed_\")
    >>> fwi.run_fwi(maxiter=50, vmin=1.5, vmax=4.5)
    """

    def __init__(self, dictionary=None, comm=None):
        """
        Initialize a FullWaveformInversion instance.

        Sets up the FWI problem with default optimization parameters and
        initializes all necessary attributes for the inversion process.

        Parameters
        ----------
        dictionary : dict, optional
            Dictionary containing parameters for the inversion.
        comm : MPI.Comm, optional
            MPI communicator for parallel execution.
        """
        super().__init__(dictionary=dictionary, comm=comm)
        default_optimization_parameters = {
            "General": {"Secant": {
                "Type": "Limited-Memory BFGS",
                "Maximum Storage": 10
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
        self.input_dictionary["inversion"].setdefault("initial_guess_model_file", None)
        self.input_dictionary["inversion"].setdefault("optimization_parameters", default_optimization_parameters)
        self.input_dictionary["inversion"].setdefault("real_shot_record_file", None)
        self.input_dictionary["inversion"].setdefault("control_output_file", "fwi/control.pvd")
        self.input_dictionary["inversion"].setdefault("gradient_output_file", "fwi/gradient.pvd")
        self.input_dictionary["inversion"].setdefault("real_velocity_model_file", None)
        inversion_dictionary = self.input_dictionary["inversion"]

        self.real_velocity_model = None
        self.real_velocity_model_file = inversion_dictionary["real_velocity_model_file"]
        self.real_shot_record_files = inversion_dictionary["real_shot_record_file"]
        self.control_out = fire.VTKFile(inversion_dictionary["control_output_file"])
        self.gradient_out = fire.VTKFile(inversion_dictionary["gradient_output_file"])
        self.guess_shot_record = None
        self.gradient = None
        self.current_iteration = 0
        self.mesh_iteration = 0
        self.iteration_limit = 100
        self.inner_product = 'L2'
        self.misfit = None
        self.guess_forward_solution = None
        self.has_gradient_mask = False
        self.gradient_mask_available = False
        self.functional_history = []

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
        if value is not None:
            if not os.path.exists(value):
                raise FileNotFoundError(f"Velocity model file '{value}' does not exist")
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

        This setter also initializes the control and gradient output files
        and loads the real shot record if a valid path is provided.

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
            # Check if it's a file prefix pattern by looking for matching files
            if not os.path.exists(value) and not glob.glob(value + "*"):
                raise FileNotFoundError(f"Shot record file '{value}' does not exist")
        self._real_shot_record_files = value
        self.control_out = fire.VTKFile("results/control.pvd")
        self.gradient_out = fire.VTKFile("results/gradient.pvd")
        if self.real_shot_record_files is not None:
            self.load_real_shot_record(file_name=self.real_shot_record_files)

    def calculate_misfit(self, c=None):
        """
        Calculate the misfit between observed and simulated data.

        Runs the forward model with the current velocity model and computes
        the difference between the simulated shot records and the real/observed
        shot records. If the forward model has already been solved, uses the
        existing solution.

        Parameters
        ----------
        c : array_like, optional
            Velocity model values to use. If provided, updates the initial
            velocity model before running the forward solve.

        Returns
        -------
        misfit : ndarray or list of ndarray
            Misfit between real and simulated shot records. Returns a list
            if using spatial parallelism with multiple sources, otherwise
            returns a single array.

        Notes
        -----
        This method also saves the current velocity model and shot records
        to disk for debugging and checkpoint purposes.
        """
        if self.mesh is None and self.guess_mesh is not None:
            self.mesh = self.guess_mesh
        if self.initial_velocity_model is None:
            self.initial_velocity_model = self.guess_velocity_model
        if c is not None:
            self.initial_velocity_model.dat.data[:] = c
        self.forward_solve()
        output = fire.File("control_" + str(self.current_iteration)+".pvd")
        output.write(self.c)
        np.save(f"control{self.comm.ensemble_comm.rank}_{self.comm.comm.rank}", self.c.dat.data[:])
        if self.parallelism_type == "spatial" and self.number_of_sources > 1:
            misfit_list = []
            guess_shot_record_list = []
            for snum in range(self.number_of_sources):
                switch_serial_shot(self, snum)
                guess_shot_record_list.append(self.forward_solution_receivers)
                misfit_list.append(self.real_shot_record[snum] - self.forward_solution_receivers)
            self.guess_shot_record = guess_shot_record_list
            self.misfit = misfit_list
        else:
            self.guess_shot_record = self.forward_solution_receivers
            self.guess_forward_solution = self.forward_solution
            self.misfit = self.real_shot_record - self.guess_shot_record
        return self.misfit

    def generate_real_shot_record(self, plot_model=False, model_filename="model.png", abc_points=None, save_shot_record=True, shot_filename="shots/shot_record_", high_resolution_model=False):
        """
        Generate synthetic shot records from the true velocity model.

        Creates a SyntheticRealAcousticWave object with the true velocity model,
        solves the forward problem, and optionally saves the shot records and
        plots the model. This is used only for synthetic test cases.

        Parameters
        ----------
        plot_model : bool, optional
            If True, plot and save the velocity model. Default is False.
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
        Wave_obj_real_velocity = SyntheticRealAcousticWave(dictionary=self.input_dictionary, comm=self.comm)
        if Wave_obj_real_velocity.mesh is None and self.real_mesh is not None:
            Wave_obj_real_velocity.mesh = self.real_mesh
        if Wave_obj_real_velocity.initial_velocity_model is None:
            Wave_obj_real_velocity.initial_velocity_model = self.real_velocity_model

        if plot_model and Wave_obj_real_velocity.comm.comm.rank == 0 and Wave_obj_real_velocity.comm.ensemble_comm.rank == 0:
            spyro_plot_model(Wave_obj_real_velocity, filename=model_filename, abc_points=abc_points, high_resolution=high_resolution_model)

        Wave_obj_real_velocity.forward_solve()
        if save_shot_record:
            save_shots(Wave_obj_real_velocity, file_name=shot_filename)
        self.real_shot_record = Wave_obj_real_velocity.real_shot_record
        self.quadrature_rule = Wave_obj_real_velocity.quadrature_rule

    def set_smooth_guess_velocity_model(self, real_velocity_model_file=None):
        """
        Set a smoothed initial guess based on the true velocity model.

        This method is intended to create a smooth initial guess from a known
        true velocity model for synthetic tests. Currently a placeholder.

        Parameters
        ----------
        real_velocity_model_file : str, optional
            Path to the file containing the true velocity model. If not provided,
            uses self.real_velocity_model_file.

        Notes
        -----
        TODOThis method currently does not implement the smoothing operation and
        may need to be completed for actual use.
        """
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
        """
        Set the true velocity model for synthetic test cases.

        This method sets the real/true velocity model that is used only for
        generating synthetic observed data. It wraps the parent class's
        set_initial_velocity_model method.

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
        super().set_initial_velocity_model(
            constant=constant,
            conditional=conditional,
            velocity_model_function=velocity_model_function,
            expression=expression,
            new_file=new_file,
            output=output,
            dg_velocity_model=dg_velocity_model,
        )
        self.real_velocity_model = self.initial_velocity_model

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
        It wraps the parent class's set_initial_velocity_model method and
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
        super().set_initial_velocity_model(
            constant=constant,
            conditional=conditional,
            velocity_model_function=velocity_model_function,
            expression=expression,
            new_file=new_file,
            output=output,
            dg_velocity_model=dg_velocity_model,
        )
        self.guess_velocity_model = self.initial_velocity_model
        self.misfit = None

    def set_real_mesh(
        self,
        user_mesh=None,
        input_mesh_parameters=None,
    ):
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
        input_mesh_parameters.setdefault("mesh_type", "firedrake_mesh")
        super().set_mesh(
            user_mesh=user_mesh,
            input_mesh_parameters=input_mesh_parameters,
        )
        self.real_mesh = self.get_mesh()

    def set_guess_mesh(
        self,
        user_mesh=None,
        input_mesh_parameters=None,
    ):
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
        super().set_mesh(
            user_mesh=user_mesh,
            input_mesh_parameters=input_mesh_parameters,
        )
        self.guess_mesh = self.get_mesh()

    def get_functional(self, c=None):
        """
        Calculate and return the objective functional value.

        Computes the misfit if needed, then evaluates the objective functional.
        Also tracks the functional history and peak memory usage.

        Parameters
        ----------
        c : array_like, optional
            Velocity model values to use for the calculation. If provided,
            updates the initial velocity model.

        Returns
        -------
        Jm : float
            The objective functional value (typically L2 norm of misfit).

        Notes
        -----
        This method writes the functional value and memory usage to text files
        for tracking convergence and resource consumption.
        """
        self.calculate_misfit(c=c)
        Jm = compute_functional(self, self.misfit)

        self.functional_history.append(Jm)
        self.functional = Jm
        peak_memory_mb = get_peak_memory()
        # Save the functional value to a text file
        if self.comm.ensemble_comm.rank == 0 and self.comm.comm.rank == 0:
            print(f"Functional: {Jm} at iteration: {self.current_iteration}", flush=True)
            with open("functional_values.txt", "a") as file:
                file.write(f"Iteration: {self.current_iteration}, Functional: {Jm}\n")

            with open("peak_memory.txt", "a") as file:
                file.write(f"Peak memory usage: {peak_memory_mb:.2f} MB \n")

        return Jm

    def get_gradient(self, c=None, save=True, calculate_functional=True):
        """
        Calculate the gradient of the objective functional.

        Computes the gradient with respect to the velocity model using the
        adjoint method. Optionally calculates the functional value first and
        saves the gradient to a VTK file.

        Parameters
        ----------
        c : array_like, optional
            Velocity model values to use. If provided and calculate_functional
            is True, updates the model before computing the functional.
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
        adjoint-state method implemented in gradient_solve().
        """
        comm = self.comm
        if getattr(self, "adjoint_type", None) is None or self.adjoint_type.name == "NONE":
            self.enable_spyro_adjoint()
        if calculate_functional:
            self.get_functional(c=c)
        comm.comm.barrier()
        self.gradient = self.gradient_solve(
            forward_solution=self.guess_forward_solution
        )
        self._apply_gradient_mask()
        if save:
            # self.gradient_out.write(dJ_total)
            output = fire.File("gradient_" + str(self.current_iteration)+".pvd")
            output.write(self.gradient)
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
            Current velocity model values.

        Returns
        -------
        functional : float
            The objective functional value.
        dJ : ndarray
            The gradient of the functional with respect to the velocity model.
        """
        self.get_gradient(c=c)
        dJ = self.gradient.dat.data[:]
        return self.functional, dJ

    def run_fwi(self, **kwargs):
        """
        Run full waveform inversion using scipy L-BFGS-B optimizer.

        Performs the complete FWI optimization using scipy.optimize.minimize
        with the L-BFGS-B method. The optimization minimizes the misfit between
        observed and simulated data by updating the velocity model.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for customizing the optimization:

            vmin : float, optional
                Minimum velocity bound. Default is 1.429 km/s.
            vmax : float, optional
                Maximum velocity bound. Default is 6.0 km/s.
            maxiter : int, optional
                Maximum number of iterations. Default is 20.
            scipy_options : dict, optional
                Additional options passed to scipy.optimize.minimize.
                Default includes disp=True, eps=1e-15, ftol=1e-11.

        Notes
        -----
        The final inverted velocity model is stored in self.vp_result and saved
        to "vp_end.pvd". The raw result array is also saved to "result.npy".

        This method uses the L-BFGS-B algorithm which is well-suited for
        large-scale bound-constrained optimization problems.

        Examples
        --------
        >>> fwi.run_fwi(maxiter=100, vmin=1.5, vmax=5.0)
        """
        parameters = {
            "vmin": kwargs.pop("vmin", 1.429),
            "vmax": kwargs.pop("vmax", 6.0),
            "scipy_options": {
                "disp": True,
                "eps": kwargs.pop("eps", 1e-15),
                "ftol": kwargs.pop("ftol", 1e-11),
                "maxiter": kwargs.pop("maxiter", 20),
            }
        }
        parameters.update(kwargs)

        vmin = parameters["vmin"]
        vmax = parameters["vmax"]
        vp_0 = self.initial_velocity_model.vector()
        bounds = [(vmin, vmax) for _ in range(len(vp_0))]
        options = parameters["scipy_options"]

        # if self.running_fwi is False:
        #     warnings.warn("Dictionary FWI options set to not run FWI.")
        # if self.current_iteration < self.iteration_limit:
        #     self.get_gradient()
        #     self.update_guess_model()
        #     self.current_iteration += 1
        # else:
        #     warnings.warn("Iteration limit reached. FWI stopped.")
        #     self.running_fwi = False
        result = scipy_minimize(
            self.return_functional_and_gradient,
            vp_0,
            method="L-BFGS-B",
            jac=True,
            tol=1e-15,
            bounds=bounds,
            options=options,
        )

        vp_end = fire.Function(self.function_space)
        vp_end.dat.data[:] = result.x
        self.vp_result = vp_end
        fire.File("vp_end.pvd").write(vp_end)
        np.save("result", result.x)

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
                Minimum velocity bound. Default is 1.429 km/s.
            vmax : float, optional
                Maximum velocity bound. Default is 6.0 km/s.
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
            }
        }
        parameters.update(kwargs)
        vmin = parameters["vmin"]
        vmax = parameters["vmax"]

        warnings.warn("This functionality is deprecated, since the pyROL library is no longer supported.")
        params = ROL.ParameterList(parameters["ROL_options"], "Parameters")

        inner_product = L2Inner(self)

        obj = Objective(inner_product, self)

        u = fire.Function(self.function_space, name="velocity").assign(self.guess_velocity_model)
        opt = FireVector(u.vector(), inner_product)

        # Add control bounds to the problem (uses more RAM)
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
        create_segy(self.vp_result, self.function_space, grid_spacing, file_name)


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
