import firedrake as fire
import warnings
from scipy.optimize import minimize as scipy_minimize
from mpi4py import MPI
import numpy as np

from .acoustic_wave import AcousticWave
from ..utils import compute_functional
from ..utils import Gradient_mask_for_pml, Mask
from ..plots import plot_model as spyro_plot_model

try:
    from ROL.firedrake_vector import FiredrakeVector as FireVector
    import ROL
    RObjective = ROL.Objective
except ImportError:
    ROL = None
    RObjective = object

# ROL = None


class L2Inner(object):
    def __init__(self, Wave_obj):
        V = Wave_obj.function_space
        # print(f"Dir {dir(Wave_obj)}", flush=True)
        dxlump = fire.dx(scheme=Wave_obj.quadrature_rule)
        self.A = fire.assemble(
            fire.TrialFunction(V) * fire.TestFunction(V) * dxlump,
            mat_type="matfree"
        )
        self.Ap = fire.as_backend_type(self.A).mat()

    def eval(self, _u, _v):
        upet = fire.as_backend_type(_u).vec()
        vpet = fire.as_backend_type(_v).vec()
        A_u = self.Ap.createVecLeft()
        self.Ap.mult(upet, A_u)
        return vpet.dot(A_u)


class Objective(RObjective):
    def __init__(self, inner_product, FWI_obj):
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
        """Compute the functional"""
        J_total = np.zeros((1))
        self.inversion_obj.misfit = None
        self.inversion_obj.reset_pressure()
        Jm = self.inversion_obj.get_functional()
        self.misfit = self.inversion_obj.misfit
        J_total[0] += Jm

        return J_total[0]

    def gradient(self, g, x, tol):
        """Compute the gradient of the functional"""
        self.inversion_obj.get_gradient(calculate_functional=False)
        dJ = self.inversion_obj.gradient
        g.scale(0)
        g.vec += dJ

    def update(self, x, flag, iteration):
        vp = self.inversion_obj.initial_velocity_model
        vp.assign(fire.Function(
            self.inversion_obj.function_space,
            x.vec,
            name="velocity")
        )


class FullWaveformInversion(AcousticWave):
    """
    The FullWaveformInversion class is a subclass of the AcousticWave class.
    It is used to perform full waveform inversion on acoustic wave data.

    Attributes:
    -----------
    dictionary: (dict)
        A dictionary containing parameters for the inversion.
    comm: MPI communicator
        A communicator for parallel execution.
    real_velocity_model:
        The real velocity model. Is used only when generating synthetic shot records
    real_velocity_model_file: (str)
        The file containing the real velocity model. Is used only when generating synthetic shot records
    guess_shot_record:
        The guess shot record.
    gradient: Firedrake function
        The most recent gradient.
    current_iteration: (int)
        The current iteration. Starts at 0.
    mesh_iteration: (int)
        The current mesh iteration when using multiscale remeshing. Starts at 0., and is not used with default FWI.
    iteration_limit: (int)
        The iteration limit. Default is 100.
    inner_product: (str)
        The inner product. Default is 'L2'.
    misfit:
        The misfit between the current forward shot record and the real observed data.
    guess_forward_solution:
        The guess forward solution.

    Methods:
    --------
    __init__(self, dictionary=None, comm=None):
        Initializes a new instance of the FullWaveformInversion class.
    calculate_misfit():
        Calculates the misfit.
    generate_real_shot_record():
        Generates the real synthetic shot record.
    set_smooth_guess_velocity_model(real_velocity_model_file=None):
        Sets the smooth guess velocity model.
    set_real_velocity_model(constant=None, conditional=None, velocity_model_function=None, expression=None, new_file=None, output=False):
        Sets the real velocity model.
    set_guess_velocity_model(constant=None, conditional=None, velocity_model_function=None, expression=None, new_file=None, output=False):
        Sets the guess velocity model.
    set_real_mesh(user_mesh=None, mesh_parameters=None):
        Sets the real mesh.
    set_guess_mesh(user_mesh=None, mesh_parameters=None):
        Sets the guess mesh.
    get_functional():
        Gets the functional.
    get_gradient(save=False):
        Gets the gradient.
    """

    def __init__(self, dictionary=None, comm=None):
        """
        Initializes a new instance of the FullWaveformInversion class.

        Parameters:
        -----------
        dictionary: (dict)
            A dictionary containing parameters for the inversion.
        comm: MPI communicator
            A communicator for parallel execution.

        Returns:
        --------
        None
        """
        super().__init__(dictionary=dictionary, comm=comm)
        if self.running_fwi is False:
            warnings.warn("Dictionary FWI options set to not run FWI.")
        self.real_velocity_model = None
        self.real_velocity_model_file = None
        self.guess_shot_record = None
        self.gradient = None
        self.current_iteration = 0
        self.mesh_iteration = 0
        self.iteration_limit = 100
        self.inner_product = 'L2'
        self.misfit = None
        self.guess_forward_solution = None
        self.has_gradient_mask = False
        self.functional_history = []
        self.control_out = fire.File("results/control.pvd")
        self.gradient_out = fire.File("results/gradient.pvd")

    def calculate_misfit(self, c=None):
        """
        Calculates the misfit, between the real shot record and the guess shot record.
        If the guess forward model has already been run it uses that value. Otherwise, it runs the forward model.
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
        self.guess_shot_record = self.forward_solution_receivers
        self.guess_forward_solution = self.forward_solution

        self.misfit = self.real_shot_record - self.guess_shot_record
        return self.misfit

    def generate_real_shot_record(self, plot_model=False, filename=None, abc_points=None):
        """
        Generates the real synthetic shot record. Only for use in synthetic test cases.
        """
        Wave_obj_real_velocity = SyntheticRealAcousticWave(dictionary=self.input_dictionary, comm=self.comm)
        if Wave_obj_real_velocity.mesh is None and self.real_mesh is not None:
            Wave_obj_real_velocity.mesh = self.real_mesh
        if Wave_obj_real_velocity.initial_velocity_model is None:
            Wave_obj_real_velocity.initial_velocity_model = self.real_velocity_model

        if plot_model and Wave_obj_real_velocity.comm.comm.rank == 0 and Wave_obj_real_velocity.comm.ensemble_comm.rank == 0:
            spyro_plot_model(Wave_obj_real_velocity, filename=filename, abc_points=abc_points)

        Wave_obj_real_velocity.forward_solve()
        self.real_shot_record = Wave_obj_real_velocity.real_shot_record
        self.quadrature_rule = Wave_obj_real_velocity.quadrature_rule

    def set_smooth_guess_velocity_model(self, real_velocity_model_file=None):
        """
        Sets the smooth guess velocity model based on the real one.

        Parameters:
        -----------
        real_velocity_model_file: (str)
            The file containing the real velocity model. Is used only when generating synthetic shot records.
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
        """"
        Sets the real velocity model. Only to be used for synthetic cases.

        Parameters:
        -----------
        conditional:  (optional)
            Firedrake conditional object.
        velocity_model_function: Firedrake function (optional)
            Firedrake function to be used as the velocity model. Has to be in the same function space as the object.
        expression:  str (optional)
            If you use an expression, you can use the following variables:
            x, y, z, pi, tanh, sqrt. Example: "2.0 + 0.5*tanh((x-2.0)/0.1)".
            It will be interpoalte into either the same function space as the object or a DG0 function space
            in the same mesh.
        new_file:  str (optional)
            Name of the file containing the velocity model.
        output:  bool (optional)
            If True, outputs the velocity model to a pvd file for visualization.
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
    ):
        """"
        Sets the initial guess.

        Parameters:
        -----------
        conditional:  (optional)
            Firedrake conditional object.
        velocity_model_function: Firedrake function (optional)
            Firedrake function to be used as the velocity model. Has to be in the same function space as the object.
        expression:  str (optional)
            If you use an expression, you can use the following variables:
            x, y, z, pi, tanh, sqrt. Example: "2.0 + 0.5*tanh((x-2.0)/0.1)".
            It will be interpoalte into either the same function space as the object or a DG0 function space
            in the same mesh.
        new_file:  str (optional)
            Name of the file containing the velocity model.
        output:  bool (optional)
            If True, outputs the velocity model to a pvd file for visualization.
        """
        super().set_initial_velocity_model(
            constant=constant,
            conditional=conditional,
            velocity_model_function=velocity_model_function,
            expression=expression,
            new_file=new_file,
            output=output,
        )
        self.guess_velocity_model = self.initial_velocity_model
        self.misfit = None

    def set_real_mesh(
        self,
        user_mesh=None,
        mesh_parameters=None,
    ):
        """
        Set the mesh for the real synthetic model.

        Parameters
        ----------
        user_mesh : spyro.Mesh, optional
            The desired mesh. The default is None.
        mesh_parameters : dict, optional
            Additional parameters for setting up the mesh. The default is an empty dictionary.

        Returns
        -------
        None
        """
        super().set_mesh(
            user_mesh=user_mesh,
            mesh_parameters=mesh_parameters,
        )
        self.real_mesh = self.get_mesh()

    def set_guess_mesh(
        self,
        user_mesh=None,
        mesh_parameters=None,
    ):
        """
        Set the mesh for the guess model.

        Parameters
        ----------
        user_mesh : spyro.Mesh, optional
            The desired mesh. The default is None.
        mesh_parameters : dict, optional
            Additional parameters for setting up the mesh. The default is an empty dictionary.

        Returns
        -------
        None
        """
        super().set_mesh(
            user_mesh=user_mesh,
            mesh_parameters=mesh_parameters,
        )
        self.guess_mesh = self.get_mesh()

    def get_functional(self, c=None):
        """
        Calculate and return the functional value.

        If the misfit is already computed, the functional value is calculated using the precomputed misfit.
        Otherwise, the misfit is calculated first and then the functional value is computed.

        Returns:
            float: The functional value.
        """
        self.calculate_misfit(c=c)
        Jm = compute_functional(self, self.misfit)

        self.functional_history.append(Jm)
        self.functional = Jm

        return Jm

    def get_gradient(self, c=None, save=True, calculate_functional=True):
        """
        Calculates the gradient of the functional with respect to the model parameters.

        Parameters:
        -----------
        save (bool, optional):
            Whether to save the gradient as a pvd file. Defaults to False.

        Returns:
        --------
        Firedrake function
        """
        comm = self.comm
        if calculate_functional:
            self.get_functional(c=c)
        comm.comm.barrier()
        dJ = self.gradient_solve(misfit=self.misfit, forward_solution=self.guess_forward_solution)
        dJ_total = fire.Function(self.function_space)
        comm.comm.barrier()
        dJ_total = comm.allreduce(dJ, dJ_total)
        dJ_total /= comm.ensemble_comm.size
        if comm.comm.size > 1:
            dJ_total /= comm.comm.size
        self.gradient = dJ_total
        self._apply_gradient_mask()
        if save and comm.comm.rank == 0:
            # self.gradient_out.write(dJ_total)
            output = fire.File("gradient_" + str(self.current_iteration)+".pvd")
            output.write(dJ_total)
            print("DEBUG")
        self.current_iteration += 1
        comm.comm.barrier()

    def return_functional_and_gradient(self, c):
        self.get_gradient(c=c)
        dJ = self.gradient.dat.data[:]
        return self.functional, dJ

    def run_fwi(self, **kwargs):
        """
        Run the full waveform inversion.
        """
        parameters = {
            "vmin": 1.429,
            "vmax": 6.0,
            "scipy_options": {
                "disp": True,
                "eps": 1e-15,
                "gtol": 1e-15, "maxiter": kwargs.pop("maxiter", 20),
            }
        }
        parameters.update(kwargs)

        vmin = parameters["vmin"]
        vmax = parameters["vmax"]
        vp_0 = self.initial_velocity_model.vector().gather()
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
        fire.File("vp_end.pvd").write(vp_end)

    def run_fwi_rol(self, **kwargs):
        """
        Run the full waveform inversion using ROL.
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
        Sets the gradient mask for zeroing gradient values outside defined boundaries.

        Args:
            boundaries (list, optional): List of boundary values for the mask. If not provided, 
                the method expects the abc_active to be True and uses PML locations for boundary
                values.

        Raises:
            ValueError: If no abc boundary is present in the object and boundaries is None.
            ValueError: If mask options do not make sense.

        Warnings:
            UserWarning: If abc_active is True and boundaries is not None, the boundaries will 
                override the PML boundaries for the mask.

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
            Applies a gradient mask to the gradient if it exists.

            If a gradient mask is available, this method applies the mask to the gradient
            using the `apply_mask` method of the `mask_obj`. If no gradient mask is available,
            this method does nothing.

            Parameters:
                None

            Returns:
                None
            """
            if self.has_gradient_mask:
                self.gradient = self.mask_obj.apply_mask(self.gradient)
            else:
                pass


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
        super().__init__(dictionary=dictionary, comm=comm)

    def forward_solve(self):
        super().forward_solve()
        self.real_shot_record = self.receivers_output
