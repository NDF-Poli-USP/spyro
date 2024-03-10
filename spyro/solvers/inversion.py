import firedrake as fire
import warnings
from ROL.firedrake_vector import FiredrakeVector as FiredrakeVector
import ROL
from scipy.optimize import minimize as scipy_minimize
from mpi4py import MPI

from .acoustic_wave import AcousticWave
from ..utils import compute_functional
from ..plots import plot_model as spyro_plot_model


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

    def calculate_misfit(self):
        """
        Calculates the misfit, between the real shot record and the guess shot record.
        If the guess forward model has already been run it uses that value. Otherwise, it runs the forward model.
        """
        if self.mesh is None and self.guess_mesh is not None:
            self.mesh = self.guess_mesh
        if self.initial_velocity_model is None:
            self.initial_velocity_model = self.guess_velocity_model
        self.forward_solve()
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

    def get_functional(self):
        """
        Calculate and return the functional value.

        If the misfit is already computed, the functional value is calculated using the precomputed misfit.
        Otherwise, the misfit is calculated first and then the functional value is computed.

        Returns:
            float: The functional value.
        """
        if self.misfit is not None:
            Jm = compute_functional(self, self.misfit)
        else:
            self.calculate_misfit()
            Jm = compute_functional(self, self.misfit)

        self.functional = Jm

        return Jm

    def get_gradient(self, save=False):
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
        if self.misfit is None:
            self.get_functional()
        dJ = self.gradient_solve(misfit=self.misfit, forward_solution=self.guess_forward_solution)
        if save:
            fire.File("gradient.pvd").write(dJ)
        dJ_total = fire.Function(self.function_space)
        comm.comm.barrier()
        dJ_total = comm.allreduce(dJ, dJ_total)
        dJ_total /= comm.ensemble_comm.size
        if comm.comm.size > 1:
            dJ_total /= comm.comm.size
        self.gradient = dJ_total

    def return_functional_and_gradient(self):
        self.get_gradient()
        dJ = self.gradient.dat.data[:]
        return self.functional, dJ

    def run_fwi(self):
        """
        Run the full waveform inversion.
        """
        # if self.running_fwi is False:
        #     warnings.warn("Dictionary FWI options set to not run FWI.")
        # if self.current_iteration < self.iteration_limit:
        #     self.get_gradient()
        #     self.update_guess_model()
        #     self.current_iteration += 1
        # else:
        #     warnings.warn("Iteration limit reached. FWI stopped.")
        #     self.running_fwi = False
        vp_0 = self.c.vector().gather()
        vmin = 1.5
        vmax = 5.0
        bounds = [(vmin, vmax) for _ in range(len(vp_0))]
        options = {
            "disp": True,
            "eps": 1e-15,
            "gtol": 1e-15, "maxiter": 5,
        }
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
