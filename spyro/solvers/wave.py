import os
from abc import abstractmethod
import warnings
import firedrake as fire
from firedrake import sin, cos, pi, tanh, sqrt  # noqa: F401
from SeismicMesh import write_velocity_model

from ..io import Model_parameters, interpolate
from .. import utils
from ..receivers.Receivers import Receivers
from ..sources.Sources import Sources
from ..domains.space import FE_method
from .solver_parameters import get_default_parameters_for_method

fire.set_log_level(fire.ERROR)


class Wave(Model_parameters):
    """
    Base class for wave equation solvers.

    Attributes:
    -----------
    comm: MPI communicator

    initial_velocity_model: firedrake function
        Initial velocity model
    function_space: firedrake function space
        Function space for the wave equation
    current_time: float
        Current time of the simulation
    solver_parameters: Python object
        Contains solver parameters
    real_shot_record: firedrake function
        Real shot record
    wavelet: list of floats
        Values at timesteps of wavelet used in the simulation
    mesh: firedrake mesh
        Mesh used in the simulation (2D or 3D)
    mesh_z: symbolic coordinate z of the mesh object
    mesh_x: symbolic coordinate x of the mesh object
    mesh_y: symbolic coordinate y of the mesh object
    sources: Sources object
        Contains information about sources
    receivers: Receivers object
        Contains information about receivers

    Methods:
    --------
    set_mesh: sets or calculates new mesh
    set_solver_parameters: sets new or default solver parameters
    get_spatial_coordinates: returns spatial coordinates of mesh
    set_initial_velocity_model: sets initial velocity model
    get_and_set_maximum_dt: calculates and/or sets maximum dt
    get_mass_matrix_diagonal: returns diagonal of mass matrix
    set_last_solve_as_real_shot_record: sets last solve as real shot record
    """

    def __init__(self, dictionary=None, comm=None):
        """Wave object solver. Contains both the forward solver
        and gradient calculator methods.

        Parameters:
        -----------
        comm: MPI communicator

        model_parameters: Python object
            Contains model parameters
        """
        super().__init__(dictionary=dictionary, comm=comm)
        self.initial_velocity_model = None

        self.function_space = None
        self.forward_solution_receivers = None
        self.current_time = 0.0
        self.set_solver_parameters()

        self.wavelet = self.get_wavelet()
        self.mesh = self.get_mesh()
        self.c = None
        if self.mesh is not None:
            self._build_function_space()
            self._map_sources_and_receivers()
        elif self.mesh_type == "firedrake_mesh":
            warnings.warn(
                "No mesh file, Firedrake mesh will be automatically generated."
            )
        else:
            warnings.warn("No mesh found. Please define a mesh.")

    @abstractmethod
    def forward_solve(self):
        """Solves the forward problem."""
        pass

    @abstractmethod
    def matrix_building(self):
        """Builds the matrix for the forward problem."""
        pass

    def set_mesh(
            self,
            user_mesh=None,
            mesh_parameters=None,
        ):
            """
            Set the mesh for the solver.

            Args:
                user_mesh (optional): User-defined mesh. Defaults to None.
                mesh_parameters (optional): Parameters for generating a mesh. Defaults to None.
            """
            super().set_mesh(
                user_mesh=user_mesh,
                mesh_parameters=mesh_parameters,
            )

            self.mesh = self.get_mesh()
            self._build_function_space()
            self._map_sources_and_receivers()

    def set_solver_parameters(self, parameters=None):
            """
            Set the solver parameters.

            Args:
                parameters (dict): A dictionary containing the solver parameters.

            Returns:
                None
            """
            if parameters is not None:
                self.solver_parameters = parameters
            elif parameters is None:
                self.solver_parameters = get_default_parameters_for_method(
                    self.method
                )

    def get_spatial_coordinates(self):
        if self.dimension == 2:
            return self.mesh_z, self.mesh_x
        elif self.dimension == 3:
            return self.mesh_z, self.mesh_x, self.mesh_y

    def set_initial_velocity_model(
        self,
        constant=None,
        conditional=None,
        velocity_model_function=None,
        expression=None,
        new_file=None,
        output=False,
        dg_velocity_model=True,
    ):
        """Method to define new user velocity model or file. It is optional.

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
        # If no mesh is set, we have to do it beforehand
        if self.mesh is None:
            self.set_mesh()
        # Resseting old velocity model
        self.initial_velocity_model = None
        self.initial_velocity_model_file = None

        if self.debug_output:
            output = True

        if conditional is not None:
            if dg_velocity_model:
                V = fire.FunctionSpace(self.mesh, "DG", 0)
            else:
                V = self.function_space
            vp = fire.Function(V, name="velocity")
            vp.interpolate(conditional)
            self.initial_velocity_model = vp
        elif expression is not None:
            z = self.mesh_z  # noqa: F841
            x = self.mesh_x  # noqa: F841
            if self.dimension == 3:
                y = self.mesh_y  # noqa: F841
            expression = eval(expression)
            V = self.function_space
            vp = fire.Function(V, name="velocity")
            vp.interpolate(expression)
            self.initial_velocity_model = vp
        elif velocity_model_function is not None:
            self.initial_velocity_model = velocity_model_function
        elif new_file is not None:
            self.initial_velocity_model_file = new_file
        elif constant is not None:
            V = self.function_space
            vp = fire.Function(V, name="velocity")
            vp.interpolate(fire.Constant(constant))
            self.initial_velocity_model = vp
        else:
            raise ValueError(
                "Please specify either a conditional, expression, firedrake \
                    function or new file name (segy or hdf5)."
            )
        if output:
            fire.File("initial_velocity_model.pvd").write(
                self.initial_velocity_model, name="velocity"
            )

    def _map_sources_and_receivers(self):
        if self.source_type == "ricker":
            self.sources = Sources(self)
        else:
            self.sources = None
        self.receivers = Receivers(self)

    def _get_initial_velocity_model(self):
        if self.initial_velocity_model is not None:
            return None

        if self.initial_velocity_model_file is None:
            raise ValueError("No velocity model or velocity file to load.")

        if self.initial_velocity_model_file.endswith(".segy"):
            vp_filename, vp_filetype = os.path.splitext(
                self.initial_velocity_model_file
            )
            warnings.warn("Converting segy file to hdf5")
            write_velocity_model(
                self.initial_velocity_model_file, ofname=vp_filename
            )
            self.initial_velocity_model_file = vp_filename + ".hdf5"

        if self.initial_velocity_model_file.endswith(".hdf5"):
            self.initial_velocity_model = interpolate(
                self,
                self.initial_velocity_model_file,
                self.function_space.sub(0),
            )

        if self.debug_output:
            fire.File("initial_velocity_model.pvd").write(
                self.initial_velocity_model, name="velocity"
            )

    def _build_function_space(self):
        self.function_space = FE_method(self.mesh, self.method, self.degree)
        if self.dimension == 2:
            z, x = fire.SpatialCoordinate(self.mesh)
            self.mesh_z = z
            self.mesh_x = x
        elif self.dimension == 3:
            z, x, y = fire.SpatialCoordinate(self.mesh)
            self.mesh_z = z
            self.mesh_x = x
            self.mesh_y = y

    def get_and_set_maximum_dt(self, fraction=0.7, estimate_max_eigenvalue=False):
        """
        Calculates and sets the maximum stable time step (dt) for the wave solver.

        Args:
            fraction (float, optional):
                Fraction of the estimated time step to use. Defaults to 0.7.
            estimate_max_eigenvalue (bool, optional):
                Whether to estimate the maximum eigenvalue. Defaults to False.

        Returns:
            float: The calculated maximum time step (dt).
        """
        # if self.method == "mass_lumped_triangle":
        #     estimate_max_eigenvalue = True
        # elif self.method == "spectral_quadrilateral":
        #     estimate_max_eigenvalue = True
        # else:

        if self.c is None:
            c = self.initial_velocity_model
        else:
            c = self.c

        dt = utils.estimate_timestep.estimate_timestep(
            self.mesh,
            self.function_space,
            c,
            estimate_max_eigenvalue=estimate_max_eigenvalue,
        )
        dt *= fraction
        nt = int(self.final_time / dt) + 1
        dt = self.final_time / (nt - 1)

        self.dt = dt

        self.wavelet = self.get_wavelet()
        return dt

    def get_mass_matrix_diagonal(self):
        """Builds a section of the mass matrix for debugging purposes."""
        A = self.solver.A
        petsc_matrix = A.petscmat
        diagonal = petsc_matrix.getDiagonal()
        return diagonal.array

    def set_last_solve_as_real_shot_record(self):
        if self.current_time == 0.0:
            raise ValueError("No previous solve to set as real shot record.")
        self.real_shot_record = self.forward_solution_receivers
