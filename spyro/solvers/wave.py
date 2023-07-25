import os
import warnings
import firedrake as fire
from firedrake import sin, cos, pi  # noqa: F401
from SeismicMesh import write_velocity_model

from ..io import Model_parameters, interpolate
from .. import utils
from ..receivers.Receivers import Receivers
from ..sources.Sources import Sources
from ..domains.space import FE_method

fire.set_log_level(fire.ERROR)


class Wave(Model_parameters):
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
        self.current_time = 0.0
        self.set_solver_parameters()
        self.real_shot_record = None

        self.wavelet = self.get_wavelet()
        self.mesh = self.get_mesh()
        if self.mesh is not None and self.mesh is not False:
            self._build_function_space()
            if self.source_type == "ricker":
                self.sources = Sources(self)
            else:
                self.sources = None
            self.receivers = Receivers(self)
        elif self.mesh_type == "firedrake_mesh":
            warnings.warn(
                "No mesh file, Firedrake mesh will be automatically generated."
            )
        else:
            warnings.warn("No mesh found. Please define a mesh.")

    def set_mesh(
        self,
        dx=None,
        user_mesh=None,
        mesh_file=None,
        length_z=None,
        length_x=None,
        length_y=None,
        periodic=False,
    ):
        super().set_mesh(
            dx=dx,
            user_mesh=user_mesh,
            mesh_file=mesh_file,
            length_z=length_z,
            length_x=length_x,
            length_y=length_y,
            periodic=periodic,
        )

        self.mesh = self.get_mesh()
        self._build_function_space()
        if self.source_type == "ricker":
            self.sources = Sources(self)
        else:
            self.sources = None
        self.receivers = Receivers(self)
        if self.dimension == 2:
            z, x = fire.SpatialCoordinate(self.mesh)
            self.mesh_z = z
            self.mesh_x = x
        elif self.dimension == 3:
            z, x, y = fire.SpatialCoordinate(self.mesh)
            self.mesh_z = z
            self.mesh_x = x
            self.mesh_y = y

    def set_solver_parameters(self, parameters=None):
        if parameters is not None:
            self.solver_parameters = parameters
        elif parameters is None:
            if self.method == "mass_lumped_triangle":
                self.solver_parameters = {
                    "ksp_type": "preonly",
                    "pc_type": "jacobi",
                }
            elif self.method == "spectral_quadrilateral":
                self.solver_parameters = {
                    "ksp_type": "preonly",
                    "pc_type": "jacobi",
                }
            else:
                self.solver_parameters = None

    def get_spatial_coordinates(self):
        if self.dimension == 2:
            x, y = fire.SpatialCoordinate(self.mesh)
            return x, y
        elif self.dimension == 3:
            x, y, z = fire.SpatialCoordinate(self.mesh)
            return x, y, z

    def set_initial_velocity_model(
        self,
        constant=None,
        conditional=None,
        velocity_model_function=None,
        expression=None,
        new_file=None,
        output=False,
    ):
        """Method to define new user velocity model or file. It is optional.

        Parameters:
        -----------
        conditional:  (optional)

        velocity_model_function:  (optional)

        expression:  str (optional)
            If you use an expression, you can use the following variables:
            x, y, z, pi

        new_file:  (optional)
        """
        # Resseting old velocity model
        self.initial_velocity_model = None
        self.initial_velocity_model_file = None

        if conditional is not None:
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

    def _get_initial_velocity_model(self):
        if self.velocity_model_type == "conditional":
            self.set_initial_velocity_model(
                conditional=self.model_parameters.velocity_conditional
            )

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
                self.model_parameters,
                self.initial_velocity_model_file,
                self.function_space.sub(0),
            )

    def _build_function_space(self):
        self.function_space = FE_method(self.mesh, self.method, self.degree)

    def get_and_set_maximum_dt(self, fraction=1.0):
        if self.method == "KMV" or (
            self.method == "CG" and self.mesh.ufl_cell() == fire.quadrilateral
        ):
            estimate_max_eigenvalue = True
        else:
            estimate_max_eigenvalue = False

        dt = utils.estimate_timestep(
            self.mesh,
            self.function_space,
            self.c,
            estimate_max_eigenvalue=estimate_max_eigenvalue,
        )
        dt *= fraction
        self.dt = dt
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