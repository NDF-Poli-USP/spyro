import os
import warnings
import firedrake as fire
from SeismicMesh import write_velocity_model

from ..io import Model_parameters, interpolate
from .. import utils
from ..receivers.Receivers import Receivers
from ..sources.Sources import Sources
from ..domains.space import FE_method

fire.set_log_level(fire.ERROR)


class Wave(Model_parameters):
    def __init__(self, dictionary= None, comm = None):
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
        
        self.wavelet = self.get_wavelet()
        self.mesh = self.get_mesh()
        if self.mesh != None and self.mesh != False:
            self._build_function_space()
            if self.source_type == 'ricker':
                self.sources = Sources(self)
            else:
                self.sources = None
            self.receivers = Receivers(self)
        else:
            warnings.warn('No mesh found. Please define a mesh.')
    
    def set_mesh(self, dx=None, user_mesh=None, mesh_file=None, length_z=None, length_x=None, length_y=None, periodic=False):
        super().set_mesh(dx=dx, user_mesh=user_mesh, mesh_file=mesh_file, length_z=length_z, length_x=length_x, length_y=length_y, periodic=periodic)

        self.mesh = self.get_mesh()
        self._build_function_space()
        if self.source_type == 'ricker':
            self.sources = Sources(self)
        else:
            self.sources = None
        self.receivers = Receivers(self)

    def set_solver_parameters(self, parameters = None):
        if   parameters != None:
            self.solver_parameters = parameters
        elif parameters == None:
            if   self.method == 'mass_lumped_triangle':
                self.solver_parameters = {"ksp_type": "preonly", "pc_type": "jacobi"}
            elif self.method == 'spectral_quadrilateral':
                self.solver_parameters = {"ksp_type": "preonly", "pc_type": "jacobi"}
            else:
                self.solver_parameters = None

    def get_spatial_coordinates(self):
        if self.dimension == 2:
            x, y = fire.SpatialCoordinate(self.mesh)
            return x, y
        elif self.dimension == 3:
            x, y, z = fire.SpatialCoordinate(self.mesh)
            return x, y, z
    
    def set_initial_velocity_model(self, constant=None, conditional= None, velocity_model_function = None, expression = None, new_file = None):
        """Method to define new user velocity model or file. It is optional.

        Parameters:
        -----------
        conditional:  (optional)

        velocity_model_functional:  (optional)

        expression:  (optional)

        new_file:  (optional)
        """
        #Resseting old velocity model
        self.initial_velocity_model = None
        self.initial_velocity_model_file = None

        if conditional != None:
            V = self.function_space
            vp = fire.Function(V)
            vp.interpolate(conditional)
            self.initial_velocity_model = vp
        elif expression != None:
            V = self.function_space
            vp = fire.Function(V)
            vp.interpolate(expression)
            self.initial_velocity_model = vp
        elif velocity_model_function != None:
            self.initial_velocity_model = velocity_model_function
        elif new_file != None:
            self.initial_velocity_model_file = new_file
        elif constant != None:
            V = self.function_space
            vp = fire.Function(V)
            vp.interpolate(fire.Constant(constant))
            self.initial_velocity_model = vp
        else:
            raise ValueError("Please specify either a conditional, expression, firedrake function or new file name (segy or hdf5).")
    
    def _get_initial_velocity_model(self):
        if self.velocity_model_type == 'conditional':
            self.set_initial_velocity_model(conditional=self.model_parameters.velocity_conditional)

        if self.initial_velocity_model != None:
            return None
        
        if self.initial_velocity_model_file == None:
            raise ValueError("No velocity model or velocity file to load.")

        if self.initial_velocity_model_file.endswith('.segy'):
            vp_filename, vp_filetype = os.path.splitext(self.initial_velocity_model_file)
            warnings.warn("Converting segy file to hdf5")
            write_velocity_model(self.initial_velocity_model_file, ofname = vp_filename)
            self.initial_velocity_model_file = vp_filename+'.hdf5'

        if self.initial_velocity_model_file.endswith('.hdf5'):
            self.initial_velocity_model = interpolate(self.model_parameters, self.initial_velocity_model_file, self.function_space.sub(0))

    def _build_function_space(self):
        self.function_space = FE_method(self.mesh,self.method,self.degree)

    def get_and_set_maximum_dt(self, fraction = 1.0):
        if self.method == 'KMV' or (self.method == 'CG' and self.mesh.ufl_cell() == fire.quadrilateral):
            estimate_max_eigenvalue = True
        else:
            estimate_max_eigenvalue = False

        dt = utils.estimate_timestep(self.mesh, self.function_space, self.c, estimate_max_eigenvalue=estimate_max_eigenvalue)
        dt *= fraction
        self.dt = dt
        return dt

    def get_mass_matrix_diagonal(self):
        """ Builds a section of the mass matrix for debugging purposes.
        """
        A= self.solver.A
        petsc_matrix = A.petscmat
        diagonal = petsc_matrix.getDiagonal()
        return diagonal.array


