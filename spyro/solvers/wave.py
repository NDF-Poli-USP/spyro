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


class Wave():
    def __init__(self, model_parameters = None, comm = None, model_dictionary = None):
        """Wave object solver. Contains both the forward solver 
        and gradient calculator methods.

        Parameters:
        -----------
        comm: MPI communicator

        model_parameters: Python object
            Contains model parameters
        """
        if comm != None:
            self.comm = comm
        if model_parameters == None:
            model_parameters = Model_parameters(dictionary=model_dictionary, comm = comm)
        if model_parameters.comm != None:
            self.comm = model_parameters.comm
        if self.comm != None:
            self.comm.comm.barrier()
        self.model_parameters = model_parameters
        self.initial_velocity_model = None
        self._unpack_parameters(model_parameters)
        self.mesh = model_parameters.get_mesh()
        self.function_space = None
        self.current_time = 0.0
        self.set_solver_parameters()
        
        self._build_function_space()
        self.sources = Sources(self)
        self.receivers = Receivers(self)
        self.wavelet = model_parameters.get_wavelet()

    def _unpack_parameters(self, model_parameters):
        self.comm = model_parameters.comm
        self.method = model_parameters.method
        self.cell_type = model_parameters.cell_type
        self.degree = model_parameters.degree
        self.dimension = model_parameters.dimension
        self.length_z = model_parameters.length_z
        self.length_x = model_parameters.length_x
        self.frequency = model_parameters.frequency

        self.abc_status = model_parameters.abc_status
        self.outer_bc = model_parameters.abc_outer_bc
        self.abc_damping_type = model_parameters.abc_damping_type
        self.abc_exponent = model_parameters.abc_exponent
        self.abc_cmax = model_parameters.abc_cmax
        self.abc_R = model_parameters.abc_R
        self.abc_lz = model_parameters.abc_lz
        self.abc_lx = model_parameters.abc_lx
        self.abc_ly = model_parameters.abc_ly

        self.velocity_model_type = model_parameters.velocity_model_type
        self.initial_velocity_model_file = model_parameters.initial_velocity_model_file

        self.final_time = model_parameters.final_time
        self.dt = model_parameters.dt

        self.output_frequency = model_parameters.output_frequency
        self.gradient_sampling_frequency = model_parameters.gradient_sampling_frequency
        
        self.automatic_adjoint = model_parameters.automatic_adjoint
        
        self.forward_output = model_parameters.forward_output
        self.fwi_velocity_model_output = model_parameters.fwi_velocity_model_output
        self.gradient_output = model_parameters.gradient_output

        self.forward_output_file = model_parameters.forward_output_file
        self.fwi_velocity_model_output_file = model_parameters.fwi_velocity_model_output_file
        self.gradient_output_file = model_parameters.gradient_output_file

        self.number_of_sources = model_parameters.number_of_sources

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
    
    def set_initial_velocity_model(self, conditional= None, velocity_model_function = None, expression = None, new_file = None):
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


