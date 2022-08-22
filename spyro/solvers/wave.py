import os
import warnings
import firedrake as fire
from firedrake.assemble import create_assembly_callable
from firedrake import Constant, dx, dot, inner, grad, ds
from warnings import warn
from SeismicMesh import write_velocity_model

from ..io.io import ensemble_propagator

from ..io import Model_parameters, interpolate
from . import helpers
from .. import utils
from ..receivers.Receivers import Receivers
from ..sources.Sources import Sources
from ..domains.space import FE_method
from ..domains.quadrature import quadrature_rules

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
        self.model_parameters = model_parameters
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

        self.velocity_model_type = model_parameters.velocity_model_type

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
    
    def forward_solve(self):
        self._get_initial_velocity_model()
        self.c = self.initial_velocity_model
        self.matrix_building()
        self.wave_propagator()

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
            return interpolate(self.model_parameters, self.initial_velocity_model_file, self.function_space.sub(0))

    def _build_function_space(self):
        self.function_space = FE_method(self.mesh,self.method,self.degree)

    def matrix_building(self):
        """ Builds solver operators. Doesn't create mass matrices if matrix_free option is on,
        which it is by default.
        """
        V = self.function_space
        quad_rule, k_rule, s_rule = quadrature_rules(V)

        # typical CG FEM in 2d/3d
        u = fire.TrialFunction(V)
        v = fire.TestFunction(V)

        u_nm1 = fire.Function(V)
        u_n = fire.Function(V)
        self.u_nm1 = u_nm1
        self.u_n = u_n

        self.current_time = 0.0
        dt = self.dt

        # -------------------------------------------------------
        m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt ** 2)) * v * dx(rule = quad_rule)
        a = self.c * self.c * dot(grad(u_n), grad(v)) * dx(rule = quad_rule)  # explicit

        B = fire.Function(V)

        form = m1 + a
        lhs = fire.lhs(form)
        rhs = fire.rhs(form)

        A = fire.assemble(lhs, mat_type="matfree")
        self.solver = fire.LinearSolver(A, solver_parameters=self.solver_parameters)

        #lterar para como o thiago fez
        self.rhs = rhs
        self.B = B
    
    @ensemble_propagator
    def wave_propagator(self, dt = None, final_time = None, source_num = 0):
        """ Propagates the wave forward in time.
        Currently uses central differences.

        Parameters:
        -----------
        dt: Python 'float' (optional)
            Time step to be used explicitly. If not mentioned uses the default,
            that was estabilished in the model_parameters.
        final_time: Python 'float' (optional)
            Time which simulation ends. If not mentioned uses the default,
            that was estabilished in the model_parameters.
        """
        excitations = self.sources
        excitations.current_source = source_num
        receivers = self.receivers
        comm = self.comm
        temp_filename = self.forward_output_file
        filename, file_extension = temp_filename.split(".")
        output_filename = filename+str(source_num)+"."+file_extension
        print(output_filename, flush = True)

        output = fire.File(output_filename)

        X = fire.Function(self.function_space)
        if final_time == None:
            final_time = self.final_time
        if dt == None:
            dt = self.dt
        t = self.current_time
        nt = int( (final_time-t) / dt)  # number of timesteps

        u_nm1 = self.u_nm1
        u_n = self.u_n
        u_np1 = fire.Function(self.function_space)

        rhs_forcing = fire.Function(self.function_space)
        usol = [fire.Function(self.function_space, name="pressure") for t in range(nt) if t % self.gradient_sampling_frequency == 0]
        usol_recv = []
        save_step = 0
        B = self.B
        rhs = self.rhs

        #assembly_callable = create_assembly_callable(rhs, tensor=B)

        for step in range(nt):
            rhs_forcing.assign(0.0)
            B = fire.assemble(rhs, tensor=B)
            f = excitations.apply_source(rhs_forcing, self.wavelet[step])
            B0 = B.sub(0)
            B0 += f
            self.solver.solve(X, B)

            u_np1.assign(X)

            usol_recv.append(self.receivers.interpolate(u_np1.dat.data_ro_with_halos[:]))

            if step % self.gradient_sampling_frequency == 0:
                usol[save_step].assign(u_np1)
                save_step += 1

            if step % self.output_frequency == 0:
                assert (
                    fire.norm(u_n) < 1
                ), "Numerical instability. Try reducing dt or building the mesh differently"
                if self.forward_output:
                    output.write(u_n, time=t, name="Pressure")
                if t > 0:
                    helpers.display_progress(self.comm, t)

            u_nm1.assign(u_n)
            u_n.assign(u_np1)

            t = step * float(dt)

        self.current_time = t
        usol_recv = helpers.fill(usol_recv, receivers.is_local, nt, receivers.num_receivers)
        usol_recv = utils.utils.communicate(usol_recv, comm)

        return usol, usol_recv

    def get_and_set_maximum_dt(self, fraction = 1.0):
        if self.method == 'KMV' or (self.method == 'CG' and self.mesh.ufl_cell() == fire.quadrilateral):
            estimate_max_eigenvalue = True
        else:
            estimate_max_eigenvalue = False

        dt = utils.estimate_timestep(self.mesh, self.function_space, self.c, estimate_max_eigenvalue=estimate_max_eigenvalue)
        dt *= fraction
        self.dt = dt
        return dt



