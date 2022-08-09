from logging import warning
from operator import methodcaller
import os
import firedrake as fire
from firedrake.assemble import create_assembly_callable
from firedrake import Constant, dx, dot, inner, grad, ds
import FIAT
import finat
from warnings import warn
import spyro
from spyro.io.model_parameters import Model_parameters
from . import helpers
from .. import utils
from spyro.utils import estimate_timestep
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
        self.mesh = model_parameters.get_mesh()
        self.method = model_parameters.method
        self.degree = model_parameters.degree
        self.dimension = model_parameters.dimension
        self.final_time = model_parameters.final_time
        self.dt = model_parameters.dt
        self.function_space = None
        self.foward_output_file = 'forward_output.pvd'
        self.current_time = 0.0

        self.comm = model_parameters.comm

        self._build_function_space()
        self.sources = spyro.Sources(self)
        self.receivers = spyro.Receivers(self)
        self.wavelet = model_parameters.get_wavelet()

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
        self.matrix_building()
        self.wave_propagator()

    def _get_initial_velocity_model(self):
        if self.initial_velocity_model != None:
            return None
        
        if self.initial_velocity_model_file == None:
            raise ValueError("No velocity model or velocity file to load.")

        if self.initial_velocity_model_file.endswith('.segy'):
            vp_filename, vp_filetype = os.path.splitext(self.initial_velocity_model_file)
            warning.warn("Converting segy file to hdf5")
            spyro.io.write_velocity_model(self.initial_velocity_model_file, ofname = vp_filename)
            self.initial_velocity_model_file = vp_filename+'.hdf5'

        if self.initial_velocity_model_file.endswith('.hdf5'):
            return spyro.io.interpolate(self.model_parameters, self.initial_velocity_model_file, self.function_space.sub(0))

    def _build_function_space(self):
        self.function_space = spyro.domains.space.FE_method(self.mesh,self.method,self.degree)

    def matrix_building(self):
        """ Builds solver operators. Doesn't create mass matrices if matrix_free option is on,
        which it is by default.
        """
        V = self.function_space

        # typical CG FEM in 2d/3d
        u = fire.TrialFunction(V)
        v = fire.TestFunction(V)

        u_nm1 = fire.Function(V)
        u_n = fire.Function(V)

        output = fire.File(self.foward_output_file)

        self.current_time = 0.0
        dt = self.dt

        # -------------------------------------------------------
        m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt ** 2)) * v * dx
        a = self.c * self.c * dot(grad(u_n), grad(v)) * dx  # explicit

        B = fire.Function(V)

        lhs = m1
        rhs = -a

        A = fire.assemble(lhs, mat_type="matfree")
        self.solver = fire.LinearSolver(A, solver_parameters=self.solver_parameters)

        #lterar para como o thiago fez
        self.rhs_assembly_callable = create_assembly_callable(rhs, tensor=B)
        self.B = B
    
    def wave_propagator(self, dt = None, final_time = None):
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
        receivers = self.receivers
        comm = self.comm

        X = fire.Function(self.function_space)
        if final_time == None:
            final_time = self.final_time
        if dt == None:
            dt = self.dt
        t = self.current_time
        nt = int( (final_time-t) / dt)  # number of timesteps

        u_nm1 = fire.Function(self.function_space)
        u_n = fire.Function(self.function_space)
        u_np1 = fire.Function(self.function_space)

        rhs_forcing = fire.Function(self.function_space)
        usol = [fire.Function(self.function_space, name="pressure") for t in range(nt) if t % self.fspool == 0]

        for step in range(nt):
            rhs_forcing.assign(0.0)
            self.rhs_assembly_callable()
            f = excitations.apply_source(rhs_forcing, self.wavelet[step])
            B0 = self.B.sub(0)
            B0 += f
            self.solver.solve(X, self.B)

            u_np1.assign(X)

            usol_recv.append(self.receivers.interpolate(u_np1.dat.data_ro_with_halos[:]))

            if step % self.fspool == 0:
                usol[save_step].assign(u_np1)
                save_step += 1

            if step % self.nspool == 0:
                assert (
                    fire.norm(u_n) < 1
                ), "Numerical instability. Try reducing dt or building the mesh differently"
                if self.output:
                    self.outfile.write(u_n, time=t, name="Pressure")
                if t > 0:
                    helpers.display_progress(self.comm, t)

            u_nm1.assign(u_n)
            u_n.assign(u_np1)

            t = step * float(dt)

        self.current_time = t
        usol_recv = helpers.fill(usol_recv, receivers.is_local, nt, receivers.num_receivers)
        usol_recv = utils.communicate(usol_recv, comm)

        return usol, usol_recv

    def get_and_set_maximum_dt(self, fraction = 1.0):
        if self.method == 'KMV' or (self.method == 'CG' and self.mesh.ufl_cell() == fire.quadrilateral):
            estimate_max_eigenvalue = True
        else:
            estimate_max_eigenvalue = False

        dt = estimate_timestep(self.mesh, self.function_space, self.c, estimate_max_eigenvalue=estimate_max_eigenvalue)
        dt *= fraction
        self.dt = dt
        return dt



