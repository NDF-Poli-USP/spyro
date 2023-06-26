import firedrake as fire
from firedrake import Constant, dx, dot, grad, sin
import numpy as np
from .CG_acoustic import AcousticWave
from ..io.io import ensemble_propagator
from . import helpers
from .. import utils

class AcousticWaveMMS(AcousticWave):

    def matrix_building(self):
        super().matrix_building()
        lhs = self.lhs
        bcs = fire.DirichletBC(self.function_space, 0.0, "on_boundary")
        A = fire.assemble(lhs, bcs=bcs, mat_type="matfree")
        self.analytical = fire.Function(self.function_space)
        self.solver = fire.LinearSolver(A, solver_parameters=self.solver_parameters)
    
    def mms_source(self, t):
        x = -self.mesh_z
        y = self.mesh_x
        return fire.Constant(sin(3*t))*(-9*x*(x - 1)*y*(y - 1) - 2*x*(x - 1) - 2*y*(y - 1))
    
    def analytical_solution(self):
        t = self.current_time
        x = -self.mesh_z
        y = self.mesh_x
        u_an = self.analytical
        u_an.interpolate(x*(x-1)*y*(y-1)*sin(3*t))
        return None

    @ensemble_propagator
    def wave_propagator(self, dt = None, final_time = None, source_num=None):
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
        receivers = self.receivers
        comm = self.comm
        temp_filename = self.forward_output_file
        filename, file_extension = temp_filename.split(".")
        output_filename = filename+"sn_mms_"+"."+file_extension
        print(output_filename, flush = True)

        output = fire.File(output_filename, comm=comm.comm)
        comm.comm.barrier()

        X = fire.Function(self.function_space)
        if final_time == None:
            final_time = self.final_time
        if dt == None:
            dt = self.dt
        t = self.current_time
        nt = int( (final_time-t) / dt) + 1 # number of timesteps

        u_nm1 = self.u_nm1
        u_n = self.u_n
        u_np1 = fire.Function(self.function_space)
        u = self.trial_function
        v = fire.TestFunction(self.function_space)

        usol = [fire.Function(self.function_space, name="pressure") for t in range(nt) if t % self.gradient_sampling_frequency == 0]
        usol_recv = []
        save_step = 0
        B = self.B
        rhs = self.rhs
        quad_rule = self.quadrature_rule

        #assembly_callable = create_assembly_callable(rhs, tensor=B)

        for step in range(nt):
            q = self.mms_source(t)
            m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt ** 2)) * v * dx(scheme = quad_rule)
            a = self.c * self.c * dot(grad(u_n), grad(v)) * dx(scheme = quad_rule)
            l = self.c * self.c * q * v * dx(scheme = quad_rule)

            form = m1 + a - l
            rhs = fire.rhs(form)
            
            B = fire.assemble(rhs, tensor=B)

            self.solver.solve(X, B)

            u_np1.assign(X)

            usol_recv.append(self.receivers.interpolate(u_np1.dat.data_ro_with_halos[:]))

            if step % self.gradient_sampling_frequency == 0:
                usol[save_step].assign(u_np1)
                save_step += 1

            if (step-1) % self.output_frequency == 0:
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
        helpers.display_progress(self.comm, t)

        usol_recv = helpers.fill(usol_recv, receivers.is_local, nt, receivers.num_receivers)
        usol_recv = utils.utils.communicate(usol_recv, comm)
        self.receivers_output = usol_recv

        return usol, usol_recv


