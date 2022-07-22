from operator import methodcaller
import firedrake as fire
from firedrake.assemble import create_assembly_callable
from firedrake import Constant, dx, dot, inner, grad, ds
import FIAT
import finat
from warnings import warn

class wave():
    def __init__(self, model_parameters = None):
        self.mesh = model_parameters.get_mesh()
        self.method = model_parameters.method
        self.degree = model_parameters.degree
        self.dimension = model_parameters.dimension
        self.final_time = model_parameters.final_time
        self.dt = model_parameters.dt
        self.initial_velocity_model = model_parameters.get_initial_velocity_model()
        self.function_space = None
        self.foward_output_file = 'forward_output.pvd'
        self.current_time = 0.0
        self.solver_parameters = model_parameters.solver_parameters
        self.c = self.initial_velocity_model
        
    def matrix_building(self):
        if self.method == 'SEM':
            element = fire.FiniteElement(self.method, self.mesh.ufl_cell(), degree=self.degree, variant="spectral")
        else:
            element = fire.FiniteElement(self.method, self.mesh.ufl_cell(), degree=self.degree)
        V = fire.FunctionSpace(self.mesh, element)
        self.function_space = V

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

        self.rhs_assembly_callable = create_assembly_callable(rhs, tensor=B)
        self.B = B
    
    def wave_propagator(self, final_time = None):
        X = fire.Function(self.function_space)
        if final_time == None:
            final_time = self.final_time
        t = self.current_time
        nt = int( (final_time-t) / self.dt)  # number of timesteps

        u_nm1 = fire.Function(self.function_space)
        u_n = fire.Function(self.function_space)
        u_np1 = fire.Function(self.function_space)

        rhs_forcing = fire.Function(self.function_space)
        usol = [fire.Function(self.function_space, name="pressure") for t in range(nt) if t % self.fspool == 0]

        for step in range(nt):
            rhs_forcing.assign(0.0)
            self.rhs_assembly_callable()
            f = self.excitations.apply_source(rhs_forcing, self.wavelet[step])
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

            t = step * float(self.dt)

        self.current_time = t
        usol_recv = helpers.fill(usol_recv, receivers.is_local, nt, receivers.num_receivers)
        usol_recv = utils.communicate(usol_recv, comm)

        return usol, usol_recv




