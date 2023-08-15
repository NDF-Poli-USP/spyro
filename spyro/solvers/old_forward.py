import math
import firedrake as fire
from firedrake import Constant, dx, ds, dot, grad, inner, as_tensor
from firedrake.assemble import create_assembly_callable

from ..io.basicio import ensemble_propagator, parallel_print
from . import helpers
from .. import utils
from .CG_acoustic import AcousticWave
from ..pml import damping
from ..domains.quadrature import quadrature_rules


class temp_pml(AcousticWave):
    def forward_solve(self):
        """Solves the forward problem.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        self._get_initial_velocity_model()
        self.c = self.initial_velocity_model
        self.matrix_building()
        self.wave_propagator()

    def matrix_building(self):
        self.current_time = 0.0
        V = self.function_space
        Z = fire.VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
        self.vector_function_space = Z
        quad_rule, k_rule, s_rule = quadrature_rules(V)
        self.quadrature_rule = quad_rule
        self.stiffness_quadrature_rule = k_rule
        self.surface_quadrature_rule = s_rule
        if self.dimension == 2:
            self.matrix_building_2d()
        elif self.dimension == 3:
            raise NotImplementedError("3D not implemented yet")
        else:
            raise ValueError("Only 2D and 3D supported")

    def matrix_building_2d(self):
        dt = self.dt
        c = self.c

        V = self.function_space
        Z = self.vector_function_space
        W = V * Z
        dxlump = dx(scheme=self.quadrature_rule)
        dslump = ds(scheme=self.surface_quadrature_rule)

        u, pp = fire.TrialFunctions(W)
        v, qq = fire.TestFunctions(W)

        X = fire.Function(W)
        X_n = fire.Function(W)
        X_nm1 = fire.Function(W)

        u_n, pp_n = X_n.split()
        u_nm1, _ = X_nm1.split()

        self.u_n = u_n
        self.X = X
        self.X_n = X_n
        self.X_nm1 = X_nm1

        sigma_x, sigma_z = damping.functions(self)
        Gamma_1, Gamma_2 = damping.matrices_2D(sigma_z, sigma_x)
        pml1 = (
            (sigma_x + sigma_z)
            * ((u - u_nm1) / Constant(2.0 * dt))
            * v
            * dxlump
        )

        # typical CG FEM in 2d/3d

        # -------------------------------------------------------
        m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt**2)) * v * dxlump
        a = c * c * dot(grad(u_n), grad(v)) * dxlump  # explicit

        nf = c * ((u_n - u_nm1) / dt) * v * dslump

        FF = m1 + a + nf

        B = fire.Function(W)

        pml2 = sigma_x * sigma_z * u_n * v * dxlump
        pml3 = inner(pp_n, grad(v)) * dxlump
        FF += pml1 + pml2 + pml3
        # -------------------------------------------------------
        mm1 = (dot((pp - pp_n), qq) / Constant(dt)) * dxlump
        mm2 = inner(dot(Gamma_1, pp_n), qq) * dxlump
        dd = c * c * inner(grad(u_n), dot(Gamma_2, qq)) * dxlump
        FF += mm1 + mm2 + dd

        lhs_ = fire.lhs(FF)
        rhs_ = fire.rhs(FF)

        A = fire.assemble(lhs_, mat_type="matfree")
        solver = fire.LinearSolver(A, solver_parameters=self.solver_parameters)
        self.solver = solver
        self.rhs = rhs_
        self.B = B

        return

    @ensemble_propagator
    def wave_propagator(self, dt=None, final_time=None, source_num=0):

        excitations = self.sources
        excitations.current_source = source_num
        receivers = self.receivers
        comm = self.comm
        temp_filename = self.forward_output_file
        filename, file_extension = temp_filename.split(".")
        output_filename = (
            filename + "sn" + str(source_num) + "." + file_extension
        )
        if self.forward_output:
            parallel_print(f"Saving output in: {output_filename}", self.comm)

        if final_time is None:
            final_time = self.final_time
        if dt is None:
            dt = self.dt
        t = self.current_time
        nt = int(final_time / dt) + 1  # number of timesteps

        X = self.X
        u_n = self.u_n
        X_n = self.X_n
        X_nm1 = self.X_nm1

        V = self.function_space
        wavelet = self.wavelet
        solver = self.solver
        rhs_ = self.rhs
        B = self.B

        nspool = self.output_frequency
        fspool = self.gradient_sampling_frequency

        usol = [
            fire.Function(V, name="pressure")
            for t in range(nt)
            if t % self.gradient_sampling_frequency == 0
        ]
        usol_recv = []
        save_step = 0

        assembly_callable = create_assembly_callable(rhs_, tensor=B)

        rhs_forcing = fire.Function(V)

        for step in range(nt):
            rhs_forcing.assign(0.0)
            assembly_callable()
            f = excitations.apply_source(rhs_forcing, wavelet[step])
            B0 = B.sub(0)
            B0 += f
            solver.solve(X, B)

            X_np1 = X

            X_nm1.assign(X_n)
            X_n.assign(X_np1)

            usol_recv.append(
                self.receivers.interpolate(X_np1.dat.data_ro_with_halos[0][:])
            )

            if step % fspool == 0:
                usol[save_step].assign(X_np1.sub(0))
                save_step += 1

            if step % nspool == 0:
                assert (
                    fire.norm(u_n) < 1
                ), "Numerical instability. Try reducing dt or building the mesh differently"
                if t > 0:
                    helpers.display_progress(comm, t)

            t = step * float(dt)

        usol_recv = helpers.fill(usol_recv, receivers.is_local, nt, receivers.number_of_points)
        usol_recv = utils.utils.communicate(usol_recv, comm)

        self.forward_solution = usol
        self.forward_solution_receivers = usol_recv

        return usol, usol_recv
