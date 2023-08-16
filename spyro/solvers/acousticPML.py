import firedrake as fire
from firedrake import Constant, dx, ds, dot, grad, inner
from firedrake.assemble import create_assembly_callable

from ..io.basicio import ensemble_propagator, parallel_print
from . import helpers
from .. import utils
from .CG_acoustic import AcousticWave
from ..pml import damping
from ..domains.quadrature import quadrature_rules


class AcousticWavePML(AcousticWave):
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
            self.matrix_building_3d()
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

    def matrix_building_3d(self):
        dt = self.dt
        c = self.c

        V = self.function_space
        Z = self.vector_function_space
        W = V * V * Z
        dxlump = dx(scheme=self.quadrature_rule)
        dslump = ds(scheme=self.surface_quadrature_rule)

        u, psi, pp = fire.TrialFunctions(W)
        v, phi, qq = fire.TestFunctions(W)

        X = fire.Function(W)
        X_n = fire.Function(W)
        X_nm1 = fire.Function(W)

        u_n, psi_n, pp_n = X_n.split()
        u_nm1, psi_nm1, _ = X_nm1.split()

        self.u_n = u_n
        self.X = X
        self.X_n = X_n
        self.X_nm1 = X_nm1

        sigma_x, sigma_y, sigma_z = damping.functions(self)
        Gamma_1, Gamma_2, Gamma_3 = damping.matrices_3D(sigma_x, sigma_y, sigma_z)

        pml1 = (
            (sigma_x + sigma_y + sigma_z)
            * ((u - u_nm1) / Constant(2.0 * dt))
            * v
            * dxlump
        )

        pml2 = (
            (sigma_x * sigma_y + sigma_x * sigma_z + sigma_y * sigma_z)
            * u_n
            * v
            * dxlump
        )

        pml3 = (sigma_x * sigma_y * sigma_z) * psi_n * v * dxlump
        pml4 = inner(pp_n, grad(v)) * dxlump

        # typical CG FEM in 2d/3d

        # -------------------------------------------------------
        m1 = ((u - 2.0 * u_n + u_nm1) / Constant(dt**2)) * v * dxlump
        a = c * c * dot(grad(u_n), grad(v)) * dxlump  # explicit

        nf = c * ((u_n - u_nm1) / dt) * v * dslump

        FF = m1 + a + nf

        B = fire.Function(W)

        FF += pml1 + pml2 + pml3 + pml4
        # -------------------------------------------------------
        mm1 = (dot((pp - pp_n), qq) / Constant(dt)) * dxlump
        mm2 = inner(dot(Gamma_1, pp_n), qq) * dxlump
        dd1 = c * c * inner(grad(u_n), dot(Gamma_2, qq)) * dxlump
        dd2 = -c * c * inner(grad(psi_n), dot(Gamma_3, qq)) * dxlump
        FF += mm1 + mm2 + dd1 + dd2

        mmm1 = (dot((psi - psi_n), phi) / Constant(dt)) * dxlump
        uuu1 = (-u_n * phi) * dxlump
        FF += mmm1 + uuu1

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
        """Propagates the wave forward in time.
        Currently uses central differences.

        Parameters:
        -----------
        dt: Python 'float' (optional)
            Time step to be used explicitly. If not mentioned uses the default,
            that was estabilished in the wave object.
        final_time: Python 'float' (optional)
            Time which simulation ends. If not mentioned uses the default,
            that was estabilished in the wave object.

        Returns:
        --------
        usol: Firedrake 'Function'
            Pressure wavefield at the final time.
        u_rec: numpy array
            Pressure wavefield at the receivers across the timesteps.
        """
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

        output = fire.File(output_filename, comm=comm.comm)
        comm.comm.barrier()

        if final_time is None:
            final_time = self.final_time
        if dt is None:
            dt = self.dt
        t = self.current_time
        nt = int(final_time / dt) + 1  # number of timesteps

        X = self.X
        X_n = self.X_n
        X_nm1 = self.X_nm1

        rhs_forcing = fire.Function(self.function_space)
        usol = [
            fire.Function(self.function_space, name="pressure")
            for t in range(nt)
            if t % self.gradient_sampling_frequency == 0
        ]
        usol_recv = []
        save_step = 0
        B = self.B
        rhs_ = self.rhs

        assembly_callable = create_assembly_callable(rhs_, tensor=B)

        for step in range(nt):
            rhs_forcing.assign(0.0)
            assembly_callable()
            f = excitations.apply_source(rhs_forcing, self.wavelet[step])
            B0 = B.sub(0)
            B0 += f
            self.solver.solve(X, B)

            X_np1 = X

            X_nm1.assign(X_n)
            X_n.assign(X_np1)

            usol_recv.append(
                self.receivers.interpolate(X_np1.dat.data_ro_with_halos[0][:])
            )

            if step % self.gradient_sampling_frequency == 0:
                usol[save_step].assign(X_np1.sub(0))
                save_step += 1

            if (step - 1) % self.output_frequency == 0:
                assert (
                    fire.norm(X_np1.sub(0)) < 1
                ), "Numerical instability. Try reducing dt or building the \
                    mesh differently"
                if self.forward_output:
                    output.write(X_np1.sub(0), time=t, name="Pressure")

                helpers.display_progress(comm, t)

            t = step * float(dt)

        self.current_time = t
        helpers.display_progress(self.comm, t)

        usol_recv = helpers.fill(
            usol_recv, receivers.is_local, nt, receivers.number_of_points
        )
        usol_recv = utils.utils.communicate(usol_recv, comm)
        self.receivers_output = usol_recv

        self.forward_solution = usol
        self.forward_solution_receivers = usol_recv

        return usol, usol_recv
