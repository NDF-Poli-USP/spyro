import firedrake as fire
from firedrake import Constant, dot, dx, grad, ds, inner
from .CG_acoustic import AcousticWave
from ..domains.quadrature import quadrature_rules
from ..pml import damping
from ..io.basicio import ensemble_propagator, parallel_print
from . import helpers
from .. import utils


class AcousticWavePML(AcousticWave):
    def matrix_building_2d(self):
        """Builds solver operators. Doesn't create mass matrices if
        matrix_free option is on,
        which it is by default.
        """
        dt = self.dt
        V = self.function_space
        Z = self.vector_function_space
        dxlump = dx(scheme=self.quadrature_rule)
        dslump = ds(scheme=self.surface_quadrature_rule)
        c = self.c

        W = V * Z
        u, pp = fire.TrialFunctions(W)
        v, qq = fire.TestFunctions(W)

        u_nm1, pp_nm1 = fire.Function(W).split()
        u_n, pp_n = fire.Function(W).split()
        self.u_nm1 = u_nm1
        self.u_n = u_n
        self.pp_nm1 = pp_nm1
        self.pp_n = pp_n

        # -------------------------------------------------------
        # Getting PML parameters
        # -------------------------------------------------------
        sigma_x, sigma_z = damping.functions(self)
        gamma_1, gamma_2 = damping.matrices_2D(sigma_z, sigma_x)

        pml_term1 = (
            (sigma_x + sigma_z)
            * ((u - u_nm1) / Constant(2.0 * dt))
            * v * dxlump
        )

        pml_term2 = sigma_x * sigma_z * u_n * v * dxlump

        pml_term3 = inner(pp_n, grad(v)) * dxlump

        mm1 = (dot((pp - pp_n), qq) / Constant(dt)) * dxlump

        mm2 = inner(dot(gamma_1, pp_n), qq) * dxlump

        dd = inner(grad(u_n), dot(gamma_2, qq)) * dxlump

        pml = pml_term1 + pml_term2 + pml_term3 + mm1 + mm2 + dd

        # -------------------------------------------------------
        # Getting wave equation form
        # -------------------------------------------------------

        m1 = (
            (1 / (c * c))
            * ((u - 2.0 * u_n + u_nm1) / Constant(dt**2))
            * v
            * dxlump
        )
        a = dot(grad(u_n), grad(v)) * dxlump  # explicit

        # -------------------------------------------------------
        # Getting other nf boundary condition
        # -------------------------------------------------------

        nf = (1 / c) * ((u_n - u_nm1) / dt) * v * dslump

        # -------------------------------------------------------
        # Building form
        # -------------------------------------------------------

        form = m1 + a + nf + pml

        B = fire.Function(W)
        # B = fire.Function(V)

        lhs = fire.lhs(form)
        rhs = fire.rhs(form)
        self.lhs = lhs

        A = fire.assemble(lhs, mat_type="matfree")
        self.solver = fire.LinearSolver(
            A, solver_parameters=self.solver_parameters
        )

        self.rhs = rhs
        self.B = B

    def matrix_building_3d(self):
        V = self.function_space
        Z = self.vector_function_space

        W = V * V * Z
        u, psi, pp = fire.TrialFunctions(W)
        v, phi, qq = fire.TestFunctions(W)
        # self.trial_function = u

        u_nm1, psi_nm1, pp_nm1 = fire.Function(W).split()
        u_n, psi_n, pp_n = fire.Function(W).split()
        self.u_nm1 = u_nm1
        self.u_n = u_n
        self.pp_nm1 = pp_nm1
        self.pp_n = pp_n
        self.psi_nm1 = psi_nm1
        self.psi_n = psi_n

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
        output_filename = (filename + "sn" + str(source_num) + "." + file_extension)
        if self.forward_output:
            parallel_print(f"Saving output in: {output_filename}", self.comm)

        output = fire.File(output_filename, comm=comm.comm)
        comm.comm.barrier()

        X = fire.Function(self.function_space)
        if final_time is None:
            final_time = self.final_time
        if dt is None:
            dt = self.dt
        t = self.current_time
        nt = int((final_time - t) / dt) + 1  # number of timesteps

        u_nm1 = self.u_nm1
        u_n = self.u_n
        u_np1 = fire.Function(self.function_space)

        rhs_forcing = fire.Function(self.function_space)
        usol = [
            fire.Function(self.function_space, name="pressure")
            for t in range(nt)
            if t % self.gradient_sampling_frequency == 0
        ]
        usol_recv = []
        save_step = 0
        B = self.B
        rhs = self.rhs

        # assembly_callable = create_assembly_callable(rhs, tensor=B)

        for step in range(nt):
            rhs_forcing.assign(0.0)
            B = fire.assemble(rhs, tensor=B)
            f = excitations.apply_source(rhs_forcing, self.wavelet[step])
            B0 = B.sub(0)
            B0 += f
            self.solver.solve(X, B)

            u_np1.assign(X)

            usol_recv.append(
                self.receivers.interpolate(u_np1.dat.data_ro_with_halos[:])
            )

            if step % self.gradient_sampling_frequency == 0:
                usol[save_step].assign(u_np1)
                save_step += 1

            if (step - 1) % self.output_frequency == 0:
                assert (
                    fire.norm(u_n) < 1
                ), "Numerical instability. Try reducing dt or building the \
                    mesh differently"
                if self.forward_output:
                    output.write(u_n, time=t, name="Pressure")

                helpers.display_progress(self.comm, t)

            u_nm1.assign(u_n)
            u_n.assign(u_np1)

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
