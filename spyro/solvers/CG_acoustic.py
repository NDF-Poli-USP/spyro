import firedrake as fire
from firedrake import Constant, dx, dot, grad
import warnings

from .wave import Wave
from ..io.basicio import ensemble_propagator, parallel_print
from . import helpers
from .. import utils
from ..domains.quadrature import quadrature_rules


class AcousticWave(Wave):
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
        """Builds solver operators. Doesn't create mass matrices if
        matrix_free option is on,
        which it is by default.
        """
        V = self.function_space
        quad_rule, k_rule, s_rule = quadrature_rules(V)
        self.quadrature_rule = quad_rule

        # typical CG FEM in 2d/3d
        u = fire.TrialFunction(V)
        self.trial_function = u
        v = fire.TestFunction(V)

        u_nm1 = fire.Function(V, name="pressure t-dt")
        u_n = fire.Function(V, name="pressure")
        self.u_nm1 = u_nm1
        self.u_n = u_n

        self.current_time = 0.0
        dt = self.dt

        # -------------------------------------------------------
        m1 = (
            (1 / (self.c * self.c))
            * ((u - 2.0 * u_n + u_nm1) / Constant(dt**2))
            * v
            * dx(scheme=quad_rule)
        )
        a = dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)  # explicit

        B = fire.Function(V)

        form = m1 + a
        lhs = fire.lhs(form)
        rhs = fire.rhs(form)
        self.lhs = lhs

        A = fire.assemble(lhs, mat_type="matfree")
        self.solver = fire.LinearSolver(
            A, solver_parameters=self.solver_parameters
        )

        # lterar para como o thiago fez
        self.rhs = rhs
        self.B = B

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
                if t > 0:
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

    def gradient_solve(self, guess=None):
        """Solves the adjoint problem to calculate de gradient.

        Parameters:
        -----------
        guess: Firedrake 'Function' (optional)
            Initial guess for the velocity model. If not mentioned uses the
            one currently in the wave object.

        Returns:
        --------
        dJ: Firedrake 'Function'
            Gradient of the cost functional.
        """
        if self.real_shot_record is None:
            warnings.warn("Please load a real shot record first")
        if self.current_time == 0.0 and guess is not None:
            self.c = guess
            warnings.warn(
                "You need to run the forward solver before the adjoint solver,\
                     will do it for you now"
            )
            self.forward_solve()
        self.misfit = self.real_shot_record - self.forward_solution_receivers
        self.wave_backward_propagator()

    def wave_backward_propagator(self):
        residual = self.misfit
        guess = self.forward_solution
        V = self.function_space
        receivers = self.receivers
        dxlump = dx(scheme=self.quadrature_rule)
        c = self.c
        final_time = self.final_time
        t = self.current_time
        dt = self.dt
        comm = self.comm
        adjoint_output = self.adjoint_output
        adjoint_output_file = self.adjoint_output_file
        if self.adjoint_output:
            print(f"Saving output in: {adjoint_output_file}", flush=True)
        output = fire.File(adjoint_output_file, comm=comm.comm)
        nt = int((final_time - t) / dt) + 1  # number of timesteps

        # Define gradient problem
        m_u = fire.Function(V)
        m_v = fire.TestFunction(V)
        mgrad = m_u * m_v * dxlump

        uuadj = fire.Function(V)  # auxiliarly function for the gradient compt.
        uufor = fire.Function(V)  # auxiliarly function for the gradient compt.

        ffG = 2.0 * c * dot(grad(uuadj), grad(uufor)) * m_v * dxlump

        lhsG = mgrad
        rhsG = ffG

        gradi = fire.Function(V)
        grad_prob = fire.LinearVariationalProblem(lhsG, rhsG, gradi)

        grad_solver = fire.LinearVariationalSolver(
            grad_prob,
            solver_parameters=self.solver_parameters,
        )

        u_nm1 = fire.Function(V)
        u_n = fire.Function(V)
        u_np1 = fire.Function(V)

        X = fire.Function(V)
        B = fire.Function(V)

        rhs_forcing = fire.Function(V)  # forcing term
        if adjoint_output:
            adjoint = [
                fire.Function(V, name="adjoint_pressure") for t in range(nt)
            ]
        for step in range(nt - 1, -1, -1):
            t = step * float(dt)
            rhs_forcing.assign(0.0)
            # Solver - main equation - (I)
            B = fire.assemble(rhsG, tensor=B)

            f = receivers.apply_receivers_as_source(
                rhs_forcing, residual, step
            )
            # add forcing term to solve scalar pressure
            B0 = B.sub(0)
            B0 += f

            # AX=B --> solve for X = B/Aˆ-1
            self.solver.solve(X, B)

            u_np1.assign(X)

            # only compute for snaps that were saved
            if step % self.gradient_sampling_frequency == 0:
                # compute the gradient increment
                uuadj.assign(u_np1)
                uufor.assign(guess.pop())

                grad_solver.solve()
                dJ += gradi

            u_nm1.assign(u_n)
            u_n.assign(u_np1)

            if step % self.output_frequency == 0:
                if adjoint_output:
                    output.write(u_n, time=t)
                if adjoint_output:
                    adjoint.append(u_n)
                helpers.display_progress(comm, t)

        self.gradient = dJ

        if adjoint_output:
            return dJ, adjoint
        else:
            return dJ
