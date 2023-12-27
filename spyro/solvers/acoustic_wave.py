import firedrake as fire
import warnings

from .wave import Wave
from .time_integration import time_integrator
from ..io.basicio import ensemble_propagator
from ..domains.quadrature import quadrature_rules
from .acoustic_solver_construction_no_pml import (
    construct_solver_or_matrix_no_pml,
)
from .acoustic_solver_construction_with_pml import (
    construct_solver_or_matrix_with_pml,
)
from . import helpers
from .. import utils


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
        if self.function_space is None:
            self.force_rebuild_function_space()

        self._get_initial_velocity_model()
        self.c = self.initial_velocity_model
        self.matrix_building()
        self.wave_propagator()

    def force_rebuild_function_space(self):
        if self.mesh is None:
            self.mesh = self.get_mesh()
        self._build_function_space()
        self._map_sources_and_receivers()

    def matrix_building(self):
        """Builds solver operators. Doesn't create mass matrices if
        matrix_free option is on,
        which it is by default.
        """
        self.current_time = 0.0
        quad_rule, k_rule, s_rule = quadrature_rules(self.function_space)
        self.quadrature_rule = quad_rule
        self.stiffness_quadrature_rule = k_rule
        self.surface_quadrature_rule = s_rule

        abc_type = self.abc_boundary_layer_type

        # Just to document variables that will be overwritten
        self.trial_function = None
        self.u_nm1 = None
        self.u_n = None
        self.lhs = None
        self.solver = None
        self.rhs = None
        self.B = None
        if abc_type is None:
            construct_solver_or_matrix_no_pml(self)
        elif abc_type == "PML":
            V = self.function_space
            Z = fire.VectorFunctionSpace(V.ufl_domain(), V.ufl_element())
            self.vector_function_space = Z
            self.X = None
            self.X_n = None
            self.X_nm1 = None
            construct_solver_or_matrix_with_pml(self)

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
        if final_time is not None:
            self.final_time = final_time
        if dt is not None:
            self.dt = dt

        usol, usol_recv = time_integrator(self, source_id=source_num)

        return usol, usol_recv

    def gradient_solve(self, guess=None, misfit=None, forward_solution=None):
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
        if misfit is not None:
            self.misfit = misfit
        if self.real_shot_record is None:
            warnings.warn("Please load or calculate a real shot record first")
        if self.current_time == 0.0 and guess is not None:
            self.c = guess
            warnings.warn(
                "You need to run the forward solver before the adjoint solver,\
                     will do it for you now"
            )
            self.forward_solve()
            self.misfit = self.real_shot_record - self.forward_solution_receivers
        return self.backward_wave_propagator()

    def backward_wave_propagator(self, dt=None):
        """Propagates the adjoint wave backwards in time.
        Currently uses central differences.

        Parameters:
        -----------
        dt: Python 'float' (optional)
            Time step to be used explicitly. If not mentioned uses the default,
            that was estabilished in the wave object for the adjoint model.
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
        if dt is not None:
            self.dt = dt

        forward_solution = self.forward_solution
        receivers = self.receivers
        residual = self.misfit
        comm = self.comm
        temp_filename = self.forward_output_file

        filename, file_extension = temp_filename.split(".")
        output_filename = "backward." + file_extension

        output = fire.File(output_filename, comm=comm.comm)
        comm.comm.barrier()

        X = fire.Function(self.function_space)
        dJ = fire.Function(self.function_space, name="gradient")

        final_time = self.final_time
        dt = self.dt
        t = self.current_time
        nt = int((final_time - 0) / dt) + 1  # number of timesteps

        u_nm1 = self.u_nm1
        u_nm1.assign(0.0)
        u_n = self.u_n
        u_n.assign(0.0)
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

        # Define a gradient problem
        m_u = fire.TrialFunction(self.function_space)
        m_v = fire.TestFunction(self.function_space)
        mgrad = m_u * m_v * fire.dx(scheme=self.quadrature_rule)
        uuadj = fire.Function(self.function_space)  # auxiliarly function for the gradient compt.
        uufor = fire.Function(self.function_space)  # auxiliarly function for the gradient compt.

        ffG = 2.0 * self.c * fire.dot(fire.grad(uuadj), fire.grad(uufor)) * m_v * fire.dx(scheme=self.quadrature_rule)

        G = mgrad - ffG
        lhsG, rhsG = fire.lhs(G), fire.rhs(G)

        gradi = fire.Function(self.function_space)
        grad_prob = fire.LinearVariationalProblem(lhsG, rhsG, gradi)
        grad_solver = fire.LinearVariationalSolver(
            grad_prob,
            solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "jacobi",
                "mat_type": "matfree",
            },
        )

        # assembly_callable = create_assembly_callable(rhs, tensor=B)

        for step in range(nt-1, -1, -1):
            rhs_forcing.assign(0.0)
            B = fire.assemble(rhs, tensor=B)
            f = receivers.apply_receivers_as_source(rhs_forcing, residual, step)
            B0 = B.sub(0)
            B0 += f
            self.solver.solve(X, B)

            u_np1.assign(X)

            usol_recv.append(
                self.receivers.interpolate(u_np1.dat.data_ro_with_halos[:])
            )

            if step % self.gradient_sampling_frequency == 0:
                uuadj.assign(u_np1)
                uufor.assign(forward_solution.pop())

                grad_solver.solve()
                dJ += gradi

            if (step) % self.output_frequency == 0:
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

        return dJ
