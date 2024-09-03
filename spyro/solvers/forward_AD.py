from firedrake import *
from firedrake.adjoint import *
from ..domains import quadrature, space
from firedrake.__future__ import interpolate
import finat
# Note this turns off non-fatal warnings
set_log_level(ERROR)


class ForwardSolver:
    """Forward solver for the acoustic wave equation.

    This forward solver is prepared to work with the automatic
    differentiation.

    Parameters
    ----------
    model : dict
        Dictionary containing the model parameters.
    mesh : Mesh
        Firedrake mesh object.
    """

    def __init__(self, model, mesh):
        self.model = model
        self.mesh = mesh
        self.element = space.FE_method(
            self.mesh, self.model["opts"]["method"],
            self.model["opts"]["degree"]
        )
        self.V = FunctionSpace(self.mesh, self.element)
        self.receiver_mesh = VertexOnlyMesh(
            self.mesh, self.model["acquisition"]["receiver_locations"])

    def execute_acoustic(
            self, c, source_number, wavelet, compute_functional=False,
            true_data_receivers=None, annotate=False
            ):
        """Time-stepping acoustic forward solver.

        Parameters
        ----------
        c : firedrake.Function
            Velocity field.
        source_number : int
            Number of the source. This is used to select the source
            location.
        wavelet : list
            Time-dependent wavelet.
        compute_functional : bool, optional
            Whether to compute the functional. If True, the true receiver
            data must be provided.
        true_data_receivers : list, optional
            True receiver data. This is used to compute the functional.
        annotate : bool, optional
            Whether to annotate the forward solver. Annotated solvers are
            used for automated adjoint computations from automatic
            differentiation.

        Returns
        -------
        (receiver_data : list, J_val : float)
            Receiver data and functional value.

        Raises
        ------
        ValueError
            If true_data_receivers is not provided when computing the
            functional.
        """
        if annotate:
            continue_annotation()
        # RHS
        source_function = Cofunction(self.V.dual())
        solver, u_np1, u_n, u_nm1 = self._acoustic_lvs(
            c, source_function)
        # Sources.
        source_mesh = VertexOnlyMesh(
            self.mesh,
            [self.model["acquisition"]["source_locations"][source_number]]
            )
        # Source function space.
        V_s = FunctionSpace(source_mesh, "DG", 0)
        d_s = Function(V_s)
        d_s.assign(1.0)
        source_cofunction = assemble(d_s * TestFunction(V_s) * dx)
        # Interpolate from the source function space to the velocity function space.
        q_s = Cofunction(self.V.dual()).interpolate(source_cofunction)

        # Receivers
        V_r = FunctionSpace(self.receiver_mesh, "DG", 0)
        # Interpolate object.
        interpolate_receivers = interpolate(u_np1, V_r)

        # Time execution.
        J_val = 0.0
        receiver_data = []
        for step in range(self.model["timeaxis"]["nt"]):
            source_cofunction.assign(wavelet(step) * q_s)
            solver.solve()
            u_nm1.assign(u_n)
            u_n.assign(u_np1)
            receiver_data.append(assemble(interpolate_receivers))
            if compute_functional:
                if not true_data_receivers:
                    raise ValueError("True receiver data is required for"
                                     "computing the functional.")
                misfit = receiver_data - true_data_receivers[step]
                J_val += assemble(0.5 * inner(misfit, misfit) * dx)

        return receiver_data, J_val

    def _acoustic_lvs(self, c, source_function):
        # Acoustic linear variational solver.
        V = self.V
        dt = self.model["timeaxis"]["dt"]
        u = TrialFunction(V)
        v = TestFunction(V)
        u_np1 = Function(V)  # timestep n+1
        u_n = Function(V)  # timestep n
        u_nm1 = Function(V)  # timestep n-1

        qr_x, qr_s, _ = quadrature.quadrature_rules(V)
        time_term = (1 / (c * c)) * (u - 2.0 * u_n + u_nm1) / \
              Constant(dt**2) * v * dx(scheme=quad_rule)

        nf = 0
        if self.model["BCs"]["outer_bc"] == "non-reflective":
            nf = (1/c) * ((u_n - u_nm1) / dt) * v * ds(scheme=qr_s)

        a = dot(grad(u_n), grad(v)) * dx(scheme=qr_x)
        F = time_term + a + nf
        lin_var = LinearVariationalProblem(
            lhs(F), rhs(F) + source_function, u_np1)
        solver = LinearVariationalSolver(
            lin_var,solver_parameters=self._solver_parameters())
        return solver, u_np1, u_n, u_nm1

    def _solver_parameters(self):
        if self.model["opts"]["method"] == "KMV":
            params = {"ksp_type": "preonly", "pc_type": "jacobi"}
        elif (
            self.model["opts"]["method"] == "CG"
            and self.mesh.ufl_cell() != quadrilateral
            and self.mesh.ufl_cell() != hexahedron
        ):
            params = {"ksp_type": "cg", "pc_type": "jacobi"}
        elif self.model["opts"]["method"] == "CG" and (
            self.mesh.ufl_cell() == quadrilateral 
            or self.mesh.ufl_cell() == hexahedron
        ):
            params = {"ksp_type": "preonly", "pc_type": "jacobi"}
        else:
            raise ValueError("method is not yet supported")

        return params
