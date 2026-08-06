from firedrake import LinearEigenproblem, LinearEigensolver
from numpy import abs, amax, array, asarray, eye, imag, real, sqrt, unique
from scipy.sparse.linalg import eigs, eigsh, lobpcg, spilu
from ...io.basicio import parallel_print as pprint
from .modal_forms_and_matrices import assemble_sparse_matrices, weak_forms
from .modal_rq_matrices import generate_eigenfunctions, matrices_rayleigh_quotient
from ...utils.error_management import (validate_data_structure, validate_firedrake_parameter,
                                       validate_numeric, validate_parameter)

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# TODO: citation
# With additions by Alexandre Olender


class Modal_Solver():
    """Class for the Modal problem with Neumann or Dirichlet boundary conditions.

    Attributes
    ----------
    AnaModSol : `modal_ana_sol.Modal_Analytical_Solver`
        An instance of the :class:`~spyro.solvers.modal.modal_ana_sol.Modal_Analytical_Solver`.
        Only initialized if `method` is "ANALYTICAL" and `calc_max_dt` is `False`.
    calc_max_dt : `bool`
        Option to estimate the maximum stable timestep for the computation of the
        transient response. Default is `False`.
    comm : `object`, optional
        An object representing the communication interface for parallel processing.
        Default is `None`.
    dimension : `int`
        Model dimension (2D or 3D). Default is 2D.
    method : `str`
        Method to use for solving the eigenvalue problem. See `valid_methods` for options.
        Default is `None`, which uses the "KRYLOVSCH_CH" method.
    valid_methods: `list`
        List of valid methods for solving the eigenproblem
        Options: "ANALYTICAL", "ARNOLDI", "LANCZOS", "LOBPCG", "KRYLOVSCH_CH",
        "KRYLOVSCH_CG", "KRYLOVSCH_GH", "KRYLOVSCH_GG" or "RAYLEIGH".
        "ANALYTICAL" method is an approximation by using homogenization techniques.
        "RAYLEIGH" method is an approximation by Rayleigh quotient.
        In "KRYLOVSCH_(K)(P)" methods, (K) indicates the Krylov solver to
        use: "C" for Conjugate Gradient (cg) or "G" for Generalized Minimal
        Residual (gmres). (P) indicates the preconditioner to use: "H" for
        Hypre (hypre) or "G" for Geometric Algebraic Multigrid (gamg). For
        example, "KRYLOVSCH_CH" uses cg solver with hypre preconditioner.

    Methods
    -------
    assemble_weak_forms()
        Build the weak forms for the modal problem solved using UFL or sparse matrices.
    estimate_timestep()
        Estimate the maximum stable timestep based on the spectral radius.
    solve_eigenproblem()
        Solve the eigenvalue problem with Neumann boundary conditions.
    solver_rayleigh_quotient()
        Solve the eigenvalue problem using the Rayleigh Quotient method for Neumann Bcs.
    solver_with_sparse_matrix()
        Solve the eigenvalue problem with sparse matrices using Scipy.
    solver_with_ufl()
            Solve the eigenvalue problem using UFL forms with SLEPc.
    """

    def __init__(self, dimension=2, method=None, calc_max_dt=False, comm=None):
        """Initialize the Modal_Solver class.

        Parameters
        ----------
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D.
        method : `str`, optional
            Method to use for solving the eigenvalue problem.
            Default is None, which uses the "KRYLOVSCH_CH" method.
            Opts: "ANALYTICAL", "ARNOLDI", "LANCZOS", "LOBPCG", "KRYLOVSCH_CH",
            "KRYLOVSCH_CG", "KRYLOVSCH_GH", "KRYLOVSCH_GG" or "RAYLEIGH".
            "ANALYTICAL" method is an approximation by using homogenization techniques.
            "RAYLEIGH" method is an approximation by Rayleigh quotient.
            In "KRYLOVSCH_(K)(P)" methods, (K) indicates the Krylov solver to
            use: "C" for Conjugate Gradient (cg) or "G" for Generalized Minimal
            Residual (gmres). (P) indicates the preconditioner to use: "H" for
            Hypre (hypre) or "G" for Geometric Algebraic Multigrid (gamg). For
            example, "KRYLOVSCH_CH" uses cg solver with hypre preconditioner.
        calc_max_dt : `bool`
            Option to estimate the maximum stable timestep for the computation
            of the transient response. Default is `False`.
        comm : `object`, optional
            An object representing the communication interface for parallel processing.
            Default is `None`.

        Returns
        -------
        None
        """

        # Dimension of the problem
        self.dimension = validate_parameter("dimension", dimension, [2, 3])

        # Option to estimate the maximum stable timestep
        self.calc_max_dt = calc_max_dt

        # Communicator MPI
        self.comm = comm

        # Valid methods for solving the eigenproblem
        def_methods = ["ANALYTICAL", "ARNOLDI", "LANCZOS", "LOBPCG"]
        self.valid_methods = def_methods + (["KRYLOVSCH_CH", "KRYLOVSCH_CG",
                                             "KRYLOVSCH_GH", "KRYLOVSCH_GG",
                                             "RAYLEIGH"] if not self.calc_max_dt else [])

        # Method for solving the eigenproblem
        method = "KRYLOVSCH_CH" if method is None else method
        self.method = validate_parameter("method", method, self.valid_methods)

        # Initializing the analytical solver
        if not self.calc_max_dt and self.method == "ANALYTICAL":
            from .modal_ana_sol import Modal_Analytical_Solver
            self.AnaModSol = Modal_Analytical_Solver(dimension=self.dimension, comm=comm)

        pprint(f"Solver Method: {self.method}", comm=self.comm)

    def solver_with_sparse_matrix(self, Asp, Msp, method, k=2, inv_oper=False):
        """Solve the eigenvalue problem with sparse matrices using Scipy.

        Parameters
        ----------
        Asp : `csr matrix`
            Sparse matrix representing the stiffness matrix.
        Msp : `csr matrix`
            Sparse matrix representing the mass matrix.
        method : `str`
            Method to use for solving the eigenvalue problem.
            Opts: "ARNOLDI", "LANCZOS" or "LOBPCG".
        k : `int`, optional
            Number of eigenvalues to compute. Default is 2.
        inv_oper : `bool`, optional
            Option to use an inverse operator for improving convergence.
            Default is `False`.

        Returns
        -------
        Lsp : `array`
            Array containing the computed eigenvalues.
        """

        # Check methods
        validate_parameter("method", method, ["ARNOLDI", "LANCZOS", "LOBPCG"])

        if method == "ARNOLDI" or method == "LANCZOS":
            # Inverse operator for improving convergence
            M_ilu = spilu(Msp) if inv_oper else None
            Minv = M_ilu.solve if inv_oper else None
            A_ilu = spilu(Asp) if inv_oper else None
            OPinv = A_ilu.solve if inv_oper else None

        if method == "ARNOLDI":
            # Solve the eigenproblem using ARNOLDI (ARPACK)
            if self.calc_max_dt:
                Lsp = eigs(Asp, k=k, M=Msp, which="LM", Minv=Minv,
                           OPinv=OPinv, return_eigenvectors=False)
            else:
                Lsp = eigs(Asp, k=k, M=Msp, sigma=0.0, Minv=Minv,
                           OPinv=OPinv, return_eigenvectors=False)

        if method == "LANCZOS":
            # Solve the eigenproblem using LANCZOS (ARPACK)
            if self.calc_max_dt:
                Lsp = eigsh(Asp, k=k, M=Msp, which="LM", Minv=Minv,
                            OPinv=OPinv, return_eigenvectors=False)
            else:
                Lsp = eigsh(Asp, k=k, M=Msp, sigma=0.0, Minv=Minv,
                            OPinv=OPinv, return_eigenvectors=False)

        if method == "LOBPCG":
            # Initialize LI vectors for LOBPCG
            X = eye(Msp.shape[0], k)

            # Solve the eigenproblem using LOBPCG (iterative method)
            it_mod = 2500
            it_ext = 2
            mag = True if self.calc_max_dt else False
            for it in range(it_ext):
                Lsp, X, resid = lobpcg(Asp, X, B=Msp, tol=5e-4, maxiter=it_mod,
                                       largest=mag, retResidualNormsHistory=True)

                it_mod //= 2  # Reduce iterations for next loop
                rmin = array(resid)[:, 1].min()
                if rmin < 5e-4 or it_mod < 20:
                    del X, resid
                    break

        return Lsp

    def solver_with_ufl(self, a, m, k=2):
        """Solve the eigenvalue problem using UFL forms with SLEPc.

        Parameters
        ----------
        a : `Firedrake.Form`
            Weak form representing the stiffness matrix.
        m : `Firedrake.Form`
            Weak form  representing the mass matrix.
        k : `int`, optional
            Number of eigenvalues to compute. Default is 2.

        Returns
        -------
        Lsp : `array`
            Array containing the computed eigenvalues.
        """

        krylovsch_config = {"KRYLOVSCH_CH": {"ksp": "cg", "pc": "hypre"},
                            "KRYLOVSCH_CG": {"ksp": "cg", "pc": "gamg"},
                            "KRYLOVSCH_GH": {"ksp": "gmres", "pc": "hypre"},
                            "KRYLOVSCH_GG": {"ksp": "gmres", "pc": "gamg"}}

        if self.method in krylovsch_config:
            ksp_type = krylovsch_config[self.method]["ksp"]
            pc_type = krylovsch_config[self.method]["pc"]

        opts = {
            "eps_gen_hermitian": None,       # Problem is Hermitian
            "eps_type": "krylovschur",       # Robust, widely used eigensolver
            "eps_tol": 1e-6,                 # Tight tolerance for accuracy
            "eps_max_it": 200,               # Reasonable iteration cap
            "st_shift": 1e-6,                # Stabilizes Neumann BC null space
            "st_type": "sinvert",            # Useful for interior eigenvalues
            "eps_monitor": "ascii",          # Print convergence info
            "ksp_type": ksp_type,            # Options for large problems
            "pc_type": pc_type               # Options for large problems
        }

        if self.calc_max_dt:
            # Largest eigenvalues magnitude
            opts.update({"eps_largest_magnitude": None})
            # subspace, arnoldi, krylovschur, lapack
        else:
            # Smallest eigenvalues magnitude
            opts.update({"eps_smallest_magnitude": None})

        eigenproblem = LinearEigenproblem(a, M=m)
        eigensolver = LinearEigensolver(eigenproblem, n_evals=k, solver_parameters=opts)
        eigensolver.solve()
        Lsp = asarray([eigensolver.eigenvalue(mod) for mod in range(k)])

        return Lsp

    def solver_rayleigh_quotient(self, c, ufl_coordinates, V,
                                 mesh_limits, k=2, quad_rule=None):
        """Solve the eigenvalue problem using the Rayleigh Quotient method for Neumann Bcs.

        Parameters
        ----------
        c : `Firedrake.Function` or `float`
            Velocity model.
        ufl_coordinates : `ufl.geometry.SpatialCoordinate`
            Domain coordinates.
        V : `Firedrake.FunctionSpace`
            Function space for the modal problem.
        mesh_limits : `tuple`, optional
            Tuple containing the minimum and maximum coordinates of the mesh.
            Structure: (min_coordinates, max_coordinates):
            - min_coordinates : `array`
                Array containing the minimum coordinates in each dimension (z, x, y).
            - max_coordinates : `array`
                Array containing the maximum coordinates in each dimension (z, x, y).
        k : `int`, optional
            Number of eigenvalues to compute. Default is 2.
        quad_rule : `dict`, optional
            Quadrature rule to use for the integration.
            Default is `None`, which uses the default quadrature rule.

        Returns
        -------
        Lsp : `array`
            Array containing the computed eigenvalues.
        """

        # Check input arguments
        validate_firedrake_parameter("c", c, "Function")
        validate_firedrake_parameter("ufl_coordinates", ufl_coordinates, "SpatialCoordinate")
        validate_firedrake_parameter("V", V, "FunctionSpace")
        validate_data_structure("mesh_limits", mesh_limits, "tuple")
        validate_data_structure("quad_rule", quad_rule, "dict", accept_parameter_as_none=True)

        # Create eigenfunctions
        eig_funcs, grad_eig = generate_eigenfunctions(ufl_coordinates, V, mesh_limits,
                                                      k=k, dimension=self.dimension)

        # Assemble matrices for generalized eigenvalue problem
        Asp, Msp = matrices_rayleigh_quotient(c, eig_funcs, grad_eig, quad_rule=quad_rule)

        # Solve the generalized eigenvalue problem
        Lsp = self.solver_with_sparse_matrix(Asp, Msp, "ARNOLDI", k=k)

        return Lsp

    def assemble_weak_forms(self, c, V, quad_rule=None, shift=0.):
        """Build the weak forms for the modal problem solved using UFL or sparse matrices.

        Parameters
        ----------
        c : `Firedrake.Function`
            Velocity model.
        V : `Firedrake.FunctionSpace`
            Function space for the modal problem.
        quad_rule : `dict`, optional
            Quadrature rule to use for the integration.
            Default is `None`, which uses the default quadrature rule.
        shift: `float`, optional
            Value to stabilize the Neumann BC null space. Default is 0.

        Returns
        -------
        a : `Firedrake.Form`
            Weak form representing the stiffness matrix.
        m : `Firedrake.Form`
            Weak form  representing the mass matrix.
        """

        # Check input arguments
        validate_firedrake_parameter("c", c, "Function")
        validate_firedrake_parameter("V", V, "FunctionSpace")
        validate_data_structure("quad_rule", quad_rule, "dict", accept_parameter_as_none=True)
        validate_numeric("shift", shift, float_num=True, integer_num=True,
                              lower_bound=0., include_lower_bound=True)

        # Get bilinear forms
        a, m = weak_forms(c, V, quad_rule=quad_rule)

        # Add shift to stabilize Neumann BC null space
        if shift > 0:
            a += shift * m

        return a, m

    def solve_eigenproblem(self, c, V=None, k=2, shift=0., quad_rule=None, inv_oper=False,
                           ufl_coordinates=None, mesh_limits=None, hyp_par=None,
                           cut_plane_percent=1., c_ref=None, V_ref=None, dof_load=None,
                           amplitude_load=None, fitting_c=(0., 0., 0., 0.)):
        """Solve the eigenvalue problem with Neumann boundary conditions.

        Parameters
        ----------
        c : `Firedrake.Function` or `float`
            Velocity model.
        V : `Firedrake.FunctionSpace`, optional
            Function space for the modal problem. Default is `None`.
        k : `int`, optional
            Number of eigenvalues to compute. Default is 2.
        shift: `float`, optional
            Value to stabilize the Neumann BC null space. Default is 0.
        quad_rule : `dict`, optional
            Quadrature rule to use for the integration.
            Default is `None`, which uses the default quadrature rule.
        inv_oper : `bool`, optional
            Option to use an inverse operator for improving convergence.
            Default is `False`.
        ufl_coordinates : `ufl.geometry.SpatialCoordinate`
            Domain coordinates.
        mesh_limits : `tuple`, optional
            Tuple containing the minimum and maximum coordinates of the mesh.
            Structure: (min_coordinates, max_coordinates):
            - min_coordinates : `array`
                Array containing the minimum coordinates in each dimension (z, x, y).
            - max_coordinates : `array`
                Array containing the maximum coordinates in each dimension (z, x, y).
        hyp_par : `tuple`, optional
            Hyperellipshape parameters. Default is `None`.
            Structure 2D: (n_hyp, a_hyp, b_hyp)
            Structure 3D: (n_hyp, a_hyp, b_hyp, c_hyp)
            - n_hyp : `float`
                Degree of the hypershape.
           - a_hyp : `float`
                Hypershape semi-axis in direction x.
            - b_hyp : `float`
                Hypershape semi-axis in direction z.
            - c_hyp : `float`
                Hypershape semi-axis in direction y (3D only).
        cut_plane_percent : `float`, optional
            Percentage of the cut plane (0 to 1). Default is 1 (no cut)
        c_ref : `Firedrake.Function` or `float`, optional
            Velocity model for the reference model without absorbing layer.
            Default is `None`.
        V_ref : `Firedrake.FunctionSpace`, optional
            Function space for the reference model without absorbing layer.
            Default is `None`.
        dof_load : `array`, optional
           Degrees of freedom (DOFs) where the dummy static load is applied for the
           calculation of the equivalent velocities when 'method' is "ANALYTICAL".
           Default is `None`.
        amplitude_load : `array`, optional
            Amplitude of the dummy static load at the specified DOFs. Default is `None`.
        fitting_c : `tuple`, optional
            Parameters for fitting equivalent velocity regression.
            Structure: (fc1, fc2, fp1, fp2). Default is (0., 0., 0., 0.)
            - fc1 : `float`
                Exponent factor for the minimum reference velocity.
            - fc2 : `float`
                Exponent factor for the maximum reference velocity.
            - fp1 : `float`
                Exponent factor for the minimum equivalent velocity.
            - fp2 : `float`
                Exponent factor for the maximum equivalent velocity.

        Returns
        -------
        Lsp : `array` or `float`
            Array containing the computed eigenvalues or the
            first eigenvalue of the model with Neumann BCs.
        """

        validate_numeric("k", k, float_num=False, integer_num=True, lower_bound=0)

        if self.method in ["ANALYTICAL", "RAYLEIGH"]:
            shift = 0.  # No shift for analytical and Rayleigh methods

        if self.method == "ANALYTICAL":

            # Compute equivalent homogenized velocities
            c_eq, c_eqref = self.AnaModSol.homogenized_velocities(
                c, V, c_ref=c_ref, V_ref=V_ref, quad_rule=quad_rule,
                dof_load=dof_load, amplitude_load=amplitude_load)

            Lsp = self.AnaModSol.solver_analytical(c_eq, hyp_par, c_eqref=c_eqref,
                                                   fitting_c=fitting_c,
                                                   cut_plane_percent=cut_plane_percent)

        elif self.method == "RAYLEIGH":
            Lsp = self.solver_rayleigh_quotient(c, ufl_coordinates, V, mesh_limits,
                                                k=k, quad_rule=quad_rule)
        else:
            # Get weak forms for the modal problem
            a, m = self.assemble_weak_forms(c, V, quad_rule=quad_rule, shift=shift)

        if self.method.startswith("KRYLOVSCH"):
            Lsp = self.solver_with_ufl(a, m, k=k)

        elif self.method in ["ARNOLDI", "LANCZOS", "LOBPCG"]:
            Asp, Msp = assemble_sparse_matrices(a, m)
            Lsp = self.solver_with_sparse_matrix(Asp, Msp, self.method,
                                                 k=k, inv_oper=inv_oper)

        Lsp -= shift if shift > 0. else 0.

        return Lsp

    def estimate_timestep(self, c, V, final_time, shift=0.,
                          quad_rule=None, inv_oper=False, fraction=0.7):
        """Estimate the maximum stable timestep based on the spectral radius.

        Optionally uses the Gershgorin Circle Theorem to estimate the
        maximum generalized eigenvalue when 'method' is "ANALYTICAL".
        Otherwise, computes the maximum generalized eigenvalue exactly.

        Parameters
        ----------
        c : `Firedrake.Function`
            Velocity model.
        V : `Firedrake.FunctionSpace`
            Function space for the modal problem.
        final_time : `float`
            Final time for the transient simulation.
        shift: `float`, optional
            Value to stabilize the Neumann BC null space. Default is 0.
        quad_rule : `dict`, optional
            Quadrature rule to use for the integration.
            Default is `None`, which uses the default quadrature rule.
        inv_oper : `bool`, optional
            Option to use an inverse operator for improving convergence.
            Default is `False`.
        fraction : `float`, optional
            Fraction of the estimated timestep to use. Defaults to 0.7.

        Returns
        -------
        max_dt : `float`
            Estimated maximum stable timestep.
        """

        # Maximum eigenvalue
        if self.method == "ANALYTICAL":
            pprint("Estimating Maximum Eigenvalue", comm=self.comm)

            a, m = self.assemble_weak_forms(c, V, quad_rule=quad_rule, shift=shift)
            Asp, Msp_inv = assemble_sparse_matrices(a, m, return_M_inv=True)
            Lsp = Msp_inv.multiply(Asp)
            max_eigval = amax(abs(Lsp.diagonal())) - shift

        else:
            pprint("Computing Exact Maximum Eigenvalue", comm=self.comm)

            # (eig = 0 is a rigid body motion)
            Lsp = self.solve_eigenproblem(
                c, V=V, shift=shift, quad_rule=quad_rule, inv_oper=inv_oper)
            max_eigval = max(unique(Lsp[(Lsp > 0.) & (imag(Lsp) == 0.)]))

        # Maximum stable timestep
        max_dt = float(real(2. / sqrt(max_eigval)))
        pprint("Maximum Stable Timestep Should Be Approximately "
               f"(ms): {1e3 * max_dt:.3f}", comm=self.comm)

        max_dt *= fraction
        nt = int(final_time / max_dt) + 1
        max_dt = final_time / (nt - 1)

        return max_dt
