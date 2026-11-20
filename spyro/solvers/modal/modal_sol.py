import warnings

import firedrake as fire
from numpy import (
    array,
    ascontiguousarray,
    asarray,
    count_nonzero,
    empty,
    eye,
    imag,
    real,
    sqrt,
    unique,
)
from petsc4py import PETSc
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, eigs, eigsh, lobpcg
from slepc4py import SLEPc
from spyro.mpi.spyro_mpi import SpyroEnsemble
from .modal_forms_and_matrices import (
    assemble_sparse_matrices,
    maximum_diagonal_ratio,
    acoustic_weak_forms_c_in_stiffness,
)
from .modal_rq_matrices import generate_eigenfunctions, matrices_rayleigh_quotient
from ...utils.error_management import (
    validate_data_structure,
    validate_firedrake_parameter,
    validate_numeric,
    validate_parameter,
)

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# TODO: citation
# With additions by Alexandre Olender

# Krylov solver and preconditioner of each "KRYLOVSCH_(K)(P)" method
KRYLOVSCH_CONFIG = {
    "KRYLOVSCH_CH": {"ksp": "cg", "pc": "hypre"},
    "KRYLOVSCH_CG": {"ksp": "cg", "pc": "gamg"},
    "KRYLOVSCH_GH": {"ksp": "gmres", "pc": "hypre"},
    "KRYLOVSCH_GG": {"ksp": "gmres", "pc": "gamg"},
}


def _petsc_preconditioner(
    Bsp: csr_matrix, pc_type: str, factor_solver_type: str | None = None
) -> LinearOperator:
    """Wrap a serial PETSc preconditioner of a sparse matrix as a SciPy operator.

    Parameters
    ----------
    Bsp : `csr matrix`
        Sparse matrix from which the preconditioner is built.
    pc_type : `str`
        PETSc preconditioner type, such as "lu" or "gamg".
    factor_solver_type : `str`, optional
        Package that factorizes the matrix when `pc_type` is a factorization,
        such as "mumps". Default is `None`.

    Returns
    -------
    `LinearOperator`
        Operator that applies the preconditioner to a vector or to each column
        of a block of vectors.

    Notes
    -----
    The SciPy eigensolvers run on a single rank, so the matrix and the
    preconditioner live on ``PETSc.COMM_SELF``. With ``pc_type="lu"`` the
    operator applies the exact inverse of the matrix.
    """
    Bsp = csr_matrix(Bsp)
    mat = PETSc.Mat().createAIJ(
        size=Bsp.shape,
        csr=(
            Bsp.indptr.astype(PETSc.IntType),
            Bsp.indices.astype(PETSc.IntType),
            Bsp.data.astype(PETSc.ScalarType),
        ),
        comm=PETSc.COMM_SELF,
    )
    mat.assemble()
    pc = PETSc.PC().create(comm=PETSc.COMM_SELF)
    pc.setOperators(mat)
    pc.setType(pc_type)
    if factor_solver_type is not None:
        pc.setFactorSolverType(factor_solver_type)
    pc.setUp()
    x, b = mat.createVecs()

    def apply(r):
        """Apply the preconditioner to a vector or a block of vectors.

        Parameters
        ----------
        r : `array`
            Vector, or block of vectors as columns.

        Returns
        -------
        `array`
            Preconditioned vector or block, with the shape of `r`.
        """
        r = asarray(r)
        cols = r.reshape(r.shape[0], -1)
        out = empty(cols.shape, dtype=PETSc.ScalarType)
        for j in range(cols.shape[1]):
            b.setArray(ascontiguousarray(cols[:, j]))
            pc.apply(b, x)
            out[:, j] = x.getArray()
        return out.reshape(r.shape)

    return LinearOperator(Bsp.shape, matvec=apply, matmat=apply, dtype=Bsp.dtype)


class Modal_Solver:
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
        "KRYLOVSCH" methods solve the problem distributed with SLEPc,
        while "ARNOLDI", "LANCZOS" and "LOBPCG" gather the whole problem
        and solve it with SciPy on one rank, which broadcasts the
        eigenvalues. "RAYLEIGH" is unavailable when `calc_max_dt` is
        `True`. With `calc_max_dt`, every "KRYLOVSCH" method solves the
        mass matrix systems with cg and a Jacobi preconditioner.

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
            "KRYLOVSCH" methods solve the problem distributed with SLEPc,
            while "ARNOLDI", "LANCZOS" and "LOBPCG" gather the whole problem
            and solve it with SciPy on one rank, which broadcasts the
            eigenvalues. "RAYLEIGH" is unavailable when `calc_max_dt` is
            `True`. With `calc_max_dt`, every "KRYLOVSCH" method solves the
            mass matrix systems with cg and a Jacobi preconditioner.
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
        self.valid_methods = (
            def_methods
            + list(KRYLOVSCH_CONFIG)
            + ([] if self.calc_max_dt else ["RAYLEIGH"])
        )

        # Method for solving the eigenproblem
        method = "KRYLOVSCH_CH" if method is None else method
        self.method = validate_parameter("method", method, self.valid_methods)

        # Initializing the analytical solver
        if not self.calc_max_dt and self.method == "ANALYTICAL":
            from .modal_ana_sol import Modal_Analytical_Solver

            self.AnaModSol = Modal_Analytical_Solver(
                dimension=self.dimension, comm=comm
            )

        SpyroEnsemble.print(f"Solver Method: {self.method}")

    @SpyroEnsemble.run_in_one_core_and_broadcast
    def solver_with_sparse_matrix(self, Asp, Msp, method, k=2):
        """Solve the eigenvalue problem with sparse matrices using Scipy.

        Every rank holds the whole matrices, so the problem is solved on a
        single rank and its eigenvalues are broadcast to the other ranks and
        ensemble members, which must all call this method.

        For the smallest eigenvalues, "ARNOLDI" and "LANCZOS" use
        shift-and-invert with sigma = 0, whose operator inv(Asp) is applied
        exactly with a MUMPS factorization of Asp: Cholesky (LDL^T) for
        "LANCZOS", which assumes a symmetric problem, and LU for "ARNOLDI".
        With `calc_max_dt` they seek the largest eigenvalues without a shift,
        on copies of the matrices without explicitly stored zeros, with the
        exact inverse of Msp. "LOBPCG" works on copies without stored zeros,
        and for the smallest eigenvalues is preconditioned with algebraic
        multigrid (GAMG).

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

        Returns
        -------
        Lsp : `array`
            Array containing the computed eigenvalues.

        Notes
        -----
        The factorization dominates the cost of "ARNOLDI" and "LANCZOS". For
        the 3D heterogeneous Q model of test_modal_solvers.py (71944 dofs, 8.5
        million stored entries, of which 0.9 million are nonzero), MUMPS
        factorizes Asp in 2.8 s against 17.3 s for SuperLU with a symmetric
        ordering, and the eigenvalues agree to the printed digits. The whole
        shift-and-invert solve takes 4.1 s with the MUMPS LU and 3.0 s with
        its Cholesky (LDL^T) factorization. The factorization keeps the stored
        zeros of Asp: without them it took 5.5 s.

        With `calc_max_dt` there is no shift, and ARPACK applies inv(Msp) at
        every iteration. SciPy would build it with SuperLU over the stored
        sparsity, which for lumped elements is the full element pattern of
        exact zeros around a diagonal: 196 s in the same model. On copies
        without the zeros, with inv(Msp) as the reciprocal of the diagonal of
        a lumped (diagonal) mass matrix, or a MUMPS LU factorization of any
        other one, the same largest eigenvalue takes 2.6 s.

        Without a preconditioner, "LOBPCG" needs thousands of iterations
        (1295 in 3D, and it stops at 2500 in 2D before reaching its
        tolerance). With GAMG it converges in tens. Asp is nearly singular,
        since its smallest eigenvalue is the shift that stabilizes the Neumann
        null space, and multigrid on Asp itself returns wrong eigenvalues. The
        preconditioner is therefore built for Asp + tau Msp, with
        tau = mean(diag(Asp) / diag(Msp)) / n**(2 / dimension) for n dofs,
        an estimate of the low end of the spectrum: the diagonal ratio scales
        as the largest eigenvalue, and the ratio of the largest to the
        smallest nonzero one as the squared number of cells per side. LOBPCG
        converges to the same eigenvalue for tau from 1e-3 to 1e2 in both
        tests. With `calc_max_dt` the largest eigenvalues are sought, which
        this preconditioner would not help, so none is used.
        """

        # Check methods
        validate_parameter("method", method, ["ARNOLDI", "LANCZOS", "LOBPCG"])

        if method == "ARNOLDI" or method == "LANCZOS":
            if self.calc_max_dt:
                # Largest eigenvalues without a shift: ARPACK applies inv(Msp)
                # at every iteration, here exactly, on zero-free copies
                Asp, Msp = Asp.copy(), Msp.copy()
                Asp.eliminate_zeros()
                Msp.eliminate_zeros()
                diag_M = Msp.diagonal()
                if Msp.nnz == count_nonzero(diag_M):  # Lumped mass matrix
                    Minv = diags(1.0 / diag_M)
                else:
                    Minv = _petsc_preconditioner(Msp, "lu", factor_solver_type="mumps")
                OPinv = None

            else:
                # Shift-and-invert with sigma = 0 applies inv(Asp) exactly.
                # LANCZOS assumes a symmetric problem, so it uses the cheaper
                # Cholesky (LDL^T) factorization; ARNOLDI keeps the general LU
                factor = "cholesky" if method == "LANCZOS" else "lu"
                OPinv = _petsc_preconditioner(Asp, factor, factor_solver_type="mumps")
                Minv = None

        if method == "ARNOLDI":
            # Solve the eigenproblem using ARNOLDI (ARPACK)
            if self.calc_max_dt:
                Lsp = eigs(
                    Asp,
                    k=k,
                    M=Msp,
                    which="LM",
                    Minv=Minv,
                    OPinv=OPinv,
                    return_eigenvectors=False,
                )
            else:
                Lsp = eigs(
                    Asp,
                    k=k,
                    M=Msp,
                    sigma=0.0,
                    Minv=Minv,
                    OPinv=OPinv,
                    return_eigenvectors=False,
                )

        if method == "LANCZOS":
            # Solve the eigenproblem using LANCZOS (ARPACK)
            if self.calc_max_dt:
                Lsp = eigsh(
                    Asp,
                    k=k,
                    M=Msp,
                    which="LM",
                    Minv=Minv,
                    OPinv=OPinv,
                    return_eigenvectors=False,
                )
            else:
                Lsp = eigsh(
                    Asp,
                    k=k,
                    M=Msp,
                    sigma=0.0,
                    Minv=Minv,
                    OPinv=OPinv,
                    return_eigenvectors=False,
                )

        if method == "LOBPCG":
            # Initialize LI vectors for LOBPCG
            X = eye(Msp.shape[0], k)

            # Lumped (spectral or KMV) elements store many exact zeros in the
            # element sparsity, which only slow down the products of LOBPCG
            Asp, Msp = Asp.copy(), Msp.copy()
            Asp.eliminate_zeros()
            Msp.eliminate_zeros()

            # Algebraic multigrid preconditioner for the smallest eigenvalues,
            # built for Asp + tau Msp, since Asp is nearly singular (see Notes)
            if self.calc_max_dt:
                precond = None
            else:
                ratio = (Asp.diagonal() / Msp.diagonal()).mean()
                tau = ratio / Asp.shape[0] ** (2.0 / self.dimension)
                precond = _petsc_preconditioner(Asp + tau * Msp, "gamg")

            # Solve the eigenproblem using LOBPCG (iterative method)
            it_mod = 2500
            it_ext = 2
            mag = True if self.calc_max_dt else False
            for it in range(it_ext):
                Lsp, X, resid = lobpcg(
                    Asp,
                    X,
                    B=Msp,
                    M=precond,
                    tol=5e-4,
                    maxiter=it_mod,
                    largest=mag,
                    retResidualNormsHistory=True,
                )

                it_mod //= 2  # Reduce iterations for next loop
                rmin = array(resid)[:, 1].min()
                if rmin < 5e-4 or it_mod < 20:
                    del X, resid
                    break

        return Lsp

    def solver_with_ufl(self, a, m, k=2):
        """Solve the eigenvalue problem using UFL forms with SLEPc.

        With `calc_max_dt`, the eigenvalues of largest magnitude are
        computed without a spectral transformation, so each iteration only
        solves a system with the mass matrix. Whatever the method, that
        system is solved with cg and a Jacobi preconditioner, which is exact
        for a lumped (diagonal) mass matrix, and the eigenvalues to a
        relative tolerance of 1e-4. Otherwise, the smallest ones are computed
        with shift-and-invert, with the Krylov solver and preconditioner of
        the method. When the shift-and-invert system is solved with a direct
        factorization, which is the SLEPc default, MUMPS factorizes it.

        Parameters
        ----------
        a : `firedrake.Form`
            Weak form representing the stiffness matrix.
        m : `firedrake.Form`
            Weak form representing the mass matrix.
        k : `int`, optional
            Number of eigenvalues to compute. Default is 2.

        Returns
        -------
        `numpy.ndarray`
            The computed eigenvalues, identical on every rank.

        Raises
        ------
        firedrake.ConvergenceError
            If SLEPc does not converge any eigenvalue.
        """

        opts = {
            "eps_gen_hermitian": None,  # Problem is Hermitian
            "eps_type": "krylovschur",  # Robust, widely used eigensolver
            "eps_max_it": 200,  # Reasonable iteration cap
        }
        if self.calc_max_dt:
            opts.update(
                {
                    # The largest eigenvalues of a uniform mesh are tightly
                    # clustered, so 1e-6 may need more than eps_max_it
                    # iterations. 1e-4 changes the time step by about 1e-5
                    "eps_tol": 1e-4,
                    "st_type": "shift",  # No inversion, only mass matrix solves
                    # Jacobi is exact for a lumped (diagonal) mass matrix, and a
                    # consistent one scaled by its diagonal is well conditioned
                    "st_ksp_type": "cg",
                    "st_pc_type": "jacobi",
                    "st_ksp_rtol": 1e-10,
                }
            )
        else:
            m_diagonal = fire.assemble(m).petscmat.getDiagonal()
            a_diagonal = fire.assemble(a).petscmat.getDiagonal()
            max_m = m_diagonal.max()[1]
            max_a = a_diagonal.max()[1]
            st_shift = max(1e-5, 1e-8 * abs(max_a / max_m))

            opts.update(
                {
                    # Smallest eigenvalues magnitude
                    "eps_tol": 1e-6,  # Tight tolerance for accuracy
                    "st_shift": st_shift,  # Stabilizes Neumann BC null space
                    "st_type": "sinvert",  # Shift-and-invert: targets small eigenvalues
                    "eps_target": 0.0,  # Converge toward eigenvalues closest to 0
                    "eps_target_magnitude": None,  # Select by magnitude
                    "st_pc_factor_shift_type": "NONZERO",  # Prevent zero-pivot
                    # The shift-and-invert factorization dominates the solve.
                    # MUMPS is several times faster than the PETSc LU in 3D
                    "st_pc_factor_mat_solver_type": "mumps",
                    "eps_monitor": "ascii",  # Print convergence info
                    "ksp_type": KRYLOVSCH_CONFIG[self.method]["ksp"],  # Large problems
                    "pc_type": KRYLOVSCH_CONFIG[self.method]["pc"],  # Large problems
                }
            )

        eigenproblem = fire.LinearEigenproblem(a, M=m)
        eigensolver = fire.LinearEigensolver(
            eigenproblem, n_evals=k, solver_parameters=opts
        )
        if self.calc_max_dt:
            # LinearEigensolver always sets a target, which makes SLEPc look
            # for the eigenvalues closest to it even when the largest are
            # requested in the options. The options are only applied when it
            # is built, so the search is changed on the SLEPc solver itself
            largest = SLEPc.EPS.Which.LARGEST_MAGNITUDE
            eigensolver.es.setWhichEigenpairs(largest)

        n_converged = eigensolver.solve()

        Lsp = asarray(
            [eigensolver.eigenvalue(mod) for mod in range(min(k, n_converged))]
        )

        return Lsp

    def solver_rayleigh_quotient(
        self, c, ufl_coordinates, V, mesh_limits, k=2, quad_rule=None
    ):
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
            Number of eigenvalues to compute. It also sets the size of the
            basis, max(2 k, 2) eigenfunctions per direction. Default is 2.
        quad_rule : `dict`, optional
            Quadrature rule to use for the integration.
            Default is `None`, which uses the default quadrature rule.

        Returns
        -------
        Lsp : `array`
            Array containing the computed eigenvalues.

        Notes
        -----
        The basis is the products of cosines that are the Neumann
        eigenfunctions of the Laplacian on the box given by `mesh_limits`,
        interpolated into `V`. The
        stiffness and mass matrices of the basis are built by
        `matrices_rayleigh_quotient` from one assembly of the stiffness and
        mass matrices of `V`, so the gradients of the eigenfunctions are not
        formed. The small generalized eigenvalue problem is then solved with
        "ARNOLDI".
        """

        # Check input arguments
        validate_firedrake_parameter("c", c, "Function")
        validate_firedrake_parameter(
            "ufl_coordinates", ufl_coordinates, "SpatialCoordinate"
        )
        validate_firedrake_parameter("V", V, "FunctionSpace")
        validate_data_structure("mesh_limits", mesh_limits, "tuple")
        validate_data_structure(
            "quad_rule", quad_rule, "dict", accept_parameter_as_none=True
        )

        # Create eigenfunctions
        eig_funcs = generate_eigenfunctions(
            ufl_coordinates, V, mesh_limits, k=k, dimension=self.dimension
        )

        # Assemble matrices for generalized eigenvalue problem
        Asp, Msp = matrices_rayleigh_quotient(c, eig_funcs, quad_rule=quad_rule)

        # Solve the generalized eigenvalue problem
        Lsp = self.solver_with_sparse_matrix(Asp, Msp, "ARNOLDI", k=k)

        return Lsp

    def assemble_weak_forms(self, c, V, quad_rule=None, shift=0.0, forms=None):
        """Build the weak forms for the modal problem solved using UFL or sparse matrices.

        Parameters
        ----------
        c : `Firedrake.Function`
            Velocity model. Ignored when ``forms`` is given.
        V : `Firedrake.FunctionSpace`
            Function space for the modal problem.
        quad_rule : `dict`, optional
            Quadrature rule to use for the integration.
            Default is `None`, which uses the default quadrature rule.
        shift: `float`, optional
            Value to stabilize the Neumann BC null space. Default is 0.
        forms : `tuple` of `Firedrake.Form`, optional
            Stiffness and mass forms ``(a, m)`` supplied by the wave
            physics. Default is `None`, which builds the scalar acoustic
            forms from ``c``.

        Returns
        -------
        a : `Firedrake.Form`
            Weak form representing the stiffness matrix.
        m : `Firedrake.Form`
            Weak form  representing the mass matrix.
        """

        # Check input arguments
        validate_firedrake_parameter("V", V, "FunctionSpace")
        validate_data_structure(
            "quad_rule", quad_rule, "dict", accept_parameter_as_none=True
        )
        validate_numeric(
            "shift",
            shift,
            float_num=True,
            integer_num=True,
            lower_bound=0.0,
            include_lower_bound=True,
        )

        # Get bilinear forms
        if forms is None:
            validate_firedrake_parameter("c", c, "Function")
            a, m = acoustic_weak_forms_c_in_stiffness(c, V, quad_rule=quad_rule)
        else:
            a, m = forms

        # Add shift to stabilize Neumann BC null space
        if shift > 0:
            a += shift * m

        return a, m

    def solve_eigenproblem(
        self,
        c,
        V=None,
        k=2,
        shift=0.0,
        quad_rule=None,
        ufl_coordinates=None,
        mesh_limits=None,
        hyp_par=None,
        cut_plane_percent=1.0,
        c_ref=None,
        V_ref=None,
        dof_load=None,
        amplitude_load=None,
        fitting_c=(0.0, 0.0, 0.0, 0.0),
        forms=None,
    ):
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
        forms : `tuple` of `Firedrake.Form`, optional
            Stiffness and mass forms ``(a, m)`` supplied by the wave
            physics. Only used by the numerical eigensolvers. Default is
            `None`, which builds the scalar acoustic forms from ``c``.

        Returns
        -------
        Lsp : `array` or `float`
            Array containing the computed eigenvalues or the
            first eigenvalue of the model with Neumann BCs.
        """

        validate_numeric("k", k, float_num=False, integer_num=True, lower_bound=0)

        if self.method in ["ANALYTICAL", "RAYLEIGH"]:
            shift = 0.0  # No shift for analytical and Rayleigh methods

        if self.method == "ANALYTICAL":
            # Compute equivalent homogenized velocities
            c_eq, c_eqref = self.AnaModSol.homogenized_velocities(
                c,
                V,
                c_ref=c_ref,
                V_ref=V_ref,
                quad_rule=quad_rule,
                dof_load=dof_load,
                amplitude_load=amplitude_load,
            )

            Lsp = self.AnaModSol.solver_analytical(
                c_eq,
                hyp_par,
                c_eqref=c_eqref,
                fitting_c=fitting_c,
                cut_plane_percent=cut_plane_percent,
            )

        elif self.method == "RAYLEIGH":
            Lsp = self.solver_rayleigh_quotient(
                c, ufl_coordinates, V, mesh_limits, k=k, quad_rule=quad_rule
            )
        else:
            # Get weak forms for the modal problem
            a, m = self.assemble_weak_forms(
                c, V, quad_rule=quad_rule, shift=shift, forms=forms
            )

        if self.method.startswith("KRYLOVSCH"):
            Lsp = self.solver_with_ufl(a, m, k=k)

        elif self.method in ["ARNOLDI", "LANCZOS", "LOBPCG"]:
            Asp, Msp = assemble_sparse_matrices(a, m)
            Lsp = self.solver_with_sparse_matrix(Asp, Msp, self.method, k=k)

        Lsp -= shift if shift > 0.0 else 0.0

        return Lsp

    def estimate_timestep(
        self,
        c,
        V,
        final_time,
        shift=0.0,
        quad_rule=None,
        fraction=0.7,
        forms=None,
    ):
        """Estimate the maximum stable timestep based on the spectral radius.

        Optionally uses the Gershgorin Circle Theorem to estimate the
        maximum generalized eigenvalue when 'method' is "ANALYTICAL".
        Otherwise, computes the maximum generalized eigenvalue exactly.

        Parameters
        ----------
        c : `Firedrake.Function` or `None`
            Velocity model. Ignored, and may be `None`, when ``forms`` is
            given.
        V : `Firedrake.FunctionSpace`
            Function space for the modal problem.
        final_time : `float`
            Final time for the transient simulation.
        shift: `float`, optional
            Value to stabilize the Neumann BC null space. Default is 0.
        quad_rule : `dict`, optional
            Quadrature rule to use for the integration.
            Default is `None`, which uses the default quadrature rule.
        fraction : `float`, optional
            Fraction of the estimated timestep to use. Defaults to 0.7.
        forms : `tuple` of `Firedrake.Form`, optional
            Stiffness and mass forms ``(a, m)`` supplied by the wave
            physics. Default is `None`, which is deprecated and builds the
            acoustic forms with the velocity in the stiffness from ``c``
            (see ``acoustic_weak_forms_c_in_stiffness``).

        Returns
        -------
        max_dt : `float`
            Estimated maximum stable timestep.
        """

        if forms is None:
            warnings.warn(
                "Calling estimate_timestep without forms builds the acoustic "
                "forms with the velocity in the stiffness "
                "(acoustic_weak_forms_c_in_stiffness), which differ from the "
                "forward problem for a heterogeneous velocity. This is "
                "deprecated; pass the forms of the wave physics instead, e.g. "
                "forms=wave.modal_weak_forms().",
                DeprecationWarning,
                stacklevel=2,
            )

        # Maximum eigenvalue
        if self.method == "ANALYTICAL":
            SpyroEnsemble.print("Estimating Maximum Eigenvalue")

            a, m = self.assemble_weak_forms(
                c, V, quad_rule=quad_rule, shift=shift, forms=forms
            )
            max_eigval = maximum_diagonal_ratio(a, m) - shift

        else:
            SpyroEnsemble.print("Computing Exact Maximum Eigenvalue")

            # (eig = 0 is a rigid body motion)
            Lsp = self.solve_eigenproblem(
                c,
                V=V,
                shift=shift,
                quad_rule=quad_rule,
                forms=forms,
            )
            max_eigval = max(unique(Lsp[(Lsp > 0.0) & (imag(Lsp) == 0.0)]))

        # Maximum stable timestep
        max_dt = float(real(2.0 / sqrt(max_eigval)))
        # Every rank must take the same number of time steps, so rank 0's
        # estimate is shared in case the ranks' eigensolves differ in the
        # last bits
        max_dt = V.mesh().comm.bcast(max_dt, root=0)
        SpyroEnsemble.print(
            "Maximum Stable Timestep Should Be Approximately "
            f"(ms): {1e3 * max_dt:.3f}"
        )

        max_dt *= fraction
        nt = round(final_time / max_dt) + 1
        max_dt = final_time / (nt - 1)

        return max_dt
