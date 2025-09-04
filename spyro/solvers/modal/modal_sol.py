import firedrake as fire
import numpy as np
import scipy.linalg as sl
import scipy.sparse as ss
from spyro.utils.error_management import value_parameter_error

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class Modal_Solver():
    '''
    Class for the Modal problem with Neumann boundary conditions

    Attributes
    ----------
    calc_max_dt : `bool`
        Option to estimate the maximum stable timestep for the computation
        of the transient response. Default is False
    dimension : `int`
        The spatial dimension of the problem
    method : `str`
        Method to use for solving the eigenvalue problem.
        Default is None, which uses as the 'ARNOLDI' method in 2D  models
        and the 'KRYLOVSCH_CH' method in 3D models
    valid_methods: `list`
        List of valid methods for solving the eigenproblem
        Options: 'ARNOLDI', 'LANCZOS', 'LOBPCG' 'KRYLOVSCH_CH',
        'KRYLOVSCH_CG', 'KRYLOVSCH_GH' or 'KRYLOVSCH_GG'.
        In 'KRYLOVSCH_(K)(P)' methods, (K) indicates the Krylov solver to
        use: 'C' for Conjugate Gradient (cg) or 'G' for Generalized Minimal
        Residual (gmres). (P) indicates the preconditioner to use: 'H' for
        Hypre (hypre) or 'G' for Geometric Algebraic Multigrid (gamg). For
        example, 'KRYLOVSCH_CH' uses cg solver with hypre preconditioner

    Methods
    -------
    assemble_sparse_matrices()
        Assemble the sparse matrices for SciPy solvers
    bilinear_forms()
        Generate the bilinear forms for the problem
    solve_eigenproblem()
        Solve the eigenvalue problem with Neumann boundary conditions
    solver_with_sparse_matrix()
        Solve the eigenvalue problem with sparse matrices using SciPy
    solver_with_ufl()
        Solve the eigenvalue problem using UFL forms with SLEPc
    '''

    def __init__(self, dimension, method=None, calc_max_dt=False):
        '''
        Initialize the Modal_Solver class

        Parameters
        ----------
        dimension : `int`
            The spatial dimension of the problem
        method : `str`, optional
            Method to use for solving the eigenvalue problem.
            Default is None, which uses as the 'ARNOLDI' method in 2D  models
            and the 'KRYLOVSCH_CH' method in 3D models.
            Options: 'ARNOLDI', 'LANCZOS', 'LOBPCG' 'KRYLOVSCH_CH',
            'KRYLOVSCH_CG', 'KRYLOVSCH_GH' or 'KRYLOVSCH_GG'.
            In 'KRYLOVSCH_(K)(P)' methods, (K) indicates the Krylov solver to
            use: 'C' for Conjugate Gradient (cg) or 'G' for Generalized Minimal
            Residual (gmres). (P) indicates the preconditioner to use: 'H' for
            Hypre (hypre) or 'G' for Geometric Algebraic Multigrid (gamg). For
            example, 'KRYLOVSCH_CH' uses cg solver with hypre preconditioner.
        calc_max_dt : `bool`, optional
            Option to estimate the maximum stable timestep for the computation
            of the transient response. Default is False

        Returns
        -------
        None
        '''

        # Dimension of the problem
        self.dimension = dimension

        # Option to estimate the maximum stable timestep
        self.calc_max_dt = calc_max_dt

        # Default methods for each dimension
        if method is None:
            if self.dimension == 2:  # 2D
                self.method = 'ARNOLDI'

            if self.dimension == 3:  # 3D
                self.method = 'LOBPCG' if self.calc_max_dt else 'KRYLOVSCH_CH'
        else:
            self.method = method

        # Valid methods for solving the eigenproblem
        self.valid_methods = ['ARNOLDI', 'LANCZOS', 'LOBPCG', 'KRYLOVSCH_CH',
                              'KRYLOVSCH_CG', 'KRYLOVSCH_GH', 'KRYLOVSCH_GG']

        if self.method not in self.valid_methods:
            value_parameter_error('method', method, self.valid_methods)

        print(f'Solver Method: {self.method}')

    @staticmethod
    def bilinear_forms(c, V, quad_rule=None):
        '''
        Generate the bilinear forms for the problem

        Parameters
        ----------
        c : `firedrake function`
            Velocity model
        V : `firedrake function space`
            Function space for the modal problem
        quad_rule : `str`, optional
            Quadrature rule to use for the integration.
            Default is None, which uses the default quadrature rule.

        Returns
        -------
        a : `firedrake form`
            Weak form representing the stiffness matrix
        m : `firedrake form`
            Weak form  representing the mass matrix
        '''

        # Functions for the problem
        u, v = fire.TrialFunction(V), fire.TestFunction(V)
        dx = fire.dx(scheme=quad_rule) if quad_rule else fire.dx

        # Bilinear forms
        a = c * c * fire.inner(fire.grad(u), fire.grad(v)) * dx
        m = fire.inner(u, v) * dx

        return a, m

    @staticmethod
    def assemble_sparse_matrices(a, m, return_M_inv=False):
        '''
        Assemble the sparse matrices for SciPy solvers

        Parameters
        ----------
        a : `firedrake form`
            Weak form representing the stiffness matrix
        m : `firedrake form`
            Weak form  representing the mass matrix
        return_M_inv : `bool`, optional
            Option to return the inverse mass matrix instead of the mass

        Returns
        -------
        Asp : `csr matrix`
            Sparse matrix representing the stiffness matrix
        Msp : `csr matrix`
            Sparse matrix representing the mass matrix
        Msp_inv : `csr matrix`
            Sparse matrix representing the inverse mass matrix
        '''

        # Assemble the stiffness matrix
        A = fire.assemble(a)
        a_ptr, a_ind, a_val = A.petscmat.getValuesCSR()
        Asp = ss.csr_matrix((a_val, a_ind, a_ptr), A.petscmat.size)

        # Assemble the mass matrix
        M = fire.assemble(m)
        m_ptr, m_ind, m_val = M.petscmat.getValuesCSR()
        Msp = ss.csr_matrix((m_val, m_ind, m_ptr), M.petscmat.size)

        if return_M_inv:
            # Assemble the inverse mass matrix
            m_val_inv = np.array(m_val)
            m_val_inv[m_val_inv != 0.] = 1. / m_val_inv[m_val_inv != 0.]
            Msp_inv = ss.csr_matrix((m_val_inv, m_ind, m_ptr), M.petscmat.size)
            return Asp, Msp_inv

        return Asp, Msp

    def solver_with_sparse_matrix(self, a, m, k=2, inv_oper=False):
        '''
        Solve the eigenvalue problem with sparse matrices using Scipy

        Parameters
        ----------
        a : `firedrake form`
            Weak form representing the stiffness matrix
        m : `firedrake form`
            Weak form  representing the mass matrix
        k : `int`, optional
            Number of eigenvalues to compute. Default is 2
        inv_oper : `bool`, optional
            Option to use an inverse operator for improving convergence.
            Default is False

        Returns
        -------
        Lsp : `array`
            Array containing the computed eigenvalues
        '''

        Asp, Msp = self.assemble_sparse_matrices(a, m)

        if self.method == 'ARNOLDI' or self.method == 'LANCZOS':
            # Inverse operator for improving convergence
            M_ilu = ss.linalg.spilu(Msp) if inv_oper else None
            Minv = M_ilu.solve if inv_oper else None
            A_ilu = ss.linalg.spilu(Asp) if inv_oper else None
            OPinv = A_ilu.solve if inv_oper else None
            sigma = None if self.calc_max_dt else 0.
            mag = 'LM' if self.calc_max_dt else 'SM'

        if self.method == 'ARNOLDI':
            # Solve the eigenproblem using ARNOLDI (ARPACK)
            Lsp = ss.linalg.eigs(
                Asp, k=k, M=Msp, sigma=sigma, which=mag,
                Minv=Minv, OPinv=OPinv, return_eigenvectors=False)

        if self.method == 'LANCZOS':
            # Solve the eigenproblem using LANCZOS (ARPACK)
            Lsp = ss.linalg.eigsh(
                Asp, k=k, M=Msp, sigma=sigma, which=mag,
                Minv=Minv, OPinv=OPinv, return_eigenvectors=False)

        if self.method == 'LOBPCG':
            # Initialize random vectors for LOBPCG
            X = sl.orth(np.random.rand(Msp.shape[0], k))

            # Solve the eigenproblem using LOBPCG (iterative method)
            it_mod = 2500
            it_ext = 2
            mag = True if self.calc_max_dt else False
            for it in range(it_ext):
                Lsp, X, resid = ss.linalg.lobpcg(Asp, X, B=Msp, tol=5e-4,
                                                 maxiter=it_mod, largest=mag,
                                                 retResidualNormsHistory=True)

                it_mod //= 2  # Reduce iterations for next loop
                rmin = np.array(resid)[:, 1].min()
                if rmin < 5e-4 or it_mod < 20:
                    del X, resid
                    break

        return Lsp

    def solver_with_ufl(self, a, m, k=2):
        '''
        Solve the eigenvalue problem using UFL forms with SLEPc

        Parameters
        ----------
        a : `firedrake form`
            Weak form representing the stiffness matrix
        m : `firedrake form`
            Weak form  representing the mass matrix
        k : `int`, optional
            Number of eigenvalues to compute. Default is 2

        Returns
        -------
        Lsp : `array`
            Array containing the computed eigenvalues
        '''

        if self.method[-2] == "C":
            ksp_type = "cg"
        elif self.method[-2] == "G":
            ksp_type = "gmres"

        if self.method[-1] == "H":
            pc_type = "hypre"
        elif self.method[-1] == "G":
            pc_type = "gamg"

        opts = {
            "eps_gen_hermitian": None,       # Problem is Hermitian
            "eps_type": "krylovschur",       # Robust, widely used eigensolver
            "eps_tol": 1e-6,                 # Tight tolerance for accuracy
            "eps_max_it": 200,               # Reasonable iteration cap
            "st_type": "sinvert",            # Useful for interior eigenvalues
            "st_shift": 1e-6,                # Stabilizes Neumann BC null space
            "eps_smallest_magnitude": None,  # Smallest eigenvalues magnitude
            "eps_monitor": "ascii",          # Print convergence info
            "ksp_type": ksp_type,            # Options for large problems
            "pc_type": pc_type               # Options for large problems
        }

        if self.calc_max_dt:
            # Largest eigenvalues magnitude
            opts.pop("eps_smallest_magnitude")
            opts.update({"eps_largest_magnitude": None})
            # subspace, arnoldi, krylovschur, lapack

        eigenproblem = fire.LinearEigenproblem(a, M=m)
        eigensolver = fire.LinearEigensolver(eigenproblem, n_evals=k,
                                             solver_parameters=opts)
        nconv = eigensolver.solve()
        Lsp = np.asarray([eigensolver.eigenvalue(mod) for mod in range(k)])

        return Lsp

    def solve_eigenproblem(self, c, V, k=2, shift=0.,
                           quad_rule=None, inv_oper=False):
        '''
        Solve the eigenvalue problem with Neumann boundary conditions

        Parameters
        ----------
        c : `firedrake function`
            Velocity model
        V : `firedrake function space`
            Function space for the modal problem
        k : `int`, optional
            Number of eigenvalues to compute. Default is 2
        shift: `float`, optional
            Value to stabilize the Neumann BC null space. Default is 0
        quad_rule : `str`, optional
            Quadrature rule to use for the integration.
            Default is None, which uses the default quadrature rule.
        inv_oper : `bool`, optional
            Option to use an inverse operator for improving convergence.
            Default is False

        Returns
        -------
        Lsp : `array`
            Array containing the computed eigenvalues
        '''

        a, m = self.bilinear_forms(c, V, quad_rule=quad_rule)

        if shift > 0:
            a += shift * m

        if self.method[:-3] == 'KRYLOVSCH':
            Lsp = self.solver_with_ufl(a, m, k=k)
        else:
            Lsp = self.solver_with_sparse_matrix(a, m, k=k, inv_oper=inv_oper)

        Lsp -= shift if shift > 0. else 0.

        return Lsp

    def estimate_timestep(self, c, V, final_time, shift=0., quad_rule=None,
                          inv_oper=False, estimate_maxeig=False, fraction=0.7):
        '''
        Estimate the maximum stable timestep based on the spectral
        radius using optionally the Gershgorin Circle Theorem to
        estimate the maximum generalized eigenvalue. Otherwise
        computes the maximum generalized eigenvalue exactly

        Parameters
        ----------
        c : `firedrake function`
            Velocity model
        V : `firedrake function space`
            Function space for the modal problem
        final_time : `float`
            Final time for the transient simulation
        shift: `float`, optional
            Value to stabilize the Neumann BC null space. Default is 0
        quad_rule : `str`, optional
            Quadrature rule to use for the integration.
            Default is None, which uses the default quadrature rule.
        inv_oper : `bool`, optional
            Option to use an inverse operator for improving convergence.
            Default is False
        estimate_maxeig : `bool`, optional
            Option to estimate the maximum eigenvalue using the
            Gershgorin Circle Theorem. Default is False
        fraction : `float`, optional
            Fraction of the estimated timestep to use. Defaults to 0.7.

        Returns
        -------
        max_dt : `float`
            Estimated maximum stable timestep
        '''

        # Maximum eigenvalue
        if estimate_maxeig:
            a, m = self.bilinear_forms(c, V, quad_rule=quad_rule)

            if shift > 0:
                a += shift * m

            Asp, Msp_inv = \
                self.assemble_sparse_matrices(a, m, return_M_inv=True)
            Lsp = Msp_inv.multiply(Asp).toarray().squeeze()
            Lsp -= shift if shift > 0. else 0.

            print(f"Estimating Maximum Eigenvalue", flush=True)
            max_eigval = np.amax(np.abs(Lsp.diagonal()))[0]

        else:
            print(f"Computing Exact Maximum Eigenvalue", flush=True)
            Lsp = self.solve_eigenproblem(
                c, V, k=2, shift=shift, quad_rule=quad_rule, inv_oper=inv_oper)
            # (eig = 0 is a rigid body motion)
            max_eigval = max(np.unique(Lsp[(Lsp > 0.) & (np.imag(Lsp) == 0.)]))

        # Maximum stable timestep
        max_dt = float(np.real(2. / np.sqrt(max_eigval)))
        print(f"Maximum Stable Timestep Should Be Approximately "
              f"(ms): {1e3 * max_dt:.3f}", flush=True)

        max_dt *= fraction
        nt = int(final_time / max_dt) + 1
        max_dt = final_time / (nt - 1)

        return max_dt
