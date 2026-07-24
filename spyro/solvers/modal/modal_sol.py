import firedrake as fire
import numpy as np
import scipy.sparse as ss
from scipy.optimize import broyden1, curve_fit
from scipy.special import (beta, betainc, gamma, jn_zeros, jnp_zeros,
                           mathieu_modcem1, spherical_jn)
from scipy.stats import norm as sn
from sys import float_info
from ...utils.error_management import value_parameter_error
from ...utils.stats_tools import coeff_of_determination
from ...io.basicio import parallel_print as pprint

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
    calc_max_dt : `bool`
        Option to estimate the maximum stable timestep for the computation of the
        transient response. Default is `False`
    comm : `object`, optional
        An object representing the communication interface for parallel processing.
        Default is `None`.
    dimension : `int`
        Model dimension (2D or 3D). Default is 2D.
    method : `str`
        Method to use for solving the eigenvalue problem.
        Default is `None`, which uses the 'KRYLOVSCH_CH' method.
    valid_methods: `list`
        List of valid methods for solving the eigenproblem
        Options: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS', 'LOBPCG', 'KRYLOVSCH_CH',
        'KRYLOVSCH_CG', 'KRYLOVSCH_GH', 'KRYLOVSCH_GG' or 'RAYLEIGH'.
        'ANALYTICAL' method is an approximation by using homogenization techniques.
        'RAYLEIGH' method is an approximation by Rayleigh quotient.
        In 'KRYLOVSCH_(K)(P)' methods, (K) indicates the Krylov solver to
        use: 'C' for Conjugate Gradient (cg) or 'G' for Generalized Minimal
        Residual (gmres). (P) indicates the preconditioner to use: 'H' for
        Hypre (hypre) or 'G' for Geometric Algebraic Multigrid (gamg). For
        example, 'KRYLOVSCH_CH' uses cg solver with hypre preconditioner.

    Methods
    -------
    assemble_sparse_matrices()
        Assemble the sparse matrices for SciPy solvers.
    estimate_timestep()
        Estimate the maximum stable timestep based on the spectral radius.
    generate_eigenfunctions()
        Generate eigenfunctions for the Rayleigh Quotient method.
    generate_norm_coords()
        Generate the normalized mesh coordinates w.r.t. the hypershape centroid.
    solve_eigenproblem()
        Solve the eigenvalue problem with Neumann boundary conditions.
    solver_rayleigh_quotient()
            Solve the eigenvalue problem using the Rayleigh Quotient method.
    solver_with_sparse_matrix()
        Solve the eigenvalue problem with sparse matrices using SciPy.
    solver_with_ufl()
        Solve the eigenvalue problem using UFL forms with SLEPc.
    weak_forms()
        Generate the bilinear forms for the modal problem.
    """

    def __init__(self, dimension=2, method=None, calc_max_dt=False, comm=None):
        """Initialize the Modal_Solver class.

        Parameters
        ----------
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D.
        method : `str`, optional
            Method to use for solving the eigenvalue problem.
            Default is None, which uses the 'KRYLOVSCH_CH' method.
            Opts: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS', 'LOBPCG', 'KRYLOVSCH_CH',
            'KRYLOVSCH_CG', 'KRYLOVSCH_GH', 'KRYLOVSCH_GG' or 'RAYLEIGH'.
            'ANALYTICAL' method is an approximation by using homogenization techniques.
            'RAYLEIGH' method is an approximation by Rayleigh quotient.
            In 'KRYLOVSCH_(K)(P)' methods, (K) indicates the Krylov solver to
            use: 'C' for Conjugate Gradient (cg) or 'G' for Generalized Minimal
            Residual (gmres). (P) indicates the preconditioner to use: 'H' for
            Hypre (hypre) or 'G' for Geometric Algebraic Multigrid (gamg). For
            example, 'KRYLOVSCH_CH' uses cg solver with hypre preconditioner.
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
        self.dimension = value_parameter_error("dimension", dimension, [2, 3])

        # Option to estimate the maximum stable timestep
        self.calc_max_dt = calc_max_dt

        # Communicator MPI
        self.comm = comm

        # Default methods
        if method is None:
            self.method = 'KRYLOVSCH_CH'
        else:
            self.method = method

        # Valid methods for solving the eigenproblem
        valid_methods = ['ANALYTICAL', 'ARNOLDI', 'LANCZOS', 'LOBPCG']
        if not self.calc_max_dt:
            valid_methods.extend(['KRYLOVSCH_CH', 'KRYLOVSCH_CG',
                                  'KRYLOVSCH_GH', 'KRYLOVSCH_GG',
                                  'RAYLEIGH'])
        self.valid_methods = value_parameter_error('method', self.method, valid_methods)

        pprint(f"Solver Method: {self.method}", comm=self.comm)

    @staticmethod
    def weak_forms(c, V, quad_rule=None, source=False, user_load=None):
        """Generate the bilinear forms for the modal problem.

        Also, it can generate a source term in weak form.

        Parameters
        ----------
        c : `Firedrake.Function`
            Velocity model.
        V : `Firedrake.FunctionSpace`
            Function space for the modal problem.
        quad_rule : `str`, optional
            Quadrature rule to use for the integration.
            Default is None, which uses the default quadrature rule.
        source : `bool`, optional
            Option to get a source term in weak form. Default is `False`
        user_load : `Firedrake.Function`, optional
            User-defined load for the source term. Default is `None`, in
            which a small constant load is applied over the entire domain.

        Returns
        -------
        a : `Firedrake.Form`
            Weak form representing the stiffness matrix.
        m : `Firedrake.Form`
            Weak form  representing the mass matrix.
        L : `Firedrake.Form`, optional
            Weak form representing a source term. Returned only if 'source' is `True`
        """

        # Functions for the problem
        u, v = fire.TrialFunction(V), fire.TestFunction(V)
        dx = fire.dx(**quad_rule) if quad_rule else fire.dx

        # Bilinear forms
        a = c * c * fire.inner(fire.grad(u), fire.grad(v)) * dx

        if source:  # Source term
            if user_load is None:
                q = fire.Constant(1.e-3)
            else:
                q = user_load

            L = q * v * dx
            return a, L

        m = fire.inner(u, v) * dx

        return a, m

    @staticmethod
    def assemble_sparse_matrices(a, m, return_M_inv=False):
        """Assemble the sparse matrices for SciPy solvers.

        Parameters
        ----------
        a : `Firedrake.Form`
            Weak form representing the stiffness matrix.
        m : `Firedrake.Form`
            Weak form  representing the mass matrix.
        return_M_inv : `bool`, optional
            Option to return the inverse mass matrix instead of the mass.

        Returns
        -------
        Asp : `csr matrix`
            Sparse matrix representing the stiffness matrix.
        Msp : `csr matrix`
            Sparse matrix representing the mass matrix.
        Msp_inv : `csr matrix`, optional
            Sparse matrix representing the inverse mass matrix.
            Returned only if 'return_M_inv' is `True`
        """

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

    def solver_with_sparse_matrix(self, Asp, Msp, method, k=2, inv_oper=False):
        """Solve the eigenvalue problem with sparse matrices using Scipy.

        Parameters
        ----------
        Asp : `csr matrix`
            Sparse matrix representing the stiffness matrix
        Msp : `csr matrix`
            Sparse matrix representing the mass matrix
        method : `str`
            Method to use for solving the eigenvalue problem.
            Opts: 'ARNOLDI', 'LANCZOS' or 'LOBPCG'
        k : `int`, optional
            Number of eigenvalues to compute. Default is 2
        inv_oper : `bool`, optional
            Option to use an inverse operator for improving convergence.
            Default is `False`.

        Returns
        -------
        Lsp : `array`
            Array containing the computed eigenvalues.
        """

        if method == 'ARNOLDI' or method == 'LANCZOS':
            # Inverse operator for improving convergence
            M_ilu = ss.linalg.spilu(Msp) if inv_oper else None
            Minv = M_ilu.solve if inv_oper else None
            A_ilu = ss.linalg.spilu(Asp) if inv_oper else None
            OPinv = A_ilu.solve if inv_oper else None

        if method == 'ARNOLDI':
            # Solve the eigenproblem using ARNOLDI (ARPACK)
            if self.calc_max_dt:
                Lsp = ss.linalg.eigs(Asp, k=k, M=Msp, which='LM', Minv=Minv,
                                     OPinv=OPinv, return_eigenvectors=False)
            else:
                Lsp = ss.linalg.eigs(Asp, k=k, M=Msp, sigma=0.0, Minv=Minv,
                                     OPinv=OPinv, return_eigenvectors=False)

        if method == 'LANCZOS':
            # Solve the eigenproblem using LANCZOS (ARPACK)
            if self.calc_max_dt:
                Lsp = ss.linalg.eigsh(Asp, k=k, M=Msp, which='LM', Minv=Minv,
                                      OPinv=OPinv, return_eigenvectors=False)
            else:
                Lsp = ss.linalg.eigsh(Asp, k=k, M=Msp, sigma=0.0, Minv=Minv,
                                      OPinv=OPinv, return_eigenvectors=False)

        if method == 'LOBPCG':
            # Initialize LI vectors for LOBPCG
            X = np.eye(Msp.shape[0], k)

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

        eigenproblem = fire.LinearEigenproblem(a, M=m)
        eigensolver = fire.LinearEigensolver(eigenproblem, n_evals=k,
                                             solver_parameters=opts)
        eigensolver.solve()
        Lsp = np.asarray([eigensolver.eigenvalue(mod) for mod in range(k)])

        return Lsp

    def generate_norm_coords(self, mesh, domain_dim, hyp_axes):
        """Generate the normalized mesh coordinates w.r.t. the hypershape centroid.

        Parameters
        ----------
        mesh : `firedrake mesh`
            Mesh for the modal problem
        domain_dim : `tuple`
            Original domain dimensions (Lx, Lz) for 2D or (Lx, Lz, Ly) for 3D
        hyp_axes : `tuple`
            Semi-axes of the hyperellipse (a, b) or hyperellipsoid (a, b, c)

        Returns
        -------
        coord_norm : `tuple`
            Normalized coordinates w.r.t. the hypershape centroid.
            Structure: (xn, zn) for 2D and (xn, zn, yn) for 3D
        """

        # Original domain dimensions
        Lx, Lz = domain_dim[:2]

        # Hypershape semi-axes
        a, b = hyp_axes[:2]

        # Mesh coordinates
        coord = fire.SpatialCoordinate(mesh)
        x, z = coord[0], coord[1]

        # Normalized coordinates w.r.t. the hypershape centroid
        x_e = (x - fire.Constant(Lx / 2.)) / fire.Constant(2. * a)
        z_e = (z + fire.Constant(Lz / 2.)) / fire.Constant(2. * b)
        coord_norm = (x_e, z_e)
        if self.dimension == 3:  # 3D
            Ly, c, y = domain_dim[2], hyp_axes[2], coord[2]
            y_e = (y - fire.Constant(Ly / 2.)) / fire.Constant(2. * c)
            coord_norm += (y_e,)

        return coord_norm

    def generate_eigenfunctions(self, coord_norm, V, k=2, bc="Neumann"):
        """Generate eigenfunctions for the Rayleigh Quotient method.

        Parameters
        ----------
        coord_norm : `tuple`
            Normalized coordinates w.r.t. the hypershape centroid.
            Structure: (xn, zn) for 2D and (xn, zn, yn) for 3D
        V : `Firedrake.FunctionSpace`
            Function space for the modal problem
        k : `int`, optional
            Number of eigenvalues to compute. Default is 2
        bc : `str`, optional
            Boundary condition type: "Dirichlet" or "Neumann".
            Default is "Neumann"

        Returns
        -------
        eig_funcs : `list`
            Eigenfunctions computed as Firedrake functions
        grad_eig : `list`
            Eigenfunction gradients computed as Firedrake functions
        """

        # Number of eigenfunctions to use
        n_eigfunc = max(2 * k, 2)

        # Mesh normalized coordinates w.r.t. the hypershape centroid
        xn, zn = coord_norm[:2]

        # Precompute cosine values for efficiency
        if bc == "Neumann":
            fi_lst = [fire.cos(i * fire.pi * xn) for i in range(n_eigfunc)]
            fj_lst = [fire.cos(j * fire.pi * zn) for j in range(n_eigfunc)]

        if bc == "Dirichlet":
            fi_lst = [fire.sin(i * fire.pi * xn) for i in range(n_eigfunc)]
            fj_lst = [fire.sin(j * fire.pi * zn) for j in range(n_eigfunc)]

        if self.dimension == 3:  # 3D
            yn = coord_norm[2]
            if bc == "Neumann":
                fk_lst = [fire.cos(k * fire.pi * yn) for k in range(n_eigfunc)]
            if bc == "Dirichlet":
                fk_lst = [fire.sin(k * fire.pi * yn) for k in range(n_eigfunc)]

        # Create eigenfunctions
        eig_funcs = []
        grad_eig = []
        for i in range(n_eigfunc):
            fi = fi_lst[i]
            for j in range(n_eigfunc):
                fj = fj_lst[j]

                if self.dimension == 2:  # 2D
                    # Eigenfunction: cos/sin(iπx/Lx) * cos/sin(jπz/Lz)
                    u_eig = fire.Function(V).interpolate(fi * fj)
                    eig_funcs.append(u_eig)
                    grad_eig.append(fire.grad(u_eig))

                if self.dimension == 3:  # 3D
                    for k in range(n_eigfunc):
                        fk = fk_lst[k]
                        u_eig = fire.Function(V).interpolate(fi * fj * fk)
                        eig_funcs.append(u_eig)
                        grad_eig.append(fire.grad(u_eig))

        return eig_funcs, grad_eig

    def solver_rayleigh_quotient(self, c, coord_norm, V, k=2, quad_rule=None):
        """Solve the eigenvalue problem using the Rayleigh Quotient method.

        Parameters
        ----------
        c : `Firedrake.Function` or `float`
            Velocity model
        coord_norm : `tuple`
            Normalized coordinates w.r.t. the hypershape centroid.
            Structure: (xn, zn) for 2D and (xn, zn, yn) for 3D
        V : `Firedrake.FunctionSpace`
            Function space for the modal problem
        k : `int`, optional
            Number of eigenvalues to compute. Default is 2
        quad_rule : `str`, optional
            Quadrature rule to use for the integration.
            Default is None, which uses the default quadrature rule.

        Returns
        -------
        Lsp : `array`
            Array containing the computed eigenvalues
        """

        # Create eigenfunctions
        eig_funcs, grad_eig = self.generate_eigenfunctions(coord_norm, V, k=k)

        # Initialize matrices for generalized eigenvalue problem
        n_funcs = len(eig_funcs)
        Asp = ss.lil_matrix((n_funcs, n_funcs))  # Stiffness matrix
        Msp = ss.lil_matrix((n_funcs, n_funcs))  # Mass matrix

        # Assemble stiffness and mass matrices
        dx = fire.dx(**quad_rule) if quad_rule else fire.dx
        for i in range(n_funcs):
            for j in range(i, n_funcs):  # Only upper triangle
                # Stiffness and mass matrix term
                A_term = fire.assemble(c * c * fire.inner(grad_eig[i],
                                                          grad_eig[j]) * dx)
                M_term = fire.assemble(fire.inner(eig_funcs[i],
                                                  eig_funcs[j]) * dx)

                # Set symmetric entries
                Asp[i, j] = A_term
                Asp[j, i] = A_term
                Msp[i, j] = M_term
                Msp[j, i] = M_term

        # Convert to CSR format for eigenvalue solver
        Asp = Asp.tocsr()
        Msp = Msp.tocsr()

        # Solve the generalized eigenvalue problem
        Lsp = self.solver_with_sparse_matrix(Asp, Msp, 'ARNOLDI', k=k)

        return Lsp

    def solve_eigenproblem(self, c, V=None, k=2, shift=0.,
                           quad_rule=None, inv_oper=False,
                           coord_norm=None, hyp_par=None,
                           cut_plane_percent=1., c_eqref=None,
                           fitting_c=(0., 0., 0., 0.),
                           static_load_for_ceq=None):
        """Solve the eigenvalue problem with Neumann boundary conditions.

        Parameters
        ----------
        c : `Firedrake.Function` or `float`
            Velocity model
        V : `Firedrake.FunctionSpace`, optional
            Function space for the modal problem. Default is None
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
        coord_norm : `tuple`
            Normalized coordinates w.r.t. the hypershape centroid.
            Structure: (xn, zn) for 2D and (xn, zn, yn) for 3D
        hyp_par : `tuple`, optional
            Hyperellipshape parameters. Default is None
            Structure 2D: (n_hyp, a_hyp, b_hyp)
            Structure 3D: (n_hyp, a_hyp, b_hyp, c_hyp)
            - n_hyp : `float`
                Degree of the hypershape
           - a_hyp : `float`
                Hypershape semi-axis in direction x
            - b_hyp : `float`
                Hypershape semi-axis in direction z
            - c_hyp : `float`
                Hypershape semi-axis in direction y (3D only)
        cut_plane_percent : `float`, optional
            Percentage of the cut plane (0 to 1). Default is 1 (no cut)
        c_eqref : `float`, optional
            Reference value for the equivalent velocity based on the original
            velocity model without an absorbing layer. Default is None
        fitting_c : `tuple`, optional
            Parameters for fitting equivalent velocity regression.
            Structure: (fc1, fc2, fp1, fp2). Default is (0., 0., 0., 0.)
            - fc1 : `float`
                Exponent factor for the minimum reference velocity..
            - fc2 : `float`
                Exponent factor for the maximum reference velocity..
            - fp1 : `float`
                Exponent factor for the minimum equivalent velocity..
            - fp2 : `float`
                Exponent factor for the maximum equivalent velocity..
        static_load_for_ceq : `Firedrake.Function`, optional
            Static load for the energy-equivalent homogenization.
            Only used if 'typ_homog'='energy'. Default is None, in which
            a small constant load is applied over the entire domain.

        Returns
        -------
        Lsp : `array` or `float`
            Array containing the computed eigenvalues or the first
            eigenvalue of the hypershape with Neumann or Dirichlet BCs
        """

        if self.method in ['ANALYTICAL', 'RAYLEIGH']:
            shift = 0.  # No shift for analytical and Rayleigh methods

        if self.method == 'ANALYTICAL':

            # Compute equivalent velocity for the hypershape
            c_eq = self.c_equivalent(c, V, quad_rule=quad_rule,
                                     static_load_for_ceq=static_load_for_ceq)

            Lsp = self.solver_analytical(c_eq, hyp_par,
                                         c_eqref=c_eqref, fitting_c=fitting_c,
                                         cut_plane_percent=cut_plane_percent)

        elif self.method == 'RAYLEIGH':
            Lsp = self.solver_rayleigh_quotient(c, coord_norm, V, k=k,
                                                quad_rule=quad_rule)
        else:
            # Get bilinear forms
            a, m = self.weak_forms(c, V, quad_rule=quad_rule)

            # Add shift to stabilize Neumann BC null space
            if shift > 0:
                a += fire.Constant(shift) * m

        if self.method.startswith('KRYLOVSCH'):
            Lsp = self.solver_with_ufl(a, m, k=k)

        elif self.method in ['ARNOLDI', 'LANCZOS', 'LOBPCG']:
            Asp, Msp = self.assemble_sparse_matrices(a, m)
            Lsp = self.solver_with_sparse_matrix(Asp, Msp, self.method,
                                                 k=k, inv_oper=inv_oper)

        Lsp -= shift if shift > 0. else 0.

        return Lsp

    def estimate_timestep(self, c, V, final_time, shift=0., quad_rule=None,
                          inv_oper=False, fraction=0.7):
        """Estimate the maximum stable timestep based on the spectral radius.

        Optionally uses the Gershgorin Circle Theorem to estimate the
        maximum generalized eigenvalue when `method` is 'ANALYTICAL'.
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
        quad_rule : `str`, optional
            Quadrature rule to use for the integration.
            Default is None, which uses the default quadrature rule.
        inv_oper : `bool`, optional
            Option to use an inverse operator for improving convergence.
            Default is False
        fraction : `float`, optional
            Fraction of the estimated timestep to use. Defaults to 0.7.

        Returns
        -------
        max_dt : `float`
            Estimated maximum stable timestep.
        """

        # Maximum eigenvalue
        if self.method == 'ANALYTICAL':
            pprint("Estimating Maximum Eigenvalue", comm=self.comm)

            a, m = self.weak_forms(c, V, quad_rule=quad_rule)

            if shift > 0:
                a += fire.Constant(shift) * m

            Asp, Msp_inv = \
                self.assemble_sparse_matrices(a, m, return_M_inv=True)
            Lsp = Msp_inv.multiply(Asp)
            max_eigval = np.amax(np.abs(Lsp.diagonal())) - shift

        else:
            pprint("Computing Exact Maximum Eigenvalue", comm=self.comm)
            Lsp = self.solve_eigenproblem(
                c, V=V, shift=shift, quad_rule=quad_rule, inv_oper=inv_oper)
            # (eig = 0 is a rigid body motion)
            max_eigval = max(np.unique(Lsp[(Lsp > 0.) & (np.imag(Lsp) == 0.)]))

        # Maximum stable timestep
        max_dt = float(np.real(2. / np.sqrt(max_eigval)))
        pprint("Maximum Stable Timestep Should Be Approximately "
               f"(ms): {1e3 * max_dt:.3f}", comm=self.comm)

        max_dt *= fraction
        nt = int(final_time / max_dt) + 1
        max_dt = final_time / (nt - 1)

        return max_dt
