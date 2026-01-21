import firedrake as fire
import numpy as np
import scipy.linalg as sl
import scipy.sparse as ss
from scipy.optimize import broyden1, curve_fit
from scipy.special import beta, betainc, gamma, jn_zeros, \
    jnp_zeros, mathieu_modcem1, spherical_jn
from scipy.stats import norm as sn
from sys import float_info
from spyro.utils.error_management import value_parameter_error
from spyro.utils.stats_tools import coeff_of_determination

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
        Default is None, which uses the 'KRYLOVSCH_CH' method
    valid_methods: `list`
        List of valid methods for solving the eigenproblem
        Options: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS', 'LOBPCG', 'KRYLOVSCH_CH',
        'KRYLOVSCH_CG', 'KRYLOVSCH_GH', 'KRYLOVSCH_GG' or 'RAYLEIGH'.
        'ANALYTICAL' method is only available for isotropic hypershapes.
        'RAYLEIGH' method is an approximation by Rayleigh quotient.
        In 'KRYLOVSCH_(K)(P)' methods, (K) indicates the Krylov solver to
        use: 'C' for Conjugate Gradient (cg) or 'G' for Generalized Minimal
        Residual (gmres). (P) indicates the preconditioner to use: 'H' for
        Hypre (hypre) or 'G' for Geometric Algebraic Multigrid (gamg). For
        example, 'KRYLOVSCH_CH' uses cg solver with hypre preconditioner

    Methods
    -------
    assemble_sparse_matrices()
        Assemble the sparse matrices for SciPy solvers
    c_equivalent()
        Compute an equivalent homogeneous velocity for an inhomogeneneous
        velocity model using an energy-equivalent homogenization
    estimate_timestep()
        Estimate the maximum stable timestep based on the spectral
        radius using optionally the Gershgorin Circle Theorem to
        estimate the maximum generalized eigenvalue. Otherwise
        computes the maximum generalized eigenvalue exactly
    freq_factor_ell()
        Compute the frequency factor for elliptical or ellipsoidal geometries
    freq_factor_hyp()
        Compute an approximate frequency factor for the hypershape with
            truncation plane at z = cut_plane_percent * b, b = Lz + pad_len
    freq_factor_rec()
        Compute the frequency factor for rectangular or prismatic geometries
    generate_eigenfunctions()
            Generate eigenfunctions for the Rayleigh Quotient method
    generate_norm_coords()
        Generate the normalized mesh coordinates w.r.t. the hypershape centroid
    reg_geometry_hyp()
        Perform the nonlinear regression for the hypershape geometry factor
    solve_eigenproblem()
        Solve the eigenvalue problem with Neumann boundary conditions
    solver_analytical()
        Compute the analytical solution for the eigenvalue problem with
        Neumann or Dirichlet boundary conditions for isotropic hypershapes
    solver_rayleigh_quotient()
            Solve the eigenvalue problem using the Rayleigh Quotient method
    solver_with_sparse_matrix()
        Solve the eigenvalue problem with sparse matrices using SciPy
    solver_with_ufl()
        Solve the eigenvalue problem using UFL forms with SLEPc
    weak_forms()
        Generate the bilinear forms for the modal problem.
        Also, it can generate a source term in weak form
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
            Default is None, which uses the 'KRYLOVSCH_CH' method.
            Opts: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS', 'LOBPCG', 'KRYLOVSCH_CH',
            'KRYLOVSCH_CG', 'KRYLOVSCH_GH', 'KRYLOVSCH_GG' or 'RAYLEIGH'.
            'ANALYTICAL' method is only available for isotropic hypershapes.
            'RAYLEIGH' method is an approximation by Rayleigh quotient.
            In 'KRYLOVSCH_(K)(P)' methods, (K) indicates the Krylov solver to
            use: 'C' for Conjugate Gradient (cg) or 'G' for Generalized Minimal
            Residual (gmres). (P) indicates the preconditioner to use: 'H' for
            Hypre (hypre) or 'G' for Geometric Algebraic Multigrid (gamg). For
            example, 'KRYLOVSCH_CH' uses cg solver with hypre preconditioner
        calc_max_dt : `bool`
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

        # Default methods
        if method is None:
            self.method = 'KRYLOVSCH_CH'
        else:
            self.method = method

        # Valid methods for solving the eigenproblem
        self.valid_methods = ['ANALYTICAL', 'ARNOLDI', 'LANCZOS', 'LOBPCG']
        if not self.calc_max_dt:
            self.valid_methods.extend(['KRYLOVSCH_CH', 'KRYLOVSCH_CG',
                                       'KRYLOVSCH_GH', 'KRYLOVSCH_GG',
                                       'RAYLEIGH'])

        if self.method not in self.valid_methods:
            value_parameter_error('method', method, self.valid_methods)

        print(f"Solver Method: {self.method}", flush=True)

    @staticmethod
    def weak_forms(c, V, quad_rule=None, source=False):
        '''
        Generate the bilinear forms for the modal problem.
        Also, it can generate a source term in weak form

        Parameters
        ----------
        c : `firedrake function`
            Velocity model
        V : `firedrake function space`
            Function space for the modal problem
        quad_rule : `str`, optional
            Quadrature rule to use for the integration.
            Default is None, which uses the default quadrature rule
        source : `bool`, optional
            Option to get a source term in weak form. Default is False

        Returns
        -------
        a : `firedrake form`
            Weak form representing the stiffness matrix
        m : `firedrake form`
            Weak form  representing the mass matrix
        L : `firedrake form`, optional
            Weak form representing a source term.
            Returned only if 'source'=True
        '''

        # Functions for the problem
        u, v = fire.TrialFunction(V), fire.TestFunction(V)
        dx = fire.dx(scheme=quad_rule) if quad_rule else fire.dx

        # Bilinear forms
        a = c * c * fire.inner(fire.grad(u), fire.grad(v)) * dx

        if source:
            L = fire.Constant(1.0) * v * dx  # Source term
            return a, L

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
        Msp_inv : `csr matrix`, optional
            Sparse matrix representing the inverse mass matrix.
            Returned only if 'return_M_inv'=True
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

    def solver_with_sparse_matrix(self, Asp, Msp, method, k=2, inv_oper=False):
        '''
        Solve the eigenvalue problem with sparse matrices using Scipy

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
            Default is False

        Returns
        -------
        Lsp : `array`
            Array containing the computed eigenvalues
        '''

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
            # Initialize random vectors for LOBPCG
            # X = sl.orth(np.random.rand(Msp.shape[0], k))
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
        nconv = eigensolver.solve()
        Lsp = np.asarray([eigensolver.eigenvalue(mod) for mod in range(k)])

        return Lsp

    def c_equivalent(self, c, V, quad_rule=None, typ_homog='energy'):
        '''
        Compute an equivalent homogeneous velocity for an inhomogeneneous
        velocity model using an energy-equivalent homogenization

        Parameters
        ----------
        c : `firedrake function`
            Velocity model
        V : `firedrake function space`
            Function space for the modal problem
        quad_rule : `str`, optional
            Quadrature rule to use for the integration.
            Default is None, which uses the default quadrature rule
        typ_homog : `str`, optional
            Type of homogenization: 'energy' or 'volume'. Default is 'energy'

        Returns
        -------
        c_eq : `float`
            Equivalent homogeneous velocity
        '''

        # Integration measure
        dx = fire.dx(scheme=quad_rule) if quad_rule else fire.dx

        if typ_homog == 'energy':
            # Equivalent velocity by energy-equivalent homogenization

            # Weak forms
            a, L = self.weak_forms(c, V, quad_rule=quad_rule, source=True)

            # Compute the energy
            u = fire.Function(V)
            fire.solve(a == L, u)
            energy = fire.assemble(
                fire.Constant(0.5) * c * c * fire.inner(fire.grad(u),
                                                        fire.grad(u)) * dx)

            # Compute the equivalent velocity
            c_eq = np.sqrt(energy / fire.assemble(
                fire.Constant(0.5) * fire.inner(fire.grad(u),
                                                fire.grad(u)) * dx))

        elif typ_homog == 'volume':
            # Equivalent velocity by volume-average homogenization

            # Compute the volume
            volume = fire.assemble(fire.Constant(1.) * (c / c) * dx)

            # Compute the equivalent velocity
            c_eq = fire.assemble(c * dx) / volume

        return c_eq

    @staticmethod
    def freq_factor_rec(hyper_axes, bc='Neumann'):
        '''
        Compute the frequency factor for rectangular or prismatic geometries
        - Rectangular layer:
            https://www.sc.ehu.es/sbweb/fisica3/ondas/membrana_1/membrana_1.html

        Parameters
        ----------
        hyper_axes : `tuple`
            Semi-axes of the hyperellipse [a, b] or hyperellipsoid [a, b, c]
        bc : `str`, optional
            Boundary condition type: 'Dirichlet' or 'Neumann'.
            Default is 'Neumann'

        Returns
        -------
        f_rec : `float`
            Fundamental frequency factor for rectangular or prismatic geometry
        '''

        if bc == 'Neumann':
            f_rec = 1. / max(hyper_axes)
        elif bc == 'Dirichlet':
            f_rec = sum(1. / np.asarray(hyper_axes)**2)**0.5
        else:
            value_parameter_error('bc', bc, ['Dirichlet', 'Neumann'])

        f_rec *= np.pi / 2.

        return f_rec

    def freq_factor_ell(self, hyper_axes, bc='Neumann', all_axes_equal=False):
        '''
        Compute the frequency factor for elliptical or ellipsoidal geometries
        - Elliptical layer:
            https://docs.scipy.org/doc/scipy/reference/generated/
            scipy.special.mathieu_modcem1.html#scipy.special.mathieu_modcem1
        - Circular layer:
            https://en.wikipedia.org/wiki/Vibrations_of_a_circular_membrane

        Parameters
        ----------
        hyper_axes : `tuple`
            Semi-axes of the hyperellipse [a, b] or hyperellipsoid [a, b, c]
        bc : `str`, optional
            Boundary condition type: 'Dirichlet' or 'Neumann'.
            Default is 'Neumann'
        all_axes_equal : `bool`, optional
            Option for circular or spherical case. Default is False

        Returns
        -------
        f_ell : `float`
            Fundamental frequency factor for elliptical or ellipsoidal geometry
        '''

        if bc not in ['Dirichlet', 'Neumann']:
            value_parameter_error('bc', bc, ['Dirichlet', 'Neumann'])

        def MMF(q):
            '''
            Compute the Modified Mathieiu's Function (MMF) or its derivative

            Parameters
            ----------
            q : `float`
                Argument of the MMF.
                q = M01 is the 1st root for the 0th-order function

            Returns
            -------
            mathieu_modcem : `float`
                Value of the MMF or its derivative at the value q
                mathieu_modcem1(m, q, psi0)[0] for the MMF and
                mathieu_modcem1(m, q, psi0)[1] for its derivative

            Examples
            -------
            mathieu_modcem1(m=0, q=2.6750449521966490,
                psi0=np.arccosh(2/(3)**0.5))[0]= 5.363046165143026e-17
            mathieu_modcem1(m=0, q=1.6748563428285737,
                psi0=0.7061880927645094)[0] = 4.036310483679603e-16
            '''

            # Eccentricity parameter: psi0 = arccosh(a/f), f = sqrt(a^2 - b^2)
            psi0 = np.arccosh(a0 / f0)
            idx = int(bc == 'Neumann')
            m = 1 if bc == 'Neumann' else 0  # Order of the MMF
            print(bc, m, psi0, q, mathieu_modcem1(m, q, psi0)[idx], flush=True)
            return mathieu_modcem1(m, q, psi0)[idx]

        def ZBF(m=0, n=1):
            '''
            Compute zeros of the Bessel's Function (BF) or its derivative

            Parameters
            ----------
            m : `int`, optional
                Order of the BF. Default is 0
            n : `int`, optional
                Number of roots to compute. Default is 1

            Returns
            -------
            Jmz : `array`
                First n zeros of the Bessel function or its derivative
            '''
            deriv = (bc == 'Neumann')
            Jmz = jnp_zeros(m, n) if deriv else jn_zeros(m, n)

            return Jmz

        def SBF(q, m=0):
            '''
            Compute the Spherical Bessel's Function (SBF) or its derivative

            Parameters
            ----------
            q : `float`
                Argument of the SBF.
                q = J01 is the 1st root for the 0th-order function

            Returns
            -------
            spherical_jn : `float`
                Value of the SBF or its derivative at the value q
                spherical_jn(m, q, derivative=False) for the SBF and
                spherical_jn(m, q, derivative=True) for its derivative
            '''
            deriv = (bc == 'Neumann')
            m = 1 if bc == 'Neumann' else 0  # Order of the SBF
            # print(bc, m, q, spherical_jn(m, q, derivative=deriv))
            return spherical_jn(m, q, derivative=deriv)

        # Semi-axes
        a, b = hyper_axes[:2]

        # Frequency factor for rectangular/prismatic case
        f_rec = self.freq_factor_rec(hyper_axes, bc=bc)

        # Initial guess
        igss = f_rec if bc == 'Neumann' else 0.

        # Circular or spherical case
        if all_axes_equal:
            # 1st root for the mth-order Bessel's function
            if self.dimension == 2:  # 2D circular
                m = 1 if bc == 'Neumann' else 0
                J01 = ZBF(m=m, n=1)[0]

            if self.dimension == 3:  # 3D spherical
                J01 = float(broyden1(SBF, igss, f_tol=1e-14))

            return J01 / a

        # Elliptical or ellipsoidal case
        if self.dimension == 2:  # 2D elliptical

            # Order semi-axes
            a, b = sorted(hyper_axes, reverse=True)

            # Ellipse eccentricity
            f0 = (a**2 - b**2)**0.5

            # 1st root or the mth-order Modified Mathieu's Function
            a0 = a
            M01 = float(broyden1(MMF, igss, f_tol=1e-14))

            return (2 / f0) * M01**0.5

        if self.dimension == 3:  # 3D ellipsoidal

            f_ell_arr = []

            # Order semi-axes
            a, b, c = sorted(hyper_axes, reverse=True)

            # Eccentricities for each pair of semi-axes
            ecc_arr = [(a, b, (a**2 - b**2)**0.5 if a > b else 0.),
                       (b, c, (b**2 - c**2)**0.5 if b > c else 0.),
                       (a, c, (a**2 - c**2)**0.5 if a > c else 0.)]

            if bc == 'Neumann':
                # Only use the pair with maximum eccentricity
                max_ecc_idx = np.argmax([ecc for _, _, ecc in ecc_arr])
                ecc_arr = [ecc_arr[max_ecc_idx]]

            for a0, b0, f0 in ecc_arr:

                if f0 == 0:  # Circular cross-section
                    # 1st root for the mth-order Bessel's function
                    J01 = ZBF(m=0, n=1)[0]
                    f_ell_arr.append((J01 / a0)**2)

                else:  # Elliptical cross-section
                    # 1st root or the mth-order Modified Mathieu's Function
                    M01 = float(broyden1(MMF, igss, f_tol=1e-14))
                    f_ell_arr.append(4 * M01 / f0**2)

            # Sum and return square root
            return sum(f_ell_arr)**0.5

    def reg_geometry_hyp(self, n_fix, cut_plane_percent=1.):
        '''
        Perform the nonlinear regression for the hypershape geometry factor

        Parameters
        ----------
        n_fix : `float`
            Fixed hypershape degree for the regression
        cut_plane_percent : `float`, optional
            Percentage of the cut plane (0 to 1). Default is 1 (no cut)

        Returns
        -------
        pn_fit : `float`
            Fitted parameter pn
        qn_fit : `float`
            Fitted parameter qn
        fr_ell : `float`
            Ratio between the area or volume of the
            truncated and full ellipse or ellipsoidal
        fr_rec : `float`
            Ratio between the area or volume of the
            truncated and full rectangle or prism
        '''

        # Verify cutting plane measurement is between 0 and 1
        cut_plane_percent = np.clip(cut_plane_percent, 0., 1.)

        def area_function(n, cut_plane_percent):
            '''
            Area function for hiperellipses
            '''
            fA = 2. * gamma(1 + 1 / n)**2 / gamma(1 + 2 / n)
            if cut_plane_percent == 1.:
                fA *= 2.
            else:
                eps = float_info.min
                w = np.maximum(cut_plane_percent ** n, eps)  # w <= 1
                p = 1 / n
                q = 1 + 1 / n
                B_w = beta(p, q) * betainc(p, q, w)  # Non-regularized Beta
                fA += (2. / n) * B_w

            return fA

        def volume_function(n, cut_plane_percent):
            '''
            Volume function for hiperellipsoids
            '''
            fV = 4. * gamma(1 + 1 / n)**3 / gamma(1 + 3 / n)
            if cut_plane_percent == 1.:
                fV *= 2.
            else:
                eps = float_info.min
                w = np.maximum(cut_plane_percent ** n, eps)  # w <= 1
                p = 1 / n
                q = 1 + 1 / n
                A_f = gamma(1 + p)**2 / gamma(q)
                B_w = beta(p, q) * betainc(p, q, w)  # Non-regularized Beta
                fV += (4. / n) * A_f * B_w

            return fV

        def fit_function(n, pn, qn):
            '''
            Define the fit function for the area or volume regression.
            Fit function:
                A or V = f_max - cn2 * (1 / (qn * n + 1 - 2 * qn)) ** pn
                cn2 = f_max - fn2
            '''

            # Constant for power-law fit
            cn2 = f_max - fn2

            return f_max - cn2 * (1. / (qn * n + 1. - 2. * qn)) ** pn

        # Regression dataset
        n_data = np.arange(2., 100., 0.1)

        # Pre-compute constants
        fax_trunc = cut_plane_percent + 1.
        if self.dimension == 2:  # 2D
            f_max = 2. * fax_trunc
            fn2 = area_function(2., cut_plane_percent)
            fr_ell = fn2 / area_function(2., 1.)
            fr_rec = area_function(100., cut_plane_percent
                                   ) / area_function(100., 1.)
            f_data = area_function(n_data, cut_plane_percent)

        if self.dimension == 3:  # 3D
            f_max = 4. * fax_trunc
            fn2 = volume_function(2., cut_plane_percent)
            fr_ell = fn2 / volume_function(2., 1.)
            fr_rec = volume_function(100., cut_plane_percent
                                     ) / area_function(100., 1.)
            f_data = volume_function(n_data, cut_plane_percent)

        # Initial guess
        init_guess = np.array([1/3, 1/3])

        # Parameter bounds pn >= 0, qn >= 0
        low_bnds = [0, 0]
        upp_bnds = [np.inf, np.inf]

        # Maximum number of iterations
        it_max = 10000

        try:
            # Perform nonlinear curve fitting
            popt, pcov = curve_fit(fit_function, n_data, f_data, p0=init_guess,
                                   bounds=(low_bnds, upp_bnds), maxfev=it_max)
            pn_fit, qn_fit = popt

            # Calculate goodness of fit metrics (RMSE: Root-Mean-Square error)
            # fit_function has only 1 predictor (x), so p=1 for R²
            f_pred = fit_function(n_data, pn_fit, qn_fit)
            residuals = f_data - f_pred
            r_squared = coeff_of_determination(f_data, f_pred, 1)
            rmse = np.sqrt(np.mean(residuals**2))

            # Calculate confidence intervals
            perr = np.sqrt(np.diag(pcov))
            delta_pn = pn_fit - sn.interval(0.95, loc=pn_fit, scale=perr[0])[0]
            delta_qn = qn_fit - sn.interval(0.95, loc=qn_fit, scale=perr[1])[0]

            print(f"Nonlinear Curve Fit Successful!", flush=True)
            print(f"Fitted Parameters: pn = {pn_fit:.6f} ± {delta_pn:.6f}, "
                  f"qn = {qn_fit:.6f} ± {delta_qn:.6f}", flush=True)
            print(f"R-Squared: {r_squared:.6f} - RMSE: {rmse:.6f}", flush=True)

            return pn_fit, qn_fit, fr_ell, fr_rec

        except fire.ConvergenceError as e:
            print(f"Nonlinear Curve Fit Failed: {e}", flush=True)

    def freq_factor_hyp(self, n_hyp, f_rec, f_ell, c_eq, bc='Neumann',
                        c_eqref=None, fitting_c=(1., 1., 0.5, 0.5),
                        cut_plane_percent=1.):
        '''
        Compute an approximate frequency factor for the hypershape with
        truncation plane at z = cut_plane_percent * b, b = Lz + pad_len.
        The fitting parameters for the equivalent velocity regression controls:
        - fc1: Magnitude order of the frequency
        - fc2: Monotonicity of the frequency
        - fp1: Rectangular domain frequency
        - fp2: Ellipsoidal domain frequency


        Parameters
        ----------
        n_hyp : `float`
            Degree of the hyperellipse
        f_rec : `float`
            Fundamental frequency factor for rectangular or prismatic geometry
        f_ell : `float`
            Fundamental frequency factor for elliptical or ellipsoidal geometry
        c_eq : `float`
            Equivalente isotropic velocity in the hypershape
        bc : `str`, optional
            Boundary condition type: 'Dirichlet' or 'Neumann'.
            Default is 'Neumann'
        c_eqref : `float`, optional
            Reference value for the equivalent velocity based on the original
            velocity model without an absorbing layer. Default is None
        fitting_c : `list`, optional
            Parameters for fitting equivalent velocity regression.
            Structure: (fc1, fc2, fp1, fp2). Default is (1., 1., 0.5, 0.5)
            - fc1 : `float`
                Exponent factor for the minimum reference velocity
            - fc2 : `float`
                Exponent factor for the maximum reference velocity
            - fp1 : `float`
                Exponent fsctor for the minimum equivalent velocity
            - fp2 : `float`
                Exponent factor for the maximum equivalent velocity
        cut_plane_percent : `float`, optional
            Percentage of the cut plane (0 to 1). Default is 1 (no cut)

        Returns
        -------
        f_hyp : `float`
            Approximate frequency factor for the hypershape
        c_reg : `float` or `None`
            Approximate equivalent velocity for the hypershape
        '''

        n_hyp = 6 if n_hyp is None else n_hyp
        c_eqref = c_eq if c_eqref is None else c_eqref

        # Regression for hypershape geometry factor
        pn, qn, fr_ell, fr_rec = self.reg_geometry_hyp(
            n_hyp, cut_plane_percent=cut_plane_percent)

        if bc == 'Dirichlet':
            f_min = f_rec / fr_rec
            cn2 = f_min - f_ell / fr_ell

        if bc == 'Neumann':
            f_min = f_rec
            cn2 = f_min - f_ell

        # Hypershape frequency factor
        pot_term = (1. / (qn * n_hyp + 1 - 2 * qn)) ** pn
        f_hyp = f_min - cn2 * pot_term

        # Adjusting equivalent velocity for the hypershape
        fc1, fc2, fp1, fp2 = fitting_c
        f1 = fr_ell / fr_rec
        f2 = f_ell / f_rec
        f3 = 1. / f2
        c_ref = max(c_eq * f3 ** fc1, c_eqref * f3 ** fc2)
        c_min = c_ref * min(f1, f2) ** (fp1 * pn)
        cc2 = c_min - c_ref * max(f1, f2) ** (fp2 * pn)

        # Equivalent velocity for the hypershape
        c_reg = c_min - cc2 * pot_term

        return f_hyp, c_reg

    def solver_analytical(self, c_eq, hyp_par, bc='Neumann', c_eqref=None,
                          fitting_c=(1., 1., 0.5, 0.5), cut_plane_percent=1.):
        '''
        Compute the analytical solution for the eigenvalue problem with
        Neumann or Dirichlet boundary conditions for isotropic hypershapes

        Parameters
        ----------
        c_eq : `float`
            Equivalente isotropic velocity in the hypershape
        hyp_par : `tuple`
            Hyperellipshape parameters.
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
        bc : `str`, optional
            Boundary condition type: 'Dirichlet' or 'Neumann'.
            Default is 'Neumann'
        c_eqref : `float`, optional
            Reference value for the equivalent velocity based on the original
            velocity model without an absorbing layer. Default is None
        fitting_c : `tuple`, optional
            Parameters for fitting equivalent velocity regression.
            Structure: (fc1, fc2, fp1, fp2). Default is (1., 1., 0.5, 0.5)
            - fc1 : `float`
                Exponent factor for the minimum reference velocity
            - fc2 : `float`
                Exponent factor for the maximum reference velocity
            - fp1 : `float`
                Exponent factor for the minimum equivalent velocity
            - fp2 : `float`
                Exponent factor for the maximum equivalent velocity
        cut_plane_percent : `float`, optional
            Percentage of the cut plane (0 to 1). Default is 1 (no cut)

        Returns
        -------
        Lsp : `float`
            First eigenvalue of the hypershape with Neumann or Dirichlet BCs
        '''

        # Check if the velocity model is a float
        if not isinstance(c_eq, float):
            raise TypeError("For 'ANALYTICAL' method, the isotropic "
                            "velocity 'c_eq' must be a float number.")

        # Hyperellipse parameters
        n_hyp, hyp_axes = hyp_par[0], hyp_par[1:]
        a, b = hyp_axes[:2]
        if self.dimension == 2:  # 2D
            all_axes_equal = (a == b)

        if self.dimension == 3:  # 3D
            c = hyp_axes[2]
            all_axes_equal = (a == b == c)

        # Frequency factors
        f_rec = self.freq_factor_rec(hyp_axes, bc=bc)
        f_ell = self.freq_factor_ell(
            hyp_axes, bc=bc, all_axes_equal=all_axes_equal)
        f_hyp, c_reg = self.freq_factor_hyp(
            n_hyp, f_rec, f_ell, c_eq, bc=bc, c_eqref=c_eqref,
            fitting_c=fitting_c, cut_plane_percent=cut_plane_percent)

        print(f"Hypershape Equivalent Velocity c_eq (km/s) = {c_eq:.3f}",
              flush=True)
        print(f"Hypershape Frequency factor f_hyp (1/km): {f_hyp:.3f}",
              flush=True)

        # Eigenvalue
        Lsp = (c_reg * f_hyp)**2

        return Lsp

    def generate_norm_coords(self, mesh, dom_dim, hyp_axes):
        '''
        Generate the normalized mesh coordinates w.r.t. the hypershape centroid

        Parameters
        ----------
        mesh : `firedrake mesh`
            Mesh for the modal problem
        dom_dim : `tuple`
            Original domain dimensions (Lx, Lz) for 2D or (Lx, Lz, Ly) for 3D
        hyp_axes : `tuple`
            Semi-axes of the hyperellipse [a, b] or hyperellipsoid [a, b, c]

        Returns
        -------
        coord_norm : `tuple`
            Normalized coordinates w.r.t. the hypershape centroid.
            Structure: (xn, zn) for 2D and (xn, zn, yn) for 3D
        '''

        # Original domain dimensions
        Lx, Lz = dom_dim[:2]

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
            Ly, c, y = dom_dim[2], hyp_axes[2], coord[2]
            y_e = (y - fire.Constant(Ly / 2.)) / fire.Constant(2. * c)
            coord_norm += (y_e,)

        return coord_norm

    def generate_eigenfunctions(self, coord_norm, V, k=2, bc='Neumann'):
        '''
        Generate eigenfunctions for the Rayleigh Quotient method

        Parameters
        ----------
        coord_norm : `tuple`
            Normalized coordinates w.r.t. the hypershape centroid.
            Structure: (xn, zn) for 2D and (xn, zn, yn) for 3D
        V : `firedrake function space`
            Function space for the modal problem
        k : `int`, optional
            Number of eigenvalues to compute. Default is 2
        bc : `str`, optional
            Boundary condition type: 'Dirichlet' or 'Neumann'.
            Default is 'Neumann'

        Returns
        -------
        eig_funcs : `list`
            Eigenfunctions computed as Firedrake functions
        grad_eig : `list`
            Eigenfunction gradients computed as Firedrake functions
        '''

        # Number of eigenfunctions to use
        n_eigfunc = max(2 * k, 2)

        # Mesh normalized coordinates w.r.t. the hypershape centroid
        xn, zn = coord_norm[:2]

        # Precompute cosine values for efficiency
        if bc == 'Neumann':
            fi_lst = [fire.cos(i * fire.pi * xn) for i in range(n_eigfunc)]
            fj_lst = [fire.cos(j * fire.pi * zn) for j in range(n_eigfunc)]

        if bc == 'Dirichlet':
            fi_lst = [fire.sin(i * fire.pi * xn) for i in range(n_eigfunc)]
            fj_lst = [fire.sin(j * fire.pi * zn) for j in range(n_eigfunc)]

        if self.dimension == 3:  # 3D
            yn = coord_norm[2]
            if bc == 'Neumann':
                fk_lst = [fire.cos(k * fire.pi * yn) for k in range(n_eigfunc)]
            if bc == 'Dirichlet':
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
        '''
        Solve the eigenvalue problem using the Rayleigh Quotient method

        Parameters
        ----------
        c : `firedrake function` or `float`
            Velocity model
        coord_norm : `tuple`
            Normalized coordinates w.r.t. the hypershape centroid.
            Structure: (xn, zn) for 2D and (xn, zn, yn) for 3D
        V : `firedrake function space`
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
        '''

        # Create eigenfunctions
        eig_funcs, grad_eig = self.generate_eigenfunctions(coord_norm, V, k=k)

        # Initialize matrices for generalized eigenvalue problem
        n_funcs = len(eig_funcs)
        Asp = ss.lil_matrix((n_funcs, n_funcs))  # Stiffness matrix
        Msp = ss.lil_matrix((n_funcs, n_funcs))  # Mass matrix

        # Assemble stiffness and mass matrices
        dx = fire.dx(scheme=quad_rule) if quad_rule else fire.dx
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
                           fitting_c=(1., 1., 0.5, 0.5)):
        '''
        Solve the eigenvalue problem with Neumann boundary conditions

        Parameters
        ----------
        c : `firedrake function` or `float`
            Velocity model
        V : `firedrake function space`, optional
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
            Structure: (fc1, fc2, fp1, fp2). Default is (1., 1., 0.5, 0.5)
            - fc1 : `float`
                Exponent factor for the minimum reference velocity
            - fc2 : `float`
                Exponent factor for the maximum reference velocity
            - fp1 : `float`
                Exponent fsctor for the minimum equivalent velocity
            - fp2 : `float`
                Exponent factor for the maximum equivalent velocity

        Returns
        -------
        Lsp : `array` or `float`
            Array containing the computed eigenvalues or the first
            eigenvalue of the hypershape with Neumann or Dirichlet BCs
        '''

        if self.method == 'ANALYTICAL':
            c_eq = self.c_equivalent(c, V, quad_rule=quad_rule)
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

        if self.method[:-3] == 'KRYLOVSCH':
            Lsp = self.solver_with_ufl(a, m, k=k)

        elif not (self.method == 'ANALYTICAL' or self.method == 'RAYLEIGH'):
            Asp, Msp = self.assemble_sparse_matrices(a, m)
            Lsp = self.solver_with_sparse_matrix(Asp, Msp, self.method,
                                                 k=k, inv_oper=inv_oper)

        Lsp -= shift if shift > 0. else 0.

        return Lsp

    def estimate_timestep(self, c, V, final_time, shift=0., quad_rule=None,
                          inv_oper=False, fraction=0.7):
        '''
        Estimate the maximum stable timestep based on the spectral radius
        using optionally the Gershgorin Circle Theorem to estimate
        the maximum generalized eigenvalue ('ANALYTICAL' method).
        Otherwise computes the maximum generalized eigenvalue exactly

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
        fraction : `float`, optional
            Fraction of the estimated timestep to use. Defaults to 0.7.

        Returns
        -------
        max_dt : `float`
            Estimated maximum stable timestep
        '''

        # Maximum eigenvalue
        if self.method == 'ANALYTICAL':
            print(f"Estimating Maximum Eigenvalue", flush=True)

            a, m = self.weak_forms(c, V, quad_rule=quad_rule)

            if shift > 0:
                a += fire.Constant(shift) * m

            Asp, Msp_inv = \
                self.assemble_sparse_matrices(a, m, return_M_inv=True)
            Lsp = Msp_inv.multiply(Asp).toarray().squeeze()
            Lsp -= shift if shift > 0. else 0.
            max_eigval = np.amax(np.abs(Lsp.diagonal()))

        else:
            print(f"Computing Exact Maximum Eigenvalue", flush=True)
            Lsp = self.solve_eigenproblem(
                c, V=V, shift=shift, quad_rule=quad_rule, inv_oper=inv_oper)
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
