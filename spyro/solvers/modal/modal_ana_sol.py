from firedrake import (assemble, ConvergenceError, dx as fire_dx,
                       Function, grad, inner, solve)
from numpy import (arange, arccosh, argmax, array, asarray,
                   diag, inf, maximum, mean, pi, sqrt)
from scipy.optimize import broyden1, curve_fit
from scipy.special import (beta, betainc, gamma, jn_zeros, jnp_zeros,
                           mathieu_modcem1, spherical_jn)
from scipy.stats import norm as sn
from sys import float_info
from ...utils.error_management import (type_data_structure_error, value_numerical_error,
                                       value_parameter_error)
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


class Modal_Analytical_Solver():
    """Class for the Modal problem with Neumann or Dirichlet boundary conditions.

    Attributes
    ----------
    calc_max_dt : `bool`
        Option to estimate the maximum stable timestep for the computation of the
        transient response. Default is `False`.
    dimension : `int`
        Model dimension (2D or 3D). Default is 2D.

    Methods
    -------
    _freq_factor_ell()
        Compute the frequency factor for elliptical or ellipsoidal geometries.
    _freq_factor_hyp()
        Compute an approximate frequency factor for a full or truncated hypershape.
    _freq_factor_rec()
        Compute the frequency factor for rectangular or prismatic geometries.
    _reg_geometry_hyp()
        Perform the nonlinear regression for the hypershape geometry factor.
    c_equivalent()
        Compute equivalent homogeneous velocity for an inhomogeneous model.
    solver_analytical()
        Compute the analytical eigenvalue for hypershapes by using homogenization.
    """

    def __init__(self, dimension=2):
        """Initialize the Modal_Analytical_Solver class.

        Parameters
        ----------
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D.
        comm : `object`, optional
            An object representing the communication interface for parallel processing.
            Default is `None`.

        Returns
        -------
        None
        """

        # Dimension of the problem
        self.dimension = value_parameter_error("dimension", dimension, [2, 3])

        # Communicator MPI
        self.comm = comm

    def c_equivalent(self, c, V, quad_rule=None, type_homog="energy",
                     static_load_for_ceq=None):
        """Compute equivalent homogeneous velocity for an inhomogeneous model.

        The method uses an energy-equivalent homogenization by default.

        Parameters
        ----------
        c : `Firedrake.Function`
            Velocity model.
        V : `Firedrake.FunctionSpace`
            Function space for the modal problem.
        quad_rule : `str`, optional
            Quadrature rule to use for the integration.
            Default is `None`, which uses the default quadrature rule.
        type_homog : `str`, optional
            Type of homogenization: "energy" or "volume". Default is "energy"
        static_load_for_ceq : `Firedrake.Function`, optional
            Static load for the energy-equivalent homogenization.
            Only used if 'type_homog' is "energy". Default is `None`, in which
            a small constant load is applied over the entire domain.

        Returns
        -------
        c_eq : `float`
            Equivalent homogeneous velocity.
        """

        # Check type of homogenization
        value_parameter_error("type_homog", type_homog, ["energy", "volume"])

        # Integration measure
        dx = fire_dx(**quad_rule) if quad_rule else fire_dx

        # State variable
        u = Function(V)

        if type_homog == "energy":
            # Equivalent velocity by energy-equivalent homogenization

            # Weak forms
            a, L = self.weak_forms(c, V, quad_rule=quad_rule, source=True,
                                   user_load=static_load_for_ceq)

            # Compute the energy
            solve(a == L, u)
            energy = assemble(0.5 * c * c * inner(grad(u), grad(u)) * dx)

            # Compute the equivalent velocity
            c_eq = sqrt(energy / assemble(bilinear_term * dx))

        elif type_homog == "volume":
            # Equivalent velocity by volume-average homogenization

            # Compute the volume
            u.assign(1.)
            volume = assemble(u * dx)

            # Compute the equivalent velocity
            c_eq = assemble(c * dx) / volume

        return c_eq

    @ staticmethod
    def _freq_factor_rec(hyper_axes, bc="Neumann"):
        """Compute the frequency factor for rectangular or prismatic geometries.

        - Rectangular layer:
            https://www.sc.ehu.es/sbweb/fisica3/ondas/membrana_1/membrana_1.html

        Parameters
        ----------
        hyper_axes : `tuple`
            Semi-axes of the rectangle [a, b] or prism [a, b, c]
        bc : `str`, optional
            Boundary condition type: "Dirichlet" or "Neumann". Default is "Neumann".

        Returns
        -------
        f_rec : `float`
            Fundamental frequency factor for rectangular or prismatic geometry.
        """

        # Compute the frequency factor for rectangular or prismatic geometries
        if bc == "Neumann":
            f_rec = 1. / max(hyper_axes)
        elif bc == "Dirichlet":
            f_rec = sum(1. / asarray(hyper_axes)**2)**0.5

        f_rec *= pi / 2.

        return f_rec

    def _freq_factor_ell(self, hyper_axes, bc="Neumann", all_axes_equal=False):
        """Compute the frequency factor for elliptical or ellipsoidal geometries.

        - Elliptical layer:
            https://docs.scipy.org/doc/scipy/reference/generated/
            scipy.special.mathieu_modcem1.html#scipy.special.mathieu_modcem1
        - Circular layer:
            https://en.wikipedia.org/wiki/Vibrations_of_a_circular_membrane

        Parameters
        ----------
        hyper_axes : `tuple`
            Semi-axes of the hyperellipse [a, b] or hyperellipsoid [a, b, c].
        bc : `str`, optional
            Boundary condition type: "Dirichlet" or "Neumann".
            Default is "Neumann".
        all_axes_equal : `bool`, optional
            Option for circular or spherical case. Default is `False`.

        Returns
        -------
        f_ell : `float`
            Fundamental frequency factor for elliptical or ellipsoidal geometry.
        """

        def MMF(q):
            """Compute the Modified Mathieiu's Function (MMF) or its derivative.

            Parameters
            ----------
            q : `float`
                Argument of the MMF. q = M01 is the 1st root for the 0th-order function.

            Returns
            -------
            mathieu_modcem : `float`
                Value of the MMF or its derivative at the value q:
                mathieu_modcem1(m, q, psi0)[0] for the MMF and
                mathieu_modcem1(m, q, psi0)[1] for its derivative

            Examples
            --------
            mathieu_modcem1(m=0, q=2.6750449521966490,
                psi0=arccosh(2/(3)**0.5))[0]= 5.363046165143026e-17
            mathieu_modcem1(m=0, q=1.6748563428285737,
                psi0=0.7061880927645094)[0] = 4.036310483679603e-16
            """

            # Eccentricity parameter: psi0 = arccosh(a/f), f = sqrt(a^2 - b^2)
            psi0 = arccosh(a0 / f0)
            idx = int(bc == "Neumann")
            m = 1 if bc == "Neumann" else 0  # Order of the MMF
            pprint(bc, m, psi0, q, mathieu_modcem1(m, q, psi0)[idx], comm=self.comm)
            return mathieu_modcem1(m, q, psi0)[idx]

        def ZBF(m=0, n=1):
            """Compute zeros of the Bessel's Function (BF) or its derivative.

            Parameters
            ----------
            m : `int`, optional
                Order of the BF. Default is 0.
            n : `int`, optional
                Number of roots to compute. Default is 1.

            Returns
            -------
            Jmz : `array`
                First n zeros of the Bessel function or its derivative.
            """
            deriv = (bc == "Neumann")
            Jmz = jnp_zeros(m, n) if deriv else jn_zeros(m, n)
            return Jmz

        def SBF(q, m=0):
            """Compute the Spherical Bessel's Function (SBF) or its derivative.

            Parameters
            ----------
            q : `float`
                Argument of the SBF. q = J01 is the 1st root for the 0th-order function.

            Returns
            -------
            spherical_jn : `float`
                Value of the SBF or its derivative at the value q:
                spherical_jn(m, q, derivative=False) for the SBF and
                spherical_jn(m, q, derivative=True) for its derivative.
            """
            deriv = (bc == "Neumann")
            m = int(deriv)  # Order of the SBF: 0 (False) or 1 (True)
            # pprint(bc, m, q, spherical_jn(m, q, derivative=deriv, comm=self.comm))
            return spherical_jn(m, q, derivative=deriv)

        # Semi-axes
        a, b = hyper_axes[: 2]

        # Frequency factor for rectangular/prismatic case
        f_rec = self._freq_factor_rec(hyper_axes, bc=bc)

        # Initial guess
        igss = f_rec if bc == "Neumann" else 0.

        # Circular or spherical case
        if all_axes_equal:
            # 1st root for the mth-order Bessel's function
            if self.dimension == 2:  # 2D circular
                m = 1 if bc == "Neumann" else 0
                J01 = ZBF(m=m, n=1)[0]

            if self.dimension == 3:  # 3D spherical
                J01 = float(broyden1(SBF, igss, f_tol=1e-14))

            return J01 / a

        # Elliptical or ellipsoidal case
        if self.dimension == 2:  # 2D elliptical

            # Order semi-axes
            a, b = sorted(hyper_axes, reverse=True)

            # Ellipse eccentricity
            f0 = (a**2 - b**2) ** 0.5

            # 1st root or the mth-order Modified Mathieu's Function
            a0 = a
            M01 = float(broyden1(MMF, igss, f_tol=1e-14))

            return (2 / f0) * M01 ** 0.5

        if self.dimension == 3:  # 3D ellipsoidal

            f_ell_arr = []

            # Order semi-axes
            a, b, c = sorted(hyper_axes, reverse=True)

            # Eccentricities for each pair of semi-axes
            ecc_arr = [(a, b, (a**2 - b**2)**0.5 if a > b else 0.),
                       (b, c, (b**2 - c**2)**0.5 if b > c else 0.),
                       (a, c, (a**2 - c**2)**0.5 if a > c else 0.)]

            if bc == "Neumann":
                # Only use the pair with maximum eccentricity
                max_ecc_idx = argmax([ecc for _, _, ecc in ecc_arr])
                ecc_arr = [ecc_arr[max_ecc_idx]]

            for a0, b0, f0 in ecc_arr:

                if f0 == 0:  # Circular cross-section
                    # 1st root for the mth-order Bessel's function
                    J01 = ZBF(m=0, n=1)[0]
                    f_ell_arr.append((J01 / a0) ** 2)

                else:  # Elliptical cross-section
                    # 1st root or the mth-order Modified Mathieu's Function
                    M01 = float(broyden1(MMF, igss, f_tol=1e-14))
                    f_ell_arr.append(4 * M01 / f0 ** 2)

            # Sum and return square root
            return sum(f_ell_arr) ** 0.5

    def _reg_geometry_hyp(self, cut_plane_percent=1.):
        """Perform the nonlinear regression for the hypershape geometry factor.

        Parameters
        ----------
        cut_plane_percent : `float`, optional
            Percentage of the cut plane (0 to 1). Default is 1 (no cut).

        Returns
        -------
        pn_fit : `float`
            Fitted parameter pn.
        qn_fit : `float`
            Fitted parameter qn.
        fr_ell : `float`
            Ratio between the area or volume of the truncated and
            full ellipse or ellipsoidal.
        fr_rec : `float`
            Ratio between the area or volume of the truncated and
            full rectangle or prism.
        """

        def area_function(n, cut_plane_percent):
            """Area function for hiperellipses."""
            fA = 2. * gamma(1 + 1 / n) ** 2 / gamma(1 + 2 / n)
            if cut_plane_percent == 1.:
                fA *= 2.
            else:
                eps = float_info.min
                w = maximum(cut_plane_percent ** n, eps)  # w <= 1
                p = 1 / n
                q = 1 + 1 / n
                B_w = beta(p, q) * betainc(p, q, w)  # Non-regularized Beta
                fA += (2. / n) * B_w

            return fA

        def volume_function(n, cut_plane_percent):
            """Volume function for hiperellipsoids."""
            fV = 4. * gamma(1 + 1 / n) ** 3 / gamma(1 + 3 / n)
            if cut_plane_percent == 1.:
                fV *= 2.
            else:
                eps = float_info.min
                w = maximum(cut_plane_percent ** n, eps)  # w <= 1
                p = 1 / n
                q = 1 + 1 / n
                A_f = gamma(1 + p)**2 / gamma(q)
                B_w = beta(p, q) * betainc(p, q, w)  # Non-regularized Beta
                fV += (4. / n) * A_f * B_w

            return fV

        def fit_function(n, pn, qn):
            """ Define the fit function for the area or volume regression.

            Fit function:
                A or V = f_max - cn2 * (1 / (qn * n + 1 - 2 * qn)) ** pn
                cn2 = f_max - fn2
            """

            # Constant for power-law fit
            cn2 = f_max - fn2

            return f_max - cn2 * (1. / (qn * n + 1. - 2. * qn)) ** pn

        # Regression dataset
        n_data = arange(2., 100., 0.1)

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
        init_guess = array([1/3, 1/3])

        # Parameter bounds pn >= 0, qn >= 0
        low_bnds = [0, 0]
        upp_bnds = [inf, inf]

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
            rmse = sqrt(mean(residuals**2))

            # Calculate confidence intervals
            perr = sqrt(diag(pcov))
            delta_pn = pn_fit - sn.interval(0.95, loc=pn_fit, scale=perr[0])[0]
            delta_qn = qn_fit - sn.interval(0.95, loc=qn_fit, scale=perr[1])[0]

            pprint("Nonlinear Curve Fit Successful!", comm=self.comm)
            pprint(f"Fitted Parameters: pn = {pn_fit:.6f} ± {delta_pn:.6f}, "
                   f"qn = {qn_fit:.6f} ± {delta_qn:.6f}", comm=self.comm)
            pprint(f"R-Squared: {r_squared:.6f} - RMSE: {rmse:.6f}", comm=self.comm)

            return pn_fit, qn_fit, fr_ell, fr_rec

        except ConvergenceError as e:
            pprint(f"Nonlinear Curve Fit Failed: {e}", comm=self.comm)

    def _freq_factor_hyp(self, n_hyp, f_rec, f_ell, c_eq, bc="Neumann", c_eqref=None,
                         fitting_c=(0., 0., 0., 0.), cut_plane_percent=1.):
        """Compute an approximate frequency factor for a full or truncated hypershape.

        The truncation plane is at z = cut_plane_percent * b, with b = Lz + pad_len.
        The fitting parameters for the equivalent velocity regression controls:
        - fc1: Magnitude order of the frequency.
        - fc2: Monotonicity of the frequency.
        - fp1: Rectangular domain frequency.
        - fp2: Ellipsoidal domain frequency.

        Parameters
        ----------
        n_hyp : `float`
            Degree of the hypershape.
        f_rec : `float`
            Fundamental frequency factor for rectangular or prismatic geometry.
        f_ell : `float`
            Fundamental frequency factor for elliptical or ellipsoidal geometry.
        c_eq : `float`
            Equivalente isotropic velocity in the hypershape.
        bc : `str`, optional
            Boundary condition type: "Dirichlet" or "Neumann". Default is "Neumann".
        c_eqref : `float`, optional
            Reference value for the equivalent velocity based on the original
            velocity model without an absorbing layer. Default is `None`.
        fitting_c : `tuple`, optional
            Parameters for fitting equivalent velocity regression.
            Structure: (fc1, fc2, fp1, fp2). Default is (0., 0., 0., 0.).
            - fc1 : `float`
                Exponent factor for the minimum reference velocity.
            - fc2 : `float`
                Exponent factor for the maximum reference velocity.
            - fp1 : `float`
                Exponent factor for the minimum equivalent velocity.
            - fp2 : `float`
                Exponent factor for the maximum equivalent velocity.
        cut_plane_percent : `float`, optional
            Percentage of the cut plane (0 to 1). Default is 1 (no cut).

        Returns
        -------
        f_hyp : `float`
            Approximate frequency factor for the hypershape.
        c_reg : `float` or `None`
            Approximate equivalent velocity for the hypershape.
        """

        # Regression for hypershape geometry factor
        pn, qn, fr_ell, fr_rec = self._reg_geometry_hyp(cut_plane_percent=cut_plane_percent)

        if bc == "Dirichlet":
            f_min = f_rec / fr_rec
            cn2 = f_min - f_ell / fr_ell

        if bc == "Neumann":
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

    def solver_analytical(self, c_eq, hyp_par, bc="Neumann", c_eqref=None,
                          fitting_c=(0., 0., 0., 0.), cut_plane_percent=1.):
        """"Compute the analytical eigenvalue for hypershapes by using homogenization.

        Support Neumann or Dirichlet boundary conditions.

        Parameters
        ----------
        c_eq : `float`
            Equivalente isotropic velocity in the hypershape.
        hyp_par : `tuple`
            Hyperellipshape parameters.
            Structure 2D: (n_hyp, a_hyp, b_hyp).
            Structure 3D: (n_hyp, a_hyp, b_hyp, c_hyp).
            - n_hyp : `float` or `None`
                Degree of the hypershape. If `None`, 'n_hyp' = 330 for rectangles or prisms.
           - a_hyp : `float`
                Hypershape semi-axis in direction x.
            - b_hyp : `float`
                Hypershape semi-axis in direction z.
            - c_hyp : `float`
                Hypershape semi-axis in direction y (3D only).
        bc : `str`, optional
            Boundary condition type: "Dirichlet" or "Neumann".
            Default is "Neumann"
        c_eqref : `float`, optional
            Reference value for the equivalent velocity based on the original
            velocity model without an absorbing layer. If `None`, 'c_eqref' = 'c_eq'.
        fitting_c : `tuple`, optional
            Parameters for fitting equivalent velocity regression.
            Structure: (fc1, fc2, fp1, fp2). Default is (0., 0., 0., 0.).
            - fc1 : `float`
                Exponent factor for the minimum reference velocity.
            - fc2 : `float`
                Exponent factor for the maximum reference velocity.
            - fp1 : `float`
                Exponent factor for the minimum equivalent velocity.
            - fp2 : `float`
                Exponent factor for the maximum equivalent velocity.
        cut_plane_percent : `float`, optional
            Percentage of the cut plane (0 to 1). Default is 1 (no cut).

        Returns
        -------
        Lsp : `float`
            First eigenvalue of the hypershape with Neumann or Dirichlet BCs.
        """

        # Check the isotropic velocity
        value_numerical_error("c_eq", c_eq, lower_bound=0., include_lower_bound=False)

        # Hyperellipse parameters
        n_hyp, hyp_axes = hyp_par[0], hyp_par[1:]

        # Check the hypershape degree
        n_hyp = 330 if n_hyp is None else value_numerical_error(
            "n_hyp", n_hyp, lower_bound=2., include_lower_bound=True)

        # Check semi-axes type
        type_data_structure_error("hyper_axes", hyper_axes, "tuple",
                                  ("float", "int"), expected_length=self.dimension)

        # Check boundary condition type
        value_parameter_error("bc", bc, ["Dirichlet", "Neumann"])

        # Check the isotropic velocity from original model without absorbing layer
        c_eqref = c_eq if c_eqref is None else value_numerical_error(
            "c_eqref", c_eqref, lower_bound=0., include_lower_bound=False)

        # Check the parameters for fitting equivalent velocity regression.
        type_data_structure_error("fitting_c", fitting_c, "tuple",
                                  ("float", "int"), expected_length=4)

        # Check the cutting plane percent is between 0 and 1
        value_numerical_error("cut_plane_percent", cut_plane_percent,
                              lower_bound=0., upper_bound=1.,
                              include_lower_bound=True, include_upper_bound=True)

        a, b = hyp_axes[: 2]
        if self.dimension == 2:  # 2D
            all_axes_equal = (a == b)

        if self.dimension == 3:  # 3D
            c = hyp_axes[2]
            all_axes_equal = (a == b == c)

        # Frequency factors
        f_rec = self._freq_factor_rec(hyp_axes, bc=bc)
        f_ell = self._freq_factor_ell(hyp_axes, bc=bc, all_axes_equal=all_axes_equal)
        f_hyp, c_reg = self._freq_factor_hyp(n_hyp, f_rec, f_ell, c_eq, bc=bc,
                                             c_eqref=c_eqref, fitting_c=fitting_c,
                                             cut_plane_percent=cut_plane_percent)

        pprint(f"Hypershape Equivalent Velocity c_eq (km/s) = {c_reg:.3f}", comm=self.comm)
        pprint(f"Hypershape Frequency Factor f_hyp (1/km): {f_hyp:.3f}", comm=self.comm)

        # Eigenvalue
        Lsp = (c_reg * f_hyp)**2

        return Lsp
