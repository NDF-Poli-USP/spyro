from firedrake import (assemble, ConvergenceError, dx as fire_dx, Function, grad,
                       inner, LinearVariationalProblem, LinearVariationalSolver)
from numpy import (arange, arccosh, argmax, array, asarray,
                   diag, inf, maximum, mean, pi, sqrt)
from scipy.optimize import curve_fit, fsolve, minimize_scalar
from scipy.special import (beta, betainc, gamma, jn_zeros, jnp_zeros,
                           mathieu_modcem1, spherical_jn)
from scipy.stats import norm as sn
from sys import float_info
from ...io.basicio import parallel_print as pprint
from .modal_forms_and_matrices import weak_forms
from ...utils.error_management import (type_data_structure_error, type_firedrake_error,
                                       value_numerical_error, value_parameter_error)
from ...utils.stats_tools import coeff_of_determination


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
    comm : `object`, optional
        An object representing the communication interface for parallel processing.
        Default is `None`.
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

    def __init__(self, dimension=2, comm=None):
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

    @staticmethod
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
            # pprint((bc, m, psi0, q, mathieu_modcem1(m, q, psi0)[idx]), comm=self.comm)
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
            # pprint((bc, m, n, Jmz), comm=self.comm)
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
            # pprint((bc, m, q, spherical_jn(m, q, derivative=deriv)), comm=self.comm)
            return spherical_jn(m, q, derivative=deriv)

        def ana_rq_ellipsoid(alpha):
            """Rayleigh quotient for ellipsoid with trial function ψ = (1 - r²)(1 + α r²).

            Parameters
            ----------
            alpha : `float`
                Parameter of the trial function ψ = (1 - r²)(1 + α r²).

            Returns
            -------
            RQ(α) : `float`
            Rayleigh quotient RQ(α) = E(α) / N(α)
            """
            # Radial integral I_r
            I_r = (alpha - 1)**2./5. - 4 * alpha * (alpha - 1.)/7. + 4. * alpha**2./9.

            # Radial integral J_r
            J_r = (1./3. + (2 * alpha - 2)/5. + (alpha**2. - 4.*alpha + 1.)/7.
                   - (2. * alpha**2. - 2. * alpha)/9. + alpha**2./11.)

            RQ = (4./3.) * (1./a**2 + 1./b**2 + 1./c**2) * (I_r / J_r)
            # pprint((bc, I_r, J_r, alpha, RQ), comm=self.comm)
            return RQ

        # Semi-axes
        a, b = hyper_axes[: 2]

        # Circular or spherical case
        if all_axes_equal:
            m = 1 if bc == "Neumann" else 0

            # 1st root for the mth-order Bessel's function
            first_root_2D = ZBF(m=m, n=1)[0]

            if self.dimension == 2:  # 2D circular
                J01 = first_root_2D

            if self.dimension == 3:  # 3D spherical
                J01 = fsolve(SBF, first_root_2D, xtol=1e-14)[0]

            return J01 / a

        # Frequency factor for rectangular/prismatic case
        f_rec = self._freq_factor_rec(hyper_axes, bc=bc)

        # Initial guess
        igss = f_rec if bc == "Neumann" else 0.

        # Elliptical or ellipsoidal case
        if self.dimension == 2:  # 2D elliptical

            # Order semi-axes
            a, b = sorted(hyper_axes, reverse=True)

            # Ellipse eccentricity
            f0 = (a**2 - b**2) ** 0.5

            # 1st root or the mth-order Modified Mathieu's Function
            a0 = a
            M01 = fsolve(MMF, igss, xtol=1e-14)[0]

            return (2 / f0) * M01 ** 0.5

        if self.dimension == 3:  # 3D ellipsoidal

            # Order semi-axes
            a, b, c = sorted(hyper_axes, reverse=True)

            # Eccentricities for each pair of semi-axes
            ecc_arr = [(a, b, (a**2 - b**2)**0.5 if a > b else 0.),
                       (b, c, (b**2 - c**2)**0.5 if b > c else 0.),
                       (a, c, (a**2 - c**2)**0.5 if a > c else 0.)]

            if bc == "Neumann":
                # Only use the pair with maximum eccentricity
                max_ecc_idx = argmax([ecc for _, _, ecc in ecc_arr])
                a0, b0, f0 = ecc_arr[max_ecc_idx]

                if f0 == 0:  # Circular cross-section
                    # 1st root for the mth-order Bessel's function
                    J01 = ZBF(m=0, n=1)[0]

                    return J01 / a

                else:  # Elliptical cross-section
                    # 1st root or the mth-order Modified Mathieu's Function
                    M01 = fsolve(MMF, igss, xtol=1e-14)[0]

                    return (2 / f0) * M01 ** 0.5

            if bc == "Dirichlet":

                # Use Rayleigh-Ritz  optimizing the parameter α in the trial function ψ.
                RQ = minimize_scalar(ana_rq_ellipsoid, bounds=(-1., 1.),
                                     method='bounded', tol=1e-14)

                return RQ.fun ** 0.5

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
            fr_rec = area_function(100., cut_plane_percent) / area_function(100., 1.)
            f_data = area_function(n_data, cut_plane_percent)

        if self.dimension == 3:  # 3D
            f_max = 4. * fax_trunc
            fn2 = volume_function(2., cut_plane_percent)
            fr_ell = fn2 / volume_function(2., 1.)
            fr_rec = volume_function(100., cut_plane_percent) / volume_function(100., 1.)
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
            Equivalente homogeneous velocity in the hypershape.
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

    def dummy_load_static_(self, V, dof_load, amplitude_load, V_ref=None):
        """Build a static load for the energy-equivalent homogenization.

        Parameters
        ----------
        V : `Firedrake.FunctionSpace`
            Function space for the modal problem.
        dof_load : `array`
           Degrees of freedom (DOFs) where the static load is applied.
        amplitude_load : `array`
            Amplitude of the static load at the specified DOFs.
        V_ref : `Firedrake.FunctionSpace`, optional
            Function space for the reference model (without absorbing layer).
            Default is `None`, in which case only the static load for the model
            with absorbing layer is returned.

        Returns
        -------
        q_dummy : `Firedrake.Function`
            Static load for model with absorbing layer.
        q_ref : `Firedrake.Function`
            Static load for reference model (without absorbing layer).
            Only returned if 'V_ref' is not `None`.
        """

        # Check imput arguments
        type_data_structure_error("dof_load", dof_load, "array",
                                  expected_type_element=("int"))
        type_data_structure_error("amplitude_load", amplitude_load, "array",
                                  expected_type_element=("float", "int"))
        type_firedrake_error("V", V, "FunctionSpace")

        # Static load for model with absorbing layer
        q_dummy = Function(V)
        q_dummy.dat.data_with_halos[dof_load] = amplitude_load

        # Static load for reference model (without absorbing layer)
        q_ref = None
        V_ref = type_firedrake_error("V_ref", V_ref, "FunctionSpace", none_default=True)
        if V_ref:
            q_ref = Function(V_ref)
            q_ref.interpolate(q_dummy, allow_missing_dofs=True)

        return (q_dummy, q_ref)

    def c_equivalent(self, c, V, quad_rule=None,
                     type_homog="energy", static_load_for_ceq=None):
        """Compute equivalent homogeneous velocity for an inhomogeneous model.

        The method uses an energy-equivalent homogenization by default.

        Parameters
        ----------
        c : `Firedrake.Function`
            Velocity model.
        V : `Firedrake.FunctionSpace`
            Function space for the modal problem.
        quad_rule : `dict`, optional
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

        # Check input arguments
        type_firedrake_error("c", c, "Function")
        type_firedrake_error("V", V, "FunctionSpace")
        type_data_structure_error("quad_rule", quad_rule, "dict", none_default=True)
        value_parameter_error("type_homog", type_homog, ["energy", "volume"])
        type_firedrake_error("static_load_for_ceq", static_load_for_ceq,
                             "Function", none_default=True)

        # Integration measure
        dx = fire_dx(**quad_rule) if quad_rule else fire_dx

        # State variable
        u = Function(V)

        if type_homog == "energy":
            # Equivalent velocity by energy-equivalent homogenization

            # Weak forms
            a, L = weak_forms(c, V, quad_rule=quad_rule, source=True,
                              user_load=static_load_for_ceq)

            # Solve static load problem for the energy-equivalent homogenization
            lin_var = LinearVariationalProblem(a, L, u, constant_jacobian=True)
            solver_param = {"ksp_type": "gmres",
                            "pc_type": "hypre",
                            "pc_hypre_type": "boomeramg",
                            "ksp_rtol": 1e-12,
                            "ksp_atol": 1e-12,
                            "ksp_gmres_restart": 100}
            LinearVariationalSolver(lin_var, solver_parameters=solver_param).solve()

            # Compute the energy
            bilinear_term = 0.5 * inner(grad(u), grad(u))
            energy = assemble(c * c * bilinear_term * dx)

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

    def homogenized_velocities(self, c, V, c_ref=None, V_ref=None, quad_rule=None,
                               dof_load=None, amplitude_load=None, type_homog="energy"):
        """Compute equivalent homogeneous velocities required for the analytical solver.

        Parameters
        ----------
        c : `Firedrake.Function` or `float`
            Velocity model for the model with absorbing layer.
        V : `Firedrake.FunctionSpace`
            Function space for the model with absorbing layer.
        c_ref : `Firedrake.Function` or `float`, optional
            Velocity model for the reference model without absorbing layer.
            Default is `None`.
        V_ref : `Firedrake.FunctionSpace`, optional
            Function space for the reference model without absorbing layer.
            Default is `None`.
        quad_rule : `dict`, optional
            Quadrature rule to use for the integration.
            Default is `None`, which uses the default quadrature rule.
        dof_load : `array`, optional
           Degrees of freedom (DOFs) where the static load is applied.
        amplitude_load : `array`, optional
            Amplitude of the static load at the specified DOFs.
        type_homog : `str`, optional
            Type of homogenization: "energy" or "volume". Default is "energy"

        Returns
        -------
        c_eq : `float`
            Equivalent homogeneous velocity for the model with absorbing layer.
        c_eqref : `float`
            Equivalent homogeneous velocity for the reference model without layer.
        """

        # Check type of homogenization
        value_parameter_error("type_homog", type_homog, ["energy", "volume"])

        # Define the load for the energy-equivalent homogenization
        c_is_float = isinstance(c, (int, float))
        dummy_load = self.dummy_load_static_(
            V, dof_load, amplitude_load, V_ref=V_ref) \
            if (type_homog == "energy" and not c_is_float) else (None, None)

        # Compute the equivalent velocity for the model with absorbing layer
        c_eq = value_numerical_error(
            "c", c, float_num=True, integer_num=True, lower_bound=0.) if c_is_float \
            else self.c_equivalent(c, V, quad_rule=quad_rule, type_homog=type_homog,
                                   static_load_for_ceq=dummy_load[0])

        # Compute the equivalent velocity for the reference model without layer
        c_eqref = None
        if V_ref is not None:
            c_ref_is_float = isinstance(c_eqref, (int, float))
            c_eqref = value_numerical_error(
                "c_ref", c_ref, float_num=True, integer_num=True, lower_bound=0.) \
                if c_ref_is_float else self.c_equivalent(c_ref, V_ref, quad_rule=quad_rule,
                                                         type_homog=type_homog,
                                                         static_load_for_ceq=dummy_load[1])

        return (c_eq, c_eqref)

    def solver_analytical(self, c_eq, hyp_par, bc="Neumann", c_eqref=None,
                          fitting_c=(0., 0., 0., 0.), cut_plane_percent=1.):
        """"Compute the analytical eigenvalue for hypershapes by using homogenization.

        Support Neumann or Dirichlet boundary conditions.

        Parameters
        ----------
        c_eq : `float`
            Equivalente homogeneous velocity in the hypershape.
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

        # Check the homogeneous velocity
        value_numerical_error("c_eq", c_eq, float_num=True, integer_num=True, lower_bound=0.)

        # Hyperellipse parameters
        n_hyp, hyper_axes = hyp_par[0], hyp_par[1:]

        # Check the hypershape degree
        value_numerical_error("n_hyp", n_hyp, float_num=True, integer_num=True,
                              none_default=True, lower_bound=2., include_lower_bound=True)
        n_hyp = 330 if n_hyp is None else n_hyp

        # Check semi-axes type
        type_data_structure_error("hyper_axes", hyper_axes, "tuple",
                                  expected_type_element=("float", "int"),
                                  expected_length=self.dimension)

        # Check boundary condition type
        value_parameter_error("bc", bc, ["Dirichlet", "Neumann"])

        # Check the homogeneous velocity from original model without absorbing layer
        value_numerical_error("c_eqref", c_eqref, float_num=True, integer_num=True,
                              none_default=True, lower_bound=0.)
        c_eqref = c_eq if c_eqref is None else c_eqref

        # Check the parameters for fitting equivalent velocity regression.
        type_data_structure_error("fitting_c", fitting_c, "tuple",
                                  expected_type_element=("float", "int"),
                                  expected_length=4)

        # Check the cutting plane percent is between 0 and 1
        value_numerical_error("cut_plane_percent", cut_plane_percent, float_num=True,
                              integer_num=False, lower_bound=0., upper_bound=1.,
                              include_lower_bound=True, include_upper_bound=True)

        a, b = hyper_axes[: 2]
        if self.dimension == 2:  # 2D
            all_axes_equal = (a == b)

        if self.dimension == 3:  # 3D
            c = hyper_axes[2]
            all_axes_equal = (a == b == c)

        # Frequency factors
        f_rec = self._freq_factor_rec(hyper_axes, bc=bc)
        f_ell = self._freq_factor_ell(hyper_axes, bc=bc, all_axes_equal=all_axes_equal)
        f_hyp, c_reg = self._freq_factor_hyp(n_hyp, f_rec, f_ell, c_eq, bc=bc,
                                             c_eqref=c_eqref, fitting_c=fitting_c,
                                             cut_plane_percent=cut_plane_percent)

        pprint(f"Hypershape Equivalent Velocity c_eq (km/s) = {c_reg:.3f}", comm=self.comm)
        pprint(f"Hypershape Frequency Factor f_hyp (1/km): {f_hyp:.3f}", comm=self.comm)

        # Eigenvalue
        Lsp = (c_reg * f_hyp)**2

        return Lsp
