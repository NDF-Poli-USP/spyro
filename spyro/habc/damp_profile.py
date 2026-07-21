from numpy import asarray, clip, cos, isreal, pi, polyfit, roots, sin, tan
from ..io.basicio import parallel_print as pprint
from spyro.utils.error_management import value_parameter_error

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class HABC_Damping():
    """ Class for HABC damping profile inside the absorbing layer.

    Attributes
    ----------
    alpha : `float`
        Ratio between the representative mesh dimensions
    comm : `object`
        An object representing the communication interface for parallel processing.
        Default is `None`.
    d_norm : `float`
        Normalized element size (lmin / pad_len).
    dimension : `int`
        Model dimension (2D or 3D). Default is 2D.
    f_correct_pwave : `float`
        Correction factor for the adimensional wavenumber p2.
    kCR : `float`
        Adimensional parameter in the reflection coefficient.
    p1 : `float`
        Hypothetical adimensional wavenumber at the original domain boundary.
    p2 : `float`
        Hypothetical adimensional wavenumber at the begining of the layer.
    variant : `string`
        Finite element type. Options: "consistent" or "lumped".

    Methods
    -------
    adim_reflection_parameters()
        Compute the adimensional parameters for the reflection coefficient
    calc_damping_properties()
        Compute the damping properties for the absorbing layer
    coeff_damp_fun()
        Compute the coefficients for quadratic damping function
    est_min_damping()
        Estimate the minimum damping ratio and the associated heuristic factor
    min_reflection()
        Compute a minimum reflection coefficiente for the quadratic damping
    psi_from_CR()
        Compute the damping ratio from the reflection coefficient
    regression_CRmin()
        Define the minimum damping ratio and the associated heuristic factor
    xCR_search_range()
        Determine the initial search range for the heuristic factor xCR
    """

    def __init__(self, domain_layer, layer_par, mesh_par, wave_par, dimension=2, comm=None):
        """
        Initialize the HABC_Damping class

        Parameters
        ----------
        domain_layer : `tuple`
            Domain dimensions with layer including the truncatiion by the free surface.
            2D: (length_z + pad_len, length_x + 2 * pad_len).
            3D: (length_z + pad_len, length_x + 2 * pad_len, length_y + 2 * pad_len).
        layer_par : `tuple`
            Parameters of the absorbing layer. (factor_length_pad, d_norm, a_par)
            - factor_length_pad : Size parameter of the absorbing layer.
            - a_par : Adimensional propagation speed parameter (a = z / f).
            - d_norm : Normalized element size (lmin / pad_len).
        mesh_par : `tuple`
            Mesh parameters. (lmin, lmax, alpha, variant)
            - lmin : Minimum mesh size.
            - lmax : Maximum mesh size.
            - alpha : Ratio between the representative mesh dimensions.
            - variant : Finite element type. Options: "consistent" or "lumped".
        wave_par : `tuple`
            Wave parameters. (freq_ref, c_ref, c_bnd)
            - freq_ref : Reference frequency of the wave at the critical poin on boundary.
            - c_ref : Minimum propagation speed among all critical points.
            - c_bnd : Propagation speed at critical point.
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D
        comm : `object`, optional
            An object representing the communication interface for parallel processing.
            Default is `None`.

        Returns
        -------
        None
        """

        # Domain dimensions with layer
        self.domain_layer = domain_layer

        # Normalized element size
        self.d_norm = layer_par[2]

        # Ratio between the representative mesh dimensions
        self.alpha = mesh_par[2]

        # Finite element type
        self.variant = mesh_par[3]

        # Model dimension
        self.dimension = dimension

        # Communicator MPI
        self.comm = comm

        # Adimensional wave numbers
        self.adim_reflection_parameters(layer_par[:2], mesh_par[:2], wave_par)

    def adim_reflection_parameters(self, layer_par, mesh_par, wave_par, vibration_mode=1):
        """Compute the adimensional parameters for the reflection coefficient.

        Parameters
        ----------
        layer_par : `tuple`
            Parameters of the absorbing layer. (factor_length_pad, d_norm, a_par)
            - factor_length_pad : Size parameter of the absorbing layer
            - a_par : Adimensional propagation speed parameter (a = z / f).
            - d_norm : Normalized element size (lmin / pad_len)
        mesh_par : `tuple`
            Mesh parameters. (lmin, lmax)
            - lmin : Minimum mesh size
            - lmax : Maximum mesh size
        wave_par : `tuple`
            Wave parameters. (freq_ref, c_ref, c_bnd)
            - freq_ref : Reference frequency of the wave at the critical poin on boundary.
            - c_ref : Minimum propagation speed among all critical points.
            - c_bnd : Propagation speed at critical point.
        vibration_mode : `int`, optional
            Vibration mode. Default is 1 (Fundamental mode).

        Returns
        -------
        None
        """

        # Layer parameters
        factor_length_pad, a_par = layer_par

        # Mesh parameters
        lmin, lmax = mesh_par

        # Wave parameters
        freq_ref, c_ref, c_bnd = wave_par

        # Dimensionless wave numbers
        self.p1 = 2. * pi * freq_ref * lmin / c_ref
        self.p2 = 2. * pi * freq_ref * lmax / c_ref
        self.f_correct_pwave = a_par / factor_length_pad

        # Adimensional parameter in the reflection coefficient
        kCR = 4. * factor_length_pad / (a_par * vibration_mode)
        self.kCR = kCR * c_bnd / c_ref

    @staticmethod
    def psi_from_CR(CR, kCR):
        """Compute the damping ratio from the reflection coefficient.

        Parameters
        ----------
        CR : `float`
            Reflection coefficient.
        kCR : `float`
            Adimensional parameter in the reflection coefficient.

        Returns
        -------
        psi_damp : `float`
            Damping ratio.
        """
        if CR == 0.:
            psi_damp = 0.
        elif CR >= 1.:
            psi_damp = 0.999
        else:
            psi_damp = kCR / (1. / CR - 1.)**0.5

        return psi_damp

    def min_reflection(self, psi_damp=None, CR_err=None, type_CR='CR_PSI'):
        """Compute a minimum reflection coefficient for the quadratic damping.

        Parameters
        ----------
        psi_damp : `float`, optional
            Damping ratio when 'type_CR' is "CR_PSI". Default is None.
        CR_err : `float`, optional
            Reflection coefficient when 'type_CR' is "CR_ERR". Default is `None`.
        type_CR : `string`, optional
            Type of reflection coefficient. Default is "CR_PSI".
            - "CR_PSI" : Minimum coefficient reflection from a damping ratio.
            - "CR_FEM" : Spourious reflection coeeficient in FEM.
            - "CR_ERR" : Correction for the minimum damping ratio.

        Returns
        -------
        CRmin : `float`
            Minimum reflection coefficient at the minimum damping.
        xCRmin : `float`
            Heuristic factor for the minimum damping ratio.
        """

        value_parameter_error("type_CR", type_CR, ["CR_PSI", "CR_FEM", "CR_ERR"])

        if type_CR == "CR_PSI":
            # Minimum coefficient reflection
            psimin = psi_damp
            CRmin = psimin**2 / (self.kCR**2 + psimin**2)

        elif type_CR == "CR_ERR":
            # Minimum damping ratio for correction in reflection parameters
            CRmin = CR_err
            psimin = self.psi_from_CR(CR_err, self.kCR)

        elif type_CR == "CR_FEM":
            # Unidimensional spourious reflection in FEM (Laier, 2020)
            # TODO: Add citation
            def Zi(p, alpha, ele_type):
                """Compute the Z parameter in the spurious reflection coefficient.

                Parameters
                ----------
                p : `float`
                    Dimensionless wavenumber.
                alpha : `float`
                    Ratio between the representative mesh dimensions.
                ele_type : `string`
                    Finite element type. Options: "consistent" or "lumped"

                Returns
                -------
                Z_fem : `float`
                    Parameter for the spurious reflection coefficient in FEM
                """
                value_parameter_error("ele_type", ele_type, ["lumped", "consistent"])

                if ele_type == "lumped":
                    m1 = 1 / 2
                    m2 = 0.
                elif ele_type == "consistent":
                    m1 = 1 / 3
                    m2 = 1 / 6

                Z_fem = m2 * (cos(alpha * p) - 1) / (m1 * (cos(alpha * p) + 1))

                return Z_fem

            # Spurious reflection coefficient in FDM (Kar and Turco, 1995)
            # TODO: Add citation
            CRfdm = tan(self.p1 / 4)**2

            # Minimum damping ratio for the spurious reflection
            psimin = self.psi_from_CR(CRfdm, self.kCR)

            # Correction for the dimensionless wavenumbers due to the damping
            self.p2 *= (1 + 1 / 8 * (psimin * self.f_correct_pwave)**2)

            # Zi parameters for the spurious reflection coefficient
            Z1 = Zi(self.p1, self.alpha, self.variant)
            Z2 = Zi(self.p2, self.alpha, self.variant)

            # Spurious reflection coefficient in FEM (Laier, 2020)
            # TODO: Add citation
            aux0 = (1 - Z1) * sin(self.p1)
            aux1 = (Z2 - 1 / self.alpha) * sin(self.alpha * self.p2)
            CRmin = abs((aux0 + aux1) / (aux0 - aux1))

        xCRmin = psimin / self.d_norm

        return CRmin, xCRmin

    def regression_CRmin(self, xCR_reg, CRmin_reg, xCR_lim):
        """Define the minimum damping ratio and the associated heuristic factor.

        Parameters
        ----------
        xCR_reg : `tuple`
            Values of the heuristic factor for regression.
        CRmin_reg : `tuple`
            Values of the minimum damping ratio for regression.
        xCR_lim : `tuple`
            Limits for the heuristic factor. (xCR_inf, xCR_sup, xCR_ini)

        Returns
        -------
        psi_min : `float`
            Minimum damping ratio of the absorbing layer (psi_min = xCR * d).
        xCR_est : `float`
            Estimated heuristic factor for the minimum damping ratio.
        CRmin : `float`
            Minimum reflection coefficient at the minimum damping ratio.
        """

        # Limits for the heuristic factor
        xCR_inf, xCR_sup, xCR_ini = xCR_lim

        # Coefficients for the quadratic equation
        z = polyfit(xCR_reg, CRmin_reg, 2)

        # Roots of the quadratic equation
        roots = roots(z)
        real_roots = [r.real for r in roots if isreal(r)]
        max_root = max(real_roots) if len(real_roots) == 2 else 0.

        # Vertex of parabola
        xCR_vtx = -z[1] / (2 * z[0]) if len(real_roots) == 2 else 0.

        # Vertex or minimum positive root
        xCR_est = xCR_vtx if xCR_vtx > xCR_inf else (max_root if max_root > xCR_inf
                                                     else xCR_ini)
        xCR_est = clip(xCR_est, xCR_inf, xCR_sup)

        # Minimum damping ratio
        psi_min = xCR_est * self.d_norm
        CRmin = self.min_reflection(psi_damp=psi_min)[0]

        return psi_min, xCR_est, CRmin

    def est_min_damping(self, psi_damp=0.999):
        """Estimate the minimum damping ratio and heuristic factor.

        The reflection coefficient is not zero because there are
        always both reflections: physical and spurious.

        Parameters
        ----------
        psi_damp : `float`, optional
            Damping ratio. Default is 0.999.

        Returns
        -------
        psi_min : `float`
            Minimum damping ratio of the absorbing layer (psi_min = xCR * d).
        xCR_est : `float`
            Estimated heuristic factor for the minimum damping ratio.
        xCR_lim : `tuple`
            Limits for the heuristic factor. (xCR_inf, xCR_sup)
        CRmin : `float`
            Minimum reflection coefficient at the minimum damping ratio.
        """

        # Lower Limit for the minimum damping ratio
        psimin_inf = psi_damp * self.d_norm**2
        CRmin_inf, xCR_inf = self.min_reflection(psi_damp=psimin_inf)

        # Upper Limit for the minimum damping ratio
        psimin_sup = psi_damp * (2. * self.d_norm - self.d_norm**2)
        CRmin_sup, xCR_sup = self.min_reflection(psi_damp=psimin_sup)

        # Initial guess
        psimin_ini = psi_damp * (self.d_norm**2 + self.d_norm) / 2.
        CRmin_ini, xCR_ini = self.min_reflection(psi_damp=psimin_ini)

        # Spurious reflection
        CRmin_fem, xCR_fem = self.min_reflection(type_CR="CR_FEM")

        # Minimum damping ratio
        xCR_reg = (xCR_inf, xCR_sup, xCR_ini, xCR_fem)
        CRmin_reg = (CRmin_inf, CRmin_sup, CRmin_ini, CRmin_fem)
        xCR_lim = (xCR_inf, xCR_sup, xCR_ini)
        psi_min, xCR_est, CRmin = self.regression_CRmin(xCR_reg, CRmin_reg, xCR_lim)

        return psi_min, xCR_est, xCR_lim[:2], CRmin

    def calc_damping_properties(self, fundam_freq, xCR_usu=None, psi_damp=0.999):
        """Compute the damping properties for the absorbing layer.

        Parameters
        ----------
        fundam_freq : `float`
            Fundamental frequency of the numerical model.
        xCR_usu : `float`, optional
            User-defined heuristic factor for the minimum damping ratio.
            Default is `None`, which defines an estimated value.
        psi_damp : `float`, optional
            Damping ratio. Default is 0.999.

        Returns
        -------
        eta_crt : `float`
            Critical damping coefficient (1/s).
        psi_min : `float`
            Minimum damping ratio of the absorbing layer (psi_min = xCR * d).
        xCR : `float`
            Heuristic factor for the minimum damping ratio.
        xCR_lim : `tuple`
            Limits for the heuristic factor. (xCR_inf, xCR_sup)
        CRmin : `float`
            Minimum reflection coefficient at the minimum damping ratio.
        """

        # Critical damping coefficient
        eta_crt = 2 * pi * fundam_freq
        eta_max = psi_damp * eta_crt
        pprint(f"Critical Damping Coefficient (1/s): {eta_crt:.5f}", self.comm)

        # Maximum damping ratio and coefficient
        pprint(f"Maximum Damping Ratio: {psi_damp:.3%}", self.comm)
        pprint(f"Maximum Damping Coefficient (1/s): {eta_max:.5f}", self.comm)

        # Minimum damping ratio and the associated heuristic factor
        psi_min, xCR_est, xCR_lim, CRmin = self.est_min_damping()
        xCR_inf, xCR_sup = xCR_lim

        # Heuristic factor for the minimum damping ratio
        if xCR_usu is None:
            xCR = xCR_est
            xcr_str = f"Estimated Heuristic Factor xCR: {xCR:.3f}"
        else:
            xCR = clip(xCR_usu, xCR_lim[0], xCR_lim[1])
            psi_min = xCR * self.d_norm
            xcr_str = f"Using User-Defined Heuristic Factor xCR: {xCR:.3f}"

        # Minimum damping ratio and coefficient
        eta_min = psi_min * eta_crt
        pprint("Minimum Damping Ratio: {psi_min:.3%}", self.comm)
        psi_str = "Range for Minimum Damping Ratio. " \
            + f"Min:{xCR_inf * self.d_norm:.5f} - Max:{xCR_sup * self.d_norm:.5f}"
        pprint(psi_str, self.comm)
        pprint(f"Minimum Damping Coefficient (1/s): {eta_min:.5f}", self.comm)

        # Heuristic factor and its range
        pprint(xcr_str, self.comm)
        xrg_str = f"Range Values for xCR Factor. Min:{xCR_inf:.3f} - Max:{xCR_sup:.3f}"
        pprint(xrg_str, self.comm)

        return eta_crt, psi_min, xCR, xCR_lim, CRmin

    def coeff_damp_fun(self, psi_min, psi_damp=0.999):
        """Compute the coefficients for quadratic damping function.

        Parameters
        ----------
        psi_min' : `float`
            Minimum damping ratio.
        psi_damp : `float`, optional
            Damping ratio. Default is 0.999.

        Returns
        -------
        aq : `float`
            Coefficient for quadratic term in the damping function.
        bq : `float`
            Coefficient bq for linear term in the damping function.
        """

        aq = (psi_min - self.d_norm * psi_damp) / (self.d_norm**2 - self.d_norm)
        bq = psi_damp - aq

        return aq, bq

    def xCR_search_range(self, CRmin, xCR_lim, f_geom):
        """Determine the initial search range for the heuristic factor xCR.

        Parameters
        ----------
        CRmin : `float`
            Minimum reflection coefficient at the minimum damping.
        xCR_lim : `tuple`
            Limits for the heuristic factor. (xCR_inf, xCR_sup)
        f_geom  : `float`
            Geometric factor for area (2D) or volume (3D) of the domain.

        Returns
        -------
        xCR_search : `tuple`
            Initial search range for the heuristic factor. (xCR_min, xCR_max)
            - xCR_min : Lower bound on the search range.
            - xCR_max : Upper bound on the search range.
        """

        # Limits for the heuristic factor
        xCR_inf, xCR_sup = xCR_lim

        # Errors: Spurious reflection rates (Matsuno, 1966)
        err1 = abs(sin(asarray([self.p1, self.p2]) / 2)).max()
        err2 = abs(-1 + cos([self.p1, self.p2])).max()

        # Correction by spurious reflection
        CR_err_min = CRmin * (1 - min(err1, err2))
        xCR_lb = self.min_reflection(CR_err=CR_err_min, type_CR="CR_ERR")[1]
        CR_err_max = CRmin * (1 + max(err1, err2))
        xCR_ub = self.min_reflection(CR_err=CR_err_max, type_CR="CR_ERR")[1]
        xCR_min = clip(max(xCR_lb, xCR_inf), xCR_inf, xCR_sup)
        xCR_max = clip(min(xCR_ub, xCR_sup), xCR_inf, xCR_sup)

        # Model dimensions
        b_rect, a_rect = self.domain_layer[0:2]

        # Axpect ratio for 2D: a/b
        Rab = a_rect / b_rect

        if self.dimension == 2:  # 2D

            # Area factor 0 < f_Ah <= 4
            f_Ah = f_geom

            # Factors and their inverses from sqrt(1/a^2 + 1/b^2)
            fa = (1 + Rab**2)**0.5  # Factoring 1/a^2
            fainv = 1 / fa
            fb = (1 + Rab**2)**0.5 / Rab  # Factoring 1/b^2
            fbinv = 1 / fb
            fmin = f_Ah / 4 * min(fainv, fbinv)
            fmax = 4 / f_Ah * max(fa, fb)

        if self.dimension == 3:  # 3D

            # Adding a dimension for 3D
            c_rect = self.domain_layer[2]

            # Aspect ratios for 3D: a/b, b/c and a/c
            Rac = a_rect / c_rect
            Rbc = b_rect / c_rect

            # Volume factor 0 < f_Vh <= 8
            f_Vh = f_geom

            # Factors and their inverses from sqrt(1/a^2 + 1/b^2 + 1/c^2)
            fa = (1 + Rab**2 + Rac**2)**0.5  # Factoring 1/a^2
            fainv = 1 / fa
            fb = (1 + Rab**2*(1 + Rbc**2))**0.5 / Rab  # Factoring 1/b^2
            fbinv = 1 / fb
            fc = (Rac**2 + (Rac * Rbc)**2 + Rbc**2)**0.5 / (Rac * Rbc)  # 1/c^2
            fcinv = 1 / fc
            fmin = f_Vh / 8 * min(fainv, fbinv, min(fc, fcinv))
            fmax = 8 / f_Vh * max(fa, fb, max(fc, 1 / fc))

        # Correction by geometry
        xCR_min = clip(xCR_min * fmin, xCR_inf, xCR_sup)
        xCR_max = clip(xCR_max * fmax, xCR_inf, xCR_sup)

        lim_str = f"Initial Range for xCR Factor. Min:{xCR_min:.3f} - Max:{xCR_max:.3f}"
        pprint(lim_str, self.comm)

        # ToDo: Use for update xCR in FWI iterations
        return (xCR_min, xCR_max)
