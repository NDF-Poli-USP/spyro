import numpy as np

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class HABC_Damping():
    '''
    Class for HABC damping profile inside the absorbing layer

    Attributes
    ----------

    Methods
    -------
    calc_damping_prop()
        Compute the damping properties for the absorbing layer
    coeff_damp_fun()
        Compute the coefficients for quadratic damping function
    est_min_damping()
        Estimate the minimum damping ratio and the associated heuristic factor
    min_reflection()
        Compute a minimum reflection coefficiente for the quadratic damping
    regression_CRmin()
        Define the minimum damping ratio and the associated heuristic factor
    xCR_search_range()
        Determine the initial search range for the heuristic factor xCR
    '''

    def __init__(self):
        '''
        Initialize the HABC_Damping class

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        pass

    def min_reflection(self, kCR, psi=None, p=None, CR_err=None, typ='CR_PSI'):
        '''
        Compute a minimum reflection coefficient for the quadratic damping.

        Parameters
        ----------
        kCR : `float`
            Adimensional parameter in reflection coefficient
        psi : `float`, optional
            Damping ratio in option 'CR_PSI'. Default is None
        p : `list`, optional
            Dimensionless wavenumbers for fundamental mode.
            p = [p1, p2, ele_type]. Default is None
            - p1 : Dimensionless wavenumber at the original domain boundary
            - p2 : Dimensionless wavenumber at the begining of absorbing layer
            - ele_type : Element type. 'consistent' or 'lumped'
        CR_err : `float`, optional
            Reflection coefficient in option 'CR_err'. Default is None
        typ : `string`, optional
            Type of reflection coefficient. Default is 'CR_PSI'.
            - 'CR_PSI' : Minimum coefficient reflection from a damping ratio
            - 'CR_FEM' : Spourious reflection coeeficient in FEM
            - 'CR_ERR' : Correction for the minimum damping ratio

        Returns
        -------
        CRmin : `float`
            Minimum reflection coefficient at the minimum damping
        xCRmin : `float`
            Heuristic factor for the minimum damping ratio
        '''

        if typ == 'CR_FEM' or typ == 'CR_ERR':
            def psi_from_CR(CR, kCR):
                '''
                Compute the damping ratio from the reflection coefficient.

                Parameters
                ----------
                CR : `float`
                    Reflection coefficient
                kCR : `float`
                    Adimensional parameter in reflection coefficient

                Returns
                -------
                psi : `float`
                    Damping ratio
                '''
                if CR == 0:
                    psi = 0
                elif CR >= 1:
                    psi = 0.999
                else:
                    psi = kCR / (1 / CR - 1)**0.5

                return psi

        if typ == 'CR_PSI':
            # Minimum coefficient reflection
            psimin = psi
            CRmin = psimin**2 / (kCR**2 + psimin**2)

        elif typ == 'CR_FEM':
            # Unidimensional spourious reflection in FEM (Laier, 2020)
            p1, p2, ele_type = p  # Dimensionless wavenumbers

            def Zi(p, alpha, ele_type):
                '''
                Compute the Z parameter in the spurious reflection coefficient.

                Parameters
                ----------
                p : `float`
                    Dimensionless wavenumber
                alpha : `float`
                    Ratio between the representative mesh dimensions
                ele_type : `string`
                    Element type. 'consistent' or 'lumped'

                Returns
                -------
                Z_fem : `float`
                    Parameter for the spurious reflection coefficient
                '''
                if ele_type == 'lumped':
                    m1 = 1 / 2
                    m2 = 0.
                elif ele_type == 'consistent':
                    m1 = 1 / 3
                    m2 = 1 / 6
                else:
                    aux0 = "Please use 'lumped' or 'consistent', "
                    UserWarning(aux0 + f"{ele_type} not supported.")

                Z_fem = m2 * (np.cos(alpha * p) - 1) / (
                    m1 * (np.cos(alpha * p) + 1))

                return Z_fem

            # Spurious reflection coefficient in FDM (Kar and Turco, 1995)
            CRfdm = np.tan(p1 / 4)**2

            # Minimum damping ratio for the spurious reflection
            psimin = psi_from_CR(CRfdm, kCR)

            # Correction for the dimensionless wavenumbers due to the damping
            p2 *= (1 + 1 / 8 * (psimin * self.a_par / self.F_L)**2)

            # Ratio between the representative mesh dimensions
            alpha = self.lmax / self.lmin

            # Zi parameters for the spurious reflection coefficient
            Z1 = Zi(p1, alpha, ele_type)
            Z2 = Zi(p2, alpha, ele_type)

            # Spurious reflection coefficient in FEM (Laier, 2020)
            aux0 = (1 - Z1) * np.sin(p1)
            aux1 = (alpha * Z2 - 1) * np.sin(alpha * p2) / alpha
            CRmin = abs((aux0 + aux1) / (aux0 - aux1))

        elif typ == 'CR_ERR':
            # Minimum damping ratio for correction in reflection parameters
            psimin = psi_from_CR(CR_err, kCR)

        xCRmin = psimin / self.d

        if typ == 'CR_PSI' or typ == 'CR_FEM':
            return CRmin, xCRmin
        else:
            return xCRmin

    def regression_CRmin(self, dat_reg, xCR_lim, kCR):
        '''
        Define the minimum damping ratio and the associated heuristic factor.

        Parameters
        ----------
        dat_reg : `list`
            Data for regression. Structure: [x_reg, y_reg]
            - x_reg : Values for the heuristic factor
            - y_reg : Values for the minimum damping ratio
        xCR_lim : `list`
            Limits for the heuristic factor. [xCR_inf, xCR_sup, xCR_ini]
        kCR : `float`
            Adimensional parameter in reflection coefficient

        Returns
        -------
        psi_min : `float`
            Minimum damping ratio of the absorbing layer (psi_min = xCR * d)
        xCR_est : `float`
            Estimated heuristic factor for the minimum damping ratio
        CRmin : `float`
            Minimum reflection coefficient at the minimum damping ratio
        '''

        # Data for regression
        x, y = dat_reg

        # Limits for the heuristic factor
        xCR_inf, xCR_sup, xCR_ini = xCR_lim

        # Coefficients for the quadratic equation
        z = np.polyfit(x, y, 2)

        # Roots of the quadratic equation
        roots = np.roots(z)

        # Vertex or minimum positive root
        xCR_vtx = -z[1] / (2 * z[0])
        max_root = max(roots)
        xCR_est = xCR_vtx if xCR_vtx > xCR_inf else (
            max_root if max_root > xCR_inf else xCR_ini)
        xCR_est = np.clip(xCR_est, xCR_inf, xCR_sup)

        # Minimum damping ratio
        psi_min = xCR_est * self.d
        CRmin = self.min_reflection(kCR, psi=psi_min)[0]

        return psi_min, xCR_est, CRmin

    def xCR_search_range(self, CRmin, kCR, p, xCR_lim):
        '''
        Determine the initial search range for the heuristic factor xCR.

        Parameters
        ----------
        CRmin : `float`
            Minimum reflection coefficient at the minimum damping
        kCR : `float`
            Adimensional parameter in reflection coefficient
        p : `list`
            Dimensionless wavenumbers for fundamental mode
        xCR_lim : `list`
            Limits for the heuristic factor. [xCR_inf, xCR_sup]

        Returns
        -------
        xCR_search : `list`
            Initial search range for the heuristic factor. [xCR_min, xCR_max]
            - xCR_min : Lower bound on the search range
            - xCR_max : Upper bound on the search range
        '''

        # Dimensionless wavenumbers
        p1, p2 = p

        # Limits for the heuristic factor
        xCR_inf, xCR_sup = xCR_lim

        # Errors: Spurious reflection rates (Matsuno, 1966)
        err1 = abs(np.sin(np.asarray([p1, p2]) / 2)).max()
        err2 = abs(-1 + np.cos([p1, p2])).max()

        # Correction by spurious reflection
        CR_err_min = CRmin * (1 - min(err1, err2))
        xCR_lb = self.min_reflection(kCR, CR_err=CR_err_min, typ='CR_ERR')
        CR_err_max = CRmin * (1 + max(err1, err2))
        xCR_ub = self.min_reflection(kCR, CR_err=CR_err_max, typ='CR_ERR')
        xCR_min = np.clip(max(xCR_lb, xCR_inf), xCR_inf, xCR_sup)
        xCR_max = np.clip(min(xCR_ub, xCR_sup), xCR_inf, xCR_sup)

        # Model dimensions
        a_rect = self.Lx_habc
        b_rect = self.Lz_habc

        # Axpect ratio for 2D: a/b
        Rab = a_rect / b_rect

        if self.dimension == 2:  # 2D

            # Area factor 0 < f_Ah <= 4
            f_Ah = self.f_Ah

            # Factors and their inverses from sqrt(1/a^2 + 1/b^2)
            fa = (1 + Rab**2)**0.5  # Factoring 1/a^2
            fainv = 1 / fa
            fb = (1 + Rab**2)**0.5 / Rab  # Factoring 1/b^2
            fbinv = 1 / fb
            fmin = f_Ah / 4 * min(fainv, fbinv)
            fmax = 4 / f_Ah * max(fa, fb)

        if self.dimension == 3:  # 3D

            # Adding a dimension for 3D
            c_rect = self.Ly_habc

            # Aspect ratios for 3D: a/b, b/c and a/c
            Rac = a_rect / c_rect
            Rbc = b_rect / c_rect

            # Volume factor 0 < f_Vh <= 8
            f_Vh = self.f_Vh

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
        xCR_min = np.clip(xCR_min * fmin, xCR_inf, xCR_sup)
        xCR_max = np.clip(xCR_max * fmax, xCR_inf, xCR_sup)

        return [xCR_min, xCR_max]

    def est_min_damping(self, psi=0.999, m=1):
        '''
        Estimate the minimum damping ratio and the associated heuristic factor.
        Obs: The reflection coefficient is not zero because there are always
        both reflections: physical and spurious

        Parameters
        ----------
        psi : `float`, optional
            Damping ratio. Default is 0.999
        m : `int`, optional
            Vibration mode. Default is 1 (Fundamental mode)

        Returns
        -------
        psi_min : `float`
            Minimum damping ratio of the absorbing layer (psi_min = xCR * d)
        xCR_est : `float`
            Estimated heuristic factor for the minimum damping ratio
        xCR_lim : `list`
            Limits for the heuristic factor. [xCR_inf, xCR_sup]
        xCR_search : `list`
            Initial search range for the heuristic factor. [xCR_min, xCR_max]
        '''

        # Dimensionless wave numbers
        c_ref = min([bnd[1] for bnd in self.eik_bnd])
        pmin = 2 * np.pi * self.freq_ref * self.lmin / c_ref
        pmax = 2 * np.pi * self.freq_ref * self.lmax / c_ref

        # Adimensional parameter in reflection coefficient
        kCR = 4 * self.F_L / (self.a_par * m)
        c_bnd = self.eik_bnd[0][1]
        kCRp = kCR * c_bnd / c_ref

        # Lower Limit for the minimum damping ratio
        psimin_inf = psi * self.d**2
        CRmin_inf, xCR_inf = self.min_reflection(kCRp, psi=psimin_inf)

        # Upper Limit for the minimum damping ratio
        psimin_sup = psi * (2. * self.d - self.d**2)
        CRmin_sup, xCR_sup = self.min_reflection(kCRp, psi=psimin_sup)

        # Initial guess
        psimin_ini = psi * (self.d**2 + self.d) / 2.
        CRmin_ini, xCR_ini = self.min_reflection(kCRp, psi=psimin_ini)

        # Spurious reflection
        p = [pmin, pmax, self.variant]
        CRmin_fem, xCR_fem = self.min_reflection(kCRp, p=p, typ='CR_FEM')

        # Minimum damping ratio
        x_reg = [xCR_inf, xCR_sup, xCR_ini, xCR_fem]
        y_reg = [CRmin_inf, CRmin_sup, CRmin_ini, CRmin_fem]
        dat_reg = [x_reg, y_reg]
        xCR_lim = [xCR_inf, xCR_sup, xCR_ini]
        psi_min, xCR_est, CRmin = self.regression_CRmin(dat_reg, xCR_lim, kCRp)

        xCR_search = self.xCR_search_range(CRmin, kCRp, p[:2], xCR_lim[:2])

        return psi_min, xCR_est, xCR_lim[:2], xCR_search

    def calc_damping_prop(self, psi=0.999, m=1, get_init_search_range=False):
        '''
        Compute the damping properties for the absorbing layer.

        Parameters
        ----------
        psi : `float`, optional
            Damping ratio. Default is 0.999
        m : `int`, optional
            Vibration mode. Default is 1 (Fundamental mode)
        get_init_search_range : `bool`, optional
            If True, returns the initial search range for the heuristic factor.
            Default is False.

        Returns
        -------
        eta_crt : `float`
            Critical damping coefficient (1/s)
        psi_min : `float`
            Minimum damping ratio of the absorbing layer (psi_min = xCR * d)
        xCR_est : `float`
            Estimated heuristic factor for the minimum damping ratio
        psimin_lim : `list`
            Limits for the minimum damping ratio. [psimin_inf, psimin_sup]
        xCR_lim : `list`
            Limits for the heuristic factor. [xCR_inf, xCR_sup]
        xCR_search : `list`, optional
            Initial search range for the heuristic factor. [xCR_min, xCR_max]
        '''

        # Critical damping coefficient
        eta_crt = 2 * np.pi * self.fundam_freq
        eta_max = psi * eta_crt
        print("Critical Damping Coefficient (1/s): {0:.5f}".format(eta_crt))
        print("Maximum Damping Ratio: {0:.3%}".format(psi))
        print("Maximum Damping Coefficient (1/s): {0:.5f}".format(eta_max))

        # Minimum damping ratio and the associated heuristic factor
        psi_min, xCR_est, xCR_lim, xCR_search = self.est_min_damping()
        xCR_inf, xCR_sup = xCR_lim
        xCR_min, xCR_max = xCR_search

        # Computed values and its range
        print("Minimum Damping Ratio: {:.3%}".format(psi_min))
        psi_str = "Range for Minimum Damping Ratio. Min:{:.5f} - Max:{:.5f}"
        print(psi_str.format(xCR_inf * self.d, xCR_sup * self.d))
        print("Estimated Heuristic Factor xCR: {:.3f}".format(xCR_est))
        xcr_str = "Range Values for xCR Factor. Min:{:.3f} - Max:{:.3f}"
        print(xcr_str.format(xCR_inf, xCR_sup))

        if get_init_search_range:
            lim_str = "Initial Range for xCR Factor. Min:{:.3f} - Max:{:.3f}"
            print(lim_str.format(xCR_min, xCR_max))
            return eta_crt, psi_min, xCR_est, xCR_lim, xCR_search

        else:
            return eta_crt, psi_min, xCR_est, xCR_lim

    def coeff_damp_fun(self, psi_min, psi=0.999):
        '''
        Compute the coefficients for quadratic damping function

        Parameters
        ----------
        psi_min' : `float`
            Minimum damping ratio
        psi : `float`, optional
            Damping ratio. Default is 0.999

        Returns
        -------
        aq : `float`
            Coefficient for quadratic term in the damping function
        bq : `float`
            Coefficient bq for linear term in the damping function
        '''

        aq = (psi_min - self.d * psi) / (self.d**2 - self.d)
        bq = psi - aq

        return aq, bq
