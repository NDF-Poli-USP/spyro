import numpy as np
from scipy.integrate import dblquad, quad
from scipy.special import beta, betainc, gamma
from spyro.utils.error_management import value_dimension_error

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class HyperLayer():
    '''
    Define a hyperlliptical layer in 2D or hyperellipsoidal in 3D.
    Hyperellipse Eq. (2D): |x/a|^n + |y/b|^n = 1.
    Hyperellipsoid Eq. (3D): |x/a|^n + |y/b|^n + |z/c|^n = 1.
        - a, b, c: semi-axes of the hypershape
        - n: degree of the hypershape
        - x, y, z: coordinates relative to the hypershape centroid.

    Attributes
    ----------
    area : `float`
        Area of the domain with hyperelliptical layer
    a_rat : `float`
        Area ratio to the area of the original domain. a_rat = area / a_orig
    dimension : `int`
        Model dimension (2D or 3D). Default is 2D
    f_Ah : `float`
        Hyperelliptical area factor. f_Ah = area / (a_hyp b_hyp)
    f_Vh : `float`
        Hyperellipsoidal volume factor. f_Vh = vol / (a_hyp b_hyp c_hyp)
    hyper_axes : `tuple`
        Semi-axes of the hypershape layer (a, b) (2D) or (a, b, c) (3D)
    n_hyp: `float`
        Degree of the hypershape pad layer (n >= 2). Default is 2
    n_bounds: `tuple`
        Bounds for the hypershape layer degree. (n_min, n_max)
        - n_min ensures to add lmin in the domain diagonal direction
        - n_max ensures to add pad_len in the domain diagonal direction
        where lmin is the minimum mesh size and pad_len is the layer size
    n_type : `str`
        Type of the hypereshape degree ('real' or 'integer'). Default is 'real'
    perim_hyp : `float`
        Perimeter of the full hyperellipse (only 2D)
    surf_hyp : `float`
        Surface area of the full hyperellipsoidf (only 3D)
    vol : `float`
        Volume of the domain with hyperellipsoidal layer
    v_rat : `float`
        Volume ratio to the volume of the original domain. v_rat = vol / v_orig

    Methods
    -------
    calc_degree_hypershape()
        Define the limits for the hypershape degree. See Salas et al (2022)
    calc_hyp_geom_prop()
        Calculate the geometric properties for the hypershape layer
    central_tendency_criteria()
        Central tendency criteria to find the hypershape degree
    define_hyperaxes()
        Define the hyperlayer semi-axes
    define_hyperlayer()
        Define the hyperlayer degree and its limits
    half_hyp_area()
        Compute half the area of the hyperellipse
    half_hyp_volume()
        Compute half the volume of the hyperellipsoid
    loop_criteria()
        Loop criteria to find the hypershape degree
    radial_parameter()
        Calculate the radial parameter for a hypershape
    trunc_half_hyp_area()
        Compute the truncated area of superellipse for 0 <= z0 / b <= 1
    trunc_half_hyp_volume()
        Compute the truncated volume of hyperellipsoid for 0 <= z0 / b <= 1
    '''

    def __init__(self, n_hyp=2, n_type='real', dimension=2):
        '''
        Initialize the HyperLayer class

        Parameters
        ----------
        n_hyp : `int`, optional
            Hypershape degree. Default is 2
        n_type : `str`, optional
            Type of the hypereshape degree ('real' or 'integer').
            Default is 'real'
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D

        Returns
        -------
        None
        '''

        # Hypershape degree
        self.n_hyp = n_hyp

        # Model dimension
        self.dimension = dimension

        # Type of the hypereshape degree
        self.n_type = n_type

        if n_type not in ['real', 'integer']:
            value_parameter_error('n_type', n_type, ['real', 'integer'])

    def define_hyperaxes(self, dom_hyp):
        '''
        Define the hyperlayer semi-axes

        Parameters
        ----------
        dom_hyp : `tuple`
            Domain dimension with layer without truncation by free surface.
            - 2D : (Lx + 2 * pad_len, Lz + 2 * pad_len)
            - 3D : (Lx + 2 * pad_len, Lz + 2 * pad_len, Ly + 2 * pad_len)

        Returns
        -------
        None
        '''

        # Hypershape semi-axes
        a_hyp = 0.5 * dom_hyp[0]
        b_hyp = 0.5 * dom_hyp[1]
        self.hyper_axes = (a_hyp, b_hyp)

        if self.dimension == 3:  # 3D
            c_hyp = 0.5 * dom_hyp[2]
            self.hyper_axes += (c_hyp,)

    def radial_parameter(self, rel_coordinates, n):
        '''
        Calculate the radial parameter for a hypershape

        Parameters
        ----------
        rel_coordinates : `tuple`
            coordinates relative to the hypershape centroid
            - 2D : (x_rel, y_rel)
            - 3D : (x_rel, y_rel, z_rel)
        n : `float`
            Degree of the hypershape

        Returns
        -------
        r : `float`
            Radial parameter
        '''

        # Hyperellipse semi-axes
        a, b = self.hyper_axes[:2]

        # Relative coordinates
        x_rel, y_rel = rel_coordinates[:2]

        # Radial parameter
        r = abs(x_rel / a)**n + abs(y_rel / b)**n

        if self.dimension == 3:  # 3D
            c = self.hyper_axes[2]
            z_rel = rel_coordinates[2]
            r += abs(z_rel / c)**n

        return r

    def central_tendency_criteria(self, spness, monitor=False):
        '''
        Central tendency criteria to find the hypershape degree.
        See Salas et al (2022)

        Parameters
        ----------
        spness : `tuple`
            Superness coordinates relative to the hypershape centroid
            - 2D : (xs, ys)
            - 3D : (xs, ys, zs)
        monitor : `bool`, optional
            Print the process on the screen. Default is False

        Returns
        -------
        h, g, z : `float`
            Degree of hypershape by applying criteria with
            harmonic, geometric, and arithmetic means
        '''

        # Hyperellipse semi-axes
        a, b = self.hyper_axes

        # Superness s = 2^(-1/n): Extreme points of the hyperellipse
        xs, ys = spness[:2]

        # Harmonic mean parameters
        sum_invhaxs = 1 / a + 1 / b
        sum_invsnss = 1 / xs + 1 / ys

        # Geometric mean parameters
        prod_haxs = a * b
        prod_snss = xs * ys

        # Arithmetic mean parameters
        sum_haxs = a + b
        sum_snss = xs + ys

        if self.dimension == 2:  # 2D
            # Harmonic mean parameters
            f_har = 1 / 2

            # Geometric mean parameters
            f_geo = 1 / 4

            # Arithmetic mean parameters
            f_ari = 1 / 2

        if self.dimension == 3:  # 3D
            zs = spness[2]

            # Harmonic mean parameters
            f_har = 1 / 3
            sum_invhaxs += 1 / c
            sum_invsnss += 1 / zs

            # Geometric mean parameters
            f_geo = 1 / 27
            prod_haxs *= c
            prod_snss *= zs

            # Arithmetic mean parameters
            f_ari = 1 / 3
            sum_haxs += c
            sum_snss += zs

        # Harmonic mean
        h = max(np.log(f_har) / np.log(sum_invhaxs / sum_invsnss), 2)
        h = np.ceil(h * 10.) / 10. if self.n_type == 'real' else np.ceil(h)
        rh = self.radial_parameter(spness, h)

        # Geometric mean
        g = max(np.log(f_geo) / np.log(prod_snss / prod_haxs), 2)
        g = np.ceil(g * 10.) / 10. if self.n_type == 'real' else np.ceil(g)
        rg = self.radial_parameter(spness, g)

        # Arithmetic mean
        z = max(np.log(f_ari) / np.log(sum_snss / sum_haxs), 2)
        z = np.ceil(z * 10.) / 10. if self.n_type == 'real' else np.ceil(z)
        rz = self.radial_parameter(spness, z)

        if monitor:
            # Central tendency criteria
            print("'Harm' Superness. r: {:>5.3f} - n: {:>.1f}".format(rh, h))
            print("'Geom' Superness. r: {:>5.3f} - n: {:>.1f}".format(rg, g))
            print("'Arit' Superness. r: {:>5.3f} - n: {:>.1f}".format(rz, z))

        return h, g, z

    def loop_criteria(self, spness, n_min=2, n_max=20, monitor=False):
        '''
        Loop criteria to find the hypershape degree

        Parameters
        ----------
        spness : `tuple`
            Superness coordinates relative to the hypershape centroid
            - 2D : (xs, ys)
            - 3D : (xs, ys, zs)
        n_min : `float`, optional
            Minimum allowed degree. Default is 2
        n_max : `float`, optional
            Maximum allowed degree. Default is 20
        monitor : `bool`, optional
            Print the process on the screen. Default is False

        Returns
        -------
        n : `float`
            Hypereshape degree
        n_min : `float`
            Minimum allowed degree for the loop criteria
        n_max : `float`
            Maximum allowed degree for the loop criteria
        '''

        # Integer limits (difference of at least 1)
        n_minint = max(int(n_min), 2)
        n_maxint = max(int(n_max + 1), n_min + 1, 20)

        # Integer loop
        r = np.inf
        n = n_minint - 1
        while r > 1 and n < n_maxint:
            n += 1
            r = self.radial_parameter(spness, n)
            if monitor:
                print("ParHypEll - r: {:>5.3f} - n: {:>.1f}".format(r, n))

        # Real limits (difference of at least 1)
        n_min = max(round(float(n_min), 1), 2.)
        n_max = max(round(float(n_max), 1), n_min + 1., 20.)

        # Real loop
        if self.n_type == 'real' and n > 2:
            r = np.inf
            n_maxreal = n
            n -= 1
            while r > 1 and n < n_maxreal:
                n += 0.1
                r = self.radial_parameter(spness, round(n, 1))
                if monitor:
                    print("ParHypEll - r: {:>5.3f} - n: {:>.1f}".format(r, n))

        return float(n), n_min, n_max

    def calc_degree_hypershape(self, spness, lim, n_min=2,
                               n_max=20, monitor=False):
        '''
        Define the limits for the hypershape degree. See Salas et al (2022).
        The condition r = 1 ensures that the point is on the layer boundary.
        The condition r < 1 ensures that the point is inside the layer

        Parameters
        ----------
        spness : `tuple`
            Superness coordinates relative to the hypershape centroid
            - 2D : (xs, ys)
            - 3D : (xs, ys, zs)
        lim : `str`
            Limit for the hypershape degree ('MIN' or 'MAX')
        n_min : `float`, optional
            Minimum allowed degree. Default is 2
        n_max : `float`, optional
            Maximum allowed degree. Default is 20
        monitor : `bool`, optional
            Print the process on the screen. Default is False

        Returns
        -------
        n : `float`
            Hypereshape degree
        '''

        # Loop criteria to find the hyperellipse degree
        n, n_min, n_max = self.loop_criteria(spness, n_min=n_min,
                                             n_max=n_max, monitor=monitor)

        # Central tendency criteria to find the hyperellipse degree
        h, g, z = self.central_tendency_criteria(spness, monitor=monitor)

        if lim == 'MIN':
            n = float(np.clip(max(n, h, g, z), n_min, n_max))
        elif lim == 'MAX':
            n = float(np.clip(min(n, h, g, z), np.floor(n_min) + 1., n_max))

        if monitor:

            # Final hypershape parameters
            r = self.radial_parameter(spness, round(n, 1))

            if self.dimension == 2:  # 2D
                shp_str = "Hyperellipse"

            if self.dimension == 3:  # 3D
                shp_str = "Hyperellipsoid"

            lim_str = "min" if lim == 'MIN' else "max"
            pr0_str = shp_str + " Parameters. r_" + lim_str
            pr1_str = ": {:>5.3f} - n_" + lim_str + ": {:>.1f}"
            print(pr0_str + pr1_str.format(r, n))

        # Superness s = 2^(-1/n): Extreme points of the hyperellipse
        snss_str = "Superness Coordinates (km): ({:5.3f}, {:5.3f})"
        if self.dimension == 3:  # 3D
            # Superness s = 3^(-1/n): Extreme points of the hyperellipsoid
            snss_str = snss_str[:-1] + ", {:5.3f})"
        print(snss_str.format(*spness))

        return n

    def define_hyperlayer(self, dom_dim, pad_len, lmin, monitor=False):
        '''
        Define the hyperlayer degree and its limits

        Parameters
        ----------
        dom_dim : `tuple`
            Original domain dimensions: (Lx, Lz) for 2D or (Lx, Lz, Ly) for 3D
        pad_len : `float`
            Size of the absorbing layer
        lmin : `float`
            Minimum mesh size
        monitor : `bool`, optional
            Print the process on the screen. Default is False

        Returns
        -------
        None
        '''

        # Domain dimensions
        Lx, Lz = dom_dim[:2]

        # Hyperellipse degree
        n_hyp = self.n_hyp

        # Verification of hypershape degree
        ndeg_str = "Checking Current Hypershape Degree n_hyp: {:>.1f}"
        print(ndeg_str.format(n_hyp))
        axes_str = "Semi-axes (km): a_hyp:{:5.3f} - b_hyp:{:5.3f}"
        if self.dimension == 3:  # 3D
            Ly = dom_dim[2]
            axes_str += " - c_hyp:{:5.3f}"
        print(axes_str.format(*self.hyper_axes))

        # Minimum allowed exponent
        # n_min ensures to add lmin in the domain diagonal direction
        print("Determining the Minimum Degree for Hypershape Layer")
        x_min = (0.5 * Lx + lmin, 0.5 * Lz + lmin)

        if self.dimension == 3:  # 3D
            x_min += (0.5 * Ly + lmin,)

        n_min = self.calc_degree_hypershape(x_min, 'MIN', monitor=monitor)
        print("Minimum Degree for Hypershape n_min: {:>.1f}".format(n_min))

        # Maximum allowed exponent
        # n_max ensures to add pad_len in the domain diagonal direction
        print("Determining the Maximum Degree for Hypershape Layer")
        theta = np.arctan2(Lz, Lx)

        if self.dimension == 2:  # 2D
            x_max = (0.5 * Lx + pad_len * np.cos(theta),
                     0.5 * Lz + pad_len * np.sin(theta))

        if self.dimension == 3:  # 3D
            phi = np.arccos(Ly / np.sqrt(Lx**2 + Lz**2 + Ly**2))
            x_max = (0.5 * Lx + pad_len * np.cos(theta) * np.sin(phi),
                     0.5 * Lz + pad_len * np.sin(theta) * np.sin(phi),
                     0.5 * Ly + pad_len * np.cos(phi))

        n_max = self.calc_degree_hypershape(
            x_max, 'MAX', n_min=n_min, monitor=monitor)
        print("Maximum Degree for Hypershape n_max: {:>.1f}".format(n_max))

        if n_min <= n_hyp <= n_max:
            print("Current Hypershape Degree n_hyp: {:>.1f}".format(n_hyp))
        else:
            hyp_str = "Degree for Hypershape Layer. Setting to"
            if n_hyp < n_min:
                print("Low", hyp_str, "n_min: {:>.1f}".format(n_min))
            elif n_hyp > n_max:
                print("High", hyp_str, "n_max: {:>.1f}".format(n_max))

        self.n_hyp = np.clip(n_hyp, n_min, n_max)
        self.n_bounds = (n_min, n_max)

    @staticmethod
    def trunc_half_hyp_area(a, b, n, z0):
        '''
        Compute the truncated area of hyperellipse for 0 <= z0 / b <= 1.
        Verify with self.trunc_half_hyp_area(1, 1, 2, 1) = pi / 2

        Parameters
        ----------
        a : `float`
            Major hyperellipse semi-axis
        b : `float`
            Minor hyperellipse semi-axis
        n : `float`
            Degree of the hyperellipse
        z0 : `float`
            Truncation plane

        Returns
        -------
        A_tr : `float`
            Truncated area of the hyperellipse
        '''

        if z0 < 0 or z0 > b:
            err = f"Truncation plane must be 0 <= z0 <= {b:.3f}, got {z0:.3f}"
            raise ValueError(err)

        w = (z0 / b) ** n  # w <= 1
        p = 1 / n
        q = 1 + 1 / n
        B_w = beta(p, q) * betainc(p, q, w)  # Non-regularized B_z(p, q)
        A_tr = (2 * a * b / n) * B_w

        return A_tr

    @staticmethod
    def half_hyp_area(a, b, n):
        '''
        Compute half the area of the hyperellipse.
        Verify with self.half_hyp_area(1, 1, 2) = pi / 2

        Parameters
        ----------
        a : `float`
            Hyperellipse semi-axis in direction 1
        b : `float`
            Hyperellipse semi-axis in direction 2
        n : `float`
            Degree of the hyperellipse

        Returns
        -------
        A_hf : `float`
            Half the area of the hyperellipse
        '''

        A_hf = 2 * a * b * gamma(1 + 1 / n)**2 / gamma(1 + 2 / n)

        return A_hf

    @staticmethod
    def trunc_half_hyp_volume(a, b, c, n, z0):
        '''
        Compute the truncated volume of hyperellipsoid for 0 <= z0 / b <= 1.
        Verify with self.trunc_half_hyp_volume(1, 1, 1, 2, 1) = 2 * pi / 3

        Parameters
        ----------
        a : `float`
            Hyperellipsoid semi-axis in direction 1
        b : `float`
            Hyperellipsoid semi-axis in truncated direction 2
        c : `float`
            Hyperellipsoid semi-axis in direction 3
        n : `float`
            Degree of the hyperellipsoid
        z0 : `float`
            Truncation plane

        Returns
        -------
        A_tr : `float`
            Truncated volume of the hyperellipsoid
        '''

        if z0 < 0 or z0 > b:
            err = f"Truncation plane must be 0 <= z0 <= {b:.3f}, got {z0:.3f}"
            raise ValueError(err)

        w = (z0 / b) ** n  # w <= 1
        p = 1 / n
        q = 1 + 2 / n
        A_f = gamma(1 + p)**2 / gamma(q)
        B_w = beta(p, q) * betainc(p, q, w)  # Non-regularized B_z(p, q)
        V_tr = (4 * a * b * c / n) * A_f * B_w

        return V_tr

    @staticmethod
    def half_hyp_volume(a, b, c, n):
        '''
        Compute half the volume of the hyperellipsoid.
        Verify with self.half_hyp_volume(1, 1, 1, 2) = 2 * pi / 3

        Parameters
        ----------
        a : `float`
            Hyperellipsoid semi-axis in direction 1
        b : `float`
            Hyperellipsoid semi-axis in direction 2
        c : `float`
            Hyperellipsoid semi-axis in direction 3
        n : `float`
            Degree of the hyperellipsoid

        Returns
        -------
        V_hf : `float`
            Half the volume of the hyperellipsoid
        '''

        V_hf = 4 * a * b * c * gamma(1 + 1 / n)**3 / gamma(1 + 3 / n)

        return V_hf

    @staticmethod
    def hyp_full_perimeter(a, b, n):
        '''
        Compute perimeter of a hyperellipse

        Parameters
        ----------
        a : `float`
            Hyperellipse semi-axis in direction 1
        b : `float`
            Hyperellipse semi-axis in direction 2
        n : `float`
            Degree of the hyperellipse

        Returns
        -------
        perim_hyp : `float`
            Perimeter of the hyperellipse
        '''

        def integrand(theta):
            '''
            Differential arc length element to compute the perimeter

            Parameters
            ----------
            theta : `float`
                Angle in radians

            Returns
            -------
            ds : `float`
                Differential arc length element
            '''
            dx = -(2 * a / n) * np.cos(theta)**(2 / n - 1) * np.sin(theta)
            dy = (2 * b / n) * np.sin(theta)**(2 / n - 1) * np.cos(theta)
            ds = np.sqrt(dx**2 + dy**2)

            return ds

        # Integrate from 0 to pi/2 and multiply by 4
        perim_hyp = 4 * quad(integrand, 0, np.pi/2)[0]

        return perim_hyp

    def hyp_full_surf_area(a, b, c, n):
        '''
        Compute the surface area of a hyperellipsoid.

        Parameters
        ----------
        a : `float`
            Hyperellipsoid semi-axis in direction 1
        b : `float`
            Hyperellipsoid semi-axis in direction 2
        c : `float`
            Hyperellipsoid semi-axis in direction 3
        n : `float`
            Degree of the hyperellipsoid

        Returns
        -------
        surf_area : `float`
            Surface area of the hyperellipsoid
        '''

        def surface_element(r, t):
            '''
            Differential surface element to compute the surface area

            Parameters
            ----------
            r : `float`
                Azimuth angle (equatorial direction)
            t : `float`
                Polar angle (meridional direction)

            Returns
            -------
            dS : `float`
                Differential surface area element
            '''

            # Trigonometric functions
            trig1 = np.cos(r) * np.sin(t)
            trig2 = np.sin(r) * np.sin(t)
            trig3 = np.cos(r) * np.cos(t)
            trig4 = np.sin(r) * np.cos(t)

            # Parametric equations (first octant, no abs/sgn needed)
            x = a * (trig1)**(2 / n)
            y = b * (trig2)**(2 / n)
            z = c * np.cos(t)**(2 / n)

            # Partial derivatives (simplified)
            dx_dr = a * (2 / n) * (trig1)**(2/n - 1) * (-trig2)
            dy_dr = b * (2 / n) * (trig2)**(2/n - 1) * (trig1)
            dz_dr = 0.0

            dx_dt = a * (2 / n) * (trig1)**(2 / n - 1) * (trig3)
            dy_dt = b * (2 / n) * (trig2)**(2 / n - 1) * (trig4)
            dz_dt = c * (2 / n) * (np.cos(t))**(2 / n - 1) * (-np.sin(t))

            # Cross product and magnitude
            F1 = dy_dr * dz_dt  # dz_dr = 0
            F2 = -dx_dr * dz_dt  # dz_dr = 0
            F3 = dx_dr * dy_dt - dy_dr * dx_dt
            dS = np.sqrt(F1**2 + F2**2 + F3**2)

            return dS

        # Integrate over first octant and multiply by 8
        # r ∈ [0, π/2] and t ∈ [0, π/2]
        surf_hyp = dblquad(lambda t, r: surface_element(r, t),
                           0, np.pi/2,  # t limits
                           lambda r: 0, lambda r: np.pi / 2)[0]  # r limits

        return 8 * surf_hyp

    def calc_hyp_geom_prop(self, dom_dim, dom_hyp, pad_len, lmin):
        '''
        Calculate the geometric properties for the hypershape layer

        Parameters
        ----------
        dom_dim : `tuple`
            Original domain dimensions: (Lx, Lz) for 2D or (Lx, Lz, Ly) for 3D
        dom_hyp : `tuple`
            Domain dimension with layer without truncation by free surface.
            - 2D : (Lx + 2 * pad_len, Lz + 2 * pad_len)
            - 3D : (Lx + 2 * pad_len, Lz + 2 * pad_len, Ly + 2 * pad_len)
        pad_len : `float`
            Size of the absorbing layer
        lmin : `float`
            Minimum mesh size

        Returns
        -------
        None
        '''

        # Domain dimensions w/o layer
        chk_domd = len(dom_dim)
        chk_habc = len(dom_hyp)
        if self.dimension != chk_domd or self.dimension != chk_habc:
            value_dimension_error(('dom_dim', 'dom_hyp'),
                                  (chk_domd, chk_habc), self.dimension)

        # Defining the hypershape semi-axes
        self.define_hyperaxes(dom_hyp)

        # Degree of the hypershape layer
        self.define_hyperlayer(dom_dim, pad_len, lmin, monitor=True)

        # Hypershape semi-axes and domain dimensions
        a_hyp, b_hyp = self.hyper_axes[:2]
        Lx, Lz = dom_dim[:2]

        # Hyperellipse degree
        n_hyp = self.n_hyp

        # Geometric properties realted to Area (2D) or Volume (3D)
        if self.dimension == 2:  # 2D

            # Area
            self.area = self.half_hyp_area(a_hyp, b_hyp, n_hyp) + \
                self.trunc_half_hyp_area(a_hyp, b_hyp, n_hyp, Lz / 2)

            # Area ratio
            self.a_rat = self.area / (Lx * Lz)
            print("Area Ratio: {:5.3f}".format(self.a_rat))

            # Area factor
            self.f_Ah = self.area / (a_hyp * b_hyp)

            # Full perimeter to estimate the mesh size
            self.perim_hyp = self.hyp_full_perimeter(a_hyp, b_hyp, n_hyp)

        if self.dimension == 3:  # 3D
            c_hyp = self.hyper_axes[2]
            Ly = dom_dim[2]

            # Volume
            self.vol = self.half_hyp_volume(a_hyp, b_hyp, c_hyp, n_hyp) + \
                self.trunc_half_hyp_volume(a_hyp, b_hyp, c_hyp, n_hyp, Lz / 2)

            # Volume ratio
            self.v_rat = self.vol / (Lx * Lz * Ly)
            print("Volume Ratio: {:5.3f}".format(self.v_rat))

            # Volume factor
            self.f_Vh = self.vol / (a_hyp * b_hyp * c_hyp)

            # Full surface area to estimate the mesh size
            self.surf_hyp = self.hyp_full_surf_area(a_hyp, b_hyp, c_hyp, n_hyp)
