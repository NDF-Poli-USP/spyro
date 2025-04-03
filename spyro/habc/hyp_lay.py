import firedrake as fire
import numpy as np
from scipy.special import gamma, beta, betainc
import ipdb
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}

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

    Attributes
    ----------
    area : `float`
        Area of the domain with hyperelliptical layer
    a_rat : `float`
        Area ratio to the area of the original domain. a_rat = area / a_orig
    dimension : int
        Model dimension (2D or 3D). Default is 2D
    domain_dim : list
        Domain dimensions [Lx, Lz] (2D) or [Lx, Lz, Ly] (3D)
    f_Ah : `float`
        Hyperelliptical area factor. f_Ah = area / (a_hyp * b_hyp)
    f_Vh : `float`
        Hyperellipsoidal volume factor. f_Vh = vol / (a_hyp * b_hyp * c_hyp)
    hyper_axes : list
        Semi-axes of the hypershape layer [a, b] (2D) or [a, b, c] (3D)
    n_hyp: `int`
        Degree of the hyperelliptical pad layer (n >= 2). Default is 2.
    n_bounds: `tuple`
        Bounds for the hypershape layer degree. (n_min, n_max)
        - n_min ensures to add lmin in the domain diagonal direction
        - n_max ensures to add pad_len in the domain diagonal direction
        where lmin is the minimum mesh size and pad_len is the layer size
    vol : `float`
        Volume of the domain with hyperellipsoidal layer
    v_rat : `float`
        Volume ratio to the volume of the original domain. v_rat = vol / v_orig

    Methods
    -------
    calc_degree_hyp2D()
        Define the limits for the hyperellipse degree. See Salas et al (2022)
    calc_degree_hyp3D()
        Define the limits for the hyperellipsoid degree
    calc_hyp_geom_prop(self):
        Calculate the geometric properties for the hypershape layer
    define_hyperaxes()
        Define the hyperlayer semi-axes
    define_hyperlayer()
        Define the hyperlayer degree and its limits
    half_hyp_area()
        Compute half the area of the hyperellipse
    half_hyp_volume()
        Compute half the volume of the hyperellipsoid
    trunc_half_hyp_area()
        Compute the truncated area of superellipse for 0 <= z0 / b <= 1
    trunc_half_hyp_volume()
        Compute the truncated volume of hyperellipsoid for 0 <= z0 / b <= 1
    '''

    def __init__(self, n_hyp=2, dimension=2):
        '''
        Initialize the HyperLayer class.

        Parameters
        ----------
        n_hyp : int, optional
            Hypershape degree. Default is 2
        dimension : int, optional
            Model dimension (2D or 3D). Default is 2D

        Returns
        -------
        None
        '''

        # Model dimension
        self.dimension = dimension

        # Hypershape degree
        self.n_hyp = n_hyp

    def define_hyperaxes(self, domain_dim, domain_hyp):
        '''
        Define the hyperlayer semi-axes.

        Parameters
        ----------
        domain_dim : list
            Domain dimensions [Lx, Lz] (2D) or [Lx, Lz, Ly] (3D)
        domain_hyp : list
            Maximum hypershape dimensions without truncations.
            - 2D: [Lx + 2 * pad_len, Lz + 2 * pad_len]
            - 3D: [Lx + 2 * pad_len, Lz + 2 * pad_len, Ly + 2 * pad_len]

        Returns
        -------
        None
        '''

        # Domain dimensions
        chk_domd = len(domain_dim)
        chk_habc = len(domain_hyp)
        if self.dimension == chk_domd and self.dimension == chk_habc:
            self.domain_dim = domain_dim

        else:
            aux0 = "Number of domain dimensions"
            aux1 = "different from the model dimension"
            aux = aux0 + " ({:0}) " + aux1 + " ({:1}D)."
            UserWarning(aux.format(min(chk_domd, chk_habc), self.dimension))

        # Hypershape semi-axes
        a_hyp = 0.5 * domain_hyp[0]
        b_hyp = 0.5 * domain_hyp[1]
        self.hyper_axes = [a_hyp, b_hyp]

        if self.dimension == 3:  # 3D
            c_hyp = 0.5 * domain_hyp[2]
            self.hyper_axes.append(c_hyp)

    def calc_degree_hyp2D(self, x_rel, lim, n_min=2, n_max=20, monitor=False):
        '''
        Define the limits for the hyperellipse degree. See Salas et al (2022).
        The condition r < 1 ensures that the point is inside the layer

        Parameters
        ----------
        x_rel : list
            Relative superness coordinates to the hyperellipse centroid
        lim : str
            Limit for the hyperellipse degree ('MIN' or 'MAX')
        n_min : int, optional
            Minimum allowed degree. Default is 2.
        n_max : int, optional
            Maximum allowed degree. Default is 20.
        monitor : bool, optional
            Print the process on the screen. Default is False.

        Returns
        -------
        n : int
            Hyperellipse degree
        '''

        # Hyperellipse semi-axes
        a, b = self.hyper_axes

        # Superness s= 2^(-1/n): Extreme points of the hyperellipse
        xs, ys = x_rel

        # Harmonic mean
        h = max(int(np.ceil(np.log(0.5) / np.log(
            (1 / a + 1 / b) / (1 / xs + 1 / ys)))), 2)
        rh = abs(xs / a)**h + abs(ys / b)**h

        # Geometric mean
        g = max(int(np.ceil(np.log(0.25) / np.log((xs * ys) / (a * b)))), 2)
        rg = abs(xs / a)**g + abs(ys / b)**g

        # Arithmetic mean
        z = max(int(np.ceil(np.log(0.5) / np.log((xs + ys) / (a + b)))), 2)
        rz = abs(xs / a)**z + abs(ys / b)**z

        r = n = 1
        while r >= 1 and n < n_max:
            n += 1
            r = abs(xs / a)**n + abs(ys / b)**n
            if monitor:
                print('ParHypEll - r: {:>5.3f} - n: {:>}'.format(r, n))

        if lim == 'MIN':
            n = np.clip(max(n, h, g, z), n_min, n_max)
        elif lim == 'MAX':
            n = np.clip(min(n, h, g, z), n_min + 1, n_max)

        if monitor:
            print("'Harm' Superness. r: {:>5.3f} - n: {:>}".format(rh, h))
            print("'Geom' Superness. r: {:>5.3f} - n: {:>}".format(rg, g))
            print("'Arit' Superness. r: {:>5.3f} - n: {:>}".format(rz, z))
            lim_str = "min" if lim == 'MIN' else "max"
            pr0_str = "Hyperellipse Parameters. r_" + lim_str
            pr1_str = ": {:>5.3f} - n_" + lim_str + ": {:>}"
            print(pr0_str + pr1_str.format(r, n))

        print("Superness Coordinates (km): ({:5.3f}, {:5.3f})".format(xs, ys))

        return n

    def calc_degree_hyp3D(self, x_rel, lim, n_min=2, n_max=20, monitor=False):
        '''
        Define the limits for the hyperellipsoid degree.
        The condition r < 1 ensures that the point is inside the layer

        Parameters
        ----------
        x_rel : list
            Relative superness coordinates to the hyperellipsoid centroid
        lim : str
            Limit for the hyperellipsoid degree ('MIN' or 'MAX')
        n_min : int, optional
            Minimum allowed degree. Default is 2.
        n_max : int, optional
            Maximum allowed degree. Default is 20.
        monitor : bool, optional
            Print the process on the screen. Default is False.

        Returns
        -------
        n : int
            Hyperellipsoid degree
        '''

        # Hyperellipsoid semi-axes
        a, b, c = self.hyper_axes

        # Superness s= 3^(-1/n): Extreme points of the hyperellipsoid
        xs, ys, zs = x_rel

        # Harmonic mean
        h = max(int(np.ceil(np.log(1 / 3) / np.log(
            (1 / a + 1 / b + 1/c) / (1 / xs + 1 / ys + 1 / zs)))), 2)
        rh = abs(xs / a)**h + abs(ys / b)**h + abs(zs / c)**h

        # Geometric mean
        g = max(int(np.ceil(np.log(1 / 27) / np.log(
            (xs * ys * zs) / (a * b * c)))), 2)
        rg = abs(xs / a)**g + abs(ys / b)**g + abs(zs / c)**g

        # Arithmetic mean
        z = max(int(np.ceil(np.log(1 / 3) / np.log(
            (xs + ys + zs) / (a + b + c)))), 2)
        rz = abs(xs / a)**z + abs(ys / b)**z + abs(zs / c)**z

        r = n = 1
        while r >= 1 and n < n_max:
            n += 1
            r = abs(xs / a)**n + abs(ys / b)**n + abs(zs / c)**n
            if monitor:
                print('ParHypEll - r: {:>5.3f} - n: {:>}'.format(r, n))

        if lim == 'MIN':
            n = np.clip(max(n, h, g, z), n_min, n_max)
        elif lim == 'MAX':
            n = np.clip(min(n, h, g, z), n_min + 1, n_max)

        if monitor:
            print("'Harm' Superness. r: {:>5.3f} - n: {:>}".format(rh, h))
            print("'Geom' Superness. r: {:>5.3f} - n: {:>}".format(rg, g))
            print("'Arit' Superness. r: {:>5.3f} - n: {:>}".format(rz, z))
            lim_str = "min" if lim == 'MIN' else "max"
            pr0_str = "Hyperellipsoid Parameters. r_" + lim_str
            pr1_str = ": {:>5.3f} - n_" + lim_str + ": {:>}"
            print(pr0_str + pr1_str.format(r, n))

        snss_str = "Superness Coordinates (km): ({:5.3f}, {:5.3f}, {:5.3f})"
        print(snss_str.format(xs, ys, zs))

        return n

    def define_hyperlayer(self, pad_len, lmin):
        '''
        Define the hyperlayer degree and its limits.

        Parameters
        ----------
        pad_len : `float`
            Size of the absorbing layer
        lmin: `float`
            Minimum mesh size

        Returns
        -------
        None
        '''

        # Hypershape semi-axes and domain dimensions
        a_hyp = self.hyper_axes[0]
        b_hyp = self.hyper_axes[1]
        Lx = self.domain_dim[0]
        Lz = self.domain_dim[1]
        axis_str = "Semi-axes (km): a_hyp:{:5.3f} - b_hyp:{:5.3f}"

        # Hyperellipse degree
        n_hyp = self.n_hyp

        # Verification of hypershape degree
        print("Checking Current Hypershape Degree n_hyp: {}".format(n_hyp))
        if self.dimension == 3:  # 3D
            c_hyp = self.hyper_axes[2]
            Ly = self.domain_dim[2]
            axis_str += " - c_hyp:{:5.3f}"
        print(axis_str.format(*self.hyper_axes))

        # Minimum allowed exponent
        # n_min ensures to add lmin in the domain diagonal direction
        print("Determining the Minimum Degree for Hypershape Layer")
        x_min = [0.5 * Lx + lmin, 0.5 * Lz + lmin]

        if self.dimension == 2:  # 2D
            n_min = self.calc_degree_hyp2D(x_min, 'MIN')

        if self.dimension == 3:  # 3D
            x_min.append(0.5 * Ly + lmin)
            n_min = self.calc_degree_hyp3D(x_min, 'MIN')
        print('Minimum Degree for Hypershape n_min: {}'.format(n_min))

        # Maximum allowed exponent
        # n_max ensures to add pad_len in the domain diagonal direction
        print("Determining the Maximum Degree for Hypershape Layer")
        theta = np.arctan2(Lz, Lx)

        if self.dimension == 2:  # 2D
            x_max = [0.5 * Lx + pad_len * np.cos(theta),
                     0.5 * Lz + pad_len * np.sin(theta)]
            n_max = self.calc_degree_hyp2D(x_max, 'MAX', n_min=n_min)

        if self.dimension == 3:  # 3D
            phi = np.arccos(Ly / np.sqrt(Lx**2 + Ly**2 + Lz**2))
            x_max = [0.5 * Lx + pad_len * np.cos(theta) * np.sin(phi),
                     0.5 * Lz + pad_len * np.sin(theta) * np.sin(phi),
                     0.5 * Ly + pad_len * np.cos(phi)]
            n_max = self.calc_degree_hyp3D(x_max, 'MAX', n_min=n_min)
        print('Maximum Degree for Hypershape n_max: {}'.format(n_max))

        if n_min <= n_hyp <= n_max:
            print("Current Hypershape Degree n_hyp: {}".format(n_hyp))
        else:
            hyp_str = 'Degree for Hypershape Layer. Setting to'
            if n_hyp < n_min:
                print('Low', hyp_str, 'n_min: {}'.format(n_min))
            elif n_hyp > n_max:
                print('High', hyp_str, 'n_max: {}'.format(n_max))

        self.n_hyp = np.clip(n_hyp, n_min, n_max)
        self.n_bounds = (n_min, n_max)

    @staticmethod
    def trunc_half_hyp_area(a, b, n, z0):
        '''
        Compute the truncated area of hyperellipse for 0 <= z0 / b <= 1.
        Verify with self.trunc_half_hyp_area(1, 1, 2, 1) = pi / 2.

        Parameters
        ----------
        a : `float`
            Major hyperellipse semi-axis
        b : `float`
            Minor hyperellipse semi-axis
        n : `int`
            Degree of the hyperellipse
        z0 : `float`
            Truncation plane

        Returns
        -------
        A_tr : `float`
            Truncated area of the hyperellipse
        '''

        if z0 < 0 or z0 > b:
            UserWarning("Truncation plane must be 0 <= h <= {:5.3f}".format(b))

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
        Verify with self.half_hyp_area(1, 1, 2) = pi / 2.

        Parameters
        ----------
        a : `float`
            Hyperellipse semi-axis in direction 1
        b : `float`
            Hyperellipse semi-axis in direction 2
        n : `int`
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
        Verify with self.trunc_half_hyp_volume(1, 1, 1, 2, 1) = 2 * pi / 3.

        Parameters
        ----------
        a : `float`
            Hyperellipsoid semi-axis in direction 1
        b : `float`
            Hyperellipsoid semi-axis in truncated direction 2
        c : `float`
            Hyperellipsoid semi-axis in direction 3
        n : `int`
            Degree of the hyperellipsoid
        z0 : `float`
            Truncation plane

        Returns
        -------
        A_tr : `float`
            Truncated volume of the hyperellipsoid
        '''

        if z0 < 0 or z0 > b:
            UserWarning("Truncation plane must be 0 <= h <= {:5.3f}".format(b))

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
        Verify with self.half_hyp_volume(1, 1, 1, 2) = 2 * pi / 3.

        Parameters
        ----------
        a : `float`
            Hyperellipsoid semi-axis in direction 1
        b : `float`
            Hyperellipsoid semi-axis in direction 2
        c : `float`
            Hyperellipsoid semi-axis in direction 3
        n : `int`
            Degree of the hyperellipsoid

        Returns
        -------
        V_hf : `float`
            Half the volume of the hyperellipsoid
        '''

        V_hf = 4 * a * b * c * gamma(1 + 1 / n)**3 / gamma(1 + 3 / n)

        return V_hf

    def calc_hyp_geom_prop(self):
        '''
        Calculate the geometric properties for the hypershape layer.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Hypershape semi-axes and domain dimensions
        a_hyp = self.hyper_axes[0]
        b_hyp = self.hyper_axes[1]
        Lx = self.domain_dim[0]
        Lz = self.domain_dim[1]

        # Hyperellipse degree
        n_hyp = self.n_hyp

        # Geometric properties realted to Area (2D) or Volume (3D)
        if self.dimension == 2:  # 2D
            self.area = self.half_hyp_area(a_hyp, b_hyp, n_hyp) + \
                self.trunc_half_hyp_area(a_hyp, b_hyp, n_hyp, Lz / 2)
            self.a_rat = self.area / (Lx * Lz)
            self.f_Ah = self.area / (a_hyp * b_hyp)

        if self.dimension == 3:  # 3D
            c_hyp = self.hyper_axes[2]
            Ly = self.domain_dim[2]
            self.vol = self.half_hyp_volume(a_hyp, b_hyp, c_hyp, n_hyp) + \
                self.trunc_half_hyp_volume(a_hyp, b_hyp, c_hyp, n_hyp, Lz / 2)
            self.v_rat = self.vol / (Lx * Lz * Ly)
            self.f_Vh = self.vol / (a_hyp * b_hyp * c_hyp)
