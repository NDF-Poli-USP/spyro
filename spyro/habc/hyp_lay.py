import firedrake as fire
from netgen.geom2d import SplineGeometry
from netgen.meshing import Element2D, FaceDescriptor, Mesh, MeshPoint
import numpy as np
from scipy.integrate import quad
from scipy.spatial import KDTree
from scipy.special import beta, betainc, gamma
# fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}

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
    dimension : `int`
        Model dimension (2D or 3D). Default is 2D
    domain_dim : list
        Domain dimensions [Lx, Lz] (2D) or [Lx, Lz, Ly] (3D)
    f_Ah : `float`
        Hyperelliptical area factor. f_Ah = area / (a_hyp * b_hyp)
    f_Vh : `float`
        Hyperellipsoidal volume factor. f_Vh = vol / (a_hyp * b_hyp * c_hyp)
    hyper_axes : `list`
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
    bnd_pnts_hyp2D()
        Generate points on the boundary of a hyperellipse
    calc_degree_hyp2D()
        Define the limits for the hyperellipse degree. See Salas et al (2022)
    calc_degree_hyp3D()
        Define the limits for the hyperellipsoid degree
    calc_hyp_geom_prop(self):
        Calculate the geometric properties for the hypershape layer
    create_bnd_mesh2D()
        Generate the boundary segment curves for the hyperellipse boundary mesh
    create_hyp_trunc_mesh2D()
        Generate the mesh for the hyperelliptical absorbing layer
    define_hyperaxes()
        Define the hyperlayer semi-axes
    define_hyperlayer()
        Define the hyperlayer degree and its limits
    half_hyp_area()
        Compute half the area of the hyperellipse
    half_hyp_volume()
        Compute half the volume of the hyperellipsoid
    merge_mesh_2D()
        Merge the rectangular and the hyperelliptical meshes
    trunc_half_hyp_area()
        Compute the truncated area of superellipse for 0 <= z0 / b <= 1
    trunc_half_hyp_volume()
        Compute the truncated volume of hyperellipsoid for 0 <= z0 / b <= 1
    trunc_hyp_bnd_points()
        Generate the boundary points for a truncated hyperellipse
    '''

    def __init__(self, n_hyp=2, dimension=2):
        '''
        Initialize the HyperLayer class.

        Parameters
        ----------
        n_hyp : `int`, optional
            Hypershape degree. Default is 2
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D

        Returns
        -------
        None
        '''

        # Model dimension
        self.dimension = dimension

        # Hypershape degree
        self.n_hyp = n_hyp if n_hyp is not None else 2

    def define_hyperaxes(self, domain_dim, domain_hyp):
        '''
        Define the hyperlayer semi-axes.

        Parameters
        ----------
        domain_dim : `list`
            Domain dimensions [Lx, Lz] (2D) or [Lx, Lz, Ly] (3D)
        domain_hyp : `list`
            Maximum hypershape dimensions without truncations.
            - 2D : [Lx + 2 * pad_len, Lz + 2 * pad_len]
            - 3D : [Lx + 2 * pad_len, Lz + 2 * pad_len, Ly + 2 * pad_len]

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
        x_rel : `list`
            Relative superness coordinates to the hyperellipse centroid
        lim : str
            Limit for the hyperellipse degree ('MIN' or 'MAX')
        n_min : `int`, optional
            Minimum allowed degree. Default is 2.
        n_max : `int`, optional
            Maximum allowed degree. Default is 20.
        monitor : `bool`, optional
            Print the process on the screen. Default is False.

        Returns
        -------
        n : `int`
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
                print("ParHypEll - r: {:>5.3f} - n: {:>}".format(r, n))

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
        x_rel : `list`
            Relative superness coordinates to the hyperellipsoid centroid
        lim : str
            Limit for the hyperellipsoid degree ('MIN' or 'MAX')
        n_min : `int`, optional
            Minimum allowed degree. Default is 2.
        n_max : `int`, optional
            Maximum allowed degree. Default is 20.
        monitor : `bool`, optional
            Print the process on the screen. Default is False.

        Returns
        -------
        n : `int`
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
                print("ParHypEll - r: {:>5.3f} - n: {:>}".format(r, n))

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
        lmin : `float`
            Minimum mesh size

        Returns
        -------
        None
        '''

        # Hypershape semi-axes and domain dimensions
        Lx = self.domain_dim[0]
        Lz = self.domain_dim[1]
        axis_str = "Semi-axes (km): a_hyp:{:5.3f} - b_hyp:{:5.3f}"

        # Hyperellipse degree
        n_hyp = self.n_hyp

        # Verification of hypershape degree
        print("Checking Current Hypershape Degree n_hyp: {}".format(n_hyp))
        if self.dimension == 3:  # 3D
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
        print("Minimum Degree for Hypershape n_min: {}".format(n_min))

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
        print("Maximum Degree for Hypershape n_max: {}".format(n_max))

        if n_min <= n_hyp <= n_max:
            print("Current Hypershape Degree n_hyp: {}".format(n_hyp))
        else:
            hyp_str = "Degree for Hypershape Layer. Setting to"
            if n_hyp < n_min:
                print("Low", hyp_str, "n_min: {}".format(n_min))
            elif n_hyp > n_max:
                print("High", hyp_str, "n_max: {}".format(n_max))

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

    @staticmethod
    def hyp_full_perimeter(a, b, n):
        '''
        Compute perimeter of a hyperellipse.

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
            Ly = self.domain_dim[2]

            # Volume
            self.vol = self.half_hyp_volume(a_hyp, b_hyp, c_hyp, n_hyp) + \
                self.trunc_half_hyp_volume(a_hyp, b_hyp, c_hyp, n_hyp, Lz / 2)

            # Volume ratio
            self.v_rat = self.vol / (Lx * Lz * Ly)
            print("Volume Ratio: {:5.3f}".format(self.v_rat))

            # Volume factor
            self.f_Vh = self.vol / (a_hyp * b_hyp * c_hyp)

            # Full surface area to estimate the mesh size
            self.surf_hyp = None

    @staticmethod
    def bnd_pnts_hyp2D(a, b, n, num_pts):
        '''
        Generate points on the boundary of a hyperellipse.

        'Parameters
        ----------
        a : `float`
            Hyperellipse semi-axis in direction 1
        b : `float`
            Hyperellipse semi-axis in direction 2
        n : `int`
            Degree of the hyperellipse
        num_pts : `int`
            Number of points to generate on the hyperellipse boundary

        Returns
        -------
        bnd_pnts : `array`
            Array of shape (num_pts, 2) containing the coordinates
            of the hyperellipse boundary points
        '''

        # Generate angle values for the parametric equations
        theta = np.linspace(0, 2 * np.pi, num_pts)

        # Especial angle values
        rc_zero = [np.pi / 2., 3 * np.pi / 2.]
        rs_zero = [0., np.pi, 2 * np.pi]

        # Trigonometric function evaluation
        cr = np.cos(theta)
        sr = np.sin(theta)
        cr = np.where(np.isin(theta, rc_zero), 0, cr)
        sr = np.where(np.isin(theta, rs_zero), 0, sr)

        # Parametric equations for the hyperellipse
        x = a * np.sign(cr) * np.abs(cr)**(2 / n)
        y = b * np.sign(sr) * np.abs(sr)**(2 / n)

        bnd_pnts = np.column_stack((x, y))

        return bnd_pnts

    def trunc_hyp_bnd_points(self, xdom, z0):
        '''
        Generate the boundary points for a truncated hyperellipse

        Parameters
        ----------
        xdom : `float`
            Maximum coordinate in normal to the truncation plane
        z0 : `float`
            Truncation plane

        Returns
        -------
        filt_bnd_pts : `array`
            Array of shape (num_bnd_pts, 2) containing the coordinates
            of the truncated hyperellipse boundary points
        trunc_feat : `list`
            List containing the truncation features:
            - ini_trunc : Index of the first point after the truncation plane
            - end_trunc : Index of the last point before the truncation plane
            - num_filt_bnd_pts : Number of filtered boundary points
            - ltrunc : Mesh size for arc length due to the truncation operation
        '''

        # Hypershape parameters
        a_hyp = self.hyper_axes[0]
        b_hyp = self.hyper_axes[1]
        n_hyp = self.n_hyp
        perimeter = self.perim_hyp

        # Boundary points: Use 16 or 24 as a minimum
        lmin = self.lmin
        num_bnd_pts = int(max(np.ceil(perimeter / lmin), 16)) - 1

        # Generate the hyperellipse boundary points
        pnt_bef_trunc = 0
        pnt_aft_trunc = 0
        pnt_str = "Number of Boundary Points for"
        while pnt_bef_trunc % 2 == 0 or pnt_aft_trunc % 2 == 0 \
                or pnt_bef_trunc < 3 or pnt_aft_trunc < 3:

            num_bnd_pts += 1
            print(pnt_str, f"Complete Hyperellipse: {num_bnd_pts}")
            bnd_pts = self.bnd_pnts_hyp2D(a_hyp, b_hyp, n_hyp, num_bnd_pts)

            # Filter hyperellipse points based on the truncation plane z0
            filt_bnd_pts = np.array([point for point in bnd_pts
                                     if point[1] <= z0])
            print(pnt_str, f"Truncated Hyperellipse: {len(filt_bnd_pts)}")

            # Identify truncation index
            ini_trunc = max(np.where(bnd_pts[:, 1] > z0)[0][0] - 1, 0)

            # Modify points to match with the truncation plane
            r0 = np.asin((z0 / b_hyp)**(n_hyp / 2))
            x0 = a_hyp * np.cos(r0)**(2 / n_hyp)
            filt_bnd_pts[ini_trunc] = np.array([x0, z0])
            filt_bnd_pts[ini_trunc + 1] = np.array([-x0, z0])

            # Insert new points to create a rectangular trunc
            new_pnts = np.array([[xdom, z0], [xdom, -z0],
                                 [-xdom, -z0], [-xdom, z0]])
            filt_bnd_pts = np.insert(filt_bnd_pts, ini_trunc + 1,
                                     new_pnts, axis=0)
            end_trunc = ini_trunc + 5

            # Points before and after truncation
            pnt_bef_trunc = len(filt_bnd_pts[:ini_trunc + 1])
            pnt_aft_trunc = len(filt_bnd_pts[end_trunc:])

        # Total number of the boundary points including the trunc
        num_filt_bnd_pts = len(filt_bnd_pts)

        # Mesh size for arc length due to the truncation operation
        ltrunc = perimeter / num_bnd_pts

        # Truncation features
        trunc_feat = [ini_trunc, end_trunc, num_filt_bnd_pts, ltrunc]

        return filt_bnd_pts, trunc_feat

    def create_bnd_mesh2D(self, geo, bnd_pts, trunc_feat, spln):
        '''
        Generate the boundary segments for the hyperellipse boundary mesh.

        Parameters
        ----------
        geo : `SplineGeometry`
            Geometry object with the data to generate the mesh
        bnd_pts : `array`
            Array of shape (num_bnd_pts, 2) containing the coordinates
            of the hyperellipse boundary points
        trunc_feat : `list`
            List containing the truncation features:
            - ini_trunc : Index of the first point after the truncation plane
            - end_trunc : Index of the last point before the truncation plane
            - num_bnd_pts : Number of filtered boundary points
            - ltrunc : Mesh size for arc length due to the truncation operation
        spln : `bool`
            Flag to indicate whether to use splines (True) or lines (False)

        Returns
        -------
        curves : `list`
            List of curves to be added to the geometry object. Each curve is
            represented as a list containing the curve type and its points.
        '''

        ini_trunc, end_trunc, num_bnd_pts, ltrunc = trunc_feat

        curves = []
        if spln:
            # Mesh with spline segments
            for idp in range(0, ini_trunc, 2):
                p1 = geo.PointData()[2][idp]
                p2 = geo.PointData()[2][idp + 1]
                p3 = geo.PointData()[2][idp + 2]
                curves.append(["spline3", p1, p2, p3, ltrunc])
                # print(p1, p2, p3)

            for idp in range(ini_trunc, end_trunc):
                p1 = geo.PointData()[2][idp]
                p2 = geo.PointData()[2][idp + 1]
                # print(p1, p2)

                if idp == ini_trunc or idp == end_trunc - 1:
                    curves.append(["line", p1, p2, ltrunc])
                else:
                    curves.append(["line", p1, p2, self.lmin])

            for idp in range(end_trunc, num_bnd_pts - 1, 2):
                p1 = geo.PointData()[2][idp]
                p2 = geo.PointData()[2][idp + 1]
                p3 = geo.PointData()[2][idp + 2]
                curves.append(["spline3", p1, p2, p3, ltrunc])
                # print(p1, p2, p3)

        else:
            # Mesh with line segments
            for idp in range(0, num_bnd_pts - 1, 2):
                p1 = geo.PointData()[2][idp]
                p2 = geo.PointData()[2][idp + 1]
                # print(p1, p2)

                if ini_trunc + 1 <= idp <= end_trunc - 2:
                    curves.append(["line", p1, p2, self.lmin])
                else:
                    curves.append(["line", p1, p2, ltrunc])

        return curves

    def create_hyp_trunc_mesh2D(self, spln=True, fmesh=1.):
        '''
        Generate the mesh for the hyperelliptical absorbing layer.

        Parameters
        ----------
        None

        Returns
        -------
        hyp_mesh : `netgen mesh`
            Generated mesh for the hyperelliptical layer
        spln : `bool`, optional
            Flag to indicate whether to use splines (True) or lines (False)
        fmesh : `float`, optional
            Mesh size factor for the hyperelliptical layer with respect to mesh
            size of the original domain. Default is 1.0.
        '''

        # Domain dimensions
        Lx = self.domain_dim[0]
        Lz = self.domain_dim[1]

        # Generate the hyperellipse boundary points
        bnd_pts, trunc_feat = self.trunc_hyp_bnd_points(Lx / 2, Lz / 2)

        # Initialize geometry
        geo = SplineGeometry()

        # Append points to the geometry
        [geo.AppendPoint(*pnt) for pnt in bnd_pts]

        #  Mesh size for arc length due to the truncation operation
        ltrunc = trunc_feat[-1]
        fmin = ltrunc / self.lmin

        # Mesh size factor for the hyperelliptical layer
        if fmesh != 1.:
            fmesh = max(fmesh, fmin)
        fm_str = "Mesh Factor Size Inside Layer (Min): {:.2f} ({:.2f})"
        print(fm_str.format(fmesh, fmin))

        while True:
            try:
                # Generate the boundary segment curves
                curves = self.create_bnd_mesh2D(geo, bnd_pts, trunc_feat, spln)
                [geo.Append(c[:-1], bc="outer", maxh=c[-1], leftdomain=1,
                            rightdomain=0) for c in curves]

                # Generate the mesh using netgen library
                hyp_mesh = geo.GenerateMesh(maxh=fmesh*self.lmin,
                                            quad_dominated=False)
                print("Hyperelliptical Mesh Generated Successfully")
                break

            except Exception as e:
                if spln:
                    print(f"Error Meshing with Splines: {e}")
                    print("Now meshing with Lines")
                    spln = False
                else:
                    UserWarning(f"Error Meshing with Lines: {e}. Exiting.")
                    break

        # Mesh is transformed into a firedrake mesh
        return fire.Mesh(hyp_mesh)

    def merge_mesh_2D(self, rec_mesh, hyp_mesh):
        '''
        Merge the rectangular and the hyperelliptical meshes.

        Parameters
        ----------
        rec_mesh : `firedrake mesh`
            Rectangular mesh representing the original domain
        hyp_mesh : `firedrake mesh`
            Hyperelliptical annular mesh representing the absorbing layer

        Returns
        -------
        final_mesh : `firedrake mesh`
            Merged final mesh
        '''

        # Create the final mesh that will contain both
        final_mesh = Mesh()
        final_mesh.dim = self.dimension

        # Get coordinates of the rectangular mesh
        coord_rec = rec_mesh.coordinates.dat.data_with_halos

        # Create KDTree for efficient nearest neighbor search
        boundary_tree = KDTree([(z, x) for z, x in
                                zip(self.bnd_nodes[0], self.bnd_nodes[1])])

        # Add all vertices from rectangular mesh and create mapping
        rec_map = {}
        boundary_coords = []
        boundary_points = []
        for i, coord in enumerate(coord_rec):
            z, x = coord
            rec_map[i] = final_mesh.Add(MeshPoint((z, x, 0.)))  # y = 0 for 2D

            # Check if the point is on the original boundary
            if boundary_tree.query(coord)[0] <= self.tol:
                boundary_coords.append(coord)
                boundary_points.append(rec_map[i])

        # Create KDTree for efficient nearest neighbor search
        boundary_tree = KDTree([coord for coord in boundary_coords])

        # Face descriptor for the rectangular mesh
        fd_rec = final_mesh.Add(FaceDescriptor(bc=1, domin=1, domout=2))

        # Get mesh cells from rectangular mesh
        rec_cells = rec_mesh.coordinates.cell_node_map().values_with_halo

        # Add all elements from rectangular mesh to the netgen mesh
        final_mesh.SetMaterial(1, "rec")
        for cell in rec_cells:
            netgen_points = [rec_map[cell[i]] for i in range(len(cell))]
            final_mesh.Add(Element2D(fd_rec, netgen_points))

        # Add all vertices from hyperelliptical mesh and create mapping
        hyp_map = {}
        coord_hyp = hyp_mesh.coordinates.dat.data_with_halos
        for i, coord in enumerate(coord_hyp):
            z, x = coord

            # Check if the point is on the original boundary
            dist, idx = boundary_tree.query(coord)

            if dist <= self.tol:
                # Reuse the existing point
                hyp_map[i] = boundary_points[idx]
            else:
                # Creating a new point (y = 0 for 2D)
                hyp_map[i] = final_mesh.Add(MeshPoint((z, x, 0.)))

        # Face descriptor for the hyperelliptical mesh
        fd_hyp = final_mesh.Add(FaceDescriptor(bc=2, domin=2, domout=0))

        # Get mesh cells from hyperelliptical mesh
        hyp_cells = hyp_mesh.coordinates.cell_node_map().values_with_halo

        # Add all elements from hyperelliptical mesh to the netgen mesh
        final_mesh.SetMaterial(2, "hyp")
        for cell in hyp_cells:
            netgen_points = [hyp_map[cell[i]] for i in range(len(cell))]
            final_mesh.Add(Element2D(fd_hyp, netgen_points))

        try:
            # Mesh is transformed into a firedrake mesh
            final_mesh.Compress()
            final_mesh = fire.Mesh(final_mesh)
            print("Merged Mesh Generated Successfully")

        except Exception as e:
            UserWarning(f"Error Generating Merged Mesh: {e}. Exiting.")

        return final_mesh
