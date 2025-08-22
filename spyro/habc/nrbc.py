import firedrake as fire
import numpy as np

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class NRBC():
    '''
    class for NRBCs applied to outer boundary absorbing layer in HABC scheme.

    Attributes
    ----------
    angle_max : `float`
        Maximum incidence angle considered. Default is pi/4 (45°).
    cos_Hig : `firedrake function`
        Profile of the cosine of incidence angle for 1^st-order Higdon BC.
        Free surfaces and interior nodes are set to 0
    cos_max : `float`
        Maximum value of the cosine of the incidence angle
    nrbc : `str`
        Type of NRBC used. Either "Higdon" or "Sommerfeld"

    Methods
    -------
    cos_ang_HigdonBC()
        Compute the cosine of the incidence angle for first order Higdon BC.
    hyperellipse_normal_vector()
        Compute the normal vector to a hyperellipse
    '''

    def __init__(self, angle_max=np.pi/4.):
        '''
        Initialize the NRBC class.

        Parameters
        ----------
        angle_max : `float`, optional
            Maximum incidence angle considered. Default is pi/4 (45°).

        Returns
        -------
        None
        '''

        # Maximum incidence angle considered
        self.angle_max = angle_max

        # Maximum value of the cosine of the incidence angle
        self.cos_max = np.cos(angle_max)

    @staticmethod
    def hypershape_normal_vector(coord_point, hyper_axes, n, dimension=2):
        '''
        Compute the normal vector to a hyperellipse |x/a|^n + |y/b|^n = 1
        or a hyperellipsoid |x/a|^n + |y/b|^n + |z/c|^n = 1 at a point.
        Both hyperellipsoid and hyperellipse have the center at the origin.

        Observations:
        Let f(x, y) = |x/a|^n - |y/b|^n -1 = 0 a level curve (level set for
        two variables) for f(x, y, z) at z = 0. The gradient of the function
        f given by ∇f(x,y) = [∂f/∂x, ∂f/∂y] is a normal vector to the curve.
        The normal vector is given by the partial derivatives of the function

        Parameters
        ----------
        coord_point : `list`
            Coordinates of the point where the normal vector is computed.
            Structure: [x, y] for 2D and [x, y, z] for 3D
        hyper_axes : `list`
            Semi-axes of the hyperellipse [a, b] or hyperellipsoid [a, b, c].
        n : `float`
            degree of the hyperellipse
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D

        Returns
        -------
        nx : `float`
            x-component of the normal vector
        ny : `float`
            y-component of the normal vector
        nz : `float`
            z-component of the normal vector
        '''

        # Point coordinates
        x, y = coord_point[:2]

        # Hypershape semi-axes
        a, b = hyper_axes[:2]

        # Compute partial derivatives
        df_dx = (n / (a**n)) * np.sign(x) * abs(x)**(n - 1)
        df_dy = (n / (b**n)) * np.sign(y) * abs(y)**(n - 1)

        if dimension == 2:
            # Normalize
            norm = (df_dx**2 + df_dy**2)**0.5

        if dimension == 3:

            # Third coordinate
            z = coord_point[2]

            # Third hypershape semi-axis
            c = hyper_axes[2]

            # Partial derivative with respect to third coordinate
            df_dz = (n / (c**n)) * np.sign(z) * abs(z)**(n - 1)

            # Normalize
            norm = (df_dx**2 + df_dy**2 + df_dz**2)**0.5

            # Third component of the normal vector
            nz = df_dz / norm

        # Components of the normal vector
        nx = df_dx / norm
        ny = df_dy / norm

        if dimension == 2:
            return nx, ny

        if dimension == 3:
            return nx, ny, nz

    def cos_ang_HigdonBC(self, sommerfeld_bc=False):
        '''
        Compute the cosine of the incidence angle for first order Higdon BC.

        Parameters
        ----------
        sommerfeld_bc : `bool`, optional
            If True, use Sommerfeld BC instead of Higdon BC. Default is False.

        Returns
        -------
        None
        '''

        # Initialize field for the cosine of the incidence angle
        nrbc_str = "Sommerfeld" if sommerfeld_bc else "Higdon"
        print("\nCreating Field for NRBC:", nrbc_str)
        self.cosHig = fire.Function(self.function_space, name='cosHig')

        # Boundary nodes
        bnd_nod = fire.DirichletBC(self.function_space, 0., "on_boundary").nodes

        # Node coordinates - To do: refactor
        z_f = fire.Function(self.function_space).interpolate(self.mesh_z)
        x_f = fire.Function(self.function_space).interpolate(self.mesh_x)
        bnd_z = z_f.dat.data_with_halos[bnd_nod]
        bnd_x = x_f.dat.data_with_halos[bnd_nod]

        # Free surfaces remain unchanged
        no_free_surf = ~(abs(bnd_z) <= self.tol)

        if sommerfeld_bc:  # Sommerfeld BC
            self.nrbc = "Sommerfeld"
            cos_Hig = 1.

        else:  # Higdon BC

            self.nrbc = "Higdon"

            # Identify source locations
            possou = self.eik_bnd[0][-1]
            psouz = possou[0]
            psoux = possou[1]

            # Components of the vector pointing to the boundary point
            ref_z = bnd_z[no_free_surf] - psouz
            ref_x = bnd_x[no_free_surf] - psoux

            # Boundary points of the hypershape centered at the origin
            z_hyp = bnd_z[no_free_surf] + self.length_z / 2
            x_hyp = bnd_x[no_free_surf] - self.length_x / 2
            coord_point = [z_hyp, x_hyp]

            # Compute cosine of the incidence angle with dot product
            if self.dimension == 2:  # 2D
                # Norm of the vector pointing to the boundary point
                norm_ref = (ref_z**2 + ref_x**2)**0.5

                # Unitary vector pointing to the boundary point
                nz_r = ref_z / norm_ref
                nx_r = ref_x / norm_ref

                # Normal vector to the boundary
                if self.layer_shape == 'rectangular':
                    # Normal vector to the boundary is a orthonormal vector,
                    # then cos on incidence angle can be estimated from a
                    # projection of the reference vector to boundary
                    # onto the orthonormal vector [1, 0]
                    cos_Hig = np.maximum.reduce([abs(nz_r), abs(nx_r)])

                elif self.layer_shape == 'hypershape':

                    # Normal vector to the boundary
                    nz_h, nx_h = self.hypershape_normal_vector(
                        coord_point, self.hyper_axes, self.n_hyp)

                    # Cosine of the incidence angle
                    cos_Hig = abs(nz_r * nz_h + nx_r * nx_h)

            if self.dimension == 3:  # 3D

                # Third component of the vector pointing to the boundary point
                y_f = fire.Function(self.function_space).interpolate(self.mesh_y)
                bnd_y = y_f.dat.data_with_halos[bnd_nod]
                psouy = possou[2]
                ref_y = bnd_y[no_free_surf] - psouy

                # Third component of the boundary points
                y_hyp = bnd_x[no_free_surf] - self.length_y / 2
                coord_point.append(y_hyp)

                # Norm of the vector pointing to the boundary point
                norm_ref = (ref_z**2 + ref_x**2 + + ref_y**2)**0.5

                # Unitary vector pointing to the boundary point
                nz_r = ref_z / norm_ref
                nx_r = ref_x / norm_ref
                ny_r = ref_y / norm_ref

                # Normal vector to the boundary
                if self.layer_shape == 'rectangular':
                    # Normal vector to the boundary is a orthonormal vector, then
                    # cosine on incidence angle can be estimated from a projection
                    # of the reference vector to boundary onto the unitary vectors
                    cos_Hig = np.maximum.reduce(
                        [abs(nz_r), abs(nx_r), abs(ny_r)])

                elif self.layer_shape == 'hypershape':

                    # Normal vector to the boundary
                    nz_h, nx_h, ny_h = \
                        self.hypershape_normal_vector(
                            coord_point, self.hyper_axes,
                            self.n_hyp, dimension=self.dimension)

                    # Cosine of the incidence angle
                    cos_Hig = abs(nz_r * nz_h + nx_r * nx_h + ny_r * ny_h)
            cos_Hig[cos_Hig < self.cos_max] = (1. - cos_Hig[
                cos_Hig < self.cos_max]**2)**0.5

        self.cosHig.dat.data_with_halos[bnd_nod[no_free_surf]] = cos_Hig

        # Save boundary profile of cosine of incidence angle
        outfile = fire.VTKFile(self.path_save + self.case_habc + "/cosHig.pvd")
        outfile.write(self.cosHig)

# dx = 0.1 km REC
# W/O = 107.72% - 0.80%
# bnd_dom = 6.40% - 24.61%
# l/4 = 7439003.04% - 38808.77%
# l/3 = 7439003.04% - 38808.77%
# l/2 = 7439003.04% - 38808.77%
# 3*l/4 =  3.44% - 6.15%
# l = 2.72% - 3.40%
# 2l =  1.75% - 3.45%
# 3l = 1.28% - 0.80%
# 4l = 1.14% - 0.80%
# 5l = 1.03% - 0.80%
# 6l = 0.53% - 0.80%
# 10l = 0.53% - 0.80%
# free_surf = 0.43% - 0.80%

# dx = 0.1 km HYP n = 2
# W/O = 323.65% - 18.53%
# bnd_dom = 7.80% - 26.49%
# l/4 = No results (Numerical Inst)
# l/3 = No results (Numerical Inst)
# l/2 = 4.32 - 7.12%
# 3*l/4 =  4.32% - 7.12%
# l = 3.18% - 5.35%
# 2l =  2.73% - 3.86%
# 3l = 2.08% - 0.76%
# 5l = 1.02% - 0.76%
# free_surf = 1.02% - 0.76%
