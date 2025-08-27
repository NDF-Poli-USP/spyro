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
        Maximum incidence angle considered. Default is pi/4 (45°)
    cos_Hig : `firedrake function`
        Profile of the cosine of incidence angle for 1^st-order Higdon BC.
        Free surfaces and interior nodes are set to 0
    cos_max : `float`
        Maximum value of the cosine of the incidence angle
    dimension : `int`
        Model dimension (2D or 3D). Default is 2D
    dom_dim : `tuple`
        Original domain dimensions: (Lx, Lz) for 2D or (Lx, Lz, Ly) for 3D
    layer_shape : `string`
        Shape type of pad layer. Options: 'rectangular' or 'hypershape'
    nrbc : `str`
        Type of NRBC used. Either "Higdon" or "Sommerfeld"
    path_save_nrbc : `str`
        Path to save field for the NRBC

    Methods
    -------
    cos_ang_HigdonBC()
        Compute the cosine of the incidence angle for first order Higdon BC
    hyperellipse_normal_vector()
        Compute the normal vector to a hyperellipse
    '''

    def __init__(self, dom_dim, layer_shape, angle_max=np.pi/4.,
                 dimension=2, output_folder=None):
        '''
        Initialize the NRBC class.

        Parameters
        ----------
        dom_dim : `tuple`
            Original domain dimensions: (Lx, Lz) for 2D or (Lx, Lz, Ly) for 3D
        layer_shape : `string`
            Shape type of pad layer. Options: 'rectangular' or 'hypershape'
        angle_max : `float`, optional
            Maximum incidence angle considered. Default is pi/4 (45°)
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D
        output_folder : str, optional
            The folder where output data will be saved. Default is None

        Returns
        -------
        None
        '''

        # Original domain dimensions
        self.dom_dim = dom_dim

        # Shape type of pad layer
        self.layer_shape = layer_shape

        # Maximum incidence angle considered
        self.angle_max = angle_max

        # Maximum value of the cosine of the incidence angle
        self.cos_max = np.cos(angle_max)

        # Model dimension
        self.dimension = dimension

        # Path to save data
        if output_folder is None:
            self.path_save_nrbc = getcwd() + "/output/"
        else:
            self.path_save_nrbc = output_folder

    def source_to_bnd_reference_vector(self, source_coord, bnd_nodes_nfs):

        # Boundary node data
        bnd_z, bnd_x = bnd_nodes_nfs[:2]

        # Source coordinates
        psouz = source_coord[0]
        psoux = source_coord[1]

        # Components of the vector pointing to the boundary point
        ref_z = bnd_z - psouz
        ref_x = bnd_x - psoux
        ref_vct = [ref_z, ref_x]

        if self.dimension == 3:  # 3D

            # Third component of the vector pointing to the boundary point
            bnd_y = bnd_nodes_nfs[2]
            psouy = source_coord[2]
            ref_y = bnd_y - psouy
            ref_vct.append(ref_y)

        # Unitary vector pointing to the boundary point
        unit_ref_vct = np.asarray(ref_vct) / np.linalg.norm(ref_vct, axis=0)

        # For rectangular layer, consider absolute values of the components
        if self.layer_shape == 'rectangular':
            unit_ref_vct = abs(unit_ref_vct)

        return unit_ref_vct

    def hypershape_normal_vector(self, coord_point, hyper_axes, n):
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

        nrm_vct = [df_dx, df_dy]

        if self.dimension == 3:  # 3D

            # Third coordinate
            z = coord_point[2]

            # Third hypershape semi-axis
            c = hyper_axes[2]

            # Partial derivative with respect to third coordinate
            df_dz = (n / (c**n)) * np.sign(z) * abs(z)**(n - 1)

            nrm_vct.append(df_dz)

        # Unitary hypershape normal vector
        unit_nrm_vct = np.asarray(nrm_vct) / np.linalg.norm(nrm_vct, axis=0)

        return unit_nrm_vct

    def cos_incidence_angle(self, unit_ref_vct):
        '''
        Compute the cosine of the incidence angle between the reference
        vector pointing to the boundary and the normal vector to the boundary.

        Parameters
        ----------
        unit_ref_vct : `array`
            Unitary reference vector pointing to the boundary point

        Returns
        -------
        cos_Hig : `array`
            Cosine of the incidence angle
        '''

        # Unitary reference vector components
        nz_r, nx_r = unit_ref_vct[:2]

        if self.dimension == 3:

            # Third component of the unitary reference vector
            ny_r = unit_ref_vct[2]

    def cos_ang_HigdonBC(self, V, source_coord, bnd_nfs,
                         bnd_nodes_nfs, sommerfeld_bc=False):
        '''
        Compute the cosine of the incidence angle for first order Higdon BC.

        Parameters
        ----------
        V : `firedrake function space`
            Function space where the cosine of the incidence angle is defined
        bnd_nfs : 'array'
            Mesh node indices on non-free surfaces
        bnd_nodes_nfs : `tuple`
            Mesh node coordinates on non-free surfaces.
            - (z_data[nfs_idx], x_data[nfs_idx]) for 2D
            - (z_data[nfs_idx], x_data[nfs_idx], y_data[nfs_idx]) for 3D
        source_coord : `tuple`
            Source coordinates
        sommerfeld_bc : `bool`, optional
            If True, use Sommerfeld BC instead of Higdon BC. Default is False

        Returns
        -------
        None
        '''

        nrbc_str = "Sommerfeld" if sommerfeld_bc else "Higdon"
        print("\nCreating Field for NRBC:", nrbc_str)

        # Initialize field for the cosine of the incidence angle
        self.cosHig = fire.Function(V, name='cosHig')

        # Boundary node data
        bnd_z, bnd_x = bnd_nodes_nfs[:2]

        if sommerfeld_bc:  # Sommerfeld BC
            self.nrbc = "Sommerfeld"
            cos_Hig = 1.

        else:  # Higdon BC

            self.nrbc = "Higdon"

            # Unitary reference vector pointing to the boundary point
            unit_ref_vct = self.source_to_bnd_reference_vector(source_coord,
                                                               bnd_nodes_nfs)

            # Normal vector to the boundary
            if self.layer_shape == 'rectangular':
                # Normal vector to the boundary is a orthonormal vector,
                # then cos on incidence angle can be estimated from a
                # projection of the reference vector to boundary
                # onto the orthonormal vector [1, 0]
                cos_Hig = np.maximum.reduce(unit_ref_vct)

                # # Normal vector to the boundary
                # if self.layer_shape == 'rectangular':
                #     # Normal vector to the boundary is a orthonormal
                #     # vector, then cosine on incidence angle can be
                #     # estimated from a projection of the reference
                #     # vector to boundary onto the unitary vectors
                #     cos_Hig = np.maximum.reduce(
                #         [abs(nz_r), abs(nx_r), abs(ny_r)])

            if self.layer_shape == 'hypershape':

                # Original domain dimensions
                Lx, Lz = self.dom_dim[:2]

                # Boundary points of the hypershape centered at the origin
                z_hyp = bnd_z + Lz / 2
                x_hyp = bnd_x - Lx / 2
                coord_point = [z_hyp, x_hyp]

                if self.dimension == 3:  # 3D

                    # Third component of the boundary points
                    Ly = self.dom_dim[2]
                    bnd_y = bnd_nodes_nfs[2]
                    y_hyp = bnd_y - Ly / 2
                    coord_point.append(y_hyp)

                # Normal vector to the boundary
                unit_nrm_vct = self.hypershape_normal_vector(coord_point,
                                                             self.hyper_axes,
                                                             self.n_hyp)
                nz_h, nx_h = unit_nrm_vct[:2]

                # Components unitary reference vector
                nz_r, nx_r = unit_ref_vct[:2]

                # Cosine of the incidence angle
                cos_Hig = abs(nz_r * nz_h + nx_r * nx_h)

                if self.dimension == 3:  # 3D

                    # Third component of the unitary reference vector
                    ny_r = unit_ref_vct[2]

                    # Third component of the normal vector to the boundary
                    ny_h = unit_nrm_vct[2]

                    # Cosine of the incidence angle
                    cos_Hig = abs(nz_r * nz_h + nx_r * nx_h + ny_r * ny_h)

            cos_Hig[cos_Hig < self.cos_max] = (
                1. - cos_Hig[cos_Hig < self.cos_max]**2)**0.5

        self.cosHig.dat.data_with_halos[bnd_nfs] = cos_Hig

        # Save boundary profile of cosine of incidence angle
        outfile = fire.VTKFile(self.path_save_nrbc + "cosHig.pvd")
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
