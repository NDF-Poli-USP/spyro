import firedrake as fire
import numpy as np
import scipy.linalg as sl
import scipy.sparse as ss
import spyro.habc.eik as eik
import spyro.habc.lay_len as lay_len
from spyro.solvers.acoustic_wave import AcousticWave
from os import getcwd
import ipdb
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class HABC_Wave(AcousticWave):
    '''
    class HABC that determines absorbing layer size and parameters to be used

    Attributes
    ----------
    a_par: `float`
        Adimensional propagation speed parameter (a = z/f).
        "z" parameter is the inverse of the minimum Eikonal (1/phi_min)
    c_habc': 'firedrake function'
        Velocity model with absorbing layer
    d_par: `float`
        Normalized element size (lmin/pad_len)
    eik_bnd: `list`
        Properties on boundaries according to minimum values of Eikonal
        Structure sublist: [pt_cr, c_bnd, eikmin, z_par, lref, sou_cr]
        - pt_cr: Critical point coordinates
        - c_bnd: Propagation speed at critical point
        - eikmin: Eikonal value in seconds
        - z_par: Inverse of minimum Eikonal (Equivalent to c_bound/lref)
        - lref: Distance to the closest source from critical point
        - sou_cr: Critical source coordinates
    eta_habc: `firedrake function`
        Damping profile within the absorbing layer
    F_L : `float`
        Size  parameter of the absorbing layer
    freq_ref: `float`
        Reference frequency of the wave at the minimum Eikonal point
    fundam_freq; `float`
        Fundamental frequency of the numerical model
    fwi_iter: `int`
        The iteration number for the Full Waveform Inversion (FWI) algorithm
    Lz_habc: `float`
        Length of the domain in the z-direction with absorbing layer
    Lx_habc: `float`
        Length of the domain in the x-direction with absorbing layer
    Ly_habc: `float`
        Length of the domain in the y-direction with absorbing (3D models)
    layer_shape: `string`
        Shape type of pad layer
    lmin: `float`
        Minimum mesh size
    lmax: `float`
        Maxmum mesh size
    nexp: `int`
        Exponent of the hyperelliptical pad layer
    pad_len : `float`
        Size of damping layer
    path_save: `string`
        Path to save data


    Methods
    -------
    calc_damping_prop()
        Compute the damping properties for the absorbing layer
    create_mesh_habc()
        Create a mesh with absorbing layer based on the determined size
    coeff_damp_fun()
        Compute the coefficients for quadratic damping function
    damping_layer()
        Set the damping profile within the absorbing layer
    det_reference_freq()
        Determine the reference frequency for a new layer size.
    fundamental_frequency()
        Compute the fundamental frequency in Hz via modal analysis
    properties_eik_mesh()
        Set the properties for the mesh used to solve the Eikonal equation
    roundFL()
        Adjust the layer parameter based on the element size
    size_habc_criterion()
        Determine the size of the absorbing layer using the Eikonal criterion
    velocity_habc()
        Set the velocity model for the model with absorbing layer
    '''

    def __init__(self, dictionary=None, comm=None,
                 layer_shape='rectangular', fwi_iter=0):
        '''
        Initializes the HABC class.

        Parameters
        ----------
        dictionary : dict, optional
            A dictionary containing the input parameters for the HABC class
        comm : object, optional
            An object representing the communication interface
        fwi_iter: int, optional
            The iteration number for the FWI algorithm

        Returns
        -------
        None

        '''

        super().__init__(dictionary=dictionary, comm=comm)

        # Layer shape
        self.layer_shape = layer_shape

        # Hyperellipse degree
        if self.layer_shape == 'rectangular':
            self.nexp = None
        elif self.layer_shape == 'hyperelliptical':
            self.nexp = 2
        else:
            aux0 = "Please use 'rectangular' or 'hyperelliptical', "
            UserWarning(aux0 + f"{self.layer_shape} not supported.")

        # Current iteration
        self.fwi_iter = fwi_iter

        # Path to save data
        self.path_save = getcwd() + "/output/"

    def properties_eik_mesh(self, p_usu=None):
        '''
        Set the properties for the mesh used to solve the Eikonal equation.

        Parameters
        ----------
        p_usu : `int`, optional
            Finite element order

        Returns
        -------
        None
        '''

        # Setting the properties of the mesh used to solve the Eikonal equation
        p = self.degree if p_usu is None else p_usu
        self.funct_space_eik = fire.FunctionSpace(self.mesh, 'CG', p)

        # Mesh cell diameters
        self.diam_mesh = fire.CellDiameter(self.mesh)

        if self.fwi_iter == 0:

            if self.dimension == 2:  # 2D
                fdim = 2**0.5
            if self.dimension == 3:  # 3D
                fdim = 3**0.5

            # Minimum and maximum mesh size
            diam = fire.Function(
                self.funct_space_eik).interpolate(self.diam_mesh)
            self.lmin = round(diam.dat.data_with_halos.min() / fdim, 6)
            self.lmax = round(diam.dat.data_with_halos.max() / fdim, 6)

    def roundFL(self):
        '''
        Adjust the layer parameter based on the element size to get
        an integer number of elements within the layer.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Adjusting the parameter size of the layer
        lref = self.eik_bnd[0][4]
        self.F_L = (self.lmin / lref) * np.ceil(lref * self.F_L / self.lmin)

        # New size of the absorving layer
        self.pad_len = self.F_L * lref

        print('\nModifying Layer Size Based on the Element Size')
        print("Modified Parameter Size FL: {:.4f}".format(self.F_L))
        print("Modified Layer Size (km): {:.4f}".format(self.pad_len))
        print('Elements ({:.3f} km) in Layer: {}'.format(
            self.lmin, int(self.pad_len / self.lmin)))

    def det_reference_freq(self, histPcrit, fpad=4):
        '''
        Determine the reference frequency for a new layer size.

        Parameters
        ----------
        histPcrit: `array`
            Transient response at the minimum Eikonal point
        fpad: `int`, optional
            Padding factor for FFT. Default is 4

        Returns
        -------
        None
        '''

        print('\nDetermining Reference Frequency')

        if self.fwi_iter > 0:  # FWI iteration

            # Zero Padding for increasing smoothing in FFT
            yt = np.concatenate([np.zeros(fpad * len(histPcrit)), histPcrit])

            # Number of sample points
            N_samples = len(yt)

            # Calculate the response in frequency domain at the critical point
            yf = fft(yt)  # FFT
            fe = 1.0 / (2.0 * self.dt)  # Nyquist frequency
            pfft = N_samples // 2 + N_samples % 2
            xf = np.linspace(0.0, fe, pfft)

            # Minimun frequency excited
            self.freq_ref = xf[np.abs(yf[0:pfft]).argmax()]

            del yt, xf, yf

        else:  # Initial guess
            # Theorical central Ricker source frequency
            self.freq_ref = self.frequency

        print("Reference Frequency (Hz): {:.4f}".format(self.freq_ref))

    def size_habc_criterion(self, Eikonal, histPcrit,
                            layer_based_on_mesh=False):
        '''
        Determine the size of the absorbing layer using the Eikonal
        criterion for HABCs. See Salas et al (2022) for details.

        Parameters
        ----------
        Eikonal : `eikonal`
            An object representing the Eikonal solver
        histPcrit: `array`
            Transient response at the minimum Eikonal point
        layer_based_on_mesh : `bool`, optional
            Adjust the layer size based on the element size. Default is False

        Returns
        -------
        None
        '''

        # Eikonal boundary conditions
        Eikonal.define_bcs(self)

        # Solving Eikonal
        Eikonal.solve_eik(self)

        # Identifying critical points
        self.eik_bnd = Eikonal.ident_crit_eik(self)

        # Determining the reference frequency
        self.det_reference_freq(histPcrit)

        # Computing layer sizes
        self.F_L, self.pad_len, self.a_par = lay_len.calc_size_lay(self)

        if layer_based_on_mesh:
            self.roundFL()

        # Normalized element size
        self.d = self.lmin / self.pad_len
        print("Normalized Element Size (adim): {0:.5f}".format(self.d))

    def create_mesh_habc(self):
        '''
        Create a mesh with absorbing layer based on the determined size.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # New geometry with layer
        Lz = self.length_z + self.pad_len
        Lx = self.length_x + 2 * self.pad_len

        # Number of elements
        n_pad = self.pad_len / self.lmin  # Elements in the layer
        nz = int(self.length_z / self.lmin) + int(n_pad)
        nx = int(self.length_x / self.lmin) + int(2 * n_pad)
        nx = nx + nx % 2

        # New mesh with layer
        self.Lz_habc = Lz
        self.Lx_habc = Lx
        if self.dimension == 2:  # 2D
            mesh_habc = fire.RectangleMesh(nz, nx, Lz, Lx)

        if self.dimension == 3:  # 3D
            Ly = self.length_y + 2 * self.pad_len
            ny = int(self.length_y / self.lmin) + int(2 * n_pad)
            ny = ny + ny % 2
            mesh_habc = fire.BoxMesh(nz, nx, ny, Lz, Lx, Ly)
            mesh_habc.coordinates.dat.data_with_halos[:, 2] -= self.pad_len
            self.Ly_habc = Ly

        # Adjusting coordinates
        mesh_habc.coordinates.dat.data_with_halos[:, 0] *= -1.0
        mesh_habc.coordinates.dat.data_with_halos[:, 1] -= self.pad_len

        # Updating the mesh with the absorbing layer
        self.set_mesh(user_mesh=mesh_habc, mesh_parameters={})

        # Save new mesh
        outfile = fire.VTKFile(self.path_save + "mesh_habc.pvd")
        outfile.write(self.mesh)

    def velocity_habc(self):
        '''
        Set the velocity model for the model with absorbing layer.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Initialize velocity field
        print('\nUpdating Velocity Profile')
        self.c_habc = fire.Function(self.function_space, name='c [km/s])')

        # Extract node positions
        z_f = fire.Function(self.function_space).interpolate(self.mesh_z)
        x_f = fire.Function(self.function_space).interpolate(self.mesh_x)
        z_data = z_f.dat.data_with_halos[:]
        x_data = x_f.dat.data_with_halos[:]

        # Points to update velocity model
        if self.dimension == 2:  # 2D
            orig_pts = np.where((z_data >= -self.length_z) & (x_data >= 0.)
                                & (x_data <= self.length_x))
        if self.dimension == 3:  # 3D
            y_f = fire.Function(self.function_space).interpolate(self.mesh_y)
            y_data = y_f.dat.data_with_halos[:]

            # Points to update velocity model
            orig_pts = np.where((z_data >= -self.length_z) & (x_data >= 0.)
                                & (x_data <= self.length_x) & (y_data >= 0.)
                                & (y_data <= self.length_y))
            ypt_to_update = y_data[orig_pts]

        zpt_to_update = z_data[orig_pts]
        xpt_to_update = x_data[orig_pts]

        # Updating velocity model
        vel_to_update = self.c_habc.dat.data_with_halos[orig_pts]
        for idp, zc in enumerate(zpt_to_update):
            xc = xpt_to_update[idp]

            if self.dimension == 2:  # 2D
                pnt_c = (zc, xc)

            if self.dimension == 3:  # 3D
                yc = ypt_to_update[idp]
                pnt_c = (zc, xc, yc)

            vel_to_update[idp] = self.c.at(pnt_c)

        self.c_habc.dat.data_with_halos[orig_pts] = vel_to_update

        # Extending velocity model in absorbing layer
        print('Extending Profile Inside Layer')

        # Points to extend velocity model
        pad_pts = np.setdiff1d(np.arange(
            self.c_habc.dat.data_with_halos.size), orig_pts)
        zpt_to_extend = z_data[pad_pts]
        xpt_to_extend = x_data[pad_pts]

        if self.dimension == 3:  # 3D
            ypt_to_extend = y_data[pad_pts]

        # Tolerance for original boundary
        tol = 10**(min(int(np.log10(self.lmin / 10)), -6))

        vel_to_extend = self.c_habc.dat.data_with_halos[pad_pts]
        for idp, z_bnd in enumerate(zpt_to_extend):

            # Find nearest point on the boundary of the original domain
            if z_bnd < -self.length_z:
                z_bnd += self.pad_len + tol

            x_bnd = xpt_to_extend[idp]
            if x_bnd < 0. or x_bnd > self.length_x:
                x_bnd -= np.sign(x_bnd) * (self.pad_len + tol)

            # Ensure that point is within the domain bounds
            z_bnd = np.clip(z_bnd, -self.length_z, 0.)
            x_bnd = np.clip(x_bnd, 0., self.length_x)

            # Set the velocity at points in layer boundary
            if self.dimension == 2:  # 2D
                pnt_c = (z_bnd, x_bnd)

            if self.dimension == 3:  # 3D
                y_bnd = ypt_to_extend[idp]

                if y_bnd < 0. or y_bnd > self.length_y:
                    y_bnd -= np.sign(y_bnd) * (self.pad_len + tol)
                y_bnd = np.clip(y_bnd, 0., self.length_y)

                pnt_c = (z_bnd, x_bnd, y_bnd)

            vel_to_extend[idp] = self.c.at(pnt_c)

        self.c_habc.dat.data_with_halos[pad_pts] = vel_to_extend

        # Save new velocity model
        outfile = fire.VTKFile(self.path_save + "c_habc.pvd")
        outfile.write(self.c_habc)

    def fundamental_frequency(self):
        '''
        Compute the fundamental frequency in Hz via modal analysis.

        Parameters
        ----------
        None

        Returns
        ----
        None

        Verification
        ------------

        f in Hz, dx in km

        * Homogeneous domain (Comsol)
            - Dirichlet:
            m  n   Theory       dx=0.05-Q   dx=0.05-T   dx=0.01-Q   dx=0.01-T
            1  1   0.62500      0.62524     0.62531     0.62501     0.62501
            2  1   0.90139      0.90216     0.90226     0.90142     0.90142
            1  2   1.06800      1.0697      1.06960     1.06810     1.06810
            3  1   1.23111      1.2336      1.23330     1.23120     1.23120
            2  2   1.25000      1.2519      1.25240     1.25010     1.25010
            3  2   1.50520      1.5084      1.50940     1.50530     1.50540

            - Neumann:
            dx=0.05-Q   dx=0.05-T   dx=0.01-Q   dx=0.01-T
            3.5236E-8   2.9779E-8i  2.2652E-7   7.7750E-8
            0.37510     0.37507     0.37500     0.37500
            0.50023     0.50016     0.50001     0.50001
            0.62524     0.62530     0.62501     0.62501
            0.75077     0.75052     0.75003     0.75002
            0.90216     0.90227     0.90142     0.90142

            - Sommerfeld:
            dx=0.05-Q   dx=0.05-T   dx=0.01-Q   dx=0.01-T
            2.3348E-8   2.8175E-8i  2.2097E-7   7.7112E-8
            0.37513     0.37508     0.37500     0.37500
            0.50032     0.50021     0.50001     0.50001
            0.62533     0.62533     0.62501     0.62501
            0.75100     0.75065     0.75003     0.75002
            0.90240     0.90234     0.90142     0.90142

        * Bimaterial domain (Comsol)
            - Dirichlet:
            dx=0.05-Q   dx=0.05-T   dx=0.01-Q   dx=0.01-T
            0.72599     0.72606     0.72562     0.72563
            1.16740     1.16750     1.16560     1.16560
            1.23700     1.23680     1.23490     1.23490
            1.59320     1.59400     1.58940     1.58950
            1.63620     1.63560     1.63030     1.63030
            1.70870     1.70800     1.70480     1.70480

            - Neumann:
            dx=0.05-Q   dx=0.05-T   dx=0.01-Q   dx=0.01-T
            4.5197E-8   3.8054E-8i  2.8719E-7   1.0084E-7
            0.54939*    0.54933*    0.54922*    0.54921*
            0.55593     0.55590     0.55570     0.55570
            0.93184     0.93186     0.93110     0.93110
            0.95198     0.95159     0.95084     0.95082
            1.04450     1.04420     1.04280     1.04280

            - Sommerfeld:
            dx=0.05-Q   dx=0.05-T   dx=0.01-Q   dx=0.01-T
            2.9482E-8   3.5721E-8i  2.7911E-7   9.8406E-8
            0.54946     0.54937     0.54922     0.54921
            0.55603     0.55594     0.55570     0.55570
            0.93209     0.93195     0.93110     0.93110
            0.95230     0.95177     0.95084     0.95082
            1.04520     1.04460     1.04280     1.04280

        * Spyro Bimaterial Neumann:
           dx=0.05-L    %Diff-Q     %Diff-T
           0.45593      17.01       17.00

           dx=0.01-L    %Diff-Q     %Diff-T
           0.47525      13.47       13.47
        '''

        V = self.function_space
        c = self.c_habc
        u, v = fire.TrialFunction(V), fire.TestFunction(V)

        # Bilinear forms
        m = fire.inner(u, v) * fire.dx
        M = fire.assemble(m)
        m_ptr, m_ind, m_val = M.petscmat.getValuesCSR()
        Msp = ss.csr_matrix((m_val, m_ind, m_ptr), M.petscmat.size)

        a = c * c * fire.inner(fire.grad(u), fire.grad(v)) * fire.dx
        A = fire.assemble(a)
        a_ptr, a_ind, a_val = A.petscmat.getValuesCSR()
        Asp = ss.csr_matrix((a_val, a_ind, a_ptr), A.petscmat.size)

        # Operator
        print('\nSolving Eigenvalue Problem')
        if self.dimension == 2:  # 2D
            Lsp = ss.linalg.eigs(Asp, k=2, M=Msp, sigma=0.0,
                                 return_eigenvectors=False)
        if self.dimension == 3:  # 3D
            # M_ilu = ss.linalg.spilu(Msp)
            # Lsp = ss.linalg.eigs(Asp, k=2, M=Msp, sigma=0.0,
            #                      return_eigenvectors=False,
            #                      OPinv=M_ilu.solve)

            # Initialize random vectors for LOBPCG
            X = sl.orth(np.random.rand(Msp.shape[0], 2))  # Assuming k=2
            Lsp = ss.linalg.lobpcg(Asp, X, M=Msp, largest=False)[0]

        # for eigval in np.unique(Lsp):
        #     print(np.sqrt(abs(eigval)) / (2 * np.pi))

        # Fundamental frequency (eig = 0 is a rigid body motion)
        min_eigval = max(np.unique(Lsp[(Lsp > 0.) & (np.imag(Lsp) == 0.)]))
        self.fundam_freq = np.real(np.sqrt(min_eigval) / (2 * np.pi))
        print("Fundamental Frequency (Hz): {0:.5f}".format(self.fundam_freq))

    def CRminQua(self, psi, kCR, wave_p=None, typ='CR_PSI'):
        '''
        Compute a minimum  reflection coefficiente in quadratic damping.

        Parameters
        ----------
        psi: `float`
            Damping ratio
        kCR: `float`
            Adimensional parameter in reflection coefficient
        wave_p: `list`, optional
            Dimensionless wavenumbers for fundamental mode.
            wave_p = [p1, p2, ele_type]. Default is None
            -p1: Dimensionless wavenumber at the original domain boundary
            -p2: Dimensionless wavenumber at the begining of absorbing layer
            -ele_type: Element type. 'consistent' or 'lumped'
        typ: `string`, optional
            Type of reflection coefficient. Default is 'CR_PSI'.
            Option 'CR_FEM' compute a spourious reflection in FEM
        '''

        if typ == 'CR_PSI':
            # Minimum coefficient reflection
            psimin = psi
            CRmin = psimin**2 / (kCR**2 + psimin**2)

        elif typ == 'CR_FEM':
            # Unidimensional spourious reflection in FEM (Laier, 2020)

            p1, p2, ele_type = wave_p  # Dimensionless wavenumbers

            def Zi(p, alpha, ele_type):
                '''
                Compute the Z parameter for the spurious reflection coefficient

                Parameters
                ----------
                p: `float`
                    Dimensionless wavenumber
                alpha: `float`
                    Ratio between the representative mesh dimensions
                ele_type: `string`
                    Element type. 'consistent' or 'lumped'

                Returns
                -------
                Z_fem: `float`
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

            if CRfdm == 0:
                psimin = 0
            else:
                psimin = kCR / (1 / CRfdm - 1)**0.5

            # Correction for the dimensionless wavenumbers due to the damping
            p2 *= (1 + 1 / 8 * (psimin * self.a_par / self.F_L)**2)

            # Ratio between the representative mesh dimensions
            alpha = self.lmax / self.lmin

            # Z's parameters for the spurious reflection coefficient
            Z1 = Zi(p1, alpha, ele_type)
            Z2 = Zi(p2, alpha, ele_type)

            # Spurious reflection coefficient in FEM (Laier, 2020)
            aux0 = (1 - Z1) * np.sin(p1)
            aux1 = (alpha * Z2 - 1) * np.sin(alpha * p2) / alpha
            CRmin = abs((aux0 + aux1) / (aux0 - aux1))

        xCRmin = psimin / self.d

        return CRmin, xCRmin

    @staticmethod
    def xCR_vert(x, y, xCR_inf):
        '''
        Perform a quadratic regression of CRmin vs xCR.
        '''

        # Coefficients for the quadratic
        z = np.polyfit(x, y, 2)
        xv = -z[1] / (2 * z[0])
        roots = np.roots(z)

        # Vertex or minimum positive root
        if xv < xCR_inf:
            ind = np.where(roots >= xCR_inf)[0][0]
            return roots[ind], 0
        else:
            return xv, -z[1]**2 / (4 * z[0]) + z[2]

    def est_min_damping(self, psi=0.999, m=1):
        '''
        Estimation of xCR
        CRref > 0, because always have both reflections: physical and spurious)
        '''

        # Dimensionless wave numbers
        c_ref = min([bnd[1] for bnd in self.eik_bnd])
        pmin = 2 * np.pi * self.freq_ref * self.lmin / c_ref
        pmax = 2 * np.pi * self.freq_ref * self.lmax / c_ref
        p = [pmin, pmax, self.variant]

        # Adimensional parameter in reflection coefficient
        kCR = 4 * self.F_L / (self.a_par * m)
        c_bnd = self.eik_bnd[0][1]
        kCRp = kCR * c_bnd / c_ref

        # Limits for the minimum damping ratio
        psimin_inf = psi * self.d**2
        CRmin_inf, xCR_inf = self.CRminQua(psimin_inf, kCRp)

        psimin_sup = psi * (2. * self.d - self.d**2)
        CRmin_sup, xCR_sup = self.CRminQua(psimin_sup, kCRp)

        # Initial guess
        psimin_ini = psi * (self.d**2 + self.d) / 2.
        CRmin_ini, xCR_ini = self.CRminQua(psimin_ini, kCRp)

        # Spurious reflection
        CRmin_fem, xCR_fem = self.CRminQua(psi, kCRp, wave_p=p, typ='CR_FEM')

        # Vertex or minimum positive root
        x_reg = [xCR_inf, xCR_sup, xCR_ini, xCR_fem]
        y_reg = [CRmin_inf, CRmin_sup, CRmin_ini, CRmin_fem]
        xCRreg, CRreg = self.xCR_vert(x_reg, y_reg, xCR_inf)

        # Heuristic factor for the minimum damping ratio
        psi_min = np.clip(psimin_ini, psimin_inf, psimin_sup)
        xCR = psi_min / self.d

        # Model dimensions
        a_rect = self.Lx_habc
        b_rect = self.Lz_habc

        if self.dimension == 2:  # 2D

            # Axpect ratio for 2D: a/b
            Ra = a_rect / b_rect
        # # Errors
            if self.layer_shape == 'rectangular':
                Fa = 4
            elif self.layer_shape == 'hyperelliptical':
                # Fa = FactA
                pass

        if self.dimension == 3:  # 3D

            # Adding a dimension for 3D
            c_rect = self.Ly_habc

            # Aspect ratio for 3D: a/b * a/c * c/b
            Rv = a_rect**2 / b_rect**2

            if self.layer_shape == 'rectangular':
                Fv = 8
            elif self.layer_shape == 'hyperelliptical':
                # Fv = FactV
                pass

        ipdb.set_trace()

        # Computed values and its range
        print('Minimum Damping Ratio: {:.5f}'.format(psi_min))
        psi_str = 'Range for Minimum Damping Ratio. Min:{:.5f} - Max:{:.5f}'
        print(psi_str.format(psimin_inf, psimin_sup))
        print('Heuristic Factor xCR: {:.3f}'.format(xCR))
        xcr_str = 'Range Values for xCR Factor. Min:{:.3f} - Max:{:.3f}'
        print(xcr_str.format(xCR_inf, xCR_sup))

        # err1 = abs(np.sin(pCR/2))
        # err2 = abs(-1 + np.cos(pCR))
        # # Reference for calculation of interval of xCR
        # if CRreg != 0:
        #     xCRref = xCRreg
        #     CRref = CRreg
        # else:
        #     CRref = CRIni
        # # Bounds: Minimum errors
        # lb1 = min(coeDampFun(CRref*(1 - err1), kCR, d, psi)[2]/d, xCRSup)
        # lb2 = min(coeDampFun(CRref*(1 - err2), kCR, d, psi)[2]/d, xCRSup)
        # if CRreg != 0:
        #     xCRmin = max(min(lb1, lb2), xCRInf)
        # else:
        #     xCRmin = xCRreg
        #     xCRref = max(min(lb1, lb2), xCRInf)
        # # Bounds: Maximum errors
        # ub1 = min(coeDampFun(CRref*(1 + err1), kCR, d, psi)[2]/d, xCRSup)
        # ub2 = min(coeDampFun(CRref*(1 + err2), kCR, d, psi)[2]/d, xCRSup)
        # xCRmax = max(ub1, ub2)

        # if Ra > 0:
        #     cad1 = 'Range Values for 1D-xCR Factor: '
        #     cad2 = 'RefMin:{:2.2f} - RefMax:{: 2.2f}'
        #     cad = cad1 + cad2
        #     mp.my_print(cad.format(xCRmin, xCRmax))
        #     # Factors: 1/sqrt(1 + Ra), Ra/sqrt(1 + Ra) and their inverses
        #     fRamin = Fa/4*min(1/(1 + Ra**2)**0.5, Ra/(1 + Ra**2)**0.5)
        #     fRamax = 4/Fa*max((1 + Ra**2)**0.5, (1 + Ra**2)**0.5/Ra)
        #     # Reference value
        #     xCRmin = max(xCRmin*fRamin, xCRInf)
        #     xCRmax = min(xCRmax*fRamax, xCRSup)

        return psi_min, xCR

    def calc_damping_prop(self, psi=0.999, m=1):
        '''
        Compute the damping properties for the absorbing layer.

        Parameters
        ----------
        psi: `float`, optional
            Damping ratio. Default is 0.999
        m: `float`, optional
            Vibration mode. Default is 1 (Fundamental mode)

        Returns
        -------
        eta_crt: `float`
            Critical damping coefficient (1/s)
        psi_min: `float`
            Minimum damping ratio of the absorbing layer
        xCR: `float`
            Heuristic factor for the minimum damping ratio (psi_min = xCR * d)
        '''

        # Critical damping coefficient
        eta_crt = 2 * np.pi * self.fundam_freq
        eta_max = psi * eta_crt
        print("Critical Damping Coefficient (1/s): {0:.5f}".format(eta_crt))
        print("Maximum Damping Coefficient (1/s): {0:.5f}".format(eta_max))

        psi_min, xCR = self.est_min_damping()

        return eta_crt, psi_min, xCR

    def coeff_damp_fun(self, psi_min, psi=0.999):
        '''
        Compute the coefficients for quadratic damping function

        Parameters
        ----------
        psi_min': `float`
            Minimum damping ratio
        psi: `float`, optional
            Damping ratio. Default is 0.999

        Returns
        -------
        aq: `float`
            Coefficient for quadratic term in the damping function
        bq: `float`
            Coefficient bq for linear term in the damping function
        '''

        aq = (psi_min - self.d * psi) / (self.d**2 - self.d)
        bq = psi - aq

        return aq, bq

    def damping_layer(self):
        '''
        Set the damping profile within the absorbing layer.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Estimating fundamental frequency
        self.fundamental_frequency()

        # Initialize damping field
        print('\nCreating Damping Profile')
        self.eta_habc = fire.Function(self.function_space, name='eta [1/s])')

        # Dimensions and node coordinates
        Lz = self.length_z
        Lx = self.length_x
        z_f = fire.Function(self.function_space).interpolate(self.mesh_z)
        x_f = fire.Function(self.function_space).interpolate(self.mesh_x)

        # Define the damping expressions
        z_pd_sqr = fire.conditional(z_f + Lz < 0, (z_f + Lz)**2, 0.)
        x_pd_sqr = fire.conditional(x_f < 0, x_f**2, 0.) + \
            fire.conditional(x_f - Lx > 0, (x_f - Lx)**2, 0.)

        # Reference distance to the original boundary
        if self.dimension == 2:  # 2D
            ref = fire.sqrt(z_pd_sqr + x_pd_sqr) / fire.Constant(self.pad_len)
        if self.dimension == 3:  # 3D
            Ly = self.length_y
            y_f = fire.Function(self.function_space).interpolate(self.mesh_y)
            y_pd_sqr = fire.conditional(y_f < 0, y_f**2, 0.) + \
                fire.conditional(y_f - Ly > 0, (y_f - Ly)**2, 0.)
            ref = fire.sqrt(z_pd_sqr + x_pd_sqr
                            + y_pd_sqr) / fire.Constant(self.pad_len)

        # Compute the minimum damping ratio and its heuristic factor
        eta_crt, psi_min, xCR = self.calc_damping_prop()

        # Compute the coefficients for quadratic damping function
        aq, bq = self.coeff_damp_fun(psi_min)

        # Apply damping profile
        expr_damp = fire.Constant(eta_crt) * (fire.Constant(aq) * ref**2
                                              + fire.Constant(bq) * ref)
        self.eta_habc.interpolate(expr_damp)

        # Save damping profile
        outfile = fire.VTKFile(self.path_save + "eta_habc.pvd")
        outfile.write(self.eta_habc)
