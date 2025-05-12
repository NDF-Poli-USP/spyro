import firedrake as fire
import scipy.linalg as sl
import scipy.sparse as ss
import spyro.habc.eik as eik
import spyro.habc.lay_len as lay_len
import numpy as np
from os import getcwd
from scipy.signal import find_peaks
from scipy.spatial import KDTree
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.habc.hyp_lay import HyperLayer
from spyro.habc.nrbc import NRBCHabc
from time import perf_counter  # For runtime
from tracemalloc import get_traced_memory, start, stop  # For memory usage
import ipdb
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


def comp_cost(flag, tRef=None):
    '''
    Estimate runtime and used memory and save them to a *.txt file.

    Parameters
    ----------
    flag : str
        Flag to indicate the action to be performed
        - 'tini' to start the timer
        - 'tfin' to finish the timer and print the results
    tRef : float, optional
        Reference time in seconds. Default is None

    Returns
    -------
    tRef : float
        Reference time in seconds. Only returned if flag is 'tini'
    '''

    if flag == 'tini':
        # Start memory usage
        start()

        # Reference time. Don't move this line!
        tRef = perf_counter()

        return tRef

    elif flag == 'tfin':

        # Separation format
        hifem_draw = 62

        # Total time
        print('\n' + hifem_draw * '-')
        print("Estimating Runtime and Used Memory")
        tTotal = perf_counter() - tRef
        val_time = [tTotal, tTotal/60, tTotal/3600]
        cad_time = 'Runtime: (s):{:3.3f}, (m):{:3.3f}, (h):{:3.3f}'
        print(cad_time.format(*val_time))

        # Memory usage
        curr, peak = get_traced_memory()
        val_memo = [curr/1024**2, peak/1024**2]
        cad_memo = "Used Memory: Current (MB):{:3.3f}, Peak (MB):{:3.3f}"
        print(cad_memo.format(*val_memo))
        print(hifem_draw * '-' + '\n')
        stop()

        # Save file for resource usage
        file_name = 'cost.txt'
        path_cost = getcwd() + "/output/" + file_name
        np.savetxt(path_cost, (*val_time, *val_memo), delimiter='\t')


class HABC_Wave(AcousticWave, HyperLayer, NRBCHabc):
    '''
    class HABC that determines absorbing layer size and parameters to be used.

    Attributes
    ----------
    a_par : `float`
        Adimensional propagation speed parameter (a = z / f).
        "z" parameter is the inverse of the minimum Eikonal (1 / phi_min)
    a_rat: `float`
        Area ratio to the area of the original domain. a_rat = area / a_orig
    area : `float`
        Area of the domain with absorbing layer
    bnds : `list` of `arrays`
        Mesh point indices on boundaries of the domain. Structure:
        - [left_boundary, right_boundary, bottom_boundary] for 2D
        - [left_boundary, right_boundary, bottom_boundary,
            left_bnd_y, right_bnd_y] for 3D
    bnd_nodes : `list` of `arrays`
        Mesh point coordinates on boundaries of the domain. Structure:
        - [z_data[bnds], x_data[bnds]] for 2D
        - [z_data[bnds], x_data[bnds], y_data[bnds]] for 3D
    c : `firedrake function`
        Velocity model without absorbing layer
    c_habc : `firedrake function`
        Velocity model with absorbing layer
    d_par : `float`
        Normalized element size (lmin / pad_len)
    eik_bnd : `list`
        Properties on boundaries according to minimum values of Eikonal
        Structure sublist: [pt_cr, c_bnd, eikmin, z_par, lref, sou_cr]
        - pt_cr : Critical point coordinates
        - c_bnd : Propagation speed at critical point
        - eikmin : Eikonal value in seconds
        - z_par : Inverse of minimum Eikonal (Equivalent to c_bound / lref)
        - lref : Distance to the closest source from critical point
        - sou_cr : Critical source coordinates
    err_habc : `list`
        Error measures at the receivers for the HABC scheme.
        Structure sublist: [errIt, errPk, pkMax]
        - errIt : Integral error
        - errPk : Peak error
        - pkMax : Maximum reference peak
    eta_habc : `firedrake function`
        Damping profile within the absorbing layer
    eta_mask : `firedrake function`
        Mask function to identify the absorbing layer domain
    F_L : `float`
        Size  parameter of the absorbing layer
    f_Ah : `float`
        Hyperelliptical area factor. f_Ah = area / (a_hyp * b_hyp).
        f_Ah is 4 for rectangular layers
    f_Vh : `float`
        Hyperellipsoidal volume factor. f_Vh = vol / (a_hyp * b_hyp * c_hyp).
        f_Vh is 8 for rectangular layers
    f_est : `float`
        Factor for the stabilizing term in Eikonal equation
    freq_ref : `float`
        Reference frequency of the wave at the minimum Eikonal point
    fundam_freq : `float`
        Fundamental frequency of the numerical model
    fwi_iter : `int`
        The iteration number for the Full Waveform Inversion (FWI) algorithm
    Lz_habc : `float`
        Length of the domain in the z-direction with absorbing layer
    Lx_habc : `float`
        Length of the domain in the x-direction with absorbing layer
    Ly_habc : `float`
        Length of the domain in the y-direction with absorbing (3D models)
    layer_shape : `string`
        Shape type of pad layer. Options: 'rectangular' or 'hypershape'
    lmin : `float`
        Minimum mesh size
    lmax : `float`
        Maxmum mesh size
    mesh_original : `firedrake mesh`
        Original mesh without absorbing layer
    n_bounds : `tuple`
        Bounds for the hypershape layer degree. (n_min, n_max)
        - n_min ensures to add lmin in the domain diagonal direction
        - n_max ensures to add pad_len in the domain diagonal direction
    n_hyp : `int`
        Degree of the hyperelliptical pad layer (n >= 2). Default is 2.
        For rectangular layers, n_hyp is set to infinity
    pad_len : `float`
        Size of the absorbing layer
    path_save : `string`
        Path to save data
    tol : `float`
        Tolerance for searching nodes on the boundary
    v_rat : `float`
        Volume ratio to the volume of the original domain. v_rat = vol / v_orig
    vol : `float`
        Volume of the domain with absorbing layer
    xCR : `float`
        Heuristic factor for the minimum damping ratio (psi_min = xCR * d)
    xCR_bounds: `list`
        Bounds for the heuristic factor. [xCR_lim, xCR_search]
        Structure: [[xCR_inf, xCR_sup], [xCR_min, xCR_max]]
        - xCR_lim : Limits for the heuristic factor.
        - xCR_search : Initial search range for the heuristic factor

    Methods
    -------
    boundary_data()
        Generate the boundary data from the original domain mesh
    calc_damping_prop()
        Compute the damping properties for the absorbing layer
    calc_rec_geom_prop()
        Calculate the geometric properties for the rectangular layer
    coeff_damp_fun()
        Compute the coefficients for quadratic damping function
    cond_marker_for_eta()
        Define the conditional expressions to identify the domain of
        the absorbing layer or the reference to the original boundary
        to compute the damping profile inside the absorbing layer
    create_mesh_habc()
        Create a mesh with absorbing layer based on the determined size
    damping_layer()
        Set the damping profile within the absorbing layer
    det_reference_freq()
        Determine the reference frequency for a new layer size
    error_measures_habc()
        Compute the error measures at the receivers for the HABC scheme.
    est_min_damping()
        Estimate the minimum damping ratio and the associated heuristic factor
    fundamental_frequency()
        Compute the fundamental frequency in Hz via modal analysis
    min_reflection()
        Compute a minimum reflection coefficiente for the quadratic damping
    preamble_mesh_operations()
        Perform mesh operations previous to size an absorbing layer
    properties_eik_mesh()
        Set the properties for the mesh used to solve the Eikonal equation
    regression_CRmin()
        Define the minimum damping ratio and the associated heuristic factor
    roundFL()
        Adjust the layer parameter based on the element size
    size_habc_criterion()
        Determine the size of the absorbing layer using the Eikonal criterion
    velocity_habc()
        Set the velocity model for the model with absorbing layer
    xCR_search_range()
        Determine the initial search range for the heuristic factor xCR
    '''

    def __init__(self, dictionary=None, f_est=0.06, fwi_iter=0, comm=None):
        '''
        Initialize the HABC class.

        Parameters
        ----------
        dictionary : dict, optional
            A dictionary containing the input parameters for the HABC class
        f_est : `float`, optional
            Factor for the stabilizing term in Eikonal equation. Default is 0.06
        fwi_iter : int, optional
            The iteration number for the FWI algorithm. Default is 0
        comm : object, optional
            An object representing the communication interface

        Returns
        -------
        None
        '''

        AcousticWave.__init__(self, dictionary=dictionary, comm=comm)
        NRBCHabc.__init__(self)

        # Layer shape
        self.layer_shape = self.abc_boundary_layer_shape
        if self.layer_shape == 'rectangular':
            print("\nAbsorbing Layer Shape: Rectangular")

        elif self.layer_shape == 'hypershape':
            HyperLayer.__init__(self, n_hyp=self.abc_deg_layer,
                                dimension=self.dimension)
            print("\nAbsorbing Layer Shape: Hypershape")

        else:
            aux0 = "Please use 'rectangular' or 'hypershape', "
            UserWarning(aux0 + f"{self.layer_shape} not supported.")

        # Factor for the stabilizing term in Eikonal equation
        self.f_est = f_est

        # Current iteration
        self.fwi_iter = fwi_iter

        # Path to save data
        self.path_save = getcwd() + "/output/"

    def preamble_mesh_operations(self, p_usu=None):
        '''
        Perform mesh operations previous to size an absorbing layer.

        Parameters
        ----------
        p_usu : `int`, optional
            Finite element order. Default is None

        Returns
        -------
        None
        '''

        # Save a copy of the original mesh
        self.mesh_original = self.mesh
        mesh_orig = fire.VTKFile(self.path_save + "mesh_orig.pvd")
        mesh_orig.write(self.mesh_original)

        # Velocity profile model
        self.c = fire.Function(self.function_space, name='c_orig [km/s])')
        self.c.interpolate(self.initial_velocity_model)

        # Save initial velocity model
        vel_c = fire.VTKFile(self.path_save + "c_vel.pvd")
        vel_c.write(self.c)

        # Mesh properties for Eikonal
        self.properties_eik_mesh(p_usu=p_usu)

        # Generating boundary data from the original domain mesh
        self.boundary_data()

    def properties_eik_mesh(self, p_usu=None):
        '''
        Set the properties for the mesh used to solve the Eikonal equation.

        Parameters
        ----------
        p_usu : `int`, optional
            Finite element order. Default is None

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

    def boundary_data(self, typ_bnd='original'):
        '''
        Generate the boundary data from the original domain mesh.

        Parameters
        ----------
        typ_bnd : str, optional
            Type of boundary data to extract. Options: 'original' for original
            domain or 'eikonal' for Eikonal analysis. Default is 'original'

        Returns
        -------
        bnds : `list` of 'arrays'
            Mesh point indices on boundaries of the domain. Structure:
            - [left_boundary, right_boundary, bottom_boundary] for 2D
            - [left_boundary, right_boundary, bottom_boundary,
                left_bnd_y, right_bnd_y] for 3D
        node_positions : `list`, optional
            List of node positions for Eikonal analysis. Structure:
            - [z_data, x_data] for 2D
            - [z_data, x_data, y_data] for 3D
        '''

        if typ_bnd == 'original':
            func_space = self.function_space
        elif typ_bnd == 'eikonal':
            func_space = self.funct_space_eik
        else:
            aux0 = "Please use 'original' or 'eikonal', "
            UserWarning(aux0 + f"{self.layer_shape} not supported.")

        # Extract node positions
        z_f = fire.Function(func_space).interpolate(self.mesh_z)
        x_f = fire.Function(func_space).interpolate(self.mesh_x)
        z_data = z_f.dat.data_with_halos[:]
        x_data = x_f.dat.data_with_halos[:]

        if self.dimension == 3:  # 3D
            y_f = fire.Function(func_space).interpolate(self.mesh_y)
            y_data = y_f.dat.data_with_halos[:]

        # Tolerance for boundary
        self.tol = 10**(min(int(np.log10(self.lmin / 10)), -6))

        # Boundaries
        left_boundary = np.where(x_data <= self.tol)
        right_boundary = np.where(x_data >= self.length_x - self.tol)
        bottom_boundary = np.where(z_data <= self.tol - self.length_z)

        bnds = [left_boundary, right_boundary, bottom_boundary]

        if self.dimension == 3:  # 3D
            left_bnd_y = np.where(y_data <= self.tol)
            right_bnd_y = np.where(y_data >= self.length_y - self.tol)
            bnds += [left_bnd_y, right_bnd_y]

        if typ_bnd == 'original':
            self.bnds = np.unique(np.concatenate(
                [idxs for idx_list in bnds for idxs in idx_list]))
            self.bnd_nodes = [z_data[self.bnds], x_data[self.bnds]]
            if self.dimension == 3:  # 3D
                self.bnd_nodes.append(y_data[self.bnds])

        elif typ_bnd == 'eikonal':
            node_positions = [z_data, x_data]
            if self.dimension == 3:  # 3D
                node_positions.append(y_data)
            return bnds, node_positions

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

        print("\nModifying Layer Size Based on the Element Size")
        print("Modified Parameter Size FL: {:.4f}".format(self.F_L))
        print("Modified Layer Size (km): {:.4f}".format(self.pad_len))
        print("Elements ({:.3f} km) in Layer: {}".format(
            self.lmin, int(self.pad_len / self.lmin)))

    def det_reference_freq(self, histPcrit, fpad=4):
        '''
        Determine the reference frequency for a new layer size.

        Parameters
        ----------
        histPcrit : `array`
            Transient response at the minimum Eikonal point
        fpad : `int`, optional
            Padding factor for FFT. Default is 4

        Returns
        -------
        None
        '''

        print("\nDetermining Reference Frequency")

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

    def calc_rec_geom_prop(self):
        '''
        Calculate the geometric properties for the rectangular layer.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        self.n_hyp = np.inf
        self.n_bounds = None

        # Geometric properties of the rectangular layer
        if self.dimension == 2:  # 2D
            self.area = self.Lx_habc * self.Lz_habc
            self.a_rat = self.area / (self.length_x * self.length_z)
            self.f_Ah = 4

        if self.dimension == 3:  # 3D
            self.vol = self.Lx_habc * self.Lz_habc * self.Ly_habc
            self.v_rat = self.vol / (self.length_x * self.length_z
                                     * self.length_y)
            self.f_Vh = 8

    def size_habc_criterion(self, Eikonal, histPcrit,
                            layer_based_on_mesh=False):
        '''
        Determine the size of the absorbing layer using the Eikonal
        criterion for HABCs. See Salas et al (2022) for details.

        Parameters
        ----------
        Eikonal : `eikonal`
            An object representing the Eikonal solver
        histPcrit : `array`
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
        Eikonal.solve_eik(self, f_est=self.f_est)

        # Identifying critical points
        self.eik_bnd = Eikonal.ident_crit_eik(self)

        # Critical point coordinates as receivers
        pcrit = [bnd[0] for bnd in self.eik_bnd]
        self.receiver_locations = pcrit + self.receiver_locations
        self.number_of_receivers = len(self.receiver_locations)

        # Determining the reference frequency
        self.det_reference_freq(histPcrit)

        # Computing layer sizes
        self.F_L, self.pad_len, self.a_par = lay_len.calc_size_lay(self)

        if layer_based_on_mesh:
            self.roundFL()

        # Normalized element size
        self.d = self.lmin / self.pad_len
        print("Normalized Element Size (adim): {0:.5f}".format(self.d))

        # New geometry with layer
        self.Lx_habc = self.length_x + 2 * self.pad_len
        self.Lz_habc = self.length_z + self.pad_len

        if self.dimension == 3:  # 3D
            self.Ly_habc = self.length_y + 2 * self.pad_len

        if self.layer_shape == 'hypershape':

            print("\nDetermining Hypershape Layer Parameters")

            # Original domain dimensions
            domain_dim = [self.length_x, self.length_z]
            domain_hyp = [self.Lx_habc, self.length_z + 2 * self.pad_len]
            if self.dimension == 3:  # 3D
                domain_dim.append(self.length_y)
                domain_hyp.append(self.Ly_habc)

            # Defining the hypershape semi-axes
            self.define_hyperaxes(domain_dim, domain_hyp)

            # Degree of the hypershape layer
            self.define_hyperlayer(self.pad_len, self.lmin)

            # Geometric properties of the hypershape layer
            self.calc_hyp_geom_prop()

        else:

            print("\nDetermining Rectangular Layer Parameters")

            # Geometric properties of the rectangular layer
            self.calc_rec_geom_prop()

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

        print("\nGenerating Mesh with Absorbing Layer")

        # New mesh with layer
        if self.layer_shape == 'rectangular':

            # New geometry with layer
            Lz = self.Lz_habc
            Lx = self.Lx_habc

            # Number of elements
            n_pad = self.pad_len / self.lmin  # Elements in the layer
            nz = int(self.length_z / self.lmin) + int(n_pad)
            nx = int(self.length_x / self.lmin) + int(2 * n_pad)
            nx = nx + nx % 2

            if self.dimension == 2:  # 2D
                mesh_habc = fire.RectangleMesh(nz, nx, Lz, Lx)

            if self.dimension == 3:  # 3D
                Ly = self.Ly_habc
                ny = int(self.length_y / self.lmin) + int(2 * n_pad)
                ny = ny + ny % 2
                mesh_habc = fire.BoxMesh(nz, nx, ny, Lz, Lx, Ly)
                mesh_habc.coordinates.dat.data_with_halos[:, 2] -= self.pad_len

            # Adjusting coordinates
            mesh_habc.coordinates.dat.data_with_halos[:, 0] *= -1.0
            mesh_habc.coordinates.dat.data_with_halos[:, 1] -= self.pad_len

            print("Extended Rectangular Mesh Generated Successfully")

        elif self.layer_shape == 'hypershape':

            # Creating the hyperellipse layer mesh
            hyp_mesh = self.create_hyp_trunc_mesh2D()

            # Adjusting coordinates
            coords = hyp_mesh.coordinates.dat.data_with_halos
            coords[:, 0], coords[:, 1] = coords[:, 1], -coords[:, 0]
            Lz_half = self.length_z / 2
            Lx_half = self.length_x / 2
            hyp_mesh.coordinates.dat.data_with_halos[:, 0] -= Lz_half
            hyp_mesh.coordinates.dat.data_with_halos[:, 1] += Lx_half
            # fire.VTKFile("output/trunc_hyp_test.pvd").write(hyp_mesh)

            # Merging the original mesh with the hyperellipse layer mesh
            mesh_habc = self.merge_mesh_2D(self.mesh_original, hyp_mesh)
            # fire.VTKFile("output/trunc_merged_test.pvd").write(mesh_habc)

        # Updating the mesh with the absorbing layer
        self.set_mesh(user_mesh=mesh_habc, mesh_parameters={})
        print("Mesh Generated Successfully")

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

        Improvements
        ------------
        dx = 0.05 km
        New approach: 1.602 1.495 1.588 mean = 1.562
        Old approach: 1.982 2.124 1.961 mean = 2.022

        dx = 0.02 km
        New approach: 5.276 5.214 6.275 mean = 5.588
        Old approach: 12.232 12.372 12.078 = 12.227
        '''

        print("\nUpdating Velocity Profile")

        # Initialize velocity field
        self.c = fire.Function(self.function_space, name='c [km/s])')

        # Assigning the original velocity model to the new mesh
        self.c.interpolate(self.initial_velocity_model, allow_missing_dofs=True)

        # Extending velocity model within the absorbing layer
        print("Extending Profile Inside Layer")

        # Extract node positions - To do: Refactor
        z_f = fire.Function(self.function_space).interpolate(self.mesh_z)
        x_f = fire.Function(self.function_space).interpolate(self.mesh_x)
        z_data = z_f.dat.data_with_halos[:]
        x_data = x_f.dat.data_with_halos[:]

        # Points to extend the velocity model
        if self.dimension == 2:  # 2D
            pad_pts = np.where((z_data < -self.length_z) | (x_data < 0.)
                               | (x_data > self.length_x))

        if self.dimension == 3:  # 3D
            y_f = fire.Function(self.function_space).interpolate(self.mesh_y)
            y_data = y_f.dat.data_with_halos[:]
            pad_pts = np.where((z_data < -self.length_z) | (x_data < 0.)
                               | (x_data > self.length_x) | (y_data < 0.)
                               | (y_data > self.length_y))
            ypt_to_extend = y_data[pad_pts]

        zpt_to_extend = z_data[pad_pts]
        xpt_to_extend = x_data[pad_pts]

        # Velocity profile inside the layer
        vel_to_extend = self.c.dat.data_with_halos[pad_pts]

        for idp, z_bnd in enumerate(zpt_to_extend):

            # Find nearest point on the boundary of the original domain
            if z_bnd < -self.length_z:
                z_bnd += self.pad_len + self.tol

            x_bnd = xpt_to_extend[idp]
            if x_bnd < 0. or x_bnd > self.length_x:
                x_bnd -= np.sign(x_bnd) * (self.pad_len + self.tol)

            # Ensure that point is within the domain bounds
            z_bnd = np.clip(z_bnd, -self.length_z, 0.)
            x_bnd = np.clip(x_bnd, 0., self.length_x)

            # Set the velocity of the nearest point on the original boundary
            if self.dimension == 2:  # 2D
                pnt_c = (z_bnd, x_bnd)

            if self.dimension == 3:  # 3D
                y_bnd = ypt_to_extend[idp]

                if y_bnd < 0. or y_bnd > self.length_y:
                    y_bnd -= np.sign(y_bnd) * (self.pad_len + self.tol)
                y_bnd = np.clip(y_bnd, 0., self.length_y)

                pnt_c = (z_bnd, x_bnd, y_bnd)

            vel_to_extend[idp] = self.initial_velocity_model.at(pnt_c)

        # Assign the extended velocity model to the absorbing layer
        self.c.dat.data_with_halos[pad_pts] = vel_to_extend

        # Save new velocity model
        outfile = fire.VTKFile(self.path_save + "c_habc.pvd")
        outfile.write(self.c)

    def fundamental_frequency(self, monitor=False):
        '''
        Compute the fundamental frequency in Hz via modal analysis.

        Parameters
        ----------
        monitor : `bool`, optional
            Print on screen the computed natural frequencies. Default is False

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
        c = self.c
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

        # Operator - To do: Slepc
        print("\nSolving Eigenvalue Problem")
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

        if monitor:
            for n_eig, eigval in enumerate(np.unique(Lsp)):
                f_eig = np.sqrt(abs(eigval)) / (2 * np.pi)
                print("Frequency {} (Hz): {0:.5f}".format(n_eig, f_eig))

        # Fundamental frequency (eig = 0 is a rigid body motion)
        min_eigval = max(np.unique(Lsp[(Lsp > 0.) & (np.imag(Lsp) == 0.)]))
        self.fundam_freq = np.real(np.sqrt(min_eigval) / (2 * np.pi))
        print("Fundamental Frequency (Hz): {0:.5f}".format(self.fundam_freq))

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

            # Z's parameters for the spurious reflection coefficient
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

    def regression_CRmin(self, data_reg, xCR_lim, kCR):
        '''
        Define the minimum damping ratio and the associated heuristic factor.

        Parameters
        ----------
        data_reg : `list`
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
            Minimum damping ratio
        xCR_min : `float`
            Heuristic factor for the minimum damping ratio
        CRmin : `float`
            Minimum reflection coefficient at the minimum damping ratio
        '''

        # Data for regression
        x, y = data_reg

        # Limits for the heuristic factor
        xCR_inf, xCR_sup, xCR_ini = xCR_lim

        # Coefficients for the quadratic equation
        z = np.polyfit(x, y, 2)

        # Roots of the quadratic equation
        roots = np.roots(z)

        # Vertex or minimum positive root
        xCR_vtx = -z[1] / (2 * z[0])
        max_root = max(roots)
        xCR_min = xCR_vtx if xCR_vtx > xCR_inf else (
            max_root if max_root > xCR_inf else xCR_ini)
        xCR_min = np.clip(xCR_min, xCR_inf, xCR_sup)

        # Minimum damping ratio
        psi_min = xCR_min * self.d
        CRmin = self.min_reflection(kCR, psi=psi_min)[0]

        return psi_min, xCR_min, CRmin

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
            fmin = f_Vh / 8 * min(fainv, fbinv, min(fc, 1 / fc))
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
            Minimum damping ratio of the absorbing layer
        xCR : `float`
            Heuristic factor for the minimum damping ratio (psi_min = xCR * d)
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
        data_reg = [x_reg, y_reg]
        xCR_lim = [xCR_inf, xCR_sup, xCR_ini]
        psi_min, xCR, CRmin = self.regression_CRmin(data_reg, xCR_lim, kCRp)

        xCR_search = self.xCR_search_range(CRmin, kCRp, p[:2], xCR_lim[:2])

        return psi_min, xCR, xCR_lim[:2], xCR_search

    def calc_damping_prop(self, psi=0.999, m=1):
        '''
        Compute the damping properties for the absorbing layer.

        Parameters
        ----------
        psi : `float`, optional
            Damping ratio. Default is 0.999
        m : `int`, optional
            Vibration mode. Default is 1 (Fundamental mode)

        Returns
        -------
        eta_crt : `float`
            Critical damping coefficient (1/s)
        psi_min : `float`
            Minimum damping ratio of the absorbing layer
        xCR : `float`
            Heuristic factor for the minimum damping ratio (psi_min = xCR * d)
        psimin_lim : `list`
            Limits for the minimum damping ratio. [psimin_inf, psimin_sup]
        xCR_lim : `list`
            Limits for the heuristic factor. [xCR_inf, xCR_sup]
        xCR_search : `list`
            Initial search range for the heuristic factor. [xCR_min, xCR_max]
        '''

        # Critical damping coefficient
        eta_crt = 2 * np.pi * self.fundam_freq
        eta_max = psi * eta_crt
        print("Critical Damping Coefficient (1/s): {0:.5f}".format(eta_crt))
        print("Maximum Damping Coefficient (1/s): {0:.5f}".format(eta_max))

        # Minimum damping ratio and the associated heuristic factor
        psi_min, xCR, xCR_lim, xCR_search = self.est_min_damping()
        xCR_inf, xCR_sup = xCR_lim
        xCR_min, xCR_max = xCR_search

        # Range for Minimum Damping Ratio. Min:0.01233 - Max:0.20967

        # Computed values and its range
        print("Minimum Damping Ratio: {:.5f}".format(psi_min))
        psi_str = "Range for Minimum Damping Ratio. Min:{:.5f} - Max:{:.5f}"
        print(psi_str.format(xCR_inf * self.d, xCR_sup * self.d))
        print("Heuristic Factor xCR: {:.3f}".format(xCR))
        xcr_str = "Range Values for xCR Factor. Min:{:.3f} - Max:{:.3f}"
        print(xcr_str.format(xCR_inf, xCR_sup))
        lim_str = "Initial Range for xCR Factor. Min:{:.3f} - Max:{:.3f}"
        print(lim_str.format(xCR_min, xCR_max))

        return eta_crt, psi_min, xCR, xCR_lim, xCR_search

    def coeff_damp_fun(self, psi_min, psi=0.999):
        '''
        Compute the coefficients for quadratic damping function.

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

    def cond_marker_for_eta(self, nodes_coord, typ_marker='damping'):
        '''
        Define the conditional expressions to identify the domain of
        the absorbing layer or the reference to the original boundary
        to compute the damping profile inside the absorbing layer.

        Parameters
        ----------
        nodes_coord : `list`
            Node coordinates. [z_f, x_f, y_f]
        typ_marker : `string`, optional
            Type of marker. Default is 'damping'.
            - 'damping' : Get the reference distance to the original boundary
            - 'mask' : Define a mask to filter the layer boundary domain

        Returns
        -------
        ref :
            - 'damping' : `ufl.conditional.Conditional`
                Reference distance to the original boundary
            - 'mask' : `ufl.algebra.Division`
                Conditional expression to identify the layer domain
        '''

        # Node coordinates
        z_f, x_f = nodes_coord[:2]

        # Dimensions
        Lz = self.length_z
        Lx = self.length_x

        # Conditional value
        val_condz = (z_f + Lz)**2 if typ_marker == 'damping' else 1.0
        val_condx1 = x_f**2 if typ_marker == 'damping' else 1.0
        val_condx2 = (x_f - Lx)**2 if typ_marker == 'damping' else 1.0

        # Define the conditional expressions for damping
        z_pd_sqr = fire.conditional(z_f + Lz < 0, val_condz, 0.)
        x_pd_sqr = fire.conditional(x_f < 0, val_condx1, 0.) + \
            fire.conditional(x_f - Lx > 0, val_condx2, 0.)

        if self.dimension == 3:  # 3D

            # 3D dimension
            Ly = self.length_y
            y_f = nodes_coord[-1]

            # Conditional value
            val_condy1 = y_f**2 if typ_marker == 'damping' else 1.0
            val_condy2 = (y_f - Ly)**2 if typ_marker == 'damping' else 1.0

            # Conditional expressions
            y_pd_sqr = fire.conditional(y_f < 0, val_condy1, 0.) + \
                fire.conditional(y_f - Ly > 0, val_condy2, 0.)

        if typ_marker == 'damping':

            # Reference distance to the original boundary
            norm_sqr = z_pd_sqr + x_pd_sqr
            if self.dimension == 3:  # 3D
                norm_sqr += y_pd_sqr

            ref = fire.sqrt(norm_sqr) / fire.Constant(self.pad_len)

        elif typ_marker == 'mask':

            # Mask filter for layer boundary domain
            ref = z_pd_sqr + x_pd_sqr
            if self.dimension == 3:  # 3D
                ref += y_pd_sqr

            ref = fire.conditional(ref > 0, 1.0, 0.0)

        return ref

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
        print("\nCreating Damping Profile")
        self.eta_habc = fire.Function(self.function_space, name='eta [1/s])')

        # Dimensions and node coordinates - To do: Refactor
        z_f = fire.Function(self.function_space).interpolate(self.mesh_z)
        x_f = fire.Function(self.function_space).interpolate(self.mesh_x)
        nodes_coord = [z_f, x_f]
        if self.dimension == 3:  # 3D
            y_f = fire.Function(self.function_space).interpolate(self.mesh_y)
            nodes_coord.append(y_f)

        # Damping mask
        mask = self.cond_marker_for_eta(nodes_coord, 'mask')

        print(type(mask))
        fnc_spc_eta_mask = fire.FunctionSpace(self.mesh, 'DG', 0)
        self.eta_mask = fire.Function(fnc_spc_eta_mask, name='eta_mask')
        self.eta_mask.interpolate(mask)

        # Save damping mask
        outfile = fire.VTKFile(self.path_save + "eta_mask.pvd")
        outfile.write(self.eta_mask)

        # Reference distance to the original boundary
        ref = self.cond_marker_for_eta(nodes_coord, 'damping')
        print(type(ref))

        # Compute the minimum damping ratio and the associated heuristic factor
        eta_crt, psi_min, xCR, xCR_lim, xCR_search = self.calc_damping_prop()

        # Heuristic factor for the minimum damping ratio
        self.xCR = xCR
        self.xCR_bounds = [xCR_lim, xCR_search]  # Bounds

        # Compute the coefficients for quadratic damping function
        aq, bq = self.coeff_damp_fun(psi_min)

        # Apply damping profile
        expr_damp = fire.Constant(eta_crt) * (fire.Constant(aq) * ref**2
                                              + fire.Constant(bq) * ref)
        self.eta_habc.interpolate(expr_damp)

        # Save damping profile
        outfile = fire.VTKFile(self.path_save + "eta_habc.pvd")
        outfile.write(self.eta_habc)

    def error_measures_habc(self):
        '''
        Compute the error measures at the receivers for the HABC scheme.
        Error measures as in Salas et al. (2022) Sec. 2.5.
        Obs: If you get an error during running in find_peaks means that
        the transient time of the simulation must be increased.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        pkMax = []
        errPk = []
        errIt = []
        dt = self.dt

        for i in range(self.number_of_receivers):

            # Transient response in receiver
            u_abc = self.receivers_output[:, i]
            u_ref = self.receivers_output[:, i]  # Reference signal (To Do)

            # Finding peaks in transient response
            u_pks = find_peaks(u_abc)

            if len(u_pks[0]) == 0:
                wrn_str0 = "No peak observed in the transient response. "
                wrn_str1 = "Increase the transient time of the simulation."
                UserWarning(wrn_str0 + wrn_str1)

            # M aximum peak
            p_abc = max(u_abc[u_pks[0]])
            p_ref = max(u_ref)
            pkMax.append(p_abc)

            if len(u_ref) < len(u_abc):
                u_ref = np.concatenate([u_ref, np.zeros(len(u_abc) - len(u_ref))])

            elif len(u_ref) > len(u_abc):
                u_abc = np.concatenate([u_abc, np.zeros(len(u_ref) - len(u_abc))])

            # Integral error
            errIt.append(np.trapz((u_abc - u_ref)**2, dx=dt)
                         / np.trapz(u_ref**2, dx=dt))

            # Peak error
            errPk.append(abs(p_abc / p_ref - 1))

        self.err_habc = [errIt, errPk, pkMax]

        np.savetxt(self.path_save + '_errs.txt',
                   (errIt, errPk, pkMax), delimiter='\t')


# # Old approach: 3.994, 3.907, 4.021 mean = 3.974
# # New approach: 4.908, 4.705, 4.699 mean = 4.772
# # Reference to resource usage
# tRef = comp_cost('tini')
# # Estimating computational resource usage
# comp_cost('tfin', tRef=tRef)
