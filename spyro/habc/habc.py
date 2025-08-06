import firedrake as fire
import numpy as np
from os import getcwd
import scipy.linalg as sl
import scipy.sparse as ss
import spyro.habc.lay_len as lay_len
from scipy.fft import fft
from scipy.signal import find_peaks
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.habc.hyp_lay import HyperLayer
from spyro.habc.nrbc import NRBCHabc
from spyro.plots.plots import plot_hist_receivers, \
    plot_rfft_receivers, plot_xCR_opt
import ipdb


# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class HABC_Wave(AcousticWave, HyperLayer, NRBCHabc):
    '''
    class HABC that determines absorbing layer size and parameters to be used.

    Attributes
    ----------
    abc_reference_freq : `str`
        Reference frequency for sizing the hybrid absorbing layer.
        Options: 'source' or 'boundary'
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
    case_habc : `str`
        Label for the output files that includes the layer shape
        ('REC' or 'HNI', I for the degree) and the reference frequency
        ('SOU' or 'BND'). Example: 'REC_SOU' or 'HN2_BND'
    c_habc : `firedrake function`
        Velocity model with absorbing layer
    d : `float`
        Normalized element size (lmin / pad_len)
    diam_mesh : `ufl.geometry.CellDiameter`
        Mesh cell diameters
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
    f_Nyq : `float`
        Nyquist frequency according to the time step. f_Nyq = 1 / (2 * dt)
    f_est : `float`
        Factor for the stabilizing term in Eikonal equation
    freq_ref : `float`
        Reference frequency of the wave at the minimum Eikonal point
    fundam_freq : `float`
        Fundamental frequency of the numerical model
    funct_space_eik: `firedrake function space`
        Function space for the Eikonal modeling
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
    lref : `float`
        Reference length for the size of the absorbing layer
    max_errIt : `float`
        Maximum integral error at the receivers for the HABC scheme
    max_errPK : `float`
        Maximum peak error at the receivers for the HABC scheme
    mesh: `firedrake mesh`
        Mesh used in the simulation (HABC or Infinite Model)
    mesh_original : `firedrake mesh`
        Original mesh without absorbing layer
    n_bounds : `tuple`
        Bounds for the hypershape layer degree. (n_min, n_max)
        - n_min ensures to add lmin in the domain diagonal direction
        - n_max ensures to add pad_len in the domain diagonal direction
    n_hyp : `int`
        Degree of the hyperelliptical pad layer (n >= 2). Default is 2.
        For rectangular layers, n_hyp is set to infinity
    number_of_receivers: `int`
        Number of receivers used in the simulation
    pad_len : `float`
        Size of the absorbing layer
    path_save : `string`
        Path to save data
    psi_min : `float`
        Minimum damping ratio of the absorbing layer (psi_min = xCR * d)
    receiver_locations: `list`
        List of receiver locations
    receivers_output : `array`
        Receiver waveform data in the HABC scheme
    receivers_out_fft : `array`
        Frequency response at the receivers in the HABC scheme
    receivers_reference : `array`
        Receiver waveform data in the reference model
    receivers_ref_fft : `array`
        Frequency response at the receivers in the reference model.
    tol : `float`
        Tolerance for searching nodes on the boundary
    v_rat : `float`
        Volume ratio to the volume of the original domain. v_rat = vol / v_orig
    vol : `float`
        Volume of the domain with absorbing layer
    xCR : `float`
        Heuristic factor for the minimum damping ratio
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
    comparison_plots()
        Plot the comparison between the HABC scheme and the reference model
    cond_marker_for_eta()
        Define the conditional expressions to identify the domain of
        the absorbing layer or the reference to the original boundary
        to compute the damping profile inside the absorbing layer
    create_mesh_habc()
        Create a mesh with absorbing layer based on the determined size
    critical_boundary_points()
        Determine the critical points on domain boundaries of the original
        model to size an absorbing layer using the Eikonal criterion for HABCs
    damping_layer()
        Set the damping profile within the absorbing layer
    det_reference_freq()
        Determine the reference frequency for a new layer size
    error_measures_habc()
        Compute the error measures at the receivers for the HABC scheme
    est_min_damping()
        Estimate the minimum damping ratio and the associated heuristic factor
    freq_response()
        Calculate the response in frequency domain of a time signal via FFT
    fundamental_frequency()
        Compute the fundamental frequency in Hz via modal analysis
    get_reference_signal()
        Acquire the reference signal to compare with the HABC scheme
    get_xCR_candidates()
        Get the heuristic factor candidates for the quadratic regression
    get_xCR_optimal()
        Get the optimal heuristic factor for the quadratic damping
    identify_habc_case()
        Generate an identifier for the current case study of the HABC scheme
    infinite_model()
        Create a reference model for the HABC scheme for comparative purposes
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
    solve_eigenproblem()
        Solve the eigenvalue problem to determine the fundamental frequency
    velocity_habc()
        Set the velocity model for the model with absorbing layer
    xCR_search_range()
        Determine the initial search range for the heuristic factor xCR
    '''

    def __init__(self, dictionary=None, f_est=0.06, fwi_iter=0,
                 comm=None, output_folder="output/"):
        '''
        Initialize the HABC class.

        Parameters
        ----------
        dictionary : dict, optional
            A dictionary containing the input parameters for the HABC class
        f_est : `float`, optional
            Factor for the stabilizing term in Eikonal Eq. Default is 0.06
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

        # Identifier for the current case study
        self.identify_habc_case()

        # Factor for the stabilizing term in Eikonal equation
        self.f_est = f_est

        # Nyquist frequency
        self.f_Nyq = 1.0 / (2.0 * self.dt)

        # Current iteration
        self.fwi_iter = fwi_iter

        # Path to save data
        self.path_save = getcwd() + "/" + output_folder

    def identify_habc_case(self):
        '''
        Generate an identifier for the current case study of the HABC scheme.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        self.layer_shape = self.abc_boundary_layer_shape
        if self.layer_shape == 'rectangular':
            print("\nAbsorbing Layer Shape: Rectangular")
            self.case_habc = 'REC'

        elif self.layer_shape == 'hypershape':
            lay_str = "\nAbsorbing Layer Shape: Hypershape"
            deg_str = " - Degree: " + str(self.abc_deg_layer)
            print(lay_str + deg_str)
            self.case_habc = 'HN' + str(self.abc_deg_layer)
            HyperLayer.__init__(self, n_hyp=self.abc_deg_layer,
                                dimension=self.dimension)

        else:
            aux0 = "Please use 'rectangular' or 'hypershape', "
            UserWarning(aux0 + f"{self.layer_shape} not supported.")

        self.case_habc += "_BND" \
            if self.abc_reference_freq == 'boundary' else "_SOU"

    def preamble_mesh_operations(self):
        '''
        Perform mesh operations previous to size an absorbing layer.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        print("\nCreating Mesh and Initial Velocity Model")

        # Save a copy of the original mesh
        self.mesh_original = self.mesh
        mesh_orig = fire.VTKFile(self.path_save + "preamble/mesh_orig.pvd")
        mesh_orig.write(self.mesh_original)

        # Velocity profile model
        self.c = fire.Function(self.function_space, name='c_orig [km/s])')
        self.c.interpolate(self.initial_velocity_model)

        # Save initial velocity model
        vel_c = fire.VTKFile(self.path_save + "preamble/c_vel.pvd")
        vel_c.write(self.c)

        # Mesh properties for Eikonal
        self.properties_eik_mesh(p_usu=self.abc_deg_eikonal)

        # Generating boundary data from the original domain mesh
        self.boundary_data()

    def properties_eik_mesh(self, p_usu=None):
        '''
        Set the properties for the mesh used to solve the Eikonal equation.

        Parameters
        ----------
        p_usu : `int`, optional
            Finite element order for the Eikonal equation. Default is None

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
        lmin = self.lmin
        lref = self.lref
        self.F_L = (lmin / lref) * np.ceil(lref * self.F_L / lmin)

        # New size of the absorving layer
        self.pad_len = self.F_L * lref

        print("\nModifying Layer Size Based on the Element Size")
        print("Modified Parameter Size FL: {:.4f}".format(self.F_L))
        print("Modified Layer Size (km): {:.4f}".format(self.pad_len))
        print("Elements ({:.3f} km) in Layer: {}".format(
            self.lmin, int(self.pad_len / self.lmin)))

    def critical_boundary_points(self, Eikonal):
        '''
        Determine the critical points on domain boundaries of the original
        model to size an absorbing layer using the Eikonal criterion for HABCs.
        See Salas et al (2022) for details.

        Parameters
        ----------
        Eikonal : `eikonal`
            An object representing the Eikonal solver

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

        # Reference length for the size of the absorbing layer
        self.lref = self.eik_bnd[0][4]

        # Critical point coordinates as receivers
        pcrit = [bnd[0] for bnd in self.eik_bnd]
        self.receiver_locations = pcrit + self.receiver_locations
        self.number_of_receivers = len(self.receiver_locations)

    def freq_response(self, signal, fpad=4, get_max_freq=False):
        '''
        Calculate the response in frequency domain of a time signal via FFT.

        Parameters
        ----------
        signal : `array`
            Signal data
        fpad : `int`, optional
            Padding factor for FFT. Default is 4
        get_max_freq : `bool`, optional
            If True, return only the maximum frequency of the spectrum.
            Default is False

        Returns
        -------
        yf : `array`
            Normalized frequency spectrum with respect to the maximum magnitude
        max_freq : `float`, optional
            Maximum frequency of the spectrum
        '''

        # Check if the signal is empty
        if signal.size == 0:

            err = "Input signal is empty. Cannot compute frequency response."
            raise ValueError(err)

        # Zero padding for increasing smoothing in FFT
        yt = np.concatenate([np.zeros(fpad * len(signal)), signal])

        # Number of sample points
        N_samples = len(yt)

        # Determine the number of samples of the spectrum
        pfft = N_samples // 2 + N_samples % 2

        # Calculate the response in frequency domain of the signal (FFT)
        yf = np.abs(fft(yt)[0:pfft])
        del yt

        # Frequency vector
        xf = np.linspace(0.0, self.f_Nyq, pfft)

        # Get the maximum frequency of the spectrum
        max_freq = xf[yf.argmax()]

        if get_max_freq:

            # Return the maximum frequency only
            return max_freq
        else:

            # Normalized frequency spectrum
            yf *= (1 / yf.max())

            # Return the normalized spectrum
            return yf

    def det_reference_freq(self, fpad=4):
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

        abc_reference_freq = self.abc_reference_freq \
            if hasattr(self, 'receivers_reference') else 'source'

        if self.abc_reference_freq == 'source':  # Initial guess
            # Theorical central Ricker source frequency
            self.freq_ref = self.frequency

        elif abc_reference_freq == 'boundary':

            # Transient response at the minimum Eikonal point
            histPcrit = self.receivers_reference[:, 0]

            # Get the minimum frequency excited at the critical point
            self.freq_ref = self.freq_response(histPcrit, get_max_freq=True)

        print("Reference Frequency (Hz): {:.5f}".format(self.freq_ref))

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
            print("Area Ratio: {:5.3f}".format(self.a_rat))

        if self.dimension == 3:  # 3D
            self.vol = self.Lx_habc * self.Lz_habc * self.Ly_habc
            self.v_rat = self.vol / (self.length_x * self.length_z
                                     * self.length_y)
            self.f_Vh = 8
            print("Volume Ratio: {:5.3f}".format(self.v_rat))

    def size_habc_criterion(self, fpad=4, n_root=1, layer_based_on_mesh=False):
        '''
        Determine the size of the absorbing layer using the Eikonal
        criterion for HABCs. See Salas et al (2022) for details.

        Parameters
        ----------
        fpad : `int`, optional
            Padding factor for FFT. Default is 4
        n_root : `int`, optional
            n-th Root selected as the size of the absorbing layer. Default is 1
        layer_based_on_mesh : `bool`, optional
            Adjust the layer size based on the element size. Default is False

        Returns
        -------
        None
        '''

        # Determining the reference frequency
        self.det_reference_freq(fpad=fpad)

        # Computing layer sizes
        self.F_L, self.pad_len, self.a_par = \
            lay_len.calc_size_lay(self, n_root=n_root)

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

    def create_mesh_habc(self, inf_model=False, spln=True, fmesh=1.):
        '''
        Create a mesh with absorbing layer based on the determined size.

        Parameters
        ----------
        inf_model : `bool`, optional
            If True, build a rectangular layer for the infinite or reference
            model (Model with "infinite" dimensions). Default is False
        spln : `bool`
            Flag to indicate whether to use splines (True) or lines (False)
            in hypershape layer generation. Default is True
        fmesh : `float`
            Mesh size factor for the hyperelliptical layer with respect to mesh
            size of the original domain. Default is 1.0.

        Returns
        -------
        None
        '''

        if inf_model:
            print("\nGenerating Mesh for Infinite Model")
            layer_shape = 'rectangular'

        else:
            print("\nGenerating Mesh with Absorbing Layer")
            layer_shape = self.layer_shape

        # New mesh with layer
        if layer_shape == 'rectangular':

            # New geometry with layer
            Lz = self.Lz_habc
            Lx = self.Lx_habc

            # Number of elements
            n_pad = self.pad_len / self.lmin  # Elements in the layer
            nz = int(self.length_z / self.lmin) + int(n_pad)
            nx = int(self.length_x / self.lmin) + int(2 * n_pad)
            nx = nx + nx % 2

            if self.dimension == 2:  # 2D
                mesh_habc = fire.RectangleMesh(nz, nx, Lz, Lx,
                                               comm=self.comm.comm)

            if self.dimension == 3:  # 3D
                Ly = self.Ly_habc
                ny = int(self.length_y / self.lmin) + int(2 * n_pad)
                ny = ny + ny % 2
                mesh_habc = fire.BoxMesh(nz, nx, ny, Lz, Lx, Ly,
                                         comm=self.comm.comm)
                mesh_habc.coordinates.dat.data_with_halos[:, 2] -= self.pad_len

            # Adjusting coordinates
            mesh_habc.coordinates.dat.data_with_halos[:, 0] *= -1.0
            mesh_habc.coordinates.dat.data_with_halos[:, 1] -= self.pad_len

            print("Extended Rectangular Mesh Generated Successfully")

        elif layer_shape == 'hypershape':

            # Creating the hyperellipse layer mesh
            hyp_mesh = self.create_hyp_trunc_mesh2D(spln=spln, fmesh=fmesh)

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

        if inf_model:
            file_name = "preamble/mesh_inf.pvd"
        else:
            file_name = self.case_habc + "/mesh_habc.pvd"

        # Save new mesh
        outfile = fire.VTKFile(self.path_save + file_name)
        outfile.write(self.mesh)

    def velocity_habc(self, inf_model=False):
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
        dx = 0.05 km (2D)
        Pts approach: 0.699 0.599 0.717 mean = 0.672
        Lst approach: 0.914 0.769 0.830 mean = 0.847
        New approach: 1.602 1.495 1.588 mean = 1.562
        Old approach: 1.982 2.124 1.961 mean = 2.022

        dx = 0.02 km
        Pts approach: 2.290 2.844 2.133 mean = 2.422
        Lst approach: 2.784 2.726 3.085 mean = 2.865
        New approach: 5.276 5.214 6.275 mean = 5.588
        Old approach: 12.232 12.372 12.078 = 12.227

        dx = 0.05 km (3D)
        Pts approach: 33.234 31.697 31.598 = 32.176
        Lst approach: 60.101 60.919 50.918 = 57.313
        '''

        print("\nUpdating Velocity Profile")

        # Initialize velocity field and assigning the original velocity model
        element_c = self.initial_velocity_model.ufl_element().family()
        p = self.initial_velocity_model.function_space().ufl_element().degree()
        V = fire.FunctionSpace(self.mesh, element_c, p)
        self.c = fire.Function(V).interpolate(self.initial_velocity_model,
                                              allow_missing_dofs=True)

        # Extending velocity model within the absorbing layer
        print("Extending Profile Inside Layer")

        # Vectorial space for auxiliar field of clipped coordinates
        W = fire.VectorFunctionSpace(self.mesh, element_c, p)

        if self.dimension == 2:  # 2D
            z, x = fire.SpatialCoordinate(self.mesh)
            coords = fire.as_vector([z, x])

        if self.dimension == 3:  # 3D
            z, x, y = fire.SpatialCoordinate(self.mesh)
            coords = fire.as_vector([z, x, y])

        # Clipping coordinates
        w_aux = fire.Function(W).interpolate(coords)
        w_arr = w_aux.dat.data_with_halos[:]
        w_arr[:, 0] = np.clip(w_arr[:, 0], -self.length_z, 0.)
        w_arr[:, 1] = np.clip(w_arr[:, 1], 0., self.length_x)
        if self.dimension == 3:  # 3D
            w_arr[:, 2] = np.clip(w_arr[:, 2], 0., self.length_y)

        # Dimensions
        Lz = self.length_z
        Lx = self.length_x

        # Mask for the layer domain
        z_pd = fire.conditional(z + Lz < 0., 1., 0.)
        x_pd = fire.conditional(x < 0., 1., 0.) + \
            fire.conditional(x - Lx > 0., 1., 0.)
        mask = z_pd + x_pd

        if self.dimension == 3:  # 3D

            # 3D dimension
            Ly = self.length_y

            # Conditional expressions for the mask
            y_pd = fire.conditional(y < 0., 1., 0.) + \
                fire.conditional(y - Ly > 0., 1., 0.)
            mask += y_pd

        # Final value for the mask
        mask = fire.conditional(mask > 0., 1., 0.)
        layer_mask = fire.Function(V, name='layer_mask')
        layer_mask.interpolate(mask)

        # Points to extend the velocity model
        pad_field = fire.Function(W).interpolate(w_aux * layer_mask)
        del w_aux

        pad_pts = pad_field.dat.data_with_halos[:]
        if self.dimension == 2:  # 2D
            ind_pts = np.where(~((abs(pad_pts[:, 0]) == 0.)
                                 & (abs(pad_pts[:, 1]) == 0.)))

        if self.dimension == 3:  # 3D
            ind_pts = np.where(~((abs(pad_pts[:, 0]) == 0.)
                                 & (abs(pad_pts[:, 1]) == 0.)
                                 & (abs(pad_pts[:, 2]) == 0.)))
        pts_to_extend = pad_pts[ind_pts]

        # # Set the velocity of the nearest point on the original boundary
        # vel_to_extend = self.initial_velocity_model.at(pts_to_extend,
        #                                                dont_raise=True)
        # del pts_to_extend

        # # Velocity profile inside the layer
        # pad_field.dat.data_with_halos[ind_pts, 0] = vel_to_extend
        # del vel_to_extend, ind_pts

        # Possible new apprach
        # Set the velocity of the nearest point on the original boundary
        pts_mesh = fire.VertexOnlyMesh(
            self.mesh_original, pts_to_extend,
            missing_points_behaviour='warn', redundant=True)
        del pts_to_extend
        V0 = fire.FunctionSpace(pts_mesh, "DG", 0)
        c_int = fire.Interpolator(self.initial_velocity_model, V0,
                                  allow_missing_dofs=True)
        c_pts = fire.assemble(c_int.interpolate())
        del c_int
        V1 = fire.FunctionSpace(pts_mesh.input_ordering, "DG", 0)
        del pts_mesh
        vel_to_extend = fire.Function(V1)
        vel_to_extend.interpolate(c_pts)
        del c_pts
        # Velocity profile inside the layer
        pad_field.dat.data_with_halos[
            ind_pts, 0] = vel_to_extend.dat.data_with_halos[:]
        del vel_to_extend, ind_pts

        # Interpolating the velocity model in the layer
        self.c.interpolate(pad_field.sub(0) * layer_mask + (
            1 - layer_mask) * self.c, allow_missing_dofs=True)
        del layer_mask, pad_field

        # Interpolating in the space function of the problem
        self.c = fire.Function(
            self.function_space, name='c [km/s])').interpolate(self.c)

        # Save new velocity model
        if inf_model:
            file_name = "preamble/c_inf.pvd"
        else:
            file_name = self.case_habc + "/c_habc.pvd"

        outfile = fire.VTKFile(self.path_save + file_name)
        outfile.write(self.c)

    def solve_eigenproblem(self, Asp, Msp, method, k=2,
                           shift=0., inv_operator=False):
        '''
        Solve the eigenvalue problem to determine the fundamental frequency.

        Parameters
        ----------
        Asp : `scipy sparse matrix`
            Sparse matrix representing the stiffness matrix
        Msp : `scipy sparse matrix`
            Sparse matrix representing the mass matrix
        method : `str`
            Method to use for solving the eigenvalue problem.
            Options: 'ARNOLDI', 'LANCZOS', 'LOBPCG' 'KRYLOVSCH_CH',
            'KRYLOVSCH_CG', 'KRYLOVSCH_GH' or 'KRYLOVSCH_GG'
        k : `int`, optional
            Number of eigenvalues to compute. Default is 2
        shift: `float`, optional
            Value to stabilize the Neumann BC null space. Default is 0
        inv_operator : `bool`, optional
            Option to use an inverse operator for improving convergence.
            Default is False

        Returns
        -------
        Lsp : `array`
            Array containing the computed eigenvalues
        '''

        if shift > 0.:
            Asp += shift * Msp

        valid_methods = ['ARNOLDI', 'LANCZOS', 'LOBPCG', 'KRYLOVSCH_CH',
                         'KRYLOVSCH_CG', 'KRYLOVSCH_GH', 'KRYLOVSCH_GG']
        if method not in valid_methods:
            met_str = ", ".join(valid_methods)
            met_str = met_str.rsplit(', ', 1)
            met_str = f"Please use {met_str[0]} or {met_str[1]}. "
            UserWarning(met_str + f"{method} not supported.")

        if method == 'ARNOLDI' or method == 'LANCZOS':
            # Inverse operator for improving convergence
            M_ilu = ss.linalg.spilu(Msp) if inv_operator else None
            Minv = M_ilu.solve if inv_operator else None
            A_ilu = ss.linalg.spilu(Asp) if inv_operator else None
            OPinv = A_ilu.solve if inv_operator else None

        if method == 'ARNOLDI':
            # Solve the eigenproblem using ARNOLDI (ARPACK)
            Lsp = ss.linalg.eigs(Asp, k=k, M=Msp, sigma=0.0, Minv=Minv,
                                 OPinv=OPinv, return_eigenvectors=False)

        if method == 'LANCZOS':
            # Solve the eigenproblem using LANCZOS (ARPACK)
            Lsp = ss.linalg.eigsh(Asp, k=k, M=Msp, sigma=0.0, Minv=Minv,
                                  OPinv=OPinv, return_eigenvectors=False)

        if method == 'LOBPCG':
            # Initialize random vectors for LOBPCG
            X = sl.orth(np.random.rand(Msp.shape[0], k))

            # Solve the eigenproblem using LOBPCG
            it_mod = 2500
            it_ext = 2
            for it in range(it_ext):
                Lsp, X, resid = ss.linalg.lobpcg(Asp, X, B=Msp, tol=5e-4,
                                                 maxiter=it_mod, largest=False,
                                                 retResidualNormsHistory=True)

                it_mod //= 2
                rmin = np.array(resid)[:, 1].min()
                if rmin < 5e-4 or it_mod < 20:
                    del X, resid
                    break

        if method[:-3] == 'KRYLOVSCH':

            if method[-2] == "C":
                ksp_type = "cg"
            elif method[-2] == "G":
                ksp_type = "gmres"

            if method[-1] == "H":
                pc_type = "hypre"
            elif method[-1] == "G":
                pc_type = "gamg"

            opts = {
                "eps_type": "krylovschur",       # Robust, widely used eigensolver
                # "eps_type": "lobpcg",          # Iterative
                "eps_tol": 1e-6,                 # Tight tolerance for accuracy
                "eps_max_it": 200,               # Reasonable iteration cap
                "st_type": "sinvert",            # Useful for interior eigenvalues
                "st_shift": 1e-6,                # Stabilizes Neumann BC null space
                "eps_smallest_magnitude": None,  # Smallest eigenvalues in magnitude
                "eps_monitor": "ascii",          # Print convergence info
                "ksp_type": ksp_type,            # Options for large problems
                "pc_type": pc_type               # Options for large problems
            }

            eigenproblem = fire.LinearEigenproblem(Asp, M=Msp)
            eigensolver = fire.LinearEigensolver(eigenproblem, n_evals=k,
                                                 solver_parameters=opts)
            nconv = eigensolver.solve()
            lam = eigensolver.eigenvalue(1)
            Lsp = np.asarray([eigensolver.eigenvalue(mod) for mod in range(k)])

        if shift > 0.:
            Lsp -= shift

        return Lsp

    def fundamental_frequency(self, method=None, monitor=False):
        '''
        Compute the fundamental frequency in Hz via modal analysis
        considering the numerical model with Neumann BCs.

        Parameters
        ----------
        method : `str`, optional
            Method to use for solving the eigenvalue problem.
            Options: 'ARNOLDI', 'LANCZOS', 'LOBPCG' 'KRYLOVSCH_CH',
            'KRYLOVSCH_CG', 'KRYLOVSCH_GH' or 'KRYLOVSCH_GG'
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

        if method is None:

            if self.dimension == 2:  # 2D
                method = 'ARNOLDI'

            if self.dimension == 3:  # 3D
                method = 'LOBPCG'

        # Function space for the problem
        V = self.function_space
        u, v = fire.TrialFunction(V), fire.TestFunction(V)
        quad_rule = self.quadrature_rule
        dx = fire.dx(scheme=quad_rule)

        # Bilinear forms
        c = self.c
        a = c * c * fire.inner(fire.grad(u), fire.grad(v)) * dx
        A = fire.assemble(a)
        m = fire.inner(u, v) * dx
        M = fire.assemble(m)

        print("\nSolving Eigenvalue Problem")

        if method[:-3] == 'KRYLOVSCH':
            # Modal solver
            Lsp = self.solve_eigenproblem(A, M, method, shift=1e-8)

        else:
            # Assembling the matrices for Scipy solvers
            m_ptr, m_ind, m_val = M.petscmat.getValuesCSR()
            Msp = ss.csr_matrix((m_val, m_ind, m_ptr), M.petscmat.size)
            a_ptr, a_ind, a_val = A.petscmat.getValuesCSR()
            Asp = ss.csr_matrix((a_val, a_ind, a_ptr), A.petscmat.size)

            # Modal solver
            Lsp = self.solve_eigenproblem(Asp, Msp, method, shift=1e-8)

        if monitor:
            for n_eig, eigval in enumerate(np.unique(Lsp)):
                f_eig = np.sqrt(abs(eigval)) / (2 * np.pi)
                print(f"Frequency {n_eig} (Hz): {f_eig:.5f}")

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
            Minimum damping ratio of the absorbing layer (psi_min = xCR * d)
        xCR_est : `float`
            Estimated heuristic factor for the minimum damping ratio
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
        data_reg = [x_reg, y_reg]
        xCR_lim = [xCR_inf, xCR_sup, xCR_ini]
        psi_min, xCR_est, CRmin = self.regression_CRmin(data_reg, xCR_lim, kCRp)

        xCR_search = self.xCR_search_range(CRmin, kCRp, p[:2], xCR_lim[:2])

        return psi_min, xCR_est, xCR_lim[:2], xCR_search

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
            Minimum damping ratio of the absorbing layer (psi_min = xCR * d)
        xCR_est : `float`
            Estimated heuristic factor for the minimum damping ratio
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
        lim_str = "Initial Range for xCR Factor. Min:{:.3f} - Max:{:.3f}"
        print(lim_str.format(xCR_min, xCR_max))

        return eta_crt, psi_min, xCR_est, xCR_lim, xCR_search

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
        ref = z_pd_sqr + x_pd_sqr

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
            ref += y_pd_sqr

        if typ_marker == 'damping':
            # Reference distance to the original boundary
            ref = fire.sqrt(ref) / fire.Constant(self.pad_len)

        elif typ_marker == 'mask':
            # Mask filter for layer boundary domain
            ref = fire.conditional(ref > 0, 1.0, 0.0)

        return ref

    def damping_layer(self, xCR_usu=None):
        '''
        Set the damping profile within the absorbing layer. Minimum damping
        ratio is computed as psi_min = xCR * d where xCR is the heuristic
        factor for the minimum damping ratio and d is the normalized element
        size (lmin / pad_len).

        Parameters
        ----------
        xCR_usu : `float`, optional
            User-defined heuristic factor for the minimum damping ratio.
            Default is None, which defines an estimated value

        Returns
        -------
        None
        '''

        # Estimating fundamental frequency
        self.fundamental_frequency(monitor=True)

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
        fnc_spc_eta_mask = fire.FunctionSpace(self.mesh, 'DG', 0)
        self.eta_mask = fire.Function(fnc_spc_eta_mask, name='eta_mask')
        self.eta_mask.interpolate(mask)

        # Save damping mask
        path_damp = self.path_save + self.case_habc
        outfile = fire.VTKFile(path_damp + "/eta_mask.pvd")
        outfile.write(self.eta_mask)

        # Reference distance to the original boundary
        ref = self.cond_marker_for_eta(nodes_coord, 'damping')

        # Compute the minimum damping ratio and the associated heuristic factor
        eta_crt, psi_min, \
            xCR_est, xCR_lim, xCR_search = self.calc_damping_prop()

        # Heuristic factor for the minimum damping ratio
        self.xCR_bounds = [xCR_lim, xCR_search]  # Bounds

        if xCR_usu is not None:
            self.xCR = np.clip(xCR_usu, xCR_lim[0], xCR_lim[1])
            self.psi_min = self.xCR * self.d
            xcr_str = "Using User-Defined Heuristic Factor xCR: {:.3f}"
            print(xcr_str.format(self.xCR))
        else:
            self.xCR = xCR_est
            self.psi_min = psi_min

        # Compute the coefficients for quadratic damping function
        aq, bq = self.coeff_damp_fun(self.psi_min)

        # Apply damping profile
        expr_damp = fire.Constant(eta_crt) * (fire.Constant(aq) * ref**2
                                              + fire.Constant(bq) * ref)
        self.eta_habc.interpolate(expr_damp)

        # Save damping profile
        outfile = fire.VTKFile(path_damp + "/eta_habc.pvd")
        outfile.write(self.eta_habc)

    def infinite_model(self):
        '''
        Create a reference model for the HABC scheme for comparative purposes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        print("\nBuilding Infinite Domain Model")

        # Size of the domain extension
        max_c = self.initial_velocity_model.dat.data_with_halos.max()
        add_dom = max_c * self.final_time / 2.

        # Distance already travelled by the wave
        if hasattr(self, 'eik_bnd'):
            # If Eikonal analysis was performed
            eikmin = self.eik_bnd[0][2]
            add_dom -= max_c * eikmin / 2.
        else:
            # If Eikonal analysis was not performed
            dist_to_bnd = np.inf
            for nsou in range(self.number_of_sources):
                psou_z = self.source_locations[nsou][0]
                psou_x = self.source_locations[nsou][1]
                delta_z = abs(psou_z - self.length_z)
                delta_x = min(abs(psou_x), abs(psou_x - self.length_x))

                if self.dimension == 2:  # 2D
                    dist_to_bnd = min(dist_to_bnd, delta_z, delta_x)

                if self.dimension == 3:  # 3D
                    psou_y = self.source_locations[nsou][2]
                    delta_y = min(abs(psou_y), abs(psou_y - self.length_y))
                    dist_to_bnd = min(dist_to_bnd, delta_z, delta_x, delta_y)

            add_dom -= dist_to_bnd

        pad_len = self.lmin * np.ceil(add_dom / self.lmin)
        self.pad_len = pad_len
        inf_str = "Infinite Domain Extension (km): {:.4f}"
        print(inf_str.format(self.pad_len))

        # New dimensions
        self.Lx_habc = self.length_x + 2 * self.pad_len
        self.Lz_habc = self.length_z + self.pad_len

        if self.dimension == 3:  # 3D
            self.Ly_habc = self.length_y + 2 * self.pad_len

        # Creating mesh for infinite domain
        self.create_mesh_habc(inf_model=True)

        # Updating velocity model
        self.velocity_habc(inf_model=True)

        # Setting no damping
        self.cosHig = fire.Constant(0.)
        self.eta_mask = fire.Constant(0.)
        self.eta_habc = fire.Constant(0.)

        # Solving the forward problem
        print("\nSolving Infinite Model")
        self.forward_solve()

        # Saving reference signal
        print("Saving Reference Output")
        pth_str = self.path_save + "preamble/"
        self.receivers_reference = self.receivers_output.copy()
        np.save(pth_str + "habc_ref.npy", self.receivers_reference)

        # Computing and saving FFT of the reference signal at receivers
        self.receivers_ref_fft = []
        for rec in range(self.number_of_receivers):
            signal = self.receivers_reference[:, rec]
            yf = self.freq_response(signal)
            self.receivers_ref_fft.append(yf)
        np.save(pth_str + "habc_fft.npy", self.receivers_ref_fft)

        # Deleting variables to be computed for the HABC scheme
        del self.pad_len, self.Lx_habc, self.Lz_habc
        del self.cosHig, self.eta_mask, self.eta_habc
        if self.dimension == 3:
            del self.Ly_habc

    def get_reference_signal(self, foldername="preamble/"):
        '''
        Acquire the reference signal to compare with the HABC scheme.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        print("\nLoading Reference Signal from Infinite Model")

        # Path to the reference data folder
        pth_str = self.path_save + foldername

        # Time domain signal
        self.receivers_reference = np.load(pth_str + "habc_ref.npy")

        # Frequency domain signal
        self.receivers_ref_fft = np.load(pth_str + "habc_fft.npy").T

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

        print("\nComputing Error Measures")

        pkMax = []
        errPk = []
        errIt = []

        for i in range(self.number_of_receivers):

            # Transient response in receiver
            u_abc = self.receivers_output[:, i]
            u_ref = self.receivers_reference[:, i]

            # Finding peaks in transient response
            u_pks = find_peaks(u_abc)
            if u_pks[0].size == 0:
                wrn_str0 = "No peak observed in the transient response. "
                wrn_str1 = "Increase the transient time of the simulation."
                UserWarning(wrn_str0 + wrn_str1)

            # Maximum peak value
            p_abc = max(abs(u_abc))
            p_ref = max(abs(u_ref))
            pkMax.append(p_ref)

            # Completing with zeros if the length of arrays is different
            delta_len = abs(len(u_abc) - len(u_ref))
            if len(u_ref) < len(u_abc):
                u_ref = np.concatenate([u_ref, np.zeros(delta_len)])
            elif len(u_ref) > len(u_abc):
                u_abc = np.concatenate([u_abc, np.zeros(delta_len)])

            # Integral error
            errIt.append(np.trapezoid((u_abc - u_ref)**2, dx=self.dt)
                         / np.trapezoid(u_ref**2, dx=self.dt))

            # Peak error
            errPk.append(abs(p_abc / p_ref - 1))

        # Final value of the dissipated energy in the HABC scheme
        final_energy = fire.assemble(self.acoustic_energy)
        self.err_habc = [errIt, errPk, pkMax, final_energy]
        self.max_errIt = max(errIt)
        self.max_errPK = max(errPk)
        print("Maximum Integral Error: {:.2%}".format(self.max_errIt))
        print("Maximum Peak Error: {:.2%}".format(self.max_errPK))
        print("Acoustic Energy: {:.2e}".format(final_energy))

        # Save error measures
        err_str = self.path_save + self.case_habc + "/habc_errs.txt"
        np.savetxt(err_str, (errIt, errPk, pkMax), delimiter='\t')

        # Append the energy value at the end
        with open(err_str, 'a') as f:
            np.savetxt(f, np.array([final_energy]), delimiter='\t')

    def comparison_plots(self, regression_xCR=False, data_regr_xCR=None):
        '''
        Plot the comparison between the HABC scheme and the reference model.

        Parameters
        ----------
        regression_xCR : `bool`, optional
            If True, Plot the regression for the error measure vs xCR
            Default is False.
        data_regr_xCR: `list`
            Data for the regression of the parameter xCR.
            Structure: [xCR, max_errIt, max_errPK, crit_opt]
            - xCR: Values of xCR used in the regression.
              The last value IS the optimal xCR
            - max_errIt: Values of the maximum integral error.
              The last value corresponds to the optimal xCR
            - max_errPK: Values of the maximum peak error.
              The last value corresponds to the optimal xCR
            - crit_opt : Criterion for the optimal heuristic factor.
              * 'error_difference' : Difference between integral and peak errors
              * 'error_integral' : Minimum integral error

        Returns
        -------
        None
        '''

        # Time domain comparison
        plot_hist_receivers(self)

        # Compute FFT for output signal at receivers
        self.receivers_out_fft = []
        for rec in range(self.number_of_receivers):
            signal = self.receivers_output[:, rec]
            yf = self.freq_response(signal)
            self.receivers_out_fft.append(yf)
        self.receivers_out_fft = np.asarray(self.receivers_out_fft).T

        # Frequency domain comparison
        plot_rfft_receivers(self)

        # Plot the error measures
        if regression_xCR:
            plot_xCR_opt(self, data_regr_xCR)

    def get_xCR_candidates(self, n_pts=3):
        '''
        Get the heuristic factor candidates for the quadratic regression.

        Parameters
        ----------
        n_pts : `int`, optional
            Number of candidates for the heuristic factor xCR.
            Default is 3. Must be an odd number

        Returns
        -------
        xCR_cand : `list`
            Candidates for the heuristic factor xCR based on the
            current xCR and its bounds. The candidates are sorted
            in ascending order and current xCR is not included
        '''

        # Setting odd number of points for regression
        n_pts = max(3, n_pts + 1 if n_pts % 2 == 0 else n_pts)

        # Limits for the heuristic factor
        xCR_inf, xCR_sup = self.xCR_bounds[0]

        # Estimated intial value
        xCR = self.xCR

        # Determining the xCR candidates for regression
        if xCR in self.xCR_bounds[0]:
            xCR_cand = list(np.linspace(xCR_inf, xCR_sup, n_pts))
            xCR_cand.remove(xCR)
        else:
            xCR_cand = list(np.linspace(xCR_inf, xCR_sup, n_pts-1))

        format_xCR = ', '.join(['{:.3f}'.format(x) for x in xCR_cand])
        print("Candidates for Heuristic Factor xCR: [{}]".format(format_xCR))

        return xCR_cand

    def get_xCR_optimal(self, dat_reg_xCR, crit_opt='error_difference'):
        '''
        Get the optimal heuristic factor for the quadratic damping.

        Parameters
        ----------
        dat_reg_xCR : `list`
            Data for the regression of the parameter xCR.
            Structure: [xCR, max_errIt, max_errPK]
        crit_opt : `string`, optional
            Criterion for the optimal heuristic factor
            Default is 'error_difference'.
            - 'error_difference' : Difference between integral and peak errors
            - 'error_integral' : Minimum integral error

        Returns
        -------
        xCR_opt : `float`, optional
            Optimal heuristic factor for the quadratic damping
        '''

        # Data for regression
        xCR = dat_reg_xCR[0]
        max_errIt = dat_reg_xCR[1]
        max_errPK = dat_reg_xCR[2]

        if crit_opt == 'error_difference':
            y_err = [eI - eP for eI, eP in zip(max_errIt, max_errPK)]

        elif crit_opt == 'error_integral':
            y_err = max_errIt

        # Limits for the heuristic factor
        xCR_inf, xCR_sup = self.xCR_bounds[0]

        # Coefficients for the quadratic equation
        eq_xCR = np.polyfit(xCR, y_err, 2)

        if crit_opt == 'error_difference':
            # Roots of the quadratic equation
            roots = np.roots(eq_xCR)
            valid_roots = [np.clip(rth, xCR_inf, xCR_sup)
                           for rth in roots if isinstance(rth, float)]

            if valid_roots:
                # Real root that provides the absolute minimum error
                min_err = [abs(np.polyval(eq_xCR, rth)) for rth in valid_roots]
                xCR_opt = valid_roots[np.argmin(min_err)]
            else:
                # Vertex when there are no real roots
                vtx = - eq_xCR[1] / (2 * eq_xCR[0])
                xCR_opt = np.clip(vtx, xCR_inf, xCR_sup)

        elif crit_opt == 'error_integral':

            # Vertex of the quadratic equation
            vtx = - eq_xCR[1] / (2 * eq_xCR[0])
            xCR_opt = np.clip(vtx, xCR_inf, xCR_sup)

        print("Optimal Heuristic Factor xCR: {:.3f}".format(xCR_opt))

        return xCR_opt
