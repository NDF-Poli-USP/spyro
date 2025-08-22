import firedrake as fire
import numpy as np
import spyro.habc.eik as eik
from os import getcwd
from scipy.signal import find_peaks
from sympy import divisors
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.meshing.meshing_habc import HABC_Mesh
from spyro.habc.hyp_lay import HyperLayer
from spyro.habc.rec_lay import RectangLayer
from spyro.habc.error_measure import HABC_Error
from spyro.habc.nrbc import NRBCHabc
from spyro.habc.lay_len import calc_size_lay
from spyro.solvers.modal.modal_sol import Modal_Solver
from spyro.plots.plots_habc import plot_function_layer_size, \
    plot_hist_receivers, plot_rfft_receivers, plot_xCR_opt
from spyro.utils.error_management import value_parameter_error
from spyro.utils.freq_tools import freq_response

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class HABC_Wave(AcousticWave, HABC_Mesh, RectangLayer,
                HyperLayer, NRBCHabc, HABC_Error):
    '''
    class HABC that determines absorbing layer size and parameters to be used.

    Attributes
    ----------
    * abc_reference_freq : `str`
        Reference frequency for sizing the hybrid absorbing layer.
        Options: 'source' or 'boundary'
    * a_par : `float`
        Adimensional propagation speed parameter (a = z / f).
        "z" parameter is the inverse of the minimum Eikonal (1 / phi_min)
    * case_habc : `str`
        Label for the output files that includes the layer shape
        ('REC' or 'HNI', I for the degree) and the reference frequency
        ('SOU' or 'BND'). Example: 'REC_SOU' or 'HN2_BND'
    * d : `float`
        Normalized element size (lmin / pad_len)
    * eik_bnd : `list`
        Properties on boundaries according to minimum values of Eikonal
        Structure sublist: [pt_cr, c_bnd, eikmin, z_par, lref, sou_cr]
        - pt_cr : Critical point coordinates
        - c_bnd : Propagation speed at critical point
        - eikmin : Eikonal value in seconds
        - z_par : Inverse of minimum Eikonal (Equivalent to c_bound / lref)
        - lref : Distance to the closest source from critical point
        - sou_cr : Critical source coordinates
    * ele_pad : `int`
        Number of elements in the layer of edge length 'lmin'
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
    * F_L : `float`
        Size parameter of the absorbing layer
    * FLpos : `list`
        Possible size parameters for the absorbing layer without rounding
    * f_Nyq : `float`
        Nyquist frequency according to the time step. f_Nyq = 1 / (2 * dt)
    * freq_ref : `float`
        Reference frequency of the wave at the minimum Eikonal point
    fundam_freq : `float`
        Fundamental frequency of the numerical model
    * fwi_iter : `int`
        The iteration number for the Full Waveform Inversion (FWI) algorithm
    * Lz_habc : `float`
        Length of the domain in the z-direction with absorbing layer
    * Lx_habc : `float`
        Length of the domain in the x-direction with absorbing layer
    * Ly_habc : `float`
        Length of the domain in the y-direction with absorbing (3D models)
    * layer_shape : `string`
        Shape type of pad layer. Options: 'rectangular' or 'hypershape'
    * lref : `float`
        Reference length for the size of the absorbing layer
    max_errIt : `float`
        Maximum integral error at the receivers for the HABC scheme
    max_errPK : `float`
        Maximum peak error at the receivers for the HABC scheme
    * mesh: `firedrake mesh`
        Mesh used in the simulation (HABC or Infinite Model)
    * number_of_receivers: `int`
        Number of receivers used in the simulation
    * pad_len : `float`
        Size of the absorbing layer
    * path_case_habc : `string`
        Path to save data for the current case study
    * path_save : `string`
        Path to save data
    psi_min : `float`
        Minimum damping ratio of the absorbing layer (psi_min = xCR * d)
    * receiver_locations: `list`
        List of receiver locations
    receivers_output : `array`
        Receiver waveform data in the HABC scheme
    receivers_out_fft : `array`
        Frequency response at the receivers in the HABC scheme
    xCR : `float`
        Heuristic factor for the minimum damping ratio
    xCR_bounds: `list`
        Bounds for the heuristic factor. [xCR_lim, xCR_search]
        Structure: [[xCR_inf, xCR_sup], [xCR_min, xCR_max]]
        - xCR_lim : Limits for the heuristic factor.
        - xCR_search : Initial search range for the heuristic factor

    Methods
    -------
    calc_damping_prop()
        Compute the damping properties for the absorbing layer
    * check_timestep()
        Check if the timestep size is appropriate for the transient response
    coeff_damp_fun()
        Compute the coefficients for quadratic damping function
    comparison_plots()
        Plot the comparison between the HABC scheme and the reference model
    cond_marker_for_eta()
        Define the conditional expressions to identify the domain of
        the absorbing layer or the reference to the original boundary
        to compute the damping profile inside the absorbing layer
    * create_mesh_habc()
        Create a mesh with absorbing layer based on the determined size
    * critical_boundary_points()
        Determine the critical points on domain boundaries of the original
        model to size an absorbing layer using the Eikonal criterion for HABCs
    damping_layer()
        Set the damping profile within the absorbing layer
    * det_reference_freq()
        Determine the reference frequency for a new layer size
    error_measures_habc()
        Compute the error measures at the receivers for the HABC scheme
    est_min_damping()
        Estimate the minimum damping ratio and the associated heuristic factor
    * fundamental_frequency()
        Compute the fundamental frequency in Hz via modal analysis
    * geometry_infinite_model()
        Determine the geometry for the infinite domain model.
    get_xCR_candidates()
        Get the heuristic factor candidates for the quadratic regression
    get_xCR_optimal()
        Get the optimal heuristic factor for the quadratic damping
    * habc_domain_dimensions()
        Determine the new dimensions of the domain with absorbing layer
    * identify_habc_case()
        Generate an identifier for the current case study of the HABC scheme
    * infinite_model()
        Create a reference model for the HABC scheme for comparative purposes
    min_reflection()
        Compute a minimum reflection coefficiente for the quadratic damping
    regression_CRmin()
        Define the minimum damping ratio and the associated heuristic factor
    * size_habc_criterion()
        Determine the size of the absorbing layer using the Eikonal criterion
    * velocity_habc()
        Set the velocity model for the model with absorbing layer
    xCR_search_range()
        Determine the initial search range for the heuristic factor xCR
    '''

    def __init__(self, dictionary=None, fwi_iter=0,
                 comm=None, output_folder="output/"):
        '''
        Initialize the HABC class.

        Parameters
        ----------
        dictionary : dict, optional
            A dictionary containing the input parameters for the HABC class
        fwi_iter : int, optional
            The iteration number for the FWI algorithm. Default is 0
        comm : object, optional
            An object representing the communication interface
            for parallel processing. Default is None
        output_folder : str, optional
            The folder where output data will be saved. Default is "output/"

        Returns
        -------
        None
        '''

        # Initializing the parent classes
        AcousticWave.__init__(self, dictionary=dictionary, comm=comm)
        HABC_Mesh.__init__(self)
        RectangLayer.__init__(self)
        NRBCHabc.__init__(self)

        # Identifier for the current case study
        self.identify_habc_case(output_folder=output_folder)

        # Current iteration
        self.fwi_iter = fwi_iter

    def identify_habc_case(self, output_folder="output/"):
        '''
        Generate an identifier for the current case study of the HABC scheme

        Parameters
        ----------
        output_folder : str, optional
            The folder where output data will be saved. Default is "output/"

        Returns
        -------
        None
        '''

        # Layer shape
        self.layer_shape = self.abc_boundary_layer_shape
        lay_str = f"\nAbsorbing Layer Shape: {self.layer_shape.capitalize()}"

        # Labeling for the layer shape
        if self.layer_shape == 'rectangular':  # Rectangular layer
            self.case_habc = 'REC'  # Label

        elif self.layer_shape == 'hypershape':  # Hypershape layer

            # Initializing the hyperelliptical layer
            HyperLayer.__init__(self, n_hyp=self.abc_deg_layer,
                                n_type=self.abc_degree_type,
                                dimension=self.dimension)

            self.case_habc = 'HN' + str(self.abc_deg_layer)  # Label
            deg_str = f" - Degree: {self.abc_deg_layer}"
            lay_str += deg_str

        else:
            value_parameter_error('layer_shape', self.layer_shape,
                                  ['rectangular', 'hypershape'])

        print(lay_str)

        # Labeling for the reference frequency for the absorbing layer
        if self.abc_reference_freq == 'boundary':
            self.case_habc += "_BND"

        elif self.abc_reference_freq == 'source':
            self.case_habc += "_SOU"

        else:
            value_parameter_error('abc_reference_freq',
                                  self.abc_reference_freq,
                                  ['boundary', 'source'])

        # Path to save data
        self.path_save = getcwd() + "/" + output_folder
        self.path_case_habc = self.path_save + self.case_habc + "/"

        # Initializing the error measure class
        HABC_Error.__init__(self, path_save=self.path_save)

    def critical_boundary_points(self):
        '''
        Determine the critical points on domain boundaries of the original
        model to size an absorbing layer using the Eikonal criterion for HABCs.
        See Salas et al (2022) for details.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Initializing Eikonal object
        Eikonal = eik.HABC_Eikonal(self)

        # Solving Eikonal
        Eikonal.solve_eik()

        # Identifying critical points
        self.eik_bnd = Eikonal.ident_crit_eik()

        # Reference length for the size of the absorbing layer
        self.lref = self.eik_bnd[0][4]

        # Critical point coordinates as receivers
        pcrit = [bnd[0] for bnd in self.eik_bnd]
        self.receiver_locations = pcrit + self.receiver_locations
        self.number_of_receivers = len(self.receiver_locations)

    def det_reference_freq(self, fpad=4):
        '''
        Determine the reference frequency for a new layer size

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

        # Nyquist frequency
        self.f_Nyq = 1.0 / (2.0 * self.dt)

        if self.abc_reference_freq == 'source':  # Initial guess

            # Theorical central Ricker source frequency
            self.freq_ref = self.frequency

        elif abc_reference_freq == 'boundary':

            # Transient response at the minimum Eikonal point
            histPcrit = self.receivers_reference[:, 0]

            # Get the minimum frequency excited at the critical point
            self.freq_ref = freq_response(
                histPcrit, self.f_Nyq, fpad=fpad, get_max_freq=True)

        print("Reference Frequency (Hz): {:.5f}".format(self.freq_ref))

    def habc_domain_dimensions(self):
        '''
        Determine the new dimensions of the domain with absorbing layer

        Parameters
        ----------
        None

        Returns
        -------
        dom_dim : `tuple`
            Original domain dimensions: (Lx, Lz) for 2D or (Lx, Lz, Ly) for 3D
        dom_lay : `tuple`
            Domain dimensions with layer. For rectangular layers, truncation
            due to the free surface is included (n = 1) while for hypershape
            layers, truncation by free surface is not included (n = 2)
            - 2D : (Lx + 2 * pad_len, Lz + n * pad_len)
            - 3D : (Lx + 2 * pad_len, Lz + n * pad_len, Ly + 2 * pad_len)
        '''

        # New geometry with layer
        self.Lx_habc = self.length_x + 2 * self.pad_len
        self.Lz_habc = self.length_z + self.pad_len

        if self.dimension == 3:  # 3D
            self.Ly_habc = self.length_y + 2 * self.pad_len

        # Original domain dimensions
        dom_dim = (self.length_x, self.length_z)

        # Domain dimension with layer without truncations

        # Labeling for the layer shape
        if self.layer_shape == 'rectangular':  # Rectangular layer
            dom_lay = (self.Lx_habc, self.Lz_habc)

        elif self.layer_shape == 'hypershape':  # Hypershape layer
            dom_lay = (self.Lx_habc, self.length_z + 2 * self.pad_len)

        if self.dimension == 3:  # 3D
            dom_dim += (self.length_y,)
            dom_lay += (self.Ly_habc,)

        return dom_dim, dom_lay

    def size_habc_criterion(self, fpad=4, n_root=1, layer_based_on_mesh=True):
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
            Adjust the layer size based on the element size. Default is True

        Returns
        -------
        None
        '''

        # Determining the reference frequency
        self.det_reference_freq(fpad=fpad)

        # Computing layer sizes
        z_par = self.eik_bnd[0][3]

        self.F_L, self.pad_len, self.ele_pad, self.d, \
            self.a_par, self.FLpos = calc_size_lay(
                self.freq_ref, z_par, self.lmin, self.lref,
                n_root=n_root, layer_based_on_mesh=layer_based_on_mesh)

        plot_function_layer_size([self.a_par, z_par],
                                 [self.freq_ref, self.frequency],
                                 [self.lmin, self.lref], self.FLpos,
                                 output_folder=self.path_case_habc)

        # New geometry with layer
        dom_dim, dom_lay = self.habc_domain_dimensions()

        if self.layer_shape == 'rectangular':

            print("\nDetermining Rectangular Layer Parameters")

            # Geometric properties of the rectangular layer
            self.calc_rec_geom_prop(dom_dim, dom_lay)

        elif self.layer_shape == 'hypershape':

            print("\nDetermining Hypershape Layer Parameters")

            # Geometric properties of the hypershape layer
            self.calc_hyp_geom_prop(dom_dim, dom_lay, self.pad_len, self.lmin)

    def create_mesh_habc(self, inf_model=False, spln=True, fmesh=1.):
        '''
        Create a mesh with absorbing layer based on the determined size.

        Parameters
        ----------
        inf_model : `bool`, optional
            If True, build a rectangular layer for the infinite or reference
            model (Model with "infinite" dimensions). Default is False
        spln : `bool`, optional
            Flag to indicate whether to use splines (True) or lines (False)
            in hypershape layer generation. Default is True
        fmesh : `float`, optional
            Mesh size factor for the hyperelliptical layer with respect to mesh
            size of the original domain. Default is 1.0.

        Returns
        -------
        None
        '''

        # Checking if the mesh for infinite model is requested
        if inf_model:
            print("\nGenerating Mesh for Infinite Model")
            layer_shape = 'rectangular'

        else:
            print("\nGenerating Mesh with Absorbing Layer")
            layer_shape = self.layer_shape

        # New mesh with layer
        if layer_shape == 'rectangular':

            mesh_habc = self.rectangular_mesh_habc()

        elif layer_shape == 'hypershape':
            dom_dim = self.habc_domain_dimensions()[0]
            hyp_par = (self.n_hyp, self.perim_hyp, *self.hyper_axes)
            mesh_habc = self.hypershape_mesh_habc(dom_dim, hyp_par,
                                                  spln=spln, fmesh=fmesh)

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

    def velocity_habc(self, inf_model=False, method='point_cloud'):
        '''
        Set the velocity model for the model with absorbing layer

        Parameters
        ----------
        inf_model : `bool`, optional
            If True, build a rectangular layer for the infinite or reference
            model (Model with "infinite" dimensions). Default is False
        method : `str`, optional
            Method to extend the velocity profile. Options:
            'point_cloud' or 'nearest_point'. Default is 'point_cloud'

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

        'point_cloud' - dx = 0.05 km (2D)
        Estimating Runtime and Used Memory
        Runtime: (s):18.437, (m):0.307, (h):0.005
        Used Memory: Current (MB):18.813, Peak (MB):25.102

        'nearest_point' - dx = 0.05 km (2D)
        Estimating Runtime and Used Memory
        Runtime: (s):20.494, (m):0.342, (h):0.006
        Used Memory: Current (MB):18.715, Peak (MB):25.298
        '''

        print("\nUpdating Velocity Profile")

        # Initialize velocity field and assigning the original velocity model
        V = fire.FunctionSpace(self.mesh, self.ele_type_c0, self.p_c0)
        self.c = fire.Function(V).interpolate(self.initial_velocity_model,
                                              allow_missing_dofs=True)

        # Domain dimensions
        dom_dim = self.habc_domain_dimensions()[0]

        # Clipping coordinates to the layer domain
        lay_field, layer_mask = self.clipping_coordinates_lay_field(dom_dim, V)

        # Extending velocity model within the absorbing layer
        self.extend_velocity_profile(lay_field, method=method)

        # Interpolating the velocity model in the layer
        self.c.interpolate(lay_field.sub(0) * layer_mask + (1 - layer_mask)
                           * self.c, allow_missing_dofs=True)
        del layer_mask, lay_field

        # Interpolating in the space function of the problem
        self.c = fire.Function(self.function_space,
                               name='c [km/s])').interpolate(self.c)

        # Save new velocity model
        if inf_model:
            file_name = "preamble/c_inf.pvd"
        else:
            file_name = self.case_habc + "/c_habc.pvd"

        outfile = fire.VTKFile(self.path_save + file_name)
        outfile.write(self.c)

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

        print("\nSolving Eigenvalue Problem")
        mod_sol = Modal_Solver(self.dimension, method=method)

        Lsp = mod_sol.solve_eigenproblem(
            self.c, self.function_space, shift=1e-8,
            quad_rule=self.quadrature_rule)

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

    def damping_layer(self, xCR_usu=None, method=None):
        '''
        Set the damping profile within the absorbing layer.
        Minimum damping ratio is computed as psi_min = xCR * d
        where xCR is the heuristic factor for the minimum damping
        ratio and d is thenormalized element size (lmin / pad_len).
        Maximum damping ratio is psi_max =  2 * pi * f_fund * psi
        where f_fund is the fundamental frequency and psi = 0.999.

        Parameters
        ----------
        xCR_usu : `float`, optional
            User-defined heuristic factor for the minimum damping ratio.
            Default is None, which defines an estimated value
        method : `str`, optional
            Method to use for solving the eigenvalue problem.
            Options: 'ARNOLDI', 'LANCZOS', 'LOBPCG' 'KRYLOVSCH_CH',
            'KRYLOVSCH_CG', 'KRYLOVSCH_GH' or 'KRYLOVSCH_GG'

        Returns
        -------
        None
        '''

        # Estimating fundamental frequency
        self.fundamental_frequency(method=method, monitor=True)

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

    def check_timestep(self, max_divisor_tf=1, set_max_dt=True):
        '''
        Check if the timestep size is appropriate for the transient response

        Parameters
        ----------
        max_divisor_tf : `int`, optional
            Index to select the maximum divisor of the final time, converted
            to an integer according to the order of magnitude of the timestep
            size. The timestep size is set to the divisor, given by the index
            in descending order, less than or equal to the user's timestep
            size. If the value is 1, the timestep size is set as the maximum
            divisor. Default is 1
        set_max_dt : `bool`, optional
            If True, set the timestep size to the selected divisor.
            Default is True

        Returns
        -------
        None
        '''

        print("\nChecking Timestep Size")

        # User timestep
        usr_dt = self.get_dt()

        # Maximum timestep size
        self.get_and_set_maximum_dt(fraction=1.0,
                                    estimate_max_eigenvalue=False)

        print("Maximum Timestep Size: {:.3f} ms".format(1e3 * self.dt))

        # Rounding power
        pot = abs(np.ceil(np.log10(self.get_dt()))) + 3

        # Maximum timestep size according to divisors of the final time
        val_int_tf = int(10**pot * self.final_time)
        val_int_dt = int(10**pot * self.get_dt())
        max_div = [d for d in divisors(val_int_tf) if d <= val_int_dt]
        index_div = min(max_divisor_tf, len(max_div))
        max_dt = 10**(-pot) * max_div[-index_div]

        # Set the timestep size
        dt = max_dt if set_max_dt else np.min(usr_dt, max_dt)
        self.set_dt(dt)

        print("Selected Timestep Size: {:.3f} ms\n".format(1e3 * self.dt))

    def geometry_infinite_model(self):
        '''
        Determine the geometry for the infinite domain model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Size of the domain extension - ToDo: Maximum boundary velocity
        add_dom = self.c_max * self.final_time / 2.

        # Distance already travelled by the wave
        if hasattr(self, 'eik_bnd'):

            # If Eikonal analysis was performed
            eikmin = self.eik_bnd[0][2]

            # Minimum distance to the nearest boundary
            dist_to_bnd = self.c_max * eikmin / 2.
        else:

            # If Eikonal analysis was not performed
            dist_to_bnd = np.inf
            for nsou in range(self.number_of_sources):
                psou_z = self.source_locations[nsou][0]
                psou_x = self.source_locations[nsou][1]
                delta_z = abs(psou_z - self.length_z)
                delta_x = min(abs(psou_x), abs(psou_x - self.length_x))

                # Candidate to minimum distance to the boundaries
                cand_dist = (dist_to_bnd, delta_z, delta_x)
                if self.dimension == 3:  # 3D
                    psou_y = self.source_locations[nsou][2]
                    delta_y = min(abs(psou_y), abs(psou_y - self.length_y))
                    cand_dist += (delta_y,)

                # Minimum distance to the nearest boundary
                dist_to_bnd = min(cand_dist)

        # Subtracting the distance already travelled by the wave
        add_dom -= dist_to_bnd

        # Pad length for the infinite domain extension
        pad_len = self.lmin * np.ceil(add_dom / self.lmin)
        self.pad_len = pad_len

        inf_str = "Infinite Domain Extension (km): {:.4f}"
        print(inf_str.format(self.pad_len))

        # Dimensions for the infinite domain
        self.Lx_habc = self.length_x + 2 * self.pad_len
        self.Lz_habc = self.length_z + self.pad_len

        if self.dimension == 3:  # 3D
            self.Ly_habc = self.length_y + 2 * self.pad_len

    def infinite_model(self, check_dt=False, max_divisor_tf=1):
        '''
        Create a reference model for the HABC scheme for comparative purposes

        Parameters
        ----------
        check_dt : `bool`, optional
            If True, check if the timestep size is appropriate for the
            transient response. Default is False
        max_divisor_tf : `int`, optional
            Index to select the maximum divisor of the final time, converted
            to an integer according to the order of magnitude of the timestep
            size. The timestep size is set to the divisor, given by the index
            in descending order, less than or equal to the user's timestep
            size. If the value is 1, the timestep size is set as the maximum
            divisor. Default is 1

        Returns
        -------
        None
        '''

        # Check the timestep size
        if check_dt:
            self.check_timestep(max_divisor_tf=max_divisor_tf)

        print("\nBuilding Infinite Domain Model")

        # Defining geometry for infinite domain
        self.geometry_infinite_model()

        # Creating mesh for infinite domain
        self.create_mesh_habc(inf_model=True)

        # Updating velocity model
        self.velocity_habc(inf_model=True)

        # Setting no damping
        self.cosHig = fire.Constant(0.)
        self.eta_mask = fire.Constant(0.)
        self.eta_habc = fire.Constant(0.)

        print("\nSolving Infinite Model")

        # Solving the forward problem
        self.forward_solve()

        # Saving reference signal
        self.save_reference_signal(self.receivers_output,
                                   self.number_of_receivers)

        # Deleting variables to be computed for the HABC scheme
        del self.pad_len, self.Lx_habc, self.Lz_habc
        del self.cosHig, self.eta_mask, self.eta_habc
        if self.dimension == 3:
            del self.Ly_habc

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
            yf = freq_response(signal, self.f_Nyq)
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
