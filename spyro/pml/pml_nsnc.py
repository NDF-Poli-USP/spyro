import firedrake as fire
import numpy as np
import spyro.habc.eik as eik
import spyro.solvers.modal.modal_sol as eigsol
from os import getcwd, path, rename
from shutil import rmtree
from sympy import divisors
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.meshing.meshing_habc import HABC_Mesh
from spyro.habc.hyp_lay import HyperLayer
from spyro.habc.rec_lay import RectangLayer
from spyro.habc.damp_profile import HABC_Damping
from spyro.habc.nrbc import NRBC
from spyro.habc.error_measure import HABC_Error
from spyro.habc.lay_len import calc_size_lay
from spyro.plots.plots_habc import plot_function_layer_size
from spyro.utils.error_management import value_parameter_error
from spyro.utils.freq_tools import freq_response

# Work from Ruben Andres Salas and Alexandre Olender


class PML_Wave(AcousticWave, HABC_Mesh, RectangLayer, HABC_Error):
    '''
    Class PML that determines PML size and parameters to be used

    Attributes
    ----------
    abc_reference_freq : `str`
        Reference frequency for sizing the hybrid absorbing layer.
        Options: 'source' or 'boundary'
    a_par : `float`
        Adimensional propagation speed parameter (a = z / f).
        "z" parameter is the inverse of the minimum Eikonal (1 / phi_min)
    case_pml : `str`
        Label for the output files that includes the layer shape
        ('REC', only available) and the reference frequency
        ('SOU' or 'BND'). Example: 'REC_SOU' or 'HN2_BND'
    crit_source : `tuple`
       Critical source coordinates
    CRmin : `float`
        Minimum reflection coefficient at the minimum damping ratio
    d : `float`
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
    ele_pad : `int`
        Number of elements in the layer of edge length 'lmin'
    sigma_pml : `firedrake function`
        Damping profile within the PML
    sigma_mask : `firedrake function`
        Mask function to identify the PML domain
    F_L : `float`
        Size parameter of the absorbing layer
    FLpos : `list`
        Possible size parameters for the absorbing layer without rounding
    f_Nyq : `float`
        Nyquist frequency according to the time step. f_Nyq = 1 / (2 * dt)
    freq_ref : `float`
        Reference frequency of the wave at the boundary
    fundam_freq : `float`
        Fundamental frequency of the numerical model
    fwi_iter : `int`
        The iteration number for the Full Waveform Inversion (FWI) algorithm
    Lx_habc : `float`
        Length of the domain in the x-direction with absorbing layer
    Ly_habc : `float`
        Length of the domain in the y-direction with absorbing layer (3D)
    Lz_habc : `float`
        Length of the domain in the z-direction with absorbing layer
    layer_shape : `string`
        Shape type of pad layer. Options: 'rectangular' or 'hypershape'
    lref : `float`
        Reference length for the size of the absorbing layer
    mesh: `firedrake mesh`
        Mesh used in the simulation (HABC or Infinite Model)
    number_of_receivers: `int`
        Number of receivers used in the simulation
    pad_len : `float`
        Size of the absorbing layer
    path_case_pml : `string`
        Path to save data for the current case study
    path_save : `string`
        Path to save data
    psi_min : `float`
        Minimum damping ratio of the absorbing layer (psi_min = xCR * d)
    receiver_locations: `list`
        List of receiver locations
    receivers_output : `array`
        Receiver waveform data in the HABC scheme
    rename_folder_habc()
        Rename the folder of results if the degree for the hypershape
        layer is out of the criterion limits
    xCR : `float`
        Heuristic factor for the minimum damping ratio
    xCR_lim: `list`
        Limits for the heuristic factor

    Methods
    -------
    check_timestep_habc()
        Check if the timestep size is appropriate for the transient response
    create_mesh_pml()
        Create a mesh with absorbing layer based on the determined size
    critical_boundary_points()
        Determine the critical points on domain boundaries of the original
        model to size an absorbing layer using the Eikonal criterion for HABCs
    pml_layer()
        Set the damping profile within the absorbing layer
    det_reference_freq()
        Determine the reference frequency for a new layer size
    fundamental_frequency()
        Compute the fundamental frequency in Hz via modal analysis
    geometry_infinite_model()
        Determine the geometry for the infinite domain model.
    pml_domain_dimensions()
        Determine the new dimensions of the domain with absorbing layer
    pml_new_geometry ()
        Determine the new domain geometry with the absorbing layer
    identify_pml_case()
        Generate an identifier for the current case study of the HABC scheme
    infinite_model()
        Create a reference model for the HABC scheme for comparative purposes
    layer_infinite_model()
        Determine the domain extension size for the infinite domain model
    nrbc_on_boundary_layer()
        Apply the Higdon ABCs on the outer boundary of the absorbing layer
    size_pml_criterion()
        Determine the size of the absorbing layer using the Eikonal criterion
    velocity_pml()
        Set the velocity model for the model with absorbing layer
    '''

    def __init__(self, dictionary=None, fwi_iter=0,
                 comm=None, output_folder=None):
        '''
        Initialize the HABC class

        Parameters
        ----------
        dictionary : `dict`, optional
            A dictionary containing the input parameters for the HABC class
        fwi_iter : int, optional
            The iteration number for the FWI algorithm. Default is 0
        comm : `object`, optional
            An object representing the communication interface
            for parallel processing. Default is None
        output_folder : `str`, optional
            The folder where output data will be saved. Default is None

        Returns
        -------
        None
        '''

        # Initializing the Wave class
        AcousticWave.__init__(self, dictionary=dictionary, comm=comm)

        # Nyquist frequency
        self.f_Nyq = 1.0 / (2.0 * self.dt)

        # Original domain dimensions
        dom_dim = self.pml_domain_dimensions(only_orig_dom=True)

        # Initializing the Mesh class
        HABC_Mesh.__init__(
            self, dom_dim, dimension=self.dimension, comm=self.comm)

        # Identifier for the current case study
        self.identify_pml_case(output_folder=output_folder)

        # Current iteration
        self.fwi_iter = fwi_iter

    def identify_pml_case(self, output_folder=None):
        '''
        Generate an identifier for the current case study of the HABC scheme

        Parameters
        ----------
        output_folder : `str`, optional
            The folder where output data will be saved. Default is None

        Returns
        -------
        None
        '''

        # Original domain dimensions
        dom_dim = self.pml_domain_dimensions(only_orig_dom=True)

        # Layer shape
        self.layer_shape = self.abc_boundary_layer_shape
        lay_str = f"\nPML Layer Shape: {self.layer_shape.capitalize()}"

        # Initializing the rectangular layer
        RectangLayer.__init__(self, dom_dim, dimension=self.dimension)
        self.case_pml = 'REC'  # Label

        value_parameter_error('layer_shape', self.layer_shape, ['rectangular'])
        print(lay_str, flush=True)

        # Labeling for the reference frequency for the absorbing layer
        if self.abc_reference_freq == 'boundary':
            self.case_pml += "_BND"

        elif self.abc_reference_freq == 'source':
            self.case_pml += "_SOU"

        else:
            value_parameter_error('abc_reference_freq',
                                  self.abc_reference_freq,
                                  ['boundary', 'source'])

        # Path to save data
        if output_folder is None:
            self.path_save = getcwd() + "/output/"
        else:
            self.path_save = getcwd() + "/" + output_folder + "/"

        self.path_case_pml = self.path_save + self.case_pml + "/"

        # Initializing the error measure class
        HABC_Error.__init__(self, self.dt, self.f_Nyq,
                            self.receiver_locations,
                            output_folder=self.path_save,
                            output_case=self.path_case_pml)

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

        # Critical point coordinates as receivers
        pcrit = [bnd[0] for bnd in self.eik_bnd]
        self.receiver_locations = pcrit + self.receiver_locations
        self.number_of_receivers = len(self.receiver_locations)

    def det_reference_freq(self, fpad=4):
        '''
        Determine the reference frequency for a new layer size

        Parameters
        ----------
        fpad : `int`, optional
            Padding factor for FFT. Default is 4

        Returns
        -------
        None
        '''

        print("\nDetermining Reference Frequency", flush=True)

        abc_reference_freq = self.abc_reference_freq \
            if hasattr(self, 'receivers_reference') else 'source'

        if self.abc_reference_freq == 'source':  # Initial guess

            # Theorical central Ricker source frequency
            self.freq_ref = self.frequency

        elif abc_reference_freq == 'boundary':

            # Reference frequency of the wave at the boundary
            self.freq_ref = np.inf

            for n_crit in range(self.number_of_receivers):

                # Transient response at each critical Eikonal point
                histPcrit = self.receivers_reference[:, n_crit]

                # Get the minimum frequency excited at each critical point
                freq_ref = freq_response(histPcrit, self.f_Nyq,
                                         fpad=fpad, get_max_freq=True)
                print("Frequency at Critical Point {:>2.0f}: {:.5f}".format(
                    n_crit, freq_ref), flush=True)

                self.freq_ref = min(self.freq_ref, freq_ref)

        print("Reference Frequency (Hz): {:.5f}".format(self.freq_ref))

    def pml_new_geometry(self):
        '''
        Determine the new domain geometry with the absorbing layer

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # New geometry with layer
        self.Lx_habc = self.length_x + 2 * self.pad_len
        self.Lz_habc = self.length_z + self.pad_len
        if self.dimension == 3:  # 3D
            self.Ly_habc = self.length_y + 2 * self.pad_len

    def pml_domain_dimensions(self, only_orig_dom=False, only_habc_dom=False):
        '''
        Determine the new dimensions of the domain with absorbing layer

        Parameters
        ----------
        only_orig_dom : `bool`, optional
            Return only the original domain dimensions. Default is False
        only_habc_dom : `bool`, optional
            Return only the domain dimensions with layer. Default is False

        Returns
        -------
        dom_dim : `tuple`
            Original domain dimensions: (Lx, Lz) for 2D or (Lx, Lz, Ly) for 3D
        dom_lay : `tuple`
            Domain dimensions with layer. The truncation due to the free
            surface is included (n = 1). Dimensions are defined as:
            - 2D : (Lx + 2 * pad_len, Lz + n * pad_len)
            - 3D : (Lx + 2 * pad_len, Lz + n * pad_len, Ly + 2 * pad_len)
        '''

        # Original domain dimensions
        dom_dim = (self.length_x, self.length_z)
        if self.dimension == 3:  # 3D
            dom_dim += (self.length_y,)

        if only_orig_dom:
            return dom_dim

        # Domain dimension with layer
        dom_lay = (self.Lx_habc, self.Lz_habc)

        if self.dimension == 3:  # 3D
            dom_lay += (self.Ly_habc,)

        if only_habc_dom:
            return dom_lay

        return dom_dim, dom_lay

    def size_pml_criterion(self, fpad=4, n_root=1, layer_based_on_mesh=True):
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
        freq_ref_lst = self.det_reference_freq(fpad=fpad)

        # Inverse of the minimum Eikonal
        z_par = self.eik_bnd[0][3]

        # Reference length for the size of the absorbing layer
        self.lref = self.eik_bnd[0][4]

        # Critical source position
        self.crit_source = self.eik_bnd[0][-1]

        # Computing layer sizes
        self.F_L, self.pad_len, self.ele_pad, self.d, \
            self.a_par, self.FLpos = calc_size_lay(
                self.freq_ref, z_par, self.lmin, self.lref,
                n_root=n_root, layer_based_on_mesh=layer_based_on_mesh)

        plot_function_layer_size([self.a_par, z_par],
                                 [self.freq_ref, self.frequency],
                                 [self.lmin, self.lref], self.FLpos,
                                 output_folder=self.path_case_pml)

        print("\nDetermining New Geometry with PML", flush=True)

        # New geometry with layer
        self.pml_new_geometry()

        # Domain dimensions without free surface truncation
        dom_lay_full = self.pml_domain_dimensions(only_habc_dom=True)

        print("Determining Rectangular Layer Parameters", flush=True)

        # Geometric properties of the rectangular layer
        self.calc_rec_geom_prop(dom_lay_full, self.pad_len)

    def create_mesh_pml(self, inf_model=False, spln=True, fmesh=1.):
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
            print("\nGenerating Mesh for Infinite Model", flush=True)
            layer_shape = 'rectangular'

        else:
            print("\nGenerating Mesh with PML", flush=True)
            layer_shape = self.layer_shape

        # New mesh with layer
        dom_lay = self.pml_domain_dimensions(only_habc_dom=True)
        mesh_pml = self.rectangular_mesh_habc(dom_lay, self.pad_len)

        # Updating the mesh with the absorbing layer
        self.set_mesh(user_mesh=mesh_pml, mesh_parameters={})
        print("Mesh Generated Successfully", flush=True)

        if inf_model:
            pth_mesh = self.path_save + "preamble/mesh_inf.pvd"
        else:
            pth_mesh = self.path_case_pml + "mesh_pml.pvd"

        # Save new mesh
        outfile = fire.VTKFile(pth_mesh)
        outfile.write(self.mesh)

    def velocity_pml(self, inf_model=False, method='point_cloud'):
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
        '''

        print("\nUpdating Velocity Profile", flush=True)

        # Initialize velocity field and assigning the original velocity model
        V = fire.FunctionSpace(self.mesh, self.ele_type_c0, self.p_c0)
        self.c = fire.Function(V).interpolate(self.initial_velocity_model,
                                              allow_missing_dofs=True)

        # Clipping coordinates to the layer domain
        lay_field, layer_mask = self.clipping_coordinates_lay_field(V)

        # Extending velocity model within the absorbing layer
        self.extend_velocity_profile(lay_field, layer_mask, method=method)

        # Interpolating the velocity model in the layer
        self.c.interpolate(lay_field.sub(0) * layer_mask + (
            1. - layer_mask) * self.c, allow_missing_dofs=True)
        del layer_mask, lay_field

        # Interpolating in the space function of the problem
        self.c = fire.Function(self.function_space,
                               name='c [km/s])').interpolate(self.c)

        # Save new velocity model
        if inf_model:
            file_name = "preamble/c_inf.pvd"
        else:
            file_name = self.case_pml + "/c_pml.pvd"

        outfile = fire.VTKFile(self.path_save + file_name)
        outfile.write(self.c)

    def calc_pml_prop(self, CR=0.001):
        '''
        Calculate the maximum damping ratio for the PML layer

        Parameters
        ----------
        CR : `float`, optional
            Desired reflection coefficient at outer boundary of PML layer.
            Default is 0.001

        Returns
        -------
        sigma_max : `float`
            Maximum damping ratio within the PML layer
        '''

        dgr_prof = 2.  # Degree of the damping profile within the PML layer
        sigma_max = (dgr_prof + 1) / (2. * self.pad_len) * np.log(1 / CR)

        return sigma_max

    def pml_layer(self):
        '''
        Set the damping profile within the PML layer.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        print("\nCreating Damping PML Profile", flush=True)

        # Compute the maximum damping coefficient
        sigma_max = calc_pml_prop(self)

        # Mesh coordinates
        coords = fire.SpatialCoordinate(self.mesh)

        # Damping mask
        V_mask = fire.FunctionSpace(self.mesh, 'DG', 0)
        self.sigma_mask = self.layer_mask_field(coords, V_mask,
                                                type_marker='mask',
                                                name_mask='sigma_mask')

        # Save damping mask
        outfile = fire.VTKFile(self.path_case_pml + "sigma_mask.pvd")
        outfile.write(self.sigma_mask)

        # Coefficients for quadratic PML function
        aq, bq = 1., 0.

        # Damping field
        damp_par = (self.pad_len, sigma_max, aq, bq)
        self.sigma_pml = self.c * self.layer_mask_field(
            coords, self.function_space, damp_par=damp_par,
            type_marker='damping', name_mask='eta [1/s])')

        # Save damping profile
        outfile = fire.VTKFile(self.path_case_pml + "sigma_pml.pvd")
        outfile.write(self.sigma_pml)

    def nrbc_on_boundary_layer(self):
        '''
        Apply the Higdon ABCs on the outer boundary of the absorbing layer

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        print("\nApplying Non-Reflecting Boundary Conditions", flush=True)

        # Getting boundary data from the layer boundaries
        bnd_nfs, bnd_nodes_nfs = self.layer_boundary_data(self.function_space)

        # Hypershape parameters
        if self.layer_shape == 'hypershape':
            hyp_par = (self.n_hyp, *self.hyper_axes)
        else:
            hyp_par = None

        # Applying Higdon ABCs
        self.cos_ang_HigdonBC(self.function_space, self.crit_source,
                              bnd_nfs, bnd_nodes_nfs, hyp_par=hyp_par)

    def check_timestep_habc(self, max_divisor_tf=1, set_max_dt=True,
                            method='ANALYTICAL', mag_add=3):
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
        method : `str`, optional
            Method to use for solving the eigenvalue problem. Default
            is 'ANALYTICAL' method that estimates the maximum eigenvalue
            using the Gershgorin Circle Theorem.
            Opts: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS' or 'LOBPCG'
        mag_add : `int`, optional
            Additional magnitude order to adjust the rounding of the timestep

        Returns
        -------
        None

        # Estimation: 2.770 (Old), 2.768 (New) (Scipy-sparse)
        # Exact: 1.842 (Old), 1.842 (New) (Scipy)
        '''

        print("\nChecking Timestep Size", flush=True)

        # User timestep
        usr_dt = self.get_dt()

        # Maximum timestep size
        dt_sol = eigsol.Modal_Solver(
            self.dimension, method=method, calc_max_dt=True)
        max_dt = dt_sol.estimate_timestep(
            self.c, self.function_space, self.final_time, shift=1e-8,
            quad_rule=self.quadrature_rule, fraction=1.)

        # Rounding power
        pot = int(abs(np.ceil(np.log10(max_dt))) + mag_add)

        # Maximum timestep size according to divisors of the final time
        val_int_tf = int(10**pot * self.final_time)
        val_int_dt = int(10**pot * max_dt)
        max_div = [d for d in divisors(val_int_tf) if d < val_int_dt]
        n_div = len(max_div)
        index_div = min(max_divisor_tf, n_div)
        max_dt = round(10**(-pot) * max_div[-index_div], pot)

        # Set the timestep size
        dt = max_dt if set_max_dt else min(usr_dt, max_dt)
        self.set_dt(dt)
        if set_max_dt:
            str_dt = "Selected Timestep Size ({} of {}): {:.{p}f} ms\n".format(
                min(max_divisor_tf, n_div), n_div, 1e3 * self.dt, p=mag_add)
        else:
            str_dt = "Selected Timestep Size: {:.{p}f} ms\n".format(
                n_div, 1e3 * self.dt, p=mag_add)

        print(str_dt, flush=True)

    def layer_infinite_model(self):
        '''
        Determine the domain extension size for the infinite domain model

        Parameters
        ----------
        None

        Returns
        -------
        infinite_pad_len : `float`
            Size of the domain extension for the infinite domain model
        '''

        # Size of the domain extension
        add_dom = self.c_bnd_max * self.final_time / 2.

        # Distance already travelled by the wave
        if hasattr(self, 'eik_bnd'):

            # If Eikonal analysis was performed
            eikmin = self.eik_bnd[0][2]

            # Minimum distance to the nearest boundary
            dist_to_bnd = self.c_bnd_max * eikmin / 2.
        else:

            # If Eikonal analysis was not performed
            sources_loc = np.array(self.source_locations)

            # Candidate to minimum distance to the boundaries
            delta_z = np.abs(sources_loc[:, 0] - self.length_z)
            delta_x = np.minimum(np.abs(sources_loc[:, 1]),
                                 np.abs(sources_loc[:, 1] - self.length_x))
            cand_dist = (delta_z, delta_x)

            if self.dimension == 3:  # 3D
                delta_y = np.minimum(np.abs(sources_loc[:, 2]),
                                     np.abs(sources_loc[:, 2] - self.length_y))
                cand_dist += (delta_y,)

            # Minimum distance to the nearest boundary
            dist_to_bnd = min(cand_dist)

        # Subtracting the distance already travelled by the wave
        add_dom -= dist_to_bnd

        # Pad length for the infinite domain extension
        infinite_pad_len = self.lmin * np.ceil(add_dom / self.lmin)

        return infinite_pad_len

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

        # Size of the domain extension
        self.pad_len = self.layer_infinite_model()

        inf_str = "Infinite Domain Extension (km): {:.4f}"
        print(inf_str.format(self.pad_len), flush=True)

        # Dimensions for the infinite domain
        self.Lx_habc = self.length_x + 2 * self.pad_len
        self.Lz_habc = self.length_z + self.pad_len

        if self.dimension == 3:  # 3D
            self.Ly_habc = self.length_y + 2 * self.pad_len

    def infinite_model(self, check_dt=False, max_divisor_tf=1,
                       method='ANALYTICAL', mag_add=3):
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
        method : `str`, optional
            Method to use for solving the eigenvalue problem. Default
            is 'ANALYTICAL' method that estimates the maximum eigenvalue
            using the Gershgorin Circle Theorem.
            Opts: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS' or 'LOBPCG'
        mag_add : `int`, optional
            Additional magnitude order to adjust the rounding of the timestep

        Returns
        -------
        None
        '''

        # Check the timestep size
        if check_dt:
            self.check_timestep_habc(
                max_divisor_tf=max_divisor_tf, method=method, mag_add=mag_add)

        print("\nBuilding Infinite Domain Model", flush=True)

        # Defining geometry for infinite domain
        self.geometry_infinite_model()

        # Creating mesh for infinite domain
        self.create_mesh_pml(inf_model=True)

        # Updating velocity model
        self.velocity_pml(inf_model=True)

        # Setting no damping
        self.cosHig = fire.Constant(0.)
        self.sigma_mask = fire.Constant(0.)
        self.sigma_pml = fire.Constant(0.)

        print("\nSolving Infinite Model", flush=True)

        # Solving the forward problem
        self.forward_solve()

        # Saving reference signal
        self.save_reference_signal()

        # Deleting variables to be computed for the HABC scheme
        del self.pad_len, self.Lx_habc, self.Lz_habc
        del self.cosHig, self.sigma_mask, self.sigma_pml
        if self.dimension == 3:
            del self.Ly_habc

    def rename_folder_habc(self):
        '''
        Rename the folder of results if the degree for the
        hypershape layer is out of the criterion limits

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        if self.n_hyp != self.abc_deg_layer and \
                self.layer_shape == 'hypershape':

            print("Output Folder for Results Will Be Renamed.", flush=True)

            # Define the current and new folder names
            old = self.path_case_pml
            new = self.path_case_pml[:-8] + f"{self.n_hyp:.1f}" + \
                self.path_case_pml[-5:]

            try:
                if path.isdir(new):
                    # Remove target directory if it exists
                    rmtree(new)

                rename(old, new)  # Rename the directory
                print(f"Folder '{old}' Successfully Renamed to '{new}'.", flush=True)

            except OSError as e:
                print(f"Error Renaming Folder: {e}", flush=True)
