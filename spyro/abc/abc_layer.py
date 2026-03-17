import firedrake as fire
import numpy as np
import spyro.habc.eik as eik
import spyro.solvers.modal.modal_sol as eigsol
from os import getcwd
from sympy import divisors
from spyro.solvers.acoustic_wave import AcousticWave
from spyro.meshing.meshing_habc import HABC_Mesh
from spyro.abc.hyp_lay import HyperLayer
from spyro.abc.rec_lay import RectangLayer
from spyro.habc.error_measure import HABC_Error
from spyro.habc.lay_len import calc_size_lay
from spyro.plots.plots_habc import plot_function_layer_size
from spyro.utils.error_management import value_parameter_error
from spyro.utils.freq_tools import freq_response


class ABC_Layer_Wave(AcousticWave, HABC_Mesh, RectangLayer,
                     HyperLayer, HABC_Error):
    '''
    Class PML that determines PML size and parameters to be used

    Attributes
    ----------
    abc_boundary_layer_shape : `string`
        Shape type of pad layer. Options: 'rectangular' or 'hypershape'
    abc_pad_length : `float`
        Size of the absorbing layer
    abc_reference_freq : `str`
        Reference frequency for sizing the hybrid absorbing layer.
        Options: 'source' or 'boundary'
    a_par : `float`
        Adimensional propagation speed parameter (a = z / f).
        "z" parameter is the inverse of the minimum Eikonal (1 / phi_min)
    case : `str`
        Label for the output files that includes the layer shape
        ('REC', only available) and the reference frequency
        ('SOU' or 'BND'). Example: 'REC_SOU' or 'REC_BND'
    crit_source : `tuple`
       Critical source coordinates
    d_norm : `float`
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
    factor_length_pad : `float`
        Size parameter of the absorbing layer
    FLpossible : `list`
        Possible size parameters for the absorbing layer without rounding
    f_Nyquist : `float`
        Nyquist frequency according to the time step. f_Nyquist = 1 / (2 * dt)
    freq_ref : `float`
        Reference frequency of the wave at the boundary
    fwi_iter : `int`
        The iteration number for the Full Waveform Inversion (FWI) algorithm
    Lx_abc : `float`
        Length of the domain in the x-direction with absorbing layer
    Ly_abc : `float`
        Length of the domain in the y-direction with absorbing layer (3D)
    Lz_abc : `float`
        Length of the domain in the z-direction with absorbing layer
    lref : `float`
        Reference length for the size of the absorbing layer
    mesh: `firedrake mesh`
        Mesh used in the simulation (ABC or Infinite Model)
    number_of_receivers: `int`
        Number of receivers used in the simulation
    path_case_abc : `string`
        Path to save data for the current case study
    path_save : `string`
        Path to save data
    receiver_locations: `list`
        List of receiver locations

    Methods
    -------
    abc_domain_dimensions()
        Determine the new dimensions of the domain with absorbing layer
    abc_new_geometry
        Determine the new domain geometry with the absorbing layer
    check_timestep_abc()
        Check if the timestep size is appropriate for the transient response
    create_mesh_with_layer()
        Create a mesh with absorbing layer based on the determined size
    critical_boundary_points()
        Determine the critical points on domain boundaries of the original
        model to size an absorbing layer using the Eikonal criterion for HABCs
    det_reference_freq()
        Determine the reference frequency for a new layer size
    formatting_abc_layer_type()
        Format a string for the ABC layer type.
    geometry_infinite_model()
        Determine the geometry for the infinite domain model.
    identify_abc_layer_case()
        Generate an identifier for the current case study of the ABC scheme
    infinite_model()
        Create a reference model for the HABC scheme for comparative purposes
    layer_infinite_model()
        Determine the domain extension size for the infinite domain model
    layer_size_criterion()
        Determine the size of the absorbing layer using the Eikonal criterion
    velocity_abc()
        Set the velocity model for the model with absorbing layer
    '''

    def __init__(self, dictionary=None, fwi_iter=0,
                 comm=None, output_folder=None):
        '''
        Initialize the ABC_Layer class

        Parameters
        ----------
        dictionary : `dict`, optional
            A dictionary containing the input parameters for the class
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
        self.f_Nyquist = 1.0 / (2.0 * self.dt)

        # Original domain dimensions
        domain_dim = self.abc_domain_dimensions(only_orig_dom=True)

        # Initializing the Mesh class for ABC scheme
        HABC_Mesh.__init__(
            self, domain_dim, dimension=self.dimension, comm=self.comm)

        # Identifier for the current case study
        self.identify_abc_layer_case(output_folder=output_folder)

        # Current iteration
        self.fwi_iter = fwi_iter

    def formatting_abc_layer_type(self, str_to_format, for_prints=True):
        '''
        Format a string for the ABC layer type.

        Parameters
        ----------
        str_to_format : `str`
            The string to format
        for_prints : `bool`, optional
            Flag to indicate whether the formatted string is for
            printing (True) or for labeling (False). Default is True

        Returns
        -------
        formatted_str : `str`
            The formatted string for the ABC layer type.
        '''

        # Layer type
        if self._abc_boundary_layer_type == "hybrid":
            abc_layer_str = "Absorbing" if for_prints else "habc"
        elif self._abc_boundary_layer_type == "PML":
            abc_layer_str = "PML" if for_prints else "pml"
        else:
            value_parameter_error('abc_layer_str',
                                  self._abc_boundary_layer_type,
                                  ["hybrid", "PML"])

        formatted_str = str_to_format.format(abc_layer_str)

        return formatted_str

    def identify_abc_layer_case(self, output_folder=None):
        '''
        Generate an identifier for the current case study of
        the ABC scheme (HABC or PML).

        Parameters
        ----------
        output_folder : `str`, optional
            The folder where output data will be saved. Default is None

        Returns
        -------
        None
        '''

        # Original domain dimensions
        domain_dim = self.abc_domain_dimensions(only_orig_dom=True)

        # Layer type
        lay_str = "\n{} Layer Shape: " + \
            f"{self.abc_boundary_layer_shape.capitalize()}"
        lay_str = self.formatting_abc_layer_type(lay_str)

        # Labeling for the layer shape
        if self.abc_boundary_layer_shape == 'rectangular':  # Rectangular layer

            # Initializing the rectangular layer
            RectangLayer.__init__(self, domain_dim, dimension=self.dimension)
            self.case = 'REC'  # Label

        elif self.abc_boundary_layer_shape == 'hypershape':  # Hypershape layer

            # Initializing the hyperelliptical layer
            HyperLayer.__init__(self, domain_dim, n_hyp=self.abc_deg_layer,
                                n_type=self.abc_degree_type,
                                dimension=self.dimension)

            self.case = 'HN' + f"{self.abc_deg_layer:.1f}"  # Label
            deg_str = f" - Degree: {self.abc_deg_layer}"
            lay_str += deg_str

        else:
            value_parameter_error('abc_boundary_layer_shape',
                                  self.abc_boundary_layer_shape,
                                  ['rectangular', 'hypershape'])
        print(lay_str, flush=True)

        # Labeling for the reference frequency for the absorbing layer
        if self.abc_reference_freq == 'boundary':
            self.case += "_BND"

        elif self.abc_reference_freq == 'source':
            self.case += "_SOU"

        else:
            value_parameter_error('abc_reference_freq',
                                  self.abc_reference_freq,
                                  ['boundary', 'source'])

        # Path to save data
        if output_folder is None:
            self.path_save = getcwd() + "/output/"
        else:
            self.path_save = getcwd() + "/" + output_folder + "/"

        self.path_case_abc = self.path_save + self.case + "/"

        # Initializing the error measure class
        HABC_Error.__init__(self, self.dt, self.f_Nyquist,
                            self.receiver_locations,
                            output_folder=self.path_save,
                            output_case=self.path_case_abc)

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
                freq_ref = freq_response(histPcrit, self.f_Nyquist,
                                         fpad=fpad, get_max_freq=True)
                print("Frequency at Critical Point {:>2.0f}: {:.5f}".format(
                    n_crit, freq_ref), flush=True)

                self.freq_ref = min(self.freq_ref, freq_ref)

        print("Reference Frequency (Hz): {:.5f}".format(self.freq_ref))

    def abc_new_geometry(self):
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
        self.Lx_abc = self.mesh_parameters.length_x + 2 * self.abc_pad_length
        self.Lz_abc = self.mesh_parameters.length_z + self.abc_pad_length

        if self.dimension == 3:  # 3D
            self.Ly_abc = self.mesh_parameters.length_y \
                + 2 * self.abc_pad_length

    def abc_domain_dimensions(self, only_orig_dom=False,
                              only_abc_dom=False, full_hyp=True):
        '''
        Determine the new dimensions of the domain with absorbing layer

        Parameters
        ----------
        only_orig_dom : `bool`, optional
            Return only the original domain dimensions. Default is False
        only_abc_dom : `bool`, optional
            Return only the domain dimensions with layer. Default is False
        full_hyp : `bool`, optional
            Option to get the domain dimensions in hypershape layers.
            If True, the domain dimensions with layer do not include truncation
            due to the free surface. If False, the domain dimensions with layer
            include truncation by free surface. Default is True.

        Returns
        -------
        domain_dim : `tuple`
            Original domain dimensions: (Lx, Lz) for 2D or (Lx, Lz, Ly) for 3D
        domain_layer : `tuple`
            Domain dimensions with layer. For rectangular layers, truncation
            due to the free surface is included (n = 1). For hypershape layers,
            truncation by free surface is not included (n = 2) if 'full_hyp' is
            True; otherwise, it is included (n = 1). Dimensions are defined as:
            - 2D : (Lx + 2 * pad_len, Lz + n * pad_len)
            - 3D : (Lx + 2 * pad_len, Lz + n * pad_len, Ly + 2 * pad_len)
        '''

        # Original domain dimensions
        domain_dim = (self.mesh_parameters.length_x,
                      self.mesh_parameters.length_z)
        if self.dimension == 3:  # 3D
            domain_dim += (self.mesh_parameters.length_y,)

        if only_orig_dom:
            return domain_dim

        # Domain dimension with layer w/ or w/o truncations
        if self.abc_boundary_layer_shape == 'rectangular':  # Rectangular layer
            domain_layer = (self.Lx_abc, self.Lz_abc)

        elif self.abc_boundary_layer_shape == 'hypershape':  # Hypershape layer
            domain_layer = (self.Lx_abc, self.mesh_parameters.length_z
                            + 2 * self.abc_pad_length) \
                if full_hyp else (self.Lx_abc, self.Lz_abc)

        if self.dimension == 3:  # 3D
            domain_layer += (self.Ly_abc,)

        if only_abc_dom:
            return domain_layer

        return domain_dim, domain_layer

    def layer_size_criterion(self, fpad=4, n_root=1, layer_based_on_mesh=True):
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

        # Inverse of the minimum Eikonal
        z_par = self.eik_bnd[0][3]

        # Reference length for the size of the absorbing layer
        self.lref = self.eik_bnd[0][4]

        # Critical source position
        self.crit_source = self.eik_bnd[0][-1]

        # Computing layer sizes
        self.factor_length_pad, self.abc_pad_length, self.ele_pad, \
            self.d_norm, self.a_par, self.FLpossible = calc_size_lay(
                self.freq_ref, z_par, self.lmin, self.lref,
                n_root=n_root, layer_based_on_mesh=layer_based_on_mesh)

        plot_function_layer_size([self.a_par, z_par],
                                 [self.freq_ref, self.frequency],
                                 [self.lmin, self.lref], self.FLpossible,
                                 output_folder=self.path_case_abc)

        # Layer type
        lay_str = "\nDetermining New Geometry with {}"
        lay_str = self.formatting_abc_layer_type(lay_str)
        print(lay_str, flush=True)

        # New geometry with layer
        self.abc_new_geometry()

        # Domain dimensions without free surface truncation
        domain_layer_full = self.abc_domain_dimensions(only_abc_dom=True)

        if self.abc_boundary_layer_shape == 'rectangular':

            print("Determining Rectangular Layer Parameters", flush=True)

            # Geometric properties of the rectangular layer
            self.calc_rec_geom_prop(domain_layer_full, self.abc_pad_length)

        elif self.abc_boundary_layer_shape == 'hypershape':

            print("Determining Hypershape Layer Parameters", flush=True)

            # Geometric properties of the hypershape layer
            self.calc_hyp_geom_prop(
                dom_lay_full, self.abc_pad_length, self.lmin)

    def create_mesh_with_layer(self, inf_model=False, spln=True):
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

        Returns
        -------
        None
        '''

        # Checking if the mesh for infinite model is requested
        if inf_model:
            print("\nGenerating Mesh for Infinite Model", flush=True)
            layer_shape = 'rectangular'

        else:
            print("\nGenerating Mesh with Absorbing Layer", flush=True)
            layer_shape = self.abc_boundary_layer_shape

        # New mesh with layer
        if layer_shape == 'rectangular':
            domain_layer = self.abc_domain_dimensions(only_abc_dom=True)
            mesh_abc = self.rectangular_mesh_habc(domain_layer,
                                                  self.abc_pad_length)

        elif layer_shape == 'hypershape':

            # Parameters for hypershape mesh
            if self.dimension == 2:  # 2D
                geometry_param = self.perim_hyp

            if self.dimension == 3:  # 3D
                geometry_param = self.surf_hyp

            hypershape_param = (self.n_hyp, geometry_param, *self.hyper_axes)
            mesh_abc = self.hypershape_mesh_habc(hypershape_param, spln=spln)

        # Updating the mesh with the absorbing layer
        self.set_mesh(user_mesh=mesh_abc)
        print("Mesh Generated Successfully")

        if inf_model:
            pth_mesh = self.path_save + "preamble/mesh_inf.pvd"
        else:
            mesh_file_name = self.formatting_abc_layer_type("mesh_{}.pvd",
                                                            for_prints=False)
            pth_mesh = self.path_case_abc + mesh_file_name

        # Save new mesh
        outfile = fire.VTKFile(pth_mesh)
        outfile.write(self.mesh)

    def velocity_abc(self, inf_model=False, method='point_cloud'):
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

        print("\nUpdating Velocity Profile", flush=True)

        # Initialize velocity field and assigning the original velocity model
        if self.quadrilateral:
            base_mesh = self.mesh._base_mesh
            base_cell = base_mesh.ufl_cell()
            element_zx = fire.FiniteElement("DQ", base_cell, 0,
                                            variant="spectral")
            element_y = fire.FiniteElement("DG", fire.interval, 0,
                                           variant="spectral")
            tensor_element = fire.TensorProductElement(element_zx, element_y)
            V = fire.FunctionSpace(self.mesh, tensor_element)
        else:
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
            c_file_name = self.formatting_abc_layer_type("/c_{}.pvd",
                                                         for_prints=False)
            file_name = self.case + c_file_name

        outfile = fire.VTKFile(self.path_save + file_name)
        outfile.write(self.c)

    def check_timestep_abc(self, max_divisor_tf=1, set_max_dt=True,
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
        dt_ms = 1e3 * self.dt
        if set_max_dt:
            str_dt = "Selected Timestep Size ({} of {}): {:.{p}f} ms".format(
                min(max_divisor_tf, n_div), n_div, dt_ms, p=mag_add)
        else:
            str_dt = "Selected Timestep Size: {:.{p}f} ms".format(dt_ms,
                                                                  p=mag_add)

        # Updating Nyquist frequency
        self.f_Nyquist = 1. / (2. * self.dt)

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
            delta_z = np.abs(sources_loc[:, 0] - self.mesh_parameters.length_z)
            delta_x = np.minimum(np.abs(sources_loc[:, 1]),
                                 np.abs(sources_loc[:, 1]
                                        - self.mesh_parameters.length_x))
            cand_dist = (delta_z, delta_x)

            if self.dimension == 3:  # 3D
                delta_y = np.minimum(np.abs(sources_loc[:, 2]),
                                     np.abs(sources_loc[:, 2]
                                            - self.mesh_parameters.length_y))
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
        self.abc_pad_length = self.layer_infinite_model()

        inf_str = "Infinite Domain Extension (km): {:.4f}"
        print(inf_str.format(self.abc_pad_length), flush=True)

        # New dimensions
        self.abc_new_geometry()

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
            self.check_timestep_abc(
                max_divisor_tf=max_divisor_tf, method=method, mag_add=mag_add)

        print("\nBuilding Infinite Domain Model", flush=True)

        # Defining geometry for infinite domain
        self.geometry_infinite_model()

        # Creating mesh for infinite domain
        self.create_mesh_with_layer(inf_model=True)

        # Updating velocity model
        self.velocity_abc(inf_model=True)

        # Setting no damping
        self.cosHig = fire.Constant(0.)
        if self._abc_boundary_layer_type == "hybrid":
            self.eta_mask = fire.Constant(0.)
            self.eta_habc = fire.Constant(0.)

        elif self._abc_boundary_layer_type == "PML":
            self.sigma_mask = fire.Constant(0.)
            self.sigma_z = fire.Constant(0.)
            self.sigma_x = fire.Constant(0.)
            if self.dimension == 3:
                self.sigma_y = fire.Constant(0.)

        print("\nSolving Infinite Model", flush=True)

        # Solving the forward problem
        self.forward_solve()

        # Saving reference signal
        self.save_reference_signal()

        # Deleting variables to be computed for the ABC scheme
        del self.Lx_abc, self.Lz_abc
        if self.dimension == 3:
            del self.Ly_abc
        if self._abc_boundary_layer_type == "hybrid":
            del self.cosHig, self.eta_mask, self.eta_habc
        elif self._abc_boundary_layer_type == "PML":
            del self.sigma_mask, self.sigma_z, self.sigma_x
            if self.dimension == 3:
                del self.sigma_y
