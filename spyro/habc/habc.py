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

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class HABC_Wave(AcousticWave, HABC_Mesh, RectangLayer,
                HyperLayer, HABC_Damping, NRBC, HABC_Error):
    '''
    Class HABC that determines absorbing layer size and parameters to be used

    Attributes
    ----------
    abc_reference_freq : `str`
        Reference frequency for sizing the hybrid absorbing layer.
        Options: 'source' or 'boundary'
    a_par : `float`
        Adimensional propagation speed parameter (a = z / f).
        "z" parameter is the inverse of the minimum Eikonal (1 / phi_min)
    case_habc : `str`
        Label for the output files that includes the layer shape
        ('REC' or 'HNI', I for the degree) and the reference frequency
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
    eta_habc : `firedrake function`
        Damping profile within the absorbing layer
    eta_mask : `firedrake function`
        Mask function to identify the absorbing layer domain
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
    path_case_habc : `string`
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
    create_mesh_habc()
        Create a mesh with absorbing layer based on the determined size
    critical_boundary_points()
        Determine the critical points on domain boundaries of the original
        model to size an absorbing layer using the Eikonal criterion for HABCs
    damping_layer()
        Set the damping profile within the absorbing layer
    det_reference_freq()
        Determine the reference frequency for a new layer size
    fundamental_frequency()
        Compute the fundamental frequency in Hz via modal analysis
    geometry_infinite_model()
        Determine the geometry for the infinite domain model.
    habc_domain_dimensions()
        Determine the new dimensions of the domain with absorbing layer
    habc_new_geometry
        Determine the new domain geometry with the absorbing layer
    identify_habc_case()
        Generate an identifier for the current case study of the HABC scheme
    infinite_model()
        Create a reference model for the HABC scheme for comparative purposes
    layer_infinite_model()
        Determine the domain extension size for the infinite domain model
    nrbc_on_boundary_layer()
        Apply the Higdon ABCs on the outer boundary of the absorbing layer
    size_habc_criterion()
        Determine the size of the absorbing layer using the Eikonal criterion
    velocity_habc()
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
        dom_dim = self.habc_domain_dimensions(only_orig_dom=True)

        # Initializing the Mesh class
        HABC_Mesh.__init__(
            self, dom_dim, dimension=self.dimension, comm=self.comm)

        # Identifier for the current case study
        self.identify_habc_case(output_folder=output_folder)

        # Current iteration
        self.fwi_iter = fwi_iter

    def identify_habc_case(self, output_folder=None):
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
        dom_dim = self.habc_domain_dimensions(only_orig_dom=True)

        # Layer shape
        self.layer_shape = self.abc_boundary_layer_shape
        lay_str = f"\nAbsorbing Layer Shape: {self.layer_shape.capitalize()}"

        # Labeling for the layer shape
        if self.layer_shape == 'rectangular':  # Rectangular layer

            # Initialing the rectangular layer
            RectangLayer.__init__(self, dom_dim, dimension=self.dimension)
            self.case_habc = 'REC'  # Label

        elif self.layer_shape == 'hypershape':  # Hypershape layer

            # Initializing the hyperelliptical layer
            HyperLayer.__init__(self, dom_dim, n_hyp=self.abc_deg_layer,
                                n_type=self.abc_degree_type,
                                dimension=self.dimension)

            self.case_habc = 'HN' + f"{self.abc_deg_layer:.1f}"  # Label
            deg_str = f" - Degree: {self.abc_deg_layer}"
            lay_str += deg_str

        else:
            value_parameter_error('layer_shape', self.layer_shape,
                                  ['rectangular', 'hypershape'])
        print(lay_str, flush=True)

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
        if output_folder is None:
            self.path_save = getcwd() + "/output/"
        else:
            self.path_save = getcwd() + "/" + output_folder + "/"

        self.path_case_habc = self.path_save + self.case_habc + "/"

        # Initializing the error measure class
        HABC_Error.__init__(self, self.dt, self.f_Nyq,
                            self.receiver_locations,
                            output_folder=self.path_save,
                            output_case=self.path_case_habc)

        # Initializing the NRBC class
        NRBC.__init__(self, dom_dim, self.layer_shape,
                      dimension=self.dimension,
                      output_folder=self.path_case_habc)

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

    def habc_new_geometry(self):
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

    def habc_domain_dimensions(self, only_orig_dom=False,
                               only_habc_dom=False, full_hyp=True):
        '''
        Determine the new dimensions of the domain with absorbing layer

        Parameters
        ----------
        only_orig_dom : `bool`, optional
            Return only the original domain dimensions. Default is False
        only_habc_dom : `bool`, optional
            Return only the domain dimensions with layer. Default is False
        full_hyp : `bool`, optional
            Option to get the domain dimensions in hypershape layers.
            If True, the domain dimensions with layer do not include truncation
            due to the free surface. If False, the domain dimensions with layer
            include truncation by free surface. Default is True.

        Returns
        -------
        dom_dim : `tuple`
            Original domain dimensions: (Lx, Lz) for 2D or (Lx, Lz, Ly) for 3D
        dom_lay : `tuple`
            Domain dimensions with layer. For rectangular layers, truncation
            due to the free surface is included (n = 1). For hypershape layers,
            truncation by free surface is not included (n = 2) if 'full_hyp' is
            True; otherwise, it is included (n = 1). Dimensions are defined as:
            - 2D : (Lx + 2 * pad_len, Lz + n * pad_len)
            - 3D : (Lx + 2 * pad_len, Lz + n * pad_len, Ly + 2 * pad_len)
        '''

        # Original domain dimensions
        dom_dim = (self.length_x, self.length_z)
        if self.dimension == 3:  # 3D
            dom_dim += (self.length_y,)

        if only_orig_dom:
            return dom_dim

        # Domain dimension with layer w/ or w/o truncations
        if self.layer_shape == 'rectangular':  # Rectangular layer
            dom_lay = (self.Lx_habc, self.Lz_habc)

        elif self.layer_shape == 'hypershape':  # Hypershape layer
            dom_lay = (self.Lx_habc, self.length_z + 2 * self.pad_len) \
                if full_hyp else (self.Lx_habc, self.Lz_habc)

        if self.dimension == 3:  # 3D
            dom_lay += (self.Ly_habc,)

        if only_habc_dom:
            return dom_lay

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
                                 output_folder=self.path_case_habc)

        print("\nDetermining New Geometry with Absorbing Layer", flush=True)

        # New geometry with layer
        self.habc_new_geometry()

        # Domain dimensions without free surface truncation
        dom_lay_full = self.habc_domain_dimensions(only_habc_dom=True)

        if self.layer_shape == 'rectangular':

            print("Determining Rectangular Layer Parameters", flush=True)

            # Geometric properties of the rectangular layer
            self.calc_rec_geom_prop(dom_lay_full, self.pad_len)

        elif self.layer_shape == 'hypershape':

            print("Determining Hypershape Layer Parameters", flush=True)

            # Geometric properties of the hypershape layer
            self.calc_hyp_geom_prop(dom_lay_full, self.pad_len, self.lmin)

        # Domain dimensions with free surface truncation
        dom_lay_trunc = self.habc_domain_dimensions(only_habc_dom=True,
                                                    full_hyp=False)

        # Layer parameters
        layer_par = (self.F_L, self.a_par, self.d)

        # mesh parameters
        mesh_par = (self.lmin, self.lmax, self.alpha, self.variant)

        # wave parameters
        c_ref = min([bnd[1] for bnd in self.eik_bnd])
        c_bnd = self.eik_bnd[0][1]
        wave_par = (self.freq_ref, c_ref, c_bnd)

        # Initializing the parent class for damping
        HABC_Damping.__init__(self, dom_lay_trunc, layer_par, mesh_par,
                              wave_par, dimension=self.dimension)

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
            print("\nGenerating Mesh for Infinite Model", flush=True)
            layer_shape = 'rectangular'

        else:
            print("\nGenerating Mesh with Absorbing Layer", flush=True)
            layer_shape = self.layer_shape

        # New mesh with layer
        if layer_shape == 'rectangular':
            dom_lay = self.habc_domain_dimensions(only_habc_dom=True)
            mesh_habc = self.rectangular_mesh_habc(dom_lay, self.pad_len)

        elif layer_shape == 'hypershape':

            # Parameters for hypershape mesh
            if self.dimension == 2:  # 2D
                par_geom = self.perim_hyp

            # if self.dimension == 3:  # 3D
            #     par_geom = self.surf_hyp

                hyp_par = (self.n_hyp, par_geom, *self.hyper_axes)
                mesh_habc = self.hypershape_mesh_habc(hyp_par,
                                                      spln=spln,
                                                      fmesh=fmesh)

            # Mesh file
            if self.dimension == 3:  # 3D
                q = {"overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)}
                mesh_habc = fire.Mesh(self.filename_mesh,
                                      distribution_parameters=q,
                                      comm=self.comm.comm)

            # Mesh data
            print(f"Mesh Created with {mesh_habc.num_vertices()} Nodes "
                  f"and {mesh_habc.num_cells()} Volume Elements", flush=True)

            # Adjusting coordinates: Swap (x, y, z) -> (z, x ,y)
            if self.dimension == 3:  # 3D
                mesh_habc.coordinates.dat.data_with_halos[:, [0, 1, 2]] = \
                    mesh_habc.coordinates.dat.data_with_halos[:, [2, 0, 1]]

        # Updating the mesh with the absorbing layer
        self.set_mesh(user_mesh=mesh_habc, mesh_parameters={})
        print("Mesh Generated Successfully", flush=True)

        if inf_model:
            pth_mesh = self.path_save + "preamble/mesh_inf.pvd"
        else:
            pth_mesh = self.path_case_habc + "mesh_habc.pvd"

        # Save new mesh
        outfile = fire.VTKFile(pth_mesh)
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
            file_name = self.case_habc + "/c_habc.pvd"

        outfile = fire.VTKFile(self.path_save + file_name)
        outfile.write(self.c)

    def fundamental_frequency(self, method=None, monitor=False,
                              fitting_c=(1., 1., 0.5, 0.5)):
        '''
        Compute the fundamental frequency in Hz via modal analysis
        considering the numerical model with Neumann BCs.

        Parameters
        ----------
        method : `str`, optional
            Method to use for solving the eigenvalue problem.
            Default is None, which uses the 'KRYLOVSCH_CH' method.
            Opts: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS', 'LOBPCG', 'KRYLOVSCH_CH',
            'KRYLOVSCH_CG', 'KRYLOVSCH_GH', 'KRYLOVSCH_GG' or 'RAYLEIGH'.
            'ANALYTICAL' method is only available for isotropic hypershapes.
            'RAYLEIGH' method is an approximation by Rayleigh quotient.
            In 'KRYLOVSCH_(K)(P)' methods, (K) indicates the Krylov solver to
            use: 'C' for Conjugate Gradient (cg) or 'G' for Generalized Minimal
            Residual (gmres). (P) indicates the preconditioner to use: 'H' for
            Hypre (hypre) or 'G' for Geometric Algebraic Multigrid (gamg). For
            example, 'KRYLOVSCH_CH' uses cg solver with hypre preconditioner.
        monitor : `bool`, optional
            Print on screen the computed natural frequencies. Default is False
        fitting_c : `tuple`, optional
            Parameters for fitting equivalent velocity regression.
            Structure: (fc1, fc2, fp1, fp2). Default is (1., 1., 0.5, 0.5)
            - fc1 : `float`
                Exponent factor for the minimum reference velocity
            - fc2 : `float`
                Exponent factor for the maximum reference velocity
            - fp1 : `float`
                Exponent fsctor for the minimum equivalent velocity
            - fp2 : `float`
                Exponent factor for the maximum equivalent velocity

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

        print("\nSolving Eigenvalue Problem", flush=True)
        mod_sol = eigsol.Modal_Solver(self.dimension, method=method)

        if method == 'ANALYTICAL':

            # Hypershape parameters
            hyp_par = (self.n_hyp, *self.hyper_axes)

            # Cut plane at free surface
            Lz = self.dom_dim[1]
            z_cut = Lz / 2.

            # Cut plane percentage
            cut_plane_perc = z_cut / self.hyper_axes[1]

            # Equivalent velocity for the original model
            c_eqref = mod_sol.c_equivalent(self.initial_velocity_model,
                                           V=self.funct_space_eik)

            Lsp = mod_sol.solve_eigenproblem(self.c, V=self.function_space,
                                             quad_rule=self.quadrature_rule,
                                             hyp_par=hyp_par, c_eqref=c_eqref,
                                             fitting_c=fitting_c,
                                             cut_plane_percent=cut_plane_perc)

        elif method == 'RAYLEIGH':

            # Normalized coordinates
            coord_norm = mod_sol.generate_norm_coords(self.mesh,
                                                      self.dom_dim,
                                                      self.hyper_axes)

            Lsp = mod_sol.solve_eigenproblem(self.c, V=self.function_space,
                                             quad_rule=self.quadrature_rule,
                                             coord_norm=coord_norm)

        else:
            Lsp = mod_sol.solve_eigenproblem(self.c,
                                             V=self.function_space, shift=1e-8,
                                             quad_rule=self.quadrature_rule)

        if monitor:
            for n_eig, eigval in enumerate(np.unique(Lsp)):
                f_eig = np.sqrt(abs(eigval)) / (2 * np.pi)
                print(f"Frequency {n_eig} (Hz): {f_eig:.5f}", flush=True)

        # Fundamental frequency (eig = 0 is a rigid body motion)
        min_eigval = max(np.unique(Lsp[(Lsp > 0.) & (np.imag(Lsp) == 0.)]))
        self.fundam_freq = np.real(np.sqrt(min_eigval) / (2 * np.pi))
        print("Fundamental Frequency (Hz): {0:.5f}".format(self.fundam_freq),
              flush=True)

    def damping_layer(self, xCR_usu=None, method=None,
                      fitting_c=(1., 1., 0.5, 0.5)):
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
            Default is None, which uses the 'KRYLOVSCH_CH' method.
            Opts: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS', 'LOBPCG', 'KRYLOVSCH_CH',
            'KRYLOVSCH_CG', 'KRYLOVSCH_GH', 'KRYLOVSCH_GG' or 'RAYLEIGH'.
            'ANALYTICAL' method is only available for isotropic hypershapes.
            'RAYLEIGH' method is an approximation by Rayleigh quotient.
            In 'KRYLOVSCH_(K)(P)' methods, (K) indicates the Krylov solver to
            use: 'C' for Conjugate Gradient (cg) or 'G' for Generalized Minimal
            Residual (gmres). (P) indicates the preconditioner to use: 'H' for
            Hypre (hypre) or 'G' for Geometric Algebraic Multigrid (gamg). For
            example, 'KRYLOVSCH_CH' uses cg solver with hypre preconditioner.
        fitting_c : `tuple`, optional
            Parameters for fitting equivalent velocity regression.
            Structure: (fc1, fc2, fp1, fp2). Default is (1., 1., 0.5, 0.5)
            - fc1 : `float`
                Exponent factor for the minimum reference velocity
            - fc2 : `float`
                Exponent factor for the maximum reference velocity
            - fp1 : `float`
                Exponent fsctor for the minimum equivalent velocity
            - fp2 : `float`
                Exponent factor for the maximum equivalent velocity

        Returns
        -------
        None
        '''

        # Estimating fundamental frequency
        self.fundamental_frequency(method=method, monitor=True,
                                   fitting_c=fitting_c)

        print("\nCreating Damping Profile", flush=True)

        # Compute the minimum damping ratio and the associated heuristic factor
        eta_crt, self.psi_min, self.xCR, self.xCR_lim, self.CRmin\
            = self.calc_damping_prop(self.fundam_freq, xCR_usu=xCR_usu)

        # Mesh coordinates
        coords = fire.SpatialCoordinate(self.mesh)

        # Damping mask
        V_mask = fire.FunctionSpace(self.mesh, 'DG', 0)
        self.eta_mask = self.layer_mask_field(coords, V_mask,
                                              type_marker='mask',
                                              name_mask='eta_mask')

        # Save damping mask
        outfile = fire.VTKFile(self.path_case_habc + "eta_mask.pvd")
        outfile.write(self.eta_mask)

        # Compute the coefficients for quadratic damping function
        aq, bq = self.coeff_damp_fun(self.psi_min)

        # Damping field
        damp_par = (self.pad_len, eta_crt, aq, bq)
        self.eta_habc = self.layer_mask_field(
            coords, self.function_space, damp_par=damp_par,
            type_marker='damping', name_mask='eta [1/s])')

        # Save damping profile
        outfile = fire.VTKFile(self.path_case_habc + "eta_habc.pvd")
        outfile.write(self.eta_habc)

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

    def infinite_model(self, check_dt=False, max_divisor_tf=1, mag_add=3):
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
        mag_add : `int`, optional
            Additional magnitude order to adjust the rounding of the timestep


        Returns
        -------
        None
        '''

        # Check the timestep size
        if check_dt:
            self.check_timestep_habc(
                max_divisor_tf=max_divisor_tf, mag_add=mag_add)

        print("\nBuilding Infinite Domain Model", flush=True)

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

        print("\nSolving Infinite Model", flush=True)

        # Solving the forward problem
        self.forward_solve()

        # Saving reference signal
        self.save_reference_signal()

        # Deleting variables to be computed for the HABC scheme
        del self.pad_len, self.Lx_habc, self.Lz_habc
        del self.cosHig, self.eta_mask, self.eta_habc
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
            old = self.path_case_habc
            new = self.path_case_habc[:-8] + f"{self.n_hyp:.1f}" + \
                self.path_case_habc[-5:]

            try:
                if path.isdir(new):
                    # Remove target directory if it exists
                    rmtree(new)

                rename(old, new)  # Rename the directory
                print(f"Folder '{old}' Successfully Renamed to '{new}'.", flush=True)

            except OSError as e:
                print(f"Error Renaming Folder: {e}", flush=True)
