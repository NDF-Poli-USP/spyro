# from firedrake import Constant, Function, VTKFile
from firedrake import Function, VTKFile
# from numpy import abs, array, ceil, inf, log10, minimum
from numpy import inf
from os import getcwd
# from sympy import divisors
from .nrbc import NRBC
from .eik_min import Minimum_Eikonal
from .lay_len import calc_size_lay
# from ..habc.error_measure import HABCError
# from ..solvers.modal.modal_sol import Modal_Solver
from ..io.basicio import parallel_print as pprint
from ..domains.space import create_function_space
from ..plots.plots_habc import plot_function_layer_size
from ..tools.habc_tools import clipping_coordinates_lay_field, extend_scalar_field_profile
from ..utils.error_management import (enum_parameter_error, value_numerical_error,
                                      value_parameter_error)
from ..utils.freq_tools import freq_response
from ..utils.typing import (BoundaryConditionsType, HyperLayerDegreeType,
                            LayerDampingType, LayerShapeType, LayerSizeRefFrequency)

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender
# TODO: Add reference


class ABCLayer(NRBC):
    """Class for ABCs based on absorbing layers.

    Attributes
    ----------
    abc_boundary_layer_shape : `typing.LayerShapeType`, optional
        Shape type of the pad layer. Options: `LayerShapeType.RECTANGULAR` or
        `LayerShapeType.HYPERSHAPE`. Default is `LayerShapeType.RECTANGULAR`.
    abc_boundary_layer_type : `typing.LayerDampingType`
        Type of the boundary layer. Options: `LayerDampingType.LOCAL`,
        `LayerDampingType.HYBRID`, `LayerDampingType.PML` or `LayerDampingType.NOABCS`.
        Default is `LayerDampingType.NOABCS` where no absorbing BCs are applied.
        Option `LayerDampingType.HYBRID` is based on paper of Salas et al. (2022).
        doi: https://doi.org/10.1016/j.apm.2022.09.014
        TODO: Add citation
    abc_pad_length : `float`
        Size of the absorbing layer
    abc_reference_freq : `typing.LayerSizeRefFrequency`, optional
        Reference frequency for sizing the absorbing layer.
        Options: 'LayerSizeRefFrequency.SOURCE' or 'LayerSizeRefFrequency.BOUNDARY'.
        Default is 'LayerSizeRefFrequency.SOURCE'.
    a_par : `float`
        Adimensional propagation speed parameter (a = z / f).
        "z" parameter is the inverse of the minimum Eikonal (1 / phi_min)
    case_abc : `str`
        Label for the output files that includes the layer shape
        ("REC", HNX.X) and the reference frequency
        ('SOU' or 'BND'). Example: "REC_SOU" or "REC_BND"
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
    frequency: `float`
        Frequency of the source.
    freq_Nyquist : `float`
        Nyquist frequency according to the time step. freq_Nyquist = 1 / (2 * dt)
    freq_ref : `float`
        Reference frequency of the wave at the boundary
    fwi_iter : `int`
        The iteration number for the Full Waveform Inversion (FWI) algorithm
    lmin : `float`
        Minimum mesh size
    length_xabc : `float`
        Length of the domain in the x-direction with absorbing layer
    length_yabc : `float`
        Length of the domain in the y-direction with absorbing layer (3D)
    length_zabc : `float`
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
    """

    def __init__(self, domain_dim, frequency, freq_Nyquist, dimension=2,
                 quadrilateral=False, func_space_type=None,
                 abc_boundary_layer_shape=LayerShapeType.RECTANGULAR,
                 abc_boundary_layer_type=LayerDampingType.HYBRID,
                 abc_reference_freq=LayerSizeRefFrequency.SOURCE,
                 abc_degree_type=HyperLayerDegreeType.REAL, abc_deg_layer=None,
                 output_folder=None, comm=None):
        """Initialize the ABCLayer class.

        Parameters
        ----------
        domain_dim : `tuple`
            Original domain dimensions: (length_z, length_x) for 2D
            or (length_z, length_x, length_y) for 3D.
        frequency: `float`
            Frequency of the source.
        freq_Nyquist : `float`
            Nyquist frequency according to the time step. freq_Nyquist = 1 / (2 * dt)
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D.
        quadrilateral : bool, optional
            Flag to indicate whether to use quadrilateral/hexahedral elements.
            Default is `False` (triangular/tetrahedral elements).
        func_space_type, `str`, optional
            Type of function space for the state variable.
            Options: 'scalar' or 'vector'. Default is None.
        abc_boundary_layer_shape : `typing.LayerShapeType`, optional
            Shape type of the pad layer. Options: `LayerShapeType.RECTANGULAR` or
            `LayerShapeType.HYPERSHAPE`. Default is `LayerShapeType.RECTANGULAR`.
        abc_boundary_layer_type : `str`, optional
            Type of the boundary layer. Options: 'hybrid' or 'PML'.
            Default is 'hybrid'. Option 'hybrid' is based on paper of Salas et al. (2022).
            doi: https://doi.org/10.1016/j.apm.2022.09.014
        abc_reference_freq : `typing.LayerSizeRefFrequency`, optional
            Reference frequency for sizing the absorbing layer.
            Options: 'LayerSizeRefFrequency.SOURCE' or 'LayerSizeRefFrequency.BOUNDARY'.
            Default is 'LayerSizeRefFrequency.SOURCE'.
        abc_degree_type : `typing.HyperLayerDegreeType`, optional
            Type of the hypereshape degree. Options: 'HyperLayerDegreeType.REAL' or
            'HyperLayerDegreeType.INTEGER'. Default is 'HyperLayerDegreeType.REAL'.
        abc_deg_layer : `int` or `float` or `None`, optional
            Hypershape degree. For hypershape layers, the degree must be greater than or
            equal to 2. `None` is used only for rectangular layers. Default is `None`.
        output_folder : `str`, optional
            The folder where output data will be saved. Default is None.
        comm : `object`, optional
            An object representing the communication interface for parallel processing.
            Default is `None`.

        Returns
        -------
        None
        """

        # Validate input arguments
        if not isinstance(domain_dim, tuple):
            raise TypeError("domain_dim must be a tuple, "
                            f"got {type(domain_dim).__name__}.")

        if output_folder is not None and not isinstance(output_folder, str):
            raise TypeError("output_folder must be a string, "
                            f"got {type(output_folder).__name__}.")

        # Original domain dimensions
        self.domain_dim = domain_dim

        # Source frequency
        self.frequency = value_numerical_error('frequency', frequency, float_num=True,
                                               integer_num=True, lower_bound=0.)

        # Nyquist frequency
        self.freq_Nyquist = value_numerical_error('freq_Nyquist', freq_Nyquist,
                                                  float_num=True, integer_num=True,
                                                  lower_bound=0.)

        # Model dimension
        self.dimension = value_parameter_error('dimension', dimension, [2, 3])

        # Quadrilateral/hexahedral elements
        self.quadrilateral = quadrilateral

        # Type of function space
        self.func_space_type = func_space_type

        # ABC layer parameters
        self.abc_boundary_layer_type = enum_parameter_error("abc_boundary_layer_type",
                                                            abc_boundary_layer_type,
                                                            LayerDampingType)
        if abc_boundary_layer_type == LayerDampingType.NOABCS:
            value_parameter_error('abc_boundary_layer_type', abc_boundary_layer_type,
                                  [LayerDampingType.HYBRID, LayerDampingType.PML])

        self.abc_boundary_layer_shape = enum_parameter_error('abc_boundary_layer_shape',
                                                             abc_boundary_layer_shape,
                                                             LayerShapeType)
        self.abc_reference_freq = enum_parameter_error('abc_reference_freq',
                                                       abc_reference_freq,
                                                       LayerSizeRefFrequency)
        self.abc_degree_type = enum_parameter_error('abc_degree_type', abc_degree_type,
                                                    HyperLayerDegreeType)

        # Layer degree
        if self.abc_boundary_layer_shape == LayerShapeType.RECTANGULAR:
            self.abc_deg_layer = None
        elif self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE:
            self.abc_deg_layer = value_numerical_error('abc_deg_layer', abc_deg_layer,
                                                       float_num=True, integer_num=True,
                                                       lower_bound=2.,
                                                       include_lower_bound=True)

        # Communicator MPI
        self.comm = comm

        # Define the shape of the absorbing layer.
        self.layer_geometry = self._define_layer_shape()

        # Create the path to save data
        self.path_to_save_abc_layer_case(output_folder=output_folder)

        # Initializing the NRBC class
        NRBC.__init__(self, self.domain_dim,
                      self.abc_boundary_layer_shape,
                      dimension=self.dimension,
                      output_folder=self.path_case_abc,
                      comm=self.comm)

        # # Initializing the error measure class
        # HABCError.__init__(self, self.dt, self.freq_Nyquist,
        #                    self.receiver_locations,
        #                    output_folder=self.path_save,
        #                    output_case=self.path_case_abc)

    def _define_layer_shape(self):
        """Define the shape of the absorbing layer.

        Parameters
        ----------
        None

        Returns
        -------
        layer_geometry : `rec_lay.RectangLayer` or `hyp_lay.HyperLayer`
            An object representing the geometry of the absorbing layer.
            Options: `rec_lay.RectangLayer` for rectangular layers or `hyp_lay.HyperLayer`
            for hypershape layers.
        """

        # Initializating the layer
        if self.abc_boundary_layer_shape == LayerShapeType.RECTANGULAR:

            # Initializing the rectangular layer
            from .rec_lay import RectangLayer
            Rectangle_layer = RectangLayer(self.domain_dim, dimension=self.dimension)

            return Rectangle_layer

        elif self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE:

            # Initializing the hyperelliptical layer
            from .hyp_lay import HyperLayer
            Hypershape_layer = HyperLayer(self.domain_dim, n_hyp=self.abc_deg_layer,
                                          n_type=self.abc_degree_type,
                                          dimension=self.dimension)
            return Hypershape_layer

    def formatting_abc_layer_type(self, str_to_format, for_prints=True):
        """Format a string for the ABC layer type.

        The formatted string can be used for printing on screen or to generate paths for
        output files. The `for_prints` parameter determines whether the formatted string
        is intended for printing or for labeling purposes.

        Parameters
        ----------
        str_to_format : `str`
            The string to format.
        for_prints : `bool`, optional
            Flag to indicate whether the formatted string is for
            printing (`True`) or for labeling (`False`). Default is `True`.

        Returns
        -------
        formatted_str : `str`
            The formatted string for the ABC layer type.
        """

        # Layer type
        if self.abc_boundary_layer_type == LayerDampingType.HYBRID:
            abc_layer_str = "Absorbing" if for_prints else "habc"
        elif self.abc_boundary_layer_type == LayerDampingType.PML:
            abc_layer_str = "PML" if for_prints else "pml"

        formatted_str = str_to_format.format(abc_layer_str)

        return formatted_str

    def identify_abc_layer_case(self):
        """Generate an identifier for the current layer geometry of the ABC.

        The identifier includes the layer shape ("REC" for rectangular layers or "HN"
        followed by the degree for hypershape layers) and the reference frequency for
        sizing the absorbing layer ('SOU': source frequency or 'BND': boundary frequency).
        Examples: "REC_SOU", "REC_BND", "HN2.4_SOU" or "HN2.4_BND".

        Parameters
        ----------
        None

        Returns
        -------
        case_abc : `str`
            Label for the output files that includes the layer shape and degree for
            hypershape layers ("REC", "HNX.Y" with X.Y as the hypershape degree with one
            decimal place precision) and the reference frequency ('SOU' or 'BND').
            Examples: "REC_SOU", "REC_BND", "HN2.4_SOU" or "HN2.4_BND".
        """

        # Labeling for the layer shape
        if self.abc_boundary_layer_shape == LayerShapeType.RECTANGULAR:
            case_abc = "REC"

        elif self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE:
            case_abc = "HN" + f"{self.abc_deg_layer:.1f}"

        # Labeling for the reference frequency for the absorbing layer
        if self.abc_reference_freq == LayerSizeRefFrequency.SOURCE:
            case_abc += "_SOU"
        elif self.abc_reference_freq == LayerSizeRefFrequency.BOUNDARY:
            case_abc += "_BND"

        # Printing layer info on screen
        layer_str = self.formatting_abc_layer_type("\n{} Layer Shape: ") + \
            f"{self.abc_boundary_layer_shape.value.capitalize()}" + \
            (f" - Degree: {self.abc_deg_layer}"
             if self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE else "")
        pprint(layer_str, comm=self.comm)

        return case_abc

    def path_to_save_abc_layer_case(self, output_folder=None):
        """Create the path to save data for the current case study of the ABC scheme.

        Parameters
        ----------
        output_folder : `str`, optional
            The folder where output data will be saved. Default is `None`.

        Returns
        -------
        None
        """

        # Identify the case of the ABC scheme for output labeling
        self.case_abc = self.identify_abc_layer_case()

        # Path to save data
        if output_folder is None:
            path_save = getcwd() + "/output/"
        else:
            path_save = getcwd() + "/" + output_folder + "/"

        path_case_abc = path_save + self.case_abc + "/"

        self.path_save = path_save
        self.path_case_abc = path_case_abc

    def critical_boundary_points(self, Wave):
        """Determine critical boundary points using the Eikonal criterion.

        Use original-domain boundaries to size the absorbing layer.
        See Salas et al (2022): Hybrid absorbing scheme based on hyperelliptical
        layers with non-reflecting boundary conditions in scalar wave equations.
        doi: https://doi.org/10.1016/j.apm.2022.09.014
        TODO: Add citation

        Parameters
        ----------
        Wave : `wave.Wave`
            An instance of the :class:`~spyro.solvers.wave.Wave`.

        Returns
        -------
        None
        """

        # Initializing Eikonal object
        Eik = Minimum_Eikonal(Wave)

        # Solving Eikonal
        Eik.solve_eik()

        # Identifying critical points
        self.eik_bnd = Eik.ident_crit_eik()

        # Critical point coordinates as receivers
        pcrit = [bnd[0] for bnd in self.eik_bnd]
        Wave.receiver_locations = pcrit + Wave.receiver_locations
        Wave.number_of_receivers = len(Wave.receiver_locations)

    def det_reference_freq(self, fpad=4):
        """Determine the reference frequency for a new layer size.

        Parameters
        ----------
        fpad : `int`, optional
            Padding factor for FFT. Default is 4.

        Returns
        -------
        None
        """

        pprint("\nDetermining Reference Frequency", comm=self.comm)

        if self.abc_reference_freq == LayerSizeRefFrequency.SOURCE:

            # Theoretical central Ricker source frequency (it can be a initial guess)
            self.freq_ref = self.frequency

        elif self.abc_reference_freq == LayerSizeRefFrequency.BOUNDARY:

            # Reference frequency of the wave at the boundary
            self.freq_ref = inf

            for n_crit in range(self.number_of_receivers):

                # Transient response at each critical Eikonal point
                histPcrit = self.receivers_reference[:, n_crit]

                # Get the minimum frequency excited at each critical point
                freq_ref = freq_response(histPcrit, self.freq_Nyquist,
                                         fpad=fpad, get_dominant_freq=True)
                pprint(f"Frequency at Critical Point {n_crit:>2.0f}: {freq_ref:.5f}",
                       comm=self.comm)

                self.freq_ref = min(self.freq_ref, freq_ref)

        pprint(f"Reference Frequency (Hz): {self.freq_ref:.5f}", comm=self.comm)

    def abc_new_geometry(self):
        """Determine the new domain geometry with the absorbing layer.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Original domain dimensions
        length_z, length_x = self.domain_dim[:2]

        # New geometry with layer
        self.length_xabc = length_x + 2 * self.abc_pad_length
        self.length_zabc = length_z + self.abc_pad_length

        if self.dimension == 3:  # 3D
            length_y = self.domain_dim[2]
            self.length_yabc = length_y + 2 * self.abc_pad_length

    def abc_domain_dimensions(self, full_hyp=True):
        """Return the new dimensions of the domain with absorbing layer as a tuple.

        Parameters
        ----------
        full_hyp : `bool`, optional
            Option to get the domain dimensions in hypershape layers.
            If `True`, the domain dimensions with layer do not include truncation
            due to the free surface. If `False`, the domain dimensions with layer
            include truncation by free surface. Default is `True`.

        Returns
        -------
        domain_layer : `tuple`
            Domain dimensions with layer. For rectangular layers, truncation
            due to the free surface is included (n = 1). For hypershape layers,
            truncation by free surface is not included (n = 2) if 'full_hyp' is
            `True`; otherwise, it is included (n = 1). (See Notes below)

        Notes
        -----
        Model dimensions are defined as:
            2D: (length_z + n * pad_len, length_x + 2 * pad_len).
            3D: (length_z + n * pad_len, length_x + 2 * pad_len, length_y + 2 * pad_len).
        """

        # Domain dimensions with layer and truncations
        domain_layer = [self.length_zabc, self.length_xabc]

        # Domain dimensions with layer without truncations only for hypershape layers
        if self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE and full_hyp:
            domain_layer[0] += self.abc_pad_length

        if self.dimension == 3:  # 3D
            domain_layer.append(self.length_yabc)

        return tuple(domain_layer)

    def layer_size_criterion(self, lmin, fpad=4, n_root=1, layer_based_on_mesh=True):
        """Determine the absorbing layer size using the Eikonal criterion for HABCs.

        See Salas et al (2022): Hybrid absorbing scheme based on hyperelliptical
        layers with non-reflecting boundary conditions in scalar wave equations.
        doi: https://doi.org/10.1016/j.apm.2022.09.014

        Parameters
        ----------
        lmin : `float`
            Minimum mesh size.
        fpad : `int`, optional
            Padding factor for FFT. Default is 4.
        n_root : `int`, optional
            n-th Root selected as the size of the absorbing layer. Default is 1.
        layer_based_on_mesh : `bool`, optional
            Adjust the layer size based on the element size. Default is `True`.

        Returns
        -------
        None
        """

        # Determining the reference frequency
        self.det_reference_freq(fpad=fpad)

        # Minimum mesh size
        self.lmin = lmin

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
        pprint(lay_str, comm=self.comm)

        # New geometry with layer
        self.abc_new_geometry()

        # Domain dimensions without free surface truncation
        domain_layer_full = self.abc_domain_dimensions()

        if self.abc_boundary_layer_shape == LayerShapeType.RECTANGULAR:

            # Geometric properties of the rectangular layer
            self.layer_geometry.calc_rec_geom_prop(
                domain_layer_full, self.abc_pad_length)

        elif self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE:

            # Geometric properties of the hypershape layer
            self.layer_geometry.calc_hyp_geom_prop(
                domain_layer_full, self.abc_pad_length, self.lmin)

    def create_mesh_with_layer(self, Wave, inf_model=False, spln=True, save_file=True):
        """Create a mesh with absorbing layer based on the determined size.

        Parameters
        ----------
        Wave : `wave.Wave`
            An instance of the :class:`~spyro.solvers.wave.Wave`.
        inf_model : `bool`, optional
            If `True`, build a rectangular layer for the infinite or reference.
            model (Model with "infinite" dimensions). Default is `False`.
        spln : `bool`, optional
            Flag to indicate whether to use splines (`True`) or lines (`False`).
            in hypershape layer generation. Default is `True`.
        save_file : `bool`, optional
            If `True`, save the mesh with absorbing layer in a .pvd file.
            Default is `True`.

        Returns
        -------
        None
        """

        # Checking if the mesh for infinite model is requested
        if inf_model:
            pprint("\nGenerating Mesh for Infinite Model", comm=self.comm)
            layer_shape = LayerShapeType.RECTANGULAR

        else:
            pprint("\nGenerating Mesh with Absorbing Layer", comm=self.comm)
            layer_shape = self.abc_boundary_layer_shape

        # New mesh with layer
        if layer_shape == LayerShapeType.RECTANGULAR:

            # Update the pad length in Wave object
            Wave.abc_pad_length = self.abc_pad_length

            # Create the mesh
            Wave.set_mesh()
            pprint("Extended Rectangular Mesh Generated Successfully", comm=self.comm)

        elif layer_shape == LayerShapeType.HYPERSHAPE:

            # Update the pad length in Wave.mesh_parameters object
            Wave.mesh_parameters.abc_pad_length = self.abc_pad_length

            # Parameters for hypershape mesh
            if self.dimension == 2:  # 2D
                geometry_param = self.layer_geometry.perim_hyp

            if self.dimension == 3:  # 3D
                geometry_param = self.layer_geometry.surf_hyp

            hypershape_param = (
                self.layer_geometry.n_hyp, geometry_param, *self.layer_geometry.hyper_axes)

            # Creating the mesh with the absorbing layer based on the hypershape geometry
            mesh_abc = Wave.mesh_ops.hypershape_mesh_habc(
                hypershape_param, Wave.mesh_original, Wave.mesh_parameters, spln=spln)

            # Updating the mesh with the absorbing layer
            Wave.set_mesh(user_mesh=mesh_abc)

        pprint("Mesh Generated Successfully", comm=self.comm)

        if save_file:
            if inf_model:
                pth_mesh = self.path_save + "preamble/mesh_inf.pvd"
            else:
                mesh_file_name = self.formatting_abc_layer_type("mesh_{}.pvd",
                                                                for_prints=False)
                pth_mesh = self.path_case_abc + mesh_file_name

            # Save new mesh
            outfile = VTKFile(pth_mesh)
            outfile.write(Wave.mesh)

    def velocity_abc(self, Wave, inf_model=False, method="point_cloud", save_file=True):
        """Set the velocity profile for the model with absorbing layer.

        Parameters
        ----------
        Wave : `wave.Wave`
            An instance of the :class:`~spyro.solvers.wave.Wave`.
        inf_model : `bool`, optional
            If `True`, build a rectangular layer for the infinite or reference
            model (Model with "infinite" dimensions). Default is `False`.
        method : `str`, optional
            Method to extend the velocity profile. Options:
            - "point_cloud" : Interpolate the field based on a point
                              cloud from the original boundary
            - "nearest_point" : Use the nearest point on the original
                                boundary to extend the field.
            Default is "point_cloud".
        save_file : `bool`, optional
            If `True`, save the velocity model with absorbing layer in a .pvd file.
            Default is `True`.

        Returns
        -------
        None

        Notes
        -----
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

        "point_cloud" - dx = 0.05 km (2D)
        Estimating Runtime and Used Memory
        Runtime: (s):18.437, (m):0.307, (h):0.005
        Used Memory: Current (MB):18.813, Peak (MB):25.102

        "nearest_point" - dx = 0.05 km (2D)
        Estimating Runtime and Used Memory
        Runtime: (s):20.494, (m):0.342, (h):0.006
        Used Memory: Current (MB):18.715, Peak (MB):25.298
        """

        pprint("\nUpdating Velocity Profile", comm=self.comm)

        # Scalar space for auxiliar field of clipped coordinates
        method_element = "DQ" if self.quadrilateral else "DG"
        V = create_function_space(Wave.mesh, method_element, 0)

        # Initialize velocity field and assigning the original velocity model
        Wave.c = Function(V).interpolate(Wave.initial_velocity_model,
                                         allow_missing_dofs=True)

        # Clipping coordinates to the layer domain
        domain_layer = self.abc_domain_dimensions(full_hyp=False)
        ufl_coordinates_habc = Wave.mesh_ops.get_spatial_coordinates_abc(Wave.mesh,
                                                                         domain_layer)
        lay_field, layer_mask = \
            clipping_coordinates_lay_field(self.domain_dim, Wave.mesh,
                                           self.dimension, ufl_coordinates_habc,
                                           V, quadrilateral=self.quadrilateral)

        # Extending velocity model within the absorbing layer
        extended_velocity = \
            extend_scalar_field_profile(Wave.mesh_original, Wave.initial_velocity_model,
                                        lay_field, layer_mask, Wave.mesh_parameters.tol,
                                        method=method, name_prop="Velocity")

        # Interpolating the velocity model in the layer
        Wave.c.interpolate(extended_velocity * layer_mask + (1. - layer_mask)
                           * Wave.c, allow_missing_dofs=True)
        del layer_mask, lay_field

        # Interpolating in the space function of the problem
        Wave.c = Function(Wave.function_space, name="c[km/s])").interpolate(Wave.c)

        # Save new velocity model
        if save_file:
            if inf_model:
                file_name = "preamble/c_inf.pvd"
            else:
                c_file_name = self.formatting_abc_layer_type("/c_{}.pvd",
                                                             for_prints=False)
                file_name = self.case_abc + c_file_name

            outfile = VTKFile(self.path_save + file_name)
            outfile.write(Wave.c)

    def nrbc_on_boundary_layer(self, Wave_object, non_reflect_bc, save_file=True):
        """Apply Non-Reflective BCs on the outer boundary of the absorbing layer.

        Parameters
        ----------
        Wave_object : `acoustic_wave.AcousticWave`
            An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
        non_reflect_bc : `typing.BoundaryConditionsType`
            Type of boundary condition to apply on the outer absorbing layer boundaries.
            - Options for Non-Reflecting BCs:
                'BoundaryConditionsType.HIGDON' or 'BoundaryConditionsType.SOMMERFELD'.
        save_file : `bool`, optional
            If `True`, save the velocity model with absorbing layer in a .pvd file.
            Default is `True`.

        Returns
        -------
        None
        """

        # Applying NRBCs on outer boundary layer
        crit_source = bnd_nod_ids_nfs = bnd_nodes_nfs = None
        if non_reflect_bc == BoundaryConditionsType.SOMMERFELD or \
                non_reflect_bc == BoundaryConditionsType.HIGDON:

            pprint("\nApplying Non-Reflecting Boundary Conditions", comm=self.comm)

            # Getting boundary data from the layer boundaries
            if non_reflect_bc == BoundaryConditionsType.SOMMERFELD:
                bnd_nod_ids_nfs = \
                    Wave_object.mesh_ops.layer_boundary_data(Wave_object.mesh,
                                                             Wave_object.function_space,
                                                             Wave_object.mesh_parameters)[0]

            if non_reflect_bc == BoundaryConditionsType.HIGDON:
                crit_source = self.crit_source
                bnd_nod_ids_nfs, bnd_nodes_nfs = \
                    Wave_object.mesh_ops.layer_boundary_data(Wave_object.mesh,
                                                             Wave_object.function_space,
                                                             Wave_object.mesh_parameters)

            # Hypershape parameters
            hyp_par = (self.layer_geometry.n_hyp, *self.layer_geometry.hyper_axes) \
                if self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE else None

            # Applying Higdon ABCs
            self.cos_ang_HigdonBC(Wave_object.function_space, crit_source,
                                  bnd_nod_ids_nfs, bnd_nodes_nfs, non_reflect_bc,
                                  hyp_par=hyp_par, save_file=save_file)
        else:
            pprint("\nNot Non-Reflecting Boundary Conditions Prescribed", comm=self.comm)

    # def check_timestep_abc(self, max_divisor_tf=1, set_max_dt=True,
    #                        method='ANALYTICAL', mag_add=3):
    #     """
    #     Check if the timestep size is appropriate for the transient response

    #     Parameters
    #     ----------
    #     max_divisor_tf : `int`, optional
    #         Index to select the maximum divisor of the final time, converted
    #         to an integer according to the order of magnitude of the timestep
    #         size. The timestep size is set to the divisor, given by the index
    #         in descending order, less than or equal to the user's timestep
    #         size. If the value is 1, the timestep size is set as the maximum
    #         divisor. Default is 1
    #     set_max_dt : `bool`, optional
    #         If `True`, set the timestep size to the selected divisor.
    #         Default is `True`
    #     method : `str`, optional
    #         Method to use for solving the eigenvalue problem. Default
    #         is 'ANALYTICAL' method that estimates the maximum eigenvalue
    #         using the Gershgorin Circle Theorem.
    #         Opts: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS' or 'LOBPCG'
    #     mag_add : `int`, optional
    #         Additional magnitude order to adjust the rounding of the timestep

    #     Returns
    #     -------
    #     None

    #     # Estimation: 2.770 (Old), 2.768 (New) (Scipy-sparse)
    #     # Exact: 1.842 (Old), 1.842 (New) (Scipy)
    #     """

    #     pprint("\nChecking Timestep Size", comm=self.comm)

    #     # User timestep
    #     usr_dt = self.get_dt()

    #     # Maximum timestep size
    #     dt_sol = Modal_Solver(self.dimension, method=method, calc_max_dt=True)
    #     max_dt = dt_sol.estimate_timestep(self.c, self.function_space, self.final_time,
    #                                       shift=1e-8, quad_rule=self.quadrature_rule,
    #                                       fraction=1.)

    #     # Rounding power
    #     pot = int(abs(ceil(log10(max_dt))) + mag_add)

    #     # Maximum timestep size according to divisors of the final time
    #     val_int_tf = int(10**pot * self.final_time)
    #     val_int_dt = int(10**pot * max_dt)
    #     max_div = [d for d in divisors(val_int_tf) if d < val_int_dt]
    #     n_div = len(max_div)
    #     index_div = min(max_divisor_tf, n_div)
    #     max_dt = round(10**(-pot) * max_div[-index_div], pot)

    #     # Set the timestep size
    #     dt = max_dt if set_max_dt else min(usr_dt, max_dt)
    #     self.set_dt(dt)
    #     dt_ms = 1e3 * self.dt
    #     if set_max_dt:
    #         str_dt = "Selected Timestep Size ({} of {}): {:.{p}f} ms".format(
    #             min(max_divisor_tf, n_div), n_div, dt_ms, p=mag_add)
    #     else:
    #         str_dt = "Selected Timestep Size: {:.{p}f} ms".format(dt_ms,
    #                                                               p=mag_add)

    #     # Updating Nyquist frequency
    #     self.freq_Nyquist = 1. / (2. * self.dt)

    #     pprint(str_dt, comm=self.comm)

    # def layer_infinite_model(self):
    #     """
    #     Determine the domain extension size for the infinite domain model

    #     Parameters
    #     ----------
    #     None

    #     Returns
    #     -------
    #     infinite_pad_len : `float`
    #         Size of the domain extension for the infinite domain model
    #     """

    #     # Size of the domain extension
    #     add_dom = self.c_bnd_max * self.final_time / 2.

    #     # Distance already travelled by the wave
    #     if hasattr(self, 'eik_bnd'):

    #         # If Eikonal analysis was performed
    #         eikmin = self.eik_bnd[0][2]

    #         # Minimum distance to the nearest boundary
    #         dist_to_bnd = self.c_bnd_max * eikmin / 2.
    #     else:

    #         # If Eikonal analysis was not performed
    #         sources_loc = array(self.source_locations)

    #         # Candidate to minimum distance to the boundaries
    #         delta_z = abs(sources_loc[:, 0] - self.mesh_parameters.length_z)
    #         delta_x = minimum(abs(sources_loc[:, 1]),
    #                           abs(sources_loc[:, 1]
    #                               - self.mesh_parameters.length_x))
    #         cand_dist = (delta_z, delta_x)

    #         if self.dimension == 3:  # 3D
    #             delta_y = minimum(abs(sources_loc[:, 2]),
    #                               abs(sources_loc[:, 2]
    #                                   - self.mesh_parameters.length_y))
    #             cand_dist += (delta_y,)

    #         # Minimum distance to the nearest boundary
    #         dist_to_bnd = min(cand_dist)

    #     # Subtracting the distance already travelled by the wave
    #     add_dom -= dist_to_bnd

    #     # Pad length for the infinite domain extension
    #     infinite_pad_len = self.lmin * ceil(add_dom / self.lmin)

    #     return infinite_pad_len

    # def geometry_infinite_model(self):
    #     """
    #     Determine the geometry for the infinite domain model.

    #     Parameters
    #     ----------
    #     None

    #     Returns
    #     -------
    #     None
    #     """

    #     # Size of the domain extension
    #     self.abc_pad_length = self.layer_infinite_model()

    #     inf_str = "Infinite Domain Extension (km): {:.4f}"
    #     pprint(inf_str.format(self.abc_pad_length), comm=self.comm)

    #     # New dimensions
    #     self.abc_new_geometry()

    # def infinite_model(self, check_dt=False, max_divisor_tf=1,
    #                    method='ANALYTICAL', mag_add=3):
    #     """
    #     Create a reference model for the HABC scheme for comparative purposes

    #     Parameters
    #     ----------
    #     check_dt : `bool`, optional
    #         If `True`, check if the timestep size is appropriate for the
    #         transient response. Default is `False`
    #     max_divisor_tf : `int`, optional
    #         Index to select the maximum divisor of the final time, converted
    #         to an integer according to the order of magnitude of the timestep
    #         size. The timestep size is set to the divisor, given by the index
    #         in descending order, less than or equal to the user's timestep
    #         size. If the value is 1, the timestep size is set as the maximum
    #         divisor. Default is 1
    #     method : `str`, optional
    #         Method to use for solving the eigenvalue problem. Default
    #         is 'ANALYTICAL' method that estimates the maximum eigenvalue
    #         using the Gershgorin Circle Theorem.
    #         Opts: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS' or 'LOBPCG'
    #     mag_add : `int`, optional
    #         Additional magnitude order to adjust the rounding of the timestep

    #     Returns
    #     -------
    #     None
    #     """

    #     # Check the timestep size
    #     if check_dt:
    #         self.check_timestep_abc(max_divisor_tf=max_divisor_tf,
    #                                 method=method, mag_add=mag_add)

    #     pprint("\nBuilding Infinite Domain Model", comm=self.comm)

    #     # Defining geometry for infinite domain
    #     self.geometry_infinite_model()

    #     # Creating mesh for infinite domain
    #     self.create_mesh_with_layer(inf_model=True)

    #     # Updating velocity model
    #     self.velocity_abc(inf_model=True)

    #     # Setting no damping
    #     if self.abc_boundary_layer_type == "hybrid":
    #         self.cosHig = Constant(0.)
    #         self.eta_mask = Constant(0.)
    #         self.eta_habc = Constant(0.)

    #     elif self.abc_boundary_layer_type == "PML":
    #         self.sigma_z = Constant(0.)
    #         self.sigma_x = Constant(0.)
    #         if self.dimension == 3:
    #             self.sigma_y = Constant(0.)

    #     pprint("\nSolving Infinite Model", comm=self.comm)

    #     # Solving the forward problem
    #     self.forward_solve()

    #     # Saving reference signal
    #     self.save_reference_signal()

    #     # Deleting variables to be computed for the ABC scheme
    #     del self.length_xabc, self.length_zabc
    #     if self.dimension == 3:
    #         del self.length_yabc
    #     if self.abc_boundary_layer_type == "hybrid":
    #         del self.cosHig, self.eta_mask, self.eta_habc
    #     elif self.abc_boundary_layer_type == "PML":
    #         del self.sigma_z, self.sigma_x
    #         if self.dimension == 3:
    #             del self.sigma_y
