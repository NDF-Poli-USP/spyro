from numpy import inf
from .abc import AbsorbingBC
from .nrbc import NRBC
from ..io.basicio import parallel_print as pprint
from .lay_len import calc_size_lay
from ..plots.plots_habc import plot_function_layer_size
from ..tools.abc_set_path_cases import formatting_abc_layer_type, path_to_save_abc_case
from ..utils.error_management import validate_enum, validate_numeric, validate_parameter
from ..utils.freq_tools import freq_response
from ..utils.typing import (AbsorbingBCsType, BoundaryConditionsType, HyperLayerDegreeType,
                            LayerShapeType, LayerSizeRefFrequency, NRBCBoundaryType)

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender
# TODO: Add reference


class ABCLayer(AbsorbingBC):
    """Class for ABCs based on absorbing layers.

    Attributes
    ----------
    a_par : `float`
        Adimensional propagation speed parameter (a = z / f).
        Also, 'z' parameter is the inverse of the minimum Eikonal (1 / phi_min).
    abc_boundary_layer_shape : `typing.LayerShapeType`
        Shape type of the pad layer. Options: `LayerShapeType.RECTANGULAR` or
        `LayerShapeType.HYPERSHAPE`. Default is `LayerShapeType.RECTANGULAR`.
    abc_boundary_layer_type : `typing.AbsorbingBCsType`
        Type of the boundary layer: `AbsorbingBCsType.HYBRID` or AbsorbingBCsType.PML`.
        Default is `AbsorbingBCsType.HYBRID`  based on paper of Salas et al. (2022).
        doi: https://doi.org/10.1016/j.apm.2022.09.014
        TODO: Add citation
    abc_deg_layer : `int` or `float` or `None`
        Hypershape degree. For hypershape layers, the degree must be greater than or
        equal to 2. `None` is used only for rectangular layers. Default is `None`.
    abc_degree_type : `typing.HyperLayerDegreeType`
        Type of the hypereshape degree. Options: 'HyperLayerDegreeType.REAL' or
        'HyperLayerDegreeType.INTEGER'. Default is 'HyperLayerDegreeType.REAL'.
    abc_pad_length : `float`
        Size of the absorbing layer
    abc_reference_freq : `typing.LayerSizeRefFrequency`
        Reference frequency for sizing the absorbing layer.
        Options: 'LayerSizeRefFrequency.SOURCE' or 'LayerSizeRefFrequency.BOUNDARY'.
        Default is 'LayerSizeRefFrequency.SOURCE'.
    case_absl : `str`
        Label for the output files that includes the layer shape and degree for
        hypershape layers ("REC", "HNX.Y" with X.Y as the hypershape degree with one
        decimal place precision) and the reference frequency ('SOU' or 'BND').
        Examples: "REC_SOU", "REC_BND", "HN2.4_SOU" or "HN2.4_BND".
    crit_source : `tuple`
       Critical source coordinates.
    d_norm : `float`
        Normalized element size (lmin / pad_len)
    ele_pad : `int`
        Number of elements in the layer of edge length equal to 'lmin'.
    factor_length_pad : `float`
        Size parameter of the absorbing layer.
    freq_ref : `float`
        Reference frequency of the wave at the critical poin on boundary.
    FLpossible : `list`
        Possible size parameters for the absorbing layer without rounding.
    layer_geometry : `rec_lay.RectangLayer` or `hyp_lay.HyperLayer`
        An object representing the geometry of the absorbing layer.
        Options: `rec_lay.RectangLayer` for rectangular layers or `hyp_lay.HyperLayer`
        for hypershape layers.
    lmin : `float`
        Minimum mesh size.
    lref : `float`
        Reference length for the size of the absorbing layer.
    path_case_absl : `string`
        Path to save data for the current case study of ABCs based on absorbing layers.
    path_save : `string`
        Path to save data.

    Methods
    -------
    _define_layer_shape()
        Define the shape of the absorbing layer.
    det_reference_freq()
        Determine the reference frequency for a new layer size.
    layer_size_criterion()
        Determine the absorbing layer size using the Eikonal criterion for HABCs.
    nrbc_on_boundary_layer()
        Apply Non-Reflective BCs on the outer boundary of the absorbing layer.
    """

    def __init__(self, domain_dim, frequency=None, dt=None,
                 dimension=2, quadrilateral=False, func_space_type=None,
                 abc_boundary_layer_shape=LayerShapeType.RECTANGULAR,
                 abc_boundary_layer_type=AbsorbingBCsType.HYBRID,
                 abc_reference_freq=LayerSizeRefFrequency.SOURCE,
                 abc_degree_type=HyperLayerDegreeType.REAL,
                 abc_deg_layer=None, output_folder=None, comm=None):
        """Initialize the ABCLayer class.

        Parameters
        ----------
        domain_dim : `tuple`
            Original domain dimensions: (length_z, length_x) for 2D
            or (length_z, length_x, length_y) for 3D.
        frequency: `float`, optional
            Frequency of the source.
        dt : `float`, optional
            Time step used in the simulation. Default is `None`.
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D.
        quadrilateral : `bool`, optional
            Flag to indicate whether to use quadrilateral/hexahedral elements.
            Default is `False` (triangular/tetrahedral elements).
        func_space_type, `str`, optional
            Type of function space for the state variable.
            Options: 'scalar' or 'vector'. Default is None.
        abc_boundary_layer_shape : `typing.LayerShapeType`, optional
            Shape type of the pad layer. Options: `LayerShapeType.RECTANGULAR` or
            `LayerShapeType.HYPERSHAPE`. Default is `LayerShapeType.RECTANGULAR`.
        abc_boundary_layer_type : `typing.AbsorbingBCsType`
            Type of the boundary layer: `AbsorbingBCsType.HYBRID` or AbsorbingBCsType.PML`.
            Default is `AbsorbingBCsType.HYBRID` based on paper of Salas et al. (2022).
            doi: https://doi.org/10.1016/j.apm.2022.09.014
            TODO: Add citation
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

        AbsorbingBC.__init__(self, domain_dim, frequency=frequency, dt=dt,
                             dimension=dimension, quadrilateral=quadrilateral,
                             func_space_type=func_space_type, comm=comm)

        # ABC layer parameters
        self.abc_boundary_layer_type = validate_enum("abc_boundary_layer_type",
                                                     abc_boundary_layer_type,
                                                     AbsorbingBCsType)
        if abc_boundary_layer_type not in [AbsorbingBCsType.HYBRID, AbsorbingBCsType.PML]:
            validate_parameter("abc_boundary_layer_type", abc_boundary_layer_type,
                               [AbsorbingBCsType.HYBRID, AbsorbingBCsType.PML])

        self.abc_boundary_layer_shape = validate_enum("abc_boundary_layer_shape",
                                                      abc_boundary_layer_shape,
                                                      LayerShapeType)
        self.abc_reference_freq = validate_enum("abc_reference_freq",
                                                abc_reference_freq,
                                                LayerSizeRefFrequency)
        self.abc_degree_type = validate_enum("abc_degree_type", abc_degree_type,
                                             HyperLayerDegreeType)

        # Layer degree
        if self.abc_boundary_layer_shape == LayerShapeType.RECTANGULAR:
            self.abc_deg_layer = None
        elif self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE:
            self.abc_deg_layer = validate_numeric('abc_deg_layer', abc_deg_layer,
                                                  float_num=True, integer_num=True,
                                                  lower_bound=2.,
                                                  include_lower_bound=True)

        # Define the shape of the absorbing layer.
        self.layer_geometry = self._define_layer_shape()

        """"
        Create the path to save data
        The required abc_type argument from path_to_save_abc_layer_case() method is set to
        self.abc_boundary_layer_type since it is an instance of `typing.AbsorbingBCsType`.
        """
        self.path_save, self.case_absl, self.path_case_absl = \
            path_to_save_abc_case(self.abc_boundary_layer_type,
                                  abc_boundary_layer_shape=self.abc_boundary_layer_shape,
                                  abc_deg_layer=self.abc_deg_layer,
                                  abc_reference_freq=self.abc_reference_freq,
                                  output_folder=output_folder)

        # Initializing the error measure class
        self.initialize_paths_for_error(output_folder=self.path_save,
                                        output_case=self.path_case_absl)

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
                                 output_folder=self.path_case_absl)

        # Layer type
        lay_str = "\nDetermining New Geometry with {}"
        lay_str = formatting_abc_layer_type(lay_str, self.abc_boundary_layer_type)
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

    def nrbc_on_boundary_layer(self, wave, non_reflect_bc, save_file=True):
        """Apply Non-Reflective BCs on the outer boundary of the absorbing layer.

        Parameters
        ----------
        wave : `acoustic_wave.AcousticWave`
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
        if non_reflect_bc in [BoundaryConditionsType.SOMMERFELD,
                              BoundaryConditionsType.HIGDON]:

            bnd_nod_ids_nfs = bnd_nodes_nfs = None

            # Getting the boundary type where NRBCs are applied
            abc_boundary_type = NRBCBoundaryType.HYPERSHAPE \
                if self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE \
                else NRBCBoundaryType.STRAIGHT

            # Initializing the NRBC class
            NRBC.__init__(self, self.domain_dim, non_reflect_bc=non_reflect_bc,
                          abc_boundary_type=abc_boundary_type,
                          dimension=self.dimension, nrbc_in_habc=True,
                          output_folder=self.path_case_absl, comm=self.comm)
            # Hypershape parameters
            hyp_par = (self.layer_geometry.n_hyp, *self.layer_geometry.hyper_axes) \
                if self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE else None

            crit_source = self.crit_source \
                if non_reflect_bc == BoundaryConditionsType.HIGDON else None

            self.nrbc_on_boundary(wave, source_coord=crit_source, hyp_par=hyp_par, save_file=save_file)

        else:
            pprint("\nNot Non-Reflecting Boundary Conditions Prescribed", comm=self.comm)
