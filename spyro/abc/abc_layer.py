from numpy import abs, array, ceil, inf, log10, minimum
from time import perf_counter
from sympy import divisors
from .abc import AbsorbingBC
from .nrbc import NRBC
from .lay_len import calc_size_lay
from ..mpi.spyro_mpi import SpyroEnsemble
from ..plots.plots_habc import plot_function_layer_size
from ..tools.abc_set_path_cases import formatting_abc_layer_type, path_to_save_abc_case
from ..utils.error_management import validate_enum, validate_numeric, validate_parameter
from ..utils.freq_tools import freq_response
from ..utils.typing import (
    AbsorbingBCsType,
    BoundaryConditionsType,
    HyperLayerDegreeType,
    LayerShapeType,
    LayerSizeRefFrequency,
    NRBCBoundaryType,
)

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
        Reference frequency of the wave at the critical point on boundary.
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
    nrbc : `nrbc.NRBC` or `None`
        Non-Reflective BCs applied on the outer boundary of the absorbing layer.
        It is `None` until `nrbc_on_boundary_layer()` is called.
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

    def __init__(
        self,
        domain_dim,
        frequency=None,
        dt=None,
        dimension=2,
        quadrilateral=False,
        func_space_type=None,
        abc_boundary_layer_shape=LayerShapeType.RECTANGULAR,
        abc_boundary_layer_type=AbsorbingBCsType.HYBRID,
        abc_reference_freq=LayerSizeRefFrequency.SOURCE,
        abc_degree_type=HyperLayerDegreeType.REAL,
        abc_deg_layer=None,
        output_folder=None,
    ):
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

        Returns
        -------
        None
        """

        AbsorbingBC.__init__(
            self,
            domain_dim,
            frequency=frequency,
            dt=dt,
            dimension=dimension,
            quadrilateral=quadrilateral,
            func_space_type=func_space_type,
        )

        # ABC layer parameters
        self.abc_boundary_layer_type = validate_enum(
            "abc_boundary_layer_type", abc_boundary_layer_type, AbsorbingBCsType
        )
        if abc_boundary_layer_type not in [
            AbsorbingBCsType.HYBRID,
            AbsorbingBCsType.PML,
        ]:
            validate_parameter(
                "abc_boundary_layer_type",
                abc_boundary_layer_type,
                [AbsorbingBCsType.HYBRID, AbsorbingBCsType.PML],
            )

        self.abc_boundary_layer_shape = validate_enum(
            "abc_boundary_layer_shape", abc_boundary_layer_shape, LayerShapeType
        )
        self.abc_reference_freq = validate_enum(
            "abc_reference_freq", abc_reference_freq, LayerSizeRefFrequency
        )
        self.abc_degree_type = validate_enum(
            "abc_degree_type", abc_degree_type, HyperLayerDegreeType
        )

        # Layer degree
        if self.abc_boundary_layer_shape == LayerShapeType.RECTANGULAR:
            self.abc_deg_layer = None
        elif self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE:
            self.abc_deg_layer = validate_numeric(
                "abc_deg_layer",
                abc_deg_layer,
                float_num=True,
                integer_num=True,
                lower_bound=2.0,
                include_lower_bound=True,
            )

        # Define the shape of the absorbing layer.
        self.layer_geometry = self._define_layer_shape()

        # Non-Reflective BCs on the outer layer boundary (see nrbc_on_boundary_layer)
        self.nrbc = None

        # Create the path to save data
        # The required abc_type argument from path_to_save_abc_layer_case() method is set to
        # self.abc_boundary_layer_type since it is an instance of `typing.AbsorbingBCsType`.
        self.path_save, self.case_absl, self.path_case_absl = path_to_save_abc_case(
            self.abc_boundary_layer_type,
            abc_boundary_layer_shape=self.abc_boundary_layer_shape,
            abc_deg_layer=self.abc_deg_layer,
            abc_reference_freq=self.abc_reference_freq,
            output_folder=output_folder,
        )

        # Initializing the error measure class
        self.initialize_paths_for_error(
            output_folder=self.path_save, output_case=self.path_case_absl
        )

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

            Hypershape_layer = HyperLayer(
                self.domain_dim,
                n_hyp=self.abc_deg_layer,
                n_type=self.abc_degree_type,
                dimension=self.dimension,
            )
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

        SpyroEnsemble.print("\nDetermining Reference Frequency")

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
                freq_ref = freq_response(
                    histPcrit, self.freq_Nyquist, fpad=fpad, get_dominant_freq=True
                )
                SpyroEnsemble.print(
                    f"Frequency at Critical Point {n_crit:>2.0f}: {freq_ref:.5f}"
                )

                self.freq_ref = min(self.freq_ref, freq_ref)

        SpyroEnsemble.print(f"Reference Frequency (Hz): {self.freq_ref:.5f}")

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
        (
            self.factor_length_pad,
            self.abc_pad_length,
            self.ele_pad,
            self.d_norm,
            self.a_par,
            self.FLpossible,
        ) = calc_size_lay(
            self.freq_ref,
            z_par,
            self.lmin,
            self.lref,
            n_root=n_root,
            layer_based_on_mesh=layer_based_on_mesh,
        )

        plot_function_layer_size(
            [self.a_par, z_par],
            [self.freq_ref, self.frequency],
            [self.lmin, self.lref],
            self.FLpossible,
            output_folder=self.path_case_absl,
        )

        # Layer type
        lay_str = "\nDetermining New Geometry with {}"
        lay_str = formatting_abc_layer_type(lay_str, self.abc_boundary_layer_type)
        SpyroEnsemble.print(lay_str)

        # New geometry with layer
        self.abc_new_geometry()

        # Domain dimensions without free surface truncation
        domain_layer_full = self.abc_domain_dimensions()

        if self.abc_boundary_layer_shape == LayerShapeType.RECTANGULAR:
            # Geometric properties of the rectangular layer
            self.layer_geometry.calc_rec_geom_prop(
                domain_layer_full, self.abc_pad_length
            )

        elif self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE:
            # Geometric properties of the hypershape layer
            self.layer_geometry.calc_hyp_geom_prop(
                domain_layer_full, self.abc_pad_length, self.lmin
            )

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
        if non_reflect_bc in [
            BoundaryConditionsType.SOMMERFELD,
            BoundaryConditionsType.HIGDON,
        ]:
            # Getting the boundary type where NRBCs are applied
            abc_boundary_type = (
                NRBCBoundaryType.HYPERSHAPE
                if self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE
                else NRBCBoundaryType.STRAIGHT
            )

            # The layer owns an NRBC object instead of being one
            self.nrbc = NRBC(
                self.domain_dim,
                frequency=self.frequency,
                dt=self.dt,
                abc_boundary_type=abc_boundary_type,
                non_reflect_bc=non_reflect_bc,
                dimension=self.dimension,
                quadrilateral=self.quadrilateral,
                nrbc_in_habc=True,
                output_folder=self.path_case_absl,
            )

            # Hypershape parameters
            hyp_par = (
                (self.layer_geometry.n_hyp, *self.layer_geometry.hyper_axes)
                if self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE
                else None
            )

            crit_source = (
                self.crit_source
                if non_reflect_bc == BoundaryConditionsType.HIGDON
                else None
            )

            self.nrbc.nrbc_on_boundary(
                wave, source_coord=crit_source, hyp_par=hyp_par, save_file=save_file
            )

        else:
            SpyroEnsemble.print("\nNot Non-Reflecting Boundary Conditions Prescribed")

    def check_timestep_abc(
        self, wave, max_divisor_tf=1, set_max_dt=True, method="ANALYTICAL", mag_add=3
    ):
        """Check if the timestep size is appropriate for the transient response.

        Parameters
        ----------
        wave : `acoustic_wave.AcousticWave`
            An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
        max_divisor_tf : `int`, optional
            Index of the selected timestep size among the divisors of the final
            time that are smaller than the estimated stable timestep size, in
            descending order: 1 selects the largest of them. The final time and the
            timestep sizes are converted to integers according to the order of
            magnitude of the estimate. Default is 1.
        set_max_dt : `bool`, optional
            If `True`, set the timestep size to the selected divisor; otherwise, set
            it to the smaller of the selected divisor and the current timestep
            size. Default is `True`.
        method : `str`, optional
            Method to use for the largest eigenvalue. Default is 'ANALYTICAL', the
            largest ratio of the stiffness and mass diagonals, which overestimates
            the stable timestep size (see Notes). 'ARNOLDI', 'LANCZOS' and 'LOBPCG'
            solve the eigenvalue problem.
        mag_add : `int`, optional
            Additional magnitude order to adjust the rounding of the timestep.

        Returns
        -------
        None

        Notes
        -----
        The stable timestep size is ``2 / sqrt(lambda_max)``, where ``lambda_max``
        is the largest generalized eigenvalue of the stiffness and mass forms of
        the wave physics (``wave.modal_weak_forms()``), estimated without any
        fraction. The timestep size is then a divisor of the final time, so that
        the simulation ends exactly at the final time, and ``max_divisor_tf``
        sets the margin below the estimate.

        'ANALYTICAL' takes ``lambda_max`` as ``max(A_ii / M_ii)``. Each ratio is
        the Rayleigh quotient of a unit vector, so it never exceeds the largest
        eigenvalue, and the estimated timestep size is never below the stable
        one. In ``test_reference_model_abc`` it exceeds it by 14 to 50 %, so
        'ANALYTICAL' may need ``max_divisor_tf > 1`` to skip the divisors above
        the stable timestep size (up to 4 there), while an exact method is
        stable with ``max_divisor_tf = 1``. There, once the forms are compiled,
        'ANALYTICAL' takes 4 to 7 ms and 'ARNOLDI' 10 to 18 ms, both negligible
        against the forward solve.

        Besides the current and the selected timestep sizes, the method prints
        the method of the estimate, the estimated timestep size and the time it
        took, the divisors of the final time from the largest down to the
        selected one, and the ratio of the selected timestep size to the
        estimate.
        """

        # Cheking input parameters
        validate_numeric(
            "max_divisor_tf",
            max_divisor_tf,
            float_num=False,
            integer_num=True,
            lower_bound=0.0,
        )
        validate_numeric(
            "mag_add",
            mag_add,
            float_num=False,
            integer_num=True,
            lower_bound=0.0,
            include_lower_bound=True,
        )

        SpyroEnsemble.print("\nChecking Timestep Size")

        # User timestep
        usr_dt = wave.dt
        SpyroEnsemble.print(f"Current Nyquist Frequency: {self.freq_Nyquist:.5f} Hz")
        SpyroEnsemble.print(f"Current Timestep Size: {1e3 * usr_dt:.{mag_add}f} ms")

        # Maximum timestep size
        SpyroEnsemble.print(f"Method for the Maximum Timestep Size: {method}")
        t_start = perf_counter()
        dt_sol = Modal_Solver(self.dimension, method=method, calc_max_dt=True)
        est_dt = dt_sol.estimate_timestep(
            wave.c,
            wave.function_space,
            wave.final_time,
            shift=1e-8,
            quad_rule=wave.quadrature_rule,
            fraction=1.0,
            forms=wave.modal_weak_forms(),
        )
        SpyroEnsemble.print(
            f"Estimated Maximum Timestep Size: {1e3 * est_dt:.{mag_add}f} ms "
            f"(computed in {perf_counter() - t_start:.3f} s)"
        )

        # Rounding power
        pot = int(abs(ceil(log10(est_dt))) + mag_add)

        # Maximum timestep size according to divisors of the final time
        val_int_tf = int(10**pot * wave.final_time)
        val_int_dt = int(10**pot * est_dt)
        max_div = [d for d in divisors(val_int_tf) if d < val_int_dt]
        n_div = len(max_div)
        index_div = min(max_divisor_tf, n_div)

        # Divisors from the largest down to the selected one
        cand_dt = [round(10 ** (-pot) * d, pot) for d in max_div[::-1][:index_div]]
        SpyroEnsemble.print(
            "Largest Divisors of the Final Time Below the Estimate (ms): "
            + ", ".join(f"{1e3 * d:.{mag_add}f}" for d in cand_dt)
        )
        max_dt = cand_dt[-1]

        # Set the timestep size
        dt = max_dt if set_max_dt else min(usr_dt, max_dt)
        wave.dt = dt
        dt_ms = 1e3 * wave.dt
        if set_max_dt:
            case_div = f"{min(max_divisor_tf, n_div)} of {n_div}"
            str_dt = f"Selected Timestep Size ({case_div}): {dt_ms:.{mag_add}f} ms"
        else:
            str_dt = f"Selected Timestep Size: {dt_ms:.{mag_add}f} ms"

        SpyroEnsemble.print(str_dt)
        SpyroEnsemble.print(
            f"Selected / Estimated Timestep Size: {wave.dt / est_dt:.3f}"
        )

        # Updating Nyquist frequency
        self.freq_Nyquist = 1.0 / (2.0 * wave.dt)
        SpyroEnsemble.print(f"New Nyquist Frequency: {self.freq_Nyquist:.5f} Hz")

    def layer_infinite_model(self, lmin, c_bnd_max, final_time, source_locations=None):
        """Determine the domain extension size for the infinite domain model.

        Parameters
        ----------
        lmin : `float`
            Minimum mesh size.
        c_bnd_max : `float`
            Maximum velocity value on the boundary of the original domain.
        final_time : `float`
            Final time of the simulation.
        source_locations: `list`, optional
            List of source locations.

        Returns
        -------
        infinite_pad_len : `float`
            Size of the domain extension for the infinite domain model.

        Notes
        -----
        The extension covers the bottom and the lateral boundaries; the top is a
        free surface. A wave that first reaches the original boundary at time
        ``t_bnd`` and is reflected by the boundary of the extension returns to the
        original boundary at ``t_bnd + 2 * pad / c``. With ``c = c_bnd_max``, the
        reflections reach no receiver of the original domain before the final
        time if ``pad >= c_bnd_max * (final_time - t_bnd) / 2``, rounded up to a
        multiple of ``lmin``.

        ``t_bnd`` is the minimum Eikonal at the critical boundary points if the
        Eikonal analysis was performed. Otherwise, it is the distance from the
        nearest source to the nearest absorbing boundary over ``c_bnd_max``, so
        half of that distance is subtracted. Source coordinates follow spyro's
        convention, with ``-length_z <= z <= 0``.
        """

        # Cheking input parameters
        validate_numeric(
            "lmin", lmin, float_num=True, integer_num=True, lower_bound=0.0
        )
        validate_numeric(
            "c_bnd_max", c_bnd_max, float_num=True, integer_num=True, lower_bound=0.0
        )
        validate_numeric(
            "final_time", final_time, float_num=True, integer_num=True, lower_bound=0.0
        )
        validate_data_structure(
            "source_locations",
            source_locations,
            "list",
            expected_type_element="tuple",
            accept_parameter_as_none=True,
        )

        # Size of the domain extension
        add_dom = c_bnd_max * final_time / 2.0

        str_pad = "Infinite Domain Extension Based on "

        # Distance already travelled by the wave
        if hasattr(self, "eik_bnd"):
            str_pad += "Minimun Eikonal at Critical Boundary Points"

            # If Eikonal analysis was performed (see `critical_boundary_points` method)
            # Structure eikmin: [pnt_crit, c_bnd, eikmin, z_par, lref, sou_crit]
            eikmin = self.eik_bnd[0][2]

            # Minimum distance to the nearest boundary
            dist_to_bnd = c_bnd_max * eikmin / 2.0
        else:
            str_pad += "Minimum Distance Source-Boundary"

            # If Eikonal analysis was not performed
            source_loc = array(source_locations)

            # Original  domain dimensions
            length_z, length_x = self.domain_dim[:2]

            # Distances from every source to the bottom (z = -length_z) and to
            # the lateral boundaries
            delta_z = abs(source_loc[:, 0] + length_z)
            delta_x = minimum(abs(source_loc[:, 1]), abs(source_loc[:, 1] - length_x))
            cand_dist = [delta_z, delta_x]

            if self.dimension == 3:  # 3D
                length_y = self.domain_dim[2]
                delta_y = minimum(
                    abs(source_loc[:, 2]), abs(source_loc[:, 2] - length_y)
                )
                cand_dist.append(delta_y)

            # Half the minimum distance from any source to the nearest boundary,
            # i.e. c_bnd_max * t_bnd / 2 with t_bnd = distance / c_bnd_max
            dist_to_bnd = float(array(cand_dist).min()) / 2.0

        SpyroEnsemble.print(str_pad)

        # Subtracting the distance already travelled by the wave
        add_dom -= dist_to_bnd

        # Pad length for the infinite domain extension
        infinite_pad_len = lmin * ceil(add_dom / lmin)

        return infinite_pad_len

    def geometry_infinite_model(self, wave):
        """Determine the geometry for the infinite domain model.

        Parameters
        ----------
        wave : `acoustic_wave.AcousticWave`
            An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.

        Returns
        -------
        None
        """

        lmin = wave.mesh_parameters.lmin if not hasattr(self, "lmin") else self.lmin
        c_bnd_max = wave.c_bnd_max
        final_time = wave.final_time
        source_locations = wave.source_locations

        # Size of the domain extension
        self.abc_pad_length = self.layer_infinite_model(
            lmin, c_bnd_max, final_time, source_locations=source_locations
        )
        SpyroEnsemble.print(
            f"Infinite Domain Extension (km): {self.abc_pad_length:.4f}"
        )

        # New dimensions
        self.abc_new_geometry()

    def infinite_model(
        self, wave, check_dt=False, max_divisor_tf=1, method="ANALYTICAL", mag_add=3
    ):
        """Create a reference model for the ABC scheme for comparative purposes.

        Parameters
        ----------
        wave : `acoustic_wave.AcousticWave`
            An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
        check_dt : `bool`, optional
            If `True`, check if the timestep size is appropriate for the transient
            response. Default is `False`.
        max_divisor_tf : `int`, optional
            Index of the selected timestep size among the divisors of the final
            time that are smaller than the estimated stable timestep size, in
            descending order: 1 selects the largest of them. Default is 1. See
            `check_timestep_abc`.
        method : `str`, optional
            Method to use for the largest eigenvalue. Default is 'ANALYTICAL', the
            largest ratio of the stiffness and mass diagonals, which overestimates
            the stable timestep size. 'ARNOLDI', 'LANCZOS' and 'LOBPCG' solve the
            eigenvalue problem. See `check_timestep_abc`.
        mag_add : `int`, optional
            Additional magnitude order to adjust the rounding of the timestep.

        Returns
        -------
        None
        """

        # Check the timestep size
        if check_dt:
            self.check_timestep_abc(
                wave, max_divisor_tf=max_divisor_tf, method=method, mag_add=mag_add
            )

        SpyroEnsemble.print("\nBuilding Infinite Domain Model")

        # Defining geometry for infinite domain
        self.geometry_infinite_model(wave)

        # Creating mesh for infinite domain
        self.create_mesh_with_layer(wave, inf_model=True)

        # Updating velocity model
        self.velocity_abc(wave, inf_model=True)

        SpyroEnsemble.print("\nSolving Infinite Model")

        # Solving the forward problem
        wave.forward_solve()

        # Saving reference signal
        output_file = self.abc_boundary_layer_type.value + "_ref"
        self.save_reference_signal(
            wave.receiver_locations,
            wave.forward_solution_receivers,
            wave.number_of_receivers,
            self.freq_Nyquist,
            output_file_prefix=output_file,
        )
