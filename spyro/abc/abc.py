from abc import ABCMeta
from firedrake import (ds as fire_ds, ds_b as quad_dsbottom, ds_t as quad_dstop,
                       ds_v as quad_ds, Function, VTKFile)
from numpy import abs, array, ceil, log10, minimum, prod, sign, sum, where
from sympy import divisors
from .eik_min import Minimum_Eikonal
from ..solvers.modal.modal_sol import Modal_Solver
from ..tools.error_measure import MeasureError
from ..domains.space import create_function_space
from ..io.basicio import parallel_print as pprint
from ..plots.plots_habc import plot_frequency_domain_receiver_responses
from ..plots.receiver_plots import plot_comparison_of_receivers_to_reference
from ..tools.abc_set_path_cases import formatting_abc_layer_type
from ..tools.habc_tools import clipping_coordinates_lay_field, extend_scalar_field_profile
from ..utils.error_management import (validate_data_structure, validate_numeric,
                                      validate_parameter)
from ..utils.freq_tools import fft_at_receivers
from ..utils.typing import AbsorbingBCsType, LayerShapeType


class AbsorbingBC(MeasureError, metaclass=ABCMeta):
    """Base class for the absorbing boundary conditions.

    Attributes
    ----------
    comm : `object`
        An object representing the communication interface for parallel processing.
        Default is `None`.
    dimension : `int`, optional
        Model dimension (2D or 3D). Default is 2D.
    domain_dim : `tuple`
        Original domain dimensions: (length_z, length_x) for 2D
        or (length_z, length_x, length_y) for 3D.
    dt : `float` or `None`
        Time step used in the simulation. It is `None` if the response is not 'transient'.
    eik_bnd : `list`
        Properties on boundaries according to minimum values of Eikonal.
        Structure sublist: [pnt_crit, c_bnd, eikmin, z_par, lref, sou_crit]
        - pnt_crit : `array`
            Critical point coordinates.
        - c_bnd :  `float`
            Propagation speed at critical point.
        - eikmin : `float`
            Minimum eikonal value in seconds.
        - z_par :  `float`
            Inverse of minimum Eikonal (Equivalent to c_bound/lref).
        - lref : `float`
            Distance to the closest source from critical point.
        - sou_crit : `tuple`
            Critical source coordinates.
    frequency: `float`
        Frequency of the source.
    freq_Nyquist : `float`
        Nyquist frequency according to the time step. freq_Nyquist = 1 / (2 * dt).
    func_space_type, `str`, optional
        Type of function space for the state variable.
        Options: 'scalar' or 'vector'. Default is None.
    length_xabc : `float`
        Length of the domain in the x-direction with absorbing layer.
    length_yabc : `float`
        Length of the domain in the y-direction with absorbing layer (3D).
    length_zabc : `float`
        Length of the domain in the z-direction with absorbing layer.
    quadrilateral : `bool`, optional
        Flag to indicate whether to use quadrilateral/hexahedral elements.
        Default is `False` (triangular/tetrahedral elements).

    Methods
    -------
    abc_domain_dimensions()
        Return the new dimensions of the domain with absorbing layer as a tuple.
    abc_new_geometry()
        Determine the new domain geometry with the absorbing layer.
    check_timestep_abc()
        Check if the timestep size is appropriate for the transient response.
    create_mesh_with_layer()
        Create a mesh with absorbing layer based on the determined size.
    critical_boundary_points()
        Determine critical boundary points using the Eikonal criterion.
    forms_acoustic_NRBCs()
        Construct the load term forms for non-reflecting boundary conditions (NRBCs).
    geometry_infinite_model()
        Determine the geometry for the infinite domain model.
    infinite_model()
        Create a reference model for the ABC scheme for comparative purposes.
    layer_infinite_model()
        Determine the domain extension size for the infinite domain model.
    comparison_plots()
        Plot the comparison between the ABC scheme and the reference model.
    velocity_abc()
        Set the velocity profile for the model with absorbing layer.
    """

    def __init__(self, domain_dim, frequency=None, dt=None, dimension=2,
                 quadrilateral=False, func_space_type=None, comm=None):
        """Initialize the AbsorbingBC class.

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
        comm : `object`, optional
            An object representing the communication interface for parallel processing.
            Default is `None`.

        Returns
        -------
        None
        """

        # Model dimension
        self.dimension = validate_parameter("dimension", dimension, [2, 3])

        # Original domain dimensions
        self.domain_dim = validate_data_structure("domain_dim", domain_dim, "tuple",
                                                  expected_type_element=("float", "int"),
                                                  expected_length=dimension)

        # Source frequency
        self.frequency = validate_numeric("frequency", frequency,
                                          float_num=True, integer_num=True,
                                          lower_bound=0., accept_parameter_as_none=True)

        # Timestep for the transient simulation
        self.dt = validate_numeric("dt", dt, float_num=True, integer_num=True,
                                   lower_bound=0., accept_parameter_as_none=True)

        # Nyquist frequency
        self.freq_Nyquist = None if self.dt is None else 1. / (2. * self.dt)

        # Quadrilateral/hexahedral elements
        self.quadrilateral = quadrilateral

        # Type of function space
        self.func_space_type = func_space_type

        # Communicator MPI
        self.comm = comm

        # Initializing the error measure class
        MeasureError.__init__(self, comm=self.comm)

    def critical_boundary_points(self, wave):
        """Determine critical boundary points using the Eikonal criterion.

        Use original-domain boundaries to size the absorbing layer.
        See Salas et al (2022): Hybrid absorbing scheme based on hyperelliptical
        layers with non-reflecting boundary conditions in scalar wave equations.
        doi: https://doi.org/10.1016/j.apm.2022.09.014
        TODO: Add citation

        Parameters
        ----------
        wave : `wave.Wave`
            An instance of the :class:`~spyro.solvers.wave.Wave`.

        Returns
        -------
        None
        """

        # Initializing Eikonal object
        eik = Minimum_Eikonal(wave)

        # Solving Eikonal
        eik.solve_eik()

        # Identifying critical points
        self.eik_bnd = eik.ident_crit_eik()

        # Critical point coordinates as receivers
        pcrit = [tuple(bnd[0]) for bnd in self.eik_bnd]
        wave.receiver_locations = pcrit + wave.receiver_locations
        wave.number_of_receivers = len(wave.receiver_locations)

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

    def create_mesh_with_layer(self, wave, inf_model=False, spln=True, save_file=True):
        """Create a mesh with absorbing layer based on the determined size.

        Parameters
        ----------
        wave : `wave.Wave`
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

            # Update the pad length in wave object
            wave.abc_pad_length = self.abc_pad_length

            # Create the mesh
            wave.set_mesh()
            pprint("Extended Rectangular Mesh Generated Successfully", comm=self.comm)

        elif layer_shape == LayerShapeType.HYPERSHAPE:

            # Update the pad length in wave.mesh_parameters object
            wave.mesh_parameters.abc_pad_length = self.abc_pad_length

            # Parameters for hypershape mesh
            if self.dimension == 2:  # 2D
                geometry_param = self.layer_geometry.perim_hyp

            if self.dimension == 3:  # 3D
                geometry_param = self.layer_geometry.surf_hyp

            hypershape_param = (
                self.layer_geometry.n_hyp, geometry_param, *self.layer_geometry.hyper_axes)

            # Creating the mesh with the absorbing layer based on the hypershape geometry
            mesh_abc = wave.mesh_ops.hypershape_mesh_habc(
                hypershape_param, wave.mesh_original, wave.mesh_parameters, spln=spln)

            # Updating the mesh with the absorbing layer
            wave.set_mesh(user_mesh=mesh_abc)

        pprint("Mesh Generated Successfully", comm=self.comm)

        if save_file:
            if inf_model:
                pth_mesh = self.path_save + "preamble/mesh_inf.pvd"
            else:
                mesh_file_name = formatting_abc_layer_type("mesh_{}.pvd",
                                                           self.abc_boundary_layer_type,
                                                           for_prints=False)
                pth_mesh = self.path_case_absl + mesh_file_name

            # Save new mesh
            outfile = VTKFile(pth_mesh)
            outfile.write(wave.mesh)

    def velocity_abc(self, wave, inf_model=False, method="point_cloud", save_file=True):
        """Set the velocity profile for the model with absorbing layer.

        Parameters
        ----------
        wave : `wave.Wave`
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
        V = create_function_space(wave.mesh, method_element, 0)

        # Initialize velocity field and assigning the original velocity model
        wave.c = Function(V).interpolate(wave.initial_velocity_model,
                                         allow_missing_dofs=True)

        # Clipping coordinates to the layer domain
        domain_layer = self.abc_domain_dimensions(full_hyp=False)
        ufl_coordinates_abc = wave.mesh_ops.get_spatial_coordinates_abc(wave.mesh,
                                                                        domain_layer)
        lay_field, layer_mask = clipping_coordinates_lay_field(self.domain_dim, wave.mesh,
                                                               self.dimension, ufl_coordinates_abc,
                                                               V, quadrilateral=self.quadrilateral)

        # Extending velocity model within the absorbing layer
        extended_velocity = extend_scalar_field_profile(wave.mesh_original, wave.initial_velocity_model,
                                                        lay_field, layer_mask, wave.mesh_parameters.tol,
                                                        method=method, name_prop="Velocity")

        # Interpolating the velocity model in the layer
        wave.c.interpolate(extended_velocity * layer_mask + (1. - layer_mask)
                           * wave.c, allow_missing_dofs=True)
        del layer_mask, lay_field

        # Interpolating in the space function of the problem
        wave.c = Function(wave.function_space, name="c[km/s])").interpolate(wave.c)

        # Save new velocity model
        if save_file:
            if inf_model:
                pth_velocity = self.path_save + "preamble/c_inf.pvd"
            else:
                c_file_name = formatting_abc_layer_type("c_{}.pvd",
                                                        self.abc_boundary_layer_type,
                                                        for_prints=False)
                pth_velocity = self.path_case_absl + c_file_name

            outfile = VTKFile(pth_velocity)
            outfile.write(wave.c)

    def check_timestep_abc(self, wave, max_divisor_tf=1,
                           set_max_dt=True, method='ANALYTICAL', mag_add=3):
        """Check if the timestep size is appropriate for the transient response.

        Parameters
        ----------
        wave : `acoustic_wave.AcousticWave`
            An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
        max_divisor_tf : `int`, optional
            Index to select the maximum divisor of the final time, converted to an
            integer according to the order of magnitude of the timestep size. The
            timestep size is set to the divisor, given by the index in descending
            order, less than or equal to the user's timestep size. If the value is 1,
            the timestep size is set as the maximum divisor. Default is 1.
        set_max_dt : `bool`, optional
            If `True`, set the timestep size to the selected divisor. Default is `True`.
        method : `str`, optional
            Method to use for solving the eigenvalue problem. Default is 'ANALYTICAL'
            method that estimates the maximum eigenvalue using the Gershgorin Circle
            Theorem. Opts: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS' or 'LOBPCG'.
        mag_add : `int`, optional
            Additional magnitude order to adjust the rounding of the timestep.

        Returns
        -------
        None
        """

        # Cheking input parameters
        validate_numeric("max_divisor_tf", max_divisor_tf,
                         float_num=False, integer_num=True, lower_bound=0.)
        validate_numeric("mag_add", mag_add, float_num=False, integer_num=True,
                         lower_bound=0., include_lower_bound=True)

        pprint("\nChecking Timestep Size", comm=self.comm)

        # User timestep
        usr_dt = wave.get_dt()
        pprint(f"Current Nyquist Frequency: {self.freq_Nyquist:.5f} Hz", comm=self.comm)
        pprint(f"Current Timestep Size: {1e3 * usr_dt:.{mag_add}f} ms", comm=self.comm)

        # Maximum timestep size
        dt_sol = Modal_Solver(self.dimension, method=method, calc_max_dt=True)
        max_dt = dt_sol.estimate_timestep(wave.c, wave.function_space, wave.final_time,
                                          shift=1e-8, quad_rule=wave.quadrature_rule,
                                          fraction=1.)

        # Rounding power
        pot = int(abs(ceil(log10(max_dt))) + mag_add)

        # Maximum timestep size according to divisors of the final time
        val_int_tf = int(10**pot * wave.final_time)
        val_int_dt = int(10**pot * max_dt)
        max_div = [d for d in divisors(val_int_tf) if d < val_int_dt]
        n_div = len(max_div)
        index_div = min(max_divisor_tf, n_div)
        max_dt = round(10**(-pot) * max_div[-index_div], pot)

        # Set the timestep size
        dt = max_dt if set_max_dt else min(usr_dt, max_dt)
        wave.set_dt(dt)
        dt_ms = 1e3 * wave.dt
        if set_max_dt:
            case_div = f"{min(max_divisor_tf, n_div)} of {n_div}"
            str_dt = f"Selected Timestep Size ({case_div}): {dt_ms:.{mag_add}f} ms"
        else:
            str_dt = f"Selected Timestep Size: {dt_ms:.{mag_add}f} ms"

        pprint(str_dt, comm=self.comm)

        # Updating Nyquist frequency
        self.freq_Nyquist = 1. / (2. * wave.dt)
        pprint(f"New Nyquist Frequency: {self.freq_Nyquist:.5f} Hz", comm=self.comm)

    def min_coord_differ_source_boundary(self, source_locations, get_crit_source=False):
        """Compute the minimum coordinate difference from sources to the nearest boundary.

        Parameters
        ----------
        source_locations: `list`
            List of source locations.
        get_crit_source: `bool`, optional
            If `True`, return the critical source location. Default is `False`.

        Returns
        -------
        min_dist_to_bnd": `float`
            minimum coordinate difference from sources to the nearest boundary.
            It is not an Euclidean distance and is calculates as:
                min_dist_to_bnd(P, Q) = min_i |x_i − y_i|
            This distance is the opposite to the Chebyshev distance defined as:
            (https://en.wikipedia.org/wiki/Chebyshev_distance) TODO: Add citation
                D_Chebyshev(P, Q) = max_i |x_i − y_i|
        critical_source: `tuple`
           Critical source location. If there are multiple sources with the
           same minimum coordinate difference, the critical source is the
           geometric mean of the critical sources.
        """

        # Source locations
        source_loc = array(source_locations)

        # Original  domain dimensions
        length_z, length_x = self.domain_dim[:2]

        def update_min_value_and_sources(candidate, delta, min_dist_to_bnd, source_cand):
            """Update candidates to minimum coordinate difference and associated sources.

            Parameters
            ----------
            candidate : `float`
                Candidate to minimum coordinate difference to the boundaries.
            delta : `arrray`
                Array of coordinate differences to the boundaries.
            min_dist_to_bnd : `float`
                Current minimum coordinate difference to the boundaries.
            source_cand : `set`
                Current set of source indices for the minimum coordinate difference.

            Returns
            -------
            min_dist_to_bnd : `float`
                Updated minimum coordinate difference to the boundaries.
            source_cand : `set`
                Updated set of source indices for the minimum coordinate difference.
            """

            source_update = where(delta == candidate)[0]
            if candidate < min_dist_to_bnd:
                min_dist_to_bnd = candidate
                source_cand = set(source_update)
            elif candidate_x == min_dist_to_bnd:
                source_cand.update(source_update)

            return min_dist_to_bnd, source_cand

        # Candidate to minimum coordinate difference to the boundaries
        delta_z = abs(source_loc[:, 0] - length_z)
        candidate_z = delta_z.min()
        min_dist_to_bnd = candidate_z
        source_cand = set(where(delta_z == candidate_z)[0])
        delta_x = minimum(abs(source_loc[:, 1]), abs(source_loc[:, 1] - length_x))
        candidate_x = delta_x.min()
        min_dist_to_bnd, source_cand = \
            update_min_value_and_sources(candidate_x, delta_x,
                                         min_dist_to_bnd, source_cand)

        if self.dimension == 3:  # 3D
            length_y = self.domain_dim[2]
            delta_y = minimum(abs(source_loc[:, 2]), abs(source_loc[:, 2] - length_y))
            candidate_y = delta_y.min()
            min_dist_to_bnd, source_cand = \
                update_min_value_and_sources(candidate_y, delta_y,
                                             min_dist_to_bnd, source_cand)
        if get_crit_source:

            if len(source_cand) > 1:

                # Critical sources
                critical_sources = source_loc[list(source_cand), :]

                # Avoiding zero values for geometric mean
                critical_sources[critical_sources == 0.] = 1e-6

                # Apparent source location as the geometric mean of the critical sources
                n_sources = critical_sources.shape[0]
                prod_coord = prod(critical_sources, axis=0)
                sign_coord = sign(sum(critical_sources, axis=0))
                critical_source = tuple(sign_coord * abs(prod_coord) ** (1. / n_sources))

            else:

                # Critical source location
                critical_source = source_locations[[*source_cand][0]]

            return min_dist_to_bnd, critical_source

        return min_dist_to_bnd

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
        """

        # Cheking input parameters
        validate_numeric("lmin", lmin, float_num=True,
                         integer_num=True, lower_bound=0.)
        validate_numeric("c_bnd_max", c_bnd_max, float_num=True,
                         integer_num=True, lower_bound=0.)
        validate_numeric("final_time", final_time, float_num=True,
                         integer_num=True, lower_bound=0.)

        # Size of the domain extension
        add_dom = c_bnd_max * final_time / 2.

        str_pad = "Infinite Domain Extension Based on "

        # Distance already travelled by the wave
        if hasattr(self, 'eik_bnd'):

            # If Eikonal analysis was performed (see `critical_boundary_points` method)
            str_pad += "Minimun Eikonal at Critical Boundary Points"

            # Structure eikmin: [pnt_crit, c_bnd, eikmin, z_par, lref, sou_crit]
            eikmin = self.eik_bnd[0][2]

            # Minimum distance to the nearest boundary
            dist_to_bnd = c_bnd_max * eikmin / 2.
        else:

            # If Eikonal analysis was not performed (see `min_coord_differ_source_boundary`)
            str_pad += "Minimum Coordinate Difference Source-Boundary"

            # Checking source locations
            validate_data_structure("source_locations", source_locations, "list",
                                    expected_type_element="tuple")

            # Minimum distance to the nearest boundary (not Euclidean distance)
            dist_to_bnd = self.min_coord_differ_source_boundary(source_locations)

        pprint(str_pad, comm=self.comm)

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

        lmin = wave.mesh_parameters.lmin if not hasattr(self, 'lmin') else self.lmin
        c_bnd_max = wave.c_bnd_max
        final_time = wave.final_time
        source_locations = wave.source_locations

        # Size of the domain extension
        self.abc_pad_length = self.layer_infinite_model(lmin, c_bnd_max, final_time,
                                                        source_locations=source_locations)
        pprint(f"Infinite Domain Extension (km): {self.abc_pad_length:.4f}", comm=self.comm)

        # New dimensions
        self.abc_new_geometry()

    def infinite_model(self, wave, check_dt=False, max_divisor_tf=1,
                       method='ANALYTICAL', mag_add=3):
        """Create a reference model for the ABC scheme for comparative purposes.

        Parameters
        ----------
        wave : `acoustic_wave.AcousticWave`
            An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
        check_dt : `bool`, optional
            If `True`, check if the timestep size is appropriate for the transient
            response. Default is `False`.
        max_divisor_tf : `int`, optional
            Index to select the maximum divisor of the final time, converted to an
            integer according to the order of magnitude of the timestep size. The
            timestep size is set to the divisor, given by the index in descending
            order, less than or equal to the user's timestep size. If the value is 1,
            the timestep size is set as the maximum divisor. Default is 1.
        method : `str`, optional
            Method to use for solving the eigenvalue problem. Default is 'ANALYTICAL'
            method that estimates the maximum eigenvalue using the Gershgorin Circle
            Theorem. Opts: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS' or 'LOBPCG'.
        mag_add : `int`, optional
            Additional magnitude order to adjust the rounding of the timestep.

        Returns
        -------
        None
        """

        # Check the timestep size
        if check_dt:
            self.check_timestep_abc(wave, max_divisor_tf=max_divisor_tf,
                                    method=method, mag_add=mag_add)

        pprint("\nBuilding Infinite Domain Model", comm=self.comm)

        # Defining geometry for infinite domain
        self.geometry_infinite_model(wave)

        # Creating mesh for infinite domain
        self.create_mesh_with_layer(wave, inf_model=True)

        # Updating velocity model
        self.velocity_abc(wave, inf_model=True)

        pprint("\nSolving Infinite Model", comm=self.comm)

        # Solving the forward problem
        wave.forward_solve()

        # Saving reference signal
        output_file = wave.abc_type.value + "_ref"
        self.save_reference_signal(
            wave.receiver_locations, wave.forward_solution_receivers,
            wave.number_of_receivers, self.freq_Nyquist, output_file=output_file)

    def forms_acoustic_NRBCs(self, wave, weak_expr_nrbc, bc_surf):
        """Build the load term weak forms for non-reflecting boundary conditions (NRBCs).

        Parameters
        ----------
        wave : `acoustic_wave.AcousticWave`
            An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
        weak_expr_nrbc : `ufl.form`
            General weak expression for the NRBCs.
        bc_surf : `tuple`
            Tuple of boundary markers where NRBCs are applied.

        Returns
        -------
        le_nrbc : `ufl.form`
            Load term for the NRBCs.
        """

        # Quadrature surface rule for NRBCs
        quad_surf = wave.surface_quadrature_rule

        # Initializing load term for NRBCs
        le_nrbc = 0.

        if self.quadrilateral and self.dimension == 3:

            # exterior_markers = set(wave.mesh.exterior_facets.unique_markers)
            # print("Available boundary markers:", exterior_markers)

            # Integer boundary IDs for 3D quadrilaterals/hexahedra meshes
            int_ids = tuple(filter(lambda k: isinstance(k, int), bc_surf))

            # Integration measure for 3D quadrilaterals/hexahedra meshes with integer ids
            ds = quad_ds(int_ids, **quad_surf) if quad_surf else quad_ds(int_ids)

            # NRBC on top boundary for 3D quadrilaterals/hexahedra
            if "top" in bc_surf:
                le_nrbc += weak_expr_nrbc * quad_dstop  # (Do not support quadrature)

            # NRBC on bottom boundary for 3D quadrilaterals/hexahedra meshes
            if "bottom" in bc_surf:
                le_nrbc += weak_expr_nrbc * quad_dsbottom  # (Do not support quadrature)

        else:

            # Integration measure for triangles/tetrahedra and 2D quadrilaterals/hexahedra
            ds = fire_ds(bc_surf, **quad_surf) if quad_surf else fire_ds(bc_surf)

        # NRBCs: Higdon or Sommerfeld
        le_nrbc += weak_expr_nrbc * ds

        return le_nrbc

    def comparison_plots(self, wave, receivers_reference, reference_receiver_fft,
                         regression_xCR=False, data_regr_xCR=None):
        """Plot the comparison between the ABC scheme and the reference model.

        Parameters
        ----------
        wave : `wave.Wave`
            An instance of the :class:`~spyro.solvers.wave.Wave`.
        receivers_reference : `array`
            Receiver waveform data in the reference model
        reference_receiver_fft : `ndarray`
            Frequency response magnitude of the reference receiver data.
        regression_xCR : `bool`, optional
            If `True`, Plot the regression for the error measure vs xCR. Default is `False`.
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
              * 'err_difference' : Difference between integral and peak errors
              * 'err_integral' : Minimum integral error

        Returns
        -------
        None
        """

        # Time domain comparison
        plot_comparison_of_receivers_to_reference(wave, receivers_reference)

        # Compute FFT for output signal at receivers
        wave.receivers_out_fft = fft_at_receivers(
            wave.number_of_receivers, wave.forward_solution_receivers, self.freq_Nyquist)

        # For NRBCs the source frequency is set as reference
        if wave.abc_type == AbsorbingBCsType.NRBC:
            self.freq_ref = self.frequency

        # Frequency parameters for plotting
        frequency_parameters = (self.freq_ref, self.frequency, self.freq_Nyquist)

        # Frequency domain comparison
        plot_frequency_domain_receiver_responses(wave, reference_receiver_fft,
                                                 frequency_parameters)

        # # Plot the error measures
        # if regression_xCR:
        #     plot_xCR_opt(self, data_regr_xCR)
