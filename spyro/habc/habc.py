from firedrake import Function  # , Constant, FunctionSpace, SpatialCoordinate, VTKFile
from numpy import abs, imag, pi, real, sqrt, unique
# from numpy import array, ceil, log10, minimum
from os import path, rename
from shutil import rmtree
from ..abc.abc_layer import ABCLayer
from .damp_profile import HABC_Damping
from ..solvers.modal.modal_sol import Modal_Solver
from ..utils.typing import (HyperLayerDegreeType, LayerDampingType,
                            LayerShapeType, LayerSizeRefFrequency)
from ..io.basicio import parallel_print as pprint
# from sympy import divisors
# from spyro.utils.error_management import value_parameter_error

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class HABCLayer(ABCLayer, HABC_Damping):
    """
    Class HABC that determines absorbing layer size and parameters to be used

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
    abc_deg_layer : `int` or `float` or `None`, optional
        Hypershape degree. For hypershape layers, the degree must be greater than or
        equal to 2. `None` is used only for rectangular layers. Default is `None`.
    abc_degree_type : `typing.HyperLayerDegreeType`, optional
        Type of the hypereshape degree. Options: 'HyperLayerDegreeType.REAL' or
        'HyperLayerDegreeType.INTEGER'. Default is 'HyperLayerDegreeType.REAL'.
    abc_reference_freq : `typing.LayerSizeRefFrequency`, optional
        Reference frequency for sizing the absorbing layer.
        Options: 'LayerSizeRefFrequency.SOURCE' or 'LayerSizeRefFrequency.BOUNDARY'.
    a_par : `float`
        Adimensional propagation speed parameter (a = z / f).
        "z" parameter is the inverse of the minimum Eikonal (1 / phi_min).
    c : `Firedrake.Function`
        Velocity model without absorbing layer.
    case_habc : `str`
        Label for the output files that includes the layer shape.
        ('REC' or 'HNI', I for the degree) and the reference frequency ('SOU' or 'BND').
        Example: 'REC_SOU' or 'HN2_BND'.
    crit_source : `tuple`
       Critical source coordinates.
    CRmin : `float`
        Minimum reflection coefficient at the minimum damping ratio.
    d_nomr : `float`
        Normalized element size (lmin / pad_len).
    eik_bnd : `list`
        Properties on boundaries according to minimum values of Eikonal.
        Structure sublist: [pt_cr, c_bnd, eikmin, z_par, lref, sou_cr]
        - pt_cr : Critical point coordinates
        - c_bnd : Propagation speed at critical point
        - eikmin : Eikonal value in seconds
        - z_par : Inverse of minimum Eikonal (Equivalent to c_bound / lref)
        - lref : Distance to the closest source from critical point
        - sou_cr : Critical source coordinates
    ele_pad : `int`
        Number of elements in the layer of edge length 'lmin'.
    eta_habc : `firedrake function`
        Damping profile within the absorbing layer.
    eta_mask : `firedrake function`
        Mask function to identify the absorbing layer domain.
    F_L : `float`
        Size parameter of the absorbing layer.
    FLpos : `list`
        Possible size parameters for the absorbing layer without rounding.
    freq_Nyq : `float`
        Nyquist frequency according to the time step. freq_Nyq = 1 / (2 * dt).
    freq_ref : `float`
        Reference frequency of the wave at the boundary.
    fundam_freq : `float`
        Fundamental frequency of the numerical model.
    fwi_iter : `int`
        The iteration number for the Full Waveform Inversion (FWI) algorithm.
    Lx_habc : `float`
        Length of the domain in the x-direction with absorbing layer.
    Ly_habc : `float`
        Length of the domain in the y-direction with absorbing layer (3D).
    Lz_habc : `float`
        Length of the domain in the z-direction with absorbing layer.
    lref : `float`
        Reference length for the size of the absorbing layer.
    mesh: `firedrake mesh`
        Mesh used in the simulation (HABC or Infinite Model).
    number_of_receivers: `int`
        Number of receivers used in the simulation.
    pad_len : `float`
        Size of the absorbing layer.
    path_case_habc : `string`
        Path to save data for the current case study.
    path_save : `string`
        Path to save data.
    psi_min : `float`
        Minimum damping ratio of the absorbing layer (psi_min = xCR * d).
    receiver_locations: `list`
        List of receiver locations.
    forward_solution_receivers : `array`
        Receiver waveform data in the HABC scheme.
    xCR : `float`
        Heuristic factor for the minimum damping ratio.
    xCR_lim: `list`
        Limits for the heuristic factor.

    Methods
    -------
    check_timestep_habc()
        Check if the timestep size is appropriate for the transient response.
    create_mesh_habc()
        Create a mesh with absorbing layer based on the determined size.
    critical_boundary_points()
        Determine the critical points on domain boundaries of the original
        model to size an absorbing layer using the Eikonal criterion for HABCs.
    damping_layer()
        Set the damping profile within the absorbing layer.
    det_reference_freq()
        Determine the reference frequency for a new layer size.
    fundamental_frequency()
        Compute the fundamental frequency in Hz via modal analysis.
    geometry_infinite_model()
        Determine the geometry for the infinite domain model.
    habc_domain_dimensions()
        Determine the new dimensions of the domain with absorbing layer.
    habc_new_geometry()
        Determine the new domain geometry with the absorbing layer.
    identify_habc_case()
        Generate an identifier for the current case study of the HABC scheme.
    infinite_model()
        Create a reference model for the HABC scheme for comparative purposes.
    layer_infinite_model()
        Determine the domain extension size for the infinite domain model.
    nrbc_on_boundary_layer()
        Apply the Higdon ABCs on the outer boundary of the absorbing layer.
    rename_folder_habc()
        Rename the folder of results if the degree for the hypershape
        layer is out of the criterion limits.
    size_habc_criterion()
        Determine the size of the absorbing layer using the Eikonal criterion.
    velocity_habc()
        Set the velocity model for the model with absorbing layer.
    """

    def __init__(self, domain_dim, frequency, f_Nyquist, abc_deg_layer,
                 dimension=2, quadrilateral=False, func_space_type=None,
                 abc_boundary_layer_shape=LayerShapeType.RECTANGULAR,
                 abc_reference_freq=LayerSizeRefFrequency.SOURCE,
                 abc_degree_type=HyperLayerDegreeType.REAL,
                 output_folder=None, comm=None):
        """
        Initialize the HABC class

        Parameters
        ----------
        domain_dim : `tuple`
            Original domain dimensions: (length_z, length_x) for 2D
            or (length_z, length_x, length_y) for 3D.
        frequency: `float`
            Frequency of the source.
        f_Nyquist : `float`
            Nyquist frequency according to the time step. f_Nyquist = 1 / (2 * dt).
        abc_deg_layer : `int` or `float` or `None`, optional
            Hypershape degree. For hypershape layers, the degree must be greater than or
            equal to 2. `None` is used only for rectangular layers. Default is `None`.
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D.
        quadrilateral : bool, optional
            Flag to indicate whether to use quadrilateral/hexahedral elements.
            Default is `False` (triangular/tetrahedral elements).
        func_space_type, `str`, optional
            Type of function space for the state variable.
            Options: 'scalar' or 'vector'. Default is `None`.
        abc_boundary_layer_shape : `typing.LayerShapeType`, optional
            Shape type of the pad layer. Options: `LayerShapeType.RECTANGULAR` or
            `LayerShapeType.HYPERSHAPE`. Default is `LayerShapeType.RECTANGULAR`.
        abc_reference_freq : `typing.LayerSizeRefFrequency`, optional
            Reference frequency for sizing the absorbing layer.
            Options: 'LayerSizeRefFrequency.SOURCE' or 'LayerSizeRefFrequency.BOUNDARY'.
        abc_degree_type : `typing.HyperLayerDegreeType`, optional
            Type of the hypereshape degree. Options: 'HyperLayerDegreeType.REAL' or
            'HyperLayerDegreeType.INTEGER'. Default is 'HyperLayerDegreeType.REAL'.
        output_folder : `str`, optional
            The folder where output data will be saved. Default is `None`.
        comm : `object`, optional
            An object representing the communication interface for parallel processing.
            Default is `None`.

        Returns
        -------
        None
        """

        # Initializing the ABCLayer class
        ABCLayer.__init__(self, domain_dim, frequency, f_Nyquist, dimension=dimension,
                          quadrilateral=quadrilateral, func_space_type=func_space_type,
                          abc_boundary_layer_shape=abc_boundary_layer_shape,
                          abc_boundary_layer_type=LayerDampingType.HYBRID,
                          abc_reference_freq=abc_reference_freq,
                          abc_degree_type=abc_degree_type, abc_deg_layer=abc_deg_layer,
                          output_folder=output_folder, comm=comm)

    def fundamental_frequency(self, Wave_obj, method=None, fitting_c=(0., 0., 0., 0.)):
        """Compute the fundamental frequency in Hz via modal analysis.

        Considering the numerical model with Neumann BCs.

        Parameters
        ----------
        Wave_obj : `wave.Wave`
            An instance of the :class:`~spyro.solvers.wave.Wave`.
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
            Structure: (fc1, fc2, fp1, fp2). Default is (0., 0., 0., 0.).
            - fc1 : `float`
                Exponent factor for the minimum reference velocity.
            - fc2 : `float`
                Exponent factor for the maximum reference velocity.
            - fp1 : `float`
                Exponent factor for the minimum equivalent velocity.
            - fp2 : `float`
                Exponent factor for the maximum equivalent velocity.

        Returns
        ----
        None

        Notes
        -----
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
        """

        pprint("\nSolving Eigenvalue Problem", comm=self.comm)
        mod_sol = Modal_Solver(self.dimension, method=method)

        if method == 'ANALYTICAL':

            # Hypershape parameters
            hyp_par = (self.layer_geometry.n_hyp, *self.layer_geometry.hyper_axes)

            # Cut plane at free surface
            length_z = self.domain_dim[0]
            z_cut = length_z / 2.

            # Cut plane percentage
            cut_plane_perc = z_cut / self.layer_geometry.hyper_axes[1]

            # Define the load for the energy-equivalent homogenization
            # Static load for HABC model
            q_lay = Wave_obj.set_material_property("q_lay", 'scalar', constant=0.)
            q_lay.dat.data_with_halos[Wave_obj.sources.cellNodeMaps.flatten().astype(
                int)] = Wave_obj.sources.cell_tabulations.flatten()

            # Static load for Reference model
            q_ref = Function(Wave_obj.mesh_parameters.funct_space_eik)
            q_ref.interpolate(q_lay, allow_missing_dofs=True)

            # Equivalent velocity for the original model
            c_eqref = mod_sol.c_equivalent(Wave_obj.initial_velocity_model,
                                           V=Wave_obj.mesh_parameters.funct_space_eik,
                                           quad_rule=Wave_obj.quadrature_rule,
                                           static_load_for_ceq=q_ref)

            Lsp = mod_sol.solve_eigenproblem(Wave_obj.c, V=Wave_obj.function_space,
                                             quad_rule=Wave_obj.quadrature_rule,
                                             hyp_par=hyp_par, c_eqref=c_eqref,
                                             fitting_c=fitting_c,
                                             cut_plane_percent=cut_plane_perc,
                                             static_load_for_ceq=q_lay)

        elif method == 'RAYLEIGH':

            # Normalized coordinates
            coord_norm = mod_sol.generate_norm_coords(Wave_obj.mesh,
                                                      self.domain_dim,
                                                      self.layer_geometry.hyper_axes)

            Lsp = mod_sol.solve_eigenproblem(Wave_obj.c, V=Wave_obj.function_space,
                                             quad_rule=Wave_obj.quadrature_rule,
                                             coord_norm=coord_norm)

        else:
            Lsp = mod_sol.solve_eigenproblem(
                Wave_obj.c, V=Wave_obj.function_space,
                shift=1e-8, quad_rule=Wave_obj.quadrature_rule)

        for n_eig, eigval in enumerate(unique(Lsp)):
            f_eig = sqrt(abs(eigval)) / (2 * pi)
            pprint(f"Frequency {n_eig} (Hz): {f_eig:.5f}", comm=self.comm)

        # Fundamental frequency (eig = 0 is a rigid body motion)
        min_eigval = max(unique(Lsp[(Lsp > 0.) & (imag(Lsp) == 0.)]))
        Wave_obj.fundam_freq = real(sqrt(min_eigval) / (2 * pi))
        pprint(f"Fundamental Frequency (Hz): {Wave_obj.fundam_freq:.5f}", comm=self.comm)

    # def damping_layer(self, xCR_usu=None, method=None,
    #                   fitting_c=(0., 0., 0., 0.)):
    #     """
    #     Set the damping profile within the absorbing layer.
    #     Minimum damping ratio is computed as psi_min = xCR * d
    #     where xCR is the heuristic factor for the minimum damping
    #     ratio and d is thenormalized element size (lmin / pad_len).
    #     Maximum damping ratio is psi_max =  2 * pi * f_fund * psi
    #     where f_fund is the fundamental frequency and psi = 0.999.

    #     Parameters
    #     ----------
    #     xCR_usu : `float`, optional
    #         User-defined heuristic factor for the minimum damping ratio.
    #         Default is None, which defines an estimated value
    #     method : `str`, optional
    #         Method to use for solving the eigenvalue problem.
    #         Default is None, which uses the 'KRYLOVSCH_CH' method.
    #         Opts: 'ANALYTICAL', 'ARNOLDI', 'LANCZOS', 'LOBPCG', 'KRYLOVSCH_CH',
    #         'KRYLOVSCH_CG', 'KRYLOVSCH_GH', 'KRYLOVSCH_GG' or 'RAYLEIGH'.
    #         'ANALYTICAL' method is only available for isotropic hypershapes.
    #         'RAYLEIGH' method is an approximation by Rayleigh quotient.
    #         In 'KRYLOVSCH_(K)(P)' methods, (K) indicates the Krylov solver to
    #         use: 'C' for Conjugate Gradient (cg) or 'G' for Generalized Minimal
    #         Residual (gmres). (P) indicates the preconditioner to use: 'H' for
    #         Hypre (hypre) or 'G' for Geometric Algebraic Multigrid (gamg). For
    #         example, 'KRYLOVSCH_CH' uses cg solver with hypre preconditioner.
    #     fitting_c : `tuple`, optional
    #         Parameters for fitting equivalent velocity regression.
    #         Structure: (fc1, fc2, fp1, fp2). Default is (1., 1., 0.5, 0.5)
    #         - fc1 : `float`
    #             Exponent factor for the minimum reference velocity
    #         - fc2 : `float`
    #             Exponent factor for the maximum reference velocity
    #         - fp1 : `float`
    #             Exponent fsctor for the minimum equivalent velocity
    #         - fp2 : `float`
    #             Exponent factor for the maximum equivalent velocity

    #     Returns
    #     -------
    #     None
    #     """

    #     # Domain dimensions with free surface truncation
    #     dom_lay_trunc = self.habc_domain_dimensions(only_habc_dom=True,
    #                                                 full_hyp=False)

    #     # Layer parameters
    #     layer_par = (self.F_L, self.a_par, self.d_norm)

    #     # mesh parameters
    #     mesh_par = (self.mesh_parameters.lmin, self.mesh_parameters.lmax,
    #                 self.mesh_parameters.alpha, self.variant)

    #     # wave parameters
    #     c_ref = min([bnd[1] for bnd in self.eik_bnd])
    #     c_bnd = self.eik_bnd[0][1]
    #     wave_par = (self.freq_ref, c_ref, c_bnd)

    #     # Initializing the parent class for damping
    #     HABC_Damping.__init__(self, dom_lay_trunc, layer_par, mesh_par,
    #                           wave_par, dimension=self.dimension, comm=self.comm)

    #    # Damping mask
    #    V_mask = create_function_space(self.mesh, "DG0", 0)
    #    self.eta_mask = self.layer_mask_field(coords, V_mask,
    #                                          type_marker='mask',
    #                                          name_mask='eta_mask')

    #     # Estimating fundamental frequency
    #     self.fundamental_frequency(method=method, monitor=True,
    #                                fitting_c=fitting_c)

    #     print("\nCreating Damping Profile", flush=True)

    #     # Compute the minimum damping ratio and the associated heuristic factor
    #     eta_crt, self.psi_min, self.xCR, self.xCR_lim, self.CRmin\
    #         = self.calc_damping_properties(self.fundam_freq, xCR_usu=xCR_usu)

    #     # Mesh coordinates
    #     coords = SpatialCoordinate(self.mesh)

    #     # Damping mask
    #     V_mask = FunctionSpace(self.mesh, 'DG', 0)
    #     self.eta_mask = self.layer_mask_field(coords, V_mask,
    #                                           type_marker='mask',
    #                                           name_mask='eta_mask')

    #     # Save damping mask
    #     outfile = VTKFile(self.path_case_habc + "eta_mask.pvd")
    #     outfile.write(self.eta_mask)

    #     # Compute the coefficients for quadratic damping function
    #     aq, bq = self.coeff_damp_fun(self.psi_min)

    #     # Damping field
    #     damp_par = (self.pad_len, eta_crt, aq, bq)
    #     self.eta_habc = self.layer_mask_field(
    #         coords, self.function_space, damp_par=damp_par,
    #         type_marker='damping', name_mask='eta [1/s])')

    #     # Save damping profile
    #     outfile = VTKFile(self.path_case_habc + "eta_habc.pvd")
    #     outfile.write(self.eta_habc)

    # def nrbc_on_boundary_layer(self):
    #     """
    #     Apply the Higdon ABCs on the outer boundary of the absorbing layer

    #     Parameters
    #     ----------
    #     None

    #     Returns
    #     -------
    #     None
    #     """

    #     print("\nApplying Non-Reflecting Boundary Conditions", flush=True)

    #     # Getting boundary data from the layer boundaries
    #     bnd_nfs, bnd_nodes_nfs = self.layer_boundary_data(self.function_space)

    #     # Hypershape parameters
    #     hyp_par = None if self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE \
    #         else (self.n_hyp, *self.hyper_axes)

    #     # Applying Higdon ABCs
    #     self.cos_ang_HigdonBC(self.function_space, self.crit_source,
    #                           bnd_nfs, bnd_nodes_nfs, hyp_par=hyp_par)

    # def check_timestep_habc(self, max_divisor_tf=1, set_max_dt=True,
    #                         method='ANALYTICAL', mag_add=3):
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
    #         If True, set the timestep size to the selected divisor.
    #         Default is True
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

    #     print("\nChecking Timestep Size", flush=True)

    #     # User timestep
    #     usr_dt = self.get_dt()

    #     # Maximum timestep size
    #     dt_sol = Modal_Solver(
    #         self.dimension, method=method, calc_max_dt=True)
    #     max_dt = dt_sol.estimate_timestep(
    #         self.c, self.function_space, self.final_time, shift=1e-8,
    #         quad_rule=self.quadrature_rule, fraction=1.)

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
    #     self.freq_Nyq = 1. / (2. * self.dt)

    #     print(str_dt, flush=True)

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
    #                              abs(sources_loc[:, 1]
    #                                     - self.mesh_parameters.length_x))
    #         cand_dist = (delta_z, delta_x)

    #         if self.dimension == 3:  # 3D
    #             delta_y = minimum(abs(sources_loc[:, 2]),
    #                                  abs(sources_loc[:, 2]
    #                                         - self.mesh_parameters.length_y))
    #             cand_dist += (delta_y,)

    #         # Minimum distance to the nearest boundary
    #         dist_to_bnd = min(cand_dist)

    #     # Subtracting the distance already travelled by the wave
    #     add_dom -= dist_to_bnd

    #     # Pad length for the infinite domain extension
    #     infinite_pad_len = self.mesh_parameters.lmin * \
    #         ceil(add_dom / self.mesh_parameters.lmin)

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
    #     self.pad_len = self.layer_infinite_model()

    #     inf_str = "Infinite Domain Extension (km): {:.4f}"
    #     print(inf_str.format(self.pad_len), flush=True)

    #     # New dimensions
    #     self.habc_new_geometry()

    # def infinite_model(self, check_dt=False, max_divisor_tf=1,
    #                    method='ANALYTICAL', mag_add=3):
    #     """
    #     Create a reference model for the HABC scheme for comparative purposes

    #     Parameters
    #     ----------
    #     check_dt : `bool`, optional
    #         If True, check if the timestep size is appropriate for the
    #         transient response. Default is False
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
    #         self.check_timestep_habc(
    #             max_divisor_tf=max_divisor_tf, method=method, mag_add=mag_add)

    #     print("\nBuilding Infinite Domain Model", flush=True)

    #     # Defining geometry for infinite domain
    #     self.geometry_infinite_model()

    #     # Creating mesh for infinite domain
    #     self.create_mesh_habc(inf_model=True)

    #     # Updating velocity model
    #     self.velocity_habc(inf_model=True)

    #     # Setting no damping
    #     self.cosHig = Constant(0.)
    #     self.eta_mask = Constant(0.)
    #     self.eta_habc = Constant(0.)

    #     print("\nSolving Infinite Model", flush=True)

    #     # Solving the forward problem
    #     self.forward_solve()

    #     # Saving reference signal
    #     self.save_reference_signal()

    #     # Deleting variables to be computed for the HABC scheme
    #     del self.pad_len, self.Lx_habc, self.Lz_habc
    #     del self.cosHig, self.eta_mask, self.eta_habc
    #     if self.dimension == 3:
    #         del self.Ly_habc

    def rename_folder_habc(self):
        """
        Rename the folder of results if the degree for the
        hypershape layer is out of the criterion limits

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if self.layer_geometry.n_hyp != self.abc_deg_layer and \
                self.abc_boundary_layer_shape == LayerShapeType.HYPERSHAPE:

            pprint(f"\nHypershape Degree Changes from {self.abc_deg_layer} "
                   f"to {self.layer_geometry.n_hyp}. "
                   "Output Folder for Results Will Be Renamed.", comm=self.comm)

            # Define the current and new folder names
            old = self.path_case_abc
            new = f"{self.path_case_abc[:-8]}{self.layer_geometry.n_hyp:.1f}" + \
                f"{self.path_case_abc[-5:]}"

            try:
                if path.isdir(new):
                    # Remove target directory if it exists
                    rmtree(new)

                rename(old, new)  # Rename the directory
                pprint(f"Folder '{old}' Successfully Renamed to '{new}'\n", comm=self.comm)

            except OSError as e:
                pprint(f"Error Renaming Folder: {e}", comm=self.comm)
