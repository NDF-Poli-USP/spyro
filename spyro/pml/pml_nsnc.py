from firedrake import as_tensor, conditional, Function, VTKFile
from numpy import log, where
from ..abc.abc_layer import ABCLayer
from ..domains.space import create_function_space
from ..io.basicio import parallel_print as pprint
from ..tools.habc_tools import layer_mask_field
from ..utils.error_management import enum_parameter_error, value_numerical_error
from ..utils.eval_functions_to_ufl import generate_ufl_functions
from ..utils.typing import (BoundaryConditionsType, LayerDampingType,
                            LayerShapeType, LayerSizeRefFrequency)


# Work from Ruben Andres Salas and Alexandre Olender
# non-split non-convolutional PML formulation
# Formulation based on:
#   "Efficient PML for the wave equation". Grote and Sim (2010)
#   "A Modified PML Acoustic Wave Equation". Kim (2019)
# TODO: Add citation


class PMLLayer(ABCLayer):
    """Class PML that determines PML size and parameters to be used.

    Attributes
    ----------
    abc_pml_cmax: float
        Maximum propagation speed (km/s) in the PML layer. Default is 4.7 km/s.
    abc_pml_exponent: int
        Exponent for the polynomial damping profile of the PML layer. Default is 2.
    abc_pml_R: float
        Theoretical reflection coefficient of the PML layer. Default is 1e-6.
    bc_boundary_pml : `typing.BoundaryConditionsType`, optional
        Type of boundary condition to apply on the PML boundaries.
        - Options for Non-Reflecting BCs:
            'BoundaryConditionsType.HIGDON' or 'BoundaryConditionsType.SOMMERFELD'.
        - Options for typical BCs:
            'BoundaryConditionsType.DIRICHLET' or 'BoundaryConditionsType.NEUMANN'
        Default is 'BoundaryConditionsType.NEUMANN'.
    pml_mask : `Firedrake.Function`
        Mask function to identify the PML domain.
    sigma_max : `float`
        Maximum damping coefficient within the PML layer.
    sigma_x : `Firedrake.Function`
        Damping profile in the x direction within the PML layer.
    sigma_y : `Firedrake.Function`
        Damping profile in the y direction within the PML layer (3D).
    sigma_z : `Firedrake.Function`
        Damping profile in the z direction within the PML layer.
    where_to_absorb : `tuple`
        Boundary ids where absorption is applied.

    Methods
    -------
    calc_pml_damping()
        Calculate the maximum damping coefficient for the PML layer.
    damping_pml_2d()
        Build damping matrices for a two-dimensional problem using PML.
    damping_pml_3d()
        Build  Damping matrices for a three-dimensional problem using PML.
    pml_layer()
        Set the damping profile within the PML layer.
    pml_sigma_field()
        Generate a damping profile for the PML.
    """

    def __init__(self, domain_dim, frequency, f_Nyquist, dimension=2,
                 quadrilateral=False, func_space_type=None,
                 bc_boundary_pml=BoundaryConditionsType.NEUMANN,
                 abc_reference_freq=LayerSizeRefFrequency.SOURCE,
                 output_folder=None, comm=None):
        """
        Initialize the PML class.

        Parameters
        ----------
        domain_dim : `tuple`
            Original domain dimensions: (length_z, length_x) for 2D
            or (length_z, length_x, length_y) for 3D.
        frequency: `float`
            Frequency of the source.
        f_Nyquist : `float`
            Nyquist frequency according to the time step. f_Nyquist = 1 / (2 * dt).
        dimension : `int`, optional
            Model dimension (2D or 3D). Default is 2D.
        quadrilateral : bool, optional
            Flag to indicate whether to use quadrilateral/hexahedral elements.
            Default is `False` (triangular/tetrahedral elements).
        func_space_type, `str`, optional
            Type of function space for the state variable.
            Options: 'scalar' or 'vector'. Default is `None`.
        bc_boundary_pml : `typing.BoundaryConditionsType`, optional
            Type of boundary condition to apply on the PML boundaries.
            - Options for Non-Reflecting BCs:
                'BoundaryConditionsType.HIGDON' or  'BoundaryConditionsType.SOMMERFELD'.
            - Options for typical BCs:
                'BoundaryConditionsType.DIRICHLET' or 'BoundaryConditionsType.NEUMANN'
            Default is 'BoundaryConditionsType.HIGDON'.
        abc_reference_freq : `typing.LayerSizeRefFrequency`, optional
            Reference frequency for sizing the absorbing layer.
            Options: 'LayerSizeRefFrequency.SOURCE' or 'LayerSizeRefFrequency.BOUNDARY'.
            Default is 'LayerSizeRefFrequency.SOURCE'.
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
                          abc_boundary_layer_shape=LayerShapeType.RECTANGULAR,
                          abc_boundary_layer_type=LayerDampingType.PML,
                          abc_reference_freq=abc_reference_freq,
                          output_folder=output_folder, comm=comm)

        # Type of boundary condition to apply on the PML boundaries
        self.bc_boundary_pml = enum_parameter_error('bc_boundary_pml', bc_boundary_pml,
                                                    BoundaryConditionsType)

    def calc_pml_damping(self, abc_pml_R, abc_pml_cmax, abc_pad_length,
                         degree_prof=2, CR_min=1e-8, CR_max=1e-3):
        """Calculate the maximum damping coefficient for the PML layer.

        Parameters
        ----------
        abc_pml_R: float
            Theoretical reflection coefficient of the PML layer.
        abc_pml_cmax: float
            Maximum propagation speed (km/s) in the PML layer.
        abc_pad_length : `float`
            Size of the absorbing layer
        degree_prof : `int`, optional
            Degree of the damping profile within the PML layer. Default is 2.
        CR_min : `float`, optional
            Minimum value for the desired reflection coefficient at outer
            boundary of PML layer. Default is 1e-8.
        CR_max : `float`, optional
            Maximum value for the desired reflection coefficient at outer
            boundary of PML layer. Default is 1e-3.

        Returns
        -------
        None
        """

        # Desired reflection coefficient at outer boundary of PML layer.
        CR = value_numerical_error("abc_pml_R", abc_pml_R, float_num=True,
                                   lower_bound=CR_min, upper_bound=CR_max,
                                   include_lower_bound=True, include_upper_bound=True)

        # Degree of the damping profile within the PML layer.
        degree_prof = value_numerical_error("degree_prof", degree_prof, integer_num=True,
                                            lower_bound=1, include_lower_bound=True)

        # Maximum damping coefficient within the PML layer
        self.sigma_max = ((degree_prof + 1.) / (2. * abc_pad_length)) * log(1. / CR)
        self.sigma_max *= abc_pml_cmax

    def pml_sigma_field(self, Wave_object, ufl_coordinates_pml, V,
                        degree_prof=2, save_file=True):
        """Generate a damping profile for the PML.

        Parameters
        ----------
        Wave_object : `acoustic_wave.AcousticWave`
            An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
        ufl_coordinates_pml : `ufl.geometry.SpatialCoordinate`
            Domain Coordinates including the absorbing layer.
        V : `Firedrake.FunctionSpace`
            Function space for the damping field
        degree_prof : `int`, optional
            Degree of the damping profile within the PML layer. Default is 2.
        save_file : `bool`, optional
            If `True`, save the mesh with absorbing layer in a .pvd file.
            Default is `True`.

        Returns
        -------
        None
        """

        # Domain dimensions
        length_z, length_x = self.domain_dim[:2]

        # UFL coordinates
        z, x = ufl_coordinates_pml[0], ufl_coordinates_pml[1]

        # Conditional expression
        condz = z < -length_z
        condx1 = x < 0.
        condx2 = x > length_x

        # Conditional value
        exprz = f"sqrt((z + {length_z})**2)"
        exprx1 = "(sqrt(x))**2"
        exprx2 = f"sqrt((x - {length_x})**2)"
        valz = generate_ufl_functions(Wave_object.mesh, exprz, self.dimension)
        valx1 = generate_ufl_functions(Wave_object.mesh, exprx1, self.dimension)
        valx2 = generate_ufl_functions(Wave_object.mesh, exprx2, self.dimension)

        # Conditional expressions for the profile
        z_pd = conditional(condz, valz, 0.)
        x_pd = conditional(condx1, valx1, 0.) + conditional(condx2, valx2, 0.)

        # Damping profiles
        ref_z = (z_pd / self.abc_pad_length) ** degree_prof
        ref_x = (x_pd / self.abc_pad_length) ** degree_prof
        self.sigma_z = Function(V, name='sigma_z[1/s]')
        self.sigma_z.interpolate(self.pml_mask * self.sigma_max * ref_z)
        self.sigma_x = Function(V, name='sigma_x[1/s]')
        self.sigma_x.interpolate(self.pml_mask * self.sigma_max * ref_x)

        if self.dimension == 3:  # 3D

            # 3D dimension
            length_y = self.domain_dim[2]
            y = ufl_coordinates_pml[2]

            # Conditional expression
            condy1 = y < 0.
            condy2 = y > length_y

            # Conditional value
            expry1 = "(sqrt(y))**2"
            expry2 = f"sqrt((y - {length_y})**2)"
            valy1 = generate_ufl_functions(Wave_object.mesh, expry1, self.dimension)
            valy2 = generate_ufl_functions(Wave_object.mesh, expry2, self.dimension)

            # Conditional expressions for the mask
            y_pd = conditional(condy1, valy1, 0.) + conditional(condy2, valy2, 0.)

            # Damping profile
            ref_y = (y_pd / self.abc_pad_length) ** degree_prof
            self.sigma_y = Function(V, name='sigma_y[1/s]')
            self.sigma_y.interpolate(self.pml_mask * self.sigma_max * ref_y)

        # Save damping profile
        if not Wave_object.abc_get_ref_model and save_file:
            outfile = VTKFile(self.path_case_abc + "sigma_pml.pvd")
            if self.dimension == 2:  # 2D
                outfile.write(self.sigma_z, self.sigma_x)
            if self.dimension == 3:  # 3D
                outfile.write(self.sigma_z, self.sigma_x, self.sigma_y)

    def pml_layer(self, Wave_object, save_file=True):
        """Set the damping profile within the PML layer.

        Parameters
        ----------
        Wave_object : `acoustic_wave.AcousticWave`
            An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
        save_file : `bool`, optional
            If `True`, save the mesh with absorbing layer in a .pvd file.
            Default is `True`.

        Returns
        -------
        None
        """

        pprint("\nCreating Damping PML Profile", comm=self.comm)

        # New geometry with layer if pad_length is provided by the user.
        if Wave_object.abc_user_pad_len:
            self.abc_pad_length = Wave_object.abc_pad_length
            self.abc_new_geometry()

        # Compute the maximum damping coefficient
        abc_pml_cmax = Wave_object.abc_pml_cmax if Wave_object.abc_user_pml_cmax \
            else Wave_object.c_bnd_max

        self.calc_pml_damping(Wave_object.abc_pml_R, abc_pml_cmax,
                              Wave_object.abc_pad_length,
                              degree_prof=Wave_object.abc_pml_exponent)

        # Scalar space for mask
        method_element = "DQ" if self.quadrilateral else "DG"
        V = create_function_space(Wave_object.mesh, method_element, 0)

        # Mesh coordinates including the absorbing layer
        domain_layer = Wave_object.layer_ops.abc_domain_dimensions(full_hyp=False)
        ufl_coordinates_pml = \
            Wave_object.mesh_ops.get_spatial_coordinates_abc(Wave_object.mesh,
                                                             domain_layer)

        # Damping mask
        self.pml_mask = layer_mask_field(self.domain_dim, Wave_object.mesh,
                                         Wave_object.dimension, ufl_coordinates_pml,
                                         V, type_marker="mask", name_mask='pml_mask')

        # Save damping mask
        save_file = save_file and not Wave_object.abc_get_ref_model
        if save_file:
            outfile = VTKFile(self.path_case_abc + "pml_mask.pvd")
            outfile.write(self.pml_mask)

        # Damping fields
        self.pml_sigma_field(Wave_object, ufl_coordinates_pml, V,
                             degree_prof=Wave_object.abc_pml_exponent)

        # Non-Reflective BCs for the PML layer if prescribed
        self.nrbc_on_boundary_layer(Wave_object, self.bc_boundary_pml, save_file=save_file)

    def damping_pml_2d(self):
        """Build damping matrices for a two-dimensional problem using PML.

        Parameters
        ----------
        sigma_z: Firedrake 'Function'
            Damping profile in the z direction
        sigma_x: Firedrake 'Function'
            Damping profile in the x direction

        Returns
        -------
        Gamma_1: Firedrake 'TensorFunction'
            First damping matrix
        Gamma_2: Firedrake 'TensorFunction'
            Second damping matrix
        """
        Gamma_1 = as_tensor([[self.sigma_z, 0.], [0., self.sigma_x]])
        Gamma_2 = as_tensor([[self.sigma_z - self.sigma_x, 0.],
                             [0., self.sigma_x - self.sigma_z]])

        return Gamma_1, Gamma_2

    def damping_pml_3d(self):
        """Build  Damping matrices for a three-dimensional problem using PML.

        Parameters
        ----------
        None

        Returns
        -------
        Gamma_1: Firedrake 'TensorFunction'
            First damping matrix
        Gamma_2: Firedrake 'TensorFunction'
            Second damping matrix
        Gamma_3: Firedrake 'TensorFunction'
            Third damping matrix
        """
        Gamma_1 = as_tensor([[self.sigma_z, 0., 0.],
                             [0., self.sigma_x, 0.],
                             [0., 0., self.sigma_y]])
        Gamma_2 = as_tensor([[self.sigma_z - self.sigma_x - self.sigma_y, 0., 0.],
                             [0., self.sigma_x - self.sigma_z - self.sigma_y, 0.],
                             [0., 0., self.sigma_y - self.sigma_x - self.sigma_z]])
        Gamma_3 = as_tensor([[self.sigma_x * self.sigma_y, 0., 0.],
                             [0., self.sigma_z * self.sigma_y, 0.],
                             [0., 0., self.sigma_z * self.sigma_x]])

        return Gamma_1, Gamma_2, Gamma_3
