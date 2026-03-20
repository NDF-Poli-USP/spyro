import firedrake as fire
import numpy as np
from spyro.abc.abc_layer import ABC_Layer_Wave
from spyro.utils.error_management import value_parameter_error

# Work from Ruben Andres Salas and Alexandre Olender
# non-split non-convolutional PML formulation
# Formulation based on:
#   "Efficient PML for the wave equation". Grote and Sim (2010)
#   "A Modified PML Acoustic Wave Equation". Kim (2019)


class PML_Wave(ABC_Layer_Wave):
    '''
    Class PML that determines PML size and parameters to be used

    Attributes
    ----------
    bc_boundary_pml : `str`
        Type of boundary condition to apply on the PML boundaries.
        Options are "Higdon" or "Sommerfeld" for Non-Reflecting BCs,
        or "Dirichlet" for Dirichlet BCs. Default is "Higdon".
    pml_mask : `firedrake function`
        Mask function to identify the PML domain
    sigma_max : `float`
        Maximum damping coefficient within the PML layer
    sigma_x : `firedrake function`
        Damping profile in the x direction within the PML layer
    sigma_y : `firedrake function`
        Damping profile in the y direction within the PML layer (3D)
    sigma_z : `firedrake function`
        Damping profile in the z direction within the PML layer
    where_to_absorb : `tuple`
        Boundary ids where absorption is applied

    Methods
    -------
    calc_pml_damping()
        Calculate the maximum damping coefficient for the PML layer
    damping_pml_2d()
        Build damping matrices for a two-dimensional problem using PML
    damping_pml_3d()
        Build  Damping matrices for a three-dimensional problem using PML
    pml_layer()
        Set the damping profile within the PML layer
    pml_parameters_boundary_conditions()
        Set the boundary conditions for the PML layer
    pml_sigma_field()
        Generate a damping profile for the PML
    '''

    def __init__(self, dictionary=None, bc_boundary_pml="Higdon",
                 fwi_iter=0, comm=None, output_folder=None):
        '''
        Initialize the HABC class

        Parameters
        ----------
        dictionary : `dict`, optional
            A dictionary containing the input parameters for the HABC class
        bc_boundary_pml : `str`, optional
            Type of boundary condition to apply on the PML boundaries.
            Options are "Higdon" or "Sommerfeld" for Non-Reflecting BCs,
            or "Dirichlet" for Dirichlet BCs. Default is "Higdon".
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
        ABC_Layer_Wave.__init__(self, dictionary=dictionary, fwi_iter=fwi_iter,
                                comm=comm, output_folder=output_folder)

        # Type of boundary condition to apply on the PML boundaries
        self.bc_boundary_pml = bc_boundary_pml

        if self.bc_boundary_pml not in ["Higdon", "Sommerfeld", "Dirichlet"]:
            value_parameter_error('bc_boundary_pml', self.bc_boundary_pml,
                                  ["Higdon", "Sommerfeld", "Dirichlet"])

    def calc_pml_damping(self, dgr_prof=2):
        '''
        Calculate the maximum damping coefficient for the PML layer.

        Parameters
        ----------
        dgr_prof : `int`, optional
            Degree of the damping profile within the PML layer.

        Returns
        -------
        None

        '''

        # Validating input parameters
        pad_len = self.abc_pad_length
        if pad_len <= 0:
            raise ValueError(f"Invalid value for 'abc_pad_length': {pad_len}. "
                             "'abc_pad_length' must be greater than zero.")

        # Desired reflection coefficient at outer boundary of PML layer.
        CR = np.clip(self.abc_R, 1e-8, 1e-3)

        # Degree of the damping profile within the PML layer.
        dgr_prof = max(1, dgr_prof)

        # Maximum damping coefficient within the PML layer
        self.sigma_max = 0. if self.abc_get_ref_model else \
            self.c_max * (dgr_prof + 1.) / (2. * pad_len) * np.log(1. / CR)

    def pml_sigma_field(self, coords, V):
        '''
        Generate a damping profile for the PML.

        Parameters
        ----------
        coords : 'ufl.geometry.SpatialCoordinate'
            Domain Coordinates including the absorbing layer
        V : `firedrake function space`
            Function space for the mask field

        Returns
        -------
        None
        '''

        pad_len = self.abc_pad_length

        # Validating input parameters
        if pad_len <= 0:
            raise ValueError(f"Invalid value for 'pad_len': {pad_len}. "
                             "'pad_len' must be greater than zero.")

        # Domain dimensions
        Lx, Lz = self.domain_dim[:2]

        # Domain coordinates
        z, x = coords[0], coords[1]

        # Conditional value
        val_condz = (z + Lz)**2
        val_condx1 = x**2
        val_condx2 = (x - Lx)**2

        # Conditional expressions for the profile
        z_sqr = fire.conditional(z < -Lz, val_condz, 0.)
        x_sqr = fire.conditional(x < 0., val_condx1, 0.) + \
            fire.conditional(x > Lx, val_condx2, 0.)

        # Quadratic damping profiles
        ref_z = z_sqr / fire.Constant(pad_len**2)
        ref_x = x_sqr / fire.Constant(pad_len**2)
        self.sigma_z = fire.Function(V, name='sigma_z [1/s]')
        self.sigma_z.interpolate(self.pml_mask * self.sigma_max * ref_z)
        self.sigma_x = fire.Function(V, name='sigma_x [1/s]')
        self.sigma_x.interpolate(self.pml_mask * self.sigma_max * ref_x)

        # Save damping profile
        outfile = fire.VTKFile(self.path_case_abc + "sigma_pml.pvd")
        if self.dimension == 2:  # 2D
            outfile.write(self.sigma_z, self.sigma_x)

        if self.dimension == 3:  # 3D

            # 3D dimension
            Ly = self.domain_dim[2]
            y = coords[2]

            # Conditional value
            val_condy1 = y**2
            val_condy2 = (y - Ly)**2

            # Conditional expressions for the profile
            y_sqr = fire.conditional(y < 0., val_condy1, 0.) + \
                fire.conditional(y > Ly, val_condy2, 0.)

            # Quadratic damping profile
            ref_y = y_sqr / fire.Constant(pad_len**2)
            self.sigma_y = fire.Function(V, name='sigma_y [1/s]')
            self.sigma_y.interpolate(self.pml_mask * self.sigma_max * ref_y)
            outfile.write(self.sigma_z, self.sigma_x, self.sigma_y)

    def pml_parameters_boundary_conditions(self):
        '''
        Set the boundary conditions for the PML layer.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Tuple of boundary ids for NRBC
        bnds = [self.absorb_top, self.absorb_bottom,
                self.absorb_right, self.absorb_left]
        if self.dimension == 3:
            bnds.extend([self.absorb_front, self.absorb_back])

        self.where_to_absorb = tuple(np.where(bnds)[0] + 1)  # ds starts at 1

        # Apply boundary conditions to the PML boundaries.
        if not self.bc_boundary_pml == "Dirichlet":

            sommerfeld_bc = True if self.bc_boundary_pml == \
                "Sommerfeld" else False

            # Applying NRBCs on outer boundary layer
            self.nrbc_on_boundary_layer(sommerfeld_bc=sommerfeld_bc)

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

        self.calc_pml_damping()

        # Mesh coordinates
        coords = fire.SpatialCoordinate(self.mesh)

        # Damping mask
        V_mask = fire.FunctionSpace(self.mesh, 'DG', 0)
        self.pml_mask = self.layer_mask_field(coords, V_mask,
                                              type_marker='mask',
                                              name_mask='pml_mask')

        # Save damping mask
        outfile = fire.VTKFile(self.path_case_abc + "pml_mask.pvd")
        outfile.write(self.pml_mask)

        # Damping fields
        self.pml_sigma_field(coords, self.function_space)

        # Boundary conditions for the PML layer
        self.pml_parameters_boundary_conditions()

    def damping_pml_2d(self):
        '''
        Build damping matrices for a two-dimensional problem using PML.

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
        '''
        Gamma_1 = fire.as_tensor([[self.sigma_z, 0.], [0., self.sigma_x]])
        Gamma_2 = fire.as_tensor([[self.sigma_z - self.sigma_x, 0.],
                                  [0., self.sigma_x - self.sigma_z]])

        return Gamma_1, Gamma_2

    def damping_pml_3d(self):
        '''
        Build  Damping matrices for a three-dimensional problem using PML.

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
        '''
        Gamma_1 = fire.as_tensor([[self.sigma_z, 0., 0.],
                                  [0., self.sigma_x, 0.],
                                  [0., 0., self.sigma_y]])
        Gamma_2 = fire.as_tensor([[self.sigma_z - self.sigma_x - self.sigma_y, 0., 0.],
                                  [0., self.sigma_x - self.sigma_z - self.sigma_y, 0.],
                                  [0., 0., self.sigma_y - self.sigma_x - self.sigma_z]])
        Gamma_3 = fire.as_tensor([[self.sigma_x * self.sigma_y, 0., 0.],
                                  [0., self.sigma_z * self.sigma_y, 0.],
                                  [0., 0., self.sigma_z * self.sigma_x]])

        return Gamma_1, Gamma_2, Gamma_3
