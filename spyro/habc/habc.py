import firedrake as fire
import spyro.habc.eik as eik
import spyro.habc.lay_len as lay_len
from spyro.solvers.acoustic_wave import AcousticWave
import numpy as np
from os import getcwd
import ipdb

# Work from Ruben Andres Salas, Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva.
# Hybrid absorbing scheme based on hyperelliptical layers with
# non-reflecting boundary conditions in scalar wave equations.
# Applied Mathematical Modelling (2022)
# doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


class HABC_Wave(AcousticWave):
    '''
    class HABC that determines absorbing layer size and parameters to be used

    Attributes
    ----------
    eik_bnd : `list`
        Properties on boundaries according to minimum values of Eikonal
        Structure sublist: [pt_cr, c_bnd, eikmin, z_par, lref, sou_cr]
        - pt_cr: Critical point coordinates
        - c_bnd: Propagation speed at critical point
        - eikmin: Eikonal value in seconds
        - z_par: Inverse of minimum Eikonal (Equivalent to c_bound/lref)
        - lref: Distance to the closest source from critical point
        - sou_cr: Critical source coordinates
    F_L : `float`
        Size  parameter of the absorbing layer
    fwi_iter: `int`
        The iteration number for the Full Waveform Inversion (FWI) algorithm.
    layer_shape: `string`
        Shape type of pad layer
    lmin: `float`
        Minimum mesh size
    nexp: `int`
        Exponent of the hyperelliptical pad layer
    pad_len : `float`
        Size of damping layer

    #############################################
    reference_frequency: `float`
        Reference frequency of the source wave
    #############################################

    Methods
    -------
    create_mesh_habc()
        Creates a mesh with absorbing layer based on the determined size
    properties_eik_mesh()
        Set the properties for the mesh used to solve the Eikonal equation
    roundFL()
        Adjust the layer parameter based on the element size
    size_habc_criterion()
        Determine the size of the absorbing layer using the Eikonal criterion
    velocity_habc()
        Set the velocity model fir the model with absorbing layer.
    '''

    def __init__(self, dictionary=None, comm=None,
                 layer_shape='rectangular', fwi_iteration=0):
        '''
        Initializes the HABC class.

        Parameters
        ----------
        dictionary : dict, optional
            A dictionary containing the input parameters for the HABC class
        comm : object, optional
            An object representing the communication interface
        fwi_iter: int, optional
            The iteration number for the FWI algorithm

        Returns
        -------
        None

        '''

        super().__init__(dictionary=dictionary, comm=comm)

        # Layer shape
        self.layer_shape = 'rectangular'
        if self.layer_shape == 'rectangular':
            self.nexp = np.nan
        elif self.layer_shape == 'hyperelliptical':
            self.nexp = 2
        else:
            aux0 = "Please use 'rectangular' or 'hyperelliptical', "
            UserWarning(aux0 + f"{self.layer_shape} not supported.")

        self.fwi_iteration = fwi_iteration

        # Path to save data
        self.path_save = getcwd() + "/output/"

    def properties_eik_mesh(self, p_usu=None):
        '''
        Set the properties for the mesh used to solve the Eikonal equation

        Parameters
        ----------
        p_usu : `int`
            Finite element order

        Returns
        -------
        None
        '''

        # Setting the properties of the mesh used to solve the Eikonal equation
        p = self.degree if p_usu is None else p_usu
        self.funct_space_eik = fire.FunctionSpace(self.mesh, 'CG', p)

        # Mesh cell diameters
        self.diam_mesh = fire.CellDiameter(self.mesh)

        if self.fwi_iteration == 0:
            # Minimum mesh size
            diam = fire.Function(
                self.funct_space_eik).interpolate(self.diam_mesh)
            self.lmin = round(diam.dat.data_with_halos.min() / 2**0.5, 6)

    def roundFL(self):
        '''
        Adjust the layer parameter based on the element size to get
        an integer number of elements within the layer.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Adjusting the parameter size of the layer
        lref = self.eik_bnd[0][4]
        self.F_L = (self.lmin / lref) * np.ceil(lref * self.F_L / self.lmin)

        # New size of the absorving layer
        self.pad_len = self.F_L * lref

        print('\nModifying Layer Size Based on the Element Size')
        print('Modified Parameter Size F_L:', round(self.F_L, 4))
        print('Modified Layer Size (km):', round(self.pad_len, 4))
        print('Elements (' + str(self.lmin), 'km) in Layer:',
              int(self.pad_len / self.lmin))

    def size_habc_criterion(self, Eikonal, layer_based_on_mesh=False):
        '''
        Determine the size of the absorbing layer using the Eikonal
        criterion for HABCs. See Salas et al (2022) for details.

        Parameters
        ----------
        Eikonal : `eikonal`
            An object representing the Eikonal solver
        layer_based_on_mesh : `bool`, optional
            Adjust the layer size based on the element size. Default is False

        Returns
        -------
        None
        '''

        # Eikonal boundary conditions
        Eikonal.define_bcs(self)

        # Solving Eikonal
        Eikonal.solve_eik(self)

        # Identifying critical points
        self.eik_bnd = Eikonal.ident_crit_eik(self)

        # Computing layer sizes
        self.F_L, self.pad_len = lay_len.calc_size_lay(self)

        if layer_based_on_mesh:
            self.roundFL()

    def create_mesh_habc(self):
        '''
        Creates a mesh with absorbing layer based on the determined size.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # New geometry with layer
        Lz = self.length_z + self.pad_len
        Lx = self.length_x + 2 * self.pad_len
        nz = int(self.length_z / self.lmin) + int(self.pad_len / self.lmin)
        nx = int(self.length_x / self.lmin) + int(2 * self.pad_len / self.lmin)
        nx = nx + nx % 2

        # New mesh with layer
        mesh_habc = fire.RectangleMesh(nz, nx, Lz, Lx)
        mesh_habc.coordinates.dat.data_with_halos[:, 0] *= -1.0
        mesh_habc.coordinates.dat.data_with_halos[:, 1] -= self.pad_len

        # Updating the mesh with the absorbing layer
        self.set_mesh(user_mesh=mesh_habc, mesh_parameters={})

        # Save new mesh
        outfile = fire.VTKFile(self.path_save + "mesh_habc.pvd")
        outfile.write(self.mesh)

    def velocity_habc(self):
        '''
        Set the velocity model fir the model with absorbing layer.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # cond = fire.conditional(self.mesh_x < 0.5, 3.0, 1.5)
        # self.set_initial_velocity_model(conditional=cond)
        # self.c_habc = self.initial_velocity_model

        # Initialize field
        self.c_habc = fire.Function(self.function_space, name='c [km/s])')

        # Extract node positions
        z_f = fire.Function(self.function_space).interpolate(self.mesh_z)
        x_f = fire.Function(self.function_space).interpolate(self.mesh_x)
        z_data = z_f.dat.data_with_halos[:]
        x_data = x_f.dat.data_with_halos[:]

        # Points to update
        orig_pts = np.where((z_data >= -self.length_z) & (x_data >= 0.)
                            & (x_data <= self.length_x))
        zpt_to_update = z_data[orig_pts]
        xpt_to_update = x_data[orig_pts]

        # Updating velocity model
        vel_to_update = self.c_habc.dat.data_with_halos[orig_pts]
        for idp, (zpt, xpt) in enumerate(zip(zpt_to_update, xpt_to_update)):
            vel_to_update[idp] = self.c.at(zpt, xpt)

        self.c_habc.dat.data_with_halos[orig_pts] = vel_to_update

        # Save new velocity model
        outfile = fire.VTKFile(self.path_save + "c_habc.pvd")
        outfile.write(self.c_habc)
