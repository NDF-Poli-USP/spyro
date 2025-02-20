import firedrake as fire
import spyro.habc.eik as eik
import spyro.habc.lay_len as lay_len
from spyro.solvers.acoustic_wave import AcousticWave
import numpy as np
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
    layer_shape: `string`
        Shape type of pad layer
    nexp: `int`
        Exponent of the hyperelliptical pad layer
    fwi_iter: `int`
        The iteration number for the Full Waveform Inversion (FWI) algorithm.
    lmin: `float`
        Minimum mesh size
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
    pad_len : `float`
        Size of damping layer

    #############################################
    reference_frequency: `float`
        Reference frequency of the source wave

    Methods
    -------
    properties_eik_mesh()
        Set the properties for the mesh used to solve the Eikonal equation
    size_habc_criterion()
        Determine the size of the absorbing layer by using the Eikonal criterion



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

        p = self.degree if p_usu is None else p_usu
        self.funct_space_eik = fire.FunctionSpace(self.mesh, 'CG', p)
        diam = fire.Function(self.funct_space_eik
                             ).interpolate(fire.CellDiameter(self.mesh))
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

    def size_habc_criterion(self, layer_based_on_mesh=False):
        '''
        Determine the size of the absorbing layer by using the Eikonal
        criterion for HABCs. See Salas et al (2022) for details.

        Parameters
        ----------
        layer_based_on_mesh : `bool`, optional
            Adjust the layer size based on the element size. Default is False

        Returns
        -------
        None
        '''

        # Solving Eikonal
        Eik_obj = eik.Eikonal(self)
        Eik_obj.solve_eik(self)

        # Identifying critical points
        self.eik_bnd = Eik_obj.ident_crit_eik(self)

        # Computing layer sizes
        self.F_L, self.pad_len = lay_len.calc_size_lay(self)

        if layer_based_on_mesh:
            self.roundFL()

    def create_mesh_habc(self):
        '''
        Creates a new mesh with the calculated layer size
        '''

        Lz = self.length_z + self.pad_len
        Lx = self.length_x + 2 * self.pad_len
        nz = int(self.length_z / self.lmin) + int(self.pad_len / self.lmin)
        nx = int(self.length_x / self.lmin) + int(2 * self.pad_len / self.lmin)
        nx = nx + nx % 2

        mesh_habc = fire.RectangleMesh(nz, nx, Lz, Lx)
        mesh_habc.coordinates.dat.data[:, 0] *= -1.0
        mesh_habc.coordinates.dat.data[:, 1] -= self.pad_len

        outfile = fire.VTKFile("/mnt/d/spyro/output/mesh_habc.pvd")
        outfile.write(mesh_habc)
