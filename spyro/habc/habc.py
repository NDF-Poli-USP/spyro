import firedrake as fire
import numpy as np
import scipy.sparse as ss
import spyro.habc.eik as eik
import spyro.habc.lay_len as lay_len
from spyro.solvers.acoustic_wave import AcousticWave
from os import getcwd
import ipdb
fire.parameters["loopy"] = {"silenced_warnings": ["v1_scheduler_fallback"]}

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
    c_habc': 'firedrake function'
        Velocity model with absorbing layer
    eik_bnd: `list`
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
        print("Modified Parameter Size F_L: {:.4f}".format(self.F_L))
        print("Modified Layer Size (km): {:.4f}".format(self.pad_len))
        print('Elements ({:.3f} km) in Layer: {}'.format(
            self.lmin, int(self.pad_len / self.lmin)))

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
        Set the velocity model for the model with absorbing layer.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Initialize field
        print('\nUpdating Velocity Profile')
        self.c_habc = fire.Function(self.function_space, name='c [km/s])')

        # Extract node positions
        z_f = fire.Function(self.function_space).interpolate(self.mesh_z)
        x_f = fire.Function(self.function_space).interpolate(self.mesh_x)
        z_data = z_f.dat.data_with_halos[:]
        x_data = x_f.dat.data_with_halos[:]

        # Points to update velocity model
        orig_pts = np.where((z_data >= -self.length_z) & (x_data >= 0.)
                            & (x_data <= self.length_x))
        zpt_to_update = z_data[orig_pts]
        xpt_to_update = x_data[orig_pts]

        # Updating velocity model
        vel_to_update = self.c_habc.dat.data_with_halos[orig_pts]
        for idp, (zpt, xpt) in enumerate(zip(zpt_to_update, xpt_to_update)):
            vel_to_update[idp] = self.c.at(zpt, xpt)

        self.c_habc.dat.data_with_halos[orig_pts] = vel_to_update

        # Extending velocity model in absorbing layer
        print('Extending Profile Inside Layer')

        # Points to extend velocity model
        pad_pts = np.setdiff1d(np.arange(
            self.c_habc.dat.data_with_halos.size), orig_pts)
        zpt_to_extend = z_data[pad_pts]
        xpt_to_extend = x_data[pad_pts]

        # Tolerance for original boundary
        tol = 10**(min(int(np.log10(self.lmin / 10)), -6))

        vel_to_extend = self.c_habc.dat.data_with_halos[pad_pts]
        for idp, (zpt, xpt) in enumerate(zip(zpt_to_extend, xpt_to_extend)):

            # Find nearest point on the boundary of the original domain
            z_bnd = zpt
            if zpt < -self.length_z:
                z_bnd += self.pad_len + tol

            x_bnd = xpt
            if xpt < 0. or xpt > self.length_x:
                x_bnd -= np.sign(xpt) * (self.pad_len + tol)

            # Ensure that point is within the domain bounds
            z_bnd = np.clip(z_bnd, -self.length_z, 0.)
            x_bnd = np.clip(x_bnd, 0., self.length_x)

            vel_to_extend[idp] = self.c.at(z_bnd, x_bnd)

        self.c_habc.dat.data_with_halos[pad_pts] = vel_to_extend

        # Save new velocity model
        outfile = fire.VTKFile(self.path_save + "c_habc.pvd")
        outfile.write(self.c_habc)

    def fundamental_frequency(self):

        V = self.function_space
        c = self.c_habc
        u, v = fire.TrialFunction(V), fire.TestFunction(V)

        # Bilinear forms
        m = fire.inner(u, v) * fire.dx
        M = fire.assemble(m)
        m_ptr, m_ind, m_val = M.petscmat.getValuesCSR()
        mv_inv = []
        mv_inv = [0.0 if value == 0 else 1 / value for value in m_val]
        Msp_inv = ss.csr_matrix((mv_inv, m_ind, m_ptr), M.petscmat.size)
        Msp = ss.csr_matrix((m_val, m_ind, m_ptr), M.petscmat.size)

        a = c * c * fire.inner(fire.grad(u), fire.grad(v)) * fire.dx
        A = fire.assemble(a)
        a_ptr, a_ind, a_val = A.petscmat.getValuesCSR()
        Asp = ss.csr_matrix((a_val, a_ind, a_ptr), A.petscmat.size)

        # Operator
        print('\nSolving Eigenvalue Problem')
        Lsp = ss.linalg.eigs(Asp, k=2, M=Msp, sigma=0.0,
                             return_eigenvectors=False)

        # for eigval in np.unique(Lsp):
        #     print(np.sqrt(abs(eigval)) / (2 * np.pi))

        min_eigval = np.unique(Lsp[(Lsp > 0.) & (np.imag(Lsp) == 0.)])[1]
        self.fundam_freq = np.real(np.sqrt(min_eigval) / (2 * np.pi))
        print("Fundamental Frequency (Hz): {0:.5f}".format(self.fundam_freq))

    def damping_layer(self):
        '''
        Set the damping profile within the absorbing layer.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        self.fundamental_frequency()
        # print(self.fundamental_freq)
        # 0.45592718619481450 dx = 0.05
        # 0.47524802941560956 dx = 0.01

        # Homogeneous domain
        # Dirichlet     m   n   f           Neumann       Sommerfeld
        # 0.62531       1   1   0.62500     4.0023E-8i    1.9418E-5i
        # 0.90226       2   1   0.90139     0.37507       0.37508
        # 1.06960       1   2   1.06800     0.50016       0.50021
        # 1.23330       3   1   1.23111     0.62530       0.62533
        # 1.25240       2   2   1.25000     0.75052       0.75065
        # 1.50940       3   2   1.50520     0.90227       0.90234

        # Bimaterial domain

        # Dirichlet     Neumann         Sommerfeld
        # 0.72606       3.2247E-5i      2.4562E-5i
        # 1.1675        0.54933         0.54937
        # 1.2368        0.55590         0.55594
        # 1.5940        0.93186         0.93195
        # 1.6356        0.95159         0.95177
        # 1.7080        1.04420         1.04460
