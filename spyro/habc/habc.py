import numpy as np
from . import eikonal as eikonal_module
from . import lenCam_spy
from .damping_spy import Damping_field_calculator
from ..meshing import AutomaticMesh
from ..solvers.acoustic_wave import AcousticWave
import firedrake as fire
from firedrake import dot, grad
import finat
import scipy

# Work from Ruben Andres Salas,
# Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva, Hybrid absorbing
# scheme based on hyperelliptical layers with non-reflecting BCs in scalar wave
# equations, Applied Mathematical
# Modelling (2022), doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender


def make_eikonal_mesh(Lz, Lx, h_min):
    """
    Creates a mesh for the eikonal solver, just a wrapper for AutomaticMesh

    Parameters
    ----------
    Lz: `float`
        Length of the mesh in z direction
    Lx: `float`
        Length of the mesh in x direction
    h_min: `float`
        Minimum mesh size

    Returns
    -------
    user_mesh: `firedrake.mesh`
        Mesh for the eikonal solver
    """
    nz = int(Lz / h_min)

    nx = int(Lx / h_min)

    user_mesh = fire.RectangleMesh(nz, nx, Lz, Lx)
    user_mesh.coordinates.dat.data[:, 0] *= -1.0

    # Automatic_mesh = AutomaticMesh(dimension=2)
    # Automatic_mesh.set_mesh_size(length_z=Lz, length_x=Lx)
    # Automatic_mesh.set_meshing_parameters(dx=h_min)
    # user_mesh = Automatic_mesh.create_mesh()

    return user_mesh


def make_eikonal_function_space(user_mesh):
    return fire.FunctionSpace(user_mesh, "CG", 1)


class HABC(AcousticWave):
    """
    class HABC that determines absorbing layer size and parameters to be used

    Attributes
    ----------
    Lz: `float`
        Length of the mesh in z direction
    Lx: `float`
        Length of the mesh in x direction
    source_position: `list`
        List of source positions
    Wave: `class`
        Contains simulation parameters and options without a pad.
    dt: `float`
        Time step size
    layer_shape: `string`
        Shape type of pad layer
    nexp: `int`
        Exponent of the hyperelliptical pad layer
    h_min: `float`
        Minimum mesh size
    fwi_iteration: `int`
        Number of FWI iteration
    initial_frequency: `float`
        Initial frequency of the source wave
    reference_frequency: `float`
        Reference frequency of the source wave

    Methods
    -------
    _minimum_h_calc()
        Calculates the minimum mesh size
    _fundamental_frequency()
        Calculates the fundamental frequency
    _store_data_without_HABC()
        Stores data without HABC
    eikonal()
        Calculates the eikonal
    habc_size()
        Calculates the size of the pad layer
    get_mesh_with_pad()
        Creates the mesh with pad layer
    get_histPcrit()
        Calculates the critical pressure history

    """

    def __init__(self, dictionary=None, comm=None, h_min=None, fwi_iteration=0):
        """
        Initializes the HABC class.

        Parameters
        ----------
        dictionary : dict, optional
            A dictionary containing the input parameters for the HABC class.
        comm : object, optional
            An object representing the communication interface.
        h_min : float, optional
            The minimum value of the mesh size.
        fwi_iteration : int, optional
            The iteration number for the Full Waveform Inversion (FWI) algorithm.

        Returns
        -------
        None

        """
        super().__init__(dictionary=dictionary, comm=comm)

        self.layer_shape = "rectangular"
        if self.layer_shape == "rectangular":
            self.nexp = np.nan
        elif self.layer_shape == "hyperelliptical":
            self.nexp = 2
        else:
            UserWarning(
                f"Please use 'rectangular' or 'hyperelliptical', {self.layer_shape} not supported."
            )
        self.h_min = h_min
        self.fwi_iteration = fwi_iteration
        self.initial_frequency = self.frequency
        self.reference_frequency = self.frequency
        # Initial noneikonal (ne) related attributes
        self.with_neik_calculation = True
        self.neik_location = None
        self.neik_point_values = []
        self.neik_time_value = None
        self.neik_velocity_value = None

    def no_boundary_forward_solve(self):
        self.forward_solve()
        self.reset_pressure()
        fref, F_L, pad_length, lref = lenCam_spy.habc_size(self)
        self.abc_pad_length = pad_length
        self.neik_reference_length = lref
        self.neik_length_factor = F_L

        print("PAUSE in noboundaryforwardsolve")

    def set_damping_field(self):
        Damp_obj = Damping_field_calculator(self)

    def _fundamental_frequency(self):
        V = self.Wave.function_space
        c = self.Wave.c
        quad_rule = finat.quadrature.make_quadrature(
            V.finat_element.cell, V.ufl_element().degree(), "KMV"
        )
        dxlump = fire.dx(rule=quad_rule)
        u, v = fire.TrialFunction(V), fire.TestFunction(V)
        A = fire.assemble(u * v * dxlump)
        ai, aj, av = A.petscmat.getValuesCSR()
        # Asp = scipy.sparse.csr_matrix((av, aj, ai))
        av_inv = []
        for value in av:
            if value == 0:
                av_inv.append(0.0)
            else:
                av_inv.append(1 / value)
        Asp_inv = scipy.sparse.csr_matrix((av_inv, aj, ai))
        K = fire.assemble(c * c * dot(grad(u), grad(v)) * dxlump)
        ai, aj, av = K.petscmat.getValuesCSR()
        Ksp = scipy.sparse.csr_matrix((av, aj, ai))

        # operator
        Lsp = Asp_inv.multiply(Ksp)
        min_eigval = np.amin(np.abs(Lsp.diagonal()))

        self.fundamental_freq = np.sqrt(min_eigval) / (2 * np.pi)

        return None

    def _minimum_h_calc(self):
        diameters = fire.CellDiameter(self.mesh)
        value = fire.assemble(diameters * fire.dx)

        return value

    def eikonal(self):
        # Eik_obj = eikCrit_spy.EikonalSolve(self)
        Eik_obj = eikonal_module.Eikonal_Solve(self, show=True)
        self.eikonal = Eik_obj
        Z = Eik_obj.Z
        posCrit = Eik_obj.min_point
        cref = Eik_obj.cref
        posCrit_x, posCrit_y = posCrit
        self.posCrit = np.asarray([posCrit_x, posCrit_y])
        self.Z = Z
        self.cref = cref
        print(f"1/Z = {1/Z}") #inverse of time
        print(f"posCrit = {posCrit}") 
        print(f"cref = {cref}")

    def habc_size(self):
        fref, F_L, pad_length, lref = lenCam_spy.habc_size(self)
        print(f"L ref = {lref}")
        self.pad_length = pad_length

        # fref, F_L, pad_length = habc_size(Lz, Lx, posCrit, source_position,
        # initial_frequency, it_fwi, lmin, Z, histPcrit=None, layer_shape='REC',
        # nexp=np.nan)

    def get_mesh_with_pad(self):
        """
        Creates a new mesh with the calculated PML length
        """
        h_min = self.h_min
        pad_length = self.pad_length

        Lz = self.Lz + pad_length
        Lx = self.Lx + 2 * pad_length
        nz = int(self.Lz / h_min) + int(pad_length / h_min)
        nx = int(self.Lx / h_min) + int(2 * pad_length / h_min)
        nx = nx + nx % 2
        mesh = fire.RectangleMesh(nz, nx, Lz, Lx, diagonal="crossed")
        mesh.coordinates.dat.data[:, 0] *= -1.0
        return mesh

    def get_histPcrit(self):
        """
        Returns pressure value at critical point
        """
        if self.fwi_iteration == 0:
            return None
        else:
            return self.Receivers.interpolate(self.Wave.u_n)
