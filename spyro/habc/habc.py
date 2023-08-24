import numpy as np
from . import eikonal as eikonal_module
from . import lenCam_spy
from ..meshing import AutomaticMesh
import firedrake as fire
from firedrake import dot, grad
import finat
import scipy

# Work from Ruben Andres Salas,
# Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva, Hybrid absorbing
# scheme based on hyperel-
# liptical layers with non-reflecting boundary conditions in scalar wave
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

    Automatic_mesh = AutomaticMesh(dimension=2)
    Automatic_mesh.set_mesh_size(length_z=Lz, length_x=Lx)
    Automatic_mesh.set_meshing_parameters(dx=h_min)
    user_mesh = Automatic_mesh.create_mesh()

    return user_mesh


def make_eikonal_function_space(user_mesh):
    return fire.FunctionSpace(user_mesh, "CG", 1)


class HABC:
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

    def __init__(self, Wave_object, h_min=None, fwi_iteration=0):
        """Initializes class and gets a wave object as an input.

        Parameters
        ----------
        Wave_object: `dictionary`
            Contains simulation parameters and options without a pad.

        h_min: `float`
            Minimum mesh size

        fwi_iteration: `int`
            Number of FWI iteration

        Returns
        -------
        None

        """

        # TODO: ajust weak implementation of boundary condition

        source_position = []

        for source in Wave_object.source_locations:
            z, x = source
            source_position.append(np.asarray([z, x]))

        self.length_z = Wave_object.length_z
        self.length_x = Wave_object.length_x
        self.source_position = source_position
        self.Wave = Wave_object
        self.dt = Wave_object.dt
        self.layer_shape = "rectangular"
        if self.layer_shape == "rectangular":
            self.nexp = np.nan
        elif self.layer_shape == "hyperelliptical":
            self.nexp = 2
        else:
            UserWarning(
                f"Please use 'rectangular' or \
                'hyperelliptical', f{self.layer_shape} not supported."
            )
        # print(f"h_min = {h_min}")
        # if h_min is None:
        #     h_min = self._minimum_h_calc()
        # print(f"h_min = {h_min}")
        self.h_min = h_min
        self.fwi_iteration = fwi_iteration
        self.initial_frequency = Wave_object.frequency
        self.reference_frequency = Wave_object.frequency
        print("Assuming initial mesh without pad")
        self._store_data_without_HABC()
        self.eikonal()
        self.habc_size()

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

    def _store_data_without_HABC(self):
        self.mesh_without_habc = make_eikonal_mesh(self.length_z, self.length_x, self.h_min)
        self.function_space_without_habc = make_eikonal_function_space(
            self.mesh_without_habc
        )
        self.c_without_habc = fire.project(
            self.Wave.c, self.function_space_without_habc
        )

        self.sources_without_habc = self.Wave.sources

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
        print(f"1/Z = {1/Z}")
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
