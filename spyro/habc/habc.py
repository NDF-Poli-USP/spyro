import numpy as np
from . import eikCrit_spy
from . import lenCam_spy
import firedrake as fire
from firedrake import dot, grad, dx
from copy import deepcopy
from ..receivers import Receivers
import spyro
import pickle
import finat
import scipy
# Work from Ruben Andres Salas,
# Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva, Hybrid absorbing scheme based on hyperel-
# liptical layers with non-reflecting boundary conditions in scalar wave equations, Applied Mathematical
# Modelling (2022), doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender

class HABC:
    """ class HABC that determines absorbing layer size and parameters to be used
    """
    def __init__(self, Wave_object, h_min=None, it_fwi=0, skip_eikonal=False):
        """Initializes class and gets a wave object as an input.

        Parameters
        ----------
        Wave_object: `dictionary`
            Contains simulation parameters and options without a pad.

        histPcrit: numpy array
            Pressure value history in critical point

        initial_freqquency : float
            FWI frequency

        it_fwi: int
            Current FWI iteration


        Returns
        -------
        pad_length; size of absorbing layer

        """

        # TODO: ajust weak implementation of boundary condition

        possou = []

        for source in Wave_object.model_parameters.source_locations:
            z, x = source
            possou.append(np.asarray([z, x]))

        self.Lz = Wave_object.length_z
        self.Lx = Wave_object.length_x
        self.possou = possou
        self.Wave = Wave_object
        self.dt = Wave_object.dt
        self.TipLay = 'rectangular'
        if self.TipLay == 'rectangular':
            self.nexp = np.nan
        elif self.TipLay == 'hyperelliptical':
            self.nexp = 2
        else:
            UserWarning(f"Please use 'rectangular' or \
                'hyperelliptical', f{self.TipLay} not supported.")
        print(f"h_min = {h_min}")
        if h_min is None:
            h_min = self._minimum_h_calc()
        print(f"h_min = {h_min}")
        self.h_min = h_min
        self.it_fwi = it_fwi
        self.initial_frequency = Wave_object.frequency
        print("Assuming initial mesh without pad")
        self._store_data_without_HABC()
        # if not skip_eikonal:
        self.eikonal()
        # elif:
        #     self._fundamental_frequency()

        self.habc_size()
        # self.reset_mesh()
        # self.get_mesh_with_pad()

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
        Asp = scipy.sparse.csr_matrix((av, aj, ai))
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
        self.mesh_without_habc = self.Wave.mesh
        self.c_without_habc = self.Wave.c
        self.function_space_without_habc = self.Wave.function_space
        self.sources_without_habc = self.Wave.sources

    # def reset_mesh(self, mesh=None, h_min=None):
    #     """ Reset mesh dependent variables
    #     """
    #     if h_min is not None:
    #         self.h_min = h_min

    #     temp_wave_object = spyro.AcousticWave(
    #         model_parameters=self.Wave.model_parameters
    #         )
    #     x, y = self.posCrit
    #     temp_wave_object.model_parameters.receiver_locations = [(x, y)]
    #     temp_wave_object.model_parameters.number_of_receivers = 1
    #     self.Receivers = Receivers(temp_wave_object)

    def _minimum_h_calc(self):
        # diameters = fire.CellDiameter(self.mesh)
        # fdrake_cell_node_map = self.Wave.function_space.cell_node_map()
        # cell_node_map = fdrake_cell_node_map.values_with_halo
        # (num_cells, nodes_per_cell) = cell_node_map.shape

        pass

    def eikonal(self):
        Z, posCrit, cref = eikCrit_spy.eikonal(self)
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

        # fref, F_L, pad_length = habc_size(Lz, Lx, posCrit, possou, 
        # initial_frequency, it_fwi, lmin, Z, histPcrit=None, TipLay='REC',
        # nexp=np.nan)

    def get_mesh_with_pad(self):
        """ 
        Creates a new mesh with the calculated PML length
        """
        h_min = self.h_min
        pad_length = self.pad_length

        Lz = self.Lz + pad_length
        Lx = self.Lx + 2*pad_length
        nz = int(self.Lz / h_min) + int(pad_length / h_min)
        nx = int(self.Lx / h_min) + int(2 * pad_length / h_min)
        nx = nx + nx % 2
        mesh = fire.RectangleMesh(nz, nx, Lz, Lx, diagonal="crossed")
        mesh.coordinates.dat.data[:, 0] *= -1.0
        return mesh

    def get_histPcrit(self):
        '''
        Returns pressure value at critical point
        '''
        if self.it_fwi == 0:
            return None
        else:
            return self.Receivers.interpolate(self.Wave.u_n)
