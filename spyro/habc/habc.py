import numpy as np
from . import eikCrit_spy
from . import lenCam_spy
import firedrake as fire
from copy import deepcopy
from ..receivers import Receivers
import spyro

# Work from Ruben Andres Salas,
# Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva, Hybrid absorbing scheme based on hyperel-
# liptical layers with non-reflecting boundary conditions in scalar wave equations, Applied Mathematical
# Modelling (2022), doi: https://doi.org/10.1016/j.apm.2022.09.014
# With additions by Alexandre Olender

class HABC:
    """ class HABC that determines absorbing layer size and parameters to be used
    """
    def __init__(self, Wave_object, h_min=None, it_fwi=0):
        """Initializes class and gets a wave object as an input.

        Parameters
        ----------
        Wave_object: `dictionary`
            Contains simulation parameters and options.

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
        self.eikonal()
        self.TipLay = 'rectangular'
        if self.TipLay == 'rectangular':
            self.nexp = np.nan
        elif self.TipLay == 'hyperelliptical':
            self.nexp = 2
        else:
            UserWarning(f"Please use 'rectangular' or \
                'hyperelliptical', f{self.TipLay} not supported.")
        if h_min is None:
            h_min = self._minimum_h_calc()
        self.h_min = h_min
        self.it_fwi = it_fwi
        self.initial_frequency = Wave_object.frequency
        self.habc_size()
        self.reset_mesh()

    def reset_mesh(self, mesh=None, h_min=None):
        """ Reset mesh dependent variables
        """
        if h_min is not None:
            self.h_min = h_min

        temp_wave_object = spyro.AcousticWave(
            model_parameters=self.Wave.model_parameters
            )
        x, y = self.posCrit
        temp_wave_object.model_parameters.receiver_locations = [(x, y)]
        temp_wave_object.model_parameters.number_of_receivers = 1
        self.Receivers = Receivers(temp_wave_object)

    def _minimum_h_calc(self):
        # diameters = fire.CellDiameter(self.mesh)
        # fdrake_cell_node_map = self.Wave.function_space.cell_node_map()
        # cell_node_map = fdrake_cell_node_map.values_with_halo
        # (num_cells, nodes_per_cell) = cell_node_map.shape

        pass

    def eikonal(self):
        Z, posCrit, cref = eikCrit_spy.eikonal(self.Wave)
        posCrit_x, posCrit_y = posCrit
        self.posCrit = np.asarray([posCrit_x, posCrit_y])
        self.Z = Z
        self.cref = cref

    def habc_size(self):
        fref, F_L, pad_length = lenCam_spy.habc_size(self)

        # fref, F_L, pad_length = habc_size(Lz, Lx, posCrit, possou, 
        # initial_frequency, it_fwi, lmin, Z, histPcrit=None, TipLay='REC',
        # nexp=np.nan)

    def get_histPcrit(self):
        '''
        Returns pressure value at critical point
        '''
        if self.it_fwi == 0:
            return None
        else:
            return self.Receivers.interpolate(self.Wave.u_n)
