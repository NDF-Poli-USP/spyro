import math
import numpy as np
from firedrake import *
from scipy.signal import butter, filtfilt
import spyro
from .Sources import Sources


class NodalSources(Sources):
    """Methods that inject a wavelet into a
    mesh directly into a node.
    """

    def __init__(self, wave_object):
        self.dof_source_locations = [0] * self.num_receivers
        self.source_locations = [0] * self.num_receivers
        super().__init__(wave_object)

    def build_maps(self):
        for rid in range(self.num_receivers):
            tolerance = 1e-6
            if self.dimension == 2:
                receiver_z, receiver_x = self.receiver_locations[rid]
                cell_id = self.mesh.locate_cell(
                    [receiver_z, receiver_x], tolerance=tolerance
                )
            elif self.dimension == 3:
                receiver_z, receiver_x, receiver_y = self.receiver_locations[
                    rid
                ]
                cell_id = self.mesh.locate_cell(
                    [receiver_z, receiver_x, receiver_y], tolerance=tolerance
                )
            self.is_local[rid] = cell_id

        (
            self.cellIDs,
            self.cellVertices,
            self.cellNodeMaps,
        ) = super.__func_receiver_locator()
        self.cell_tabulations = None

        self.num_receivers = len(self.receiver_locations)

        self.dof_source_locations = self.source_move_and_locate()

    def source_move_and_locate(self):
        for source_id in range(self.num_receivers):
            source_x, source_y = self.receiver_locations[source_id][0]
            distance_from_node = 1e3
            j = 999
            nodes = self.cellNodeMaps[source_id, :]
            for i in range(len(nodes)):
                node_id = nodes[i]
                node_x = self.node_locations[node_id, 0]
                node_y = self.node_locations[node_id, 1]
                d = math.sqrt(
                    (source_x - node_x) ** 2 + (source_y - node_y) ** 2
                )

                if d < distance_from_node:
                    distance_from_node = d
                    j = node_id
                    moved_x = node_x
                    moved_y = node_y
            if j == 999:
                raise ValueError("Source not found in cell")
            else:
                self.dof_source_locations[source_id] = j
                self.source_locations[source_id] = (moved_x, moved_y)

    def apply_source(self, rhs_forcing, value):
        """Applies source in a assembled right hand side."""
        for source_id in range(self.num_receivers):
            if self.is_local[source_id] and source_id == self.current_source:
                rhs_forcing.dat.data_with_halos[
                    int(self.dof_source_locations[source_id])
                ] = value
            else:
                for i in range(len(self.cellNodeMaps[source_id])):
                    tmp = rhs_forcing.dat.data_with_halos[0]

        return rhs_forcing
