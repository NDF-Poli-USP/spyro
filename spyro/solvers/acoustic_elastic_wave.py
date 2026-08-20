import firedrake as fire
import warnings
import os

from .wave import Wave
from .acoustic_elastic_solver_no_pml import construct_acoustic_elastic_no_pml # to implement
from ..utils.typing import override, WaveType
from ..domains.space import create_function_space
from ..domains.quadrature import quadrature_rules
from ..plots.general_plots import plot acoustic_elastic_snapshot # to implement

def _extract_interface_markers(parent_mesh, child_mesh):
    parent_mesh = {int(m) for m in parent_mesh.exterior_facets.unique_markers}
    child_mesh  = {int(m) for m in child_mesh.exterior_facets.unique_markers}
    return tuple(sorted(child_mesh - parent_mesh))

class AcousticElasticWave(Wave):
    def __init__(self, dictionary, comm=None):
        self.acoustic_id = dictionary["mesh"].get("acoustic_id", 1)
        self.elastic_id = dictionary["mesh"].get("elastic_id", 2)
        self.interface_x = dictionary["mesh"].get("interface_x", None)

        super().__init__(dictionary, comm=comm)
        self.wave_type = WaveType.ISOTROPIC_ACOUSTIC_ELASTIC # to implement
