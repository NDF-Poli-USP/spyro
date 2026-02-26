import firedrake as fire
import warnings
import os

from .wave import Wave

from ..io.basicio import ensemble_gradient
from ..io import interpolate
from .acoustic_elastic_solver_construction_no_pml import {
    construct_acoustic_elastic_solver_no_pml,
    }
from .backward_time_integration import {
    backward_wave_propagator,
    }
from ..domains.space import create_function_space
from ..utils.typing import override
from .functionals import acoustic_energy

try:
    from SeismicMesh import write_velocity_model
    SEISMIC_MESH_AVAILABLE = True
except ImportError:
    SEISMIC_MESH_AVAILABLE = False


class AcousticElasticWave(Wave):
    def __init__(self, dictionary, comm=None):
        super().__init__(dictionary, comm=comm)

        self.interface_id = dictionary.get("interface_id", 3)

        self.acoustic_part = AcousticWave(dictionary, comm=comm)
        self.elastic_part = IsotropicWave(dictionary, comm=comm)

    @override
    def _create_function_space(self):
        self.acoustic_part.function_space = create_functionn_space(self.mesh, "CG", self.degree)
        self.elastic_part.function_space = create_function_space(self.mesh, "CG", self.degree, dim=self.dimension)
        return self.acoustic_part.function_space

    @override
    def matrix_building(self):
        self.current_time = 0.0

        self.acoustic_part.matrix_building()
        self.elastic_part.matrix_building()

        
