from firedrake import *  # noqa: F403
from FIAT.reference_element import (
    UFCTriangle,
    UFCTetrahedron,
    UFCQuadrilateral,
)
from FIAT.reference_element import UFCInterval
from FIAT import GaussLobattoLegendre as GLLelement
from FIAT.tensor_product import TensorProductElement
from FIAT.kong_mulder_veldhuizen import KongMulderVeldhuizen as KMV
from FIAT.lagrange import Lagrange as CG
from FIAT.discontinuous_lagrange import DiscontinuousLagrange as DG

import numpy as np

class Delta_projector:
    def __init__(self, wave_object):
        my_ensemble = wave_object.comm
        if wave_object.automatic_adjoint:
            self.automatic_adjoint = True
        else:
            self.automatic_adjoint = False

        self.mesh = wave_object.mesh
        self.space = wave_object.function_space.sub(0)
        self.my_ensemble = my_ensemble
        self.dimension = wave_object.dimension
        self.degree = wave_object.degree

        self.point_locations = None
        self.number_of_points = None
