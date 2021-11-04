import math
import numpy as np
from copy import deepcopy

import firedrake as fire

import spyro

def test_marmousi_mesh():
    vp_filename = "velocity_models/vp_marmousi-ii.segy"

    model = {}
    model["opts"] = {
        "method": "KMV",  # either CG or KMV
        "quadratrue": "KMV",  # Equi or KMV
        "degree": 4,  # p order
        "dimension": 2,  # dimension
    }
    model["mesh"] = {
        "Lz": 3.5,  # depth in km - always positive
        "Lx": 17.0,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
    }
    model["parallelism"] = {
        "type": "spatial",
    }
    model["BCs"] = {
        "status": True,  # True or false
        "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
        "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
        "exponent": 2,  # damping layer has a exponent variation
        "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
        "R": 1e-6,  # theoretical reflection coefficient
        "lz": 0.287,  # thickness of the PML in the z-direction (km) - always positive
    }

    comm = spyro.utils.mpi_init(model)
    mesh = spyro.build_mesh(model,comm,'test_mesh',vp_filename=vp_filename)

    return True

if __name__ == "__main__":
    test_marmousi_mesh()
