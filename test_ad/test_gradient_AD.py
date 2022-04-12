from firedrake import *
from scipy.optimize import *
import pytest
import spyro
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import meshio
import SeismicMesh
import finat
import pytest

#from ..domains import quadrature, space
# @pytest.mark.skip(reason="no way of currently testing this")
@pytest.mark.mpi_skip()
def test_gradient_AD():
    model = {}

    model["opts"] = {
        "method": "KMV",  # either CG or KMV
        "quadrature": "KMV", # Equi or KMV
        "degree": 1,  # p order
        "dimension": 2,  # dimension
        "regularization": False,  # regularization is on?
        "gamma": 1e-5, # regularization parameter
    }

    model["parallelism"] = {
        "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
    }

    # Define the domain size without the ABL.
    model["mesh"] = {
        "Lz": 1.5,  # depth in km - always positive
        "Lx": 1.5,  # width in km - always positive
        "Ly": 0.0,  # thickness in km - always positive
        "meshfile": "not_used.msh",
        "initmodel": "not_used.hdf5",
        "truemodel": "not_used.hdf5",
    }

    # Specify a 250-m Absorbing Boundary Layer (ABL) on the three sides of the domain to damp outgoing waves.
    model["BCs"] = {
        "status": False,  # True or False, used to turn on any type of BC
        "outer_bc": "non-reflective", #  none or non-reflective (outer boundary condition)
        "abl_bc": "none",  # none, gaussian-taper, or alid
        "lz": 0.25,  # thickness of the ABL in the z-direction (km) - always positive
        "lx": 0.25,  # thickness of the ABL in the x-direction (km) - always positive
        "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
    }

    model["acquisition"] = {
        "source_type": "Ricker",
        "source_pos": [(0.75, 0.75)],
        "frequency": 10.0,
        "delay": 1.0,
        "receiver_locations": spyro.create_transect(
        (0.9, 0.2), (0.9, 0.8), 10
        ),
    }
    model["aut_dif"] = {
        "status": True, 
    }

    model["timeaxis"] = {
        "t0": 0.0,  #  Initial time for event
        "tf": 0.8,  # Final time for event (for test 7)
        "dt": 0.001,  # timestep size (divided by 2 in the test 4. dt for test 3 is 0.00050)
        "amplitude": 1,  # the Ricker has an amplitude of 1.
        "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
        "fspool": 1,  # how frequently to save solution to RAM
    }

    comm = spyro.utils.mpi_init(model)
    mesh = RectangleMesh(100, 100, 1.5, 1.5) # to test FWI, mesh aligned with interface

    element = spyro.domains.space.FE_method(
        mesh, model["opts"]["method"], model["opts"]["degree"]
    )

    V    = FunctionSpace(mesh, element)
    z, x = SpatialCoordinate(mesh)

    vp_exact = Function(V).interpolate( 1.0 + 0.0*x)
    vp_guess = Function(V).interpolate( 0.8 + 0.0*x)

    spyro.tools.gradient_test_acoustic_ad(
                                model, 
                                mesh, 
                                V, 
                                comm, 
                                vp_exact, 
                                vp_guess)

