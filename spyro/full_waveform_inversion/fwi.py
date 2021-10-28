from firedrake import *
import numpy as np
import finat
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
from mpi4py import MPI

import meshio
import SeismicMesh

import spyro

class FWI():
    def __init__(self, model):
        self.model = model
        self.dimension = model["opts"]["dimension"]
        self.method = model["opts"]["method"]
        self.degree = model["opts"]["degree"]
        self.comm = spyro.utils.mpi_init(model)

        if model["mesh"]["meshfile"] != None:
            mesh, V = spyro.io.read_mesh(model, self.comm)
            self.mesh = mesh
            self.space = V
        else:
            mesh, V = self.build_initial_mesh()
            self.mesh = mesh
            self.space = V

    def build_inital_mesh(self):
        print('Entering mesh generation', flush = True)
        M = cells_per_wavelength(self.model)
        minimum_mesh_velocity = 1.429
        frequency = self.model["acquisition"]['frequency']
        lbda = minimum_mesh_velocity/frequency

        bbox = getbbox(self.model)

        if self.dimension == 2:

        

