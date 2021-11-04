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
    """Runs a standart FWI gradient based optimization.
    """
    def __init__(self, model, iteration_limit = 100, params = None):
        self.model = model
        self.dimension = model["opts"]["dimension"]
        self.method = model["opts"]["method"]
        self.degree = model["opts"]["degree"]
        self.comm = spyro.utils.mpi_init(model)
        self.shot_record = spyro.io.load_shots(model, self.comm)

        if params == None:
            params = {
                "General": {"Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 10}},
                "Step": {
                    "Type": "Augmented Lagrangian",
                    "Augmented Lagrangian": {
                        "Subproblem Step Type": "Line Search",
                        "Subproblem Iteration Limit": 5.0,
                    },
                    "Line Search": {"Descent Method": {"Type": "Quasi-Newton Step"}},
                },
                "Status Test": {
                    "Gradient Tolerance": 1e-16,
                    "Iteration Limit": iteration_limit,
                    "Step Tolerance": 1.0e-16,
                },
            }
        
        self.parameters = ROL.ParameterList(params, "Parameters")

        if model["mesh"]["meshfile"] != None:
            mesh, V = spyro.io.read_mesh(model, self.comm)
            self.mesh = mesh
            self.space = V
        else:
            mesh, V = self.build_initial_mesh()
            self.mesh = mesh
            self.space = V

        vp = self.run_FWI()


    def build_inital_mesh(self):
        print('Entering mesh generation', flush = True)
        M = cells_per_wavelength(self.model)
        mesh = build_mesh(model, vp = 'default')
        element = domains.space.FE_method(mesh, method, degree)
        space = fire.FunctionSpace(mesh, element)
        return mesh, space
    


        

