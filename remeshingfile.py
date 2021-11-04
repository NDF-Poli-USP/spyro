from firedrake import *

import SeismicMesh
import meshio

import numpy as np
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
import spyro
from mpi4py import MPI

def remesh(fname, freq, mesh_iter, comm):
    """for now some hardcoded options"""
    if comm.ensemble_comm.rank == 0:
        bbox = (-4000.0, 0.0, -500.0, 17500.0)

        wl = 2.67

        # Desired minimum mesh size in domain
        hmin = 1500.0 / (wl * freq)

        rectangle = SeismicMesh.Rectangle(bbox)

        # Construct mesh sizing object from velocity model
        ef = SeismicMesh.get_sizing_function_from_segy(
            fname, bbox, hmin=hmin, wl=wl, freq=freq, dt=0.001, comm=comm.comm
        )

        SeismicMesh.write_velocity_model(
            fname,
            ofname="velocity_models/mm_GUESS" + str(mesh_iter),
            comm=comm.comm,
        )

        points, cells = SeismicMesh.generate_mesh(
            domain=rectangle, edge_length=ef, comm=comm.comm
        )

        if comm.comm.rank == 0:
            meshio.write_points_cells(
                "meshes/mm_GUESS" + str(mesh_iter) + ".msh",
                points / 1000,
                [("triangle", cells)],
                file_format="gmsh22",
                binary=False,
            )
            # for visualization
            meshio.write_points_cells(
                "meshes/mm_GUESS" + str(mesh_iter) + ".vtk",
                points / 1000.0,
                [("triangle", cells)],
            )

