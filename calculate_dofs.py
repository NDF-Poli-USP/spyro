from firedrake import *
import numpy as np
from model_for_mesh import create_model_for_grid_point_calculation as model
import spyro

scales = [1,2,3]
comm = spyro.utils.mpi_init(model(scales[0]))
for scale in scales:
    if comm.comm.rank == 0:
        print(f"For scale of {scale}", flush = True)
    mesh = Mesh(
        "meshes/homogeneous_3D_scale"+str(scale)+".msh",
        comm=comm.comm,
        distribution_parameters={
            "overlap_type": (DistributedMeshOverlapType.NONE, 0)
            },
        )
    element = FiniteElement('KMV', mesh.ufl_cell(), degree=3, variant = 'KMV')

    V = FunctionSpace(mesh, element)
    comm.comm.barrier()
    print("DoFs as Vdim", flush=True)
    print(V.dim(), flush = True)
    comm.comm.barrier()
    u = Function(V)
    print("DoFs as data with halos", flush=True)
    comm.comm.barrier()
    udat = u.dat.data_with_halos[:]
    print(np.shape(udat), flush = True)
    comm.comm.barrier()
    print("DoFs as data", flush=True)
    comm.comm.barrier()
    udat = u.dat.data[:]
    print(np.shape(udat), flush = True)
    comm.comm.barrier()
