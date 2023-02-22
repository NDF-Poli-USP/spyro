from firedrake import *
import numpy as np
from model_for_mesh import create_model_for_grid_point_calculation as model
import spyro

# scales = [1,2,3]
# scales = [50, 60, 70, 80, 90, 130]#, 160, 180]
# scales = [100, 150, 200]
# scales = [160, 180, 200]
scales = [21, 22, 23, 24, 26, 28, 42, 44, 46, 48]
comm = spyro.utils.mpi_init(model(scales[0]))
for scale in scales:
    if comm.comm.rank == 0:
        print(f"For scale of {scale} by 100", flush = True)
    mesh = Mesh(
        "meshes/homogeneous_3D_scale"+str(scale)+"by100.msh",
        comm=comm.comm,
        distribution_parameters={
            "overlap_type": (DistributedMeshOverlapType.NONE, 0)
            },
        )
    element = FiniteElement('KMV', mesh.ufl_cell(), degree=3, variant = 'KMV')

    V = FunctionSpace(mesh, element)
    comm.comm.barrier()
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print("DoFs as Vdim", flush=True)
    print(V.dim(), flush = True)
    comm.comm.barrier()
    u = Function(V)
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print("DoFs as data with halos", flush=True)
    comm.comm.barrier()
    udat = u.dat.data_with_halos[:]
    print(np.shape(udat), flush = True)
    comm.comm.barrier()
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print("DoFs as data", flush=True)
    comm.comm.barrier()
    udat = u.dat.data[:]
    print(np.shape(udat), flush = True)
    comm.comm.barrier()
    if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
        print("\n\n\n", flush = True)
