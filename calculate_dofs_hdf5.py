from firedrake import *
import numpy as np
from model_for_mesh import create_model_for_grid_point_calculation as model
import spyro
import h5py

scales = [10, 20, 30, 40, 50, 60, 70, 80, 90, 130]
comm = spyro.utils.mpi_init(model(scales[0]))
with h5py.File('output2.h5', 'w') as f:
    for scale in scales:
        if comm.comm.rank == 0:
            print(f"For scale of {scale} by 100", flush=True)

        mesh = Mesh("meshes/homogeneous_3D_scale" + str(scale) + "by100.msh",
                    comm=comm.comm,
                    distribution_parameters={"overlap_type": (DistributedMeshOverlapType.NONE, 0)})

        element = FiniteElement('KMV', mesh.ufl_cell(), degree=3, variant='KMV')
        V = FunctionSpace(mesh, element)

        comm.comm.barrier()
        if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
            print("DoFs as Vdim", flush=True)
        f.create_group(f"scale_{scale}")
        f[f"scale_{scale}"].create_dataset("Vdim", data=V.dim())

        comm.comm.barrier()
        u = Function(V)
        if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
            print("DoFs as data with halos", flush=True)
        comm.comm.barrier()
        udat = u.dat.data_with_halos[:]
        f[f"scale_{scale}"].create_dataset("data_with_halos", data=udat)

        comm.comm.barrier()
        if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
            print("DoFs as data", flush=True)
        comm.comm.barrier()
        udat = u.dat.data[:]
        f[f"scale_{scale}"].create_dataset("data", data=udat)

        comm.comm.barrier()
        if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
            print("\n\n\n", flush=True)
