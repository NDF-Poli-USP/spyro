import numpy as np
import h5py
import firedrake as fire
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


# class Data_only_wave_model:
#     def __init__(self, velocity_filename, mesh_filename):


plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 12

velocity_filename = "/media/alexandre/T7 Shield/common_files/velocity_models/vp_marmousi-ii.hdf5"
mesh_filename = "automatic_mesh.msh"

mesh = fire.Mesh(mesh_filename)

minimum_z = np.amin(mesh.coordinates.dat.data[:, 0])
maximun_z = np.amax(mesh.coordinates.dat.data[:, 0])
minimum_x = np.amin(mesh.coordinates.dat.data[:, 1])
maximum_x = np.amax(mesh.coordinates.dat.data[:, 1])

V = fire.FunctionSpace(mesh, "KMV", 4)
W = fire.VectorFunctionSpace(mesh, "KMV", 4)
dof_coordinates = fire.interpolate(mesh.coordinates, W)
dofs_z, dofs_x = dof_coordinates.dat.data[:, 0], dof_coordinates.dat.data[:, 1]

with h5py.File(velocity_filename, "r") as f:
    Z = np.asarray(f.get("velocity_model")[()])

    nrow, ncol = Z.shape
    z = np.linspace(minimum_z, maximun_z, nrow)
    x = np.linspace(minimum_x, maximum_x, ncol)

    # make sure no out-of-bounds
    dofs_z2 = [
        minimum_z if z < minimum_z else maximun_z if z > maximun_z else z for z in dofs_z
    ]
    dofs_x2 = [
        minimum_x if x < minimum_x else maximum_x if x > maximum_x else x for x in dofs_x
    ]

    interpolant = RegularGridInterpolator((z, x), Z)
    tmp = interpolant((dofs_z2, dofs_x2))
c = fire.Function(V)
c.dat.data[:] = tmp

if min(c.dat.data[:]) > 100.0:
    # data is in m/s but must be in km/s
    if fire.COMM_WORLD.rank == 0:
        print("INFO: converting from m/s to km/s", flush=True)
    c.assign(c / 1000.0)  # meters to kilometers

print("END")
