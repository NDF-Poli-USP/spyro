import numpy as np
import h5py
import firedrake as fire
import matplotlib.pyplot as plt
import copy
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

V = fire.FunctionSpace(mesh, "KMV", 5)
W = fire.VectorFunctionSpace(mesh, "KMV", 5)
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
coordinates = copy.deepcopy(mesh.coordinates.dat.data)
mesh.coordinates.dat.data[:, 0] = coordinates[:, 1]
mesh.coordinates.dat.data[:, 1] = coordinates[:, 0]


fig, axes = plt.subplots()
im = fire.tripcolor(c, axes=axes, cmap='coolwarm')
# axes.axis("equal")
axes.set_aspect("equal", "box")
plt.title("Velocity field")

cbar = fig.colorbar(im, orientation="vertical", fraction=0.046)
cbar.ax.set_xlabel("velocity (km/s)")
axes.set_xticks([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.3])
fig.set_size_inches(13, 10)
plt.savefig("velocity_field_marmousi.png")

plt.show()
print("END")
