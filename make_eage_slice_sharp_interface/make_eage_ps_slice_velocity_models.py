# import meshio
import segyio
import matplotlib.pyplot as plt

import numpy as np

import zipfile
import SeismicMesh as sm


# bottom left corner is at (-4200.0, 0.0, 0.0) *in meters
# top right corner is at (0.0, 13520.0, 13520.0)

bbox = (-4200.0, 0.0, 0.0, 13520.0, 0.0, 13520.0)
nx, ny, nz = 676, 676, 210

grid_spacing_z = 4200 / 210
grid_spacing_x = 13520.0 / 676
grid_spacing_y = 13520.0 / 676


def _create_segy(velocity, filename):
    """Write the velocity data into a segy file named filename"""
    spec = segyio.spec()

    velocity = np.flipud(velocity.T)

    spec.sorting = 2  # not sure what this means
    spec.format = 1  # not sure what this means
    spec.samples = range(velocity.shape[0])
    spec.ilines = range(velocity.shape[1])
    spec.xlines = range(velocity.shape[0])

    assert np.sum(np.isnan(velocity[:])) == 0

    with segyio.create(filename, spec) as f:
        for tr, il in enumerate(spec.ilines):
            f.trace[tr] = velocity[:, tr]


def _read_bin(filename, nz, nx, ny, byte_order, axes_order, axes_order_sort):
    axes = [nz, nx, ny]
    ix = np.argsort(axes_order)
    axes = [axes[o] for o in ix]
    with open(filename, "r") as file:
        print("Reading binary file: " + filename)
        if byte_order == "big":
            vp = np.fromfile(file, dtype=np.dtype("float32").newbyteorder(">"))
        elif byte_order == "little":
            vp = np.fromfile(file, dtype=np.dtype("float32").newbyteorder("<"))
        else:
            raise ValueError("Please specify byte_order as either: little or big.")
        vp = vp.reshape(*axes, order=axes_order_sort)
        return np.flipud(vp.transpose((*axes_order)))


# wget https://s3.amazonaws.com/open.source.geoscience/open_data/seg_eage_models_cd/Salt_Model_3D.tar.gz
# tar -xvf Salt_Model_3D.tar.gz

# This file is in a big Endian binary format, so we must tell the program the shape of the velocity model.
path = "Salt_Model_3D/3-D_Salt_Model/VEL_GRIDS/"
# Extract binary file Saltf@@ from SALTF.ZIP
zipfile.ZipFile(path + "SALTF.ZIP", "r").extract("Saltf@@", path=path)

fname = path + "Saltf@@"

# Dimensions of model (number of grid points in z, x, and y)
byte_order = "big"
axes_order = (2, 0, 1)  # order for EAGE (x, y, z) to default order (z,x,y)
axes_order_sort = "F"  # binary is packed in a FORTRAN-style

vp_as_np_array = _read_bin(fname, nz, nx, ny, byte_order, axes_order, axes_order_sort)

y = np.linspace(0, 13520.0, 676)
z = np.linspace(-4200.0, 0.0, 210)
yg, zg = np.meshgrid(y, z)
true_slice = vp_as_np_array[:, :, 300]
# make a linear variation of velocity
# linear = np.flipud(np.linspace(2000.0, 4000.0, true_slice.shape[0]))
C0 = 1500.0
M = 0.60
linear = C0 + M * np.abs(np.linspace(-4200.0, 0, true_slice.shape[0]))
linear = np.tile(linear, (true_slice.shape[1], 1)).T
linear_wo_salt = linear.copy()
print(np.amax(linear))

linear[true_slice > 4400.0] = 4500.0

plt.pcolor(y, z, linear)
plt.title("EAGE Salt, Slice y-axis #300, True model")
plt.axis("equal")
cb = plt.colorbar()
cb.set_label("P-wave velocity (m/s)")
plt.ylabel("z-axis (m)")
plt.xlabel("y-axis (m)")
plt.show()

_create_segy(linear.T, "eage_slice_ps_true.segy")


# create initial model
p = np.vstack((zg.ravel(), yg.ravel())).T
initial_slice = linear_wo_salt.copy()
# initial_slice[:] = 2000.0
#disk = sm.Disk((-1690.0, 7890.0), 500.0, stretch=[0.001, 5.0], translate=[0.0, -30710])
disk = sm.Disk((-1690.0, 7890.0), 550.0, stretch=[0.001, 5.0], translate=[0.0, -31710])

# points,cells = sm.generate_mesh(domain=disk, edge_length=100)
# meshio.write_points_cells('mesh.vtk', points, [('triangle', cells)])
# quit()
# disk.show(samples=5e6)
d = disk.eval(p)
d = np.reshape(d, initial_slice.shape)
initial_slice[d < 0] = 4500.0

plt.pcolor(y, z, initial_slice)
plt.title("EAGE Salt, Initial model")
plt.axis("equal")
cb = plt.colorbar()
cb.set_label("P-wave velocity (m/s)")
plt.ylabel("z-axis (m)")
plt.xlabel("y-axis (m)")
plt.show()

_create_segy(initial_slice.T, "eage_slice_ps_guess.segy")
