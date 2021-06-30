import firedrake
import matplotlib.pyplot as plt
import copy
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size']   = 12

mesh = firedrake.Mesh("meshes/heterogeneous10.08.msh")

coordinates = copy.deepcopy(mesh.coordinates.dat.data)
print(coordinates)
mesh.coordinates.dat.data[:,0]=coordinates[:,1]
mesh.coordinates.dat.data[:,1]=coordinates[:,0]

DG0 = firedrake.FunctionSpace(mesh, "DG", 0)
f = firedrake.interpolate(firedrake.CellSize(mesh), DG0)

fig, axes = plt.subplots()
im = firedrake.tricontourf(f, axes=axes)
# axes.axis("equal")
axes.set_aspect("equal", "box")
plt.xlabel("X (km)")
plt.ylabel("Z (km)")
plt.title("BP2004 mesh resolution with C = 2.03")

cbar = fig.colorbar(im, orientation="horizontal")
cbar.ax.set_xlabel("circumcircle radius (km)")
fig.set_size_inches(13,10)

plt.show()