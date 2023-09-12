import spyro
import firedrake as fire
import numpy as np


def circle_circunscribe(T):
    (x1, y1), (x2, y2), (x3, y3) = T
    A = np.array([[x3-x1,y3-y1],[x3-x2,y3-y2]])
    Y = np.array([(x3**2 + y3**2 - x1**2 - y1**2),(x3**2+y3**2 - x2**2-y2**2)])
    if np.linalg.det(A) == 0:
        return False
    Ainv = np.linalg.inv(A)
    X = 0.5*np.dot(Ainv,Y)
    x,y = X[0],X[1]
    r = np.sqrt((x-x1)**2+(y-y1)**2)
    return (x,y),r


Lz = 1.0
Lx = 2.0
c = 1.5
freq = 5.0
lbda = c/freq
pad = 0.3
cpw = 3

Mesh_obj = spyro.meshing.AutomaticMesh(
    dimension=2,
    abc_pad=pad,
    mesh_type="SeismicMesh"
)
Mesh_obj.set_mesh_size(length_z=Lz, length_x=Lx)
Mesh_obj.set_seismicmesh_parameters(cpw=cpw, edge_length=lbda/cpw, output_file_name="test.msh")

mesh = Mesh_obj.create_mesh()

V = fire.FunctionSpace(mesh, "CG", 1)
z_mesh, x_mesh = fire.SpatialCoordinate(mesh)
uz = fire.Function(V).interpolate(z_mesh)
ux = fire.Function(V).interpolate(x_mesh)

z = uz.dat.data[:]
x = ux.dat.data[:]

# Testing if boundaries are correct
test1 = (np.isclose(np.amin(z), -Lz-pad))
test1 = test1 and (np.isclose(np.amax(x), Lx+pad))
test1 = test1 and (np.isclose(np.amax(z), 0.0))
test1 = test1 and (np.isclose(np.amin(x), -pad))

# Checking cell diameter of an interior cell
node_ids = V.cell_node_list[300]
p0 = (z[node_ids[0]], x[node_ids[0]])
p1 = (z[node_ids[1]], x[node_ids[1]])
p2 = (z[node_ids[2]], x[node_ids[2]])

print("end")
