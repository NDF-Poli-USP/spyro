import spyro
import firedrake as fire
import numpy as np


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

print("end")
