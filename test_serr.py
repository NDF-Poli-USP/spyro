import firedrake as fire
import numpy as np
from netgen.meshing import *
from netgen.csg import *
import meshio
import ipdb


def inside_hyp(x, y, z, a, b, c):
    return abs((x - edge / 2.) / a) ** n + \
        abs((y - edge / 2.) / b) ** n + \
        abs((z + edge / 2.) / c) ** n < 1.


edge = 1.
pad = 0.5
Lx = edge + 2 * pad
Ly = edge + 2 * pad
Lz = edge + pad
a = edge / 2. + pad
b = edge / 2. + pad
c = edge / 2. + pad
n = 2.8

# Mesh
rec_mesh = fire.BoxMesh(12, 12, 8, Lx, Ly, Lz)
rec_mesh.coordinates.dat.data_with_halos[:, 0] -= pad
rec_mesh.coordinates.dat.data_with_halos[:, 1] -= pad
rec_mesh.coordinates.dat.data_with_halos[:, 2] *= -1.
# rec_mesh = fire.Mesh("ellipsoid_mesh.msh")

# Create the final mesh that will contain both
final_mesh = Mesh()
final_mesh.dim = 3

# Get coordinates of the rectangular mesh
coord_rec = rec_mesh.coordinates.dat.data_with_halos

# Add all vertices from rectangular mesh and create mapping
rec_map = {}
for i, coord in enumerate(coord_rec):
    x, y, z = coord
    if inside_hyp(x, y, z, a, b, c):
        rec_map[i] = final_mesh.Add(MeshPoint((x, y, z)))

# Face descriptor for the rectangular mesh
fd_rec = final_mesh.Add(FaceDescriptor(bc=1, domin=1, domout=0))

# Get mesh cells from rectangular mesh
rec_cells = rec_mesh.coordinates.cell_node_map().values_with_halo

# Add all elements from rectangular mesh to the netgen mesh
final_mesh.SetMaterial(1, "rec")
for cell in rec_cells:
    # netgen_points = [rec_map[cell[i]] for i in range(len(cell))]
    netgen_points = [rec_map.get(cell[i]) for i in range(len(cell))]
    if not any(p is None for p in netgen_points):
        final_mesh.Add(Element3D(fd_rec, netgen_points))

# Mesh data
final_mesh.Compress()
print(f"Mesh created with {len(final_mesh.Points())} points "
      f"and {len(final_mesh.Elements3D())} elements", flush=True)

q = {"overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)}
final_mesh = fire.Mesh(final_mesh, distribution_parameters=q)


# Create mesh and save with explicit Gmsh format
coords = final_mesh.coordinates.dat.data
cells = [("tetra", final_mesh.coordinates.cell_node_map().values)]
mesh = meshio.Mesh(coords, cells)
output_filename = "box_gmsh.msh"

# Use Gmsh format 2.2 which is more widely compatible
meshio.write(output_filename, mesh, file_format="gmsh22")
# meshio.write("ellipsoid_mesh.msh", mesh, file_format="gmsh22", binary=False)

# Or try ASCII format
# meshio.write(output_filename, mesh, file_format="gmsh-ascii")

print("Box Mesh Generated Successfully", flush=True)
fire.VTKFile("box_netgen.pvd").write(final_mesh)
