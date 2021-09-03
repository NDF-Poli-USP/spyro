from SeismicMesh.sizing.mesh_size_function import write_velocity_model
from mpi4py import MPI
import meshio

import sys

from SeismicMesh import get_sizing_function_from_segy, generate_mesh, Rectangle

comm = MPI.COMM_WORLD

"""
Build a mesh of the Marmousi elastic benchmark velocity model in serial or parallel
Takes roughly 1 minute with 2 processors and less than 1 GB of RAM.
"""

# Name of SEG-Y file containg velocity model (S-wave speed)
#fname = "./velocity_models/elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy"
#fname = "./velocity_models/elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy"
fname = "./velocity_models/elastic-marmousi-model/model/MODEL_DENSITY_1.25m.segy"

# Bounding box describing domain extents (corner coordinates)
bbox = (-3500.0, 0.0, 0.0, 17000.0) # this includes a 450-m thick water layer
#bbox = (-3500.0, -450.0, 0.0, 17000.0) # removing water layer

# Desired minimum mesh size in domain
hmin = 50.0

rectangle = Rectangle(bbox)

write_vel_mod=0
if write_vel_mod==1:
    write_velocity_model(fname)
    sys.exit("Mesh not generated") 

# Construct mesh sizing object from velocity model
#ef = get_sizing_function_from_segy(
#    fname,
#    bbox,
#    hmin=hmin,
#    wl=10,
#    freq=4,
#    dt=0.001,
#    grade=0.15,
#    domain_pad=1e3,
#    pad_style="edge",
#)
ef = get_sizing_function_from_segy(
    fname,
    bbox,
    hmin=hmin,
    wl=10,
    freq=4,
    dt=0.001,
    grade=0.15
)

points, cells = generate_mesh(domain=rectangle, edge_length=ef)

if comm.rank == 0:
    # Write the mesh in a vtk format for visualization in ParaView
    # NOTE: SeismicMesh outputs assumes the domain is (z,x) so for visualization
    # in ParaView, we swap the axes so it appears as in the (x,z) plane.
    meshio.write_points_cells(
        "meshes/marmousi_elastic_with_water_layer.msh",
        points[:] / 1000, # do not swap here
        [("triangle", cells)],
        file_format="gmsh22",
        binary=False
    )
    meshio.write_points_cells(
        "meshes/marmousi_elastic_with_water_layer.vtk",
        points[:, [1, 0]] / 1000,
        [("triangle", cells)],
        file_format="vtk",
        binary=False
    )

