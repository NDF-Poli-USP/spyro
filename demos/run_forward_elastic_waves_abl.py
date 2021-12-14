# script used in tests 3, 4, and 5 (BC tests), and 7 (mu=0 and different rho)

from firedrake import (
    RectangleMesh,
    FunctionSpace,
    Function,
    SpatialCoordinate,
    conditional,
    File,
    Constant,
    Mesh,
    exp
)
import spyro
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import meshio
import SeismicMesh

model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV", # Equi or KMV
    "degree": 3,  # p order
    "dimension": 2,  # dimension
}

model["parallelism"] = {
    "type": "spatial",  # options: automatic (same number of cores for evey processor) or spatial
}

# Define the domain size without the ABL.
model["mesh"] = {
    "Lz": 1.5,  # depth in km - always positive
    "Lx": 1.5,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

# Specify a 250-m Absorbing Boundary Layer (ABL) on the three sides of the domain to damp outgoing waves.
model["BCs"] = {
    "status": True,  # True or False, used to turn on any type of BC 
    "outer_bc": "non-reflective", #  none or non-reflective (outer boundary condition)
    "abl_bc": "none",  # none, gaussian-taper, or alid
    "lz": 0.25,  # thickness of the ABL in the z-direction (km) - always positive
    "lx": 0.25,  # thickness of the ABL in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": [(-0.75, 0.75)],
    "frequency": 10.0,
    "delay": 1.0,
    "num_receivers": 100,
    "receiver_locations": spyro.create_transect(
        (-1.25, -0.25), (-1.25, 1.75), 100
    ),
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 0.0005*4000,  # Final time for event
    #"tf": 0.0005*1600,  # Final time for event (for test 7)
    "dt": 0.00050,  # timestep size (divided by 2 in the test 4. dt for test 3 is 0.00050)
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 100,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)

mesher="FM" # either SM (SeismicMesh), FM (Firedrake mesher), or RM (read an existing mesh)
if mesher=="FM":
    #                            (X     Y)  ->  (Z  X) in spyro
    mesh = RectangleMesh(30, 30, 1.75, 2.0, diagonal="crossed")
    #mesh = RectangleMesh(85, 85, 1.5, 1.5, diagonal="crossed") # to test water layer
    #mesh = RectangleMesh(80, 80, 1.5, 1.5, diagonal="crossed") # to test water layer, mesh aligned with interface
elif mesher=="SM":
    raise ValueError("check this first")
    bbox = (0.0, 1.5, 0.0, 1.5)
    rect = SeismicMesh.Rectangle(bbox)
    points, cells = SeismicMesh.generate_mesh(domain=rect, edge_length=0.025)
    mshname = "meshes/test_mu=0.msh"
    meshio.write_points_cells(
        mshname,
        points[:], # do not swap here
        [("triangle", cells)],
        file_format="gmsh22",
        binary=False
    )
    mesh = Mesh(mshname)
elif mesher=="RM":
    mshname = "meshes/test_mu=0_unstructured.msh" # to test water layer
    mesh = Mesh(mshname)
else:
    raise ValueError("mesher not yet supported")

if model["BCs"]["status"]:
    mesh.coordinates.dat.data[:, 0] -= 1.75 # x -> z
    mesh.coordinates.dat.data[:, 1] -= 0.25 # y -> x
else:
    mesh.coordinates.dat.data[:, 0] -= 1.5
    mesh.coordinates.dat.data[:, 1] -= 0.0

element = spyro.domains.space.FE_method(
    mesh, model["opts"]["method"], model["opts"]["degree"]
)
V = FunctionSpace(mesh, element)

z, x = SpatialCoordinate(mesh) 

water_layer=0 # for test 7
if water_layer==0:
    lamb = Constant(1./2.)
    mu = Constant(1./4.)
else:
    lambfield = conditional(z <= -1.05, 1./2., 1.)
    mufield   = conditional(z <= -1.05, 1./4., 0.)
    
    element0 = spyro.domains.space.FE_method(
                    mesh, "DG", 0)
    V0 = FunctionSpace(mesh, element0)
    
    #lamb = Function(V, name="lamb").interpolate(lambfield)
    #mu   = Function(V, name="mu").interpolate(mufield)
    lamb = Function(V0, name="lamb").interpolate(lambfield)
    mu   = Function(V0, name="mu").interpolate(mufield)
    File("lamb.pvd").write(lamb)
    File("mu.pvd").write(mu)

if 0:
    rho = Constant(1.) # for test 3 and 7 (constant cp and cd)
else:
    rhofield = conditional(z <= -1.05, 0.25, 1.0)
    rho = Function(V, name="rho").interpolate(rhofield) # for test 4 (discontinuity in cp and cs)
    File("rho.pvd").write(rho)

#sys.exit("exiting without running")

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
                        dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"]
                        )

post_process=False
vtkfiles=True
shotfiles=True
if post_process==False:
    start = time.time()
    u_field, uz_at_recv, ux_at_recv, uy_at_recv = spyro.solvers.forward_elastic_waves(
        model, mesh, comm, rho, lamb, mu, sources, wavelet, receivers, output=vtkfiles
    )
    end = time.time()
    print(round(end - start,2))
    
    u_at_recv = (uz_at_recv**2. + ux_at_recv**2.)**0.5
    if shotfiles==True:
        cmin=-1e-4
        cmax=1e-4
        spyro.plots.plot_shots(model, comm, u_at_recv, show=True, vmin=cmin, vmax=cmax)

        #filename="./shots/test_4/test_4_kmv_p3_cp.dat"
        #spyro.io.save_shots(model, comm, u_at_recv, file_name=filename)
else:
    tn="test_4"
    u_kmv = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_kmv_p3_nobc.dat")
    cmin=-1e-4
    cmax=1e-4
    spyro.plots.plot_shots(model, comm, u_kmv, show=True, vmin=cmin, vmax=cmax)
    #spyro.plots.plot_shots(model, comm, u_kmv, show=True)
