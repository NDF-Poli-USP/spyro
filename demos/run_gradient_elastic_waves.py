# script used to test the gradient and adjoint problems (elastic waves)

from firedrake import (
    RectangleMesh,
    FunctionSpace,
    Function,
    SpatialCoordinate,
    conditional,
    File,
    Constant,
    Mesh,
    exp,
    DumbCheckpoint,
    FILE_CREATE
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
    "status": False,  # True or False, used to turn on any type of BC 
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
    "num_receivers": 3,
    "receiver_locations": spyro.create_transect(
       # (-0.4, 0.375), (-0.4, 1.125), 2
       # (-1.1, 0.375), (-1.1, 1.125), 2
       (-1.1, 0.375), (-1.1, 1.125), 3
       # (-1.25, -0.25), (-1.25, 1.75), 100 for the case with ABL
    ),
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    #"tf": 0.0005*4000,  # Final time for event
    "tf": 0.0005*1600,  # Final time for event (for test 7)
    "dt": 0.00050,  # timestep size (divided by 2 in the test 4. dt for test 3 is 0.00050)
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 100,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)

mesher="FM" # either SM (SeismicMesh), FM (Firedrake mesher), or RM (read an existing mesh)
if mesher=="FM":
    #                            (X     Y)  ->  (Z  X) in spyro
    #mesh = RectangleMesh(30, 30, 1.75, 2.0, diagonal="crossed")
    #mesh = RectangleMesh(85, 85, 1.5, 1.5, diagonal="crossed") # to test water layer
    mesh = RectangleMesh(80, 80, 1.5, 1.5, diagonal="crossed") # to test water layer, mesh aligned with interface
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

run_guess=1 
if run_guess:
    lamb = Constant(1./3.) # guess
    mu = Constant(1./4.)
else:
    lamb = Constant(1./2.) # exact
    mu = Constant(1./4.)

rho = Constant(1.) # for test 3 and 7 (constant cp and cd)

sources     = spyro.Sources(model, mesh, V, comm)
receivers   = spyro.Receivers(model, mesh, V, comm)
AD = True
if AD:
    point_cloud = receivers.setPointCloudRec(comm,paralel_z=True)

wavelet = spyro.full_ricker_wavelet(
                dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"]
            )

run_forward = 1
if run_forward:
    print("Starting forward computation")
    start = time.time()
    uz_exact = spyro.io.load_shots(model, comm, file_name="./shots/test_grad/uz_at_recv_exact.dat")
    ux_exact = spyro.io.load_shots(model, comm, file_name="./shots/test_grad/ux_at_recv_exact.dat")
    true_rec = [uz_exact, ux_exact]
    u_field, uz_at_recv, ux_at_recv, uy_at_recv, J = spyro.solvers.forward_elastic_waves(
        model, mesh, comm, rho, lamb, mu, sources, wavelet, receivers, true_rec, output=False
    )
    end = time.time()
    print(round(end - start,2))

    if 0:
        cmin=-1e-4
        cmax=1e-4
        u_at_recv = (uz_at_recv**2. + ux_at_recv**2.)**0.5
        spyro.plots.plot_shots(model, comm, u_at_recv, show=True, vmin=cmin, vmax=cmax)
        sys.exit("exiting without running gradient")

    if run_guess:
        spyro.io.save_shots(model, comm, uz_at_recv, file_name="./shots/test_grad/uz_at_recv_guess.dat")
        spyro.io.save_shots(model, comm, ux_at_recv, file_name="./shots/test_grad/ux_at_recv_guess.dat")
    else:
        spyro.io.save_shots(model, comm, uz_at_recv, file_name="./shots/test_grad/uz_at_recv_exact.dat")
        spyro.io.save_shots(model, comm, ux_at_recv, file_name="./shots/test_grad/ux_at_recv_exact.dat")
        sys.exit("exiting without running gradient")
    
    #chk = DumbCheckpoint("dump", mode=FILE_CREATE)
    #chk.store(u_field) $ u_field is a list of firedrake functions, checkpoint only saves functions
    
uz_guess = spyro.io.load_shots(model, comm, file_name="./shots/test_grad/uz_at_recv_guess.dat")
ux_guess = spyro.io.load_shots(model, comm, file_name="./shots/test_grad/ux_at_recv_guess.dat")
    
residual_z = spyro.utils.evaluate_misfit(model, uz_guess, uz_exact)
residual_x = spyro.utils.evaluate_misfit(model, ux_guess, ux_exact)
residual_y = []

if 0:
    if 1:
        plt.title("u_z")
        plt.plot(uz_exact,label='exact')
        plt.plot(uz_guess,label='guess')
        plt.plot(uz_exact-uz_guess,label='exact - guess',marker="+")
        plt.plot(residual_z,label='residual')
    else:
        plt.title("u_x")
        plt.plot(ux_exact,label='exact')
        plt.plot(ux_guess,label='guess')
        plt.plot(ux_exact-ux_guess,label='exact - guess',marker="+")
        plt.plot(residual_x,label='residual')
    plt.legend()
    plt.show()
    sys.exit("exiting without running gradient")

print("Starting gradient computation")
start      = time.time()
dJdl, dJdm = spyro.solvers.gradient_elastic_waves(
    model, mesh, comm, rho, lamb, mu, receivers, u_field, residual_z, residual_x, residual_y, output=True
)
end = time.time()
print(round(end - start,2))
File("dJdl.pvd").write(dJdl)
File("dJdm.pvd").write(dJdm)
