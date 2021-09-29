# script used in tests 1 and 2 (p-extension analysis)

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
    "method": "CG",  # either CG or KMV
    "quadratrue": "CG", # Equi or KMV
    "degree": 2,  # p order
    "dimension": 2,  # dimension
}

model["parallelism"] = {
    "type": "spatial",  # options: automatic (same number of cores for evey processor) or spatial
}

model["mesh"] = {
    "Lz": 1.5,  # depth in km - always positive
    "Lx": 1.5,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

model["BCs"] = {
    "status": False,  # True or False, used to turn on any type of BC 
    "outer_bc": "non-reflective", #  none or non-reflective (outer boundary condition)
    "abl_bc": "gaussian-taper",  # none, gaussian-taper, or alid
    "lz": 0.0,  # thickness of the ABL in the z-direction (km) - always positive
    "lx": 0.0,  # thickness of the ABL in the x-direction (km) - always positive
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
        (-0.50, 0.1), (-0.50, 1.4), 100
    ),
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 0.0005*1600,  # Final time for event
    "dt": 0.00050,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool":  20,  # how frequently to output solution to pvds
    "fspool": 100,  # how frequently to save solution to RAM
}
comm = spyro.utils.mpi_init(model)

mesher="FM" # either SM (SeismicMesh) or FM (Firedrake mesher) 
if mesher=="FM":
    mesh = RectangleMesh(30, 30, 1.5, 1.5, diagonal="crossed")
    #mesh = RectangleMesh(60, 60, 1.5, 1.5, diagonal="crossed") # reference mesh 1
    #mesh = RectangleMesh(120, 120, 1.5, 1.5, diagonal="crossed") # reference mesh 2 only to test KMV
    #mesh = RectangleMesh(50, 50, 1.5, 1.5)
elif mesher=="SM":
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
else:
    raise ValueError("mesher not yet supported")    

mesh.coordinates.dat.data[:, 0] -= 1.5
mesh.coordinates.dat.data[:, 1] -= 0.0

element = spyro.domains.space.FE_method(
    mesh, model["opts"]["method"], model["opts"]["degree"]
)

V = FunctionSpace(mesh, element)
#print(V.dof_count*2)

lamb = Constant(1./2.) 
mu = Constant(1./4.)  
#lamb = Constant(1.) 
#mu = Constant(0.)  
rho = Constant(1.) 
#sys.exit("exiting without running")

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
                        dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"]
                        )

post_process=True
vtkfiles=False
shotfiles=False
if post_process==False:
    start = time.time()
    u_field, u_at_recv = spyro.solvers.forward_elastic_waves(
        model, mesh, comm, rho, lamb, mu, sources, wavelet, receivers, output=vtkfiles
    )
    end = time.time()
    print(round(end - start,2))

    if shotfiles==True:
        cmin=-1e-4
        cmax=1e-4
        spyro.plots.plot_shots(model, comm, u_at_recv, show=True, vmin=cmin, vmax=cmax)

        #filename="./shots/test_1/test_1_cg_p5_h25m.dat"
        #filename="./shots/test_1/test_1_kmv_p5_h25m.dat"
        filename="./shots/test_1/test_1_kmv_p5.dat"
        spyro.io.save_shots(model, comm, u_at_recv, file_name=filename)

else:
    tn="test_1"
    u_cg_ref = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_cg_p5_h25m.dat")
    u_kmv_ref = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_kmv_p5_h25m.dat")
   
    u_cg_p2 = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_cg_p2.dat")
    u_cg_p3 = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_cg_p3.dat")
    u_cg_p4 = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_cg_p4.dat")
    u_cg_p5 = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_cg_p5.dat")
    
    u_kmv_p2 = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_kmv_p2.dat")
    u_kmv_p3 = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_kmv_p3.dat")
    u_kmv_p4 = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_kmv_p4.dat")
    u_kmv_p5 = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_kmv_p5.dat")
    
    # CG
    L2_cg_ref = (u_cg_ref)**2.
    L2_cg_ref = L2_cg_ref.sum()**0.5
    
    L2_cg_p2 = (u_cg_p2-u_cg_ref)**2. 
    L2_cg_p2 = 100*(L2_cg_p2.sum()**0.5)/L2_cg_ref
    
    L2_cg_p3 = (u_cg_p3-u_cg_ref)**2. 
    L2_cg_p3 = 100*(L2_cg_p3.sum()**0.5)/L2_cg_ref
    
    L2_cg_p4 = (u_cg_p4-u_cg_ref)**2. 
    L2_cg_p4 = 100*(L2_cg_p4.sum()**0.5)/L2_cg_ref
    
    L2_cg_p5 = (u_cg_p5-u_cg_ref)**2. 
    L2_cg_p5 = 100*(L2_cg_p5.sum()**0.5)/L2_cg_ref
    
    # KMV
    L2_kmv_ref = (u_kmv_ref)**2.
    L2_kmv_ref = L2_kmv_ref.sum()**0.5
    
    L2_kmv_p2 = (u_kmv_p2-u_kmv_ref)**2. 
    L2_kmv_p2 = 100*(L2_kmv_p2.sum()**0.5)/L2_kmv_ref
    
    L2_kmv_p3 = (u_kmv_p3-u_kmv_ref)**2. 
    L2_kmv_p3 = 100*(L2_kmv_p3.sum()**0.5)/L2_kmv_ref
    
    L2_kmv_p4 = (u_kmv_p4-u_kmv_ref)**2. 
    L2_kmv_p4 = 100*(L2_kmv_p4.sum()**0.5)/L2_kmv_ref
    
    L2_kmv_p5 = (u_kmv_p5-u_kmv_ref)**2. 
    L2_kmv_p5 = 100*(L2_kmv_p5.sum()**0.5)/L2_kmv_ref

    # prepare to plot
    L2_cg = np.array([L2_cg_p2, L2_cg_p3, L2_cg_p4, L2_cg_p5])
    dof_cg = np.array([14642, 32762, 58082, 90602])
    t_cg = np.array([11.58 , 28.49, 92.8, 272.47]) # for test 2

    L2_kmv = np.array([L2_kmv_p2, L2_kmv_p3, L2_kmv_p4, L2_kmv_p5])
    dof_kmv = np.array([21842, 47162, 79682, 155402]) 
    t_kmv = np.array([3.89, 6.58, 12.43, 32.31]) # for test 2 

    p = np.array([2, 3, 4, 5])

    plt.plot(dof_cg, L2_cg, label='Lagrange', marker="o")
    plt.plot(dof_kmv, L2_kmv, label='KMV', marker="o")
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("relative L2 norm (%)", fontsize=14)
    plt.xlabel("DOFs", fontsize=14)
    plt.title("Lagrange x KMV elements", fontsize=16)
    plt.legend()
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor')
    plt.show()

    
    plt.plot(p, L2_cg, label='Lagrange',marker="o")
    plt.plot(p, L2_kmv, label='KMV',marker="o")
    plt.ylabel("relative L2 norm (%)", fontsize=14)
    plt.xlabel("polynomial order", fontsize=14)
    plt.title("Lagrange x KMV elements", fontsize=16)
    plt.legend()
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor')
    plt.show()
    
    
    plt.plot(dof_cg, t_cg, label='Lagrange',marker="o")
    plt.plot(dof_kmv, t_kmv, label='KMV',marker="o")
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("simulation time (s)",fontsize=14)
    plt.xlabel("DOFs", fontsize=14)
    plt.title("Lagrange x KMV elements", fontsize=16)
    plt.legend()
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor')
    plt.show()
