from firedrake import (
    RectangleMesh,
    FunctionSpace,
    Function,
    SpatialCoordinate,
    conditional,
    File,
    Constant,
    exp
)
import spyro
import time
import sys
import matplotlib.pyplot as plt
import numpy as np

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
    "dt": 0.0005,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool":  20,  # how frequently to output solution to pvds
    "fspool": 100,  # how frequently to save solution to RAM
}

mesh = RectangleMesh(50, 50, 1.5, 1.5)
mesh.coordinates.dat.data[:, 0] -= 1.5
mesh.coordinates.dat.data[:, 1] -= 0.0

comm = spyro.utils.mpi_init(model)

element = spyro.domains.space.FE_method(
    mesh, model["opts"]["method"], model["opts"]["degree"]
)

V = FunctionSpace(mesh, element)
#print(V.dof_count*2)

lamb = Constant(1./2.) 
mu = Constant(1./4.)  
rho = Constant(1.) 
#sys.exit("exiting without running")

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
                        dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"]
                        )

post_process=True
if post_process==False:
    start = time.time()
    u_field, u_at_recv = spyro.solvers.forward_elastic_waves(
        model, mesh, comm, rho, lamb, mu, sources, wavelet, receivers, output=False
    )
    end = time.time()
    print(round(end - start,2))

    #cmin=-1e-4
    #cmax=1e-4
    #spyro.plots.plot_shots(model, comm, u_at_recv, show=True, vmin=cmin, vmax=cmax)

    #filename="./shots/test_2_cg_p7.dat"
    #spyro.io.save_shots(model, comm, u_at_recv, file_name=filename)

else:
    tn="test_1"
    u_ref = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_cg_p7.dat")
   
    u_cg_p2 = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_cg_p2.dat")
    u_cg_p3 = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_cg_p3.dat")
    u_cg_p4 = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_cg_p4.dat")
    u_cg_p5 = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_cg_p5.dat")
    
    u_kmv_p2 = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_kmv_p2.dat")
    u_kmv_p3 = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_kmv_p3.dat")
    u_kmv_p4 = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_kmv_p4.dat")
    u_kmv_p5 = spyro.io.load_shots(model, comm, file_name="./shots/"+tn+"/"+tn+"_kmv_p5.dat")
   
    L2_ref = (u_ref)**2.
    L2_ref = L2_ref.sum()**0.5
    
    # CG
    L2_cg_p2 = (u_cg_p2-u_ref)**2. 
    L2_cg_p2 = 100*(L2_cg_p2.sum()**0.5)/L2_ref
    
    L2_cg_p3 = (u_cg_p3-u_ref)**2. 
    L2_cg_p3 = 100*(L2_cg_p3.sum()**0.5)/L2_ref
    
    L2_cg_p4 = (u_cg_p4-u_ref)**2. 
    L2_cg_p4 = 100*(L2_cg_p4.sum()**0.5)/L2_ref
    
    L2_cg_p5 = (u_cg_p5-u_ref)**2. 
    L2_cg_p5 = 100*(L2_cg_p5.sum()**0.5)/L2_ref
    
    # KMV
    L2_kmv_p2 = (u_kmv_p2-u_ref)**2. 
    L2_kmv_p2 = 100*(L2_kmv_p2.sum()**0.5)/L2_ref
    
    L2_kmv_p3 = (u_kmv_p3-u_ref)**2. 
    L2_kmv_p3 = 100*(L2_kmv_p3.sum()**0.5)/L2_ref
    
    L2_kmv_p4 = (u_kmv_p4-u_ref)**2. 
    L2_kmv_p4 = 100*(L2_kmv_p4.sum()**0.5)/L2_ref
    
    L2_kmv_p5 = (u_kmv_p5-u_ref)**2. 
    L2_kmv_p5 = 100*(L2_kmv_p5.sum()**0.5)/L2_ref

    # prepare to plot
    L2_cg = np.array([L2_cg_p2, L2_cg_p3, L2_cg_p4, L2_cg_p5])
    dof_cg = np.array([20402, 45602, 80802, 126002])
    t_cg = np.array([14.97 , 36.38, 127.37, 356.32])

    L2_kmv = np.array([L2_kmv_p2, L2_kmv_p3, L2_kmv_p4, L2_kmv_p5])
    dof_kmv = np.array([30402, 65602, 110802, 216002])
    t_kmv = np.array([4.77, 8.87, 16.62, 48.2])

    p = np.array([2, 3, 4, 5])

    plt.plot(dof_cg, L2_cg, label='L2 cg',marker="o")
    plt.plot(dof_kmv, L2_kmv, label='L2 kmv',marker="o")
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("relative L2 norm (%)")
    plt.xlabel("DOFs")
    if tn=="test_1":
        plt.title("CG x KMV (test 1, only P wave)")
    else:
        plt.title("CG x KMV (test 2, P and S waves)")
    
    plt.legend()
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor')
    plt.show()

    
    plt.plot(p, L2_cg, label='L2 cg',marker="o")
    plt.plot(p, L2_kmv, label='L2 kmv',marker="o")
    plt.ylabel("relative L2 norm (%)")
    plt.xlabel("polynomial order")
    if tn=="test_1":
        plt.title("CG x KMV (test 1, only P wave)")
    else:
        plt.title("CG x KMV (test 2, P and S waves)")
    plt.legend()
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor')
    plt.show()
    
    
    plt.plot(dof_cg, t_cg, label='time cg',marker="o")
    plt.plot(dof_kmv, t_kmv, label='time kmv',marker="o")
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("simulation time (s)")
    plt.xlabel("DOFs")
    if tn=="test_1":
        plt.title("CG x KMV (test 1, only P wave)")
    else:
        plt.title("CG x KMV (test 2, P and S waves)")
    plt.legend()
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor')
    plt.show()
