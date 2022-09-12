from firedrake import *
from scipy.optimize import * 
#from movement import *
import spyro
import time
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import h5py
import meshio
#import SeismicMesh
import weakref
import finat
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
from scipy.ndimage import gaussian_filter
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from spyro.io import write_function_to_grid, create_segy
#from ..domains import quadrature, space

# define the model parameters {{{
model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadrature": "KMV", # Equi or KMV #FIXME it will be removed
    "degree": 2,  # p order
    "dimension": 2,  # dimension
    "regularization": False,  # regularization is on?
    "gamma": 1e-5, # regularization parameter
}

model["parallelism"] = {
    "type": "automatic", # options: automatic (same number of cores for evey processor) or spatial
    #"type": "spatial",  # 
    #"custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    #"num_cores_per_shot": 1 #FIXME this is not used
}

model["mesh"] = {
    #"Lz": 1.5,  # depth in km - always positive
    "Lz": 2.00-.45,  # depth in km - always positive (waterbottom at z=-0.45 km)
    #"Lx": 2.0,  # width in km - always positive
    "Lx": 4.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "not_used.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "not_used.hdf5",
}

model["BCs"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "abl_bc": "none",  # none, gaussian-taper, or alid
    "lz": 0.0,  # thickness of the ABL in the z-direction (km) - always positive
    "lx": 0.0,  # thickness of the ABL in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the ABL in the y-direction (km) - always positive
}

model["acquisition"] = {
    "source_type": "Ricker",
    "frequency": 3.0, # 3 Hz for sigma=300, 5 Hz for sigma=100 
    #"frequency": 5.0, # 3 Hz for sigma=300, 5 Hz for sigma=100 
    #"frequency": 7.0, # 3 Hz for sigma=300, 5 Hz for sigma=100 
    "delay": 1.0, # FIXME check this
    "num_sources": 4, #FIXME not used (remove it, and update an example script)
    "source_pos": spyro.create_transect((0.5, -0.01-0.45), (3.5, -0.01-0.45), 4), # waterbottom at z=-0.45 km
    "amplitude": 1.0, #FIXME check this
    "num_receivers": 100, #FIXME not used (remove it, and update an example script)
    "receiver_locations": spyro.create_transect((0.1, -0.10-0.45), (3.9, -0.10-0.45), 100), # waterbottom at z=-0.45 km REC1
    #"receiver_locations": spyro.create_transect((0.1, -1.9), (3.9, -1.9), 100), # receivers at the bottom of the domain (z=-1.9 km) REC2 
    #"receiver_locations": spyro.create_2d_grid(1, 3, -1.4, -1, 10) # 10^2 points REC3
}

model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 2.5, # Final time for event 
    "dt": 0.00025,  # timestep size 
    "nspool":  20,  # (20 for dt=0.00050) how frequently to output solution to pvds
    "fspool": 10,  # how frequently to save solution to RAM
}
#}}}
# make vp, vs or rho {{{
def _make_field(V, guess=False, field="vp"):
    
    path = "/share/tdsantos/velocity_models/elastic-marmousi-model/model/"
    if guess: # interpolate from a smoothed field
        sys.exit("not implemented yet")
        fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain_smoothed_sigma=300.segy.hdf5" # domain 4 x 2 km2 (x, y) 
    else: # interpolate from the exact field
        if field=="vp":
            fname = path + "MODEL_P-WAVE_VELOCITY_1.25m_small_domain.segy.hdf5" # domain 4 x 2 km2 (x, y) 
        elif field=="vs": 
            fname = path + "MODEL_S-WAVE_VELOCITY_1.25m_small_domain.segy.hdf5" # domain 4 x 2 km2 (x, y) 
        elif field=="rho":
            fname = path + "MODEL_DENSITY_1.25m_small_domain.segy.hdf5" # domain 4 x 2 km2 (x, y) 
        else:
            sys.exit("field not found")

    if field=="vp" or field=="vs":
        _field = "velocity_model"
        _scale = 1000
    elif field=="rho":
        _field = "density"
        _scale = 1
    else:
        sys.exit("field not found")
    
    with h5py.File(fname, "r") as f:
        Zo = np.asarray(f.get(_field)[()]) # original Marmousi data/domain
        nrow, ncol = Zo.shape
        Lz = 2 # Value defined by the velocity model
        Lx = 4 # Value defined by the velocity model
        zo = np.linspace(-Lz, 0.0, nrow) # original Marmousi data/domain
        xo = np.linspace(0.0,  Lx, ncol) # original Marmousi data/domain
        interpolant = RegularGridInterpolator((xo, zo), np.transpose(Zo))

        m = V.ufl_domain()
        W = VectorFunctionSpace(m, V.ufl_element())
        coords = interpolate(m.coordinates, W)
        xq, zq = coords.dat.data[:, 0], coords.dat.data[:, 1]

        _fd = interpolant((xq, zq))
        if field == "vs": # FIXME it should be fixed using a different domain
            _fd[np.where(_fd < 0.001)] = 300 # small part of water layer 
        
        fd = Function(V)
        fd.dat.data[:] = _fd / _scale # m/s -> km/s for vp and vs (rho is in g/cm3==Gt/km3)

        if guess:
            File("guess_"+field+".pvd").write(fd)
        else:
            File("exact_"+field+".pvd").write(fd)
    
    return fd
#}}}
comm = spyro.utils.mpi_init(model)
distribution_parameters={"partition": True,
                         "overlap_type": (DistributedMeshOverlapType.VERTEX, 60)}

REF = 0
# run reference model {{{
if REF:
    nx = 200
    ny = math.ceil( 100*(model["mesh"]["Lz"]-0.45)/model["mesh"]["Lz"] ) # (Lz-0.45)/Lz
    mesh_ref = RectangleMesh(nx, ny, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                            distribution_parameters=distribution_parameters)
    mesh_ref.coordinates.dat.data[:, 0] -= 0.0 
    mesh_ref.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"] + 0.45 # waterbottom at z=-0.450 km

    # for the exact model, use a higher-order element
    element = spyro.domains.space.FE_method(mesh_ref, model["opts"]["method"], model["opts"]["degree"])
    V_ref = FunctionSpace(mesh_ref, element) # FIXME come back here

    element_DG = spyro.domains.space.FE_method(mesh_ref, "DG", 2) # here, it could be 2 too
    V_DG = FunctionSpace(mesh_ref, element_DG)

    sources = spyro.Sources(model, mesh_ref, V_ref, comm)
    receivers = spyro.Receivers(model, mesh_ref, V_ref, comm)
    wavelet = spyro.full_ricker_wavelet(dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"])

    # generate the Lamé parameter
    vp_ref  = _make_field(V_DG, guess=False, field="vp")
    vs_ref  = _make_field(V_DG, guess=False, field="vs")
    rho_ref = _make_field(V_DG, guess=False, field="rho")
    
    mu_ref   = Function(V_DG).interpolate(rho_ref * vs_ref ** 2.)
    lamb_ref = Function(V_DG).interpolate(rho_ref * (vp_ref ** 2. - 2. * vs_ref ** 2.))
   
    File("mu_ref.pvd").write(mu_ref)
    File("lamb_ref.pvd").write(lamb_ref)
    
    #sys.exit("exit")
    
    print("Starting forward computation of the exact model",flush=True) 
    start = time.time()
    u_ref, uz_ref, ux_ref, uy_ref = spyro.solvers.forward_elastic_waves(
        model, mesh_ref, comm, rho_ref, lamb_ref, mu_ref, sources, wavelet, receivers, output=False
    )
    end = time.time()
    print(round(end - start,2),flush=True)
    File("u_ref.pvd").write(u_ref[-1])

    spyro.io.save_shots(model, comm, uz_ref, file_name="./shots/uz_ref_recv1")
    spyro.io.save_shots(model, comm, ux_ref, file_name="./shots/ux_ref_recv1")

#}}}
#sys.exit("exit")
if REF==0: 
    uz_ref = spyro.io.load_shots(model, comm, file_name="/share/tdsantos/shots/uz_ref_recv1")
    ux_ref = spyro.io.load_shots(model, comm, file_name="/share/tdsantos/shots/ux_ref_recv1")

# now, prepare to run with different mesh resolutions
FIREMESH = 0
# generate or read a mesh {{{
if FIREMESH: 
    nx = 200
    ny = math.ceil( 100*(model["mesh"]["Lz"]-0.45)/model["mesh"]["Lz"] ) # (Lz-0.45)/Lz
    nx = 180 # FIXME
    ny = 64
    mesh = RectangleMesh(nx, ny, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                            distribution_parameters=distribution_parameters)
    mesh.coordinates.dat.data[:, 0] -= 0.0 
    mesh.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"] + 0.45 # waterbottom at z=-0.450 km

    element = spyro.domains.space.FE_method(mesh, model["opts"]["method"], model["opts"]["degree"])
    V = FunctionSpace(mesh, element)
else:
    p = 2
    M = 7.02
    #model["mesh"]["meshfile"] = "/share/tdsantos/meshes/fwi_amr_marmousi_small_p=" + str(p) + "_M=" + str(M) + ".msh"
    #model["mesh"]["meshfile"] = "/share/tdsantos/meshes/marmousi_small_no_water_h_150m.msh"
    #model["mesh"]["meshfile"] = "/share/tdsantos/meshes/marmousi_small_no_water_h_100m.msh"
    model["mesh"]["meshfile"] = "/share/tdsantos/meshes/marmousi_small_no_water_h_50m.msh"
    mesh, V = spyro.io.read_mesh(model, comm, distribution_parameters=distribution_parameters)
#}}}

AMR = 1
# adapt the mesh using the exact vp, if requested {{{
if AMR:
    nx = 200
    ny = math.ceil( 100*(model["mesh"]["Lz"]-0.45)/model["mesh"]["Lz"] ) # (Lz-0.45)/Lz
    mesh_grid = RectangleMesh(nx, ny, model["mesh"]["Lx"], model["mesh"]["Lz"], diagonal="crossed", comm=comm.comm,
                            distribution_parameters=distribution_parameters)
    mesh_grid.coordinates.dat.data[:, 0] -= 0.0 
    mesh_grid.coordinates.dat.data[:, 1] -= model["mesh"]["Lz"] + 0.45 # waterbottom at z=-0.45 km
    V_grid = FunctionSpace(mesh_grid, "CG", 2)
    V_grid_DG = FunctionSpace(mesh_grid, "DG", 2)
    V_vec_grid = VectorFunctionSpace(mesh_grid, "CG", 2)
  
    vs_grid = _make_field(V_grid_DG, guess=False, field="vs")
    grad_vs_grid = Function(V_vec_grid)

    u_cts = TrialFunction(V_vec_grid)
    v_cts = TestFunction(V_vec_grid)
    a = inner(v_cts, u_cts)*dx
    L = inner(v_cts, grad(vs_grid))*dx
    _cg = {
        "ksp_type": "cg",
        "pc_type": "bjacobi",
        "pc_sub_type": "ilu",
    }
    _problem = LinearVariationalProblem(a, L, grad_vs_grid, bcs=None)
    _solver = LinearVariationalSolver(_problem, solver_parameters=_cg) 
    _solver.solve()

    File("grad_vs_grid.pvd").write(grad_vs_grid)
    File("vs_grid.pvd").write(vs_grid)

    # Huang type monitor function
    E1 = sqrt( inner( grad_vs_grid, grad_vs_grid ) ) # gradient based estimate
    E2 = vs_grid.vector().gather().max() / vs_grid - 1 # a priori error estimate (it starts on 1, so it could be better)

    #beta = 0.5 # (0, 1) # for E1
    E = E1
    beta = 0.5 # (0, 1) # for E2
    phi = sqrt( 1 + E*E ) - 1
    phi_hat = assemble(phi*dx(domain=mesh_grid)) / assemble(Constant(1.0)*dx(domain=mesh_grid))
    alpha = beta / ( phi_hat * ( 1 - beta ) )
    M1 = 1 + alpha * phi
   
    E = E2
    beta = 0.5 # (0, 1) # for E2
    phi = sqrt( 1 + E*E ) - 1
    phi_hat = assemble(phi*dx(domain=mesh_grid)) / assemble(Constant(1.0)*dx(domain=mesh_grid))
    alpha = beta / ( phi_hat * ( 1 - beta ) )
    M2 = 1 + alpha * phi

    M = max_value(M1,M2)

    # Define the monitor function to be projected onto the adapted mesh
    Mfunc = Function(V_grid)
    Mfunc.interpolate(M)

    # smooth the monitor function
    if 1: # {{{
        lamb = 0.005

        u = TrialFunction(V_grid)
        v = TestFunction(V_grid)
        u_n = Function(V_grid)

        m = inner((u - u_n), v) * dx 
        a = lamb * inner(grad(u), grad(v)) * dx # implicit
        F = m + a

        lhs_ = lhs(F)
        rhs_ = rhs(F)

        X = Function(V_grid)
        B = Function(V_grid)
        A = assemble(lhs_)

        params = {"ksp_type": "preonly", "pc_type": "lu"}
        solver = LinearSolver(A, solver_parameters=params)

        u_n.assign(Mfunc)
        nt = 1
        for step in range(nt):
            B = assemble(rhs_, tensor=B)
            solver.solve(X, B)
            u_n.assign(X)

        Mfunc.assign(u_n)
    #}}}

    File("Mfunc.pvd").write(Mfunc)
    #sys.exit("exit")

    def monitor_function(mesh): # here, mesh is the physical doman, i.e., x (=xi+Grad phi)
        # project onto "mesh" that is being adapted (i.e., mesh_x)
        _P1 = FunctionSpace(mesh, "CG", 1) 
        _M = Function(_P1)
        spyro.mesh_to_mesh_projection(Mfunc, _M, degree=5)
        File("Mfunc_x.pvd").write(_M)
        return _M
    
    V_DG = FunctionSpace(mesh, "DG", 2)
    _vs = _make_field(V_DG, guess=False, field="vs")
    File("vp_before_amr.pvd").write(_vs)
    #sys.exit("exit")
        
    def mask_receivers(mesh):
        _x,_y = mesh.coordinates
        g = conditional(_y < -1.99, 0, 1) # 0 apply BC
        g = conditional(_y > -0.01, 0, g)
        g = conditional(_x < 0.01, 0, g)
        g = conditional(_x > 3.99, 0, g)
        return g
    
    def mask_dummy(mesh):
        return Constant(1.)

    #fix_boundary_nodes = False
    mask = mask_dummy
    #if FIREMESH==0:
    #    fix_boundary_nodes = True
    #    mask = mask_receivers

    mesh._parallel_compatible = {weakref.ref(mesh_grid)}
    start = time.time()
    step = spyro.monge_ampere_solver(mesh, monitor_function, p=1, mask=mask) #fix_boundary_nodes=fix_boundary_nodes) 
    end = time.time()
    print(round(end - start,2),flush=True)

    _vs = _make_field(V_DG, guess=False, field="vs")
    File("vp_after_amr.pvd").write(_vs)
#}}}
sys.exit("exit")

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(dt=model["timeaxis"]["dt"], tf=model["timeaxis"]["tf"], freq=model["acquisition"]["frequency"])

V_DG = FunctionSpace(mesh, "DG", 2)
# generate the Lamé parameters
vp  = _make_field(V_DG, guess=False, field="vp")
vs  = _make_field(V_DG, guess=False, field="vs")
rho = _make_field(V_DG, guess=False, field="rho")

mu   = Function(V_DG).interpolate(rho * vs ** 2.)
lamb = Function(V_DG).interpolate(rho * (vp ** 2. - 2. * vs ** 2.))

u, uz, ux, uy = spyro.solvers.forward_elastic_waves(
    model, mesh, comm, rho, lamb, mu, sources, wavelet, receivers, output=False
)
        
J_scale = sqrt(1.e14)
misfit_uz = spyro.utils.evaluate_misfit(model, uz, uz_ref)# ds_exact[:ll] - guess
misfit_ux = spyro.utils.evaluate_misfit(model, ux, ux_ref)# ds_exact[:ll] - guess

J_total = np.zeros((1))
J_ref   = np.zeros((1))
J_total[0] += spyro.utils.compute_functional(model, J_scale * misfit_uz) # J += residual[ti][rn] ** 2 (and J *= 0.5))
J_total[0] += spyro.utils.compute_functional(model, J_scale * misfit_ux) # J += residual[ti][rn] ** 2 (and J *= 0.5)

print(J_total[0],flush=True)

J_ref[0]   += spyro.utils.compute_functional(model, J_scale * uz_ref) # J += p_ref_recv[ti][rn] ** 2 (and J *= 0.5)
J_ref[0]   += spyro.utils.compute_functional(model, J_scale * ux_ref) # J += p_ref_recv[ti][rn] ** 2 (and J *= 0.5)

# divide
J_total[0] /= J_ref[0]

J_total = COMM_WORLD.allreduce(J_total, op=MPI.SUM)
J_total[0] /= comm.ensemble_comm.size # paralelismo ensemble (fontes)
if comm.comm.size > 1:
    J_total[0] /= comm.comm.size # paralelismo espacial

E = sqrt(J_total[0]) # relative error as defined in Spyro paper
print(E, flush=True)

write_files=1
if comm.ensemble_comm.rank == 1 and write_files==1:
    uz_rec_ref = []
    uz_rec = []
    nt = int(model["timeaxis"]["tf"] / model["timeaxis"]["dt"])
    rn = 0
    for ti in range(nt):
        uz_rec_ref.append(uz_ref[ti][rn])
        uz_rec.append(uz[ti][rn])
    plt.title("uz at receiver")
    plt.plot(uz_rec_ref,label='uz ref')
    plt.plot(uz_rec,label='uz')
    plt.legend()
    plt.savefig('/home/tdsantos/uz_rec_1.png')
    plt.close()

  
