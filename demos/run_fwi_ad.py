
import firedrake as fire
import time as tm
import spyro
import model_set
from scipy.ndimage import gaussian_filter
from scipy import optimize
from mpi4py import MPI
from spyro.io import ensemble_solvers_ad

outdir = "fwi/"
global nshots

vel_model = "circle"
model = model_set.model_settings(vel_model)
comm = spyro.utils.mpi_init(model)
nshots = model["acquisition"]["num_sources"]  
obj = []

mesh, V = spyro.io.read_mesh(model, comm)

sources = spyro.Sources(model, mesh, V, comm)
receivers = spyro.Receivers(model, mesh, V, comm)
wavelet = spyro.full_ricker_wavelet(
                dt=model["timeaxis"]["dt"], 
                tf=model["timeaxis"]["tf"], 
                freq=model["acquisition"]["frequency"]
            )

p_exact_recv = None

if vel_model == "circle":
    vp_guess = model_set._make_vp_circle(V, mesh, vp_guess=True)

if (vel_model == "marmousi" or vel_model == "br_model"):
    vp_guess = spyro.io.interpolate(
                model, mesh, V, smooth_vel=True, sigma=100
                )  

if comm.ensemble_comm.rank == 0:
    control_file = fire.File(outdir + "control.pvd", comm=comm.comm)
    grad_file = fire.File(outdir + "grad.pvd", comm=comm.comm)

solver_ad = spyro.solvers.solver_ad.solver_ad(model, mesh, V)
wp = solver_ad.wave_propagate


def runfwi(solver_type, tot_source_num, comm, xi, sn=0):
    solver_ad.source_num = sn
    aut_dif = model["aut_dif"]["status"]
    if aut_dif:
        import firedrake_adjoint as fire_adj
    
    local_mesh_index = mesh.coordinates.node_set.halo.local_to_global_numbering
    vp_guess = spyro.utils.scatter_data_function(xi, V, comm, local_mesh_index, name="vp_guess")
    if comm.ensemble_comm.rank == 0:
        control_file.write(vp_guess)
        # fire.File("guess_br_ad.pvd", comm=comm.comm).write(vp_guess)
    
    print('######## Running the guess model ########')
    
    if aut_dif:
        out = wp(
                comm, vp_guess, sources, receivers, wavelet,
                calc_functional=True, true_rec=p_exact_recv,
                output=True
                )
        Jm = out[0]
        # quit()
        
        control = fire_adj.Control(vp_guess)
        # J_hat = fire_adj.ReducedFunctional(Jm, control)
        # fire_adj.minimize(J_hat, options={'disp': True, "maxiter": 5})

        comp_grad = fire_adj.compute_gradient
        
        dJ = comp_grad(Jm, control, "riesz_representation" == "L2")
        fire_adj.get_working_tape().clear_tape()

    else:
        out = wp(
                comm, vp_guess, true_rec=p_exact_recv,
                calc_functional=True, save_misfit=True,
                save_p=True
                )
        p_guess = out[0]
        Jm = out[1]
        misfit = out[2]
        solver_ad.solver = "bwd"
        out = wp(comm, vp_guess, misfit=misfit, p_guess=p_guess)
        dJ = out[0]
    return Jm, dJ


def run_source(xi):
    solver_type = "fwi"
    Jm, dJ_local = runfwi(solver_type, sources.num_receivers, comm, xi)
    dJ = fire.Function(V, name="gradient")
    if comm.ensemble_comm.size > 1:
        comm.allreduce(dJ_local, dJ)
    else:
        dJ = dJ_local
    J_total = fire.COMM_WORLD.allreduce(Jm, op=MPI.SUM)
    # dJ /= comm.ensemble_comm.size

    # if comm.ensemble_comm.rank == 0:
    #     grad_file.write(dJ)

    return J_total, dJ.dat.data


# if __name__ == "__main__":  
p_exact_recv = spyro.io.load_shots(model, comm)
                            
vmax = 3.5
vmin = 1.5
m0 = vp_guess.vector().gather()
bounds = [(vmin, vmax) for _ in range(len(m0))]

solver_type = "fwi"
# runfwi(solver_type, sources.num_receivers, comm, m0, sn=0)

# start = tm.time()
result_da = optimize.minimize(
                run_source, m0, method='L-BFGS-B',
                jac=True, tol=1e-15, bounds=bounds,
                options={"disp": True, "eps": 1e-15, "gtol": 1e-15, "maxiter": 5}
            )

    # rank = comm.comm.rank
    # size = comm.comm.size

    # vp_end = fire.Function(V) 
    # n = len(vp_end.dat.data[:])
    # N = [comm.comm.bcast(n, r) for r in range(size)]
    # indices = np.insert(np.cumsum(N), 0, 0)
    # vp_end.dat.data[:] = result_da.x[indices[rank]:indices[rank+1]]
    # fire.File("vp_scipy_mm_ad.pvd", comm=comm.comm).write(vp_end)