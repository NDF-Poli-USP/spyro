import firedrake as fire
import spyro
import model_set
from spyro.io import ensemble_solvers_ad
from spyro.solvers.solver_ad import solver_ad

OMP_NUM_THREADS = 1
# from spyro.solvers.forward_AD import solver_ad

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

if vel_model == "circle":
    vp_exact = model_set._make_vp_circle(V, mesh, vp_guess=False)  # exact  
elif (vel_model == "marmousi" or vel_model == "br_model"):
    vp_exact = spyro.io.interpolate(model, mesh, V)
else:
    AssertionError("It is necessary to define the velocity field")              

fire.File("exact_vel.pvd").write(vp_exact)

solver_ad = solver_ad(model, mesh, V)


@ensemble_solvers_ad
def run_forward_true(solver_type, tot_source_num, comm, sn=0):
    print('######## Running the exact model ########')
    solver_ad.source_num = sn
    wp = solver_ad.wave_propagate
    output = wp(
                comm, vp_exact, sources, receivers, wavelet,
                output=True, save_rec_data=True
                )
    p_exact_recv = output[0]

    if comm.comm.rank == 0:
        spyro.io.save_shots(
                    model, comm, p_exact_recv
                    )


solver_type = "fwd"
for i in range(sources.num_receivers):
    run_forward_true(solver_type, sources.num_receivers, comm, sn=i)
