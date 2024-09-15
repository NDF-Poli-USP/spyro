import firedrake as fire
import spyro
from demos.with_automatic_differentiation.utils import \
      model_settings, make_c_camembert
import os
os.environ["OMP_NUM_THREADS"] = "1"

# --- Basid setup to run a forward simulation with AD --- #

model = model_settings()

# Use emsemble parallelism.
M = model["parallelism"]["num_spacial_cores"]
my_ensemble = fire.Ensemble(fire.COMM_WORLD, M)
mesh = fire.UnitSquareMesh(50, 50, comm=my_ensemble.comm)
element = fire.FiniteElement(
    model["opts"]["method"], mesh.ufl_cell(), degree=model["opts"]["degree"],
    variant=model["opts"]["quadrature"]
    )
V = fire.FunctionSpace(mesh, element)


forward_solver = spyro.solvers.forward_ad.ForwardSolver(model, mesh, V)

c_true = make_c_camembert(mesh, V)
# Ricker wavelet
wavelet = spyro.full_ricker_wavelet(
    model["timeaxis"]["dt"], model["timeaxis"]["tf"],
    model["acquisition"]["frequency"],
)

if model["parallelism"]["type"] is None:
    outfile = fire.VTKFile("solution.pvd")
    for sn in range(len(model["acquisition"]["source_pos"])):
        rec_data, _ = forward_solver.execute(c_true, sn, wavelet)
        sol = forward_solver.solution
        outfile.write(sol)
else:
    # source_number based on the ensemble.ensemble_comm.rank
    source_number = my_ensemble.ensemble_comm.rank
    rec_data, _ = forward_solver.execute_acoustic(
        c_true, source_number, wavelet)
    sol = forward_solver.solution
    fire.VTKFile(
        "solution_" + str(source_number) + ".pvd", comm=my_ensemble.comm
        ).write(sol)
