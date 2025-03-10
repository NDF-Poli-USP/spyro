import firedrake as fire
from spyro.solvers import DifferentiableWaveEquation
from demos.with_automatic_differentiation.utils import \
    model_settings, make_c_camembert
import os
os.environ["OMP_NUM_THREADS"] = "1"

# --- Basid setup to run a forward simulation with AD --- #

model = model_settings()

# Use emsemble parallelism.
M = model["parallelism"]["num_spacial_cores"]
my_ensemble = fire.Ensemble(fire.COMM_WORLD, M)
# my_ensemble.comm is the spatial communicator
mesh = fire.UnitSquareMesh(80, 80, comm=my_ensemble.comm)

wave_equation = DifferentiableWaveEquation(model, mesh)
c_true = make_c_camembert(mesh, wave_equation.function_space)

if model["parallelism"]["type"] is None:
    outfile = fire.VTKFile("solution.pvd")
    for sn in range(len(model["acquisition"]["source_pos"])):
        wave_equation.acoustic_solver(c_true, sn)
        sol = wave_equation.solution
        outfile.write(sol)
else:
    # source_number based on the ensemble.ensemble_comm.rank
    source_number = my_ensemble.ensemble_comm.rank
    wave_equation.acoustic_solver(c_true, source_number)
    sol = wave_equation.solution
    fire.VTKFile(
        "solution_" + str(source_number) + ".pvd", comm=my_ensemble.comm
    ).write(sol)
