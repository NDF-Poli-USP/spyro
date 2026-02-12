import firedrake as fire
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import spyro
from demos.with_automatic_differentiation.utils import \
    model_settings, make_c_camembert

os.environ["OMP_NUM_THREADS"] = "1"

# --- Basid setup to run a forward simulation with AD --- #

model = model_settings()

# Use emsemble parallelism.
M = model["parallelism"]["num_spacial_cores"]
my_ensemble = fire.Ensemble(fire.COMM_WORLD, M)
forward_solver = spyro.solvers.forward_ad.DifferentiableForwardSolver(model, comm=my_ensemble.comm)
print("Forward solver created")
