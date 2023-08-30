from copy import deepcopy
from firedrake import *
import spyro

from test.model import dictionary as model
model["acquisition"]["source_type"] = "MMS"


def run_solve(model):
    testmodel = deepcopy(model)

    Wave_obj = spyro.AcousticWaveMMS(dictionary=testmodel)
    Wave_obj.set_mesh(dx=0.02)
    Wave_obj.set_initial_velocity_model(expression="1 + sin(pi*-z)*sin(pi*x)")
    Wave_obj.forward_solve()

    u_an = Wave_obj.analytical
    u_num = Wave_obj.u_n

    return errornorm(u_num, u_an)


model["cell_type"] = "triangles"
model["variant"] = "lumped"
error = run_solve(model)
