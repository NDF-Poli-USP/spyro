import firedrake as fire
from irksome import GaussLegendre, Dt, MeshConstant, StageDerivativeNystromTimeStepper


mesh = UnitSquareMesh(100,100)
V = FunctionSpace(mesh, "KMV", 4)

u = fire.TrialFunction(V)
v = fire.TestFunction(V)

u_nm1 = fire.Function(V, name="pressure t-dt")
u_n = fire.Function(V, name="pressure")
u_np1 = fire.Function(V, name="pressure t+dt")
Wave_object.u_nm1 = u_nm1
Wave_object.u_n = u_n
Wave_object.u_np1 = u_np1

Wave_object.current_time = 0.0
dt = Wave_object.dt

# -------------------------------------------------------
m1 = (
    (1 / (Wave_object.c * Wave_object.c))
    * Dt(u, 2)
    * v
    * dx(scheme=quad_rule)
)
a = dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)  # explicit

le = 0
q = Wave_object.source_expression
if q is not None:
    le += q * v * dx(scheme=quad_rule)

B = fire.Cofunction(V.dual())

