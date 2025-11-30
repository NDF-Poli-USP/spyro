import firedrake as fire
from firedrake import Function, TestFunction, TrialFunction, Cofunction
from firedrake import Constant, Identity
from firedrake import exp, dot, grad, dx, dS, div, inner, sym
import math
import numpy as np

# time
T  = 1.0
dt = fire.Constant(5e-4)
t  = fire.Constant(0)

# fluid
rho_f = 1.0
K     = 4.0

# solid
rho_s = 2.0
mu    = 8.0
lam   = 12.0

# mesh
mesh = fire.Mesh("acoustic_elastic.msh")
fluid_id     = 1
solid_id     = 2
interface_id = 3

# ===============================================================================
# function spaces:

V_F = fire.FunctionSpace(mesh, "KMV", 2)
V_S = fire.VectorFunctionSpace(mesh, "KMV", 2)

# ===============================================================================
# functions:

# fluid
p       = Function(V_F, name="p")
p_n     = Function(V_F)
p_nm1   = Function(V_F)
p_trial = TrialFunction(V_F)
q       = TestFunction(V_F)

# solid
u       = Function(V_S, name="u")
u_n     = Function(V_S)
u_nm1   = Function(V_S)
u_trial = TrialFunction(V_S)
v       = TestFunction(V_S)

# ===============================================================================
# normal unit vector:

n   = fire.FacetNormal(mesh)
n_f = n("+")
n_s = n("-")

# ===============================================================================
# source:

freq = 6

def RickerWavelet(t, freq, amp=1.0):
    t_shifted = t - 1.0/freq
    return amp * (1 - 2*(math.pi*freq*t_shifted)**2) * exp(-(math.pi*freq*t_shifted)**2)

def delta(time_expr, q, mesh, source_locations):
    """Creates a point source using VertexOnlyMesh approach."""
    vom = fire.VertexOnlyMesh(mesh, source_locations)
    # Scalar function space
    P0 = fire.FunctionSpace(vom, "DG", 0)
    Fvom = fire.Cofunction(P0.dual()).assign(1)
    return fire.interpolate(time_expr * q, Fvom)

x, y   = fire.SpatialCoordinate(mesh)
# source_location =(2.0, 2.5)
source_location =(1.98, 2.51962)

ricker = RickerWavelet(t, freq, amp=1.0)
F_s = delta(ricker, q, mesh, [source_location])

# ===============================================================================
# solvers of equations:

def eps(x):
    return sym(grad(x))

def sigma(x):
    return lam * div(x) * Identity(mesh.geometric_dimension()) + 2 * mu * eps(x)

# fluid
# ddot_p = (p_trial - 2.0*p_n + p_nm1) / Constant(dt * dt)

# F_p = (1/K) * ddot_p * q * dx(domain=mesh, scheme="lump")\
#       + q("+") * dot(sigma(u_n("-")) * n_f, n_f) * dS(interface_id) \
#       + (1/rho_f) * dot(grad(q), grad(p_n)) * dx(fluid_id, scheme="lump") \
#       - F_s

# a_p, r_p = fire.lhs(F_p), fire.rhs(F_p)

a_p = (1/K) * (p_trial) / Constant(dt * dt) * q * dx(domain=mesh, scheme="lump")
r_p = -(1/K) * (- 2.0*p_n + p_nm1) / Constant(dt * dt) * q * dx(domain=mesh, scheme="lump")\
      - q("+") * dot(sigma(u_n("-")) * n_f, n_f) * dS(interface_id) \
      - (1/rho_f) * dot(grad(q), grad(p_n)) * dx(fluid_id, scheme="lump") \
      + F_s

A_p = fire.assemble(a_p)
solver_f = fire.LinearSolver(A_p, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
R_f = Cofunction(V_F.dual())

# solid

# ddot_u = (u_trial - 2.0*u_n + u_nm1) / Constant(dt * dt)

# F_u = rho_s * dot(v, ddot_u) * dx(domain=mesh, scheme="lump") \
#       - p_n("+") * dot(v("-"), n_s) * dS(interface_id) \
#       + inner(eps(v), sigma(u_n)) * dx(solid_id, scheme="lump")

# a_u, r_u = fire.lhs(F_u), fire.rhs(F_u)

a_u = rho_s * dot(v, (u_trial) / Constant(dt * dt)) * dx(domain=mesh, scheme="lump")
r_u = -rho_s * dot(v, (- 2.0*u_n + u_nm1) / Constant(dt * dt)) * dx(domain=mesh, scheme="lump") \
    + p_n("+") * dot(v("-"), n_s) * dS(interface_id) \
    - inner(eps(v), sigma(u_n)) * dx(solid_id, scheme="lump")

A_u = fire.assemble(a_u)
solver_s = fire.LinearSolver(A_u, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
R_s = Cofunction(V_S.dual())

# ===============================================================================
# store data:
outfile = fire.VTKFile("results_acoustic_elastic/acoustic_elastic.pvd")

# ===============================================================================
# initial conditions:

p.assign(0)
p_n.assign(0)
p_nm1.assign(0)
u.assign(Constant((0,0)))
u_n.assign(Constant((0,0)))
u_nm1.assign(Constant((0,0)))

# ===============================================================================
# computation loop:
step = 0
while float(t) < T:
    step += 1
    # ricker.assign(RickerWavelet(t, freq))

    # fluid
    R_f = fire.assemble(r_p, tensor=R_f)
    solver_f.solve(p, R_f)

    # solid
    R_s = fire.assemble(r_u, tensor=R_s)
    solver_s.solve(u, R_s)

    p_nm1.assign(p_n)
    p_n.assign(p)
    u_nm1.assign(u_n)
    u_n.assign(u)

    t.assign(float(t) + float(dt))
    if step % 10 == 0:
        print("Elapsed time is: " +str(float(t)))
        assdebug = fire.assemble(F_s)
        outfile.write(p, u, time=float(t))    

# ===============================================================================
