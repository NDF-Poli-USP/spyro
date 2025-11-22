# imports:

from firedrake import *
import math
import numpy as np

# ===============================================================================
# parameters:

# time
T  = 1.0
dt = 5e-4
t  = 0

# fluid
rho_f = 1.0
K     = 4.0

# solid
rho_s = 2.0
mu    = 8.0
lam   = 12.0

# mesh
mesh = Mesh("acoustic_elastic.msh")
fluid_id     = 1
solid_id     = 2
interface_id = 3

# ===============================================================================
# function spaces:

V_F = FunctionSpace(mesh, "CG", 2)
V_S = VectorFunctionSpace(mesh, "CG", 2)

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

n   = FacetNormal(mesh)
n_f = n("+")
n_s = n("-")

# ===============================================================================
# source:

freq = 6

def RickerWavelet(t, freq, amp=1.0):
    t_shifted = t - 1.0 / freq
    factor = 1 - 2 * math.pi**2 * (freq**2) * (t_shifted**2)
    envelope = math.exp(-math.pi**2 * (freq**2) * (t_shifted**2))
    return amp * factor * envelope

def delta_expr(x0, x, y, sigma_x=2000.0):
    sigma_x = Constant(sigma_x)
    return exp(-sigma_x * ((x - x0[0])**2 + (y - x0[1])**2))

x, y   = SpatialCoordinate(mesh)
source = Constant([2.0, 2.5])
ricker = Constant(0.0)
ricker.assign(RickerWavelet(t, freq))

f = delta_expr(source, x, y) * ricker

# ===============================================================================
# solvers of equations:

def eps(x):
    return sym(grad(x))

def sigma(x):
    return lam * div(x) * Identity(mesh.geometric_dimension()) + 2 * mu * eps(x)

# fluid
ddot_p = (p_trial - 2.0*p_n + p_nm1) / Constant(dt * dt)

F_p = (1/K) * ddot_p * q * dx(domain=mesh)\
      + q("+") * dot(sigma(u_n("-")) * n_f, n_f) * dS(interface_id) \
      + (1/rho_f) * dot(grad(q), grad(p_n)) * dx(fluid_id) \
      - (f/K) * q * dx(fluid_id)

a_p, r_p = lhs(F_p), rhs(F_p)
A_p = assemble(a_p)
solver_f = LinearSolver(A_p, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
R_f = Cofunction(V_F.dual())

# solid


ddot_u = (u_trial - 2.0*u_n + u_nm1) / Constant(dt * dt)

F_u = rho_s * dot(v, ddot_u) * dx(domain=mesh) \
      - p_n("+") * dot(v("-"), n_s) * dS(interface_id) \
      + inner(eps(v), sigma(u_n)) * dx(solid_id)

a_u, r_u = lhs(F_u), rhs(F_u)
A_u = assemble(a_u)
solver_s = LinearSolver(A_u, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
R_s = Cofunction(V_S.dual())

# ===============================================================================
# store data:
outfile = File("results_acoustic_elastic/acoustic_elastic.pvd")

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
while t < T:
    step += 1
    ricker.assign(RickerWavelet(t, freq))

    # fluid
    R_f = assemble(r_p, tensor=R_f)
    solver_f.solve(p, R_f)

    # solid
    R_s = assemble(r_u, tensor=R_s)
    solver_s.solve(u, R_s)

    p_nm1.assign(p_n)
    p_n.assign(p)
    u_nm1.assign(u_n)
    u_n.assign(u)

    t += dt
    if step % 10 == 0:
        print("Elapsed time is: " +str(t))
        outfile.write(p, u, time=t)    

# ===============================================================================


