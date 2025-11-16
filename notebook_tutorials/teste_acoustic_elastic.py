# imports:

from firedrake import *
import math
import numpy as np
import finat

# ===============================================================================
# parameters:

# time
T = 1
dt = 0.001
t = 0
step = 0

Lx = 4.0
Lz = 4.0

# fluid
rho_f = 1.0
K = 1.0

# solid
rho_s = 2.0
lam = 1e7
mu = 1e7

# mesh
mesh = Mesh("acoustic_elastic.msh")

fluid_id = 1
solid_id = 2
interface_id = 3

coupling = True

# ===============================================================================
# function spaces:

V_F = FunctionSpace(mesh, "CG", 2)
V_S = VectorFunctionSpace(mesh, "CG", 2)
mixed_V = V_F * V_S

# ===============================================================================
# functions:

# fluid
p = Function(V_F, name="p")
trial_F = TrialFunction(V_F)
q_F = TestFunction(V_F)

# solid
u = Function(V_S, name="u")
trial_S = TrialFunction(V_S)
v_S = TestFunction(V_S)

# mixed domain
trial_f, trial_s = TrialFunctions(mixed_V)
q_f, v_s = TestFunctions(mixed_V)
temp_f = Function(V_F)
temp_s = Function(V_S)
result_mixed = Function(mixed_V)

# ===============================================================================
# auxiliary indicator functions (0 in one domain and 1 in the other):
V_DG0_F = FunctionSpace(mesh, "DG", 0)
V_DG0_S = FunctionSpace(mesh, "DG", 0)

# fluid
I_F = Function(V_DG0_F)
par_loop(("{[i] : 0 <= i < f.dofs}", "f[i, 0] = 1.0"),
         dx(fluid_id),
         {"f": (I_F, WRITE)})
I_cg_F = Function(V_F)
par_loop(("{[i] : 0 <= i < A.dofs}", "A[i, 0] = fmax(A[i, 0], B[0, 0])"),
         dx,
         {"A": (I_cg_F, RW), "B": (I_F, READ)})

# solid
I_S = Function(V_DG0_S)
par_loop(("{[i] : 0 <= i < f.dofs}", "f[i, 0] = 1.0"),
         dx(solid_id),
         {"f": (I_S, WRITE)})
I_cg_S = Function(V_S)
par_loop(("{[i, j] : 0 <= i < A.dofs and 0 <= j < 2}", "A[i, j] = fmax(A[i, j], B[0, 0])"),
         dx,
         {"A": (I_cg_S, RW), "B": (I_S, READ)})

# ===============================================================================
# normal unit vector:

n_vec = FacetNormal(mesh)
n_int = I_S("+") * n_vec("+") + I_S("-") * n_vec("-")

# ===============================================================================
# source:

freq = 6
c = Constant(1.5)

def RickerWavelet(t, freq, amp=1.0):
    t_shifted = t - 1.0 / freq
    factor = 1 - 2 * math.pi**2 * (freq**2) * (t_shifted**2)
    envelope = math.exp(-math.pi**2 * (freq**2) * (t_shifted**2))
    return amp * factor * envelope

def delta_expr(x0, x, y, sigma_x=2000.0):
    sigma_x = Constant(sigma_x)
    return exp(-sigma_x * ((x - x0[0])**2 + (y - x0[1])**2))

# ===============================================================================
# special boundary conditions (solvers only to the appropriate subdomains):


# ===============================================================================
# source injection:
x, y = SpatialCoordinate(mesh)
source = Constant([2.0, 2.5])
ricker = Constant(0.0)
ricker.assign(RickerWavelet(t, freq))

# ===============================================================================
# cofunctions:
R_f = Cofunction(V_F.dual())
R_s = Cofunction(V_S.dual())

# ===============================================================================
# solvers of equations:

p_np1 = Function(V_F)
p_n = Function(V_F)
p_nm1 = Function(V_F)
ddot_p = (p - 2.0*p_n + p_nm1) / Constant(dt * dt)

u_np1 = Function(V_S)
u_n = Function(V_S)
u_nm1 = Function(V_S)
ddot_u = (u - 2.0*u_n + u_nm1) / Constant(dt * dt)

f = delta_exp(source, x, y)*ricker

def eps(x):
    return sym(grad(x))

def sigma(x):
    return lam * div(x) * Identity(mesh.geometric_dimension()) + 2 * mu * eps(x)

# fluid
F_p = (1/K) * ddot_p * q_F * dx(fluid_id)\
      + q_F * dot(ddot_u, n_int) * dS(interface_id) \
      + (1/rho_f) * dot(grad(q_F), grad(p_n)) * dx(fluid_id) \
      - (f/K) * q_F * dx(fluid_id)
a_p, r_p = lhs(F_p), rhs(F_p)
A_p = assemble(a_p)
solver = LinearSolver(A_p)

# solid
F_u = rho_s * dot(v_S, ddot_u) * dx(solid_id) \
      + p_n * dot(v_S, n_int) * dS(interface_id) \
      + inner(eps(v_S), sigma(u_n)) * dx(solid_id)
a_u, r_u = lhs(F_u), rhs(F_u)
A_u = assemble(a_u)
solver = LinearSolver(A_u)

# ===============================================================================
# store data:
V_out = V_F * V_S
out = Functioo(V_out, name="acoustic_elastic")
p_out, u_out = out.split()
outfile = VTKFile("results_acoustic_elastic/acoustic_elastic.pvd")

# ===============================================================================
# computation loop:
step = 0
while t < T:
    step += 1
    ricker.assign(RickerWavelet(t, freq))
    
    R_f = assemble(r_p, tensor=R_f)
    R_s = assemble(r_u, tensor=R_s)
    
    solver.solve(p_np1, R_f)
    solver.solve(u_np1, R_s)

    p_nm1.assign(p_n)
    p_n.assign(p_np1)
    u_nm1.assign(u_n)
    u_n.assign(u_np1)

    t += dt
    if step % 10 == 0:
        print("Elapsed time is: " +str(t))
        p_out.assign(p_n)
        u_out.assign(u_n)
        outfile.write(out, time=t)    

# ===============================================================================


