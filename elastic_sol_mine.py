import firedrake as fire
from firedrake import Constant, TrialFunction, TestFunction
from firedrake import Function, as_vector
from firedrake import grad, inner, div, exp, dx, sqrt, dot
import finat
import numpy as np
import math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"ufl\.utils\.sorting")

# Domain and mesh
Lx, Ly = 1.5, 1.5
Nx, Ny = 100, 100
mesh = fire.PeriodicRectangleMesh(Nx, Ny, Lx, Ly)
x, y = fire.SpatialCoordinate(mesh)

# Function spaces and mass-lumped quadrature
degree = 2
V = fire.VectorFunctionSpace(mesh, "KMV", degree)
V_scalar = fire.FunctionSpace(mesh, "KMV", degree)

quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, degree, "KMV")
dxlump = dx(scheme=quad_rule)

# Time and physical parameters
T  = 1.0
dt = fire.Constant(0.001)
t  = fire.Constant(0.0)

rho   = Constant(0.00015)
lmbda = Constant(0.00033075)
mu    = Constant(3.375e-06)

alpha = float(np.sqrt((float(lmbda) + 2.0*float(mu))/float(rho)))
beta  = float(np.sqrt(float(mu)/float(rho)))
print(f"P-wave speed alpha ~ {alpha:.2f}, S-wave speed beta ~ {beta:.2f}")

A_P = Constant(1.0)
A_S = Constant(0.7)

# Unknowns, output files, and helper fields
u = TrialFunction(V)
v = TestFunction(V)
u_np1 = Function(V, name="u")
u_n = Function(V)
u_nm1 = Function(V)

outdir = os.path.join(".", "outputs", "scalar_wave_equation-out")
os.makedirs(outdir, exist_ok=True)
vtk = fire.VTKFile(os.path.join(outdir, "scalar_wave_equation.pvd"))

umag = Function(V_scalar, name="umag")
P_ind = Function(V_scalar, name="P_indicator")
S_ind = Function(V_scalar, name="S_indicator")

def RickerWavelet(t, freq, amp=1.0, integral=True):
    t_shifted = t - 1.0/freq
    if integral:
        return t_shifted*exp((-1.0) * (math.pi * freq * t_shifted) ** 2) * amp
    else:
        return amp * (1 - 2*(math.pi*freq*t_shifted)**2) * exp(-(math.pi*freq*t_shifted)**2)

#VertexOnly functions
def delta(time_expr, v, mesh, source_locations):
    """Creates a point source using VertexOnlyMesh approach."""
    vom = fire.VertexOnlyMesh(mesh, source_locations)
    if v.ufl_shape == ():
        # Scalar function space
        P0 = fire.FunctionSpace(vom, "DG", 0)
        Fvom = fire.Cofunction(P0.dual()).assign(1)
    else:
        # Vector function space
        P0_vec = fire.VectorFunctionSpace(vom, "DG", 0)
        Fvom = fire.Cofunction(P0_vec.dual())
        Fvom_x = Fvom.sub(1)
        Fvom_x.assign(1)
    return fire.interpolate(time_expr * v, Fvom)

freq = 5.0
amp = 1.0
source_location = (0.75, 0.75+0.15)

# Variational forms and solver
def eps(w):
    return 0.5*(grad(w) + grad(w).T)

F_m = (rho/(dt*dt)) * inner(u - 2*u_n + u_nm1, v) * dxlump
F_k = lmbda*div(u_n)*div(v)*dxlump + 2.0*mu*inner(eps(u_n), eps(v))*dxlump

# # Forcing term Gaussian from Jessica

# ricker = Constant(0.0)
# gauss_xy = gauss_expr(source_location, x, y)
# b_vec = gauss_forcing()
# F_s = inner(b_vec, v) * dxlump

# Forcing term vertexonlymesh
ricker = RickerWavelet(t, freq, amp=1.0)
F_s = delta(ricker, v, mesh, [source_location])

# Final FORM
# F = F_m + F_k - F_s
# lhs = fire.lhs(F)
# rhs = fire.rhs(F)
lhs = (rho/(dt*dt)) * inner(u, v) * dxlump
rhs = -(rho/(dt*dt)) * inner(- 2*u_n + u_nm1, v) * dx - lmbda*div(u_n)*div(v)*dx - 2.0*mu*inner(eps(u_n), eps(v))*dx + F_s

A = fire.assemble(lhs)
solver = fire.LinearSolver(A, solver_parameters={"ksp_type": "preonly", "pc_type": "jacobi"})

# Rreceiver
receptor_coords = (2.15, 2.0)
uz_numerical_history = []
ux_numerical_history = []
time_points = []
receiver_evaluator = fire.PointEvaluator(mesh, [receptor_coords])

# Time loop

step = 0
while float(t) < T - 1e-12:
    # ricker.assign(RickerWavelet(t, freq, amp=amp))

    # R = fire.assemble(fire.rhs(F) + F_s)
    R = fire.assemble(rhs)
    solver.solve(u_np1, R)

    t.assign(float(t) + float(dt))
    step += 1
    u_nm1.assign(u_n)
    u_n.assign(u_np1)

    uz_val = float(
        receiver_evaluator.evaluate(u_n.sub(0))[0]
    )
    ux_val = float(
        receiver_evaluator.evaluate(u_n.sub(1))[0]
    )
    uz_numerical_history.append(uz_val)
    ux_numerical_history.append(ux_val)
    time_points.append(float(t))

    if step % 100 == 0:
        umag.interpolate(sqrt(dot(u_n, u_n)))
        P_ind.interpolate(div(u_n))
        S_ind.interpolate(u_n[1].dx(0) - u_n[0].dx(1))
        vtk.write(u_n, umag, P_ind, S_ind, time=float(t))
        print(f"Elapsed time is: {float(t):.3f}")
        # Rdat = R.dat.data[:]
        # print("DEBUG")


def plot_comparison(time_points, u_numerical_history, save_path=None, show=True):
    min_len = min(len(time_points), len(u_numerical_history) )
    time = time_points[:min_len]
    u_num = np.array(u_numerical_history[:min_len])
    plt.figure(figsize=(12, 6))
    plt.plot(time, u_num, label='Numerical', linewidth=2)
    plt.title('')
    plt.xlabel('Time (s)'); plt.ylabel('Amplitude')
    plt.legend(loc='lower right'); plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()


plot_path = "debug_x0.png"
plot_comparison(time_points, uz_numerical_history, save_path=plot_path)

plot_path = "debug_x1.png"
plot_comparison(time_points, ux_numerical_history, save_path=plot_path)

print("END")
