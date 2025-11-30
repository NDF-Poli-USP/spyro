import firedrake

from firedrake import *
import finat
import math
import numpy as np
from scipy.signal import convolve
import os
from firedrake import VTKFile
from scipy.signal import convolve
import matplotlib.pyplot as plt
from firedrake import PointEvaluator
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"ufl\.utils\.sorting")

mesh = RectangleMesh(100, 100, 0.0, 0.0, 4.0, 4.0)

V = FunctionSpace(mesh, "KMV", 2)

u = TrialFunction(V)
v = TestFunction(V)

u_np1 = Function(V)
u_n = Function(V)
u_nm1 = Function(V)

T = 1.0
dt = 0.001
t = 0
step = 0

root = "."
outdir = os.path.join(root, "outputs", "solution_acoustic_wave-homogeneous_medium")
os.makedirs(outdir, exist_ok=True)
vtk = VTKFile(os.path.join(outdir, "solution_acoustic_wave-homogeneous_medium.pvd"))
fig_path = os.path.join(outdir, "comparison_num_vs_analytic.png")

freq = 6
c = Constant(1.5)
#source_location = (0.75, 0.75+0.15)
source_location = (2.0, 2.0)

def RickerWavelet(t, freq, amp=1.0):
    t_shifted = t - 1.0 / freq
    factor = 1 - 2 * math.pi**2 * (freq**2) * (t_shifted**2)
    envelope = math.exp(-math.pi**2 * (freq**2) * (t_shifted**2))
    return amp * factor * envelope

def delta_expr(x0, x, y, sigma_x=2000.0):
    sigma_x = Constant(sigma_x)
    return exp(-sigma_x * ((x - x0[0]) ** 2 + (y - x0[1]) ** 2))

def delta(time_expr, v, mesh, source_locations):
    vom = VertexOnlyMesh(mesh, source_locations)
    if v.ufl_shape == ():
        P0 = FunctionSpace(vom, "DG", 0)
        Fvom = Cofunction(P0.dual()).assign(1)
    else:
        P0_vec = VectorFunctioSpace(vom, "DG", 0)
        Fvom = Cofunction(P0_vec.dual())
        Fvom_x = Fvom.sub(1)
        Fvom_x.assign(1)
    return interpolate(time_expr * v, Fvom)

quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")

dxlump = dx(scheme=quad_rule)

##m = (u - 2.0 * u_n + u_nm1) / Constant(dt * dt) * v * dxlump
##
##a = c * c * dot(grad(u_n), grad(v)) * dx

##x, y = SpatialCoordinate(mesh)
#source = Constant([2.0, 2.0])
##ricker = Constant(0.0)
##ricker.assign(RickerWavelet(t, freq))

##ricker = RickerWavelet(t, freq, amp=1.0)
##F_s = delta(ricker, v, mesh, [source_location])

##R = Cofunction(V.dual())

##F = m + a - delta_expr(source, x, y) * ricker * v * dx
##F = m + a - F_s
##a, r = lhs(F), rhs(F)

##lhs = (1 / Constant(dt * dt)) * inner(u, v) * dxlump
##rhs = -(1 / Constant(dt * dt)) * inner(- 2.0 * u_n + u_nm1, v) * dx - c * c * dot(grad(u_n), grad(v)) * dx + F_s

#=====
ricker = Constant(0.0)
R = Cofunction(V.dual())

lhs = (1 / Constant(dt * dt)) * inner(u, v) * dxlump
A = assemble(lhs)
solver = LinearSolver(A, solver_parameters={"ksp_type": "preonly", "pc_type": "jacobi"})
#=====




A = assemble(lhs)
solver = LinearSolver(A, solver_parameters={"ksp_type": "preonly", "pc_type": "jacobi"})

#P = 0.0011
P = 1.0
def analitic_solution_2D_green(mesh, t):
    x = SpatialCoordinate(mesh)
    r = sqrt((x[0] - source_location[0])**2 + (x[1] - source_location[1])**2)
    ct = c * t
    phi_analytical = conditional(lt(r, ct), (c * P / (2.0 * pi)) / sqrt(ct**2 - r**2), Constant(0.0))
    return phi_analytical

receptor_coords = (2.15, 2.0)
amp = 1.0
u_numerical_history = []
G_history = []
time_points = []
time_const = Constant(0.0)

phi_fun = Function(V)
point_eval_G = PointEvaluator(mesh, [receptor_coords])
point_eval_u = PointEvaluator(mesh, [receptor_coords])

step = 0
while t < T:
    step += 1
    ricker.assign(RickerWavelet(t, freq, amp=amp))

    F_s = delta(ricker, v, mesh, [source_location])
    rhs = (1 / Constant(dt * dt)) * inner(2.0 * u_n - u_nm1, v) * dxlump \
          - c * c * dot(grad(u_n), grad(v)) * dx \
          + F_s
    
    ##R = assemble(r, tensor=R)
    R = assemble(rhs, tensor=R)
    solver.solve(u_np1, R)
    t += dt
    time_const.assign(t)
    vals_u = point_eval_u.evaluate(u_np1)
    u_numerical_history.append(vals_u[0])
    phi_expr = analitic_solution_2D_green(mesh, time_const)
    phi_fun.interpolate(phi_expr)
    vals_G = point_eval_G.evaluate(phi_fun)
    G_history.append(vals_G[0])
    time_points.append(t)
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    if step % 10 == 0:
        print("Elapsed time is: " + str(t))
        vtk.write(u_n, time=t)

def perform_convolution(G_history, R_history, dt):
    conv_result = convolve(G_history, R_history, mode='full') * dt
    return conv_result[:len(R_history)]

def calculate_L2_error(u_numerical_history, u_analytical_convolved):
    min_len = min(len(u_numerical_history), len(u_analytical_convolved))
    u_num = np.array(u_numerical_history[:min_len])
    u_conv = np.array(u_analytical_convolved[:min_len])
    error_vector = u_num - u_conv
    L2_error = np.linalg.norm(error_vector) / np.linalg.norm(u_conv)
    return L2_error

def plot_comparison(time_points, u_numerical_history, u_analytical_convolved, receptor_coords, L2_error, save_path=None):
    min_len = min(len(time_points), len(u_numerical_history), len(u_analytical_convolved))
    time = time_points[:min_len]
    u_num = np.array(u_numerical_history[:min_len])
    u_conv = np.array(u_analytical_convolved[:min_len])
    plt.figure(figsize=(12, 6))
    plt.plot(time, u_num, 'b-', label='Numerical Solution (Firedrake)', linewidth=2)
    plt.plot(time, u_conv, 'r--', label='Analytical Solution (Convolved)', linewidth=2.5, alpha=0.7)
    plt.title('Comparison of Numerical vs. Analytical Solution', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.text(0.05, 0.9, f'Relative L2 Error: {L2_error:.4e}', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

R_history = np.array([RickerWavelet(t_val, freq, amp) for t_val in time_points])
u_analytical_convolved = perform_convolution(G_history, R_history, dt)

#====

min_len = min(len(u_numerical_history), len(u_analytical_convolved))
u_num = np.array(u_numerical_history[:min_len])
u_conv = np.array(u_analytical_convolved[:min_len])


alpha = np.dot(u_num, u_conv) / np.dot(u_conv, u_conv)
print("Best-fit alpha =", alpha)


u_analytical_convolved_scaled = alpha * u_analytical_convolved
#====


L2_error = calculate_L2_error(u_numerical_history, u_analytical_convolved_scaled)
plot_comparison(time_points, u_numerical_history, u_analytical_convolved_scaled, receptor_coords, L2_error, save_path=fig_path)
