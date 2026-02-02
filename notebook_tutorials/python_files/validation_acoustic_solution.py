# ====================================================================================

# Validation of the numerical results of the acoustic wave solution with the
# analytical solution and Gar6more2D

# ====================================================================================

# Imports:
from firedrake import *
import finat
import math
import numpy as np
from scipy.signal import convolve
import os
import matplotlib.pyplot as plt

# ====================================================================================

# Parameters:

# Mesh:
mesh = RectangleMesh(150, 150, 0.0, 0.0, 4.0, 4.0)

# Time:
time = 0
final_time = 1.0
dt = 0.001
step = 0

# ====================================================================================

# Function space:
V = FunctionSpace(mesh, "KMV", 2)

u = TrialFunction(V)
v = TestFunction(V)

u_np1 = Function(V)  # timestep n+1
u_n = Function(V)    # timestep n
u_nm1 = Function(V)  # timestep n-1

# ====================================================================================

# Output directory setup:
root = "."
outdir = os.path.join(root, "outputs", "solution_acoustic_wave")
os.makedirs(outdir, exist_ok=True)
vtk = VTKFile(os.path.join(outdir, "solution_acoustic_wave.pvd"))

# ====================================================================================

# Source:
frequency = 6
c = Constant(1.5)

def ricker_wavelet(time, frequency, amplitude=1.0):
    shifted_time = time - 1.0 / frequency
    factor = 1 - 2 * math.pi**2 * (frequency**2) * (shifted_time**2)
    envelope = math.exp(-math.pi**2 * (frequency**2) * (shifted_time**2))
    return amplitude * factor * envelope

def dirac_delta_approximated_by_gaussian(x0, mesh, V, sigma_x=2000.0):
    x = SpatialCoordinate(mesh)
    expr = exp(-Constant(sigma_x) *
               ((x[0] - x0[0])**2 + (x[1] - x0[1])**2))
    F = Function(V)
    F.interpolate(expr)
    mass = assemble(F * dx)
    F.assign(F / mass)
    final_mass = assemble(F * dx)
    assert abs(final_mass - 1.0) < 1e-10, \
        f"Normalization failed: {final_mass}"
    return F

# ====================================================================================

# Numerical integration:
quadrature_rule = finat.quadrature.make_quadrature(V.finat_element.cell,
                                                   V.ufl_element().degree(), "KMV")
dxlump=dx(scheme=quadrature_rule)

m = (u - 2.0 * u_n + u_nm1) / Constant(dt * dt) * v * dxlump

a = c*c*dot(grad(u_n), grad(v)) * dx

x, y = SpatialCoordinate(mesh)
source = Constant([2.0, 2.0])
ricker = Constant(0.0)
ricker.assign(ricker_wavelet(time, frequency))

R = Cofunction(V.dual())

delta_spatial = dirac_delta_approximated_by_gaussian(source, mesh, V)

F = m + a - ricker * delta_spatial * v * dx

a, r = lhs(F), rhs(F)
A = assemble(a)
solver = LinearSolver(A, solver_parameters={"ksp_type": "preonly",
                                            "pc_type": "jacobi"})

# ====================================================================================

# Analytical solution:
P = 1.0
def analytical_solution_2d(mesh, time):
    x = SpatialCoordinate(mesh)
    r = sqrt((x[0] - source[0])**2 + (x[1] - source[1])**2)
    ct = c * time
    phi_analytical = conditional(lt(r, ct),
                                 (c * P / (2.0 * pi)) / sqrt(ct**2 - r**2),
                                 Constant(0.0))
    return phi_analytical

# ====================================================================================

# Time-stepping loop:
receptor_coords = (2.15, 2.0)
amplitude = 1.0
u_numerical_history = []
G_history = []
time_points = []
time_const = Constant(0.0)

evaluator = PointEvaluator(mesh, [receptor_coords])
while time < final_time:
    step += 1
    ricker.assign(ricker_wavelet(time, frequency, amplitude=amplitude))
    R = assemble(r, tensor=R)
    solver.solve(u_np1, R)
    time += dt
    time_const.assign(time)
    u_numerical_history.append(evaluator.evaluate(u_np1)[0])
    phi_expr = analytical_solution_2d(mesh, time_const)
    phi_func = assemble(interpolate(phi_expr, V))
    G_history.append(evaluator.evaluate(phi_func)[0])
    time_points.append(time)
    u_nm1.assign(u_n)
    u_n.assign(u_np1)
    if step % 10 == 0:
        print("Elapsed time is: "+str(time))
        vtk.write(u_n, time=time)

# ====================================================================================

# Convolution of the analytical solution:
def perform_convolution(G_history, R_history, dt):
    conv_result = convolve(G_history, R_history, mode='full') * dt
    return conv_result[:len(R_history)]

R_history = np.array([ricker_wavelet(t_val, frequency, amplitude) for t_val in time_points])
u_analytical_convolved = perform_convolution(G_history, R_history, dt)

# ====================================================================================

# Numerical results scaling:
min_len = min(len(u_numerical_history), len(u_analytical_convolved))

u_num = np.asarray(u_numerical_history[:min_len])
u_conv = np.asarray(u_analytical_convolved[:min_len])

alpha = np.dot(u_conv, u_num) / np.dot(u_num, u_num)
u_numerical_convolved_scaled = alpha * u_num

# ====================================================================================

# Gar6more2D:
try:
    gar6_path = "P.dat" 
    gar6_data = np.loadtxt(gar6_path)
    
    t_gar6 = gar6_data[:, 0]
    u_gar6 = gar6_data[:, 1]
    
    u_gar6_sync = np.interp(time_points, t_gar6, u_gar6)

    # Scaling:
    alpha_gar6 = np.dot(u_conv, u_gar6_sync) / np.dot(u_gar6_sync, u_gar6_sync)
    u_gar6_final = alpha_gar6 * u_gar6_sync
    
except Exception as e:
    print(f"Error loading Gar6: {e}")
    u_gar6_final = None

# ====================================================================================

# Error L2:
def calculate_L2_error(u_numerical_history, u_analytical_convolved):
    min_len = min(len(u_numerical_history), len(u_analytical_convolved))
    u_num = np.array(u_numerical_history[:min_len])
    u_conv = np.array(u_analytical_convolved[:min_len])
    error_vector = u_num - u_conv
    L2_error = np.linalg.norm(error_vector) / np.linalg.norm(u_conv)
    return L2_error

L2_error = calculate_L2_error(u_numerical_convolved_scaled, u_analytical_convolved)
    
# ====================================================================================

# Plotting the results:
def plot_comparison(time_points, u_numerical_history, u_analytical_convolved,
                    u_gar6, receptor_coords, L2_error):
    min_len = min(len(time_points), len(u_numerical_history), len(u_analytical_convolved))
    time = time_points[:min_len]
    u_num = np.array(u_numerical_history[:min_len])
    u_conv = np.array(u_analytical_convolved[:min_len])
    plt.figure(figsize=(14, 7))
    plt.plot(time, u_num, 'b-', label='Numerical', linewidth=2.5)
    plt.plot(time, u_conv, 'r--', label='Analytical', linewidth=2.0, alpha=0.8)
    if u_gar6 is not None:
        u_gar6_plot = u_gar6[:min_len]
        plt.plot(time, u_gar6_plot, 'g:', label='Gar6more2D', linewidth=2.5)
    plt.title(f'Validation of the numerical solution - Acoustic Wave', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "validation_acoustic_solution.png"), dpi=300)
    plt.show()
    plt.close()

plot_comparison(time_points,
                u_numerical_convolved_scaled,
                u_analytical_convolved,
                u_gar6_final,
                receptor_coords,
                L2_error)

# =======================================END==========================================
