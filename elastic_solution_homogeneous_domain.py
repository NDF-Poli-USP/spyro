from firedrake import *
import finat
import numpy as np
import math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Domain and mesh
Lx, Ly = 2.0, 2.0
Nx, Ny = 50, 50
mesh = RectangleMesh(Nx, Ny, Lx, Ly)
x, y = SpatialCoordinate(mesh)

# Function spaces and mass-lumped quadrature
degree = 2
V = VectorFunctionSpace(mesh, "KMV", degree)
V_scalar = FunctionSpace(mesh, "KMV", degree)

quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, degree, "KMV")
dxlump = dx(scheme=quad_rule)

# Time and physical parameters
T  = 1.0
dt = 0.001
t  = 0.0

rho   = Constant(1.0)
lmbda = Constant(1.3)
mu    = Constant(0.4)

alpha = float(np.sqrt((float(lmbda) + 2.0*float(mu))/float(rho)))
beta  = float(np.sqrt(float(mu)/float(rho)))
print(f"P-wave speed alpha ~ {alpha:.2f}, S-wave speed beta ~ {beta:.2f}")

A_P = Constant(1.0)
A_S = Constant(0.7)

# Unknowns, output files, and helper fields
u     = TrialFunction(V)
v     = TestFunction(V)
u_np1 = Function(V, name="u")
u_n   = Function(V)
u_nm1 = Function(V)

outdir = os.path.join(".", "outputs", "scalar_wave_equation-out")
os.makedirs(outdir, exist_ok=True)
vtk = VTKFile(os.path.join(outdir, "scalar_wave_equation.pvd"))

umag  = Function(V_scalar, name="umag")
P_ind = Function(V_scalar, name="P_indicator")
S_ind = Function(V_scalar, name="S_indicator")

# Source definition
def RickerWavelet(t, freq, amp=1.0):
    t_shifted = t - 1.0/freq
    return amp * (1 - 2*(pi*freq*t_shifted)**2) * exp(-(pi*freq*t_shifted)**2)

def delta(v, mesh, source_locations):
    vom = VertexOnlyMesh(mesh, source_locations)
    P0_vec = VectorFunctionSpace(vom, "DG", 0)
    Fvom = Cofunction(P0_vec.dual()).assign(1)
    return interpolate(v, Fvom)

freq   = 5.0
source_locations = [(1.0, 1.0)]

# Variational forms and solver
def eps(w):
    return 0.5*(grad(w) + grad(w).T)

F_m = (rho/Constant(dt*dt)) * inner(u - 2*u_n + u_nm1, v) * dxlump
F_k = lmbda*div(u_n)*div(v)*dx + 2.0*mu*inner(eps(u_n), eps(v))*dx

F_s = delta(v*RickerWavelet(t, freq), mesh, source_locations)

F = F_m + F_k - F_s

A = assemble(lhs(F))
solver = LinearSolver(A, solver_parameters={"ksp_type": "preonly", "pc_type": "jacobi"})

# Analytical and receiver
c = Constant(1.5)
Pp = 0.0011
source = source_location

def analitic_solution_2D_green(t):
    r = sqrt((x - source[0])**2 + (y - source[1])**2)
    eps0 = 1e-30
    tP = alpha * t
    phiP = conditional(tP > r,
                       (alpha * A_P / (2.0 * pi)) / sqrt(tP**2 - r**2 + eps0),
                       0.0)
    tS = beta * t
    phiS = conditional(tS > r,
                       (beta * A_S / (2.0 * pi)) / sqrt(tS**2 - r**2 + eps0),
                       0.0)
    return phiP + phiS

receptor_coords = (2.15, 2.0)
u_numerical_history = []
G_history = []
time_points = []

# Time loop
step = 0
while t < T - 1e-12:
    ricker.assign(RickerWavelet(t, freq, amp=amp))

    R = assemble(rhs(F))
    solver.solve(u_np1, R)

    t += dt
    step += 1
    u_nm1.assign(u_n)
    u_n.assign(u_np1)

    ux_val = float(u_n.at(receptor_coords)[0])
    u_numerical_history.append(ux_val)
    time_points.append(t)

    phi_fun = Function(V_scalar)
    phi_fun.interpolate(analitic_solution_2D_green(t))
    G_history.append(float(phi_fun.at(receptor_coords)))

    if step % 10 == 0:
        umag.interpolate(sqrt(dot(u_n, u_n)))
        P_ind.interpolate(div(u_n))
        S_ind.interpolate(u_n[1].dx(0) - u_n[0].dx(1))
        vtk.write(u_n, umag, P_ind, S_ind, time=t)
        print(f"Elapsed time is: {t:.3f}")

# Postprocessing functions
def perform_convolution(G_history, R_history, dt):
    from scipy.signal import convolve
    conv_result = convolve(G_history, R_history, mode='full') * dt
    return conv_result[:len(R_history)]

def calculate_L2_error(u_numerical_history, u_analytical_convolved):
    min_len = min(len(u_numerical_history), len(u_analytical_convolved))
    u_num = np.array(u_numerical_history[:min_len])
    u_conv = np.array(u_analytical_convolved[:min_len])
    error_vector = u_num - u_conv
    return np.linalg.norm(error_vector) / (np.linalg.norm(u_conv) + 1e-15)

def plot_comparison(time_points, u_numerical_history, u_analytical_convolved, L2_error, save_path=None):
    min_len = min(len(time_points), len(u_numerical_history), len(u_analytical_convolved))
    time = time_points[:min_len]
    u_num = np.array(u_numerical_history[:min_len])
    u_conv = np.array(u_analytical_convolved[:min_len])
    plt.figure(figsize=(12, 6))
    plt.plot(time, u_num,  label='Numerical', linewidth=2)
    plt.title('')
    plt.xlabel('Time (s)'); plt.ylabel('Amplitude')
    plt.legend(loc='lower right'); plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.close()

# Postprocessing
R_history = np.array([RickerWavelet(ti, freq, amp=amp) for ti in time_points])
u_analytical_convolved = perform_convolution(G_history, R_history, dt)
L2_error = calculate_L2_error(u_numerical_history, u_analytical_convolved)
plot_path = os.path.join(outdir, "comparison_plot.png")
plot_comparison(time_points, u_numerical_history, u_analytical_convolved, L2_error, save_path=plot_path)

print(f"Simulation completed. L2 error: {L2_error:.6f}")