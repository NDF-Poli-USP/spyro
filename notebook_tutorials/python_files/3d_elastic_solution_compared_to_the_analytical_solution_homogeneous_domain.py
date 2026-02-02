import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"ufl\.utils\.sorting")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"firedrake\.interpolation")

from firedrake import *
import finat
import numpy as np
import math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Configuração da Malha 3D
Lx, Ly, Lz = 4.0, 4.0, 4.0
Nx, Ny, Nz = 50, 50, 50 
mesh = BoxMesh(Nx, Ny, Nz, Lx, Ly, Lz)
x, y, z = SpatialCoordinate(mesh)

V = VectorFunctionSpace(mesh, "KMV", 2)
V_scalar = FunctionSpace(mesh, "KMV", 2)

deg = V.ufl_element().sub_elements[0].degree()
quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, deg, "KMV")
dxlump = dx(scheme=quad_rule)

T  = 1.0
dt = 0.001
t  = 0.0

rho   = Constant(1.0)
lmbda = Constant(1.0)
mu    = Constant(0.25)

alpha = float(np.sqrt((float(lmbda) + 2.0*float(mu))/float(rho)))
beta  = float(np.sqrt(float(mu)/float(rho)))

A_P = Constant(1.0)
A_S = Constant(0.7)

u     = TrialFunction(V)
v     = TestFunction(V)
u_np1 = Function(V, name="u")
u_n   = Function(V)
u_nm1 = Function(V)

outdir = os.path.join(".", "outputs", "elastic_3d_out")
os.makedirs(outdir, exist_ok=True)
vtk = VTKFile(os.path.join(outdir, "elastic_solution.pvd"))

umag  = Function(V_scalar, name="umag")
P_ind = Function(V_scalar, name="P_indicator")
S_ind = Function(V_scalar, name="S_indicator")

def RickerWavelet(t, freq, amp=1.0):
    t0 = 1.0/freq
    a = math.pi * freq * (t - t0)
    return amp * (1 - 2*a**2) * math.exp(-a**2)

def delta_expr(x0, mesh, V, sigma_x=2000.0):
    X = SpatialCoordinate(mesh)
    expr = exp(-Constant(sigma_x) * ((X[0]-x0[0])**2 + (X[1]-x0[1])**2 + (X[2]-x0[2])**2))
    F = Function(V)
    F.interpolate(expr)
    mass = assemble(F * dx)
    F.assign(F / mass)
    return F

freq   = 6.0
amp    = 5.0
source_coords = np.array([2.0, 2.0, 2.0])
source = Constant(source_coords)
ricker = Constant(0.0)

# Usaremos SOURCE_MODE "x" para validar com a solução de Stokes (força pontual em i)
SOURCE_MODE = "x"
delta_xyz = delta_expr(source, mesh, V_scalar)

def forcing_vec():
    return as_vector((delta_xyz * ricker, 0.0, 0.0))

def eps(w):
    return 0.5*(grad(w) + grad(w).T)

F_m = (rho/Constant(dt*dt)) * inner(u - 2*u_n + u_nm1, v) * dxlump
F_k = lmbda*div(u_n)*div(v)*dx + 2.0*mu*inner(eps(u_n), eps(v))*dx
b_vec = forcing_vec()
F_s = inner(b_vec, v) * dx

F = F_m + F_k - F_s
A = assemble(lhs(F))
solver = LinearSolver(A, solver_parameters={"ksp_type": "preonly", "pc_type": "jacobi"})

# --- Configuração da Solução Analítica ---
receptor_coords = np.array([2.15, 2.0, 2.0]) # Receptor deslocado no eixo X
x_r = receptor_coords - source_coords
r_dist = np.linalg.norm(x_r)

u_numerical_history = []
time_points = []

# Loop de tempo
step = 0
while t < T - 1e-12:
    ricker.assign(RickerWavelet(t, freq, amp=amp))
    
    R = assemble(rhs(F))
    solver.solve(u_np1, R)

    t += dt
    step += 1
    u_nm1.assign(u_n)
    u_n.assign(u_np1)

    # Pegando componente X da solução numérica
    u_numerical_history.append(float(u_n.at(receptor_coords)[0]))
    time_points.append(t)

    if step % 20 == 0:
        umag.interpolate(sqrt(dot(u_n, u_n)))
        P_ind.interpolate(div(u_n))
        S_ind.interpolate(sqrt(dot(curl(u_n), curl(u_n))))
        vtk.write(u_n, umag, P_ind, S_ind, time=t)
        print(f"t = {t:.3f}")

# --- Implementação da sua função analytical_solution ---
def get_stokes_analytical(time_list, x_r, i, j, f0, amp, rho, alpha, beta):
    nt = len(time_list)
    u_total = np.zeros(nt)
    r = np.linalg.norm(x_r)
    gamma_i = x_r[i]/r
    gamma_j = x_r[j]/r
    delta_ij = 1 if i == j else 0
    t0 = 1.0/f0

    def X0(t_val):
        if t_val < 0: return 0
        a = np.pi * f0 * (t_val - t0)
        return amp * (1 - 2*a**2) * np.exp(-a**2)

    for k, tk in enumerate(time_list):
        # Termo Near-field (Integral)
        res, _ = quad(lambda tau: tau * X0(tk - tau), r/alpha, r/beta)
        u_near = (1./(4*np.pi*rho)) * (3*gamma_i*gamma_j - delta_ij) * (1./r**3) * res
        
        # Termos Far-field
        P_far = (1./(4*np.pi*rho*alpha**2)) * (gamma_i*gamma_j) * (1./r) * X0(tk - r/alpha)
        S_far = (1./(4*np.pi*rho*beta**2)) * (gamma_i*gamma_j - delta_ij) * (1./r) * X0(tk - r/beta)
        
        u_total[k] = u_near + P_far - S_far
    return u_total

#=================

# ... (após o término do loop while t < T)

# 1. Calcular a solução analítica de Stokes (que já retorna o sinal u_j devido à força em i)
# i=0 (Força em X), j=0 (Medição em X)
u_analytical = get_stokes_analytical(time_points, x_r, 0, 0, freq, amp, float(rho), alpha, beta)

# 2. Preparar os dados para comparação (seu bloco solicitado)
min_len = min(len(u_numerical_history), len(u_analytical))

u_num = np.asarray(u_numerical_history[:min_len])
u_conv = np.asarray(u_analytical[:min_len]) # Aqui u_conv é a analítica de Stokes

# Cálculo do fator de escala (Best-fit L2)
# Isso ajusta a amplitude da analítica para casar com a numérica
scaling_factor = np.dot(u_num, u_conv) / (np.dot(u_conv, u_conv) + 1e-15)
print(f"Best-fit alpha (Scaling factor) = {scaling_factor:.4f}")

u_analytical_scaled = scaling_factor * u_conv

# 3. Cálculo do erro L2 relativo
error_l2 = np.linalg.norm(u_num - u_analytical_scaled) / (np.linalg.norm(u_num) + 1e-15)
print(f"Relative L2 Error: {error_l2:.4%}")

# 4. Plotagem atualizada
plt.figure(figsize=(12, 6))
plt.plot(time_points[:min_len], u_num, label='Numerical (Firedrake 3D)', linewidth=2)
plt.plot(time_points[:min_len], u_analytical_scaled, '--', label='Analytical', linewidth=1.5)
plt.title(f'Comparison at Receptor {receptor_coords}\nL2 Relative Error: {error_l2:.2%}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (u_x)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plot_path = os.path.join(outdir, "comparison_elastic_3d.png")
plt.savefig(plot_path, dpi=200)
print(f"Gráfico salvo em: {plot_path}")
#=================

##
##
### Calculando a analítica (i=0 para força em X, j=0 para componente u_x)
##u_analytical = get_stokes_analytical(time_points, x_r, 0, 0, freq, amp, float(rho), alpha, beta)
##
### Plotagem final
##plt.figure(figsize=(10, 5))
##plt.plot(time_points, u_numerical_history, label="Numérica (Firedrake)")
##plt.plot(time_points, u_analytical, '--', label="Analítica (Stokes)")
##plt.legend()
##plt.savefig(os.path.join(outdir, "validacao.png"))
