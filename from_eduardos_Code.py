'''
The analytical solutions are provided by the book:
Quantitative Seismology (2nd edition) from Aki and Richards
'''
import numpy as np
import os
import spyro
import time
import matplotlib.pyplot as plt

from firedrake.petsc import PETSc
from math import pi as PI
from mpi4py import MPI
from numpy.linalg import norm
from scipy.integrate import quad


opts = PETSc.Options()

moment_tensor = opts.getBool("moment_tensor", False)

L = opts.getReal("L", default=450)  # Length (m)
N = opts.getInt("N", default=30)    # Number of elements in each direction
h = L/N                             # Element size (m)

alpha = opts.getReal("alpha", default=1500)  # P-wave velocity
beta = opts.getReal("beta", default=1000)    # S-wave velocity
rho = opts.getReal("rho", default=2000)      # Density (kg/m3)

smag = opts.getReal("amplitude", default=1e3)  # Source amplitude
f0 = opts.getReal("frequency", default=20)     # Frequency (Hz)
t0 = 1/f0                                      # Time delay

tn = opts.getReal("total_time", default=0.3)  # Simulation time (s)
nt = opts.getInt("time_steps", default=750)   # Number of time steps
time_step = (tn/nt)
out_freq = int(0.01/time_step)

x_s = np.r_[-L/2, L/2, L/2]   # Source location (m)
x_r = np.r_[opts.getReal("receiver_x", default=100),  # Receiver relative location (m)
            opts.getReal("receiver_y", default=0),
            opts.getReal("receiver_z", default=100)]


def analytical_solution(i, j):
    t = np.linspace(0, tn, nt)

    u_near = np.zeros(nt)  # near field contribution
    P_far = np.zeros(nt)   # P-wave far-field
    S_far = np.zeros(nt)   # S-wave far field

    r = np.linalg.norm(x_r)
    gamma_i = x_r[i]/r
    gamma_j = x_r[j]/r
    delta_ij = 1 if i == j else 0

    def X0(t):
        a = PI*f0*(t - t0)
        return (1 - 2*a**2)*np.exp(-a**2)

    for k in range(nt):
        res = quad(lambda tau: tau*X0(t[k] - tau), r/alpha, r/beta)
        u_near[k] = smag*(1./(4*PI*rho))*(3*gamma_i*gamma_j - delta_ij)*(1./r**3)*res[0]
        P_far[k] = smag*(1./(4*PI*rho*alpha**2))*gamma_i*gamma_j*(1./r)*X0(t[k] - r/alpha)
        S_far[k] = smag*(1./(4*PI*rho*beta**2))*(gamma_i*gamma_j - delta_ij)*(1./r)*X0(t[k] - r/beta)

    return u_near + P_far - S_far


def explosive_source_analytical(i):
    t = np.linspace(0, tn, nt)

    P_mid = np.zeros(nt)  # P wave intermediate field
    P_far = np.zeros(nt)  # P wave far field

    r = np.linalg.norm(x_r)
    gamma_i = x_r[i]/r

    def w(t):
        a = PI*f0*(t - t0)
        return (t - t0)*np.exp(-a**2)

    def w_dot(t):
        a = PI*f0*(t - t0)
        return (1 - 2*a**2)*np.exp(-a**2)

    for k in range(nt):
        P_mid[k] = smag*(gamma_i/(4*PI*rho*alpha**2))*(1./r**2)*w(t[k] - r/alpha)
        P_far[k] = smag*(gamma_i/(4*PI*rho*alpha**3))*(1./r)*w_dot(t[k] - r/alpha)

    return P_mid + P_far


def numerical_solution(j):
    if moment_tensor:
        A0 = smag*np.eye(3)
    else:
        A0 = np.zeros(3)
        A0[j] = smag

    d = {}

    d["options"] = {
        "cell_type": "T",
        "variant": "lumped",
        "degree": 3,
        "dimension": 3,
    }

    d["parallelism"] = {
        "type": "automatic",
    }

    d["mesh"] = {
        "Lz": L,
        "Lx": L,
        "Ly": L,
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    }

    d["acquisition"] = {
        "source_type": "ricker",
        "source_locations": [x_s.tolist()],
        "frequency": f0,
        "delay": t0,
        "delay_type": "time",
        "amplitude": A0,
        "receiver_locations": [(x_s + x_r).tolist()],
    }

    d["synthetic_data"] = {
        "type": "object",
        "density": rho,
        "p_wave_velocity": alpha,
        "s_wave_velocity": beta,
        "real_velocity_file": None,
    }

    d["time_axis"] = {
        "initial_time": 0,
        "final_time": tn,
        "dt": time_step,
        "output_frequency": out_freq,
        "gradient_sampling_frequency": 1,
    }

    d["visualization"] = {
        "forward_output": True,
        "forward_output_filename": "results/elastic_analytical.pvd",
        "fwi_velocity_model_output": False,
        "gradient_output": False,
        "adjoint_output": False,
        "debug_output": False,
    }

    d["absorving_boundary_conditions"] = {
        "status": True,
        "damping_type": "local",
    }

    wave = spyro.IsotropicWave(d)
    wave.set_mesh(mesh_parameters={'edge_length': h})
    wave.forward_solve()

    return wave.receivers_output.reshape(nt + 1, 3)[0:-1, :]


def err(u_a, u_n):
    return norm(u_a - u_n)/norm(u_a)


if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        start = time.time()

    j = 0
    U_x = analytical_solution(0, j)
    U_y = analytical_solution(1, j)
    U_z = analytical_solution(2, j)

    # u_n = numerical_solution(j)
    # u_x = u_n[:, 0]
    # u_y = u_n[:, 1]
    # u_z = u_n[:, 2]

    if rank == 0:
        end = time.time()
        print(f"Elapsed time: {end - start:.2f} s")
        # print(f"err_x = {err(U_x, u_x)}")
        # print(f"err_y = {err(U_y, u_y)}")
        # print(f"err_z = {err(U_z, u_z)}")

        # Create time array for plotting
        t = np.linspace(0, tn, nt)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot all three components
        plt.subplot(3, 1, 1)
        plt.plot(t, U_x, 'b-', linewidth=2, label='Ux (displacement in x)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Displacement Component Ux')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(t, U_y, 'r-', linewidth=2, label='Uy (displacement in y)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Displacement Component Uy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(t, U_z, 'g-', linewidth=2, label='Uz (displacement in z)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Displacement Component Uz')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Also create a combined plot
        plt.figure(figsize=(12, 6))
        plt.plot(t, U_x, 'b-', linewidth=2, label='Ux')
        plt.plot(t, U_y, 'r-', linewidth=2, label='Uy')
        plt.plot(t, U_z, 'g-', linewidth=2, label='Uz')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('All Displacement Components')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if moment_tensor:
            basename = "ExplosiveSource"
        else:
            basename = "ForceSource"
        
        # Save plots in current directory
        plt.figure(1)  # Select the first figure (subplots)
        plt.savefig(f"{basename}_displacement_components_separated.png", dpi=300, bbox_inches='tight')
        
        plt.figure(2)  # Select the second figure (combined)
        plt.savefig(f"{basename}_displacement_components_combined.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
        np.save(f"{basename}_x_analytical.npy", U_x)
        np.save(f"{basename}_y_analytical.npy", U_y)
        np.save(f"{basename}_z_analytical.npy", U_z)
        # np.save(f"{basename}_numerical.npy", u_n)

        print(f"Data and plots saved to current directory with prefix: {basename}_")