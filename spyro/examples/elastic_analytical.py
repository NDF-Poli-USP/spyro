'''
The analytical solutions are provided by the book:
Quantitative Seismology (2nd edition) from Aki and Richards
'''
import argparse
import numpy as np
import spyro
import sys

from math import pi as PI
from scipy.integrate import quad

parser = argparse.ArgumentParser()
parser.add_argument("-L", default=450, type=float, metavar='<value>',
                    help="size of the edge of the computational domain (cube)")
parser.add_argument("-N", default=30, type=int, metavar='<value>',
                    help="number of divisions in each direction")
parser.add_argument("-alpha", default=1500, type=float, metavar='<value>',
                    help="P wave velocity")
parser.add_argument("-beta", default=1000, type=float, metavar='<value>',
                    help="S wave velocity")
parser.add_argument("-rho", default=2000, type=float, metavar='<value>',
                    help="density")
parser.add_argument("-amplitude", default=1e3, type=float, metavar='<value>',
                    help="amplitude of the wavelet")
parser.add_argument("-frequency", default=20, type=float, metavar='<value>',
                    help="frequency of the wavelet")
parser.add_argument("-total_time", default=0.3, type=float, metavar='<value>',
                    help="total simulation time")
parser.add_argument("-time_steps", default=750, type=int, metavar='<value>',
                    help="number of time steps")
parser.add_argument("-receiver_x", default=100, type=float, metavar='<value>',
                    help="receiver position in x direction")
parser.add_argument("-receiver_y", default=0, type=float, metavar='<value>',
                    help="receiver position in y direction")
parser.add_argument("-receiver_z", default=100, type=float, metavar='<value>',
                    help="receiver position in z direction")

if "pytest" in sys.argv[0]:
    args = parser.parse_args([])
else:
    args = parser.parse_args()

L = args.L  # Length (m)
N = args.N  # Number of elements in each direction
h = L/N     # Element size (m)

alpha = args.alpha  # P-wave velocity
beta = args.beta    # S-wave velocity
rho = args.rho      # Density (kg/m3)

smag = args.amplitude  # Source amplitude
f0 = args.frequency    # Frequency (Hz)
t0 = 1/f0              # Time delay

tn = args.total_time  # Simulation time (s)
nt = args.time_steps
time_step = (tn/nt)
out_freq = int(0.01/time_step)

x_s = np.r_[-L/2, L/2, L/2]   # Source location (m)
x_r = np.r_[args.receiver_x,  # Receiver relative location (m)
            args.receiver_y,
            args.receiver_z]


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


def numerical_solution(j):
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
        "h": h,
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
    wave.set_mesh(mesh_parameters={'dx': h})
    wave.forward_solve()

    return wave.receivers_output.reshape(nt + 1, 3)[0:-1, :]
