import numpy as np
import os
import spyro
import time
import gmsh

from firedrake.petsc import PETSc
from math import pi as PI
from mpi4py import MPI
from numpy.linalg import norm
from scipy.integrate import quad

from spyro.utils.file_utils import mkdir_timestamp

opts = PETSc.Options()

moment_tensor = opts.getBool("moment_tensor", False)
length = 1500
h = 25
L = opts.getReal("L", default=length)  # Length (m)
h = opts.getReal("h", default=h)    # Number of elements in each direction

alpha = opts.getReal("alpha", default=1500)  # P-wave velocity
beta = opts.getReal("beta", default=150)    # S-wave velocity
rho = opts.getReal("rho", default=150)      # Density (kg/m3)

smag = opts.getReal("amplitude", default=1)  # Source amplitude
f0 = opts.getReal("frequency", default=5)     # Frequency (Hz)
t0 = 1/f0                                      # Time delay

final_time = 3.0
dt = 0.0001
tn = opts.getReal("total_time", default=final_time)  # Simulation time (s)
nt = opts.getInt("time_steps", default=final_time/dt+1)   # Number of time steps
time_step = (tn/(nt - 1))
out_freq = int(100)

x_s = np.r_[-L/2, L/2]   # Source location (m)
x_r = np.r_[opts.getReal("receiver_x", default=0.0),  # Receiver relative location (m)
            opts.getReal("receiver_z", default=150.0)]



def generate_gmsh_mesh(Lx, Ly, edge_size, radius=200,
                       center=None,
                       output_file="mesh.msh"):

    gmsh.initialize()
    gmsh.model.add("refined_mesh")

    # ----------------------------
    # 1. Center of refinement
    # ----------------------------
    if center is None:
        cx, cy = -Lx/2, Ly/2
    else:
        cx, cy = center

    # ----------------------------
    # 2. Create rectangle (OCC)
    # ----------------------------
    p1 = gmsh.model.occ.addPoint(-Lx, 0, 0)
    p2 = gmsh.model.occ.addPoint(0,   0, 0)
    p3 = gmsh.model.occ.addPoint(0,  Ly, 0)
    p4 = gmsh.model.occ.addPoint(-Lx, Ly, 0)

    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p1)

    loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.occ.addPlaneSurface([loop])

    # Refinement target point (must be embedded)
    p_center = gmsh.model.occ.addPoint(cx, cy, 0)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.embed(0, [p_center], 2, surf)

    # ----------------------------
    # 3. Attractor field
    # ----------------------------
    f_attr = gmsh.model.mesh.field.add("Attractor")
    gmsh.model.mesh.field.setNumbers(f_attr, "NodesList", [p_center])

    # ----------------------------
    # 4. Threshold field (refinement)
    # ----------------------------
    f_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_attr)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", edge_size * 0.2)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", edge_size)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", radius * 0.3)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", radius)

    gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)

    # Safe options that exist in all modern Gmsh versions
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", edge_size * 0.2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", edge_size)

    # ----------------------------
    # 5. Generate mesh
    # ----------------------------
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")

    gmsh.write(output_file)
    gmsh.finalize()

    print(f"Mesh saved: {output_file}")


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
        A0 = smag*np.eye(2)
    else:
        A0 = np.zeros(2)
        A0[j] = smag

    # Generate mesh using gmsh
    mesh_file = "elastic_mesh.msh"
    generate_gmsh_mesh(L, L, h, output_file=mesh_file)

    d = {}

    d["options"] = {
        "cell_type": "T",
        "variant": "lumped",
        "degree": 4,
        "dimension": 2,
    }

    d["parallelism"] = {
        "type": "automatic",
    }

    d["mesh"] = {
        "Lz": L,
        "Lx": L,
        "Ly": 0.0,
        "mesh_type": "file",
        "mesh_file": mesh_file,
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
        "output_frequency": 100,
        "gradient_sampling_frequency": 100,
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
        "status": False,
    }

    wave = spyro.IsotropicWave(d)
    wave.forward_solve()
    spyro.plots.plot_receiver(wave, xi=0, filename="rec_at_dir0.png")
    spyro.plots.plot_receiver(wave, xi=1, filename="rec_at_dir1.png")

    return wave.receivers_output


def err(u_a, u_n):
    return norm(u_a - u_n)/norm(u_a)


if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        start = time.time()

    j = 0
    U_x = analytical_solution(0, j)
    U_z = analytical_solution(1, j)

    u_n = numerical_solution(j)
    u_x = u_n[:, 0, 0]
    u_z = u_n[:, 0, 1]

    if rank == 0:
        end = time.time()
        print(f"Elapsed time: {end - start:.2f} s")
        print(f"err_x = {err(U_x, u_x)}")
        print(f"err_z = {err(U_z, u_z)}")

        if moment_tensor:
            basename = "ExplosiveSource"
        else:
            basename = "ForceSource"
        output_dir = mkdir_timestamp(basename)
        np.save(os.path.join(output_dir, "x_analytical.npy"), U_x)
        np.save(os.path.join(output_dir, "z_analytical.npy"), U_z)
        np.save(os.path.join(output_dir, "numerical.npy"), u_n)

        print(f"Data saved to: {output_dir}")
