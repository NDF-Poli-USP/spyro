from mpi4py.MPI import COMM_WORLD
from mpi4py import MPI
import debugpy
debugpy.listen(3000 + COMM_WORLD.rank)
debugpy.wait_for_client()

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
from PIL import Image
from copy import deepcopy
from firedrake import File
import firedrake as fire
import spyro


def error_calc(p_numerical, p_analytical, nt):
    norm = np.linalg.norm(p_numerical, 2) / np.sqrt(nt)
    error_time = np.linalg.norm(p_analytical - p_numerical, 2) / np.sqrt(nt)
    div_error_time = error_time / norm
    return div_error_time


final_time = 1.0

dictionary = {}
dictionary["options"] = {
    "cell_type": "Q",  # simplexes such as triangles or tetrahedra (T) or quadrilaterals (Q)
    "variant": "lumped",  # lumped, equispaced or DG, default is lumped
    "degree": 4,  # p order
    "dimension": 2,  # dimension
}
dictionary["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor) or spatial
}
dictionary["mesh"] = {
    "Lz": 3.0,  # depth in km - always positive   # Como ver isso sem ler a malha?
    "Lx": 3.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "mesh_file": None,
    "mesh_type": "firedrake_mesh",
}
dictionary["acquisition"] = {
    "source_type": "ricker",
    "source_locations": [(-1.1, 1.2), (-1.1, 1.5), (-1.1, 1.8)],
    "frequency": 5.0,
    "delay": 0.2,
    "delay_type": "time",
    "receiver_locations": spyro.create_transect((-1.3, 1.2), (-1.3, 1.8), 301),
}
dictionary["time_axis"] = {
    "initial_time": 0.0,  # Initial time for event
    "final_time": final_time,  # Final time for event
    "dt": 0.001,  # timestep size
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "output_frequency": 100,  # how frequently to output solution to pvds - Perguntar Daiane ''post_processing_frequnecy'
    "gradient_sampling_frequency": 1,
}
dictionary["visualization"] = {
    "forward_output": False,
    "forward_output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
}

Wave_obj = spyro.AcousticWave(dictionary=dictionary)
Wave_obj.set_mesh(mesh_parameters={"dx": 0.1})

mesh_z = Wave_obj.mesh_z
cond = fire.conditional(mesh_z < -1.5, 3.5, 1.5)
Wave_obj.set_initial_velocity_model(conditional=cond, output=True)

# Wave_obj.source_locations.append((-1.1, 1.5))
# Wave_obj.source_locations.append((-1.1, 1.8))

# mesh = fire.RectangleMesh(30, 30, 3.0, 3.0, quadrilateral=True)
# x_mesh, y_mesh = fire.SpatialCoordinate(mesh)
# V = fire.FunctionSpace(mesh, "DG", 0)
# vp = fire.Function(V)
# new_cond = fire.conditional(y_mesh < 1.5, 3.5, 1.5)
# vp.interpolate(new_cond)

# fig = plt.figure(figsize=(7, 12))
# axes = fig.add_subplot(111)
# vp_image = fire.tripcolor(vp, axes=axes)
# sources = [(1.2, 3-1.1), (1.5, 3-1.1), (1.8, 3-1.1)]
# receivers = spyro.create_transect((1.2, 3-1.3), (1.8, 3-1.3), 15)
# for source in sources:
#     z, x = source
#     plt.scatter(z, x, c="green")
# for receiver in receivers:
#     z, x = receiver
#     plt.scatter(z, x, c="red")
# abc_points = [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]
# if abc_points is not None:
#     zs = []
#     xs = []

#     first = True
#     for point in abc_points:
#         z, x = point
#         zs.append(z)
#         xs.append(x)
#         if first:
#             z_first = z
#             x_first = x
#         first = False
#     zs.append(z_first)
#     xs.append(x_first)
#     plt.plot(zs, xs, "--")

# cbar = plt.colorbar(vp_image, orientation="horizontal")
# # axes.axis('equal')
# plt.show()
# plt.savefig("experiment_setup.png")

# print("TEST")

# spyro.plots.plot_model(
#     Wave_obj,
#     filename="experiment_setup.png",
#     abc_points=[(-1.0, 1.0), (-2.0, 1.0), (-2.0, 2.0), (-1.0, 2.0)])

Wave_obj.forward_solve()

comm = Wave_obj.comm

arr = Wave_obj.receivers_output
nt = int(Wave_obj.final_time / Wave_obj.dt) + 1  # number of timesteps
num_recvs = Wave_obj.number_of_receivers
x_space = np.linspace(0, num_recvs, num_recvs)
t_space = np.linspace(0.0, Wave_obj.final_time, nt)
x_rec = x_space
t_rec = t_space[190:900]
X, Y = np.meshgrid(x_rec, t_rec)

fig = plt.figure(figsize=(5, 10))
cmap = plt.get_cmap("gray")
plt.contourf(X, Y, arr[190:900, :], 700, cmap=cmap)
plt.gca().invert_yaxis()
plt.title("Shot record for source 0")
plt.xlabel("Receiver id")
plt.ylabel("Time (s)")

if comm.ensemble_comm.rank == 0:
    analytical_p = spyro.utils.nodal_homogeneous_analytical(
        Wave_obj, 0.2, 1.5, n_extra=100
    )
else:
    analytical_p = None
analytical_p = comm.ensemble_comm.bcast(analytical_p, root=0)

# Checking if error before reflection matches
if comm.ensemble_comm.rank == 0:
    rec_id = 0
elif comm.ensemble_comm.rank == 1:
    rec_id = 150
elif comm.ensemble_comm.rank == 2:
    rec_id = 300

arr0 = arr[:, rec_id]
arr0 = arr0.flatten()

error = error_calc(arr0[:430], analytical_p[:430], 430)
error_all = COMM_WORLD.allreduce(error, op=MPI.SUM)
error_all /= 3

test = np.abs(error_all) < 0.01

assert test

print("END")
