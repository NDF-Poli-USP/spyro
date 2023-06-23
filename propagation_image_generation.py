import numpy as np
import matplotlib.pyplot as plt

analytical_file = "analytical_solution_dt_0.0005.npy"
interior_file =  "interior_quads_rec_out0.0005.npy"
dof_file = "dof_quads_rec_out0.0005.npy"

dt = 0.0005

p_analytical = np.load(analytical_file)
p_interior = np.load(interior_file)
p_dof = np.load(dof_file)
p_interior = p_interior.flatten()
p_dof = p_dof.flatten()

time = np.linspace(0.0, 1.0, int(1.0/dt)+1)

p_analytical = p_analytical[400:]
p_interior = p_interior[400:]
p_dof = p_dof[400:]
time = time[400:]

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(time, p_interior, label="Interior $u_i$", color="red")
ax[0].plot(time, p_analytical, label="Analytical $u_a$", color="black", linestyle="--")
ax[0].plot(time, 500*(p_interior - p_analytical), label="$500|u_i-u_a|$", color="blue", linestyle="-.")
ax[0].legend()
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Pressure (Pa)")

ax[1].plot(time, p_dof, label="DOF $u_{DOF} $", color="red")
ax[1].plot(time, p_analytical, label="Analytical $u_a$", color="black", linestyle="--")
ax[1].plot(time, 500*(p_dof - p_analytical), label="$500|u_{DOF} - u_a|$", color="blue", linestyle="-.")
ax[1].legend()
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Pressure (Pa)")

# plt.plot(time, p_numerical, label="Numerical", color="red")
# plt.plot(time, p_analytical, label="Analytical", color="black", linestyle="--")
# plt.title(f"dt = {dts[i]}")
# plt.legend()
# plt.plot(time, -(p_analytical - p_numerical))
# plt.xlabel("Time (s)")
# plt.ylabel("Pressure (Pa)")
fig.suptitle("Source propagation comparison")
plt.show()
# nt = len(time)
# error_time = np.linalg.norm(p_analytical - p_numerical, 2) / np.sqrt(nt)
# errors.append(error_time)
# print(f"dt = {dts[i]}")
# print(f"Error = {error_time}")


print("End of script")