import numpy as np
import matplotlib.pyplot as plt

# analytical_files =[
#     "analytical_solution_dt_0.0005.npy",
#     "analytical_solution_dt_0.0006.npy",
#     "analytical_solution_dt_0.00075.npy",
#     "analytical_solution_dt_0.0008.npy",
#     "analytical_solution_dt_0.001.npy",
# ]
# numerical_files = [
#     "rec_out0.0005.npy",
#     "rec_out0.0006.npy",
#     "rec_out0.00075.npy",
#     "rec_out0.0008.npy",
#     "rec_out0.001.npy",
# ]
# dts = [
#     0.0005,
#     0.0006,
#     0.00075,
#     0.0008,
#     0.001,
# ]

def error_calc(p_numerical, p_analytical, nt):
    norm = np.linalg.norm(p_numerical, 2) / np.sqrt(nt)
    error_time = np.linalg.norm(p_analytical - p_numerical, 2) / np.sqrt(nt)
    div_error_time = error_time / norm
    return div_error_time
        

analytical_files =[
    "analytical_solution_dt_5e-05.npy",
    "analytical_solution_dt_0.0001.npy",
    "analytical_solution_dt_0.0005.npy",
]
numerical_dof_files = [
    "dof_quads_rec_out5e-05.npy",
    "dof_quads_rec_out0.0001.npy",
    "dof_quads_rec_out0.0005.npy",
]
numerical_interior_files = [
    "interior_quads_rec_out5e-05.npy",
    "interior_quads_rec_out0.0001.npy",
    "interior_quads_rec_out0.0005.npy",
]
dts = [
    5e-05,
    0.0001,
    0.0005,
]

analytical_files.reverse()
numerical_dof_files.reverse()
numerical_interior_files.reverse()
dts.reverse()

errors_dof = []
errors_interior = []
errors_between = []

for i in range(len(analytical_files)):
    p_analytical = np.load(analytical_files[i])
    p_numerical_dof = np.load(numerical_dof_files[i])
    p_numerical_interior = np.load(numerical_interior_files[i])
    p_numerical_dof = p_numerical_dof.flatten()
    p_numerical_interior = p_numerical_interior.flatten()
    time = np.linspace(0.0, 1.0, int(1.0/dts[i])+1)
    plt.plot(time, p_numerical_dof, label="Numerical DoF propagation", color="red")
    plt.plot(time, p_numerical_interior, '-.', label="Numerical Interior propagation", color="blue")
    plt.plot(time, p_analytical, label="Analytical", color="black", linestyle="--")
    plt.title(f"dt = {dts[i]}")
    plt.legend()
    # plt.plot(time, -(p_analytical - p_numerical))
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (Pa)")
    plt.show()
    nt = len(time)
    error_dof = error_calc(p_numerical_dof, p_analytical, nt)
    error_interior = error_calc(p_numerical_interior, p_analytical, nt)
    error_between = error_calc(p_numerical_interior, p_numerical_dof, nt)

    errors_dof.append(error_dof)
    errors_interior.append(error_interior)
    errors_between.append(error_between)

    print(f"dt = {dts[i]}")
    print("Error DOF = {:.4e}".format(error_dof))
    print("Error interior = {:.4e}".format(error_interior))
    print("Error between = {:.4e}".format(error_between))

plt.loglog(dts, errors_dof, '-o', label='numerical dof propagation')
plt.loglog(dts, errors_interior, '-.*', label='numerical interior propagation')

theory = [t**2 for t in dts]
theory = [errors_dof[0]*th/theory[0] for th in theory]

plt.loglog(dts, theory, '--^', label='theoretical 2nd order in time')
for x, y, a in zip([t for t in dts], theory, [('dt = {:.0e} s'.format(t)) for t in dts]):
            plt.annotate(a, xy=(x, y), xytext=(4, 2),
                         textcoords='offset points',
                         horizontalalignment='left', verticalalignment='top')

plt.legend()
plt.title(f"Convergence with central difference in time scheme")
plt.xlabel("dt [s]")
plt.ylabel(r'error $\frac{{||u_{num} - u_{an}||_2}}{{||u_{an}||_2}}$')

# theory = [t for t in dts]
# theory = [errors[0]*th/theory[0] for th in theory]

# plt.loglog(dts, theory, '-.')

plt.show()

# p_analytical = np.load("analytical_solution_dt_0.0005.npy")

# p_4_full = np.load("rec_out.npy")
# p_4 = p_4_full.flatten()
# time = np.linspace(0.0, 1.0, int(1.0/0.0005))
# nt = len(time)

# plt.plot(time, p_analytical, '--', label="Analytical")
# plt.plot(time, p_4, label="4th order")

# error_time = np.linalg.norm(p_analytical - p_4, 2) / np.sqrt(nt)
# print(error_time)

# # plt.plot(time, 100 *(p_analytical- p_4), '-b', label='difference x100')

# plt.show()

print("End of script")