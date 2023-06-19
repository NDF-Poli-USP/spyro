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

analytical_files =[
    "analytical_solution_dt_5e-05.npy",
    "analytical_solution_dt_0.0001.npy",
    "analytical_solution_dt_0.0005.npy",
]
numerical_files = [
    "rec_out5e-05.npy",
    "rec_out0.0001.npy",
    "rec_out0.0005.npy",
]
dts = [
    5e-05,
    0.0001,
    0.0005,
]

analytical_files.reverse()
numerical_files.reverse()
dts.reverse()

errors = []

for i in range(len(analytical_files)):
    p_analytical = np.load(analytical_files[i])
    p_numerical = np.load(numerical_files[i])
    p_numerical = p_numerical.flatten()
    time = np.linspace(0.0, 1.0, int(1.0/dts[i]))
    nt = len(time)
    error_time = np.linalg.norm(p_analytical - p_numerical, 2) / np.sqrt(nt)
    errors.append(error_time)
    print(error_time)

plt.loglog(dts, errors)

theory = [t**2 for t in dts]
theory = [errors[0]*th/theory[0] for th in theory]

plt.loglog(dts, theory, '--')

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