import numpy as np
import matplotlib.pyplot as plt



dts = [
    0.001,
    0.0005,
    0.0001,
]

# errors = [
#     1.07e-3,
#     2.15e-4,
#     1.07e-4,
# ]

# errors = [
#     9.257970007003882e-04,
#     1.8510170642787083e-04,
#     9.254710294530231e-05,
# ]

# errors = [
#     1.4092664660258035e-08,
#     1.7643136210189937e-09,
#     2.7512883603466392e-11,
# ]

errors = [
    4.287914887554447e-09,
    5.363542490228256e-10,
    1.0802242342626882e-10,
]


plt.loglog(dts, errors)

theory = [t**2 for t in dts]
theory = [errors[0]*th/theory[0] for th in theory]

plt.loglog(dts, theory, '--')

theory = [t**3 for t in dts]
theory = [errors[0]*th/theory[0] for th in theory]

plt.loglog(dts, theory, '-.')

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