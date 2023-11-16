import numpy as np
import matplotlib.pyplot as plt


dts = [
    5e-05,
    0.0001,
    0.0005,
]

dts.reverse()

errors = []

for dt in dts:
    analytical_filename = "analytical_solution_dt"+str(dt)+".npy"
    numerical_filename = "forward_output_dt"+str(dt)+".npy"
    p_analytical = np.load(analytical_filename)
    p_numerical = np.load(numerical_filename)
    time = np.linspace(0.0, 1.0, int(1.0/dt)+1)
    nt = len(time)
    a = np.shape(p_analytical)
    print(f"P analytical shape {a}")
    a = np.shape(p_numerical)
    print(f"P numerical shape {a}")
    error_time = np.linalg.norm(p_analytical - p_numerical, 2) / np.sqrt(nt)
    errors.append(error_time)
    print(error_time)

plt.loglog(dts, errors)

theory = [t**2 for t in dts]
theory = [errors[0]*th/theory[0] for th in theory]

plt.loglog(dts, theory, '--')

plt.show()
