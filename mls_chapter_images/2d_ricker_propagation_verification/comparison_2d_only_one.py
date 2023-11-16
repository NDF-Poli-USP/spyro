import numpy as np
import matplotlib.pyplot as plt


dt = 0.0005

errors = []

analytical_filename = "analytical_solution_dt"+str(dt)+".npy"
numerical_filename = "forward_output_dt"+str(dt)+".npy"
p_analytical = np.load(analytical_filename)
p_numerical = np.load(numerical_filename)
time = np.linspace(0.0, 1.0, int(1.0/dt)+1)

plt.plot(time, p_numerical, label='Numerical interior propagation')
plt.plot(time, p_analytical, '-.', label='Analytical solution')

plt.legend()
plt.title("Ricker interior point propagation convergence in time")
plt.xlabel("dt [s]")
plt.ylabel("Pressure")
plt.savefig("mls_ricker_2d_propagation_verification_figA.png")
plt.show()
