#!/usr/bin/env python3
from matplotlib import use
import numpy as np
import matplotlib.pyplot as plt
use("agg")


# Results
cpw = np.array([
    5.500e+00,
    4.583e+00,
    3.438e+00,
    2.750e+00,
    2.521e+00,
    2.292e+00,
    2.108e+00,
    1.971e+00,
    1.788e+00,
    2.017e+00,
    2.200e+00,
    2.383e+00,
    2.521e+00,
    2.613e+00,
    2.704e+00,
    2.796e+00
])

error = np.array([
    1.117542e-02,
    1.138238e-02,
    1.184404e-02,
    1.592501e-02,
    3.977317e-02,
    1.905037e-01,
    1.958987e-01,
    3.550383e-01,
    1.144726e+00,
    8.018754e-01,
    1.356192e-01,
    6.306767e-02,
    3.977317e-02,
    2.047810e-02,
    2.972936e-02,
    2.086649e-02
])

cpw = cpw / 2.0
# Sort by cpw for a cleaner plot
idx = np.argsort(cpw)
cpw = cpw[idx]
error = error[idx]


# Plot
fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    cpw,
    error,
    s=60,
    label="Numerical results",
)

ax.plot(
    cpw,
    error,
    linewidth=1,
    alpha=0.6,
)
# Reference error threshold
ax.axhline(
    y=5e-2,
    linestyle="--",
    linewidth=1.5,
    label="Error threshold = $5\\times10^{-2}$",
)


ax.set_xlabel("Cells per wavelength (cpw)")
ax.set_ylabel("Normalized $L^2$ error")
ax.set_title("4th order mlt analytical cpw")
# ax.set_title("Error vs. cells per wavelength")

# Error spans almost two orders of magnitude
ax.set_yscale("log")

ax.grid(True, which="both", alpha=0.3)
ax.legend()

fig.tight_layout()
plt.savefig("mlt_analytical_2d_cpw.png")