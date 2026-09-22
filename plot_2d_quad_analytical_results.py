#!/usr/bin/env python3
from matplotlib import use
import numpy as np
import matplotlib.pyplot as plt
use("agg")


# Results
cpw = np.array([
    3.071e+00,
    2.979e+00,
    2.796e+00,
    2.750e+00,
    2.704e+00,
    2.613e+00,
    2.521e+00,
    2.383e+00,
    2.292e+00,
    2.200e+00,
    2.108e+00,
    2.017e+00,
    1.971e+00,
    1.833e+00,
    1.788e+00,
    3.988e+00,
    3.483e+00,
])

error = np.array([
    1.669422e-02,
    3.271060e-02,
    2.506636e-02,
    3.082833e-02,
    8.239186e-02,
    5.599806e-02,
    5.695384e-02,
    1.001744e-01,
    5.180735e-01,
    4.866781e-01,
    3.555042e-01,
    1.823181e+00,
    6.607366e-01,
    1.172493e+00,
    3.321154e+00,
    1.125592e-02,
    1.323520e-02,
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
ax.set_title("4th order quadrilateral analytical cpw")
# ax.set_title("Error vs. cells per wavelength")

# Error spans almost two orders of magnitude
ax.set_yscale("log")

ax.grid(True, which="both", alpha=0.3)
ax.legend()

fig.tight_layout()
plt.savefig("quad_analytical_2d_cpw.png")