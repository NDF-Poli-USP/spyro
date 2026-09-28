import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use
use("agg")

ml4_results = np.array([
    [1.1, 0.8127286],
    [1.2, 0.3641711],
    [1.4, 0.2947500],
    [1.6, 0.0950266],
    [1.8, 0.1628653],
    [2.0, 0.1768795],
    [2.1, 0.1138699],
    [2.2, 0.0428772],
    [2.5, 0.0443505],
    [2.8, 0.0170727],
])

ml6_results = np.array([
    [1.1, 0.2161752],
    [1.2, 0.0602536],
    [1.4, 0.0619151],
    [1.5, 0.0168581],
    [1.6, 0.0321273],
    [1.8, 0.0267796],
    [2.0, 0.0360494],
])

sem4_unstructured_results = np.array([
    [1.0, 2.5273657],
    [1.2, 1.3767337],
    [1.4, 0.7735119],
    [1.5, 0.3077950],
    [1.6, 0.4306072],
    [1.8, 0.6150052],
    [2.0, 0.2867249],
    [2.2, 0.2577596],
    [2.4, 0.1848269],
    [2.6, 0.0753155],
    [2.8, 0.0643854],
    [3.0, 0.0593463],
    [3.2, 0.0492546],
    [3.3, 0.1170831],
    [3.4, 0.0623501],
    [3.5, 0.0474101],
])

sem4_structured_winslow_results = np.array([
    [1.0, 2.0016434],
    [1.2, 1.0126616],
    [1.4, 1.3444935],
    [1.6, 0.6310205],
    [1.8, 0.5377709],
    [2.0, 0.4101345],
    [2.2, 0.1931828],
    [2.4, 0.3257839],
    [2.6, 0.1717001],
    [2.8, 0.1243243],
    [3.0, 0.1471740],
    [3.1, 0.1670331],
    [3.2, 0.1012959],
    [3.3, 0.0849120],
    [3.4, 0.0702761],
    [3.5, 0.0676565],
])

sem4_unstructured_results = sem4_unstructured_results[np.argsort(sem4_unstructured_results[:, 0])]
sem4_structured_winslow_results = sem4_structured_winslow_results[np.argsort(sem4_structured_winslow_results[:, 0])]
ml4_results = ml4_results[np.argsort(ml4_results[:, 0])]
ml6_results = ml6_results[np.argsort(ml6_results[:, 0])]

plt.figure(figsize=(8, 6))

plt.plot(
    sem4_unstructured_results[:, 0],
    sem4_unstructured_results[:, 1],
    "s-",
    label="SEM4 unstructured",
)

plt.plot(
    sem4_structured_winslow_results[:, 0],
    sem4_structured_winslow_results[:, 1],
    "^-",
    label="SEM4 structured Winslow",
)

plt.plot(
    ml4_results[:, 0],
    ml4_results[:, 1],
    "^-",
    label="ML4",
)

plt.plot(
    ml6_results[:, 0],
    ml6_results[:, 1],
    "^-",
    label="ML6",
)


# Error threshold
plt.axhline(
    5e-2,
    linestyle="--",
    linewidth=1.5,
    label=r"Error = $5\times10^{-2}$",
)

# Error threshold
plt.axhline(
    10e-2,
    linestyle="--",
    linewidth=1.5,
    label=r"Error = $1\times10^{-2}$",
)


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

plt.xlabel("Cells per wavelength (CPW)")
plt.ylabel("Normalized $L^2$ error in receivers")
plt.title("Heterogeneous isotropic elastic CPW vs error")

plt.yscale("log")

plt.grid(
    True,
    which="both",
    linestyle=":",
    alpha=0.5,
)

plt.legend()

plt.tight_layout()
plt.savefig("2d_heterogeneous_cpw.png")
