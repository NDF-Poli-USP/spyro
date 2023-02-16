import matplotlib.pyplot as plt
import numpy as np

dof_elastic = np.array([605106, 1309630, 2215140, 4326566])
#time_elastic = np.array([229.3, 517.5, 1047.3]) bkp
time_elastic = np.array([36.3, 82.0, 166.8, 432.9])

dof_acoustic = np.array([302553, 654815, 1107570, 2163283])
#time_acoustic = np.array([116.4, 273.6, 555.6]) bkp
time_acoustic = np.array([19.4, 43.7, 89.7, 232.0])

plt.plot(dof_elastic, time_elastic, label='Elastic',marker="o")
plt.plot(dof_acoustic, time_acoustic, label='Acoustic',marker="o")
plt.yscale('log')
plt.xscale('log')
plt.ylabel("simulation time (s)",fontsize=14)
plt.xlabel("DOFs", fontsize=14)
plt.title("Acoustic model x Elastic model", fontsize=16)
plt.legend(loc='lower right')
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor')
plt.annotate(" p=2", (dof_elastic[0], 0.8*time_elastic[0]), color='tab:blue', fontsize=14)
plt.annotate(" 3", (dof_elastic[1], 0.8*time_elastic[1]), color='tab:blue', fontsize=14)
plt.annotate(" 4", (dof_elastic[2], 0.8*time_elastic[2]), color='tab:blue', fontsize=14)
plt.annotate("p=2", (dof_acoustic[0], 1.2*time_acoustic[0]), color='tab:orange', fontsize=14)
plt.annotate(" 3", (dof_acoustic[1], 0.8*time_acoustic[1]), color='tab:orange', fontsize=14)
plt.annotate(" 4", (dof_acoustic[2], 0.8*time_acoustic[2]), color='tab:orange', fontsize=14)
plt.show()
