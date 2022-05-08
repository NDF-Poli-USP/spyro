# strong scaling analysis performed on Intel10 (NDF/Poli)

import matplotlib.pyplot as plt
import numpy as np

num_cores = np.array([1, 2, 4, 8, 16])

time_p2   = np.array([175.1, 86.1, 44.1, 28.2, 14.9])
time_p2_t = np.array([175.1, 175.1/2, 175.1/4, 175.1/8, 175.1/16])
dofs_p2   = np.array([24855851, 24855851/2., 24855851/4., 24855851/8., 24855851/16.])
cells_p2  = 8282542 

time_p3   = np.array([109.2, 52.8, 28.2, 15.8, 9.2])
time_p3_t = np.array([109.2, 109.2/2, 109.2/4, 109.2/8, 109.2/16])
dofs_p3   = np.array([14929633, 14929633/2., 14929633/4., 14929633/8., 14929633/16.])
cells_p3  = 2295867 

time_p4   = np.array([80.3, 40.5, 22.7, 12.1, 8.0])
time_p4_t = np.array([80.3, 80.3/2, 80.3/4, 80.3/8, 80.3/16])
dofs_p4   = np.array([13124867, 13124867/2., 13124867/4., 13124867/8., 13124867/16.])
cells_p3  = 1192602

time_p5   = np.array([113.8, 56.9, 29.1, 19.8, 11.3])
time_p5_t = np.array([113.8, 113.8/2, 113.8/4, 113.8/8, 113.8/16])
dofs_p5   = np.array([14838012, 14838012/2., 14838012/4., 14838012/8., 14838012/16.])
cells_p5  = 689864

plt.plot(num_cores, time_p2, label='P2',marker="o", linestyle='')
plt.plot(num_cores, time_p3, label='P3',marker="o", linestyle='')
plt.plot(num_cores, time_p4, label='P4',marker="o", linestyle='')
plt.plot(num_cores, time_p5, label='P5',marker="o", linestyle='')

plt.plot(num_cores, time_p2_t, label='',marker="", linestyle='--', color='tab:blue')
plt.plot(num_cores, time_p3_t, label='',marker="", linestyle='--', color='tab:orange')
plt.plot(num_cores, time_p4_t, label='',marker="", linestyle='--', color='tab:green')
plt.plot(num_cores, time_p5_t, label='',marker="", linestyle='--', color='tab:red')

plt.yscale('log')
plt.xscale('log')
plt.ylabel("simulation time (s)",fontsize=14)
plt.xlabel("number of cores", fontsize=14)
plt.title("Strong scaling analysis (2D ML elements)", fontsize=16)
plt.legend(loc='upper right')
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor')

plt.annotate(str(round(dofs_p2[0])), (num_cores[0], 1.05*time_p2[0]), color='tab:blue', fontsize=8)
plt.annotate(str(round(dofs_p2[1])), (num_cores[1], 1.05*time_p2[1]), color='tab:blue', fontsize=8)
plt.annotate(str(round(dofs_p2[2])), (num_cores[2], 1.05*time_p2[2]), color='tab:blue', fontsize=8)
plt.annotate(str(round(dofs_p2[3])), (num_cores[3], 1.05*time_p2[3]), color='tab:blue', fontsize=8)
plt.annotate(str(round(dofs_p2[4])), (num_cores[4], 1.05*time_p2[4]), color='tab:blue', fontsize=8)

plt.annotate(str(round(dofs_p3[0])), (num_cores[0], 0.9*time_p3[0]), color='tab:orange', fontsize=8)
plt.annotate(str(round(dofs_p3[1])), (num_cores[1], 0.9*time_p3[1]), color='tab:orange', fontsize=8)
plt.annotate(str(round(dofs_p3[2])), (num_cores[2], 0.9*time_p3[2]), color='tab:orange', fontsize=8)
plt.annotate(str(round(dofs_p3[3])), (num_cores[3], 0.9*time_p3[3]), color='tab:orange', fontsize=8)
plt.annotate(str(round(dofs_p3[4])), (num_cores[4], 0.9*time_p3[4]), color='tab:orange', fontsize=8)

plt.annotate(str(round(dofs_p4[0])), (num_cores[0], 0.8*time_p4[0]), color='tab:green', fontsize=8)
plt.annotate(str(round(dofs_p4[1])), (num_cores[1], 0.8*time_p4[1]), color='tab:green', fontsize=8)
plt.annotate(str(round(dofs_p4[2])), (num_cores[2], 0.8*time_p4[2]), color='tab:green', fontsize=8)
plt.annotate(str(round(dofs_p4[3])), (num_cores[3], 0.8*time_p4[3]), color='tab:green', fontsize=8)
plt.annotate(str(round(dofs_p4[4])), (num_cores[4], 0.8*time_p4[4]), color='tab:green', fontsize=8)

plt.annotate(str(round(dofs_p5[0])), (num_cores[0], 1.05*time_p5[0]), color='tab:red', fontsize=8)
plt.annotate(str(round(dofs_p5[1])), (num_cores[1], 1.05*time_p5[1]), color='tab:red', fontsize=8)
plt.annotate(str(round(dofs_p5[2])), (num_cores[2], 1.05*time_p5[2]), color='tab:red', fontsize=8)
plt.annotate(str(round(dofs_p5[3])), (num_cores[3], 1.05*time_p5[3]), color='tab:red', fontsize=8)
plt.annotate(str(round(dofs_p5[4])), (num_cores[4], 1.05*time_p5[4]), color='tab:red', fontsize=8)

plt.show()
