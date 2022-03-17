import matplotlib.pyplot as plt
import numpy as np

num_cores = np.array([1, 2, 4, 8, 16])

time_p2   = np.array([345.0, 180.2, 90.5, 50.5, 27.9])
time_p2_t = np.array([345.0, 345.0/2, 345.0/4, 345.0/8, 345.0/16])
dofs_p2   = np.array([49711702, 49711702/2., 49711702/4., 49711702/8., 49711702/16.])
cells_p2  = 8282542

time_p3   = np.array([201.1, 100.0, 50.5, 28.2, 20.3])
time_p3_t = np.array([201.1, 201.1/2, 201.1/4, 201.1/8, 201.1/16])
dofs_p3   = np.array([29859266, 29859266/2., 29859266/4., 29859266/8., 29859266/16.])
cells_p3  = 2295867

time_p4   = np.array([172.5, 85.9, 42.1, 24.4, 14.9])
time_p4_t = np.array([172.5, 172.5/2, 172.5/4, 172.5/8, 172.5/16])
dofs_p4   = np.array([26249734, 26249734/2., 26249734/4., 26249734/8., 26249734/16.])
cells_p3  = 1192602

time_p5   = np.array([226.1, 112.9, 61.6, 33.3, 22.8])
time_p5_t = np.array([226.1, 226.1/2, 226.1/4, 226.1/8, 226.1/16])
dofs_p5   = np.array([29676024, 29676024/2., 29676024/4., 29676024/8., 29676024/16.])
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
