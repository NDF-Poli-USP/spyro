# strong scaling analysis performed on Ogun (SENAI/CIMATEC) April/2022

import matplotlib.pyplot as plt
import numpy as np

num_cores = np.array([1, 2, 4, 8, 16, 32, 64])

time_p2   = np.array([318.8, 159.1, 79.3, 40.6, 20.6, 10.6, 5.6]) # max cores per node = 16
#time_p2   = np.array([318.8, 159.1, 79.3, 40.6, 20.6, 11.7, 6.2]) # max cores per node = 32
time_p2_t = time_p2[0]/num_cores
dofs_p2   = 49711702/num_cores 
cells_p2  = 8282542

time_p3   = np.array([164.2, 80.7, 40.0, 20.7, 11.0, 5.9, 3.2]) # max cores per node = 16
#time_p3   = np.array([164.2, 80.7, 40.0, 20.7, 11.0, 6.8, 3.7]) # max cores per node = 32
time_p3_t = time_p3[0]/num_cores 
dofs_p3   = 29859266/num_cores 
cells_p3  = 2295867

time_p4   = np.array([143.8, 71.3, 34.3, 18.5, 10.2, 5.2, 2.9]) # max cores per node = 16
#time_p4   = np.array([143.8, 71.3, 34.3, 18.5, 10.2, 6.1, 3.4]) # max cores per node = 32
time_p4_t = time_p4[0]/num_cores 
dofs_p4   = 26249734/num_cores 
cells_p3  = 1192602

time_p5   = np.array([192.6, 95.8, 48.1, 24.5, 12.9, 6.9, 3.9]) # max cores per node = 16
#time_p5   = np.array([192.6, 95.8, 48.1, 24.5, 12.9, 8.0, 4.4]) # max cores per node = 32
time_p5_t = time_p5[0]/num_cores
dofs_p5   = 29676024/num_cores
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
plt.legend(prop={"size":14})
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor')

#for i in range(len(num_cores)):
#    plt.annotate(str(round(dofs_p2[i])), (num_cores[i], 1.05*time_p2[i]), color='tab:blue', fontsize=8)
#    plt.annotate(str(round(dofs_p3[i])), (num_cores[i], 0.9*time_p3[i]), color='tab:orange', fontsize=8)
#    plt.annotate(str(round(dofs_p4[i])), (num_cores[i], 0.8*time_p4[i]), color='tab:green', fontsize=8)
#    plt.annotate(str(round(dofs_p5[i])), (num_cores[i], 1.05*time_p5[i]), color='tab:red', fontsize=8)

plt.show()
