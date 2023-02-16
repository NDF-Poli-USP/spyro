# results obtained with demos/run_fwi_acoustic_moving_mesh_circle_case.py

import matplotlib.pyplot as plt
import numpy as np

it = np.arange(11)

#J_no_amr_smooth_p2 = np.array([, ])
#J_no_amr_smooth_p3 = np.array([, ])
#J_no_amr_smooth_p4 = np.array([, ])
#J_amr_smooth_p2 = np.array([, ])
#J_amr_smooth_p3 = np.array([, ])
#J_amr_smooth_p4 = np.array([, ])

J_no_amr_sharp_p2 = np.array([1.756261e-02, 0.006291904284134328, 0.003761983199818125, 0.002627595097744294, 0.0022609683153755455, 0.0021975644642715083, 0.0021784898403480425, 0.0021774338839960056, 0.002177433883996435, 0.002177433883996435, 0.002177433883996435])
J_no_amr_sharp_p3 = np.array([1.756056e-02, 0.004346242285032501, 0.0031030734056154653, 0.002762585454125219, 0.0027067255077575696, 0.0027008187628750173, 0.0027008187628804097, 0.0027008187628804097, 0.0027008187628804097, 0.0027008187628804097, 0.0027008187628804097])
J_no_amr_sharp_p4 = np.array([1.761855e-02, 0.005314874251128027, 0.0034408505487979403, 0.0028856330560917487, 0.0028145846964039017, 0.0027862255203450668, 0.002785692193187278, 0.002785692193190838, 0.002785692193190838, 0.002785692193190838, 0.002785692193190838])

J_amr_sharp_p2 = np.array([1.756261e-02,])

J_amr_sharp_p3 = np.array([1.756056e-02 ,0.004346242285032501, 0.003088123502885359, 0.0024994368573648743, 0.002368671232322726, 0.0023598885775466165, 0.002359871830165363, 0.0023598718301682206, 0.0023598718301682206, 0.0023598718301682206, 0.0023598718301682206])
J_amr_sharp_p4 = np.array([1.761855e-02, 0.005314874251128027, 0.003383378198957358, 0.002826223946024245, 0.002711683644697945, 0.0026888126987486905, 0.0026887221810075437, 0.0026887221810080316, 0.0026887221810080316, 0.0026887221810080316, 0.0026887221810080316])

plt.plot(it, J_no_amr_sharp_p2, label='no AMR, p=2',marker="o", linestyle='-', color='tab:blue')
plt.plot(it, J_no_amr_sharp_p3, label='no AMR, p=3',marker="o", linestyle='-', color='tab:orange')
plt.plot(it, J_no_amr_sharp_p4, label='no AMR, p=4',marker="o", linestyle='-', color='tab:green')
#plt.plot(it, J_amr_sharp_p2, label='AMR, p=2',marker="o", linestyle='--', color='tab:blue')
plt.plot(it, J_amr_sharp_p3, label='AMR, p=3',marker="o", linestyle='--', color='tab:orange')
plt.plot(it, J_amr_sharp_p4, label='AMR, p=4',marker="o", linestyle='--', color='tab:green')

plt.ylabel("J total",fontsize=14)
plt.xlabel("Loop iteration", fontsize=14)
plt.title("FWI with AMR (circle case)", fontsize=16)
plt.legend(loc='upper right')
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor')

plt.show()


