import matplotlib.pyplot as plt
import numpy as np


p2_J_no_amr = np.array([2.303198e-01, 0.002556973998088136, 0.0006536535245262438, 0.0003448399853630157, 0.00024164788340689074])
p2_J_amr_1  = np.array([2.303198e-01, 0.002556973998088136, 0.0006529824415775318, 0.0003901106821485473, 0.0002598930447076839])
p2_J_amr_2  = np.array([2.344967e-01, 0.002900499985120751, 0.0006682827818409827, 0.0003895897296198926, 0.0002567261520124321])

max_J = 2.303198e-01

fwi_it = np.array([0, 20, 40, 60, 80])

plt.plot(fwi_it, p2_J_no_amr/max_J, label="p=2 (no AMR)",marker="^", linestyle="-", color='tab:red')
plt.plot(fwi_it, p2_J_amr_1/max_J, label="p=2 (AMR 1)",marker=">", linestyle="-", color='tab:green')
plt.plot(fwi_it, p2_J_amr_2/max_J, label="p=2 (AMR 2)",marker="o", linestyle="-", color='tab:blue')

plt.ylabel("J",fontsize=14)
plt.xlabel("# FWI ", fontsize=14)
plt.yscale('log')
#plt.xscale('log')
plt.title("Acoustic FWI with AMR (Marmousi)", fontsize=16)
plt.legend(loc='upper right')
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor')

plt.show()

