import matplotlib.pyplot as plt
import numpy as np

# P=2
# C=7.02 (based on Spyro paper to achieve ~5% of error)
# vp_bar=3 km/s (Marmousi - small)

# Freq=3 Hz, h_bar=150 m
# Freq=5 Hz, h_bar=100 m
# Freq=7 Hz, h_bar=50 m
E_ref = np.array([0.06663764588270706, 0.10299328406992929, 0.04700121819572666])*100
E_M0_Linf = np.array([0.10561743264595633, 0.11633638877261812, float("NaN")])*100
E_M1_Linf = np.array([0.0815676289680304, 0.09847245934907256, 0.05888168848143848])*100
x_f = np.array([3, 5, 7])


plt.plot(x_f, E_ref, label="E_ref (no AMR)",marker="X", linestyle="", color='tab:green')
plt.plot(x_f, E_M0_Linf, label="E_M0 (AMR w/ M0)",marker="s", linestyle="", color='tab:orange')
plt.plot(x_f, E_M1_Linf, label="E_M1 (AMR w/ M1)",marker="o", linestyle="", color='tab:red')


plt.ylabel("E (%)",fontsize=14)
plt.xlabel("Frequency", fontsize=14)
#plt.yscale('log')
plt.title("Acoustic forward with AMR (Marmousi)", fontsize=16)
plt.legend(loc='upper right')
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor')

plt.show()
