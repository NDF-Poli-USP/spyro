import matplotlib.pyplot as plt
import numpy as np

# Reference model (structured mesh):
# Freq=6Hz (peak)
# h=20m
# p=5
# DOF=1342991
# Nelem=62400

# Models with structured meshes
# Errors computed on Rec1 (close to sources line)
# monitor function: max(M1,M2), beta1=beta2=0.5
# smoothing factor = 0.001, nt = 1

p2_E_no_amr = np.array([48.31, 14.99, 7.86,  2.26])
p2_E_wi_amr = np.array([52.05, 14.11, 6.77,  1.65])
p2_dofs     = np.array([1977,  7793,  12141, 29983])
p2_cells    = np.array([640,   2560,  4000,  9920])
p2_h        = np.array([200,   100,   80,    50])

p3_E_no_amr = np.array([22.53, 3.46,  2.85,  0.86])
p3_E_wi_amr = np.array([20.42, 2.56,  1.7,   0.88])
p3_dofs     = np.array([4245,  16809, 26211, 64814])
p3_cells    = np.array([640,   2560,  4000,  9920])
p3_h        = np.array([200,   100,   80,    50])

p4_E_no_amr = np.array([8.1,  2.0,   1.24])
p4_E_wi_amr = np.array([7.35, 1.61,  1.09])
p4_dofs     = np.array([7153, 28385, 44281])
p4_cells    = np.array([640,  2560,  4000])
p4_h        = np.array([200,  100,   80])

p4_quad_E_no_amr = np.array([2.43,  1.69])
p4_quad_E_wi_amr = np.array([1.98,  1.53])
p4_quad_dofs     = np.array([16281, 40125])
p4_quad_cells    = np.array([1000,  2480])
p4_quad_h        = np.array([80,    50])

plt.plot(p2_dofs, p2_E_no_amr, label="p=2 (no AMR)",marker="o", linestyle="-", color='tab:green')
plt.plot(p2_dofs, p2_E_wi_amr, label="p=2 (w/ AMR)",marker="", linestyle="--", color='tab:green')

plt.plot(p3_dofs, p3_E_no_amr, label="p=3 (no AMR)",marker="", linestyle="-", color='tab:orange')
plt.plot(p3_dofs, p3_E_wi_amr, label="p=3 (w/ AMR)",marker="", linestyle="--", color='tab:orange')

plt.plot(p4_dofs, p4_E_no_amr, label="p=4 (no AMR)",marker="", linestyle="-", color='tab:blue')
plt.plot(p4_dofs, p4_E_wi_amr, label="p=4 (w/ AMR)",marker="", linestyle="--", color='tab:blue')

plt.plot(p4_quad_dofs, p4_quad_E_no_amr, label="p=4 quad (no AMR)",marker="", linestyle="-", color='tab:red')
plt.plot(p4_quad_dofs, p4_quad_E_wi_amr, label="p=4 quad (w/ AMR)",marker="", linestyle="--", color='tab:red')

# E X DOFs
plt.ylabel("E (%)",fontsize=14)
plt.xlabel("DOFs", fontsize=14)
plt.yscale('log')
plt.xscale('log')
plt.title("Acoustic forward with AMR (Marmousi)", fontsize=16)
plt.legend(loc='upper right')
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor')

plt.show()

# E X Nelem
plt.plot(p2_cells, p2_E_no_amr, label="p=2 (no AMR)",marker="", linestyle="-", color='tab:green')
plt.plot(p2_cells, p2_E_wi_amr, label="p=2 (w/ AMR)",marker="", linestyle="--", color='tab:green')

plt.plot(p3_cells, p3_E_no_amr, label="p=3 (no AMR)",marker="", linestyle="-", color='tab:orange')
plt.plot(p3_cells, p3_E_wi_amr, label="p=3 (w/ AMR)",marker="", linestyle="--", color='tab:orange')

plt.plot(p4_cells, p4_E_no_amr, label="p=4 (no AMR)",marker="", linestyle="-", color='tab:blue')
plt.plot(p4_cells, p4_E_wi_amr, label="p=4 (w/ AMR)",marker="", linestyle="--", color='tab:blue')

plt.plot(p4_quad_cells, p4_quad_E_no_amr, label="p=4 quad (no AMR)",marker="", linestyle="-", color='tab:red')
plt.plot(p4_quad_cells, p4_quad_E_wi_amr, label="p=4 quad (w/ AMR)",marker="", linestyle="--", color='tab:red')

plt.ylabel("E (%)",fontsize=14)
plt.xlabel("Nelem", fontsize=14)
plt.yscale('log')
plt.xscale('log')
plt.title("Acoustic forward with AMR (Marmousi)", fontsize=16)
plt.legend(loc='upper right')
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor')

plt.show()
