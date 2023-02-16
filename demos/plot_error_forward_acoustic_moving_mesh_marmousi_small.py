import matplotlib.pyplot as plt
import numpy as np

# Reference model (structured mesh):
# Freq=6Hz (peak)
# h=20m
# p=5
# DOF=1342991
# Nelem=62400

# Reference model 2 (structured mesh):
# Freq=6Hz (peak)
# h=20m
# p=4
# DOF=687513
# Nelem=62400


# Models with structured meshes
# Errors computed on Rec1 (close to sources line)
# monitor function: max(M1,M2), beta1=beta2=0.5
# smoothing factor = 0.001, nt = 1
# Neumann-type source

p2_E_no_amr = np.array([70.03,  65.77, 36.60,  21.12, 8.17, 5.07,   0.98]) 
p2_E_wi_amr = np.array([68.54,  64.92, 32.98,  22.5,  8.83, 4.34,   0.96]) 
p2_dofs     = np.array([509,    1049,   1977,  3071,  7793,  12141,  29983])
p2_cells    = np.array([160,    336,    640,   1000,  2560,  4000,   9920])
p2_h        = np.array([400,    286,    200,   160,   100,   80,     50])

p3_E_no_amr = np.array([49.38, 20.55, 12.24, 6.07,  1.53,  1.40,  0.53]) 
p3_E_wi_amr = np.array([47.11, 19.68, 11.65, 5.13,  1.55,  0.96,  0.43]) 
p3_dofs     = np.array([1083,  2245,  4245,  6606,  16809, 26211, 64814])
p3_cells    = np.array([160,   336,   640,   1000,  2560,  4000,  9920])
p3_h        = np.array([400,   286,   200,   160,   100,   80,    50])

p4_E_no_amr = np.array([20.91, 12.77, 4.12,  2.84,  1.16,  0.70,   0.39]) 
p4_E_wi_amr = np.array([21.09, 10.86, 3.40,  2.33,  1.00,  0.72,   0.38])  
p4_dofs     = np.array([1817,  3777,  7153,  11141, 28385, 44281,  109565])
p4_cells    = np.array([160,   336,   640,   1000,  2560,  4000,   9920])
p4_h        = np.array([400,   286,   200,   160,   100,   80,     50])

# for p-extension (p2, p3, p4)
h200_E_no_amr = np.array([36.60, 12.24,  4.12])
h200_E_wi_amr = np.array([32.98, 11.65,  3.40])
h200_dofs     = np.array([1977,  4245,   7153])

h100_E_no_amr = np.array([8.17,  1.53,  1.16])
h100_E_wi_amr = np.array([8.83,  1.55,  1.00])
h100_dofs     = np.array([7793,  16809, 28385])


# spectral
p4_quad_E_no_amr = np.array([20.1,  30.0,  14.24, 5.19,  2.42,  1.48,  0.85]) 
p4_quad_E_wi_amr = np.array([27.38, 30.19, 12.25, 5.86,  2.1,   1.37,  0.83])
p4_quad_dofs     = np.array([697,   1425,  2673,  4141,  10465, 16281, 40125])
p4_quad_cells    = np.array([40,    84,    160,   250,   640,   1000,  2480])
p4_quad_h        = np.array([400,   286,   200,   160,   100,   80,    50])

##########################################################################
# E X DOFs
plt.plot(p2_dofs, p2_E_no_amr, label="p=2 (no AMR)",marker="o", linestyle="-", color='tab:green')
plt.plot(p2_dofs, p2_E_wi_amr, label="p=2 (w/ AMR)",marker="o", linestyle="--", color='tab:green')

plt.plot(p3_dofs, p3_E_no_amr, label="p=3 (no AMR)",marker="o", linestyle="-", color='tab:orange')
plt.plot(p3_dofs, p3_E_wi_amr, label="p=3 (w/ AMR)",marker="o", linestyle="--", color='tab:orange')

plt.plot(p4_dofs, p4_E_no_amr, label="p=4 (no AMR)",marker="o", linestyle="-", color='tab:blue')
plt.plot(p4_dofs, p4_E_wi_amr, label="p=4 (w/ AMR)",marker="o", linestyle="--", color='tab:blue')

plt.plot(p4_quad_dofs, p4_quad_E_no_amr, label="p=4 quad (no AMR)",marker="o", linestyle="-", color='tab:red')
plt.plot(p4_quad_dofs, p4_quad_E_wi_amr, label="p=4 quad (w/ AMR)",marker="o", linestyle="--", color='tab:red')

# E x DOFs (p-extension)
plt.plot(h200_dofs, h200_E_no_amr, label="h=200 m (no AMR)",marker="o", linestyle="-", color='darkgray')
plt.plot(h200_dofs, h200_E_wi_amr, label="h=200 m (w/ AMR)",marker="o", linestyle="--", color='darkgray')

#plt.plot(h100_dofs, h100_E_no_amr, label="h=100 m (no AMR)",marker="o", linestyle="-", color='lightgray')
#plt.plot(h100_dofs, h100_E_wi_amr, label="h=100 m (w/ AMR)",marker="o", linestyle="--", color='lightgray')

plt.ylabel("E (%)",fontsize=14)
plt.xlabel("DOFs", fontsize=14)
plt.yscale('log')
plt.xscale('log')
plt.title("Acoustic forward with AMR (Marmousi)", fontsize=16)
plt.legend(loc='upper right')
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor')

plt.show()

##########################################################################
# E X Nelem
plt.plot(p2_cells, p2_E_no_amr, label="p=2 (no AMR)",marker="o", linestyle="-", color='tab:green')
plt.plot(p2_cells, p2_E_wi_amr, label="p=2 (w/ AMR)",marker="o", linestyle="--", color='tab:green')

plt.plot(p3_cells, p3_E_no_amr, label="p=3 (no AMR)",marker="o", linestyle="-", color='tab:orange')
plt.plot(p3_cells, p3_E_wi_amr, label="p=3 (w/ AMR)",marker="o", linestyle="--", color='tab:orange')

plt.plot(p4_cells, p4_E_no_amr, label="p=4 (no AMR)",marker="o", linestyle="-", color='tab:blue')
plt.plot(p4_cells, p4_E_wi_amr, label="p=4 (w/ AMR)",marker="o", linestyle="--", color='tab:blue')

plt.plot(p4_quad_cells, p4_quad_E_no_amr, label="p=4 quad (no AMR)",marker="o", linestyle="-", color='tab:red')
plt.plot(p4_quad_cells, p4_quad_E_wi_amr, label="p=4 quad (w/ AMR)",marker="o", linestyle="--", color='tab:red')

plt.ylabel("E (%)",fontsize=14)
plt.xlabel("Nelem", fontsize=14)
plt.yscale('log')
plt.xscale('log')
plt.title("Acoustic forward with AMR (Marmousi)", fontsize=16)
plt.legend(loc='upper right')
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor')

plt.show()


