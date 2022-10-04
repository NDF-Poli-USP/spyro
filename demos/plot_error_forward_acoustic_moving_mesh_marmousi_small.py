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

p2_E_no_amr = np.array([48.31, 36.39, 14.99, 7.85,  2.26]) # compared with p=4
p2_E_wi_amr = np.array([52.05, 37.01, 14.11, 6.77,  1.65]) # compared with p=4
p2_dofs     = np.array([1977,  3071,  7793,  12141, 29983])
p2_cells    = np.array([640,   1000,  2560,  4000,  9920])
p2_h        = np.array([200,   160,   100,   80,    50])

p3_E_no_amr = np.array([43.27, 22.53, 11.79, 3.46,  2.85,  0.86]) # compared to p=4 and p=5
p3_E_wi_amr = np.array([43.41, 20.42, 10.54, 2.56,  1.7,   0.88]) # compared to p=4 and p=5 
p3_dofs     = np.array([2245,  4245,  6606,  16809, 26211, 64814])
p3_cells    = np.array([336,   640,   1000,  2560,  4000,  9920])
p3_h        = np.array([286,   200,   160,   100,   80,    50])

p4_E_no_amr = np.array([47.05, 21.16, 8.1,  6.4, 1.99,  1.22]) # compared to p=4 
p4_E_wi_amr = np.array([46.47, 21.81, 7.35, 4.02, 1.61,  1.03]) 
p4_dofs     = np.array([1817,  3777,  7153, 11141, 28385, 44281])
p4_cells    = np.array([160,   336,   640,  1000, 2560,  4000])
p4_h        = np.array([400,   286,   200,  160, 100,   80])

#running with nx = 14  # nx=14  => dx = dz = 285.71 m

# recomputed
p4_quad_E_no_amr = np.array([41.69, 22.53, 13.14, 4.67,  2.41,  1.69]) 
p4_quad_E_wi_amr = np.array([43.95, 21.56, 11.72, 3.59,  1.98,  1.54])
p4_quad_dofs     = np.array([1425,  2673,  4141,  10465, 16281, 40125])
p4_quad_cells    = np.array([84,    160,   250,   640,   1000,  2480])
p4_quad_h        = np.array([286,   200,   160,   100,   80,    50])

plt.plot(p2_dofs, p2_E_no_amr, label="p=2 (no AMR)",marker="o", linestyle="-", color='tab:green')
plt.plot(p2_dofs, p2_E_wi_amr, label="p=2 (w/ AMR)",marker="o", linestyle="--", color='tab:green')

plt.plot(p3_dofs, p3_E_no_amr, label="p=3 (no AMR)",marker="o", linestyle="-", color='tab:orange')
plt.plot(p3_dofs, p3_E_wi_amr, label="p=3 (w/ AMR)",marker="o", linestyle="--", color='tab:orange')

plt.plot(p4_dofs, p4_E_no_amr, label="p=4 (no AMR)",marker="o", linestyle="-", color='tab:blue')
plt.plot(p4_dofs, p4_E_wi_amr, label="p=4 (w/ AMR)",marker="o", linestyle="--", color='tab:blue')

plt.plot(p4_quad_dofs, p4_quad_E_no_amr, label="p=4 quad (no AMR)",marker="o", linestyle="-", color='tab:red')
plt.plot(p4_quad_dofs, p4_quad_E_wi_amr, label="p=4 quad (w/ AMR)",marker="o", linestyle="--", color='tab:red')

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
