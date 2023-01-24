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
# Using uniform vp = 2 km/s
# Errors computed on Rec1 (close to sources line)
# monitor function: max(M1,M2), beta1=beta2=0.5
# smoothing factor = 0.001, nt = 1
# Neumann-type source

#p2_E_no_amr = np.array([57.6574,  48.3585, 21.7051,  15.0551, 5.7402,  2.3266,  0.0938]) # REC1 
#p2_E_wi_amr = np.array([61.9695,  49.3974,  23.878,   18.5576,   8.5203,  4.1247,  0.3752]) # REC1 
p2_E_no_amr = np.array([68.2602,  45.3347,  29.4067,  16.4659,   4.9675,  1.8736,  0.3204]) # REC3
p2_E_wi_amr = np.array([68.7562,  48.687,   33.4811,  21.4072,   6.9166,  3.6597,  0.6938]) # REC3
p2_dofs     = np.array([509,      1049,     1977,     3071,      7793,    12141,   29983])
p2_cells    = np.array([160,      336,      640,      1000,      2560,    4000,    9920])
p2_h        = np.array([400,      286,      200,      160,       100,     80,      50])

#p3_E_no_amr = np.array([42.6053, 17.4426, 7.8737,  2.3134,  0.3866,  0.1097,  0.0147]) # REC1 
#p3_E_wi_amr = np.array([40.0968, 19.0065, 9.7972,  4.3621,  0.6366,  0.2681,  0.0196]) # REC1
p3_E_no_amr = np.array([45.0321, 19.1389, 7.8078,  2.6635,  0.4418,  0.1252,  0.0219]) # REC3
p3_E_wi_amr = np.array([45.5614, 22.7423,   9.8768,   4.7006,    0.8669,    0.3219,    0.048]) # REC3 
p3_dofs     = np.array([1083,  2245,    4245,    6606,    16809,   26211,   64814])
p3_cells    = np.array([160,   336,     640,     1000,    2560,    4000,    9920])
p3_h        = np.array([400,   286,     200,     160,     100,     80,      50])

#p4_E_no_amr = np.array([15.2292, 6.8914, 1.3202,  0.5791,  0.02,  0.0139,   0.0005]) # REC1 
#p4_E_wi_amr = np.array([18.8967, 7.3662, 2.2049,  0.8241,  0.0431,0.0139,   0.0008]) # REC1 
p4_E_no_amr = np.array([25.2084, 5.6282, 1.5486,  0.4274,  0.0403,  0.0104,   0.0015]) # REC3 
p4_E_wi_amr = np.array([25.4597, 7.402,  2.381,   0.9326,  0.1134,  0.0355,   0.0046]) # REC3  
p4_dofs     = np.array([1817,    3777,   7153,    11141,   28385,   44281,    109565])
p4_cells    = np.array([160,     336,    640,     1000,    2560,    4000,     9920])
p4_h        = np.array([400,     286,    200,     160,     100,     80,       50])

# for p-extension (p2, p3, p4) FIXME not updated
h200_E_no_amr = np.array([36.60, 12.24,  4.12])
h200_E_wi_amr = np.array([32.98, 11.65,  3.40])
h200_dofs     = np.array([1977,  4245,   7153])

h100_E_no_amr = np.array([8.17,  1.53,  1.16])
h100_E_wi_amr = np.array([8.83,  1.55,  1.00])
h100_dofs     = np.array([7793,  16809, 28385])


#p4_quad_E_no_amr = np.array([18.7773,   21.5936,    7.7225,   2.0825,   0.1264,   0.0577,   0.0002]) # REC1
#p4_quad_E_wi_amr = np.array([32.4204,   25.6368,    10.5422,  4.5162,   0.6629,   0.5382,   0.2342]) # REC1
p4_quad_E_no_amr = np.array([47.9162,   20.6738,    8.1959,   2.5104,   0.1272,   0.0408,   0.0035]) # REC3
p4_quad_E_wi_amr = np.array([50.1849,   23.4344,    11.2775,  6.0811,   2.5025,   1.8469,   0.7612]) # REC3
p4_quad_dofs     = np.array([697,       1425,       2673,     4141,     10465,    16281,    40125])
p4_quad_cells    = np.array([40,        84,         160,      250,      640,      1000,     2480])
p4_quad_h        = np.array([400,       286,        200,      160,      100,      80,       50])

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
#plt.plot(h200_dofs, h200_E_no_amr, label="h=200 m (no AMR)",marker="o", linestyle="-", color='darkgray')
#plt.plot(h200_dofs, h200_E_wi_amr, label="h=200 m (w/ AMR)",marker="o", linestyle="--", color='darkgray')

#plt.plot(h100_dofs, h100_E_no_amr, label="h=100 m (no AMR)",marker="o", linestyle="-", color='lightgray')
#plt.plot(h100_dofs, h100_E_wi_amr, label="h=100 m (w/ AMR)",marker="o", linestyle="--", color='lightgray')

plt.ylabel("E (%)",fontsize=14)
plt.xlabel("DOFs", fontsize=14)
plt.yscale('log')
plt.xscale('log')
plt.title("Acoustic forward with AMR (vp = 2 km/s)", fontsize=16)
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
plt.title("Acoustic forward with AMR (vp = 2 km/s)", fontsize=16)
plt.legend(loc='upper right')
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor')

plt.show()


