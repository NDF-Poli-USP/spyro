import matplotlib.pyplot as plt
import numpy as np

# Reference model (structured mesh):
# Freq=6Hz (peak)
# h=20m
# p=4
# DOF=XX FIXME check it
# Nelem=XX FIXME check it

# Models with structured meshes
# Errors computed on Rec1 (close to sources line)
# monitor function: max(M1,M2), beta1=beta2=0.5
# smoothing factor = 0.003, nt = 1

# with 4 shots (keeping it for a while)
#p2_E_no_amr = np.array([164.1,  101.24,  59.78,      45.9]) 
#p2_E_wi_amr = np.array([286.87,  83.39,  76.93,     39.92]) 
#p2_dofs     = np.array([12141,   29983,  47079,     83363]) 
#p2_cells    = np.array([4000,     9920,  15600,     27664]) 
#p2_h        = np.array([80,         50,     40,   30.0752])

# REC1
p2_E_no_amr = np.array([16.06,   16.53,    14.37,   9.42,     5.99,   4.24]) 
p2_E_wi_amr = np.array([17.99,   15.04,    10.48,   7.14,     4.53,   3.56]) 
p2_dofs     = np.array([7793,    12141,    17449,  29983,    47079,  61879]) 
p2_cells    = np.array([2560,     4000,     5760,   9920,    15600,  20520]) 
p2_h        = np.array([100,        80,   66.667,     50,       40, 35.088])

p3_E_no_amr = np.array([11.68, 9.88,    7.25,    5.66,    3.75])
p3_E_wi_amr = np.array([10.95, 8.95,    6.49,     4.9,     2.8]) 
p3_dofs     = np.array([10952, 16809,  26211,   37693,   64814])
p3_cells    = np.array([1664,  2560,    4000,    5760,    9920])
p3_h        = np.array([125,   100,       80,  66.667,      50])

p4_E_no_amr = np.array([11.35, 7.00,     4.3,      3.53])
p4_E_wi_amr = np.array([11.4,  5.64,    4.02,      2.69]) 
p4_dofs     = np.array([11141, 28385,  44281,     63697])
p4_cells    = np.array([1000,  2560,    4000,      5760])
p4_h        = np.array([160,   100,       80,    66.667])

#p4_quad_E_no_amr = np.array([ ]) 
#p4_quad_E_wi_amr = np.array([ ])
#p4_quad_dofs     = np.array([ ])
#p4_quad_cells    = np.array([ ])
#p4_quad_h        = np.array([ ])

# for p-extension (p2, p3, p4)
h67_E_no_amr = np.array([14.37,   5.66,     3.53])
h67_E_wi_amr = np.array([10.48,    4.9,     2.69])
h67_dofs     = np.array([17449,  37693,    63697])

h80_E_no_amr = np.array([16.53,   7.25,     4.3])
h80_E_wi_amr = np.array([15.04,   6.49,    4.02])
h80_dofs     = np.array([12141,  26211,   44281])

h100_E_no_amr = np.array([16.06,   9.88,    7.00])
h100_E_wi_amr = np.array([17.99,   8.95,    5.64])
h100_dofs     = np.array([7793,   16809,   28385])


plt.plot(p2_dofs, p2_E_no_amr, label="p=2 (no AMR)",marker="o", linestyle="-", color='tab:green')
plt.plot(p2_dofs, p2_E_wi_amr, label="p=2 (w/ AMR)",marker="o", linestyle="--", color='tab:green')

plt.plot(p3_dofs, p3_E_no_amr, label="p=3 (no AMR)",marker="o", linestyle="-", color='tab:orange')
plt.plot(p3_dofs, p3_E_wi_amr, label="p=3 (w/ AMR)",marker="o", linestyle="--", color='tab:orange')

plt.plot(p4_dofs, p4_E_no_amr, label="p=4 (no AMR)",marker="o", linestyle="-", color='tab:blue')
plt.plot(p4_dofs, p4_E_wi_amr, label="p=4 (w/ AMR)",marker="o", linestyle="--", color='tab:blue')

#plt.plot(p4_quad_dofs, p4_quad_E_no_amr, label="p=4 quad (no AMR)",marker="o", linestyle="-", color='tab:red')
#plt.plot(p4_quad_dofs, p4_quad_E_wi_amr, label="p=4 quad (w/ AMR)",marker="o", linestyle="--", color='tab:red')

# E x DOFs (p-extension)
plt.plot(h67_dofs, h67_E_no_amr, label="h=67 m (no AMR)",marker="o", linestyle="-", color='gray')
plt.plot(h67_dofs, h67_E_wi_amr, label="h=67 m (w/ AMR)",marker="o", linestyle="--", color='gray')

#plt.plot(h80_dofs, h80_E_no_amr, label="h=80 m (no AMR)",marker="o", linestyle="-", color='darkgray')
#plt.plot(h80_dofs, h80_E_wi_amr, label="h=80 m (w/ AMR)",marker="o", linestyle="--", color='darkgray')

#plt.plot(h100_dofs, h100_E_no_amr, label="h=100 m (no AMR)",marker="o", linestyle="-", color='lightgray')
#plt.plot(h100_dofs, h100_E_wi_amr, label="h=100 m (w/ AMR)",marker="o", linestyle="--", color='lightgray')

# E X DOFs
plt.ylabel("E (%)",fontsize=14)
plt.xlabel("DOFs", fontsize=14)
plt.yscale('log')
plt.xscale('log')
plt.title("Elastic forward with AMR (Marmousi)", fontsize=16)
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

#plt.plot(p4_quad_cells, p4_quad_E_no_amr, label="p=4 quad (no AMR)",marker="o", linestyle="-", color='tab:red')
#plt.plot(p4_quad_cells, p4_quad_E_wi_amr, label="p=4 quad (w/ AMR)",marker="o", linestyle="--", color='tab:red')

plt.ylabel("E (%)",fontsize=14)
plt.xlabel("Nelem", fontsize=14)
plt.yscale('log')
plt.xscale('log')
plt.title("Elastic forward with AMR (Marmousi)", fontsize=16)
plt.legend(loc='upper right')
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor')

plt.show()
