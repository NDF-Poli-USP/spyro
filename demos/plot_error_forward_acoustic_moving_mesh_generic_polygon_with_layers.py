import matplotlib.pyplot as plt
import numpy as np

# triangle P4
# vp in CG
# rec1
p4_E_no_amr_rec1_vpCG = np.array([1.5437,  1.6542,  0.5225,  0.3104,   0.1608]) 
p4_E_wi_amr_rec1_vpCG = np.array([0.8568,  0.5964,  0.2921,  0.2811,   0.1216])   
# rec2
p4_E_no_amr_rec2_vpCG = np.array([1.305,   1.6568,  0.5345,  0.3025,   0.2316]) 
p4_E_wi_amr_rec2_vpCG = np.array([0.9541,  0.8901,  0.4511,  0.2828,   0.1718]) 
# rec_total
p4_E_no_amr_rect_vpCG = np.array([1.5187,  1.6565,  0.5216,  0.3093,   0.1688]) 
p4_E_wi_amr_rect_vpCG = np.array([0.8634,  0.6329,  0.312,   0.2787,   0.1287]) 

# vp in DG 
# rec1
p4_E_no_amr_rec1_vpDG = np.array([3.6930,  5.9762,  2.6680,   1.3692,   0.5442])  
p4_E_wi_amr_rec1_vpDG = np.array([2.8851,  2.5231,  1.0932,   0.8041,   0.3810])  
# rec2
p4_E_no_amr_rec2_vpDG = np.array([4.2050,  8.7017,  3.3082,  1.8597,   0.7660]) 
p4_E_wi_amr_rec2_vpDG = np.array([3.9838,  3.9172,  1.5104,  1.1345,   0.5988])  
# rec_total
p4_E_no_amr_rect_vpDG = np.array([3.7362,  6.3076,  2.7158,  1.3928,   0.5634]) 
p4_E_wi_amr_rect_vpDG = np.array([3.0108,  2.7063,  1.1430,  0.8372,   0.4077])  

# dofs
p4_dofs               = np.array([17761,  27701, 70721, 110401,  282241])
p4_cells              = np.array([1600,   2500,  6400,  10000,   25600])
p4_h                  = np.array([50,     40,    25,    20,      12.5])


# spectral/quadrilateral P4
# vp in CG
# rec1
p4_quad_E_no_amr_rec1_vpCG = np.array([1.9868, 4.1972,  0.8752,  0.538,   0.3460])  
p4_quad_E_wi_amr_rec1_vpCG = np.array([2.3597, 1.5560,  0.7078,  0.4321,  0.2323]) 
# rec2
p4_quad_E_no_amr_rec2_vpCG = np.array([2.7905, 6.0543,  0.8121,  0.6277,  0.5316]) 
p4_quad_E_wi_amr_rec2_vpCG = np.array([2.9522, 2.2815,  1.1727,  0.6902,  0.3361]) 
# rec_total
p4_quad_E_no_amr_rect_vpCG = np.array([2.0751, 4.4246,  0.8753,  0.5498,  0.3701]) 
p4_quad_E_wi_amr_rect_vpCG = np.array([2.4195, 1.651,   0.7697,  0.4618,  0.2452]) 

# vp in DG
# rec1
p4_quad_E_no_amr_rec1_vpDG = np.array([8.8723,  11.6621,  2.7221,  3.1216,  1.5416]) 
p4_quad_E_wi_amr_rec1_vpDG = np.array([9.2872,  6.7284,   3.0241,  1.7919,  1.1370])
# rec2
p4_quad_E_no_amr_rec2_vpDG = np.array([7.4936,  23.8400,  4.3622,  3.5192,  1.9252]) 
p4_quad_E_wi_amr_rec2_vpDG = np.array([14.7816, 9.2492,   4.4071,  2.9608,  1.9738])
# rec_total
p4_quad_E_no_amr_rect_vpDG = np.array([8.7660,  13.3436,  2.8904,  3.1322,  1.5721]) 
p4_quad_E_wi_amr_rect_vpDG = np.array([9.8452,  7.0747,   3.2042,  1.9629,  1.2562])

# dofs
p4_quad_dofs     = np.array([6561,  10201,    25921,   40401,   103041])
p4_quad_cells    = np.array([400,   625,      1600,    2500,    6400])
p4_quad_h        = np.array([50,    40,       25,      20,      12.5])


##########################################################################
# E X DOFs
# REC1, CG/VP
plt.plot(p4_dofs, p4_E_no_amr_rec1_vpCG, label="CG/VP (no AMR)",marker="o", linestyle="-", color='tab:blue')
plt.plot(p4_dofs, p4_E_wi_amr_rec1_vpCG, label="CG/VP (w/ AMR)",marker="o", linestyle="--", color='tab:blue')

# REC1, DG/VP
plt.plot(p4_dofs, p4_E_no_amr_rec1_vpDG, label="DG/VP (no AMR)",marker="o", linestyle="-", color='tab:green')
plt.plot(p4_dofs, p4_E_wi_amr_rec1_vpDG, label="DG/VP (w/ AMR)",marker="o", linestyle="--", color='tab:green')

# REC1, CG/VP, QUAD
plt.plot(p4_quad_dofs, p4_quad_E_no_amr_rec1_vpCG, label="Quad CG/VP (no AMR)",marker="o", linestyle="-", color='tab:red')
plt.plot(p4_quad_dofs, p4_quad_E_wi_amr_rec1_vpCG, label="Quad CG/VP (w/ AMR)",marker="o", linestyle="--", color='tab:red')

# REC1, DG/VP, QUAD
plt.plot(p4_quad_dofs, p4_quad_E_no_amr_rec1_vpDG, label="Quad DG/VP (no AMR)",marker="o", linestyle="-", color='tab:orange')
plt.plot(p4_quad_dofs, p4_quad_E_wi_amr_rec1_vpDG, label="Quad DG/VP (w/ AMR)",marker="o", linestyle="--", color='tab:orange')



plt.ylabel("E (%)",fontsize=14)
plt.xlabel("DOFs", fontsize=14)
plt.yscale('log')
plt.xscale('log')
plt.title("Acoustic forward with AMR (generic salt body with layers)", fontsize=16)
plt.legend(loc='upper right')
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor')

plt.show()

