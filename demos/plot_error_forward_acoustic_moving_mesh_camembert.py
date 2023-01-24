import matplotlib.pyplot as plt
import numpy as np

# triangle P4
# vp in CG
# rec1
p4_E_no_amr_rec1_vpCG = np.array([0.5999,  0.4178,  0.1623,  0.2187,   0.0382]) 
p4_E_wi_amr_rec1_vpCG = np.array([0.4772,  0.2182,  0.1256,  0.1186,   0.0423])  
# rec2
p4_E_no_amr_rec2_vpCG = np.array([1.3912,  0.7667,  0.3049,  0.2567,   0.0619]) 
p4_E_wi_amr_rec2_vpCG = np.array([1.0768,  0.4761,  0.2991,  0.4211,   0.0626])  
# rec_total
p4_E_no_amr_rect_vpCG = np.array([0.7465,  0.4776,  0.1852,  0.2219,   0.0418])  
p4_E_wi_amr_rect_vpCG = np.array([0.5884,  0.2698,  0.1606,  0.1911,   0.045])  

# vp in DG
# rec1
p4_E_no_amr_rec1_vpDG = np.array([2.6805,  2.3739,  1.0639,  0.9471,   0.4448]) 
p4_E_wi_amr_rec1_vpDG = np.array([1.8442,  1.1562,  0.306,   0.2878,   0.2214])  
# rec2
p4_E_no_amr_rec2_vpDG = np.array([5.1673,  6.5356,  1.722,   2.2863,   0.8635]) 
p4_E_wi_amr_rec2_vpDG = np.array([2.3106,  2.1165,  0.7919,  0.5057,   0.5577])  
# rec_total
p4_E_no_amr_rect_vpDG = np.array([3.087,  3.2444,  1.1514,   1.2419,   0.5181])  
p4_E_wi_amr_rect_vpDG = np.array([1.8849,  1.3186,  0.4088,  0.3213,   0.2922])  

# dofs
p4_dofs               = np.array([17761,  27701, 70721, 110401,  282241])
p4_cells              = np.array([1600,   2500,  6400,  10000,   25600])
p4_h                  = np.array([50,     40,    25,    20,      12.5])


# spectral/quadrilateral P4
# vp in CG
# rec1
p4_quad_E_no_amr_rec1_vpCG = np.array([1.0527, 0.8383,  0.2278,  0.4905,  0.0784]) 
p4_quad_E_wi_amr_rec1_vpCG = np.array([1.0018, 0.5159,  0.1824,  0.1382,  0.0976])
# rec2
p4_quad_E_no_amr_rec2_vpCG = np.array([2.4916, 0.9843,  0.3967,  0.9508,  0.1464]) 
p4_quad_E_wi_amr_rec2_vpCG = np.array([2.3516, 0.8697,  0.4744,  0.1377,  0.2204])
# rec_total
p4_quad_E_no_amr_rect_vpCG = np.array([1.3377, 0.8494,  0.2515,  0.58,  0.0918]) 
p4_quad_E_wi_amr_rect_vpCG = np.array([1.2563, 0.5721,  0.2437,  0.1357,  0.1203])

# vp in DG
# rec1
p4_quad_E_no_amr_rec1_vpDG = np.array([4.8806, 3.9881,  1.8706,  1.9858,  0.7713]) 
p4_quad_E_wi_amr_rec1_vpDG = np.array([3.2351, 2.212,  1.0252,  0.6297,  0.4516])
# rec2
p4_quad_E_no_amr_rec2_vpDG = np.array([6.5555, 8.8878,  2.8348,  4.3623,  1.7444]) 
p4_quad_E_wi_amr_rec2_vpDG = np.array([6.2538, 2.8619,  1.7049,  1.1712,  0.9251])
# rec_total
p4_quad_E_no_amr_rect_vpDG = np.array([5.1751, 4.8467,  2.018,  2.476,  0.9676]) 
p4_quad_E_wi_amr_rect_vpDG = np.array([3.7319, 2.2871,  1.1308,  0.7253,  0.5406])

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
plt.title("Acoustic forward with AMR (Camembert)", fontsize=16)
plt.legend(loc='upper right')
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor')

plt.show()

