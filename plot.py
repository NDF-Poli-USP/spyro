import matplotlib.pyplot      as plt
import numpy as np


fwd_tape_t = np.array([124, 100, 91, 143])
grad_aut = np.array([426, 364, 368, 403])
grad_impl = np.array([196])
p_order = np.array([2, 3, 4, 5])
fwi_aut_mem = np.array([])
fwi_impl_mem = np.array([])

# plt.plot(extension,polyfit(damp), '--k')
# plt.plot(extension, habc, 'ok', mec='b', ms=9, mew=3, lw=2,label=r'\huge{HABC-A1}')
# plt.plot(extension,polyfit(habc), '--k')
# plt.plot(extension, higdon, '*k', mec='g', ms=9, mew=3, lw=2,label=r'\huge{HABC-Higdon}')
# plt.plot(extension,polyfit(higdon), '--k')
# plt.plot(extension, pml, '^k', mec='c', ms=9, mew=3, lw=2,label=r'\huge{PML}')
# plt.plot(extension,polyfit(pml), '--k')
# plt.plot(extension, cpml, 'Hk', mec='r', ms=9, mew=3, lw=2,label=r'\huge{CPML}')
# plt.plot(extension,polyfit(cpml), '--k',label=r'\huge{Fit curve}')

