import numpy as np
import matplotlib.pyplot as plt
# Use latex fonts
plt.rcParams.update({"font.family": "serif"})
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \usepackage{amsmath}'

#
FL = np.linspace(0.001, 1, int(1e3))

#
m = 1

#
eps = 0.999

#
c = 1.5

#
L = 1.2

#
fs = np.array([2.25, 5])

#
label = ['2.25', '5.00']

#
color = ['C0', 'C1']

#
R_list = []

#
C_list = []


for i, f in enumerate(fs):

    #
    a = c / (L * f)

    #
    A = 1 + (1 / 8) * (eps * m * a / FL)**2

    #
    R = np.abs(np.exp(
        -eps * m * np.pi * A) * (np.cos(m * np.pi * A * (1 - eps**2)**0.5)
                                 + (eps / (1 - eps**2)**0.5) * np.sin(
            m * np.pi * A * (1 - eps**2)**0.5)) * np.cos(2 * np.pi * FL/a * A))

    # Append R to list
    R_list.append(R)

for i, f in enumerate(fs):

    #
    a = c / (L * f)

    #
    C = np.abs(eps**2/(eps**2 + (4*FL/(m*a))**2))

    # Append C to list
    C_list.append(C)

plt.figure(figsize=(12, 6))


# Plot the difference
for i, k in enumerate(C_list):
    plt.plot(FL, np.abs(k) - R_list[i], color=color[i],
             label=r'$\Psi_{{F_L}}(f={}\text{{Hz}})$'.format(label[i]))


FL_str = r'$F_L \; (L_{{\xi}} \; = \; L_{{ref}} \, F_L \;$'
lref_str = r'$\therefore \; L_{{ref}} \; = \; {:.1f}\text{{km}})$'
plt.xlabel((FL_str + lref_str).format(L))
plt.ylabel(r'$\Psi_{{F_L}} \; = \; |C_{Rmin}| \; - \; R$')

plt.text(x=0.44, y=-0.07, s='0.4259', horizontalalignment='center',
         verticalalignment='bottom',
         bbox=dict(facecolor=color[0], alpha=0.8))

plt.text(x=0.59, y=-0.07, s='0.5959', horizontalalignment='center',
         verticalalignment='bottom',
         bbox=dict(facecolor=color[0], alpha=0.8))

plt.text(x=0.66, y=-0.07, s='0.6624', horizontalalignment='center',
         verticalalignment='bottom',
         bbox=dict(facecolor=color[0], alpha=0.8))

plt.text(x=0.89, y=-0.07, s='0.9197', horizontalalignment='center',
         verticalalignment='bottom',
         bbox=dict(facecolor=color[0], alpha=0.8))

plt.text(x=0.96, y=-0.07, s='0.9431', horizontalalignment='center',
         verticalalignment='bottom',
         bbox=dict(facecolor=color[0], alpha=0.8))

plt.text(x=0.175, y=0.035, s='0.1917', horizontalalignment='center',
         verticalalignment='bottom',
         bbox=dict(facecolor=color[1], alpha=0.8))

plt.text(x=0.245, y=0.035, s='0.2682', horizontalalignment='center',
         verticalalignment='bottom',
         bbox=dict(facecolor=color[1], alpha=0.8))

plt.text(x=0.315, y=0.035, s='0.2981', horizontalalignment='center',
         verticalalignment='bottom',
         bbox=dict(facecolor=color[1], alpha=0.8))

plt.text(x=0.385, y=0.035, s='0.4130', horizontalalignment='center',
         verticalalignment='bottom',
         bbox=dict(facecolor=color[1], alpha=0.8))

plt.text(x=0.455, y=0.035, s='0.4244', horizontalalignment='center',
         verticalalignment='bottom',
         bbox=dict(facecolor=color[1], alpha=0.8))

plt.scatter([0.4259, 0.5959, 0.6624, 0.9179, 0.9431], np.zeros(5), color=color[0])
plt.scatter([0.1917, 0.2682, 0.2981, 0.4130, 0.4244], np.zeros(5), color=color[1])

plt.xticks(np.arange(0, 1.05, 0.05))
plt.xlim((-0.01, 1.01))
plt.ylim((-0.18, 1.02))
plt.grid()
plt.legend()
plt.savefig('benchmark_lay.pdf', bbox_inches='tight')
# plt.show()
plt.close()
