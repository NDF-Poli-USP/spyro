import matplotlib.pyplot as plt
import matplotlib

plt.rcParams["font.family"] = "Times New Roman"

cmap0 = matplotlib.cm.get_cmap("Oranges")
cmap1 = matplotlib.cm.get_cmap("Blues")
cmap2 = matplotlib.cm.get_cmap("Reds")
cmap3 = matplotlib.cm.get_cmap("Purples")
cmap4 = matplotlib.cm.get_cmap("Greys")

colors_regular = [cmap0(0.2), cmap1(0.3), cmap2(0.5), cmap3(0.7), cmap4(0.9)]
colors_dark = [cmap0(0.3), cmap1(0.4), cmap2(0.6), cmap3(0.8), cmap4(1.0)]

fig, ax = plt.subplots(1, 1)
leters_size = 20
y_limits = (1, 200)

p = [2, 4, 6, 8]
cpwl = [
    [2.20, 2.42, 2.66, 2.92, 3.22, 3.54, 3.89, 4.28, 4.71, 5.18, 5.70, 6.27, 6.89, 7.00, 7.1, 7.16, 7.23, 7.5, 7.53, 7.58,],
    [1.0, 1.4, 1.8, 2.2, 2.35, 2.38, 2.41, 2.42, 2.44, 2.46],
    [1.1, 1.21, 1.33, 1.35, 1.37, 1.38, 1.40, 1.45],
    [0.8052, 0.95, 0.98, 1.0],
]


err = [
    [1.0316, 0.8882, 0.8524, 0.83925, 0.7718, 0.5821, 0.5302, 0.4514, 0.3041, 0.2197, 0.1511, 0.1069, 0.0818, 0.06521, 0.0692, 0.05456, 0.0576, 0.0552, 0.042391, 0.04098],
    [0.71395, 0.5181, 0.2106, 0.1012, 0.0670, 0.06340, 0.05691, 0.0569, 0.05513, 0.046885],
    [0.1954, 0.1399, 0.0774, 0.06115, 0.06087, 0.0575, 0.0536328, 0.040839],
    [0.1382, 0.0598, 0.05155, 0.048806],
]

cont = 0
for degree in p:
    m = cpwl[cont]

    error_percent = [i * 100 for i in err[cont]]

    ax.plot(m, error_percent, "o", label="SEM" + str(degree)+"quad", color=colors_dark[cont])
    cont += 1


ax.plot([0.5, 8.0], [5, 5], 'k--')
# ax .set(xlabel = "Grid-points-per-wavelength (G)", ylabel = "E %")
ax.set_title("Error with varying C")
ax.set(xlabel="Cells-per-wavelength (C)", ylabel="E %")
ax.set_yticks([3, 5, 10, 30, 100])
ax.set_xticks([1, 2, 4, 6, 8])
ax.set_xlim((0.0, 8.5))
ax.set_ylim(y_limits)
ax.set_yscale("log")
ax.legend()
ax.set_yticks([3, 5, 10, 30, 100])
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax.text(-0.1, 1.0, '(b)', transform=ax .transAxes, size=leters_size)
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
fig.set_size_inches(9, 5)
plt.show()

# cont = 0
# p = [1, 2, 3, 4, 5]
# for degree in p:
#     g = copy.deepcopy(gpwl[cont])
#     for ite in range(len(gpwl[cont])):
#         g[ite] = old_to_new_g_converter("KMV", degree, gpwl[cont][ite])

#     error_percent = [i * 100 for i in err[cont]]

#     plt.plot(g, error_percent, "o-", label="KMV" + str(degree) + "tri")
#     cont += 1

# plt.plot([3.0, 13.0], [5, 5], "k--")
# plt.xlabel("Grid-points-per-wavelength (G)")
# plt.ylabel("E %")
# plt.title("Error with varying G")
# plt.legend(loc="lower left")
# plt.yscale("log")
# plt.yticks([1, 5, 10, 100])
# ax = plt.gca()
# ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
# plt.show()

# dp = 2
# print("for p=" + str(dp))
# gorigin = 11.7
# gnew = old_to_new_g_converter("KMV", dp, gorigin)
# print(gnew)

# dp = 3
# print("for p=" + str(dp))
# gorigin = 10.5
# gnew = old_to_new_g_converter("KMV", dp, gorigin)
# print(gnew)

# dp = 4
# print("for p=" + str(dp))
# gorigin = 10.5
# gnew = old_to_new_g_converter("KMV", dp, gorigin)
# print(gnew)

# dp = 5
# print("for p=" + str(dp))
# gorigin = 8.4
# gnew = old_to_new_g_converter("KMV", dp, gorigin)
# print(gnew)
