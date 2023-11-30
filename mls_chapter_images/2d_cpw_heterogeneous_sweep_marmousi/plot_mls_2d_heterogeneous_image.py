import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib
import copy


def line_it(cs, error, ax, color, op=-1):
    coeffs = np.polyfit(np.array(cs), np.log(np.array(error)), 1)
    x = [cs[0], cs[-1]]
    y = [0.0, 0.0]
    y[0] = coeffs[0]*x[0]+coeffs[1]
    y[1] = coeffs[0]*x[1]+coeffs[1]
    # ax.plot(x,np.exp(y),'--', color = color)
    if op == -1:
        text_location = (cs[-1]-0.4, error[-1]*0.5)
    elif op == 0:
        text_location = (cs[-1]-0.85, error[-1]*0.6)
    elif op == 1:
        text_location = (cs[-1]-0.4, error[-1]*0.6)
    elif op == 2:
        text_location = (cs[-1]-0.4, error[-1]*0.6)
    elif op == 3:
        text_location = (cs[-1]-0.3, error[-1]*0.5)
    elif op == 4:
        text_location = (cs[-1]-0.8, error[-1]*0.5)
    #ax.annotate(str(round(coeffs[0],2)), xy = text_location, color = color 


def get_minumum_cpw_indice(cpws, errs):
    errs_array = np.array(errs)
    cpws_array = np.array(cpws)
    acceptable_indices = np.where(errs_array < 0.05)
    found_any = np.any(acceptable_indices)
    if found_any:
        acceptable_cpws = cpws_array[np.where(errs_array < 0.05)]
        min_cpw = np.min(acceptable_cpws)

        indice_array = np.where(cpws_array == min_cpw)

        min_indice = indice_array[0][0]
        return min_indice
    else:
        return np.nan


# Load the pickle file
with open('ml_results.pkl', 'rb') as f:
    data = pickle.load(f)

ml2t = data["ml2t"]
ml3t = data["ml3t"]
ml4t = data["ml4t"]
ml5t = data["ml5t"]
ml6t = data["ml6t"]

plt.rcParams["font.family"] = "Times New Roman"

cmap0 = matplotlib.cm.get_cmap("Oranges")
cmap1 = matplotlib.cm.get_cmap("Blues")
cmap2 = matplotlib.cm.get_cmap("Reds")
cmap3 = matplotlib.cm.get_cmap("Purples")
cmap4 = matplotlib.cm.get_cmap("Greys")
cmap5 = matplotlib.cm.get_cmap("Greens")

colors_regular = [cmap0(0.2), cmap1(0.3), cmap2(0.5), cmap3(0.7), cmap4(0.9), cmap5(1.0)]
colors_dark = [cmap0(0.3), cmap1(0.4), cmap2(0.6), cmap3(0.8), cmap4(1.0), cmap5(1.0)]

fig, ax = plt.subplots(1, 2)
leters_size = 20
y_limits = (1, 200)
alphas = [1.7326282059345566, 2.5500981353665586, 3.31722783359841, 4.637348434180895, 5.385721910756255]

p = [2, 3, 4, 5, 6]
cpwl = [
    ml2t['c'].tolist(),
    ml3t['c'].tolist(),
    ml4t['c'].tolist(),
    ml5t['c'].tolist(),
    ml6t['c'].tolist(),
]

err = [
    ml2t['error'].tolist(),
    ml3t['error'].tolist(),
    ml4t['error'].tolist(),
    ml5t['error'].tolist(),
    ml6t['error'].tolist(),
]

dts = [
    ml2t['dt'].tolist(),
    ml3t['dt'].tolist(),
    ml4t['dt'].tolist(),
    ml5t['dt'].tolist(),
    ml6t['dt'].tolist(),
]

runtimes = [
    ml2t['runtime'].tolist(),
    ml3t['runtime'].tolist(),
    ml4t['runtime'].tolist(),
    ml5t['runtime'].tolist(),
    ml6t['runtime'].tolist(),
]

cont = 0
for degree in p:
    m = cpwl[cont]

    error_percent = [i * 100 for i in err[cont]]

    ax[0].plot(m, error_percent, "o", label="MLT" + str(degree)+"tri", color=colors_dark[cont])
    cont += 1


ax[0].plot([0.5, 12.0], [5, 5], 'k--')
# ax[0] .set(xlabel = "Grid-points-per-wavelength (G)", ylabel = "E %")
ax[0].set_title("Error with varying C")
ax[0].set(xlabel="Cells-per-wavelength (C)", ylabel="E %")
ax[0].set_yticks([3, 5, 10, 30, 100])
ax[0].set_xticks([1, 2, 4, 6, 8, 10, 12])
ax[0].set_xlim((0.0, 12.5))
ax[0].set_ylim(y_limits)
ax[0].set_yscale("log")
ax[0].legend()
ax[0].set_yticks([3, 5, 10, 30, 100])
ax[0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[0].text(-0.1, 1.0, '(a)', transform=ax[0].transAxes, size=leters_size)
ax[0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

cont = 0
for degree in p:
    g = copy.deepcopy(cpwl[cont])
    for ite in range(len(cpwl[cont])):
        g[ite] = g[ite]*alphas[cont]

    error_percent = [i * 100 for i in err[cont]]

    ax[1].plot(g, error_percent, "o", label="MLT" + str(degree)+"tri", color=colors_dark[cont])
    # line_it(g, error_percent, ax[1], color=colors_regular[cont], op=cont)
    cont += 1

ax[1].plot([3.0, 13.0], [5, 5], 'k--')
ax[1].set_title("Error with varying G")
ax[1].set(xlabel="Grid-points-per-wavelength (G)", ylabel="E %")
ax[1].set_xticks([2, 4, 6, 8, 10, 12])
ax[1].set_xlim((1.5, 12.5))
ax[1].set_ylim(y_limits)

ax[1].set_yscale("log")
ax[1].set_yticks([3, 5, 10, 30, 100])
ax[1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1].text(-0.1, 1.0, '(b)', transform=ax[1].transAxes, size=leters_size)

fig.set_size_inches(9, 5)

print("Table results:")
print("Element \tminimum C \tminimum G \ttime-step (s) \t runtime (s)")
print("------------------------------------------")
for i in range(len(p)):
    p_i = p[i]
    cpw_i = cpwl[i]
    err_i = err[i]
    dts_i = dts[i]
    runtimes_i = runtimes[i]
    indice = get_minumum_cpw_indice(cpw_i, err_i)

    g = copy.deepcopy(cpw_i)
    for ite in range(len(cpw_i)):
        g[ite] = g[ite]*alphas[i]
    if np.isnan(indice):
        print("ML" + str(p_i) + "tri \t\t" + "DNF" + "\t\t" + "DNF")
    else:
        # print(f"ML{p_i}tri \t\t{cpw_i[indice]:.2f}\t\t{g[indice]:.2f}")
        print(f"ML{p_i}tri & ${cpw_i[indice]:.2f}$ & ${g[indice]:.2f}$ & ${dts_i[indice]:.2e}$ & ${runtimes_i[indice]:.1f}$ \\\\")

fig.savefig("mls_cpw_results_heterogeneous_mltriangles.png")
plt.show()
