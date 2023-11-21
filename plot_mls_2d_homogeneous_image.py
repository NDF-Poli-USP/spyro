import matplotlib.pyplot as plt
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


plt.rcParams["font.family"] = "Times New Roman"

cmap0 = matplotlib.cm.get_cmap("Oranges")
cmap1 = matplotlib.cm.get_cmap("Blues")
cmap2 = matplotlib.cm.get_cmap("Reds")
cmap3 = matplotlib.cm.get_cmap("Purples")
cmap4 = matplotlib.cm.get_cmap("Greys")

colors_regular = [cmap0(0.2), cmap1(0.3), cmap2(0.5), cmap3(0.7), cmap4(0.9)]
colors_dark = [cmap0(0.3), cmap1(0.4), cmap2(0.6), cmap3(0.8), cmap4(1.0)]

fig, ax = plt.subplots(1, 2)
leters_size = 20
y_limits = (1, 200)
alphas = [0.707813887967734, 1.7326282059345566, 2.5500981353665586, 3.31722783359841, 4.637348434180895]

p = [1, 2, 3, 4, 5]
cpwl = [
    [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5],
    [2.2, 2.4200000000000004, 2.6620000000000004, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.05, 6.0, 5.0, 5.5, 6.0, 5.91],
    [1.0, 1.1, 1.2100000000000002, 1.3310000000000002, 1.4641000000000002, 1.61051, 1.7715610000000002, 1.9487171, 2.1435888100000002, 2.357947691, 2.5937424601, 2.8531167061100002, 2.99],
    [1.0, 1.1, 1.2100000000000002, 1.3310000000000002, 1.4641000000000002, 1.61051, 1.7715610000000002, 1.9487171, 2.1435888100000002, 2.357947691, 2.1, 2.2, 2.3000000000000003, 2.23],
    [1.0, 1.1, 1.2100000000000002, 1.3310000000000002, 1.4641000000000002, 1.61051, 1.7715610000000002, 1.6, 1.7000000000000002],
]

err = [
    [1.1196722745000114, 1.0861844627392327, 1.1658242652847741, 1.1725990450320878, 1.4610549216926918, 1.2636021531550985, 1.3162856757695631, 1.2674128684113877, 1.4476399372679631, 1.2521781800970782, 1.2584334384376774, 1.1944918660429025, 1.2694782927590846, 1.1293195136501304, 1.109266191054108, 1.0494832256025122, 1.0641217642860485, 0.9853301570626605, 0.9378566988065031, 0.8856095566165014],
    [0.822339169500742, 0.7886866875687579, 0.6908961452505153, 0.5270629117777847, 0.40984721073990144, 0.2687142737121448, 0.1783032311459387, 0.10941013530088237, 0.0693246368040273, 0.041007091172497755, 0.04451200527818815, 0.10941013530088237, 0.0693246368040273, 0.04451200527818815, 0.04799713291575475],
    [0.8415241121582129, 0.8301105394289546, 0.7967853827408525, 0.7324768440473581, 0.6874775868744354, 0.5646827685523109, 0.39462731283819136, 0.3024501087599718, 0.2798604445038005, 0.17168231222328753, 0.1102209752703036, 0.07753974748838786, 0.046298167655249634],
    [0.5960931610457801, 0.576940253506065, 0.4413778168966874, 0.3507859571386087, 0.27353628447755024, 0.20630666808065928, 0.17610246876987454, 0.1089761318406112, 0.060879683369979294, 0.035507817488064426, 0.06754651329155496, 0.05983289279287807, 0.041660852428524485, 0.04826218313246773],
    [0.4015731769058424, 0.3615186685309648, 0.25351567610438097, 0.15252501842912994, 0.0904280052582628, 0.05629167718704417, 0.03573254284202123, 0.06841592289968317, 0.04781376363965674],
]

dts = [
    [0.011494252873563218, 0.008583690987124463, 0.007633587786259542, 0.0061823802163833074, 0.005714285714285714, 0.004761904761904762, 0.0045662100456621, 0.003898635477582846, 0.0038022813688212928, 0.0032921810699588477, 0.003257328990228013, 0.0028591851322373124, 0.0028469750889679717, 0.0025173064820641915, 0.002530044275774826, 0.002257336343115124, 0.0022766078542970974, 0.002039775624681285, 0.0020693222969477496, 0.001863932898415657],
    [0.0024242424242424242, 0.002196595277320154, 0.002171552660152009, 0.001924001924001924, 0.001514004542013626, 0.0014404033129276198, 0.0011754334410813989, 0.0011510791366906475, 0.0009587727708533077, 0.0009505703422053232, 0.000958543014617781, 0.0011510791366906475, 0.0009587727708533077, 0.000958543014617781, 0.0008906702293475841],
    [0.003367003367003367, 0.00306044376434583, 0.0025559105431309905, 0.0022766078542970974, 0.0022896393817973667, 0.002073613271124935, 0.0017043033659991478, 0.0015414258188824663, 0.001404001404001404, 0.001272264631043257, 0.0011655011655011655, 0.0011651616661811825, 0.0010022550739163117],
    [0.0017338534893801473, 0.0015760441292356187, 0.001324942033786022, 0.0011788977306218685, 0.0011795930404010617, 0.0010686615014694097, 0.0008828073273008167, 0.0007992007992007992, 0.0007276696379843551, 0.0006597394029358403, 0.0007649646203863071, 0.000716974368166338, 0.0006836438215689625, 0.0007059654076950229],
    [0.0007026172492534692, 0.0006386715631486508, 0.0005368406925244934, 0.0004774979109466396, 0.0004780114722753346, 0.00043308791684711995, 0.0003575259206292456, 0.00040052067687994393, 0.00037112636852848393],
]

runtimes = [
    [1.3626694679260254, 1.7792153358459473, 1.462381362915039, 2.6182034015655518, 2.439692735671997, 3.948869466781616, 4.058746099472046, 6.229112148284912, 7.445000171661377, 9.45028829574585, 10.587958574295044, 13.026362657546997, 14.433992624282837, 18.088027477264404, 20.808316946029663, 24.2312593460083, 26.615440607070923, 31.52055025100708, 33.811195611953735, 40.94170641899109],
    [13.013147354125977, 17.139479637145996, 21.191476345062256, 39.60166788101196, 58.87876057624817, 83.76258969306946, 115.33708429336548, 148.76553988456726, 237.45892333984375, 280.88322401046753, 228.9335060119629, 127.60914063453674, 184.7234206199646, 207.07557368278503, 252.07376646995544],
    [10.431009292602539, 8.564547061920166, 17.13622236251831, 44.329190731048584, 31.125303745269775, 38.018271923065186, 34.10909557342529, 52.48997616767883, 60.012927770614624, 83.77455735206604, 118.04440140724182, 115.85980796813965, 155.62800455093384],
    [14.505264043807983, 17.703612327575684, 28.289212942123413, 38.25164842605591, 46.186219692230225, 111.35497188568115, 98.58344340324402, 116.20505619049072, 178.62511682510376, 229.5563042163849, 163.38052129745483, 163.56908702850342, 245.71752667427063, 169.17672967910767],
    [135.09528350830078, 117.91736268997192, 185.8162567615509, 252.2658712863922, 273.51905965805054, 345.56841802597046, 493.0372796058655, 331.13982915878296, 417.5062177181244],
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