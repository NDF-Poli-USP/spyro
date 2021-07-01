import matplotlib.pyplot as plt
import matplotlib
import copy

plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size']   = 16

def grid_point_to_mesh_point_converter_for_seismicmesh(method, degree , G):
    if method == 'KMV':
        if degree == 1:
            M = G
        if degree == 2:
            M = 0.5*G
        if degree == 3:
            M = 0.2934695559090401*G
        if degree == 4:
            M = 0.21132486540518713*G
        if degree == 5:
            M = 0.20231237605867816*G

    if method == 'CG':
        if degree == 1:
            M = G
        if degree == 2:
            M = 0.5*G
        if degree == 3:
            M = 0.333333333333333*G
        if degree == 4:
            M = 0.25*G
        if degree == 5:
            M = 0.2*G

    if method == 'spectral':
        if degree == 1:
            M = G
        if degree == 2:
            M = 0.5*G
        if degree == 3:
            M = 0.27639320225002106*G
        if degree == 4:
            M = 0.32732683535398854*G
        if degree == 5:
            M = 0.23991190372440996*G

    return M

def old_to_new_g_converter(method, degree, Gold):
    if method == 'KMV':
        if degree == 1:
            G = 0.707813887967734*Gold
        if degree == 2:
            G = 0.8663141029672784*Gold
        if degree == 3:
            G = 0.7483761673104953*Gold
        if degree == 4:
            G = 0.7010127254535244*Gold
        if degree == 5:
            G = 0.9381929803311276*Gold

    return G


# heterogeneous and KMV 
# For p of 1
# 	G		Error 
# 	8		1.2797546602482666 
# 	9		1.2901729000446824 
# 	10		1.2498834332645652 
# 	11		1.2172132027385325 
# 	12		1.1613843203491774 
# For p of 2
# 	G		Error 
# 	8		0.36577520510863876 
# 	9		0.25058624713364136 
# 	10		0.16307536942425765 
# 	11		0.10889470303672913 
# 	12		0.07591378759966216 
# For p of 3
# 	G		Error 
# 	8		0.21989411648458276 
# 	9		0.12411093437021095 
# 	10		0.11323136685893356 
# 	11		0.06059389910151199 
# 	12		0.05150149054386856 
# For p of 4
# 	G		Error 
# 	8		0.20271323332978108 
# 	9		0.14725676038956462 
# 	10		0.07547072137668284 
# 	11		0.06171113085811123 
# 	12		0.0358982589253432 

fig, ax = plt.subplots(2,2)
leters_size = 36

## Homogeneous
gpwl_homogeneous = [
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],#, 13],# 14, 15, 16, 17],
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [3, 4, 5, 6, 7, 8, 9, 10, 11],
    [3, 4, 5, 6, 7, 8, 9, 10, 11],
    [4, 5, 6, 7, 8, 9]
]

err_homogeneous = [
    [1.1648811814215285,  1.462880070429223,   1.3187042521844574,  1.4474382039628124,   1.2487413308791995,   1.24531593991547,    1.0578457988047862 ,  0.9861895930782795 ,  0.8143390891456412 , 0.7169644686997928],# , 0.5682106142990109],# , 0.4692416333337447 , 0.35153413401259487 , 0.2664619530345574 , 0.18013170223717548   ],
    [1.022599700382224,   1.2398833269021887,  0.7183083467603064,  0.5711340959374986,   0.45672494105083966,  0.27858824134793325, 0.15225356339066667 , 0.10136053021693772 , 0.07167966029261491, 0.030036466329772582     ],
    [0.9827437584395736,  0.7986689626562492,  0.5684974828465463,  0.4169565650748226,   0.2758150075157688,   0.16770053909795496, 0.09810480401532358 , 0.06965956764786514,  0.048635539022396194     ],
	[0.8558802863265129,  0.7364316605977338,  0.5383343316446062,  0.39695542059556965,  0.26693788113662026,  0.20854480331343098, 0.11845380627809417 , 0.05468725536899697,  0.040122777468893124     ],
    [0.6476193161471047,  0.40112260136703387, 0.24901597141013956, 0.12159676205077578,  0.06222608484075993,  0.02903638466877757     ]
]

cont = 0
p = [1, 2, 3, 4, 5]
for degree in p:
    g = copy.deepcopy(gpwl_homogeneous[cont])
    for ite in range(len(gpwl_homogeneous[cont])):
        g[ite]= old_to_new_g_converter('KMV', degree, gpwl_homogeneous[cont][ite])

    error_percent = [i*100 for i in err_homogeneous[cont]]

    ax[0,1].plot(g, error_percent, "o-", label="KMV" + str(degree)+"tri")
    cont +=1

ax[0,1].plot([3.0,13.0],[5, 5], 'k--')
ax[0,1].set(xlabel = "Grid-points-per-wavelength (G)", ylabel = "E %")
ax[0,1].set_title("Error with varying G\n\n\n")
ax[0,1].set_xticks([2, 4, 6, 8, 10, 12])
ax[0,1].set_xlim((1.5, 12.5))

ax[0,1].legend(loc='lower left')
ax[0,1].set_yscale("log")
ax[0,1].set_yticks([3, 5, 10, 100])
ax[0,1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[0,1].text(-0.1, 1.1, '(b)', transform=ax[0,1].transAxes, 
            size=leters_size)

cont = 0
p = [1, 2, 3, 4, 5]
for degree in p:
    m = copy.deepcopy(gpwl_homogeneous[cont])
    for ite in range(len(gpwl_homogeneous[cont])):
        m[ite]= grid_point_to_mesh_point_converter_for_seismicmesh('KMV', degree ,gpwl_homogeneous[cont][ite])

    error_percent = [i*100 for i in err_homogeneous[cont]]

    ax[0,0].plot(m, error_percent, "o-", label="KMV" + str(degree)+"tri")
    cont +=1

ax[0,0].plot([1.0,12.0],[5, 5], 'k--')

ax[0,0].set(xlabel = "Cells-per-wavelength (C)", ylabel = "E %")
ax[0,0].set_title("Error with varying C\n\n\n")
ax[0,0].legend(loc='lower right')
ax[0,0].set_yscale("log")
ax[0,0].set_yticks([3, 5, 10, 100])
ax[0,0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[0,0].text(-0.1, 1.1, '(a)', transform=ax[0,0].transAxes, 
            size=leters_size)


## Heterogeneous
p = [1, 2, 3, 4, 5]
gpwl_heterogeneous = [
    [8, 9, 10, 11, 12],
    [8, 9, 10, 11, 12, 12.5, 13, 13.5, 14],
    [8, 9, 10, 11, 12, 12.5],
    [8, 9, 10, 11, 12],
    [8, 8.5, 9, 9.5]
]

err_heterogeneous = [
    [1.2797546602482666 , 1.2901729000446824  , 1.2498834332645652  , 1.2172132027385325  , 1.1613843203491774],
    [0.36577520510863876, 0.25058624713364136 , 0.16307536942425765 , 0.10889470303672913 , 0.07591378759966216, 0.06466627849630871, 0.058196710034454174, 0.051600252120609674, 0.04491917551712101],
    [0.21989411648458276, 0.12411093437021095 , 0.11323136685893356 , 0.06059389910151199 , 0.05150149054386856, 0.04087487684809255],
	[0.20271323332978108, 0.14725676038956462 , 0.07547072137668284 , 0.06171113085811123 , 0.0358982589253432],
    [0.10704348273076325, 0.08267927690965501 , 0.05759702141756072 , 0.044996842569362545]
]

cont = 0
p = [1, 2, 3, 4, 5]
for degree in p:
    g = copy.deepcopy(gpwl_heterogeneous[cont])
    for ite in range(len(gpwl_heterogeneous[cont])):
        g[ite]= old_to_new_g_converter('KMV', degree, gpwl_heterogeneous[cont][ite])

    error_percent = [i*100 for i in err_heterogeneous[cont]]

    ax[1,1].plot(g, error_percent, "o-", label="KMV" + str(degree)+"tri")
    cont +=1

ax[1,1].plot([3.0,13.0],[5, 5], 'k--')
ax[1,1].set(xlabel = "Grid-points-per-wavelength (G)", ylabel = "E %")
#ax[1,1].set_title("Error with varying G")

ax[1,1].legend(loc='lower left')
ax[1,1].set_xticks([2, 4, 6, 8, 10, 12])
ax[1,1].set_xlim((1.5, 12.5))
ax[1,1].set_yscale("log")
ax[1,1].set_yticks([3, 5, 10, 100])
ax[1,1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1,1].text(-0.1, 1.1, '(d)', transform=ax[1,1].transAxes, 
            size=leters_size)

cont = 0
p = [1, 2, 3, 4, 5]
for degree in p:
    m = copy.deepcopy(gpwl_heterogeneous[cont])
    for ite in range(len(gpwl_heterogeneous[cont])):
        m[ite]= grid_point_to_mesh_point_converter_for_seismicmesh('KMV', degree ,gpwl_heterogeneous[cont][ite])

    error_percent = [i*100 for i in err_heterogeneous[cont]]

    ax[1,0].plot(m, error_percent, "o-", label="KMV" + str(degree)+"tri")
    cont +=1

ax[1,0].plot([1.0,12.0],[5, 5], 'k--')

ax[1,0].set(xlabel = "Cells-per-wavelength (C)", ylabel = "E %")
#ax[1,0].set_title("Error with varying C")
ax[1,0].legend(loc='lower right')
ax[1,0].set_yscale("log")
ax[1,0].set_yticks([3, 5, 10, 100])
ax[1,0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1,0].text(-0.1, 1.1, '(c)', transform=ax[1,0].transAxes, 
            size=leters_size)
fig.set_size_inches(13,10)

rows = ['Homogeneous   ', 'Heterogeneous   ']
pad = 1
for a, row in zip(ax[:,0], rows):
    a.annotate(row, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - pad, 0),
                xycoords=a.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
fig.tight_layout(pad=pad)
plt.show()