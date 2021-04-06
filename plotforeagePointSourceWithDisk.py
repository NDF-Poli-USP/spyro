import matplotlib.pyplot as plt
import copy

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
 
# Heterogeneous and KMV 
# For p of 2
# 	G		Error 
# 	8		0.1802427789996952 
# 	9		0.09997697105642048 
# 	10		0.09487529666357421 
# 	11		0.09196043299030913 
# 	12		0.03837062673141954 
# For p of 3
# 	G		Error 
# 	8		0.16811414034904365 
# 	9		0.1263102110253845 
# 	10		0.12492484599935488 
# 	11		0.098311370456542 
# 	12		0.1876845361177833 
# For p of 4
# 	G		Error 
# 	8		0.14627341712221967 
# 	9		0.10829407293647764 
# 	10		0.0647596913329896 
# 	11		0.062105469182119705 
# 	12		0.06886232422217914 
# For p of 5
# 	G		Error 
# 	8		0.07031227187250048 
# 	9		0.046131041226128426 
# 	10		0.0 
# 	11		0.0 
# 	12		0.0 
 
p = [2, 3, 4, 5]
gpwl = [8, 9, 10, 11, 12]
err = [
    [0.1802427789996952, 0.09997697105642048 , 0.09487529666357421 , 0.09196043299030913 , 0.03837062673141954 ],
	[0.16811414034904365, 0.1263102110253845 , 0.12492484599935488 , 0.098311370456542 , 0.1876845361177833  ],
	[0.14627341712221967, 0.10829407293647764 , 0.0647596913329896 , 0.062105469182119705 , 0.06886232422217914 ],
	[0.07031227187250048, 0.046131041226128426 , 0.0 , 0.0 , 0.0  ]
]
for p, e in enumerate(err):
    plt.plot(gpwl, e, "o-", label="P=" + str(p + 1))

plt.plot([8.0,12.0],[0.01, 0.01], 'k--')
plt.xlabel("grid-points-per-wavelength")
plt.ylabel("error (pressure)")
plt.title("Error with varying grid point density sigma = 500")
plt.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5
)
plt.yscale("log")
plt.show()

cont = 0
p = [2, 3, 4, 5]
for degree in p:
    m = copy.deepcopy(gpwl)
    for ite in range(len(gpwl)):
        m[ite]= grid_point_to_mesh_point_converter_for_seismicmesh('KMV', degree ,gpwl[ite])

    plt.plot(m, err[cont], "o-", label="P=" + str(degree))
    cont +=1

plt.plot([1.0,7.0],[0.01, 0.01], 'k--')

plt.xlabel("elements-per-wavelength")
plt.ylabel("error (pressure)")
plt.title("Error with varying element density")
plt.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5
)
plt.yscale("log")
plt.show()

