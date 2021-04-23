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
# 	8		0.1802427790345947 
# 	9		0.09997697109961355 
# 	10		0.0948752966930467 
# 	11		0.09196043303666974 
# 	12		0.038370626845302104 
# For p of 3
# 	G		Error 
# 	8		0.1681141403770824 
# 	9		0.12631021107537393 
# 	10		0.12492484605435965 
# 	11		0.09831137050467163 
# 	12		0.1876845361331776 
# For p of 4
# 	G		Error 
# 	8		0.1462734172232458 
# 	9		0.10829407310869334 
# 	10		0.06475969150230607 
# 	11		0.062105469368590537 
# 	12		0.06886232435241654 
# For p of 5
# 	G		Error 
# 	8		0.07031232975858367 
# 	9		0.04613109080919571 
# 	10		0.032471455822937444 
# 	11		0.02659394063145809 
# 	12		0.02333458190577292 
 
p = [2, 3, 4, 5]
gpwl = [8, 9, 10, 11, 12]
err = [
    [0.1802427790345947, 0.09997697109961355 , 0.0948752966930467 , 0.09196043303666974 , 0.038370626845302104 ],
	[0.1681141403770824, 0.12631021107537393 , 0.12492484605435965 , 0.09831137050467163 , 0.1876845361331776   ],
	[0.1462734172232458, 0.10829407310869334 , 0.06475969150230607 , 0.062105469368590537 , 0.06886232435241654 ],
	[0.07031232975858367, 0.04613109080919571 , 0.032471455822937444 , 0.02659394063145809 , 0.02333458190577292   ]
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

