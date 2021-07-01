import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

plt.rcParams["font.family"] = "Times New Roman"


#from mpi4py import MPI
import meshio

from SeismicMesh import (
   get_sizing_function_from_segy,
   generate_mesh,
   Rectangle,
)
import firedrake as fire
import spyro

model = spyro.tools.create_model_for_grid_point_calculation(5.0, 2, "KMV", 1.429, experiment_type = 'homogeneous', receiver_type = 'near')
comm = spyro.utils.mpi_init(model)

"""
Build a mesh of the BP2004 benchmark velocity model in serial or parallel
Takes roughly 1 minute with 2 processors and less than 1 GB of RAM.
"""

err = [
    [ 1.1648811814215285, 1.462880070429223, 1.3187042521844574, 1.4474382039628124, 1.2487413308791995, 1.24531593991547, 1.0578457988047862, 0.9861895930782795, 0.8143390891456412, 0.7169644686997928, 0.5682106142990109, 0.4692416333337447, 0.35153413401259487, 0.2664619530345574, 0.18013170223717548    ],
    [1.022599700382224, 1.2398833269021887, 0.7183083467603064, 0.5711340959374986, 0.45672494105083966, 0.27858824134793325, 0.15225356339066667 , 0.10136053021693772 , 0.07167966029261491 , 0.0461, 0.030036466329772582     ],
    [0.9827437584395736, 0.7986689626562492, 0.5684974828465463, 0.4169565650748226, 0.2758150075157688, 0.16770053909795496, 0.09810480401532358 , 0.06965956764786514 , 0.0487, 0.048635539022396194     ],
    [0.8558802863265129, 0.7364316605977338, 0.5383343316446062, 0.39695542059556965, 0.26693788113662026, 0.20854480331343098, 0.11845380627809417 , 0.05468725536899697 , 0.0424, 0.040122777468893124 ],
    [ 0.6476193161471047, 0.40112260136703387, 0.24901597141013956 , 0.12159676205077578 , 0.06222608484075993 , 0.040, 0.02903638466877757 , 0.01879546007599548 , 0.009546506182400107   ],
]


results = [
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    [1.5,2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 5.85, 6],
    [0.88041, 1.17388, 1.46735, 1.76082, 2.05429, 2.34776, 2.64123, 2.9347, 3.081435, 3.22817],
    [0.6339745962155614, 0.8452994616207485, 1.0566243270259357, 1.2679491924311228, 1.47927405783631, 1.690598923241497, 1.9019237886466842, 2.1132486540518713, 2.218911086754465, 2.3245735194570587],
    [0.8092495042347126, 1.0115618802933908, 1.213874256352069, 1.416186632410747, 1.6184990084694253, 1.6994239588928965, 1.8208113845281035, 2.0231237605867816, 2.2254361366454596],
]
#
# Name of SEG-Y file containg velocity model.
fname = "vel_z6.25m_x12.5m_exact.segy"

# Bounding box describing domain extents (corner coordinates)
bbox = (-12000.0, 0.0, 0.0, 67000.0)

rectangle = Rectangle(bbox)

freq = 5
deg = [1, 2, 3, 4, 5]
# dofs = []
# for p, result in zip(deg, results):
#    dof = []
#    print(p)
#    for g in result:
#        hmin = 1500 / (freq * g)

#        ef = get_sizing_function_from_segy(
#            fname,
#            bbox,
#            hmin=hmin,
#            wl=g,
#            freq=freq,
#            grade=0.15,
#        )
#        #print('TEST0')

#        points, cells = generate_mesh(domain=rectangle, edge_length=ef, verbose=0, mesh_improvement = False)
#        #print('TEST1')

#        meshio.write_points_cells(
#            "tmp.msh",
#            points[:, [1, 0]],
#            [("triangle", cells)],
#            file_format="gmsh22",
#            binary=False,
#        )
#        #print('TEST2')
#        mesh = fire.Mesh("tmp.msh")
#        V = fire.FunctionSpace(
#            mesh,
#            "KMV",
#            p,
#        )
#        #print('TEST3')
#        dof.append(V.dim())
#        print(f"The number of dof are: {V.dim()}")
#    dofs.append(dof)
# print(dofs)


# dofs = [
#     [24831, 43425, 67140, 96753, 131218, 170638, 215362, 265384, 320404, 381051],
#     [37877, 66721, 103421, 147015, 198649, 257947, 325631, 399595, 484299, 576639],
#     [28770, 51428, 75984, 110225, 150523, 196700, 241570, 299218, 366567, 424045],
#     [25318, 43031, 68282, 95942, 132948, 169517, 220143, 263823, 326599, 381198],
#     [46546, 80936, 123214, 174287, 232469, 300065, 380107, 489003, 586339, 689325],
# ]
# dofs = [[24831, 43425, 67140, 96753, 131218, 170638, 215362, 265384, 320404, 381051, 446771, 518337, 594376, 675651, 762168], 
# [147015, 257947, 399595, 576639, 782771, 1018657, 1286343, 1585855, 1915343, 2165815, 2278567], 
# [317876, 558018, 864708, 1248092, 1694492, 2205367, 2785134, 3433870, 3780405, 4147550], 
# [537414, 943638, 1462479, 2111112, 2866381, 3730768, 4711735, 5809429, 6395784, 7017025], 
# [1842552, 2856198, 4123534, 5599278, 7288315, 8032177, 9205196, 11350252, 13710108]]
dofs = [[24831, 43425, 67140, 96753, 131218, 170638, 215362, 265384, 320404, 381051, 446771, 518337, 594376, 675651, 762168], [37877, 66721, 103421, 147015, 198649, 257947, 325631, 399595, 484299, 548161, 576639], [28788, 51438, 75974, 110210, 150518, 196690, 241557, 299142, 331663, 366572], [25318, 43031, 68282, 95942, 132948, 169517, 220143, 263823, 288763, 326599], [80936, 123214, 174287, 232469, 300065, 345921, 380107, 489003, 586339]]

for i, stuff in enumerate(zip(results, dofs)):
    result, dof = stuff
    plt.plot(result, [d / 1e6 for d in dof], "o-", label="P=" + str(i + 1))
    e = np.array(err[i])
    result = np.array(result)
    dof = np.array(dof) / 1e6
    plt.plot(result[e < 0.05], dof[e < 0.05], "kx")
plt.title("Number of degrees-of-freedom given varying cell density")
plt.legend()
axes = plt.gca()
plt.ylabel("Millions of DoF")
plt.xlabel("Cells per wavelength (G)")
# axes.yaxis.set_major_formatter(FormatStrFormatter("%"))
axes.yaxis.grid()
axes.set_frame_on(False)

# plt.show()
plt.tight_layout(pad=3.0)
plt.savefig("disp2.png", transparent=True, bbox_inches="tight", pad_inches=0, dpi=1200)
